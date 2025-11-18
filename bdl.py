"""
BDL main module migrated into the `bae` package.

This file is a near-exact copy of the original top-level `bdl.py` moved
into the `bae` package so the project's core demo and training logic
live with the `bae` code. The top-level `bdl.py` is left as a small
wrapper to preserve backwards compatibility.
"""

import numpy as np
import os
import json
from math import sqrt
from copy import deepcopy
from typing import List, Tuple, Optional, Dict

# -----------------------------
# Config / Hyperparameters
# -----------------------------
SEED = 42
np.random.seed(SEED)

# Topology
N_NODES = 7
TRIADS = [(i, (i+2) % N_NODES, (i+4) % N_NODES) for i in range(N_NODES)]
CANONICAL_TRIADS = [tuple(sorted(t)) for t in TRIADS]
CANONICAL_TRIADS = list(dict.fromkeys(CANONICAL_TRIADS))

N_TRIADS = len(CANONICAL_TRIADS)
TRIAD_INPUTS = 3

# Learning & Osmotic
ETA_BASE = 0.05
LAMBDA = 0.15
C_TARGET = 1.0

PRESSURE_SENS = 0.05
LATENCY_DECAY = 0.95
JITTER_THRESHOLD = 0.30
LATENCY_WINDOW = 10

ENTROPY_MIN = 0.2
JITTER_STORM_MAG = 0.8
BETRAYAL_WINDOW = 50
ANTI_HEBB_STRENGTH = 0.02

MIN_WEIGHT_NORM = 0.05
# Pressure threshold for optional safety adjustments (large default to avoid false positives)
PRESSURE_THRESHOLD = 100.0
# New Self-Play Hyperparameters
COMP_LR_FACTOR = 0.1   # Competitive learning rate = ETA_BASE * COMP_LR_FACTOR
DISCERNMENT_TARGET = 0.7 

# Training
DEFAULT_EPOCHS = 200
SELF_PLAY_BATCH = 8

# Persistence
SAVE_DIR = "bdl0_outputs"
WEIGHTS_FILE = os.path.join(SAVE_DIR, "weights_layer0.npy")
STATE_FILE = os.path.join(SAVE_DIR, "bdl_state.json")

# Diagnostics sampling
DIAG_INTERVAL = 10

# -----------------------------
# Utilities
# -----------------------------
def ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def softmax_vec(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    s = e.sum()
    if s == 0:
        return np.ones_like(x) / x.size
    return e / s

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(num / den)

def compute_entropy_from_vec(vec: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector."""
    vec = np.clip(vec, eps, 1.0)
    return -float(np.sum(vec * np.log(vec)))

# -----------------------------
# Core Classes (Unchanged: TriadicMesh, LatencyField, ContextModulator, MetricsLogger, MemoryCore)
# -----------------------------

class TriadicMesh:
    def __init__(self, canonical_triads: List[Tuple[int,int,int]]):
        self.indices = canonical_triads
        self.n = len(canonical_triads)
        self.W = np.zeros((self.n, 3), dtype=float)

    def initialize_small_random(self, scale=0.1):
        for t in range(self.n):
            v = 0.1 + np.random.rand(3) * scale
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v /= nrm
            self.W[t, :] = v

    def norm_per_triad(self) -> np.ndarray:
        return np.linalg.norm(self.W, axis=1)

    def sparsity_count(self, threshold=1e-6) -> int:
        return int(np.sum(np.abs(self.W) > threshold))

    def save(self, path: str):
        ensure_save_dir()
        np.save(path, self.W)

    def load(self, path: str):
        if os.path.exists(path):
            self.W = np.load(path)
            if self.W.shape[0] != self.n or self.W.shape[1] != 3:
                raise ValueError("Weights shape mismatch on load.")
            return True
        return False

class LatencyField:
    def __init__(self, n_nodes: int, window: int = LATENCY_WINDOW):
        self.n = n_nodes
        self.latency = np.zeros(n_nodes, dtype=float)
        self.window = window
        self.history = np.zeros((n_nodes, window), dtype=float)
        self.idx = 0
        self.jitter = np.zeros(n_nodes, dtype=float)
        self.pressure = np.zeros((n_nodes, n_nodes), dtype=float)

    def add_latency(self, node_idx: int, delta: float):
        if 0 <= node_idx < self.n:
            self.latency[node_idx] += float(delta)
            self.history[node_idx, self.idx % self.window] = self.latency[node_idx]

    def step_jitter(self):
        count = min(self.idx + 1, self.window)
        for i in range(self.n):
            if count < 3:
                self.jitter[i] = 0.0
                continue
            h = self.history[i, :count]
            m = h.mean()
            var = ((h - m) ** 2).mean()
            self.jitter[i] = sqrt(var) if var >= 0 else 0.0
        self.idx += 1

    def compute_pressure(self):
        for i in range(self.n):
            for j in range(self.n):
                self.pressure[i,j] = 0.0 if i == j else (self.latency[i] - self.latency[j])

    def dissipate(self, decay=LATENCY_DECAY):
        self.latency *= decay

    def mean_abs_pressure(self):
        s = 0.0
        cnt = 0
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                s += abs(self.pressure[i,j])
                cnt += 1
        return s / (cnt + 1e-12)

    def mean_pressure_matrix(self) -> np.ndarray:
        """Returns the mean pressure exerted on each triad."""
        # Simple approximation: returns the pressure matrix itself for now
        return self.pressure

class SynapticDynamics:
    def __init__(self, mesh: TriadicMesh, latency_field: LatencyField):
        self.mesh = mesh
        self.lat = latency_field
        self.eta = ETA_BASE
        self.lambda_reg = LAMBDA
        self.c_target = C_TARGET
        self.pressure_sens = PRESSURE_SENS
        self.anti_hebb = ANTI_HEBB_STRENGTH

    def update_triad(self, pattern: np.ndarray, triad_idx: int, target: float,
                     reward: float = 0.0, adaptive_eta: Optional[float] = None):
        if adaptive_eta is not None:
            eta = adaptive_eta
        else:
            eta = self.eta

        tri = self.mesh.indices[triad_idx]
        w_vec = self.mesh.W[triad_idx].copy()
        prod = float(pattern[tri[0]] * pattern[tri[1]] * pattern[tri[2]])

        delta = np.zeros(3)
        for k in range(3):
            delta[k] = eta * pattern[tri[k]] * prod

        # 1. Latency Update (REQUIRED)
        mag = np.sum(np.abs(delta))
        for k in range(3):
            self.lat.add_latency(tri[k], mag / 3.0)

        # Apply anti-Hebbian if negative reward
        if reward < 0.0:
            for k in range(3):
                w_vec[k] -= self.anti_hebb * abs(reward) * pattern[tri[k]] * prod

        # 2. Homeostatic Pressure Calculation
        self.lat.compute_pressure()
        avg_pressure = 0.0
        for k in range(3):
            i = tri[k]
            avg_pressure += np.mean(np.abs(self.lat.pressure[i, :]))
        avg_pressure /= 3.0

        # 3. Homeostatic Depression Factor (sigmoid-based)
        depression = self.pressure_sens / (1.0 + np.exp(-avg_pressure))
        depression = clamp(depression, 0.0, 0.9)

        # 4. Apply Update and Depression
        for k in range(3):
            w_star = w_vec[k] + reward * delta[k]
            w_star *= (1.0 - depression)
            w_vec[k] = w_star

        # Optional: Monitor overall system pressure and adjust learning params
        total_pressure = float(np.sum(np.abs(self.lat.pressure)))
        if total_pressure > PRESSURE_THRESHOLD:
            self.adjust_learning_params(total_pressure)

        # Norm Regulation
        norm = np.linalg.norm(w_vec)
        R = norm - self.c_target
        if abs(R) > 1e-12:
            w_vec[:] = w_vec - self.lambda_reg * w_vec * R

        nrm = np.linalg.norm(w_vec)
        if nrm < 1e-12:
            w_vec[:] = 1e-3 * (0.5 - np.random.rand(3))

        self.mesh.W[triad_idx] = w_vec

    def set_eta(self, val: float):
        self.eta = float(val)

    def adjust_learning_params(self, total_pressure: float):
        """Mild safety mechanism to reduce learning aggressiveness when pressure is high.
        This is intentionally conservative to avoid destabilizing tests or normal runs.
        """
        # Reduce eta slightly but keep a reasonable floor
        old_eta = self.eta
        self.eta = max(1e-6, self.eta * 0.9)
        # Slightly increase regularization to dampen growth
        self.lambda_reg = min(1.0, self.lambda_reg * 1.02)
        # Optionally emit a lightweight debug message (noisy in tests so commented)
        # print(f"Adjusting learning params due to pressure={total_pressure:.3f}: eta {old_eta:.6f}->{self.eta:.6f}")

class TransformOps:
    @staticmethod
    def invert(x: np.ndarray) -> np.ndarray:
        return -x

    @staticmethod
    def reverse(x: np.ndarray) -> np.ndarray:
        return x[::-1]

    @staticmethod
    def transpose(x: np.ndarray, shift: int) -> np.ndarray:
        return np.roll(x, shift)

    @staticmethod
    def random_permutation(x: np.ndarray) -> np.ndarray:
        p = np.random.permutation(len(x))
        return x[p]

    @staticmethod
    def apply_sequence(x: np.ndarray, seq: List[Tuple[str, Optional[int]]]) -> np.ndarray:
        y = x.copy()
        for op, arg in seq:
            if op == "invert":
                y = TransformOps.invert(y)
            elif op == "reverse":
                y = TransformOps.reverse(y)
            elif op == "transpose":
                y = TransformOps.transpose(y, arg if arg is not None else 1)
            elif op == "perm":
                y = TransformOps.random_permutation(y)
        return y

class ContextModulator:
    def __init__(self, history_len=20):
        self.history_len = history_len
        self.history = []

    def push(self, pattern: np.ndarray):
        self.history.append(pattern.copy())
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def coherence(self, pattern: np.ndarray) -> float:
        if not self.history:
            return 1.0
        mean_hist = np.mean(self.history, axis=0)
        cos = cosine_similarity(pattern, mean_hist)
        return float(1.0 + cos)

class MetricsLogger:
    def __init__(self):
        self.logs = {
            'triad_norms': [],
            'sparsity': [],
            'entropy': [],
            'mean_jitter': [],
            'mean_pressure': [],
            'energy': [],
            'discreteness': [], # New Metric
        }

    def record(self, mesh: TriadicMesh, latency: LatencyField, entropy: float, discreteness: float):
        triad_norms = mesh.norm_per_triad().tolist()
        sparsity = mesh.sparsity_count()
        mean_jit = float(np.mean(latency.jitter))
        mean_press = float(latency.mean_abs_pressure())
        energy = float(np.sum(mesh.W ** 2))

        self.logs['triad_norms'].append(triad_norms)
        self.logs['sparsity'].append(sparsity)
        self.logs['entropy'].append(entropy)
        self.logs['mean_jitter'].append(mean_jit)
        self.logs['mean_pressure'].append(mean_press)
        self.logs['energy'].append(energy)
        self.logs['discreteness'].append(discreteness)

        return {
            'triad_norms': triad_norms,
            'sparsity': sparsity,
            'entropy': entropy,
            'mean_jitter': mean_jit,
            'mean_pressure': mean_press,
            'energy': energy,
            'discreteness': discreteness
        }

    def dump(self, path: str):
        ensure_save_dir()
        with open(path, 'w') as f:
            json.dump(self.logs, f, indent=2)

class MemoryCore:
    def __init__(self, mesh: TriadicMesh, path: str = WEIGHTS_FILE, state_path: str = STATE_FILE):
        self.mesh = mesh
        self.path = path
        self.state_path = state_path

    def save(self, metadata: Optional[dict] = None):
        ensure_save_dir()
        self.mesh.save(self.path)
        m = metadata or {}
        with open(self.state_path, 'w') as f:
            json.dump(m, f, indent=2)

    def load(self) -> bool:
        return self.mesh.load(self.path)

# -----------------------------
# System / Layers
# -----------------------------

class DorsicLayer:
    def __init__(self, mesh: TriadicMesh, synapse: SynapticDynamics, context: ContextModulator):
        self.mesh = mesh
        self.syn = synapse
        self.context = context

    def forward(self, pattern: np.ndarray) -> np.ndarray:
        out = np.zeros(self.mesh.n)
        for t in range(self.mesh.n):
            tri = self.mesh.indices[t]
            node_acts = np.array([pattern[tri[0]], pattern[tri[1]], pattern[tri[2]]])
            net = np.dot(self.mesh.W[t], node_acts)
            out[t] = 1.0 / (1.0 + np.exp(-10.0 * net))
        return out

    def train_on_pattern(self, pattern: np.ndarray, target: float = 1.0,
                         reward: float = 1.0, adaptive_boost: Optional[float] = None):
        original_eta = self.syn.eta
        if adaptive_boost is not None:
            self.syn.set_eta(original_eta * adaptive_boost)

        coherence = self.context.coherence(pattern)
        effective_eta = self.syn.eta * coherence
        
        # Reward is now scaled by the internal signal (discreteness from Trainer)
        effective_reward = reward 

        for tri_idx in range(self.mesh.n):
            self.syn.set_eta(effective_eta)
            self.syn.update_triad(pattern, tri_idx, target=target, reward=effective_reward)

        self.context.push(pattern)
        self.syn.set_eta(original_eta)

class IonicLayer:
    def __init__(self, n_triads: int, pressure_field: LatencyField, mesh_indices: Optional[List[Tuple[int,int,int]]] = None):
        self.n = n_triads
        self.pressure_field = pressure_field
        # Triad indices used for pressure modulation; default to global TRIADS for backwards compatibility
        self.mesh_indices = mesh_indices if mesh_indices is not None else TRIADS
        
        # Trainable Weights (Decoupled from constants)
        # W_lateral: Competition matrix (Identity + off-diagonal inhibition)
        self.W_lateral = np.eye(self.n) * 1.0 + (1.0 - np.eye(self.n)) * (-0.3)
        # W_feedback: Top-down policy reinforcement vector
        self.W_feedback = np.ones(self.n) * 0.5 

    def forward(self, dorsic_out: np.ndarray, corinthian_out: np.ndarray) -> np.ndarray:
        
        # 1. Lateral Competition (Dorsic -> Ionic)
        lateral_net = self.W_lateral @ dorsic_out
        
        # 2. Pressure Modulation (Homeostatic regulation)
        # Use mean pressure (a simplification for now, but a matrix multiplication is cleaner)
        self.pressure_field.compute_pressure()
        pressure_mod = np.ones_like(dorsic_out)
        
        # Calculate individual triad pressure modulation (sum of pressure from all nodes involved)
        for t in range(self.n):
            tri = self.mesh_indices[t]
            # Sum of mean absolute pressure exerted on nodes in this triad
            triad_pressure = np.sum([np.mean(np.abs(self.pressure_field.pressure[n, :])) for n in tri]) / 3.0
            pressure_mod[t] = 1.0 / (1.0 + triad_pressure * 0.1)

        lateral_net *= pressure_mod

        # 3. Policy Feedback (Corinthian -> Ionic)
        feedback_net = self.W_feedback * corinthian_out
        
        s = lateral_net + feedback_net
        return 1.0 / (1.0 + np.exp(-10.0 * s))


class CorinthianLayer:
    def __init__(self, n_triads: int):
        self.n = n_triads

    def forward(self, ionic_in: np.ndarray) -> np.ndarray:
        # Remains the softmax decision layer
        return softmax_vec(ionic_in)

# -----------------------------
# Trainer & Experiment Suite (Modified)
# -----------------------------

class Trainer:
    def __init__(self,
                 dorsic: DorsicLayer,
                 ionic: IonicLayer,
                 corinthian: CorinthianLayer,
                 mesh: TriadicMesh,
                 latency: LatencyField,
                 metrics: MetricsLogger,
                 memory: MemoryCore):
        self.dorsic = dorsic
        self.ionic = ionic
        self.corinthian = corinthian
        self.mesh = mesh
        self.lat = latency
        self.metrics = metrics
        self.mem = memory
        self.epoch = 0
        self.core_history = [[] for _ in range(self.mesh.n)]

    def interware_cycle(self, patterns: List[np.ndarray], epochs: int = 100):
        # ... (Unchanged)
        for e in range(epochs):
            if e % 4 == 0:
                yield [patterns[i] for i in np.random.permutation(len(patterns))]
            elif e % 4 == 1:
                yield [p[::-1] for p in patterns]
            elif e % 4 == 2:
                shift = (e % N_NODES)
                yield [np.roll(p, shift) for p in patterns]
            else:
                yield [-p for p in patterns]

    def corinthian_to_nodes(self, cor_out: np.ndarray) -> np.ndarray:
        node_pattern = np.zeros(N_NODES)
        counts = np.zeros(N_NODES)
        for idx, tri in enumerate(self.mesh.indices):
            for k in range(3):
                node_pattern[tri[k]] += cor_out[idx]
                counts[tri[k]] += 1
        counts[counts == 0] = 1.0
        return node_pattern / counts

    def measure_discreteness(self, cor_out: np.ndarray) -> float:
        """Measures how discrete (sharp) the policy decision is."""
        sorted_acts = np.sort(cor_out)[::-1]
        
        if len(sorted_acts) < 2: return 0.0

        # Separation: Winner's margin (how much it won by)
        separation = sorted_acts[0] - sorted_acts[1] 
        
        # Suppression: Loser depression (how close the weakest is to 0)
        suppression = 1.0 - sorted_acts[-1] 
        
        return np.mean([separation, suppression])

    def train_competitive_layers(self, dorsic_out: np.ndarray, cor_out: np.ndarray, discreteness: float):
        """Updates Ionic (L1) and Corinthian (L2) weights based on self-generated 'discreteness' reward."""
        
        # Loss Signal: difference from the target sharpness
        loss_signal = discreteness - DISCERNMENT_TARGET 
        LR_COMP = self.dorsic.syn.eta * COMP_LR_FACTOR
        winner = np.argmax(cor_out)
        
        # If the decision is already sharp enough, reinforce; if fuzzy, penalize/depress.
        reinforce_factor = np.tanh(loss_signal * 5.0) # Smooth factor for reinforcement/depression

        for t in range(self.mesh.n):
            for s_idx in range(self.mesh.n):
                # 1. Update W_lateral (Self-Excitation/Lateral Inhibition)
                
                # Update magnitude scales with input and current decision strength
                update_mag = LR_COMP * dorsic_out[s_idx] * cor_out[s_idx]
                
                if t == s_idx: # Self-Excitation Term
                    # Reinforce self-excitation if policy is sharp
                    self.ionic.W_lateral[t, s_idx] += reinforce_factor * update_mag
                else: # Lateral Inhibition Term
                    # Strengthen inhibition if policy is sharp (sharpening boundaries)
                    # or weaken inhibition if policy is too depressed (allowing new competition)
                    self.ionic.W_lateral[t, s_idx] -= reinforce_factor * update_mag * 0.5

            # 2. Update W_feedback (Top-down Policy Reinforcement)
            # Policy feedback is strengthened/weakened globally based on decision quality.
            self.ionic.W_feedback[t] += reinforce_factor * LR_COMP

        # Clip weights to maintain stability and prevent runaway values
        self.ionic.W_lateral = np.clip(self.ionic.W_lateral, -1.0, 1.0)
        self.ionic.W_feedback = np.clip(self.ionic.W_feedback, 0.0, 1.0)


    def single_epoch(self, batch: List[np.ndarray], motif_train: bool = True,
                     rival: Optional[np.ndarray] = None):
        self.lat.step_jitter()
        trans = np.sum(self.lat.jitter > JITTER_THRESHOLD) > (self.lat.n // 2)
        adaptive_boost = 1.0 + self.lat.mean_abs_pressure() * 0.1 if trans else None

        representative_pattern = batch[0] if batch else np.ones(N_NODES)
        
        # 1. Forward Pass
        dorsic_out = self.dorsic.forward(representative_pattern)
        # Recurrent call using last policy decision (simplification: start with zeros for a clean pass)
        ionic_out = self.ionic.forward(dorsic_out, np.zeros(self.mesh.n)) 
        cor_out = self.corinthian.forward(ionic_out)

        # 2. Self-Play Signal Generation
        discreteness = self.measure_discreteness(cor_out)
        # Use discreteness to modulate the reward signal for Layer 0 (Dorsic)
        # Reward is high when the internal policy decision is sharp (self-consistent)
        effective_reward = 1.0 + np.tanh((discreteness - DISCERNMENT_TARGET) * 2.0)
        
        # 3. Dorsic (Layer 0) Training (Hebbian/Homeostatic)
        if motif_train:
            self.dorsic.train_on_pattern(representative_pattern, target=1.0, 
                                         reward=effective_reward, adaptive_boost=adaptive_boost)
        if rival is not None:
            self.dorsic.train_on_pattern(rival, target=0.0, 
                                         reward=-0.5, adaptive_boost=adaptive_boost)

        # 4. Competitive Layer (L1/L2) Training (Discrete Discernment)
        self.train_competitive_layers(dorsic_out, cor_out, discreteness) 

        self.lat.dissipate()

        entropy = compute_entropy_from_vec(ionic_out)
        collapsed = False
        if entropy < ENTROPY_MIN:
            self.lat.latency[:] += JITTER_STORM_MAG
            collapsed = True

        core_mask = dorsic_out > 0.7
        for t in range(self.mesh.n):
            self.core_history[t].append(float(core_mask[t]))
            if len(self.core_history[t]) > BETRAYAL_WINDOW:
                recent_core = np.mean(self.core_history[t][-BETRAYAL_WINDOW:]) > 0.5
                currently_inactive = dorsic_out[t] < 0.1
                if recent_core and currently_inactive:
                    nrm = np.linalg.norm(self.mesh.W[t])
                    if nrm > MIN_WEIGHT_NORM:
                        self.mesh.W[t] *= 0.8
                        # Slight damping applied to prevent runaway core neurons

        # 7. Log Metrics & Update Memory State
        discreteness_value = self.measure_discreteness(cor_out)
        self.metrics.record(self.mesh, self.lat, entropy, discreteness_value)

        # Save the memory state
        self.mem.save(metadata={'epoch': self.epoch, 'collapsed': collapsed})

        # Increment Epoch counter
        self.epoch += 1

    def sample_batch(self, patterns: List[np.ndarray], batch_size: int = SELF_PLAY_BATCH) -> List[np.ndarray]:
        """Return a randomized batch of patterns (with replacement) for self-play."""
        if not patterns:
            return [np.ones(N_NODES) for _ in range(batch_size)]
        idx = np.random.choice(len(patterns), size=batch_size, replace=True)
        return [patterns[i] for i in idx]

    def run(self, patterns: List[np.ndarray], epochs: int = DEFAULT_EPOCHS, save_interval: int = DIAG_INTERVAL, verbose: bool = True):
        """High-level training loop that iterates over transformed sequences from
        `interware_cycle`, calls `single_epoch`, and periodically saves metrics
        and memory. This completes the truncated Trainer class.
        """
        ensure_save_dir()
        gen = self.interware_cycle(patterns, epochs=epochs)
        for e, batch_seq in enumerate(gen):
            # `interware_cycle` yields a list of patterns for the epoch; use that
            batch = batch_seq if isinstance(batch_seq, list) else [batch_seq]

            # Optionally sample a smaller self-play batch for intra-epoch training
            sp_batch = self.sample_batch(batch)

            # Run a single epoch using the sampled batch as the representative set
            self.single_epoch(sp_batch, motif_train=True)

            # Verbose status every ~10% of progress
            if verbose and (e % max(1, (epochs // 10))) == 0:
                d_out = self.dorsic.forward(sp_batch[0])
                i_out = self.ionic.forward(d_out, np.zeros(self.mesh.n))
                ent = compute_entropy_from_vec(i_out)
                disc = self.measure_discreteness(self.corinthian.forward(i_out))
                print(f"Epoch {e+1}/{epochs}: entropy={ent:.3f}, discreteness={disc:.3f}")

            # Periodic persistence
            if (e + 1) % save_interval == 0:
                self.metrics.dump(os.path.join(SAVE_DIR, "metrics.json"))
                self.mem.save(metadata={'epoch': self.epoch})

        # Final save and short summary
        self.metrics.dump(os.path.join(SAVE_DIR, "metrics.json"))
        self.mem.save(metadata={'epoch': self.epoch})
        if verbose:
            print("Training complete. Metrics and memory saved to output directory.")