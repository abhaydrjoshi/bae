BAE 
=====================================

Overview
--------
This package contains a compact, readable implementation of a triadic/self-play neural engine (BAE) consolidated from several experimental files in the repository. It's intentionally minimal so you can iterate quickly.

What you'll find
-----------------
- `engine.py` — the canonical consolidated engine with key classes: `TriadicMesh`, `LatencyField`, `SynapticDynamics`, `ContextModulator`, `DorsicLayer`, `IonicLayer`, `CorinthianLayer`, `Trainer`, `MetricsLogger`, `FullMCTS`, and `ExperimentSuite`.
- `run_bae.py` — a small runner for quick smoke tests and examples.
- Outputs are written to the `bdl0_outputs/` directory:
  - `weights_layer0.npy` — numpy array with layer-0 weights (triads x 3).
  - `metrics.json` — recorded time-series metrics (triad norms, entropy, jitter, pressure, energy).
  - `bdl_state.json` — small state file with epoch or meta info.

Quick start
-----------
(From the repository root, Windows PowerShell)

```powershell
python -m pip install -r requirements.txt
python run_bae.py
```

Simple usage (from Python)
--------------------------
```python
import bae
print('BAE version:', bae.__version__)

# Run a quick demo (builds engine and runs a short training loop)
trainer = bae.main_demo(epochs=20, batch=8)

# Or construct components manually and use Trainer
from bae import TriadicMesh, LatencyField, SynapticDynamics, ContextModulator, DorsicLayer, IonicLayer, CorinthianLayer, Trainer
mesh = TriadicMesh([(0,2,4),(1,3,5),(2,4,6)])  # example triads
mesh.initialize_small_random()
lat = LatencyField(7)
syn = SynapticDynamics(mesh, lat)
ctx = ContextModulator()
dorsic = DorsicLayer(mesh, syn, ctx)
ionic = IonicLayer(mesh.n, lat, mesh.indices)
cor = CorinthianLayer(mesh.n)
trainer = Trainer(dorsic, ionic, cor, mesh, lat)
trainer.train([np.random.randn(7) for _ in range(8)], epochs=10)
```

Key classes / API
------------------
- `TriadicMesh` — container for triadic weights and save/load helpers.
- `LatencyField` — maintains per-node latency, jitter and pairwise pressure.
- `SynapticDynamics` — Hebbian/anti-Hebbian updates for triads.
- `DorsicLayer`, `IonicLayer`, `CorinthianLayer` — layer forward/update mechanics.
- `Trainer` — main training loop; use `Trainer.train(patterns, epochs)`.
- `FullMCTS` — optional MCTS-like policy wrapper (experimental).
- `MetricsLogger` — collects and dumps training metrics to `metrics.json`.

Notes and tips
-------------
- Dependencies: `numpy` (see `requirements.txt`).
- The optional `FullMCTS` scoring is experimental and can be replaced by providing an alternative `mcts_impl` to `Trainer`.
- To resume training, keep `bdl0_outputs/weights_layer0.npy` — `engine.py` tries to load it at startup.

Troubleshooting
---------------
- If `run_bae.py` fails due to missing packages, run the install command above.
- If weights fail to load, `engine.py` prints a warning and continues with random init.

License & provenance
---------------------
This package consolidates experimental code from the repository for easier iteration. Feel free to adapt and add tests.
