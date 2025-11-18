BAE — Local Training and Self-Play

Minimal, local-first conversational agent with a simple 4->1 scorer trained on qualitative data or self-play. TensorBoard is used as the visualizer.

What’s included

- Torch path (primary): rule-driven responder + trainable scorer (now ~100k params by default) with a production-ready trainer.
- `baeo/` path (experimental): a numpy-only triadic self-play core showcasing the essential intention (homeostasis, LTP/LTD, discreteness).

Research flow (local)

- Self-play (default): trains the scorer on synthetic prompts; good for quick iterations.
- Dataset run (optional): if datasets are present, trains on qualitative corpora; else falls back to synthetic seeds.
- Tune scorer capacity via env to explore ~100k parameter regimes.

Linux/macOS:
```bash
python3 -m venv ~/.venvs/bae && source ~/.venvs/bae/bin/activate
pip install -U pip setuptools wheel && pip install -r requirements.txt
export PYTHONPATH=$(pwd)
export BAE_SCORER_HIDDEN=320
export BAE_SCORER_LAYERS=2
export BAE_VERBOSE_MODEL=1
python selfplay_training.py
tensorboard --logdir runs/bae_selfplay
```

Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
$env:PYTHONPATH=(Get-Location)
$env:BAE_SCORER_HIDDEN=320
$env:BAE_SCORER_LAYERS=2
$env:BAE_VERBOSE_MODEL=1
python .\selfplay_training.py
tensorboard --logdir runs\bae_selfplay
```

Setup (Pop!_OS / Ubuntu)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
python3 -m venv ~/.venvs/bae
source ~/.venvs/bae/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Quick Start — Self-Play Training

Self-play generates synthetic prompts and trains the scorer.

```bash
source ~/.venvs/bae/bin/activate
PYTHONPATH=$(pwd) python selfplay_training.py
```

- Checkpoints: checkpoints_selfplay/
- TensorBoard logs: runs/bae_selfplay/

Launch visualizer:
```bash
tensorboard --logdir runs/bae_selfplay
```

Windows (PowerShell) quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
$env:PYTHONPATH=(Get-Location)
python .\selfplay_training.py
# visualize
tensorboard --logdir runs\bae_selfplay
```

Dataset-based Training (optional)

EmpatheticDialogues repo is included under data/EmpatheticDialogues/ (no CSVs by default). The runner falls back to DailyDialog (if present) or a small synthetic set.

```bash
source ~/.venvs/bae/bin/activate
PYTHONPATH=$(pwd) python run_training.py
tensorboard --logdir runs/bae
```

Minimal Agent

bae.py exposes bae() with:
- respond(text: str) -> str
- scorer: Torch module taking 4 features to 1 score

Visualizer (CLI)

A simple visualizer emits CSV from the latest checkpoint history and prints a brief summary.

```bash
source ~/.venvs/bae/bin/activate
PYTHONPATH=$(pwd) python visualizer.py --checkpoints checkpoints_selfplay --out runs/metrics_selfplay.csv
# or for dataset run
PYTHONPATH=$(pwd) python visualizer.py --checkpoints checkpoints --out runs/metrics.csv
```

Project Structure

- bae.py: minimal agent
- trainer.py: training loop for the scorer
- dataset.py: loaders for qualitative corpora (EmpatheticDialogues/DailyDialog)
- run_training.py: dataset-based training entrypoint
- selfplay_training.py: self-supervised play training
- visualizer.py: export/print training metrics from checkpoints
- experimental/: isolated nanochat-derived scripts (not required locally)
 - e_e.py: eternal enhancements (EWC, curriculum, meta). Uses continuous_rl_bae.py
 - continuous_rl_bae.py: minimal base RL agent for e_e.py

OrbChat integration (hmmyn-main)

- Service:
  - FastAPI app at `app/server.py`
  - Run locally:
    - Linux/macOS:
      ```bash
      export PYTHONPATH=$(pwd)
      uvicorn app.server:app --host 0.0.0.0 --port 8080 --reload
      ```
    - Windows PowerShell:
      ```powershell
      $env:PYTHONPATH=(Get-Location)
      uvicorn app.server:app --host 0.0.0.0 --port 8080 --reload
      ```
- Client:
  - TypeScript client at `clients/typescript/orbchat.ts`:
    ```ts
    import {respond} from "./clients/typescript/orbchat"
    const out = await respond("http://localhost:8080", "session-123", "I feel overwhelmed.")
    ```
- Config (env):
  - `BAE_SCORER_HIDDEN`, `BAE_SCORER_LAYERS` — scorer size
  - `BAE_ENABLE_SAFETY` (default: true) — basic profanity/PII masking
  - `BAE_MEMORY_LEN` (default: 5) — per-session FIFO length
  - `BAE_HOST`, `BAE_PORT`, `BAE_DEBUG` — server settings

Scorer model size and 100k params

- By default, the scorer is an MLP (~100k parameters) that maps 4 engineered features → 1 score.
- You can tune its size via environment variables:

Linux/macOS:
```bash
export BAE_SCORER_HIDDEN=320     # hidden width (default 320)
export BAE_SCORER_LAYERS=2       # number of hidden layers (default 2)
export BAE_VERBOSE_MODEL=1       # prints parameter count
```

Windows PowerShell:
```powershell
$env:BAE_SCORER_HIDDEN=320
$env:BAE_SCORER_LAYERS=2
$env:BAE_VERBOSE_MODEL=1
```

Notes on `baeo/` (experimental)

- `baeo/` contains a compact, numpy-only “triadic” core that explores the essential intention: discrete decisions via homeostasis and LTP/LTD.
- It is educational and self-contained. The primary training flow (torch + trainer) remains the recommended path for optimization at ~100k params.

Tips

- Make PYTHONPATH persistent:
  ```bash
  echo 'export PYTHONPATH=$(pwd):$PYTHONPATH' >> ~/.bashrc && source ~/.bashrc
  ```
- GPU build of PyTorch (optional):
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```



