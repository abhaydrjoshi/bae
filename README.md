BAE 
=====================================

Overview
--------
This package contains a compact, readable implementation of a triadic/self-play neural engine (BAE) consolidated from several experimental files in the repository. It's intentionally minimal so you can iterate quickly.

BAE package

===========


Overview

--------

This repository contains three related modules packaged together under `bae`:


- `bae.py` — a small conversational agent (`Bae` class) intended as a lightweight example/utility.

- `bdl.py` — the behavioral/dorsic learning (BDL) engine with the triadic mesh, latency field, synaptic dynamics, layers and `Trainer`.

- `engine.py` — a token-generation helper (the `Engine` class and `KVCache`) used for efficient generation and tooling.


What to import

---------------

You can import the package root to get convenient access to the common classes:


```python

import bae


print(bae.__version__)


# conversational agent

from bae import Bae, create_bae


# core BDL types

from bae import TriadicMesh, LatencyField, SynapticDynamics, Trainer


# engine helpers

from bae import Engine, KVCache

```


Quick start (example flows)

--------------------------


1) Run or test the conversational agent


```powershell

python -c "from bae import create_bae; b=create_bae(); print(b.respond('Why do you love music?'))"

```


2) Create and initialize a triadic mesh and run a short training loop


```python

import numpy as np

from bae import TriadicMesh, LatencyField, SynapticDynamics, ContextModulator, DorsicLayer, IonicLayer, CorinthianLayer, Trainer, MetricsLogger, MemoryCore


mesh = TriadicMesh([(0,2,4),(1,3,5),(2,4,6)])

mesh.initialize_small_random()

lat = LatencyField(7)

syn = SynapticDynamics(mesh, lat)

ctx = ContextModulator()

dorsic = DorsicLayer(mesh, syn, ctx)

ionic = IonicLayer(mesh.n, lat, mesh.indices)

cor = CorinthianLayer(mesh.n)

metrics = MetricsLogger()

mem = MemoryCore(mesh)

trainer = Trainer(dorsic, ionic, cor, mesh, lat, metrics, mem)

trainer.run([np.random.randn(7) for _ in range(8)], epochs=5)

```


3) Use the `Engine`/`KVCache` utilities (for token generation tooling)


```python

from bae import Engine, KVCache

# Engine expects a model+tokenizer — see engine.py for its API and examples

```


Notes

-----

- Dependencies used in this repository include `numpy` and `torch` for the `bdl` and `engine` modules; install them as needed.

- The repository is intentionally experimental; if you add scripts or runners, update the README accordingly.


Tests

-----

`test_engine.py` contains a unit test that verifies `KVCache` resizing behavior.


License & provenance

---------------------

This code consolidates experimental modules for easier iteration and testing. Contributions and tests are welcome.
