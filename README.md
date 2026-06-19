# Wavesim

A compact, validated **FDTD electromagnetic solver** in Python + NumPy, with an
optional multithreaded **Numba** backend (~10–12× faster, same results).

Wavesim integrates Maxwell's equations on a Yee grid with a functional design:
one `FDTDGrid` state object and a set of pure functions that advance it. The full
3D arrays can run as a thin 2D slice (`Nz = 1`) or as a true 3D domain (`Nz > 1`)
on the same code. Boundaries are **CPML** (convolutional perfect matched layer)
and **PEC** (perfect electric conductor); sources are soft-injected Gaussian
pulses.

The import package is named `wavesim`.

---

## Features

- **Functional core** — `FDTDGrid` dataclass + pure update functions; you write
  the time loop, there is no hidden framework. An optional `Simulation` class
  runs the canonical loop for you (same physics, bit-identical results) when you
  want to skip the boilerplate.
- **Full 3D Yee curl** operators, vectorised with NumPy slicing (no cell loops).
- **Optional Numba backend** — JIT, multithreaded drop-in kernels for the hot
  update functions (`Simulation(backend='numba')`), **~10–12× faster** than NumPy
  at 3D sizes and **bit-identical** to it. NumPy stays the default and the
  reference; Numba is opt-in.
- **CPML boundaries** (Roden–Gedney) selectable **per face**, so PEC walls and
  symmetry planes are easy to combine with absorbing ends.
- **PEC** domain faces and interior conductor masks (boxes, cylinders, coax).
- **Gaussian sources** (baseband envelope + narrowband/CW recipe), plus a v2
  `Source` abstraction (`PointSource`, `ArraySource`, or subclass your own).
- **Diagnostics** — point field, `|E|`/`|H|` magnitude, 2D snapshots, and total
  energy monitors.
- **Visualisation** — grid/material/PML plots, field snapshots, animations, and
  full-3D helpers (orthogonal XY/XZ/YZ slice triptych + multi-plane animation).
- **Validated** — each subsystem is checked against analytic results.

---

## Quickstart

```bash
# 1. environment (see docs/HOW_TO_SET_UP.md for details)
conda create -n wavesim python=3.11 -y
conda activate wavesim
conda install -n wavesim numpy matplotlib scipy pillow -y
pip install numba          # optional — only for the faster backend='numba'

# 2. clone
git clone https://github.com/itaischles/Wavesim.git
cd Wavesim
```

A minimal free-space pulse:

```python
import numpy as np
from wavesim.grid import create_grid
from wavesim.materials import set_vacuum
from wavesim.update import update_H, update_E
from wavesim.pml import init_cpml, update_H_pml, update_E_pml
from wavesim.pec import apply_pec_mask
from wavesim.sources import GaussianSource, gaussian_pulse
from wavesim.monitors import SnapshotMonitor, record_snapshot

grid = create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)   # dt set automatically
grid = set_vacuum(grid)
cpml = init_cpml(grid, d_pml=10)                       # absorb on all 4 faces
src  = GaussianSource(t0=4/(2*np.pi*10e9), width=1/(2*np.pi*10e9))
snap = SnapshotMonitor(component='Ez', at_z=0.0, every_N_steps=20)

for n in range(2000):
    t = n * grid.dt
    grid = update_H(grid);  grid, cpml = update_H_pml(grid, cpml)
    grid = update_E(grid);  grid, cpml = update_E_pml(grid, cpml)
    grid = apply_pec_mask(grid)
    grid.Ez[100, 100, 0] += gaussian_pulse(src, t)     # soft injection
    record_snapshot(snap, grid)
    grid.time_step += 1
```

Prefer to skip the loop? The same run via the v2 orchestration layer:

```python
from wavesim.simulation import Simulation
from wavesim.sources import PointSource, make_source_for_fmax

sim  = Simulation(grid, cpml=cpml)                     # backend='numba' for ~10× speed
sim.add_source(PointSource('Ez', 50e-3, 50e-3, 0.0, make_source_for_fmax(10e9)))
snap = sim.add_monitor(SnapshotMonitor('Ez', at_z=0.0, every_N_steps=20))
sim.run(2000)                                          # bit-identical to the loop
```

For large 3D runs, pass `backend='numba'` to `Simulation` (requires `pip install
numba`) — multithreaded JIT kernels that are bit-identical to the NumPy default.
The stencil is memory-bandwidth-bound, so ~4–6 threads is the sweet spot
(`numba.set_num_threads(4)`). See [ROADMAP.md](ROADMAP.md) §3.

See the **[API Guide](docs/API_GUIDE.md)** for the full reference, the canonical
loop, and the conventions worth committing to memory.

---

## Repository layout

```
Wavesim/
├── README.md
├── wavesim/             # solver package
│   ├── grid.py       # FDTDGrid dataclass + create_grid
│   ├── materials.py  # vacuum / box / cylinder / coax / raw-array builders
│   ├── update.py        # E and H field updates (3D curl) — NumPy reference
│   ├── pml.py           # CPML init + corrections (per-face selectable)
│   ├── pec.py           # PEC faces and interior conductor mask
│   ├── backend_numba.py # optional Numba JIT/multithreaded drop-in for update.py + pml.py
│   ├── sources.py       # Gaussian waveform + Source / PointSource / ArraySource
│   ├── monitors.py      # field / magnitude / snapshot / energy monitors
│   ├── simulation.py    # Simulation class — runs the canonical loop (backend='numpy'|'numba')
│   ├── viz.py           # all plotting and animation (2D + full-3D helpers)
│   └── constants.py     # C0, EPS0, MU0, ETA0
└── docs/             # API_GUIDE.md, HOW_TO_SET_UP.md, design/dev notes
```

---

## Documentation

- **[docs/API_GUIDE.md](docs/API_GUIDE.md)** — comprehensive user-facing API
  reference with worked examples.
- **[docs/HOW_TO_SET_UP.md](docs/HOW_TO_SET_UP.md)** — environment setup (conda
  + VS Code) and how to run a simulation.
- **[ROADMAP.md](ROADMAP.md)** — what's planned next (nonuniform grid, mode
  solver) and what's landed (v2 `Simulation`/`Source` layer, the Numba backend).

---

## Status

v1: a validated 2D-in-3D solver (CPML + PEC), plus full 3D — coaxial TEM (axial
propagation) and a rectangular PEC cavity (a genuinely volumetric mode varying in
all three axes). 3D is validated against analytic ground truth, profiled, and
visualised with dedicated orthogonal-slice helpers in `viz.py`.

## What's next

See **[ROADMAP.md](ROADMAP.md)** — making full 3D first-class, a nonuniform
rectilinear grid, and a waveguide-port mode solver with modal injection. (The v2
`Simulation`/`Source` orchestration layer and the optional ~10–12× Numba backend
have landed.)