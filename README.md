# Wavesim

A compact, validated **FDTD electromagnetic solver** in Python + NumPy.

Wavesim integrates Maxwell's equations on a Yee grid with a functional design:
one `FDTDGrid` state object and a set of pure functions that advance it. Tests
00–04 run the full 3D arrays as a thin 2D slice (`Nz = 1`); Test 05 is the first
true 3D run (`Nz > 1`), exercising the 3D curl and z-face CPML on the same code.
Boundaries are **CPML** (convolutional perfect matched layer) and **PEC**
(perfect electric conductor); sources are soft-injected Gaussian pulses.

The import package is named `wavesim`.

---

## Features

- **Functional core** — `FDTDGrid` dataclass + pure update functions; you write
  the time loop, there is no hidden framework. An optional `Simulation` class
  runs the canonical loop for you (same physics, bit-identical results) when you
  want to skip the boilerplate.
- **Full 3D Yee curl** operators, vectorised with NumPy slicing (no cell loops).
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

# 2. clone & run
git clone https://github.com/itaischles/Wavesim.git
cd Wavesim
python tests/test_02_free_space.py
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
snap = SnapshotMonitor(component='Ez', k_slice=0, interval=20)

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

sim  = Simulation(grid, cpml=cpml)
sim.add_source(PointSource('Ez', 100, 100, 0, make_source_for_fmax(10e9)))
snap = sim.add_monitor(SnapshotMonitor('Ez', k_slice=0, interval=20))
sim.run(2000)                                          # bit-identical to the loop
```

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
│   ├── update.py     # E and H field updates (3D curl)
│   ├── pml.py        # CPML init + corrections (per-face selectable)
│   ├── pec.py        # PEC faces and interior conductor mask
│   ├── sources.py    # Gaussian waveform + Source / PointSource / ArraySource
│   ├── monitors.py   # field / magnitude / snapshot / energy monitors
│   ├── simulation.py # Simulation class — runs the canonical time loop for you
│   ├── viz.py        # all plotting and animation (2D + full-3D helpers)
│   └── constants.py  # C0, EPS0, MU0, ETA0
├── tests/            # validated example simulations (run in order)
├── tools/            # profile_3d.py — memory & runtime profiling harness
└── docs/             # API_GUIDE.md, HOW_TO_SET_UP.md, design/dev notes
```

---

## Tests

Run from the project root, in order — each validates one subsystem.

| Test | What it validates | Status |
|------|-------------------|--------|
| `test_00_grid_viz.py`   | Yee grid / material / PML visualisation (no physics) | ✅ |
| `test_01_source_viz.py` | Gaussian source waveform & bandwidth | ✅ |
| `test_02_free_space.py` | Pulse propagation + CPML absorption + symmetry | ✅ |
| `test_03_pec_cavity.py` | PEC cavity resonances vs analytic TM modes (<0.04%) | ✅ |
| `test_04_waveguide.py`  | Waveguide cutoff: evanescence below, phase velocity above | ✅ |
| `test_05_coax_tem.py`   | Coaxial TEM mode (first full 3D run, `Nz>1`): 1/r profile, `Z=η₀`, `v=c` | ✅ |
| `test_06_box_cavity_3d.py` | Volumetric 3D PEC cavity: 14 analytic resonances within 1.5% (10 with `p≥1`, a half-wave along `z`) | ✅ |
| `test_07_simulation_api.py` | v2 `Simulation`/`Source` API tutorial: open-boundary absorption + symmetry, bit-identical to the manual loop | ✅ |

Tests 02–07 also save an animated GIF alongside their PNG (both git-ignored,
regenerated on each run).

A profiling harness, `tools/profile_3d.py`, sweeps cube sizes and reports
throughput (µs per cell-step) and memory (bytes per cell) for the pure-NumPy
backend — see [ROADMAP.md](ROADMAP.md) §1/§3 for the headline numbers.

---

## Documentation

- **[docs/API_GUIDE.md](docs/API_GUIDE.md)** — comprehensive user-facing API
  reference with worked examples.
- **[docs/HOW_TO_SET_UP.md](docs/HOW_TO_SET_UP.md)** — environment setup (conda
  + VS Code) and how to run the tests.
- **[ROADMAP.md](ROADMAP.md)** — what's planned next (full 3D, JAX, nonuniform
  grid, mode solver) and what's landed (v2 `Simulation`/`Source` layer).
- **[DEBUG_NOTES_test02_pml.md](DEBUG_NOTES_test02_pml.md)**,
  **[PML_NOTES_2026-06-12_independent_faces.md](PML_NOTES_2026-06-12_independent_faces.md)**
  — CPML implementation notes.

---

## Status

v1: a validated 2D-in-3D solver (CPML + PEC) through Test 04, plus full 3D —
Test 05 (coaxial TEM, axial propagation) and Test 06 (rectangular PEC cavity, a
genuinely volumetric mode varying in all three axes). 3D is validated against
analytic ground truth, profiled (`tools/profile_3d.py`), and visualised with
dedicated orthogonal-slice helpers in `viz.py`.

## What's next

See **[ROADMAP.md](ROADMAP.md)** — making full 3D first-class, a JAX performance
backend, a nonuniform rectilinear grid, and a waveguide-port mode solver with
modal injection. (The v2 `Simulation`/`Source` orchestration layer has landed —
see `test_07_simulation_api.py`.)