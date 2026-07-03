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
  runs the canonical loop for you (same physics, bit-identical results).
- **Full 3D Yee curl** operators, vectorised with NumPy slicing (no cell loops).
- **Optional Numba backend** — JIT, multithreaded drop-in kernels for the hot
  update functions (`Simulation(backend='numba')`), **~10–12× faster** than NumPy
  at 3D sizes and **bit-identical** to it. NumPy stays the default and the
  reference; Numba is opt-in.
- **CPML boundaries** (Roden–Gedney) selectable **per face**, so PEC walls and
  symmetry planes are easy to combine with absorbing ends.
- **PEC** domain faces and interior conductor masks (boxes, cylinders, coax).
- **Gaussian sources** (baseband envelope + narrowband/CW recipe), plus a
  `Source` abstraction — `PointSource`, `LineSource`, `PlaneSource`,
  `VolumeSource`, `ArraySource`, or subclass your own.
- **Lumped V-I-Z elements** — `LineSource` places a Thevenin/Norton source,
  ideal voltage/current source, or passive resistor on a line between two
  points (semi-implicit, stable for any Z > 0) and self-records its port
  V(t)/I(t) for impedance and S-parameter extraction.
- **2D TEM mode solver** — finds the PEC conductor cross-sections on a grid plane
  and solves each supported TEM mode (ε-weighted electrostatic BVP), reporting
  per-unit-length C, L, Z₀, phase velocity and ε_eff. A solved mode launches
  straight into the run as a directional input port (`TEMMode.to_source`).
- **Diagnostics** — point field, `|E|`/`|H|` magnitude, 2D snapshots (XY/XZ/YZ
  slice planes), line-integral voltage (∫E·dl) and current (∮H·dl) monitors, and
  total energy.
- **Visualisation** — grid/material/PML plots, field snapshots, animations,
  voltage/current time series, TEM-mode profiles, and full-3D helpers (orthogonal
  XY/XZ/YZ slice triptych + multi-plane animation).
- **Validated** — each subsystem is checked against analytic results.

---

## Quickstart

The only hard dependencies are `numpy`, `scipy`, `matplotlib`, and `pillow`
(`numba` is optional, for the faster backend). Use conda **or** a plain-Python
`venv` + `pip` — either works. See [docs/HOW_TO_SET_UP.md](docs/HOW_TO_SET_UP.md)
for details.

```bash
# Option A — conda
conda create -n wavesim python=3.11 -y
conda activate wavesim
conda install -n wavesim numpy scipy matplotlib pillow -y   # scipy: TEM mode solver
pip install numba          # optional — only for the faster backend='numba'
```

```bash
# Option B — plain Python 3.10+ with a venv (no conda)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pillow
pip install numba          # optional — only for the faster backend='numba'
```

```bash
# then, either way, clone
git clone https://github.com/itaischles/Wavesim.git
cd Wavesim
```

A minimal free-space pulse:

```python
import wavesim as ws

grid = ws.create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)   # dt set automatically
grid = ws.set_vacuum(grid)
cpml = ws.init_cpml(grid, d_pml=10)                       # absorb on all 4 faces
src  = ws.GaussianPulse.for_fmax(10e9)                    # callable waveform src(t)
snap = ws.SnapshotMonitor(component='Ez', at_z=0.0, every_N_steps=20)

for n in range(2000):
    t = n * grid.dt
    grid = ws.update_H(grid);  grid, cpml = ws.update_H_pml(grid, cpml)
    grid = ws.update_E(grid);  grid, cpml = ws.update_E_pml(grid, cpml)
    grid = ws.apply_pec_mask(grid)
    grid.Ez[100, 100, 0] += src(t)                        # soft injection
    ws.record_snapshot(snap, grid)
    grid.time_step += 1
```

Prefer to skip the loop? The same run via the Simulation orchestration layer:

```python
import wavesim as ws

sim  = ws.Simulation(grid, cpml=cpml)                     # backend='numba' for ~10× speed
sim.add_source(ws.PointSource('Ez', 50e-3, 50e-3, 0.0, ws.GaussianPulse.for_fmax(10e9)))
snap = sim.add_monitor(ws.SnapshotMonitor('Ez', at_z=0.0, every_N_steps=20))
sim.run(2000)                                             # bit-identical to the loop
```

For large 3D runs, pass `backend='numba'` to `Simulation` (requires `pip install
numba`) — multithreaded JIT kernels that are bit-identical to the NumPy default.
The stencil is memory-bandwidth-bound, so ~4–6 threads is the sweet spot
(`numba.set_num_threads(4)`).

Every module carries a thorough docstring covering the public API, the canonical
loop, and the conventions worth committing to memory — start with
`wavesim/__init__.py`, `wavesim/simulation.py`, and `wavesim/mode_solver.py`.

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
│   ├── sources.py       # waveforms + Source / Point / Line / Plane / Volume / Array
│   ├── mode_solver.py   # 2D TEM mode solver + TEMMode.to_source port launch
│   ├── monitors.py      # field / magnitude / snapshot / energy / voltage / current
│   ├── simulation.py    # Simulation class — runs the canonical loop (backend='numpy'|'numba')
│   ├── viz.py           # all plotting and animation (2D + full-3D helpers)
│   └── constants.py     # C0, EPS0, MU0, ETA0
└── docs/             # HOW_TO_SET_UP.md
```

---

## Documentation

- **[docs/HOW_TO_SET_UP.md](docs/HOW_TO_SET_UP.md)** — environment setup (conda
  + VS Code) and how to run a simulation.
- **Module docstrings** — each `wavesim/*.py` module documents its own public API
  with worked examples; `__init__.py` lists everything re-exported under `ws.`.

---