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
- **SPICE co-simulation** — `SpicePort` terminates a port with a full ngspice
  circuit (authored in LTspice or any tool), solved in lockstep with the FDTD
  loop. Bidirectional: the port presents its Thevenin equivalent to SPICE and
  injects the returned current each step. See [SPICE co-simulation](#spice-co-simulation-ngspice).
- **2D TEM mode solver** — finds the PEC conductor cross-sections on a grid plane
  and solves each supported TEM mode (ε-weighted electrostatic BVP), reporting
  per-unit-length C, L, Z₀, phase velocity and ε_eff. A solved mode launches
  straight into the run as a fixed directional source (`TEMMode.to_source`) or as
  a **`TEMPort`** — a circuit-driven, matched modal port: a Thevenin `(Vs, Z₀)` (or
  a `SpicePort(mode=…)` co-sim) drives the frozen mode profile, reading back the
  modal voltage by ε-weighted overlap projection and launching one-way by also
  driving the paired H sheet from the same port current.
- **Diagnostics** — point field, `|E|`/`|H|` magnitude, 2D snapshots (XY/XZ/YZ
  slice planes), line-integral voltage (∫E·dl) and current (∮H·dl) monitors, and
  total energy. Snapshots are collocated off the staggered Yee points onto cell
  centres and H is averaged onto the E timebase, so every component of a frame
  shares one coordinate grid and one instant (frames are one cell shorter than
  the grid per in-plane axis — see `SnapshotMonitor`).
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

## SPICE co-simulation (ngspice)

`SpicePort` terminates an FDTD port with an arbitrary ngspice circuit, solved in
lockstep with the time loop. It is a `LineSource` whose per-step circuit law is
replaced by a live ngspice solve: each step the port hands ngspice its Thevenin
equivalent (the time-centred port voltage `v_mid` behind the discrete
self-coupling `κ/2`) and injects the returned branch current. If the circuit
reduces to a Thevenin `(Vs, Z)` this is bit-for-bit identical to
`LineSource(voltage=Vs, impedance=Z)` — the equivalence is validated to ~1e-12,
and nonlinear terminations (diodes, transistors) run through the same path.

### Install ngspice + PySpice

Runtime is **ngspice** driven through **PySpice**. Author the netlist anywhere
(e.g. LTspice) — it is only the interchange format; LTspice itself cannot
lockstep.

```bash
pip install PySpice                      # into the wavesim env
```

PySpice needs the ngspice **shared library** (`ngspice.dll`), *not* the console
build. Download the **"ngspice as a DLL"** package (`ngspice-XX_dll_64.zip`) from
<https://ngspice.sourceforge.io/download.html> and unzip it; the library is at
`…\Spice64_dll\dll-vs\ngspice.dll`. Pass that path to `SpicePort(library_path=…)`
(wavesim then auto-derives `SPICE_LIB_DIR` from the package layout), or set the
`NGSPICE_LIBRARY_PATH` / `SPICE_LIB_DIR` environment variables yourself.

> PySpice's bundled `pyspice-post-installation --install-ngspice-dll` also
> fetches the DLL, but it downloads from SourceForge over TLS — behind a strict
> proxy the manual download above is more reliable.

### Netlist authoring contract

- **Name two port nodes** where the FDTD structure connects (e.g. `port1p`,
  `port1n`); pass them as `nodes=(plus, minus)`. `plus` is the FDTD `+` terminal
  (`p0`). wavesim splices the Thevenin companion across them — you place *no*
  port component yourself.
- **Provide a ground.** SPICE needs a DC path to node `0`; the simplest port ties
  the minus terminal to ground (`nodes=("port1p", "0")`).
- **Standard primitives only.** R/L/C, independent + controlled sources,
  diodes/BJT/MOSFET with explicit `.model`/`.subckt` cards are portable to
  ngspice; avoid LTspice-only library parts and behavioural `A`-devices.
- **No analysis cards.** wavesim owns the transient — any `.tran`/`.ac`/`.op`/
  `.probe`/`.end` in your file is stripped; `.model`/`.subckt`/`.include`/`.param`
  are kept.

### Example

```python
import wavesim as ws

grid = ws.set_vacuum(ws.create_grid(Nx=140, Ny=80, Nz=1, dx=0.5e-3))
cpml = ws.init_cpml(grid, d_pml=10)
sim  = ws.Simulation(grid, cpml=cpml)

# launch a pulse down a parallel-plate line …
sim.add_source(ws.LineSource(p0=(15e-3, 15e-3, 0.0), p1=(15e-3, 25e-3, 0.0),
                             voltage=ws.GaussianPulse.for_fmax(20e9), impedance=50.0))

# … terminated by a SPICE circuit (driver.net names nodes port1p / 0).
port = ws.SpicePort(p0=(50e-3, 15e-3, 0.0), p1=(50e-3, 25e-3, 0.0),
                    netlist="driver.net", nodes=("port1p", "0"),
                    library_path=r"C:\ngspice\Spice64_dll\dll-vs\ngspice.dll")
sim.add_source(port)
sim.run(2000)

# port.times / port.voltages / port.currents are the co-simulated port record.
```

To drive a **distributed TEM mode** rather than a single line, solve the mode and
pass it as `mode=` (put the matched source resistance `Z₀` in the netlist):

```python
mode = ws.solve_tem_modes(grid, normal='z', position=30e-3)[0]
port = ws.SpicePort(mode=mode, nodes=("port1p", "0"), netlist="driver.net",
                    library_path=r"C:\ngspice\Spice64_dll\dll-vs\ngspice.dll")
# analytic equivalent, no SPICE:  ws.TEMPort(mode=mode, voltage=Vs)  # impedance defaults to Z₀
```

The port reads the modal voltage by an ε-weighted overlap projection (rejecting
non-modal content), presents `Z₀` to the field (pre-compensated by the discrete
`κ/2`), and — with `directional=True` (default) — drives the paired H sheet from
the same port current for a one-way launch. `κ/2` must stay below `Z₀`; a
low-impedance mode on a coarse transverse grid needs a finer cross-section.

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
│   ├── sources.py       # waveforms + Source / Point / Line / Plane / Volume / Array / SpicePort
│   ├── spice.py         # ngspice co-simulation coupler (SpicePort backend, PySpice)
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