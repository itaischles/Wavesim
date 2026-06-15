# FDTD Engine — User API Guide (v1)

A practical guide to building and running 2D-in-3D FDTD electromagnetic
simulations with this engine. It documents every public function you need,
the canonical simulation loop, the conventions that bite if you get them
wrong, and three complete worked examples drawn from the validated test suite.

> **Scope.** This covers the current API: a functional NumPy solver on full
> 3D arrays — usually run as a thin `Nz=1` slice, but also full 3D (`Nz>1`, see
> `test_05_coax_tem.py`) — with CPML and PEC boundaries, Gaussian sources,
> time/snapshot/energy monitors, and visualisation helpers. It also documents the
> v2 [`Simulation`](#simulation) / [`Source`](#sources) orchestration layer,
> which runs the canonical loop for you on top of that same functional core.

---

## Contents

1. [Mental model](#1-mental-model)
2. [Setup & running](#2-setup--running)
3. [Quickstart](#3-quickstart)
4. [The simulation loop (canonical pattern)](#4-the-simulation-loop-canonical-pattern)
5. [Conventions you must know](#5-conventions-you-must-know)
6. [API reference](#6-api-reference)
   - [grid](#grid) · [materials](#materials) · [sources](#sources) ·
     [pml](#pml) · [pec](#pec) · [monitors](#monitors) ·
     [simulation](#simulation) · [viz](#viz) · [constants](#constants)
7. [Worked examples](#7-worked-examples)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Mental model

The engine is **functional**: there is one state object, `FDTDGrid`, and a set
of pure-ish functions that take it and return it (mutating its arrays in place).
*You* can write the time loop in your script — and understanding that loop is the
whole mental model. A thin **[`Simulation`](#simulation) class** (v2) can run the
exact same loop for you once you know what it does; it orchestrates these same
functions and changes no physics (see [§6 simulation](#simulation)).

```
create_grid ──► set materials ──► init boundaries / sources / monitors
                                          │
                                          ▼
                        ┌── for n in range(N_STEPS): ──┐
                        │   advance H, E (+ CPML)       │
                        │   enforce PEC                 │
                        │   inject source               │
                        │   record monitors             │
                        └───────────────────────────────┘
                                          │
                                          ▼
                              plot / animate results
```

Everything operates on full 3D arrays of shape `(Nx, Ny, Nz)`. Most v1 tests keep
`Nz = 1`; the third dimension is carried so the same code runs full 3D simply by
setting `Nz > 1` (no restructuring) — `test_05_coax_tem.py` does exactly that.
With `Nz=1` and an `Ez` source you are simulating the **TM_z** polarisation: the
live fields are `Ez`, `Hx`, `Hy` (all z-derivatives vanish automatically). With
`Nz > 1` all six components and all three curl terms are live, and the z-faces
can carry CPML (`init_cpml(..., faces=(...,'z0','z1'))`).

---

## 2. Setup & running

The engine needs **NumPy**, **Matplotlib**, and (for some analysis in the tests)
**SciPy**. Use the dedicated conda environment:

```bash
# the env that has the dependencies
C:\Users\itais\miniconda3\envs\wavesim\python.exe  your_script.py
```

On the Windows console, set UTF-8 first if your script prints non-ASCII glyphs:

```bash
set PYTHONIOENCODING=utf-8
```

Import from the `wavesim` package (run scripts from the repo root, or add it to
`sys.path` as the tests do):

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # if in tests/
```

---

## 3. Quickstart

A complete free-space pulse in ~20 lines:

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
    grid = apply_pec_mask(grid)                        # no-op without PEC
    grid.Ez[100, 100, 0] += gaussian_pulse(src, t)     # soft injection
    record_snapshot(snap, grid)
    grid.time_step += 1
```

`snap.snapshots` now holds a list of `(Nx, Ny)` `Ez` slices you can plot or
animate (see [viz](#viz)).

---

## 4. The simulation loop (canonical pattern)

**The order of operations is fixed. Do not reorder it.** Each step depends on
the previous one being applied to the correct half-update.

```python
for n in range(N_STEPS):
    t = n * grid.dt

    # 1. H update (interior, full curl)
    grid = update_H(grid)
    # 2. CPML H correction (only if using CPML)
    grid, cpml = update_H_pml(grid, cpml)

    # 3. E update (interior, full curl)
    grid = update_E(grid)
    # 4. CPML E correction
    grid, cpml = update_E_pml(grid, cpml)

    # 5. Enforce PEC — ALWAYS after the E update (+ CPML)
    grid = apply_pec_faces(grid, faces=('y0', 'y1'))   # omit if no PEC walls
    grid = apply_pec_mask(grid)                         # no-op if pec_mask is None

    # 6. Inject source (soft, additive)
    grid.Ez[i_src, j_src, 0] += gaussian_pulse(source, t)

    # 7. Record monitors
    record_field(fmon, grid)
    record_energy(emon, grid)
    record_snapshot(snap_mon, grid)

    # 8. Advance the step counter (monitors read it for their time axis)
    grid.time_step += 1
```

| # | Step | Skip when… |
|---|------|-----------|
| 1–2 | H update + CPML | never advance H; CPML optional (lossless cavity) |
| 3–4 | E update + CPML | CPML optional |
| 5 | PEC faces / mask | no PEC walls / no conductors |
| 6 | Source injection | — |
| 7 | Monitors | — |
| 8 | `time_step += 1` | **never** — monitors timestamp from it |

If you are **not** using CPML (e.g. a closed PEC cavity), simply drop steps 2
and 4 and don't create a `cpml` object.

> **Don't want to type this loop?** The [`Simulation`](#simulation) class runs
> exactly these eight steps for you in this fixed order — `Simulation(grid,
> cpml=...).run(n_steps)`. It is pure orchestration over the same functions
> (bit-identical results), so reach for it once you understand the loop above.

---

## 5. Conventions you must know

These are the things that silently produce wrong physics if you ignore them.

### Units are SI (metres, seconds)
All geometry and timing inputs are in **metres** and **seconds**. `dx=0.5e-3`
is 0.5 mm. `set_box(grid, 0.02, 0.04, ...)` is 20–40 mm. Plot helpers *display*
in mm/ns, but every API input is base SI.

### Materials are *relative* (`eps_r`, `mu_r`)
`grid.eps_*` and `grid.mu_*` store relative permittivity/permeability. Vacuum is
`1.0`. The physical constants `EPS0`/`MU0` are applied inside the update
functions — you never multiply them in yourself.

### `dt` is computed for you
`create_grid` sets `grid.dt` from the conservative 3D CFL condition
(`CFL = 0.99`). **Never set `dt` manually** — doing so risks instability
(energy blow-up) or wasted resolution.

### Soft injection only (`+=`, never `=`)
Add the source to the existing field value:
`grid.Ez[i, j, 0] += gaussian_pulse(...)`. A hard assignment (`=`) acts like a
PEC sheet and reflects every wave that reaches it.

### Effective domain size: the `(N−1)` rule
`apply_pec_faces` zeroes the field on the **node planes** `i=0` and `i=Nx-1`.
A standing wave therefore spans the distance *between* those planes,
`(Nx−1)·dx`, **not** `Nx·dx`. This matters whenever you compare against an
analytic resonance or cutoff:

```python
a_eff = (Nx - 1) * grid.dx        # cavity width seen by the fields
b_eff = (Ny - 1) * grid.dy
```

Using the nominal `Nx·dx` injects a ~1% error — enough to fail a 1% tolerance.
(See the cavity and waveguide examples.)

### `gaussian_pulse` is a *baseband* envelope
It returns `amplitude · exp(-½((t-t0)/width)²)` — a pulse centred at DC with
bandwidth `≈ 1/(2π·width)`. For a **single-frequency / narrowband** excitation
(e.g. testing above/below a waveguide cutoff) multiply by a carrier yourself:

```python
g = np.sin(2*np.pi*f0*(t - t0)) * np.exp(-0.5*((t - t0)/tau)**2)
grid.Ez[i_src, :, 0] += g
```

### CPML only on open faces
Put CPML on faces that should *absorb*. Faces that are PEC walls or symmetry
planes must be **excluded** from CPML, or they will wrongly absorb the guided
or standing mode. Select faces explicitly:

```python
cpml = init_cpml(grid, d_pml=10, faces=('x0', 'x1'))   # open ends only
```

### Monitors timestamp from `grid.time_step`
Every `record_*` call stores `grid.time_step * grid.dt` as the time. Increment
`grid.time_step` once per loop (step 8). `SnapshotMonitor` records a frame only
when `time_step % interval == 0`.

---

## 6. API reference

Import paths are `from wavesim.<module> import <name>`.

### grid

```python
@dataclass
class FDTDGrid:
    Ex, Ey, Ez, Hx, Hy, Hz          # field arrays, shape (Nx, Ny, Nz)
    eps_x, eps_y, eps_z             # relative permittivity per component
    mu_x,  mu_y,  mu_z              # relative permeability per component
    dx, dy, dz, dt                  # spacing (m) and timestep (s)
    Nx, Ny, Nz                      # cell counts
    pec_mask                        # bool array or None
    time_step = 0                   # integer step counter
```

```python
create_grid(Nx, Ny, Nz, dx, dy=None, dz=None) -> FDTDGrid
```
Allocate a grid with all fields zero and all materials vacuum (`eps_r=mu_r=1`).
`dy`/`dz` default to `dx` (cubic cells). `dt` is set automatically from the CFL
condition. **This is your starting point for every simulation.**

```python
grid = create_grid(Nx=100, Ny=80, Nz=1, dx=1e-3)
print(grid.dt)          # ~1.9 ps for 1 mm cells
```

Access fields/materials directly as attributes: `grid.Ez[i, j, k]`,
`grid.eps_z[...] = 4.0`, etc.

---

### materials

Always `set_vacuum` first, then place geometry.

```python
set_vacuum(grid) -> grid
```
Reset the entire domain to `eps_r = mu_r = 1`.

```python
set_box(grid, x0, x1, y0, y1, z0, z1, eps_r, mu_r=1.0, pec=False) -> grid
```
Fill an axis-aligned box (corners in **metres**, snapped to the nearest cell)
with a uniform material. With `pec=True` the region is marked in `grid.pec_mask`
instead (a solid conductor), and `eps_r`/`mu_r` are ignored.

```python
set_cylinder(grid, cx, cy, radius, z0, z1, eps_r, mu_r=1.0, pec=False) -> grid
```
Fill a Z-aligned cylinder. `cx, cy, radius` in **metres**. `pec=True` marks it
as a conductor.

```python
set_coax(grid, cx, cy, r_inner, r_outer, eps_r_fill=1.0) -> grid
```
Build a coaxial cross-section: inner conductor (PEC), dielectric fill
(`eps_r_fill`) in the annulus, and everything at `r ≥ r_outer` marked PEC as the
outer wall. Radii in **metres**.

```python
set_material_arrays(grid, eps_x, eps_y, eps_z, mu_x, mu_y, mu_z,
                    pec_mask=None) -> grid
```
Assign pre-computed `(Nx, Ny, Nz)` arrays directly (validated for shape). Use
this when you have built material distributions yourself.

```python
grid = set_vacuum(grid)
grid = set_box(grid, 0.03, 0.07, 0.03, 0.05, 0, grid.dz, eps_r=4.0)  # dielectric
grid = set_cylinder(grid, 0.05, 0.04, 0.005, 0, grid.dz, eps_r=1, pec=True)  # PEC rod
```

---

### sources

Two layers live here: **waveforms** (the time part of an excitation) and
**`Source` objects** (the v2 *where + when + which-component* injection
abstraction). They compose; you can use either on its own.

#### Waveforms (time part)

```python
@dataclass
class GaussianSource:
    t0: float          # pulse centre time (s)
    width: float       # std-dev (s); bandwidth ≈ 1/(2π·width)
    amplitude = 1.0
    def __call__(self, t) -> float        # == gaussian_pulse(self, t)
```
The built-in baseband pulse. It is **callable**, so an instance is itself a valid
waveform (`f(t) -> float`) and can be passed straight to a `Source`.

```python
gaussian_pulse(source, t) -> float
```
Evaluate the baseband Gaussian envelope at time `t`.

```python
make_source_for_fmax(f_max, amplitude=1.0) -> GaussianSource
```
Convenience constructor that picks `width = 1/(2π f_max)` and `t0 = 4·width` so
the pulse is fully contained in the window and carries energy up to ≈ `f_max`.

```python
src = make_source_for_fmax(5e9)                 # content up to ~5 GHz
grid.Ez[100, 100, 0] += gaussian_pulse(src, t)  # functional, in the loop
```

A waveform is **any** `callable(t) -> float`. For a narrowband/CW excitation,
pass your own carrier-modulated lambda (see
[Conventions](#5-conventions-you-must-know)):

```python
f0, tau, t0 = 9e9, 6/9e9, 3.5*(6/9e9)
cw = lambda t: np.sin(2*np.pi*f0*(t-t0)) * np.exp(-0.5*((t-t0)/tau)**2)
```

#### Source objects (v2 injection abstraction)

A `Source` bundles **which component** it drives, **where** (`spatial_profile`),
and **when** (`time_function`), and exposes `inject(grid, t)` — the soft
(additive) write the time loop calls. [`Simulation`](#simulation) injects every
registered source each step; you can also call `inject` from a hand-written loop.

```python
class Source(ABC):
    component: str                          # 'Ex'..'Hz'
    time_function(self, t) -> float         # abstract
    spatial_profile(self, grid) -> ndarray  # abstract; (Nx,Ny,Nz) weights
    inject(self, grid, t) -> None           # grid.<component> += time*profile
```
Base class. Subclass it for a fully custom excitation; the profile is built once
and cached. Two ready-made subclasses cover the common cases:

```python
PointSource(component, i, j, k, waveform)
```
Soft point excitation at one cell — the object form of
`grid.<component>[i,j,k] += waveform(t)`. (`waveform` is any `callable(t)->float`,
e.g. a `GaussianSource`.)

```python
ArraySource(component, profile, waveform)
```
Distributed excitation from a user-supplied `profile` array of shape
`(Nx, Ny, Nz)`: the step adds `waveform(t) * profile`. Use it for line, shaped,
annular or modal drives (a single nonzero z-plane gives a planar source, etc.).
The profile shape is validated against the grid.

```python
from wavesim.sources import PointSource, ArraySource, make_source_for_fmax

pt = PointSource('Ez', 100, 100, 0, make_source_for_fmax(10e9))

prof = np.zeros((Nx, Ny, Nz)); prof[20, :, 0] = 1.0      # transverse line
ln = ArraySource('Ez', prof, cw)                          # cw from above
```

---

### pml

CPML (convolutional PML) absorbing boundaries. Create one `CPMLArrays` object
once, then call the two correction functions inside the loop.

```python
ALL_FACES = ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')

init_cpml(grid, d_pml=10, faces=ALL_FACES) -> CPMLArrays
```
Allocate auxiliary arrays and precompute absorption profiles. `d_pml` is the
PML thickness in cells (8–12 recommended; thicker = lower reflection, more
memory). **`faces`** selects which boundaries absorb — pass a subset to leave
PEC-wall or symmetry faces transparent. `'x0'` is the face at `i=0`, `'x1'` at
`i=Nx-1`, etc. Unknown face names raise `ValueError`. (For `Nz=1`, the z-faces
are automatically inert.)

```python
update_H_pml(grid, cpml) -> (grid, cpml)
update_E_pml(grid, cpml) -> (grid, cpml)
```
Apply the CPML correction on top of `update_H` / `update_E`. **Call order is
fixed:** `update_H → update_H_pml → update_E → update_E_pml`. Both return the
updated `(grid, cpml)` tuple — reassign both.

```python
cpml = init_cpml(grid, d_pml=10)                       # absorb everywhere
cpml = init_cpml(grid, d_pml=10, faces=('x0', 'x1'))   # waveguide: open ends only
```

---

### pec

Perfect-electric-conductor enforcement. Both functions zero E components and run
**after** the E update (and CPML correction).

```python
apply_pec_faces(grid, faces=('x0', 'x1', 'y0', 'y1')) -> grid
```
Zero the tangential E components on the named domain faces — i.e. make those
walls perfect conductors. Subset of `('x0','x1','y0','y1','z0','z1')`.

```python
apply_pec_mask(grid) -> grid
```
Zero all E components in cells where `grid.pec_mask` is `True` (interior
conductors placed by `set_box`/`set_cylinder`/`set_coax`). A no-op when
`pec_mask` is `None`, so it is always safe to call.

```python
grid = apply_pec_faces(grid, faces=('y0', 'y1'))   # waveguide side walls
grid = apply_pec_mask(grid)                          # interior conductors
```

---

### monitors

Each monitor is a dataclass holding configuration plus accumulated data lists;
each `record_*` appends the current value and returns the monitor (it also
mutates in place, so the return value is optional).

```python
FieldMonitor(component, i, j, k)            # component: 'Ex'..'Hz'
record_field(monitor, grid) -> monitor      # -> monitor.times, monitor.values
```
Record one field component at a fixed cell.

```python
MagnitudeMonitor(field, i, j, k)            # field: 'E' or 'H'
record_magnitude(monitor, grid) -> monitor
```
Record `|E|` or `|H|` (vector magnitude) at a fixed cell.

```python
SnapshotMonitor(component, k_slice, interval)
record_snapshot(monitor, grid) -> monitor    # -> monitor.snapshots, .snap_times
```
Capture the 2D `(Nx, Ny)` slice of `component` at `k_slice`, every `interval`
steps (records when `time_step % interval == 0`). Stored as copies.

```python
EnergyMonitor()
record_energy(monitor, grid) -> monitor      # -> monitor.times, monitor.values
```
Total EM energy `U = ½ Σ(eps·|E|² + mu·|H|²)·dV`. In a stable run it must not
grow without bound; a steadily rising curve means a CFL/stability problem.

```python
fmon = FieldMonitor(component='Ez', i=150, j=100, k=0)
emon = EnergyMonitor()
snap = SnapshotMonitor(component='Ez', k_slice=0, interval=20)
# in the loop:
record_field(fmon, grid); record_energy(emon, grid); record_snapshot(snap, grid)
```

---

### simulation

A thin orchestration layer (v2) that runs the [canonical loop](#4-the-simulation-loop-canonical-pattern)
for you. It **only orchestrates** the existing pure functions — same physics,
bit-for-bit identical results (verified by `test_07_simulation_api.py`). Use it to
drop the per-script loop boilerplate; keep writing the loop by hand whenever you
want full control.

```python
Simulation(grid, cpml=None, sources=(), monitors=(), pec_faces=())
```
Wrap a grid and its components. `cpml=None` skips the CPML correction steps (for a
closed, lossless cavity). `pec_faces` (e.g. `('y0','y1')`) are held as PEC walls
each step; `apply_pec_mask` always runs too, so interior conductors from the
material helpers are enforced automatically.

```python
sim.add_source(source) -> source        # register a Source; returns it
sim.add_monitor(monitor) -> monitor      # any Field/Magnitude/Snapshot/Energy monitor
```
Build the simulation up incrementally; `add_monitor` returns the monitor so you
can read its `.values` / `.snapshots` after the run.

```python
sim.step() -> grid                        # one timestep (the canonical loop body)
sim.run(n_steps, callback=None) -> grid   # run n_steps; final grid in sim.grid
```
`run` executes the fixed order `update_H → update_H_pml → update_E → update_E_pml
→ apply_pec_faces → apply_pec_mask → sources.inject → monitors.record →
time_step += 1`. The optional `callback(sim, n)` runs after each step (progress,
custom logic). Source time is `grid.time_step * grid.dt`, identical to the
hand-written `t = n*dt`.

```python
from wavesim.simulation import Simulation
from wavesim.sources import PointSource, make_source_for_fmax
from wavesim.monitors import SnapshotMonitor

grid = create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)
grid = set_vacuum(grid)
cpml = init_cpml(grid, d_pml=10)

sim  = Simulation(grid, cpml=cpml)
sim.add_source(PointSource('Ez', 100, 100, 0, make_source_for_fmax(10e9)))
snap = sim.add_monitor(SnapshotMonitor('Ez', k_slice=0, interval=20))
sim.run(2000)                              # snap.snapshots holds the frames
```

A closed PEC cavity (no CPML, four PEC walls) is just:

```python
sim = Simulation(grid, cpml=None, pec_faces=('x0','x1','y0','y1'))
```

See `tests/test_07_simulation_api.py` for a fully commented tutorial.

---

### viz

All plotting lives here. Functions that draw a figure return `(fig, ax)`;
`animate_snapshots` returns a Matplotlib `FuncAnimation`. Use the non-interactive
backend (`matplotlib.use('Agg')`) when saving to file in a headless run.

```python
plot_grid_xy(grid, cpml=None, ax=None) -> (fig, ax)
```
Yee cell grid with staggered E/H marker positions; shades the PML region if
`cpml` is given.

```python
plot_materials_xy(grid, component='eps_z', cpml=None, ax=None) -> (fig, ax)
```
Colour map of a material array; overlays PML shading and hatches PEC cells.
`component` is one of `'eps_x'..'mu_z'`.

```python
plot_source_waveform(source, dt, n_steps, ax=None) -> (fig, ax)
```
Plot the Gaussian pulse over the run window; prints the estimated bandwidth and
the residual amplitude at both ends (to confirm the pulse fits).

```python
plot_field_snapshot(snapshot_array, grid, timestep, component='Ez', ax=None)
```
Render a single 2D snapshot with a zero-centred diverging colour map.

```python
animate_snapshots(snapshot_monitor, grid, interval_ms=50) -> FuncAnimation
anim.save('out.gif', writer='pillow', fps=20)
```
Animate a `SnapshotMonitor`'s frames.

```python
plot_monitor_time_series(monitor, dt, ax=None) -> (fig, ax)   # Field/Magnitude
plot_energy(monitor, dt, ax=None) -> (fig, ax)                # log-scale energy
```

```python
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plot_materials_xy(grid, component='eps_z', cpml=cpml)
plt.savefig('materials.png', dpi=120)
```

**Full-3D helpers.** The functions above show a single XY (`k`) slice. For an
`Nz>1` run use the orthogonal-slice triptych and the multi-plane animator. Both
accept either a component name (`'Ex'..'Hz'`, resolved against the grid) or a raw
`(Nx,Ny,Nz)` array — e.g. an `|E|` envelope or a temporal-DFT mode shape.

```python
plot_field_slices_3d(data, grid, component='', i=None, j=None, k=None,
                     cmap=None, symmetric=None, fig=None, axes=None) -> (fig, axes)
```
XY/XZ/YZ slices through `(i, j, k)` (default: domain centre), one shared colour
scale, crosshairs marking the other cuts. `cmap`/`symmetric` auto-pick a diverging
or sequential map from the data's sign. Pass `axes=(ax_xy, ax_xz, ax_yz)` to embed
the triptych in a larger figure (as Test 06 does under its spectrum panel).

```python
animate_field_slices_3d(panels, times=None, interval_ms=60) -> FuncAnimation
```
Animate one or more oriented 2D-plane sequences side by side. Each `panel` is a
dict: `frames` (list of pre-oriented 2D arrays), `extent` (mm), `xlabel`,
`ylabel`, `title`, `cmap`, `symmetric`, `aspect`, and optional `vlines`/`hlines`
markers. Generalises `animate_snapshots` to arbitrary orthogonal cuts; Tests 05
and 06 build their GIFs with it.

```python
# Inspect a 3D field component through the domain centre:
fig, axes = plot_field_slices_3d('Ez', grid)
plt.savefig('slices.png', dpi=120)
```

---

### constants

```python
from wavesim.constants import C0, EPS0, MU0, ETA0
# C0   = 299792458.0      speed of light (m/s)
# EPS0 = 8.8541878e-12    vacuum permittivity (F/m)
# MU0  = 1.2566370e-6     vacuum permeability (H/m)
# ETA0 = 376.730313       free-space impedance (Ω)
```

Use these for analytic comparisons (e.g. `f = C0/(2*b_eff)`).

---

## 7. Worked examples

Each corresponds to a validated test in `tests/`. Only the distinctive parts are
shown; the loop body follows [section 4](#4-the-simulation-loop-canonical-pattern).

### 7.1 Free-space pulse + absorbing boundaries (`test_02`)

CPML on all four faces; soft `Ez` point source; check the wavefront is absorbed
(energy decays, no reflections).

```python
grid = create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)
grid = set_vacuum(grid)
cpml = init_cpml(grid, d_pml=10)                       # all faces absorb
src  = make_source_for_fmax(10e9)
# loop: H, H_pml, E, E_pml, apply_pec_mask (no-op),
#       grid.Ez[100,100,0] += gaussian_pulse(src, t)
```

### 7.2 Closed PEC cavity resonance (`test_03`)

**No CPML** (the cavity is lossless); PEC on all four faces; broadband pulse
rings as a sum of eigenmodes; FFT the field monitors to read the resonances.

```python
grid = create_grid(Nx=100, Ny=80, Nz=1, dx=1e-3)
grid = set_vacuum(grid)
# NO init_cpml; drop the *_pml calls from the loop
# loop: update_H, update_E,
#       apply_pec_faces(grid, faces=('x0','x1','y0','y1')),
#       grid.Ez[23,17,0] += gaussian_pulse(src, t)

# analytic check uses the EFFECTIVE dimensions:
a_eff, b_eff = (100-1)*grid.dx, (80-1)*grid.dx
f_mn = 0.5*C0*np.sqrt((m/a_eff)**2 + (n/b_eff)**2)     # m,n >= 1
```

### 7.3 Rectangular waveguide dispersion (`test_04`)

PEC side walls at `y0/y1`; CPML on the propagation-axis faces **only**;
narrowband (modulated-Gaussian) source on a transverse line; below cutoff the
field is evanescent, above cutoff it propagates.

```python
grid = create_grid(Nx=200, Ny=50, Nz=1, dx=0.5e-3)
grid = set_vacuum(grid)
cpml = init_cpml(grid, d_pml=10, faces=('x0', 'x1'))   # y-faces are PEC walls

b_eff = (50-1)*grid.dx                                   # effective width
f_c   = C0 / (2*b_eff)                                   # ~6.12 GHz

f0, tau, t0 = 9e9, 6/9e9, 3.5*(6/9e9)                    # narrowband above cutoff
# in the loop, after the *_pml calls:
grid = apply_pec_faces(grid, faces=('y0', 'y1'))         # waveguide walls
g = np.sin(2*np.pi*f0*(t - t0)) * np.exp(-0.5*((t - t0)/tau)**2)
grid.Ez[20, :, 0] += g                                   # line source
```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Energy grows without bound; fields → NaN | `dt` set manually / CFL violated | Let `create_grid` set `dt`; never override it |
| Strong reflections off the domain edge | No CPML, or CPML correction omitted | `init_cpml` + call `update_H_pml`/`update_E_pml` in the right order |
| Guided/standing mode is damped near a wall | CPML active on a PEC-wall face | Exclude that face: `init_cpml(..., faces=(...))` |
| Source reflects waves back | Hard injection (`=`) | Use soft injection (`+=`) |
| Measured resonance/cutoff off by ~1% | Used nominal `N·d` instead of effective `(N−1)·d` | Use `(N-1)*dx` for the mode span |
| Single-frequency test has huge bandwidth | Used bare `gaussian_pulse` (baseband) | Multiply by a `sin(2π f0 t)` carrier |
| `SnapshotMonitor` is empty | `time_step` never incremented, or `interval` > run length | Increment `grid.time_step` each step; check `interval` |
| Monitor time axis is all zeros | `grid.time_step += 1` missing | Add step 8 of the loop |
| Wave won't propagate (waveguide) | Driving below cutoff | Drive above `f_c = c/(2·b_eff)`, or expect evanescence |

---

*This guide documents the v1 engine as built. The fixed loop order, SI/relative-
material conventions, the `(N−1)` effective-dimension rule, and per-face CPML
selection are the four things most worth committing to memory.*
