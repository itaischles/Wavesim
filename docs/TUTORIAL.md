# Building Simulations with Wavesim — A Hands-On Tutorial

This is a **build-along** tutorial. By the end you will have written, from a blank
file, a handful of complete FDTD scripts of your own and you will understand
*what* each line does, *why* it is there, and *how* it fits the whole. It exists
so you (and anyone new to the codebase) can take ownership of the engine instead
of copy-pasting from examples.

> **How this differs from the other docs.** [`API_GUIDE.md`](API_GUIDE.md) is the
> *reference* — every function, every argument, terse and exhaustive. This file is
> the *teacher* — fewer facts, more reasoning, built up in order. When you want
> "what are all the arguments to `init_cpml`?" go to the API guide. When you want
> "why is there a PML at all and how do I think about it?" stay here.
> [`HOW_TO_SET_UP.md`](HOW_TO_SET_UP.md) is how you get Python + the deps running.

**Prerequisites.** A working `wavesim` conda environment (see
[`HOW_TO_SET_UP.md`](HOW_TO_SET_UP.md)) and the ability to run `python yourscript.py`
from the repository root.

---

## Contents

- [0. The 60-second mental model](#0-the-60-second-mental-model)
- [1. Your first simulation — the hand-written loop](#1-your-first-simulation--the-hand-written-loop)
- [2. The same thing, the easy way — the `Simulation` class](#2-the-same-thing-the-easy-way--the-simulation-class)
- [3. Sources — where, when, and which field](#3-sources--where-when-and-which-field)
- [4. Materials & geometry — putting things in the box](#4-materials--geometry--putting-things-in-the-box)
- [5. Boundaries — PEC walls and PML absorbers](#5-boundaries--pec-walls-and-pml-absorbers)
- [6. Monitors — measuring what happened](#6-monitors--measuring-what-happened)
- [7. Visualisation — seeing what happened](#7-visualisation--seeing-what-happened)
- [8. Going fully 3D](#8-going-fully-3d)
- [9. Going fast — the Numba backend](#9-going-fast--the-numba-backend)
- [10. Capstone — build your own scattering experiment](#10-capstone--build-your-own-scattering-experiment)
- [Appendix: a setup checklist](#appendix-a-setup-checklist)

---

## 0. The 60-second mental model

FDTD ("Finite-Difference Time-Domain") solves Maxwell's equations by leap-frogging
the electric and magnetic fields forward in time on a staggered grid. Everything in
this engine flows through one pipeline:

```
  create_grid ──► set materials ──► place sources / boundaries / monitors
                                              │
                                              ▼
                          ┌──────── for each timestep n ────────┐
                          │  1. advance H   (+ PML correction)   │
                          │  2. advance E   (+ PML correction)   │
                          │  3. enforce PEC (walls + conductors) │
                          │  4. inject source                    │
                          │  5. record monitors                  │
                          │  6. time_step += 1                   │
                          └──────────────────────────────────────┘
                                              │
                                              ▼
                                  plot / animate the results
```

Two facts make the rest of the tutorial make sense:

**(a) The state lives in one object, `FDTDGrid`.** It holds the six field arrays
(`Ex, Ey, Ez, Hx, Hy, Hz`), the material arrays (`eps_*`, `mu_*`), the cell spacing
(`dx, dy, dz`), the timestep (`dt`), and a step counter (`time_step`). Every array
has shape `(Nx, Ny, Nz)` — **always 3D**, even for a 2D simulation (you just set
`Nz = 1`). The functions in the engine take a grid and mutate its arrays in place.

**(b) E and H are staggered in space and time (the Yee cell).** They are not stored
at the same points; they are offset by half a cell. This is *why* FDTD is accurate
and stable, and it is *why* the update order is fixed. For a 2D `Nz=1` slice driven
by an `Ez` source, only three components are alive — `Ez`, `Hx`, `Hy` — laid out
like this:

```
        Hx          Hx          Hx           A 2D "TMz" Yee cell.
    ┌────┼──────┬────┼──────┬────┼────┐      Ez sits on the nodes;
    │    │      │    │      │    │    │      Hx/Hy sit on the edges
   Hy───Ez────Hy───Ez────Hy───Ez───Hy       between them, half a cell
    │    │      │    │      │    │    │       away.  Each H is curl-fed
    │    Hx     │    Hx     │    Hx   │       by its neighbouring Ez,
   Hy───Ez────Hy───Ez────Hy───Ez───Hy        and vice versa — that
    │    │      │    │      │    │    │        staggering is the whole
    └────┴──────┴────┴──────┴────┴────┘        trick.
```

You never have to manage the staggering yourself — `update_H` and `update_E` do it.
But knowing it exists explains the rules you'll meet: *soft* injection, the fixed
H-then-E order, and the "(N−1)" cavity rule.

That's the whole model. Now let's build.

---

## 1. Your first simulation — the hand-written loop

We will simulate a **pulse expanding in empty space**, absorbed at the edges so it
doesn't bounce back. We write the time loop by hand first — not because you'll
always do it that way, but because the loop *is* the mental model, and the
`Simulation` class in §2 will mean nothing until you've typed it once.

Create a file `scripts/sim01_freespace.py` (make the `scripts/` folder if needed).
We'll build it in pieces; the complete file is assembled at the end of this section.

### 1.1 Imports and the grid

```python
import numpy as np
from wavesim.grid import create_grid
from wavesim.materials import set_vacuum
from wavesim.update import update_H, update_E
from wavesim.pml import init_cpml, update_H_pml, update_E_pml
from wavesim.pec import apply_pec_mask
from wavesim.sources import GaussianPulse
from wavesim.monitors import SnapshotMonitor, record_snapshot

grid = create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)
grid = set_vacuum(grid)
```

**What:** `create_grid` allocates a 200×200×1 grid of 0.5 mm cells (a 100 mm × 100 mm
2D domain). `set_vacuum` fills every cell with `eps_r = mu_r = 1`.

**Why this order:** the grid is your blank canvas; `set_vacuum` paints it before
you place any geometry. Always set materials before sources/boundaries.

**How — the arguments that matter:**

- `Nx, Ny, Nz` — cell counts. `Nz=1` makes this a 2D slice (see §0).
- `dx` — cell size in **metres** (this engine is SI throughout; `0.5e-3` is 0.5 mm).
  `dy`/`dz` default to `dx`, giving cubic cells. Smaller cells = finer detail but
  more cells and a smaller `dt`.
- **You do not set `dt`.** `create_grid` computes it from the CFL stability condition
  (`CFL = 0.99`). Setting `dt` yourself risks the simulation blowing up (energy → ∞,
  fields → `NaN`). Print it if you're curious: `print(grid.dt)`.

> **Rule of thumb for `dx`:** you want roughly 10–20 cells per wavelength at your
> highest frequency of interest. At 10 GHz in vacuum, λ = c/f ≈ 30 mm, so
> 0.5–3 mm cells are sensible.

### 1.2 Absorbing boundary (CPML)

```python
cpml = init_cpml(grid, d_pml=10)        # absorb on all four faces
```

**What:** a *Convolutional Perfectly Matched Layer* — a band of artificially lossy
cells around the domain edge that swallows outgoing waves with almost no reflection.

**Why:** without it, the domain edge acts like a mirror and your "free space" fills
up with reflections. CPML makes a finite box behave like open space.

**How:** `d_pml=10` is the layer thickness in cells (8–12 is the usual range; thicker
= less reflection but more memory). By default it wraps **all** faces. You can pick a
subset — crucial when some faces are real walls (we'll do that in §5).

```
        x0 ┌───────────────────────┐ x1      Face names. 'x0' is the face
   y1      │░░░░░ PML border ░░░░░░░│         at i=0, 'x1' at i=Nx-1, and
      ┌────┼───────────────────────┼────┐    likewise y0/y1, z0/z1.  For
      │░░░░│                       │░░░░│     Nz=1 the z-faces are inert.
      │░░░░│   interior (vacuum)   │░░░░│
      │░░░░│                       │░░░░│
      └────┼───────────────────────┼────┘
   y0      │░░░░░░░░░░░░░░░░░░░░░░░░░│
          └───────────────────────┘
```

### 1.3 The source waveform

```python
src = GaussianPulse.for_fmax(10e9)      # a Gaussian pulse with energy up to ~10 GHz
```

**What:** a time waveform — the shape of the "kick" we inject each step. A
`GaussianPulse` is a smooth bump in time; its Fourier transform is a smooth bump in
frequency, so it excites a band of frequencies up to roughly `f_max`.

**Why `GaussianPulse.for_fmax`:** it picks the pulse `width` and centre time `t0` for
you so the pulse (i) carries energy up to `f_max` and (ii) starts from ~zero at
`t=0` (a pulse that's already half-on at `t=0` injects a nasty step). You *can*
build `GaussianPulse(t0=..., width=...)` by hand, but let the helper do it.

A waveform is just **any callable `f(t) -> float`**. For a single-frequency
(narrowband) experiment you supply your own carrier-modulated lambda — we do that
in §3.

### 1.4 A monitor to capture frames

```python
snap = SnapshotMonitor(component='Ez', k_slice=0, interval=20)
```

**What:** records a 2D `(Nx, Ny)` picture of `Ez` every 20 steps, into `snap.snapshots`.
`k_slice=0` is the only z-plane we have (`Nz=1`).

**Why:** the fields are gone the instant the next step overwrites them; a monitor is
how you keep anything. `interval=20` keeps memory and animation length sane.

### 1.5 The loop — *the order is sacred*

```python
N_STEPS = 2000
i_src, j_src = 100, 100      # inject at the centre

for n in range(N_STEPS):
    t = n * grid.dt

    grid = update_H(grid)                       # 1. advance H (full curl)
    grid, cpml = update_H_pml(grid, cpml)       # 2. PML correction for H

    grid = update_E(grid)                       # 3. advance E (full curl)
    grid, cpml = update_E_pml(grid, cpml)       # 4. PML correction for E

    grid = apply_pec_mask(grid)                 # 5. enforce conductors (no-op here)

    grid.Ez[i_src, j_src, 0] += src(t)                   # 6. SOFT injection (+=)

    record_snapshot(snap, grid)                 # 7. record
    grid.time_step += 1                         # 8. advance the clock
```

This is the canonical loop. Three things will bite you if you ignore them:

**(1) Order is fixed: H → H-PML → E → E-PML → PEC → source → record → tick.**
This isn't style; it's the leapfrog. H is advanced using the *current* E, then E is
advanced using the *just-updated* H. Reordering corrupts the half-step relationship:

```
   time:  n            n+½           n+1          n+1½
          E(n) ───────► H(n+½) ─────► E(n+1) ────► H(n+1½)
                 uses          uses          uses
                 E(n)          H(n+½)         E(n+1)
```

**(2) Inject with `+=`, never `=`.** A soft source (`+=`) adds energy and lets waves
pass through it. A hard assignment (`=`) pins the cell to a value — which is exactly
what a metal sheet does, so it reflects everything. Always `+=`.

**(3) `grid.time_step += 1` every step, no exceptions.** Monitors read `time_step`
to timestamp their data and to decide when to snapshot. Forget it and your time axis
is all zeros and your snapshot list is empty.

The PML calls (steps 2 and 4) and the PEC call (step 5) are optional in general —
here PEC is a harmless no-op because we placed no conductors, and we keep it only so
the loop matches the canonical shape. (In a closed cavity you'd drop the PML calls
entirely — see §5.)

### 1.6 Look at it

```python
import matplotlib
matplotlib.use('Agg')                # save to file, no popup window
import matplotlib.pyplot as plt
from wavesim.viz import plot_field_snapshot, animate_snapshots

# a single mid-run frame
mid = len(snap.snapshots) // 2
plot_field_snapshot(snap.snapshots[mid], grid, snap.snap_times[mid], component='Ez')
plt.savefig('sim01_frame.png', dpi=120)

# the whole movie
anim = animate_snapshots(snap, grid, interval_ms=40)
anim.save('sim01.gif', writer='pillow', fps=25)
print('wrote sim01_frame.png and sim01.gif')
```

Run it:

```bash
python scripts/sim01_freespace.py
```

You should see a ring expanding from the centre and vanishing into the edges with no
visible bounce-back. **That clean disappearance is the PML working.** If the ring
reflects off the walls, your PML is missing or the `*_pml` calls are out of order.

<details>
<summary><b>Complete <code>sim01_freespace.py</code></b></summary>

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavesim.grid import create_grid
from wavesim.materials import set_vacuum
from wavesim.update import update_H, update_E
from wavesim.pml import init_cpml, update_H_pml, update_E_pml
from wavesim.pec import apply_pec_mask
from wavesim.sources import GaussianPulse
from wavesim.monitors import SnapshotMonitor, record_snapshot
from wavesim.viz import plot_field_snapshot, animate_snapshots

grid = create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)
grid = set_vacuum(grid)
cpml = init_cpml(grid, d_pml=10)

src  = GaussianPulse.for_fmax(10e9)
snap = SnapshotMonitor(component='Ez', k_slice=0, interval=20)

N_STEPS = 2000
i_src, j_src = 100, 100
for n in range(N_STEPS):
    t = n * grid.dt
    grid = update_H(grid);  grid, cpml = update_H_pml(grid, cpml)
    grid = update_E(grid);  grid, cpml = update_E_pml(grid, cpml)
    grid = apply_pec_mask(grid)
    grid.Ez[i_src, j_src, 0] += src(t)
    record_snapshot(snap, grid)
    grid.time_step += 1

mid = len(snap.snapshots) // 2
plot_field_snapshot(snap.snapshots[mid], grid, snap.snap_times[mid], component='Ez')
plt.savefig('sim01_frame.png', dpi=120)
anim = animate_snapshots(snap, grid, interval_ms=40)
anim.save('sim01.gif', writer='pillow', fps=25)
print('wrote sim01_frame.png and sim01.gif')
```
</details>

---

## 2. The same thing, the easy way — the `Simulation` class

Now that you've typed the loop, you never have to again. The `Simulation` class runs
**exactly** those eight steps in the same fixed order — it's pure orchestration over
the same functions, so the results are bit-for-bit identical. Here is §1 rewritten:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavesim.grid import create_grid
from wavesim.materials import set_vacuum
from wavesim.pml import init_cpml
from wavesim.sources import PointSource, GaussianPulse
from wavesim.monitors import SnapshotMonitor
from wavesim.simulation import Simulation
from wavesim.viz import animate_snapshots

grid = create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)
grid = set_vacuum(grid)
cpml = init_cpml(grid, d_pml=10)

sim  = Simulation(grid, cpml=cpml)
sim.add_source(PointSource('Ez', 100, 100, 0, GaussianPulse.for_fmax(10e9)))
snap = sim.add_monitor(SnapshotMonitor('Ez', k_slice=0, interval=20))

sim.run(2000, verbose=1)             # verbose=1 prints a live progress line

anim = animate_snapshots(snap, grid, interval_ms=40)
anim.save('sim02.gif', writer='pillow', fps=25)
```

**What changed:** the hand loop and the bare `grid.Ez[...] += ...` injection are
gone, replaced by a `Simulation`, a `PointSource`, and `sim.run(...)`.

**Why use it:** less boilerplate, no chance of mis-ordering the loop, and you get
`verbose=1` progress (`step n/N | steps/s | sim-time | ETA`) for free. The trade is
slightly less moment-to-moment control.

**How — the constructor options (every one):**

```python
Simulation(grid, cpml=None, sources=(), monitors=(), pec_faces=(), backend='numpy')
```

| argument | meaning | when to change it |
|---|---|---|
| `grid` | the state object, already given materials | always required |
| `cpml` | from `init_cpml`, or `None` | `None` for a closed lossless cavity (skips the PML steps) |
| `sources` | iterable of `Source` | or add later with `add_source` |
| `monitors` | iterable of monitors | or add later with `add_monitor` |
| `pec_faces` | tuple like `('y0','y1')` | the domain faces that are metal walls |
| `backend` | `'numpy'` or `'numba'` | `'numba'` for large 3D runs (§9) |

And the methods:

- `sim.add_source(src)` — register a source; returns it.
- `sim.add_monitor(mon)` — register a monitor; **returns it**, so capture the return
  value (`snap = sim.add_monitor(...)`) to read `.snapshots`/`.values` after the run.
- `sim.step()` — advance exactly one timestep (the loop body).
- `sim.run(n_steps, callback=None, verbose=0)` — run many. `verbose=1` for the live
  status line. `callback(sim, n)` runs after each step — use it for custom logic
  without unrolling the loop, e.g. printing energy or stopping early:

  ```python
  def on_step(sim, n):
      if n % 500 == 0:
          print(f"  step {n}: peak |Ez| = {abs(sim.grid.Ez).max():.3e}")
  sim.run(2000, callback=on_step)
  ```

The whole rest of the tutorial uses `Simulation` because it's less to read. Anything
shown with `Simulation` can be done in a hand loop — and vice versa.

---

## 3. Sources — where, when, and which field

A source answers three questions: **which** field component it drives, **where** in
space, and **when** in time (the waveform). The waveform and the placement are
separate, composable pieces.

### 3.1 The waveform (the "when")

Two built-in options plus "roll your own":

**Broadband Gaussian** (one pulse, many frequencies) — use this to ring a cavity, see
a whole spectrum, or watch a wavefront:

```python
from wavesim.sources import GaussianPulse

wf = GaussianPulse.for_fmax(10e9)                   # auto width & t0 up to 10 GHz
wf = GaussianPulse(t0=0.5e-9, width=0.05e-9)        # or set them by hand
```

`GaussianPulse` is **callable** — `wf(t)` gives the value — so it plugs straight into
any source object.

**Narrowband / single-frequency** (a tone) — use this to excite one waveguide mode or
test behaviour at one frequency. There's no built-in for it; you multiply a carrier
by a Gaussian envelope yourself, which is genuinely just a function:

```python
import numpy as np
f0, tau, t0 = 9e9, 6/9e9, 3.5*(6/9e9)
cw = lambda t: np.sin(2*np.pi*f0*(t - t0)) * np.exp(-0.5*((t - t0)/tau)**2)
```

**Why the distinction matters:** a bare `GaussianPulse` is a *baseband* pulse
centred at DC. If you wanted "9 GHz only" and used a bare Gaussian, you'd inject a
huge band from DC upward and your result would be a smear. Multiplying by
`sin(2π f0 t)` shifts the energy up to `f0`.

A waveform is just a callable, so you can sample it directly to inspect it before
committing to a long run:

```python
import numpy as np
t = np.arange(2000) * grid.dt
values = np.array([wf(ti) for ti in t])             # sample the pulse to plot / check
```

### 3.2 The placement (the "where" + "which field")

```python
from wavesim.sources import PointSource, ArraySource
```

**`PointSource(component, i, j, k, waveform)`** — a single cell. The object form of
`grid.<component>[i,j,k] += waveform(t)`:

```python
pt = PointSource('Ez', 100, 100, 0, wf)
```

**`ArraySource(profiles, waveform)`** — a *distributed* drive. You supply a
`{component: (Nx, Ny, Nz)}` mapping of weights (or a single `(component, array)`
pair); each step adds `waveform(t) * weights` to every named component. This is how
you make line sources, planar sources, shaped beams, modal or multi-component drives:

```python
import numpy as np
profile = np.zeros((grid.Nx, grid.Ny, grid.Nz))
profile[20, :, 0] = 1.0                  # a transverse LINE at i=20 (a waveguide feed)
line = ArraySource({'Ez': profile}, cw)
```

```
   ArraySource profile examples (the nonzero cells = where energy is added):

   point             line              plane (3D)        shaped / modal
   ┌─────────┐      ┌─────────┐       ┌─────────┐        ┌─────────┐
   │         │      │ █       │       │█████████│        │  ▁▃▅▃▁   │
   │    •    │      │ █       │       │ (a whole │       │ ▃▅███▅▃  │
   │         │      │ █       │       │  z-plane)│       │  ▁▃▅▃▁   │
   └─────────┘      └─────────┘       └─────────┘        └─────────┘
   PointSource      profile[20,:,0]   profile[:,:,k]=1   profile = mode shape
```

**Which component?** `'Ez'` for a 2D TMz slice is the natural choice (it's the live E
component). In full 3D you might drive `'Ex'`/`'Ey'`/`'Ez'` to set polarisation, or
even an H component. Each key must name a real field array on the grid — and an
`ArraySource` can name **several at once** (e.g. `{'Ex': wx, 'Ey': wy}`) to drive a
vector mode like a coaxial TEM feed.

### 3.3 Fully custom sources

If neither point nor array fits, subclass `Source` and implement
`spatial_profiles(grid)` — a `{component: weights}` mapping computed from the grid.
The base class caches the profiles, multiplies by `waveform(t)`, and does the soft
add for you. You rarely need this — `ArraySource` covers most "shaped" cases — but
it's there when you want the geometry computed from the grid itself.

---

## 4. Materials & geometry — putting things in the box

So far the box has been empty vacuum. Real experiments have dielectrics and metal.
**The workflow is always: `set_vacuum` first (clean slate), then stamp geometry on
top.** Materials are stored as *relative* `eps_r` / `mu_r` (vacuum = 1.0); the physical
constants are applied inside the solver, so you never multiply by `EPS0` yourself.

```python
from wavesim.materials import set_vacuum, set_box, set_cylinder, set_coax

grid = set_vacuum(grid)                                       # 1. clean slate
grid = set_box(grid, 0.03, 0.07, 0.03, 0.05, 0, grid.dz, eps_r=4.0)   # 2. dielectric slab
grid = set_cylinder(grid, 0.05, 0.04, 0.005, 0, grid.dz, eps_r=1, pec=True)  # 3. metal rod
```

**The geometry helpers (every one):**

| helper | shape | key arguments | `pec=True`? |
|---|---|---|---|
| `set_box(grid, x0,x1, y0,y1, z0,z1, eps_r, mu_r=1, pec=False)` | axis-aligned box | corners in **metres** | yes — marks a solid conductor |
| `set_cylinder(grid, cx,cy, radius, z0,z1, eps_r, mu_r=1, pec=False)` | z-aligned cylinder | centre + radius in metres | yes |
| `set_coax(grid, cx,cy, r_inner, r_outer, eps_r_fill=1)` | coaxial cross-section | inner PEC, dielectric annulus, outer PEC wall | built-in |
| `set_material_arrays(grid, eps_x..mu_z, pec_mask=None)` | anything | pre-computed `(Nx,Ny,Nz)` arrays | via `pec_mask` |

**The `eps_r` vs `pec` distinction is the important one:**

- A **dielectric** (`eps_r=4.0`, `pec=False`) slows and bends waves but lets them
  through. It changes the `eps_*` arrays.
- A **conductor** (`pec=True`) is a perfect mirror. It doesn't touch `eps_r` at all;
  instead it sets cells in `grid.pec_mask`, and `apply_pec_mask` zeroes the E-field
  there every step. `eps_r`/`mu_r` are ignored when `pec=True`.

```
   set_box(..., eps_r=4.0)            set_cylinder(..., pec=True)
   ┌───────────────────────┐          ┌───────────────────────┐
   │                       │          │                       │
   │      ▓▓▓▓▓▓▓▓▓        │          │          ███          │
   │      ▓ eps_r=4 ▓      │          │        ███████        │   ███ = pec_mask
   │      ▓▓▓▓▓▓▓▓▓        │          │          ███          │   (perfect mirror)
   │   (wave bends/slows)  │          │   (wave scatters off) │
   └───────────────────────┘          └───────────────────────┘
```

**Always confirm your geometry before a long run** — it's the cheapest bug to catch:

```python
from wavesim.viz import plot_materials_xy
plot_materials_xy(grid, component='eps_z', cpml=cpml)   # colour map + PEC hatching
```

`set_material_arrays` is the escape hatch: build the `eps_*`/`mu_*` arrays however you
like (a gradient-index lens, anisotropy via different `eps_x`/`eps_y`, a CSV import)
and assign them directly. It validates shapes against the grid.

---

## 5. Boundaries — PEC walls and PML absorbers

Every face of your box is *something*. The two questions are: **does it reflect (PEC)
or absorb (PML)?** Get this wrong and the physics is silently wrong.

### 5.1 PEC — perfect electric conductor (a mirror)

```python
from wavesim.pec import apply_pec_faces, apply_pec_mask
```

- `apply_pec_faces(grid, faces=('y0','y1'))` — make the named **domain faces** metal
  walls (zeroes tangential E there). This is how you build waveguide side walls and
  closed cavities.
- `apply_pec_mask(grid)` — zero E inside cells flagged by `grid.pec_mask` (the
  conductors you placed with `set_box(...pec=True)` etc.). It's a **no-op when there's
  no mask**, so it's always safe to call.

Both run **after** the E update (and its PML correction) — they overwrite E with the
conductor's boundary condition. With the `Simulation` class you don't call them by
hand; you pass `pec_faces=(...)` and the conductor mask is enforced automatically.

### 5.2 PML — the absorber (open space), revisited

We met `init_cpml(grid, d_pml=10)` in §1. The one option that matters now is
**`faces`** — *which* faces absorb:

```python
from wavesim.pml import init_cpml, ALL_FACES   # ALL_FACES = ('x0','x1','y0','y1','z0','z1')

cpml = init_cpml(grid, d_pml=10)                       # absorb everywhere (free space)
cpml = init_cpml(grid, d_pml=10, faces=('x0','x1'))    # absorb only the x-ends
```

**The golden rule: PML only on faces that should be open. Never put PML on a face
that is a PEC wall or a symmetry plane** — it will wrongly soak up the guided or
standing mode you're trying to measure.

### 5.3 The three canonical boundary recipes

This is where it all clicks. Three classic setups, each a different combination:

```
  (a) FREE SPACE                (b) CLOSED CAVITY            (c) WAVEGUIDE
      PML on all faces              PEC on all faces             PEC top/bottom,
                                    (no PML at all)              PML on the ends

   ░░░░░░░░░░░░░░░░░          ████████████████████        ████████████████████
   ░               ░         █                    █       █                    █
   ░     ((•))     ░         █     ((•))          █      ░░    →→→ wave →→→   ░░
   ░               ░         █  (rings forever)   █       ░░  (open ←  → open)░░
   ░░░░░░░░░░░░░░░░░          ████████████████████        ████████████████████
   ░ = PML (absorb)          █ = PEC (reflect)            mix: ██ walls, ░░ ends
```

**(a) Free space** — `cpml = init_cpml(grid, d_pml=10)`, no PEC. (Our §1/§2 sim.)

**(b) Closed cavity** — *no CPML at all*, PEC on all four faces. The pulse rings as a
sum of the cavity's eigenmodes forever (lossless). With `Simulation`:

```python
sim = Simulation(grid, cpml=None, pec_faces=('x0','x1','y0','y1'))
```

Note `cpml=None` — the PML steps are simply skipped. **The "(N−1)" rule lives here:**
PEC faces zero the field on the node planes `i=0` and `i=Nx-1`, so a standing wave
spans `(Nx−1)·dx`, *not* `Nx·dx`. Use the effective size for analytic checks:

```python
a_eff = (grid.Nx - 1) * grid.dx        # cavity width the fields actually see
```

Using `Nx·dx` gives a ~1% error — enough to fail a 1% tolerance test.

**(c) Waveguide** — PEC on the two side walls, PML on the two propagation-axis ends:

```python
cpml = init_cpml(grid, d_pml=10, faces=('x0','x1'))      # open ends only
sim  = Simulation(grid, cpml=cpml, pec_faces=('y0','y1'))  # side walls
```

If you put PML on `y0/y1` here, it would absorb the very mode bouncing between the
walls — the waveguide would "leak" and you'd measure nonsense. This is the single
most common boundary mistake.

---

## 6. Monitors — measuring what happened

Fields are overwritten every step; monitors are how you keep data. There are four,
each a small dataclass you create and register (with `sim.add_monitor`) or feed by
hand (`record_*` in the loop). **Capture the return of `add_monitor`** so you can read
the data afterwards.

```python
from wavesim.monitors import (FieldMonitor, MagnitudeMonitor,
                              SnapshotMonitor, EnergyMonitor)

ez   = sim.add_monitor(FieldMonitor('Ez', i=150, j=100, k=0))   # one component, one cell
mag  = sim.add_monitor(MagnitudeMonitor('E', i=150, j=100, k=0))# |E| at one cell
snap = sim.add_monitor(SnapshotMonitor('Ez', k_slice=0, interval=20))  # 2D frames
en   = sim.add_monitor(EnergyMonitor())                          # total energy each step
```

| monitor | records | stored in | use it for |
|---|---|---|---|
| `FieldMonitor(component, i,j,k)` | one component at one cell over time | `.times`, `.values` | time-trace → **FFT for spectra/resonances** |
| `MagnitudeMonitor(field, i,j,k)` | `\|E\|` or `\|H\|` at one cell | `.times`, `.values` | envelope / amplitude at a probe |
| `SnapshotMonitor(component, k_slice, interval)` | a 2D `(Nx,Ny)` slice every `interval` steps | `.snapshots`, `.snap_times` | movies and still frames |
| `EnergyMonitor()` | total `½Σ(ε\|E\|²+μ\|H\|²)·dV` | `.times`, `.values` | **stability/sanity check** |

**The `EnergyMonitor` is your smoke detector.** In a stable run energy should decay
(with PML) or stay bounded (lossless cavity). If it grows without bound, your `dt` is
wrong or the loop order is broken — stop and fix it before trusting anything else.

**`SnapshotMonitor.interval`** is a memory/time trade: it records only when
`time_step % interval == 0`. `interval=20` over 2000 steps = 100 frames, plenty for a
smooth GIF without hoarding gigabytes of arrays.

**A `FieldMonitor` + FFT** is how you turn a time-domain run into a spectrum (cavity
resonances, waveguide cutoff):

```python
import numpy as np
vals = np.array(ez.values);  dt = grid.dt
freqs = np.fft.rfftfreq(len(vals), dt)
spectrum = np.abs(np.fft.rfft(vals))
# peaks in `spectrum` are the cavity's resonant frequencies
```

---

## 7. Visualisation — seeing what happened

All plotting lives in `wavesim.viz`. Functions that draw return `(fig, ax)`;
animators return a Matplotlib `FuncAnimation`. **For file output (headless/scripts)
set the Agg backend first:**

```python
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wavesim.viz as viz
```

Two families: **setup checks** (run these *before* a long simulation) and **result
plots** (after).

### 7.1 Setup checks — catch mistakes before they cost you minutes

```python
viz.plot_grid_xy(grid, cpml=cpml)                 # Yee cell layout; shades the PML
viz.plot_materials_xy(grid, component='eps_z', cpml=cpml)  # geometry + PEC hatch
plt.savefig('setup.png', dpi=120)
```

If `plot_materials_xy` doesn't show your slab where you expect, you saved the run.
To check the pulse fits the window, sample the waveform yourself — `wf(0.0)` and
`wf(n_steps*grid.dt)` should both be a small fraction of the peak (a fat residual
means the pulse is clipped).

### 7.2 Result plots — 2D

```python
# a single frame from a SnapshotMonitor
viz.plot_field_snapshot(snap.snapshots[-1], grid, snap.snap_times[-1], component='Ez')
plt.savefig('frame.png', dpi=120)

# the movie
anim = viz.animate_snapshots(snap, grid, interval_ms=40)
anim.save('field.gif', writer='pillow', fps=25)

# time-series and energy
viz.plot_monitor_time_series(ez, grid.dt)         # FieldMonitor / MagnitudeMonitor
viz.plot_energy(en, grid.dt)                       # log-scale energy curve
plt.savefig('energy.png', dpi=120)
```

`plot_field_snapshot` uses a zero-centred diverging colour map (blue/red) so you can
see field polarity. `plot_energy` is log-scale precisely so a stability blow-up is
obvious as a rising line.

### 7.3 Result plots — 3D (preview for §8)

For `Nz>1` runs, a single XY slice isn't enough. Two helpers handle volumetric data
and accept either a component name (`'Ez'`) or a raw `(Nx,Ny,Nz)` array:

```python
viz.plot_field_slices_3d('Ez', grid)              # XY / XZ / YZ triptych through centre
# or an oriented multi-panel animation — see §8
```

---

## 8. Going fully 3D

Here is the engine's best feature: **the same code runs in 3D.** The arrays were
always `(Nx, Ny, Nz)`; a 2D run just had `Nz=1`. Set `Nz>1` and you get a full
vector simulation — all six components, all three curl terms, z-faces that can carry
PML — *with no restructuring*.

```python
grid = create_grid(Nx=60, Ny=60, Nz=60, dx=1e-3)     # a real 3D cube
grid = set_vacuum(grid)
cpml = init_cpml(grid, d_pml=10, faces=ALL_FACES)     # z-faces now matter too

sim = Simulation(grid, cpml=cpml)
sim.add_source(PointSource('Ez', 30, 30, 30, GaussianPulse.for_fmax(20e9)))
snap = ...                                            # see below
sim.run(400, verbose=1)
```

**What's different in practice:**

- **Cost explodes.** Doubling resolution in 3D is ~8× the cells *and* a smaller `dt`,
  so ~16× the work. Start small (`60³`), and reach for the Numba backend (§9).
- **All six components are live.** Your source's component now sets polarisation.
- **z-faces are real.** Include `'z0','z1'` in `init_cpml` faces (or `ALL_FACES`).
- **Visualisation needs slices.** A 3D field can't be one image. Use
  `plot_field_slices_3d` for an XY/XZ/YZ triptych through a chosen `(i,j,k)`
  (defaults to the domain centre):

```python
viz.plot_field_slices_3d('Ez', grid, i=30, j=30, k=30)
plt.savefig('slices3d.png', dpi=120)
```

```
        plot_field_slices_3d  — three orthogonal cuts through one point
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │   XY     │   │   XZ     │   │   YZ     │     crosshairs on each panel
   │   ┼      │   │   ┼      │   │   ┼      │     mark where the other two
   │  (z=k)   │   │  (y=j)   │   │  (x=i)   │     planes slice through.
   └──────────┘   └──────────┘   └──────────┘
```

A coaxial TEM line and a rectangular PEC box cavity are natural full 3D examples
(including building animations with `animate_field_slices_3d`) — try one once you've
done your own small 3D run.

---

## 9. Going fast — the Numba backend

The NumPy solver is the readable reference. For large 3D grids it's also the slow
one. `Simulation` can swap in a Numba-JIT-compiled, multithreaded implementation of
the four hot update functions with a single argument:

```python
sim = Simulation(grid, cpml=cpml, backend='numba')   # ~10–12× faster on 3D sizes
```

**What you need to know:**

- It is **bit-for-bit identical** to `'numpy'` — same physics, just faster.
  PEC, sources, and monitors are backend-independent.
- The **first step pays a one-time JIT compile** (a few seconds). Don't benchmark step 1.
- It's only worth it for 3D / large grids. For a 2D `200²` slice, NumPy is fine and
  the compile overhead isn't worth paying.
- The stencil is **memory-bandwidth-bound**, so more threads ≠ faster past a point.
  ~4–6 threads is typically the sweet spot; using all cores can be *slower*:

  ```python
  import numba; numba.set_num_threads(4)
  ```

- Requires `pip install numba` (it's optional — see [`HOW_TO_SET_UP.md`](HOW_TO_SET_UP.md)).

To use Numba in a hand-written loop instead of `Simulation`, just import the four
functions from `wavesim.backend_numba` in place of `wavesim.update` / `wavesim.pml`.

---

## 10. Capstone — build your own scattering experiment

Time to take ownership. We'll combine everything into one script that you write:
**a plane-wave-like pulse scattering off a metal cylinder in free space.** This uses a
line `ArraySource`, a PEC `set_cylinder`, all-faces CPML, and both a snapshot and an
energy monitor. Try to write it from the pieces above *before* reading the solution.

The recipe:

1. `create_grid` — 240×180×1 at 0.5 mm.
2. `set_vacuum`, then `set_cylinder(..., pec=True)` a metal rod right of centre.
3. `init_cpml` on all faces (it's open free space).
4. An `ArraySource` driving a **vertical line** of `Ez` near the left edge — that
   launches a roughly planar wavefront travelling in +x.
5. A broadband `GaussianPulse.for_fmax` waveform.
6. A `SnapshotMonitor` and an `EnergyMonitor`.
7. `plot_materials_xy` to confirm geometry, then `sim.run`, then a GIF.

<details>
<summary><b>Solution — <code>sim10_scatter.py</code></b></summary>

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavesim.grid import create_grid
from wavesim.materials import set_vacuum, set_cylinder
from wavesim.pml import init_cpml
from wavesim.sources import ArraySource, GaussianPulse
from wavesim.monitors import SnapshotMonitor, EnergyMonitor
from wavesim.simulation import Simulation
import wavesim.viz as viz

# 1–2. grid + geometry
grid = create_grid(Nx=240, Ny=180, Nz=1, dx=0.5e-3)
grid = set_vacuum(grid)
grid = set_cylinder(grid, cx=0.075, cy=0.045, radius=0.012,
                    z0=0, z1=grid.dz, eps_r=1.0, pec=True)   # metal rod

# 3. open free space
cpml = init_cpml(grid, d_pml=12)

# 4–5. a transverse line source near the left edge -> a quasi-plane wave in +x
profile = np.zeros((grid.Nx, grid.Ny, grid.Nz))
profile[15, :, 0] = 1.0
wave = GaussianPulse.for_fmax(12e9)
plane = ArraySource({'Ez': profile}, wave)

# confirm geometry BEFORE running
viz.plot_materials_xy(grid, component='eps_z', cpml=cpml)
plt.savefig('sim10_geometry.png', dpi=120); plt.close()

# 6. assemble and run
sim  = Simulation(grid, cpml=cpml)
sim.add_source(plane)
snap = sim.add_monitor(SnapshotMonitor('Ez', k_slice=0, interval=15))
en   = sim.add_monitor(EnergyMonitor())
sim.run(1400, verbose=1)

# 7. results
anim = viz.animate_snapshots(snap, grid, interval_ms=40)
anim.save('sim10_scatter.gif', writer='pillow', fps=25)
viz.plot_energy(en, grid.dt); plt.savefig('sim10_energy.png', dpi=120)
print('done: sim10_geometry.png, sim10_scatter.gif, sim10_energy.png')
```

Watch the GIF: a flat wavefront marches in from the left, hits the rod, and throws a
circular scattered wave back and around it — the classic "shadow + ripple" pattern.
The energy curve rises while the source is on, then decays as everything drains into
the PML. If instead it *grows* late in the run, something is unstable.
</details>

When that runs and looks right, you own this engine. Change the rod to a dielectric
(`pec=False, eps_r=9.0`) and watch it become a lens. Add a second `FieldMonitor` in
the shadow and FFT it. Make it 3D. The pieces all compose.

---

## Appendix: a setup checklist

Before every `run`, walk this list — it catches almost every "wrong physics" bug:

- [ ] **`set_vacuum` called first**, before any geometry.
- [ ] **Never set `dt` by hand** — `create_grid` did it from the CFL condition.
- [ ] **Materials are relative** (`eps_r=4`, not `4·EPS0`).
- [ ] **Geometry in metres**; confirmed with `plot_materials_xy`.
- [ ] **Source is soft** (`+=` / `PointSource` / `ArraySource`, never `=`).
- [ ] **Narrowband?** multiply a carrier by the Gaussian envelope.
- [ ] **PML only on open faces**; PEC walls/symmetry planes excluded from `faces`.
- [ ] **Analytic comparison?** use the effective size `(N−1)·dx`, not `N·dx`.
- [ ] **`EnergyMonitor`** present and bounded/decaying (your stability smoke test).
- [ ] Hand loop only: **order fixed** (H→H-PML→E→E-PML→PEC→source→record) and
      **`time_step += 1`** every step.

For the exhaustive argument-by-argument reference, see [`API_GUIDE.md`](API_GUIDE.md).
For installation, [`HOW_TO_SET_UP.md`](HOW_TO_SET_UP.md).
```
