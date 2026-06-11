# FDTD Engine v1 — Tactical Build Plan

## Purpose of this document

This is a detailed, step-by-step implementation plan for building a first working version of a 3D FDTD electromagnetic solver in Python. The immediate target is a validated 2D solver running as a thin 3D slice (`Nz=1`). All design decisions have been made deliberately to support a clean upgrade path to full 3D later.

This document is intended as input context for AI-assisted coding sessions. Read it in full before writing any code.

---

## Agreed design decisions (non-negotiable)

| Topic | Decision |
|---|---|
| Language | Python + NumPy |
| Code style | Functional — pure functions operating on a state dataclass |
| Dimensionality | Full 3D arrays `(Nx, Ny, Nz)` from day one; `Nz=1` for all v1 work |
| Field update operators | Full 3D curl — no 2D-specific operators |
| PML | CPML (Convolutional PML), Roden-Gedney parameters |
| Boundaries | PEC + CPML only for v1 |
| PEC body | Boolean `pec_mask` array in `FDTDGrid`; zeroed after every E update via `apply_pec_mask()` |
| Sources | Soft additive injection, Gaussian pulse time function hard-coded in tests; `gaussian_pulse(t)` reusable callable in `sources.py` |
| Diagnostics | Field monitors (time series), \|E\| and \|H\| magnitude monitors, snapshots, energy monitor |
| No v1 scope | DFT monitors, flux monitors, S-parameters, far-field, symmetry planes |
| Simulation loop | Lives in examples/tests — not inside any module |
| Future refactor | `Simulation` class (v2) wrapping grid + sources + monitors + `run()` method |
| Performance | NumPy sufficient for v1; JAX migration documented at 100³ cell threshold |
| Comment convention | Mark every place needing 3D attention with `# 3D-UPGRADE:` |

---

## Repository structure

```
fdtd/
├── grid.py          # FDTDGrid dataclass + construction helpers
├── materials.py     # eps/mu array builders; test scaffolding geometry primitives
├── update.py        # E and H field update functions (pure)
├── pml.py           # CPML auxiliary arrays, coefficients, update functions
├── sources.py       # gaussian_pulse() time function; soft injection helpers
├── pec.py           # PEC enforcement: domain faces and interior mask
├── monitors.py      # FieldMonitor, MagnitudeMonitor, SnapshotMonitor, EnergyMonitor
└── viz.py           # All plotting: grid, materials (with PML overlay), sources, fields

tests/
├── test_00_grid_viz.py       # Grid, material, and PML region visualisation
├── test_01_source_viz.py     # Source waveform visualisation
├── test_02_free_space.py     # Gaussian pulse in vacuum
├── test_03_pec_cavity.py     # PEC cavity resonance
├── test_04_waveguide.py      # Rectangular waveguide mode
└── test_05_coax_tem.py       # Coaxial TEM mode (vs MEEP reference)

examples/
└── coax_tem.py               # Standalone coaxial simulation script
```

All modules are flat collections of functions. No deep class hierarchies. No circular imports. The `FDTDGrid` dataclass is the only shared object.

---

## Module specifications

### `grid.py` — FDTDGrid dataclass

This is the central state object. Every function in every module takes it as input and returns a modified copy (or the same object for in-place operations where safe).

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FDTDGrid:
    # Field arrays — shape (Nx, Ny, Nz) always
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray
    Hz: np.ndarray

    # Material arrays — shape (Nx, Ny, Nz)
    # Store as separate x/y/z tensors for future anisotropic support
    eps_x: np.ndarray   # relative permittivity seen by Ex
    eps_y: np.ndarray
    eps_z: np.ndarray
    mu_x: np.ndarray    # relative permeability seen by Hx
    mu_y: np.ndarray
    mu_z: np.ndarray

    # Grid spacing
    dx: float
    dy: float
    dz: float           # # 3D-UPGRADE: set dz = dx for uniform 3D; for Nz=1 slices dz=dx is fine
    dt: float           # Computed from CFL condition, do not set manually

    # Domain size (derived, stored for convenience)
    Nx: int
    Ny: int
    Nz: int             # # 3D-UPGRADE: set Nz > 1 for full 3D

    # PEC body mask — shape (Nx, Ny, Nz), dtype bool
    # True = cell is a perfect electric conductor.
    # E components inside or on the surface of PEC cells are zeroed after every E update.
    # Set by pec.py geometry helpers or directly by the future CAD importer.
    pec_mask: np.ndarray = field(default_factory=lambda: None)

    # Simulation time
    time_step: int = 0
```

Construction helper:

```python
def create_grid(Nx, Ny, Nz, dx, dy, dz) -> FDTDGrid:
    """
    Allocate a grid with all fields and materials initialised to zero/vacuum.
    dt is set automatically from the 3D CFL condition:
        dt = CFL / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
    with CFL = 0.99 (conservative).
    # 3D-UPGRADE: CFL formula is already correct for full 3D — no changes needed.
    """
```

**Key rule:** `dt` is always computed from the full 3D CFL condition, even for `Nz=1`. This ensures correctness when `Nz` is later increased without touching `grid.py`.

---

### `materials.py` — Material array builders

This module has two clearly separated roles:

**Production path** (called by the future FreeCAD CAD importer and by any test):

```python
def set_vacuum(grid) -> FDTDGrid:
    """Set entire domain to eps_r=1, mu_r=1 (vacuum). Always call first."""

def set_material_arrays(grid, eps_x, eps_y, eps_z, mu_x, mu_y, mu_z, pec_mask=None) -> FDTDGrid:
    """
    Directly assign pre-computed material arrays to the grid.
    This is the function the future CAD importer will call after voxelising
    a FreeCAD geometry into NumPy arrays.
    All arrays must have shape (Nx, Ny, Nz).
    If pec_mask is provided, it is written into grid.pec_mask.
    """
```

**Test scaffolding** (used only in tests and examples — not part of the production pipeline):

```python
def set_box(grid, x0, x1, y0, y1, z0, z1, eps_r, mu_r=1.0, pec=False) -> FDTDGrid:
    """
    Fill an axis-aligned box with a uniform material, or mark as PEC if pec=True.
    Coordinates in metres; snapped to nearest cell.
    If pec=True, writes into grid.pec_mask instead of eps/mu arrays.
    # 3D-UPGRADE: z0/z1 range is already honoured — no changes needed.
    """

def set_cylinder(grid, cx, cy, radius, z0, z1, eps_r, mu_r=1.0, pec=False) -> FDTDGrid:
    """
    Fill a cylindrical rod aligned with Z, or mark as PEC if pec=True.
    # 3D-UPGRADE: z0/z1 range is already honoured.
    """

def set_coax(grid, cx, cy, r_inner, r_outer, eps_r_fill=1.0) -> FDTDGrid:
    """
    Build a coaxial cross-section in the XY plane.
    Inner conductor: marked PEC in grid.pec_mask.
    Outer conductor: marked PEC in grid.pec_mask.
    Dielectric fill between conductors: written to eps arrays.
    Calls set_cylinder internally.
    """
```

**Key rule:** all geometry functions ultimately call `set_material_arrays` or write directly to `grid.pec_mask`. The future CAD importer bypasses the scaffolding functions entirely and calls `set_material_arrays` directly.

---

### `update.py` — Field update functions

The core physics. Full 3D curl operators, vectorised over the entire grid using NumPy slicing. No loops over cells.

```python
def update_H(grid) -> FDTDGrid:
    """
    Advance H fields by half a timestep using the full 3D curl of E.

    Hx: dHz/dy - dHy/dz  (all three terms present)
    Hy: dHx/dz - dHz/dx
    Hz: dHy/dx - dHx/dy

    Uses centred finite differences on the Yee grid.
    # 3D-UPGRADE: no changes needed — curl is already 3D.
    """

def update_E(grid) -> FDTDGrid:
    """
    Advance E fields by a full timestep using the full 3D curl of H.
    # 3D-UPGRADE: no changes needed — curl is already 3D.
    """
```

**Yee grid staggering — standard 3D convention (Taflove):**

Each field component is located at a different sub-cell position. For cell index `(i, j, k)`:

```
Ex[i,j,k]  →  (i,    j+½,  k+½) · (dx, dy, dz)
Ey[i,j,k]  →  (i+½,  j,    k+½) · (dx, dy, dz)
Ez[i,j,k]  →  (i+½,  j+½,  k  ) · (dx, dy, dz)

Hx[i,j,k]  →  (i+½,  j,    k  ) · (dx, dy, dz)
Hy[i,j,k]  →  (i,    j+½,  k  ) · (dx, dy, dz)
Hz[i,j,k]  →  (i,    j,    k+½) · (dx, dy, dz)
```

Sanity check — the H update for `Hx` uses `(∇×E)_x = ∂Ez/∂y - ∂Ey/∂z`:
- `Hx` lives at `(i+½, j, k)`
- `∂Ez/∂y` differences `Ez[i,j,k]` at `y=j+½` and `Ez[i,j+1,k]` at `y=j+3/2` → centred at `y=j+1` ✓ (half-cell offset from Hx, consistent with Yee leapfrog)
- `∂Ey/∂z` differences `Ey[i,j,k]` at `z=k+½` and `Ey[i,j,k+1]` at `z=k+3/2` → centred at `z=k+1` ✓

The full set of update equations (H step, Faraday's law, `Δt` absorbed into coefficients):

```
Hx[i,j,k] -= (dt/mu_x[i,j,k]) * (
    (Ez[i,j+1,k] - Ez[i,j,k]) / dy  -  (Ey[i,j,k+1] - Ey[i,j,k]) / dz )

Hy[i,j,k] -= (dt/mu_y[i,j,k]) * (
    (Ex[i,j,k+1] - Ex[i,j,k]) / dz  -  (Ez[i+1,j,k] - Ez[i,j,k]) / dx )

Hz[i,j,k] -= (dt/mu_z[i,j,k]) * (
    (Ey[i+1,j,k] - Ey[i,j,k]) / dx  -  (Ex[i,j+1,k] - Ex[i,j,k]) / dy )
```

And the E step (Ampere's law):

```
Ex[i,j,k] += (dt/eps_x[i,j,k]) * (
    (Hz[i,j,k] - Hz[i,j-1,k]) / dy  -  (Hy[i,j,k] - Hy[i,j,k-1]) / dz )

Ey[i,j,k] += (dt/eps_y[i,j,k]) * (
    (Hx[i,j,k] - Hx[i,j,k-1]) / dz  -  (Hz[i,j,k] - Hz[i-1,j,k]) / dx )

Ez[i,j,k] += (dt/eps_z[i,j,k]) * (
    (Hy[i,j,k] - Hy[i-1,j,k]) / dx  -  (Hx[i,j,k] - Hx[i,j-1,k]) / dy )
```

NumPy slicing implementation for the H update (vectorised, no cell loops):

```python
# Hx update — interior cells only (avoids boundary edge)
grid.Hx[:, :-1, :-1] -= (dt / mu_x[:, :-1, :-1]) * (
    (grid.Ez[:, 1:, :-1] - grid.Ez[:, :-1, :-1]) / dy
  - dEy_dz  # zero for Nz=1, see guard below
)
```

The slicing pattern `[:-1]` / `[1:]` implements the forward difference and automatically stays within bounds.

Array slicing convention for a forward difference `∂F/∂x`:
```python
(F[1:, :, :] - F[:-1, :, :]) / dx   # result shape (Nx-1, Ny, Nz)
```
Boundary rows/columns are handled by PML and PEC — do not add special cases in `update.py`.

**Implementation note for Nz=1:** When `Nz=1`, all z-derivatives (`∂/∂z` terms) evaluate to zero automatically because the arrays have shape `(Nx, Ny, 1)` and the difference `grid.Ex[:,:,1:] - grid.Ex[:,:,:-1]` produces a `(Nx, Ny, 0)` array. Add a guard:

```python
# 3D-UPGRADE: remove this guard when Nz > 1
dHz_dz = np.zeros_like(grid.Hz)
if grid.Nz > 1:
    dHz_dz[:, :, :-1] = (grid.Hz[:, :, 1:] - grid.Hz[:, :, :-1]) / grid.dz
```

This keeps the update loop physically correct in both modes.

---

### `pml.py` — CPML implementation

CPML (Convolutional PML) is the boundary absorption layer. It adds auxiliary field arrays and modifies the curl update inside the PML region only. The PML region is visualised as a shaded overlay on the grid and material plots in `viz.py` — there is no standalone PML profile plot.

**Theory summary for the implementing agent:**

CPML replaces each spatial derivative `∂F/∂x` inside the PML with:

```
∂F/∂x → (1/κ_x) * ∂F/∂x + ψ_x
```

where `ψ_x` is a recursive convolution variable updated each timestep:

```
ψ_x^{n+1} = b_x * ψ_x^n + c_x * ∂F/∂x
```

The scalars `b_x` and `c_x` are precomputed from the profile functions:

```
σ_x(x),  κ_x(x),  α_x(x)
```

Parameter profiles (Roden-Gedney defaults, validated):
- `σ_x(x) = σ_max * (x / d_pml)^m`, with `m = 3`, `σ_max = 0.8*(m+1) / (η₀ * dx)`
- `κ_x(x) = 1 + (κ_max - 1) * (x / d_pml)^m`, `κ_max = 1` for v1 (propagating fields only)
- `α_x(x) = α_max * (1 - x/d_pml)`, `α_max = 0.05`
- `η₀ = 377 Ω` (free-space impedance)

Precomputed scalars (per cell, per axis):
```
b_x = exp(-(σ_x/κ_x + α_x) * dt / ε₀)
c_x = σ_x / (σ_x*κ_x + κ_x²*α_x) * (b_x - 1)
```

**Implementation notes (validated against Test 02 — read before editing `pml.py`):**

The two subtleties below are not optional; getting either wrong silently breaks
absorption or symmetry. Both were live bugs caught in Test 02.

1. **The CPML correction carries `MU0` / `EPS0`, because the material arrays are
   *relative*.** The `ψ` correction is part of the *same* curl as the interior
   update, so it must use the *identical* coefficient as `update.py`:
   `dt/(MU0·mu)` for H corrections, `dt/(EPS0·eps)` for E corrections. Dropping
   the constants makes the correction ~1e-6 (H) / ~1e-11 (E) too small and the
   PML stops absorbing entirely.

2. **The profile coordinate must follow the differencing convention of
   `update.py`.** E updates use *backward* differences, H updates use *forward*
   differences (see the E/H update equations above). Consequently a derivative
   driving an E-field component sits half a cell *below* its index, and one
   driving an H-field component sits on the integer node:
   - E-field (`*_E`) profiles sample the **staggered** coordinate `(i − ½)·ds`.
   - H-field (`*_H`) profiles sample the **integer**  coordinate `i·ds`.

   Using `(i + ½)` for the staggered grid mis-aligns the two PML slabs in
   *opposite* directions and produces phase-reversed reflections off opposite
   walls (x0 vs x1, y0 vs y1).

3. **CPML is additive on top of the interior update.** With `κ = 1` the interior
   update already supplies the `(1/κ)·∂F` term, so `update_H_pml` /
   `update_E_pml` only advance `ψ` and add the `ψ` correction. Loop order is
   fixed: `update_H → update_H_pml → update_E → update_E_pml`.

A residual reflection of ≈ −60 dB is normal for a 10-cell CPML and is not a bug;
it appears amplified in animations only because of global colour-scale autoscaling.

**Data structures:**

```python
@dataclass
class CPMLArrays:
    # Auxiliary E-curl correction variables (one per PML face per axis)
    psi_Ex_y: np.ndarray   # correction to dEx/dy in Hz update
    psi_Ex_z: np.ndarray   # correction to dEx/dz in Hy update
    psi_Ey_x: np.ndarray
    psi_Ey_z: np.ndarray
    psi_Ez_x: np.ndarray
    psi_Ez_y: np.ndarray
    # Same for H-curl corrections
    psi_Hx_y: np.ndarray
    # ... etc

    # Precomputed b, c profile arrays for each axis
    bx_E: np.ndarray   # shape (Nx,) — one value per x-cell in PML region
    cx_E: np.ndarray
    # ... etc

    d_pml: int         # PML thickness in cells (typically 8–12)
```

**Functions:**

```python
def init_cpml(grid, d_pml=10) -> CPMLArrays:
    """
    Allocate auxiliary arrays and precompute b, c profiles for all 6 faces.
    # 3D-UPGRADE: z-face PML arrays are allocated but zeroed when Nz=1.
    #             Set Nz > 1 and they activate automatically.
    """

def update_H_pml(grid, cpml) -> tuple[FDTDGrid, CPMLArrays]:
    """Update ψ arrays for H-field and apply CPML correction to H update."""

def update_E_pml(grid, cpml) -> tuple[FDTDGrid, CPMLArrays]:
    """Update ψ arrays for E-field and apply CPML correction to E update."""
```

**Main loop integration pattern:**

```python
grid = update_H(grid)
grid, cpml = update_H_pml(grid, cpml)   # CPML correction applied on top
grid = update_E(grid)
grid, cpml = update_E_pml(grid, cpml)
```

**Recommended PML thickness:** 8–12 cells. Thicker = better absorption, more memory. Start with 10.

---

### `pec.py` — PEC enforcement

Handles two distinct PEC operations. Both ultimately zero tangential E-field components.

**Domain face PEC** (boundary condition — walls of the simulation box):

```python
def apply_pec_faces(grid, faces=('x0', 'x1', 'y0', 'y1')) -> FDTDGrid:
    """
    Zero tangential E-field components on specified domain faces.
    faces: any subset of ('x0','x1','y0','y1','z0','z1')
    Called once per timestep, after E update and CPML correction.
    # 3D-UPGRADE: z faces already supported — no changes needed.
    """
```

**Interior PEC body** (material — solid conductors inside the domain):

```python
def apply_pec_mask(grid) -> FDTDGrid:
    """
    Zero all E-field components inside cells where grid.pec_mask is True.
    Called every timestep after apply_pec_faces.
    If grid.pec_mask is None or all-False, this is a no-op.

    Implementation: for each E component, zero cells where pec_mask is True.
    For surface accuracy, also zero the component on the faces of PEC cells
    (forward and backward neighbours). See note below.
    # 3D-UPGRADE: no changes needed — operates on full 3D mask.
    """
```

**Surface treatment note:** Zeroing only interior cells leaves a thin non-zero shell at the conductor surface. For v1, zeroing all E components where `pec_mask[i,j,k] == True` is sufficient. A more accurate surface treatment (zeroing only tangential components at the exact Yee face) can be added in v2 if needed.

**Correct timestep order** (from the main loop):
```
E update → CPML E correction → apply_pec_faces → apply_pec_mask → monitors
```
PEC enforcement must come after every E update, including after CPML corrections.

---

### `sources.py` — Source time function

For v1, spatial injection is hard-coded in each test/example (one or two lines directly in the time loop). The only reusable abstraction is the time function itself.

```python
@dataclass
class GaussianSource:
    t0: float        # pulse centre time (s)
    width: float     # pulse half-width (s); spectral bandwidth ≈ 1 / (2π · width)
    amplitude: float = 1.0

def gaussian_pulse(source, t) -> float:
    """
    Evaluate Gaussian pulse at time t.
    Returns: amplitude * exp(-0.5 * ((t - t0) / width)^2)
    """
    return source.amplitude * np.exp(-0.5 * ((t - source.t0) / source.width) ** 2)
```

**Choosing parameters for a target maximum frequency `f_max`:**
```python
width = 1.0 / (2 * np.pi * f_max)   # -3 dB bandwidth ≈ f_max
t0    = 4 * width                    # pulse fully risen by t=0 within 1% of peak
```

**Injection in the time loop** (test hard-codes this directly — not abstracted):
```python
# Soft additive injection — add to whatever field value already exists
grid.Ez[i, j, k] += gaussian_pulse(source, t)
```

Soft injection is transparent to passing waves (no impedance mismatch artifact). Hard injection (`=` instead of `+=`) reflects waves and must not be used.

**Future v2 source abstraction** will define a `Source` base class with `spatial_profile(grid)` and `time_function(t)` methods, allowing arbitrary user-defined functions. The current design does not preclude this — the injection line in the loop is unchanged.

---

### `monitors.py` — Diagnostics

```python
@dataclass
class FieldMonitor:
    """Record a single field component at a fixed cell location."""
    component: str      # 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'
    i: int
    j: int
    k: int
    times: list = field(default_factory=list)
    values: list = field(default_factory=list)

def record_field(monitor, grid) -> FieldMonitor:
    """Append current field value and timestep to the monitor."""


@dataclass
class MagnitudeMonitor:
    """
    Record |E| or |H| magnitude at a fixed cell location.
    |E| = sqrt(Ex^2 + Ey^2 + Ez^2)
    |H| = sqrt(Hx^2 + Hy^2 + Hz^2)
    Useful for near-field diagnostics without needing to know
    which component dominates.
    """
    field: str          # 'E' or 'H'
    i: int
    j: int
    k: int
    times: list = field(default_factory=list)
    values: list = field(default_factory=list)

def record_magnitude(monitor, grid) -> MagnitudeMonitor:
    """Compute and append |E| or |H| at the monitor location."""


@dataclass
class SnapshotMonitor:
    """Capture a 2D slice of a field component at regular intervals."""
    component: str
    k_slice: int        # which z-index to capture (use 0 for Nz=1)
    interval: int       # record every N timesteps
    snapshots: list = field(default_factory=list)
    snap_times: list = field(default_factory=list)

def record_snapshot(monitor, grid) -> SnapshotMonitor:
    """Append a 2D slice to the snapshot list if this is a recording timestep."""


@dataclass
class EnergyMonitor:
    """
    Track total electromagnetic energy in the domain.
    U = 0.5 * sum(eps * |E|^2 + mu * |H|^2) * dx*dy*dz
    Must not grow over time in a stable simulation.
    """
    times: list = field(default_factory=list)
    values: list = field(default_factory=list)

def record_energy(monitor, grid) -> EnergyMonitor:
    """Compute total field energy and append to time series."""
```

---

### `viz.py` — Visualisation

All matplotlib calls live here. No plotting in any other module.

**Infrastructure visualisations (no physics required):**

```python
def plot_grid_xy(grid, cpml=None, ax=None):
    """
    Draw the Yee cell grid in the XY plane (k=0 slice).
    Show E and H component locations as staggered markers per the Yee convention.
    Annotate cell dimensions dx, dy and total domain size in metres.
    If cpml is provided, shade the PML region with a semi-transparent overlay
    and label its thickness in cells.
    """

def plot_materials_xy(grid, component='eps_z', cpml=None, ax=None):
    """
    2D colour map of a material array (eps or mu) in the XY plane.
    Show cell boundaries as thin grid lines.
    Annotate with colour bar and physical dimensions (metres).
    If cpml is provided, overlay the PML region as a shaded border.
    Mark PEC cells (grid.pec_mask) with a distinct hatch or solid colour.
    """

def plot_source_waveform(source, dt, n_steps, ax=None):
    """
    1D plot of the Gaussian pulse time function over the simulation duration.
    X-axis: time in nanoseconds. Y-axis: normalised amplitude.
    Mark t0 and the ±2σ width. Print estimated bandwidth to stdout.
    """
```

**Field visualisations:**

```python
def plot_field_snapshot(snapshot_array, grid, timestep, ax=None):
    """
    2D colour map of a single field snapshot (a 2D NumPy array).
    Use diverging colourmap (RdBu) centred at zero.
    Annotate with physical dimensions (metres) and timestep number.
    """

def animate_snapshots(snapshot_monitor, grid, interval_ms=50):
    """
    Animate a sequence of field snapshots.
    Returns a matplotlib FuncAnimation object.
    Save with anim.save('out.gif') or display inline in Jupyter.
    """

def plot_monitor_time_series(monitor, dt, ax=None):
    """
    Plot a FieldMonitor or MagnitudeMonitor time series.
    X-axis: time in nanoseconds. Y-axis: field value or magnitude in SI units.
    Label with component name and monitor location.
    """

def plot_energy(monitor, dt, ax=None):
    """
    Plot total energy vs time on a log Y-axis.
    Flat = lossless interior; decaying = PML absorbing outgoing waves.
    A rising curve indicates numerical instability — simulation must be stopped.
    """
```

---

## Test sequence

Tests must be run in order. Each test validates one subsystem and is a prerequisite for the next.

### Test 00 — Grid, material, and PML region visualisation

**No physics. No time loop.**

Steps:
1. Create a grid with `Nx=50, Ny=50, Nz=1, dx=1e-3` (1 mm cells, 5 cm domain)
2. Call `init_cpml(grid, d_pml=8)` to create the CPML object
3. Call `plot_grid_xy(grid, cpml)` — verify E and H positions are correctly staggered per the Yee table above; verify PML region is shaded on all 4 sides with correct thickness
4. Place a dielectric box (eps_r=4) in the centre using `set_box()`
5. Call `plot_materials_xy(grid, cpml=cpml)` — verify the box appears at the correct location, PML shading visible, colour bar correct
6. Place a coaxial cross-section using `set_coax()`
7. Call `plot_materials_xy(grid, cpml=cpml)` — verify inner and outer PEC regions shown with distinct hatch, dielectric fill visible

**Pass criteria:** visual inspection — geometry matches intent, dimensions correct, PML overlay visible and correctly sized, PEC cells distinctly marked.

---

### Test 01 — Source waveform visualisation

**No physics. No time loop.**

Steps:
1. Create a `GaussianSource` targeting `f_max = 10 GHz`
2. Call `plot_source_waveform(source, grid.dt, n_steps=2000)`
3. Verify the pulse is fully contained within the simulation window (< 1% amplitude at both ends)
4. Confirm printed bandwidth matches `f_max`

**Pass criteria:** pulse fits within window, bandwidth matches target.

---

### Test 02 — Free-space Gaussian pulse propagation

**First physics test.**

Setup:
- Grid: `Nx=200, Ny=200, Nz=1, dx=0.5e-3` (0.5 mm cells, 10 cm square domain)
- Material: vacuum everywhere (`set_vacuum`)
- Boundaries: CPML on all 4 faces (no PEC faces, no PEC mask)
- Source: soft Ez injection at centre cell `(100, 100, 0)`, Gaussian pulse `f_max = 5 GHz`; injection is a single line in the loop: `grid.Ez[100, 100, 0] += gaussian_pulse(source, t)`
- Monitors: `FieldMonitor` at 4 symmetric points equidistant from source; `EnergyMonitor`; `SnapshotMonitor` on Ez every 20 steps
- Run: 2000 timesteps

Validation checks:
1. Animate snapshots — circular wavefront propagates outward from source
2. Field monitors — pulse arrives at all 4 points at the same time (symmetry check)
3. Arrival time matches `t = r/c` where `r` is monitor distance from source (measure to within ±2 timesteps)
4. Energy monitor — rises while pulse is injected, then decays monotonically as wavefront enters PML; no late-time growth
5. No visible reflections after wavefront has been absorbed

**Pass criteria:** all 5 checks pass. Visible late-time reflections → check CPML σ_max and d_pml. Energy growth → CFL violation, check dt.

---

### Test 03 — PEC cavity resonance

Setup:
- Grid: `Nx=100, Ny=80, Nz=1, dx=1e-3` (1 mm cells, 10 cm × 8 cm cavity)
- Material: vacuum
- Boundaries: `apply_pec_faces(grid, faces=('x0','x1','y0','y1'))` — no CPML (resonance must not be absorbed)
- Source: soft Ez injection at `(23, 17, 0)` (off-centre to avoid modal nodes), short Gaussian pulse
- Monitors: `FieldMonitor` at 3 points; `SnapshotMonitor` every 50 steps
- Run: 10000 timesteps

Validation checks:
1. FFT of FieldMonitor time series — peaks at cavity TM resonant frequencies
2. Analytic formula: `f_mn = (c/2) * sqrt((m/a)² + (n/b)²)`, `a=0.1 m`, `b=0.08 m`
3. Measured vs analytic: agreement within < 1%
4. Snapshot animation shows standing wave patterns

**Pass criteria:** at least 3 resonant peaks identified with < 1% frequency error.

---

### Test 04 — Rectangular waveguide TE10 mode

Setup:
- Grid: `Nx=200, Ny=50, Nz=1, dx=0.5e-3` (propagation in X, cross-section in Y)
- Width: 25 mm → `f_c10 = c / (2 × 0.025) ≈ 6 GHz`
- Material: vacuum
- Boundaries: `apply_pec_faces` on y=0 and y=Ny (waveguide walls); CPML on x=0 and x=Nx
- Source: soft Ez injection along a vertical line `x=20, j=0..Ny-1, k=0`; loop line: `grid.Ez[20, :, 0] += gaussian_pulse(source, t)`
- Monitors: `FieldMonitor` at two x-locations on the centreline; `SnapshotMonitor` every 20 steps
- Run: 3000 timesteps

Validation checks:
1. Below cutoff: field amplitude decays exponentially in X — no propagation
2. Above cutoff: wavefront propagates, phase velocity `v_ph = c / sqrt(1 - (f_c/f)²)` within 2%
3. Snapshot shows half-sine transverse field profile (TE10 mode shape)

**Pass criteria:** exponential decay below cutoff confirmed; phase velocity within 2% above cutoff.

---

### Test 05 — Coaxial TEM mode (MEEP reference comparison)

Setup:
- Grid: `Nx=200, Ny=200, Nz=1, dx=0.25e-3` (0.25 mm, 5 cm × 5 cm domain)
- Geometry: coaxial cross-section in XY plane using `set_coax(grid, cx=100, cy=100, r_inner=14, r_outer=32)` (in cells; ≈3.5 mm and 8 mm radii → Z₀ ≈ 50 Ω vacuum fill)
- Boundaries: CPML on all 4 domain faces; `apply_pec_mask` enforces inner and outer conductors
- Source: soft Ez injection on an annular ring of cells between inner and outer conductor radii at one face
- Monitors: `FieldMonitor` for Ez and Hz at 3 radii between conductors; `MagnitudeMonitor` for |E|; `SnapshotMonitor` every 20 steps
- Run: 2000 timesteps

Validation checks:
1. Ez snapshot profile: field magnitude decays as `1/r` between conductors
2. Wave impedance ratio: `|Ez| / |Hφ| ≈ η₀ = 377 Ω` at all monitor radii (within 5%)
3. Compare Ez and Hz snapshot profiles against existing MEEP reference simulation

**Pass criteria:** `1/r` radial profile confirmed; impedance ratio within 5%; profile shapes match MEEP reference within 5%.

---

## The main simulation loop (reference pattern)

Every example and test should follow this pattern exactly. Do not deviate.

```python
import numpy as np
from fdtd.grid import create_grid
from fdtd.materials import set_vacuum
from fdtd.pml import init_cpml, update_H_pml, update_E_pml
from fdtd.pec import apply_pec_faces, apply_pec_mask
from fdtd.sources import GaussianSource, gaussian_pulse
from fdtd.monitors import (FieldMonitor, MagnitudeMonitor, SnapshotMonitor,
                            EnergyMonitor, record_field, record_magnitude,
                            record_snapshot, record_energy)
from fdtd.update import update_H, update_E

# --- Setup ---
grid = create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3, dy=0.5e-3, dz=0.5e-3)
grid = set_vacuum(grid)
cpml = init_cpml(grid, d_pml=10)

source   = GaussianSource(t0=30*grid.dt, width=10*grid.dt)
fmon     = FieldMonitor(component='Ez', i=150, j=100, k=0)
magmon   = MagnitudeMonitor(field='E', i=150, j=100, k=0)
snap_mon = SnapshotMonitor(component='Ez', k_slice=0, interval=20)
emon     = EnergyMonitor()

# --- Time loop ---
N_STEPS = 2000
for n in range(N_STEPS):
    t = n * grid.dt

    # 1. Update H (interior, full 3D curl)
    grid = update_H(grid)
    # 2. CPML H correction
    grid, cpml = update_H_pml(grid, cpml)

    # 3. Update E (interior, full 3D curl)
    grid = update_E(grid)
    # 4. CPML E correction
    grid, cpml = update_E_pml(grid, cpml)

    # 5. Enforce PEC — always after E update and CPML correction
    grid = apply_pec_faces(grid, faces=('y0', 'y1'))   # omit if no PEC walls
    grid = apply_pec_mask(grid)                         # no-op if pec_mask is None

    # 6. Inject source (soft, additive — hard-coded per test)
    grid.Ez[100, 100, 0] += gaussian_pulse(source, t)

    # 7. Record monitors
    fmon     = record_field(fmon, grid)
    magmon   = record_magnitude(magmon, grid)
    snap_mon = record_snapshot(snap_mon, grid)
    emon     = record_energy(emon, grid)

    grid.time_step += 1
```

**Update order is fixed and must not change:**
1. H update (interior)
2. CPML H correction
3. E update (interior)
4. CPML E correction
5. PEC enforcement (faces, then mask)
6. Source injection
7. Monitor recording

---

## 3D upgrade checklist (future reference)

When the time comes to move from `Nz=1` to full 3D, search for `# 3D-UPGRADE:` comments throughout the codebase. The main changes will be:

- Remove z-derivative guards in `update.py`
- Allocate CPML z-face arrays in `pml.py` (already structured for this)
- Add z-face PEC support in `pec.py` (already parametrised)
- Increase `Nz` in grid construction — everything else follows automatically
- Profile performance at target grid size; if > 100³ cells, evaluate JAX migration for `update.py` only

---

## JAX migration notes (for future reference)

The functional architecture chosen for v1 is designed to make a JAX migration straightforward. When the time comes:

1. Register `FDTDGrid` and `CPMLArrays` as JAX pytrees using `jax.tree_util.register_pytree_node`
2. Replace `import numpy as np` with `import jax.numpy as jnp` in `update.py` and `pml.py`
3. Replace in-place mutation patterns with returned new arrays (already the design — pure functions)
4. Wrap the time loop body in `@jax.jit`
5. The z-derivative guards in `update.py` must be rewritten using `jax.lax.cond` — flag these with `# JAX-UPGRADE:`

The main structural obstacle is CPML auxiliary array updates — they carry state across timesteps. In JAX this is handled by `jax.lax.scan` over the time loop, passing the full `(grid, cpml)` tuple as carry state.

---

## Physical constants (use these values throughout)

```python
C0   = 299792458.0      # speed of light, m/s
EPS0 = 8.8541878e-12    # vacuum permittivity, F/m
MU0  = 1.2566370e-6     # vacuum permeability, H/m
ETA0 = 376.730313       # free-space impedance, Ω
```

---

*End of plan. Start with Test 00 and build upward.*
