# PML Change Notes — Independent per-face CPML

**Date:** 2026-06-12
**Subsystem:** `wavesim/pml.py` (CPML, Roden–Gedney)
**Driven by:** `tests/test_04_waveguide.py` (rectangular waveguide dominant mode)
**Status:** ✅ Implemented; Test 04 passes; Tests 00/02/03 unaffected (backward compatible).

---

## Summary

`init_cpml` previously made **every** domain face absorbing. That is wrong for
any problem where a face is a PEC wall or a symmetry plane — most immediately the
rectangular waveguide of Test 04, whose y-faces are PEC walls. With an absorbing
y-PML in place, the transverse standing mode (the half-sine that *defines* the
waveguide mode) gets damped near the walls and the dispersion is corrupted.

`pml.py` now lets the caller choose which of the six faces are CPML. Faces left
out are transparent.

| Item | Before | After |
|------|--------|-------|
| `init_cpml` signature | `init_cpml(grid, d_pml=10)` | `init_cpml(grid, d_pml=10, faces=('x0','x1','y0','y1','z0','z1'))` |
| Face selection | all 6 always on | any subset of the 6 |
| Default behaviour | all 6 on | all 6 on (unchanged) |

---

## Why all-faces was a problem (Test 04)

Test 04 is a 2D TMz waveguide: propagation along x, PEC walls at y0/y1, open
(absorbing) ends at x0/x1. The dominant mode is

```
Ez(x, y, t) ~ sin(pi * y / b) * exp(j(omega t - beta x))
```

The `sin(pi y / b)` transverse profile is sustained by reflection off the two
PEC y-walls. An absorbing y-PML eats that reflection, so the mode cannot form.

`init_cpml` builds each axis profile with `_calc_profile_1d`, which is inert only
when the axis is too thin to host two slabs (`N <= 2*d_pml`). For the waveguide
`Ny = 50`, `d_pml = 10` → `50 > 20`, so the **y-PML was active** and absorbing —
exactly the wrong thing on PEC walls.

### Interim workaround (now removed)

Before the API existed, Test 04 neutralised the y-profiles by hand after
`init_cpml`:

```python
cpml = init_cpml(grid, d_pml=10)
cpml.by_E[:] = 1.0; cpml.cy_E[:] = 0.0   # b=1, c=0 -> psi stays 0 -> no y-correction
cpml.by_H[:] = 1.0; cpml.cy_H[:] = 0.0
```

This works (with `b=1, c=0` the convolution variable `psi` never leaves 0, so no
correction is added) but it reaches into CPML internals from the test. It has
been replaced by the supported API below.

---

## The change

### 1. `_calc_profile_1d` — optional per-slab construction

Each 1D axis profile already contains a **low-index slab** (absorbing towards
index 0) and a **high-index slab** (absorbing towards index N-1) — i.e. the two
opposite faces of that axis. Two boolean flags now gate them independently:

```python
def _calc_profile_1d(N, ds, dt, d_pml, staggered, low=True, high=True):
    ...
    if N <= 2 * d_pml or not (low or high):
        return b, c                      # identity -> transparent face
    ...
    if low:
        # build low-index (face at index 0) slab
    if high:
        # build high-index (face at index N-1) slab
```

A disabled slab simply leaves that region at `(b, c) = (1, 0)`.

### 2. `init_cpml` — `faces` argument

```python
ALL_FACES = ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')

def init_cpml(grid, d_pml=10, faces=ALL_FACES) -> CPMLArrays:
    bad = set(faces) - set(ALL_FACES)
    if bad:
        raise ValueError(...)
    x_lo, x_hi = 'x0' in faces, 'x1' in faces
    y_lo, y_hi = 'y0' in faces, 'y1' in faces
    z_lo, z_hi = 'z0' in faces, 'z1' in faces
    # low slab  = face at index 0   ('*0')
    # high slab = face at index N-1 ('*1')
    bx_E, cx_E = _calc_profile_1d(grid.Nx, grid.dx, grid.dt, d_pml, True,  x_lo, x_hi)
    ...
```

Face-to-slab mapping: `'x0'/'y0'/'z0'` → low-index slab; `'x1'/'y1'/'z1'` →
high-index slab. Unknown face strings raise `ValueError`.

The `update_H_pml` / `update_E_pml` functions are **unchanged** — they already
just advance `psi` and add the correction; a transparent face contributes zero
through `c = 0`.

---

## Usage

```python
# Free space / cavity-in-vacuum — absorb on all sides (default, as before)
cpml = init_cpml(grid, d_pml=10)

# Waveguide — PEC side walls at y0/y1, open ends at x0/x1
cpml = init_cpml(grid, d_pml=10, faces=('x0', 'x1'))

# Half-space with a PEC ground plane at y0
cpml = init_cpml(grid, d_pml=10, faces=('x0', 'x1', 'y1'))
```

---

## Verification

| Test | Faces used | Result |
|------|-----------|--------|
| `test_02_free_space.py` | default (all 6) | PASS — arrival symmetry, r/c, PML absorption, stability all unchanged |
| `test_04_waveguide.py` (manual neutralise) | x-only via `cpml.by_*` hack | PASS |
| `test_04_waveguide.py` (new `faces=('x0','x1')`) | x-only via API | PASS — **bit-for-bit identical** numbers |

Test 04 metrics (both methods identical):

```
Below cutoff (4 GHz):  alpha err 0.2%,  far/near 4.78e-3   -> evanescent  PASS
Above cutoff (9 GHz):  v_ph  err 0.06%, far/near 1.003     -> propagating PASS
Transverse profile:    corr 1.0000 with sin(pi y / b)      -> TE10 shape  PASS
```

The default-argument path reproduces the previous all-faces behaviour exactly,
so Tests 00/02/03 need no changes.

---

## Conventions preserved

The fixes documented in `DEBUG_NOTES_test02_pml.md` are untouched and still hold:

1. CPML correction carries `MU0` (H) / `EPS0` (E) — relative material arrays.
2. E-field profiles sample the staggered coord `(i − ½)·ds`; H-field profiles
   sample the integer coord `i·ds`.
3. CPML is additive on the interior update; loop order
   `update_H → update_H_pml → update_E → update_E_pml`.

This change only gates **which slabs are built**; it does not alter the profile
maths, coefficients, or staggering for the slabs that remain enabled.
