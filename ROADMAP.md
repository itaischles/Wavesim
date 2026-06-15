# Wavesim — Roadmap

Forward-looking plan for the engine. For *current* usage see
[docs/API_GUIDE.md](docs/API_GUIDE.md); for setup see
[docs/HOW_TO_SET_UP.md](docs/HOW_TO_SET_UP.md). This file replaces the original
v1 build plan, whose implementation details now live in the code and the docs.

## Where things stand

A validated, functional NumPy FDTD solver: Yee-grid `update_E`/`update_H`, CPML
(Roden–Gedney, per-face selectable), PEC faces + interior conductor masks,
Gaussian sources, point/magnitude/snapshot/energy monitors, and visualisation.
Tests 00–04 run as a thin `Nz=1` slice; **Test 05 is the first full 3D run
(`Nz>1`)** — a coaxial TEM mode validating the 3D curl and z-face CPML on the
same code (1/r profile, `Z=η₀`, `v=c`).

Items below are roughly in priority order; each is independent unless noted.

---

## 1. Make full 3D first-class

Test 05 proves the 3D code paths work, but they are still gated by `Nz>1`
branches and `# 3D-UPGRADE:` guards written for the 2D-slice era. Promote 3D from
"works when you set `Nz>1`" to the default, well-exercised mode.

Checklist (search the codebase for `# 3D-UPGRADE:`):

- [x] z-face CPML allocated and active for `Nz>1` (`pml.py`) — exercised by Test 05.
- [x] z-face PEC supported (`pec.py` is already parametrised over all six faces).
- [ ] Remove / simplify the `Nz>1` z-derivative guards in `update.py` once 3D is
      the default path (keep them only if the 2D-slice fast path stays valuable).
- [ ] Add a genuinely volumetric 3D validation test (e.g. a rectangular cavity
      with at least one `z`-varying mode, or a 3D dipole radiation pattern) so 3D
      correctness does not rest on the coax alone.
- [ ] 3D visualisation helpers in `viz.py` (orthogonal slice views / isosurfaces);
      the current plot helpers assume a single `k`-slice.
- [ ] Profile memory and runtime at representative 3D sizes; document practical
      grid-size limits for pure NumPy (see §3).

---

## 2. `Simulation` class (v2 API)

Today the time loop lives in each test/example by design. A thin orchestration
layer would remove that boilerplate without hiding the physics:

- A `Simulation` object wrapping `grid` + `cpml` + sources + monitors, with a
  `run(n_steps)` method that executes the canonical loop
  (`update_H → update_H_pml → update_E → update_E_pml → PEC → sources → monitors`).
- A `Source` base class with `spatial_profile(grid)` and `time_function(t)`,
  so arbitrary user-defined excitations drop in (the current soft-injection line
  in the loop is already compatible with this).
- Keep the functional core intact: the class only *orchestrates* the existing
  pure functions, so scripts that write their own loop keep working.

---

## 3. Performance — JAX migration

NumPy is sufficient for v1. Beyond roughly **100³ cells** the per-step array work
dominates; that is the point to evaluate a JAX backend. The functional design was
chosen to make this migration mechanical:

1. Register `FDTDGrid` and `CPMLArrays` as JAX pytrees via
   `jax.tree_util.register_pytree_node`.
2. Swap `import numpy as np` → `import jax.numpy as jnp` in `update.py` and
   `pml.py` (the hot modules only).
3. Replace in-place mutation with returned new arrays — already the design
   (pure functions returning the grid).
4. Wrap the time-loop body in `@jax.jit`.
5. Rewrite the `Nz>1` z-derivative guards in `update.py` with `jax.lax.cond`
   (flag these with `# JAX-UPGRADE:`).

The one structural obstacle is the CPML auxiliary-array state carried across
timesteps. In JAX this is handled by `jax.lax.scan` over the time loop, passing
the full `(grid, cpml)` tuple as the carry.

---

## 4. Nonuniform rectilinear grid

Allow per-axis variable cell sizes (`dx_i`, `dy_j`, `dz_k`) so fine features can
be resolved without paying for a globally fine grid. Touches the curl stencils in
`update.py` (per-cell spacings), the CPML profile coordinates in `pml.py`, the
CFL/`dt` computation in `grid.py`, and the geometry snapping in `materials.py`.

---

## 5. Waveguide-port mode solver + modal injection

A 2D cross-section eigenmode solver to compute guided-mode field profiles, then
inject a chosen mode at a port (clean single-mode excitation instead of relying
on a shaped soft source). Natural prerequisite for S-parameter extraction later.

---

*Out-of-scope items from v1 (DFT/flux monitors, S-parameters, far-field,
symmetry planes) may return as dedicated roadmap entries once the items above
land.*
