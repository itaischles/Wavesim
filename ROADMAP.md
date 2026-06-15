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
- [x] Add a genuinely volumetric 3D validation test — **Test 06** (rectangular
      PEC cavity) matches 14 analytic resonances within 1.5%, 10 of them with a
      half-wave along `z` (`p>=1`), so 3D correctness no longer rests on the
      coax's extruded TEM mode alone.
- [x] 3D visualisation helpers in `viz.py` — `plot_field_slices_3d`
      (orthogonal XY/XZ/YZ triptych) and `animate_field_slices_3d` (general
      multi-plane time animation). Tests 05 and 06 both render through them.
      (Isosurfaces deferred — slice views cover current needs.)
- [x] Profile memory and runtime at representative 3D sizes — `tools/profile_3d.py`.
      Findings: **~0.33 µs/cell-step**, **192 bytes/cell** (half of it the 12
      full-grid CPML `psi` arrays). Practical pure-NumPy ceiling for runs in the
      3–5 min budget is ≈100³ (e.g. 96³ fits ~600 steps in 3 min) — the
      acceleration trigger point addressed by the Numba backend in §3. See §3.
- [ ] Remove / simplify the `Nz>1` z-derivative guards in `update.py` once 3D is
      the default path. **Deliberately kept for now**: the `Nz=1` fast path still
      benefits Tests 00–04 and quick iteration. Note the Numba backend (§3)
      already collapses both paths into one 3D kernel guarded by a plain `if Nz>1`,
      so this NumPy duplication can be retired when Numba becomes the default.
- [x] Allocate the CPML `psi` arrays as boundary slabs instead of full volume.
      Each `psi` is now compressed along its derivative axis to just the active
      PML cells (`sel_*` in `pml.py`), cutting their footprint to ~`2·d_pml/N` of
      full volume — bit-identical to the old allocation since the dropped cells
      held 0 forever (the recursion is local; `c=0`/`b·0=0` in the interior).

---

## 2. `Simulation` class (v2 API) — **done**

A thin orchestration layer that removes the per-script time-loop boilerplate
without hiding the physics. Lives in `wavesim/simulation.py` and
`wavesim/sources.py`; see `docs/API_GUIDE.md` §6 and the worked tutorial in
`tests/test_07_simulation_api.py`.

- [x] A `Simulation` object wrapping `grid` + `cpml` + sources + monitors, with a
      `run(n_steps)` method that executes the canonical loop
      (`update_H → update_H_pml → update_E → update_E_pml → PEC → sources → monitors`).
      `step()` exposes a single iteration; `run(..., callback=...)` allows
      per-step hooks (progress, etc.).
- [x] A `Source` base class with `spatial_profile(grid)`, `time_function(t)` and
      a soft-additive `inject(grid, t)`. Concrete `PointSource` and `ArraySource`
      cover the common cases; `GaussianSource` is now callable so it doubles as a
      waveform. Custom excitations subclass `Source`.
- [x] The functional core is untouched: the class only *orchestrates* the
      existing pure functions, so scripts that write their own loop keep working.
      Verified bit-for-bit identical to a hand-written loop (Test 07, Check 3).

---

## 3. Performance — Numba acceleration — **done** (was: JAX)

NumPy is sufficient for v1 but single-threaded: at representative 3D sizes the
solver ran at <10% CPU utilisation (~1 of 12 cores), ~0.33 µs/cell-step
(`tools/profile_3d.py`), so a 96³ grid fit only ~600 steps in a 3-minute budget.

**JAX was the original plan but rejected for this machine.** Native-Windows JAX
is CPU-only (GPU needs WSL2), and the "mechanical swap" premise was false: the
update functions *mutate arrays in place* (`grid.Hx[...] -= …`), which JAX forbids
— a full `.at[].set()` rewrite of the whole pipeline. The real, measured problem
(single-thread NumPy) is better solved by **Numba**.

Delivered (`wavesim/backend_numba.py`):

- The four hot functions (`update_H/E`, `update_H_pml/E_pml`) reimplemented as
  explicit-loop `@njit(parallel=True, cache=True)` kernels that mutate the same
  NumPy arrays in place — **signature-compatible** drop-ins. A single 3D kernel
  subsumes the `Nz=1` fast path via an `if Nz>1` branch, so the parallel NumPy
  twin in `update.py`/`pml.py` is no longer the only correct path.
- Selected per-run via `Simulation(backend='numba')` (default `'numpy'`); the
  NumPy reference is untouched and serves as the validation oracle.
- **Parity**: `tests/test_08_numba_parity.py` — bit-identical to NumPy
  (`max|diff| == 0`) on both the 2D-slice and full-3D paths (no parallel
  reductions, identical float64 arithmetic).
- **Speedup** (`tools/benchmark_numba.py`, GTX-1660 box, 12 cores):
  **~10–12.5× over NumPy** across 48³–128³. The win is dominated by op-fusion
  (single-thread already ~9×, no NumPy temporaries); the stencil is
  memory-bandwidth-bound, so threading adds only ~1.4–1.5× and **peaks at 4–6
  threads** (12 threads is slightly *slower* than 6 — prefer
  `numba.set_num_threads(4..6)`).

Follow-ups: optional `numba.cuda` GPU kernels for the GTX 1660 (native Windows,
no WSL2); migrating monitors into the kernels. The slab-compressed CPML `psi`
layout from §1 carried over unchanged.

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
