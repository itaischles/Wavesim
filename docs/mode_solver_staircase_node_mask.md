# The mode solver's staircase conductor is half a cell too small

**Status:** open, not fixed. Found 2026-08-05 while validating the electrostatic
solver (`wavesim/electrostatics.py`) against `solve_tem_modes`.
**Affects:** `wavesim/mode_solver.py`, staircase (non-conformal) path only.
**Does not affect:** the conformal path, which is the validated one.

---

## The finding

`solve_tem_modes` builds its conductor mask by slicing the cell-centred
`grid.pec_mask` and using the result as if it were indexed by *node*:

```python
# wavesim/mode_solver.py:694-698
if fa_full is not None:
    pec_full = _conformal_node_pec(fa_full, fb_full)     # conformal: correct
elif grid.pec_mask is None:
    pec_full = np.zeros(eps_a_full.shape, dtype=bool)
else:
    pec_full = _slice(grid.pec_mask, normal, k).astype(bool)   # <-- here
```

φ lives on the nodes, but `pec_mask` is indexed by cell, and cell `i` spans nodes
`i` → `i+1`. Treating the arrays as interchangeable keeps each conductor's
low-side surface node and silently drops its high-side one. A block occupying
cells 2..5 physically spans `x[2] … x[6]`; this marks nodes 2..5, putting the far
surface at `x[5]`. **Every conductor is one cell short along each axis, on its
high side only** — so it is both undersized and asymmetric.

The degenerate case makes it plainest: a conductor one cell thick becomes a
node mask one node thick, i.e. a sheet of zero thickness sitting on its low face.

This is precisely the trap `wavesim/pec.py::apply_pec_mask` documents for the
FDTD update itself, where the same mistake once cost 6.8% in ε_eff on an
RG58-like coax. The fix there was to zero an E-edge when *any* of the four cells
touching it is PEC (`build_pec_edge_masks`, a dilation in the two perpendicular
axes). The mode solver never received the equivalent treatment on its staircase
path.

## Why this counts as a bug rather than a choice

The module's own design principle, from its docstring:

> When the grid carries cut-cell open fractions the solver switches to them
> wholesale — conductor mask, stencil, energy integral and launched ê — so the
> port is solved on the *same* geometry the FDTD steps. Without that the Z₀ the
> port presents stops being the Z₀ the run presents.

That reasoning applies just as much to the staircase path, and there it is not
honoured: the FDTD's staircase conductor (after the `build_pec_edge_masks`
dilation) is *not* the mode solver's staircase conductor. A port solved on
geometry the run does not step is the exact failure the conformal work existed
to prevent.

## Minimal reproduction

Rectangular geometry, where staircasing is exact and the node-mask convention is
therefore the only variable. Two full-width PEC strips one cell thick, at
y-cells 2 and 15, so the metal faces are at `y[3]` and `y[15]` — a true gap of
**12** cells:

```python
import numpy as np, warnings
import wavesim as ws
from wavesim.parts import pec_node_mask

ds = 1e-3
g = ws.set_vacuum(ws.create_grid(12, 20, 1, ds, ds, ds))
ws.set_box(g, 0, 12e-3,  2e-3,  3e-3, 0, 1e-3, 1.0, pec=True, name='lo')
ws.set_box(g, 0, 12e-3, 15e-3, 16e-3, 0, 1e-3, 1.0, pec=True, name='hi')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m = ws.solve_tem_modes(g, normal='z', position=0.0, boundary='neumann')[0]
    sol = (ws.Electrostatics(g)
             .set_potential('lo', 0.0).set_potential('hi', 1.0)
             .solve(boundary='neumann', method='direct'))

print(np.round(m.phi[6, :], 4))            # mode solver
print(np.round(sol.phi[6, :, 0], 4))       # electrostatics
print(g.pec_mask[6, :, 0].astype(int))     # what the mode solver pins
print(pec_node_mask(g)[6, :, 0].astype(int))   # what the FDTD actually shorts
```

Observed:

```
mode solver phi : 0 0 0 .0769 .1538 ... .9231 1 1 1 1 1     <- 1/13 per step
electrostatics  : 0 0 0 0 .0833 .1667 ... .9167 1 1 1 1 1   <- 1/12 per step
mode solver pins: 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0   <- one node per strip
FDTD shorts     : 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0   <- both surfaces
```

The mode solver spreads the drop over **13** intervals against a true gap of
**12**. For a parallel-plate line, `C' ∝ 1/d`, so it reads `12/13` of the correct
capacitance — **−7.7%** at this mesh, and correspondingly high in Z₀.

## Magnitude on a real structure

Coax, a = 3 mm, b = 10 mm, ε_r = 2.3, C'_analytic = 106.277 pF/m. Refining the
mesh, comparing `solve_tem_modes` against the electrostatic solver (which derives
its node mask from `build_pec_edge_masks`):

| cells per inner radius | mode solver | error | electrostatic | error |
|---:|---:|---:|---:|---:|
| 3  |  97.660 pF/m | −8.1% | 123.702 pF/m | +16.4% |
| 6  | 100.614 pF/m | −5.3% | 113.148 pF/m |  +6.5% |
| 12 | 102.110 pF/m | −3.9% | 108.270 pF/m |  +1.9% |
| 18 | 103.857 pF/m | −2.3% | 107.970 pF/m |  +1.6% |

Both converge, so neither is *broken*; the mode solver simply approaches from
below because its conductors are undersized, and converges more slowly. The
error is a first-order geometry offset, exactly what a half-cell surface
displacement produces.

## Proposed fix

Replace the staircase branch with the node mask derived from the FDTD's own
edge masks — the function already exists and is tested:

```python
from wavesim.parts import pec_node_mask
...
else:
    pec_full = _slice(pec_node_mask(grid), normal, k)
```

`pec_node_mask` calls `build_conformal_edge_masks` when the grid carries cut
cells and `build_pec_edge_masks` otherwise, so it returns the conformal answer
on a conformal grid. The conformal branch above it therefore becomes redundant
and could collapse into it — verify that `_conformal_node_pec(fa, fb)` and the
sliced `pec_node_mask` agree on a cut-cell grid **before** removing anything;
they are derived differently and only believed to be equivalent.

Note `pec_node_mask` operates on the full 3D grid and is then sliced, whereas
`_conformal_node_pec` works within the plane. The concern is that the dilation
along the *normal* might leak into the in-plane result. Checked on a PEC block
filling cells 2..5 in all three axes: the sliced mask comes out as nodes 2..6 in
both transverse axes for `normal='x'`, `'y'` and `'z'` alike — the correct closed
node box, with no dependence on the normal. So the slice-then-use order is safe;
what remains unverified is the conformal equivalence below.

## Risks, and why this was not fixed on the spot

The mode solver is the most heavily validated module in the repo, and this
change moves every staircase-path number it produces. Specifically:

- **Z₀ and C shift by several percent** on any staircase model. Any test with a
  hard-coded expected impedance will move. That is the *point* of the fix, but
  each such test needs re-deriving rather than re-baselining, or the fix cannot
  be distinguished from a regression.
- **The conformal validation must not move at all.** `tests/test_conformal_mode_solver.py`
  holds the −0.8% Z₀ result on the reference coax; if that changes by any amount,
  the equivalence assumed above is false and the change is wrong.
- **`tests/test_homogeneous_fill.py`** is the sharpest available gate — ε_eff must
  stay exactly ε_r regardless of conductor geometry. It caught two solver bugs
  before and should be run first.
- **The launched port profile** feeds `build_port_kernel` and the TEM/Spice
  ports. Enlarging the conductor by half a cell changes which edges carry the
  launched ê, so `tests/test_directional_launch.py`, `test_tem_port_impedance.py`
  and `test_modal_port*.py` all need to pass unchanged in the conformal case and
  be re-reasoned in the staircase case.

Suggested order: (1) confirm the conformal equivalence, (2) change the staircase
branch, (3) run the conformal suite expecting *zero* movement, (4) re-derive the
staircase expectations analytically, not from the new output.

## What is definitely not affected

- The conformal (Dey–Mittra) path. It already builds its node mask from cut
  geometry via `_conformal_node_pec`, which is why that path validates to −0.8%.
- The FDTD update itself. `apply_pec_mask` has used the dilation since the fix
  its docstring describes.
- `wavesim/electrostatics.py`, which took the correct route from the start —
  this discrepancy is how the issue was found.

## See also

- `wavesim/pec.py::apply_pec_mask` — the same trap, already fixed there, with
  the measured cost of getting it wrong.
- `wavesim/parts.py` — `pec_node_mask` and why connectivity is read off the edge
  masks rather than off `pec_mask`.
- `tests/test_electrostatics.py::test_both_solvers_bracket_the_analytic_coax` —
  a `@pytest.mark.slow` test that pins the current discrepancy, so it is
  recorded rather than hidden. **Delete or invert it when this is fixed**, since
  the two solvers should then agree closely instead of bracketing.
