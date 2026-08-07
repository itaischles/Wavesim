# The mode solver's staircase conductor was half a cell too small

**Status:** fixed 2026-08-07. Found 2026-08-05 while validating the electrostatic
solver (`wavesim/electrostatics.py`) against `solve_tem_modes`.
**Affected:** `wavesim/mode_solver.py`, staircase (non-conformal) path only.
**Did not affect:** the conformal path, which is unmoved to the last digit.

---

## The finding

`solve_tem_modes` built its conductor mask by slicing the cell-centred
`grid.pec_mask` and using the result as if it were indexed by *node*:

```python
# wavesim/mode_solver.py, before the fix
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
cells 2..5 physically spans `x[2] … x[6]`; this marked nodes 2..5, putting the far
surface at `x[5]`. **Every conductor was one cell short along each axis, on its
high side only** — so it was both undersized and asymmetric.

The degenerate case makes it plainest: a conductor one cell thick became a
node mask one node thick, i.e. a sheet of zero thickness sitting on its low face.

This is precisely the trap `wavesim/pec.py::apply_pec_mask` documents for the
FDTD update itself, where the same mistake once cost 6.8% in ε_eff on an
RG58-like coax. The fix there was to zero an E-edge when *any* of the four cells
touching it is PEC (`build_pec_edge_masks`, a dilation in the two perpendicular
axes). The mode solver never received the equivalent treatment on its staircase
path.

## Why this counted as a bug rather than a choice

The module's own design principle, from its docstring:

> When the grid carries cut-cell open fractions the solver switches to them
> wholesale — conductor mask, stencil, energy integral and launched ê — so the
> port is solved on the *same* geometry the FDTD steps. Without that the Z₀ the
> port presents stops being the Z₀ the run presents.

That reasoning applies just as much to the staircase path, and there it was not
honoured: the FDTD's staircase conductor (after the `build_pec_edge_masks`
dilation) was *not* the mode solver's staircase conductor.

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
print(m.pec[6, :].astype(int))             # what the mode solver pins
print(pec_node_mask(g)[6, :, 0].astype(int))   # what the FDTD actually shorts
```

Before — the mode solver spread the drop over **13** intervals against a true gap
of **12**, i.e. `12/13` of the correct capacitance, **−7.7%** at this mesh:

```
mode solver phi : 0 0 0 .0769 .1538 ... .9231 1 1 1 1 1     <- 1/13 per step
electrostatics  : 0 0 0 0 .0833 .1667 ... .9167 1 1 1 1 1   <- 1/12 per step
mode solver pins: 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0   <- one node per strip
FDTD shorts     : 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0   <- both surfaces
```

After — the two solvers agree exactly, both on φ and on the pinned set.

## The fix

Two changes, and the second is the one that was not obvious.

**1. The conductor mask** (`solve_tem_modes`) — the node mask derived from the
FDTD's own edge masks:

```python
else:
    pec_full = _slice(pec_node_mask(grid), normal, k)
```

`pec_node_mask` calls `build_conformal_edge_masks` when the grid carries cut
cells and `build_pec_edge_masks` otherwise. It is only ever reached on the
staircase branch, so which rule it picks is not load-bearing here.

**2. The launched ê** (`TEMMode._staggered_port_fields`) — masked on the *edges*
the run zeroes rather than on the nodes φ is pinned at:

```python
m_a, m_b = _plane_edge_pec(grid, cfg, self.normal, k)
Ea[m_a] = 0.0
Eb[m_b] = 0.0
```

The two sets are not the same, and the difference is exactly what change 1
exposed. A node on a conductor surface owns a live edge running out into the gap
— on a coax that edge carries the **largest field on the plane** — and the old
`Ea[self.pec] = 0.0` deleted it as soon as `pec` started marking surface nodes.
With change 1 alone the launched/absorbed profile stopped being the mode the run
carries: the reference-coax `ModalPort` termination went from −34 dB to −13.5 dB,
and no admittance scale could recover it (sweeping `s` over ±20% moved the floor
by less than 1 dB — it was a shape error, not an amplitude one). With both
changes it is **−64 dB**.

The same change makes `numerical_admittance_scale` collapse to 1 on the staircase
path, which is a derivation rather than a coincidence: `G` sums `ê²` over the
edges and `Z₀` sums `(Δφ)²` over the face coefficients of those same edges, so
`s = 1/(Z₀·G)` is 1 exactly once `ê` vanishes on exactly the edges whose
coefficient contributes nothing. The residual 3.2e-9 is tabulated η₀ against
1/(ε₀c₀). The staircase path used to read 1.0058, and that was this mismatch —
not the discretisation floor it was recorded as.

`Ha[pec]`/`Hb[pec]` are gone too: Ĥ is built from the already-masked ê and
inherits its zeros, so masking it again could only ever remove more than the run
does.

### The conformal branch was *not* collapsed into it

The proposal was to let `pec_node_mask` serve both branches. It agrees with
`_conformal_node_pec` on the reference coax at 1.5 mm and 0.75 mm (identical node
sets), but the two rules are not the same rule: `pec_node_mask` also counts the
**longitudinal** edge, so it marks a node that lies strictly inside metal even
when all four transverse edges meeting it are merely partially covered. A
sub-cell blob sitting on a node separates them (checked: `_conformal_node_pec`
marks nothing, `pec_node_mask` marks the node). The conformal branch deliberately
leaves such a node free — its `1/L` weight already places it the correct sub-cell
distance from the metal — so the two branches stay separate.

## What moved

The mode solver's staircase Z₀ now *is* the Z₀ the line presents. Mid-line V/I on
the reference coax (`tests/reference_coax.py`), which is independent of the mask
convention:

| cell | measured V/I | mode Z₀ (new) | error | mode Z₀ (old) | error |
|---:|---:|---:|---:|---:|---:|
| 0.50 mm | 61.53 Ω | 62.42 Ω | +1.4% | 70.39 Ω | +13.4% |
| 0.25 mm | 64.07 Ω | 64.17 Ω | +0.2% | 68.24 Ω | +6.1% |

Against the *analytic* 65.871 Ω the staircase column changed sign, from +6.86% to
−5.24% at 0.5 mm: the conductor the run steps is the slightly fat one, and the
solver now reports the line's own error instead of cancelling it against its own.

Other measured effects, all improvements:

- `ModalPort` termination on the staircase coax: −34 dB → **−64 dB**.
- `TEMMode.to_source` launch calibration: 0.953 → **1.002** forward volts.
- Loop-independence of the launched Ĥ (`∮Ĥ·dl·Z₀` on ±8/10/12-cell contours):
  spread 3.6e-3 → **8.5e-5**, now better than the conformal path's 1.3e-4. The
  old drift was read as spurious staircase curl; it was the undersized conductor.
- `numerical_admittance_scale` on the staircase path: 1.0058 → **1 − 3.2e-9**.
- The 2D mode solver and the 3D electrostatic solver now return the **same** C′
  on the same coax, to ~4e-15 relative — they used to bracket the analytic value.

### One thing got stricter: a one-cell gap

`solve_tem_modes` labels conductors with `ndimage.label` on the node mask, and
two conductors separated by exactly one cell of dielectric now have
*index-adjacent* surface nodes — cells 0..8 own nodes 0..9, cells 10.. own nodes
10.. — so the labelling fuses them. On the repro geometry above with the gap
narrowed to one cell the solver warns "found 1 conductor(s)" and returns no
modes, where before it returned a mode with a (doubly wrong) 2-cell gap. Two and
three-cell gaps are unaffected.

The conformal path has always had this property, and `_conformal_node_pec`
documents it: "two conductors closer than one cell would be merged by the index
adjacency". It is a loud failure rather than a silent wrong answer, and a
one-cell gap carries no useful capacitance anyway, so it is recorded here rather
than worked around. The real fix, if it is ever wanted, is the one
`wavesim/parts.py::conductor_bodies` already uses: label the **covered-edge
graph** instead of the node mask, which keeps the two apart because the edge in
the gap is not zeroed. That would change conductor labelling on the conformal
path too, so it is a separate decision.

## Tests re-derived

Nothing was re-baselined from the new output; each expectation below was
re-derived and then checked.

- `tests/test_conformal_mode_solver.py::_binary_fractions` — now states the
  staircase conductor as 0/1 fractions using `build_pec_edge_masks`, the rule the
  run applies. `binary` is once again bit-identical to `staircase`, which is what
  makes it a reduction test.
- `…::test_node_mask_round_trips_the_edge_fractions` — round-trips against
  `pec_node_mask` instead of `pec_mask`.
- `…::test_admittance_scale_is_unity_for_a_homogeneous_fill` — asserts 1 on
  *both* paths now (see the derivation above) instead of a staircase/conformal
  contrast.
- `…::test_launched_h_sheet_carries_the_exact_modal_current` — asserts
  loop-independence for both paths; the "staircase drifts 10× more" contrast is
  gone because it is no longer true.
- `tests/test_mode_solver_spacing.py::_parallel_plate` — returns the two metal
  *face nodes*, so `d = y[hi] − y[lo]` is the true separation. The exact
  `C = ε₀ε_r·W/d` gate holds again at `rel=1e-12`.
- `tests/test_electrostatics.py::test_both_solvers_bracket_the_analytic_coax` →
  `…agree_on_the_coax_to_round_off`, inverted as the doc's own note asked.
- `tests/test_conformal_mode_solver.py` conformal numbers are unchanged to every
  digit recorded (−0.79 / −0.50 / −0.27 / −0.17 %), which was the condition for
  believing any of the above.

Full suite: 306 passed.

## What was definitely not affected

- The conformal (Dey–Mittra) path.
- The FDTD update itself. `apply_pec_mask` has used the dilation since the fix
  its docstring describes.
- `wavesim/electrostatics.py`, which took the correct route from the start —
  this discrepancy is how the issue was found.

## See also

- `wavesim/pec.py::apply_pec_mask` — the same trap, with the measured cost of
  getting it wrong.
- `wavesim/parts.py` — `pec_node_mask` and why connectivity is read off the edge
  masks rather than off `pec_mask`.
