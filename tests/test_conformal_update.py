"""S2 — the conformal (Dey–Mittra) Faraday update, NumPy reference.

The H update becomes the same contour integral with the covered parts of the
contour removed and the open face area as the denominator. Two properties carry
the whole design:

  * **absent fractions ⇒ bit-identical** to the staircase path (plan V2), which
    is a dispatch property and therefore exact;
  * **all-ones fractions ⇒ algebraically identical**, which is only true to
    round-off, because ``(E₁·d − E₀·d)/(d·d')`` and ``(E₁ − E₀)/d'`` do not
    agree bit-for-bit in floating point.

The E update is unchanged in form: E on a partially covered edge is the unknown
representing the field on the *open* part of that edge.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import MU0
from wavesim.grid import create_grid, create_grid_rectilinear
from wavesim.update import update_H, update_E
from wavesim.pml import init_cpml, update_H_pml, update_E_pml


FRACTION_KEYS = ('pec_edge_open_x', 'pec_edge_open_y', 'pec_edge_open_z',
                 'pec_face_open_x', 'pec_face_open_y', 'pec_face_open_z')


def _make(shape=(9, 8, 7), uniform=True, conformal=None, seed=0, fields=True):
    """Grid with reproducible random fields; ``conformal`` is a dict of overrides
    on the all-ones fractions, or None for the staircase path."""
    nx, ny, nz = shape
    if uniform:
        grid = create_grid(nx, ny, nz, 1e-3, 1.3e-3, 0.7e-3)
    else:
        def ax(n, d):
            return np.concatenate([[0.0], np.cumsum(d * 1.12 ** np.arange(n))])
        grid = create_grid_rectilinear(ax(nx, 1e-3), ax(ny, 1.3e-3), ax(nz, 0.7e-3))

    if fields:
        rng = np.random.default_rng(seed)
        for nm in ('Ex', 'Ey', 'Ez'):
            getattr(grid, nm)[...] = rng.standard_normal(getattr(grid, nm).shape)

    ones = lambda: np.ones(shape)
    kw = {}
    if conformal is not None:
        kw = {k: np.ones(shape, dtype=np.float32) for k in FRACTION_KEYS}
        for k, v in conformal.items():
            kw[k] = kw[k].copy()
            kw[k][v[0]] = v[1]
    ws.set_material_arrays(grid, ones(), ones(), ones(), ones(), ones(), ones(), **kw)
    return grid


def _H(grid):
    return {n: getattr(grid, n).copy() for n in ('Hx', 'Hy', 'Hz')}


# ---------------------------------------------------------------------- #
# V2 — the legacy path is untouched
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "graded"])
def test_absent_fractions_are_bit_identical(uniform):
    """Plan V2. No fraction arrays ⇒ the staircase branch runs, unmodified."""
    a, b = _make(uniform=uniform), _make(uniform=uniform)
    update_H(a)
    update_H(b)
    for n, v in _H(a).items():
        assert np.array_equal(v, getattr(b, n)), n


# ---------------------------------------------------------------------- #
# All-ones reduces to the staircase form
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "graded"])
@pytest.mark.parametrize("shape", [(9, 8, 7), (9, 8, 1)], ids=["3d", "nz1"])
def test_all_ones_matches_the_staircase_update(uniform, shape):
    """Includes the Nz=1 fast path, which has its own branch in both forms."""
    ref = _make(shape, uniform=uniform)
    got = _make(shape, uniform=uniform, conformal={})
    update_H(ref)
    update_H(got)
    for n, v in _H(ref).items():
        scale = np.abs(v).max()
        assert scale > 0.0
        assert np.abs(v - getattr(got, n)).max() < 8 * np.spacing(scale), n


def test_all_ones_matches_across_a_full_run_with_pml():
    """Round-off does not compound: the two paths track each other over many
    steps, through the PML correction as well as the interior kernel."""
    def run(conformal):
        grid = _make((20, 18, 16), conformal={} if conformal else None, fields=False)
        cpml = init_cpml(grid, d_pml=5)
        i, j, k = grid.Nx // 2, grid.Ny // 2, grid.Nz // 2
        for n in range(60):
            update_H(grid); update_H_pml(grid, cpml)
            update_E(grid); update_E_pml(grid, cpml)
            grid.Ez[i, j, k] += float(np.exp(-((n - 20) / 6.0) ** 2))
        return grid

    ref, got = run(False), run(True)
    assert np.abs(ref.Ez).max() > 1.0
    # Scale per field *kind*, not per component, so a component that happens to
    # be numerical grass is not normalised against itself. 60 steps of 1–2 ULP
    # per-step disagreement accumulates to ~1e-12 relative; the bound is set an
    # order above that, and still ~6 orders below anything physical.
    for kind in (('Ex', 'Ey', 'Ez'), ('Hx', 'Hy', 'Hz')):
        scale = max(np.abs(getattr(ref, n)).max() for n in kind)
        for n in kind:
            a, b = getattr(ref, n), getattr(got, n)
            assert np.abs(a - b).max() <= 1e-11 * scale, n


# ---------------------------------------------------------------------- #
# The conformal terms themselves
# ---------------------------------------------------------------------- #

def test_cut_edge_scales_its_contour_contribution():
    """Halving one edge's open length halves that edge's term in its face's
    contour integral — and touches nothing else."""
    ref = _make(conformal={})
    cut = _make(conformal={'pec_edge_open_y': ((3, 2, 1), 0.5)})
    update_H(ref)
    update_H(cut)

    g = ws.conformal_geometry(cut)
    dt = cut.dt
    # Going counterclockwise about +z, the Hz face at [i,j,k] carries Ey[i+1,j,k]
    # as its right edge (+Ey·Ly in the contour) and Ey[i,j,k] as its left edge
    # (−Ey·Ly). ΔHz = −(dt/(μ·A))·∮, so shortening Ey[3,2,1] by half raises the
    # face that has it on the right and lowers the one that has it on the left.
    for (i, j, k), sign in (((2, 2, 1), +1.0), ((3, 2, 1), -1.0)):
        delta = sign * (dt / MU0) * 0.5 * cut.Ey[3, 2, 1] * cut.dyp[2] * g.inv_Az[i, j, k]
        assert cut.Hz[i, j, k] == pytest.approx(ref.Hz[i, j, k] + delta, rel=1e-12)

    # An Hy contour is (Ex, Ex, Ez, Ez) — no Ey at all — so a y-edge cut cannot
    # reach it. (Hx *is* affected: its contour carries Ey[i,j,k] and Ey[i,j,k+1].)
    assert np.array_equal(ref.Hy, cut.Hy)
    assert ref.Hz[4, 2, 1] == cut.Hz[4, 2, 1]      # no other Hz face touches it


def test_smaller_open_area_amplifies_the_update():
    """The defining property of a cut cell: the same contour drives a smaller
    face harder, because A_open is the denominator."""
    ref = _make(conformal={})
    cut = _make(conformal={'pec_face_open_z': ((3, 2, 1), 0.25)})
    update_H(ref)
    update_H(cut)
    dH_ref = ref.Hz[3, 2, 1]
    dH_cut = cut.Hz[3, 2, 1]
    assert dH_ref != 0.0
    assert dH_cut == pytest.approx(dH_ref / 0.25, rel=1e-12)


def test_fully_covered_face_freezes_h_without_dividing_by_zero():
    """A_open = 0 must give inv_A = 0 (H frozen inside the metal), not inf/nan."""
    cut = _make(conformal={'pec_face_open_z': ((3, 2, 1), 0.0)})
    update_H(cut)
    assert cut.Hz[3, 2, 1] == 0.0
    for n in ('Hx', 'Hy', 'Hz'):
        assert np.all(np.isfinite(getattr(cut, n))), n


def test_fully_covered_edge_drops_out_of_the_contour():
    """L = 0 removes that edge's term entirely — the conformal analogue of the
    staircase mask zeroing E on the edge."""
    zeroed = _make(conformal={'pec_edge_open_y': ((3, 2, 1), 0.0)})
    killed = _make()
    killed.Ey[3, 2, 1] = 0.0                      # same thing, done to the field
    ws.set_material_arrays(killed, *[np.ones((9, 8, 7))] * 6,
                           **{k: np.ones((9, 8, 7), np.float32) for k in FRACTION_KEYS})
    update_H(zeroed)
    update_H(killed)
    for n in ('Hx', 'Hy', 'Hz'):
        assert np.abs(getattr(zeroed, n) - getattr(killed, n)).max() == 0.0, n


# ---------------------------------------------------------------------- #
# PML (plan R3 — conformal inside the absorbing shell)
# ---------------------------------------------------------------------- #

def test_pml_correction_uses_the_conformal_derivative():
    """A cut inside the absorbing shell must reach the psi recursion; if the PML
    kept the staircase derivative, psi would be identical to the uncut run."""
    def step(fraction):
        conf = {} if fraction is None else {'pec_edge_open_z': ((2, 3, 5), fraction)}
        grid = _make((20, 18, 16), conformal=conf, seed=3)
        cpml = init_cpml(grid, d_pml=6)
        update_H(grid)
        update_H_pml(grid, cpml)
        return cpml.psi_Ez_y.copy()

    assert not np.array_equal(step(None), step(0.5))


def test_pml_is_unaffected_by_a_cut_outside_the_shell():
    """Sanity on the previous test: an interior cut leaves the psi slabs alone,
    so what it detected really was the shell cell."""
    def step(fraction):
        conf = {} if fraction is None else {'pec_edge_open_z': ((10, 9, 8), fraction)}
        grid = _make((20, 18, 16), conformal=conf, seed=3)
        cpml = init_cpml(grid, d_pml=6)
        update_H(grid)
        update_H_pml(grid, cpml)
        return cpml.psi_Ez_y.copy()

    assert np.array_equal(step(None), step(0.5))
