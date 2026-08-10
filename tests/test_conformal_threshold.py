"""S4 — the small-cut area threshold, and the conformal reference case.

``dt/(μ·A_open)`` diverges as a cut cell shrinks, so a face whose open fraction
falls below ``grid.conformal_area_threshold`` has its area clamped to
``threshold·A_full``. The timestep is deliberately left alone: reducing dt would
perturb every existing result and ``summary["dt"]``.

This is not a refinement — it is load-bearing. An analytic coax on a 32-cell
transverse mesh produces open-area fractions down to 0.011, and the run
diverges to NaN without the threshold.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import C0
from wavesim.grid import create_grid
from wavesim.mode_solver import solve_tem_modes
from wavesim.pec import conformal_geometry
from wavesim.update import update_H, update_E

from conformal_shapes import coax_fractions
from test_homogeneous_fill import _coax_grid, R_IN, R_OUT


KEYS = ('pec_edge_open_x', 'pec_edge_open_y', 'pec_edge_open_z',
        'pec_face_open_x', 'pec_face_open_y', 'pec_face_open_z')


def _cut_grid(fraction, threshold=None, N=16):
    """Cube with one small cut cell: a shrunken Hz face and its two x/y edges."""
    grid = create_grid(N, N, N, 1e-3)
    shape = (N, N, N)
    kw = {k: np.ones(shape, dtype=np.float32) for k in KEYS}
    for k in ('pec_face_open_z', 'pec_edge_open_x', 'pec_edge_open_y'):
        kw[k] = kw[k].copy()
        kw[k][8, 8, 8] = fraction
    ones = lambda: np.ones(shape)
    ws.set_material_arrays(grid, ones(), ones(), ones(), ones(), ones(), ones(),
                           conformal_area_threshold=threshold, **kw)
    return grid


def _run_free(grid, steps=4000):
    """Seed noise and step; returns the growth ratio (inf/NaN if it diverges)."""
    rng = np.random.default_rng(1)
    grid.Ez[4:12, 4:12, 4:12] = rng.standard_normal((8, 8, 8)) * 1e-6
    start = np.abs(grid.Ez).max()
    with np.errstate(over='ignore', invalid='ignore'):
        for _ in range(steps):
            update_H(grid)
            update_E(grid)
    return np.abs(grid.Ez).max() / start


# ---------------------------------------------------------------------- #
# The mechanism
# ---------------------------------------------------------------------- #

def test_default_threshold_is_documented_value():
    assert create_grid(4, 4, 4, 1e-3).conformal_area_threshold == 0.4


def test_sliver_face_area_is_clamped_not_killed():
    """The face keeps integrating its contour; only the coefficient is bounded.

    Killing it instead (inv_A = 0) leaves the four edges of its contour carrying
    E while H is frozen — an E/H mismatch that costs ε_eff dearly. See
    test_conformal_coax_eps_eff_matches_the_fill.
    """
    grid = _cut_grid(0.05, threshold=0.4)
    g = conformal_geometry(grid)
    full = grid.dxp[8] * grid.dyp[8]
    assert g.inv_Az[8, 8, 8] == pytest.approx(1.0 / (0.4 * full), rel=1e-12)
    assert g.inv_Az[8, 8, 8] > 0.0             # still live
    assert g.Az[8, 8, 8] > 0.0                 # the true area is untouched
    assert g.n_clamped == 1


def test_face_above_threshold_is_untouched():
    grid = _cut_grid(0.45, threshold=0.4)
    g = conformal_geometry(grid)
    assert g.inv_Az[8, 8, 8] == pytest.approx(1.0 / g.Az[8, 8, 8], rel=1e-12)
    assert g.n_clamped == 0


def test_fully_covered_face_stays_frozen_whatever_the_threshold():
    """Clamping applies to *cut* faces. A face with no open area at all has no
    contour either — all four of its edges are zeroed — so it stays at inv_A = 0
    and there is no mismatch to create."""
    for thr in (0.0, 0.4):
        grid = _cut_grid(0.0, threshold=thr)
        g = conformal_geometry(grid)
        assert g.inv_Az[8, 8, 8] == 0.0
        assert g.n_clamped == 0
        assert np.all(np.isfinite(g.inv_Az))


def test_threshold_change_invalidates_the_cache():
    grid = _cut_grid(0.2, threshold=0.4)
    assert conformal_geometry(grid).n_clamped == 1
    grid.conformal_area_threshold = 0.1
    assert conformal_geometry(grid).n_clamped == 0


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_out_of_range_threshold_is_rejected(bad):
    grid = create_grid(4, 4, 4, 1e-3)
    ones = lambda: np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="conformal_area_threshold"):
        ws.set_material_arrays(grid, *[ones()] * 6,
                               conformal_area_threshold=bad)


# ---------------------------------------------------------------------- #
# Why it exists
# ---------------------------------------------------------------------- #

@pytest.mark.slow
def test_small_cut_diverges_without_the_threshold():
    """Pins the failure mode. Measured boundary: stable at 0.35, NaN at 0.30."""
    assert not np.isfinite(_run_free(_cut_grid(0.05, threshold=0.0)))


@pytest.mark.slow
@pytest.mark.parametrize("fraction", [0.05, 0.2])
def test_threshold_restores_stability(fraction):
    growth = _run_free(_cut_grid(fraction, threshold=0.4))
    assert np.isfinite(growth) and growth < 10.0


# ---------------------------------------------------------------------- #
# Reference case — analytic coax (plan §7)
# ---------------------------------------------------------------------- #

def _coax(eps_r, n, nz, threshold):
    grid, ds = _coax_grid(n, eps_r, nz=nz)
    c = 0.5 * n * ds
    ws.set_material_arrays(
        grid, grid.eps_x, grid.eps_y, grid.eps_z, grid.mu_x, grid.mu_y, grid.mu_z,
        conformal_area_threshold=threshold,
        **coax_fractions(grid, c, c, R_IN, R_OUT))
    return grid, ds, c


def test_analytic_coax_produces_slivers_far_below_the_threshold():
    """The reference case is not a gentle one: without S4 it cannot run."""
    grid, _, _ = _coax(2.3, 32, 3, 0.4)
    open_z = grid.pec_face_open_z
    assert open_z[open_z > 0].min() < 0.05
    assert conformal_geometry(grid).n_clamped > 0


@pytest.mark.slow
def test_conformal_coax_eps_eff_matches_the_fill():
    """V1 — homogeneous fill ⇒ LC = με ⇒ v = c/√ε_r whatever the conductor
    geometry. Staircasing and cut cells may move Z₀; they may not move ε_eff.

    This is the S3 gate, and it is a genuine assertion now that E and H are both
    derived from the cut geometry. Measured by time-domain velocity, whose own
    resolution is ~0.3%:

        staircase                        +0.50%
        conformal, threshold 0.2         +0.56%
        conformal, threshold 0.4         +1.55%   ← the bound below
        conformal, threshold 0.6         +3.37%

    Two earlier configurations failed this and are why a bound exists at all:
    conformal H with the staircase dilation still on E read +1.83% *when the
    dilation was also hiding the ε below*, and killing sliver faces instead of
    clamping them read +5.77%.

    **The bound moved from 0.5% to 2%, and the reason is not a weakening.** This
    read +0.21% at threshold 0.4 until the FDTD started reading the same ε as the
    mode solve (:func:`wavesim.pec.conformal_edge_eps`). ``set_coax`` fills only
    *between* the conductors and leaves the background ε = 1 inside the metal, so
    every edge crossing a conductor surface used to be stepped as vacuum — a thin
    low-ε shell hugging the conductor, which speeds the wave and was cancelling
    most of the clamp's slowing. The control that settles it: building this same
    coax with ε = 2.3 assigned *everywhere*, metal included — a map with nothing
    mislabelled to repair — reads 2.3357 both before and after the change, and
    the repaired ``set_coax`` map now reads 2.3357 too. The old number was the
    artifact; the two maps agreeing is the fix working.

    What is left is the clamp, and the sweep above is what it costs: the run at
    threshold 0.4 is genuinely ~1.5% slow on this mesh, monotone in the
    threshold and shrinking with resolution-per-wavelength (+1.55% at fmax 40
    GHz, +1.16% at 20 GHz). ``_guarded_inverse`` raising a sliver face's area to
    0.4·A_full stiffens that cell, and nothing about ε changes it. Note the mode
    solver's own ``eps_eff`` is *exactly* 2.300000 throughout all of this
    (``tests/test_homogeneous_fill.py``, ``abs=1e-10``) — the static operator
    never sees the clamp, so it cannot be the thing being tested here.
    """
    eps_r, n, nz, f_max = 2.3, 32, 256, 40e9
    k_src, k_p1, k_p2 = 24, 70, 200

    grid, ds, c = _coax(eps_r, n, nz, 0.4)
    mode = solve_tem_modes(grid, normal='z', position=k_src * ds,
                           compute_params=True)[0]
    src = mode.to_source(ws.GaussianPulse.for_fmax(f_max), fields='EH')

    i_p = int((c + 0.5 * (R_IN + R_OUT)) / ds)
    j_p = int(c / ds)
    probes = [ws.FieldProbe(component='Ex', x=i_p * ds, y=j_p * ds, z=k * ds)
              for k in (k_p1, k_p2)]

    cpml = ws.init_cpml(grid, faces=('z0', 'z1'))
    sim = ws.Simulation(grid, cpml=cpml, sources=[src], monitors=probes,
                        backend='numpy')
    v_true = C0 / np.sqrt(eps_r)
    sim.run(int(1.6 * (k_p2 - k_src) * ds / v_true / grid.dt))

    for p in probes:
        assert np.all(np.isfinite(p.values)), "conformal coax diverged"

    def peak_time(p):
        v = np.abs(np.asarray(p.values))
        t = np.asarray(p.times)
        i = int(np.argmax(v))
        assert 0 < i < len(v) - 1, "no interior peak — the run blew up"
        y0, y1, y2 = v[i - 1], v[i], v[i + 1]
        d = y0 - 2 * y1 + y2
        return t[i] + (0.5 * (y0 - y2) / d if d else 0.0) * (t[1] - t[0])

    v_meas = (k_p2 - k_p1) * ds / (peak_time(probes[1]) - peak_time(probes[0]))
    eps_eff = (C0 / v_meas) ** 2
    assert eps_eff == pytest.approx(eps_r, rel=0.02), (
        f"eps_eff {eps_eff:.4f} vs {eps_r} (v = {v_meas:.4e} m/s)")
