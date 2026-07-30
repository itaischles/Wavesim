"""S4 — the small-cut area threshold, and the conformal reference case.

``dt/(μ·A_open)`` diverges as a cut cell shrinks, so faces whose open fraction
falls below ``grid.conformal_area_threshold`` are treated as fully PEC. The
timestep is deliberately left alone: reducing dt would perturb every existing
result and ``summary["dt"]``.

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


def test_sliver_face_is_suppressed_and_counted():
    grid = _cut_grid(0.05, threshold=0.4)
    g = conformal_geometry(grid)
    assert g.inv_Az[8, 8, 8] == 0.0            # treated as fully PEC
    assert g.Az[8, 8, 8] > 0.0                 # the area itself is untouched
    assert g.n_suppressed == 1


def test_face_above_threshold_is_untouched():
    grid = _cut_grid(0.45, threshold=0.4)
    g = conformal_geometry(grid)
    assert g.inv_Az[8, 8, 8] == pytest.approx(1.0 / g.Az[8, 8, 8], rel=1e-12)
    assert g.n_suppressed == 0


def test_threshold_zero_keeps_every_cut_but_still_guards_zero_area():
    grid = _cut_grid(0.0, threshold=0.0)
    g = conformal_geometry(grid)
    assert g.inv_Az[8, 8, 8] == 0.0            # fully covered, not 1/0
    assert g.n_suppressed == 0                 # nothing *thresholded* away
    assert np.all(np.isfinite(g.inv_Az))


def test_threshold_change_invalidates_the_cache():
    grid = _cut_grid(0.2, threshold=0.4)
    assert conformal_geometry(grid).n_suppressed == 1
    grid.conformal_area_threshold = 0.1
    assert conformal_geometry(grid).n_suppressed == 0


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
    assert conformal_geometry(grid).n_suppressed > 0


@pytest.mark.slow
def test_conformal_coax_runs_stably_and_records_eps_eff():
    """Homogeneous fill ⇒ LC = με ⇒ v = c/√ε_r whatever the conductor geometry.

    Stability is the S4 gate and is asserted tightly. ε_eff is only *recorded*
    here, because it cannot be exact yet: H integrates the conformal contour
    while ``apply_pec_mask`` still zeroes E by the staircase dilation, so E and H
    see different geometry — the very inconsistency S3 exists to remove. Measured
    today: staircase +0.29%, conformal +1.83% (threshold 0.4), +2.91% (0.5).

    The bound below is a regression floor, not a target. **S3 must bring this to
    the staircase level or better**; if it does not, S3 is wrong.
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
    assert eps_eff == pytest.approx(eps_r, rel=0.03), (
        f"eps_eff {eps_eff:.4f} vs {eps_r} (v = {v_meas:.4e} m/s)")
