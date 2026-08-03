"""S7 — make the conformal clamp threshold safe by itself.

S4's threshold bounds the ``1/A_open`` coefficient, but 0.4 is not safe on real
geometry: the plan's reference coax diverges at a 0.25 mm transverse cell with no
sources and no ports, while the *finer* 0.1875 mm mesh runs. Stability is not
monotone in resolution, so the setting cannot be reasoned about — it has to be
measured, which is what :mod:`wavesim.stability` does.

Two measurements, deliberately independent, and the tests below pin them against
each other:

* :func:`probe_growth` seeds noise and steps the real scheme;
* :func:`max_stable_dt` takes ``2/√λ_max`` of the discrete curl-curl by Lanczos.

Measured on the reference coax through both (the grids are far too large for this
suite; the numbers are recorded in the module docstring of ``wavesim/stability``):

    0.5000 mm  thr 0.4    margin 1.00585   growth 1.000   stable
    0.2500 mm  thr 0.4    margin 0.99795   growth 1.1358  diverges
    0.2500 mm  thr 0.5    margin 1.01072   growth 1.000   stable

and the auto-raise lands the 0.25 mm case on 0.5 — the same threshold the plan's
re-baseline table had to pick by hand for that row.
"""

import warnings

import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import C0
from wavesim.grid import create_grid
from wavesim.stability import GROWTH_TRIGGER, _pin_unupdated_edges

from test_conformal_threshold import _cut_grid


def _vacuum(n=16, ds=1e-3):
    grid = create_grid(n, n, n, ds)
    ws.set_vacuum(grid)
    return grid


# ---------------------------------------------------------------------- #
# The fast probe
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("grid_fn", [
    _vacuum,
    lambda: _cut_grid(0.05, threshold=0.4),
    lambda: _cut_grid(0.20, threshold=0.4),
])
def test_probe_is_quiet_on_a_stable_grid(grid_fn):
    """A lossless leapfrog conserves energy, so the fit must read 1.

    Measured floor across these three and a PEC-walled box: |growth - 1| <=
    7.4e-6, against the 5e-5 trigger.
    """
    probe = ws.probe_growth(grid_fn())
    assert probe.stable
    assert abs(probe.growth - 1.0) < GROWTH_TRIGGER
    assert probe.margin is None       # a stable run bounds dt from below only


def test_probe_catches_the_unclamped_sliver():
    """The S4 failure mode, detected in 100 steps instead of 4000."""
    probe = ws.probe_growth(_cut_grid(0.05, threshold=0.0))
    assert not probe.stable
    assert probe.growth > 10.0
    assert probe.steps < ws.stability.PROBE_STEPS      # bailed out early


def test_probe_leaves_the_grid_alone():
    """Fields, threshold and geometry all survive the measurement."""
    grid = _cut_grid(0.05, threshold=0.4)
    grid.Ez[3, 4, 5] = 1.25
    ws.probe_growth(grid)
    assert grid.Ez[3, 4, 5] == 1.25
    assert np.count_nonzero(grid.Ez) == 1
    assert not np.any(grid.Hx)
    assert grid.conformal_area_threshold == 0.4
    assert grid.time_step == 0


def test_pinned_edges_are_exactly_the_ones_no_update_writes():
    """``update_E`` writes Ex[:, 1:, 1:] and cyclic partners.

    Seeding the rest is not harmless: a frozen E edge feeds Faraday every step
    and ramps H linearly, which read as 9x energy growth over 500 steps on an
    open grid before this was pinned.
    """
    grid = _vacuum()
    grid.Ex[...] = grid.Ey[...] = grid.Ez[...] = 1.0
    _pin_unupdated_edges(grid)
    assert not np.any(grid.Ex[:, 0, :]) and not np.any(grid.Ex[:, :, 0])
    assert not np.any(grid.Ey[0, :, :]) and not np.any(grid.Ey[:, :, 0])
    assert not np.any(grid.Ez[0, :, :]) and not np.any(grid.Ez[:, 0, :])
    assert np.all(grid.Ex[:, 1:, 1:] == 1.0)
    assert np.all(grid.Ey[1:, :, 1:] == 1.0)
    assert np.all(grid.Ez[1:, 1:, :] == 1.0)


# ---------------------------------------------------------------------- #
# The definition, and that the two agree
# ---------------------------------------------------------------------- #

def test_max_stable_dt_reproduces_the_courant_condition():
    """Uniform vacuum: ``dt_max = h/(c√3)``, i.e. λ_max = 12c²/h².

    Comes out 0.5% high because the domain is finite — the outermost slabs are
    frozen, which puts λ_max just below the infinite-grid value. That is the
    operator the scheme actually steps, so it is the right answer.
    """
    grid = _vacuum(20)
    assert ws.max_stable_dt(grid) == pytest.approx(1e-3 / (C0 * np.sqrt(3)),
                                                   rel=0.01)
    assert ws.stability_margin(grid) == pytest.approx(1 / 0.99, rel=0.01)


def test_growth_rate_and_eigenvalue_agree_on_the_unstable_grid():
    """The cross-check that makes either measurement trustworthy.

    ``z + 1/z = 2 - dt²λ`` ties the per-step amplification of a diverging run to
    the eigenvalue of the operator that produced it, so a time-domain run and a
    Lanczos solve must land on the same margin. Measured 0.52139 against
    0.52182 here, and 0.99802 against 0.99795 on the reference coax at 0.25 mm —
    the case this whole section exists for.
    """
    grid = _cut_grid(0.05, threshold=0.0)
    probe = ws.probe_growth(grid)
    assert probe.margin == pytest.approx(ws.stability_margin(grid), rel=2e-3)
    assert probe.margin < 1.0


def test_margins_bracket_the_threshold_that_fixes_the_sliver():
    """Below the clamp the operator is over the limit; above it, under."""
    assert ws.stability_margin(_cut_grid(0.05, threshold=0.0)) < 1.0
    assert ws.stability_margin(_cut_grid(0.05, threshold=0.4)) > 1.0


# ---------------------------------------------------------------------- #
# The remedy
# ---------------------------------------------------------------------- #

def test_safe_area_threshold_climbs_until_the_probe_is_quiet():
    grid = _cut_grid(0.05, threshold=0.0)
    threshold, first = ws.safe_area_threshold(grid)
    assert threshold > 0.0
    assert not first.stable            # reports what was wrong, not the rung that worked
    assert grid.conformal_area_threshold == 0.0     # decided, not applied
    grid.conformal_area_threshold = threshold
    assert ws.probe_growth(grid).stable


def test_auto_raise_fixes_an_unstable_grid_and_says_so():
    grid = _cut_grid(0.05, threshold=0.0)
    with pytest.warns(RuntimeWarning, match="clamp threshold raised"):
        ws.Simulation(grid)
    assert grid.conformal_area_threshold > 0.0
    assert ws.probe_growth(grid).stable


def test_auto_raise_is_a_no_op_when_the_threshold_already_works():
    """The check must be invisible to every run that was already fine — it may
    cost time, never accuracy."""
    grid = _cut_grid(0.05, threshold=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # any warning here is a failure
        ws.Simulation(grid)
    assert grid.conformal_area_threshold == 0.4


def test_staircase_grids_are_never_probed(monkeypatch):
    """No cut cells, no cost: the S7 machinery must not touch a legacy run."""
    import wavesim.stability as stability

    def explode(*args, **kwargs):
        raise AssertionError("a staircase grid was probed")

    monkeypatch.setattr(stability, "probe_growth", explode)
    monkeypatch.setattr(stability, "ensure_stable_threshold", explode)
    ws.Simulation(_vacuum())


def test_off_leaves_an_unstable_grid_exactly_as_it_was():
    grid = _cut_grid(0.05, threshold=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ws.Simulation(grid, conformal_stability='off')
    assert grid.conformal_area_threshold == 0.0


def test_warn_reports_without_changing_anything():
    grid = _cut_grid(0.05, threshold=0.0)
    with pytest.warns(RuntimeWarning, match="diverges at clamp threshold"):
        ws.Simulation(grid, conformal_stability='warn')
    assert grid.conformal_area_threshold == 0.0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="conformal_stability"):
        ws.Simulation(_cut_grid(0.05, threshold=0.4), conformal_stability='yes')


def test_cuda_conformal_is_refused_rather_than_mismeasured():
    """A CUDA probe would step the staircase kernel (R7's CudaResident hole) and
    report a reassuring 1.000 for a conformal grid whose conformal update never
    ran. Refuse instead — the backend's own guard, raised one step earlier."""
    pytest.importorskip("wavesim.backend_cuda")
    with pytest.raises(NotImplementedError, match="conformal"):
        ws.Simulation(_cut_grid(0.05, threshold=0.4), backend='cuda')


def test_clamping_cannot_always_save_it():
    """A grid that is unstable for a reason clamping does not reach must fail
    loudly rather than climb to a threshold that ruins the geometry."""
    grid = _cut_grid(0.05, threshold=0.4)
    grid.dt *= 2.0                      # nothing to do with cut cells
    with pytest.raises(RuntimeError, match="unstable at every clamp threshold"):
        ws.safe_area_threshold(grid)
