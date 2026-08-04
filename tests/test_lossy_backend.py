"""The lossy E update in the Numba backend.

The NumPy implementation in :mod:`wavesim.update` / :mod:`wavesim.pml` stays the
oracle; these kernels must track it bit-for-bit. They can: the coefficients are
precomputed once in :mod:`wavesim.loss` and both backends read the *same* Ca/Cb
arrays, so there is no arithmetic left to regroup and no round-off to allow for.
That is a stronger guarantee than the conformal kernel offers (which agrees only
to 1-2 ULP, because it distributes a multiplication the reference does not).

Two ways this could go wrong quietly, and one test each:

  * the interior kernel ignores sigma and runs the model lossless;
  * the CPML correction keeps ``dt/(eps0*eps)`` where the interior uses ``Cb``,
    which shows up only inside the absorbing slabs.

Both are caught by comparing whole fields, PML shell included, rather than an
interior window.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim import backend_numba as nb
from wavesim.grid import create_grid
from wavesim.pml import init_cpml, update_E_pml, update_H_pml
from wavesim.update import update_E, update_H

FIELDS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def _lossy_grid(shape=(18, 17, 16), ds=0.8e-3, sigma=0.35, eps_r=2.5,
                seed=0, graded=False):
    """Random fields on a lossy grid; ``graded`` exercises the non-uniform path.

    The conductivity is deliberately *not* uniform — a constant sigma would let
    a kernel that indexed Ca/Cb wrongly still agree with the reference.
    """
    if graded:
        def ax(n, d):
            return np.concatenate([[0.0], np.cumsum(d * 1.1 ** np.arange(n))])
        grid = ws.create_grid_rectilinear(
            ax(shape[0], ds), ax(shape[1], ds * 1.3), ax(shape[2], ds * 0.7))
    else:
        grid = create_grid(*shape, ds, ds * 1.3, ds * 0.7)
    ws.set_vacuum(grid)

    rng = np.random.default_rng(seed)
    eps = eps_r + rng.random(shape)
    sig = sigma * rng.random(shape)
    ws.set_material_arrays(
        grid, eps, eps * 1.1, eps * 0.9,
        np.ones(shape), np.ones(shape), np.ones(shape),
        sigma_x=sig, sigma_y=sig * 1.4, sigma_z=sig * 0.6)
    for name in FIELDS:
        getattr(grid, name)[...] = rng.standard_normal(shape)
    return grid


def _snapshot(grid):
    return {n: getattr(grid, n).copy() for n in FIELDS}


def _assert_identical(a, b, what):
    for name in FIELDS:
        assert np.array_equal(a[name], b[name]), f"{what}: {name} differs"


# ---------------------------------------------------------------------- #
# Interior update
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize('shape', [(18, 17, 16), (20, 19, 1)])
def test_numba_lossy_update_matches_the_numpy_reference(shape):
    """Bit-for-bit, on the 3D path and on the Nz=1 fast path."""
    ref, got = _lossy_grid(shape), _lossy_grid(shape)
    for _ in range(3):
        update_H(ref);   update_E(ref)
        nb.update_H(got); nb.update_E(got)
    _assert_identical(_snapshot(ref), _snapshot(got), 'interior')


def test_numba_lossy_update_matches_on_a_graded_mesh():
    ref, got = _lossy_grid(graded=True), _lossy_grid(graded=True)
    for _ in range(3):
        update_H(ref);   update_E(ref)
        nb.update_H(got); nb.update_E(got)
    _assert_identical(_snapshot(ref), _snapshot(got), 'graded')


def test_numba_lossy_update_actually_damps():
    """Guard against a kernel that agrees with the reference by ignoring sigma.

    Without this, a `_update_E_lossy` that dropped the Ca term would still pass
    every comparison above if the reference had the same bug.
    """
    lossy, lossless = _lossy_grid(), _lossy_grid(sigma=0.0)
    for _ in range(20):
        nb.update_H(lossy);    nb.update_E(lossy)
        nb.update_H(lossless); nb.update_E(lossless)
    assert np.abs(lossy.Ez).max() < 0.98 * np.abs(lossless.Ez).max()


# ---------------------------------------------------------------------- #
# CPML correction
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize('shape', [(18, 17, 16), (20, 19, 1)])
def test_numba_lossy_pml_matches_the_numpy_reference(shape):
    """Whole fields, PML shell included — the slabs are where Cb could go missing."""
    ref, got = _lossy_grid(shape), _lossy_grid(shape)
    cref, cgot = init_cpml(ref, d_pml=4), init_cpml(got, d_pml=4)

    for _ in range(3):
        update_H(ref);    ref, cref = update_H_pml(ref, cref)
        update_E(ref);    ref, cref = update_E_pml(ref, cref)
        nb.update_H(got); got, cgot = nb.update_H_pml(got, cgot)
        nb.update_E(got); got, cgot = nb.update_E_pml(got, cgot)

    _assert_identical(_snapshot(ref), _snapshot(got), 'pml')
    for name in ('psi_Hz_y', 'psi_Hy_z', 'psi_Hx_z',
                 'psi_Hz_x', 'psi_Hy_x', 'psi_Hx_y'):
        assert np.array_equal(getattr(cref, name), getattr(cgot, name)), name


def test_lossless_numba_path_is_untouched():
    """A grid with no sigma must step exactly as it did before loss existed."""
    ref, got = _lossy_grid(sigma=0.0), _lossy_grid(sigma=0.0)
    ref.sigma_x = ref.sigma_y = ref.sigma_z = None      # the pre-loss grid
    ref._loss_cache = None
    assert not ref.is_lossy and got.is_lossy

    cref, cgot = init_cpml(ref, d_pml=4), init_cpml(got, d_pml=4)
    for _ in range(3):
        nb.update_H(ref); ref, cref = nb.update_H_pml(ref, cref)
        nb.update_E(ref); ref, cref = nb.update_E_pml(ref, cref)
        nb.update_H(got); got, cgot = nb.update_H_pml(got, cgot)
        nb.update_E(got); got, cgot = nb.update_E_pml(got, cgot)
    _assert_identical(_snapshot(ref), _snapshot(got), 'zero sigma')


# ---------------------------------------------------------------------- #
# End to end
# ---------------------------------------------------------------------- #

def test_simulation_backends_agree_on_a_lossy_run():
    """The whole loop, through Simulation, sources and PEC included."""
    def run(backend):
        grid = create_grid(40, 32, 1, 1e-3)
        ws.set_vacuum(grid)
        ws.set_box(grid, 15e-3, 30e-3, 0.0, 32e-3, 0.0, 1e-3,
                   eps_r=3.0, sigma=0.08)
        cpml = ws.init_cpml(grid, d_pml=8, faces=('x0', 'x1'))
        sim = ws.Simulation(grid, cpml=cpml, pec_faces=('y0', 'y1'),
                            backend=backend)
        sim.add_source(ws.GaussianBeam(
            'x0', angle=0.0, waveform=ws.GaussianPulse.for_fmax(30e9),
            waist=10.0, d_pml=8, directional=True))
        sim.run(200)
        return _snapshot(sim.grid)

    _assert_identical(run('numpy'), run('numba'), 'Simulation')
