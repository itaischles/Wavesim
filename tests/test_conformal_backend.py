"""Phase 4 — the conformal H update in the Numba backend, and the CUDA guard.

Until this landed, a conformal grid on ``backend='numba'`` was a **silently
wrong answer**: the Numba H update integrated the full face area while
``apply_pec_mask`` zeroed E by the cut-cell rule, so E and H saw different
conductors. Nothing raised, nothing warned, and the run looked healthy. That is
exactly the failure the homogeneous-fill invariant exists to catch, and it must
not be reachable by accident — hence both halves of this module: the Numba
kernel now tracks the NumPy reference, and CUDA (which has no conformal kernel)
refuses the grid outright rather than staircasing it.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim import backend_numba as nb
from wavesim.grid import create_grid
from wavesim.pml import init_cpml, update_H_pml, update_E_pml
from wavesim.update import update_H, update_E

from conformal_shapes import coax_fractions

KEYS = ('pec_edge_open_x', 'pec_edge_open_y', 'pec_edge_open_z',
        'pec_face_open_x', 'pec_face_open_y', 'pec_face_open_z')


def _cut_coax(n=20, nz=24, ds=0.6e-3, threshold=0.4):
    """A coax with genuine cut cells — the fractions vary continuously, so the
    open lengths and areas exercise every term of the conformal contour."""
    grid = create_grid(n, n, nz, ds)
    ws.set_vacuum(grid)
    c = 0.5 * n * ds
    ws.set_material_arrays(
        grid, grid.eps_x, grid.eps_y, grid.eps_z,
        grid.mu_x, grid.mu_y, grid.mu_z,
        conformal_area_threshold=threshold,
        **coax_fractions(grid, c, c, 1.2e-3, 4.8e-3))
    return grid


def _drive(grid, n):
    i, j, k = grid.Nx // 2, grid.Ny // 2, grid.Nz // 2
    grid.Ez[i, j, k] += float(np.exp(-((n - 12) / 5.0) ** 2))


# ---------------------------------------------------------------------- #
# Numba == NumPy on cut geometry
# ---------------------------------------------------------------------- #

def test_numba_conformal_matches_the_numpy_reference():
    """Bit-for-bit, with and without a cut cell in play.

    The kernel forms ``E·L`` inline rather than materialising the three ``E*L``
    arrays the reference builds, but the arithmetic grouping is otherwise
    identical, so the two agree exactly rather than to a tolerance.
    """
    def run(numba):
        grid = _cut_coax()
        for n in range(40):
            (nb.update_H if numba else update_H)(grid)
            (nb.update_E if numba else update_E)(grid)
            _drive(grid, n)
        return grid

    ref, got = run(False), run(True)
    assert np.abs(ref.Ez).max() > 1.0             # the run actually did something
    assert np.all(np.isfinite(ref.Ez))
    for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        assert np.array_equal(getattr(ref, name), getattr(got, name)), name


def test_numba_conformal_matches_numpy_through_the_pml():
    """A conductor crossing the absorbing shell is the case S6 chose to handle
    conformally rather than staircase, so the PML correction has to feed the psi
    recursion the conformal derivative too. The coax runs the full length of the
    domain and straight through both z shells, which is the geometry that would
    expose a mismatch at the interface."""
    def run(numba):
        grid = _cut_coax(nz=28)
        cpml = init_cpml(grid, faces=('z0', 'z1'), d_pml=6)
        for n in range(40):
            (nb.update_H if numba else update_H)(grid)
            (nb.update_H_pml if numba else update_H_pml)(grid, cpml)
            (nb.update_E if numba else update_E)(grid)
            (nb.update_E_pml if numba else update_E_pml)(grid, cpml)
            _drive(grid, n)
        return grid

    ref, got = run(False), run(True)
    assert np.abs(ref.Ez).max() > 1.0
    for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        assert np.array_equal(getattr(ref, name), getattr(got, name)), name


def test_numba_staircase_path_is_untouched():
    """No fraction arrays ⇒ the legacy kernel, unchanged. The conformal work
    added a dispatch and six arguments to the PML kernel; a model with no
    conductors must still take exactly the path it took before."""
    def run(numba):
        grid = create_grid(12, 12, 12, 1e-3)
        ws.set_vacuum(grid)
        cpml = init_cpml(grid, d_pml=4)
        for n in range(30):
            (nb.update_H if numba else update_H)(grid)
            (nb.update_H_pml if numba else update_H_pml)(grid, cpml)
            (nb.update_E if numba else update_E)(grid)
            (nb.update_E_pml if numba else update_E_pml)(grid, cpml)
            _drive(grid, n)
        return grid

    ref, got = run(False), run(True)
    assert np.abs(ref.Ez).max() > 1.0
    for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        assert np.array_equal(getattr(ref, name), getattr(got, name)), name


def test_conformal_dispatch_actually_changes_the_answer():
    """Guards the guard: if the conformal arrays were ignored, every test above
    would still pass by comparing two identically-wrong runs. The cut geometry
    must visibly move the fields away from the staircase result."""
    def run(conformal):
        grid = _cut_coax()
        if not conformal:
            for key in KEYS:
                setattr(grid, key, None)
        assert grid.is_conformal is conformal
        for n in range(40):
            nb.update_H(grid)
            nb.update_E(grid)
            _drive(grid, n)
        return grid

    stair, conf = run(False), run(True)
    rel = np.abs(conf.Hz - stair.Hz).max() / np.abs(stair.Hz).max()
    assert rel > 1e-3, f"conformal geometry moved Hz by only {rel:.2e}"


# ---------------------------------------------------------------------- #
# CUDA refuses rather than staircasing
# ---------------------------------------------------------------------- #

def test_cuda_refuses_a_conformal_grid():
    """CUDA has no conformal kernel and is deliberately out of scope, so it must
    fail loudly. Importing the backend does not need a GPU — the guard is in the
    host-side wrapper, ahead of any launch."""
    cu = pytest.importorskip("wavesim.backend_cuda")
    grid = _cut_coax()
    for fn, args in ((cu.update_H, (grid,)),
                     (cu.update_H_pml, (grid, None))):
        with pytest.raises(NotImplementedError, match="conformal"):
            fn(*args)


def test_cuda_still_accepts_a_staircase_grid():
    """The guard must key on the fraction arrays, not on ``pec_mask``: a plain
    staircased conductor is still a supported CUDA run."""
    cu = pytest.importorskip("wavesim.backend_cuda")
    grid = create_grid(8, 8, 8, 1e-3)
    ws.set_vacuum(grid)
    grid.pec_mask = np.zeros((8, 8, 8), dtype=bool)
    grid.pec_mask[3:5, 3:5, 3:5] = True
    cu._refuse_conformal(grid)          # does not raise
