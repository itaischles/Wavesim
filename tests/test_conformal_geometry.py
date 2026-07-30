"""S1 — conformal PEC grid + material plumbing.

Covers the data contract with the FreeCAD workbench (six dimensionless
open-fraction arrays in ``materials.npz``) and the metre-valued geometry the
solver derives from it. No field update is exercised here — that is S2.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.grid import create_grid, create_grid_rectilinear
from wavesim.pec import build_conformal_geometry, conformal_geometry, count_cut_cells


N = (4, 5, 3)


def _grid(uniform=True):
    if uniform:
        return create_grid(*N, 1e-3)
    ax = lambda n, d0: np.concatenate([[0.0], np.cumsum(d0 * 1.3 ** np.arange(n))])
    return create_grid_rectilinear(ax(N[0], 1e-3), ax(N[1], 2e-3), ax(N[2], 5e-4))


def _vacuum(grid):
    ones = lambda: np.ones((grid.Nx, grid.Ny, grid.Nz))
    return dict(eps_x=ones(), eps_y=ones(), eps_z=ones(),
                mu_x=ones(), mu_y=ones(), mu_z=ones())


def _fractions(grid, value=1.0):
    return {k: np.full((grid.Nx, grid.Ny, grid.Nz), value, dtype=np.float32)
            for k in ('pec_edge_open_x', 'pec_edge_open_y', 'pec_edge_open_z',
                      'pec_face_open_x', 'pec_face_open_y', 'pec_face_open_z')}


# ---------------------------------------------------------------------- #
# Contract: absent ⇒ legacy, all six or none, fractions in [0, 1]
# ---------------------------------------------------------------------- #

def test_absent_fractions_leave_grid_on_the_staircase_path():
    grid = ws.set_material_arrays(_grid(), **_vacuum(_grid()))
    assert grid.is_conformal is False
    assert conformal_geometry(grid) is None
    assert count_cut_cells(grid) == 0
    for name in ('pec_edge_open_x', 'pec_face_open_z'):
        assert getattr(grid, name) is None


def test_partial_set_is_rejected():
    grid = _grid()
    fr = _fractions(grid)
    fr.pop('pec_face_open_y')
    with pytest.raises(ValueError, match="complete set of six"):
        ws.set_material_arrays(grid, **_vacuum(grid), **fr)
    # and nothing was written on the way to the error
    assert grid.is_conformal is False


def test_out_of_range_fraction_is_rejected():
    grid = _grid()
    fr = _fractions(grid)
    fr['pec_edge_open_y'][1, 1, 1] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ws.set_material_arrays(grid, **_vacuum(grid), **fr)


def test_wrong_shape_is_rejected():
    grid = _grid()
    fr = _fractions(grid)
    fr['pec_face_open_x'] = np.ones((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="expected shape"):
        ws.set_material_arrays(grid, **_vacuum(grid), **fr)


def test_pec_mask_is_independent_of_the_fractions():
    """pec_mask stays the fully-covered test and the legacy path (§3)."""
    grid = _grid()
    mask = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool)
    mask[2, 2, 1] = True
    grid = ws.set_material_arrays(grid, **_vacuum(grid), pec_mask=mask,
                                  **_fractions(grid, 0.5))
    assert grid.pec_mask[2, 2, 1]
    assert grid.is_conformal


# ---------------------------------------------------------------------- #
# Derived geometry
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "graded"])
def test_all_ones_reproduces_the_full_cell_geometry(uniform):
    """No conductor ⇒ L is the full edge length and A the full face area.

    Primary widths, because the Faraday contour of an H face is bounded by
    nodes — the definition S0 pins down and the workbench shares.
    """
    grid = _grid(uniform)
    grid = ws.set_material_arrays(grid, **_vacuum(grid), **_fractions(grid, 1.0))
    g = conformal_geometry(grid)

    dxp, dyp, dzp = grid.dxp, grid.dyp, grid.dzp
    assert np.allclose(g.Lx, dxp[:, None, None])
    assert np.allclose(g.Ly, dyp[None, :, None])
    assert np.allclose(g.Lz, dzp[None, None, :])
    assert np.allclose(g.Ax, dyp[None, :, None] * dzp[None, None, :])
    assert np.allclose(g.Ay, dzp[None, None, :] * dxp[:, None, None])
    assert np.allclose(g.Az, dxp[:, None, None] * dyp[None, :, None])

    assert count_cut_cells(grid) == 0


def test_fractions_scale_the_geometry_linearly():
    grid = _grid(uniform=False)
    fr = _fractions(grid, 1.0)
    fr['pec_edge_open_x'][1, 2, 0] = 0.25
    fr['pec_face_open_z'][1, 2, 0] = 0.75
    grid = ws.set_material_arrays(grid, **_vacuum(grid), **fr)
    g = conformal_geometry(grid)

    assert g.Lx[1, 2, 0] == pytest.approx(0.25 * grid.dxp[1])
    assert g.Az[1, 2, 0] == pytest.approx(0.75 * grid.dxp[1] * grid.dyp[2])
    # neighbours untouched
    assert g.Lx[2, 2, 0] == pytest.approx(grid.dxp[2])


def test_fully_covered_edges_and_faces_are_zero_not_reciprocal():
    grid = _grid()
    fr = _fractions(grid, 1.0)
    fr['pec_face_open_y'][0, 0, 0] = 0.0
    fr['pec_edge_open_z'][0, 0, 0] = 0.0
    grid = ws.set_material_arrays(grid, **_vacuum(grid), **fr)
    g = conformal_geometry(grid)
    assert g.Ay[0, 0, 0] == 0.0
    assert g.Lz[0, 0, 0] == 0.0
    assert np.all(np.isfinite(g.Ay))


def test_cut_cell_count_excludes_fully_open_and_fully_closed():
    grid = _grid()
    fr = _fractions(grid, 1.0)
    fr['pec_face_open_x'][0, 0, 0] = 0.0      # fully covered — not a cut cell
    fr['pec_face_open_x'][1, 0, 0] = 0.3      # cut
    fr['pec_face_open_z'][2, 1, 1] = 0.9      # cut
    grid = ws.set_material_arrays(grid, **_vacuum(grid), **fr)
    assert count_cut_cells(grid) == 2


# ---------------------------------------------------------------------- #
# Cache
# ---------------------------------------------------------------------- #

def test_geometry_is_cached_and_invalidated_by_replacement():
    grid = _grid()
    grid = ws.set_material_arrays(grid, **_vacuum(grid), **_fractions(grid, 1.0))

    first = conformal_geometry(grid)
    assert conformal_geometry(grid) is first          # cached, not rebuilt

    grid.pec_edge_open_x = np.full_like(grid.pec_edge_open_x, 0.5)
    second = conformal_geometry(grid)
    assert second is not first                        # identity change invalidates
    assert np.allclose(second.Lx, 0.5 * grid.dxp[:, None, None])


def test_builder_is_pure():
    """build_conformal_geometry bypasses the cache and never mutates the grid."""
    grid = _grid()
    grid = ws.set_material_arrays(grid, **_vacuum(grid), **_fractions(grid, 0.5))
    a = build_conformal_geometry(grid)
    b = build_conformal_geometry(grid)
    assert a is not b
    assert np.array_equal(a.Lx, b.Lx)
    assert not hasattr(grid, '_conformal_cache')


def test_fractions_are_stored_as_float64_geometry():
    """Contract says float32 on the wire; geometry is float64 inside, like dxp."""
    grid = _grid()
    grid = ws.set_material_arrays(grid, **_vacuum(grid), **_fractions(grid, 1.0))
    assert grid.pec_edge_open_x.dtype == np.float64
    assert conformal_geometry(grid).Ax.dtype == np.float64
