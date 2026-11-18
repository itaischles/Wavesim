"""Test geometry for :mod:`wavesim.magnetostatics`.

The workhorse is a **shorted coax with a port across its open end**, which is
the structure whose inductance has a closed form, ``L = µℓ/(2π)·ln(b/a)``.
It is laid out along z so that the box's own flux walls do the shorting:

    node plane 0        zmin flux wall  — the short
    cells 0 … ke-1      inner conductor (PEC or σ), radius a
    node plane ke       inner end face  — terminal a
    cells ke            the gap         — the port channel
    node plane ke+1     zmax flux wall  — terminal b (= Nz-1, the last node)

with the outer conductor (``r ≥ b``) running the full length. The last cell
layer, ``Nz-1``, lies beyond the last carried node and is outside the solve.
The flux walls, outer conductor and (PEC) inner are one group, so the port
returns through PEC; with a σ inner the current runs short → inner → gap.

Static TEM is exact all the way into both ends: the short is a flat flux wall
carrying a radial current ``I/(2πr)``, and the channel is an axial current of
the inner conductor's own cross-section. So the field is ``µI/(2πr)``
everywhere in ``a < r < b`` and

    ``L ≈ µ/(2π)·ln(b/a)·(ℓ + g) + (channel's internal inductance)``

The channel term (~``µg/8π`` for uniform current) is removed exactly by
differencing two lengths, which leaves ``L′`` free of every end effect.
"""

import numpy as np

import wavesim as ws
from wavesim.materials import _sigma_arrays
from wavesim.parts import name_pec_region, pec_node_mask

from conformal_shapes import coax_fractions

A_IN, B_OUT, SPAN = 3.0e-3, 9.0e-3, 24.0e-3


def _closure(cells2d):
    """Node closure of a 2D cell mask: every corner of every cell."""
    out = cells2d.copy()
    out[1:, :] |= cells2d[:-1, :]
    out[:, 1:] |= out[:, :-1].copy()
    return out


def z_nodes(n_line, dz, gap=None, gap_dz=None):
    """Node coordinates along z: ``n_line`` cells of ``dz`` for the inner
    conductor, one gap cell, and one dummy cell beyond the last carried node.
    """
    g = dz if gap_dz is None else gap_dz
    widths = [dz] * n_line + [g, g]
    return np.concatenate([[0.0], np.cumsum(widths)])


def coax_port(n, zs, inner='pec', geometry='staircase', sigma=1.0e6,
              mu_r=1.0, square=False):
    """Build the shorted coax and its port.

    Parameters
    ----------
    n : int
        Cells across the 24 mm box (transverse cell ``24/n`` mm).
    zs : array
        z node coordinates from :func:`z_nodes`.
    inner : 'pec' or 'sigma'
        A PEC inner conductor (flux expelled) or a σ one (B penetrates).
    geometry : 'staircase' or 'conformal'
        Cut-cell fractions for the PEC (inner and outer). A σ inner is always
        a cell mask.
    square : bool
        Make the inner a square bar of side ``2a`` (cell-aligned when ``2a``
        is a whole number of cells) instead of a round rod. Used to measure a
        resistance that is exactly ``ℓ/(σA)``.

    Returns ``(grid, (term_a, term_b, channel), info)`` with ``info`` holding
    ``ell`` (inner length), ``gap`` and ``ke``.
    """
    ds = SPAN / n
    xy = np.arange(n + 1) * ds
    grid = ws.set_vacuum(ws.create_grid_rectilinear(xy, xy, zs))
    if mu_r != 1.0:
        ws.set_dielectric(grid, 1.0)
        for arr in (grid.mu_x, grid.mu_y, grid.mu_z):
            arr[...] = mu_r
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    ke = Nz - 2
    c = 0.5 * SPAN

    xc, yc = grid.xc - c, grid.yc - c
    if square:
        in_xy = (np.abs(xc)[:, None] < A_IN) & (np.abs(yc)[None, :] < A_IN)
    else:
        in_xy = np.hypot(xc[:, None], yc[None, :]) < A_IN
    out_xy = np.hypot(xc[:, None], yc[None, :]) > B_OUT
    k = np.arange(Nz)
    inner_cells = in_xy[:, :, None] & (k < ke)[None, None, :]
    outer_cells = np.broadcast_to(out_xy[:, :, None], (Nx, Ny, Nz)).copy()

    name_pec_region(grid, outer_cells, 'outer')
    if inner == 'pec':
        name_pec_region(grid, inner_cells, 'inner')
    elif inner == 'sigma':
        for arr in _sigma_arrays(grid):
            arr[inner_cells] = sigma
    else:
        raise ValueError(inner)

    if geometry == 'conformal':
        a = A_IN if inner == 'pec' else 0.0
        fin = coax_fractions(grid, c, c, a, B_OUT)
        fout = coax_fractions(grid, c, c, 0.0, B_OUT)
        spans = k < ke            # elements spanning k..k+1
        plane = k <= ke           # elements lying in node plane k
        sel = {'pec_edge_open_z': spans, 'pec_face_open_x': spans,
               'pec_face_open_y': spans, 'pec_edge_open_x': plane,
               'pec_edge_open_y': plane, 'pec_face_open_z': plane}
        fr = {key: np.where(sel[key][None, None, :], fin[key], fout[key])
              for key in fin}
        ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                               grid.mu_x, grid.mu_y, grid.mu_z,
                               sigma_x=grid.sigma_x, sigma_y=grid.sigma_y,
                               sigma_z=grid.sigma_z, **fr)
    elif geometry != 'staircase':
        raise ValueError(geometry)

    r_node = np.hypot((grid.x[:Nx] - c)[:, None], (grid.y[:Ny] - c)[None, :])
    if inner == 'pec':
        face = pec_node_mask(grid)[:, :, ke] & (r_node < 0.5 * (A_IN + B_OUT))
    else:
        face = _closure(in_xy)
    term_a = np.zeros((Nx, Ny, Nz), dtype=bool)
    term_a[:, :, ke] = face
    ch2 = face[:-1, :-1] & face[1:, :-1] & face[:-1, 1:] & face[1:, 1:]
    channel = np.zeros((Nx, Ny, Nz), dtype=bool)
    channel[:-1, :-1, ke] = ch2
    foot = np.zeros((Nx, Ny), dtype=bool)
    foot[:-1, :-1] = ch2
    term_b = np.zeros((Nx, Ny, Nz), dtype=bool)
    term_b[:, :, ke + 1] = _closure(foot)

    info = dict(ell=float(zs[ke] - zs[0]), gap=float(zs[ke + 1] - zs[ke]),
                ke=ke, ds=ds,
                area=float(in_xy.sum()) * ds * ds)
    return grid, (term_a, term_b, channel), info


def coax_cross_section(n, geometry='staircase', nz=4):
    """The same coax cross-section, extruded for :mod:`wavesim.electrostatics`.

    Identical cell masks (cell-centre radius) and identical cut fractions to
    :func:`coax_port`, so the electrostatic C′ and the magnetostatic L′ are
    measured on one conductor. ``set_coax`` would not do: it voxelises the
    circle by node coordinates, a different staircase. Inner part ``'inner'``,
    outer ``'outer'``; ``nz`` cells deep with the end caps left Neumann.
    """
    ds = SPAN / n
    grid = ws.set_vacuum(ws.create_grid(n, n, nz, ds, ds, ds))
    c = 0.5 * SPAN
    r = np.hypot((grid.xc - c)[:, None], (grid.yc - c)[None, :])
    full = (n, n, nz)
    name_pec_region(grid, np.broadcast_to((r < A_IN)[:, :, None], full).copy(),
                    'inner')
    name_pec_region(grid, np.broadcast_to((r > B_OUT)[:, :, None], full).copy(),
                    'outer')
    if geometry == 'conformal':
        fr = coax_fractions(grid, c, c, A_IN, B_OUT)
        ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                               grid.mu_x, grid.mu_y, grid.mu_z, **fr)
    return grid
