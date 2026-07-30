"""Analytic conformal-PEC open fractions for test geometry.

The conformal solver work is developed against fractions computed directly from
the exact shape, so it does not wait on — or inherit bugs from — the FreeCAD
workbench voxeliser that will produce them in production. The contract is the
same either way: six dimensionless arrays in [0, 1] (see
:class:`wavesim.grid.FDTDGrid`).
"""

import numpy as np


def _segment_open(u0, u1, v, a, b):
    """Length of the segment ``[u0, u1]`` (at offset ``v``) inside the annulus
    ``a ≤ √(u² + v²) ≤ b``.

    Closed form rather than sampling: ``r ≥ a`` ⟺ ``|u| ≥ √(a² − v²)`` and
    ``r ≤ b`` ⟺ ``|u| ≤ √(b² − v²)``, so the open set is the pair of intervals
    ``±[s, t]`` and the answer is its overlap with ``[u0, u1]``.
    """
    v2 = v * v
    if v2 > b * b:
        return 0.0
    t = np.sqrt(max(b * b - v2, 0.0))
    s = np.sqrt(max(a * a - v2, 0.0))

    def overlap(lo, hi):
        return max(0.0, min(u1, hi) - max(u0, lo))

    return overlap(s, t) + overlap(-t, -s)


def coax_fractions(grid, cx, cy, r_inner, r_outer, nsub=64):
    """Open fractions for a z-invariant coax: conductor where r < a or r > b.

    z-invariance buys an exactness that a general sampler could not give: the Hx
    face spans (y, z) at x-node i, and the geometry does not vary in z, so its
    open *area* fraction equals the open *length* fraction of the Ey edge at the
    same (i, j) — the two arrays are the same object here rather than two
    independent estimates. Likewise Hy and Ex.

    Only the Hz face genuinely needs two dimensions, and it is sub-sampled.
    That is fine for the homogeneous-fill invariant, which must hold *whatever*
    the conductor geometry is — including an imperfectly resolved one.

    Returns the six-key dict accepted by ``set_material_arrays``.
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)
    x, y = grid.x - cx, grid.y - cy
    a, b = r_inner, r_outer

    fx = np.zeros(shape)                     # Ex edge, and Hy face
    fy = np.zeros(shape)                     # Ey edge, and Hx face
    for i in range(grid.Nx):
        for j in range(grid.Ny):
            fx[i, j, :] = _segment_open(x[i], x[i + 1], y[j], a, b) / grid.dxp[i]
            fy[i, j, :] = _segment_open(y[j], y[j + 1], x[i], a, b) / grid.dyp[j]

    # Ez runs along z at fixed (x, y), so r is constant on it: fully in or out.
    r_node = np.hypot(x[:grid.Nx, None], y[None, :grid.Ny])
    fz = np.broadcast_to(((r_node >= a) & (r_node <= b))[:, :, None],
                         shape).astype(float)

    faz = np.zeros(shape)                    # Hz face — 2D, sub-sampled
    q = (np.arange(nsub) + 0.5) / nsub
    for i in range(grid.Nx):
        xs = x[i] + q * grid.dxp[i]
        for j in range(grid.Ny):
            ys = y[j] + q * grid.dyp[j]
            r = np.hypot(xs[:, None], ys[None, :])
            faz[i, j, :] = np.mean((r >= a) & (r <= b))

    # A fully open edge can land a ULP or two above 1.0 (the closed-form length
    # divided by the cell width), which the contract rightly rejects.
    fx, fy = np.clip(fx, 0.0, 1.0), np.clip(fy, 0.0, 1.0)

    return dict(pec_edge_open_x=fx, pec_edge_open_y=fy, pec_edge_open_z=fz,
                pec_face_open_x=fy, pec_face_open_y=fx, pec_face_open_z=faz)
