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


def binary_fractions(mask):
    """0/1 open fractions describing exactly the staircase ``mask``.

    One convention, applied to both kinds of array: **covered iff the metal
    reaches it**, i.e. iff any cell touching it is PEC. For an edge that is the
    four cells around it (:func:`~wavesim.pec.build_pec_edge_masks`) and for a
    face the two it separates. Both readings say the same thing — a Yee element
    lying *on* a staircased conductor's surface is in the closure of the metal,
    and carries no tangential field — and it is the rule the run itself applies.
    :func:`~wavesim.mode_solver._conformal_node_pec` and
    :func:`~wavesim.parts.pec_node_mask` on these fractions both return the
    closed node box, so the staircase and conformal code paths describe the same
    conductor and must land on precisely the same potential.

    Keeping the two consistent matters, because
    :func:`~wavesim.pec.build_conformal_edge_masks` reads the *faces* to find
    edges tangent to a grid-aligned surface. With this convention that clause
    reproduces the staircase dilation exactly — ``Ez`` is covered iff
    ``dilate(mask, (x, y))``, which is what an Hx face dilated along y and an Hy
    face dilated along x come to. Describing the faces some other way (this
    carried the coax's z-invariant identity ``face_x ≡ edge_y`` for a while,
    which is only true for a z-invariant solid) makes the fixture claim metal
    where there is none, and the edge rule believes it.

    Stating the edge rule as "covered iff both end nodes are PEC cells" instead
    (which is what this did while the staircase mode solve sliced ``pec_mask`` as
    if it were a node mask) describes a conductor one cell smaller on every high
    side — see ``docs/mode_solver_staircase_node_mask.md``.

    This is how the conformal *code path* is separated from the cut cells
    themselves: whatever it computes on these fractions is what the staircase
    assembly computes, or the reduction property has been broken.
    """
    from wavesim.pec import build_pec_edge_masks, _dilate

    ex, ey, ez = build_pec_edge_masks(mask)
    return dict(
        pec_edge_open_x=(~ex).astype(float),
        pec_edge_open_y=(~ey).astype(float),
        pec_edge_open_z=(~ez).astype(float),
        # Hx separates cells (i-1, i); Hy (j-1, j); Hz (k-1, k).
        pec_face_open_x=(~_dilate(mask, (0,))).astype(float),
        pec_face_open_y=(~_dilate(mask, (1,))).astype(float),
        pec_face_open_z=(~_dilate(mask, (2,))).astype(float))
