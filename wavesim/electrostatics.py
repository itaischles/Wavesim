"""
electrostatics.py — 3D electrostatic (Poisson) solver on the FDTD grid.

Hold some conductors at some potentials and ask what the field does. Formally

    ``∇·(ε ∇φ) = -ρ``

over the dielectric, with each conductor an equipotential surface. This is a
boundary-value problem, not a time stepping one: nothing here reads ``dt``,
``Ex..Hz`` or the CFL condition, and running it neither needs nor disturbs a
simulation. It shares the grid because the *geometry* is the same geometry —
``set_box``, ``pec_mask``, ``eps_*`` and the cut-cell fractions all mean here
exactly what they mean to the field solver, so a model is built once.

Space charge (``ρ``) is carried through the formulation and the right-hand side
but is not implemented: today every source is a conductor potential, which makes
the equation Laplace's. See :meth:`Electrostatics.solve`.

Relationship to the mode solver
-------------------------------
:mod:`wavesim.mode_solver` already solves this equation in 2D — a TEM mode's
transverse field *is* the electrostatic field of the cross-section. This module
is the same finite-volume discretisation with a 7-point stencil instead of a
5-point one, and it deliberately reuses that module's two hard-won rules: the
node-centred control volume (:func:`_node_dual`) and the one-sided permittivity
at a conductor surface (:func:`_face_eps`). The two implementations are kept
separate for now rather than unified behind a shared N-D core, because the mode
solver is validated to −0.8% Z₀ on the conformal coax and a generalisation is
exactly the kind of change that quietly reintroduces the primary/dual pairing
bug its docstrings describe. Unification is a later job with those tests as the
gate.

Where the potential lives
-------------------------
φ sits on the Yee **nodes**, which is what makes the whole thing fit: the
difference of φ across an edge is that edge's E component, so ``E = -∇φ`` lands
exactly on the ``Ex``/``Ey``/``Ez`` staggering and every existing monitor and
plotting helper can read it. Node ``(i,j,k)`` is the low corner of cell
``(i,j,k)``, and as everywhere else in wavesim the ``N``-th node of each axis is
not carried — ``N`` cells have ``N+1`` nodes but the arrays hold ``N``.

Which nodes are metal is *not* read off ``pec_mask`` directly. It comes from
:func:`wavesim.parts.pec_node_mask`, i.e. from the same zeroed-E-edge masks the
FDTD applies, so the electrostatic conductor and the FDTD conductor are the same
object. See that module for why the shortcut is wrong.

Cut cells
---------
On a grid carrying conformal (Dey–Mittra) open fractions the solve runs on the
cut geometry, not on its staircased approximation, so the conductor here is the
one the FDTD steps at sub-cell resolution too. The whole of that support is one
substitution — :func:`_open_lengths` — because the fraction belongs on the
node-to-node **distance** and nowhere else: a node just outside the metal sits
the open part of an edge away from the surface, so its coupling grows as
``1/f``, which is the correct Dirichlet behaviour, and a fully covered edge
(``f = 0``) drops out of the operator entirely with its two endpoints pinned
instead. That is :mod:`wavesim.mode_solver`'s derivation from the conformal
Faraday contour, unchanged; every weight below is the staircase weight times an
open fraction, so an all-1.0 grid reproduces the staircase assembly bit for bit.

The face *area* is deliberately not scaled. Beyond the derivation saying so, the
grid does not carry the number that would be needed: ``pec_face_open_*`` are the
open areas of the **primary** H faces, which span nodes, whereas the control
volume here is a node's *dual* cell, whose faces are a different set of surfaces
the voxeliser never measured. Scaling by the H-face fraction would be using a
plausible-looking array for a quantity it is not.

Boundary conditions
-------------------
PML is meaningless in statics — there is no wave to absorb — so the domain box
gets one of two conditions per face:

**Dirichlet**, ``φ = V``: the face is an equipotential. The field arrives
perpendicular to it, so this is a conducting wall (a ground plane at ``V = 0``,
or a driven plate).

**Neumann**, ``∂φ/∂n = 0``: no flux crosses the face. The field is purely
tangential there, so this is a mirror — a symmetry plane, or the magnetic wall
of microwave usage.

There is no open-boundary condition and cannot cheaply be one: the exterior of a
finite box is an infinite domain, and no local rule at the wall reproduces it.
An isolated charged object is modelled by putting a Dirichlet ``φ = 0`` box far
enough away that moving it further stops changing the answer — which is a
convergence study the caller owns, not a setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Tuple
import warnings

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu, cg, LinearOperator

from wavesim.constants import EPS0
from wavesim.grid import FDTDGrid
from wavesim.parts import (body_parts, conductor_bodies, part_id,
                           pec_node_mask)

FACES = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')

# ---------------------------------------------------------------------- #
# Which linear solver, and why it is not a size-dependent choice.
#
# The system is sparse and symmetric positive definite, so there are two
# families: factorise it once (``splu``), or refine a guess using only
# matrix-vector products (conjugate gradients). The 2D mode solver factorises,
# and the obvious plan was to inherit that for small 3D problems and fall back
# to CG for large ones. Measured on a cubic parallel-plate model, single solve,
# CG at rtol=1e-10 with Jacobi preconditioning:
#
#     unknowns     direct        CG
#        3,072      0.04 s     0.004 s
#       11,520      0.33 s     0.01 s
#       28,672      2.93 s     0.03 s
#       57,600     15.56 s     0.07 s
#      101,376     59.00 s     0.15 s
#
# There is no crossover to find: CG is ahead at every size, and the gap widens
# from 10x to 400x because 3D elimination fill-in grows superlinearly while CG
# stays roughly proportional to the grid. The plan was wrong in the direction
# nobody checks, which is why it was measured rather than reasoned about.
#
# The second argument for factorising was multiple right-hand sides — the
# capacitance matrix needs one solve per conductor, and re-using a factorisation
# should make the extras nearly free. It does make them cheap; it does not make
# them cheaper than CG:
#
#     unknowns   LU factor   extra RHS   |   CG per RHS
#       12,096      0.44 s     0.008 s   |     0.012 s
#       29,696      3.69 s     0.028 s   |     0.037 s
#       59,200     16.71 s     0.099 s   |     0.088 s
#
# An extra LU solve costs about what a whole CG solve costs, so the
# factorisation never amortises — it would take of order a hundred conductors
# to pay back, and models do not have a hundred conductors.
#
# So 'auto' means CG, at every size. The direct path stays available because it
# is exact and has no tolerance to tune, which makes it the right reference when
# a discretisation is under test rather than a model, and the right retry if CG
# ever converges badly on a nastier permittivity contrast than has been tried.
# Selecting it by problem size was rejected on top of all this: it would mean
# refining a mesh silently changes the solver, and with it the last digits of
# the answer, for no gain.
# ---------------------------------------------------------------------- #
AUTO_METHOD = 'cg'


# ====================================================================== #
# Geometry weights
# ====================================================================== #

def _node_dual(w: np.ndarray) -> np.ndarray:
    """Dual-cell width owned by each node, from the per-cell primary widths.

    Node ``i`` sits between cells ``i-1`` and ``i`` and owns half of each, so its
    control volume spans ``(w[i-1] + w[i])/2``. **Both** end nodes lie on the
    domain wall and own a single half cell.

    The high end is where this parts company with
    :func:`wavesim.mode_solver._node_dual`, which gives node ``N-1`` a full dual
    width. The faces of this operator span ``x[0] … x[N-1]`` — there is no face
    beyond the last carried node, since ``N`` cells give ``N+1`` nodes and the
    arrays hold ``N`` — so the dual cells have to tile that same interval, and a
    full width at the end overshoots it by half a cell. The mode solver never
    noticed because its default grounds the whole edge ring, whose coefficients
    are then never assembled.

    Here it is visible, because a Neumann wall is a first-class boundary and a
    symmetry plane is the main reason to want one. With a full width at the top
    the two ends weight their faces differently, so mirroring a symmetric
    problem about the high wall does not reproduce the full solve — which is
    exactly what ``test_a_neumann_face_reproduces_the_mirrored_full_solve``
    measures.

    A single-cell axis (``Nz=1``, the quasi-2D case) has no faces along it at
    all and keeps the full cell width, making the solve one of unit depth rather
    than of zero depth. φ cannot see the choice — a common factor on every face
    of an axis cancels out of the operator — but the energy and capacitance
    integrals built on it read as per-unit-length, which is what a
    cross-sectional solve should report.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.size == 1:
        return w.copy()
    out = np.empty_like(w)
    out[0] = 0.5 * w[0]
    out[1:-1] = 0.5 * (w[:-2] + w[1:-1])
    out[-1] = 0.5 * w[-2]
    return out


def _face_eps(eps: np.ndarray, node_pec: np.ndarray, axis: int) -> np.ndarray:
    """Permittivity of each face — the stored edge value, not an average.

    The face between node ``i`` and node ``i+1`` along ``axis`` *is* the Yee edge
    ``E<axis>[i]``, and ``eps_<axis>[i]`` is by definition the permittivity that
    edge sees. Using it directly rather than averaging neighbours keeps a
    material interface exactly on the face it actually lies on.

    A face straddling a conductor surface is the exception, and it is
    load-bearing. ε inside metal is not a material property — it is whatever the
    voxeliser happened to leave there, normally 1.0 — so such a face borrows the
    ε of the next face *outward*, which lies wholly in the dielectric. Without
    it the filled and vacuum operators stop being exact scalar multiples of each
    other and a homogeneously filled structure no longer reports ε_eff = ε_r
    exactly. Same rule as :func:`wavesim.mode_solver._face_eps`, generalised to
    three axes.
    """
    n = eps.shape[axis]
    take = lambda a, sl: a[(slice(None),) * axis + (sl,)]
    face = take(eps, slice(0, n - 1)).astype(np.float64)
    if n < 2:
        return face

    lo = take(node_pec, slice(0, n - 1))
    hi = take(node_pec, slice(1, n))
    outward_hi = take(eps, slice(1, n))                       # face i+1
    outward_lo = np.concatenate(                              # face i-1, clamped
        [take(eps, slice(0, 1)), take(eps, slice(0, n - 2))], axis=axis)
    face = np.where(lo & ~hi, outward_hi, face)
    face = np.where(hi & ~lo, outward_lo, face)
    return face


def _face_slices(axis: int):
    """``(low, high)`` index tuples pairing each node with its ``+axis`` neighbour.

    Every face-wise loop in this module — assembly, flux, energy, the gradient —
    walks the same pairing, so it is written once. ``arr[low]`` and ``arr[high]``
    are the values at the two ends of every face along ``axis``.
    """
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[axis] = slice(0, -1)
    hi[axis] = slice(1, None)
    return tuple(lo), tuple(hi)


# The smallest open fraction the operator will carry. This is *not* the
# covered/not-covered test — that stays exactly ``f == 0.0``, so the set of
# edges this module calls metal is the identical set
# :func:`wavesim.pec.build_conformal_edge_masks` hands the field update, and the
# two solvers cannot drift apart on the question of where the conductor is.
#
# It is a floor on the coefficient. A conductor face lying on the node ruler
# voxelises to a fraction like 1.7e-15 rather than to 0 — round-off, not
# geometry — and such an edge is alive by the test above while its ``1/f`` weight
# is 1e15 times its neighbours'. The FDTD never notices (it multiplies by the
# open length, so a vanishing edge simply contributes nothing); a linear solve
# does, because that weight lands on a matrix row. Clamping the *length* from
# below leaves the edge live and still overwhelmingly strongly coupled — 1e9 is
# a Dirichlet pin by any measure — without putting an infinity in the operator.
#
# Distinct from :data:`wavesim.grid.FDTDGrid.conformal_area_threshold`, which
# exists because ``dt/(μA)`` diverges on a sliver face. There is no dt here and
# no instability to cure; this is conditioning alone, and it clamps a length
# rather than an area.
MIN_OPEN_FRACTION = 1e-9


def _open_lengths(grid: FDTDGrid) -> Tuple[Tuple[np.ndarray, ...], int]:
    """Open node-to-node distance along each axis in metres, and a sliver count.

    Returns ``((Lx, Ly, Lz), n_clamped)``. Each ``L`` is indexed like the edge it
    measures — ``Lx[i,j,k]`` is the open length of the Yee edge ``Ex[i,j,k]``,
    joining node ``(i,j,k)`` to ``(i+1,j,k)`` — and is the single quantity that
    carries cut-cell geometry into this module. The operator divides by it, and
    so does ``E = −∇φ``, which is what keeps the field consistent with the
    equations that produced it.

    Without cut geometry this is just the primary width, returned **broadcast**
    rather than expanded: the staircase path keeps a length-``N`` array reshaped
    to ``(N,1,1)``, since materialising three full grids of a constant would cost
    hundreds of megabytes on a large model to say nothing new.

    A fully covered edge keeps ``L = 0`` exactly, which drops it from the
    operator; only a strictly positive fraction below :data:`MIN_OPEN_FRACTION`
    is clamped, and ``n_clamped`` counts those.
    """
    primary = (grid.dxp, grid.dyp, grid.dzp)
    fracs = (grid.pec_edge_open_x, grid.pec_edge_open_y, grid.pec_edge_open_z)

    out, n_clamped = [], 0
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = primary[axis].size
        width = np.asarray(primary[axis], dtype=np.float64).reshape(shape)
        if not grid.is_conformal:
            out.append(width)
            continue
        f = np.asarray(fracs[axis], dtype=np.float64)
        sliver = (f > 0.0) & (f < MIN_OPEN_FRACTION)
        n_clamped += int(np.count_nonzero(sliver))
        out.append(np.where(sliver, MIN_OPEN_FRACTION, f) * width)
    return tuple(out), n_clamped


def _face_coefs(grid: FDTDGrid, node_pec: np.ndarray,
                lengths: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    """Per-face conductances of the ε-weighted Laplacian, one array per axis.

    ``cx[i,j,k]`` couples node ``(i,j,k)`` to ``(i+1,j,k)`` and is

        ``ε_face · face_area / centre_distance``

    The control volume of the equation at a node is its *dual* cell, so the face
    area is the product of the two **dual** widths on the other axes, while the
    distance is the **primary** width along the coupling axis — the node-to-node
    separation, which is what ``E = Δφ/d`` divides by.

    That pairing is the one thing here worth double-checking against intuition.
    Getting it backwards is invisible on a uniform mesh, where primary and dual
    widths are equal to the last bit, and costs a full order of accuracy on a
    graded one; :mod:`wavesim.mode_solver` carries the same note because the
    same mistake was made and measured there.

    ``lengths`` supplies that distance, from :func:`_open_lengths`, and is the
    only route by which cut cells enter: under conformal PEC it is the *open*
    part of the edge, so the coefficient rises as ``1/f``. A fully covered edge
    has zero length and is given a zero coefficient rather than an infinite one —
    it carries no flux equation at all, and the constraint it stands for (equal φ
    at its two ends) is imposed by those nodes being pinned instead.
    """
    dual = (_node_dual(grid.dxp), _node_dual(grid.dyp), _node_dual(grid.dzp))
    eps = (grid.eps_x, grid.eps_y, grid.eps_z)

    def spread(vec, axis):
        """``vec`` laid out along ``axis``, broadcastable against a 3D array."""
        shape = [1, 1, 1]
        shape[axis] = vec.size
        return vec.reshape(shape)

    out = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        area = spread(dual[others[0]], others[0]) * spread(dual[others[1]],
                                                           others[1])
        lo, _ = _face_slices(axis)
        distance = np.broadcast_to(lengths[axis], grid.eps_x.shape)[lo]
        numerator = _face_eps(eps[axis], node_pec, axis) * area
        out.append(np.divide(numerator, distance, out=np.zeros_like(numerator),
                             where=distance > 0.0))
    return tuple(out)


# ====================================================================== #
# Sparse assembly
# ====================================================================== #

def _assemble(coefs, fixed: np.ndarray):
    """Build the symmetric positive-definite system over the free nodes.

    Discretises ``-∇·(ε∇φ)`` with a 7-point variable-coefficient finite-volume
    stencil. The equation at node ``n`` is ``Σ_faces c_f (φ_n − φ_nbr)``, which
    integrates the flux out of that node's dual cell; a neighbour that is off
    the end of the array simply contributes no face, which *is* the zero-flux
    (Neumann) wall, so the default boundary needs no code at all.

    Returns ``(A, B, free_idx)``: ``A`` (free × free) is the operator, ``B``
    (free × fixed) collects the couplings to pinned nodes so the right-hand side
    is ``b = B @ φ_fixed``, and ``free_idx`` maps ``(i,j,k)`` to its row (−1
    where pinned).

    The sign is chosen so ``A`` comes out **positive** definite (positive
    diagonal), unlike the mode solver's otherwise identical assembly. That is
    not cosmetic: conjugate gradients requires it, and CG is the only way this
    scales to a 3D grid.
    """
    shape = fixed.shape
    free_mask = ~fixed
    n_free = int(free_mask.sum())
    free_idx = -np.ones(shape, dtype=np.int64)
    free_idx[free_mask] = np.arange(n_free)

    fixed_idx = -np.ones(shape, dtype=np.int64)
    fixed_idx[fixed] = np.arange(int(fixed.sum()))

    diag = np.zeros(shape, dtype=np.float64)
    rows_A, cols_A, data_A = [], [], []
    rows_B, cols_B, data_B = [], [], []

    # Each face is visited twice — once from each side — sharing one coefficient
    # array, which is what makes A symmetric by construction rather than by
    # arithmetic luck.
    for axis, coef in enumerate(coefs):
        if coef.size == 0:
            continue
        lo, hi = _face_slices(axis)
        for src, nbr in ((lo, hi), (hi, lo)):
            i_src = free_idx[src]
            live = i_src >= 0
            diag[src] += np.where(live, coef, 0.0)

            i_nbr = free_idx[nbr]
            to_free = live & (i_nbr >= 0)
            rows_A.append(i_src[to_free])
            cols_A.append(i_nbr[to_free])
            data_A.append(-coef[to_free])

            j_nbr = fixed_idx[nbr]
            to_fixed = live & (j_nbr >= 0)
            rows_B.append(i_src[to_fixed])
            cols_B.append(j_nbr[to_fixed])
            data_B.append(coef[to_fixed])

    rows_A.append(free_idx[free_mask])
    cols_A.append(free_idx[free_mask])
    data_A.append(diag[free_mask])

    cat = lambda parts: (np.concatenate(parts) if parts else np.empty(0))
    A = coo_matrix((cat(data_A), (cat(rows_A), cat(cols_A))),
                   shape=(n_free, n_free)).tocsr()
    B = coo_matrix((cat(data_B), (cat(rows_B), cat(cols_B))),
                   shape=(n_free, int(fixed.sum()))).tocsr()
    return A, B, free_idx


# ====================================================================== #
# Derived quantities
#
# Everything below is computed from the *same* face coefficients the operator
# was assembled from, never from a fresh discretisation of the same integral.
# That is what makes the answers consistent with each other rather than merely
# close: the reported charge is exactly the flux the solved equations balanced,
# and the reported energy is exactly the quadratic form of the operator that
# produced φ. Re-deriving either — integrating ½ε|E|² over cells, say — would
# introduce a second discretisation whose disagreement with the first is
# indistinguishable from a bug.
# ====================================================================== #

def _node_flux(coefs, phi: np.ndarray) -> np.ndarray:
    """Net outward ε-weighted flux from each node's dual cell.

    ``Σ_faces c_f (φ_n − φ_nbr)`` at every node, which is Gauss's law over that
    node's control volume divided by ε₀: multiply by ε₀ and it is the enclosed
    charge in coulombs.

    Free nodes come out at zero to solver accuracy — that *is* the equation that
    was solved — so the charge lands entirely on pinned nodes, conductor surfaces
    and Dirichlet walls. That makes this its own consistency check.
    """
    out = np.zeros_like(phi)
    for axis, coef in enumerate(coefs):
        if coef.size == 0:
            continue
        lo, hi = _face_slices(axis)
        flux = coef * (phi[lo] - phi[hi])
        out[lo] += flux
        out[hi] -= flux
    return out


def _field_energy(coefs, phi: np.ndarray) -> float:
    """Electrostatic energy in joules, ``½ε₀ Σ_faces c_f (Δφ_f)²``.

    Each term is ``ε·(A·d)·(Δφ/d)² = ε·V_face·E²``, so the sum is the usual
    ``∫ε|E|²`` with every face weighted by the volume it owns — the quadratic
    form of the operator, evaluated on its own solution.
    """
    total = 0.0
    for axis, coef in enumerate(coefs):
        if coef.size == 0:
            continue
        lo, hi = _face_slices(axis)
        total += float(np.sum(coef * (phi[hi] - phi[lo]) ** 2))
    return 0.5 * EPS0 * total


def _gradient(grid: FDTDGrid, phi: np.ndarray,
              lengths: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    """``E = −∇φ`` on the Yee E-edges, as three ``(Nx, Ny, Nz)`` arrays.

    ``Ex[i,j,k] = −(φ[i+1,j,k] − φ[i,j,k]) / Lx[i,j,k]`` — the edge joining the
    two nodes, which is exactly where ``Ex`` lives, so this needs no
    interpolation and is not an approximation of the staggering but a statement
    of it.

    The divisor is the edge's **open** length from :func:`_open_lengths`, which
    on a staircase grid is just its width. That is the same field the conformal
    FDTD carries on that edge (:meth:`wavesim.mode_solver.TEMMode._staggered_port_fields`
    divides by the identical quantity) and the same one the operator's
    coefficients were built from, so the potential, the field and the charge are
    three readings of one discretisation rather than three discretisations.

    A fully covered edge has no open length and is left at zero — E genuinely
    vanishes inside a conductor. So is the last index along each axis, which has
    no node beyond it to difference against, matching the FDTD arrays, whose
    final edge lies on the domain boundary.
    """
    out = []
    for axis in range(3):
        E = np.zeros_like(phi)
        if phi.shape[axis] > 1:
            lo, hi = _face_slices(axis)
            L = np.broadcast_to(lengths[axis], phi.shape)[lo]
            np.divide(-(phi[hi] - phi[lo]), L, out=E[lo], where=L > 0.0)
        out.append(E)
    return tuple(out)


def _to_nodes(components: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    """Collocate three edge-centred components onto the nodes.

    The three components live on three different edge families, so any question
    about the vector at a point — its magnitude, its direction, a plot of one
    component against another's coordinates — needs them brought to one place
    first. Each node averages the two collinear edges meeting there.

    The walls are one-sided rather than averaged. Node 0 has no edge below it,
    and the last node's edge *above* it is not carried by the arrays (``N``
    cells, ``N`` slots, the ``N``-th node absent), so at both ends the single
    edge that exists is the answer. Averaging the top node against the absent
    edge instead would read exactly half the field there — the same missing-end
    trap :func:`_node_dual` documents for the control volumes.

    This is display-grade interpolation. Every quantitative result in this
    module uses the components where they actually live.
    """
    out = []
    for axis, E in enumerate(components):
        centred = np.array(E, dtype=np.float64)
        n = E.shape[axis]
        if n > 1:
            lo, hi = _face_slices(axis)
            centred[hi] = 0.5 * (E[hi] + E[lo])
            end = [slice(None)] * 3
            end[axis] = -1
            prev = [slice(None)] * 3
            prev[axis] = n - 2
            centred[tuple(end)] = E[tuple(prev)]
        out.append(centred)
    return tuple(out)


def _pad_face_array(face: np.ndarray, axis: int, shape) -> np.ndarray:
    """Grow a per-face array to full grid shape, zero in the missing last slot."""
    out = np.zeros(shape, dtype=np.float64)
    lo, _ = _face_slices(axis)
    if face.size:
        out[lo] = face
    return out


# ====================================================================== #
# Boundary conditions
# ====================================================================== #

def _resolve_boundary(spec) -> Dict[str, object]:
    """Normalise a boundary specification into ``{face: 'neumann' | volts}``.

    Accepts ``'neumann'`` or ``'ground'`` for all six faces at once, or a dict
    keyed by face name with ``'*'`` supplying the default for any face left out.
    A number means Dirichlet at that potential; ``'ground'`` is shorthand for
    ``0.0``.
    """
    if isinstance(spec, str) or isinstance(spec, (int, float)):
        spec = {'*': spec}
    if not isinstance(spec, dict):
        raise TypeError(f"boundary must be a string, a number or a dict, "
                        f"got {type(spec).__name__}")

    unknown = set(spec) - set(FACES) - {'*'}
    if unknown:
        raise ValueError(f"unknown boundary face(s) {sorted(unknown)}; "
                         f"expected any of {list(FACES)} or '*'")

    default = spec.get('*', 'neumann')
    return {f: _one_bc(spec.get(f, default), f) for f in FACES}


def _one_bc(value, face: str):
    """Validate a single face's condition."""
    if isinstance(value, str):
        if value == 'neumann':
            return 'neumann'
        if value == 'ground':
            return 0.0
        raise ValueError(
            f"boundary[{face!r}] = {value!r}: expected 'neumann', 'ground', or "
            f"a number (Dirichlet at that potential in volts)")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"boundary[{face!r}] = {value!r}: expected 'neumann', "
                         f"'ground', or a number")
    return float(value)


def _boundary_layer(shape, face: str):
    """Index tuple of the node layer belonging to ``face``.

    ``xmax`` is node ``Nx-1``, the last node the arrays carry — half a cell
    inside the geometric domain edge, in exactly the way every other wavesim
    array is.
    """
    axis = {'x': 0, 'y': 1, 'z': 2}[face[0]]
    sl = [slice(None)] * 3
    sl[axis] = 0 if face[1:] == 'min' else shape[axis] - 1
    return tuple(sl)


# ====================================================================== #
# Result
# ====================================================================== #

@dataclass
class ElectrostaticSolution:
    """Result of one electrostatic solve.

    ``phi`` is the potential in volts on the Yee nodes, shape ``(Nx, Ny, Nz)``.
    The fields and integrals derived from it are computed on first access and
    cached; all of them reuse ``coefs``, the face conductances the operator was
    built from, so they agree with φ and with each other by construction.
    """
    phi: np.ndarray
    grid: FDTDGrid
    potentials: Dict[str, float]
    boundary: Dict[str, object]
    grounded_bodies: int
    method: str
    n_unknowns: int
    iterations: int = 0
    node_pec: np.ndarray = field(default=None, repr=False)
    coefs: Tuple[np.ndarray, ...] = field(default=None, repr=False)
    open_lengths: Tuple[np.ndarray, ...] = field(default=None, repr=False)
    body_labels: np.ndarray = field(default=None, repr=False)
    part_body: Dict[str, int] = field(default_factory=dict, repr=False)

    # -- potential ------------------------------------------------------ #

    def potential_at(self, x: float, y: float, z: float) -> float:
        """φ at the node nearest a physical position, in volts."""
        return float(self.phi[self.grid.position_to_index(x, y, z)])

    # -- fields --------------------------------------------------------- #

    @cached_property
    def E(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Ex, Ey, Ez)`` in V/m, on the Yee E-edges.

        Same staggering and same shapes as ``grid.Ex``/``Ey``/``Ez``, so these
        drop straight into the existing monitors and plotting helpers. E comes
        out identically zero inside a conductor without being masked: every edge
        there joins two nodes of one body, which the solve holds at one
        potential. Under conformal PEC it is the *open* part of each edge that
        carries the field, which is what the divisor accounts for.
        """
        return _gradient(self.grid, self.phi, self._lengths)

    @property
    def _lengths(self) -> Tuple[np.ndarray, ...]:
        """The open edge lengths the operator was built from.

        Recomputed only for a solution assembled by hand rather than by
        :meth:`Electrostatics.solve`; the geometry is a property of the grid, so
        the two agree.
        """
        if self.open_lengths is None:
            return _open_lengths(self.grid)[0]
        return self.open_lengths

    @cached_property
    def D(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Dx, Dy, Dz)`` in C/m², ``D = ε₀ ε_r E`` on the same edges.

        The permittivity is the *face* permittivity :func:`_face_eps` produced
        for the operator, not the raw ``eps_*`` array. On a face straddling a
        conductor surface those differ — the raw array holds whatever the
        voxeliser left inside the metal — and using the raw value would make the
        flux through the conductor surface, hence its charge, inconsistent with
        the equations that were actually solved.
        """
        eps = (self.grid.eps_x, self.grid.eps_y, self.grid.eps_z)
        shape = self.phi.shape
        out = []
        for axis, E in enumerate(self.E):
            face = _face_eps(eps[axis], self.node_pec, axis)
            out.append(EPS0 * _pad_face_array(face, axis, shape) * E)
        return tuple(out)

    @cached_property
    def E_nodes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Ex, Ey, Ez)`` in V/m collocated onto the nodes, for display.

        The same three components as :attr:`E`, brought to one sample point by
        :func:`_to_nodes` so that they share φ's coordinate grid and can be
        drawn, differenced or combined against each other. Interpolated, so not
        what to measure with — see :attr:`E`.
        """
        return _to_nodes(self.E)

    @cached_property
    def D_nodes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Dx, Dy, Dz)`` in C/m² collocated onto the nodes, for display."""
        return _to_nodes(self.D)

    def E_magnitude(self) -> np.ndarray:
        """``|E|`` in V/m on the nodes."""
        return np.sqrt(sum(c ** 2 for c in self.E_nodes))

    def D_magnitude(self) -> np.ndarray:
        """``|D|`` in C/m² on the nodes."""
        return np.sqrt(sum(c ** 2 for c in self.D_nodes))

    # -- integrals ------------------------------------------------------ #

    @cached_property
    def energy(self) -> float:
        """Total electrostatic energy in the domain, in joules."""
        return _field_energy(self.coefs, self.phi)

    @cached_property
    def node_charge(self) -> np.ndarray:
        """Charge in coulombs enclosed by each node's dual cell.

        Nonzero only on conductor surfaces and Dirichlet walls; free nodes carry
        no net charge, which is the equation that was solved.
        """
        return EPS0 * _node_flux(self.coefs, self.phi)

    def charge(self, name: str) -> float:
        """Total charge in coulombs on the named conductor.

        Gauss's law over a surface enclosing the part's body: the flux out of
        every node the conductor occupies, which telescopes to the flux through
        its surface because interior nodes are surrounded by their own potential.
        """
        body = self.part_body.get(name)
        if body is None:
            known = ", ".join(sorted(self.part_body)) or "none"
            raise KeyError(f"no conductor named {name!r} in this solution; "
                           f"available: {known}")
        return float(self.node_charge[self.body_labels == body].sum())


# ====================================================================== #
# Solver
# ====================================================================== #

class Electrostatics:
    """Assign potentials to named PEC parts and solve for the field.

        >>> es = Electrostatics(grid)
        >>> es.set_potential('trace', 5.0)
        >>> es.set_potential('ground_plane', 0.0)
        >>> sol = es.solve(boundary='ground')

    Potentials are assigned to parts (see :mod:`wavesim.parts`) but applied to
    *bodies*: if two named parts turn out to be the same lump of metal, they
    hold one potential, because that is what a conductor does. Assigning them
    different ones is refused rather than averaged — it is a modelling error,
    usually two CAD solids clearing each other by less than a cell.

    A conductor nobody assigned a potential to is grounded at 0 V, with a
    warning saying how many were. That is the useful default (an enclosure or a
    shield is normally exactly that) and it is also what a mistyped part name
    looks like, so it is worth saying out loud.
    """

    def __init__(self, grid: FDTDGrid):
        self.grid = grid
        self.potentials: Dict[str, float] = {}

    # -- setup ---------------------------------------------------------- #

    def set_potential(self, name: str, volts: float) -> 'Electrostatics':
        """Hold the named PEC part at ``volts``. Returns self, so calls chain."""
        part_id(self.grid, name)          # raises, listing the names that exist
        self.potentials[name] = float(volts)
        return self

    def ground(self, name: str) -> 'Electrostatics':
        """Hold the named part at 0 V — ``set_potential(name, 0.0)``."""
        return self.set_potential(name, 0.0)

    # -- solve ---------------------------------------------------------- #

    def solve(self, boundary='ground', method: str = 'auto',
              rtol: float = 1e-10, maxiter: int = None,
              rho=None) -> ElectrostaticSolution:
        """Solve for the potential.

        Parameters
        ----------
        boundary : str, number or dict
            Condition on the domain box. ``'neumann'`` (zero normal field — a
            symmetry plane) or ``'ground'`` (φ = 0) applies to all six faces; a
            number applies Dirichlet at that potential; a dict keyed by
            ``'xmin'``…``'zmax'`` sets faces individually, with ``'*'`` as the
            default for the rest. Defaults to a grounded box, matching
            :func:`wavesim.mode_solver.solve_tem_modes`.
        method : {'auto', 'direct', 'cg'}
            ``'auto'`` (the default) means conjugate gradients at every size —
            measured to beat the direct factorisation by 10x to 400x in 3D, with
            no crossover in between. ``'direct'`` factorises instead: exact, with
            no tolerance to tune, which makes it the reference to compare
            against when a discretisation is under test, and the thing to retry
            if CG converges badly. See the note beside :data:`AUTO_METHOD` for
            the numbers.
        rtol, maxiter : float, int
            Conjugate-gradient tolerance and iteration cap. Ignored by the
            direct path, which has no tolerance to set.
        rho : None
            Space charge density in C/m³. **Not implemented.** The formulation
            and the right-hand side carry the term — it enters as
            ``ρ·V_dual/ε₀`` at each free node, the operator being assembled in
            *relative* permittivity — so adding it is a small change, but
            nothing has needed it yet and an untested path is worse than an
            absent one.

        Returns
        -------
        ElectrostaticSolution
        """
        if rho is not None:
            raise NotImplementedError(
                "space charge (rho) is not implemented; every source today is a "
                "conductor potential, which makes this Laplace's equation. The "
                "right-hand side hook is in place: rho enters as rho*V_dual/eps0 "
                "at each free node.")

        grid = self.grid
        chosen = self._choose(method)          # validate before doing any work
        bc = _resolve_boundary(boundary)
        node_pec = pec_node_mask(grid)

        labels, n_bodies = conductor_bodies(grid)
        occupants = body_parts(grid)
        part_body = {name: b + 1
                     for b, names in enumerate(occupants) for name in names}

        fixed, phi_fixed, n_grounded = self._pin_conductors(
            labels, n_bodies, occupants)
        self._pin_boundary(bc, fixed, phi_fixed)

        if not fixed.any():
            raise ValueError(
                "the problem is singular: no conductor carries a potential and "
                "every domain face is Neumann, so phi is only determined up to "
                "an additive constant. Ground a part, or make at least one face "
                "Dirichlet (e.g. boundary='ground').")

        lengths, n_clamped = _open_lengths(grid)
        if n_clamped:
            warnings.warn(
                f"{n_clamped} cut edge(s) were open by less than "
                f"{MIN_OPEN_FRACTION:g} of their length and were clamped to it. "
                f"A conductor face lying on the node ruler voxelises to a "
                f"round-off fraction rather than to zero, and that is what this "
                f"usually is — harmless, since such an edge is a Dirichlet pin "
                f"either way. Genuine slivers that fine are below the resolution "
                f"at which a cut cell means anything; move the geometry off the "
                f"ruler or coarsen the mesh.", stacklevel=2)

        coefs = _face_coefs(grid, node_pec, lengths)
        A, B, free_idx = _assemble(coefs, fixed)
        n_free = A.shape[0]

        phi = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=np.float64)
        phi[fixed] = phi_fixed[fixed]

        iterations = 0
        if n_free:
            b = B @ phi_fixed[fixed]
            x, iterations = _linear_solve(A, b, chosen, rtol, maxiter)
            phi[~fixed] = x

        return ElectrostaticSolution(
            phi=phi, grid=grid, potentials=dict(self.potentials), boundary=bc,
            grounded_bodies=n_grounded, method=chosen, n_unknowns=n_free,
            iterations=iterations, node_pec=node_pec, coefs=coefs,
            open_lengths=lengths, body_labels=labels, part_body=part_body)

    # -- pinning -------------------------------------------------------- #

    def _pin_conductors(self, labels, n_bodies, occupants):
        """Pin every conductor node to its body's potential.

        Works body by body rather than part by part, because a body is the thing
        that physically holds one potential. A body with no named part is
        grounded; a body with one is held at it, *including* any unnamed metal
        fused to it — that metal is the same conductor, so grounding it instead
        would be inventing a short. A body claimed at two different potentials
        has no solution and raises.
        """
        grid = self.grid
        shape = (grid.Nx, grid.Ny, grid.Nz)
        fixed = np.zeros(shape, dtype=bool)
        phi_fixed = np.zeros(shape, dtype=np.float64)

        if n_bodies == 0:
            return fixed, phi_fixed, 0

        n_grounded = 0
        for body in range(1, n_bodies + 1):
            names = [n for n in occupants[body - 1] if n in self.potentials]
            values = {self.potentials[n] for n in names}
            if len(values) > 1:
                detail = ", ".join(f"{n}={self.potentials[n]} V" for n in names)
                raise ValueError(
                    f"parts {names} are electrically the same conductor but "
                    f"were given different potentials ({detail}). A conductor "
                    f"holds one potential, so this has no solution — the usual "
                    f"cause is two solids that clear each other in CAD by less "
                    f"than a cell. wavesim.parts.check_shorts() lists every "
                    f"such join.")
            if values:
                volts = values.pop()
            else:
                volts = 0.0
                n_grounded += 1
            here = labels == body
            fixed |= here
            phi_fixed[here] = volts

        if n_grounded:
            warnings.warn(
                f"{n_grounded} conductor(s) had no assigned potential and were "
                f"grounded at 0 V. That is usually intended (a shield or an "
                f"enclosure); if it is not, it is what a mistyped part name "
                f"looks like — wavesim.parts.describe_conductors(grid) lists "
                f"what is in the model.", stacklevel=3)

        return fixed, phi_fixed, n_grounded

    def _pin_boundary(self, bc, fixed, phi_fixed):
        """Pin the Dirichlet domain faces, refusing to short a live conductor."""
        shape = fixed.shape
        for face, value in bc.items():
            if value == 'neumann':
                continue
            layer = _boundary_layer(shape, face)
            clash = fixed[layer] & (phi_fixed[layer] != value)
            if clash.any():
                volts = np.unique(phi_fixed[layer][clash])
                raise ValueError(
                    f"boundary[{face!r}] holds that face at {value} V, but a "
                    f"conductor at {volts.tolist()} V touches it. That is a "
                    f"short with no solution. Either move the conductor off the "
                    f"face, make the face Neumann, or set it to the same "
                    f"potential.")
            fixed[layer] = True
            phi_fixed[layer] = value

    @staticmethod
    def _choose(method: str) -> str:
        if method not in ('auto', 'direct', 'cg'):
            raise ValueError(f"method must be 'auto', 'direct' or 'cg', "
                             f"got {method!r}")
        return AUTO_METHOD if method == 'auto' else method


def _linear_solve(A: csr_matrix, b: np.ndarray, method: str,
                  rtol: float, maxiter) -> Tuple[np.ndarray, int]:
    """Solve ``A x = b`` for a symmetric positive-definite ``A``."""
    if method == 'direct':
        return splu(A.tocsc()).solve(b), 0

    # Jacobi (diagonal) preconditioning: the cheapest thing that helps, and it
    # helps most where it matters, because a strong permittivity contrast shows
    # up directly on the diagonal. If a large model ever converges too slowly
    # this is the knob to replace (algebraic multigrid), not the method.
    d = A.diagonal()
    M = LinearOperator(A.shape, matvec=lambda v: v / d)

    count = 0

    def tick(_):
        nonlocal count
        count += 1

    x, info = cg(A, b, rtol=rtol, maxiter=maxiter, M=M, callback=tick)
    if info > 0:
        raise RuntimeError(
            f"conjugate gradients did not converge in {count} iterations "
            f"(requested rtol={rtol:g}). Raise maxiter, relax rtol, or use "
            f"method='direct' if the problem is small enough.")
    if info < 0:
        raise RuntimeError(f"conjugate gradients failed with info={info}")
    return x, count


# ====================================================================== #
# Capacitance
# ====================================================================== #

@dataclass
class CapacitanceMatrix:
    """Capacitance between a set of named conductors, in farads.

    ``maxwell[i][j]`` is ``∂Q_i/∂V_j`` with every other conductor held at 0 V —
    the *Maxwell* (short-circuit) matrix, which is what a field solver measures
    directly and what circuit extraction consumes. Its diagonal is positive, its
    off-diagonals negative (raising one conductor pulls negative charge onto its
    neighbours), and each row sums to the capacitance from that conductor to
    whatever plays the role of ground.

    :meth:`mutual` converts to the two-terminal capacitances people draw as
    lumped components between pairs of pins. The two are routinely confused, and
    reporting one under the other's name is off by more than a sign, so both are
    named explicitly rather than one being "the" capacitance matrix.
    """
    names: Tuple[str, ...]
    maxwell: np.ndarray

    def mutual(self) -> np.ndarray:
        """Two-terminal capacitances: off-diagonal ``−C_ij``, diagonal to ground.

        ``mutual[i][j]`` (i ≠ j) is the lumped capacitor between conductors i and
        j; ``mutual[i][i]`` is the one from conductor i to ground, i.e. the row
        sum of the Maxwell matrix.
        """
        out = -self.maxwell.copy()
        np.fill_diagonal(out, self.maxwell.sum(axis=1))
        return out

    def between(self, a: str, b: str) -> float:
        """Two-terminal capacitance in farads between two named conductors."""
        i, j = self.names.index(a), self.names.index(b)
        return float(self.mutual()[i, j])

    def to_ground(self, a: str) -> float:
        """Capacitance in farads from one conductor to ground."""
        i = self.names.index(a)
        return float(self.maxwell[i].sum())


def capacitance_matrix(grid: FDTDGrid, names=None, *, boundary='ground',
                       **solve_kw) -> CapacitanceMatrix:
    """Extract the capacitance matrix by energising one conductor at a time.

    Runs one solve per conductor with that conductor at 1 V and the rest at 0 V,
    reading a column of the matrix off the resulting charges. There is no
    cheaper route: the columns are genuinely independent solutions, and reusing
    a factorisation across them was measured not to pay (see :data:`AUTO_METHOD`).

    Parameters
    ----------
    names : sequence of str, optional
        Conductors to include, defaulting to every named part. Parts that turn
        out to be the same body are refused, since they cannot be driven
        independently.
    boundary : str, number or dict
        As :meth:`Electrostatics.solve`. Defaults to a grounded box, which makes
        the box the reference conductor and gives every conductor a capacitance
        to ground. With an all-Neumann box instead, the rows sum to zero and the
        Maxwell matrix is rank-deficient — correctly so, since there is no
        ground to have a capacitance to — while the mutual capacitances remain
        exact. That is the normal way to measure a shielded structure, not an
        error. Note that a conductor touching a Dirichlet face is shorted to it
        as soon as this drives that conductor, and says so.
    **solve_kw
        Passed through to :meth:`Electrostatics.solve` (``method``, ``rtol``, …).

    Returns
    -------
    CapacitanceMatrix
    """
    names = tuple(names if names is not None else sorted(grid.pec_names or {}))
    if len(names) < 1:
        raise ValueError("no named PEC parts to extract capacitance between; "
                         "name them at placement (see wavesim.parts)")

    bodies = body_parts(grid)
    owner = {n: b for b, group in enumerate(bodies) for n in group}
    seen = {}
    for n in names:
        if n not in owner:
            part_id(grid, n)      # raises with the list of names, if unknown
            raise ValueError(f"part {n!r} occupies no conductor body")
        if owner[n] in seen:
            raise ValueError(
                f"parts {seen[owner[n]]!r} and {n!r} are the same conductor, so "
                f"they cannot be driven independently and have no capacitance "
                f"between them. Drop one, or check wavesim.parts.check_shorts().")
        seen[owner[n]] = n

    n_cond = len(names)
    C = np.zeros((n_cond, n_cond), dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for j, driven in enumerate(names):
            es = Electrostatics(grid)
            for n in names:
                es.set_potential(n, 1.0 if n == driven else 0.0)
            sol = es.solve(boundary=boundary, **solve_kw)
            for i, probe in enumerate(names):
                C[i, j] = sol.charge(probe)
    # The per-solve grounding notice is identical every time; say it once.
    for w in caught[:1]:
        warnings.warn(w.message, UserWarning, stacklevel=2)

    # A matrix whose rows sum to zero is *not* an error: with no path to ground
    # the Maxwell matrix is genuinely rank-deficient, and the mutual
    # capacitances in it are still exactly right — a shielded coax is the
    # ordinary case. What is useless is a matrix that is numerically all zero,
    # which means nothing in the model couples to anything: a lone conductor in
    # a Neumann box, whose field has nowhere to go. Scaled against ε₀·(domain
    # size), the capacitance any real structure of this size would have, so the
    # comparison does not depend on the units the model was built in.
    span = max(float(grid.x[-1] - grid.x[0]), float(grid.y[-1] - grid.y[0]),
               float(grid.z[-1] - grid.z[0]))
    if np.abs(C).max() <= 1e-9 * EPS0 * span:
        raise ValueError(
            "every extracted capacitance is zero: nothing in the model couples "
            "to anything else. A single conductor inside an all-Neumann box has "
            "no capacitance, because no flux can leave it. Give the field "
            "somewhere to terminate — boundary='ground', or a second conductor.")
    return CapacitanceMatrix(names=names, maxwell=C)
