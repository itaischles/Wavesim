"""
magnetostatics.py — 3D magnetostatic solver on the FDTD grid.

Drive some currents and ask what field they make, and how much flux each one
links. Formally

    ``∇×(ν ∇×A) = J``,   ``ν = 1/(µ₀ µ_r)``,   ``B = ∇×A``

for the magnetic vector potential ``A``, with perfect conductors expelling the
field. Like :mod:`wavesim.electrostatics` this is a boundary-value problem, not
a time-stepping one: nothing here reads ``dt``, ``Ex..Hz`` or the CFL
condition. It shares the grid because the geometry is the same geometry, so a
model is built once and either static solver, or the FDTD, can run on it.

Where the potential lives
-------------------------
``A`` sits on the Yee **E edges** and ``B`` on the **H faces**. That makes the
operator the FDTD's own: ``B = ∇×A`` is the Faraday contour around each H face
(:func:`wavesim.update.update_H`, conformal form included), and Ampère's law
``∇×H = J`` is the dual contour around each E edge. The solve is the static
limit of the two curl operators the time-stepper alternates between.

The unknown is not ``A`` itself but its **line integral along the edge**,
``u = A·L`` in webers, ``L`` being the edge's open length — the same ``E·L`` the
conformal H update differences. In those variables the curl is a pure ±1
incidence matrix ``D`` (face × edge):

    ``Φ_f = Σ_edges ±u_e``              flux through face f, exactly
    ``(Dᵀ W Φ)_e = I_e``                Ampère around edge e's dual face

with ``W = diag(ν_f h_f / A_f)``: ``A_f`` the face's open area, ``h_f`` the
dual length through it (the node dual width along the face normal, half a cell
on a wall), and ``I_e`` the current, in amperes, crossing the edge's dual face.
``W`` makes ``Dᵀ W Φ`` the circulation ``Σ H_f h_f`` of ``H = νB = νΦ/A``
around the dual face, so the second line *is* the integral Ampère law, and
``K = Dᵀ W D`` is symmetric positive semi-definite by construction. Its
quadratic form ``½ uᵀKu = ½ Σ_f ν Φ_f² h_f / A_f`` is the field energy
``½∫B·H dV``; the source term ``uᵀI`` is ``∫A·J dV``.

Cut cells enter in exactly the two places the conformal H update puts them.
``pec_face_open_*`` scales the face area ``A_f``, and ``pec_edge_open_*``
scales the edge length ``L`` that turns ``u`` back into ``A``. The incidence
``D`` is topology and never changes. Fractions of 1.0 everywhere therefore
multiply every weight by exactly 1.0, so the conformal path reproduces the
staircase assembly bit for bit. Unlike the FDTD, the face area is **not**
clamped at ``conformal_area_threshold``. That clamp bounds ``dt/(µA)`` for
stability, and statics has no ``dt``; :mod:`wavesim.electrostatics` made the
same call for the same reason. A face is dropped from the operator only when
:func:`wavesim.pec.build_conformal_edge_masks` would call it covered (open
fraction ≤ ``COVERED_FRACTION_TOL``), and then all four of its edges are
pinned anyway.

Perfect conductors expel flux
-----------------------------
PEC is the high-frequency (skin-effect) limit: no field inside, so the
inductance reported is the **external** inductance only. It is imposed by
pinning ``u = 0`` on every edge the FDTD zeroes E on, taken from the same
``wavesim.parts`` route electrostatics uses to find conductor nodes
(:func:`wavesim.parts.pec_node_mask` is built from the same masks), so the
three solvers agree on where the metal is. Tangential ``A = 0`` on a surface
makes ``B·n = 0`` there, which is flux expulsion.

The current a PEC body carries is never imposed. It is an **output**: the
jump of tangential H across the surface, read off as Ampère's law evaluated on
the pinned edges, where the solve does not enforce it
(:meth:`MagnetostaticSolution.surface_current`).

A set of nodes held together by pinned edges — a PEC body, a flux-wall box
face, a port's terminal sheet, or any of these touching — is called a
**group** below. Groups are the magnetostatic counterpart of electrostatic
conductors, and they carry the one solvability condition: the net source
current into each group must vanish (KCL). A current that ran into a
perfect conductor and nowhere out of it would need a time-varying charge
pile-up, which statics cannot represent.

Gauge
-----
``K`` is singular: any gradient ``u = Gᵀφ`` has zero curl, so it adds nothing
to B and is invisible to ``K``. A **grad-div penalty** removes that freedom:

    ``K + γ Mᵀ diag(τ) M``

``M`` is the discrete divergence of ``A`` — the flux of ``A`` out of each
node's dual cell, ``Σ ± u_e S_e/L_e`` with ``S_e`` the dual-face area.
``τ = ν/V`` uses the node's dual volume, and ``γ`` is the user's ``gauge``
weight. On a uniform mesh ``Dᵀ D + Gᵀ G`` is the discrete vector Laplacian,
so the penalised operator is Laplacian-like and conjugate gradients behaves
as it does for the electrostatic one. This is the Coulomb gauge imposed
softly.

``B`` does not depend on ``γ``, and the reason is exact, not asymptotic.
Suppose the source is discretely divergence-free: zero at every free node,
and summing to zero over every group. Then some ``u*`` satisfies both
``K u* = I`` and ``M u* = 0``. That ``u*`` is the unique solution of the
penalised system for **every** ``γ > 0``, and the penalty only selects
which member of the gradient family ``u* + Gᵀφ`` is returned. One detail
makes it exact. Each group has a single constant ``φ``, so a group gets
**one** divergence row, the sum over its nodes, not one row per node. That
is the electrostatic floating-body collapse (see
:mod:`wavesim.electrostatics`, "Floating conductors") applied to the gauge,
and it matches the constraints one for one with the gradients they
suppress. ``test_B_does_not_depend_on_the_gauge_weight`` measures it.

Sources
-------
**Closed filament loops** are a polyline in metres and a current. The edge
current is ``I`` times the signed number of times the curve crosses each
edge's **dual face**. Those faces tile the dual-cell boundaries, so a closed
curve leaves every dual cell exactly as often as it enters. The discrete
divergence is then zero at every node, in integers, before ``I`` multiplies
it. The crossings are found by walking the curve through the dual cells and
tracking the current cell explicitly: each plane crossing moves it one face,
so even a curve through a dual-cell corner is counted consistently.
(:func:`wavesim.monitors._bin_crossings` bins a path onto edges for a
different purpose and is not divergence-free, which is why it is not reused.)

A filament's **self**-inductance depends on the mesh. On the grid it is a wire
roughly a fraction of a cell thick, and the self-inductance of a thin wire
diverges logarithmically as it thins. Mutual inductance between separate loops
converges normally. Use filaments to excite and to couple, and do not read an
absolute self-inductance off one.

**Current ports** are two terminal faces facing each other across a gap in a
conductor path, a channel of cells spanning the gap, and a current ``I``
positive from terminal a to terminal b. The caller supplies the terminals as
boolean **node** masks (voxelised CAD faces) and the channel as a boolean
**cell** mask. Each terminal is treated as a perfectly conducting contact
sheet in both solves: its in-plane edges are pinned here, and its nodes are an
equipotential in the current-flow solve. Inside the channel the source current
is a uniform-conductivity flow from a to b, one Laplace solve over the
channel's nodes. The return current through the conductor comes from the
physics, and how depends on what the conductor is:

* **PEC**. The terminals sit on one PEC group. The channel current enters it
  at b and leaves at a, and the curl-curl solve decides how it flows in
  between, as a surface current. Inductance is external only.
* **Finite σ**. A DC current-flow pre-solve ``∇·(σ∇V) = 0`` runs over the
  conductor's nodes, with the terminals as contacts. It is the electrostatic
  assembly with σ in place of ε. Any PEC piece in that circuit is a floating
  equipotential carrying no net current, via
  :func:`wavesim.electrostatics._floating_reduction`, the same collapse that
  fixes a floating conductor's charge. ``J = −σ∇V`` lands on the edges, as
  ``E = −∇φ`` does. B penetrates a σ body, so the inductance includes the
  internal part, and the same solve gives the DC resistance.

Lossy **dielectrics** carry σ too, and a σ-weighted flow through an FR4
substrate is not a return path anyone meant. The flow region is therefore
restricted twice: to a ``conductor`` cell mask, defaulting to cells with σ
above :data:`CONDUCTOR_SIGMA_THRESHOLD`, and to the part of it connected to
the port's terminals.

Before anything is solved, the total source current is checked for zero
discrete divergence at every free node, and for zero net current into every
group. A port whose terminals land on two bodies with no σ path between them
fails the second check. It is refused by name, before any solve, rather than
handed to a solver that would wander.

Boundary conditions
-------------------
Each face of the domain box gets one of two conditions:

**flux_wall** (the default): tangential ``A = 0``, so ``B·n = 0``. A
superconducting wall, which pushes the field back as an image current of
opposite sense. For this solver's KCL check, the flux-wall faces together
count as one PEC group, since they are pinned together along the box edges
they share.

**symmetry**: tangential ``H = 0``, the natural (do-nothing) condition of
the operator, exactly as Neumann is the do-nothing condition in
electrostatics. A magnetic mirror: field lines cross it at right angles. The
plane of a current loop is one, as is the midplane between two equal
co-directed loops.

There is no open boundary. As in electrostatics, an isolated structure is
approximated by a flux wall far enough away that moving it further stops
changing the answer. Flux walls *lower* inductance (their image currents
oppose the source), so a too-small box underestimates L.

The domain is the node box ``x[0] … x[N-1]`` on each axis, as in
electrostatics: an edge along an axis exists for indices ``0 … N-2`` there,
and the last slot of every edge and face array is always zero.

Linear solve
------------
SciPy only. :func:`wavesim.electrostatics._linear_solve` provides both paths.
The preconditioner is one replaceable function,
:func:`_preconditioner`. ``'auto'`` means conjugate gradients at every size;
the measurements behind that are beside :data:`AUTO_METHOD`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator

from wavesim.constants import MU0
from wavesim.grid import FDTDGrid
from wavesim.parts import _edge_masks, conductor_bodies, body_parts, part_id
from wavesim.pec import COVERED_FRACTION_TOL
from wavesim.electrostatics import (
    FACES, _assemble, _check_floating_coupled, _face_slices,
    _floating_reduction, _linear_solve, _node_dual, _node_flux,
    _open_lengths, _pad_face_array, _to_nodes, jacobi_preconditioner)

BOUNDARY_KINDS = ('flux_wall', 'symmetry')

# ---------------------------------------------------------------------- #
# Which cells count as a conductor for a port's current-flow solve, when
# the caller does not say. The cell value is the mean of the three
# conductivities stored on the cell's own edges (sigma_x/y/z[i,j,k]), which
# is what set_box writes for a box.
#
# 1e3 S/m sits above every lossy dielectric wavesim models (FR4 and other
# tan δ substrates are below 1e-2 S/m; silicon substrates, seawater and
# tissue are 0.1–10 S/m) and below anything one would call a conductor
# (graphite ~1e5, solder ~7e6, copper 5.8e7). Heavily doped silicon (up to
# ~1e4 S/m) sits near the line on purpose: whether it is a return path is a
# modelling decision, so pass ``conductor=`` explicitly there.
# ---------------------------------------------------------------------- #
CONDUCTOR_SIGMA_THRESHOLD = 1.0e3

# ---------------------------------------------------------------------- #
# The tolerance on the source's discrete divergence, in amperes per ampere
# of source current. A filament's divergence is exactly zero (integer
# crossing counts times I). A port's channel and conductor flows are
# divergence-free to their linear solve's accuracy. That is round-off
# (~1e-15) on the direct path, and FLOW_RTOL on CG. A genuine KCL violation
# is O(1). The gap between the two is twelve orders of magnitude, so the
# threshold is not delicate.
# ---------------------------------------------------------------------- #
DIVERGENCE_RTOL = 1e-9
FLOW_RTOL = 1e-13

# ---------------------------------------------------------------------- #
# Which linear solver 'auto' picks.
#
# The plan was to take a direct factorisation below a measured size and
# CG above it. The electrostatic module had already found no such size for
# its Laplacian (see the note beside wavesim.electrostatics.AUTO_METHOD),
# but this operator is different: three unknowns per node, a curl-curl plus
# a gauge penalty, and wider coupling. So it was measured again rather than
# assumed. Shorted coax with a port (tests/magnetostatic_shapes.py,
# staircase, uniform mesh), one solve, CG at rtol=1e-10 with Jacobi, direct
# on the augmented matrix of _GaugedOperator, on this machine:
#
#     unknowns     direct        CG    (iterations)
#        1,069     0.058 s    0.007 s     47
#        2,921     0.107 s    0.014 s     63
#        8,285      1.15 s    0.038 s     90
#       19,481      9.72 s    0.080 s    120
#       39,113     18.3 s     0.125 s    150
#       66,781     86.8 s     0.33 s     182
#      155,797       —        1.93 s     241
#
# There is no crossover. CG is 8x ahead at a thousand unknowns and 260x at
# sixty thousand, for the reason electrostatics gives (3D fill-in grows
# superlinearly, CG roughly linearly). Its iteration count grows like
# the mesh's linear size, as it does for a Laplacian, which is the grad-div
# gauge doing its job. 'auto' therefore means CG at every size. 'direct'
# stays as the exact, tolerance-free reference for tests.
#
# One trap found on the way, and the reason _GaugedOperator exists. The
# first assembly multiplied the gauge's per-group rows out, and on the
# 48-cell coax that gave 666 nonzeros per row instead of ~30. A flux wall's
# row touches thousands of edges, so its rank-one term is a dense block.
# CG ran 20x slower per iteration and the direct path ~4x slower, so both
# columns above were wrong until the group rows were kept factored.
# ---------------------------------------------------------------------- #
AUTO_METHOD = 'cg'


def _preconditioner(K: csr_matrix):
    """The CG preconditioner for the curl-curl operator.

    One function so it can be replaced wholesale. Algebraic multigrid (an
    auxiliary-space AMG in the Hiptmair–Xu sense is the textbook choice for a
    curl-curl operator) would go here. Today it is Jacobi, which is enough
    because the grad-div gauge makes the operator Laplacian-like.
    """
    return jacobi_preconditioner(K)


# ====================================================================== #
# Index helpers
# ====================================================================== #

def _others(axis: int) -> Tuple[int, int]:
    """The two axes ``(b, c)`` cyclically after ``axis``.

    ``(a, b, c)`` is then a right-handed triple, so a face normal to ``a`` has
    its Faraday contour running ``+b, +c, −b, −c``. Every curl in this module
    is written once in terms of it.
    """
    return (axis + 1) % 3, (axis + 2) % 3


def _spread(vec: np.ndarray, axis: int) -> np.ndarray:
    """``vec`` laid out along ``axis``, broadcastable against a 3D array."""
    shape = [1, 1, 1]
    shape[axis] = vec.size
    return np.asarray(vec, dtype=np.float64).reshape(shape)


def _edge_exists(shape) -> Tuple[np.ndarray, ...]:
    """Which slots of an edge array are real edges: index ``< N-1`` along the
    edge's own axis, whose far node is carried."""
    out = []
    for a in range(3):
        m = np.zeros(shape, dtype=bool)
        sl = [slice(None)] * 3
        sl[a] = slice(0, shape[a] - 1)
        m[tuple(sl)] = True
        out.append(m)
    return tuple(out)


def _face_box(shape, axis: int):
    """``(slices, sub_shape)`` of the faces normal to ``axis`` that exist.

    A face normal to ``a`` spans nodes ``n … n+e_b+e_c``, so it exists for
    indices ``< N-1`` along both ``b`` and ``c`` and for every index along
    ``a``. Returns ``None`` for a family with no faces at all (a degenerate
    ``N = 1`` axis among ``b``, ``c``).
    """
    b, c = _others(axis)
    sub = list(shape)
    sub[b] -= 1
    sub[c] -= 1
    if min(sub) <= 0:
        return None
    sl = [slice(None)] * 3
    sl[b] = slice(0, shape[b] - 1)
    sl[c] = slice(0, shape[c] - 1)
    return tuple(sl), tuple(sub)


def _shift_sum(arr: np.ndarray, axes) -> np.ndarray:
    """``out[n] = Σ arr[n − Σ_{ax∈S} e_ax]`` over all subsets ``S`` of ``axes``.

    The summing twin of :func:`wavesim.pec._dilate`. With the two axes
    perpendicular to an edge it collects the (up to) four cells around that
    edge, and with the two in-plane axes of a face family it collects the four
    faces meeting at a node.
    """
    out = np.array(arr, dtype=np.float64)
    for ax in axes:
        src = out.copy()
        dst = [slice(None)] * 3
        sl = [slice(None)] * 3
        dst[ax] = slice(1, None)
        sl[ax] = slice(0, -1)
        out[tuple(dst)] += src[tuple(sl)]
    return out


def _node_divergence(I: Tuple[np.ndarray, ...]) -> np.ndarray:
    """Net current (A) flowing *out of* each node's dual cell.

    ``I`` is edge-staggered in amperes: ``I[a][n]`` crosses the dual face of
    the edge from node ``n`` to ``n + e_a``, positive along ``+a``.
    """
    div = np.zeros(I[0].shape, dtype=np.float64)
    for a in range(3):
        lo, hi = _face_slices(a)
        div[lo] += I[a][lo]
        div[hi] -= I[a][lo]
    return div


# ====================================================================== #
# Geometry weights
# ====================================================================== #

def _dual_widths(grid: FDTDGrid):
    return (_node_dual(grid.dxp), _node_dual(grid.dyp), _node_dual(grid.dzp))


def _face_weights(grid: FDTDGrid) -> Tuple[np.ndarray, ...]:
    """``W`` per face family: ``ν·h / A_open`` in 1/H, shape ``(Nx,Ny,Nz)``.

    ``A_open`` is the primary face area ``dp_b·dp_c`` (the H face spans nodes,
    so its sides are primary widths, as in :func:`wavesim.pec.build_conformal_geometry`)
    times the open fraction on a conformal grid. ``h`` is the **dual** width
    along the face normal, the length of the dual edge piercing the face,
    which is what the Ampère contour around a neighbouring E edge runs along.

    A face the conformal rule calls covered gets weight zero. All four of its
    contour edges are pinned then (the tangency clause of
    :func:`wavesim.pec.build_conformal_edge_masks`), so the face could only
    ever carry ``Φ = 0``. No floor is applied to the open area of a live face.
    A face open by more than ``COVERED_FRACTION_TOL`` is real geometry, and
    its ``1/A`` coefficient is a strong coupling, not a hazard: there is no
    ``dt`` here for it to destabilise.

    The order of operations is fixed so that a conformal grid with every
    fraction at 1.0 produces bit-identical weights to the staircase path:
    ``(dp_b·dp_c)·1.0`` is exact.
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)
    dp = (grid.dxp, grid.dyp, grid.dzp)
    dual = _dual_widths(grid)
    mus = (grid.mu_x, grid.mu_y, grid.mu_z)
    fracs = (grid.pec_face_open_x, grid.pec_face_open_y, grid.pec_face_open_z)

    out = []
    for a in range(3):
        b, c = _others(a)
        area = _spread(dp[b], b) * _spread(dp[c], c)
        live = np.ones(shape, dtype=bool)
        if grid.is_conformal:
            f = np.asarray(fracs[a], dtype=np.float64)
            area = area * f
            live = f > COVERED_FRACTION_TOL
        nu_h = _spread(dual[a], a) / (MU0 * np.asarray(mus[a], np.float64))
        nu_h = np.broadcast_to(nu_h, shape)
        area = np.broadcast_to(area, shape)
        w = np.zeros(shape, dtype=np.float64)
        np.divide(nu_h, area, out=w, where=live & (area > 0.0))
        out.append(w)
    return tuple(out)


def _open_area(grid: FDTDGrid, axis: int) -> np.ndarray:
    """Open area (m²) of the faces normal to ``axis``, full shape."""
    shape = (grid.Nx, grid.Ny, grid.Nz)
    dp = (grid.dxp, grid.dyp, grid.dzp)
    b, c = _others(axis)
    area = _spread(dp[b], b) * _spread(dp[c], c)
    if grid.is_conformal:
        f = (grid.pec_face_open_x, grid.pec_face_open_y,
             grid.pec_face_open_z)[axis]
        area = area * np.asarray(f, dtype=np.float64)
    return np.broadcast_to(area, shape)


def _dual_areas(grid: FDTDGrid) -> Tuple[np.ndarray, ...]:
    """Area (m²) of each edge's dual face, broadcastable, one per axis.

    The product of the two node dual widths across the edge, which is the
    area :func:`wavesim.electrostatics._face_coefs` calls a face. It is the
    same dual face, seen from the edge it is pierced by.
    """
    dual = _dual_widths(grid)
    out = []
    for a in range(3):
        b, c = _others(a)
        out.append(_spread(dual[b], b) * _spread(dual[c], c))
    return tuple(out)


# ====================================================================== #
# Discrete operators
# ====================================================================== #

def _curl_incidence(shape):
    """The ±1 face × edge incidence ``D``, and the row layout of the faces.

    Columns are **global** edge ids, ``a·N + ravel(n)`` for an edge along
    ``a`` at node ``n``, so the edge vector is the three edge arrays raveled
    and concatenated. Rows are the existing faces, family by family, each
    family in C order over its :func:`_face_box`.

    Row ``f`` (normal ``a``) reads ``Φ = u_c[n+e_b] − u_c[n] − u_b[n+e_c] +
    u_b[n]``, the Faraday contour of :func:`wavesim.update.update_H`, so
    ``D`` is the FDTD's own curl, sign included.
    """
    N = int(np.prod(shape))
    rows, cols, vals, layout = [], [], [], []
    n_rows = 0
    for a in range(3):
        box = _face_box(shape, a)
        if box is None:
            layout.append(None)
            continue
        sl, sub = box
        idx = np.indices(sub).reshape(3, -1)
        nf = idx.shape[1]
        r = np.arange(n_rows, n_rows + nf, dtype=np.int64)
        b, c = _others(a)

        def gid(axis, shift=None):
            p = idx
            if shift is not None:
                p = idx.copy()
                p[shift] += 1
            return axis * N + np.ravel_multi_index(tuple(p), shape)

        for axis, shift, sign in ((c, b, 1.0), (c, None, -1.0),
                                  (b, c, -1.0), (b, None, 1.0)):
            rows.append(r)
            cols.append(gid(axis, shift))
            vals.append(np.full(nf, sign))
        layout.append((n_rows, nf, sl, sub))
        n_rows += nf
    D = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n_rows, 3 * N)).tocsr()
    return D, layout


def _faces_to_vector(arrays, layout) -> np.ndarray:
    """Three full-shape face arrays → the row vector of :func:`_curl_incidence`."""
    parts = [arrays[a][box[2]].ravel() for a, box in enumerate(layout)
             if box is not None]
    return np.concatenate(parts) if parts else np.empty(0)


def _vector_to_faces(vec, layout, shape) -> Tuple[np.ndarray, ...]:
    """Inverse of :func:`_faces_to_vector`; faces that do not exist read 0."""
    out = []
    for box in layout:
        arr = np.zeros(shape, dtype=np.float64)
        if box is not None:
            start, nf, sl, sub = box
            arr[sl] = vec[start:start + nf].reshape(sub)
        out.append(arr)
    return tuple(out)


def _edges_to_vector(arrays) -> np.ndarray:
    return np.concatenate([np.asarray(a, np.float64).ravel() for a in arrays])


def _vector_to_edges(vec, shape) -> Tuple[np.ndarray, ...]:
    N = int(np.prod(shape))
    return tuple(vec[a * N:(a + 1) * N].reshape(shape).copy() for a in range(3))


# ====================================================================== #
# Pinned edges and groups
# ====================================================================== #

def _boundary_layer(shape, face: str):
    axis = {'x': 0, 'y': 1, 'z': 2}[face[0]]
    sl = [slice(None)] * 3
    sl[axis] = 0 if face[1:] == 'min' else shape[axis] - 1
    return axis, tuple(sl)


def _pinned_edges(grid: FDTDGrid, bc, sheets) -> Tuple[np.ndarray, ...]:
    """Edges carrying ``u = 0``: PEC, flux-wall tangentials, terminal sheets.

    PEC edges come from :func:`wavesim.parts._edge_masks`, the masks the
    FDTD zeroes E on: the conformal rule on a cut-cell grid, the staircase
    dilation otherwise. A flux wall pins the edges lying *in* its node layer.
    A terminal sheet pins every edge whose two end nodes are both in the
    terminal's node mask.
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)
    exists = _edge_exists(shape)
    pinned = [np.array(m, dtype=bool) for m in _edge_masks(grid)]
    for face, kind in bc.items():
        if kind != 'flux_wall':
            continue
        axis, layer = _boundary_layer(shape, face)
        for a in range(3):
            if a != axis:
                pinned[a][layer] = True
    for sheet in sheets:
        for a in range(3):
            lo, hi = _face_slices(a)
            pinned[a][lo] |= sheet[lo] & sheet[hi]
    return tuple(p & e for p, e in zip(pinned, exists))


def _groups(pinned, shape) -> Tuple[np.ndarray, int]:
    """Label nodes by the connected component of the pinned-edge graph.

    Returns ``(labels, n_groups)`` with ``labels`` ``-1`` on nodes no pinned
    edge touches. Same construction as :func:`wavesim.parts.conductor_bodies`,
    over this solve's pinned set (PEC plus walls plus sheets) instead of PEC
    alone.
    """
    n_nodes = int(np.prod(shape))
    idx = np.arange(n_nodes, dtype=np.int64).reshape(shape)
    rows, cols = [], []
    on = np.zeros(shape, dtype=bool)
    for a in range(3):
        lo, hi = _face_slices(a)
        live = pinned[a][lo]
        rows.append(idx[lo][live])
        cols.append(idx[hi][live])
        on[lo] |= live
        on[hi] |= live
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    labels = np.full(shape, -1, dtype=np.int64)
    if rows.size == 0:
        return labels, 0
    graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)),
                       shape=(n_nodes, n_nodes))
    _, comp = connected_components(graph, directed=False)
    comp = comp.reshape(shape)
    keep = np.unique(comp[on])
    renumber = np.full(comp.max() + 1, -1, dtype=np.int64)
    renumber[keep] = np.arange(keep.size)
    labels[on] = renumber[comp[on]]
    return labels, int(keep.size)


class _GroupNames:
    """Human-readable names for groups, computed only when a message needs one.

    A group is named by what it is made of: the PEC parts in it (via
    :func:`wavesim.parts.body_parts`), "unnamed metal", the flux-wall faces,
    and port terminals.
    """

    def __init__(self, grid, labels, bc, ports):
        self.grid, self.labels, self.bc, self.ports = grid, labels, bc, ports
        self._cache = None

    def _build(self):
        grid, labels = self.grid, self.labels
        shape = labels.shape
        n = int(labels.max()) + 1 if labels.size else 0
        names = [[] for _ in range(max(n, 0))]
        body_labels, n_bodies = conductor_bodies(grid)
        occupants = body_parts(grid) if n_bodies else []
        for body in range(1, n_bodies + 1):
            gs = np.unique(labels[body_labels == body])
            for g in gs[gs >= 0]:
                parts = occupants[body - 1]
                names[g].extend(repr(p) for p in parts)
                if not parts:
                    names[g].append("unnamed metal")
        for face, kind in self.bc.items():
            if kind != 'flux_wall':
                continue
            _, layer = _boundary_layer(shape, face)
            gs = np.unique(labels[layer])
            for g in gs[gs >= 0]:
                names[g].append(f"the {face} flux wall")
        for pname, port in self.ports.items():
            for key, mask in (('a', port.terminal_a), ('b', port.terminal_b)):
                gs = np.unique(labels[mask])
                for g in gs[gs >= 0]:
                    names[g].append(f"terminal {key} of port {pname!r}")
        self._cache = [", ".join(dict.fromkeys(ns)) or "an unnamed group"
                       for ns in names]

    def __call__(self, g: int) -> str:
        if self._cache is None:
            self._build()
        return self._cache[g]


# ====================================================================== #
# Boundary conditions
# ====================================================================== #

def _resolve_boundary(spec) -> Dict[str, str]:
    """Normalise to ``{face: 'flux_wall' | 'symmetry'}``.

    Accepts one of the two names for all six faces, or a dict keyed by face
    with ``'*'`` as the default for faces left out (``'flux_wall'`` if absent),
    the same shape :meth:`wavesim.electrostatics.Electrostatics.solve` takes.
    """
    if isinstance(spec, str):
        spec = {'*': spec}
    if not isinstance(spec, dict):
        raise TypeError(f"boundary must be a string or a dict, got "
                        f"{type(spec).__name__}")
    unknown = set(spec) - set(FACES) - {'*'}
    if unknown:
        raise ValueError(f"unknown boundary face(s) {sorted(unknown)}; "
                         f"expected any of {list(FACES)} or '*'")
    default = spec.get('*', 'flux_wall')
    out = {}
    for f in FACES:
        v = spec.get(f, default)
        if v not in BOUNDARY_KINDS:
            raise ValueError(
                f"boundary[{f!r}] = {v!r}: expected 'flux_wall' (tangential "
                f"A = 0, B·n = 0) or 'symmetry' (tangential H = 0). There is "
                f"no open boundary; put a flux wall far away instead.")
        out[f] = v
    return out


# ====================================================================== #
# Sources: filaments
# ====================================================================== #

def _filament_counts(grid: FDTDGrid, path) -> Tuple[np.ndarray, ...]:
    """Signed dual-face crossing counts of a closed polyline, per edge.

    Walks the curve through the dual cells (cell ``n`` along an axis is the
    node ``n``'s slab, bounded by the planes at the cell centres
    ``xc[n-1]``, ``xc[n]``) and records each plane crossing on the dual face
    it passes through. Which face that is comes from the *tracked* current
    cell, not from re-binning the crossing point. A curve through a corner,
    where several planes are crossed at one parameter, still moves one face at
    a time and stays consistent. Since every step leaves one dual cell and
    enters its neighbour, a closed curve makes the integer divergence vanish
    at every node exactly.

    Membership is half-open, ``x ≥ plane`` counting as the high side, for
    the start cell and every crossing alike.
    """
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("a filament path must be an (n, 3) array of at least "
                         "three points, in metres")
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=0.0):
        pts = np.vstack([pts, pts[:1]])

    shape = (grid.Nx, grid.Ny, grid.Nz)
    coords = (grid.x, grid.y, grid.z)
    centres = (grid.xc, grid.yc, grid.zc)
    planes = tuple(np.asarray(centres[a][:shape[a] - 1], np.float64)
                   for a in range(3))
    for a, name in enumerate('xyz'):
        lo, hi = coords[a][0], coords[a][shape[a] - 1]
        if shape[a] > 1 and (pts[:, a].min() < lo or pts[:, a].max() > hi):
            raise ValueError(
                f"filament leaves the domain along {name}: it must stay "
                f"within [{lo:g}, {hi:g}] m, the span of the carried nodes")

    def cell(p):
        return [int(np.searchsorted(planes[a], p[a], side='right'))
                for a in range(3)]

    counts = [np.zeros(shape, dtype=np.int64) for _ in range(3)]
    cur = cell(pts[0])
    start = list(cur)
    for p, q in zip(pts[:-1], pts[1:]):
        n0, n1 = cell(p), cell(q)
        events = []
        for a in range(3):
            d = q[a] - p[a]
            if n1[a] > n0[a]:
                for i in range(n0[a], n1[a]):
                    events.append(((planes[a][i] - p[a]) / d, a, i, 1))
            elif n1[a] < n0[a]:
                for i in range(n0[a] - 1, n1[a] - 1, -1):
                    events.append(((planes[a][i] - p[a]) / d, a, i, -1))
        events.sort()
        for _, a, i, s in events:
            edge = list(cur)
            edge[a] = i
            counts[a][tuple(edge)] += s
            cur[a] = i + 1 if s > 0 else i
    assert cur == start, "filament walk did not close"   # a bug, never input
    return tuple(counts)


# ====================================================================== #
# Sources: current-flow solves (port channel and σ conductor)
# ====================================================================== #

def _cell_conductance(grid: FDTDGrid, s_cell: np.ndarray,
                      lengths) -> Tuple[np.ndarray, ...]:
    """Edge conductances ``Σ_q σ_q·(quarter area)_q / L`` from a cell field.

    The dual face of an edge is shared by the (up to) four cells around it,
    a quarter each. Summing ``σ`` times each quarter's area gives the
    conductance of the face, and dividing by the open edge length gives the
    coefficient of :func:`wavesim.electrostatics._assemble`. That is σ in
    place of ε, for a σ that is defined per cell, as a conductor is. A
    rectangular bar therefore carries exactly its own cross-section: an edge
    on its surface sees half the face, an edge on its corner a quarter.

    Returned in the per-face layout ``_assemble`` takes (length ``N-1`` along
    each coefficient's own axis).
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)
    dp = (grid.dxp, grid.dyp, grid.dzp)
    out = []
    for a in range(3):
        b, c = _others(a)
        half = [None, None, None]
        for ax in (b, c):
            w = np.asarray(dp[ax], np.float64)
            half[ax] = w if shape[ax] == 1 else 0.5 * w
        quarter = s_cell * _spread(half[b], b) * _spread(half[c], c)
        g = _shift_sum(quarter, (b, c))
        L = np.broadcast_to(lengths[a], shape)
        coef = np.zeros(shape)
        np.divide(g, L, out=coef, where=L > 0.0)
        lo, _ = _face_slices(a)
        out.append(coef[lo])
    return tuple(out)


def _flow_solve(coefs, fixed, V_fixed, float_groups, float_names,
                method, maxiter):
    """Solve ``∇·(σ∇V) = 0`` with ``fixed`` nodes held and groups floating.

    ``fixed`` marks nodes held at ``V_fixed``, including every node outside
    the flow region, which has no coefficient and is just parked.
    ``float_groups`` labels floating equipotentials, 0… or −1. Built entirely
    from the electrostatic assembly, reduction and solver.
    """
    A, B, free_idx = _assemble(coefs, fixed)
    n_free = A.shape[0]
    n_float = len(float_names)
    P, n_unk = _floating_reduction(float_groups, free_idx, n_free, n_float)
    if method == 'auto':
        method = AUTO_METHOD
    V = np.zeros(fixed.shape, dtype=np.float64)
    V[fixed] = V_fixed[fixed]
    if n_free:
        b = B @ V_fixed[fixed]
        if P is None:
            x, _ = _linear_solve(A, b, method, FLOW_RTOL, maxiter)
        else:
            A_red = (P.T @ A @ P).tocsr()
            _check_floating_coupled(A_red, n_float, float_names)
            x_red, _ = _linear_solve(A_red, P.T @ b, method, FLOW_RTOL,
                                     maxiter)
            x = P @ x_red
        V[~fixed] = x
    return V


def _flow_currents(coefs, V, shape) -> Tuple[np.ndarray, ...]:
    """Edge currents ``c·(V_lo − V_hi)`` in the full edge layout."""
    out = []
    for a, coef in enumerate(coefs):
        lo, hi = _face_slices(a)
        out.append(_pad_face_array(coef * (V[lo] - V[hi]), a, shape))
    return tuple(out)


# ====================================================================== #
# Source records
# ====================================================================== #

@dataclass
class _Loop:
    path: np.ndarray
    current: float


@dataclass
class _Port:
    terminal_a: np.ndarray
    terminal_b: np.ndarray
    channel: np.ndarray
    current: float
    conductor: Optional[np.ndarray]


# ====================================================================== #
# Result
# ====================================================================== #

@dataclass
class MagnetostaticSolution:
    """Result of one magnetostatic solve.

    ``u`` is the solved unknown, the line integral of A along each edge in
    webers, as three edge-staggered arrays. Everything else is derived from it
    on first access and cached, using the same operator weights the solve was
    assembled from. The energy, the flux linkages and the surface currents
    are therefore readings of one discretisation, not re-discretisations of
    the same integrals.

    Edge arrays (``A``, ``J``, ``edge_current``, ``surface_current``) have the
    shape and staggering of ``grid.Ex``/``Ey``/``Ez``; face arrays (``B``,
    ``H``, ``flux``) those of ``grid.Hx``/``Hy``/``Hz``. The last slot along
    an edge's own axis, and along a face's two in-plane axes, is outside the
    node box and always zero.
    """
    u: Tuple[np.ndarray, ...]
    grid: FDTDGrid
    boundary: Dict[str, str]
    method: str
    n_unknowns: int
    iterations: int
    gauge: float
    currents: Dict[str, float]
    source_edges: Dict[str, Tuple[np.ndarray, ...]] = field(repr=False)
    resistances: Dict[str, Optional[float]] = field(default_factory=dict)
    pinned: Tuple[np.ndarray, ...] = field(default=None, repr=False)
    group_labels: np.ndarray = field(default=None, repr=False)
    open_lengths: Tuple[np.ndarray, ...] = field(default=None, repr=False)
    weights: Tuple[np.ndarray, ...] = field(default=None, repr=False)
    D: csr_matrix = field(default=None, repr=False)
    layout: list = field(default=None, repr=False)

    @property
    def _shape(self):
        return (self.grid.Nx, self.grid.Ny, self.grid.Nz)

    # -- potential ------------------------------------------------------ #

    @cached_property
    def A(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Ax, Ay, Az)`` in T·m (Wb/m) on the E edges: ``u / L_open``.

        Zero on pinned edges (inside PEC, in flux walls and terminal sheets),
        where it is held zero. Gauge-dependent: only its curl, and its
        integral against a divergence-free current, mean anything physically.
        """
        out = []
        for a in range(3):
            L = np.broadcast_to(self.open_lengths[a], self._shape)
            Aa = np.zeros(self._shape)
            np.divide(self.u[a], L, out=Aa, where=L > 0.0)
            out.append(Aa)
        return tuple(out)

    # -- flux and fields ------------------------------------------------ #

    @cached_property
    def _phi_vec(self) -> np.ndarray:
        return self.D @ _edges_to_vector(self.u)

    @cached_property
    def flux(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Φx, Φy, Φz)`` in Wb through each open H face: the circulation of
        A around it, exactly. Their signed sum over any cell's six faces is
        zero to round-off (div B = 0 is topological here)."""
        return _vector_to_faces(self._phi_vec, self.layout, self._shape)

    @cached_property
    def B(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Bx, By, Bz)`` in T on the H faces, ``Φ / A_open``.

        On a cut face this is the field over its *open* part, the same
        reading the conformal H update gives its unknown. A covered face reads
        zero.
        """
        out = []
        for a, phi in enumerate(self.flux):
            area = _open_area(self.grid, a)
            Ba = np.zeros(self._shape)
            np.divide(phi, area, out=Ba, where=self.weights[a] > 0.0)
            out.append(Ba)
        return tuple(out)

    @cached_property
    def H(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Hx, Hy, Hz)`` in A/m, ``ν B`` with the face's own µ_r."""
        mus = (self.grid.mu_x, self.grid.mu_y, self.grid.mu_z)
        return tuple(B / (MU0 * np.asarray(mu, np.float64))
                     for B, mu in zip(self.B, mus))

    @cached_property
    def J(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Jx, Jy, Jz)`` in A/m², the total *imposed* source current density.

        Filaments plus port channels plus σ-conductor return flows, each edge
        current divided by its dual-face area. PEC surface currents are not
        imposed and are not in here: see :meth:`surface_current`.
        """
        dual = _dual_areas(self.grid)
        tot = self._source_total
        return tuple(np.divide(tot[a], np.broadcast_to(dual[a], self._shape))
                     for a in range(3))

    @cached_property
    def _source_total(self) -> Tuple[np.ndarray, ...]:
        tot = [np.zeros(self._shape) for _ in range(3)]
        for edges in self.source_edges.values():
            for a in range(3):
                tot[a] += edges[a]
        return tuple(tot)

    @cached_property
    def edge_current(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ampère's law at every edge: ``∮H·dl`` around its dual face, in A.

        ``Dᵀ W Φ``. On a free edge this is the imposed source current there,
        to solver accuracy (it is the equation that was solved). On a pinned
        edge it is the current the conductor or wall carries through that dual
        face, which the solve was never told: the induced surface current.
        """
        y = _faces_to_vector(self.weights, self.layout) * self._phi_vec
        return _vector_to_edges(self.D.T @ y, self._shape)

    def surface_current(self, name: str = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Induced current on the pinned edges, in amperes per edge.

        The jump of tangential H across the conductor surface, integrated over
        each pinned edge's dual face: :attr:`edge_current` restricted to the
        edges the solve pinned. Summing it over the edges a cutting plane
        crosses gives the total current through that section of the conductor.
        Dividing an edge's value by the width of its dual face gives a
        surface-current density in A/m, for display.

        ``name`` restricts it to the group (see the module docstring) holding
        the named PEC part. ``None`` returns every pinned edge, flux walls
        included.
        """
        cur = self.edge_current
        keep = self.pinned
        if name is not None:
            part_id(self.grid, name)                    # raises, listing names
            body_labels, _ = conductor_bodies(self.grid)
            names = body_parts(self.grid)
            bodies = [b + 1 for b, ns in enumerate(names) if name in ns]
            on = np.isin(body_labels, bodies)
            gs = np.unique(self.group_labels[on])
            in_group = np.isin(self.group_labels, gs[gs >= 0])
            keep = tuple(p & in_group for p in self.pinned)
        return tuple(np.where(k, c, 0.0) for k, c in zip(keep, cur))

    # -- node collocation (display) ------------------------------------- #

    @cached_property
    def A_nodes(self):
        """``A`` collocated onto the nodes, for display (see
        :func:`wavesim.electrostatics._to_nodes`)."""
        return _to_nodes(self.A)

    @cached_property
    def J_nodes(self):
        """``J`` collocated onto the nodes, for display."""
        return _to_nodes(self.J)

    @cached_property
    def B_nodes(self):
        """``B`` collocated onto the nodes: the mean of the (up to four)
        faces of each family meeting at the node. Display-grade."""
        return self._faces_to_nodes(self.B)

    @cached_property
    def H_nodes(self):
        """``H`` collocated onto the nodes, for display."""
        return self._faces_to_nodes(self.H)

    def _faces_to_nodes(self, F):
        shape = self._shape
        out = []
        for a in range(3):
            b, c = _others(a)
            exists = np.zeros(shape)
            box = _face_box(shape, a)
            if box is not None:
                exists[box[0]] = 1.0
            total = _shift_sum(F[a], (b, c))
            count = _shift_sum(exists, (b, c))
            out.append(np.divide(total, np.maximum(count, 1.0)))
        return tuple(out)

    def A_magnitude(self) -> np.ndarray:
        """``|A|`` in T·m on the nodes."""
        return np.sqrt(sum(c ** 2 for c in self.A_nodes))

    def B_magnitude(self) -> np.ndarray:
        """``|B|`` in T on the nodes."""
        return np.sqrt(sum(c ** 2 for c in self.B_nodes))

    def H_magnitude(self) -> np.ndarray:
        """``|H|`` in A/m on the nodes."""
        return np.sqrt(sum(c ** 2 for c in self.H_nodes))

    def J_magnitude(self) -> np.ndarray:
        """``|J|`` in A/m² on the nodes."""
        return np.sqrt(sum(c ** 2 for c in self.J_nodes))

    def B_at(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """``B`` (T) at the node nearest a physical position, from
        :attr:`B_nodes`."""
        n = self.grid.position_to_index(x, y, z)
        return tuple(float(c[n]) for c in self.B_nodes)

    # -- integrals ------------------------------------------------------ #

    @cached_property
    def energy(self) -> float:
        """Magnetic energy ``½∫B·H dV`` in joules, ``½ Σ_f W_f Φ_f²``.

        The quadratic form of the operator on its own solution. Equal to
        :attr:`energy_AJ`, ``½∫A·J``, to the linear solve's accuracy, which
        makes the pair a check.
        """
        w = _faces_to_vector(self.weights, self.layout)
        return 0.5 * float(np.sum(w * self._phi_vec ** 2))

    @cached_property
    def energy_AJ(self) -> float:
        """``½∫A·J dV`` in joules, ``½ Σ_e u_e I_e`` over the source currents."""
        return 0.5 * float(sum(np.sum(self.u[a] * self._source_total[a])
                               for a in range(3)))

    def flux_linkage(self, name: str) -> float:
        """Flux linked by the named source, ``(1/I)∫A·J_source dV``, in Wb.

        For a filament that is the flux through the curve, and for a port the
        current-weighted average flux through the loops its current takes.
        Alone in the model, ``flux_linkage(name) / I`` is the source's
        self-inductance. With others driven, it includes their mutual
        contributions. :func:`inductance_matrix` separates the two.
        """
        if name not in self.source_edges:
            known = ", ".join(sorted(self.source_edges)) or "none"
            raise KeyError(f"no source named {name!r}; sources: {known}")
        I = self.currents[name]
        if I == 0.0:
            raise ValueError(f"source {name!r} carries no current, so its "
                             f"flux linkage per ampere is undefined")
        src = self.source_edges[name]
        return float(sum(np.sum(self.u[a] * src[a]) for a in range(3))) / I

    def resistance(self, name: str) -> Optional[float]:
        """DC resistance in ohms of the named port's conductor path.

        From the current-flow pre-solve: the voltage across the σ path per
        ampere through it. ``None`` for a port whose terminals are joined by
        PEC (zero resistance, nothing to report) and for a filament.
        """
        if name not in self.source_edges:
            known = ", ".join(sorted(self.source_edges)) or "none"
            raise KeyError(f"no source named {name!r}; sources: {known}")
        return self.resistances.get(name)


# ====================================================================== #
# The assembled system (shared by solve and inductance_matrix)
# ====================================================================== #

class _GaugedOperator(LinearOperator):
    """``K = K_s + Mgᵀ diag(τ_g) Mg``, kept factored.

    ``K_s`` is sparse: the curl-curl plus the per-free-node gauge rows. ``Mg``
    holds the one divergence row per group, each touching every free edge that
    leaves the group. Multiplying those out would add a dense block per group,
    so they are applied as a low-rank update instead. CG only ever needs
    ``K @ x`` and the diagonal (for Jacobi), and both are cheap in this form.

    The direct path cannot factorise an operator, so :meth:`augmented` gives
    the equivalent sparse saddle-point matrix with one multiplier per group,

        ``[K_s  Mgᵀ; Mg  −diag(1/τ_g)] [u; λ] = [I; 0]``

    whose second row says ``λ = τ_g Mg u``, so the first is ``K u = I``. It is
    symmetric but indefinite, which ``splu`` handles and CG would not, and
    CG never sees it.
    """

    def __init__(self, Ks: csr_matrix, Mg: csr_matrix, tau: np.ndarray):
        super().__init__(dtype=np.float64, shape=Ks.shape)
        self.Ks, self.Mg, self.tau = Ks, Mg, np.asarray(tau, np.float64)

    def _matvec(self, x):
        x = np.ravel(x)
        out = self.Ks @ x
        if self.tau.size:
            out = out + self.Mg.T @ (self.tau * (self.Mg @ x))
        return out

    def _rmatvec(self, x):
        return self._matvec(x)

    def diagonal(self) -> np.ndarray:
        d = self.Ks.diagonal()
        if self.tau.size:
            d = d + np.asarray(self.Mg.multiply(self.Mg).T @ self.tau).ravel()
        return d

    def augmented(self) -> csr_matrix:
        if not self.tau.size:
            return self.Ks
        return bmat([[self.Ks, self.Mg.T],
                     [self.Mg, diags(-1.0 / self.tau)]], format='csc')

@dataclass
class _System:
    grid: FDTDGrid
    bc: Dict[str, str]
    shape: tuple
    pinned: tuple
    labels: np.ndarray
    n_groups: int
    free_cols: np.ndarray
    K: '_GaugedOperator'
    D: csr_matrix
    layout: list
    weights: tuple
    lengths: tuple
    unit_sources: Dict[str, Tuple[np.ndarray, ...]]   # edge currents at 1 A
    resistances: Dict[str, Optional[float]]
    method: str
    gauge: float

    def rhs(self, currents: Dict[str, float]) -> np.ndarray:
        total = np.zeros(3 * int(np.prod(self.shape)))
        for name, I in currents.items():
            total += I * _edges_to_vector(self.unit_sources[name])
        return total[self.free_cols]

    def solve(self, b, rtol, maxiter, progress):
        if self.method == 'direct':
            aug = self.K.augmented()
            b_aug = np.concatenate([b, np.zeros(aug.shape[0] - b.size)])
            x, iters = _linear_solve(aug, b_aug, 'direct', rtol, maxiter,
                                     progress)
            x = x[:b.size]
        else:
            x, iters = _linear_solve(self.K, b, 'cg', rtol, maxiter,
                                     progress, preconditioner=_preconditioner)
        full = np.zeros(3 * int(np.prod(self.shape)))
        full[self.free_cols] = x
        return _vector_to_edges(full, self.shape), iters


# ====================================================================== #
# Solver
# ====================================================================== #

class Magnetostatics:
    """Drive currents and solve for the static magnetic field.

        >>> ms = Magnetostatics(grid)
        >>> ms.add_loop('coil', path, current=1.0)
        >>> ms.add_port('feed', term_a, term_b, channel, current=1.0)
        >>> sol = ms.solve(boundary='flux_wall')
        >>> sol.energy, sol.flux_linkage('feed'), sol.resistance('feed')

    Sources are named, like electrostatic conductors, so results can be
    addressed by name, and :func:`inductance_matrix` can energise them one at
    a time. PEC in the model expels flux. σ in the model matters only where a
    port's current flows through it.
    """

    def __init__(self, grid: FDTDGrid):
        self.grid = grid
        self.loops: Dict[str, _Loop] = {}
        self.ports: Dict[str, _Port] = {}

    @property
    def names(self) -> Tuple[str, ...]:
        """Source names, in the order they were added."""
        return tuple(list(self.loops) + list(self.ports))

    def _check_name(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError(f"source name must be a non-empty string, "
                             f"got {name!r}")
        if name in self.loops or name in self.ports:
            raise ValueError(f"a source named {name!r} already exists")

    # -- setup ---------------------------------------------------------- #

    def add_loop(self, name: str, path, current: float) -> 'Magnetostatics':
        """A closed filament along ``path`` (``(n, 3)`` points, metres).

        The polyline is closed automatically if the last point is not the
        first. ``current`` (A) flows in the direction the points are listed.
        Refused at solve time if it runs through PEC or along a flux wall,
        where the solve could not carry it. Returns self.
        """
        self._check_name(name)
        pts = np.asarray(path, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
            raise ValueError("path must be an (n, 3) array of at least three "
                             "points, in metres")
        self.loops[name] = _Loop(path=pts.copy(), current=float(current))
        return self

    def add_port(self, name: str, terminal_a, terminal_b, channel,
                 current: float, conductor=None) -> 'Magnetostatics':
        """A current source across a gap in a conductor path.

        Parameters
        ----------
        terminal_a, terminal_b : bool array, shape ``(Nx, Ny, Nz)``, **nodes**
            The two contact faces, as node masks: node ``(i,j,k)`` is the low
            corner of cell ``(i,j,k)``. Each must be one connected sheet of
            nodes, or lie on one conductor.
        channel : bool array, shape ``(Nx, Ny, Nz)``, **cells**
            The gap volume the source current crosses, from a to b. Its node
            closure must touch both terminals and no other conductor.
        current : float
            Amperes, positive flowing from terminal a to terminal b *across
            the gap* (and so from b back to a through the conductor).
        conductor : bool array, shape ``(Nx, Ny, Nz)``, **cells**, optional
            Cells allowed to carry the return current as a σ conductor.
            Defaults to cells with σ above :data:`CONDUCTOR_SIGMA_THRESHOLD`.
            Either way only the part connected to this port's terminals is
            used. Irrelevant when the terminals are joined by PEC.
        """
        self._check_name(name)
        shape = (self.grid.Nx, self.grid.Ny, self.grid.Nz)

        def mask(m, what):
            m = np.asarray(m)
            if m.shape != shape:
                raise ValueError(f"{what}: expected shape {shape}, got "
                                 f"{m.shape}")
            m = m.astype(bool)
            if not m.any():
                raise ValueError(f"{what} is empty")
            return m

        a = mask(terminal_a, "terminal_a")
        b = mask(terminal_b, "terminal_b")
        ch = mask(channel, "channel")
        if (a & b).any():
            raise ValueError(f"port {name!r}: terminals a and b share "
                             f"{int((a & b).sum())} node(s); they must be two "
                             f"separate faces across a gap")
        if self.grid.pec_mask is not None and (ch & self.grid.pec_mask).any():
            raise ValueError(f"port {name!r}: the channel overlaps PEC in "
                             f"{int((ch & self.grid.pec_mask).sum())} cell(s); "
                             f"it must be the gap between the terminals")
        cond = None if conductor is None else mask(conductor, "conductor")
        self.ports[name] = _Port(terminal_a=a, terminal_b=b, channel=ch,
                                 current=float(current), conductor=cond)
        return self

    # -- solve ---------------------------------------------------------- #

    def solve(self, boundary='flux_wall', method: str = 'auto',
              rtol: float = 1e-10, maxiter: int = None, gauge: float = 1.0,
              progress=None) -> MagnetostaticSolution:
        """Solve for the field of every source at its own current.

        Parameters
        ----------
        boundary : str or dict
            ``'flux_wall'`` (default) or ``'symmetry'`` for all six faces, or
            a dict keyed by ``'xmin'``…``'zmax'`` with ``'*'`` as the default.
        method : {'auto', 'direct', 'cg'}
            ``'auto'`` (the default) is Jacobi-preconditioned conjugate
            gradients at every size, measured 8x to 260x ahead of the direct
            factorisation with no crossover (see :data:`AUTO_METHOD`).
            ``'direct'`` is exact and tolerance-free, which makes it the
            reference for tests.
        rtol, maxiter : float, int
            CG tolerance and iteration cap; ignored by the direct path.
        gauge : float
            Weight of the grad-div gauge term, relative to ``ν/V``. B, H,
            energies, flux linkages and inductances do not depend on it; A
            does. Change it only to study conditioning.
        progress : callable, optional
            ``progress(fraction)`` over the main solve, as electrostatics.

        Returns
        -------
        MagnetostaticSolution
        """
        if not self.names:
            raise ValueError("no sources: add a loop or a port first")
        system = self._system(boundary, method, maxiter, gauge)
        currents = {n: s.current for n, s in
                    list(self.loops.items()) + list(self.ports.items())}
        u, iters = system.solve(system.rhs(currents), rtol, maxiter, progress)
        if progress is not None:
            progress(1.0)
        return self._solution(system, u, iters, currents)

    def _solution(self, system, u, iters, currents) -> MagnetostaticSolution:
        return MagnetostaticSolution(
            u=u, grid=self.grid, boundary=system.bc, method=system.method,
            n_unknowns=int(system.free_cols.size), iterations=iters,
            gauge=system.gauge, currents=dict(currents),
            source_edges={n: tuple(I * e for e in system.unit_sources[n])
                          for n, I in currents.items()},
            resistances=dict(system.resistances), pinned=system.pinned,
            group_labels=system.labels, open_lengths=system.lengths,
            weights=system.weights, D=system.D, layout=system.layout)

    # -- assembly ------------------------------------------------------- #

    def _system(self, boundary, method, maxiter, gauge) -> _System:
        if method not in ('auto', 'direct', 'cg'):
            raise ValueError(f"method must be 'auto', 'direct' or 'cg', "
                             f"got {method!r}")
        if not gauge > 0.0:
            raise ValueError(f"gauge must be positive, got {gauge!r}")
        grid = self.grid
        shape = (grid.Nx, grid.Ny, grid.Nz)
        N = int(np.prod(shape))
        bc = _resolve_boundary(boundary)

        sheets = [m for p in self.ports.values()
                  for m in (p.terminal_a, p.terminal_b)]
        pinned = _pinned_edges(grid, bc, sheets)
        labels, n_groups = _groups(pinned, shape)
        names = _GroupNames(grid, labels, bc, self.ports)
        lengths, _ = _open_lengths(grid)

        # Sources at 1 A, built before the operator so a bad one is refused
        # before any expensive work.
        unit, resistances = {}, {}
        flow_method = method
        for lname, loop in self.loops.items():
            unit[lname] = self._loop_edges(lname, loop, pinned)
        for pname, port in self.ports.items():
            unit[pname], resistances[pname] = self._port_edges(
                pname, port, pinned, labels, names, lengths,
                flow_method, maxiter)
        self._check_divergence(unit, labels, n_groups, names)

        exists = _edge_exists(shape)
        free = [e & ~p for e, p in zip(exists, pinned)]
        free_cols = np.flatnonzero(_edges_to_vector(free) > 0.5)
        if free_cols.size == 0:
            raise ValueError(
                "every edge is pinned, so there is nothing to solve for. A "
                "degenerate (N = 1) axis with a flux wall on it pins the whole "
                "layer; make that axis's faces 'symmetry'.")

        weights = _face_weights(grid)
        D, layout = _curl_incidence(shape)
        w_vec = _faces_to_vector(weights, layout)
        Df = D[:, free_cols]
        Kc = (Df.T @ diags(w_vec) @ Df).tocsr()
        Kf, Mg, tau_g = self._gauge_operator(
            free, free_cols, labels, n_groups, lengths, shape, N)
        K = _GaugedOperator((Kc + gauge * Kf).tocsr(), Mg, gauge * tau_g)

        chosen = AUTO_METHOD if method == 'auto' else method
        return _System(grid=grid, bc=bc, shape=shape, pinned=pinned,
                       labels=labels, n_groups=n_groups, free_cols=free_cols,
                       K=K, D=D, layout=layout, weights=weights,
                       lengths=lengths, unit_sources=unit,
                       resistances=resistances, method=chosen, gauge=gauge)

    def _gauge_operator(self, free, free_cols, labels, n_groups, lengths,
                        shape, N):
        """``Mᵀ diag(τ) M``: one divergence row per free node and per group.

        ``M[row, e] = ±S_e/L_e``, the flux of ``A = u/L`` through the dual
        face, out of the edge's low node and into its high one. A group's
        nodes share one row, as discussed under "Gauge" in the module
        docstring. ``τ = ν/V``, the node's dual volume and the mean reluctivity
        of the three faces it is the low corner of. For a group, the volume
        is summed and ν averaged. Any positive τ gives the same B.
        """
        grid = self.grid
        dual = _dual_widths(grid)
        V = (_spread(dual[0], 0) * _spread(dual[1], 1)
             * _spread(dual[2], 2)) * np.ones(shape)
        mu_node = (np.asarray(grid.mu_x) + np.asarray(grid.mu_y)
                   + np.asarray(grid.mu_z)) / 3.0
        nu = 1.0 / (MU0 * mu_node)

        flat = labels.ravel()
        grouped = flat >= 0
        row = np.empty(N, dtype=np.int64)
        row[grouped] = flat[grouped]
        row[~grouped] = n_groups + np.arange(int((~grouped).sum()))
        n_rows = n_groups + int((~grouped).sum())

        tau = np.empty(n_rows)
        tau[n_groups:] = (nu.ravel() / V.ravel())[~grouped]
        if n_groups:
            vol = np.bincount(flat[grouped], weights=V.ravel()[grouped],
                              minlength=n_groups)
            nus = np.bincount(flat[grouped], weights=nu.ravel()[grouped],
                              minlength=n_groups)
            cnt = np.bincount(flat[grouped], minlength=n_groups)
            tau[:n_groups] = (nus / cnt) / vol

        dual_area = _dual_areas(grid)
        col_of = np.full(3 * N, -1, dtype=np.int64)
        col_of[free_cols] = np.arange(free_cols.size)
        node = np.arange(N, dtype=np.int64).reshape(shape)
        rows, cols, vals = [], [], []
        for a in range(3):
            lo, hi = _face_slices(a)
            live = free[a][lo]
            L = np.broadcast_to(lengths[a], shape)[lo][live]
            S = np.broadcast_to(dual_area[a], shape)[lo][live]
            c = np.divide(S, L, out=np.zeros_like(S), where=L > 0.0)
            n_lo = node[lo][live]
            n_hi = node[hi][live]
            col = col_of[a * N + n_lo]
            rows += [row[n_lo], row[n_hi]]
            cols += [col, col]
            vals += [c, -c]
        M = coo_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_rows, free_cols.size)).tocsr()
        # The free-node rows touch at most six edges each, so their product is
        # a sparse 7-point-like block. A group's row touches every free edge
        # leaving the group, thousands for a flux wall, and MᵀτM would turn
        # it into a dense block of that size squared (measured: 666 nonzeros
        # per row on a 48-cell coax, the matvec 20x slower). So the group rows
        # stay factored, as a low-rank term — see _GaugedOperator.
        Mf, Mg = M[n_groups:], M[:n_groups]
        return (Mf.T @ diags(tau[n_groups:]) @ Mf).tocsr(), Mg, tau[:n_groups]

    # -- sources -------------------------------------------------------- #

    def _loop_edges(self, name, loop, pinned):
        counts = _filament_counts(self.grid, loop.path)
        for a in range(3):
            bad = (counts[a] != 0) & pinned[a]
            if bad.any():
                n = tuple(int(i[0]) for i in np.nonzero(bad))
                pos = (float(self.grid.x[n[0]]), float(self.grid.y[n[1]]),
                       float(self.grid.z[n[2]]))
                raise ValueError(
                    f"filament {name!r} passes through PEC or lies along a "
                    f"flux wall (first at the {'xyz'[a]} edge from node {n}, "
                    f"near ({pos[0]:.4g}, {pos[1]:.4g}, {pos[2]:.4g}) m). "
                    f"Tangential A is held at zero there, so no current can "
                    f"be carried. Route it through free space, or use a port "
                    f"to drive current through a conductor.")
        return tuple(c.astype(np.float64) for c in counts)

    def _port_edges(self, name, port, pinned, labels, names, lengths,
                    flow_method, maxiter):
        """Unit-current edge currents for one port, and its DC resistance."""
        grid = self.grid
        shape = labels.shape
        a_nodes, b_nodes = port.terminal_a, port.terminal_b

        def one_group(nodes, key):
            gs = np.unique(labels[nodes])
            if gs.size != 1 or gs[0] < 0:
                raise ValueError(
                    f"port {name!r}: terminal {key} is not one connected "
                    f"contact. Its {int(nodes.sum())} node(s) are not all "
                    f"joined to each other by grid edges or by one conductor. "
                    f"A terminal must be a single sheet of edge-connected "
                    f"nodes (a lone node, or nodes touching only diagonally, "
                    f"do not qualify), or lie on a single conductor.")
            return int(gs[0])

        ga, gb = one_group(a_nodes, 'a'), one_group(b_nodes, 'b')

        # -- channel flow: uniform conductivity, a at 1 V, b at 0 V -------- #
        ch_cells = port.channel.astype(np.float64)
        coefs = _cell_conductance(grid, ch_cells, lengths)
        coefs = tuple(np.where(pinned[a][_face_slices(a)[0]], 0.0, c)
                      for a, c in enumerate(coefs))
        region = np.zeros(shape, dtype=bool)
        for a, c in enumerate(coefs):
            lo, hi = _face_slices(a)
            region[lo] |= c > 0.0
            region[hi] |= c > 0.0
        stray = region & (labels >= 0) & ~np.isin(labels, [ga, gb])
        if stray.any():
            g = int(np.unique(labels[stray])[0])
            raise ValueError(
                f"port {name!r}: the channel touches {names(g)}, which is "
                f"neither terminal. The channel must span only the gap "
                f"between the two terminal faces.")
        if not (region & a_nodes).any() or not (region & b_nodes).any():
            raise ValueError(
                f"port {name!r}: the channel does not reach both terminals. "
                f"Its cells must span the gap from terminal a to terminal b, "
                f"with each terminal's nodes on the channel's surface.")
        fixed = ~region | a_nodes | b_nodes
        V_fix = np.where(a_nodes, 1.0, 0.0)
        V = _flow_solve(coefs, fixed, V_fix, np.full(shape, -1), [],
                        flow_method, maxiter)
        G = float(_node_flux(coefs, V)[a_nodes].sum())
        if not G > 0.0:
            raise ValueError(f"port {name!r}: no current flows through the "
                             f"channel from terminal a to terminal b")
        channel = tuple(I / G for I in _flow_currents(coefs, V, shape))

        if ga == gb:                     # terminals joined by PEC
            return channel, None

        # -- σ conductor return path --------------------------------------- #
        cond = port.conductor
        s_cell = self._cell_sigma()
        if cond is None:
            cond = s_cell > CONDUCTOR_SIGMA_THRESHOLD
        cond = cond & ~port.channel
        if grid.pec_mask is not None:
            cond = cond & ~grid.pec_mask
        coefs = _cell_conductance(grid, np.where(cond, s_cell, 0.0), lengths)
        coefs = tuple(np.where(pinned[a][_face_slices(a)[0]], 0.0, c)
                      for a, c in enumerate(coefs))

        comp = self._circuit_component(coefs, pinned, labels, ga, shape)
        if not comp[b_nodes].any():
            raise ValueError(
                f"port {name!r} has no return path: terminal a sits on "
                f"{names(ga)} and terminal b on {names(gb)}, and no conductor "
                f"joins them. The current driven across the gap would pile up "
                f"on each side, which has no static solution (KCL). Close the "
                f"loop with PEC, with a σ body above "
                f"{CONDUCTOR_SIGMA_THRESHOLD:g} S/m, or by passing conductor=.")
        kept = []
        for a, c in enumerate(coefs):
            lo, hi = _face_slices(a)
            kept.append(np.where(comp[lo] & comp[hi], c, 0.0))
        coefs = tuple(kept)

        fixed = ~comp | (labels == ga) | (labels == gb)
        V_fix = np.where(labels == gb, 1.0, 0.0)
        floaters = [g for g in np.unique(labels[comp & ~fixed]) if g >= 0]
        float_groups = np.full(shape, -1, dtype=np.int64)
        for i, g in enumerate(floaters):
            float_groups[labels == g] = i
        V = _flow_solve(coefs, fixed, V_fix, float_groups,
                        [(names(g),) for g in floaters],
                        flow_method, maxiter)
        G = float(_node_flux(coefs, V)[labels == gb].sum())
        if not G > 0.0:
            raise ValueError(f"port {name!r}: no current flows through the "
                             f"conductor between its terminals")
        ret = _flow_currents(coefs, V, shape)
        return (tuple(c + r / G for c, r in zip(channel, ret)), 1.0 / G)

    def _cell_sigma(self) -> np.ndarray:
        g = self.grid
        shape = (g.Nx, g.Ny, g.Nz)
        if not g.is_lossy:
            return np.zeros(shape)
        return (np.asarray(g.sigma_x, np.float64) + g.sigma_y + g.sigma_z) / 3.0

    @staticmethod
    def _circuit_component(coefs, pinned, labels, ga, shape):
        """Nodes connected to group ``ga`` by σ edges or pinned edges."""
        n_nodes = int(np.prod(shape))
        idx = np.arange(n_nodes, dtype=np.int64).reshape(shape)
        rows, cols = [], []
        for a in range(3):
            lo, hi = _face_slices(a)
            live = (coefs[a] > 0.0) | pinned[a][lo]
            rows.append(idx[lo][live])
            cols.append(idx[hi][live])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)),
                           shape=(n_nodes, n_nodes))
        _, comp = connected_components(graph, directed=False)
        comp = comp.reshape(shape)
        return comp == comp[labels == ga][0]

    @staticmethod
    def _check_divergence(unit, labels, n_groups, names):
        """Refuse a source that is not discretely divergence-free.

        Zero at every free node, to round-off. Zero net into every group,
        which is KCL for each PEC body, flux-wall set and terminal sheet.
        """
        # Every source in ``unit`` carries exactly 1 A, so the tolerance is
        # absolute in amperes per ampere of source. Scaling it by the largest
        # *edge* current instead made it shrink with the mesh (a port's 1 A
        # spreads over more edges), and a 96-cell model tripped on CG noise.
        tol = DIVERGENCE_RTOL
        for name, I in unit.items():
            div = _node_divergence(I)
            free_err = float(np.abs(div[labels < 0]).max(initial=0.0))
            if free_err > tol:
                raise AssertionError(
                    f"source {name!r}: discrete divergence {free_err:.3g} A at "
                    f"a free node exceeds round-off ({tol:.3g} A). This is a "
                    f"solver bug, not an input error.")
            if n_groups:
                net = np.bincount(labels[labels >= 0],
                                  weights=div[labels >= 0],
                                  minlength=n_groups)
                for g in np.flatnonzero(np.abs(net) > tol):
                    raise ValueError(
                        f"source {name!r} drives a net {-net[g]:.4g} A into "
                        f"{names(int(g))} (per ampere of source), with no way "
                        f"out. A perfect conductor cannot accumulate charge "
                        f"in statics (KCL), so the problem has no solution. A "
                        f"source must close on itself or return through the "
                        f"conductor it leaves.")


# ====================================================================== #
# Inductance
# ====================================================================== #

@dataclass
class InductanceMatrix:
    """Inductance between named sources, in henries.

    ``matrix[i][j] = ∫A_j·J_i dV`` with source ``j`` driven at 1 A and every
    other source open: an unexcited port carries no current, an unexcited
    filament none either. The diagonal holds the self-inductances, and the
    off-diagonal entries the mutuals. Symmetric up to the linear solve's
    accuracy (exactly the same operator inverse is applied on both sides),
    and positive definite, since ``½ IᵀLI`` is the energy of currents ``I``.
    """
    names: Tuple[str, ...]
    matrix: np.ndarray
    resistances: Dict[str, Optional[float]] = field(default_factory=dict)

    def between(self, a: str, b: str) -> float:
        """Mutual inductance in henries between two named sources."""
        return float(self.matrix[self.names.index(a), self.names.index(b)])

    def self_(self, a: str) -> float:
        """Self-inductance in henries of a named source."""
        i = self.names.index(a)
        return float(self.matrix[i, i])

    def coupling(self, a: str, b: str) -> float:
        """Coupling coefficient ``M/√(L_a L_b)``, in [−1, 1]."""
        return self.between(a, b) / np.sqrt(self.self_(a) * self.self_(b))

    def energy(self, currents) -> float:
        """``½ IᵀLI`` in joules for ``currents`` keyed by name or in order."""
        if isinstance(currents, dict):
            I = np.array([currents.get(n, 0.0) for n in self.names])
        else:
            I = np.asarray(currents, dtype=np.float64)
        return 0.5 * float(I @ self.matrix @ I)


def inductance_matrix(ms: Magnetostatics, names=None, *,
                      boundary='flux_wall', progress=None,
                      **solve_kw) -> InductanceMatrix:
    """Extract the inductance matrix by driving one source at a time at 1 A.

    The operator is assembled once and solved once per source. Every source's
    geometry (including every port's terminal sheets) is in the operator for
    every solve, so the columns describe one fixed structure. The currents set
    on the sources are ignored; each is driven at 1 A in turn.

    Parameters
    ----------
    ms : Magnetostatics
        Carrying the sources. A bare grid is not accepted, because the sources
        are what is being measured, and they live on the solver, not the grid.
    names : sequence of str, optional
        Sources to include, defaulting to all of them in the order added.
    boundary : str or dict
        As :meth:`Magnetostatics.solve`.
    progress : callable, optional
        ``progress(fraction)`` over the whole extraction, each solve an equal
        share.
    **solve_kw
        ``method``, ``rtol``, ``maxiter``, ``gauge``, as for
        :meth:`Magnetostatics.solve`.
    """
    if isinstance(ms, FDTDGrid):
        raise TypeError(
            "inductance_matrix needs the sources: build Magnetostatics(grid), "
            "add loops/ports to it, and pass that")
    names = tuple(names if names is not None else ms.names)
    if not names:
        raise ValueError("no sources to extract inductance between")
    for n in names:
        if n not in ms.loops and n not in ms.ports:
            known = ", ".join(ms.names) or "none"
            raise KeyError(f"no source named {n!r}; sources: {known}")

    method = solve_kw.pop('method', 'auto')
    rtol = solve_kw.pop('rtol', 1e-10)
    maxiter = solve_kw.pop('maxiter', None)
    gauge = solve_kw.pop('gauge', 1.0)
    if solve_kw:
        raise TypeError(f"unexpected keyword(s) {sorted(solve_kw)}")

    system = ms._system(boundary, method, maxiter, gauge)
    n = len(names)
    rhs = [system.rhs({nm: 1.0}) for nm in names]
    L = np.zeros((n, n))
    for j, nm in enumerate(names):
        sub = None
        if progress is not None:
            def sub(frac, j=j):
                progress((j + frac) / n)
        u, _ = system.solve(rhs[j], rtol, maxiter, sub)
        uv = _edges_to_vector(u)[system.free_cols]
        for i in range(n):
            L[i, j] = float(rhs[i] @ uv)
    if progress is not None:
        progress(1.0)
    return InductanceMatrix(
        names=names, matrix=L,
        resistances={nm: system.resistances.get(nm) for nm in names})
