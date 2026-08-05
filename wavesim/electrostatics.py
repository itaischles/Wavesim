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
from typing import Dict, Tuple
import warnings

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu, cg, LinearOperator

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


def _face_coefs(grid: FDTDGrid, node_pec: np.ndarray) -> Tuple[np.ndarray, ...]:
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
    """
    dual = (_node_dual(grid.dxp), _node_dual(grid.dyp), _node_dual(grid.dzp))
    primary = (grid.dxp, grid.dyp, grid.dzp)
    eps = (grid.eps_x, grid.eps_y, grid.eps_z)

    def spread(vec, axis):
        """``vec`` laid out along ``axis``, broadcastable against a 3D array."""
        shape = [1, 1, 1]
        shape[axis] = vec.size
        return vec.reshape(shape)

    out = []
    for axis in range(3):
        n = grid.eps_x.shape[axis]
        others = [a for a in range(3) if a != axis]
        area = spread(dual[others[0]], others[0]) * spread(dual[others[1]],
                                                           others[1])
        distance = spread(primary[axis][:n - 1], axis)
        out.append(_face_eps(eps[axis], node_pec, axis) * area / distance)
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
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        for src, nbr in ((tuple(lo), tuple(hi)), (tuple(hi), tuple(lo))):
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
    Derived fields (E, D, energy, charge, capacitance) are added in the next
    commit; everything needed to compute them is retained here.
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

    def potential_at(self, x: float, y: float, z: float) -> float:
        """φ at the node nearest a physical position, in volts."""
        return float(self.phi[self.grid.position_to_index(x, y, z)])


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

        fixed, phi_fixed, n_grounded = self._pin_conductors(node_pec)
        self._pin_boundary(bc, fixed, phi_fixed)

        if not fixed.any():
            raise ValueError(
                "the problem is singular: no conductor carries a potential and "
                "every domain face is Neumann, so phi is only determined up to "
                "an additive constant. Ground a part, or make at least one face "
                "Dirichlet (e.g. boundary='ground').")

        coefs = _face_coefs(grid, node_pec)
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
            iterations=iterations, node_pec=node_pec)

    # -- pinning -------------------------------------------------------- #

    def _pin_conductors(self, node_pec):
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

        labels, n_bodies = conductor_bodies(grid)
        if n_bodies == 0:
            return fixed, phi_fixed, 0

        occupants = body_parts(grid)
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
