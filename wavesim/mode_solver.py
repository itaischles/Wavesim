"""
mode_solver.py — 2D TEM (transverse electromagnetic) mode solver.

Given a grid face (or a rectangular subset of one), this finds the PEC conductor
cross-sections lying on that plane and solves the transverse-static field of each
TEM mode the structure supports. The resulting :class:`TEMMode` carries the 2D
transverse E (and H) profiles, which can be scaled and launched as an input port
via :meth:`TEMMode.to_source` (a :class:`~wavesim.sources.PlaneSource`).

Physics
-------
A TEM mode's transverse field is electrostatic in the cross-section: ``E_t = -∇φ``
where ``φ`` solves the ε-weighted 2D Laplace equation ``∇·(ε ∇φ) = 0`` over the
dielectric, with each conductor held at a constant potential. This is a
*boundary-value problem*, not an eigenvalue problem — a cross-section with *M*
disjoint conductors supports *M − 1* independent TEM modes. We pick one conductor
(or the grounded outer shield) as the 0 V reference, raise one other conductor to
1 V, ground the rest, and solve once per signal conductor.

The magnetic field follows from the TEM relation ``H_t = (n̂ × E_t) / η`` with the
local wave impedance ``η = η₀·√(μ_r/ε_r)`` and ``n̂`` the +propagation direction,
giving the H profile needed to launch a directional (one-way) wave.

Per mode we also report the (per-unit-length) capacitance, inductance, phase
velocity, effective permittivity and characteristic impedance, obtained from the
field-energy integral and a companion air-filled solve.

Conventions match the rest of wavesim: all positions in metres; the transverse
plane is sliced exactly as :mod:`wavesim.monitors` does (``normal='z'`` → XY, etc.).
SciPy provides the sparse solve (:func:`scipy.sparse.linalg.splu`) and the
connected-conductor labelling (:func:`scipy.ndimage.label`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import warnings

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from wavesim.constants import EPS0, ETA0, C0
from wavesim.grid import FDTDGrid


# ====================================================================== #
# Per-normal geometry: how a slice maps to transverse axes / components.
#
# For each propagation normal we record, in slice (a, b) order:
#   axes   — the two transverse axis letters,
#   ds     — attribute names of the two cell sizes,
#   eps    — material attrs seen by the two transverse E components,
#   mu     — material attr seen by the H_a component (for the wave impedance),
#   E, H   — the field-component names driven on the plane,
#   h_sign — (sa, sb) so that  H_a = sa·E_b/η,  H_b = sb·E_a/η  (= n̂ × E_t / η).
# ====================================================================== #
_NORMAL_CFG = {
    'z': dict(axes=('x', 'y'), ds=('dx', 'dy'),
              eps=('eps_x', 'eps_y'), mu='mu_x',
              E=('Ex', 'Ey'), H=('Hx', 'Hy'), h_sign=(-1.0, +1.0)),
    'y': dict(axes=('x', 'z'), ds=('dx', 'dz'),
              eps=('eps_x', 'eps_z'), mu='mu_x',
              E=('Ex', 'Ez'), H=('Hx', 'Hz'), h_sign=(+1.0, -1.0)),
    'x': dict(axes=('y', 'z'), ds=('dy', 'dz'),
              eps=('eps_y', 'eps_z'), mu='mu_y',
              E=('Ey', 'Ez'), H=('Hy', 'Hz'), h_sign=(-1.0, +1.0)),
}


def _plane_to_grid(normal: str, k: int, a: np.ndarray, b: np.ndarray):
    """Map transverse-plane indices ``(a, b)`` on the slice to full 3D grid
    indices, inverting :func:`_slice` (same (a, b) axis order)."""
    kk = np.full(a.shape, k, dtype=a.dtype)
    if normal == 'z':      # plane axes (x, y); slice along z
        return a, b, kk
    if normal == 'y':      # plane axes (x, z); slice along y
        return a, kk, b
    return kk, a, b        # normal == 'x': plane axes (y, z); slice along x


@dataclass
class TEMMode:
    """One solved TEM mode on a transverse plane.

    The field profiles are stored at full transverse-plane resolution (the shape
    of the corresponding grid slice), zero outside any sub-rectangle that was
    solved, so they drop straight into a :class:`~wavesim.sources.PlaneSource`.
    """
    # --- where the mode lives ---------------------------------------------- #
    normal: str                       # propagation axis ('x'/'y'/'z')
    position: float                   # metres along ``normal``
    slice_index: int                  # cell index of the plane along ``normal``
    transverse_axes: Tuple[str, str]  # the two axes ⟂ to ``normal``, slice order
    da: float                         # cell size along first transverse axis (m)
    db: float                         # cell size along second transverse axis (m)

    # --- field shapes (full transverse-plane 2D arrays) -------------------- #
    phi: np.ndarray                   # electrostatic potential (V), V=1 drive
    E: Dict[str, np.ndarray]          # transverse E profiles, keyed by component
    H: Dict[str, np.ndarray]          # transverse H profiles, keyed by component
    pec: np.ndarray                   # PEC mask on the plane (for plotting)

    # --- identity & per-unit-length parameters ----------------------------- #
    conductor_id: int                 # label of the energized (1 V) conductor
    capacitance: float = None         # C (F/m)
    inductance: float = None          # L (H/m)
    impedance: float = None           # Z₀ (Ω)
    v_phase: float = None             # phase velocity (m/s)
    eps_eff: float = None             # effective permittivity (C / C_air)

    def to_source(self, waveform: Callable[[float], float],
                  amplitude: float = 1.0, fields: str = 'EH'):
        """Build a :class:`~wavesim.sources.PlaneSource` that launches this mode.

        Parameters
        ----------
        waveform : Callable[[float], float]
            Temporal profile (e.g. a :class:`~wavesim.sources.GaussianPulse`).
        amplitude : float
            Scalar the mode profiles are multiplied by (the mode is normalised to
            a 1 V drive; scale it to whatever excitation you want).
        fields : str
            ``'EH'`` injects both transverse E and H (directional launch);
            ``'E'`` injects only E (simpler, bidirectional).
        """
        from wavesim.sources import PlaneSource  # local import avoids a cycle
        profiles: Dict[str, np.ndarray] = {}
        if 'E' in fields:
            for comp, arr in self.E.items():
                profiles[comp] = amplitude * arr
        if 'H' in fields:
            for comp, arr in self.H.items():
                profiles[comp] = amplitude * arr
        return PlaneSource(waveform, axis=self.normal, position=self.position,
                           profiles=profiles)

    def build_port_kernel(self, grid: FDTDGrid, *,
                          directional: bool = True) -> dict:
        """Compile this mode into a distributed lumped-port kernel.

        The modal generalisation of
        :meth:`wavesim.sources.LineSource._build_port`: it replaces the straight
        p0→p1 path with the frozen transverse mode profile, so a
        :class:`~wavesim.sources.TEMPort` / :class:`~wavesim.sources.SpicePort`
        can still expose a single scalar ``(V, I)`` pair to a circuit / SPICE
        solve. With ``Ê`` the 1 V-normalised modal E profile and
        ``S = Σ ε_r Ê²`` summed over the transverse-plane cells (both components):

        * **voltage read-back** ``V* = Σ (ε_r Ê / S)·E`` — an ε-weighted overlap
          projection: reads 1 V for the pure mode and rejects non-modal content;
        * **current injection** ``E += κ·Ê·I`` — launches the mode shape;
        * **modal self-coupling** ``κ = dt / (ε₀·dV·S)`` — ohms, the change in
          ``V*`` per unit injected current per step (exactly ``LineSource``'s κ).

        The returned dict mirrors ``LineSource._build_port`` (``edges``/``kappa``)
        so the existing time-centred (Piket-May) injection runs unchanged. When
        ``directional`` the same scalar also drives the paired H sheet
        ``H += κ·Ĥ·I`` (the EH launch of :meth:`to_source`), biasing energy into
        +normal. That term is added *after* the implicit ``V*→I`` solve, so it
        does not enter κ or the stability condition ``κ/2 < Z₀``.
        """
        cfg = _NORMAL_CFG[self.normal]
        dV = grid.dx * grid.dy * grid.dz
        eps_of = {'Ex': grid.eps_x, 'Ey': grid.eps_y, 'Ez': grid.eps_z}
        k = self.slice_index

        # Gather nonzero plane cells per E component; accumulate S = Σ ε_r Ê².
        gathered = {}
        S = 0.0
        for comp in cfg['E']:
            Ehat2d = self.E[comp]
            a, b = np.nonzero(Ehat2d)
            if a.size == 0:
                continue
            ii, jj, kk = _plane_to_grid(self.normal, k, a, b)
            Ehat = Ehat2d[a, b]
            epsr = eps_of[comp][ii, jj, kk]
            gathered[comp] = (ii, jj, kk, Ehat, epsr)
            S += float(np.sum(epsr * Ehat ** 2))
        if not gathered or S <= 0.0:
            raise ValueError(
                "TEM mode has no transverse E energy on the plane; cannot build "
                "a port kernel.")

        kappa = grid.dt / (EPS0 * dV * S)
        edges = {}
        for comp, (ii, jj, kk, Ehat, epsr) in gathered.items():
            w = epsr * Ehat / S            # projection weight (metres)
            coef = kappa * Ehat            # E-injection coefficient
            edges[comp] = (ii, jj, kk, w, coef)

        hedges = {}
        if directional:
            for comp in cfg['H']:
                Hhat2d = self.H[comp]
                a, b = np.nonzero(Hhat2d)
                if a.size == 0:
                    continue
                ii, jj, kk = _plane_to_grid(self.normal, k, a, b)
                hedges[comp] = (ii, jj, kk, kappa * Hhat2d[a, b])

        return {'edges': edges, 'kappa': kappa, 'hedges': hedges,
                'z0': self.impedance}


# ====================================================================== #
# Public entry point
# ====================================================================== #

def solve_tem_modes(grid: FDTDGrid, *,
                    normal: str = 'z', position: float = 0.0,
                    bounds: Tuple[float, float, float, float] = None,
                    ground='auto', boundary: str = 'ground',
                    compute_params: bool = True) -> List[TEMMode]:
    """Solve the TEM modes of the PEC cross-section on a grid plane.

    Parameters
    ----------
    grid : FDTDGrid
    normal : {'x', 'y', 'z'}
        Propagation axis; the solve is done on the plane perpendicular to it.
    position : float
        Position (metres) of the plane along ``normal``, snapped to a cell.
    bounds : (a0, a1, b0, b1), optional
        Rectangular subset of the plane (metres) in the two transverse axes
        (slice order, see ``transverse_axes``). ``None`` ⇒ the whole face.
    ground : 'auto' or int
        Reference-conductor selection. ``'auto'`` makes the ground node the
        outer shield: with ``boundary='ground'`` that is the domain edge together
        with every PEC region touching it; otherwise the largest PEC region. An
        explicit integer forces that conductor label into the ground node.
    boundary : {'ground', 'neumann'}
        Outer boundary condition on the solve region's edge. ``'ground'``
        (default) pins φ=0 there (a grounded shield, correct for enclosed/shielded
        structures); ``'neumann'`` imposes zero normal flux (open/symmetry).
    compute_params : bool
        Also compute C, L, Z₀, v, ε_eff per mode (needs one extra air-filled
        solve). Set False to skip.

    Returns
    -------
    list[TEMMode]
        One mode per signal conductor. Empty (with a warning) if the cross-section
        has fewer than two conductors — a single conductor supports no TEM mode.
    """
    if normal not in _NORMAL_CFG:
        raise ValueError(f"normal must be 'x', 'y' or 'z', got {normal!r}")
    cfg = _NORMAL_CFG[normal]

    k = grid.axis_index(normal, position)
    da = getattr(grid, cfg['ds'][0])
    db = getattr(grid, cfg['ds'][1])

    # --- slice the plane: eps (per transverse component), mu, PEC ----------- #
    eps_a_full = _slice(getattr(grid, cfg['eps'][0]), normal, k)
    eps_b_full = _slice(getattr(grid, cfg['eps'][1]), normal, k)
    mu_a_full = _slice(getattr(grid, cfg['mu']), normal, k)
    if grid.pec_mask is None:
        pec_full = np.zeros(eps_a_full.shape, dtype=bool)
    else:
        pec_full = _slice(grid.pec_mask, normal, k).astype(bool)

    full_shape = eps_a_full.shape  # the PlaneSource-compatible 2D shape

    # --- optional rectangular subset --------------------------------------- #
    if bounds is not None:
        a0, a1, b0, b1 = bounds
        ia0 = grid.axis_index(cfg['axes'][0], a0)
        ia1 = grid.axis_index(cfg['axes'][0], a1)
        ib0 = grid.axis_index(cfg['axes'][1], b0)
        ib1 = grid.axis_index(cfg['axes'][1], b1)
    else:
        ia0, ia1, ib0, ib1 = 0, full_shape[0], 0, full_shape[1]
    sub = np.s_[ia0:ia1, ib0:ib1]

    eps_a = np.ascontiguousarray(eps_a_full[sub], dtype=np.float64)
    eps_b = np.ascontiguousarray(eps_b_full[sub], dtype=np.float64)
    mu_a = np.ascontiguousarray(mu_a_full[sub], dtype=np.float64)
    pec = np.ascontiguousarray(pec_full[sub])

    # --- conductors & reference node --------------------------------------- #
    labels, n_cond = ndimage.label(pec)          # 4-connectivity (default)
    signals, ground_labels = _classify_conductors(labels, n_cond, boundary, ground)

    if not signals:
        warnings.warn(
            f"TEM mode solver found {n_cond} conductor(s) on the plane and no "
            f"signal conductor relative to the reference — a TEM mode needs at "
            f"least two conductors. Returning no modes.")
        return []

    # Cells whose potential is pinned: all PEC, plus the grounded edge ring.
    fixed = pec.copy()
    if boundary == 'ground':
        fixed[0, :] = True; fixed[-1, :] = True
        fixed[:, 0] = True; fixed[:, -1] = True

    # --- factorise the weighted Laplacian once, reuse for every mode -------- #
    lu, B, free_idx, fixed_cells = _factor_laplacian(eps_a, eps_b, da, db, fixed)
    # Air-filled companion (ε≡1) for the per-unit-length parameters.
    if compute_params:
        lu_air, B_air, _, _ = _factor_laplacian(
            np.ones_like(eps_a), np.ones_like(eps_b), da, db, fixed)

    modes: List[TEMMode] = []
    for Ls in signals:
        phi = _solve_one(lu, B, free_idx, fixed_cells, labels, fixed, Ls)
        mode = _build_mode(phi, eps_a, eps_b, mu_a, pec, da, db, cfg,
                           normal, position, k, full_shape, (ia0, ib0), Ls)
        if compute_params:
            phi_air = _solve_one(lu_air, B_air, free_idx, fixed_cells,
                                 labels, fixed, Ls)
            _attach_params(mode, phi, phi_air, eps_a, eps_b, da, db)
        modes.append(mode)

    return modes


# ====================================================================== #
# Plane slicing (mirrors wavesim.monitors._slice)
# ====================================================================== #

def _slice(arr: np.ndarray, normal: str, idx: int) -> np.ndarray:
    """The 2D plane of ``arr`` perpendicular to ``normal`` at cell ``idx``."""
    if normal == 'z':
        return arr[:, :, idx]
    if normal == 'y':
        return arr[:, idx, :]
    return arr[idx, :, :]


# ====================================================================== #
# Conductor classification
# ====================================================================== #

def _classify_conductors(labels, n_cond, boundary, ground):
    """Split labelled PEC regions into (signal conductors, ground-node labels).

    The ground node is the 0 V reference. With ``boundary='ground'`` it is the
    grounded shield: the domain edge plus any conductor touching it. Otherwise a
    reference conductor is needed; an explicit ``ground`` label, else the largest
    region, becomes it. Everything not in the ground node is a signal conductor
    and gets its own mode.
    """
    all_labels = set(range(1, n_cond + 1))
    edge_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])))
    edge_labels.discard(0)

    ground_labels = set()
    if boundary == 'ground':
        ground_labels |= edge_labels
    if isinstance(ground, (int, np.integer)) and not isinstance(ground, bool):
        if int(ground) in all_labels:
            ground_labels.add(int(ground))
        else:
            raise ValueError(f"ground={ground} is not a conductor label "
                             f"(1..{n_cond}).")
    if boundary != 'ground' and not ground_labels and all_labels:
        # No grounded shield and no explicit reference: ground the largest
        # conductor so the remaining conductors are measured against it.
        counts = np.bincount(labels.ravel(), minlength=n_cond + 1)
        ground_labels = {int(np.argmax(counts[1:])) + 1}

    signals = sorted(all_labels - ground_labels)
    return signals, ground_labels


# ====================================================================== #
# Sparse weighted-Laplacian assembly and solve
# ====================================================================== #

def _factor_laplacian(eps_a, eps_b, da, db, fixed):
    """Assemble and LU-factorise the ε-weighted 2D Laplacian over free cells.

    Discretises ``∂_a(ε_a ∂_a φ) + ∂_b(ε_b ∂_b φ) = 0`` with a 5-point
    variable-coefficient stencil; the face permittivity is the arithmetic mean of
    the two adjoining cells. Out-of-array neighbours are simply omitted, which is
    the natural zero-flux (Neumann) edge; a grounded edge is instead handled by
    the caller marking the ring as ``fixed``.

    Returns ``(lu, B, free_idx, fixed_cells)`` where ``lu`` solves ``A x = b`` over
    the free cells, ``B`` (free × fixed) maps pinned potentials into the RHS via
    ``b = -B @ φ_fixed``, ``free_idx`` is the (Na,Nb) int map to free-cell indices
    (−1 where fixed), and ``fixed_cells`` lists the (p, q) of each pinned cell.
    """
    Na, Nb = eps_a.shape
    free_mask = ~fixed
    n_free = int(free_mask.sum())
    free_idx = -np.ones((Na, Nb), dtype=np.int64)
    free_idx[free_mask] = np.arange(n_free)

    fixed_pq = np.argwhere(fixed)
    n_fixed = len(fixed_pq)
    fixed_idx = -np.ones((Na, Nb), dtype=np.int64)
    fixed_idx[fixed] = np.arange(n_fixed)

    inv_da2 = 1.0 / (da * da)
    inv_db2 = 1.0 / (db * db)

    # The 5-point stencil is built one *face direction* at a time (4 vectorised
    # passes), not cell-by-cell. For each direction the in-bounds region is a
    # whole-array slice ``src`` paired with its neighbour slice ``nbr``; the face
    # permittivity is the arithmetic mean of the two, exactly as before. Omitting
    # the out-of-bounds border rows/columns reproduces the zero-flux (Neumann)
    # edge. Off-diagonal couplings split by whether the neighbour is free (→ A) or
    # pinned (→ B); the diagonal accumulates −Σ(face coef) over the same faces.
    diag = np.zeros((Na, Nb), dtype=np.float64)
    rows_A, cols_A, data_A = [], [], []
    rows_B, cols_B, data_B = [], [], []

    directions = (
        (np.s_[0:Na - 1, :], np.s_[1:Na, :],   eps_a, inv_da2),  # +a face
        (np.s_[1:Na, :],     np.s_[0:Na - 1, :], eps_a, inv_da2),  # -a face
        (np.s_[:, 0:Nb - 1], np.s_[:, 1:Nb],   eps_b, inv_db2),  # +b face
        (np.s_[:, 1:Nb],     np.s_[:, 0:Nb - 1], eps_b, inv_db2),  # -b face
    )
    for src, nbr, eps, invd2 in directions:
        if eps[src].size == 0:
            continue
        coef = 0.5 * (eps[src] + eps[nbr]) * invd2
        i_src = free_idx[src]
        free_src = i_src >= 0
        # diagonal: only free source cells own a row (fixed-cell diag is unused).
        diag[src] -= np.where(free_src, coef, 0.0)
        jn_free = free_idx[nbr]
        nbr_free = jn_free >= 0
        m_AA = free_src & nbr_free                 # free ↔ free  → A
        rows_A.append(i_src[m_AA]); cols_A.append(jn_free[m_AA]); data_A.append(coef[m_AA])
        m_AB = free_src & ~nbr_free                # free ↔ fixed → B
        rows_B.append(i_src[m_AB]); cols_B.append(fixed_idx[nbr][m_AB]); data_B.append(coef[m_AB])

    # Diagonal entries (A[i, i] = diag) appended last; COO sums duplicates, so the
    # off-diagonal and diagonal contributions accumulate just like the old ``+=``.
    rows_A.append(np.arange(n_free)); cols_A.append(np.arange(n_free)); data_A.append(diag[free_mask])

    A = coo_matrix((np.concatenate(data_A),
                    (np.concatenate(rows_A), np.concatenate(cols_A))),
                   shape=(n_free, n_free)).tocsc()
    B = coo_matrix((np.concatenate(data_B),
                    (np.concatenate(rows_B), np.concatenate(cols_B))),
                   shape=(n_free, n_fixed)).tocsr()

    lu = splu(A)
    return lu, B, free_idx, fixed_pq


def _solve_one(lu, B, free_idx, fixed_cells, labels, fixed, energized_label):
    """Solve for φ with ``energized_label`` at 1 V and all other fixed cells at 0."""
    # Pinned potentials, ordered like ``fixed_cells``.
    phi_fixed = np.zeros(len(fixed_cells))
    for n, (p, q) in enumerate(fixed_cells):
        if labels[p, q] == energized_label:
            phi_fixed[n] = 1.0
    b = -(B @ phi_fixed)
    x = lu.solve(b)

    phi = np.zeros(free_idx.shape)
    free_mask = free_idx >= 0
    phi[free_mask] = x[free_idx[free_mask]]
    for n, (p, q) in enumerate(fixed_cells):
        phi[p, q] = phi_fixed[n]
    return phi


# ====================================================================== #
# Field construction and per-unit-length parameters
# ====================================================================== #

def _transverse_E(phi, da, db, pec):
    """``E_t = -∇φ`` on the cross-section (centred differences), zeroed in PEC."""
    dphi_da = np.gradient(phi, da, axis=0)
    dphi_db = np.gradient(phi, db, axis=1)
    Ea = -dphi_da
    Eb = -dphi_db
    Ea[pec] = 0.0
    Eb[pec] = 0.0
    return Ea, Eb


def _build_mode(phi, eps_a, eps_b, mu_a, pec, da, db, cfg,
                normal, position, k, full_shape, offset, label):
    """Assemble a :class:`TEMMode` (fields embedded into the full transverse plane)."""
    Ea, Eb = _transverse_E(phi, da, db, pec)

    # H_t = (n̂ × E_t) / η,  η = η₀·√(μ_r/ε_r)  (local wave impedance).
    eta = ETA0 * np.sqrt(mu_a / np.where(eps_a > 0, eps_a, 1.0))
    sa, sb = cfg['h_sign']
    Ha = sa * Eb / eta
    Hb = sb * Ea / eta
    Ha[pec] = 0.0
    Hb[pec] = 0.0

    ia0, ib0 = offset
    sub = np.s_[ia0:ia0 + phi.shape[0], ib0:ib0 + phi.shape[1]]

    def _embed(arr2d):
        full = np.zeros(full_shape, dtype=np.float64)
        full[sub] = arr2d
        return full

    phi_full = _embed(phi)
    pec_full = np.zeros(full_shape, dtype=bool)
    pec_full[sub] = pec

    E = {cfg['E'][0]: _embed(Ea), cfg['E'][1]: _embed(Eb)}
    H = {cfg['H'][0]: _embed(Ha), cfg['H'][1]: _embed(Hb)}

    return TEMMode(
        normal=normal, position=position, slice_index=k,
        transverse_axes=cfg['axes'], da=da, db=db,
        phi=phi_full, E=E, H=H, pec=pec_full, conductor_id=int(label))


def _attach_params(mode: TEMMode, phi, phi_air, eps_a, eps_b, da, db):
    """Fill C, L, Z₀, v, ε_eff from the field energy of the filled & air solves."""
    # Energy integral uses the gradient over the whole cross-section (V = 1 V):
    #   C = ε₀ ∫ (ε_a E_a² + ε_b E_b²) dA.
    dA = da * db
    Ea = -np.gradient(phi, da, axis=0)
    Eb = -np.gradient(phi, db, axis=1)
    C = EPS0 * np.sum(eps_a * Ea**2 + eps_b * Eb**2) * dA

    Ea_air = -np.gradient(phi_air, da, axis=0)
    Eb_air = -np.gradient(phi_air, db, axis=1)
    C_air = EPS0 * np.sum(Ea_air**2 + Eb_air**2) * dA

    if C > 0 and C_air > 0:
        mode.capacitance = float(C)
        mode.inductance = float(1.0 / (C0**2 * C_air))
        mode.impedance = float(1.0 / (C0 * np.sqrt(C * C_air)))
        mode.v_phase = float(C0 * np.sqrt(C_air / C))
        mode.eps_eff = float(C / C_air)
