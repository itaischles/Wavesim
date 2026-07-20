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
#   ds     — attribute names of the two scalar cell sizes (min width per axis;
#            kept for legacy display only — the solve uses the per-cell arrays),
#   dp     — attribute names of the two primary-width arrays (per-cell widths),
#   cen    — attribute names of the two cell-center coordinate arrays,
#   node   — attribute names of the two node-coordinate arrays (length N+1),
#   eps    — material attrs seen by the two transverse E components,
#   mu     — material attr seen by the H_a component (for the wave impedance),
#   E, H   — the field-component names driven on the plane,
#   h_sign — (sa, sb) so that  H_a = sa·E_b/η,  H_b = sb·E_a/η  (= n̂ × E_t / η).
#
# The rectilinear (non-uniform) rehaul (``docs/nonuniform_grid_plan.md`` Session
# 6) drives every transverse derivative and area weight off the per-cell ``dp``
# widths / ``cen`` coordinates instead of the scalar ``ds``, so the solve is
# correct on a graded transverse mesh; a uniform mesh reduces to the old result.
# ====================================================================== #
_NORMAL_CFG = {
    'z': dict(axes=('x', 'y'), ds=('dx', 'dy'),
              dp=('dxp', 'dyp'), cen=('xc', 'yc'), node=('x', 'y'),
              eps=('eps_x', 'eps_y'), mu='mu_x',
              E=('Ex', 'Ey'), H=('Hx', 'Hy'), h_sign=(-1.0, +1.0)),
    'y': dict(axes=('x', 'z'), ds=('dx', 'dz'),
              dp=('dxp', 'dzp'), cen=('xc', 'zc'), node=('x', 'z'),
              eps=('eps_x', 'eps_z'), mu='mu_x',
              E=('Ex', 'Ez'), H=('Hx', 'Hz'), h_sign=(+1.0, -1.0)),
    'x': dict(axes=('y', 'z'), ds=('dy', 'dz'),
              dp=('dyp', 'dzp'), cen=('yc', 'zc'), node=('y', 'z'),
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


def _normal_width(grid: FDTDGrid, normal: str) -> np.ndarray:
    """Per-cell primary widths along ``normal`` (the propagation axis)."""
    return {'x': grid.dxp, 'y': grid.dyp, 'z': grid.dzp}[normal]


def numerical_velocity(v: float, dn: float, dt: float,
                       frequency: float = None) -> float:
    """Phase velocity a wave *actually* travels at on the grid.

    A Yee mesh is dispersive: the discrete wave runs slightly slower than the
    medium's ``v``, by an amount depending on the Courant number and on how many
    cells resolve a wavelength. Inverting the 1D dispersion relation

        sin(ω·dt/2) / (v·dt) = sin(k·dn/2) / dn

    for ``k`` gives ``v_num = ω/k``. Returns ``v`` unchanged when ``frequency``
    is ``None`` (broadband drive — no single frequency to tune to) or when the
    frequency lies outside the grid's numerical passband, where the relation has
    no real solution and the wave is evanescent rather than propagating.
    """
    if not frequency or frequency <= 0.0 or v <= 0.0:
        return v
    omega = 2.0 * np.pi * frequency
    s = (dn / (v * dt)) * np.sin(omega * dt / 2.0)
    if not -1.0 < s < 1.0:
        return v
    k = (2.0 / dn) * np.arcsin(s)
    return omega / k if k > 0.0 else v


def _launch_time_shift(dt: float, dn: float, v: float,
                       frequency: float = None) -> float:
    """Time shift (s) for the H sheet of a directional launch, ``≤ 0``.

    ``dt/2`` undoes the leapfrog stagger (E and H are stored half a step apart);
    ``dn/(2·v)`` undoes the half-cell the H sheet sits behind the E plane. With
    the sheet placed *behind*, the two subtract, so the result is negative for
    any stable Courant number and only past drive values are needed.
    """
    if v is None or v <= 0.0:
        v = C0
    return dt / 2.0 - dn / (2.0 * numerical_velocity(v, dn, dt, frequency))


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
    da: float                         # representative cell size, axis a (m; mean
                                      #   width — the mesh may be non-uniform)
    db: float                         # representative cell size, axis b (m)

    # --- field shapes (full transverse-plane 2D arrays) -------------------- #
    phi: np.ndarray                   # electrostatic potential (V), V=1 drive
    E: Dict[str, np.ndarray]          # transverse E profiles, keyed by component
    H: Dict[str, np.ndarray]          # transverse H profiles, keyed by component
    pec: np.ndarray                   # PEC mask on the plane (for plotting)

    # --- identity & per-unit-length parameters ----------------------------- #
    conductor_id: int                 # label of the energized (1 V) conductor

    # --- transverse node coordinates (metres) for the full plane ----------- #
    # Length (Na+1, Nb+1); the true cell boundaries along each transverse axis.
    # Carried so a non-uniform mesh plots with correct physical extents (viz)
    # rather than assuming the constant da/db above. ``None`` ⇒ derive a uniform
    # ruler from da/db (legacy).
    a_nodes: np.ndarray = None
    b_nodes: np.ndarray = None

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

        Notes
        -----
        The ``'EH'`` launch here is the *naive* pairing: both sheets go on the
        same slice and share one waveform, which biases energy into +normal but
        only rejects backwards by roughly -18 dB. The corrected pairing — H half
        a cell behind, driven with a compensating time shift — currently lives in
        :meth:`build_port_kernel` (used by
        :class:`~wavesim.sources.TEMPort`), because applying it needs the grid's
        ``dt`` and cell size, which a :class:`~wavesim.sources.PlaneSource` only
        sees lazily at first injection.
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
                          directional: bool = True,
                          frequency: float = None) -> dict:
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
        * **modal self-coupling** ``κ = dt / (ε₀·Σ_c dV_c·ε_r·Ê_c²)`` — ohms, the
          change in ``V*`` per unit injected current per step. ``dV_c`` is the
          **local Yee cell volume** at cell ``c`` (the product of the primary
          widths ``dxp·dyp·dzp`` there, matching the all-primary divisors of
          :func:`wavesim.update.update_E`, exactly as ``LineSource._build_port``
          does). On a uniform grid ``dV_c`` is the constant ``dx·dy·dz`` and this
          reduces to the old ``κ = dt/(ε₀·dV·S)``; on a rectilinear mesh each
          cell carries its own volume so κ tracks the local spacing.

        The returned dict mirrors ``LineSource._build_port`` (``edges``/``kappa``)
        so the existing time-centred (Piket-May) injection runs unchanged. When
        ``directional`` the same scalar also drives the paired H sheet
        ``H += κ·Ĥ·I``, biasing energy into +normal. That term is added *after*
        the implicit ``V*→I`` solve, so it does not enter κ or the stability
        condition ``κ/2 < Z₀``.

        Placing that H sheet correctly is what makes the launch one-way. The two
        sheets cancel backwards only if they represent the *same* incident wave,
        and on a Yee grid they do not sample it at the same point: ``H`` sits
        half a cell along the normal from ``E`` and half a timestep away in the
        leapfrog. Both offsets are corrected here:

        * the sheet goes at ``k-1``, i.e. half a cell **behind** the E plane
          relative to +normal propagation (``H`` is stored at ``+½`` cell, so
          index ``k-1`` lands at ``-½``). Behind rather than ahead is what makes
          the required time shift *negative*, so a circuit-driven port can build
          it from past currents instead of future ones;
        * ``h_tau = dt/2 - dn/(2·v)`` (seconds, ≤ 0 for any stable Courant
          number) is returned for the caller to apply to the H drive.

        Measured backward rejection on a 1D vacuum test: ≈ -18 dB uncorrected,
        ≈ -150 dB with both offsets applied. The E/H *amplitude* ratio needs no
        correction — the continuum ``1/η`` is right to within 0.3% across
        Courant numbers 0.3-0.99 and 10-40 cells per wavelength.

        Parameters
        ----------
        directional : bool
            Build the paired H sheet for a one-way launch.
        frequency : float, optional
            Drive frequency (Hz) used to evaluate the *numerical* phase velocity
            in ``h_tau``. Omit for a broadband drive: the continuum velocity is
            then used, which is a weak approximation here (``h_tau`` varies only
            ~3% over a 4× frequency range) and still rejects to roughly -55 dB.
        """
        cfg = _NORMAL_CFG[self.normal]
        eps_of = {'Ex': grid.eps_x, 'Ey': grid.eps_y, 'Ez': grid.eps_z}
        k = self.slice_index

        # Gather nonzero plane cells per E component. ``S = Σ ε_r Ê²`` normalises
        # the read-back projection (dimensionless, so V*=1 for the pure mode);
        # ``Sv = Σ dV_c ε_r Ê²`` volume-weights the energy for κ (per-cell local
        # Yee volume, all-primary as in update_E). On a uniform grid Sv = dV·S.
        gathered = {}
        S = 0.0
        Sv = 0.0
        for comp in cfg['E']:
            Ehat2d = self.E[comp]
            a, b = np.nonzero(Ehat2d)
            if a.size == 0:
                continue
            ii, jj, kk = _plane_to_grid(self.normal, k, a, b)
            Ehat = Ehat2d[a, b]
            epsr = eps_of[comp][ii, jj, kk]
            dV_c = grid.dxp[ii] * grid.dyp[jj] * grid.dzp[kk]
            gathered[comp] = (ii, jj, kk, Ehat, epsr)
            S += float(np.sum(epsr * Ehat ** 2))
            Sv += float(np.sum(dV_c * epsr * Ehat ** 2))
        if not gathered or S <= 0.0 or Sv <= 0.0:
            raise ValueError(
                "TEM mode has no transverse E energy on the plane; cannot build "
                "a port kernel.")

        kappa = grid.dt / (EPS0 * Sv)
        edges = {}
        for comp, (ii, jj, kk, Ehat, epsr) in gathered.items():
            w = epsr * Ehat / S            # projection weight (metres)
            coef = kappa * Ehat            # E-injection coefficient
            edges[comp] = (ii, jj, kk, w, coef)

        hedges = {}
        h_tau = 0.0
        if directional:
            if k < 1:
                raise ValueError(
                    f"A directional launch needs its H sheet one cell behind the "
                    f"E sheet, but the mode plane sits at {self.normal}-index "
                    f"{k}. Move the port at least one cell into the domain.")
            dn = float(_normal_width(grid, self.normal)[k - 1])
            h_tau = _launch_time_shift(grid.dt, dn, self.v_phase, frequency)
            for comp in cfg['H']:
                Hhat2d = self.H[comp]
                a, b = np.nonzero(Hhat2d)
                if a.size == 0:
                    continue
                ii, jj, kk = _plane_to_grid(self.normal, k - 1, a, b)
                hedges[comp] = (ii, jj, kk, kappa * Hhat2d[a, b])

        return {'edges': edges, 'kappa': kappa, 'hedges': hedges,
                'h_tau': h_tau, 'z0': self.impedance}


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

    # --- transverse spacing (per-cell — rectilinear/non-uniform aware) ------ #
    # ``da_w``/``db_w`` are the per-cell primary widths on the solved sub-rect;
    # ``a_c``/``b_c`` the matching cell-center coordinates for gradients. The
    # full-plane node coordinates are carried onto each mode for correct viz
    # extents. On a uniform grid these are constant and reproduce the old result.
    da_w = np.ascontiguousarray(getattr(grid, cfg['dp'][0])[ia0:ia1], np.float64)
    db_w = np.ascontiguousarray(getattr(grid, cfg['dp'][1])[ib0:ib1], np.float64)
    a_c = np.ascontiguousarray(getattr(grid, cfg['cen'][0])[ia0:ia1], np.float64)
    b_c = np.ascontiguousarray(getattr(grid, cfg['cen'][1])[ib0:ib1], np.float64)
    a_nodes = np.asarray(getattr(grid, cfg['node'][0]), np.float64)
    b_nodes = np.asarray(getattr(grid, cfg['node'][1]), np.float64)

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
    lu, B, free_idx, fixed_cells = _factor_laplacian(eps_a, eps_b, da_w, db_w,
                                                     fixed, pec)
    # Air-filled companion (ε≡1) for the per-unit-length parameters. The PEC
    # one-sided rule is a no-op here (ε is uniformly 1), which is precisely why
    # applying it to the filled solve restores φ == φ_air on a homogeneous fill.
    if compute_params:
        lu_air, B_air, _, _ = _factor_laplacian(
            np.ones_like(eps_a), np.ones_like(eps_b), da_w, db_w, fixed, pec)

    modes: List[TEMMode] = []
    for Ls in signals:
        phi = _solve_one(lu, B, free_idx, fixed_cells, labels, fixed, Ls)
        mode = _build_mode(phi, eps_a, eps_b, mu_a, pec, da_w, db_w, a_c, b_c,
                           cfg, normal, position, k, full_shape, (ia0, ib0), Ls,
                           a_nodes, b_nodes)
        if compute_params:
            phi_air = _solve_one(lu_air, B_air, free_idx, fixed_cells,
                                 labels, fixed, Ls)
            _attach_params(mode, phi, phi_air, eps_a, eps_b, da_w, db_w, a_c, b_c,
                           pec)
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

def _factor_laplacian(eps_a, eps_b, da_w, db_w, fixed, pec=None):
    """Assemble and LU-factorise the ε-weighted 2D Laplacian over free cells.

    Discretises ``∂_a(ε_a ∂_a φ) + ∂_b(ε_b ∂_b φ) = 0`` with a 5-point
    variable-coefficient **finite-volume** stencil on a rectilinear (possibly
    non-uniform) transverse mesh. ``da_w``/``db_w`` are the per-cell primary
    widths along the two transverse axes. The flux across the face between two
    cells is ``ε_face·(Δφ / centre_distance)·face_length`` with the face
    permittivity the arithmetic mean of the two adjoining cells; ``centre_distance``
    is the dual width ``(w[i]+w[i+1])/2`` and ``face_length`` is the cell width
    on the *other* axis. Faces onto a PEC cell take the free cell's ε one-sided
    rather than the mean (``pec``, optional — see the inline note at the stencil
    assembly). Out-of-array neighbours are simply omitted, which is the
    natural zero-flux (Neumann) edge; a grounded edge is instead handled by the
    caller marking the ring as ``fixed``. On a uniform mesh every coefficient
    reduces to a constant multiple of the old ``ε/da²`` stencil (a global row
    scaling that leaves φ unchanged).

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

    # Centre-to-centre distances (dual widths) between adjacent cells per axis.
    dac = 0.5 * (da_w[:-1] + da_w[1:])          # length Na-1, face i↔i+1
    dbc = 0.5 * (db_w[:-1] + db_w[1:])          # length Nb-1
    DA = da_w[:, None]                          # (Na, 1) a-widths (b-face length)
    DB = db_w[None, :]                          # (1, Nb) b-widths (a-face length)

    # Per-direction face weights, already shaped like the ``src`` slice: an
    # a-face carries ``face_length(=db_w) / centre_distance(=dac)``, a b-face
    # ``da_w / dbc``. ``+a``/``-a`` reference the same physical faces (rows
    # 0..Na-2), so both use ``dac``; likewise ``±b`` share ``dbc``.
    wa = DB / dac[:, None]                       # (Na-1, Nb)
    wb = DA / dbc[None, :]                        # (Na, Nb-1)

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
        (np.s_[0:Na - 1, :], np.s_[1:Na, :],     eps_a, wa),  # +a face
        (np.s_[1:Na, :],     np.s_[0:Na - 1, :], eps_a, wa),  # -a face
        (np.s_[:, 0:Nb - 1], np.s_[:, 1:Nb],     eps_b, wb),  # +b face
        (np.s_[:, 1:Nb],     np.s_[:, 0:Nb - 1], eps_b, wb),  # -b face
    )
    for src, nbr, eps, w in directions:
        if eps[src].size == 0:
            continue
        # Face permittivity: arithmetic mean of the two adjoining cells, EXCEPT
        # where one of them is PEC. ε inside a conductor is not a material
        # property — it is whatever the voxeliser happened to leave there (1.0)
        # — so averaging it in makes conductor-adjacent faces carry a different
        # ε ratio than interior faces. The filled and air systems then stop being
        # scalar multiples of one another, φ ≠ φ_air, and the exact cancellation
        # in ε_eff = C/C_air breaks. Taking the free cell's ε one-sided is both
        # physically right (the dielectric runs up to the conductor surface) and
        # restores A_filled = ε_r·A_air for a homogeneous fill.
        eps_face = 0.5 * (eps[src] + eps[nbr])
        if pec is not None:
            pec_src, pec_nbr = pec[src], pec[nbr]
            eps_face = np.where(pec_nbr & ~pec_src, eps[src], eps_face)
            eps_face = np.where(pec_src & ~pec_nbr, eps[nbr], eps_face)
        coef = eps_face * w
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

def _transverse_E(phi, a_c, b_c, pec):
    """``E_t = -∇φ`` on the cross-section (centred differences), zeroed in PEC.

    ``a_c``/``b_c`` are the cell-center coordinates along the two transverse
    axes, so :func:`numpy.gradient` uses the true (possibly non-uniform) spacing
    (2nd order in the interior). On a uniform mesh this matches the old scalar-Δ
    gradient.
    """
    dphi_da = np.gradient(phi, a_c, axis=0)
    dphi_db = np.gradient(phi, b_c, axis=1)
    Ea = -dphi_da
    Eb = -dphi_db
    Ea[pec] = 0.0
    Eb[pec] = 0.0
    return Ea, Eb


def _build_mode(phi, eps_a, eps_b, mu_a, pec, da_w, db_w, a_c, b_c, cfg,
                normal, position, k, full_shape, offset, label,
                a_nodes, b_nodes):
    """Assemble a :class:`TEMMode` (fields embedded into the full transverse plane)."""
    Ea, Eb = _transverse_E(phi, a_c, b_c, pec)

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
        transverse_axes=cfg['axes'],
        da=float(np.mean(da_w)), db=float(np.mean(db_w)),
        phi=phi_full, E=E, H=H, pec=pec_full, conductor_id=int(label),
        a_nodes=a_nodes, b_nodes=b_nodes)


def _attach_params(mode: TEMMode, phi, phi_air, eps_a, eps_b, da_w, db_w, a_c, b_c,
                   pec=None):
    """Fill C, L, Z₀, v, ε_eff from the field energy of the filled & air solves."""
    # Energy integral uses the gradient over the whole cross-section (V = 1 V):
    #   C = ε₀ ∫ (ε_a E_a² + ε_b E_b²) dA,  with a per-cell area dA_ij = da_i·db_j
    #   and the gradient taken against the true (non-uniform) centre coordinates.
    #
    # E must be zeroed inside PEC first — exactly as _transverse_E does. φ is
    # pinned there, but np.gradient is a centred difference, so a conductor cell
    # bordering a free cell still reports a nonzero E. That phantom energy is
    # weighted by the conductor's meaningless ε (1.0) in the filled solve and by
    # 1.0 in the air solve, so it does not cancel in C/C_air and biases ε_eff low.
    dA = da_w[:, None] * db_w[None, :]

    def _grad(p):
        Ea = -np.gradient(p, a_c, axis=0)
        Eb = -np.gradient(p, b_c, axis=1)
        if pec is not None:
            Ea[pec] = 0.0
            Eb[pec] = 0.0
        return Ea, Eb

    Ea, Eb = _grad(phi)
    C = EPS0 * np.sum((eps_a * Ea**2 + eps_b * Eb**2) * dA)

    Ea_air, Eb_air = _grad(phi_air)
    C_air = EPS0 * np.sum((Ea_air**2 + Eb_air**2) * dA)

    if C > 0 and C_air > 0:
        mode.capacitance = float(C)
        mode.inductance = float(1.0 / (C0**2 * C_air))
        mode.impedance = float(1.0 / (C0 * np.sqrt(C * C_air)))
        mode.v_phase = float(C0 * np.sqrt(C_air / C))
        mode.eps_eff = float(C / C_air)
