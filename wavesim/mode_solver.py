"""
mode_solver.py — 2D TEM (transverse electromagnetic) mode solver.

Given a grid face (or a rectangular subset of one), this finds the PEC conductor
cross-sections lying on that plane and solves the transverse-static field of each
TEM mode the structure supports. The resulting :class:`TEMMode` carries the 2D
transverse E (and H) profiles, which can be scaled and launched as an input port
via :meth:`TEMMode.to_source`.

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

Conformal (Dey–Mittra) PEC
-------------------------
When the grid carries cut-cell open fractions the solver switches to them
wholesale — conductor mask, stencil, energy integral and launched ê — so the
port is solved on the *same* geometry the FDTD steps. Without that the Z₀ the
port presents stops being the Z₀ the run presents, which is the consistency the
design exists to protect (no separate mode mesh; the mode is solved on the run's
grid). On the reference coax it takes the modal Z₀ error from +14.4% to −0.8%
and makes the launched profile exactly mirror-symmetric. The one substitution it
all rests on is derived at :func:`_plane_open_fractions` below; the gate is
``tests/test_conformal_mode_solver.py``.

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
from wavesim.parts import pec_node_mask
from wavesim.pec import (build_pec_edge_masks,
                         COVERED_FRACTION_TOL as _COVERED_FRACTION_TOL)


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
#   edge   — conformal open-fraction attrs for the two transverse E components,
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
              sigma=('sigma_x', 'sigma_y'),
              edge=('pec_edge_open_x', 'pec_edge_open_y'),
              E=('Ex', 'Ey'), H=('Hx', 'Hy'), h_sign=(-1.0, +1.0)),
    'y': dict(axes=('x', 'z'), ds=('dx', 'dz'),
              dp=('dxp', 'dzp'), cen=('xc', 'zc'), node=('x', 'z'),
              eps=('eps_x', 'eps_z'), mu='mu_x',
              sigma=('sigma_x', 'sigma_z'),
              edge=('pec_edge_open_x', 'pec_edge_open_z'),
              E=('Ex', 'Ez'), H=('Hx', 'Hz'), h_sign=(+1.0, -1.0)),
    'x': dict(axes=('y', 'z'), ds=('dy', 'dz'),
              dp=('dyp', 'dzp'), cen=('yc', 'zc'), node=('y', 'z'),
              eps=('eps_y', 'eps_z'), mu='mu_y',
              sigma=('sigma_y', 'sigma_z'),
              edge=('pec_edge_open_y', 'pec_edge_open_z'),
              E=('Ey', 'Ez'), H=('Hy', 'Hz'), h_sign=(-1.0, +1.0)),
}


# ====================================================================== #
# Conformal (Dey–Mittra) PEC on the mode plane
#
# The FDTD's conformal Faraday update integrates ``E·L`` around the open part
# of each H face's contour. A TEM mode is exactly the transverse field that
# makes the longitudinal H face's contour integral vanish:
#
#   (E_b·L_b)[i+1,j] − (E_b·L_b)[i,j] − (E_a·L_a)[i,j+1] + (E_a·L_a)[i,j] = 0
#
# which is solved identically by ``E_a·L_a = φ[i,j] − φ[i+1,j]`` for a node
# potential φ. So the conformal transverse field is **the ordinary gradient of
# φ divided by the OPEN edge length instead of the full one** — the open
# fraction lands on the stencil's centre distance, not on its face length. The
# rest follows: a fully covered edge (``L = 0``) forces φ equal at its two
# endpoints, which is the conductor equipotential condition. Nothing has to be
# assumed about which side of a cut edge the metal lies on.
#
# ``L = 0`` is not the whole of what the run holds at zero, though — an edge
# lying *in* a grid-aligned conductor surface has a full open length and is still
# a tangential E on PEC. The conductor φ is pinned on therefore comes from
# :func:`~wavesim.parts.pec_node_mask`, i.e. from the edge masks the FDTD
# actually applies (:func:`wavesim.pec.build_conformal_edge_masks`), and not from
# the fractions directly; the two agree wherever the geometry genuinely cuts.
#
# Every weight below is the legacy weight times an open fraction, so a grid
# whose fractions are all 1.0 reproduces the staircase assembly bit-for-bit on
# any mesh (``tests/test_conformal_mode_solver.py``).
# ====================================================================== #


def _plane_open_fractions(grid: FDTDGrid, cfg: dict, normal: str, k: int):
    """``(f_a, f_b)`` open fractions of the two transverse E edges on the plane.

    ``(None, None)`` when the grid carries no cut-cell geometry, which is the
    single switch between the conformal and the legacy assembly.
    """
    if not grid.is_conformal:
        return None, None
    return (np.asarray(_slice(getattr(grid, cfg['edge'][0]), normal, k), float),
            np.asarray(_slice(getattr(grid, cfg['edge'][1]), normal, k), float))


def _plane_edge_pec(grid: FDTDGrid, cfg: dict, normal: str, k: int):
    """``(m_a, m_b)`` — the two transverse E-edge masks the FDTD zeroes, sliced.

    The staircase counterpart of :func:`_plane_open_fractions`: exactly the edges
    :func:`wavesim.pec.apply_pec_mask` holds at zero, so ``ê`` can be masked on
    the *edges* the run masks rather than on the nodes ``φ`` is pinned at. The
    two are not the same set — a node on a conductor surface owns a live edge
    running out into the gap, and on a coax that edge carries the largest field
    on the plane. Zeroing it (which masking by :attr:`TEMMode.pec` does) throws
    that away and the launched/absorbed profile stops being the mode the run
    carries.
    """
    if grid.pec_mask is None:
        return None, None
    masks = dict(zip(('Ex', 'Ey', 'Ez'), build_pec_edge_masks(grid.pec_mask)))
    return (_slice(masks[cfg['E'][0]], normal, k),
            _slice(masks[cfg['E'][1]], normal, k))


def _conformal_node_pec(f_a: np.ndarray, f_b: np.ndarray) -> np.ndarray:
    """Nodes lying inside the conductor, from the *transverse* cut fractions.

    A node is in the metal iff at least one edge meeting it is *fully* covered:
    a covered edge lies wholly inside the conductor, so both of its endpoints
    do too. Every such node is index-adjacent to the far end of that edge, so
    :func:`scipy.ndimage.label` on this mask recovers the equipotential groups
    — the connected components of the fully-covered-edge graph — and the rest
    of the solver (conductor labelling, ground selection, pinning) is unchanged.

    This replaces the cell-centred ``pec_mask``, which is half a cell away from
    the nodes φ actually lives on. A *partially* covered edge is deliberately
    not counted: its open part carries field, and the ``1/L`` weight already
    places the node the correct sub-cell distance from the metal surface.

    Two conductors closer than one cell would be merged by the index adjacency
    (the labelling cannot tell a partially-open edge from a gap); that is below
    the resolution at which a cut cell means anything.

    Not what :func:`_build_mode` pins on any more — see
    :func:`~wavesim.parts.pec_node_mask`, which asks the same question of the
    run's own edge masks and so also catches an edge lying *in* a grid-aligned
    surface, whose own open fraction is a full 1.0. Kept because it is the plane-
    local statement of the rule, and because the two agreeing on a genuinely cut
    geometry is worth being able to assert.
    """
    covered = (f_a == 0.0) | (f_b == 0.0)
    covered[1:, :] |= (f_a[:-1, :] == 0.0)
    covered[:, 1:] |= (f_b[:, :-1] == 0.0)
    return covered


# The covered-fraction threshold this module used to carry for the port rule
# below, now :data:`wavesim.pec.COVERED_FRACTION_TOL` — the FDTD's edge masking
# is where it belongs, and one definition is the point: an edge this module calls
# covered has to be one the run holds at zero. Re-exported under the old name.
COVERED_FRACTION_TOL = _COVERED_FRACTION_TOL


def port_plane_pinned_nodes(grid: FDTDGrid, normal: str, k: int) -> np.ndarray:
    """Nodes on the plane at ``normal``-index ``k`` that lie on or inside metal.

    Literally the set :func:`_build_mode` pins ``φ`` at rather than solves for —
    the plane slice of :func:`~wavesim.parts.pec_node_mask`, asked of the same
    grid — and it has to stay literal. At a pinned node the discrete divergence
    ``∇·(ε ê)``, which the Laplacian drives to zero at every *free* node, is
    unconstrained; what it holds instead is the mode's induced **surface
    charge**, physically real and nonzero. That is harmless in the bulk, where
    the leapfrog carries it, and is the whole problem on a
    :class:`~wavesim.sources.ModalPort`'s ghost-H plane, where there is no
    leapfrog to carry it. A pin computed from a *different* rule than the solve
    used misses exactly the nodes the two disagree about, which is where the
    residual is.

    Deriving it from the transverse open fractions is not the same question and
    was the earlier answer here: it cannot see a node held by an edge that lies
    *in* a grid-aligned conductor surface, whose own open fraction is a full 1.0.
    """
    return _slice(pec_node_mask(grid), normal, k)


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
        """Build an amplitude-calibrated source that launches this mode.

        Parameters
        ----------
        waveform : Callable[[float], float]
            Temporal profile (e.g. a :class:`~wavesim.sources.GaussianPulse`).
        amplitude : float
            Forward-wave voltage in volts. The mode is normalised to a 1 V drive,
            so ``amplitude=1`` launches a wave a downstream
            :class:`~wavesim.monitors.VoltageMonitor` reads as ``waveform(t)``
            volts; scale it for any other level.
        fields : str
            ``'EH'`` (default) launches a directional (one-way, +normal) wave with
            the paired E and H sheets; ``'E'`` drives only the E footprint, a
            simpler bidirectional launch that radiates both ways.

        Notes
        -----
        The launch is *calibrated* — it impresses the modal current that a matched
        line turns into ``amplitude·waveform(t)`` volts forward, using the same
        current kernel a :class:`~wavesim.sources.TEMPort` uses
        (:meth:`build_port_kernel`), so the amplitude is correct on any grid or
        fill permittivity. (The earlier engine wrote the mode profile straight
        into the field arrays, ignoring the FDTD update coefficient, and came out
        √ε_r / S_c too large.)

        The directional (``'EH'``) launch reuses the port's corrected sheet
        pairing: the paired ``H = (n̂ × E)/η`` sheet sits one cell *behind* the E
        plane and is driven by the impressed current lagged onto its own
        space-time sample point, so the backward lobe cancels rather than merely
        shrinking. That places the port plane one cell inside the domain — a mode
        solved against a boundary (``position`` on the domain edge) has no room
        for the H sheet; move it at least one cell in. A waveform advertising a
        ``center_frequency`` (e.g. :class:`~wavesim.sources.Sinusoid`) tunes the
        lag to the numerical phase velocity at that frequency.

        Unlike a :class:`~wavesim.sources.TEMPort`, this is a pure soft source: it
        launches but does not absorb the returning wave. Use a ``TEMPort`` (a
        matched Thévenin drive) when you also want the port to terminate.
        """
        from wavesim.sources import _ModalLaunch  # local import avoids a cycle
        if 'E' not in fields:
            raise ValueError("fields must contain 'E' (an H-only launch is not "
                             "a valid source).")
        return _ModalLaunch(self, waveform, amplitude=amplitude,
                            directional=('H' in fields))

    def _staggered_port_fields(self, grid: FDTDGrid):
        """Discrete (Yee-staggered) transverse mode fields for the port kernel.

        Returns ``(E, H)``, each a ``{component: full-plane 2D array}`` mapping,
        with ``ê`` built as a **forward difference of φ landed on the Yee edges**:
        ``Ex[i,j] = −(φ[i+1,j] − φ[i,j]) / dxd[i]`` (dual/centre-to-centre width),
        which is exactly where :func:`wavesim.update.update_E` reads Ex. This is
        the discretisation that makes ``ê`` a null vector of the grid's transverse
        divergence — the collocated ``np.gradient`` field in ``self.E`` is not, and
        injecting it charges the domain. ``Ĥ = (n̂ × ê)/η`` is rebuilt from the same
        staggered ``ê`` so the directional E/H pairing stays consistent.

        Under conformal PEC the divisor becomes the **open** edge length
        ``f·d``: ``ê·L = -Δφ`` is precisely the condition that makes the
        conformal Faraday contour of the longitudinal H face vanish, i.e. that
        makes ``ê`` a genuine TEM (``H_n = 0``) field of the *cut* grid rather
        than of the staircased one. Fully covered edges divide by zero open
        length and are simply zeroed; a partially covered edge must keep its
        field, which is why no mask is applied on top.

        Both paths still come out zero on exactly the edge set the run holds at
        zero — but they get there differently, and the difference matters. The
        staircase path masks explicitly, via :func:`_plane_edge_pec`. The
        conformal path masks nothing: ``φ`` is pinned on the node mask the FDTD
        shorts (:func:`~wavesim.parts.pec_node_mask`), so ``Δφ`` already vanishes
        on every such edge, including one lying *in* a grid-aligned surface whose
        own open length is full. Masking ``ê`` there afterwards instead would
        zero the edge without telling ``φ``, and ``ê`` would stop being a null
        vector of the transverse divergence at the free nodes beside it — which
        is the one property a modal sheet cannot do without, since that
        divergence is what its ghost H deposits every step.

        Zeroing it on the *nodes* — which is what ``Ea[self.pec] = 0`` did —
        deletes the live edge a surface node owns out into the gap, the one
        carrying the largest field on a coax plane; see
        ``docs/mode_solver_staircase_node_mask.md``.
        """
        cfg = _NORMAL_CFG[self.normal]
        phi = self.phi
        Na, Nb = phi.shape
        # PRIMARY widths: φ sits on nodes, so consecutive φ are one primary
        # width apart. This is the same separation :func:`_face_coefs` divides
        # by, which is what keeps ê an exact null vector of the operator that
        # produced φ. (It read the dual widths before S5b — uniform-mesh
        # equivalent, wrong on a graded one.)
        prim = {'x': grid.dxp, 'y': grid.dyp, 'z': grid.dzp}
        da = prim[cfg['axes'][0]][:Na - 1][:, None]
        db = prim[cfg['axes'][1]][:Nb - 1][None, :]
        k = self.slice_index
        f_a, f_b = _plane_open_fractions(grid, cfg, self.normal, k)
        Ea = np.zeros_like(phi)
        Eb = np.zeros_like(phi)
        if f_a is None:
            Ea[:-1, :] = -(phi[1:, :] - phi[:-1, :]) / da
            Eb[:, :-1] = -(phi[:, 1:] - phi[:, :-1]) / db
            # Zero the *edges* the FDTD zeroes, not the nodes φ is pinned at:
            # a surface node owns a live edge running out into the gap, and
            # that edge carries the largest field on a coax plane. Masking by
            # ``self.pec`` (which is now the closed node box, see
            # :func:`solve_tem_modes`) would delete it. Δφ already vanishes on
            # the edges buried inside one conductor, so this only removes the
            # ones the run really holds at zero.
            m_a, m_b = _plane_edge_pec(grid, cfg, self.normal, k)
            if m_a is not None:
                Ea[m_a] = 0.0
                Eb[m_b] = 0.0
        else:
            La, Lb = f_a[:Na - 1, :] * da, f_b[:, :Nb - 1] * db
            np.divide(-(phi[1:, :] - phi[:-1, :]), La,
                      out=Ea[:-1, :], where=La > 0.0)
            np.divide(-(phi[:, 1:] - phi[:, :-1]), Lb,
                      out=Eb[:, :-1], where=Lb > 0.0)
            # Nothing is masked afterwards on this path, and nothing needs to
            # be: φ is pinned on the node mask the FDTD shorts, so Δφ already
            # vanishes on every edge the run holds at zero — including an edge
            # lying *in* a grid-aligned surface, which is fully open by its own
            # measure. Post-masking ``ê`` instead would zero the edge without
            # telling φ, and ``ê`` would stop being a null vector of the
            # transverse divergence at the free nodes next to it — which is the
            # one property a modal sheet cannot do without.

        # η = η₀·√(μ_r/ε_r) on the plane, exactly as :func:`_build_mode`.
        eps_a = _slice(getattr(grid, cfg['eps'][0]), self.normal, k)
        mu_a = _slice(getattr(grid, cfg['mu']), self.normal, k)
        eta = ETA0 * np.sqrt(mu_a / np.where(eps_a > 0, eps_a, 1.0))
        sa, sb = cfg['h_sign']
        # Ĥ is built from the already-masked ê, so it inherits its zeros on
        # both paths — no separate masking, which would only ever remove more
        # than the run does.
        Ha = sa * Eb / eta
        Hb = sb * Ea / eta

        E = {cfg['E'][0]: Ea, cfg['E'][1]: Eb}
        H = {cfg['H'][0]: Ha, cfg['H'][1]: Hb}
        return E, H

    def numerical_admittance_scale(self, grid: FDTDGrid) -> float:
        """Discrete numerical-admittance correction ``s`` for a modal impedance
        sheet (:class:`~wavesim.sources.ModalPort`).

        A matched impedance sheet writes ghost H ``= s·(1/η)·(n̂×ê)·V``. For the
        continuum mode ``s = 1`` exactly; on the grid it departs from 1 only
        through **transverse-discretisation error** (staircased cross-section,
        electrostatic ``ê`` vs the true discrete propagating mode), shrinking back
        toward 1 as the cross-section is refined. It is derived here from power
        balance rather than tuned: the sheet dissipates ``P = s·G·V²`` with the
        **discrete modal conductance**

            ``G = Σ_cells (ê² / η)·dA``   (both transverse E components),

        and a matched wave of modal voltage ``V`` carries ``P = V²/Z₀``, so a
        no-reflection sheet needs ``s = 1/(Z₀·G)``. Both ``Z₀`` (the energy-
        integral characteristic impedance) and ``G`` are computed on the *same*
        staggered fields the sheet uses, so their discretisation errors cancel.

        For a homogeneous fill the cancellation is exact on **both** paths: Z₀
        and G are then two readings of one energy integral, ``G = C·c₀/√ε_r``
        against ``Z₀ = √ε_r/(c₀·C)``, so ``s = 1`` to round-off. That holds only
        because ``ê`` vanishes on exactly the edges whose face coefficient
        contributes nothing to the energy — masking it by the node mask instead
        left the staircase path reading ≈1.006, which looked like a
        discretisation floor and was not (see
        ``docs/mode_solver_staircase_node_mask.md``). An inhomogeneous
        cross-section still has a genuine residue.

        Requires the mode's ``impedance`` (solve with ``compute_params=True``).
        """
        if self.impedance is None or not self.impedance > 0:
            raise ValueError(
                "numerical_admittance_scale needs the mode's Z₀; solve with "
                "compute_params=True or pass admittance_scale= to ModalPort.")
        cfg = _NORMAL_CFG[self.normal]
        E_stag, _H = self._staggered_port_fields(grid)
        k = self.slice_index
        # Transverse per-cell primary widths → cell area dA on the plane. Under
        # conformal PEC each component's area is scaled by its own edge's open
        # fraction, so ``G`` integrates over the open cross-section — the same
        # open-area weighting the mode's Z₀ comes from, which is what lets the
        # two discretisation errors keep cancelling in ``s = 1/(Z₀·G)``.
        prim = {'x': grid.dxp, 'y': grid.dyp, 'z': grid.dzp}
        wa = prim[cfg['axes'][0]]
        wb = prim[cfg['axes'][1]]
        dA = wa[:, None] * wb[None, :]
        f_open = dict(zip(cfg['E'],
                          _plane_open_fractions(grid, cfg, self.normal, k)))
        mu_a = _slice(getattr(grid, cfg['mu']), self.normal, k)
        G = 0.0
        eps_of = {'Ex': grid.eps_x, 'Ey': grid.eps_y, 'Ez': grid.eps_z}
        for comp in cfg['E']:
            ehat = E_stag[comp]
            dA_c = dA if f_open[comp] is None else dA * f_open[comp]
            a, b = np.nonzero(ehat)
            if a.size == 0:
                continue
            ii, jj, kk = _plane_to_grid(self.normal, k, a, b)
            epsr = eps_of[comp][ii, jj, kk]
            eta = ETA0 * np.sqrt(mu_a[a, b] / np.where(epsr > 0, epsr, 1.0))
            G += float(np.sum(ehat[a, b] ** 2 / eta * dA_c[a, b]))
        if G <= 0.0:
            raise ValueError("Mode has no transverse E energy; cannot scale.")
        return 1.0 / (self.impedance * G)

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

        # Port fields are the DISCRETE (Yee-staggered) mode, not the collocated
        # ``self.E`` used for plotting / per-unit-length energy. Building ê as a
        # forward difference of φ landed on the Yee edges makes it an exact null
        # vector of the grid's transverse divergence (∇·(ε ê) = 0 to round-off in
        # the dielectric), so ``E += κ·ê·I`` deposits NO charge — even under a
        # DC-containing pulse. The collocated centred-gradient field is ~20%
        # divergent on the Yee grid and would slowly charge the domain. See
        # :meth:`_staggered_port_fields`.
        E_stag, H_stag = self._staggered_port_fields(grid)

        # Gather nonzero plane cells per E component. ``S = Σ ε_r Ê²`` normalises
        # the read-back projection (dimensionless, so V*=1 for the pure mode);
        # ``Sv = Σ dV_c ε_r Ê²`` volume-weights the energy for κ (per-cell local
        # Yee volume, all-primary as in update_E). On a uniform grid Sv = dV·S.
        gathered = {}
        S = 0.0
        Sv = 0.0
        f_open = dict(zip(cfg['E'],
                          _plane_open_fractions(grid, cfg, self.normal, k)))
        for comp in cfg['E']:
            Ehat2d = E_stag[comp]
            a, b = np.nonzero(Ehat2d)
            if a.size == 0:
                continue
            ii, jj, kk = _plane_to_grid(self.normal, k, a, b)
            Ehat = Ehat2d[a, b]
            epsr = eps_of[comp][ii, jj, kk]
            dV_c = grid.dxp[ii] * grid.dyp[jj] * grid.dzp[kk]
            if f_open[comp] is not None:   # cut cells store energy only where open
                dV_c = dV_c * f_open[comp][a, b]
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
                Hhat2d = H_stag[comp]
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

def _warn_if_lossy_plane(grid: FDTDGrid, cfg: dict, normal: str, k: int) -> None:
    """Warn when the port plane carries conductivity, and solve lossless anyway.

    The static solve is ``∇·(ε∇φ) = 0`` over a **real** ε. Conductivity makes it
    ``∇·(ε̃∇φ) = 0`` with ``ε̃ = ε − jσ/(ωε₀)``, so φ, Z₀ and γ all go complex and
    become frequency-dependent — a different solver, not a coefficient change.

    Solving on Re(ε) is the right default rather than a refusal: a port plane
    normally sits in low-loss dielectric, where the correction to Z₀ is second
    order in tan δ (0.5% at tan δ = 0.1), and the alternative is refusing to
    launch into any model that has a lossy substrate somewhere. But the number
    returned is the *lossless* Z₀ of that cross-section, so it is said out loud.
    """
    if not grid.is_lossy:
        return
    worst = max(float(np.max(_slice(getattr(grid, name), normal, k)))
                for name in cfg['sigma'])
    if worst <= 0.0:
        return
    warnings.warn(
        f"solve_tem_modes: the port plane carries conductivity (max sigma = "
        f"{worst:.4g} S/m). The transverse-static solve uses the real part of "
        f"eps only, so the reported Z0, C, L and eps_eff are those of the "
        f"lossless cross-section; the true ones are complex and "
        f"frequency-dependent. Fine when the port sits in low-loss dielectric "
        f"(the usual case) -- place the port plane clear of the lossy region if "
        f"you need the loss reflected in Z0.",
        RuntimeWarning, stacklevel=2)


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
    _warn_if_lossy_plane(grid, cfg, normal, k)

    # Conformal cut-cell geometry, if the grid carries it. The conductor mask
    # then comes from the cut edges rather than from the cell-centred
    # ``pec_mask``, because φ lives on the nodes those edges connect. On the
    # staircase path the same reasoning applies, and the node mask is the one
    # the FDTD itself shorts — :func:`~wavesim.parts.pec_node_mask`, i.e. the
    # end points of every edge ``build_pec_edge_masks`` zeroes. Slicing
    # ``pec_mask`` and calling it a node mask instead kept each conductor's
    # low-side surface node and dropped its high-side one, leaving every
    # staircase conductor a cell short along each axis on its high side only:
    # −7.7% in C' on a parallel-plate line, −8.1% on a coarse coax, and a port
    # solved on geometry the run does not step. See
    # ``docs/mode_solver_staircase_node_mask.md``.
    #
    # Both paths ask :func:`~wavesim.parts.pec_node_mask`, which reads whichever
    # edge rule the grid is actually running. Asking the conformal fractions
    # directly (:func:`_conformal_node_pec`) is the same answer on a cut cell and
    # the wrong one where the conductor is tangent to the grid: an edge lying in
    # a grid-aligned surface is fully open by its own measure, the run holds it
    # at zero all the same, and a φ that does not know that comes out with a
    # potential drop along a conductor surface.
    fa_full, fb_full = _plane_open_fractions(grid, cfg, normal, k)
    if fa_full is None and grid.pec_mask is None:
        pec_full = np.zeros(eps_a_full.shape, dtype=bool)
    else:
        pec_full = _slice(pec_node_mask(grid), normal, k)

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
    f_a = None if fa_full is None else np.ascontiguousarray(fa_full[sub])
    f_b = None if fb_full is None else np.ascontiguousarray(fb_full[sub])

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
                                                     fixed, pec, f_a, f_b)
    # Air-filled companion (ε≡1) for the per-unit-length parameters. The PEC
    # one-sided rule is a no-op here (ε is uniformly 1), which is precisely why
    # applying it to the filled solve restores φ == φ_air on a homogeneous fill.
    if compute_params:
        lu_air, B_air, _, _ = _factor_laplacian(
            np.ones_like(eps_a), np.ones_like(eps_b), da_w, db_w, fixed, pec,
            f_a, f_b)

    modes: List[TEMMode] = []
    for Ls in signals:
        phi = _solve_one(lu, B, free_idx, fixed_cells, labels, fixed, Ls)
        mode = _build_mode(phi, eps_a, eps_b, mu_a, pec, da_w, db_w, a_c, b_c,
                           cfg, normal, position, k, full_shape, (ia0, ib0), Ls,
                           a_nodes, b_nodes)
        if compute_params:
            phi_air = _solve_one(lu_air, B_air, free_idx, fixed_cells,
                                 labels, fixed, Ls)
            _attach_params(mode, phi, phi_air, eps_a, eps_b, da_w, db_w,
                           pec, f_a, f_b)
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

def _node_dual(w: np.ndarray) -> np.ndarray:
    """Dual-cell width owned by each node, from the per-cell primary widths.

    Node ``j`` owns the cell centres either side of it, ``(w[j-1]+w[j])/2`` —
    which is the grid's own ``dyd[j-1]``, recomputed here because the solver
    only ever receives the primary widths of its (possibly sub-rectangular)
    solve region. Node 0 sits on the region's edge and owns only the half cell
    ``w[0]/2``.

    That half cell is inert for ``boundary='ground'``, where the whole edge ring
    is pinned and its coefficients are never assembled; it matters only under
    ``'neumann'``, where a boundary node really does own half a control volume.
    """
    out = np.empty_like(w)
    out[0] = 0.5 * w[0]
    out[1:] = 0.5 * (w[:-1] + w[1:])
    return out


def _face_eps(eps: np.ndarray, pec, axis: int) -> np.ndarray:
    """Permittivity of each face — the stored edge value, not an average.

    ``eps_a[i,j]`` **is** ``eps_x[i,j]``, the permittivity wavesim keeps on the
    Ex edge joining node ``i`` to node ``i+1`` — precisely the face being
    weighted. Averaging it with its neighbour, as this did before S5c, smears
    data that is already in the right place: on a two-layer parallel plate the
    material interface then lands mid-face and the capacitance comes out 0.82%
    wrong, against **exact** when the stored value is used.

    A face straddling the conductor surface is the exception. ε there is not a
    material property — it is whatever the voxeliser left inside the metal
    (1.0) — so such a face borrows the ε of the next face **outward**, which
    lies wholly in the dielectric. That is the old cell-wise one-sided rule
    shifted onto the edges where it belongs, and it is load-bearing: it is what
    keeps the filled and air operators exact scalar multiples of one another,
    hence φ = φ_air and ε_eff = ε_r to round-off. Dropping it (direct ε alone)
    reads 2.273 on a homogeneous coax against a true 2.300.
    """
    n = eps.shape[axis]
    part = (lambda a, s: a[s]) if axis == 0 else (lambda a, s: a[:, s])
    face = part(eps, np.s_[:n - 1]).copy()
    if pec is None or face.size == 0:
        return face
    lo, hi = part(pec, np.s_[:n - 1]), part(pec, np.s_[1:n])
    outward_hi = part(eps, np.s_[1:n])                       # face i+1
    outward_lo = np.concatenate(                             # face i-1, clamped
        [part(eps, np.s_[:1]), part(eps, np.s_[:n - 2])], axis=axis)
    face = np.where(lo & ~hi, outward_hi, face)
    face = np.where(hi & ~lo, outward_lo, face)
    return face


def _face_coefs(eps_a, eps_b, da_w, db_w, pec=None, f_a=None, f_b=None):
    """Per-face conductances of the ε-weighted transverse Laplacian.

    Returns ``(ca, cb)``: ``ca[i,j]`` (shape ``(Na-1, Nb)``) is the coefficient
    of the face between cells ``(i,j)`` and ``(i+1,j)``, ``cb[i,j]`` (shape
    ``(Na, Nb-1)``) that of the face between ``(i,j)`` and ``(i,j+1)``. Each is

        ``ε_face · face_length / centre_distance``

    φ lives on the Yee **nodes**, so the control volume of the equation at node
    ``(i,j)`` is the *dual* cell spanning ``xc[i-1]..xc[i]`` — hence
    ``face_length`` is a **dual** width on the other axis and
    ``centre_distance`` is the **primary** width along the coupling axis (the
    node-to-node separation, which is what ``E = Δφ/d`` divides by).

    This pairing was originally the other way round — primary face length
    against dual centre distance — which is S0's bug living in the mode solver
    rather than in the kernels. A uniform mesh hides it exactly
    (``dxp == dxd`` to the last ULP), so nothing caught it until the conformal
    derivation made the node picture explicit. On a graded mesh it costs an
    order: the discrete field of a parallel-plate line stops being uniform, so
    even that exactly-solvable case comes out wrong
    (``tests/test_mode_solver_spacing.py``).

    **Face permittivity** comes from :func:`_face_eps` — the stored edge value,
    used directly rather than averaged.

    **Conformal PEC**: ``f_a``/``f_b`` scale the centre distance down to the
    *open* edge length, ``L = f·primary width``, so the coefficient grows as
    ``1/f`` — a node just outside the metal sits a short distance from the
    surface and is therefore strongly coupled to it, which is the correct
    Dirichlet behaviour. With the pairing above this ``L`` is now literally the
    open edge length the conformal derivation asks for, rather than the
    uniform-mesh-equivalent ``f·dual`` S5 settled for. A fully covered edge (``f = 0``) gets coefficient 0
    and carries no flux equation at all; both of its endpoints are pinned by
    :func:`~wavesim.parts.pec_node_mask`, so the constraint it stands for (equal
    φ) is imposed by the pinning instead. An edge lying *in* a grid-aligned
    surface keeps a full ``f`` and so keeps its flux equation — but the same
    pinning holds both its endpoints too, so ``Δφ`` is zero across it and the
    equation is satisfied trivially.

    Both arrays are returned once per face, not once per direction, and the
    assembly walks each face twice — the two views share the same coefficient.
    The energy integral in :func:`_attach_params` reuses them, which is what
    makes the reported capacitance exactly the quadratic form of the operator
    that was actually solved.
    """
    Na, Nb = eps_a.shape
    DA = _node_dual(da_w)[:, None]              # (Na, 1) dual a-widths (b-face)
    DB = _node_dual(db_w)[None, :]              # (1, Nb) dual b-widths (a-face)

    # Coupling distance: the node-to-node separation, i.e. the PRIMARY width of
    # the edge joining them — and under conformal PEC only its open part.
    la = np.broadcast_to(da_w[:Na - 1, None], (max(Na - 1, 0), Nb)).astype(np.float64)
    lb = np.broadcast_to(db_w[None, :Nb - 1], (Na, max(Nb - 1, 0))).astype(np.float64)
    if f_a is not None:
        la = la * f_a[:Na - 1, :]
        lb = lb * f_b[:, :Nb - 1]

    ca = np.divide(np.broadcast_to(DB, la.shape), la,
                   out=np.zeros_like(la), where=la > 0.0) * _face_eps(eps_a, pec, 0)
    cb = np.divide(np.broadcast_to(DA, lb.shape), lb,
                   out=np.zeros_like(lb), where=lb > 0.0) * _face_eps(eps_b, pec, 1)
    return ca, cb


def _factor_laplacian(eps_a, eps_b, da_w, db_w, fixed, pec=None,
                      f_a=None, f_b=None):
    """Assemble and LU-factorise the ε-weighted 2D Laplacian over free cells.

    Discretises ``∂_a(ε_a ∂_a φ) + ∂_b(ε_b ∂_b φ) = 0`` with a 5-point
    variable-coefficient **finite-volume** stencil on a rectilinear (possibly
    non-uniform) transverse mesh. ``da_w``/``db_w`` are the per-cell primary
    widths along the two transverse axes; the per-face conductances come from
    :func:`_face_coefs`, which also carries the PEC ε rule and the conformal
    open-length weighting (``f_a``/``f_b``). Out-of-array neighbours are simply
    omitted, which is the natural zero-flux (Neumann) edge; a grounded edge is
    instead handled by the caller marking the ring as ``fixed``. On a uniform
    mesh every coefficient reduces to a constant multiple of the old ``ε/da²``
    stencil (a global row scaling that leaves φ unchanged).

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

    ca, cb = _face_coefs(eps_a, eps_b, da_w, db_w, pec, f_a, f_b)

    # The 5-point stencil is built one *face direction* at a time (4 vectorised
    # passes), not cell-by-cell. For each direction the in-bounds region is a
    # whole-array slice ``src`` paired with its neighbour slice ``nbr``; ``+a``
    # and ``-a`` are the same physical faces seen from opposite sides, so they
    # share one coefficient array. Omitting the out-of-bounds border rows/columns
    # reproduces the zero-flux (Neumann) edge. Off-diagonal couplings split by
    # whether the neighbour is free (→ A) or pinned (→ B); the diagonal
    # accumulates −Σ(face coef) over the same faces.
    diag = np.zeros((Na, Nb), dtype=np.float64)
    rows_A, cols_A, data_A = [], [], []
    rows_B, cols_B, data_B = [], [], []

    directions = (
        (np.s_[0:Na - 1, :], np.s_[1:Na, :],     ca),  # +a face
        (np.s_[1:Na, :],     np.s_[0:Na - 1, :], ca),  # -a face
        (np.s_[:, 0:Nb - 1], np.s_[:, 1:Nb],     cb),  # +b face
        (np.s_[:, 1:Nb],     np.s_[:, 0:Nb - 1], cb),  # -b face
    )
    for src, nbr, coef in directions:
        if coef.size == 0:
            continue
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


def _attach_params(mode: TEMMode, phi, phi_air, eps_a, eps_b, da_w, db_w,
                   pec=None, f_a=None, f_b=None):
    """Fill C, L, Z₀, v, ε_eff from the field energy of the filled & air solves.

    Both capacitances are the quadratic form of the operator that was actually
    solved (:func:`_fv_energy`). Until S5d the staircase path instead integrated
    a collocated ``np.gradient`` field over per-cell areas, which is a different
    and worse discretisation: it reads 3.9% low on an exactly-solvable parallel
    plate, and on the reference coax it was most of the gap between the +14.4%
    staircase Z₀ error and the +6.9% the same staircased conductor gives through
    this integral.
    """
    C = _fv_energy(phi, eps_a, eps_b, da_w, db_w, pec, f_a, f_b)
    C_air = _fv_energy(phi_air, np.ones_like(eps_a), np.ones_like(eps_b),
                       da_w, db_w, pec, f_a, f_b)
    _set_params(mode, C, C_air)


def _fv_energy(phi, eps_a, eps_b, da_w, db_w, pec, f_a, f_b) -> float:
    """Capacitance as the quadratic form of the conformal operator: ``-φᵀAφ``.

    Writing the face conductance as ``ε·face_length/L`` with ``L`` the *open*
    edge length, each face contributes

        ``ε·(Δφ)²·face_length/L  =  ε·E²·(L · face_length)``

    — the field energy on the face's **open area**, since ``E = Δφ/L`` is
    exactly the field the conformal FDTD carries on that edge. So this is the
    open-area energy integral the plan asks for, and it comes out of the
    operator rather than being re-derived beside it: reusing
    :func:`_face_coefs` guarantees ``C`` is the energy of the system that was
    actually solved, which is what keeps ``C = ε_r·C_air`` exact for a
    homogeneous fill (the filled and air operators are then scalar multiples,
    so φ = φ_air and every face coefficient scales by ε_r).

    Both paths use it since S5d; ``f_a``/``f_b`` of ``None`` reduce the open
    lengths to the full ones, so the staircase path is the same integral over
    uncut geometry.
    """
    ca, cb = _face_coefs(eps_a, eps_b, da_w, db_w, pec, f_a, f_b)
    e = 0.0
    if ca.size:
        e += float(np.sum(ca * (phi[1:, :] - phi[:-1, :]) ** 2))
    if cb.size:
        e += float(np.sum(cb * (phi[:, 1:] - phi[:, :-1]) ** 2))
    return EPS0 * e


def _set_params(mode: TEMMode, C: float, C_air: float) -> None:
    """Fill the per-unit-length parameters from the two capacitances."""
    if C > 0 and C_air > 0:
        mode.capacitance = float(C)
        mode.inductance = float(1.0 / (C0**2 * C_air))
        mode.impedance = float(1.0 / (C0 * np.sqrt(C * C_air)))
        mode.v_phase = float(C0 * np.sqrt(C_air / C))
        mode.eps_eff = float(C / C_air)
