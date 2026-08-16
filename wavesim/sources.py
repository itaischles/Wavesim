"""
sources.py — excitation waveforms and the Source injection abstraction.

Two layers live here, and they compose:

1. Waveforms — the *time* part of an excitation. A waveform is any callable
   ``f(t) -> float``. :class:`GaussianPulse` is the built-in baseband pulse and
   :class:`Sinusoid` the built-in CW drive (both are callable); any
   ``lambda t: ...`` works anywhere a waveform is expected. Prefer
   :class:`Sinusoid` over a hand-rolled ``lambda t: sin(2*pi*f*t)`` — it ramps
   the turn-on, which the lambda does not.

2. Source objects — the injection abstraction. A :class:`Source` bundles *where*
   and *which components* (``spatial_profiles(grid)`` → ``{component: weights}``),
   *when* (``waveform(t)``), and exposes ``inject(grid, t)`` that performs the
   soft, additive write. ``Simulation`` calls ``inject`` once per timestep; you
   can also call it yourself from a hand-written loop.

A Source captures the three things every excitation has:
    * **location** — which cells it occupies (held by each subclass's ctor args);
    * **spatial profile** — per-cell additive weights, *per field component*, so
      a single source can drive several components at once (a coaxial TEM mode is
      a radial E → ``Ex`` and ``Ey``; a waveguide port carries several transverse
      components);
    * **temporal profile** — the shared ``waveform(t)``.

Soft injection (+=) is transparent to passing waves (no impedance mismatch).
Hard injection (=) reflects waves; every Source here adds (+=), with one
documented exception: :class:`LineSource` in ideal-voltage mode (``voltage=``
with no load) pins ∫E·dl on its line, which is a hard write — the
physically correct behaviour of a zero-impedance source.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple, Union
import warnings
import weakref

import numpy as np

from wavesim.constants import C0, EPS0, ETA0
from wavesim.grid import FDTDGrid
from wavesim.lumped import LumpedNetwork
# Shared with VoltageMonitor so a LineSource and a monitor on the same path
# snap to identical Yee E-edges and agree bit-for-bit on ∫E·dl.
from wavesim.monitors import _build_path_quadrature
from wavesim.pec import conformal_edge_eps


# ====================================================================== #
# Waveforms (the time part)
# ====================================================================== #

class Waveform(ABC):
    """Abstract temporal profile: a callable ``f(t) -> float``.

    Any plain callable (e.g. ``lambda t: np.sin(2*np.pi*f*t)``) is equally
    acceptable wherever a waveform is expected — subclassing is only a
    convenience for parameterised, self-describing pulses like
    :class:`GaussianPulse`.
    """

    @abstractmethod
    def __call__(self, t: float) -> float:
        """Scalar waveform value at time ``t`` (seconds)."""


@dataclass
class GaussianPulse(Waveform):
    """Gaussian pulse waveform.

    Callable: ``GaussianPulse(...)(t)`` returns the pulse value at ``t``, so an
    instance can be passed directly as the ``waveform`` of a :class:`Source`.

    Parameters
    ----------
    t0 : float
        Pulse centre time (s).
    width : float
        Pulse half-width / standard deviation (s). Spectral bandwidth (-3 dB)
        ≈ ``1 / (2π · width)``.
    amplitude : float
        Peak amplitude.
    """
    t0: float
    width: float
    amplitude: float = 1.0

    def __call__(self, t: float) -> float:
        return self.amplitude * np.exp(-0.5 * ((t - self.t0) / self.width) ** 2)

    @classmethod
    def for_fmax(cls, f_max: float, amplitude: float = 1.0) -> "GaussianPulse":
        """Build a pulse targeting ``f_max`` Hz.

        ``width`` is chosen so the -3 dB bandwidth ≈ ``f_max``, and ``t0`` so the
        pulse has fully risen by ``t = 0`` (amplitude there is <1% of peak),
        keeping the excitation contained in the simulation window.
        """
        width = 1.0 / (2.0 * np.pi * f_max)
        t0 = 4.0 * width
        return cls(t0=t0, width=width, amplitude=amplitude)


@dataclass
class Sinusoid(Waveform):
    """Continuous-wave (CW) sinusoid with a smooth turn-on ramp.

    Callable, like :class:`GaussianPulse`, so an instance can be passed directly
    as the ``waveform`` of a :class:`Source`.

    The ramp is the point of this class. A bare ``sin(ωt)`` switched on at t=0
    starts at zero *amplitude* but at maximum *slope*, and that kink is a
    broadband impulse: it injects energy far outside the intended line, excites
    resonances that have nothing to do with the drive frequency, and can leave a
    slowly-decaying static residue. Multiplying by a raised-cosine envelope over
    the first ``ramp_cycles`` periods makes both the value and its derivative
    continuous at turn-on, so the spectrum stays where it belongs.

    Parameters
    ----------
    frequency : float
        Drive frequency (Hz).
    amplitude : float
        Steady-state peak amplitude (reached after the ramp).
    phase : float
        Phase offset (radians). The default 0 starts the carrier at zero.
    ramp_cycles : float
        Length of the raised-cosine turn-on, in periods. Set to 0 to disable the
        ramp and start the carrier abruptly — only sensible when ``phase`` leaves
        the waveform continuous at t=0, and it forfeits the protection above.

    Notes
    -----
    Output is identically zero for ``t <= 0``.
    """
    frequency: float
    amplitude: float = 1.0
    phase: float = 0.0
    ramp_cycles: float = 3.0

    def __call__(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        envelope = 1.0
        if self.ramp_cycles > 0.0:
            t_ramp = self.ramp_cycles / self.frequency
            if t < t_ramp:
                # Raised cosine: 0 → 1 with zero slope at both ends.
                envelope = 0.5 * (1.0 - np.cos(np.pi * t / t_ramp))
        return self.amplitude * envelope * np.sin(
            2.0 * np.pi * self.frequency * t + self.phase)

    @property
    def center_frequency(self) -> float:
        """Spectral centre (Hz) — here simply the carrier frequency.

        Read by machinery that has to tune itself to the drive frequency (the
        numerical-impedance correction of a directional launch). Waveforms
        without this attribute fall back to frequency-independent behaviour.
        """
        return self.frequency


# ====================================================================== #
# Source objects (the injection abstraction)
# ====================================================================== #

class Source(ABC):
    """
    Base class for excitations.

    A Source knows two things:
      * ``waveform(t)`` — the scalar temporal profile (any callable ``f(t)``).
      * ``spatial_profiles(grid)`` — a mapping ``{component: weights}`` where
        each ``weights`` array has the grid shape (Nx, Ny, Nz). Every step, each
        named component gets ``waveform(t) * weights`` *added* to it.

    Returning a mapping (rather than one component) lets a single source drive
    several field arrays at once — e.g. a coaxial TEM mode whose radial E maps to
    both ``Ex`` and ``Ey``, or a waveguide port carrying several transverse
    components.

    ``inject(grid, t)`` performs the soft (additive) write and is what the time
    loop calls. The profiles are built once and cached (``_profiles``): geometry
    is fixed for a run, so the per-cell weight arrays — which may be full-grid —
    are computed lazily on the first ``inject`` and reused thereafter rather than
    reallocated every timestep. Subclasses implement ``spatial_profiles``; a
    cheap/point source may override ``inject`` to skip building any array (see
    :class:`PointSource`).
    """

    def __init__(self, waveform: Callable[[float], float]) -> None:
        self.waveform = waveform
        self._profiles: Dict[str, np.ndarray] | None = None  # built once, cached

    @abstractmethod
    def spatial_profiles(self, grid: FDTDGrid) -> Dict[str, np.ndarray]:
        """Per-component additive weights, each broadcastable to (Nx, Ny, Nz)."""

    def inject(self, grid: FDTDGrid, t: float) -> None:
        """Soft-add ``waveform(t) * weights`` into every driven component."""
        if self._profiles is None:
            self._profiles = self.spatial_profiles(grid)
        amp = self.waveform(t)
        for component, profile in self._profiles.items():
            getattr(grid, component)[...] += amp * profile


class PointSource(Source):
    """
    Soft point excitation: one cell of one component, driven by a waveform.

    Equivalent to the one-liner ``grid.<component>[i, j, k] += waveform(t)``, but
    as a reusable object. Overrides ``inject`` so no full-grid profile is
    allocated (the profile is a single cell).

    Parameters
    ----------
    component : str
        Field component to drive ('Ex'..'Hz').
    x, y, z : float
        Physical position of the injection point in metres, snapped to the
        nearest cell against the grid (use z=0 for an Nz=1 slice).
    waveform : Callable[[float], float]
        Time function, e.g. a :class:`GaussianPulse` instance or a custom lambda.
    """

    def __init__(self, component: str, x: float, y: float, z: float,
                 waveform: Callable[[float], float]) -> None:
        super().__init__(waveform)
        self.component = component
        self.x, self.y, self.z = x, y, z

    def spatial_profiles(self, grid: FDTDGrid) -> Dict[str, np.ndarray]:
        """Full-grid profile with a single 1.0 at the source cell — for inspection."""
        i, j, k = grid.position_to_index(self.x, self.y, self.z)
        prof = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=np.float64)
        prof[i, j, k] = 1.0
        return {self.component: prof}

    def inject(self, grid: FDTDGrid, t: float) -> None:
        i, j, k = grid.position_to_index(self.x, self.y, self.z)
        getattr(grid, self.component)[i, j, k] += self.waveform(t)


class ArraySource(Source):
    """
    Distributed soft excitation from user-supplied spatial profiles.

    The multi-component workhorse: covers line/shaped/annular/modal drives — any
    excitation whose per-cell weights you can express as arrays. Each step every
    given component is updated as ``component += waveform(t) * profile``.

    Parameters
    ----------
    profiles : mapping or tuple
        Either ``{component: ndarray(Nx, Ny, Nz)}`` driving one or more
        components, or a single ``(component, ndarray)`` pair for convenience.
        Zero cells are not driven; a single nonzero z-plane gives a planar
        source, two components with a radial shape give a coax-TEM-like mode, etc.
    waveform : Callable[[float], float]
        Shared time function (e.g. a :class:`GaussianPulse` or carrier lambda).

    Notes
    -----
    Each profile's shape is validated against the grid on first injection.
    """

    def __init__(self,
                 profiles: Union[Mapping[str, np.ndarray],
                                 Tuple[str, np.ndarray]],
                 waveform: Callable[[float], float]) -> None:
        super().__init__(waveform)
        # Accept a single (component, array) pair as a convenience.
        if isinstance(profiles, tuple) and len(profiles) == 2 \
                and isinstance(profiles[0], str):
            profiles = {profiles[0]: profiles[1]}
        self.profiles = {comp: np.asarray(arr, dtype=np.float64)
                         for comp, arr in dict(profiles).items()}

    def spatial_profiles(self, grid: FDTDGrid) -> Dict[str, np.ndarray]:
        expected = (grid.Nx, grid.Ny, grid.Nz)
        for comp, arr in self.profiles.items():
            if arr.shape != expected:
                raise ValueError(
                    f"ArraySource profile for {comp!r} has shape {arr.shape}, "
                    f"which does not match grid shape {expected}.")
        return self.profiles


# ====================================================================== #
# Planned source families — API locked, implementation pending.
# These reserve the constructor signatures so calling code and docs can be
# written against the final API; the bodies raise NotImplementedError until built.
# ====================================================================== #

class PlaneSource(Source):
    """
    Planar excitation over a full slice normal to one axis — *not yet implemented*.

    Intended for plane waves and waveguide ports. ``profiles=None`` gives a
    uniform plane wave; otherwise a mapping of 2D transverse mode profiles
    (placed on the slice) defines a port mode.

    Parameters
    ----------
    waveform : Callable[[float], float]
        Shared time function.
    axis : str
        Slice normal, one of 'x', 'y', 'z'.
    position : float
        Physical position (metres) along ``axis`` where the slice sits, snapped
        to the nearest cell against the grid.
    profiles : mapping, optional
        ``{component: 2D-array}`` transverse mode profiles; ``None`` ⇒ uniform.
    """

    def __init__(self, waveform: Callable[[float], float], *,
                 axis: str, position: float,
                 profiles: Mapping[str, np.ndarray] | None = None) -> None:
        super().__init__(waveform)
        self.axis = axis
        self.position = position
        self.profiles = profiles

    def spatial_profiles(self, grid: FDTDGrid) -> Dict[str, np.ndarray]:
        """Place each 2D transverse profile onto the slice at ``position``.

        A uniform plane wave (``profiles=None``) is not implemented yet; the
        port-mode path (``profiles`` given, a ``{component: 2D-array}`` transverse
        mode profile) maps each onto a full-grid weight array, nonzero only on the
        slice perpendicular to ``axis`` at the snapped cell.
        """
        if self.profiles is None:
            raise NotImplementedError(
                "PlaneSource uniform plane wave (profiles=None) is not "
                "implemented yet; pass transverse mode profiles.")

        k = grid.axis_index(self.axis, self.position)
        # Shape of the slice perpendicular to ``axis`` (same orientation as
        # SnapshotMonitor / the mode solver).
        if self.axis == 'z':
            slice_shape = (grid.Nx, grid.Ny)
        elif self.axis == 'y':
            slice_shape = (grid.Nx, grid.Nz)
        elif self.axis == 'x':
            slice_shape = (grid.Ny, grid.Nz)
        else:
            raise ValueError(f"axis must be 'x', 'y' or 'z', got {self.axis!r}")

        out: Dict[str, np.ndarray] = {}
        for comp, prof in self.profiles.items():
            prof = np.asarray(prof, dtype=np.float64)
            if prof.shape != slice_shape:
                raise ValueError(
                    f"PlaneSource profile for {comp!r} has shape {prof.shape}, "
                    f"which does not match the {self.axis}-slice shape "
                    f"{slice_shape}.")
            full = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=np.float64)
            if self.axis == 'z':
                full[:, :, k] = prof
            elif self.axis == 'y':
                full[:, k, :] = prof
            else:
                full[k, :, :] = prof
            out[comp] = full
        return out


# ====================================================================== #
# Plane-wave / full-slice directional launch
# ====================================================================== #

# Per boundary face: the propagation normal, which side of the box it is, and the
# ordered transverse pair (a, b). The pair is chosen so **a × b = the inward
# propagation direction**, which makes (â, b̂, n̂) right-handed on *every* face —
# so the launch's magnetic field ``H = (n̂ × E)/η`` needs no per-face sign table.
#
# The price of that uniformity: the same physical polarization takes a DIFFERENT
# ``angle`` value on opposite faces. E.g. a wave polarized along +z is angle=90°
# on x0 (pair y→z) but angle=0° on x1 (pair z→y). This is deliberate — read the
# angle as "measured from â towards b̂ in this face's own right-handed frame".
_FACE_CFG = {
    'x0': dict(normal='x', side='low',  a='y', b='z'),   # y × z = +x  (into +x)
    'x1': dict(normal='x', side='high', a='z', b='y'),   # z × y = -x  (into -x)
    'y0': dict(normal='y', side='low',  a='z', b='x'),   # z × x = +y
    'y1': dict(normal='y', side='high', a='x', b='z'),   # x × z = -y
    'z0': dict(normal='z', side='low',  a='x', b='y'),   # x × y = +z
    'z1': dict(normal='z', side='high', a='y', b='x'),   # y × x = -z
}


def _plane_slice(arr: np.ndarray, normal: str, k: int) -> np.ndarray:
    """The 2D plane of ``arr`` perpendicular to ``normal`` at cell ``k``
    (same orientation as :mod:`wavesim.monitors` / the mode solver)."""
    if normal == 'z':
        return arr[:, :, k]
    if normal == 'y':
        return arr[:, k, :]
    return arr[k, :, :]


# The two array axes of a ``_plane_slice`` perpendicular to each normal, in the
# slice's own order (axis 0, axis 1) — see :func:`_plane_slice`.
_TRANSVERSE_AXES = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}


def _cell_centers(nodes: np.ndarray) -> np.ndarray:
    """Cell-centre coordinates (length N) from the N+1 node coordinates — the
    positions the cell-centred field/material slices actually sit on."""
    return 0.5 * (nodes[:-1] + nodes[1:])


def _gaussian_aperture(coord0: np.ndarray, coord1: np.ndarray,
                       d_pml: int, waist: float) -> np.ndarray:
    """2D transverse Gaussian apodization, hard-zeroed over the PML cells.

    Returns ``exp(-r²/w₀²)`` on the transverse plane, centred on the plane and
    then set to zero over the ``d_pml`` outermost cells at each transverse edge.
    ``coord0``/``coord1`` are the physical cell-centre coordinates (metres) of the
    plane's two array axes and ``waist`` is w₀, the 1/e field radius.

    A flat-phase sheet apodized this way launches a Gaussian beam whose waist w₀
    sits at the launch plane. Zeroing the PML cells also stops a DC-containing
    waveform from accumulating without bound in the corner where the sheet meets
    the transverse absorber — choose ``waist`` small enough that the beam is
    already negligible there and the hard cut adds no diffraction of its own.
    """
    mid0 = 0.5 * (coord0[0] + coord0[-1])
    mid1 = 0.5 * (coord1[0] + coord1[-1])
    r2 = ((coord0 - mid0) ** 2)[:, None] + ((coord1 - mid1) ** 2)[None, :]
    w = np.exp(-r2 / (waist * waist))
    for axis, coord in enumerate((coord0, coord1)):
        n = coord.size
        if n - 2 * d_pml <= 0:             # no interior — leave the Gaussian
            continue
        keep = np.zeros(n, dtype=bool)
        keep[d_pml:n - d_pml] = True
        w = w * (keep[:, None] if axis == 0 else keep[None, :])
    return w


class _PlaneLaunch(Source):
    """Full-slice E (and, when directional, paired H) launch with the corrected
    co-indexed H time shift. The engine behind :class:`GaussianBeam`. It writes the
    profiles straight into the field arrays each step, so — unlike the calibrated
    current kernel a :class:`TEMPort` / :meth:`TEMMode.to_source` uses — the
    launched amplitude is *not* calibrated (it scales as ≈ 1/S_n × the waveform).

    Both sheets sit on the *same* slice (index ``k``). E is driven by
    ``waveform(t)``; the directional H sheet by ``waveform(t + τ)`` with

        τ = dt/2 + p · dn/(2·v_num)

    where ``p = +1`` for a launch into +normal (low faces / a +normal mode) and
    ``p = -1`` into -normal (high faces). ``dt/2`` undoes the leapfrog stagger and
    ``dn/(2·v_num)`` the half-cell that the co-indexed H sheet sits ahead of E
    along +normal (``dn`` = the normal cell width, ``v_num`` the numerical phase
    velocity). Because the correction is a *positive* shift — H sampled in the
    future — it can only be built from an analytic waveform, which is exactly why
    a circuit-driven port (:meth:`TEMMode.build_port_kernel`) instead puts its H
    sheet one cell behind and lags it. The two are the same launch shifted by one
    cell (``+dn/v``): both reject the backward wave to ≈ -150 dB on a 1D test.

    Subclasses supply the geometry lazily (the grid is only seen at first
    ``inject``): ``_plane_index(grid)`` and ``_transverse_profiles(grid)``.
    """

    def __init__(self, waveform: Callable[[float], float], *,
                 normal: str, directional: bool, v_medium: float,
                 prop_sign: float = 1.0,
                 e_profiles: Mapping[str, np.ndarray] | None = None,
                 h_profiles: Mapping[str, np.ndarray] | None = None,
                 position: float | None = None) -> None:
        super().__init__(waveform)
        self.normal = normal
        self.directional = bool(directional)
        self.v_medium = float(v_medium)
        self.prop_sign = float(prop_sign)
        self.position = position
        self._e2d = e_profiles
        self._h2d = h_profiles
        self._e_full: Dict[str, np.ndarray] | None = None
        self._h_full: Dict[str, np.ndarray] = {}
        self._tau = 0.0

    # --- geometry hooks (overridable) ---------------------------------- #
    def _plane_index(self, grid: FDTDGrid) -> int:
        return grid.axis_index(self.normal, self.position)

    def _transverse_profiles(self, grid: FDTDGrid):
        """Return ``(E2d, H2d)`` transverse-plane profile dicts."""
        return dict(self._e2d or {}), dict(self._h2d or {})

    # --- lazy build ---------------------------------------------------- #
    def _embed(self, grid: FDTDGrid, k: int, prof2d: np.ndarray) -> np.ndarray:
        full = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=np.float64)
        if self.normal == 'z':
            full[:, :, k] = prof2d
        elif self.normal == 'y':
            full[:, k, :] = prof2d
        else:
            full[k, :, :] = prof2d
        return full

    def _build(self, grid: FDTDGrid) -> None:
        k = self._plane_index(grid)
        E2d, H2d = self._transverse_profiles(grid)
        self._e_full = {c: self._embed(grid, k, np.asarray(p, np.float64))
                        for c, p in E2d.items()}
        if self.directional and H2d:
            self._h_full = {c: self._embed(grid, k, np.asarray(p, np.float64))
                            for c, p in H2d.items()}
            dn = float({'x': grid.dxp, 'y': grid.dyp,
                        'z': grid.dzp}[self.normal][k])
            from wavesim.mode_solver import numerical_velocity
            freq = getattr(self.waveform, 'center_frequency', None)
            v_num = numerical_velocity(self.v_medium, dn, grid.dt, freq)
            self._tau = grid.dt / 2.0 + self.prop_sign * dn / (2.0 * v_num)
        else:
            self._h_full = {}
            self._tau = 0.0

    def spatial_profiles(self, grid: FDTDGrid) -> Dict[str, np.ndarray]:
        """Full-grid E (and H) weight arrays, for inspection. ``inject`` drives
        the two sheets at different times, so it does not use this directly."""
        if self._e_full is None:
            self._build(grid)
        return {**self._e_full, **self._h_full}

    def inject(self, grid: FDTDGrid, t: float) -> None:
        if self._e_full is None:
            self._build(grid)
        ae = self.waveform(t)
        for comp, prof in self._e_full.items():
            getattr(grid, comp)[...] += ae * prof
        if self._h_full:
            ah = self.waveform(t + self._tau)
            for comp, prof in self._h_full.items():
                getattr(grid, comp)[...] += ah * prof


class GaussianBeam(_PlaneLaunch):
    """A directional Gaussian beam launched from one boundary face.

    Drives a cross-section one PML-depth inside a boundary face, biased into the
    domain: an E sheet plus the paired ``H = (n̂ × E)/η`` sheet
    (:class:`_PlaneLaunch`). The transverse amplitude is a Gaussian
    ``exp(-r²/w₀²)`` centred on the face; because the sheet is driven with a flat
    phase front, this launches a Gaussian beam whose **waist w₀ sits at the launch
    plane** and then diverges downstream by the usual Gaussian-beam laws. Taking
    ``waist`` large relative to the aperture recovers a (finite-aperture) plane
    wave. The waveform carries the amplitude — there is deliberately no
    ``amplitude`` parameter, as for every other source. The launched field is
    *not* amplitude-calibrated (its peak scales as ≈ ``1/S_n`` × the waveform,
    ``S_n`` the Courant number along the normal); use a monitor to normalise if
    you need an absolute level.

    The sheet is always zeroed over the transverse PML slabs. This matters for a
    DC-containing waveform (e.g. a unipolar :class:`GaussianPulse`): a sheet that
    overlapped the transverse absorber would inject a DC bias into the corner
    cells there, which neither propagate nor absorb DC, so the field would grow
    without bound and swamp the energy monitor. Keep ``waist`` comfortably smaller
    than the interior half-width so the beam is already negligible at that edge
    and the hard cut adds no diffraction of its own.

    Parameters
    ----------
    face : str
        Boundary face to launch from — one of ``'x0','x1','y0','y1','z0','z1'``
        (``'x0'`` = the low-x face, propagating into +x; ``'x1'`` = high-x, into
        -x; etc.). The wave propagates *into* the domain.
    angle : float
        Polarization angle (radians) of E, measured from the face's first
        transverse axis ``â`` towards its second ``b̂``: ``E ∝ cos(angle)·â +
        sin(angle)·b̂``. The (a, b) pair is right-handed with the propagation
        normal (see ``_FACE_CFG``), so the SAME physical polarization needs a
        DIFFERENT ``angle`` on opposite faces — e.g. +z-polarized light is 90° on
        x0 but 0° on x1.
    waveform : Callable[[float], float]
        Time function (e.g. a :class:`Sinusoid` or :class:`GaussianPulse`). A
        waveform advertising a ``center_frequency`` tunes the H time shift to the
        numerical phase velocity at that frequency; otherwise the continuum
        velocity is used.
    waist : float
        Beam waist w₀ in metres — the 1/e radius of the transverse E amplitude
        (1/e² in intensity), located at the launch plane. Centred on the face.
    d_pml : int
        PML thickness in cells (default 10, matching :func:`init_cpml`). The E
        sheet is placed on the first interior cell — index ``d_pml`` on a low
        face, ``N-1-d_pml`` on a high face — so the backward lobe is launched
        straight into the absorber, and the same depth is zeroed off every
        transverse edge.
    directional : bool
        Pair the E sheet with an H sheet for a one-way launch (default True).
        ``False`` gives a bare E sheet, which radiates symmetrically both ways.

    Notes
    -----
    There are no periodic/Bloch boundaries. A beam whose waist approaches the
    transverse aperture will diffract off the finite, PML-masked edge; keep the
    waist well inside it.
    """

    def __init__(self, face: str, angle: float,
                 waveform: Callable[[float], float], waist: float, *,
                 d_pml: int = 10, directional: bool = True) -> None:
        if face not in _FACE_CFG:
            raise ValueError(
                f"face must be one of {sorted(_FACE_CFG)}, got {face!r}.")
        if not waist > 0:
            raise ValueError(f"waist must be positive, got {waist!r}.")
        cfg = _FACE_CFG[face]
        super().__init__(waveform, normal=cfg['normal'], directional=directional,
                         v_medium=C0,
                         prop_sign=(1.0 if cfg['side'] == 'low' else -1.0))
        self.face = face
        self.angle = float(angle)
        self.waist = float(waist)
        self.d_pml = int(d_pml)

    def _plane_index(self, grid: FDTDGrid) -> int:
        N = {'x': grid.Nx, 'y': grid.Ny, 'z': grid.Nz}[self.normal]
        if _FACE_CFG[self.face]['side'] == 'low':
            return self.d_pml
        return N - 1 - self.d_pml

    def _transverse_profiles(self, grid: FDTDGrid):
        cfg = _FACE_CFG[self.face]
        a_ax, b_ax = cfg['a'], cfg['b']
        k = self._plane_index(grid)

        # Local wave impedance per transverse cell. The (E_b, H_a) pair carries
        # power along n̂, as does (E_a, H_b); each uses η = η₀·√(μ/ε) built from
        # the permeability the H component sees and the permittivity its partner
        # E component sees. Uniform (vacuum) grids give η₀ everywhere.
        eps_a = _plane_slice(getattr(grid, 'eps_' + a_ax), self.normal, k)
        eps_b = _plane_slice(getattr(grid, 'eps_' + b_ax), self.normal, k)
        mu_a = _plane_slice(getattr(grid, 'mu_' + a_ax), self.normal, k)
        mu_b = _plane_slice(getattr(grid, 'mu_' + b_ax), self.normal, k)
        eta_a = ETA0 * np.sqrt(mu_a / np.where(eps_b > 0, eps_b, 1.0))
        eta_b = ETA0 * np.sqrt(mu_b / np.where(eps_a > 0, eps_a, 1.0))

        ca, sa = np.cos(self.angle), np.sin(self.angle)
        # Transverse Gaussian apodization, zeroed over the PML slabs. Built on the
        # plane's own two array axes (see _TRANSVERSE_AXES / _plane_slice) so it
        # lines up with the eps/mu slices above.
        ax0, ax1 = _TRANSVERSE_AXES[self.normal]
        c0, c1 = (_cell_centers(grid._coords(ax0)),
                  _cell_centers(grid._coords(ax1)))
        w = _gaussian_aperture(c0, c1, self.d_pml, self.waist)
        # E = cos·â + sin·b̂;  H = (n̂ × E)/η = (cos·b̂ - sin·â)/η
        E = {'E' + a_ax: ca * w, 'E' + b_ax: sa * w}
        H = {'H' + a_ax: -sa * w / eta_a, 'H' + b_ax: ca * w / eta_b}
        return E, H


class LineSource(Source):
    """
    Lumped V-I-Z element on a straight line between two endpoints.

    The line from ``p0`` to ``p1`` is rasterised onto Yee E-edges with the same
    quadrature as :class:`~wavesim.monitors.VoltageMonitor`, so the element's
    port voltage V(t) = ∫E·dl (p0 → p1) is exactly what a VoltageMonitor on the
    same path reads. ``p0`` is the "+" terminal; positive port current I(t) is
    delivered out of ``p0`` into the surrounding structure.

    The **load** is any one of ``resistance`` / ``inductance`` / ``capacitance``,
    or several of them wired ``topology='series'`` (default) or ``'parallel'``
    — the lumped R, L, C and their combinations, without needing the ngspice
    round trip of a :class:`SpicePort`. Exactly one of ``voltage`` / ``current``
    selects the drive, and composes with the load (or omit both, for a purely
    passive element):

    ==========================  =============================================
    Arguments                   Element
    ==========================  =============================================
    ``voltage=Vs``              Ideal voltage source — pins ∫E·dl = Vs(t)
                                each step (a *hard* write; reflects incident
                                waves, as a zero-impedance source must).
    ``voltage=Vs`` + load       Thevenin source: V = Vs(t) − V_load, the load
                                in series with the EMF.
    ``current=Is``              Ideal current source — soft impressed current
                                Is(t) along the line.
    ``current=Is`` + load       Norton source: the load sits in parallel with
                                the impressed current.
    load only                   Passive lumped element: a resistor
                                ``I = −V/R`` (e.g. a matched termination), a
                                capacitor, an inductor, or a network of them.
    ==========================  =============================================

    Reactive branches are integrated with the trapezoidal companion model —
    each becomes a resistance plus a history source built from its own previous
    ``(I, V)`` sample, so the per-step law keeps the single-solve form below
    (see :mod:`wavesim.lumped`). Every branch resistance is positive and finite,
    so no value of L or C imposes a timestep limit of its own.

    Unlike the static sources above, the injection depends on the local field
    each step (an impedance/feedback relationship), so ``inject`` is overridden.
    An impressed current I spread along the line adds
    ``E_a += dt · I · dl_a / (ε_a · dV_cell)`` on each occupied edge, which
    changes the port voltage by ``κ·I`` with ``κ = Σ dt·dl²/(ε·dV)`` (the line's
    self-coupling, ohm-like). Loaded modes use the semi-implicit current

        I = (Vs(t) + V_hist − (Vⁿ + Vⁿ⁺¹)/2) / Z_eq    (Norton: Vs ≡ Z_eq·Is)

    solved for the injected I as ``(Vs + V_hist − (Vⁿ + V*)/2)/(Z_eq + κ/2)``,
    where ``V*`` is the just-curl-updated line voltage and ``Vⁿ`` the voltage at
    the end of the previous step. ``(Z_eq, V_hist)`` is the load's companion
    pair for this step — for a plain resistor ``(R, 0)``, recovering the
    familiar ``I = (Vs − V)/R``. This is the standard Piket-May semi-implicit
    lumped element, time-centred across the whole step, and is stable for any
    Z_eq > 0. (Centring on V* alone — or the naive explicit I = (Vs−V)/Z —
    couples unstably with the leapfrog update once Z is below a few hundred
    ohms.)

    The element self-records its port quantities each step — ``times`` (s),
    ``voltages`` (V, post-injection, what a co-located VoltageMonitor reads) and
    ``currents`` (A, the injected impressed current; in ideal-voltage mode the
    equivalent current (Vs − V_before)/κ that produces the imposed field change)
    — so it doubles as a port for impedance / S-parameter extraction.

    Two discretisation caveats, both standard for FDTD lumped elements:

    * **The cell is bridged, not replaced.** To the surrounding field the element
      presents exactly ``Z_eq`` — the κ/2 in the solve below is the stability term
      of the implicit averaging and does *not* appear in the presented impedance
      (:mod:`tests.test_lumped_element_impedance` measures it spectrally: a 25 Ω
      resistor reads 25 Ω, not ``R + κ/2`` = 58 Ω). What the element does *not*
      do is remove its own Yee cell: the cell's gap capacitance
      ``C_cell = dt/κ = ε·dA/dl`` sits in parallel, as it does with or without the
      element, so the total across the gap is ``Z_eq ‖ C_cell``. For a component
      bridging a modelled gap that is the physical answer, since the gap has that
      capacitance in reality too. For a sub-cell component whose value already
      accounts for its own body it is not, and the total then moves with the mesh
      (``C_cell ∝ dA/dl``); refine the port cell transversely to shrink it.
      The recorded V(t)/I(t) are exact either way, so port extraction is
      unaffected.
    * **Co-located elements.** Elements sharing line edges inject
      sequentially, not as a jointly solved circuit, so each contributes its
      own κ/2 in series (a 2-element voltage divider on one line settles to
      ``Vs·Z_L/(Z + Z_L + κ)``). Use one element with a ``topology=`` network
      instead — those branches *are* solved jointly.

    The line typically spans the gap between two conductors; endpoints may sit
    just inside PEC (as with the monitors), but keep the driven gap itself in
    dielectric — E on PEC edges is zeroed every step.

    Parameters
    ----------
    p0, p1 : tuple of float
        Endpoints ``(x, y, z)`` in metres; ``p0`` is the "+" terminal. Any
        orientation (oblique lines are split per-axis onto staggered edges).
    voltage : Callable[[float], float], optional
        Source voltage Vs(t) in volts (Thevenin open-circuit value when a load
        is given).
    current : Callable[[float], float], optional
        Source current Is(t) in amperes (Norton short-circuit value when a load
        is given). Mutually exclusive with ``voltage``.
    resistance : float, optional
        Resistance R in ohms (> 0).
    inductance : float, optional
        Inductance L in henries (> 0).
    capacitance : float, optional
        Capacitance C in farads (> 0).
    topology : {'series', 'parallel'}
        How several of R/L/C are wired between the two terminals; irrelevant
        with a single one. Default ``'series'``.
    """

    #: Fixed series impedance presented by subclasses that bypass this
    #: ``__init__`` (a mode's Z₀, or None for an ideal impressed source).
    #: :class:`LineSource` itself uses ``_element`` instead.
    impedance: float | None = None
    _element: LumpedNetwork | None = None

    def __init__(self, *,
                 p0: Tuple[float, float, float], p1: Tuple[float, float, float],
                 voltage: Callable[[float], float] | None = None,
                 current: Callable[[float], float] | None = None,
                 resistance: float | None = None,
                 inductance: float | None = None,
                 capacitance: float | None = None,
                 topology: str = 'series') -> None:
        if voltage is not None and current is not None:
            raise ValueError(
                "LineSource takes either voltage= or current=, not both.")
        has_load = (resistance is not None or inductance is not None
                    or capacitance is not None)
        if voltage is None and current is None and not has_load:
            raise ValueError(
                "LineSource needs a drive (voltage= or current=) and/or a load "
                "(resistance=, inductance=, capacitance=; a load alone gives a "
                "passive lumped element).")
        drive = voltage if voltage is not None else current
        super().__init__(drive if drive is not None else (lambda t: 0.0))
        self.p0 = tuple(p0)
        self.p1 = tuple(p1)
        self.voltage = voltage
        self.current = current
        self.resistance = resistance
        self.inductance = inductance
        self.capacitance = capacitance
        # The load, as a per-step (Z_eq, V_hist) companion pair; None for an
        # ideal (unloaded) source, which takes the hard-write / impressed-
        # current paths in ``inject``. Value validation lives in LumpedNetwork.
        self._element = (
            LumpedNetwork(resistance=resistance, inductance=inductance,
                          capacitance=capacitance, topology=topology)
            if has_load else None)
        self.topology = topology
        # Port record (see class docstring).
        self.times: list = []
        self.voltages: list = []
        self.currents: list = []
        self._port: dict | None = None      # quadrature + coefficients, built once
        self._v_prev = 0.0                  # port V at end of previous step (Vⁿ)
        self._h_lag_steps = 0.0             # directional H-sheet shift (see below)

    def _lagged_current(self, lag_steps: float) -> float:
        """Port current ``lag_steps`` (in units of dt, ≥ 0) into the past.

        A directional launch's H sheet must be driven by the incident wave as
        sampled at *its* place and time, which trails the E sheet's by a fraction
        of a step (:meth:`~wavesim.mode_solver.TEMMode.build_port_kernel`). The
        port current is only known implicitly at the present step, so the shift
        is built by interpolating the recorded history — which is exactly why the
        H sheet is placed behind the E plane rather than ahead, making the
        required shift a lag instead of a lead. Call this only after the present
        step's current has been appended; values from before the run read zero.
        """
        hist = self.currents
        if not hist:
            return 0.0
        if lag_steps <= 0.0:
            return hist[-1]
        whole = int(np.floor(lag_steps))
        frac = lag_steps - whole
        near = hist[-1 - whole] if len(hist) > whole else 0.0
        far = hist[-2 - whole] if len(hist) > whole + 1 else 0.0
        return (1.0 - frac) * near + frac * far

    # ------------------------------------------------------------------ #
    # Geometry compilation (once per grid)
    # ------------------------------------------------------------------ #
    def _build_port(self, grid: FDTDGrid) -> dict:
        """Compile the line into per-edge quadrature and injection coefficients.

        For each occupied edge: ``w`` is its path length dl_a (metres, signed),
        shared with VoltageMonitor; ``coef = dt·w/(ε·dV)`` is the E-change per
        unit impressed current. ``kappa = Σ w·coef`` is then the port-voltage
        change per unit current, and ``wsq = Σ w²`` normalises the hard
        (ideal-voltage) write ``E_a = Vs·w_a/wsq``.

        ``dV`` is the **local Yee cell volume at each edge** — the product of the
        primary cell widths at that index (``dxp[i]·dyp[j]·dzp[k]``), matching the
        all-primary divisors of :func:`wavesim.update.update_E`. On a uniform grid
        this is the constant ``dx*dy*dz``; on a rectilinear grid it varies per
        edge, so κ and the injection stay physically correct. ``wsq`` is purely
        geometric (physical lengths) and is unchanged.
        """
        quad = _build_path_quadrature([self.p0, self.p1], grid, 'E', close=False)
        # The ε the E update will actually divide by, which on a conformal grid
        # is not the stored array (:func:`wavesim.pec.conformal_edge_eps`); κ is
        # this source's model *of* that update and has to track it.
        eps_of = dict(zip(('Ex', 'Ey', 'Ez'), conformal_edge_eps(grid)))
        edges = {}
        kappa = 0.0
        wsq = 0.0
        for comp, (ii, jj, kk, w) in quad.items():
            dV = grid.dxp[ii] * grid.dyp[jj] * grid.dzp[kk]   # per-edge local volume
            coef = grid.dt * w / (EPS0 * eps_of[comp][ii, jj, kk] * dV)
            edges[comp] = (ii, jj, kk, w, coef)
            kappa += float(np.dot(w, coef))
            wsq += float(np.dot(w, w))
        return {'edges': edges, 'kappa': kappa, 'wsq': wsq}

    def self_coupling(self, grid: FDTDGrid) -> float:
        """κ in ohms: the port-voltage change per unit injected current per
        step, ``Σ dt·dl²/(ε·dV)`` over the line's edges.

        Equivalently ``κ = dt/C_cell``: it *is* the port cell's own gap
        capacitance, in the units the update works in, and ``κ/2 = dt/(2·C_cell)``
        is that capacitance's trapezoidal companion resistance. It is a stability
        term of the semi-implicit solve, not a series parasitic — the element
        presents ``Z_eq`` to the field, with ``C_cell`` in parallel as background
        (see the class docstring)."""
        if self._port is None:
            self._port = self._build_port(grid)
        return self._port['kappa']

    def spatial_profiles(self, grid: FDTDGrid) -> Dict[str, np.ndarray]:
        """Geometric footprint for inspection: full-grid arrays holding each
        occupied edge's path length dl_a (metres). ``inject`` does not use
        this — the injection is field-dependent."""
        if self._port is None:
            self._port = self._build_port(grid)
        out: Dict[str, np.ndarray] = {}
        for comp, (ii, jj, kk, w, _coef) in self._port['edges'].items():
            full = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=np.float64)
            full[ii, jj, kk] = w
            out[comp] = full
        return out

    # ------------------------------------------------------------------ #
    # Per-step injection
    # ------------------------------------------------------------------ #
    def inject(self, grid: FDTDGrid, t: float) -> None:
        if self._port is None:
            self._port = self._build_port(grid)
        edges = self._port['edges']
        kappa = self._port['kappa']
        elem = self._element
        if elem is None:
            # No R/L/C network: either an ideal source, or a subclass that
            # bypasses __init__ and presents a fixed series impedance of its
            # own (a mode's Z₀; None for a pure impressed source).
            Z, v_hist = self.impedance, 0.0
        else:
            # The load, linearised over this step: V_load = −Z·I + V_hist.
            # Reactive branches hide their memory in V_hist, so the solve below
            # keeps the plain resistive form (see :mod:`wavesim.lumped`).
            Z, v_hist = elem.companion(grid.dt)

        # Port voltage before injection: V = Σ E·dl (p0 → p1).
        v_before = 0.0
        for comp, (ii, jj, kk, w, _coef) in edges.items():
            v_before += float(np.dot(getattr(grid, comp)[ii, jj, kk], w))

        if self.voltage is not None and Z is None:
            # Ideal voltage source: hard-set the line edges so ∫E·dl = Vs(t).
            vs = self.waveform(t)
            wsq = self._port['wsq']
            for comp, (ii, jj, kk, w, _coef) in edges.items():
                getattr(grid, comp)[ii, jj, kk] = vs * w / wsq
            v_after = vs
            i_port = (vs - v_before) / kappa    # equivalent impressed current
        else:
            # Time-centred circuit law: the "old" voltage is Vⁿ from the end of
            # the previous step (the line edges are untouched between then and
            # this step's curl update), the "new" is v_before + κ·I.
            v_mid = 0.5 * (self._v_prev + v_before)
            drive = self.waveform(t)
            if self.voltage is not None:        # Thevenin: load in series
                i_port = (drive + v_hist - v_mid) / (Z + 0.5 * kappa)
            elif Z is None:                     # ideal current source
                i_port = drive
            else:                               # Norton: load in parallel
                i_port = (Z * drive + v_hist - v_mid) / (Z + 0.5 * kappa)
            for comp, (ii, jj, kk, w, coef) in edges.items():
                getattr(grid, comp)[ii, jj, kk] += coef * i_port
            v_after = v_before + kappa * i_port

            if elem is not None:
                # Advance the load's integrator with this step's solved sample.
                # What the *load* sees is not always the port pair: a Norton
                # drive puts its source current in parallel with the load, a
                # Thevenin drive its EMF in series with it. The mid-step port
                # voltage is v_mid + κ·I/2 — i.e. (Vⁿ + Vⁿ⁺¹)/2, the very
                # quantity the law above was centred on.
                v_port_mid = v_mid + 0.5 * kappa * i_port
                if self.voltage is not None:
                    elem.update(grid.dt, i_port, v_port_mid - drive)
                else:
                    elem.update(grid.dt, i_port - drive, v_port_mid)

        self._v_prev = v_after
        self.times.append(t)
        self.voltages.append(v_after)
        self.currents.append(i_port)

    def _inject_directional_h(self, grid: FDTDGrid) -> None:
        """Drive the paired directional H sheet, if the port has one.

        Shared by :class:`TEMPort` and :class:`_ModalLaunch`. The sheet is
        driven by the port current sampled at *its* space-time point, which
        trails the E plane by ``_h_lag_steps`` (built from the kernel's
        ``h_tau``); the shift is read out of the recorded current history, so
        this must be called only after the present step's current has been
        appended (see :meth:`_lagged_current`)."""
        hedges = self._port.get('hedges') if self._port else None
        if not hedges:
            return
        i_port = self._lagged_current(self._h_lag_steps)
        for comp, (ii, jj, kk, coefH) in hedges.items():
            getattr(grid, comp)[ii, jj, kk] += coefH * i_port


class TEMPort(LineSource):
    """Distributed TEM-mode port: a Thévenin ``(Vs, Z₀)`` drive of a solved mode.

    Where :class:`LineSource` drives a straight p0→p1 line, a :class:`TEMPort`
    drives the frozen transverse profile of a
    :class:`~wavesim.mode_solver.TEMMode`. The mode is solved once; each step the
    port reads the modal voltage (an ε-weighted overlap projection of the plane
    field onto the mode), runs the same time-centred (Piket-May) circuit law with
    series impedance ``Z₀`` (the mode's characteristic impedance by default), and
    injects the resulting impressed current back over the whole profile —
    launching / terminating the mode. See
    :meth:`~wavesim.mode_solver.TEMMode.build_port_kernel`.

    The port presents its internal series resistance ``z_int = Z₀`` to the field
    in both roles — as a *terminator* (a wave arriving sees ``z_int``) and as a
    *source* (its own launched wave divides by ``z_int``) — so a matched line sees
    a matched source with no κ correction. The semi-implicit denominator carries a
    ``κ/2`` stability term, but it self-cancels for smooth excitation and does not
    appear in the presented impedance; a spectral mid-line reflection sweep puts
    the matched (``Γ = −1/3``) value at ``z_int = Z₀`` to within a few percent.
    (An earlier revision pre-compensated to ``Z₀ − κ/2`` on the theory that an
    arriving wave sees ``z_int + κ/2``; a clean spectral measurement did not bear
    that out and the compensation *under*-matched the terminator by ~κ/2.)

    ``voltage`` is the launched **forward-wave** voltage, not the raw Thévenin
    EMF: ``V_fwd = Vs·Z_load/(z_int + Z_load)`` where ``Z_load`` is what the source
    drives into — the one-way line ``Z₀`` (directional) or the two halves in
    parallel ``Z₀/2`` (bidirectional). The port drives its EMF at the reciprocal
    ``(z_int + Z_load)/Z_load`` so ``voltage(t)`` lands on the line — with
    ``z_int = Z₀`` the clean, geometry-independent factors 2 (directional) and 3
    (bidirectional) — matching
    :meth:`~wavesim.mode_solver.TEMMode.to_source`'s ``amplitude`` convention.

    With ``directional=True`` (default) the port also drives a paired H sheet,
    biasing energy into +normal. That sheet sits one cell *behind* the E plane and
    is driven by the port current lagged onto its own space-time sample point,
    which is what makes the backward wave cancel rather than merely shrink — see
    :meth:`~wavesim.mode_solver.TEMMode.build_port_kernel`. Measured backward
    rejection on a driven coax: ≈ -30 dB with the sheets naively co-indexed and
    unlagged, ≈ -48 dB corrected. A passive matched termination (no drive) is
    usually best left bidirectional (``directional=False``).

    Driving with a waveform that advertises a ``center_frequency`` (e.g.
    :class:`Sinusoid`) tunes the lag to the numerical phase velocity at that
    frequency; a broadband drive falls back to the continuum velocity, which costs
    little — the lag varies only ~3% over a 4× frequency range.

    Parameters
    ----------
    mode : TEMMode
        A mode from :func:`~wavesim.mode_solver.solve_tem_modes` (solve with
        ``compute_params=True`` for its ``impedance``/Z₀).
    voltage, current : Callable[[float], float], optional
        Drive (mutually exclusive); omit both for a passive matched termination.
        ``voltage`` is the launched forward-wave voltage (see above), amplitude-
        calibrated like ``to_source``. ``current`` is the raw Norton ``Is(t)``.
    impedance : float, optional
        Series/source impedance in ohms; defaults to the mode's ``Z₀``. Present
        this = the line's Z₀ for a matched port (the default already does).
    directional : bool
        Also drive the H sheet for a one-way launch (default True).
    """

    def __init__(self, *, mode,
                 voltage: Callable[[float], float] | None = None,
                 current: Callable[[float], float] | None = None,
                 impedance: float | None = None,
                 directional: bool = True) -> None:
        if voltage is not None and current is not None:
            raise ValueError(
                "TEMPort takes either voltage= or current=, not both.")
        z0 = impedance if impedance is not None else getattr(mode, 'impedance', None)
        if z0 is None or not z0 > 0:
            raise ValueError(
                "TEMPort needs a positive impedance: the mode has no Z₀ (solve "
                "with compute_params=True) or pass impedance= explicitly.")
        # ``voltage`` is the launched forward-wave voltage, not the raw Thévenin
        # EMF; the EMF is scaled up by the launch-divider reciprocal (2 directional
        # / 3 bidirectional) so it lands ``voltage`` volts forward. The scaling is
        # applied in _build_port alongside the impedance; until then ``waveform``
        # holds the raw drive.
        drive = voltage if voltage is not None else current
        Source.__init__(self, drive if drive is not None else (lambda t: 0.0))
        self.mode = mode
        self.voltage = voltage
        self.current = current
        self._z0 = float(z0)
        self.directional = bool(directional)
        self.impedance = None       # finalised in _build_port (= Z₀)
        self.p0 = self.p1 = None    # not a straight-line port
        self.times: list = []
        self.voltages: list = []
        self.currents: list = []
        self._port: dict | None = None
        self._v_prev = 0.0

    def _build_port(self, grid: FDTDGrid) -> dict:
        # A Sinusoid (or any waveform advertising a spectral centre) lets the
        # launch tune its H-sheet shift to the numerical phase velocity at that
        # frequency; a broadband drive falls back to the continuum velocity.
        drive = self.voltage if self.voltage is not None else self.current
        freq = getattr(drive, 'center_frequency', None)
        kernel = self.mode.build_port_kernel(
            grid, directional=self.directional, frequency=freq)
        self._h_lag_steps = -kernel.get('h_tau', 0.0) / grid.dt
        # The port presents its internal series resistance z_int = Z₀ to the field
        # in BOTH roles — as a terminator (a wave arriving sees z_int) and as a
        # source (its own launched wave divides by z_int). The semi-implicit
        # denominator's κ/2 is an internal stability term that self-cancels for
        # smooth excitation and does NOT appear in the presented impedance: a
        # spectral mid-line reflection sweep puts the matched (Γ = −1/3) value at
        # z_int = Z₀ to within a few %. So no κ/2 pre-compensation, and with
        # z_int = Z₀ > 0 the scheme is stable on any grid.
        self.impedance = self._z0
        z_int = self.impedance
        # Forward-volts calibration: the port's own launched wave divides by z_int,
        #   V_fwd = Vs·Z_load/(z_int + Z_load),
        # with Z_load = Z₀ (directional, one-way line) or Z₀/2 (bidirectional, the
        # two halves in parallel). Drive the EMF at the reciprocal so ``voltage``
        # lands forward; with z_int = Z₀ these are the clean factors 2 and 3.
        if self.voltage is not None:
            z_load = self._z0 if self.directional else 0.5 * self._z0
            scale = (z_int + z_load) / z_load
            raw = self.voltage
            self.waveform = lambda t, _r=raw, _s=scale: _s * _r(t)
        return kernel

    def inject(self, grid: FDTDGrid, t: float) -> None:
        # Modal V* read-back, Piket-May law, E-injection and recording are all
        # inherited from LineSource; only the paired directional H sheet is new.
        super().inject(grid, t)
        self._inject_directional_h(grid)


class _ModalLaunch(LineSource):
    """Amplitude-calibrated impressed launch of a solved TEM mode.

    The engine behind :meth:`~wavesim.mode_solver.TEMMode.to_source`. Unlike a
    :class:`TEMPort` — a *terminated* Thévenin/Norton port that also absorbs the
    returning wave — this is a pure soft source: it impresses the modal current
    that a matched line turns into the requested forward voltage, and reflects
    nothing.

    Calibration comes from the very same
    :meth:`~wavesim.mode_solver.TEMMode.build_port_kernel` current kernel a port
    uses (``E += κ·Ê·I``, the paired one-cell-behind, lagged ``H += κ·Ĥ·I``), so
    the launched amplitude is correct by construction on any grid or fill — the
    older additive field write (a straight ``E += waveform·Ê``) ignored the FDTD
    update coefficient ``dt/(ε·dV)`` and came out √ε_r / S_c too large.

    The forward voltage of a matched-line launch relates to the impressed modal
    current by ``V = I·Z₀`` for a directional (one-way) launch and ``V = I·Z₀/2``
    for a bidirectional one (the current splits both ways). So to put
    ``amplitude·waveform(t)`` volts into the forward wave the impressed current is
    ``amplitude·waveform(t)/Z₀`` directional, twice that bidirectional. Because
    the mode is normalised to a 1 V drive, ``amplitude`` is just that forward
    voltage in volts.

    Like a port it self-records ``times``/``voltages``/``currents`` (the injected
    impressed current and the resulting plane voltage), so it can double as a
    launch monitor.
    """

    def __init__(self, mode, waveform: Callable[[float], float], *,
                 amplitude: float = 1.0, directional: bool = True) -> None:
        z0 = getattr(mode, 'impedance', None)
        if z0 is None or not z0 > 0:
            raise ValueError(
                "TEMMode.to_source needs the mode's characteristic impedance to "
                "calibrate the launch amplitude; solve with compute_params=True.")
        # Impressed modal current I(t) = amplitude·waveform(t)·scale/Z₀ that a
        # matched line turns into amplitude·waveform(t) volts forward.
        scale = 1.0 if directional else 2.0
        current = lambda t: amplitude * scale * waveform(t) / z0
        # Bypass LineSource.__init__ (its p0/p1/load validation): this is an
        # ideal impressed *current* source (Z=None) on a modal footprint, not a
        # straight line. Mirror the state _build_port / inject need.
        Source.__init__(self, current)
        self.mode = mode
        self.base_waveform = waveform
        self.amplitude = float(amplitude)
        self.directional = bool(directional)
        self._z0 = float(z0)
        self.voltage = None
        self.current = current
        self.impedance = None       # ideal (soft) current source: no absorption
        self.p0 = self.p1 = None    # not a straight-line port
        self.times: list = []
        self.voltages: list = []
        self.currents: list = []
        self._port: dict | None = None
        self._v_prev = 0.0
        self._h_lag_steps = 0.0

    def _build_port(self, grid: FDTDGrid) -> dict:
        freq = getattr(self.base_waveform, 'center_frequency', None)
        kernel = self.mode.build_port_kernel(
            grid, directional=self.directional, frequency=freq)
        self._h_lag_steps = -kernel.get('h_tau', 0.0) / grid.dt
        return kernel

    def inject(self, grid: FDTDGrid, t: float) -> None:
        # Ideal-current E-injection, V read-back and recording come from
        # LineSource; the paired directional H sheet is added on top.
        super().inject(grid, t)
        self._inject_directional_h(grid)


# --------------------------------------------------------------------------- #
# Co-planar ModalPort bookkeeping
# --------------------------------------------------------------------------- #
# Every ModalPort *assigns* its sheet onto the ghost H plane, because the sheet
# must replace what ``update_H`` left there rather than add to it (``update_H``
# writes that plane too, in both the staircase and the conformal branch). With
# two ports on the same face — the multi-conductor S-parameter case — plain
# assignment in ``Simulation.step``'s boundary loop makes the last port in the
# list erase every earlier one: not a degradation but a total suppression, since
# co-planar modes generally span the same cells. The suppressed port then loses
# its *termination* as well as its drive, while still recording a plausible
# V(t)/I(t) (both are read-only projections of planes nobody clobbers) — which is
# why the failure reads as a quiet port rather than as an error.
#
# The fix is clear-then-sum across co-planar ports only: the plane carries
#
#     H_ghost = Σ_m s_m·(V̄_m − 2·a_m)·ĥ_m
#
# Ports sharing a write target find each other through ``_GHOST_GROUPS``, keyed
# on the grid and the plane actually written (``_h_k``, which is ``k`` for a high
# face and ``k−1`` for a low one). Each port drops its scalar amplitude into a
# per-step accumulator and then flushes the running sum, so the result does not
# depend on boundary-list order and is correct even on the first step, where a
# port may flush before a later one has run its lazy ``_setup``: whoever flushes
# last writes the full sum, and the E update only sees the end of the loop.
#
# The grid is a mutable dataclass — unhashable, so no WeakKeyDictionary — hence
# the id key plus a weakref used both to detect id reuse and to drop the entry
# when the grid dies.
_GHOST_GROUPS: dict = {}


class _GhostPlaneGroup:
    """The ModalPorts writing one ghost H plane, and their sum for this step."""

    __slots__ = ('t', 'contrib', 'plans')

    def __init__(self) -> None:
        self.t = None
        self.contrib: dict = {}    # port -> scalar sheet amplitude this step
        self.plans: dict = {}      # frozenset(ports) -> scatter plan

    def open_step(self, port, t: float) -> None:
        """Start a new step's accumulation if this is the first contribution.

        ``t`` advances every step, so a differing ``t`` marks a new step. A port
        contributing twice under the same ``t`` marks one too: each port applies
        once per step, so the repeat can only be a fresh run on a re-used grid
        whose step counter was reset.
        """
        if self.t != t or port in self.contrib:
            self.t = t
            self.contrib.clear()

    def plan(self, shape: Tuple[int, int, int]) -> dict:
        """Scatter plan for the current contributor set (built once, cached).

        Per component: the grid indices of the **union** of the contributors'
        nonzero cells, and each port's offsets into that union. Only the union is
        written — cells where every mode is zero keep whatever ``update_H`` put
        there, as they always have.
        """
        key = frozenset(self.contrib)
        plan = self.plans.get(key)
        if plan is not None:
            return plan
        ports = sorted(self.contrib, key=id)
        comps = {c for p in ports for c in p._h}
        plan = {}
        for comp in comps:
            members = [(p, np.ravel_multi_index(p._h[comp][:3], shape))
                       for p in ports if comp in p._h]
            union = np.unique(np.concatenate([f for _, f in members]))
            plan[comp] = (np.unravel_index(union, shape), union.size,
                          [(p, np.searchsorted(union, f)) for p, f in members])
        self.plans[key] = plan
        return plan


def _ghost_group(grid: FDTDGrid, normal: str, h_k: int) -> _GhostPlaneGroup:
    """The accumulator shared by every ModalPort writing this grid's ``h_k``
    plane along ``normal`` (created on first use)."""
    key = (id(grid), normal, h_k)
    entry = _GHOST_GROUPS.get(key)
    if entry is not None and entry[0]() is grid:
        return entry[1]
    ref = weakref.ref(grid, lambda _r, k=key: _GHOST_GROUPS.pop(k, None))
    group = _GhostPlaneGroup()
    _GHOST_GROUPS[key] = (ref, group)
    return group


class ModalPort:
    """One-way modal impedance-sheet port: a TEM absorber / launcher on a face.

    Where a :class:`TEMPort` is a *lumped, distributed* Thévenin drive injected on
    an interior plane, a ``ModalPort`` is an **impedance-sheet boundary** placed on
    a domain face. Each step it sets the ghost tangential H just outside the face
    to the value a matched continuation of the mode would carry, so the face
    absorbs the mode with **no reflection and no DC error** — replacing PML for a
    closed cross-section, and (unlike PML) exact at DC. With ``amplitude > 0`` the
    *same* sheet also launches the mode inward, so one object both drives and
    terminates a port (the CST "waveguide port" model).

    It is registered with :meth:`~wavesim.simulation.Simulation.add_boundary`, not
    ``add_source``: the sheet writes the ghost H that the *next* E-update consumes,
    so it must run **between** the H and E updates. A source hook (after the E
    update) would be clobbered by the following step's H update before it is ever
    read.

    The rule, per face cell, is

        ``H_ghost = ±s · Y₀ · (V̄ − 2a) · (n̂ × ê)``

    where ``ê`` is the 1 V-normalised staggered mode E-profile
    (:meth:`~wavesim.mode_solver.TEMMode._staggered_port_fields`), ``Y₀ = 1/η`` the
    local wave admittance, ``V̄`` the modal voltage (ε-weighted overlap
    projection, :meth:`build_port_kernel`) sampled at ``n + h_tau`` — the E↔H
    space-time offset ``dt/2 − dn/(2·v)`` the launch already applies to its H
    sheet, interpolated from the read-back history — ``a = amplitude·waveform(t)``
    the drive, and ``s`` the numerical-admittance correction ``admittance_scale``.
    Sampling ``V̄`` at the shifted instant rather than the naïve ``½(Vⁿ+Vⁿ⁻¹)``
    (n−½) removes an O(ω·dt) phase error and is the difference between a ~−25 dB
    and a ~−33 dB coax termination; it is exact at DC, so DC-exactness stands.

    The ``−2a`` term makes the sheet radiate a
    forward wave of ``a`` volts inward *and* absorb whatever returns, from one
    expression. The sign ``±`` and the ghost-H plane index depend on the face:
    the high-index face (e.g. ``z1``) writes the ghost H at the mode's own index
    ``k`` with ``+``; the low-index face (``z0``) writes it at ``k-1`` with ``−``
    (the Yee z-curl does not update the ``k=0`` E-plane, so a low face must sit at
    least one cell in).

    **Co-planar ports.** A multi-conductor cross-section is terminated by one
    ``ModalPort`` per conductor mode, all writing the *same* ghost plane. They
    superpose there,

        ``H_ghost = Σ_m s_m·(V̄_m − 2·a_m)·ĥ_m``

    which each port arranges for itself — register them as separate boundaries in
    any order and the plane carries the sum (see ``_GhostPlaneGroup``). Each
    ``V̄_m`` is that port's own modal projection, so this is the whole of the
    coupling the sheet needs.

    One accuracy caveat, and it is *not* about the summation: the conductor-basis
    modes :func:`~wavesim.mode_solver.solve_tem_modes` returns (energize conductor
    ``id``, ground the rest) are not mutually orthogonal, so ``V̄_m`` picks up
    some of mode ``n``'s content and the termination carries a cross-coupling
    error. That is a property of the modal basis, not of the port, and it is
    unaddressed here.

    **Port record.** Each step appends to ``times``, ``voltages`` and
    ``currents``, all co-timed at step ``n``. ``V`` is the ε-weighted modal
    projection of the plane E (``Eⁿ``); ``I`` is the modal projection of the H
    plane **one cell inside** the ghost plane, signed **positive into the
    domain**, so ``V·I`` is the power the port delivers inward. Its quadrature is
    the Poynting pairing

        ``I = ±Σ_c ê_c · H_c · dA_open,c``

    over the transverse cells (``ê`` the same 1 V-normalised staggered profile
    the sheet itself is built from, ``±`` the face sign). For a pure mode of
    modal voltage ``V`` propagating along +normal this returns ``G·V``, with

        ``G = Σ_c (ê_c²/η_c)·dA_open,c``

    the **discrete modal conductance** of
    :meth:`~wavesim.mode_solver.TEMMode.numerical_admittance_scale`. Being the
    Poynting pairing rather than a least-squares fit, ``V·I`` *is* the modal
    power through the plane; and because ``G`` is a pure profile/grid quadrature,
    the record needs no ``Z₀`` — it works on a mode solved with
    ``compute_params=False``. ``H`` is stored at ``n±½``, so ``I`` is the average
    of the two half-step projections straddling ``n`` (the first step averages
    against zero, where the fields are ~0).

    ``reference_impedance`` is the impedance the ``(V, I)`` pair is
    self-consistent against, ``Z_ref = 1/(s·G)``: the sheet writes
    ``H = s·V·ĥ``, so ``s·G`` is its terminating admittance *as this same
    read-back measures it*. Wave amplitudes follow as

        ``a = (V + Z_ref·I)/2``   (incident, into the domain)
        ``b = (V − Z_ref·I)/2``   (outgoing, into the port)

    so ``S_ji = FFT(b_j)/FFT(a_i)``. Referencing to ``Z_ref`` rather than to
    ``Z₀`` is what makes a matched absorber read ``a = 0`` to round-off instead
    of carrying ``(1−s)/2`` of spurious reflection; with ``s`` derived (the
    default) the two coincide, ``Z_ref == Z₀`` identically.

    Parameters
    ----------
    mode : TEMMode
        A mode from :func:`~wavesim.mode_solver.solve_tem_modes` (``compute_params
        =True`` gives its ``impedance``, not required for a pure absorber). Solve
        it *on the face plane* you want to terminate.
    amplitude : float
        Forward-wave launch voltage in volts (0 ⇒ pure absorber). Calibrated like
        :meth:`~wavesim.mode_solver.TEMMode.to_source`: ``amplitude=1`` launches a
        wave a downstream :class:`~wavesim.monitors.VoltageMonitor` reads as
        ``waveform(t)`` volts.
    waveform : Callable[[float], float], optional
        Temporal profile of the launch; required if ``amplitude != 0``.
    face : str, optional
        ``'z0'``/``'z1'`` (or ``x``/``y`` variants). ``None`` (default) picks the
        low/high face from the mode's ``slice_index`` relative to the grid size.
    admittance_scale : float, optional
        The discrete numerical-admittance correction ``s``. ``None`` (default)
        derives it from the mode on the grid (see
        :meth:`~wavesim.mode_solver.TEMMode.numerical_admittance_scale`); pass a
        float to override. It is ``1.0`` for an ideal 1-D mode and departs from it
        only through transverse-discretisation error, shrinking toward 1 as the
        cross-section is refined.
    """

    def __init__(self, mode, *, amplitude: float = 0.0,
                 waveform: Callable[[float], float] | None = None,
                 face: str | None = None,
                 admittance_scale: float | None = None) -> None:
        if amplitude != 0.0 and waveform is None:
            raise ValueError(
                "ModalPort with amplitude != 0 needs a waveform= to launch.")
        self.mode = mode
        self.amplitude = float(amplitude)
        self.waveform = waveform
        self.face = face
        self._scale_override = admittance_scale
        self._ready = False
        self.times: list = []
        self.voltages: list = []
        self.currents: list = []
        self.modal_conductance: float | None = None   # G, set at setup
        self.reference_impedance: float | None = None  # 1/(s·G), set at setup

    # -- one-time compile ---------------------------------------------------- #
    def _setup(self, grid: FDTDGrid) -> None:
        from wavesim.mode_solver import (_plane_to_grid, _launch_time_shift,
                                         _normal_width, _plane_open_fractions,
                                         _slice, _NORMAL_CFG,
                                         port_plane_pinned_nodes,
                                         port_sheet_divergence,
                                         _SHEET_DIVERGENCE_TOL)

        normal = self.mode.normal
        k = self.mode.slice_index
        n_along = {'x': grid.Nx, 'y': grid.Ny, 'z': grid.Nz}[normal]

        # Resolve the face → outward direction. High-index face (z1-like) has the
        # ghost H at the mode's own index k and sign +; low-index face (z0-like)
        # at k-1 and sign − (see the class docstring for the Yee-stencil reason).
        if self.face is not None:
            outward = +1 if self.face.endswith('1') else -1
        else:
            outward = +1 if k >= n_along // 2 else -1
        if outward > 0:
            self._h_k = k
            self._sign = +1.0
        else:
            if k < 1:
                raise ValueError(
                    f"A low-index ModalPort needs its ghost-H plane one cell "
                    f"inside the domain, but the mode sits at {normal}-index {k}. "
                    f"Solve the mode at least one cell in.")
            self._h_k = k - 1
            self._sign = -1.0

        # V read-back edges (ε-weighted modal projection at the face plane k).
        ker = self.mode.build_port_kernel(grid, directional=False)
        self._edges = ker['edges']

        # Staggered H pattern (n̂ × ê)/η, mapped to grid indices on the ghost plane.
        E_stag, H = self.mode._staggered_port_fields(grid)
        self._h = {}
        for comp, arr2d in H.items():
            a, b = np.nonzero(arr2d)
            if a.size == 0:
                continue
            ii, jj, kk = _plane_to_grid(normal, self._h_k, a, b)
            self._h[comp] = (ii, jj, kk, arr2d[a, b])

        # Ports sharing this ghost plane must sum onto it, not overwrite each
        # other; the group is how they find one another. Keyed on the plane
        # *written*, which is why it can only be joined here, after ``_h_k``.
        self._shape = (grid.Nx, grid.Ny, grid.Nz)
        self._group = _ghost_group(grid, normal, self._h_k)

        # Ghost-plane normal-E edges to hold at zero (conformal grids only).
        # See :meth:`apply_post_E` for what goes wrong without this.
        self._pin = None
        if grid.is_conformal:
            pinned = port_plane_pinned_nodes(grid, normal, k)
            a, b = np.nonzero(pinned)
            if a.size:
                ii, jj, kk = _plane_to_grid(normal, self._h_k, a, b)
                self._pin = ({'x': 'Ex', 'y': 'Ey', 'z': 'Ez'}[normal],
                             ii, jj, kk)

        # The sheet must be divergence-free where the mode solve was free to make
        # it so; nothing downstream can repair it if it is not, because the ghost
        # plane runs open loop. Measured, not assumed — see
        # :func:`~wavesim.mode_solver.port_sheet_divergence` for what the number
        # means and why round-off is thirteen orders below a real failure.
        self.sheet_divergence = port_sheet_divergence(self.mode, grid)
        if self.sheet_divergence > _SHEET_DIVERGENCE_TOL:
            warnings.warn(
                f"ModalPort on the {normal}-plane at index {k}: the injected "
                f"sheet has a transverse divergence of {self.sheet_divergence:.3g} "
                f"(relative; round-off is ~1e-14) at nodes where the mode solver "
                f"solved for phi and so drove it to zero. The ghost plane is open "
                f"loop, so this will integrate into a static field on both port "
                f"planes rather than radiate away. The usual cause is a grid whose "
                f"permittivity disagrees with its own cut-cell geometry: an edge "
                f"the open fractions call dielectric carrying the background eps "
                f"the voxeliser left inside the metal. Check eps_x/eps_y/eps_z "
                f"against pec_edge_open_* before trusting this port.",
                RuntimeWarning, stacklevel=2)

        # Amplitude calibration of the launch. ``V̄ − 2a`` radiates a forward wave
        # whose modal voltage equals ``a`` for a matched sheet, so ``amplitude`` is
        # already the forward volts — no extra factor (validated in tests).
        if self._scale_override is not None:
            self._scale = float(self._scale_override)
        else:
            self._scale = self.mode.numerical_admittance_scale(grid)

        # Modal current read-back (see "Port record" in the class docstring).
        # The H plane read is the one a cell *inside* the ghost plane — the
        # ghost plane's own H is written by this port every step and would only
        # play back the sheet law. High face: ghost at k, interior at k-1; low
        # face: ghost at k-1, interior at k. Both are ``_h_k − sign``.
        self._i_k = self._h_k - int(self._sign)
        if not 0 <= self._i_k < n_along:
            raise ValueError(
                f"A ModalPort needs an interior H plane to read its current "
                f"from, but the mode at {normal}-index {k} puts it at "
                f"{self._i_k}. Solve the mode at least one cell in.")
        # Weights ±ê·dA on the *paired* H component: Ĥa = sa·êb/η and
        # Ĥb = sb·êa/η, so η·Ĥ·H·dA — the Poynting pairing — is (s·ê)·H·dA with
        # no η left in it. Summed against a pure mode (H = V·Ĥ) this returns
        # G·V, with G the same open-area conductance quadrature
        # ``numerical_admittance_scale`` builds; G is accumulated here from the
        # same terms so the two cannot drift apart.
        cfg = _NORMAL_CFG[normal]
        prim = {'x': grid.dxp, 'y': grid.dyp, 'z': grid.dzp}
        dA = prim[cfg['axes'][0]][:, None] * prim[cfg['axes'][1]][None, :]
        f_open = dict(zip(cfg['E'],
                          _plane_open_fractions(grid, cfg, normal, k)))
        mu_p = _slice(getattr(grid, cfg['mu']), normal, k)
        eps_of = dict(zip(('Ex', 'Ey', 'Ez'), conformal_edge_eps(grid)))
        pair = {cfg['E'][0]: (cfg['H'][1], cfg['h_sign'][1]),
                cfg['E'][1]: (cfg['H'][0], cfg['h_sign'][0])}
        self._i_edges = {}
        G = 0.0
        for e_comp, ehat2d in E_stag.items():
            a, b = np.nonzero(ehat2d)
            if a.size == 0:
                continue
            h_comp, s_h = pair[e_comp]
            dA_c = dA if f_open[e_comp] is None else dA * f_open[e_comp]
            ehat = ehat2d[a, b]
            area = dA_c[a, b]
            ii, jj, kk = _plane_to_grid(normal, self._i_k, a, b)
            self._i_edges[h_comp] = (ii, jj, kk, s_h * ehat * area)
            i2, j2, k2 = _plane_to_grid(normal, k, a, b)
            epsr = eps_of[e_comp][i2, j2, k2]
            eta = ETA0 * np.sqrt(mu_p[a, b] / np.where(epsr > 0, epsr, 1.0))
            G += float(np.sum(ehat ** 2 / eta * area))
        if G <= 0.0:
            raise ValueError(
                "TEM mode has no transverse E energy on the plane; cannot read "
                "back a modal current.")
        self.modal_conductance = G
        # The sheet's own terminating admittance as this read-back measures it
        # is s·G, so that — not Z₀ — is the reference an ``a`` of exactly zero
        # falls out of. They are the same number whenever s was derived.
        self.reference_impedance = 1.0 / (self._scale * G)
        self._i_prev = 0.0        # previous half-step current, for centring

        # Read-back time-shift. The ghost H written here is consumed by the E
        # update as H^{n+½}, so the matched modal voltage must be sampled at
        # n + h_tau, where ``h_tau = dt/2 − dn/(2·v)`` is the *same* E↔H
        # space-time offset the directional launch already applies to its H
        # sheet (:meth:`build_port_kernel`). The naive n−½ average left an
        # O(ω·dt) phase error that was the dominant reflection floor (~−25 dB on
        # a coax; ~−33 dB once shifted). ``read_tau`` is that offset in steps
        # (≤ 0). DC-safe: at DC ``V`` is constant, so the shift changes nothing
        # and the DC-exact termination is preserved. ``frequency=None`` keeps the
        # broadband continuum velocity, matching the mode's launch counterpart.
        v_ph = self.mode.v_phase if self.mode.v_phase else C0
        dn = float(_normal_width(grid, normal)[self._h_k])
        self._read_tau = _launch_time_shift(grid.dt, dn, v_ph, None) / grid.dt
        self._hist_len = int(np.ceil(max(-self._read_tau, 0.0))) + 2
        self._v_hist = []
        self._ready = True

    # -- per-step boundary hook (runs between update_H and update_E) ---------- #
    def apply(self, grid: FDTDGrid, t: float) -> None:
        if not self._ready:
            self._setup(grid)
        V = 0.0
        for comp, (ii, jj, kk, w, coef) in self._edges.items():
            V += float(np.dot(getattr(grid, comp)[ii, jj, kk], w))
        # Modal current on the interior H plane. This hook runs between the H
        # and E updates, so ``grid.H`` holds H^{n+½} and the projection is the
        # half-step current; averaging it with the previous half step lands I at
        # n, alongside V. Read before the ghost write below — a different plane,
        # so only for clarity. The face sign makes I positive *into* the domain:
        # the raw quadrature is positive along +normal, which leaves the domain
        # through a high face and enters it through a low one.
        i_raw = 0.0
        for comp, (ii, jj, kk, w) in self._i_edges.items():
            i_raw += float(np.dot(getattr(grid, comp)[ii, jj, kk], w))
        i_half = -self._sign * i_raw
        i_port = 0.5 * (i_half + self._i_prev)
        self._i_prev = i_half
        # Modal voltage sampled at n + read_tau (read_tau ≤ 0), by linear
        # interpolation over the recent history (newest sample = V^n). Reading a
        # past instant needs only stored samples — no extrapolation. Startup
        # (history shorter than the offset) clamps to the oldest sample, where
        # the fields are ~0 and the choice is immaterial.
        self._v_hist.append(V)
        if len(self._v_hist) > self._hist_len:
            del self._v_hist[0]
        s = -self._read_tau
        i0 = int(np.floor(s))
        frac = s - i0
        h = self._v_hist
        n = len(h)
        v_shift = ((1.0 - frac) * h[-1 - min(i0, n - 1)]
                   + frac * h[-1 - min(i0 + 1, n - 1)])
        a = self.amplitude * (self.waveform(t) if self.waveform is not None else 0.0)
        amp = self._sign * self._scale * (v_shift - 2.0 * a)
        self._write_ghost_h(grid, t, amp)
        self.times.append(t)
        self.voltages.append(V)
        self.currents.append(i_port)

    def _write_ghost_h(self, grid: FDTDGrid, t: float, amp: float) -> None:
        """Put this port's sheet on the ghost plane, summed with any co-planar
        peers' (see the ``_GhostPlaneGroup`` notes above ``ModalPort``).

        The write stays a fancy-index **assignment** on ``grid.<comp>``, exactly
        as the single-port path always did, so it makes no new demand of the
        backend. Every path that runs boundaries runs them through
        :meth:`~wavesim.simulation.Simulation.step`, where ``grid.Hx`` &c. are
        host numpy arrays on all three backends — the CUDA one included, since it
        keeps the fields on the host between steps. (``run(backend='cuda')`` takes
        the resident fast path, which does not run boundaries at all: a
        pre-existing limitation, unrelated to this write.)
        """
        group = self._group
        group.open_step(self, t)
        group.contrib[self] = amp
        if len(group.contrib) == 1:
            # The common case, and the one that has to stay bit-identical: one
            # port on this plane writes its own pattern, unsummed.
            for comp, (ii, jj, kk, vals) in self._h.items():
                getattr(grid, comp)[ii, jj, kk] = amp * vals
            return
        for comp, (idx, n, members) in group.plan(self._shape).items():
            total = np.zeros(n)
            for port, off in members:
                total[off] += group.contrib[port] * port._h[comp][3]
            getattr(grid, comp)[idx] = total

    # -- per-step post-E hook (runs after update_E and the PEC masking) ------- #
    def apply_post_E(self, grid: FDTDGrid, t: float) -> None:
        """Hold the port-normal E at zero on the ghost-H plane's conductor nodes.

        The ghost plane is the one place in the domain where the leapfrog is
        **open loop**: :meth:`apply` overwrites its tangential H every step, so
        whatever E the next Ampère update writes there can never act back on that
        H. Anything the E update deposits on that plane therefore *integrates*,
        step after step, with no restoring force.

        What it deposits is the sheet's discrete transverse divergence. Writing
        ``ĥ = (n̂ × ê)/η`` on the plane, the normal-E update reads its transverse
        curl, which is ``∇_t·ê/η``. The mode solver's Laplacian drives that to
        round-off at every node where it *solved* for ``φ`` — but not at the nodes
        where it **pinned** ``φ``, i.e. the nodes on the conductor. There the
        residual is the mode's induced surface charge, which is physically real
        and of order the field itself.

        That residual is not new and is not conformal: it is the same size in both
        formulations (2.4e3 in ``V⁻¹m⁻¹`` on the reference coax either way). The
        staircase run never saw it only because
        :func:`wavesim.pec.build_pec_edge_masks` **dilates**, so every one of those
        conductor-node edges was already zeroed for an unrelated reason. The
        conformal rule as first written — zero an edge iff its own open length is
        zero — left them alive, because an edge running *along* a grid-aligned
        conductor surface is fully open even though both of its endpoints are on
        the metal. That turned out to be the defect and not the price of being
        exact: such an edge is a tangential E on a PEC boundary.
        :func:`wavesim.pec.build_conformal_edge_masks` now finds it through the
        fully covered H face on the metal side, and the same integral landing on
        the *transverse* E of the mode plane — which is inside the domain, and is
        what every downstream monitor sees — went with it.

        What is left for this hook is the case that rule cannot see: it needs a
        covered face, and a conductor that varies along the port normal need not
        present one. On every cross-section in the test suite the two agree and
        this is redundant (``tests/test_modal_port_ghost_plane.py``); it is kept
        because the plane it guards is open loop, where being wrong is unbounded.

        Measured on the plan's reference coax at ``d`` = 0.5 mm with a 1 GHz
        modulated Gaussian: ``max|Ez|`` on the ghost plane reached 6.87e3 V/m
        against 0.35 V/m one cell in — five orders — and was reproduced to
        3.6e-12 V/m by ``(dt/(ε₀ε))·∇_t·ê/η·Σₙ ampₙ``, i.e. it is exactly this
        integral and nothing else. It survives ``amplitude = 0`` on both ports, so
        it is a property of the sheet, not of the drive.

        The condition is **geometric alignment**, not cut-cell size: it fires only
        where a conductor surface lands on the node ruler. Shifting the same coax
        a quarter cell off-lattice takes the count of affected edges to zero; a
        grid-aligned *rectangular* conductor, where every surface node qualifies,
        takes it from 6 to 117.

        The nodes are the ones :func:`~wavesim.mode_solver._build_mode` pinned
        ``φ`` at, read back from the same
        :func:`~wavesim.mode_solver.port_plane_pinned_nodes`. That has to be the
        *same* set, not a second opinion assembled from the open fractions: the
        residual lives exactly where φ was pinned, so a pin computed some other
        way misses precisely the nodes the two rules disagree about.

        Zeroing here rather than in :meth:`apply` is deliberate: ``apply`` runs
        *before* the E update, so it would leave one step's worth of the residual
        (~60 V/m on that case) alive for the monitors and for the next H update.
        This hook runs after the E update and after the PEC masking, so the plane
        is clean when anything reads it.
        """
        if self._pin is None:
            return
        comp, ii, jj, kk = self._pin
        getattr(grid, comp)[ii, jj, kk] = 0.0


class SpicePort(LineSource):
    """Lumped port coupled to an ngspice circuit (SPICE co-simulation).

    A :class:`SpicePort` is a :class:`LineSource` whose per-step circuit law is
    replaced by a live ngspice solve. Geometry, the self-coupling κ, the
    time-centred (Piket-May) injection and the port recording are all inherited
    unchanged — the *only* difference is where the impressed current comes from:
    each step the FDTD port hands ngspice its Thévenin equivalent (voltage
    ``v_mid`` behind ``κ/2``) and reads the resulting branch current back (see
    :mod:`wavesim.spice`). If the ngspice circuit reduces to a Thévenin
    ``(Vs, Z)`` the injected current matches ``LineSource(voltage=Vs,
    resistance=Z)`` exactly — the golden equivalence test.

    The port geometry is **either** a straight line (``p0``/``p1``, as in
    :class:`LineSource`) **or** a solved TEM mode (``mode=``, as in
    :class:`TEMPort`, with the same distributed projection / κ / directional H
    sheet). Exactly one of the two must be given.

    The two ``nodes`` must already exist in the netlist (the user's circuit
    connects to them); wavesim splices the Thévenin companion across them.
    ``p0``/``nodes[0]`` are the "+" terminal. For a ``mode=`` port put the
    matched source resistance ``Z₀`` in the netlist itself.

    Parameters
    ----------
    netlist : str
        Path to the SPICE netlist file.
    nodes : (str, str)
        Port node names ``(plus, minus)`` in the netlist.
    p0, p1 : tuple of float, optional
        Line endpoints ``(x, y, z)`` in metres (``p0`` is the "+" terminal).
        Mutually exclusive with ``mode``.
    mode : TEMMode, optional
        A solved mode to drive as a distributed port. Mutually exclusive with
        ``p0``/``p1``.
    directional : bool
        For a ``mode=`` port, also drive the paired H sheet (one-way launch);
        ignored for a line port. Default True.
    library_path : str, optional
        Full path to the ngspice shared library (else PySpice's own search /
        ``NGSPICE_LIBRARY_PATH``).
    sign : float
        ±1 branch-current orientation (fixed by the golden test); default +1.
    uic : bool
        Pass ``uic`` to ngspice's ``.tran`` (skip the DC operating point).
    """

    def __init__(self, *,
                 netlist: str, nodes: Tuple[str, str],
                 p0: Tuple[float, float, float] | None = None,
                 p1: Tuple[float, float, float] | None = None,
                 mode=None, directional: bool = True,
                 library_path: str | None = None,
                 sign: float = 1.0, uic: bool = False) -> None:
        # Bypass LineSource.__init__ (which validates the analytic modes'
        # voltage/current/load); replicate just the state _build_port / inject
        # need. The drive is supplied by ngspice, so there is no waveform.
        Source.__init__(self, lambda t: 0.0)
        if mode is None and (p0 is None or p1 is None):
            raise ValueError(
                "SpicePort needs either mode= or both p0= and p1=.")
        if mode is not None and (p0 is not None or p1 is not None):
            raise ValueError(
                "SpicePort takes either mode= or p0=/p1=, not both.")
        self.mode = mode
        self.directional = bool(directional)
        self.p0 = tuple(p0) if p0 is not None else None
        self.p1 = tuple(p1) if p1 is not None else None
        self.voltage = None
        self.current = None
        self.impedance = None
        self.times: list = []
        self.voltages: list = []
        self.currents: list = []
        self._port: dict | None = None
        self._v_prev = 0.0
        # SPICE side.
        self.netlist = netlist
        self.nodes = (str(nodes[0]), str(nodes[1]))
        self.library_path = library_path
        self.sign = float(sign)
        self.uic = bool(uic)
        self._coupler = None    # wavesim.spice.SpiceCoupler, built on first inject

    def _build_port(self, grid: FDTDGrid) -> dict:
        if self.mode is not None:
            # The drive is a netlist, so there is no single frequency to tune
            # the launch to; the continuum velocity is used (see build_port_kernel).
            kernel = self.mode.build_port_kernel(
                grid, directional=self.directional)
            self._h_lag_steps = -kernel.get('h_tau', 0.0) / grid.dt
            return kernel
        return super()._build_port(grid)

    def inject(self, grid: FDTDGrid, t: float) -> None:
        if self._port is None:
            self._port = self._build_port(grid)
        if self._coupler is None:
            # Import here so `import wavesim` never requires PySpice/ngspice.
            from wavesim.spice import SpiceCoupler
            self._coupler = SpiceCoupler(
                netlist=self.netlist, nodes=self.nodes,
                kappa=self._port['kappa'], dt=grid.dt,
                library_path=self.library_path, sign=self.sign, uic=self.uic)
            self._coupler.start()

        edges = self._port['edges']
        kappa = self._port['kappa']

        # Port voltage before injection: V* = Σ E·dl (p0 → p1), post curl update.
        v_before = 0.0
        for comp, (ii, jj, kk, w, _coef) in edges.items():
            v_before += float(np.dot(getattr(grid, comp)[ii, jj, kk], w))

        # Time-centred port voltage handed to ngspice as the Thévenin source.
        v_mid = 0.5 * (self._v_prev + v_before)
        i_port = self._coupler.advance(v_mid, grid.dt)

        for comp, (ii, jj, kk, w, coef) in edges.items():
            getattr(grid, comp)[ii, jj, kk] += coef * i_port
        v_after = v_before + kappa * i_port

        self._v_prev = v_after
        self.times.append(t)
        self.voltages.append(v_after)
        self.currents.append(i_port)

        # Directional (EH) launch for a mode port: the paired H sheet, driven by
        # the port current lagged onto the sheet's own space-time sample point.
        # Placed after the history append so _lagged_current sees this step.
        hedges = self._port.get('hedges')
        if hedges:
            i_h = self._lagged_current(self._h_lag_steps)
            for comp, (ii, jj, kk, coefH) in hedges.items():
                getattr(grid, comp)[ii, jj, kk] += coefH * i_h

    def close(self) -> None:
        """Tear down the ngspice instance (optional; also freed on GC)."""
        if self._coupler is not None:
            self._coupler.close()
            self._coupler = None


class VolumeSource(Source):
    """
    Volumetric excitation / field seeding over a box region — *not yet
    implemented*.

    Intended for full-3D initialisation of fields inside a sub-domain.

    Parameters
    ----------
    waveform : Callable[[float], float]
        Shared time function.
    bounds : tuple of float
        Box extent in metres ``(x0, x1, y0, y1, z0, z1)``, snapped to the
        nearest cells against the grid.
    profiles : mapping, optional
        ``{component: array}`` weights over the region; ``None`` ⇒ uniform.
    """

    def __init__(self, waveform: Callable[[float], float], *,
                 bounds: Tuple[float, float, float, float, float, float],
                 profiles: Mapping[str, np.ndarray] | None = None) -> None:
        super().__init__(waveform)
        self.bounds = bounds
        self.profiles = profiles

    def spatial_profiles(self, grid: FDTDGrid) -> Dict[str, np.ndarray]:
        raise NotImplementedError("VolumeSource is not implemented yet.")
