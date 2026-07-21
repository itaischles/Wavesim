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
with no ``impedance=``) pins ∫E·dl on its line, which is a hard write — the
physically correct behaviour of a zero-impedance source.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple, Union
import numpy as np

from wavesim.constants import C0, EPS0, ETA0
from wavesim.grid import FDTDGrid
# Shared with VoltageMonitor so a LineSource and a monitor on the same path
# snap to identical Yee E-edges and agree bit-for-bit on ∫E·dl.
from wavesim.monitors import _build_path_quadrature


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
        port-mode path (``profiles`` given, e.g. from
        :meth:`~wavesim.mode_solver.TEMMode.to_source`) maps each
        ``{component: 2D-array}`` onto a full-grid weight array, nonzero only on
        the slice perpendicular to ``axis`` at the snapped cell.
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


class _PlaneLaunch(Source):
    """Full-slice E (and, when directional, paired H) launch with the corrected
    co-indexed H time shift. Shared engine behind :class:`PlaneWave` and
    :meth:`~wavesim.mode_solver.TEMMode.to_source`.

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


class PlaneWave(_PlaneLaunch):
    """A directional plane wave launched from one boundary face.

    Drives the full cross-section one PML-depth inside a boundary face with a
    uniform transverse field, biased into the domain: an E sheet plus the paired
    ``H = (n̂ × E)/η`` sheet (:class:`_PlaneLaunch`). The waveform carries the
    amplitude — there is deliberately no ``amplitude`` parameter, as for every
    other source. The launched field is *not* amplitude-calibrated (it scales as
    ≈ ``1/S_n`` × the waveform, ``S_n`` the Courant number along the normal); use
    a monitor to normalise if you need an absolute level.

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
    d_pml : int
        PML thickness in cells (default 10, matching :func:`init_cpml`). The E
        sheet is placed on the first interior cell — index ``d_pml`` on a low
        face, ``N-1-d_pml`` on a high face — so the backward lobe is launched
        straight into the absorber.
    directional : bool
        Pair the E sheet with an H sheet for a one-way launch (default True).
        ``False`` gives a bare E sheet, which radiates symmetrically both ways.

    Notes
    -----
    There are no periodic/Bloch boundaries, so a truly infinite plane wave is not
    reachable: expect edge effects where the sheet meets the transverse PMLs.
    """

    def __init__(self, face: str, angle: float,
                 waveform: Callable[[float], float], *,
                 d_pml: int = 10, directional: bool = True) -> None:
        if face not in _FACE_CFG:
            raise ValueError(
                f"face must be one of {sorted(_FACE_CFG)}, got {face!r}.")
        cfg = _FACE_CFG[face]
        super().__init__(waveform, normal=cfg['normal'], directional=directional,
                         v_medium=C0,
                         prop_sign=(1.0 if cfg['side'] == 'low' else -1.0))
        self.face = face
        self.angle = float(angle)
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
        ones = np.ones_like(eps_a)
        # E = cos·â + sin·b̂;  H = (n̂ × E)/η = (cos·b̂ - sin·â)/η
        E = {'E' + a_ax: ca * ones, 'E' + b_ax: sa * ones}
        H = {'H' + a_ax: -sa / eta_a, 'H' + b_ax: ca / eta_b}
        return E, H


class LineSource(Source):
    """
    Lumped V-I-Z element on a straight line between two endpoints.

    The line from ``p0`` to ``p1`` is rasterised onto Yee E-edges with the same
    quadrature as :class:`~wavesim.monitors.VoltageMonitor`, so the element's
    port voltage V(t) = ∫E·dl (p0 → p1) is exactly what a VoltageMonitor on the
    same path reads. ``p0`` is the "+" terminal; positive port current I(t) is
    delivered out of ``p0`` into the surrounding structure.

    Exactly one of ``voltage`` / ``current`` selects the drive (or neither, for
    a passive resistor); ``impedance`` composes with either:

    ==========================  =============================================
    Arguments                   Element
    ==========================  =============================================
    ``voltage=Vs``              Ideal voltage source — pins ∫E·dl = Vs(t)
                                each step (a *hard* write; reflects incident
                                waves, as a zero-impedance source must).
    ``voltage=Vs, impedance=Z`` Thevenin source: V = Vs(t) − I·Z.
    ``current=Is``              Ideal current source — soft impressed current
                                Is(t) along the line.
    ``current=Is, impedance=Z`` Norton source: I = Is(t) − V/Z.
    ``impedance=Z`` only        Passive lumped resistor: I = −V/Z
                                (e.g. a matched termination).
    ==========================  =============================================

    Unlike the static sources above, the injection depends on the local field
    each step (an impedance/feedback relationship), so ``inject`` is overridden.
    An impressed current I spread along the line adds
    ``E_a += dt · I · dl_a / (ε_a · dV_cell)`` on each occupied edge, which
    changes the port voltage by ``κ·I`` with ``κ = Σ dt·dl²/(ε·dV)`` (the line's
    self-coupling, ohm-like). Impedance modes use the semi-implicit current

        I = (Vs(t) − (Vⁿ + Vⁿ⁺¹)/2) / Z      (Norton: Vs ≡ Z·Is)

    solved for the injected I as ``(Vs − (Vⁿ + V*)/2)/(Z + κ/2)``, where ``V*``
    is the just-curl-updated line voltage and ``Vⁿ`` the voltage at the end of
    the previous step. This is the standard Piket-May semi-implicit lumped
    element, time-centred across the whole step, and is stable for any Z > 0.
    (Centring on V* alone — or the naive explicit I = (Vs−V)/Z — couples
    unstably with the leapfrog update once Z is below a few hundred ohms.)

    The element self-records its port quantities each step — ``times`` (s),
    ``voltages`` (V, post-injection, what a co-located VoltageMonitor reads) and
    ``currents`` (A, the injected impressed current; in ideal-voltage mode the
    equivalent current (Vs − V_before)/κ that produces the imposed field change)
    — so it doubles as a port for impedance / S-parameter extraction.

    Two discretisation caveats, both standard for FDTD lumped elements:

    * **Effective impedance.** To the surrounding field the element presents
      ≈ ``Z + κ/2``, not Z — the κ/2 is the parasitic of the stable implicit
      averaging (verified here by matched-launch tests on a parallel-plate
      line). ``self_coupling(grid)`` returns κ so you can pre-compensate
      (``impedance = Z_target − κ/2``, only possible while that stays > 0) or
      de-embed. The recorded V(t)/I(t) are exact regardless, so port
      extraction is unaffected.
    * **Co-located elements.** Elements sharing line edges inject
      sequentially, not as a jointly solved circuit, so each contributes its
      own κ/2 in series (a 2-element voltage divider on one line settles to
      ``Vs·Z_L/(Z + Z_L + κ)``). Combine them into a single equivalent
      element instead.

    The line typically spans the gap between two conductors; endpoints may sit
    just inside PEC (as with the monitors), but keep the driven gap itself in
    dielectric — E on PEC edges is zeroed every step.

    Parameters
    ----------
    p0, p1 : tuple of float
        Endpoints ``(x, y, z)`` in metres; ``p0`` is the "+" terminal. Any
        orientation (oblique lines are split per-axis onto staggered edges).
    voltage : Callable[[float], float], optional
        Source voltage Vs(t) in volts (Thevenin open-circuit value when
        ``impedance`` is given).
    current : Callable[[float], float], optional
        Source current Is(t) in amperes (Norton short-circuit value when
        ``impedance`` is given). Mutually exclusive with ``voltage``.
    impedance : float, optional
        Resistive impedance Z in ohms (> 0).
    """

    def __init__(self, *,
                 p0: Tuple[float, float, float], p1: Tuple[float, float, float],
                 voltage: Callable[[float], float] | None = None,
                 current: Callable[[float], float] | None = None,
                 impedance: float | None = None) -> None:
        if voltage is not None and current is not None:
            raise ValueError(
                "LineSource takes either voltage= or current=, not both.")
        if voltage is None and current is None and impedance is None:
            raise ValueError(
                "LineSource needs a drive (voltage= or current=) and/or an "
                "impedance= (impedance alone gives a passive resistor).")
        if impedance is not None and not impedance > 0:
            raise ValueError(
                f"impedance must be a positive resistance in ohms, "
                f"got {impedance!r}.")
        drive = voltage if voltage is not None else current
        super().__init__(drive if drive is not None else (lambda t: 0.0))
        self.p0 = tuple(p0)
        self.p1 = tuple(p1)
        self.voltage = voltage
        self.current = current
        self.impedance = impedance
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
        eps_of = {'Ex': grid.eps_x, 'Ey': grid.eps_y, 'Ez': grid.eps_z}
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
        step, ``Σ dt·dl²/(ε·dV)`` over the line's edges. The element's
        effective impedance to the field is ≈ ``impedance + κ/2`` (see class
        docstring)."""
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
        Z = self.impedance

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
            if self.voltage is not None:        # Thevenin
                i_port = (self.waveform(t) - v_mid) / (Z + 0.5 * kappa)
            elif Z is None:                     # ideal current source
                i_port = self.waveform(t)
            else:                               # Norton (resistor when Is ≡ 0)
                i_port = (Z * self.waveform(t) - v_mid) / (Z + 0.5 * kappa)
            for comp, (ii, jj, kk, w, coef) in edges.items():
                getattr(grid, comp)[ii, jj, kk] += coef * i_port
            v_after = v_before + kappa * i_port

        self._v_prev = v_after
        self.times.append(t)
        self.voltages.append(v_after)
        self.currents.append(i_port)


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

    The port presents ``Z₀`` to the field: the internal series resistance is
    pre-compensated to ``Z₀ − κ/2`` (κ = modal self-coupling), so a matched line
    sees a matched source. If ``κ/2`` exceeds ``Z₀`` (a low-impedance mode on a
    coarse transverse grid) the semi-implicit scheme cannot be stabilised — refine
    the transverse grid to lower κ.

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
        Thévenin ``Vs(t)`` or Norton ``Is(t)`` drive (mutually exclusive); omit
        both for a passive matched termination.
    impedance : float, optional
        Series/source impedance in ohms; defaults to the mode's ``Z₀``.
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
        drive = voltage if voltage is not None else current
        Source.__init__(self, drive if drive is not None else (lambda t: 0.0))
        self.mode = mode
        self.voltage = voltage
        self.current = current
        self._z0 = float(z0)
        self.directional = bool(directional)
        self.impedance = None       # finalised (pre-compensated) in _build_port
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
        half_kappa = 0.5 * kernel['kappa']
        z_int = self._z0 - half_kappa
        if not z_int > 0:
            raise ValueError(
                f"TEM port κ/2 = {half_kappa:.4g} Ω exceeds the target Z₀ = "
                f"{self._z0:.4g} Ω; the semi-implicit lumped scheme is unstable "
                f"there — refine the transverse grid to lower κ.")
        self.impedance = z_int
        return kernel

    def inject(self, grid: FDTDGrid, t: float) -> None:
        # Modal V* read-back, Piket-May law, E-injection and recording are all
        # inherited from LineSource; only the paired directional H sheet is new.
        super().inject(grid, t)
        hedges = self._port.get('hedges')
        if hedges:
            i_port = self._lagged_current(self._h_lag_steps)
            for comp, (ii, jj, kk, coefH) in hedges.items():
                getattr(grid, comp)[ii, jj, kk] += coefH * i_port


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
    impedance=Z)`` exactly — the golden equivalence test.

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
        # Bypass LineSource.__init__ (which validates voltage/current/impedance
        # for the analytic modes); replicate just the state _build_port / inject
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
