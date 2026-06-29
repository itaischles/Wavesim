"""
sources.py — excitation waveforms and the Source injection abstraction.

Two layers live here, and they compose:

1. Waveforms — the *time* part of an excitation. A waveform is any callable
   ``f(t) -> float``. :class:`GaussianPulse` is the built-in baseband pulse (and
   is itself callable); for a narrowband/CW drive just pass your own
   ``lambda t: ...`` anywhere a waveform is expected.

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
Hard injection (=) reflects waves — do not use; every Source here adds (+=).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple, Union
import numpy as np

from wavesim.grid import FDTDGrid


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


class LineSource(Source):
    """
    Lumped V-I-Z (Thevenin) line source between two endpoints — *not yet
    implemented*.

    Unlike the soft sources above, the injected excitation depends on the *local
    field each step* (a feedback/impedance relationship), so this class will
    override ``inject`` rather than supply a static ``spatial_profiles``.

    Parameters
    ----------
    waveform : Callable[[float], float]
        Open-circuit drive (Thevenin source voltage) as a function of time.
    p0, p1 : tuple of float
        Endpoint positions ``(x, y, z)`` of the line in metres, each snapped to
        the nearest cell against the grid.
    impedance : float, optional
        Series source impedance Z (ohms); ``None`` ⇒ ideal voltage source.
    """

    def __init__(self, waveform: Callable[[float], float], *,
                 p0: Tuple[float, float, float], p1: Tuple[float, float, float],
                 impedance: float | None = None) -> None:
        super().__init__(waveform)
        self.p0 = p0
        self.p1 = p1
        self.impedance = impedance

    def spatial_profiles(self, grid: FDTDGrid) -> Dict[str, np.ndarray]:
        raise NotImplementedError("LineSource is not implemented yet.")

    def inject(self, grid: FDTDGrid, t: float) -> None:
        raise NotImplementedError("LineSource is not implemented yet.")


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
