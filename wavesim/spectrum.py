"""
spectrum.py — Fourier transforms of recorded time series, and Z(f) / Y(f).

The workflow this exists for: excite a sub-wavelength structure at a port with
a broadband pulse, record V(t) and I(t) there (and at any other port), and read
off the frequency-domain response the structure presents — the raw material for
fitting a lumped-element model.

    sim.run(4000)
    Z = ws.impedance(port)                    # V(f) / I(f) at the driven port
    H = ws.transfer_function(v_out, v_in)     # port 2 / port 1

Everything funnels through one transform, :func:`spectrum`, which accepts any
of the recorded objects in the package — a :class:`~wavesim.monitors.VoltageMonitor`
or :class:`~wavesim.monitors.CurrentMonitor`, a :class:`~wavesim.monitors.FieldProbe`,
a self-recording port (:class:`~wavesim.sources.LineSource`,
:class:`~wavesim.sources.TEMPort`, :class:`~wavesim.sources.ModalPort`,
:class:`~wavesim.sources.SpicePort`), or a bare ``(times, values)`` pair — and
returns a :class:`Spectrum`.

Three things about FDTD data that this module handles, and which hand-rolled
``np.fft.rfft`` calls at the call site usually get wrong:

**The half-step stagger.** E and H leapfrog, so the E- and H-derived quantities
recorded on the same step are half a timestep apart. In :meth:`Simulation.step
<wavesim.simulation.Simulation.step>` the monitors run after both updates, when
E has advanced to ``(n+1)·dt`` but H only to ``(n+½)·dt``, and both get stamped
``n·dt``. Divide the two spectra naively and Z(f) picks up a spurious
``exp(+jπ·f·dt)``: harmless at low frequency, a 9° phase error at f = Nyquist/10,
90° at Nyquist. So every series carries a ``stagger`` (its true sample time minus
its stamp, in units of dt), inferred from what produced it, and the transform
undoes it. E-derived quantities are the time reference (``stagger = 0``);
H-derived ones are ``-0.5``. Pass ``stagger=`` explicitly to override.

**Truncation.** A low-loss resonant structure — exactly the kind whose lumped
model you are after — rings for a long time. If the run ends while it is still
ringing, the transform sees a rectangular-windowed sinusoid and the resonance
smears into a sinc. :func:`spectrum` measures the tail against the peak and warns
when the signal has not decayed, because no amount of post-processing recovers
information the run did not capture; ``window=`` is there for when a longer run
is not an option (it trades resolution for leakage, and biases the amplitude —
see :func:`spectrum`).

**Dividing noise by noise.** V(f)/I(f) is only meaningful where the excitation
put energy. Outside its band both numerator and denominator are round-off, and
their ratio is a garbage number that plots as a wild excursion. :func:`impedance`,
:func:`admittance` and :func:`transfer_function` mask those bins to NaN.

What this module does *not* do: multi-port Z/Y matrices, S-parameters, and
fitting an equivalent circuit to the result. Those build on this one.

.. note::
   A port's Z(f) includes the port cell's own gap capacitance
   ``C_cell = ε·dA/dl`` in parallel with the structure — see the discretisation
   caveats in :class:`~wavesim.sources.LineSource`. It is physical for a gap that
   is really there, and it moves with the mesh. Nothing here subtracts it.
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np


__all__ = [
    "Spectrum", "spectrum", "transfer_function", "impedance", "admittance",
]


# ======================================================================= #
# Spectrum — the result type
# ======================================================================= #

@dataclass(frozen=True)
class Spectrum:
    """
    A one-sided (real-input) spectrum: complex ``values`` on positive ``freqs``.

    The scaling is that of the continuous transform — the DFT sum times ``dt``
    — so ``values`` are per-hertz densities (V/Hz for a voltage) and are
    independent of the timestep and record length. That matters when comparing
    two runs at different resolutions; for ratios like Z(f) it cancels.

    Attributes
    ----------
    freqs : ndarray
        Frequency bins (Hz), from DC to Nyquist.
    values : ndarray
        Complex spectrum, same length as ``freqs``. May contain NaN where a
        ratio was masked as out-of-band.
    dt : float
        The timestep the series was sampled at (s).
    label : str
        Free-text name used by the plotting helpers ('V', 'I', 'Z', …).
    unit : str
        Physical unit of the underlying time series ('V', 'A', 'Ω', …).
    """
    freqs: np.ndarray
    values: np.ndarray
    dt: float
    label: str = ''
    unit: str = ''

    # --- views ---------------------------------------------------------- #

    @property
    def magnitude(self) -> np.ndarray:
        """``|X(f)|``."""
        return np.abs(self.values)

    @property
    def phase(self) -> np.ndarray:
        """Unwrapped phase in radians (NaN bins break the unwrap, by design —
        they mark where the phase is genuinely unknown)."""
        return np.unwrap(np.angle(self.values))

    @property
    def phase_deg(self) -> np.ndarray:
        """Unwrapped phase in degrees."""
        return np.degrees(self.phase)

    @property
    def db(self) -> np.ndarray:
        """``20·log10|X(f)|``, with zeros floored rather than -inf."""
        mag = self.magnitude
        floor = np.nanmax(mag) * 1e-300 if np.any(np.isfinite(mag)) else 1.0
        return 20.0 * np.log10(np.maximum(mag, max(floor, np.finfo(float).tiny)))

    @property
    def real(self) -> np.ndarray:
        """Real part — resistance R(f) for an impedance, conductance for a Y."""
        return self.values.real

    @property
    def imag(self) -> np.ndarray:
        """Imaginary part — reactance X(f) for an impedance, susceptance for Y."""
        return self.values.imag

    # --- selection ------------------------------------------------------ #

    def band(self, f_lo: float = 0.0, f_hi: float = np.inf) -> "Spectrum":
        """A new Spectrum restricted to ``f_lo <= f <= f_hi`` (Hz)."""
        keep = (self.freqs >= f_lo) & (self.freqs <= f_hi)
        return replace(self, freqs=self.freqs[keep], values=self.values[keep])

    def at(self, f) -> np.ndarray:
        """Complex value(s) at frequency ``f`` (Hz), linearly interpolated.

        Interpolating real and imaginary parts separately, which is right for a
        smooth spectrum sampled well above its features and wrong near a sharp
        resonance — there, refine the bin spacing with a longer run rather than
        trusting the interpolation.
        """
        f = np.asarray(f, dtype=float)
        re = np.interp(f, self.freqs, self.values.real, left=np.nan, right=np.nan)
        im = np.interp(f, self.freqs, self.values.imag, left=np.nan, right=np.nan)
        return (re + 1j * im)[()] if f.ndim else complex(re + 1j * im)

    def __len__(self) -> int:
        return len(self.freqs)


# ======================================================================= #
# Time-series adapters
# ======================================================================= #

# Stagger, in units of dt, of a recorded quantity's true sample time relative to
# its timestamp. E-derived quantities define the reference; H-derived ones are
# half a step behind. See the module docstring.
_STAGGER_E = 0.0
_STAGGER_H = -0.5


def _norm_quantity(quantity):
    """Normalise a user-supplied quantity name to 'V', 'I', or None."""
    if quantity is None:
        return None
    key = str(quantity).upper()
    if key in ('V', 'VOLTAGE', 'VOLTAGES'):
        return 'V'
    if key in ('I', 'CURRENT', 'CURRENTS'):
        return 'I'
    raise ValueError(f"quantity must be 'V' or 'I', got {quantity!r}.")


def _series_from_port(obj, quantity):
    """``(times, values, unit, stagger)`` from a self-recording port, or None."""
    if not (hasattr(obj, 'voltages') and hasattr(obj, 'currents')):
        return None
    key = _norm_quantity(quantity)
    if key == 'V':
        return obj.times, obj.voltages, 'V', _STAGGER_E
    if key == 'I':
        return obj.times, obj.currents, 'A', _STAGGER_H
    raise ValueError(
        f"{type(obj).__name__} records both V and I; say which one: "
        f"spectrum(port, 'V') or spectrum(port, 'I').")


# Monitors that record one named quantity: class name -> (kind, unit, stagger).
_MONITOR_KINDS = {
    'VoltageMonitor': ('V', 'V', _STAGGER_E),
    'CurrentMonitor': ('I', 'A', _STAGGER_H),
}


def _series_from_monitor(obj, quantity):
    """``(times, values, unit, stagger)`` from a monitor, or None.

    A monitor records exactly one quantity, so ``quantity`` is redundant — but
    naming it (``impedance((vmon, 'V'), (imon, 'I'))``) is natural enough to
    allow, as long as it agrees with what the monitor actually holds.
    """
    if not (hasattr(obj, 'times') and hasattr(obj, 'values')):
        return None
    known = _MONITOR_KINDS.get(type(obj).__name__)
    if known is None:
        # FieldProbe and anything else field-shaped: staggered by which field
        # it is. 'V'/'I' means nothing here, so an explicit quantity is a
        # mistake rather than a redundancy.
        comp = str(getattr(obj, 'component', 'E'))
        if quantity is not None:
            raise ValueError(
                f"{type(obj).__name__} records {comp!r}, not a port quantity; "
                f"drop the {quantity!r} argument.")
        stagger = _STAGGER_H if comp.strip('|')[0].upper() == 'H' else _STAGGER_E
        return obj.times, obj.values, comp, stagger
    kind, unit, stagger = known
    asked = _norm_quantity(quantity)
    if asked is not None and asked != kind:
        raise ValueError(
            f"{type(obj).__name__} records {kind}, but {quantity!r} was asked "
            f"for.")
    return obj.times, obj.values, unit, stagger


def _as_series(data, quantity):
    """Normalise any accepted input to ``(times, values, unit, stagger)``.

    Accepts a port, a monitor, or a ``(times, values)`` pair. A raw pair has no
    provenance, so its stagger is 0 — pass ``stagger=`` if it is H-derived.
    """
    for adapter in (_series_from_port, _series_from_monitor):
        got = adapter(data, quantity)
        if got is not None:
            return got
    if isinstance(data, (tuple, list)) and len(data) == 2:
        t, v = data
        return t, v, '', 0.0
    raise TypeError(
        "spectrum() takes a monitor, a port, or a (times, values) pair; got "
        f"{type(data).__name__}.")


def _uniform_dt(times) -> float:
    """The sampling interval of ``times``, checked for uniformity.

    Every recorder in the package samples every step, so the intervals are
    identical to round-off. A non-uniform record (a monitor attached partway
    through a run, two runs concatenated) would silently produce a wrong
    frequency axis, so it is rejected rather than averaged over.
    """
    t = np.asarray(times, dtype=float)
    if t.size < 2:
        raise ValueError("Need at least two samples to transform.")
    d = np.diff(t)
    dt = float(d.mean())
    if dt <= 0.0:
        raise ValueError("Timestamps must be strictly increasing.")
    if np.max(np.abs(d - dt)) > 1e-6 * dt:
        raise ValueError(
            "Sample times are not uniformly spaced — spectrum() assumes a "
            "constant dt. Was the record built from more than one run?")
    return dt


# ======================================================================= #
# Windows
# ======================================================================= #

def _window(name, n: int, alpha: float, tau: float) -> np.ndarray:
    """Window of length ``n``. Implemented here to keep this module numpy-only."""
    if name is None or name == 'none' or name == 'boxcar':
        return np.ones(n)
    k = np.arange(n)
    if name == 'hann':
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * k / max(n - 1, 1)))
    if name == 'hamming':
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * k / max(n - 1, 1))
    if name == 'tukey':
        # Cosine-tapered box: flat over the middle, Hann lobes over a fraction
        # ``alpha`` of the length. alpha=0 is a box, alpha=1 a Hann.
        if alpha <= 0.0:
            return np.ones(n)
        if alpha >= 1.0:
            return _window('hann', n, alpha, tau)
        w = np.ones(n)
        edge = int(np.floor(alpha * (n - 1) / 2.0)) + 1
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
        w[:edge] = ramp
        w[n - edge:] = ramp[::-1]
        return w
    if name == 'exponential':
        # exp(-t/tau) with tau in units of the record length. The natural choice
        # for a decaying resonator: it forces the tail to zero while leaving the
        # early, high-amplitude part almost untouched, at the cost of broadening
        # every resonance by 1/(2π·tau·T).
        return np.exp(-k / (tau * max(n - 1, 1)))
    raise ValueError(
        f"Unknown window {name!r}; expected one of None, 'hann', 'hamming', "
        f"'tukey', 'exponential'.")


def _warn_if_undecayed(values: np.ndarray, tail_frac: float, tol: float,
                       windowed: bool) -> None:
    """Warn when the record ends while the signal is still going.

    Compares the largest excursion in the final ``tail_frac`` of the record
    against the largest anywhere. A response that has rung down leaves a tail of
    numerical dust; one that has not leaves a tail comparable to the peak, and
    the transform of that is a rectangular-windowed sinusoid whose resonances
    are smeared across neighbouring bins.
    """
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak == 0.0:
        return
    n_tail = max(int(round(tail_frac * values.size)), 1)
    tail = float(np.max(np.abs(values[-n_tail:])))
    if tail <= tol * peak:
        return
    advice = ("run longer so it rings down"
              if windowed else
              "run longer so it rings down, or pass window='tukey' (or "
              "'exponential') to suppress the discontinuity")
    warnings.warn(
        f"Time series has not decayed: the last {100 * tail_frac:.0f}% of the "
        f"record still reaches {tail / peak:.1%} of the peak (tolerance "
        f"{tol:.1%}). Truncating there smears sharp features across "
        f"neighbouring bins — {advice}.",
        RuntimeWarning, stacklevel=3)


# ======================================================================= #
# The transform
# ======================================================================= #

def spectrum(data, quantity=None, *, window=None, alpha: float = 0.1,
             tau: float = 0.25, pad: int = 1, stagger=None, subtract_mean=False,
             warn_undecayed: bool = True, tail_frac: float = 0.05,
             decay_tol: float = 0.01, label=None) -> Spectrum:
    """
    Fourier transform of a recorded time series.

    Parameters
    ----------
    data : monitor | port | (times, values)
        What to transform. A :class:`~wavesim.monitors.VoltageMonitor`,
        :class:`~wavesim.monitors.CurrentMonitor` or
        :class:`~wavesim.monitors.FieldProbe`; a port that self-records
        (``LineSource``, ``TEMPort``, ``ModalPort``, ``SpicePort``); or a pair
        of arrays.
    quantity : {'V', 'I'}, optional
        Required for a port, which records both. Omit for a monitor.
    window : {None, 'hann', 'hamming', 'tukey', 'exponential'}
        Taper applied before transforming. ``None`` (the default) transforms the
        record as-is, which is correct for a signal that has decayed and is the
        only choice that preserves absolute amplitudes. A window suppresses the
        leakage from an undecayed tail, but it broadens every feature and scales
        the magnitude down by the window's mean — so use one for reading
        resonant *frequencies* off a truncated record, not for reading
        amplitudes.

        **Which one, for an FDTD port record.** These records are front-loaded:
        the drive arrives in the first few percent of the run and everything
        after it is decay. A symmetric taper (``'hann'``, ``'hamming'``) is
        built for a signal centred in its record and here lands its steep rising
        edge right on top of the excitation — it does not merely rescale the
        series, it reshapes it, and a ratio taken through one shifts by tens of
        percent (measured: 18% on a Hann-windowed R∥C in
        ``tests/test_spectrum.py``). Prefer ``'tukey'``, which is flat over the
        record and tapers only the last few percent where the ringing actually
        is, or ``'exponential'``. Both leave a ratio essentially untouched.

        Ratios (Z, Y, H) are otherwise safe as long as both series get the same
        window, which :func:`impedance` and friends enforce.
    alpha : float
        Tapered fraction for ``window='tukey'`` (0 = boxcar, 1 = Hann). The
        default 0.1 tapers 5% at each end.
    tau : float
        Decay constant for ``window='exponential'``, in units of the record
        length. The default 0.25 brings the record's end down by e⁻⁴.
    pad : int
        Zero-pad to ``pad`` times the record length. Interpolates the frequency
        axis to a finer grid; it adds no information and cannot resolve two
        features the record length does not already separate, but it does make a
        peak easier to locate. Zeros are appended after windowing, so pad only a
        decayed or windowed record.
    stagger : float, optional
        Override the inferred sample-time offset, in units of dt (see the module
        docstring). ``0`` for an E-derived quantity, ``-0.5`` for H-derived.
    subtract_mean : bool
        Remove the record's mean before transforming. Kills the DC bin and the
        leakage a nonzero offset spreads into the lowest few bins. Off by
        default because for a port the DC value is often the answer you want.
    warn_undecayed : bool
        Emit a ``RuntimeWarning`` when the record ends mid-ring. See
        ``tail_frac`` / ``decay_tol``.
    tail_frac, decay_tol : float
        The decay check: warn if the peak excursion over the final
        ``tail_frac`` of the record exceeds ``decay_tol`` times the overall peak.
    label : str, optional
        Name for plots. Defaults to the quantity or component name.

    Returns
    -------
    Spectrum
    """
    times, values, unit, inferred = _as_series(data, quantity)
    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"Expected a 1D time series, got shape {v.shape}.")
    dt = _uniform_dt(times)
    if stagger is None:
        stagger = inferred

    if warn_undecayed:
        _warn_if_undecayed(v, tail_frac, decay_tol, window is not None)

    if subtract_mean:
        v = v - v.mean()
    v = v * _window(window, v.size, alpha, tau)

    n = int(v.size * pad)
    if n < v.size:
        raise ValueError(f"pad must be >= 1, got {pad}.")
    freqs = np.fft.rfftfreq(n, dt)
    X = np.fft.rfft(v, n=n) * dt

    # Undo the sampling-time offset: a series sampled at t + stagger·dt but
    # stamped t transforms to X(f)·exp(+j2πf·stagger·dt), so divide it out.
    if stagger:
        X = X * np.exp(-2j * np.pi * freqs * (stagger * dt))

    return Spectrum(freqs=freqs, values=X, dt=dt,
                    label=label if label is not None else unit, unit=unit)


# ======================================================================= #
# Ratios — transfer function, Z(f), Y(f)
# ======================================================================= #

def _in_band(num: Spectrum, den: Spectrum, floor: float) -> np.ndarray:
    """Bins where *both* spectra carry real signal, as a boolean mask.

    A ratio is only meaningful where the excitation put energy. Outside its band
    both spectra are round-off, and their quotient is a number with no physical
    content that nonetheless plots at full scale. Each spectrum is measured
    against its own peak so the test is scale-free.
    """
    a, b = np.abs(num.values), np.abs(den.values)
    return (a >= floor * np.nanmax(a)) & (b >= floor * np.nanmax(b))


def _unpack(arg):
    """Split an optional ``(data, quantity)`` pair into ``(data, quantity)``."""
    if isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[1], str):
        return arg
    return arg, None


def _ratio(num, den, *, quantity_num, quantity_den, floor, band, label, unit,
           **kw) -> Spectrum:
    """Shared body of transfer_function / impedance / admittance.

    ``num`` and ``den`` may be spectra already, or anything :func:`spectrum`
    accepts — in which case both are transformed with the *same* options, which
    is what makes windowing safe for a ratio.
    """
    if not isinstance(num, Spectrum):
        num = spectrum(num, quantity_num, **kw)
    if not isinstance(den, Spectrum):
        den = spectrum(den, quantity_den, **kw)
    if num.freqs.shape != den.freqs.shape or not np.allclose(num.freqs, den.freqs):
        raise ValueError(
            "Numerator and denominator are on different frequency axes — they "
            "must come from records of the same length and dt (and the same "
            "pad=).")

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num.values / den.values
    ratio = np.where(_in_band(num, den, floor), ratio, np.nan + 0j)

    out = Spectrum(freqs=num.freqs, values=ratio, dt=num.dt,
                   label=label, unit=unit)
    return out.band(*band) if band is not None else out


def transfer_function(out, in_, *, floor: float = 1e-3, band=None,
                      label: str = 'H', **kw) -> Spectrum:
    """
    H(f) = out(f) / in(f) — the response at one place per unit drive at another.

    The two-port measurement this module was written for: drive port 1 with a
    broadband pulse, record something at port 2, and divide. Both arguments may
    be monitors/ports (transformed here with identical options) or ready-made
    :class:`Spectrum` objects.

    Parameters
    ----------
    out, in_ : monitor | port | Spectrum
        Response and drive. For a port, pass a ``(port, 'V')`` pair — or
        transform it yourself and pass the Spectrum.
    floor : float
        Out-of-band cutoff: bins where either spectrum falls below ``floor``
        times its own peak are masked to NaN, rather than reporting the ratio of
        two round-off numbers. Raise it if the result is still noisy at the band
        edges; lower it to see further into the skirts of the excitation.
    band : (f_lo, f_hi), optional
        Restrict the result to this range (Hz).
    **kw
        Passed to :func:`spectrum` for both series (``window=``, ``pad=``, …).

    Returns
    -------
    Spectrum
        Dimensionless, with NaN outside the excitation band.
    """
    num, den = _unpack(out), _unpack(in_)
    return _ratio(num[0], den[0], quantity_num=num[1], quantity_den=den[1],
                  floor=floor, band=band, label=label, unit='', **kw)


def _port_pair(v, i, name):
    """Resolve the ``(V, I)`` argument pair, splitting a lone port into both."""
    if i is not None:
        return _unpack(v), _unpack(i)
    if isinstance(v, Spectrum) or not (hasattr(v, 'voltages')
                                       and hasattr(v, 'currents')):
        raise TypeError(
            f"{name}() with one argument needs a port that records both V and "
            f"I (LineSource, TEMPort, ModalPort, SpicePort); got "
            f"{type(v).__name__}. Pass the voltage and current separately.")
    return (v, 'V'), (v, 'I')


def impedance(v, i=None, *, floor: float = 1e-3, band=None, label: str = 'Z',
              **kw) -> Spectrum:
    """
    Z(f) = V(f) / I(f) at a port.

    Parameters
    ----------
    v : port | monitor | Spectrum
        A self-recording port on its own supplies both V and I:
        ``impedance(port)``. Otherwise this is the voltage and ``i`` the current.
    i : monitor | Spectrum, optional
        The current, when it comes from somewhere other than ``v``.
    floor, band, **kw
        As :func:`transfer_function`.

    Returns
    -------
    Spectrum
        Ohms, with NaN outside the excitation band.

    Notes
    -----
    The sign convention is the recorders': a ``VoltageMonitor`` integrated from
    the "+" conductor to the reference, and a current positive *into* the
    structure, give a passive Z with ``Re Z >= 0``. A Z that comes out
    consistently negative means one of the two is reversed — flip the monitor
    path rather than the sign of the answer, so ``V·I`` stays the delivered
    power.

    The port's own cell capacitance is in this number; see the module docstring.
    """
    num, den = _port_pair(v, i, 'impedance')
    return _ratio(num[0], den[0], quantity_num=num[1], quantity_den=den[1],
                  floor=floor, band=band, label=label, unit='Ω', **kw)


def admittance(v, i=None, *, floor: float = 1e-3, band=None, label: str = 'Y',
               **kw) -> Spectrum:
    """
    Y(f) = I(f) / V(f) at a port — the reciprocal of :func:`impedance`.

    Worth computing directly rather than inverting Z: the masking then keys off
    the same in-band test, and a near-open port whose Z runs to huge numbers
    reads as a small, well-conditioned Y. Arguments are as :func:`impedance`.

    Returns
    -------
    Spectrum
        Siemens, with NaN outside the excitation band.
    """
    den, num = _port_pair(v, i, 'admittance')
    return _ratio(num[0], den[0], quantity_num=num[1], quantity_den=den[1],
                  floor=floor, band=band, label=label, unit='S', **kw)
