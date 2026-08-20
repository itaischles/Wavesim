"""Fourier transforms of recorded time series, and Z(f) / Y(f).

Most of these are pure-signal tests: synthetic V(t) and I(t) with a known
analytic transform, so the assertions can be exact rather than "close enough for
a field solve". The two that matter most:

* **The half-step de-stagger.** E and H are recorded half a timestep apart but
  stamped identically, so a naive V(f)/I(f) carries a spurious ``exp(+jπ·f·dt)``.
  :func:`test_stagger_recovers_the_true_phase` feeds in a series *built* with
  that offset and checks the transform puts it back; the companion test checks
  the uncorrected ratio is wrong by exactly the expected factor, so the
  correction can't be quietly deleted without a failure.

* **The decay warning.** A resonance that is still ringing at the end of the run
  is the dominant error in this whole workflow, and it is invisible in the
  output — the transform of a truncated sinusoid looks perfectly plausible. The
  warning is the only thing standing between the user and a smeared Q.

:func:`test_a_lossless_gap_reads_as_lossless` is the one real FDTD run, and the
only test here that can catch the stagger being *inferred* wrongly rather than
*applied* wrongly: a port across a vacuum gap is a lossless capacitor, so its Z
must sit at -90°, and the half-step error turns that tilt into a resistance the
structure does not have.
"""
import numpy as np
import pytest

import wavesim as ws
from wavesim.spectrum import Spectrum, spectrum, transfer_function, impedance, \
    admittance


DT = 1e-12
N = 4096


def _t(n=N, dt=DT):
    return np.arange(n) * dt


def _decayed_pulse(n=N, dt=DT, t0=400e-12, width=40e-12, amp=1.0):
    """A Gaussian: decayed at both ends, so no window is needed."""
    t = _t(n, dt)
    return t, amp * np.exp(-0.5 * ((t - t0) / width) ** 2)


class _FakeMonitor:
    """Stand-in with a monitor's duck-type, so the adapters can be tested
    without running a solve. The class *name* is what selects the stagger."""
    def __init__(self, times, values):
        self.times, self.values = list(times), list(values)


class VoltageMonitor(_FakeMonitor):
    pass


class CurrentMonitor(_FakeMonitor):
    pass


class _FakePort:
    """Stand-in for LineSource/TEMPort/ModalPort/SpicePort's port record."""
    def __init__(self, times, voltages, currents):
        self.times = list(times)
        self.voltages = list(voltages)
        self.currents = list(currents)


# ======================================================================= #
# The transform itself
# ======================================================================= #

def test_gaussian_matches_its_analytic_transform():
    # FT{A·exp(-½((t-t0)/w)²)} = A·w·√(2π)·exp(-½(2πf w)²)·exp(-j2πf t0)
    t0, w, amp = 400e-12, 40e-12, 3.0
    t, v = _decayed_pulse(t0=t0, width=w, amp=amp)
    S = spectrum((t, v))

    f = S.freqs
    exact = (amp * w * np.sqrt(2 * np.pi)
             * np.exp(-0.5 * (2 * np.pi * f * w) ** 2)
             * np.exp(-2j * np.pi * f * t0))
    # Compare where the pulse has real content; the skirts are below round-off.
    keep = np.abs(exact) > 1e-5 * np.abs(exact).max()
    assert np.allclose(S.values[keep], exact[keep], rtol=2e-3, atol=0)


def test_scaling_is_per_hertz_and_independent_of_dt():
    # The same physical pulse sampled twice as finely must transform to the
    # same density — that is the point of the dt factor.
    coarse = spectrum(_decayed_pulse(n=N, dt=DT))
    fine = spectrum(_decayed_pulse(n=2 * N, dt=DT / 2))
    probe = np.linspace(0.0, 5e9, 25)
    assert np.allclose(coarse.at(probe), fine.at(probe), rtol=1e-3)


def test_dc_bin_is_the_time_integral():
    t, v = _decayed_pulse()
    S = spectrum((t, v))
    assert S.freqs[0] == 0.0
    assert S.values[0].real == pytest.approx(np.trapezoid(v, t), rel=1e-6)


def test_pad_refines_the_axis_without_changing_the_curve():
    # Zero-padding interpolates the same underlying transform: every original
    # bin must survive untouched, with three new ones between each pair.
    t, v = _decayed_pulse()
    plain, padded = spectrum((t, v)), spectrum((t, v), pad=4)
    assert len(padded) == pytest.approx(4 * len(plain), rel=0.01)
    assert np.allclose(padded.freqs[::4], plain.freqs, rtol=1e-12)
    assert np.allclose(padded.values[::4], plain.values, rtol=1e-9, atol=1e-24)


def test_non_uniform_sampling_is_rejected():
    t, v = _decayed_pulse()
    t = t.copy()
    t[N // 2:] += 3 * DT                      # a gap, as if two runs were joined
    with pytest.raises(ValueError, match="uniformly spaced"):
        spectrum((t, v))


def test_subtract_mean_removes_the_dc_bin():
    t, v = _decayed_pulse()
    v = v + 0.5
    # The offset itself trips the decay check (a record that ends at 0.5 has
    # not settled) — true, and beside the point here.
    S = spectrum((t, v), subtract_mean=True, warn_undecayed=False)
    assert abs(S.values[0]) < 1e-12


# ======================================================================= #
# Spectrum views
# ======================================================================= #

def test_spectrum_views_and_selection():
    t, v = _decayed_pulse()
    S = spectrum((t, v))
    assert np.allclose(S.magnitude, np.abs(S.values))
    nz = S.magnitude > 0                       # db floors exact zeros, not -inf
    assert np.allclose(S.db[nz], 20 * np.log10(S.magnitude[nz]))
    assert np.all(np.isfinite(S.db))
    sub = S.band(1e9, 4e9)
    assert sub.freqs.min() >= 1e9 and sub.freqs.max() <= 4e9
    assert len(sub) < len(S)


def test_at_interpolates_between_bins():
    t, v = _decayed_pulse()
    S = spectrum((t, v))
    f = 0.5 * (S.freqs[10] + S.freqs[11])
    assert S.at(f) == pytest.approx(0.5 * (S.values[10] + S.values[11]), rel=1e-12)
    assert np.isnan(S.at(-1e9))               # outside the axis, not extrapolated


# ======================================================================= #
# The half-step stagger — the correctness core of this module
# ======================================================================= #

def _rc_series(R=25.0, C=1.3e-12, stagger_i=-0.5):
    """V(t) and I(t) through a known R∥C, with I sampled ``stagger_i`` steps
    off V's stamp — i.e. exactly how the solver records them. The R∥C corner
    (1/2πRC ≈ 4.9 GHz) sits inside the drive's band on purpose: a Z that is flat
    across the band has no phase for the stagger test to get wrong.

    Built in the frequency domain so both series are consistent to round-off:
    pick V(t) as a Gaussian, get I from Y(f)·V(f), and re-sample I at the offset
    time by applying the corresponding phase ramp before transforming back.
    """
    t, v = _decayed_pulse()
    f = np.fft.rfftfreq(v.size, DT)
    Y = 1.0 / R + 2j * np.pi * f * C
    # I(t + stagger·dt): a time shift is a phase ramp on the spectrum.
    I = np.fft.irfft(np.fft.rfft(v) * Y * np.exp(2j * np.pi * f * stagger_i * DT),
                     n=v.size)
    return t, v, I, R, C


def test_stagger_recovers_the_true_phase():
    t, v, i, R, C = _rc_series()
    Z = impedance(VoltageMonitor(t, v), CurrentMonitor(t, i))

    f = Z.freqs
    exact = 1.0 / (1.0 / R + 2j * np.pi * f * C)
    keep = np.isfinite(Z.values)
    assert keep.sum() > 40
    assert np.allclose(Z.values[keep] / exact[keep], 1.0, rtol=1e-9, atol=1e-9)


def test_without_the_stagger_the_phase_is_wrong_by_a_known_factor():
    # Same data, stagger forced to zero: the answer must be off by exactly the
    # half-step phase ramp. If someone deletes the correction, this fails too.
    t, v, i, R, C = _rc_series()
    Z_ok = impedance(VoltageMonitor(t, v), CurrentMonitor(t, i))
    Z_raw = impedance((t, v), (t, i))         # bare arrays carry no provenance

    keep = np.isfinite(Z_ok.values) & np.isfinite(Z_raw.values)
    ramp = np.exp(1j * np.pi * Z_ok.freqs * DT)
    assert np.allclose(Z_raw.values[keep], (Z_ok.values * ramp)[keep], rtol=1e-6)

    # And it is a real error, not a formality. Over this drive's band it is a
    # couple of degrees; it grows linearly with f, to 90° at Nyquist.
    err = np.degrees(np.abs(np.angle(Z_raw.values[keep] / Z_ok.values[keep])))
    assert err.max() > 2.0
    assert err.max() == pytest.approx(180.0 * Z_ok.freqs[keep].max() * DT, rel=1e-6)


def test_stagger_can_be_overridden():
    t, v, i, R, C = _rc_series()
    forced = impedance((t, v), ((t, i)), stagger=-0.5)
    # Overriding applies to *both* series, so the relative offset is unchanged.
    plain = impedance((t, v), (t, i))
    keep = np.isfinite(plain.values)
    assert np.allclose(forced.values[keep], plain.values[keep], rtol=1e-9)


def test_current_monitor_and_port_current_agree():
    # The port record and the CurrentMonitor must be assigned the same stagger,
    # or the same physical measurement would read differently depending on which
    # object it came from.
    t, v, i, R, C = _rc_series()
    from_monitors = impedance(VoltageMonitor(t, v), CurrentMonitor(t, i))
    from_port = impedance(_FakePort(t, v, i))
    keep = np.isfinite(from_port.values)
    assert np.allclose(from_monitors.values[keep], from_port.values[keep])


# ======================================================================= #
# Ratios
# ======================================================================= #

def test_transfer_function_of_a_pure_delay():
    t, v = _decayed_pulse()
    delay = 37 * DT
    out = np.interp(t - delay, t, v, left=0.0, right=0.0)
    H = transfer_function((t, out), (t, v))

    keep = np.isfinite(H.values)
    assert np.allclose(np.abs(H.values[keep]), 1.0, atol=2e-3)
    # Linear phase with slope -2π·delay.
    slope = np.polyfit(H.freqs[keep], np.unwrap(np.angle(H.values[keep])), 1)[0]
    assert slope == pytest.approx(-2 * np.pi * delay, rel=2e-3)


def test_admittance_is_the_reciprocal_of_impedance():
    t, v, i, R, C = _rc_series()
    Z = impedance(_FakePort(t, v, i))
    Y = admittance(_FakePort(t, v, i))
    keep = np.isfinite(Z.values) & np.isfinite(Y.values)
    assert np.allclose(Y.values[keep], 1.0 / Z.values[keep], rtol=1e-9)
    assert Z.unit == 'Ω' and Y.unit == 'S'


def test_out_of_band_bins_are_masked_not_garbage():
    t, v, i, R, C = _rc_series()
    Z = impedance(_FakePort(t, v, i))
    # The drive is a 40 ps Gaussian: nothing above ~20 GHz. Those bins must be
    # NaN rather than a ratio of two round-off numbers.
    assert np.all(np.isnan(Z.values[Z.freqs > 40e9]))
    assert np.all(np.isfinite(Z.values[Z.freqs < 4e9]))
    # In-band values stay finite and physical.
    assert np.nanmax(np.abs(Z.values)) < 2 * R


def test_masked_bins_are_nan_in_both_parts():
    # ``np.nan + 0j`` is complex(nan, 0.0) — a masked bin written that way keeps
    # a perfectly good zero imaginary part, and an out-of-band reactance then
    # draws as X = 0, which reads as a resonance instead of as no data.
    t, v, i, R, C = _rc_series()
    Z = impedance(_FakePort(t, v, i))
    out = Z.freqs > 40e9
    assert np.all(np.isnan(Z.values[out].real))
    assert np.all(np.isnan(Z.values[out].imag))
    assert np.all(np.isnan(Z.real[out])) and np.all(np.isnan(Z.imag[out]))


def test_series_are_labelled_by_quantity_not_by_unit():
    # 'I', not 'A': the label names the quantity and rides into plot legends.
    t, v, i, R, C = _rc_series()
    port = _FakePort(t, v, i)
    assert (spectrum(port, 'V').label, spectrum(port, 'V').unit) == ('V', 'V')
    assert (spectrum(port, 'I').label, spectrum(port, 'I').unit) == ('I', 'A')
    assert spectrum(CurrentMonitor(t, i)).label == 'I'
    assert spectrum(port, 'I', label='port 2').label == 'port 2'


def test_floor_controls_how_much_band_survives():
    t, v, i, R, C = _rc_series()
    wide = impedance(_FakePort(t, v, i), floor=1e-8)
    narrow = impedance(_FakePort(t, v, i), floor=1e-1)
    assert np.isfinite(wide.values).sum() > np.isfinite(narrow.values).sum()


def test_band_restricts_the_result():
    t, v, i, R, C = _rc_series()
    Z = impedance(_FakePort(t, v, i), band=(1e9, 5e9))
    assert Z.freqs.min() >= 1e9 and Z.freqs.max() <= 5e9


def test_mismatched_frequency_axes_are_rejected():
    t, v = _decayed_pulse()
    t2, v2 = _decayed_pulse(n=N // 2)
    with pytest.raises(ValueError, match="different frequency axes"):
        transfer_function((t, v), (t2, v2))


# ======================================================================= #
# Truncation: the warning and the windows
# ======================================================================= #

def _ringing(n=N, dt=DT, f0=5e9, q_decay=0.0):
    t = _t(n, dt)
    return t, np.exp(-q_decay * t) * np.sin(2 * np.pi * f0 * t)


def test_undecayed_record_warns():
    t, v = _ringing()                          # never decays
    with pytest.warns(RuntimeWarning, match="has not decayed"):
        spectrum((t, v))


def test_decayed_record_does_not_warn():
    t, v = _decayed_pulse()
    with warnings_as_errors():
        spectrum((t, v))


def test_the_warning_can_be_switched_off():
    t, v = _ringing()
    with warnings_as_errors():
        spectrum((t, v), warn_undecayed=False)


def test_a_slowly_decaying_ring_still_warns():
    # Decays, but not by the end of the record: 1/e every quarter of the run.
    t, v = _ringing(q_decay=4.0 / (N * DT))
    with pytest.warns(RuntimeWarning, match="has not decayed"):
        spectrum((t, v))


# Far-out leakage each window is expected to leave, as a fraction of the
# rectangular transform's. The cosine tapers are worth an order of magnitude or
# more; 'exponential' is a mild taper aimed at a *decaying* resonator's tail
# rather than at leakage, and only halves it.
@pytest.mark.parametrize("window,budget", [('hann', 0.01), ('hamming', 0.1),
                                           ('tukey', 0.15), ('exponential', 0.6)])
def test_windows_suppress_truncation_leakage(window, budget):
    # A truncated sinusoid's rectangular transform leaks as 1/Δf; a tapered one
    # falls off faster. Measure the floor well away from the line.
    t, v = _ringing(f0=5e9)
    plain = spectrum((t, v), warn_undecayed=False)
    tapered = spectrum((t, v), window=window, warn_undecayed=False)
    far = plain.freqs > 15e9
    assert np.max(tapered.magnitude[far]) < budget * np.max(plain.magnitude[far])


def test_a_flat_window_preserves_the_ratio():
    # A window multiplies both series by the same taper, so where the taper is
    # flat the ratio is untouched. This is why impedance() insists on applying
    # the same options to numerator and denominator.
    t, v, i, R, C = _rc_series()
    plain = impedance(_FakePort(t, v, i))
    tapered = impedance(_FakePort(t, v, i), window='tukey')
    keep = np.isfinite(plain.values) & np.isfinite(tapered.values)
    assert keep.sum() > 40
    assert np.allclose(tapered.values[keep], plain.values[keep], rtol=1e-5)


def test_a_symmetric_window_wrecks_a_front_loaded_record():
    # The documented trap: an FDTD port record is front-loaded (drive first,
    # decay after), so a Hann taper's rising edge lands on the excitation itself
    # and reshapes it rather than rescaling it. Pinned so the docstring's
    # recommendation of tukey/exponential can't drift away from the behaviour.
    t, v, i, R, C = _rc_series()
    plain = impedance(_FakePort(t, v, i))
    hann = impedance(_FakePort(t, v, i), window='hann')
    keep = np.isfinite(plain.values) & np.isfinite(hann.values)
    assert np.max(np.abs(hann.values[keep] / plain.values[keep] - 1.0)) > 0.1


def test_tukey_spans_boxcar_to_hann():
    n = 512
    from wavesim.spectrum import _window
    assert np.allclose(_window('tukey', n, 0.0, 0.25), np.ones(n))
    assert np.allclose(_window('tukey', n, 1.0, 0.25), _window('hann', n, 1.0, 0.25))


def test_unknown_window_is_rejected():
    t, v = _decayed_pulse()
    with pytest.raises(ValueError, match="Unknown window"):
        spectrum((t, v), window='blackman-harris-7')


# ======================================================================= #
# Input adapters
# ======================================================================= #

def test_port_needs_a_named_quantity():
    t, v, i, R, C = _rc_series()
    with pytest.raises(ValueError, match="say which one"):
        spectrum(_FakePort(t, v, i))


def test_monitor_rejects_a_contradictory_quantity():
    t, v = _decayed_pulse()
    with pytest.raises(ValueError, match="records V"):
        spectrum(VoltageMonitor(t, v), 'I')


def test_impedance_of_one_argument_needs_a_port():
    t, v = _decayed_pulse()
    with pytest.raises(TypeError, match="records both V and I"):
        impedance(VoltageMonitor(t, v))


def test_field_probe_stagger_follows_the_field():
    t, v = _decayed_pulse()
    probe_e = ws.FieldProbe('Ez', 0.0, 0.0, 0.0)
    probe_h = ws.FieldProbe('Hy', 0.0, 0.0, 0.0)
    for p in (probe_e, probe_h):
        p.times, p.values = list(t), list(v)
    ratio = transfer_function(probe_h, probe_e)
    keep = np.isfinite(ratio.values)
    # Identical samples, but H is half a step earlier: the ratio is the ramp.
    expected = np.exp(1j * np.pi * ratio.freqs * DT)
    assert np.allclose(ratio.values[keep], expected[keep], rtol=1e-9)


def test_rejects_unrecognised_input():
    with pytest.raises(TypeError, match="monitor, a port"):
        spectrum(object())


# ======================================================================= #
# A real FDTD run: the port timing convention
# ======================================================================= #

@pytest.mark.slow
def test_a_lossless_gap_reads_as_lossless():
    """The port timing convention, measured on a live run.

    Two plates a cell apart, bridged by a driven port. Below the structure's
    first resonance that is a lossless capacitor, so Z must sit at -90°: all
    reactance, no resistance. The half-step stagger tilts it by π·f·dt, which
    manufactures a *resistance* ``|Z|·sin(π·f·dt)`` out of nothing — at 3 GHz on
    this grid, a 0.4 Ω series loss in a structure that has none. That is the
    failure mode this matters for: fitting a lumped model to the uncorrected Z
    hands you a spurious series R (or a dielectric loss tangent) that is purely
    an artefact of when the two recorders happened to sample.

    So the test asserts the corrected phase is -90° to a twentieth of a degree,
    and separately that the *un*corrected data would fail — pinning the
    inferred stagger to the one the solver actually uses, not merely to a
    self-consistent story about it.
    """
    dl = 0.5e-3
    grid = ws.set_vacuum(ws.create_grid(Nx=40, Ny=40, Nz=40, dx=dl, dy=dl, dz=dl))
    # Two plates normal to z, one cell of vacuum between them.
    grid = ws.set_box(grid, 12 * dl, 28 * dl, 12 * dl, 28 * dl, 19 * dl, 20 * dl,
                      1.0, pec=True, name='bot')
    grid = ws.set_box(grid, 12 * dl, 28 * dl, 12 * dl, 28 * dl, 21 * dl, 22 * dl,
                      1.0, pec=True, name='top')

    # Endpoints on the conductor surfaces, as LineSource requires.
    port = ws.LineSource(p0=(20 * dl, 20 * dl, 20 * dl),
                         p1=(20 * dl, 20 * dl, 21 * dl),
                         voltage=ws.GaussianPulse.for_fmax(30e9),
                         resistance=50.0)
    sim = ws.Simulation(grid, cpml=ws.init_cpml(grid, d_pml=8))
    sim.add_source(port)
    sim.run(3000)

    # Well below the plate resonance near 7 GHz, where the gap is still a plain
    # capacitor. No warn_undecayed override: this run does ring down, and if it
    # ever stops doing so the warning should surface rather than be silenced.
    band = (0.7e9, 3e9)
    Z = impedance(port, band=band)
    keep = np.isfinite(Z.values)
    assert keep.sum() >= 5

    z = Z.values[keep]
    assert np.allclose(np.degrees(np.angle(z)), -90.0, atol=0.05)
    assert np.max(z.real / np.abs(z)) < 2e-3

    # The same record read with the stagger switched off shows the artificial
    # loss, an order of magnitude larger.
    raw = impedance(port, band=band, stagger=0.0, warn_undecayed=False)
    r = raw.values[np.isfinite(raw.values)]
    assert np.max(r.real / np.abs(r)) > 5e-3


# ----------------------------------------------------------------------- #

import contextlib
import warnings as _warnings


@contextlib.contextmanager
def warnings_as_errors():
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        yield
