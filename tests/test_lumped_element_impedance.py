"""What impedance a :class:`~wavesim.sources.LineSource` element presents to the field.

:mod:`tests.test_lumped_rlc` proves the element's *recorded* port traces obey its
terminal law exactly, and deliberately leaves open the separate question this
module answers: what the surrounding field actually sees. That needed a spectral
reflection sweep, because the two are not the same measurement — the recorded
V(t)/I(t) are exact whatever the field sees.

The method
----------
A 2D (``Nz=1``) parallel-plate line between real PEC plates, PML-matched at both
ends, with the element shunting the line mid-span. An element-free reference run
is subtracted, which isolates the scattered wave; taking the monitor at the
element's *own* plane then makes the measurement free of any de-embedding, since
a shunt discontinuity radiates the same scattered voltage both ways:

    Gamma = V_scat/V_inc = -Z_env*y / (1 + Z_env*y)   ->   Z_env*y = -Gamma/(1 + Gamma)

for a shunt of admittance ``y``. ``Z_env`` — what the shunt sees looking both ways,
nominally Z0/2 — is not assumed but *calibrated out* by two resistor standards:
``Z_env*y_i = Z_env*(G_i + s*C_par)`` is linear in ``G_i``, so a pair of resistors
determines both ``Z_env(f)`` and any parasitic shunt capacitance ``C_par(f)``.
Anything the element adds beyond its own admittance has to show up in ``C_par``.

What it finds
-------------
The element contributes **exactly its own companion admittance**, to four
significant figures across 4-30 GHz. On this line (Z0 = 94.2 ohm, kappa = 65.9
ohm, cell capacitance C_cell = 7.08 fF) that rejects both descriptions that had
been in circulation:

* **No series kappa/2.** A 25 ohm resistor presenting ``R + kappa/2`` = 58 ohm
  would read 17.3 mS; it reads 40.000 mS. (The ``Z + kappa/2`` wording in
  LineSource's docstring was wrong and is now corrected; TEMPort's spectral sweep
  had already reached the same conclusion by a different route.)
* **No parallel cell capacitance either.** ``C_par`` comes out at ~0.001 fF
  against a C_cell of 7.08 fF. The cell's capacitance is *background* — it is
  there with or without the element, in both runs, and so is not part of what the
  element contributes. A lumped element bridges its cell; it does not replace it.

The only residual is the trapezoidal companion's own frequency warp,
``s -> j(2/dt)tan(w*dt/2)``, which :meth:`LumpedNetwork.impedance_at` already
documents as the scheme's exact behaviour: a 100 fF capacitor reads 0.064 fF high
at 30 GHz, matching the warp's ``(w*dt/2)^2/3`` to two digits, and a 1 nH inductor
shows a *frequency-flat* 0.018 fF equivalent, which a real shunt capacitance
cannot be.

Runtime is ~0.8 s per run and seven runs are cached across the module.
"""
import numpy as np
import pytest

import wavesim as ws

ETA0 = 376.730313668
EPS0 = 8.8541878128e-12

# --- the line -------------------------------------------------------------- #
DX = DY = 0.2e-3
DZ = 4.0e-3                     # the plate width; sets Z0 with the gap
NY = 9
J0, J1 = 2, 7                   # gap cells in y; the plates are the rest
GAP = (J1 - J0) * DY
# The path quadrature bins an edge by its *nearest* sample point, so endpoints
# half a cell inside the gap select exactly the five gap edges at full weight —
# no PEC edge in the path. Including one would both inflate kappa and let the
# injection charge an edge the PEC mask clears next step, which reads back as a
# spurious series capacitance at the port (~100 fF here).
YLO, YHI = (J0 - 0.5) * DY, (J1 - 0.5) * DY
NX, I_SRC, I_PORT, D_PML = 130, 25, 100, 12
NSTEPS = 2500
F_PK = 12e9                     # spectral peak of the (DC-free) drive

Z0 = ETA0 * GAP / DZ            # 94.18 ohm
C_CELL = EPS0 * DX * DZ / GAP   # 7.08 fF — the port cell's own gap capacitance
BAND = (4e9, 30e9)


def _dgauss(f_pk):
    """Gaussian derivative: DC-free, so no static charge is left on the line."""
    w = 1.0 / (2.0 * np.pi * f_pk)
    t0 = 4.0 * w
    return lambda t: -((t - t0) / w) * np.exp(-0.5 * ((t - t0) / w) ** 2)


def _path(i):
    return ((i * DX, YLO, 0.0), (i * DX, YHI, 0.0))


def _run(**elem):
    """Launch a pulse down the line; return the port-plane voltage trace."""
    grid = ws.create_grid(Nx=NX, Ny=NY, Nz=1, dx=DX, dy=DY, dz=DZ)
    ws.set_vacuum(grid)
    xs = NX * DX
    ws.set_box(grid, 0.0, xs, 0.0, J0 * DY, 0.0, DZ, eps_r=1.0, pec=True)
    ws.set_box(grid, 0.0, xs, J1 * DY, NY * DY, 0.0, DZ, eps_r=1.0, pec=True)
    sim = ws.Simulation(grid, cpml=ws.init_cpml(grid, d_pml=D_PML,
                                                faces=('x0', 'x1')))
    p0, p1 = _path(I_SRC)
    # An ideal impressed current is transparent to the returning scattered wave,
    # so the launcher itself never re-reflects and Gamma stays first-order.
    sim.add_source(ws.LineSource(p0=p0, p1=p1, current=_dgauss(F_PK)))
    if elem:
        q0, q1 = _path(I_PORT)
        sim.add_source(ws.LineSource(p0=q0, p1=q1, **elem))
    mon = sim.add_monitor(ws.VoltageMonitor(path=_path(I_PORT)))
    for _ in range(NSTEPS):
        sim.step()
    return np.asarray(mon.values), grid.dt


_CACHE: dict = {}


def _gamma(**elem):
    """Scattered/incident voltage ratio at the element's own plane."""
    key = tuple(sorted(elem.items()))
    if key not in _CACHE:
        if 'ref' not in _CACHE:
            v0, dt = _run()
            _CACHE['ref'] = (v0, np.fft.rfft(v0), np.fft.rfftfreq(NSTEPS, dt), dt)
        v0, V0, _, _ = _CACHE['ref']
        _CACHE[key] = np.fft.rfft(_run(**elem)[0] - v0) / V0
    return _CACHE[key]


def _freqs():
    _gamma(resistance=100.0)
    return _CACHE['ref'][2]


def _band():
    f = _freqs()
    return (f >= BAND[0]) & (f <= BAND[1])


def _calibrate():
    """(Z_env, C_par) per frequency, from two resistor standards.

    ``u = -Gamma/(1 + Gamma) = Z_env*(G + s*C_par)`` is linear in the standard's
    conductance, so two of them separate the environment from any parasitic shunt
    capacitance the element brings with it.
    """
    if 'cal' not in _CACHE:
        r1, r2 = 100.0, 400.0
        u = lambda g: -g / (1.0 + g)
        u1, u2 = u(_gamma(resistance=r1)), u(_gamma(resistance=r2))
        z_env = (u1 - u2) / (1.0 / r1 - 1.0 / r2)
        with np.errstate(divide='ignore', invalid='ignore'):
            c_par = (u1 / z_env - 1.0 / r1) / (2j * np.pi * _freqs())
        _CACHE['cal'] = (z_env, c_par)
    return _CACHE['cal']


def _admittance(**elem):
    """The element's own shunt admittance y(f), environment calibrated out."""
    g = _gamma(**elem)
    return (-g / (1.0 + g)) / _calibrate()[0]


# --------------------------------------------------------------------------- #
# The harness itself
# --------------------------------------------------------------------------- #

def test_the_calibration_recovers_the_line_it_was_run_on():
    """Z_env must come out as Z0/2, real — the guard on the whole measurement.

    Nothing downstream is trustworthy if the environment is not a matched line:
    a port path that strays onto a PEC edge, or plates built as boundary faces
    rather than material, both show up here as a large series reactance while
    leaving the extracted element admittance superficially plausible.
    """
    z_env = _calibrate()[0][_band()]
    assert np.allclose(z_env.real, 0.5 * Z0, rtol=0.02), (
        f"Z_env = {np.mean(z_env.real):.2f}, expected Z0/2 = {0.5 * Z0:.2f}")
    assert np.max(np.abs(z_env.imag)) < 0.05 * 0.5 * Z0, (
        f"Z_env is reactive (max |Im| = {np.max(np.abs(z_env.imag)):.2f} ohm)")


def test_the_reflection_being_measured_is_not_negligible():
    """Guard the null result: the resistors really do scatter."""
    g = np.abs(_gamma(resistance=25.0)[_band()])
    assert np.min(g) > 0.4, f"min |Gamma| = {np.min(g):.3f} — too weak to conclude from"


# --------------------------------------------------------------------------- #
# What the element presents
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("R", [25.0, 100.0, 400.0])
def test_resistor_presents_exactly_its_own_conductance(R):
    """y = 1/R across the band — no kappa/2 in series with it.

    kappa/2 is 33 ohm on this line, so the discredited ``R + kappa/2`` reading
    would be low by 57% at R = 25, 25% at R = 100 and 7.6% at R = 400. The
    measurement lands on 1/R to better than 0.5%, and the susceptance stays at
    the noise floor.
    """
    y = _admittance(resistance=R)[_band()]
    half_kappa = 0.5 * _CACHE['ref'][3] / C_CELL
    assert np.allclose(y.real, 1.0 / R, rtol=5e-3), (
        f"R={R}: measured {1.0 / np.mean(y.real):.2f} ohm")
    # ... and specifically not the series-parasitic reading: the data sits far
    # closer to 1/R than the two hypotheses sit to each other.
    err = abs(np.mean(y.real) - 1.0 / R)
    gap = abs(1.0 / R - 1.0 / (R + half_kappa))
    assert err < 0.05 * gap, (
        f"R={R}: off 1/R by {err:.3e} S, which is not small against the "
        f"{gap:.3e} S gap to R + kappa/2 = {R + half_kappa:.1f} ohm")
    assert np.max(np.abs(y.imag)) < 5e-3 / R


def test_the_element_adds_no_shunt_cell_capacitance():
    """The two-standard solve returns C_par ~ 0, not the cell's 7.08 fF.

    This is the substantive finding: the Yee cell's own capacitance is present in
    the reference run too, so it is background, not something the element brings.
    An element bridges its cell rather than replacing it.
    """
    c_par = _calibrate()[1][_band()].real
    assert np.max(np.abs(c_par)) < 0.02 * C_CELL, (
        f"excess shunt C = {np.max(np.abs(c_par)) * 1e15:.3f} fF "
        f"(cell capacitance is {C_CELL * 1e15:.2f} fF)")


def test_capacitor_delivers_the_capacitance_it_was_asked_for():
    """A 100 fF cap reads 100 fF, not 100 + C_cell.

    The residual is the trapezoidal companion's frequency warp — it grows as f^2
    (0.003 fF at 4 GHz to 0.064 fF at 30 GHz), where a parallel cell capacitance
    would be a flat 7.08 fF offset at every frequency.
    """
    C = 100e-15
    f = _freqs()[_band()]
    w = 2.0 * np.pi * f
    excess = (_admittance(capacitance=C)[_band()].imag - w * C) / w
    assert np.max(np.abs(excess)) < 0.02 * C_CELL, (
        f"capacitor reads {np.max(np.abs(excess)) * 1e15:.3f} fF off")
    # The warp is the (w*dt/2)^2/3 of the trapezoidal rule, not a fixed offset.
    x = 0.5 * w * _CACHE['ref'][3]
    assert np.allclose(excess, C * x ** 2 / 3.0, atol=0.002e-15)


def test_inductor_does_not_resonate_with_the_cell_capacitance():
    """10 nH shunting this line would resonate at 18.9 GHz against C_cell.

    That is inside the measured band, so the parallel-cell-capacitance model is
    not merely a small correction here — it predicts the susceptance passing
    through zero and changing sign mid-band. It does not: the element stays
    inductive and tracks 1/(jwL) to better than 1%.
    """
    L = 10e-9
    f_res = 1.0 / (2.0 * np.pi * np.sqrt(L * C_CELL))
    assert BAND[0] < f_res < BAND[1], "the discriminating resonance left the band"

    w = 2.0 * np.pi * _freqs()[_band()]
    y = _admittance(inductance=L)[_band()]
    assert np.all(y.imag < 0.0), "susceptance changed sign — it resonated"
    assert np.allclose(y.imag, -1.0 / (w * L), rtol=1e-2), (
        f"inductor off by {np.max(np.abs(y.imag * w * L + 1.0)) * 100:.2f}%")


def test_a_flat_equivalent_capacitance_is_not_a_capacitance():
    """1 nH's residual is frequency-flat at 0.018 fF, which pins it as the warp.

    A shunt capacitance C contributes a susceptance w*C, i.e. a *constant* C when
    divided out; the trapezoidal warp on an inductor instead contributes a
    constant when divided by w — the two have opposite frequency signatures, and
    the measurement follows the warp.
    """
    L = 1e-9
    w = 2.0 * np.pi * _freqs()[_band()]
    excess = (_admittance(inductance=L)[_band()].imag + 1.0 / (w * L)) / w
    assert np.allclose(excess, (0.5 * w * _CACHE['ref'][3]) ** 2 / (3.0 * w ** 2 * L),
                       atol=0.01e-15)
    assert np.max(np.abs(excess)) < 0.01 * C_CELL
