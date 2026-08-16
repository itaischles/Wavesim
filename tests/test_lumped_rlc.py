"""Lumped R / L / C loads on a :class:`~wavesim.sources.LineSource`.

The element solves its circuit law semi-implicitly, balancing the impressed
current at ``n+½`` against the mid-step port voltage ``(Vⁿ + Vⁿ⁺¹)/2``. Both
samples sit at the same instant, which is what lets a reactive branch be carried
by a trapezoidal companion — a resistance plus a history source — without
disturbing that single-solve structure (:mod:`wavesim.lumped`).

Two layers are tested, and they answer different questions:

* **The companion algebra alone.** Does a branch actually integrate its ODE?
  Driven with a prescribed current (series) or voltage (parallel), the reported
  terminal pair is compared against the closed-form V(t) of the same network,
  and the error is required to fall as dt². Sign conventions and factors of two
  have nowhere to hide here — a flipped sign or a 2L/dt written as L/(2dt)
  changes the answer by more than the tolerance at every dt.
* **The element inside a live FDTD run.** The port's recorded traces must obey
  the load's terminal law *exactly*, and that check can be written free of κ:
  eliminating the pre-injection voltage V* between ``V ⁿ⁺¹ = V* + κ·I`` and the
  solve leaves ``Z_eq·I + (Vⁿ + Vⁿ⁺¹)/2 = Vs + V_hist``, which involves only
  recorded quantities. So these assertions are machine-precision, on a real
  grid, for R, L, C and combinations.

What is deliberately *not* asserted here is the impedance the element presents
to the surrounding field — a different measurement, since the recorded V(t)/I(t)
tested below are exact whatever the field sees. That one needed a spectral
reflection sweep and now lives in :mod:`tests.test_lumped_element_impedance`,
which settles it: the element contributes exactly its companion admittance, with
no κ/2 in series and no cell capacitance of its own.
"""
import math

import numpy as np
import pytest

import wavesim as ws
from wavesim.lumped import LumpedNetwork


# --------------------------------------------------------------------------- #
# Companion algebra
# --------------------------------------------------------------------------- #

def _drive_series(net, dt, n_steps, current):
    """Push a prescribed current through a series network.

    Returns the terminal voltage sampled at each ``n+½``, in the port sign
    convention (``V = −Z_eq·I + V_hist``) that the FDTD element uses.
    """
    out = []
    for n in range(n_steps):
        i_n = current((n + 0.5) * dt)
        z, vh = net.companion(dt)
        v = -z * i_n + vh
        net.update(dt, i_n, v)
        out.append(v)
    return np.asarray(out)


def _drive_parallel(net, dt, n_steps, voltage):
    """Impose a prescribed voltage across a parallel network; return the total
    current it draws, port sign convention (``I = (V_hist − V)/Z_eq``)."""
    out = []
    for n in range(n_steps):
        v_n = voltage((n + 0.5) * dt)
        z, vh = net.companion(dt)
        i = (vh - v_n) / z
        net.update(dt, i, v_n)
        out.append(i)
    return np.asarray(out)


def test_resistor_companion_is_the_plain_ohmic_law():
    """R has no memory: (Z, V_hist) = (R, 0) at every step, whatever happened."""
    net = LumpedNetwork(resistance=50.0)
    for i_n in (0.0, 1e-3, -7.0):
        assert net.companion(1e-13) == (50.0, 0.0)
        net.update(1e-13, i_n, -50.0 * i_n)
    assert net.companion(1e-13) == (50.0, 0.0)


@pytest.mark.parametrize("kind, value", [("inductance", 2e-9),
                                         ("capacitance", 0.5e-12)])
def test_reactive_branch_integrates_its_own_ode(kind, value):
    """A prescribed sinusoidal current reproduces V = −L·dI/dt / −(1/C)∫I dt.

    Second-order convergence is the assertion, not a fixed tolerance: the
    trapezoidal rule's error must fall by ~4× per halving of dt. A sign or
    factor error would show as a *constant* error instead.
    """
    f = 5e9
    w = 2.0 * math.pi * f
    i_of_t = lambda t: 0.01 * math.sin(w * t)

    errors = []
    for dt in (2e-13, 1e-13, 0.5e-13):
        n_steps = int(round(2.0 / (f * dt)))          # two periods
        net = LumpedNetwork(**{kind: value})
        v = _drive_series(net, dt, n_steps, i_of_t)
        t_mid = (np.arange(n_steps) + 0.5) * dt
        if kind == "inductance":
            exact = -value * 0.01 * w * np.cos(w * t_mid)
        else:
            # ∫I dt from rest = (0.01/w)(1 − cos wt); the port sign flips it.
            exact = -(0.01 / (value * w)) * (1.0 - np.cos(w * t_mid))
        scale = np.max(np.abs(exact))
        errors.append(np.max(np.abs(v - exact)) / scale)

    assert errors[0] < 2e-3
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.15)


def test_series_rlc_matches_the_closed_form_terminal_voltage():
    """All three branches at once: V = −(R·I + L·dI/dt + (1/C)∫I dt)."""
    R, L, C = 20.0, 1e-9, 0.4e-12
    f, dt = 4e9, 1e-13
    w = 2.0 * math.pi * f
    n_steps = int(round(2.0 / (f * dt)))
    i_of_t = lambda t: 0.02 * math.sin(w * t)

    net = LumpedNetwork(resistance=R, inductance=L, capacitance=C)
    v = _drive_series(net, dt, n_steps, i_of_t)

    t_mid = (np.arange(n_steps) + 0.5) * dt
    exact = -(R * 0.02 * np.sin(w * t_mid)
              + L * 0.02 * w * np.cos(w * t_mid)
              + (0.02 / (C * w)) * (1.0 - np.cos(w * t_mid)))
    assert np.max(np.abs(v - exact)) / np.max(np.abs(exact)) < 3e-3


def test_parallel_rc_draws_the_sum_of_its_branch_currents():
    """Under an imposed voltage the branches are independent: I = V/R + C·dV/dt."""
    R, C = 1e3, 0.2e-12
    f, dt = 4e9, 1e-13
    w = 2.0 * math.pi * f
    n_steps = int(round(2.0 / (f * dt)))
    v_of_t = lambda t: 0.5 * math.sin(w * t)

    net = LumpedNetwork(resistance=R, capacitance=C, topology='parallel')
    i = _drive_parallel(net, dt, n_steps, v_of_t)

    t_mid = (np.arange(n_steps) + 0.5) * dt
    exact = -(0.5 * np.sin(w * t_mid) / R + C * 0.5 * w * np.cos(w * t_mid))
    assert np.max(np.abs(i - exact)) / np.max(np.abs(exact)) < 3e-3


def test_topology_changes_the_answer():
    """Guard against 'parallel' being silently ignored: at the same dt an R‖C
    is far stiffer than the same parts in series."""
    dt = 1e-13
    series = LumpedNetwork(resistance=1e3, capacitance=0.2e-12).companion(dt)[0]
    parallel = LumpedNetwork(resistance=1e3, capacitance=0.2e-12,
                             topology='parallel').companion(dt)[0]
    assert series > parallel
    assert parallel < 1e3            # the cap shunts the resistor


def test_discrete_impedance_matches_the_trapezoidal_mapping():
    """``impedance_at(f, dt)`` is the scheme's own reactance, s → j(2/dt)tan(ωdt/2),
    and approaches the continuum value as the frequency is resolved."""
    L, f, dt = 1e-9, 2e9, 1e-13
    net = LumpedNetwork(inductance=L)
    ideal = net.impedance_at(f)
    discrete = net.impedance_at(f, dt)
    assert ideal == pytest.approx(2j * math.pi * f * L)
    assert discrete.imag / ideal.imag == pytest.approx(1.0, rel=1e-3)
    # Coarsening dt stretches the reactance upward, never down.
    assert net.impedance_at(f, 20 * dt).imag > discrete.imag


@pytest.mark.parametrize("kwargs, message", [
    (dict(), "at least one"),
    (dict(resistance=0.0), "positive"),
    (dict(inductance=-1e-9), "positive"),
    (dict(capacitance=1e-12, topology='parallell'), "topology"),
])
def test_rejects_impossible_networks(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LumpedNetwork(**kwargs)


# --------------------------------------------------------------------------- #
# The element inside a run
# --------------------------------------------------------------------------- #

def _run_line_source(**kwargs):
    """A short driven line in a small vacuum box; returns the element."""
    ds = 1e-3
    grid = ws.create_grid(Nx=16, Ny=16, Nz=16, dx=ds, dy=ds, dz=ds)
    ws.set_vacuum(grid)
    src = ws.LineSource(p0=(8 * ds, 6 * ds, 8 * ds),
                        p1=(8 * ds, 10 * ds, 8 * ds), **kwargs)
    sim = ws.Simulation(grid, sources=[src])
    sim.run(60)
    return src, grid


def _mid_voltages(src):
    """The mid-step port voltages (Vⁿ + Vⁿ⁺¹)/2 the circuit law was centred on.

    ``src.voltages[n]`` is Vⁿ⁺¹ for step n, and V⁰ = 0 (fields start at rest).
    """
    v = np.asarray(src.voltages)
    v_prev = np.concatenate(([0.0], v[:-1]))
    return 0.5 * (v_prev + v)


def test_resistive_load_obeys_ohm_in_the_run():
    """R·I + V_mid = Vs(t), exactly — the κ-free form of the solve."""
    wf = ws.GaussianPulse.for_fmax(20e9)
    src, _ = _run_line_source(voltage=wf, resistance=50.0)
    v_mid = _mid_voltages(src)
    i = np.asarray(src.currents)
    vs = np.array([wf(t) for t in src.times])
    assert np.max(np.abs(50.0 * i + v_mid - vs)) < 1e-12 * max(1.0, np.max(np.abs(vs)))


def test_capacitive_load_obeys_its_ode_in_the_run():
    """A passive cap: the mid-step voltage steps by −(dt/2C)(Iⁿ + Iⁿ⁻¹).

    Driven only by the field a neighbouring source puts on the line, so the
    trace is whatever the run produced — the point is that the element
    integrated it correctly.
    """
    C = 0.3e-12
    src, grid = _run_line_source(voltage=ws.GaussianPulse.for_fmax(20e9),
                                 capacitance=C)
    v_mid = _mid_voltages(src)
    i = np.asarray(src.currents)
    # Element voltage = port mid-step voltage less the series EMF.
    vs = np.array([src.voltage(t) for t in src.times])
    v_elem = v_mid - vs
    lhs = np.diff(v_elem)
    rhs = -(grid.dt / (2.0 * C)) * (i[1:] + i[:-1])
    assert np.max(np.abs(lhs - rhs)) < 1e-9 * np.max(np.abs(v_elem))
    assert np.max(np.abs(i)) > 0.0        # the run was not trivially quiet


def test_inductive_load_obeys_its_ode_in_the_run():
    """A series L: (Vⁿ + Vⁿ⁻¹)_elem = −(2L/dt)(Iⁿ − Iⁿ⁻¹)."""
    L = 2e-9
    src, grid = _run_line_source(voltage=ws.GaussianPulse.for_fmax(20e9),
                                 inductance=L)
    v_mid = _mid_voltages(src)
    i = np.asarray(src.currents)
    vs = np.array([src.voltage(t) for t in src.times])
    v_elem = v_mid - vs
    lhs = v_elem[1:] + v_elem[:-1]
    rhs = -(2.0 * L / grid.dt) * (i[1:] - i[:-1])
    assert np.max(np.abs(lhs - rhs)) < 1e-9 * np.max(np.abs(v_elem))


def test_norton_drive_takes_its_source_current_out_of_the_load():
    """With ``current=``, the load carries I − Is(t), not I.

    The distinction only bites for a *stateful* load, which is why it is tested
    here and not against a resistor: feeding the port current to the capacitor's
    integrator instead of the branch current gives a trace that fails the ODE
    check below while still looking plausible.
    """
    C = 0.3e-12
    wf = ws.GaussianPulse.for_fmax(20e9)
    src, grid = _run_line_source(current=lambda t: 1e-3 * wf(t), capacitance=C)
    v_mid = _mid_voltages(src)
    i_elem = np.asarray(src.currents) - np.array([src.current(t) for t in src.times])
    lhs = np.diff(v_mid)
    rhs = -(grid.dt / (2.0 * C)) * (i_elem[1:] + i_elem[:-1])
    assert np.max(np.abs(lhs - rhs)) < 1e-9 * np.max(np.abs(v_mid))


def test_a_capacitor_blocks_where_a_resistor_conducts():
    """Physics sanity, independent of the algebra: under the same drive a small
    series capacitor passes far less charge than a 50 Ω resistor."""
    wf = ws.GaussianPulse.for_fmax(20e9)
    res, _ = _run_line_source(voltage=wf, resistance=50.0)
    cap, _ = _run_line_source(voltage=wf, capacitance=1e-15)
    assert np.max(np.abs(cap.currents)) < 0.1 * np.max(np.abs(res.currents))


def test_unloaded_ideal_sources_are_untouched():
    """The load is optional: with no R/L/C the element still hard-writes an
    ideal voltage drive and impresses an ideal current one."""
    wf = ws.GaussianPulse.for_fmax(20e9)
    ideal_v, _ = _run_line_source(voltage=wf)
    assert np.allclose(ideal_v.voltages, [wf(t) for t in ideal_v.times])

    drive = lambda t: 1e-3 * wf(t)
    ideal_i, _ = _run_line_source(current=drive)
    assert np.allclose(ideal_i.currents, [drive(t) for t in ideal_i.times])


def test_line_source_still_needs_a_drive_or_a_load():
    with pytest.raises(ValueError, match="drive"):
        ws.LineSource(p0=(0, 0, 0), p1=(0, 1e-3, 0))
