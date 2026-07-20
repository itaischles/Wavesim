"""Directional (one-way) launch: the E/H sheet pairing.

A single impressed sheet has no notion of "forward" — it radiates both ways.
Pairing it with an H sheet at ``H = (n̂ × E)/η`` makes the two contributions add
forwards and cancel backwards. On a Yee grid that cancellation is only as good
as the bookkeeping: H is stored half a cell along the normal from E and half a
timestep away in the leapfrog, so the two sheets sample the incident wave at
different space-time points. Correcting both offsets is what turns a rough bias
into a real null.

The numbers below come from a 1D vacuum study (see ``build_port_kernel``):
uncorrected ≈ -18 dB, corrected ≈ -150 dB, with the E/H *amplitude* ratio needing
no correction at all (continuum 1/η is right to 0.3% over Courant 0.3-0.99 and
10-40 cells per wavelength). In 3D the achievable null is set by other error
sources — PML reflection, staircased conductors, transverse mode discretisation
— so the coax test below asserts a far looser bound than the physics allows.
"""
import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import C0
from wavesim.mode_solver import (
    solve_tem_modes, numerical_velocity, _launch_time_shift,
)
from wavesim.sources import LineSource

R_IN, R_OUT = 0.405e-3, 1.475e-3


# ---------------------------------------------------------------------- #
# Numerical phase velocity and the resulting time shift
# ---------------------------------------------------------------------- #

def test_numerical_velocity_is_continuum_without_a_frequency():
    """A broadband drive has no single frequency to tune to."""
    assert numerical_velocity(C0, 1e-3, 2e-12, None) == C0


def test_grid_wave_is_slower_than_the_medium():
    """Yee dispersion always retards the wave; never speeds it up."""
    v = numerical_velocity(C0, 1e-3, 2e-12, 15e9)
    assert v < C0
    assert v == pytest.approx(C0, rel=5e-3)      # ~0.3% at 20 cells/wavelength


def test_dispersion_vanishes_with_resolution():
    """Finer sampling in time and space converges back to the continuum."""
    coarse = numerical_velocity(C0, 1e-3, 2e-12, 30e9)
    fine = numerical_velocity(C0, 1e-4, 2e-13, 30e9)
    assert abs(fine - C0) < abs(coarse - C0)


def test_time_shift_is_a_lag_for_every_stable_courant():
    """Negative shift is what lets a circuit port build it from past values."""
    dz = 1e-3
    for S in (0.99, 0.7, 0.5, 0.3):
        dt = S * dz / C0
        assert _launch_time_shift(dt, dz, C0, 15e9) <= 0.0


def test_time_shift_vanishes_at_the_magic_timestep():
    """At S=1 the half-cell and half-step offsets cancel exactly."""
    dz = 1e-3
    dt = dz / C0
    assert _launch_time_shift(dt, dz, C0, None) == pytest.approx(0.0, abs=1e-18)


# ---------------------------------------------------------------------- #
# History interpolation
# ---------------------------------------------------------------------- #

def _lag(currents, lag_steps):
    """Call the helper without going through a full port constructor."""
    obj = LineSource.__new__(LineSource)
    obj.currents = list(currents)
    return LineSource._lagged_current(obj, lag_steps)


def test_zero_lag_is_the_present_current():
    assert _lag([1.0, 2.0, 3.0], 0.0) == 3.0


def test_whole_step_lag_walks_back_the_history():
    assert _lag([1.0, 2.0, 3.0], 1.0) == pytest.approx(2.0)
    assert _lag([1.0, 2.0, 3.0], 2.0) == pytest.approx(1.0)


def test_fractional_lag_interpolates_linearly():
    assert _lag([1.0, 2.0, 3.0], 0.25) == pytest.approx(2.75)
    assert _lag([1.0, 2.0, 3.0], 1.5) == pytest.approx(1.5)


def test_history_before_the_run_reads_zero():
    assert _lag([], 0.0) == 0.0
    assert _lag([5.0], 1.0) == pytest.approx(0.0)      # only one step recorded


# ---------------------------------------------------------------------- #
# Sheet placement
# ---------------------------------------------------------------------- #

def _coax(n=28, nz=60, eps_r=2.3):
    ds = (2.6 * R_OUT) / n
    grid = ws.create_grid(Nx=n, Ny=n, Nz=nz, dx=ds, dy=ds, dz=ds)
    ws.set_vacuum(grid)
    c = 0.5 * n * ds
    ws.set_coax(grid, cx=c, cy=c, r_inner=R_IN, r_outer=R_OUT, eps_r_fill=eps_r)
    return grid, ds


def test_h_sheet_sits_one_cell_behind_the_e_plane():
    """H is stored at +½ cell, so index k-1 puts it half a cell *behind* E."""
    grid, ds = _coax()
    k = 20
    mode = solve_tem_modes(grid, normal='z', position=k * ds,
                           compute_params=True)[0]
    kernel = mode.build_port_kernel(grid, directional=True, frequency=20e9)
    assert kernel['h_tau'] <= 0.0
    for _comp, (_ii, _jj, kk, _w) in kernel['hedges'].items():
        assert np.all(kk == k - 1)
    for _comp, (_ii, _jj, kk, _w, _c) in kernel['edges'].items():
        assert np.all(kk == k)


def test_bidirectional_kernel_has_no_h_sheet_or_shift():
    grid, ds = _coax()
    mode = solve_tem_modes(grid, normal='z', position=20 * ds,
                           compute_params=True)[0]
    kernel = mode.build_port_kernel(grid, directional=False)
    assert kernel['hedges'] == {}
    assert kernel['h_tau'] == 0.0


def test_a_port_against_the_boundary_is_rejected():
    """The H sheet would land outside the grid — say so, don't silently clip."""
    grid, ds = _coax()
    mode = solve_tem_modes(grid, normal='z', position=0.0,
                           compute_params=True)[0]
    with pytest.raises(ValueError, match="one cell behind"):
        mode.build_port_kernel(grid, directional=True)


# ---------------------------------------------------------------------- #
# End-to-end: a driven coax port
# ---------------------------------------------------------------------- #

@pytest.mark.slow
def test_corrected_launch_rejects_the_backward_wave():
    """Measured on a real coax TEMPort: ≈ -30 dB before the fix, ≈ -48 dB after.

    Bounds are deliberately loose — the point is the *improvement*, and the 3D
    floor is set by PML reflection and conductor staircasing, not by the sheet
    pairing.
    """
    n, nz, d_pml = 28, 150, 10
    k_port, k_bwd, k_fwd = 40, 25, 110
    freq, nsteps = 20e9, 1400

    def run(emulate_old):
        grid, ds = _coax(n=n, nz=nz)
        mode = solve_tem_modes(grid, normal='z', position=k_port * ds,
                               compute_params=True)[0]
        cpml = ws.init_cpml(grid, d_pml=d_pml, faces=('z0', 'z1'))
        sim = ws.Simulation(grid, cpml=cpml)
        port = sim.add_source(ws.TEMPort(
            mode=mode, voltage=ws.Sinusoid(frequency=freq), directional=True))
        c = 0.5 * n * ds
        pi = int((c + 0.5 * (R_IN + R_OUT)) / ds)
        pj = int(c / ds)
        fwd, bwd = np.zeros(nsteps), np.zeros(nsteps)
        for step in range(nsteps):
            if step == 1 and emulate_old:
                # Pre-fix behaviour: H sheet at index k, driven with no shift.
                port._h_lag_steps = 0.0
                port._port['hedges'] = {
                    comp: (ii, jj, kk + 1, w)
                    for comp, (ii, jj, kk, w) in port._port['hedges'].items()}
            g = sim.step()
            fwd[step] = g.Ex[pi, pj, k_fwd]
            bwd[step] = g.Ex[pi, pj, k_bwd]
        dt = grid.dt
        ps = 1.0 / (freq * dt)
        n_win = int(round(int((nsteps - 700) / ps) * ps))
        idx = np.arange(nsteps - n_win, nsteps)
        ref = np.exp(-1j * 2 * np.pi * freq * idx * dt)

        def amp(sig):
            return abs(2.0 * np.sum(sig[idx] * ref) / n_win)
        return 20 * np.log10(amp(bwd) / amp(fwd))

    corrected = run(emulate_old=False)
    old = run(emulate_old=True)
    assert corrected < -42.0, f"backward rejection only {corrected:.1f} dB"
    assert old - corrected > 10.0, (
        f"correction gained only {old - corrected:.1f} dB "
        f"({old:.1f} -> {corrected:.1f})")
