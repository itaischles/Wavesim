"""Regression gate for the :class:`TEMPort` launch/termination impedance.

The port is a semi-implicit (Piket-May) Thevenin element on a solved mode. It
presents its internal series resistance ``z_int`` to the field in *both* roles —
as a terminator (a wave arriving sees ``z_int``) and as a source (its own
launched wave divides by ``z_int``) — so it is Thevenin-consistent: one
resistance, matched at ``z_int = Z₀``. The semi-implicit denominator carries a
``κ/2`` stability term, but it self-cancels for smooth excitation and does not
appear in the presented impedance: a spectral mid-line reflection sweep puts the
matched (``Γ = −1/3``) value at ``z_int = Z₀`` to within a few percent. (An
earlier revision pre-compensated to ``Z₀ − κ/2``; a clean spectral measurement
did not bear that out — it *under*-matched the terminator by ~κ/2.)

``voltage`` is the launched forward-wave voltage: with ``z_int = Z₀`` a matched
Thevenin source delivers ``Vs/2`` forward (directional) or ``Vs/3`` each way
(bidirectional), so the EMF is driven up by the exact reciprocal — clean factors
2 and 3, κ- and geometry-independent — to land ``voltage(t)`` on the line,
matching ``TEMMode.to_source``'s amplitude convention.

What the tests lock in:

* ``z_int = Z₀`` and the EMF scale is 2 (directional) / 3 (bidirectional).
* A bare ``TEMPort(voltage=A)`` lands ``A`` forward volts on any geometry.
* A passive port terminates a mid-line at ≈Z₀ (Γ ≈ −1/3); the earlier
  over-compensation (``z_int = Z₀ − κ/2``) presents a lower impedance and
  reflects clearly more.
* As a Thevenin generator it drains a closed cavity a soft launch leaves trapped.

Everything 3D runs on the numba backend (bit-for-bit identical to numpy, ~10×
faster). Derivation lives in the :class:`~wavesim.sources.TEMPort` docstring.
"""
import numpy as np
import pytest

import wavesim as ws
from wavesim.mode_solver import solve_tem_modes

# RG58-ish coax cross-section (shared with test_directional_launch).
R_IN, R_OUT = 0.405e-3, 1.475e-3
EPS_FILL = 2.3
N_XY = 24
F_MAX = 30e9
BACKEND = 'numba'
RG58 = (N_XY, R_IN, R_OUT, EPS_FILL)


def _coax_cfg(cfg, nz):
    n, r_in, r_out, eps_r = cfg
    ds = (2.6 * r_out) / n
    grid = ws.create_grid(Nx=n, Ny=n, Nz=nz, dx=ds, dy=ds, dz=ds)
    ws.set_vacuum(grid)
    c = 0.5 * n * ds
    ws.set_coax(grid, cx=c, cy=c, r_inner=r_in, r_outer=r_out, eps_r_fill=eps_r)
    return grid, ds, c, r_out


def _coax(nz):
    grid, ds, c, _ = _coax_cfg(RG58, nz)
    return grid, ds, c


def _mode(grid, ds, k):
    return solve_tem_modes(grid, normal='z', position=k * ds,
                           compute_params=True)[0]


def _pulse():
    return ws.GaussianPulse.for_fmax(F_MAX)


def _l2(x):
    return float(np.sqrt(np.sum(np.square(x))))


def _field_energy(grid):
    return float(sum(np.sum(eps * getattr(grid, comp) ** 2)
                     for comp, eps in (('Ex', grid.eps_x),
                                       ('Ey', grid.eps_y),
                                       ('Ez', grid.eps_z))))


# --------------------------------------------------------------------------- #
# Fast: z_int = Z₀, and the EMF scale is the clean 2 (dir.) / 3 (bidir.).
# --------------------------------------------------------------------------- #

def test_internal_resistance_and_emf_scale():
    """``z_int = Z₀`` (no κ/2 compensation) and the drive is scaled so the
    forward wave lands ``voltage`` volts — a clean 2 (directional) / 3
    (bidirectional). No timestepping needed.

    κ/2 is ~19% of Z₀ here, so the clean 2 is well clear of the ``2 − κ/2Z₀ ≈
    1.81`` a resurrected compensation would produce.
    """
    grid, ds, c = _coax(nz=60)
    mode = _mode(grid, ds, 20)
    z0 = mode.impedance
    half_kappa = 0.5 * mode.build_port_kernel(
        grid, directional=True, frequency=F_MAX)['kappa']
    assert 0.05 < half_kappa / z0 < 0.5             # the effect is not rounding

    wf = _pulse()
    port = ws.TEMPort(mode=mode, voltage=wf, directional=True)
    port._port = port._build_port(grid)             # finalises impedance + scale
    assert port.impedance == pytest.approx(z0), (
        f"z_int={port.impedance:.3f} should be Z₀={z0:.3f}")

    port_bi = ws.TEMPort(mode=mode, voltage=wf, directional=False)
    port_bi._port = port_bi._build_port(grid)
    for t in (0.3e-9, 0.7e-9):
        assert port.waveform(t) == pytest.approx(2.0 * wf(t))
        assert port_bi.waveform(t) == pytest.approx(3.0 * wf(t))


# --------------------------------------------------------------------------- #
# Slow: a bare port lands forward volts, on any geometry.
# --------------------------------------------------------------------------- #

def _forward_ratio(cfg, k_src=14, k_mon=28, nz=80, nsteps=700, d_pml=10):
    """||bare TEMPort(voltage=w)|| / ||to_source(amplitude=1, w)|| at a downstream
    monitor. The calibrated soft launch lands exactly ``w`` forward volts, so this
    ratio is the port's launched forward volts per requested volt — 1 if fixed."""
    def launch(make_source):
        grid, ds, c, r_out = _coax_cfg(cfg, nz)
        cpml = ws.init_cpml(grid, d_pml=d_pml, faces=('z0', 'z1'))
        sim = ws.Simulation(grid, cpml=cpml, backend=BACKEND)
        sim.add_source(make_source(grid, ds))
        zc = k_mon * ds
        mon = sim.add_monitor(ws.VoltageMonitor(
            path=((c, c, zc), (c + r_out + ds, c, zc))))
        for _ in range(nsteps):
            sim.step()
        return np.asarray(mon.values)

    v_cal = launch(lambda g, ds: _mode(g, ds, k_src).to_source(
        _pulse(), amplitude=1.0, fields='EH'))
    v_prt = launch(lambda g, ds: ws.TEMPort(
        mode=_mode(g, ds, k_src), voltage=_pulse(), directional=True))
    return _l2(v_prt) / _l2(v_cal)


@pytest.mark.slow
def test_bare_port_lands_forward_volts_on_any_geometry():
    """``TEMPort(voltage=A)`` launches ``A`` forward volts with no rescale.

    Two coaxes with very different κ/Z₀ (RG58 ε=2.3 → ~19%; a fatter air line →
    ~29%). Both land the forward voltage to within a few percent and agree — the
    residual is discretisation noise, not a κ/Z₀ term.
    """
    r_rg58 = _forward_ratio(RG58)
    r_fat = _forward_ratio((30, 0.30e-3, 2.10e-3, 1.0))
    assert r_rg58 == pytest.approx(1.0, abs=0.05), f"RG58 forward ratio {r_rg58:.3f}"
    assert r_fat == pytest.approx(1.0, abs=0.05), f"air-line forward ratio {r_fat:.3f}"
    assert abs(r_rg58 - r_fat) < 0.05, (
        f"forward ratio drifts with geometry ({r_rg58:.3f} vs {r_fat:.3f})")


# --------------------------------------------------------------------------- #
# Slow: a passive port terminates a mid-line at ≈Z₀  (Γ ≈ −1/3).
# --------------------------------------------------------------------------- #

def _midline_gamma(impedance, k_src=14, k_mon=63, k_port=65, nz=100, nsteps=1500,
                   d_pml=10, f_max=12e9):
    """Spectral reflection off a passive mid-line port, by reference subtraction.

    A directional pulse is launched down a PML-matched coax; a passive port sits
    at ``k_port`` with the line (and PML) continuing past it. Subtracting the
    otherwise-identical no-port run isolates the reflected wave at the adjacent
    upstream monitor; |Γ| is its band-centre spectral magnitude over the
    incident's. A matched Z₀ shunt on a continuing Z₀ line gives 1/3; a lower
    shunt impedance gives more. Lower ``f_max`` + adjacent monitor keep dispersion
    bias well under the κ/2 effect."""
    def run(with_port):
        grid, ds, c, r_out = _coax_cfg(RG58, nz)
        cpml = ws.init_cpml(grid, d_pml=d_pml, faces=('z0', 'z1'))
        sim = ws.Simulation(grid, cpml=cpml, backend=BACKEND)
        sim.add_source(_mode(grid, ds, k_src).to_source(
            ws.GaussianPulse.for_fmax(f_max), amplitude=1.0, fields='EH'))
        if with_port:
            sim.add_source(ws.TEMPort(mode=_mode(grid, ds, k_port),
                                      impedance=impedance, directional=False))
        zc = k_mon * ds
        mon = sim.add_monitor(ws.VoltageMonitor(
            path=((c, c, zc), (c + r_out + ds, c, zc))))
        for _ in range(nsteps):
            sim.step()
        return np.asarray(mon.values), grid.dt

    incident, dt = run(with_port=False)
    reflected = run(with_port=True)[0] - incident
    freqs = np.fft.rfftfreq(len(incident), dt)
    b = int(np.argmin(np.abs(freqs - f_max)))
    return abs(np.fft.rfft(reflected)[b]) / abs(np.fft.rfft(incident)[b])


@pytest.mark.slow
def test_passive_port_terminates_at_z0():
    """The default passive port reflects like a matched Z₀ shunt (|Γ| ≈ 1/3).

    A Z₀ shunt on a continuing Z₀ line gives |Γ| = 1/3; a *lower* shunt impedance
    reflects more. The default (``z_int = Z₀``) lands near 1/3, while the earlier
    over-compensation (``z_int = Z₀ − κ/2``) presents a lower impedance and
    reflects clearly more. The exact ``z_int = Z₀`` is pinned by the unit test.
    """
    grid, ds, c = _coax(nz=100)
    mode = _mode(grid, ds, 65)
    z0 = mode.impedance
    half_kappa = 0.5 * mode.build_port_kernel(
        grid, directional=True, frequency=12e9)['kappa']

    g_matched = _midline_gamma(impedance=None)              # z_int = Z₀
    g_undercomp = _midline_gamma(impedance=z0 - half_kappa)  # z_int = Z₀ − κ/2

    assert g_matched == pytest.approx(1.0 / 3.0, abs=0.03), (
        f"default passive port |Γ|={g_matched:.3f}, expected ≈ 1/3 (Z₀ shunt)")
    assert g_undercomp > g_matched + 0.02, (
        f"over-compensation not reflecting more: default |Γ|={g_matched:.3f}, "
        f"z_int=Z₀−κ/2 |Γ|={g_undercomp:.3f}")


# --------------------------------------------------------------------------- #
# Slow: the matched Thevenin port drains DC/trapped energy a soft launch freezes.
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_matched_port_drains_a_closed_cavity():
    """A closed (PEC-ended, lossless) coax has exactly one loss path: the port.

    Driven from a boundary ``TEMPort`` the pulse's energy bleeds back out through
    the source resistance (the whole point of the matched-port change — a real
    resistance conducts at DC). Driven by the soft, ideal-current launch the same
    energy has nowhere to go and stays trapped. The port therefore retains a
    markedly smaller fraction of its mid-run energy at late time.
    """
    k_port, nz, nsteps = 4, 64, 2600
    mid = slice(int(0.40 * nsteps), int(0.55 * nsteps))
    late = slice(int(0.85 * nsteps), None)

    def cavity(make_source):
        grid, ds, c = _coax(nz=nz)
        sim = ws.Simulation(grid, backend=BACKEND)     # no cpml -> PEC ends
        sim.add_source(make_source(grid, ds))
        energy = np.empty(nsteps)
        for step in range(nsteps):
            energy[step] = _field_energy(sim.step())
        return float(np.mean(energy[late]) / np.mean(energy[mid]))

    retain_soft = cavity(lambda g, ds: _mode(g, ds, k_port).to_source(
        _pulse(), amplitude=1.0, fields='EH'))
    retain_port = cavity(lambda g, ds: ws.TEMPort(
        mode=_mode(g, ds, k_port), voltage=_pulse(), directional=True))

    # Soft launch traps: with no loss path its late energy is ≥ its mid energy
    # (the pulse just sloshes). The matched port visibly decays below it.
    assert retain_soft > 0.95, (
        f"soft launch unexpectedly lost energy (retain={retain_soft:.3f}); "
        f"the closed cavity should be lossless for it")
    assert retain_port < 0.90, (
        f"matched port did not drain (retain={retain_port:.3f})")
    assert retain_port < retain_soft, (
        f"port retains {retain_port:.3f}, soft {retain_soft:.3f} — "
        f"the matched port should drain more")
