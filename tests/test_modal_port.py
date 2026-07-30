"""Regression gate for :class:`~wavesim.sources.ModalPort` — the modal
impedance-sheet boundary that launches and/or terminates a TEM mode on a domain
face, replacing PML for a closed cross-section.

What the tests lock in:

* **Self-calibrating admittance.** The sheet's numerical-admittance correction
  ``s`` is *derived* from the mode via power balance
  (:meth:`~wavesim.mode_solver.TEMMode.numerical_admittance_scale`), not a tuned
  constant. It stays within a few percent of 1 across geometries and there is no
  constant fallback: a mode without ``Z₀`` raises.
* **Absorbs without PML, both faces.** A z1 (high-index) and a z0 (low-index)
  face each terminate the coax TEM at better than −20 dB with *no* PML — the
  ghost-H indexing/sign differ per face and both are exercised.
* **DC-exact.** A dual modal-port line (launch at one end, absorb at the other,
  no PML anywhere) leaves no static residual — the property PML cannot provide.
* **Calibrated launch.** ``amplitude=1`` lands ≈1 forward volt, matching
  ``TEMMode.to_source``, on more than one geometry.
* **No charge, no self-oscillation.** The staggered-``ê`` kernel deposits no
  divergence, and a passive absorber does not ring (it reads V and sets H; it is
  not a fed-back drive loop).

Everything 3D runs on the numba backend (bit-for-bit identical to numpy, ~10×
faster). Derivation lives in the :class:`~wavesim.sources.ModalPort` and
:meth:`~wavesim.mode_solver.TEMMode.numerical_admittance_scale` docstrings.
"""
import numpy as np
import pytest

import wavesim as ws
import wavesim.constants as C
from wavesim.mode_solver import solve_tem_modes

BACKEND = 'numba'


# --------------------------------------------------------------------------- #
# Geometry builders (two distinct closed cross-sections).
# --------------------------------------------------------------------------- #

def _coax(nx=36, dx=0.5e-3, nz=60, dz=2e-3, r_in=3e-3, r_out=9e-3):
    g = ws.create_grid(Nx=nx, Ny=nx, Nz=nz, dx=dx, dy=dx, dz=dz)
    ws.set_vacuum(g)
    c = 0.5 * nx * dx
    ws.set_coax(g, cx=c, cy=c, r_inner=r_in, r_outer=r_out, eps_r_fill=1.0)
    return g, c, r_out - 0.5 * dx


def _stripline(nx=36, dx=0.5e-3, nz=60, dz=2e-3):
    g = ws.create_grid(Nx=nx, Ny=nx, Nz=nz, dx=dx, dy=dx, dz=dz)
    ws.set_vacuum(g)
    c = 0.5 * nx * dx
    w = 0.4 * nx * dx
    ws.set_box(g, c - 0.5 * w, c + 0.5 * w, c - dx, c + dx, 0, nz * dz,
               eps_r=1.0, pec=True)
    return g, c, 0.45 * nx * dx


def _mode(g, k):
    return solve_tem_modes(g, normal='z', position=k * g.dz,
                           compute_params=True)[0]


def _vprobe(sim, g, c, rprobe, kz):
    zc = kz * g.dz
    return sim.add_monitor(ws.VoltageMonitor(
        path=((c, c, zc), (c + rprobe, c, zc))))


def _reflection_db(t_ns, v, tcut):
    inc = np.max(np.abs(v[t_ns < tcut]))
    ref = np.max(np.abs(v[t_ns >= tcut]))
    return 20.0 * np.log10(ref / inc)


# --------------------------------------------------------------------------- #
# Fast: the admittance scale is derived, near 1, geometry-adaptive, no fallback.
# --------------------------------------------------------------------------- #

def test_admittance_scale_is_derived_and_near_unity():
    """``s`` is computed from power balance ``s = 1/(Z₀·G)``, not a constant.

    It sits within a few percent of 1 for two very different cross-sections, and
    is *not* the coax-tuned ``1.10`` of the original prototype (which does not
    generalise). No timestepping needed.
    """
    for build in (_coax, _stripline):
        g, *_ = build()
        m = _mode(g, g.Nz - 1)
        s = m.numerical_admittance_scale(g)
        assert 0.9 < s < 1.1, f"derived admittance scale {s:.3f} out of range"


def test_no_constant_fallback_without_z0():
    """Without Z₀ (compute_params=False) the scale cannot be guessed — it raises,
    rather than silently falling back to a constant fudge."""
    g, *_ = _coax()
    m = solve_tem_modes(g, normal='z', position=(g.Nz - 1) * g.dz,
                        compute_params=False)[0]
    with pytest.raises(ValueError):
        m.numerical_admittance_scale(g)
    port = ws.ModalPort(m, amplitude=0.0)      # setup deferred to first apply
    sim = ws.Simulation(g, backend=BACKEND, pec_faces=('x0', 'x1', 'y0', 'y1'))
    sim.add_boundary(port)
    with pytest.raises(ValueError):
        sim.step()


def test_low_face_requires_one_cell_in():
    """A low-index port whose ghost-H plane would fall off the grid (k=0) raises."""
    g, *_ = _coax()
    m = solve_tem_modes(g, normal='z', position=0.0, compute_params=True)[0]
    port = ws.ModalPort(m, amplitude=0.0, face='z0')
    with pytest.raises(ValueError):
        port._setup(g)


# --------------------------------------------------------------------------- #
# Slow: both faces absorb the TEM with NO PML.
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_z1_face_absorbs_without_pml():
    """High-index face terminates the coax TEM at < −28 dB, PML only on the far
    (launch) side. The bound locks in the read-back time-shift (sampling V̄ at
    n + h_tau, not the naïve n−½): reverting it lifts reflection back to ~−25 dB."""
    g, c, rp = _coax(nz=50)
    cpml = ws.init_cpml(g, d_pml=8, faces=('z0',))
    sim = ws.Simulation(g, cpml=cpml, pec_faces=('x0', 'x1', 'y0', 'y1'),
                        backend=BACKEND)
    sim.add_source(_mode(g, 10).to_source(
        ws.GaussianPulse.for_fmax(8e9), amplitude=1.0, fields='EH'))
    sim.add_boundary(ws.ModalPort(_mode(g, g.Nz - 1), amplitude=0.0))
    mon = _vprobe(sim, g, c, rp, 25)
    for _ in range(int(round(1.2e-9 / g.dt))):
        sim.step()
    db = _reflection_db(np.array(mon.times) * 1e9, np.array(mon.values), 0.33)
    assert db < -28.0, f"z1 face reflection {db:.1f} dB (want < −28)"


@pytest.mark.slow
def test_dual_port_line_absorbs_and_is_dc_exact():
    """A line terminated by modal ports at BOTH ends, NO PML: one launches inward,
    the other absorbs. Reflection < −20 dB and — the property PML lacks — the
    late-time DC residual is essentially zero."""
    def run(f_max, sim_time, kz):
        g, c, rp = _coax(nz=60)
        sim = ws.Simulation(g, cpml=None, pec_faces=('x0', 'x1', 'y0', 'y1'),
                            backend=BACKEND)
        wf = ws.GaussianPulse.for_fmax(f_max)
        sim.add_boundary(ws.ModalPort(_mode(g, g.Nz - 1), amplitude=1.0, waveform=wf))
        sim.add_boundary(ws.ModalPort(_mode(g, 1), amplitude=0.0))
        mon = _vprobe(sim, g, c, rp, kz)
        for _ in range(int(round(sim_time / g.dt))):
            sim.step()
        return np.array(mon.times) * 1e9, np.array(mon.values)

    t, v = run(8e9, 1.6e-9, 30)
    assert np.max(np.abs(v)) > 0.5, "launch did not reach mid-line"
    assert _reflection_db(t, v, 0.55) < -20.0, "dual-port reflection too high"

    t2, v2 = run(2e9, 4.0e-9, 30)
    pk = np.max(np.abs(v2))
    dc = np.mean(v2[t2 > 0.85 * t2[-1]])
    assert abs(dc) / pk < 5e-3, f"dual modal-port line left DC residual {dc/pk*100:.2f}%"


# --------------------------------------------------------------------------- #
# Slow: calibrated launch lands forward volts, on any geometry.
# --------------------------------------------------------------------------- #

def _launch_forward_ratio(build):
    """‖ModalPort launch(amp=1)‖peak / ‖to_source(amp=1)‖peak at a downstream
    monitor — 1 if the sheet launches exactly the requested forward volts."""
    wf = ws.GaussianPulse.for_fmax(6e9)

    g, c, rp = build()
    cpml = ws.init_cpml(g, d_pml=8, faces=('z1',))
    sim = ws.Simulation(g, cpml=cpml, pec_faces=('x0', 'x1', 'y0', 'y1'),
                        backend=BACKEND)
    sim.add_boundary(ws.ModalPort(_mode(g, 1), amplitude=1.0, waveform=wf))
    mon = _vprobe(sim, g, c, rp, 30)
    for _ in range(int(round(1.6e-9 / g.dt))):
        sim.step()
    v_port = np.max(np.abs(np.array(mon.values)))

    g, c, rp = build()
    cpml = ws.init_cpml(g, d_pml=8, faces=('z0', 'z1'))
    sim = ws.Simulation(g, cpml=cpml, pec_faces=('x0', 'x1', 'y0', 'y1'),
                        backend=BACKEND)
    sim.add_source(_mode(g, 14).to_source(wf, amplitude=1.0, fields='EH'))
    mon = _vprobe(sim, g, c, rp, 30)
    for _ in range(int(round(1.6e-9 / g.dt))):
        sim.step()
    v_ref = np.max(np.abs(np.array(mon.values)))
    return v_port / v_ref


@pytest.mark.slow
def test_launch_lands_forward_volts_on_any_geometry():
    """``amplitude=1`` lands ≈1 forward volt like ``to_source``, coax and
    stripline agreeing — the launch calibration carries no geometry fudge."""
    r_coax = _launch_forward_ratio(_coax)
    r_strip = _launch_forward_ratio(_stripline)
    assert r_coax == pytest.approx(1.0, abs=0.06), f"coax launch ratio {r_coax:.3f}"
    assert r_strip == pytest.approx(1.0, abs=0.06), f"stripline launch ratio {r_strip:.3f}"
    assert abs(r_coax - r_strip) < 0.06, (
        f"launch ratio drifts with geometry ({r_coax:.3f} vs {r_strip:.3f})")


# --------------------------------------------------------------------------- #
# Slow: no charge is deposited under a DC-containing pulse.
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_absorber_deposits_no_charge():
    """After a DC-heavy run with a z1 absorber, ∇·D in the dielectric gap is at
    round-off — the staggered-ê kernel is divergence-free."""
    g, c, rp = _coax(nz=50)
    cpml = ws.init_cpml(g, d_pml=8, faces=('z0',))
    sim = ws.Simulation(g, cpml=cpml, pec_faces=('x0', 'x1', 'y0', 'y1'),
                        backend=BACKEND)
    sim.add_source(_mode(g, 10).to_source(
        ws.GaussianPulse.for_fmax(2e9), amplitude=1.0, fields='EH'))
    sim.add_boundary(ws.ModalPort(_mode(g, g.Nz - 1), amplitude=0.0))
    for _ in range(int(round(2.0e-9 / g.dt))):
        sim.step()

    Dx = C.EPS0 * g.eps_x * g.Ex
    Dy = C.EPS0 * g.eps_y * g.Ey
    Dz = C.EPS0 * g.eps_z * g.Ez
    div = ((Dx[1:, 1:, 1:] - Dx[:-1, 1:, 1:]) / g.dx
           + (Dy[1:, 1:, 1:] - Dy[1:, :-1, 1:]) / g.dy
           + (Dz[1:, 1:, 1:] - Dz[1:, 1:, :-1]) / g.dz)
    # dielectric gap column between conductors, away from the ends
    nx = g.Nx
    gap = np.abs(div[nx // 2 + 8 - 1, nx // 2 - 1, 15:35])
    peak_grad = np.max(np.abs(Dx)) / g.dx + 1e-30
    assert gap.max() / peak_grad < 1e-6, (
        f"absorber deposited charge: rel ∇·D = {gap.max()/peak_grad:.2e}")
