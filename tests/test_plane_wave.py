"""Boundary-face plane-wave source (:class:`wavesim.sources.PlaneWave`).

A ``PlaneWave`` drives the full cross-section one PML-depth inside a boundary
face with a uniform transverse field, biased into the domain by the paired
``H = (n̂ × E)/η`` sheet. Two things are asserted here:

* **Convention** — the ordered transverse pair (a, b) per face is right-handed
  with the inward normal, so the magnetic field needs no per-face sign table but
  the same physical polarization takes a *different* ``angle`` on opposite faces.
* **Directionality** — the corrected co-indexed H sheet, driven a fraction of a
  step ahead (``τ = dt/2 + p·dn/(2·v_num)``), cancels the backward wave. On a
  clean 2D slab this measures ≈ -96 dB on both a low and a high face, versus
  ≈ -18 dB for the naive same-time pairing and 0 dB (symmetric) for E alone. The
  per-face sign of ``p`` is essential: flipping it collapses the null to ≈ -16 dB.
"""
import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import C0, ETA0
from wavesim.mode_solver import numerical_velocity
from wavesim.sources import PlaneWave, _FACE_CFG


# ---------------------------------------------------------------------- #
# Face convention: (a, b) is right-handed with the inward normal
# ---------------------------------------------------------------------- #

def _axis_vec(letter):
    return {'x': np.array([1., 0, 0]),
            'y': np.array([0, 1., 0]),
            'z': np.array([0, 0, 1.])}[letter]


@pytest.mark.parametrize('face', sorted(_FACE_CFG))
def test_transverse_pair_is_right_handed_with_propagation(face):
    """a × b must equal the inward propagation direction on every face."""
    cfg = _FACE_CFG[face]
    a, b = _axis_vec(cfg['a']), _axis_vec(cfg['b'])
    n = _axis_vec(cfg['normal']) * (1.0 if cfg['side'] == 'low' else -1.0)
    assert np.allclose(np.cross(a, b), n)


def _driven(grid, pw):
    """{component: scalar} of the (uniform) driven amplitude, E and H."""
    pw._build(grid)
    out = {}
    for comp, prof in {**pw._e_full, **pw._h_full}.items():
        nz = prof[np.nonzero(prof)]
        out[comp] = float(nz.flat[0]) if nz.size else 0.0
    return out


def test_angle_zero_drives_the_first_transverse_axis():
    """angle=0 ⇒ E along â; the H partner is E_a/η on the b-axis component."""
    g = ws.create_grid(Nx=16, Ny=16, Nz=16, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    pw = PlaneWave('z0', angle=0.0, waveform=ws.Sinusoid(frequency=10e9), d_pml=4)
    d = _driven(g, pw)
    assert d['Ex'] == pytest.approx(1.0)          # a = x on z0
    assert d.get('Ey', 0.0) == pytest.approx(0.0)
    assert d['Hy'] == pytest.approx(1.0 / ETA0)    # H = (n̂ × E)/η on b = y
    assert d.get('Hx', 0.0) == pytest.approx(0.0)


def test_same_polarization_different_angle_on_opposite_faces():
    """A +z-polarized wave is angle=π/2 on x0 (pair y→z) but angle=0 on x1."""
    g = ws.create_grid(Nx=16, Ny=16, Nz=16, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    wf = ws.Sinusoid(frequency=10e9)
    on_x0 = _driven(g, PlaneWave('x0', angle=np.pi / 2, waveform=wf, d_pml=4))
    on_x1 = _driven(g, PlaneWave('x1', angle=0.0, waveform=wf, d_pml=4))
    for d in (on_x0, on_x1):
        assert d['Ez'] == pytest.approx(1.0)
        assert d.get('Ey', 0.0) == pytest.approx(0.0)


def test_unknown_face_is_rejected():
    with pytest.raises(ValueError, match='face must be one of'):
        PlaneWave('z2', angle=0.0, waveform=ws.Sinusoid(frequency=10e9))


# ---------------------------------------------------------------------- #
# Sheet placement and the H time shift
# ---------------------------------------------------------------------- #

def test_e_sheet_lands_on_the_first_interior_cell():
    g = ws.create_grid(Nx=8, Ny=8, Nz=64, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    low = PlaneWave('z0', angle=0.0, waveform=ws.Sinusoid(frequency=10e9), d_pml=10)
    high = PlaneWave('z1', angle=0.0, waveform=ws.Sinusoid(frequency=10e9), d_pml=10)
    assert low._plane_index(g) == 10
    assert high._plane_index(g) == 64 - 1 - 10


def test_h_sheet_is_co_indexed_with_e():
    """Unlike a TEMPort (H one cell behind), the plane wave keeps H on E's slice."""
    g = ws.create_grid(Nx=8, Ny=8, Nz=64, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    pw = PlaneWave('z0', angle=0.3, waveform=ws.Sinusoid(frequency=10e9), d_pml=10)
    pw._build(g)
    k = pw._plane_index(g)
    for prof in pw._e_full.values():
        assert prof[:, :, k].any() and not np.any(np.delete(prof, k, axis=2))
    for prof in pw._h_full.values():
        assert prof[:, :, k].any() and not np.any(np.delete(prof, k, axis=2))


@pytest.mark.parametrize('face,sign', [('z0', +1.0), ('z1', -1.0)])
def test_time_shift_matches_the_launch_formula(face, sign):
    """τ = dt/2 + p·dn/(2·v_num); p = +1 into +normal, −1 into −normal."""
    g = ws.create_grid(Nx=8, Ny=8, Nz=64, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    freq = 15e9
    pw = PlaneWave(face, angle=0.0, waveform=ws.Sinusoid(frequency=freq), d_pml=10)
    pw._build(g)
    k = pw._plane_index(g)
    dn = float(g.dzp[k])
    v_num = numerical_velocity(C0, dn, g.dt, freq)
    assert pw._tau == pytest.approx(g.dt / 2.0 + sign * dn / (2.0 * v_num))
    assert pw.prop_sign == sign


def test_bidirectional_plane_wave_has_no_h_sheet():
    g = ws.create_grid(Nx=8, Ny=8, Nz=64, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    pw = PlaneWave('z0', angle=0.0, waveform=ws.Sinusoid(frequency=10e9),
                   d_pml=10, directional=False)
    pw._build(g)
    assert pw._h_full == {}
    assert pw._tau == 0.0


# ---------------------------------------------------------------------- #
# End-to-end directional launch (clean 2D slab)
# ---------------------------------------------------------------------- #

def _backward_rejection(face, *, angle, comp, freq=15e9, N=200, Ny=32,
                        nt=1200, d_pml=12, flip_sign=False, naive=False):
    """dB ratio of the backward to the forward launched amplitude.

    The source sits mid-domain (placed via ``d_pml``) so both sides are clean
    vacuum; the forward/backward probes are 50 cells either way, sampled after
    the wave has filled the window. In-plane (TE_z) polarization only — the
    out-of-plane component is degenerate on an Nz=1 slice.
    """
    g = ws.create_grid(Nx=N, Ny=Ny, Nz=1, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    kmid = N // 2
    place = kmid if face == 'x0' else N - 1 - kmid
    pw = PlaneWave(face, angle=angle, waveform=ws.Sinusoid(frequency=freq),
                   d_pml=place, directional=True)
    cpml = ws.init_cpml(g, d_pml=d_pml, faces=('x0', 'x1'))
    sim = ws.Simulation(g, cpml=cpml, sources=[pw], pec_faces=('y0', 'y1'))
    if flip_sign:
        pw.prop_sign = -pw.prop_sign
    pw._build(g)
    if naive:
        pw._tau = 0.0
    k0 = pw._plane_index(g)
    jc = Ny // 2
    # Forward is +x on a low face, −x on a high face.
    if face == 'x0':
        j_fwd, j_bwd = k0 + 50, k0 - 50
    else:
        j_fwd, j_bwd = k0 - 50, k0 + 50
    fwd = np.zeros(nt)
    bwd = np.zeros(nt)
    for s in range(nt):
        sim.step()
        fwd[s] = getattr(g, comp)[j_fwd, jc, 0]
        bwd[s] = getattr(g, comp)[j_bwd, jc, 0]
    dt = g.dt
    idx = np.arange(nt - 400, nt)
    ref = np.exp(-1j * 2 * np.pi * freq * idx * dt)
    amp = lambda sig: abs(np.sum(sig[idx] * ref))
    return 20.0 * np.log10(amp(bwd) / amp(fwd))


@pytest.mark.slow
@pytest.mark.parametrize('face,angle,comp', [
    ('x0', 0.0, 'Ey'),          # low face,  in-plane E along a = y
    ('x1', np.pi / 2, 'Ey'),    # high face, in-plane E along b = y
])
def test_corrected_launch_rejects_the_backward_wave(face, angle, comp):
    """The corrected pairing nulls the backward wave on both low and high faces.

    Bounds are loose relative to the ≈ -96 dB actually measured — the point is a
    deep, robust null, and that the per-face time-shift sign is what delivers it.
    """
    corrected = _backward_rejection(face, angle=angle, comp=comp)
    naive = _backward_rejection(face, angle=angle, comp=comp, naive=True)
    flipped = _backward_rejection(face, angle=angle, comp=comp, flip_sign=True)
    assert corrected < -40.0, f"{face}: backward rejection only {corrected:.1f} dB"
    assert corrected < naive - 20.0, (
        f"{face}: correction gained only {naive - corrected:.1f} dB "
        f"({naive:.1f} -> {corrected:.1f})")
    assert corrected < flipped - 20.0, (
        f"{face}: wrong-sign shift rejects {flipped:.1f} dB vs {corrected:.1f}")


@pytest.mark.slow
def test_bidirectional_launch_is_symmetric():
    """An E-only sheet has no preferred direction: forward ≈ backward (≈ 0 dB)."""
    g = ws.create_grid(Nx=200, Ny=32, Nz=1, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    freq, nt = 15e9, 1200
    pw = PlaneWave('x0', angle=0.0, waveform=ws.Sinusoid(frequency=freq),
                   d_pml=100, directional=False)
    cpml = ws.init_cpml(g, d_pml=12, faces=('x0', 'x1'))
    sim = ws.Simulation(g, cpml=cpml, sources=[pw], pec_faces=('y0', 'y1'))
    fwd = np.zeros(nt)
    bwd = np.zeros(nt)
    for s in range(nt):
        sim.step()
        fwd[s] = g.Ey[150, 16, 0]
        bwd[s] = g.Ey[50, 16, 0]
    idx = np.arange(nt - 400, nt)
    ref = np.exp(-1j * 2 * np.pi * freq * idx * g.dt)
    amp = lambda sig: abs(np.sum(sig[idx] * ref))
    assert 20.0 * np.log10(amp(bwd) / amp(fwd)) == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------- #
# TEMMode.to_source: an amplitude-calibrated modal launch
#
# The launch impresses the modal current a matched line turns into the
# requested forward voltage, using the same calibrated current kernel a TEMPort
# uses (build_port_kernel). It is NOT the uncalibrated _PlaneLaunch used by
# PlaneWave: a straight ``E += waveform*Ehat`` field write ignored the FDTD
# update coefficient and came out √ε_r / S_c too large.
# ---------------------------------------------------------------------- #

R_IN, R_OUT = 0.405e-3, 1.475e-3


def _coax(n=28, nz=64, eps_r=2.3):
    ds = (2.6 * R_OUT) / n
    grid = ws.create_grid(Nx=n, Ny=n, Nz=nz, dx=ds, dy=ds, dz=ds)
    ws.set_vacuum(grid)
    c = 0.5 * n * ds
    ws.set_coax(grid, cx=c, cy=c, r_inner=R_IN, r_outer=R_OUT, eps_r_fill=eps_r)
    return grid, ds


def test_to_source_eh_builds_a_calibrated_directional_launch():
    """'EH' returns a directional _ModalLaunch backed by the port current kernel:
    an H sheet one cell *behind* the E plane, driven by a lagged current."""
    from wavesim.mode_solver import solve_tem_modes
    from wavesim.sources import _ModalLaunch
    grid, ds = _coax()
    k = grid.axis_index('z', 20 * ds)
    mode = solve_tem_modes(grid, normal='z', position=20 * ds,
                           compute_params=True)[0]
    src = mode.to_source(ws.Sinusoid(frequency=20e9), fields='EH')
    assert isinstance(src, _ModalLaunch) and src.directional
    kernel = src._build_port(grid)
    assert kernel['hedges'] and src._h_lag_steps >= 0.0     # H sheet + lag
    for _comp, (_ii, _jj, kk, _c) in kernel['hedges'].items():
        assert np.all(kk == k - 1)                          # one cell behind E
    for _comp, (_ii, _jj, kk, _w, _c) in kernel['edges'].items():
        assert np.all(kk == k)


def test_to_source_e_only_is_bidirectional_with_no_h_sheet():
    from wavesim.mode_solver import solve_tem_modes
    from wavesim.sources import _ModalLaunch
    grid, ds = _coax()
    mode = solve_tem_modes(grid, normal='z', position=20 * ds,
                           compute_params=True)[0]
    src = mode.to_source(ws.GaussianPulse.for_fmax(20e9), fields='E')
    assert isinstance(src, _ModalLaunch) and not src.directional
    kernel = src._build_port(grid)
    assert kernel['hedges'] == {}
    assert src._h_lag_steps == 0.0


def test_to_source_needs_the_mode_impedance_to_calibrate():
    """Without Z₀ (compute_params=False) the amplitude cannot be calibrated."""
    from wavesim.mode_solver import solve_tem_modes
    grid, ds = _coax()
    mode = solve_tem_modes(grid, normal='z', position=20 * ds,
                           compute_params=False)[0]
    with pytest.raises(ValueError, match='impedance'):
        mode.to_source(ws.Sinusoid(frequency=20e9))


def test_to_source_rejects_an_h_only_launch():
    from wavesim.mode_solver import solve_tem_modes
    grid, ds = _coax()
    mode = solve_tem_modes(grid, normal='z', position=20 * ds,
                           compute_params=True)[0]
    with pytest.raises(ValueError, match="must contain 'E'"):
        mode.to_source(ws.Sinusoid(frequency=20e9), fields='H')


def _launched_modal_amplitude(eps_r, directional, *, freq=20e9, nsteps=2200):
    """Forward-wave amplitude of a 1 V ``to_source`` launch, read downstream with
    the mode's own ε-weighted overlap (reads 1.0 for the pure mode)."""
    from wavesim.mode_solver import solve_tem_modes, _NORMAL_CFG, _plane_to_grid
    grid, ds = _coax(nz=160, eps_r=eps_r)
    k_port, k_mon = 30, 110
    mode = solve_tem_modes(grid, normal='z', position=k_port * ds,
                           compute_params=True)[0]
    cpml = ws.init_cpml(grid, d_pml=10, faces=('z0', 'z1'))
    sim = ws.Simulation(grid, cpml=cpml)
    sim.add_source(mode.to_source(ws.Sinusoid(frequency=freq),
                                  fields='EH' if directional else 'E'))

    eps_of = {'Ex': grid.eps_x, 'Ey': grid.eps_y, 'Ez': grid.eps_z}
    gathered, S = {}, 0.0
    for comp in _NORMAL_CFG['z']['E']:
        a, b = np.nonzero(mode.E[comp])
        if a.size == 0:
            continue
        ii, jj, _ = _plane_to_grid('z', k_mon, a, b)
        km = np.full_like(ii, k_mon)
        Ehat, epsr = mode.E[comp][a, b], eps_of[comp][ii, jj, km]
        gathered[comp] = (ii, jj, km, epsr * Ehat)
        S += float(np.sum(epsr * Ehat ** 2))

    vstar = np.zeros(nsteps)
    for s in range(nsteps):
        g = sim.step()
        vstar[s] = sum(float(np.sum((w / S) * getattr(g, c)[ii, jj, km]))
                       for c, (ii, jj, km, w) in gathered.items())
    idx = np.arange(nsteps - 600, nsteps)
    ref = np.exp(-1j * 2 * np.pi * freq * idx * grid.dt)
    return abs(2.0 * np.sum(vstar[idx] * ref) / len(idx))


@pytest.mark.slow
@pytest.mark.parametrize('directional', [True, False])
def test_to_source_launches_one_volt_independent_of_fill(directional):
    """The core fix: a 1 V mode launches ≈ 1 V, on vacuum and on an ε_r = 2.3
    fill alike. The pre-fix additive write gave √ε_r / S_c (≈ 2.3 V on the coax,
    1/S_c ≈ 1.7 V in vacuum) — a fill/Courant-dependent error, not ≈ 1."""
    a_vac = _launched_modal_amplitude(1.0, directional)
    a_fill = _launched_modal_amplitude(2.3, directional)
    assert a_vac == pytest.approx(1.0, abs=0.08), f"vacuum launched {a_vac:.3f} V"
    assert a_fill == pytest.approx(1.0, abs=0.08), f"fill launched {a_fill:.3f} V"
    assert a_fill == pytest.approx(a_vac, rel=0.05)   # not scaling with √ε_r/S_c
