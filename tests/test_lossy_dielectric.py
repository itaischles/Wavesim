"""Lossy dielectrics — the two-coefficient E update (wavesim.loss).

Four properties, in increasing order of how much of the solver they involve:

  * **sigma = 0 is bit-identical**, whether the arrays are absent (a dispatch
    property, exact by construction) or present and zero (an arithmetic
    property, exact because ``k`` is exactly 0 and the coefficients collapse to
    ``1`` and ``dt/(eps0*eps)``);
  * **the coefficients are the Pade form** and converge on ``exp(-dt/tau)`` at
    second order;
  * **a plane wave decays at the analytic rate** in a uniform lossy medium;
  * **the CPML still absorbs inside that medium**, which is the gate on the one
    change that is easy to get wrong quietly — the psi correction must carry
    ``Cb``, not ``dt/(eps0*eps)``.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import C0, EPS0, MU0
from wavesim.grid import create_grid
from wavesim.loss import build_loss_coefficients, loss_coefficients


# ====================================================================== #
# sigma = 0 changes nothing
# ====================================================================== #

def _pulse_run(sigma, n_steps=300, shape=(60, 8, 1), ds=1e-3):
    """A short CPML run driven by a point source; returns the final fields."""
    grid = create_grid(*shape, ds)
    ws.set_vacuum(grid)
    if sigma is not None:
        z = np.full(shape, sigma, dtype=grid.eps_x.dtype)
        ws.set_material_arrays(grid, *(np.ones(shape) for _ in range(6)),
                               sigma_x=z, sigma_y=z.copy(), sigma_z=z.copy())
    cpml = ws.init_cpml(grid, d_pml=8, faces=('x0', 'x1'))
    sim = ws.Simulation(grid, cpml=cpml)
    sim.add_source(ws.PointSource('Ez', 30 * ds, 4 * ds, 0.0,
                                  ws.GaussianPulse.for_fmax(60e9)))
    sim.run(n_steps)
    return {n: getattr(sim.grid, n).copy()
            for n in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')}


def test_absent_sigma_takes_the_lossless_path():
    grid = create_grid(4, 4, 1, 1e-3)
    ws.set_vacuum(grid)
    assert not grid.is_lossy
    assert loss_coefficients(grid) is None


def test_zero_sigma_is_bit_identical_to_no_sigma():
    """All-zero sigma arrays step the lossy branch and must not perturb a bit."""
    lossless = _pulse_run(None)
    zeroed = _pulse_run(0.0)
    for name, ref in lossless.items():
        assert np.array_equal(ref, zeroed[name]), f"{name} differs at sigma=0"


def test_zero_sigma_coefficients_are_exact():
    grid = create_grid(5, 4, 3, 1e-3)
    ws.set_vacuum(grid)
    grid.eps_x[:] = 2.5                      # non-unit eps, so Cb is not trivial
    z = np.zeros((5, 4, 3))
    grid.sigma_x, grid.sigma_y, grid.sigma_z = z, z.copy(), z.copy()

    c = build_loss_coefficients(grid)
    assert np.array_equal(c.Ca_x, np.ones((5, 4, 3)))
    # Character-for-character the expression update.py evaluates.
    assert np.array_equal(c.Cb_x, grid.dt / (EPS0 * grid.eps_x))


# ====================================================================== #
# The coefficients themselves
# ====================================================================== #

def test_coefficients_match_the_closed_form():
    grid = create_grid(4, 4, 1, 1e-3)
    ws.set_vacuum(grid)
    ws.set_box(grid, 0.0, 4e-3, 0.0, 4e-3, 0.0, 1e-3, eps_r=4.0, sigma=0.3)

    dt, eps, sigma = grid.dt, 4.0, 0.3
    k = sigma * dt / (2 * EPS0 * eps)
    c = loss_coefficients(grid)
    assert np.allclose(c.Ca_z, (1 - k) / (1 + k), rtol=0, atol=1e-15)
    assert np.allclose(c.Cb_z, (dt / (EPS0 * eps)) / (1 + k), rtol=1e-15)


@pytest.mark.parametrize('refine', [1, 2, 4, 8])
def test_relaxation_converges_at_second_order(refine):
    """``Ca**n`` is the (1,1) Pade of ``exp(-dt/tau)``: error falls as dt^2.

    Checked as a plain scalar recursion — this is a property of the coefficient,
    not of the curl, and mixing the two would only add dispersion error.
    """
    eps_r, sigma = 3.0, 0.05
    tau = EPS0 * eps_r / sigma
    t_end = 0.5 * tau

    dt = (1e-12) / refine
    n = int(round(t_end / dt))
    k = sigma * dt / (2 * EPS0 * eps_r)
    err = abs(((1 - k) / (1 + k)) ** n - np.exp(-t_end / tau))
    # Second order: err*refine^2 is constant. Pin the constant loosely and let
    # the parametrisation prove the slope.
    assert err * refine ** 2 < 1.5e-2
    assert err < 1.5e-2 / refine ** 2


def test_metal_scale_sigma_warns():
    """Ca < 0 is the regime where the update stops resembling a decay."""
    grid = create_grid(4, 4, 1, 1e-4)
    ws.set_vacuum(grid)
    with pytest.warns(RuntimeWarning, match="alternates sign"):
        ws.set_box(grid, 0.0, 4e-4, 0.0, 4e-4, 0.0, 1e-4,
                   eps_r=1.0, sigma=5.8e7)
        loss_coefficients(grid)


# ====================================================================== #
# Physics: plane-wave attenuation in a uniform lossy medium
# ====================================================================== #

def _analytic_alpha(f, eps_r, sigma, mu_r=1.0):
    """Attenuation constant (Np/m) of a plane wave in a lossy dielectric."""
    w = 2 * np.pi * f
    eps, mu = EPS0 * eps_r, MU0 * mu_r
    return w * np.sqrt(mu * eps / 2) * np.sqrt(np.sqrt(1 + (sigma / (w * eps)) ** 2) - 1)


F_CW = 15e9
DS = 1e-3
D_PML = 12


def _launch(grid, waveform, d_pml=D_PML):
    """A directional +x plane-wave launch on a lossy slice, and its Simulation.

    The recipe is ``tests/test_poynting.py``'s: a very wide-waist GaussianBeam
    against PEC y-walls excites Ey/Hz only, which is the parallel-plate TEM mode
    — exactly 1D, no cutoff, and one-way, so the measurement window sees only
    the outgoing wave.

    Deliberately *not* a soft PointSource sheet. ``update_E`` never writes
    ``Ez[:, 0, :]``, so a soft source on that edge accumulates forever and drives
    the run instead of exciting it (the trap behind
    ``stability._pin_unupdated_edges``); an earlier draft of this test measured a
    *negative* alpha for exactly that reason.
    """
    beam = ws.GaussianBeam('x0', angle=0.0, waveform=waveform,
                           waist=10.0, d_pml=d_pml, directional=True)
    cpml = ws.init_cpml(grid, d_pml=d_pml, faces=('x0', 'x1'))
    return ws.Simulation(grid, cpml=cpml, sources=[beam],
                         pec_faces=('y0', 'y1'))


@pytest.mark.slow
def test_plane_wave_attenuates_at_the_analytic_rate():
    """Steady-state |Ey| envelope decays as exp(-alpha x) with the textbook alpha.

    tan delta = 0.1, so the medium is lossy enough for the envelope to fall 12x
    across the window and mild enough that the launch's vacuum-impedance pairing
    stays clean. Measured error is 0.8%, essentially all of it the 20-cell
    wavelength.
    """
    eps_r, sigma = 1.0, 0.0834               # tan delta = 0.1 at 15 GHz
    Nx, Ny = 260, 24

    grid = create_grid(Nx, Ny, 1, DS)
    ws.set_vacuum(grid)
    ws.set_dielectric(grid, eps_r, sigma=sigma)
    sim = _launch(grid, ws.Sinusoid(frequency=F_CW))

    # Run past the domain transit (13 periods), then take the envelope over one
    # full period.
    n_period = int(round(1.0 / (F_CW * grid.dt)))
    sim.run(25 * n_period)
    envelope = np.zeros(Nx)
    for _ in range(n_period + 1):
        sim.step()
        envelope = np.maximum(envelope, np.abs(sim.grid.Ey[:, Ny // 2, 0]))

    # Clear of the launch and of the far absorber.
    i0, i1 = 40, 200
    x = np.arange(i0, i1) * DS
    alpha = -np.polyfit(x, np.log(envelope[i0:i1]), 1)[0]

    expected = _analytic_alpha(F_CW, eps_r, sigma)
    assert abs(alpha - expected) / expected < 0.02, (
        f"alpha = {alpha:.4f} Np/m, analytic {expected:.4f} Np/m")


# ====================================================================== #
# The CPML correction carries Cb, not dt/(eps0*eps)
# ====================================================================== #

def test_pml_psi_correction_carries_Cb():
    """The psi memory term is part of the same curl, so it takes the same Cb.

    Pinned exactly rather than through a physical observable, because the
    discrepancy is small: Cb and dt/(eps0*eps) differ by 1/(1+k) with
    k = sigma*dt/(2*eps) = dt/(2*tau), which is ~2e-3 for a well-resolved lossy
    dielectric and only approaches 1 in the regime wavesim.loss already refuses
    to be quiet about. A reflection-level test cannot see 0.2%; this can.

    The construction: start both a lossy and a lossless grid from E = 0 with
    identical H and identical (zeroed) psi. The psi recursion depends only on H
    and the (b, c) profiles, so both grids build the *same* psi. With E = 0 the
    Ca term drops out, so a whole step of update_E + update_E_pml leaves

        E_lossless = (dt/(eps0*eps)) * (curl + psi)
        E_lossy    = Cb             * (curl + psi)      = E_lossless / (1 + k)

    — one uniform ratio over every cell, interior and PML slab alike. Applying
    dt/(eps0*eps) to the psi term instead breaks it only inside the slabs, which
    is exactly the error being excluded.
    """
    shape, sigma, eps_r = (24, 22, 20), 0.6, 2.5
    k = sigma * 0.5 / (EPS0 * eps_r)          # dt factored in below

    def stepped(lossy):
        grid = create_grid(*shape, 1e-3)
        ws.set_vacuum(grid)
        ws.set_dielectric(grid, eps_r, sigma=sigma if lossy else 0.0)
        rng = np.random.default_rng(7)
        for name in ('Hx', 'Hy', 'Hz'):
            getattr(grid, name)[...] = rng.standard_normal(shape)
        cpml = ws.init_cpml(grid, d_pml=6)
        grid = ws.update_E(grid)
        grid, _ = ws.update_E_pml(grid, cpml)
        return grid

    lossy, lossless = stepped(True), stepped(False)
    ratio = 1.0 / (1.0 + k * lossy.dt)

    for name in ('Ex', 'Ey', 'Ez'):
        got, ref = getattr(lossy, name), getattr(lossless, name)
        assert np.abs(ref).max() > 0.0
        np.testing.assert_allclose(got, ref * ratio, rtol=1e-12, atol=0.0,
                                   err_msg=f"{name}: psi correction is not Cb")


# ====================================================================== #
# The CPML must still absorb inside a lossy medium
# ====================================================================== #

@pytest.mark.slow
def test_cpml_absorbs_inside_a_lossy_medium():
    """The absorber still works when the medium it terminates is lossy.

    A short domain and a long one are driven identically and probed at the same
    physical point. In the long domain the wave never comes back, so any late
    difference at the probe *is* the short domain's PML reflection.
    """
    eps_r, sigma = 1.0, 0.02                 # tan delta ~ 0.02 at 15 GHz
    Ny, probe_i = 24, 60
    Nx_short, Nx_long = 150, 600

    def run(Nx, n_steps):
        grid = create_grid(Nx, Ny, 1, DS)
        ws.set_vacuum(grid)
        ws.set_dielectric(grid, eps_r, sigma=sigma)
        sim = _launch(grid, ws.GaussianPulse.for_fmax(20e9))
        probe = sim.add_monitor(
            ws.FieldProbe('Ey', probe_i * DS, (Ny // 2) * DS, 0.0))
        sim.run(n_steps)
        return np.asarray(probe.values)

    n_steps = 1200
    short, long = run(Nx_short, n_steps), run(Nx_long, n_steps)

    incident = np.max(np.abs(long))
    # The incident pulse has long since passed the probe by the halfway mark;
    # what is left in the short run is the round trip off the right-hand PML.
    tail = slice(n_steps // 2, None)
    reflection = np.max(np.abs(short[tail] - long[tail])) / incident
    assert reflection < 5e-3, f"PML reflection in lossy medium: {reflection:.2e}"


# ====================================================================== #
# API guards
# ====================================================================== #

def test_pec_and_sigma_are_mutually_exclusive():
    grid = create_grid(8, 8, 1, 1e-3)
    ws.set_vacuum(grid)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ws.set_box(grid, 0.0, 4e-3, 0.0, 4e-3, 0.0, 1e-3,
                   eps_r=1.0, pec=True, sigma=1.0)


def test_subpixel_refuses_a_lossy_shape():
    grid = create_grid(8, 8, 1, 1e-3)
    ws.set_vacuum(grid)
    with pytest.raises(NotImplementedError, match="frequency-dependent"):
        ws.set_box(grid, 1e-3, 4e-3, 1e-3, 4e-3, 0.0, 1e-3,
                   eps_r=4.0, sigma=0.1, subpixel=True)


def test_partial_sigma_set_is_rejected():
    shape = (6, 5, 4)
    grid = create_grid(*shape, 1e-3)
    ones = [np.ones(shape) for _ in range(6)]
    with pytest.raises(ValueError, match="all three components"):
        ws.set_material_arrays(grid, *ones, sigma_x=np.zeros(shape))


def test_negative_sigma_is_rejected():
    shape = (6, 5, 4)
    grid = create_grid(*shape, 1e-3)
    ones = [np.ones(shape) for _ in range(6)]
    bad = np.zeros(shape)
    bad[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        ws.set_material_arrays(grid, *ones, sigma_x=bad,
                               sigma_y=np.zeros(shape), sigma_z=np.zeros(shape))


def test_set_vacuum_clears_conductivity():
    grid = create_grid(8, 8, 1, 1e-3)
    ws.set_vacuum(grid)
    ws.set_box(grid, 0.0, 8e-3, 0.0, 8e-3, 0.0, 1e-3, eps_r=4.0, sigma=0.5)
    assert grid.is_lossy
    ws.set_vacuum(grid)
    assert not grid.is_lossy


def test_placing_lossless_material_clears_a_lossy_region():
    """sigma=0 over a lossy region must actually zero it, not leave it behind."""
    grid = create_grid(8, 8, 1, 1e-3)
    ws.set_vacuum(grid)
    ws.set_box(grid, 0.0, 8e-3, 0.0, 8e-3, 0.0, 1e-3, eps_r=4.0, sigma=0.5)
    ws.set_box(grid, 0.0, 4e-3, 0.0, 8e-3, 0.0, 1e-3, eps_r=2.0)
    assert np.all(grid.sigma_z[:4] == 0.0)
    assert np.all(grid.sigma_z[4:] == 0.5)


# ====================================================================== #
# Interaction with the rest of the solver
# ====================================================================== #

def test_pec_mask_wins_over_conductivity():
    """Overlap is resolved by apply_pec_mask, which runs after the E update."""
    grid = create_grid(8, 8, 1, 1e-3)
    ws.set_vacuum(grid)
    ws.set_box(grid, 0.0, 8e-3, 0.0, 8e-3, 0.0, 1e-3, eps_r=4.0, sigma=0.5)
    grid.pec_mask = np.zeros((8, 8, 1), dtype=bool)
    grid.pec_mask[2:5, 2:5, :] = True
    grid.Ez[...] = 1.0

    sim = ws.Simulation(grid)
    sim.step()
    assert np.all(sim.grid.Ez[2:5, 2:5, :] == 0.0)


def test_max_stable_dt_ignores_conductivity():
    """The CFL limit is a property of the curl; sigma damps but does not bound it.

    Also the regression guard on _curl_curl_operator: with sigma left live the
    operator is not symmetric and eigsh would silently solve a different problem.
    """
    def margin(sigma):
        grid = create_grid(12, 11, 10, 1e-3)
        ws.set_vacuum(grid)
        ws.set_dielectric(grid, 2.0, sigma=sigma)
        return ws.stability_margin(grid, tol=1e-6)

    assert margin(0.5) == pytest.approx(margin(0.0), rel=1e-6)


def test_probe_growth_sees_a_lossy_grid_as_stable():
    grid = create_grid(16, 15, 1, 1e-3)
    ws.set_vacuum(grid)
    ws.set_dielectric(grid, 2.0, sigma=0.2)
    probe = ws.probe_growth(grid, steps=200, chunk=50)
    assert probe.stable
    assert probe.growth < 1.0            # it is a decay, not merely bounded


def test_loss_cache_tracks_dt():
    """dt is in the cache key; a copied grid with a different dt must not reuse."""
    import copy

    grid = create_grid(6, 5, 4, 1e-3)
    ws.set_vacuum(grid)
    ws.set_dielectric(grid, 2.0, sigma=0.4)
    first = loss_coefficients(grid)
    assert loss_coefficients(grid) is first          # cached, not rebuilt

    other = copy.copy(grid)
    other.dt = grid.dt * 0.5
    assert not np.array_equal(loss_coefficients(other).Ca_x, first.Ca_x)
