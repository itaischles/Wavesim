"""The CPML profile knows what it is filled with.

``sigma_max`` used to be one number per face, computed from the vacuum impedance.
:mod:`wavesim.pml` now scales it per cell by ``(mu_r/eps_r)**0.25`` — the module
docstring derives the exponent and records the reflection measurements behind it.
Two things have to hold for that to be worth anything:

* the factor lands in the coefficients the solver actually steps with, exactly,
  on both the E and H side and on every backend; and
* a conductor sitting in the absorber does not get to set the profile with the
  filler permittivity that happens to be inside it.

The reflection gate at the bottom is the end-to-end one: a dielectric-filled PML
absorbs measurably better than the vacuum-tuned profile it replaced.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import EPS0
from wavesim.pml import _material_scale


# ====================================================================== #
# The factor itself
# ====================================================================== #
def _uniform(eps_r=1.0, mu_r=1.0, n=12):
    g = ws.create_grid(n, n, n, 1e-3, 1e-3, 1e-3)
    ws.set_vacuum(g)
    ws.set_material_arrays(g, *[np.full((n, n, n), v) for v in
                                (eps_r, eps_r, eps_r, mu_r, mu_r, mu_r)])
    return g


@pytest.mark.parametrize('eps_r, mu_r', [(1.0, 1.0), (2.3, 1.0), (16.0, 1.0),
                                         (1.0, 4.0), (4.0, 4.0)])
def test_scale_is_the_fourth_root_of_the_impedance_ratio(eps_r, mu_r):
    n = _material_scale(_uniform(eps_r, mu_r))
    assert n == pytest.approx((mu_r / eps_r) ** 0.25, rel=1e-12)


def test_vacuum_is_exactly_one():
    """The vacuum case must not merely be close to 1 — a solver that was correct
    yesterday has to stay bit-identical today."""
    assert np.all(_material_scale(_uniform()) == 1.0)


# ====================================================================== #
# ...reaching the coefficients
# ====================================================================== #
def _sigma_alpha_from_bc(b, c, dt):
    """Recover (sigma, alpha) from a CPML coefficient pair.

    b = exp(-(sigma+alpha)*dt/EPS0) and c = sigma/(sigma+alpha)*(b-1), so
    -ln(b)*EPS0/dt is (sigma+alpha) and c/(b-1) is sigma's share of it. Inverting
    rather than recomputing keeps the test independent of how pml.py factors the
    profile — it only asserts what the recursion will do.
    """
    total = -np.log(b) * EPS0 / dt
    share = np.where(b != 1.0, c / (b - 1.0), 0.0)
    return total * share, total * (1.0 - share)


def test_sigma_scales_by_the_material_and_alpha_does_not():
    """Half the domain in eps_r=16: its sigma is halved, cell for cell, and the
    alpha grading is untouched."""
    N = 60
    g = ws.create_grid(N, N, 1, 1e-3, 1e-3, 1e-3)
    ws.set_vacuum(g)
    g.eps_x[:, N // 2:, :] = g.eps_y[:, N // 2:, :] = 16.0
    g.eps_z[:, N // 2:, :] = 16.0
    cpml = ws.init_cpml(g, d_pml=10)

    # x-axis slabs run across the material step, so one array carries both halves.
    for b_s, c_s, b1, c1 in ((cpml.bxE_s, cpml.cxE_s, cpml.bx_E, cpml.cx_E),
                             (cpml.bxH_s, cpml.cxH_s, cpml.bx_H, cpml.cx_H)):
        sel = cpml.sel_xE if b_s is cpml.bxE_s else cpml.sel_xH
        sig, alp = _sigma_alpha_from_bc(b_s, c_s, g.dt)
        sig0, alp0 = _sigma_alpha_from_bc(b1[sel], c1[sel], g.dt)
        sig0 = sig0.reshape(-1, 1, 1)

        air, diel = np.s_[:, :N // 2, :], np.s_[:, N // 2:, :]
        assert sig[air] == pytest.approx(np.broadcast_to(sig0, sig.shape)[air],
                                         rel=1e-10)
        assert sig[diel] == pytest.approx(
            0.5 * np.broadcast_to(sig0, sig.shape)[diel], rel=1e-10)
        assert alp == pytest.approx(
            np.broadcast_to(alp0.reshape(-1, 1, 1), alp.shape), rel=1e-10)


def test_vacuum_run_is_unchanged_by_the_feature():
    """A vacuum model steps exactly as it did when the profile was 1D."""
    g = ws.create_grid(40, 40, 1, 1e-3, 1e-3, 1e-3)
    ws.set_vacuum(g)
    cpml = ws.init_cpml(g, d_pml=10)
    assert np.array_equal(cpml.bxE_s,
                          np.broadcast_to(cpml.bx_E[cpml.sel_xE].reshape(-1, 1, 1),
                                          cpml.bxE_s.shape))
    assert np.array_equal(cpml.byH_s,
                          np.broadcast_to(cpml.by_H[cpml.sel_yH].reshape(1, -1, 1),
                                          cpml.byH_s.shape))


# ====================================================================== #
# PEC in the absorber
# ====================================================================== #
def test_conductor_in_the_pml_takes_the_material_around_it():
    """A PEC bar crossing the absorber carries whatever permittivity the
    voxeliser left inside it — here 1.0, in a substrate of 16. That filler is not
    a material and must not reach the profile: the metal cells grade with the
    substrate."""
    N = 48
    g = ws.create_grid(N, N, N, 1e-3, 1e-3, 1e-3)
    ws.set_vacuum(g)
    ws.set_dielectric(g, 16.0)
    # A bar along x, running out through both x faces and their PML shells.
    ws.set_box(g, 0.0, N * 1e-3, 20e-3, 24e-3, 20e-3, 24e-3,
               eps_r=16.0, pec=True)
    # Whatever a voxeliser leaves inside the metal — vacuum here, but it is
    # arbitrary — must not reach the profile.
    for comp in ('eps_x', 'eps_y', 'eps_z'):
        getattr(g, comp)[g.pec_mask] = 1.0
    assert g.eps_z[2, 22, 22] == 1.0          # the filler really is in there

    n = _material_scale(g)
    assert n[2, 22, 22] == pytest.approx(16.0 ** -0.25, rel=1e-12)
    assert np.all(n == pytest.approx(16.0 ** -0.25, rel=1e-12))


def test_a_conductor_ending_in_the_pml_stays_bounded():
    """PEC through the absorber, driven hard, with the whole model in a
    substrate: the run must decay, not grow. A conductor in a PML is the classic
    place for a slow instability to hide, and the material-scaled profile is a
    new coefficient on exactly those cells."""
    N = 40
    g = ws.create_grid(N, N, N, 1e-3, 1e-3, 1e-3)
    ws.set_vacuum(g)
    ws.set_dielectric(g, 9.0)
    ws.set_box(g, 0.0, N * 1e-3, 18e-3, 22e-3, 18e-3, 22e-3,
               eps_r=1.0, pec=True)
    cpml = ws.init_cpml(g, d_pml=8)
    sim = ws.Simulation(g, cpml=cpml)
    sim.add_source(ws.PointSource('Ez', 20e-3, 26e-3, 20e-3,
                                  ws.GaussianPulse.for_fmax(30e9)))

    energy = []
    for n in range(600):
        sim.step()
        energy.append(float(np.sum(g.Ez ** 2 + g.Ey ** 2 + g.Ex ** 2)))
    energy = np.array(energy)
    assert np.all(np.isfinite(energy))
    # The floor is not zero and should not be: a unipolar pulse deposits net
    # charge on the bar, and the static field that leaves behind is physical —
    # no absorber removes it. What must not happen is growth. Measured: a
    # monotone decay to 16.22 by step 1200, from a peak of 20.94.
    half = len(energy) // 2
    assert energy[-1] <= energy[half] * (1 + 1e-6)
    assert energy.max() == pytest.approx(energy[:half].max(), rel=1e-12)


# ====================================================================== #
# Backends
# ====================================================================== #
def test_numba_matches_numpy_on_a_multi_material_grid():
    """The per-cell coefficients are indexed by three subscripts in the kernels
    and by broadcasting in the reference. Same answer, to the bit."""
    numba = pytest.importorskip('numba')          # noqa: F841

    def build():
        g = ws.create_grid(30, 28, 26, 1e-3, 1e-3, 1e-3)
        ws.set_vacuum(g)
        ws.set_box(g, 5e-3, 25e-3, 0.0, 28e-3, 0.0, 26e-3, eps_r=9.0)
        return g

    out = []
    for backend in ('numpy', 'numba'):
        g = build()
        sim = ws.Simulation(g, cpml=ws.init_cpml(g, d_pml=6), backend=backend)
        sim.add_source(ws.PointSource('Ez', 15e-3, 14e-3, 13e-3,
                                      ws.GaussianPulse.for_fmax(30e9)))
        for _ in range(40):
            sim.step()
        out.append(g.Ez.copy())
    assert np.array_equal(out[0], out[1])


# ====================================================================== #
# What it is all for
# ====================================================================== #
def _reflection_db(eps_r, material_aware, N=100, D=10, pad=110):
    """Residual at a probe beside the absorber, against an extended-domain
    reference, in dB relative to the incident peak.

    The reference domain is padded far enough that its own boundary cannot answer
    within the window, so the difference between the two traces is the test
    domain's PML reflection and nothing else.
    """
    import wavesim.pml as pml

    src, probe = N // 2, N - D - 12
    n_idx = np.sqrt(eps_r)
    nt, fmax = int(500 * n_idx), 20e9 / n_idx     # same wavelength in cells

    def trace(size, s, p, aware):
        orig = pml._material_scale
        if not aware:
            pml._material_scale = lambda g: np.ones((g.Nx, g.Ny, g.Nz))
        try:
            g = ws.create_grid(size, size, 1, 1e-3, 1e-3, 1e-3)
            ws.set_vacuum(g)
            ws.set_dielectric(g, eps_r)
            sim = ws.Simulation(g, cpml=ws.init_cpml(g, d_pml=D),
                                backend='numba')
            sim.add_source(ws.PointSource('Ez', s * g.dx, s * g.dy, 0.0,
                                          ws.GaussianPulse.for_fmax(fmax)))
            out = np.empty(nt)
            for i in range(nt):
                sim.step()
                out[i] = g.Ez[p, s, 0]
            return out
        finally:
            pml._material_scale = orig

    ref = trace(N + 2 * pad, src + pad, probe + pad, True)
    got = trace(N, src, probe, material_aware)
    return 20 * np.log10(np.abs(got - ref).max() / np.abs(ref).max())


@pytest.mark.slow
def test_dielectric_filled_pml_reflects_less_than_a_vacuum_profile():
    """eps_r=9 fill: the material-scaled profile is several dB quieter than the
    vacuum-tuned one it replaced. Measured here: -84.7 dB against -78.9 dB (and
    -85.6 against -81.9 at eps_r=4). The gate keeps 3 dB of that."""
    pytest.importorskip('numba')
    aware = _reflection_db(9.0, True)
    vacuum = _reflection_db(9.0, False)
    assert aware < vacuum - 3.0, f"material {aware:.1f} dB, vacuum {vacuum:.1f} dB"
