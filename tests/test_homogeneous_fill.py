"""Homogeneous-fill identities — zero-tolerance invariants.

A transmission line whose cross-section is filled with a *single* dielectric
satisfies LC = με exactly, so ε_eff = ε_r **regardless of the conductor
geometry**. Staircasing, wrong radii, an asymmetric PEC rasterisation — none of
it can move ε_eff; it can only move Z₀. That makes ε_eff an exact assertion
rather than a tolerance, and it is the sharpest available probe of the two
solvers agreeing about the same structure.

Two real bugs were found by this invariant and are pinned here:

* the 2D mode solver averaged ε across faces onto PEC cells, whose ε is not a
  material property but whatever the voxeliser left there. That broke the scalar
  -multiple relationship between the filled and air systems, so φ ≠ φ_air and
  the cancellation in ε_eff = C/C_air failed (ε_eff read ~2.14 at ε_r = 2.3).
  ``_attach_params`` compounded it by integrating spurious ``np.gradient``
  energy from inside the conductors.
* ``apply_pec_mask`` zeroed only the three E-edges carrying a PEC cell's own
  index, leaving nine of its twelve edges alive — so E survived half a cell
  inside the metal on each conductor's high-x/y/z faces. E and H then saw
  different geometry and the wave ran 6.8% slow (ε_eff 2.65 against 2.30).
"""
import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import C0
from wavesim.mode_solver import solve_tem_modes
from wavesim.pec import build_pec_edge_masks

# RG58-like coax.
R_IN = 0.405e-3      # id 0.81 mm
R_OUT = 1.475e-3     # od 2.95 mm


def _coax_grid(n_transverse, eps_r, nz=3):
    ds = (2.6 * R_OUT) / n_transverse
    grid = ws.create_grid(Nx=n_transverse, Ny=n_transverse, Nz=nz,
                          dx=ds, dy=ds, dz=ds)
    ws.set_vacuum(grid)
    c = 0.5 * n_transverse * ds
    ws.set_coax(grid, cx=c, cy=c, r_inner=R_IN, r_outer=R_OUT, eps_r_fill=eps_r)
    return grid, ds


# ---------------------------------------------------------------------- #
# 2D mode solver
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [33, 66, 132])
def test_eps_eff_exact_at_every_resolution(n):
    """ε_eff = ε_r to machine precision, independent of mesh density.

    This is the zero-tolerance form: the identity does not converge with
    resolution, it holds outright. Before the face-ε and energy-masking fixes
    this read 2.152 / 2.218 / 2.257 for n = 33 / 66 / 132.
    """
    eps_r = 2.3
    grid, ds = _coax_grid(n, eps_r)
    mode = solve_tem_modes(grid, normal='z', position=1.5 * ds,
                           compute_params=True)[0]
    assert mode.eps_eff == pytest.approx(eps_r, abs=1e-10)


@pytest.mark.parametrize("eps_r", [1.0, 2.3, 4.4, 10.2])
def test_eps_eff_exact_across_permittivities(eps_r):
    """The identity holds for any fill value, including vacuum."""
    grid, ds = _coax_grid(48, eps_r)
    mode = solve_tem_modes(grid, normal='z', position=1.5 * ds,
                           compute_params=True)[0]
    assert mode.eps_eff == pytest.approx(eps_r, abs=1e-10)


def test_phase_velocity_matches_eps_eff():
    """v_phase, ε_eff and the C/L pair stay mutually consistent."""
    eps_r = 2.3
    grid, ds = _coax_grid(64, eps_r)
    mode = solve_tem_modes(grid, normal='z', position=1.5 * ds,
                           compute_params=True)[0]
    assert mode.v_phase == pytest.approx(C0 / np.sqrt(eps_r), rel=1e-10)
    assert 1.0 / np.sqrt(mode.inductance * mode.capacitance) == pytest.approx(
        C0 / np.sqrt(eps_r), rel=1e-10)


def test_impedance_converges_to_analytic():
    """Z₀ *does* depend on rasterisation — it converges rather than being exact.

    Guards against 'fixing' ε_eff by breaking the capacitance. First-order
    convergence toward the analytic coax value is genuine staircase physics.
    """
    eps_r = 2.3
    z_analytic = (376.730313 / (2 * np.pi * np.sqrt(eps_r))) * np.log(R_OUT / R_IN)
    errs = []
    for n in (48, 96, 192):
        grid, ds = _coax_grid(n, eps_r)
        mode = solve_tem_modes(grid, normal='z', position=1.5 * ds,
                               compute_params=True)[0]
        errs.append(abs(mode.impedance - z_analytic))
    assert errs[0] > errs[1] > errs[2], f"Z0 not converging: {errs}"
    assert errs[-1] / z_analytic < 0.10


# ---------------------------------------------------------------------- #
# PEC edge rasterisation
# ---------------------------------------------------------------------- #

def test_pec_edge_masks_cover_all_twelve_edges():
    """One PEC cell owns twelve E-edges; all of them must be masked.

    With Ex[i,j,k] spanning node (i,j,k) → (i+1,j,k), cell (i,j,k) has x-edges
    at (j,k), (j+1,k), (j,k+1), (j+1,k+1), and so on by symmetry.
    """
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    ex, ey, ez = build_pec_edge_masks(mask)

    assert set(map(tuple, np.argwhere(ex))) == {
        (2, 2, 2), (2, 3, 2), (2, 2, 3), (2, 3, 3)}
    assert set(map(tuple, np.argwhere(ey))) == {
        (2, 2, 2), (3, 2, 2), (2, 2, 3), (3, 2, 3)}
    assert set(map(tuple, np.argwhere(ez))) == {
        (2, 2, 2), (2, 3, 2), (3, 2, 2), (3, 3, 2)}
    assert ex.sum() == ey.sum() == ez.sum() == 4


def test_edge_mask_count_for_a_block():
    """A 4×4×4 PEC block owns 4×5×5 = 100 x-edges, not 64.

    The cell-wise rule masked one edge per cell; the correct count is the number
    of *edges*, which exceeds the cell count on every conductor.
    """
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[6:10, 6:10, 6:10] = True
    ex, ey, ez = build_pec_edge_masks(mask)
    assert mask.sum() == 64
    assert ex.sum() == ey.sum() == ez.sum() == 4 * 5 * 5


def test_apply_pec_mask_leaves_no_field_on_conductor_edges():
    """After apply_pec_mask no E-edge touching a PEC cell may be nonzero."""
    grid = ws.create_grid(Nx=8, Ny=8, Nz=8, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(grid)
    grid.pec_mask = np.zeros((8, 8, 8), dtype=bool)
    grid.pec_mask[3:5, 3:5, 3:5] = True

    rng = np.random.default_rng(0)
    for comp in ('Ex', 'Ey', 'Ez'):
        getattr(grid, comp)[...] = rng.standard_normal(grid.Ex.shape)

    ws.apply_pec_mask(grid)

    ex, ey, ez = build_pec_edge_masks(grid.pec_mask)
    assert not grid.Ex[ex].any()
    assert not grid.Ey[ey].any()
    assert not grid.Ez[ez].any()
    # and nothing outside the conductor was touched
    assert grid.Ex[~ex].all()


# ---------------------------------------------------------------------- #
# 3D propagation — the end-to-end form of the same identity
# ---------------------------------------------------------------------- #

@pytest.mark.slow
def test_cuda_pec_matches_numpy():
    """The device-side PEC kernel must apply the same edge rule as the host.

    ``CudaResident`` runs PEC on the GPU, so it needs the per-component masks
    too — it takes them from :func:`build_pec_edge_masks` on the host, which is
    what makes this parity exact rather than approximate.
    """
    from conftest import cuda_available
    if not cuda_available():
        pytest.skip("no CUDA device")

    n, steps = 24, 40

    def run(backend):
        grid = ws.create_grid(Nx=n, Ny=n, Nz=n, dx=1e-3, dy=1e-3, dz=1e-3)
        ws.set_vacuum(grid)
        grid.pec_mask = np.zeros((n, n, n), dtype=bool)
        grid.pec_mask[9:13, 9:13, 9:13] = True
        src = ws.PointSource('Ez', 5e-3, 5e-3, 5e-3,
                             ws.GaussianPulse.for_fmax(60e9))
        ws.Simulation(grid, sources=[src], backend=backend).run(steps)
        return grid

    a, b = run('numpy'), run('cuda')
    for comp in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        x, y = getattr(a, comp), getattr(b, comp)
        scale = max(np.abs(x).max(), 1e-30)
        assert np.abs(x - y).max() / scale < 1e-10, f"{comp} diverged"

    ex, ey, ez = build_pec_edge_masks(a.pec_mask)
    for g in (a, b):
        assert not g.Ex[ex].any() and not g.Ey[ey].any() and not g.Ez[ez].any()


@pytest.mark.slow
def test_coax_propagation_velocity_matches_fill():
    """A pulse on a uniform coax travels at c/√ε_r, whatever the staircase.

    Deliberately run on a coarse transverse mesh (the inner conductor is only a
    few cells across) precisely because velocity must not care. Before the
    apply_pec_mask fix this measured 1.842e8 m/s — 6.8% slow, ε_eff 2.65.
    """
    eps_r = 2.3
    n, nz, f_max = 32, 256, 40e9
    k_src, k_p1, k_p2 = 24, 70, 200

    grid, ds = _coax_grid(n, eps_r, nz=nz)
    mode = solve_tem_modes(grid, normal='z', position=k_src * ds,
                           compute_params=True)[0]
    src = mode.to_source(ws.GaussianPulse.for_fmax(f_max), fields='EH')

    c = 0.5 * n * ds
    i_p = int((c + 0.5 * (R_IN + R_OUT)) / ds)
    j_p = int(c / ds)
    probes = [ws.FieldProbe(component='Ex', x=i_p * ds, y=j_p * ds, z=k * ds)
              for k in (k_p1, k_p2)]

    cpml = ws.init_cpml(grid, faces=('z0', 'z1'))
    sim = ws.Simulation(grid, cpml=cpml, sources=[src], monitors=probes,
                        backend='numpy')
    v_true = C0 / np.sqrt(eps_r)
    sim.run(int(1.6 * (k_p2 - k_src) * ds / v_true / grid.dt))

    def peak_time(p):
        v = np.abs(np.asarray(p.values))
        t = np.asarray(p.times)
        i = int(np.argmax(v))
        y0, y1, y2 = v[i - 1], v[i], v[i + 1]
        d = y0 - 2 * y1 + y2
        return t[i] + (0.5 * (y0 - y2) / d if d else 0.0) * (t[1] - t[0])

    v_meas = (k_p2 - k_p1) * ds / (peak_time(probes[1]) - peak_time(probes[0]))
    assert v_meas == pytest.approx(v_true, rel=0.01), (
        f"v = {v_meas:.4e} m/s vs {v_true:.4e} "
        f"(eps_eff {(C0 / v_meas) ** 2:.4f} vs {eps_r})")
