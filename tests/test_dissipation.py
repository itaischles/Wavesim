"""Ohmic dissipation, and what the rest of the solver does about a lossy model.

:class:`~wavesim.monitors.DissipationMonitor` records P = sum(sigma*|E|^2*dV),
the term that makes a lossy run's total energy legitimately decay. The test that
matters is the balance: in a closed PEC cavity with no sources and no PML, the
only place energy can go is the conductivity, so

    U(0) - U(t)  ==  integral of P dt

to the scheme's order. Without that identity a lossy run has no way to tell
absorbed power from a solver leak, which is the whole reason the monitor exists.

How well it closes depends on whether the field is resolved in time, and the two
balance tests below pin both ends of that: 0.1% from a smooth initial field, 22%
high from white noise. The monitor records ``sigma*|E^n|^2`` while the scheme
dissipates ``sigma*|(E^{n+1}+E^n)/2|^2``, and on a Nyquist mode those are not the
same number. Documented on the monitor, and asserted here so it stays known.

Also here: the mode solver warns rather than silently reporting a lossless Z0 as
though it were the real one.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.grid import create_grid
from wavesim.monitors import DissipationMonitor, record_dissipation


# ====================================================================== #
# The integral itself
# ====================================================================== #

def test_power_is_the_sigma_E_squared_integral():
    """Static uniform E in a uniform lossy fill: P = sigma*|E|^2*V, no factor 1/2."""
    shape, ds = (6, 5, 4), 1e-3
    sigma, Ez = 0.25, 3.0

    grid = create_grid(*shape, ds)
    ws.set_vacuum(grid)
    ws.set_dielectric(grid, 1.0, sigma=sigma)
    grid.Ez[...] = Ez

    mon = record_dissipation(DissipationMonitor(), grid)
    volume = np.prod(shape) * ds ** 3
    assert mon.values[0] == pytest.approx(sigma * Ez ** 2 * volume,
                                          rel=1e-12, abs=0.0)


def test_anisotropic_sigma_is_honoured_per_component():
    shape, ds = (4, 4, 4), 1e-3
    grid = create_grid(*shape, ds)
    ws.set_vacuum(grid)
    s = np.ones(shape)
    ws.set_material_arrays(grid, *(np.ones(shape) for _ in range(6)),
                           sigma_x=s * 1.0, sigma_y=s * 2.0, sigma_z=s * 4.0)
    grid.Ex[...], grid.Ey[...], grid.Ez[...] = 1.0, 1.0, 1.0

    mon = record_dissipation(DissipationMonitor(), grid)
    volume = np.prod(shape) * ds ** 3
    assert mon.values[0] == pytest.approx((1.0 + 2.0 + 4.0) * volume,
                                          rel=1e-12, abs=0.0)


def test_lossless_grid_records_zero_rather_than_refusing():
    grid = create_grid(4, 4, 4, 1e-3)
    ws.set_vacuum(grid)
    grid.Ez[...] = 5.0
    mon = record_dissipation(DissipationMonitor(), grid)
    assert mon.values == [0.0]


def test_region_rejects_a_bad_value():
    with pytest.raises(ValueError, match="must be 'full' or 'interior'"):
        DissipationMonitor(region='middle')


# ====================================================================== #
# Energy balance — the reason the monitor exists
# ====================================================================== #

SHAPE, DS = (24, 22, 20), 1e-3


def _cavity_balance(smooth: bool, steps: int = 400):
    """(energy lost, energy dissipated) for a closed lossy PEC cavity.

    No sources and no PML, so conductivity is the only place energy can go.
    """
    grid = create_grid(*SHAPE, DS)
    ws.set_vacuum(grid)
    ws.set_dielectric(grid, 2.0, sigma=0.02)

    if smooth:
        ix, iy, _ = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing='ij')
        grid.Ez[...] = (np.sin(np.pi * ix / (SHAPE[0] - 1))
                        * np.sin(np.pi * iy / (SHAPE[1] - 1)))
    else:
        rng = np.random.default_rng(3)
        for name in ('Ex', 'Ey', 'Ez'):
            getattr(grid, name)[...] = rng.standard_normal(SHAPE)

    energy, power = ws.EnergyMonitor(), DissipationMonitor()
    sim = ws.Simulation(grid, pec_faces=ws.ALL_FACES, monitors=[energy, power])
    sim.run(steps)

    lost = energy.values[0] - energy.values[-1]
    # Trim P to the same samples the energy drop spans.
    dissipated = np.trapezoid(power.values[:-1], power.times[:-1])
    return lost, dissipated


def test_dissipated_energy_accounts_for_the_cavity_loss():
    """Closed PEC cavity, no sources, no PML: U(0) - U(t) == integral of P dt.

    ``abs=0`` matters here and is not decoration: these energies are ~1e-17 J,
    and pytest.approx's default absolute tolerance of 1e-12 would pass this
    assertion no matter what the solver did.
    """
    lost, dissipated = _cavity_balance(smooth=True)
    assert lost > 0.0
    assert dissipated == pytest.approx(lost, rel=0.005, abs=0.0)


def test_unresolved_fields_over_report_dissipation():
    """The documented limitation, pinned so nobody 'fixes' it by accident.

    P is recorded as sigma*|E^n|^2, while the scheme dissipates
    sigma*|(E^{n+1}+E^n)/2|^2. White noise carries Nyquist modes where
    E^{n+1} ~= -E^n, so the average is near zero exactly where |E^n|^2 peaks and
    the monitor reads high. Same cavity, same solver, same monitor as the test
    above, which closes to 0.1% on a resolved field.
    """
    lost, dissipated = _cavity_balance(smooth=False)
    assert dissipated > 1.10 * lost


def test_energy_helper_matches_an_explicit_trapezoid():
    grid = create_grid(10, 10, 10, 1e-3)
    ws.set_vacuum(grid)
    ws.set_dielectric(grid, 1.0, sigma=0.05)
    grid.Ez[...] = 1.0

    mon = DissipationMonitor()
    sim = ws.Simulation(grid, monitors=[mon])
    sim.run(20)
    assert mon.energy() == pytest.approx(
        np.trapezoid(mon.values, mon.times), rel=1e-12, abs=0.0)
    assert DissipationMonitor().energy() == 0.0


# ====================================================================== #
# Simulation plumbing
# ====================================================================== #

def test_interior_region_is_autofilled_from_the_cpml():
    """The trio is filled from the run's CPML, exactly as EnergyMonitor's is."""
    grid = create_grid(40, 40, 1, 1e-3)
    ws.set_vacuum(grid)
    ws.set_dielectric(grid, 1.0, sigma=0.01)
    cpml = ws.init_cpml(grid, d_pml=8, faces=('x0', 'x1'))

    mon = DissipationMonitor(region='interior')
    ws.Simulation(grid, cpml=cpml, monitors=[mon])
    assert mon.d_pml == 8
    assert mon.faces == ('x0', 'x1')


def test_interior_region_excludes_the_pml_shell():
    grid = create_grid(40, 40, 1, 1e-3)
    ws.set_vacuum(grid)
    ws.set_dielectric(grid, 1.0, sigma=0.5)
    grid.Ez[...] = 1.0
    cpml = ws.init_cpml(grid, d_pml=8, faces=('x0', 'x1'))

    full = DissipationMonitor(region='full')
    interior = DissipationMonitor(region='interior')
    # Constructed only to autofill d_pml/faces; recorded by hand so the uniform
    # field survives (a step would make it non-uniform and blur the ratio).
    ws.Simulation(grid, cpml=cpml, monitors=[full, interior])
    record_dissipation(full, grid)
    record_dissipation(interior, grid)

    # 16 of 40 x-cells trimmed; the field is uniform so the ratio is exact.
    assert interior.values[0] == pytest.approx(full.values[0] * 24 / 40,
                                               rel=1e-12, abs=0.0)


def test_monitor_is_registered_with_the_simulation():
    grid = create_grid(8, 8, 8, 1e-3)
    ws.set_vacuum(grid)
    ws.set_dielectric(grid, 1.0, sigma=0.1)
    grid.Ez[...] = 1.0

    sim = ws.Simulation(grid)
    mon = sim.add_monitor(ws.DissipationMonitor())
    sim.run(5)
    assert len(mon.values) == len(mon.times) == 5
    assert all(v > 0.0 for v in mon.values)


# ====================================================================== #
# Mode solver on a lossy cross-section
# ====================================================================== #

def _coax(sigma):
    grid = create_grid(40, 40, 3, 0.5e-3)
    ws.set_vacuum(grid)
    c = 10e-3
    ws.set_coax(grid, c, c, r_inner=2e-3, r_outer=9e-3, eps_r_fill=2.0)
    if sigma:
        ws.set_cylinder(grid, c, c, 9e-3, 0.0, 1.5e-3,
                        eps_r=2.0, sigma=sigma)
    return grid


def test_mode_solver_warns_on_a_lossy_port_plane():
    with pytest.warns(RuntimeWarning, match="lossless cross-section"):
        modes = ws.solve_tem_modes(_coax(0.05), normal='z', position=0.5e-3)
    assert len(modes) == 1


def test_mode_solver_is_silent_on_a_lossless_plane():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ws.solve_tem_modes(_coax(0.0), normal='z', position=0.5e-3)


def test_lossy_plane_reports_the_lossless_Z0():
    """The documented behaviour: solved on Re(eps), so Z0 is sigma-independent."""
    with pytest.warns(RuntimeWarning):
        lossy = ws.solve_tem_modes(_coax(0.05), normal='z', position=0.5e-3)
    lossless = ws.solve_tem_modes(_coax(0.0), normal='z', position=0.5e-3)
    assert lossy[0].impedance == pytest.approx(lossless[0].impedance, rel=1e-12)
    assert lossy[0].eps_eff == pytest.approx(lossless[0].eps_eff, rel=1e-12)


# ====================================================================== #
# viz
# ====================================================================== #

def test_plot_materials_rejects_sigma_on_a_lossless_grid():
    grid = create_grid(8, 8, 1, 1e-3)
    ws.set_vacuum(grid)
    with pytest.raises(ValueError, match="no conductivity"):
        ws.plot_materials_xy(grid, component='sigma_z')
