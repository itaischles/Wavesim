"""
test_electrostatics.py — the 3D Poisson solve (wavesim.electrostatics).

The tests are ordered by how much they would tell us if they broke. First the
cases with a closed-form answer the discretisation should reproduce *exactly* —
a parallel plate is a linear ramp and a two-layer plate is a known kink, both
representable on the mesh, so anything but round-off agreement is a real defect
rather than discretisation error. Then the invariants, which need no analytic
solution and catch the subtler class of bug: a uniform permittivity must scale
out of the answer completely, and a symmetry plane must reproduce the mirrored
full solve. Only then the guard rails.

Capacitance, energy and the E/D fields arrive in the next commit; everything
here is about phi.
"""

import warnings

import numpy as np
import pytest

import wavesim as ws
from wavesim.electrostatics import Electrostatics, _resolve_boundary


DS = 1e-3


def _grid(Nx=12, Ny=12, Nz=20, ds=DS):
    return ws.set_vacuum(ws.create_grid(Nx, Ny, Nz, ds, ds, ds))


def _plates(g, eps_r=1.0):
    """Two full-width plates normal to z: metal in cells k=2 and k=15.

    Node sets are therefore 2..3 and 15..16, leaving nodes 3..15 — twelve
    intervals — spanning the gap.
    """
    if eps_r != 1.0:
        ws.set_dielectric(g, eps_r)
    ws.set_box(g, 0, 12e-3, 0, 12e-3, 2e-3, 3e-3, 1.0, pec=True, name="bot")
    ws.set_box(g, 0, 12e-3, 0, 12e-3, 15e-3, 16e-3, 1.0, pec=True, name="top")
    return g


def _solve(g, potentials, boundary='neumann', method='direct', **kw):
    """Solve, defaulting to the *exact* linear solver.

    The assertions below are about the discretisation, not about the linear
    algebra, so they run on the factorisation — it has no tolerance, which lets
    them demand round-off agreement and mean it. Every test problem here is a
    few thousand unknowns, where that costs milliseconds. The production default
    (conjugate gradients) is covered by the two tests that compare the solvers
    against each other.
    """
    es = Electrostatics(g)
    for name, volts in potentials.items():
        es.set_potential(name, volts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return es.solve(boundary=boundary, method=method, **kw)


# ====================================================================== #
# Exactly representable solutions
# ====================================================================== #

def test_parallel_plate_potential_is_an_exact_linear_ramp():
    """φ is linear in the gap, so the FV stencil should be exact, not close."""
    g = _plates(_grid())
    phi = _solve(g, {"bot": 0.0, "top": 1.0}).phi

    expected = np.clip((np.arange(20) - 3) / 12.0, 0.0, 1.0)
    assert phi[6, 6, :] == pytest.approx(expected, abs=1e-12)


def test_parallel_plate_is_uniform_across_the_transverse_plane():
    """Nothing varies in x or y, so nothing in the answer may either."""
    g = _plates(_grid())
    phi = _solve(g, {"bot": 0.0, "top": 1.0}).phi
    for k in range(20):
        assert np.ptp(phi[:, :, k]) == pytest.approx(0.0, abs=1e-12)


def test_two_layer_dielectric_reproduces_the_series_capacitor():
    """D is continuous across the interface, so φ kinks by a known amount.

    This is the test that pins :func:`_face_eps` down: the interface sits
    exactly on a face, and averaging the permittivity across it instead of using
    the stored edge value would smear the kink and miss φ at the interface.
    """
    eps1, eps2 = 2.0, 8.0
    g = _grid()
    # Layer 1 on cells 3..8 -> nodes 3..9; layer 2 on cells 9..14 -> nodes 9..15.
    ws.set_box(g, 0, 12e-3, 0, 12e-3, 3e-3, 9e-3, eps1)
    ws.set_box(g, 0, 12e-3, 0, 12e-3, 9e-3, 15e-3, eps2)
    _plates(g)

    phi = _solve(g, {"bot": 0.0, "top": 1.0}).phi

    d1 = d2 = 6 * DS
    E1 = 1.0 / (d1 + eps1 * d2 / eps2)
    assert phi[6, 6, 9] == pytest.approx(E1 * d1, rel=1e-12)
    # Linear within each layer, with the two slopes in the ratio eps2:eps1.
    lo = np.diff(phi[6, 6, 3:10])
    hi = np.diff(phi[6, 6, 9:16])
    assert lo == pytest.approx(np.full(6, E1 * DS), rel=1e-12)
    assert hi == pytest.approx(np.full(6, E1 * DS * eps1 / eps2), rel=1e-12)


def test_conductors_come_out_exactly_equipotential():
    g = _grid()
    ws.set_box(g, 3e-3, 9e-3, 3e-3, 9e-3, 8e-3, 12e-3, 1.0, pec=True, name="blob")
    sol = _solve(g, {"blob": 7.0}, boundary='ground')
    assert np.all(sol.phi[sol.node_pec] == 7.0)


# ====================================================================== #
# Invariants — no analytic solution needed
# ====================================================================== #

def test_uniform_permittivity_scales_out_of_the_potential():
    """A constant ε multiplies the whole operator, so φ cannot depend on it.

    The 3D form of the homogeneous-fill probe that has caught two solver bugs
    already. Zero tolerance is the point: any dependence at all is a bug.
    """
    def phi_for(eps_r):
        g = _grid()
        if eps_r != 1.0:
            ws.set_dielectric(g, eps_r)
        ws.set_box(g, 3e-3, 9e-3, 3e-3, 9e-3, 8e-3, 12e-3, 1.0,
                   pec=True, name="c")
        return _solve(g, {"c": 3.0}, boundary='ground').phi

    assert phi_for(9.0) == pytest.approx(phi_for(1.0), abs=1e-12)


def test_permittivity_left_inside_the_metal_does_not_leak_out():
    """The one-sided ε rule, stated as an invariant rather than a formula.

    ε inside a conductor is not a material property — it is whatever the
    voxeliser left there. A filled model whose metal cells were stamped back to
    ε=1 must give the same φ as one where they were not, because no field lives
    in there to care.
    """
    def phi_for(stamp_metal):
        g = _grid()
        ws.set_dielectric(g, 6.0)
        ws.set_box(g, 3e-3, 9e-3, 3e-3, 9e-3, 8e-3, 12e-3, 1.0,
                   pec=True, name="c")
        if stamp_metal:
            for arr in (g.eps_x, g.eps_y, g.eps_z):
                arr[g.pec_mask] = 1.0
        return _solve(g, {"c": 3.0}, boundary='ground').phi

    assert phi_for(True) == pytest.approx(phi_for(False), abs=1e-12)


def test_a_neumann_face_reproduces_the_mirrored_full_solve():
    """A symmetry plane is a Neumann wall — so half the problem must agree.

    Independent of any closed form: it only asserts that the discrete operator
    respects the symmetry of the geometry it was built from.
    """
    # Symmetry is about *nodes*, not cells: 17 cells carry nodes 0..16, whose
    # mid-plane is node 8, and the conductor spans cells 6..9 = nodes 6..10,
    # centred on it. Both walls are then eight node-steps from the centre.
    full = _grid(Nx=17, Ny=8, Nz=8)
    ws.set_box(full, 6e-3, 10e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0,
               pec=True, name="c")
    phi_full = _solve(full, {"c": 1.0},
                      boundary={'xmin': 'ground', 'xmax': 'ground',
                                '*': 'neumann'}).phi
    assert phi_full == pytest.approx(phi_full[::-1], abs=1e-12)

    # The half domain stops on the mid-plane: 9 cells carry nodes 0..8, and half
    # the conductor is cells 6..7 = nodes 6..8.
    half = _grid(Nx=9, Ny=8, Nz=8)
    ws.set_box(half, 6e-3, 8e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0,
               pec=True, name="c")
    phi_half = _solve(half, {"c": 1.0},
                      boundary={'xmin': 'ground', '*': 'neumann'}).phi

    assert phi_half == pytest.approx(phi_full[:9], abs=1e-10)


def test_swapping_the_two_electrode_potentials_flips_the_field():
    """Linearity: φ(V1, V2) is affine, so 1−φ solves the swapped problem."""
    g = _plates(_grid())
    a = _solve(g, {"bot": 0.0, "top": 1.0}).phi
    b = _solve(g, {"bot": 1.0, "top": 0.0}).phi
    assert a + b == pytest.approx(np.ones_like(a), abs=1e-12)


# ====================================================================== #
# The two linear solvers must agree
# ====================================================================== #

def test_cg_matches_the_direct_factorisation():
    """Same operator, two solvers — the choice must not be visible in φ."""
    g = _grid()
    ws.set_box(g, 3e-3, 9e-3, 3e-3, 9e-3, 8e-3, 12e-3, 1.0, pec=True, name="c")
    ws.set_box(g, 0, 12e-3, 0, 12e-3, 18e-3, 19e-3, 1.0, pec=True, name="lid")

    direct = _solve(g, {"c": 1.0, "lid": 0.0}, method='direct')
    iterative = _solve(g, {"c": 1.0, "lid": 0.0}, method='cg', rtol=1e-12)

    assert direct.method == 'direct' and iterative.method == 'cg'
    assert iterative.iterations > 0
    assert iterative.phi == pytest.approx(direct.phi, abs=1e-9)


def test_auto_means_cg_at_every_size():
    """Measured: CG beats the factorisation 10x-400x in 3D, with no crossover.

    Size-dependent selection was rejected as well as unnecessary — it would mean
    refining a mesh silently changes the solver, and with it the last digits.
    """
    small = _plates(_grid())
    assert _solve(small, {"bot": 0.0, "top": 1.0}, method='auto').method == 'cg'

    tiny = _grid(6, 6, 6)
    ws.set_box(tiny, 2e-3, 4e-3, 2e-3, 4e-3, 2e-3, 4e-3, 1.0, pec=True, name="c")
    assert _solve(tiny, {"c": 1.0}, boundary='ground', method='auto').method == 'cg'


# ====================================================================== #
# Boundary conditions
# ====================================================================== #

def test_dirichlet_faces_hold_their_potential():
    g = _grid(8, 8, 8)
    ws.set_box(g, 3e-3, 5e-3, 3e-3, 5e-3, 3e-3, 5e-3, 1.0, pec=True, name="c")
    sol = _solve(g, {"c": 0.0}, boundary={'zmin': 2.0, '*': 'neumann'})
    assert np.all(sol.phi[:, :, 0] == 2.0)


def test_a_neumann_box_with_one_conductor_is_constant_everywhere():
    """No flux out, one potential in: there is nothing to make a gradient."""
    g = _grid(8, 8, 8)
    ws.set_box(g, 3e-3, 5e-3, 3e-3, 5e-3, 3e-3, 5e-3, 1.0, pec=True, name="c")
    sol = _solve(g, {"c": 2.5}, boundary='neumann')
    assert sol.phi == pytest.approx(np.full(sol.phi.shape, 2.5), abs=1e-12)


@pytest.mark.parametrize("spec,expect", [
    ('neumann', {f: 'neumann' for f in
                 ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')}),
    ('ground', {f: 0.0 for f in
                ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')}),
])
def test_scalar_boundary_specs_apply_to_all_six_faces(spec, expect):
    assert _resolve_boundary(spec) == expect


def test_star_supplies_the_default_face():
    bc = _resolve_boundary({'zmin': 'ground', 'zmax': 5.0, '*': 'neumann'})
    assert bc['zmin'] == 0.0 and bc['zmax'] == 5.0
    assert bc['xmin'] == bc['ymax'] == 'neumann'


def test_unspecified_faces_default_to_neumann():
    assert _resolve_boundary({'zmin': 'ground'})['xmax'] == 'neumann'


@pytest.mark.parametrize("bad,match", [
    ({'zup': 'ground'}, "unknown boundary face"),
    ({'zmin': 'dirichlet'}, "expected 'neumann'"),
    ({'zmin': None}, "expected 'neumann'"),
])
def test_bad_boundary_specs_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        _resolve_boundary(bad)


def test_boundary_must_be_a_recognised_type():
    with pytest.raises(TypeError, match="boundary must be"):
        _resolve_boundary(['ground'])


# ====================================================================== #
# Guard rails
# ====================================================================== #

def test_an_all_neumann_box_with_no_conductor_is_refused_as_singular():
    """φ is determined only up to a constant — better to say so than to guess."""
    g = _grid(8, 8, 8)
    with pytest.raises(ValueError, match="singular"):
        _solve(g, {}, boundary='neumann')


def test_shorted_parts_at_different_potentials_have_no_solution():
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0, pec=True, name="a")
    ws.set_box(g, 6e-3, 10e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0, pec=True, name="b")
    with pytest.raises(ValueError, match="same conductor"):
        _solve(g, {"a": 1.0, "b": 0.0})


def test_shorted_parts_at_the_same_potential_are_fine():
    """Naming each sub-piece of one electrode is legitimate, not an error."""
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0, pec=True, name="a")
    ws.set_box(g, 6e-3, 10e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0, pec=True, name="b")
    sol = _solve(g, {"a": 1.0, "b": 1.0}, boundary='ground')
    assert np.all(sol.phi[sol.node_pec] == 1.0)


def test_a_conductor_shorted_to_a_dirichlet_face_is_refused():
    g = _grid()
    ws.set_box(g, 0, 4e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0, pec=True, name="a")
    with pytest.raises(ValueError, match="short"):
        _solve(g, {"a": 5.0}, boundary={'xmin': 'ground', '*': 'neumann'})


def test_a_conductor_on_a_face_at_its_own_potential_is_fine():
    g = _grid()
    ws.set_box(g, 0, 4e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0, pec=True, name="a")
    sol = _solve(g, {"a": 0.0}, boundary={'xmin': 'ground', '*': 'neumann'})
    assert np.all(sol.phi == 0.0)


def test_unassigned_conductors_are_grounded_with_a_warning():
    g = _grid()
    ws.set_box(g, 3e-3, 6e-3, 3e-3, 6e-3, 3e-3, 6e-3, 1.0, pec=True, name="live")
    ws.set_box(g, 8e-3, 11e-3, 3e-3, 6e-3, 3e-3, 6e-3, 1.0, pec=True)  # unnamed

    es = Electrostatics(g).set_potential("live", 1.0)
    with pytest.warns(UserWarning, match="grounded at 0 V"):
        sol = es.solve(boundary='ground')
    assert sol.grounded_bodies == 1


def test_unnamed_metal_fused_to_a_part_takes_the_part_potential():
    """It is the same conductor, so grounding it would invent a short."""
    g = _grid()
    ws.set_box(g, 3e-3, 6e-3, 3e-3, 6e-3, 3e-3, 6e-3, 1.0, pec=True, name="live")
    ws.set_box(g, 6e-3, 9e-3, 3e-3, 6e-3, 3e-3, 6e-3, 1.0, pec=True)  # fused

    es = Electrostatics(g).set_potential("live", 4.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # no "grounded" warning may fire
        sol = es.solve(boundary='ground')
    assert sol.grounded_bodies == 0
    assert np.all(sol.phi[sol.node_pec] == 4.0)


def test_setting_a_potential_on_an_unknown_part_fails_immediately():
    g = _plates(_grid())
    with pytest.raises(KeyError, match="bot"):
        Electrostatics(g).set_potential("bott", 1.0)


def test_space_charge_is_explicitly_not_implemented():
    g = _plates(_grid())
    es = Electrostatics(g).set_potential("bot", 0.0).set_potential("top", 1.0)
    with pytest.raises(NotImplementedError, match="rho"):
        es.solve(rho=np.zeros((12, 12, 20)))


def test_unknown_method_is_refused():
    g = _plates(_grid())
    with pytest.raises(ValueError, match="method must be"):
        _solve(g, {"bot": 0.0, "top": 1.0}, method='multigrid')


# ====================================================================== #
# Housekeeping
# ====================================================================== #

def test_the_solve_does_not_touch_the_fdtd_state():
    """A boundary-value problem has no business perturbing a time-domain run."""
    g = _plates(_grid())
    g.Ez[:] = 1.234
    before = {name: getattr(g, name).copy() for name in
              ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')}
    dt, step = g.dt, g.time_step

    _solve(g, {"bot": 0.0, "top": 1.0})

    for name, arr in before.items():
        assert np.array_equal(getattr(g, name), arr)
    assert g.dt == dt and g.time_step == step


def test_potential_at_reads_the_nearest_node():
    g = _plates(_grid())
    sol = _solve(g, {"bot": 0.0, "top": 1.0})
    assert sol.potential_at(6e-3, 6e-3, 9e-3) == pytest.approx(0.5, abs=1e-12)
