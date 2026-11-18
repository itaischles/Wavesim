"""
test_magnetostatics.py — the 3D curl-curl solve (wavesim.magnetostatics).

Ordered as the electrostatic tests are, by how much a failure would say.
First the identities that hold to round-off whatever the geometry is:

* discrete Ampère and div B;
* the energy pair ½∫B·H = ½∫A·J;
* gauge independence;
* the staircase reduction;
* the electrostatic cross-check L′C′ = µε.

Then the analytic references, each with its error budget stated. Last, the
guard rails.

Two error sources recur in the references and are kept apart on purpose:

* **The source, not the solver.** A filament is carried on the edges the
  curve's dual-face crossings select, a staircase through the nearest nodes,
  and its enclosed area is 1–3% short of πR² at 6–12 cells of radius. So a
  filament is compared twice: tightly against Biot–Savart / Neumann of the
  *discrete* path it was given (which tests the solve), and loosely against
  the circle (which tests the whole chain).
* **The cross-section, not the solver.** A σ conductor is a cell mask. Its
  internal inductance converges to the continuum value only at first order
  (the current density jumps at the surface), which an independent 5-point
  finite-difference solve of the same 2D problem reproduces digit for digit
  (0.28098 vs 0.28100 at 6 cells across a square bar, 0.27820 at 96).
"""

import warnings

import numpy as np
import pytest
from scipy.special import ellipe, ellipk, gamma

import wavesim as ws
from wavesim.constants import EPS0, MU0
from wavesim.electrostatics import Electrostatics
from wavesim.magnetostatics import (
    Magnetostatics, InductanceMatrix, inductance_matrix, _filament_counts,
    _node_divergence, _resolve_boundary, _face_weights)
from wavesim.parts import name_pec_region

from conformal_shapes import binary_fractions
from magnetostatic_shapes import (A_IN, B_OUT, SPAN, coax_cross_section,
                                  coax_port, z_nodes)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _box(n=14, nz=None, ds=1e-3):
    nz = n if nz is None else nz
    return ws.set_vacuum(ws.create_grid(n, n, nz, ds, ds, ds))


def _circle(R, cx, cy, z, n=720):
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + R * np.cos(t), cy + R * np.sin(t), np.full(n, z)]


def _graded(h, core, outer, ratio=1.2):
    """Symmetric node coordinates: uniform ``h`` out to ``±core``, then
    geometric growth to ``±outer``."""
    pos = list(np.arange(0.0, core + h / 2, h))
    w = h
    while pos[-1] < outer:
        w *= ratio
        pos.append(pos[-1] + w)
    pos = np.array(pos)
    return np.concatenate([-pos[:0:-1], pos])


def _port_model(n, n_line=6, inner='pec', geometry='staircase', dz=3e-3,
                **kw):
    g, (ta, tb, ch), info = coax_port(
        n, z_nodes(n_line, dz, gap_dz=SPAN / n), inner=inner,
        geometry=geometry, **kw)
    return g, Magnetostatics(g).add_port('p', ta, tb, ch, 1.0), info


def _free(sol):
    return [~p for p in sol.pinned]


def _ampere_residual(sol):
    """max |∮H·dl − I| over the free edges, relative to the largest current."""
    cur, src = sol.edge_current, sol._source_total
    scale = max(np.abs(c).max() for c in src)
    return max(np.abs((cur[a] - src[a])[f]).max()
               for a, f in enumerate(_free(sol))) / scale


def _div_B(sol):
    """Net flux out of every cell whose six faces exist, relative to max |Φ|."""
    F = sol.flux
    d = ((F[0][1:, :-1, :-1] - F[0][:-1, :-1, :-1])
         + (F[1][:-1, 1:, :-1] - F[1][:-1, :-1, :-1])
         + (F[2][:-1, :-1, 1:] - F[2][:-1, :-1, :-1]))
    return np.abs(d).max() / max(np.abs(f).max() for f in F)


ANALYTIC_LP = MU0 / (2 * np.pi) * np.log(B_OUT / A_IN)     # coax L′


def _L_prime(n, geometry='staircase', inner='pec', **kw):
    """L′ from two line lengths: every end effect cancels in the difference."""
    L, ell = [], []
    for n_line in (6, 12):
        _, ms, info = _port_model(n, n_line, inner=inner, geometry=geometry,
                                  **kw)
        L.append(ms.solve(rtol=1e-12).flux_linkage('p'))
        ell.append(info['ell'])
    return (L[1] - L[0]) / (ell[1] - ell[0])


# ====================================================================== #
# 1. Discrete identities: Ampère, div B, energy, gauge
# ====================================================================== #

def test_filament_divergence_is_exactly_zero_in_integers():
    """Crossing counts of a closed curve: divergence zero at every node, as
    integers, including a curve threading cell corners and leaving the plane."""
    g = _box(10)
    t = np.linspace(0, 2 * np.pi, 97, endpoint=False)
    path = np.c_[4.5e-3 + 3e-3 * np.cos(t), 4.5e-3 + 3e-3 * np.sin(3 * t),
                 4.5e-3 + 2.5e-3 * np.sin(2 * t)]
    path[::7, :] = np.round(path[::7, :] * 2e3) / 2e3     # exact on planes
    counts = _filament_counts(g, path)
    assert any(c.any() for c in counts)
    assert np.array_equal(_node_divergence(counts), np.zeros((10, 10, 10)))


@pytest.mark.parametrize("case", ["filament", "pec_port", "sigma_port"])
def test_discrete_ampere_and_div_B(case):
    """∮H·dl around every free edge's dual face is the current through it,
    and the flux out of every cell is zero, both to round-off."""
    if case == "filament":
        g = _box(14)
        ms = Magnetostatics(g).add_loop(
            'L', _circle(4e-3, 6.5e-3, 6.5e-3, 6.3e-3), 2.0)
    else:
        g, ms, _ = _port_model(12, 6,
                               inner='pec' if case == 'pec_port' else 'sigma')
    sol = ms.solve(method='direct')
    assert _ampere_residual(sol) < 1e-11
    assert _div_B(sol) < 1e-13


def test_return_current_is_read_off_the_conductors():
    """Ampère around a whole conductor section: the inner surface carries the
    port's +1 A up, the outer surface carries it back. Neither was imposed."""
    g, ms, info = _port_model(12, 6)
    sol = ms.solve(method='direct')
    k = info['ke'] // 2
    c = 0.5 * SPAN
    r = np.hypot((g.x[:g.Nx] - c)[:, None], (g.y[:g.Ny] - c)[None, :])
    Iz = sol.surface_current()[2][:, :, k]
    assert Iz[r < 0.5 * (A_IN + B_OUT)].sum() == pytest.approx(1.0, abs=1e-10)
    assert Iz[r > 0.5 * (A_IN + B_OUT)].sum() == pytest.approx(-1.0, abs=1e-10)
    # Restricting to a named part returns the group holding it.
    assert sol.surface_current('inner')[2][:, :, k].sum() == pytest.approx(
        0.0, abs=1e-10)               # inner, walls and outer are one group


@pytest.mark.parametrize("case", ["filament", "pec_port", "sigma_port"])
def test_field_energy_equals_half_A_dot_J(case):
    """½∫B·H (the operator's quadratic form) = ½∫A·J (the source pairing)."""
    if case == "filament":
        ms = Magnetostatics(_box(12)).add_loop(
            'L', _circle(3e-3, 5.5e-3, 5.5e-3, 5e-3), 1.5)
    else:
        _, ms, _ = _port_model(12, 6,
                               inner='pec' if case == 'pec_port' else 'sigma')
    sol = ms.solve(method='direct')
    assert sol.energy == pytest.approx(sol.energy_AJ, rel=1e-12)


@pytest.mark.parametrize("case", ["filament", "pec_port", "sigma_port"])
def test_B_does_not_depend_on_the_gauge_weight(case):
    """Over five decades of gauge weight B moves by round-off only. It is
    exact, not asymptotic: the group-collapsed divergence rows match the
    gradient null space one for one."""
    if case == "filament":
        ms = Magnetostatics(_box(12)).add_loop(
            'L', _circle(3e-3, 5.5e-3, 5.5e-3, 5e-3), 1.0)
    else:
        _, ms, _ = _port_model(12, 6,
                               inner='pec' if case == 'pec_port' else 'sigma')
    ref = ms.solve(method='direct', gauge=1.0)
    scale = max(np.abs(b).max() for b in ref.B)
    for w in (1e-3, 1e2):
        other = ms.solve(method='direct', gauge=w)
        for a in range(3):
            assert np.abs(other.B[a] - ref.B[a]).max() < 1e-10 * scale
        assert other.energy == pytest.approx(ref.energy, rel=1e-10)


def test_cg_agrees_with_the_direct_solve():
    _, ms, _ = _port_model(12, 6)
    a = ms.solve(method='direct')
    b = ms.solve(method='cg', rtol=1e-12)
    assert b.iterations > 0
    assert b.flux_linkage('p') == pytest.approx(a.flux_linkage('p'), rel=1e-9)


def test_auto_means_cg():
    _, ms, _ = _port_model(12, 6)
    assert ms.solve().method == 'cg'


def test_progress_is_reported_and_ends_at_one():
    _, ms, _ = _port_model(12, 6)
    seen = []
    ms.solve(progress=seen.append)
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert all(b >= a for a, b in zip(seen, seen[1:]))


# ====================================================================== #
# Reduction: the conformal branch contains the staircase one
# ====================================================================== #

def test_open_fractions_of_one_reproduce_the_staircase_weights():
    """Bit for bit, on a graded mesh with varying µ: the fraction multiplies
    the face area and nothing else."""
    rng = np.random.default_rng(0)
    shape = (7, 5, 4)
    nodes = [np.concatenate([[0.0], np.cumsum(rng.uniform(1e-4, 3e-4, n))])
             for n in shape]
    g = ws.set_vacuum(ws.create_grid_rectilinear(*nodes))
    for ax in 'xyz':
        getattr(g, 'mu_' + ax)[...] = rng.uniform(1.0, 4.0, shape)
    stair = _face_weights(g)
    ones = {f'pec_{k}_open_{a}': np.ones(shape)
            for k in ('edge', 'face') for a in 'xyz'}
    ws.set_material_arrays(g, g.eps_x, g.eps_y, g.eps_z,
                           g.mu_x, g.mu_y, g.mu_z, **ones)
    for s, c in zip(stair, _face_weights(g)):
        assert np.array_equal(s, c)


def test_binary_fractions_reproduce_the_staircase_solve():
    """The same staircased conductor by both routes ⇒ identical answers."""
    g, ms, _ = _port_model(12, 6)
    a = ms.solve(method='direct')
    fr = binary_fractions(g.pec_mask)
    ws.set_material_arrays(g, g.eps_x, g.eps_y, g.eps_z,
                           g.mu_x, g.mu_y, g.mu_z, **fr)
    b = ms.solve(method='direct')
    assert all(np.array_equal(p, q) for p, q in zip(a.pinned, b.pinned))
    assert b.flux_linkage('p') == pytest.approx(a.flux_linkage('p'),
                                                rel=1e-13)
    assert b.energy == pytest.approx(a.energy, rel=1e-13)


# ====================================================================== #
# 2. Circular filament: on-axis B
# 3. Two coaxial filaments: mutual inductance
# ====================================================================== #

def _biot_savart(g, counts, P):
    """B at ``P`` from the straight edge segments a filament was put on."""
    X = (g.x, g.y, g.z)
    B = np.zeros(3)
    for a in range(3):
        for n in np.argwhere(counts[a] != 0):
            n1 = n.copy()
            n1[a] += 1
            p0 = np.array([X[i][n[i]] for i in range(3)])
            p1 = np.array([X[i][n1[i]] for i in range(3)])
            r0, r1, dl = P - p0, P - p1, p1 - p0
            c = np.cross(dl, r0)
            f = (dl @ r0 / np.linalg.norm(r0)
                 - dl @ r1 / np.linalg.norm(r1)) / (c @ c)
            B += MU0 * counts[a][tuple(n)] / (4 * np.pi) * f * c
    return B


def _neumann_segments(g, c1, c2):
    """Neumann double integral over the discrete paths (8-point Gauss)."""
    xg, wg = np.polynomial.legendre.leggauss(8)
    xg, wg = (xg + 1) / 2, wg / 2
    X = (g.x, g.y, g.z)

    def segs(c):
        out = []
        for a in range(3):
            for n in np.argwhere(c[a] != 0):
                n1 = n.copy()
                n1[a] += 1
                out.append((a, np.array([X[i][n[i]] for i in range(3)]),
                            np.array([X[i][n1[i]] for i in range(3)]),
                            c[a][tuple(n)]))
        return out

    M = 0.0
    for a, p0, p1, s in segs(c1):
        P = p0 + np.outer(xg, p1 - p0)
        for b, q0, q1, t in segs(c2):
            if a != b:
                continue
            Q = q0 + np.outer(xg, q1 - q0)
            r = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
            M += (s * t * np.linalg.norm(p1 - p0) * np.linalg.norm(q1 - q0)
                  * np.sum(wg[:, None] * wg[None, :] / r))
    return MU0 / (4 * np.pi) * M


@pytest.fixture(scope='module')
def two_loops():
    """Loops of radius 6 and 4 mm, 4 mm apart, on a 1 mm core graded out to
    flux walls at ten radii. Axis through cell centres, so Bz on the axis is
    read on Hz faces without interpolation."""
    R1, R2, d = 6e-3, 4e-3, 4e-3
    xy = (_graded(1.0, 9.5, 60.0) + 0.5) * 1e-3
    zz = _graded(1.0, 9.0, 60.0) * 1e-3
    g = ws.set_vacuum(ws.create_grid_rectilinear(xy, xy, zz))
    i0 = int(np.argmin(np.abs(g.xc)))
    k0 = int(np.argmin(np.abs(g.z + d / 2)))
    k1 = int(np.argmin(np.abs(g.z - d / 2)))
    pa = _circle(R1, g.xc[i0], g.yc[i0], g.z[k0])
    pb = _circle(R2, g.xc[i0], g.yc[i0], g.z[k1])
    ms = Magnetostatics(g).add_loop('a', pa, 1.0).add_loop('b', pb, 1.0)
    L = inductance_matrix(ms, rtol=1e-11)
    sol_a = Magnetostatics(g).add_loop('a', pa, 1.0).solve(rtol=1e-11)
    return dict(g=g, i0=i0, k0=k0, k1=k1, R1=R1, R2=R2, pa=pa, pb=pb, L=L,
                sol_a=sol_a)


@pytest.mark.slow
def test_circular_filament_on_axis_field(two_loops):
    """Bz on the axis of a loop, at the centre and out to 1.5 R.

    Against Biot–Savart of the edges the filament was actually put on, the
    solve is within 1.5% (it converges with h: 1.1% → 0.16% at the centre on
    halving it). Against the circle ``µIR²/(2(R²+z²)^{3/2})`` add the
    staircase's own 1–2% (the lattice path's area deficit), so 4%."""
    t = two_loops
    g, sol, i0, k0, R = t['g'], t['sol_a'], t['i0'], t['k0'], t['R1']
    counts = _filament_counts(g, t['pa'])
    for zr in (0.0, 0.5, 1.0, 1.5):
        k = k0 + int(np.argmin(np.abs(g.z[k0:] - g.z[k0] - zr * R)))
        z = g.z[k] - g.z[k0]
        Bz = sol.B[2][i0, i0, k]
        circle = MU0 * R ** 2 / (2 * (R ** 2 + z ** 2) ** 1.5)
        discrete = _biot_savart(g, counts, np.array([g.xc[i0], g.yc[i0],
                                                     g.z[k]]))[2]
        assert Bz == pytest.approx(discrete, rel=0.015)
        assert Bz == pytest.approx(circle, rel=0.04)


@pytest.mark.slow
def test_mutual_inductance_of_coaxial_loops(two_loops):
    """M against the Neumann formula.

    The solve matches the Neumann double integral over the two *discrete*
    paths to 0.5% (measured −0.28%). The circles' closed form
    ``µ√(R₁R₂)[(2/k − k)K − (2/k)E]`` is further off by the staircase source
    error, which at 6 and 4 cells of radius is −8.4% by itself (measured,
    independent of the solver), so it gets 10%.
    """
    t = two_loops
    g, L = t['g'], t['L']
    d = g.z[t['k1']] - g.z[t['k0']]
    R1, R2 = t['R1'], t['R2']
    m = 4 * R1 * R2 / ((R1 + R2) ** 2 + d ** 2)
    k = np.sqrt(m)
    M_circle = MU0 * np.sqrt(R1 * R2) * ((2 / k - k) * ellipk(m)
                                         - (2 / k) * ellipe(m))
    M_paths = _neumann_segments(g, _filament_counts(g, t['pa']),
                                _filament_counts(g, t['pb']))
    assert L.between('a', 'b') == pytest.approx(M_paths, rel=0.005)
    assert L.between('a', 'b') == pytest.approx(M_circle, rel=0.10)
    assert L.between('a', 'b') == pytest.approx(L.between('b', 'a'),
                                                rel=1e-8)


# ====================================================================== #
# 4. Shorted PEC coax with a port; the cross-solver invariant
# ====================================================================== #

def test_conformal_coax_inductance_matches_the_analytic_line():
    """L′ = µ/(2π)·ln(b/a) to 0.3% at 24 cells across (measured +0.29%), and
    the whole L to µ(ℓ+g)/(2π)·ln(b/a) plus the channel's own small internal
    inductance (measured +1.5%)."""
    assert _L_prime(24, 'conformal') == pytest.approx(ANALYTIC_LP, rel=0.004)
    _, ms, info = _port_model(24, 12, geometry='conformal')
    L = ms.solve(rtol=1e-12).flux_linkage('p')
    assert L == pytest.approx(ANALYTIC_LP * (info['ell'] + info['gap']),
                              rel=0.02)


@pytest.mark.parametrize("geometry", ["conformal", "staircase"])
def test_L_prime_times_C_prime_is_mu_eps(geometry):
    """The cross-solver invariant: this module's L′ × electrostatics' C′ on
    the same cross-section = µ₀ε₀, to round-off.

    It is exact, not approximate, and it holds on the staircase too, because
    in a z-invariant section the two problems are the same 2D operator. φ on
    nodes with ε on edges, and A_z on z-edges with ν on the faces those
    edges' differences cross, share one set of weights, cut fractions
    included. So the agreement says the two modules discretise one geometry,
    not that either is accurate (the staircase L′ is 13% off on its own). A
    staircase built a different way (``set_coax`` voxelises by node
    coordinates) breaks it at the percent level, which is why
    :func:`coax_cross_section` exists.
    """
    n = 24
    Lp = _L_prime(n, geometry)
    es = Electrostatics(coax_cross_section(n, geometry))
    es.set_potential('inner', 1.0).set_potential('outer', 0.0)
    sol = es.solve(boundary='neumann', method='direct')
    Cp = sol.charge('inner') / (3 * SPAN / n)
    assert Lp * Cp / (MU0 * EPS0) == pytest.approx(1.0, abs=1e-9)


# ====================================================================== #
# 5. σ conductors: resistance and internal inductance
# ====================================================================== #

def test_rectangular_sigma_bar_resistance_is_exact():
    """R = ℓ/(σA) for a square bar between a flux-wall contact and a port
    terminal: the flow is uniform and the quarter-area weights give the bar
    exactly its own cross-section."""
    sigma = 3.0e6
    _, ms, info = _port_model(24, 6, inner='sigma', square=True, sigma=sigma)
    sol = ms.solve(method='direct')
    assert sol.resistance('p') == pytest.approx(
        info['ell'] / (sigma * (2 * A_IN) ** 2), rel=1e-12)


def test_pec_port_has_no_resistance():
    _, ms, _ = _port_model(12, 6)
    assert ms.solve(method='direct').resistance('p') is None


def test_internal_inductance_of_a_square_bar():
    """L_σ − L_PEC per length against ``µ/(2π)·ln(R_cap/GMD)``, exact for a
    square (``R_cap = Γ(¼)²s/(4π^{3/2})``, ``GMD = 0.44705 s``).

    Measured +1.6% at 6 cells across the bar, which an independent 5-point
    finite-difference solve of the same cross-section reproduces (0.28098
    against this module's 0.28100). It converges at first order, reaching
    0.27820 at 96 cells. So the 2% bound is the mesh, not a modelling
    error."""
    exact = MU0 / (2 * np.pi) * np.log(gamma(0.25) ** 2 / (4 * np.pi ** 1.5)
                                       / 0.447049)
    internal = (_L_prime(24, inner='sigma', square=True)
                - _L_prime(24, inner='pec', square=True))
    assert internal == pytest.approx(exact, rel=0.02)


def test_internal_inductance_of_the_round_coax():
    """L_σ − L_PEC on the round coax, against ``µℓ/(8π)``.

    The σ rod is a cell mask, a pixel disk of radius 3 cells, and only a
    round wire has internal inductance exactly µ/(8π). A pixel disk's
    differs, the way a square bar's is 1.11× it. Measured 1.138× here and
    1.129× at twice the resolution, so 15%. The exact identities this rests
    on (Ampère, energy, R) are tested above. :func:`test_internal_inductance_of_a_square_bar`
    is the quantitative check."""
    internal = _L_prime(24, inner='sigma') - _L_prime(24, inner='pec')
    assert internal == pytest.approx(MU0 / (8 * np.pi), rel=0.15)
    assert internal > MU0 / (8 * np.pi)


# ====================================================================== #
# 6. µ_r filling
# ====================================================================== #

@pytest.mark.parametrize("model", ["port", "filament"])
def test_uniform_mu_r_scales_inductance_exactly(model):
    mu_r = 3.7
    if model == "port":
        L = []
        for m in (1.0, mu_r):
            _, ms, _ = _port_model(12, 6, mu_r=m)
            L.append(ms.solve(method='direct').flux_linkage('p'))
    else:
        L = []
        for m in (1.0, mu_r):
            g = _box(12)
            for arr in (g.mu_x, g.mu_y, g.mu_z):
                arr[...] = m
            ms = Magnetostatics(g).add_loop(
                'L', _circle(3e-3, 5.5e-3, 5.5e-3, 5e-3), 1.0)
            L.append(ms.solve(method='direct').flux_linkage('L'))
    assert L[1] / L[0] == pytest.approx(mu_r, rel=1e-12)


# ====================================================================== #
# 7. Symmetry (and flux-wall) planes reproduce the mirrored full model
# ====================================================================== #

def _mirror_grids(n=12, nz_half=8, ds=1e-3):
    """A half grid with its zmin node plane at z = 0, and its full mirror."""
    xy = np.arange(n + 1) * ds
    half_nodes = np.arange(nz_half) * ds                  # carried, from 0
    half = np.concatenate([half_nodes, [half_nodes[-1] + ds]])
    full = np.concatenate([-half_nodes[:0:-1], half_nodes,
                           [half_nodes[-1] + ds]])
    return (ws.set_vacuum(ws.create_grid_rectilinear(xy, xy, half)),
            ws.set_vacuum(ws.create_grid_rectilinear(xy, xy, full)))


@pytest.mark.parametrize("wall, sign", [("symmetry", 1.0),
                                        ("flux_wall", -1.0)])
def test_a_mirror_plane_reproduces_the_full_model(wall, sign):
    """Two coaxial loops at ±d. Co-directed, the midplane carries no
    tangential H (a symmetry wall). Counter-directed, no normal B (a flux
    wall). The half model's self-inductance equals ``L₁₁ ± M`` of the full
    one."""
    half, full = _mirror_grids()
    c, R, d = 5.5e-3, 3.5e-3, 3e-3
    up = _circle(R, c, c, d)
    down = _circle(R, c, c, -d)
    L = inductance_matrix(
        Magnetostatics(full).add_loop('up', up, 1.0).add_loop('down', down,
                                                              1.0),
        method='direct')
    sol = Magnetostatics(half).add_loop('up', up, 1.0).solve(
        boundary={'*': 'flux_wall', 'zmin': wall}, method='direct')
    assert sol.flux_linkage('up') == pytest.approx(
        L.self_('up') + sign * L.between('up', 'down'), rel=1e-11)


# ====================================================================== #
# 8. Inductance matrix
# ====================================================================== #

def test_inductance_matrix_is_symmetric_and_carries_the_energy():
    """A port and a filament loop threading the coax's annulus. L is
    symmetric to round-off, and ½IᵀLI is the energy of a solve with both
    driven."""
    g, ms, info = _port_model(12, 6)
    c = 0.5 * SPAN
    x0, x1 = c + 4.2e-3, c + 6.2e-3
    z0, z1 = 3.5e-3, 12.5e-3
    rect = np.array([[x0, c, z0], [x1, c, z0], [x1, c, z1], [x0, c, z1]])
    ms.add_loop('loop', rect, 1.0)
    L = inductance_matrix(ms, method='direct')
    assert isinstance(L, InductanceMatrix)
    assert L.names == ('loop', 'p')
    assert L.between('p', 'loop') == pytest.approx(L.between('loop', 'p'),
                                                   rel=1e-11)
    assert abs(L.coupling('p', 'loop')) > 0.05
    I = {'p': 2.0, 'loop': -0.5}
    ms.ports['p'].current, ms.loops['loop'].current = 2.0, -0.5
    sol = ms.solve(method='direct')
    assert L.energy(I) == pytest.approx(sol.energy, rel=1e-11)
    assert np.all(np.linalg.eigvalsh(0.5 * (L.matrix + L.matrix.T)) > 0)


def test_inductance_matrix_refuses_a_bare_grid():
    with pytest.raises(TypeError, match="needs the sources"):
        inductance_matrix(_box(6))


# ====================================================================== #
# 9. Guard rails
# ====================================================================== #

def _two_blocks(gap=1):
    """Two separate PEC blocks ``gap`` cells apart, and a port across them."""
    g = _box(12)
    m1 = np.zeros((12, 12, 12), bool)
    m2 = np.zeros((12, 12, 12), bool)
    top = 5 + gap
    m1[3:9, 3:9, 2:5] = True
    m2[3:9, 3:9, top:top + 3] = True
    name_pec_region(g, m1, 'lower')
    name_pec_region(g, m2, 'upper')
    ta = np.zeros_like(m1)
    tb = np.zeros_like(m1)
    ta[3:10, 3:10, 5] = True
    tb[3:10, 3:10, top] = True
    ch = np.zeros_like(m1)
    ch[3:9, 3:9, 5:top] = True
    return g, ta, tb, ch


def test_port_across_two_disconnected_bodies_is_refused_by_name():
    g, ta, tb, ch = _two_blocks()
    ms = Magnetostatics(g).add_port('gap', ta, tb, ch, 1.0)
    with pytest.raises(ValueError, match="no return path") as err:
        ms.solve(boundary='symmetry')
    assert "'lower'" in str(err.value) and "'upper'" in str(err.value)


def test_the_flux_wall_counts_as_one_body_for_the_return_path():
    """The same blocks touching the box's flux walls: the walls close the
    loop, so the port is accepted."""
    g, ta, tb, ch = _two_blocks()
    g.pec_mask[3:9, 3:9, 0:2] = True                     # lower → zmin wall
    g.pec_mask[3:9, 3:9, 9:12] = True                    # upper → zmax wall
    sol = Magnetostatics(g).add_port('gap', ta, tb, ch, 1.0).solve(
        method='direct')
    assert sol.flux_linkage('gap') > 0.0


def test_filament_through_pec_is_refused():
    g = _box(12)
    m = np.zeros((12, 12, 12), bool)
    m[5:7, 5:7, :] = True
    name_pec_region(g, m, 'post')
    path = np.array([[2e-3, 2e-3, 6e-3], [9e-3, 9e-3, 6e-3],
                     [2e-3, 9e-3, 6e-3]])
    with pytest.raises(ValueError, match="passes through PEC"):
        Magnetostatics(g).add_loop('L', path, 1.0).solve()


def test_filament_along_a_flux_wall_is_refused():
    g = _box(8)
    path = np.array([[0, 1e-3, 1e-3], [0, 6e-3, 1e-3], [0, 6e-3, 6e-3]])
    with pytest.raises(ValueError, match="flux wall"):
        Magnetostatics(g).add_loop('L', path, 1.0).solve()
    # The same path is fine against a symmetry wall.
    Magnetostatics(g).add_loop('L', path, 1.0).solve(
        boundary={'*': 'flux_wall', 'xmin': 'symmetry'}, method='direct')


def test_a_lossy_dielectric_is_not_a_return_path():
    """A σ = 0.5 S/m slab bridging the two blocks is a lossy dielectric, below
    the conductor threshold, so the port still has no return path. Passed
    explicitly as ``conductor=`` it becomes one, with a finite R."""
    g, ta, tb, ch = _two_blocks()
    ws.set_box(g, 0, 3e-3, 0, 12e-3, 2e-3, 9e-3, 4.0, sigma=0.5)
    with pytest.raises(ValueError, match="no return path"):
        Magnetostatics(g).add_port('gap', ta, tb, ch, 1.0).solve(
            boundary='symmetry')
    slab = np.zeros((12, 12, 12), bool)
    slab[0:3, :, 2:9] = True
    sol = Magnetostatics(g).add_port('gap', ta, tb, ch, 1.0,
                                     conductor=slab).solve(
        boundary='symmetry', method='direct')
    assert sol.resistance('gap') > 0.0
    assert _ampere_residual(sol) < 1e-11


def test_a_channel_touching_a_third_conductor_is_refused():
    """A three-cell gap with a stray PEC cell beside the channel, touching
    the channel's surface but neither terminal."""
    g, ta, tb, ch = _two_blocks(gap=3)
    m = np.zeros((12, 12, 12), bool)
    m[9, 5, 6] = True
    name_pec_region(g, m, 'stray')
    ms = Magnetostatics(g).add_port('gap', ta, tb, ch, 1.0)
    with pytest.raises(ValueError, match="'stray'"):
        ms.solve(boundary='symmetry')


def test_input_validation():
    g = _box(6)
    ms = Magnetostatics(g)
    with pytest.raises(ValueError, match="no sources"):
        ms.solve()
    with pytest.raises(ValueError, match="expected shape"):
        ms.add_port('p', np.ones((5, 5, 5)), np.ones((6, 6, 6)),
                    np.ones((6, 6, 6)), 1.0)
    t = np.zeros((6, 6, 6), bool)
    t[1, 1, 1] = True
    with pytest.raises(ValueError, match="share"):
        ms.add_port('p', t, t, t, 1.0)
    ms.add_loop('L', _circle(1e-3, 2.5e-3, 2.5e-3, 2.5e-3), 1.0)
    with pytest.raises(ValueError, match="already exists"):
        ms.add_loop('L', _circle(1e-3, 2.5e-3, 2.5e-3, 2.5e-3), 1.0)
    with pytest.raises(ValueError, match="leaves the domain"):
        Magnetostatics(g).add_loop(
            'far', _circle(9e-3, 2.5e-3, 2.5e-3, 2.5e-3), 1.0).solve()
    with pytest.raises(ValueError, match="gauge"):
        ms.solve(gauge=0.0)


def test_boundary_spec_resolution():
    assert set(_resolve_boundary('symmetry').values()) == {'symmetry'}
    bc = _resolve_boundary({'zmin': 'symmetry'})
    assert bc['zmin'] == 'symmetry' and bc['xmax'] == 'flux_wall'
    with pytest.raises(ValueError, match="no open boundary"):
        _resolve_boundary('open')
    with pytest.raises(ValueError, match="unknown boundary face"):
        _resolve_boundary({'top': 'symmetry'})


# ====================================================================== #
# 10. Conformal against staircase
# ====================================================================== #

@pytest.mark.slow
def test_conformal_converges_faster_than_the_staircase():
    """L′ error against µ/(2π)·ln(b/a) (measured):

        cells across    staircase    conformal
            24           −13.70%      +0.292%
            48            −4.62%      +0.071%

    The conformal error falls 4.1× per halving, second order. The staircase
    is erratic and an order of magnitude or more worse. These are the
    electrostatic C′ errors reflected through L′C′ = µε."""
    err = {(geo, n): abs(_L_prime(n, geo) / ANALYTIC_LP - 1)
           for geo in ('conformal', 'staircase') for n in (24, 48)}
    for n in (24, 48):
        assert err[('conformal', n)] < err[('staircase', n)] / 10
    assert err[('conformal', 24)] / err[('conformal', 48)] > 3.0
    assert err[('conformal', 48)] < 1e-3
