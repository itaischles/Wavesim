"""S5 — the conformal (Dey–Mittra) TEM mode solver.

The FDTD steps the cut geometry after S2–S4, but the port is solved by
``mode_solver.py``, and until it sees the *same* cut geometry the Z₀ the port
presents is not the Z₀ the run presents. This is the step that closes that gap,
and it is the one that pays: on the plan's reference coax the modal Z₀ error
goes from **+14.4% to −0.8%** at the coarsest resolution.

The whole conformal branch follows from one identity. A TEM mode is the
transverse field whose longitudinal H stays zero, i.e. whose *conformal*
Faraday contour integral vanishes:

    (E_b·L_b)[i+1,j] − (E_b·L_b)[i,j] − (E_a·L_a)[i,j+1] + (E_a·L_a)[i,j] = 0

which is solved identically by ``E_a·L_a = φ[i,j] − φ[i+1,j]``. So the field is
the gradient of a node potential divided by the **open** edge length: the open
fraction lands on the stencil's centre distance, not on its face length (the
plan §S5 guessed the latter). Everything else — the energy integral's open
areas, the modal conductance, the launched ê — is that one substitution carried
through consistently.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.mode_solver import (_conformal_node_pec, _face_coefs,
                                 solve_tem_modes)
from wavesim.parts import pec_node_mask

from conformal_shapes import coax_fractions, binary_fractions as _binary_fractions

# The plan §7 reference coax: air, a = 3 mm, b = 9 mm, dz = 1 mm.
A_IN, B_OUT, DZ = 3.0e-3, 9.0e-3, 1.0e-3
Z0_ANALYTIC = 59.9585 * np.log(B_OUT / A_IN)          # 65.871 Ω


def _coax(cell, geometry='conformal', eps_r=1.0, centre_shift=0.0):
    """Reference coax at transverse cell size ``cell``.

    ``geometry`` selects how the conductor reaches the solver:

    * ``'staircase'`` — the cell-centred ``pec_mask`` alone (the legacy path);
    * ``'binary'``    — the *same* staircased conductor expressed as 0/1 open
      fractions, so it runs the conformal code path over uncut geometry. This
      is what separates the new energy integral from the cut cells themselves;
    * ``'conformal'`` — the true analytic cut fractions.

    ``centre_shift`` moves the coax centre by that many cells. The default
    build puts it on node ``n/2`` while the PEC walls (the grounded ring at
    nodes 0 and n−1) are symmetric about node ``(n−1)/2`` — half a cell apart.
    Shifting by −0.5 lines the two up, which is what the symmetry test needs.
    """
    n = int(round(2 * B_OUT / cell))
    grid = ws.create_grid(Nx=n, Ny=n, Nz=3, dx=cell, dy=cell, dz=DZ)
    ws.set_vacuum(grid)
    for axis in 'xyz':
        getattr(grid, 'eps_' + axis)[...] = eps_r
    c = (0.5 * n + centre_shift) * cell
    r2 = ((grid.xc[:, None, None] - c) ** 2 + (grid.yc[None, :, None] - c) ** 2)
    grid.pec_mask = np.broadcast_to(
        (r2 < A_IN ** 2) | (r2 > B_OUT ** 2), (n, n, 3)).copy()

    if geometry == 'staircase':
        return grid
    if geometry == 'conformal':
        fr = coax_fractions(grid, c, c, A_IN, B_OUT)
    else:
        fr = _binary_fractions(grid.pec_mask)
    ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                           grid.mu_x, grid.mu_y, grid.mu_z, **fr)
    return grid


def _mode(grid, **kw):
    return solve_tem_modes(grid, normal='z', position=DZ,
                           compute_params=True, **kw)[0]


def _z0_err(cell, geometry):
    return abs(_mode(_coax(cell, geometry)).impedance / Z0_ANALYTIC - 1.0)


# ---------------------------------------------------------------------- #
# Reduction: the conformal branch must contain the legacy one
# ---------------------------------------------------------------------- #

def test_node_mask_round_trips_the_edge_fractions():
    """``_conformal_node_pec`` inverts ``_binary_fractions`` exactly.

    Both sides are the conductor the *run* steps: the end points of every edge
    the staircase rule zeroes, which is what :func:`~wavesim.parts.pec_node_mask`
    returns and what the staircase mode solve pins.
    """
    grid = _coax(1.5e-3, 'staircase')
    fr = _binary_fractions(grid.pec_mask)
    node = _conformal_node_pec(fr['pec_edge_open_x'][:, :, 1],
                               fr['pec_edge_open_y'][:, :, 1])
    assert np.array_equal(node, pec_node_mask(grid)[:, :, 1])


def test_all_open_fractions_mean_no_conductor():
    """Fractions of 1.0 everywhere describe empty space — the conformal mask
    ignores ``pec_mask`` entirely, so there is no conductor and no TEM mode."""
    grid = _coax(1.5e-3, 'staircase')
    ones = {k: np.ones(grid.pec_mask.shape) for k in
            ('pec_edge_open_x', 'pec_edge_open_y', 'pec_edge_open_z',
             'pec_face_open_x', 'pec_face_open_y', 'pec_face_open_z')}
    ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                           grid.mu_x, grid.mu_y, grid.mu_z, **ones)
    with pytest.warns(UserWarning, match="at least two conductors"):
        assert solve_tem_modes(grid, normal='z', position=DZ) == []


def test_open_fractions_reproduce_the_legacy_face_coefficients():
    """With nothing cut, every stencil coefficient is the legacy one bit-for-bit
    — on a *non-uniform* transverse mesh too, because the open fraction scales
    the existing centre distance rather than replacing it."""
    rng = np.random.default_rng(0)
    Na, Nb = 7, 5
    eps_a, eps_b = rng.uniform(1, 4, (Na, Nb)), rng.uniform(1, 4, (Na, Nb))
    da_w, db_w = rng.uniform(1e-4, 3e-4, Na), rng.uniform(1e-4, 3e-4, Nb)
    pec = rng.random((Na, Nb)) < 0.2
    legacy = _face_coefs(eps_a, eps_b, da_w, db_w, pec)
    opened = _face_coefs(eps_a, eps_b, da_w, db_w, pec,
                         np.ones((Na, Nb)), np.ones((Na, Nb)))
    for lo, op in zip(legacy, opened):
        assert np.array_equal(lo, op)


def test_binary_fractions_reproduce_the_staircase_potential():
    """Same conductor, two routes into the solver ⇒ the identical φ.

    Z₀ is *not* asserted equal: the conformal branch reports the capacitance as
    the quadratic form of the operator it solved (the open-area energy), while
    the legacy branch keeps its collocated ``np.gradient`` integral so every
    recorded staircase result stays bit-identical. The two are different
    discretisations of the same C, and the operator is where the reduction
    property has to hold.
    """
    staircase = _mode(_coax(1.5e-3, 'staircase'))
    binary = _mode(_coax(1.5e-3, 'binary'))
    assert np.array_equal(staircase.pec, binary.pec)
    assert np.abs(staircase.phi - binary.phi).max() == 0.0
    assert binary.eps_eff == pytest.approx(staircase.eps_eff, rel=1e-12)


# ---------------------------------------------------------------------- #
# V1 — the homogeneous-fill invariant still holds through the new operator
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("cell", [0.5e-3, 0.375e-3, 0.25e-3])
def test_eps_eff_is_exact_under_conformal_pec(cell):
    """ε_eff = ε_r to round-off at every resolution, cut cells and all.

    It survives because the filled and air operators stay exact scalar
    multiples: the conformal weighting multiplies both by the same open
    fraction, and the one-sided PEC ε rule keeps conductor-adjacent faces from
    averaging in the meaningless ε inside the metal. φ = φ_air follows, then
    C = ε_r·C_air face by face — which is only true because the capacitance is
    read off the very operator that was solved.
    """
    mode = _mode(_coax(cell, 'conformal', eps_r=2.3))
    assert mode.eps_eff == pytest.approx(2.3, rel=1e-12)


# ---------------------------------------------------------------------- #
# V3 — the payoff
# ---------------------------------------------------------------------- #

def test_conformal_impedance_beats_the_staircase_target():
    """Z₀ error < 2% at the coarsest resolution — the plan's V3 gate.

    Measured against the analytic 65.871 Ω (error, |Z₀/Z_analytic − 1|):

        cell (mm)   staircase = binary   conformal
        0.5000            -5.24%           -0.79%
        0.3750            -5.79%           -0.50%
        0.2500            -2.59%           -0.27%
        0.1875            -2.57%           -0.17%

    ``binary`` — the staircase conductor pushed through the conformal code path
    — is *identical* to ``staircase``, because S5d gave both the same
    finite-volume energy integral and both now read the conductor off the same
    edge rule. Before S5d, staircase read +14.37% here and the two columns
    differed; that gap was the old collocated ``np.gradient`` capacitance, not
    the geometry.

    The staircase column changed sign when the node-mask bug was fixed
    (``docs/mode_solver_staircase_node_mask.md``): its conductor used to be a
    cell short on every high side, which widened the coax gap and read Z₀ high
    (+6.86% at 0.5 mm). The dilated conductor the run actually steps is the
    slightly *fat* one, so the error is now negative — and it is the error the
    FDTD line itself has, which is the point: mid-line V/I on the reference coax
    measures 61.53 Ω against the mode's 62.42 Ω at 0.5 mm (+1.4%), where the old
    mask claimed 70.39 Ω (+13.4%).

    What is left is the geometry alone, and it is worth reading the whole column
    rather than the endpoints: only the cut cells make the sequence monotone.
    Staircase wobbles (5.79 after 5.24) because refining a staircase
    re-rasterises the conductor rather than resolving it.
    """
    assert _z0_err(0.5e-3, 'conformal') < 0.02
    assert _z0_err(0.5e-3, 'staircase') > 0.05


def test_conformal_impedance_converges_faster_than_first_order():
    """The staircase error is O(h); the conformal error is not.

    Measured order over a 2× refinement is ≈1.55 (0.79% → 0.27%) against ≈1.02
    for the staircase (5.24% → 2.59%). It is not the O(h²) the plan hoped for,
    and the residue is the mode solver's remaining half-cell bookkeeping — φ
    lives on nodes while the legacy stencil pairs primary face lengths with
    dual centre distances, invisible on a uniform mesh but not free of error.
    Asserted loosely: what matters is that it is decisively past first order.
    """
    coarse, fine = _z0_err(0.5e-3, 'conformal'), _z0_err(0.25e-3, 'conformal')
    order = np.log(coarse / fine) / np.log(2.0)
    assert order > 1.3, f"conformal Z0 convergence order {order:.2f}"

    s_order = np.log(_z0_err(0.5e-3, 'staircase')
                     / _z0_err(0.25e-3, 'staircase')) / np.log(2.0)
    assert s_order < 1.1, f"staircase order {s_order:.2f} — baseline moved"


# ---------------------------------------------------------------------- #
# The launched profile
# ---------------------------------------------------------------------- #

def _mirror_asymmetry(grid):
    """Worst mirror-pair discrepancy in the launched ê, relative to its peak.

    φ lives on nodes 0..n−1 (the PEC walls are the grounded ring), so the
    symmetry centre is node (n−1)/2. ``Ex[i,j]`` spans nodes i, i+1 at node j,
    so it mirrors onto edge n−2−i at node n−1−j; ``Ey`` is the transpose.
    """
    E, _H = _mode(grid)._staggered_port_fields(grid)
    worst = 0.0
    for comp, arr in E.items():
        a = np.abs(arr)
        core, mir = ((a[:-1, :], a[-2::-1, ::-1]) if comp == 'Ex'
                     else (a[:, :-1], a[::-1, -2::-1]))
        worst = max(worst, np.abs(core - mir).max() / core.max())
    return worst


def test_conformal_launch_profile_is_exactly_symmetric():
    """A mirror-symmetric coax must launch a mirror-symmetric ê. Conformally it
    does, to round-off; the staircase path is 100% asymmetric on the same grid.

    The cause is not the outermost dropped edge the plan suspected — that edge
    lies outside the PEC walls and is correctly zero, and its mirror partner
    does not exist. It is ``Ea[pec] = 0`` in the legacy path: a **node**-indexed
    conductor mask applied to an **edge**-indexed field. Node (i,j) and edge
    (i,j) mirror to n−1−i and n−2−i respectively, so the masking zeroes one end
    of each straddling edge pair and not the other. Removing the masking from
    the legacy path drops its asymmetry from 1.0 to 4e-15, which is how this was
    pinned down.

    The conformal path never has the problem: it zeroes an edge by that edge's
    own open fraction, which is indexed like the edge.
    """
    assert _mirror_asymmetry(_coax(0.5e-3, 'conformal', centre_shift=-0.5)) \
        < 1e-12
    assert _mirror_asymmetry(_coax(0.5e-3, 'staircase')) > 0.5


def _circulation(H, grid, ic, jc, m):
    """``∮Ĥ·dl`` on a square Yee contour of half-width ``m`` cells.

    ``Hx[i,j]`` is an x-directed edge from node i to i+1 at node j, ``Hy[i,j]``
    a y-directed edge from node j to j+1 at node i, so this rectangle closes
    exactly on the staggered grid.
    """
    i = np.arange(ic - m, ic + m)
    j = np.arange(jc - m, jc + m)
    return ((H['Hx'][i, jc - m] * grid.dxp[i]).sum()
            + (H['Hy'][ic + m, j] * grid.dyp[j]).sum()
            - (H['Hx'][i, jc + m] * grid.dxp[i]).sum()
            - (H['Hy'][ic - m, j] * grid.dyp[j]).sum())


def test_launched_h_sheet_carries_the_exact_modal_current():
    """``∮Ĥ·dl`` = V/Z₀ on every contour enclosing the inner conductor.

    Ampère says the enclosed current is the same on any such loop, so the
    **loop-independence is the real assertion** — it is what says Ĥ is curl-free
    in the gap, i.e. a valid magnetostatic field. Measured on the reference coax
    at 0.5 mm, loops of ±8/10/12 cells:

        staircase   1.00025 → 1.00032 → 1.00024   (spread 8.5e-5)
        conformal   1.00036 → 1.00049 → 1.00038   (spread 1.3e-4)

    Both paths carry the modal current to better than 0.05% and both are
    curl-free in the gap to ~1e-4. The staircase column used to read
    1.0052 → 1.0034 → 1.0016 (spread 3.6e-3), and that drift was read as
    spurious staircase curl; it was the undersized conductor
    (``docs/mode_solver_staircase_node_mask.md``). With the conductor the run
    actually steps, the staircase Ĥ is as loop-independent as the conformal one,
    so no contrast is asserted here — what R8 needs of the conformal launch is
    tested where it is used, in ``tests/test_modal_port.py``.

    Before S5d both ratios sat ~7% high (1.0758 → 1.0720 for staircase). That
    offset was the *capacitance integral*, not the launch: it inflated Z₀ by the
    same 7%. Only the spread was ever the physical signal.
    """
    cell = 0.5e-3
    ic = jc = int(round(0.5 * int(round(2 * B_OUT / cell))))
    for geometry in ('staircase', 'conformal'):
        grid = _coax(cell, geometry)
        mode = _mode(grid)
        _E, H = mode._staggered_port_fields(grid)
        r = np.array([_circulation(H, grid, ic, jc, m) * mode.impedance
                      for m in (8, 10, 12)])
        assert np.abs(r - 1.0).max() < 5e-3, f"{geometry} modal current off: {r}"
        assert np.ptp(r) < 5e-4, f"{geometry} Ĥ not curl-free in the gap: {r}"


def test_admittance_scale_is_unity_for_a_homogeneous_fill():
    """``s = 1/(Z₀·G)`` collapses to 1 — the two are now the same integral.

    A matched modal sheet needs the discrete conductance G to equal 1/Z₀. Both
    are now read off the open-area energy of the solved operator, so for a
    homogeneous fill they agree identically: G = C·c₀/√ε_r and Z₀ = √ε_r/(c₀C).
    The residual ~3e-9 is the tabulated η₀ against 1/(ε₀c₀), not discretisation.

    The **staircase** path collapses to the same 1 — it did not while ``ê`` was
    masked by the node mask. G sums ``ê²`` over the edges, Z₀ sums ``(Δφ)²`` over
    the face coefficients of the same edges, so the two are one integral only if
    ``ê`` vanishes on exactly the edges whose coefficient contributes nothing.
    Masking ``ê`` by :attr:`TEMMode.pec` killed live surface-to-gap edges the
    energy still counted, and the leftover 1.0058 at this resolution was that
    mismatch, not a discretisation floor. Both paths now mask ``ê`` on the edge
    set their own PEC rule zeroes, so both give 1 — see
    ``docs/mode_solver_staircase_node_mask.md``.
    """
    for geometry in ('conformal', 'staircase'):
        for eps_r in (1.0, 2.3):
            grid = _coax(0.5e-3, geometry, eps_r=eps_r)
            assert _mode(grid).numerical_admittance_scale(grid) == \
                pytest.approx(1.0, rel=1e-6)
