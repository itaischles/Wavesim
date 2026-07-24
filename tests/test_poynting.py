"""Poynting-vector power-flow monitor (:class:`wavesim.monitors.PoyntingMonitor`).

The monitor is the power-flow companion to ``SnapshotMonitor``: same slice
geometry and cadence, but it records S = E × H as a vector at cell centres,
with H averaged onto the E timebase so the cross product is formed at one
instant. Three things are checked:

* **Cross product / timing** — with static, uniform fields the half-step H
  average is exact, so a frame is E × H to machine precision, and its timestamp
  follows the ``(time_step + 1)·dt`` recording rule.
* **Output contract** — frames are ``(Nx-1, Ny-1, 3)`` (collocated + cropped),
  and ``snapshots`` / ``snap_times`` stay the same length.
* **Physics** — a directional plane wave carries power along its propagation
  direction and (for an on-axis launch) nowhere else.
"""
import numpy as np
import pytest

import wavesim as ws
from wavesim.monitors import PoyntingMonitor, record_poynting


# ---------------------------------------------------------------------- #
# Cross product and timing, on static uniform fields
# ---------------------------------------------------------------------- #

def test_static_field_frame_is_exact_cross_product():
    """Ex=2, Hy=3 (constant) ⇒ S = (0, 0, Ex·Hy) = (0, 0, 6) at every cell."""
    g = ws.create_grid(Nx=8, Ny=8, Nz=1, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    g.Ex[...] = 2.0
    g.Hy[...] = 3.0

    mon = PoyntingMonitor(at_z=0.0, every_N_steps=1, normal='z')
    # Step 0 stashes; step 1 completes frame 0 (fields are static, so the
    # t±dt/2 H average is exact).
    g.time_step = 0
    record_poynting(mon, g)
    assert mon.snapshots == []              # still pending, nothing emitted yet
    g.time_step = 1
    record_poynting(mon, g)

    assert len(mon.snapshots) == 1
    frame = mon.snapshots[0]
    assert frame.shape == (g.Nx - 1, g.Ny - 1, 3)
    assert np.allclose(frame[..., 0], 0.0)
    assert np.allclose(frame[..., 1], 0.0)
    assert np.allclose(frame[..., 2], 6.0)
    # Timestamp is the E instant of the recording step (step 0): (0+1)·dt.
    assert mon.snap_times[0] == pytest.approx(g.dt)


def test_snapshots_and_times_stay_the_same_length():
    """A run that ends on a recording step drops the un-completed carry."""
    g = ws.create_grid(Nx=8, Ny=8, Nz=1, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    g.Ex[...] = 1.0
    g.Hy[...] = 1.0

    mon = PoyntingMonitor(at_z=0.0, every_N_steps=5, normal='z')
    for n in range(21):                     # stop ON recording step 20
        g.time_step = n
        record_poynting(mon, g)
    # Steps 0,5,10,15 completed (4 frames); step 20 is stashed but never
    # partnered, so it is dropped — the two lists stay equal length.
    assert len(mon.snapshots) == len(mon.snap_times) == 4
    assert mon._pending is not None         # step 20's frame awaits its H partner


# ---------------------------------------------------------------------- #
# End-to-end physics: a plane wave carries power downstream
# ---------------------------------------------------------------------- #

def test_plane_wave_carries_power_in_propagation_direction():
    """A +x directional launch gives Sx > 0 downstream and Sy ≈ Sz ≈ 0."""
    g = ws.create_grid(Nx=160, Ny=24, Nz=1, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(g)
    pw = ws.PlaneWave('x0', angle=0.0, waveform=ws.Sinusoid(frequency=15e9),
                      d_pml=12, directional=True)
    cpml = ws.init_cpml(g, d_pml=12, faces=('x0', 'x1'))
    mon = ws.PoyntingMonitor(at_z=0.0, every_N_steps=10, normal='z')
    sim = ws.Simulation(g, cpml=cpml, sources=[pw], pec_faces=('y0', 'y1'),
                        monitors=[mon])
    sim.run(700)

    S = np.array(mon.snapshots)                     # (nframes, Nx-1, Ny-1, 3)
    assert S.shape[1:] == (g.Nx - 1, g.Ny - 1, 3)
    assert len(mon.snapshots) == len(mon.snap_times)

    # Interior window: past the source (cell 12), short of the far PML.
    late = S.shape[0] // 2
    win = (slice(late, None), slice(40, 120), slice(None))
    Sx, Sy, Sz = S[..., 0][win], S[..., 1][win], S[..., 2][win]

    assert Sx.mean() > 0.0                          # power flows +x
    scale = np.abs(Sx).max()
    # On-axis launch excites only Ey, Hz ⇒ Sy and Sz are identically zero.
    assert np.abs(Sy).max() < 1e-6 * scale
    assert np.abs(Sz).max() < 1e-6 * scale
