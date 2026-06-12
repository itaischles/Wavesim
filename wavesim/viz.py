"""
viz.py — All visualisation functions. No plotting in any other module.

Functions split into two groups:

INFRASTRUCTURE (no physics required):
    plot_grid_xy()         — Yee cell grid with staggered E/H positions
    plot_materials_xy()    — 2D colour map of eps/mu + PML overlay + PEC hatch
    plot_source_waveform() — Gaussian pulse time function

FIELD DIAGNOSTICS:
    plot_field_snapshot()  — single 2D field snapshot
    animate_snapshots()    — animation of SnapshotMonitor data
    plot_monitor_time_series() — FieldMonitor or MagnitudeMonitor time series
    plot_energy()          — total energy vs time (log scale)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from wavesim.grid import FDTDGrid
from wavesim.sources import GaussianSource, gaussian_pulse


# ======================================================================= #
# INFRASTRUCTURE VISUALISATIONS
# ======================================================================= #

def plot_grid_xy(grid: FDTDGrid, cpml=None, ax=None):
    """
    Draw the Yee cell grid in the XY plane (k=0 slice).

    Shows E and H component locations as staggered markers per the Yee
    convention. Annotates cell dimensions dx, dy and total domain size
    in metres.

    If cpml is provided, shades the PML region with a semi-transparent
    overlay and labels its thickness in cells.

    Yee positions (relative to cell corner at (i*dx, j*dy)):
        Ex: (i,    j+½)  → centre of bottom edge
        Ey: (i+½,  j  )  → centre of left edge
        Ez: (i+½,  j+½)  → cell centre

        Hx: (i+½,  j  )  → same as Ey (different component)
        Hy: (i,    j+½)  → same as Ex (different component)
        Hz: (i,    j  )  → cell corner
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    Nx, Ny = grid.Nx, grid.Ny
    dx, dy = grid.dx, grid.dy

    # Draw cell grid lines
    step = 1
    for i in range(0, Nx + 1, step):
        ax.axvline(i * dx * 1e3, color='lightgray', lw=0.5, zorder=1)
    for j in range(0, Ny + 1, step):
        ax.axhline(j * dy * 1e3, color='lightgray', lw=0.5, zorder=1)

    # Plot staggered field positions for a small representative patch
    # Show a 4x4 block in the interior (away from PML)
    d = cpml.d_pml if cpml is not None else 0
    i0 = d + 2
    j0 = d + 2
    n_show = min(4, Nx - d - i0, Ny - d - j0)

    marker_kw = dict(s=1, zorder=5)
    for di in range(n_show):
        for dj in range(n_show):
            i = i0 + di
            j = j0 + dj
            x0, y0 = i * dx * 1e3, j * dy * 1e3
            hx, hy = dx * 1e3, dy * 1e3

            # Ez at cell centre
            ax.scatter(x0 + 0.5*hx, y0 + 0.5*hy, marker='o',
                       color='blue', label='Ez' if (di==0 and dj==0) else '', **marker_kw)
            # Ex at (i, j+½)
            ax.scatter(x0, y0 + 0.5*hy, marker='^',
                       color='green', label='Ex' if (di==0 and dj==0) else '', **marker_kw)
            # Ey at (i+½, j)
            ax.scatter(x0 + 0.5*hx, y0, marker='>',
                       color='red', label='Ey' if (di==0 and dj==0) else '', **marker_kw)
            # Hz at corner (i, j)
            ax.scatter(x0, y0, marker='s',
                       color='purple', label='Hz' if (di==0 and dj==0) else '', **marker_kw)
            # Hx at (i+½, j)
            ax.scatter(x0 + 0.5*hx, y0, marker='D',
                       color='orange', label='Hx' if (di==0 and dj==0) else '', **marker_kw)
            # Hy at (i, j+½)
            ax.scatter(x0, y0 + 0.5*hy, marker='P',
                       color='brown', label='Hy' if (di==0 and dj==0) else '', **marker_kw)

    # PML overlay
    if cpml is not None:
        _draw_pml_overlay(ax, grid, cpml.d_pml, units='mm')

    ax.set_xlim(0, Nx * dx * 1e3)
    ax.set_ylim(0, Ny * dy * 1e3)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Yee Grid — XY plane\n'
                 f'Nx={Nx}, Ny={Ny}, dx={dx*1e3:.2f} mm, dy={dy*1e3:.2f} mm\n'
                 f'Domain: {Nx*dx*1e3:.1f} mm × {Ny*dy*1e3:.1f} mm')
    ax.set_aspect('equal')

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper right', fontsize=8, markerscale=1.2)

    plt.tight_layout()
    return fig, ax


def plot_materials_xy(grid: FDTDGrid, component: str = 'eps_z',
                      cpml=None, ax=None):
    """
    2D colour map of a material array (eps or mu) in the XY plane.

    Shows cell boundaries as thin grid lines. Annotates with colour bar
    and physical dimensions.

    If cpml is provided, overlays the PML region as a shaded border.

    PEC cells (grid.pec_mask) are marked with a distinct hatch pattern.

    Parameters
    ----------
    component : str
        One of: 'eps_x', 'eps_y', 'eps_z', 'mu_x', 'mu_y', 'mu_z'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    Nx, Ny = grid.Nx, grid.Ny
    dx, dy = grid.dx, grid.dy

    arr = getattr(grid, component)[:, :, 0]   # 2D slice at k=0

    # Physical extent for imshow
    extent = [0, Nx * dx * 1e3, 0, Ny * dy * 1e3]  # mm

    im = ax.imshow(arr.T, origin='lower', extent=extent,
                   cmap='plasma', aspect='equal',
                   vmin=arr.min(), vmax=max(arr.max(), arr.min() + 1e-10))
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(component, fontsize=10)

    # PEC hatch overlay
    if grid.pec_mask is not None:
        pec_2d = grid.pec_mask[:, :, 0]
        pec_rgba = np.zeros((*pec_2d.T.shape, 4))
        pec_rgba[pec_2d.T, :] = [0.2, 0.2, 0.2, 0.6]   # dark grey, semi-opaque
        ax.imshow(pec_rgba, origin='lower', extent=extent, aspect='equal',
                  zorder=3)
        # Hatch via contourf is tricky with imshow; use a patch legend entry instead
        pec_patch = mpatches.Patch(color='dimgray', alpha=0.6, label='PEC')
        ax.legend(handles=[pec_patch], loc='upper right', fontsize=9)

    # PML overlay
    if cpml is not None:
        _draw_pml_overlay(ax, grid, cpml.d_pml, units='mm')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Material map: {component} (k=0 slice)\n'
                 f'Domain: {Nx*dx*1e3:.1f} mm × {Ny*dy*1e3:.1f} mm')

    plt.tight_layout()
    return fig, ax


def plot_source_waveform(source: GaussianSource, dt: float, n_steps: int, ax=None):
    """
    1D plot of the Gaussian pulse time function over the simulation duration.

    X-axis: time in nanoseconds. Y-axis: normalised amplitude.
    Marks t0 and the ±2σ width. Prints estimated bandwidth to stdout.

    Parameters
    ----------
    source  : GaussianSource
    dt      : float   timestep (seconds)
    n_steps : int     total number of timesteps to plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    t = np.arange(n_steps) * dt
    t_ns = t * 1e9
    values = np.array([gaussian_pulse(source, ti) for ti in t])
    values_norm = values / (values.max() + 1e-30)

    ax.plot(t_ns, values_norm, color='steelblue', lw=1.5, label='Gaussian pulse')
    ax.axvline(source.t0 * 1e9, color='orange', lw=1.2, ls='--', label=f't₀ = {source.t0*1e9:.2f} ns')
    ax.axvline((source.t0 - 2*source.width) * 1e9, color='green', lw=1.0, ls=':',
               label=f'±2σ = ±{2*source.width*1e9:.2f} ns')
    ax.axvline((source.t0 + 2*source.width) * 1e9, color='green', lw=1.0, ls=':')
    ax.axhline(0.01, color='gray', lw=0.8, ls='--', alpha=0.5, label='1% level')

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Normalised amplitude')
    ax.set_title('Gaussian Source Waveform')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_ns[-1])

    # Report bandwidth
    bw_hz = 1.0 / (2.0 * np.pi * source.width)
    bw_ghz = bw_hz / 1e9
    print(f"Source bandwidth (-3 dB): {bw_ghz:.2f} GHz  (f_max ≈ {bw_ghz:.2f} GHz)")
    print(f"Pulse window check: "
          f"amplitude at t=0 = {abs(gaussian_pulse(source, 0.0)):.2e}, "
          f"at t_end = {abs(gaussian_pulse(source, n_steps*dt)):.2e}")

    plt.tight_layout()
    return fig, ax


# ======================================================================= #
# FIELD VISUALISATIONS
# ======================================================================= #

def plot_field_snapshot(snapshot_array: np.ndarray, grid: FDTDGrid,
                        timestep: int, component: str = 'Ez', ax=None):
    """
    2D colour map of a single field snapshot (a 2D NumPy array).

    Uses diverging colourmap (RdBu) centred at zero.
    Annotates with physical dimensions (metres) and timestep number.

    Parameters
    ----------
    snapshot_array : np.ndarray  shape (Nx, Ny)
    grid           : FDTDGrid
    timestep       : int
    component      : str   label for the colour bar
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    Nx, Ny = grid.Nx, grid.Ny
    extent = [0, Nx * grid.dx * 1e3, 0, Ny * grid.dy * 1e3]

    vmax = np.max(np.abs(snapshot_array))
    if vmax < 1e-30:
        vmax = 1.0

    im = ax.imshow(snapshot_array.T, origin='lower', extent=extent,
                   cmap='RdBu_r', aspect='equal',
                   vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f'{component} (V/m or A/m)', fontsize=9)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'{component} snapshot — timestep {timestep}\n'
                 f't = {timestep * grid.dt * 1e9:.3f} ns')
    plt.tight_layout()
    return fig, ax


def animate_snapshots(snapshot_monitor, grid: FDTDGrid, interval_ms: int = 50):
    """
    Animate a sequence of field snapshots from a SnapshotMonitor.

    Returns a matplotlib FuncAnimation object.
    Save with:  anim.save('out.gif', writer='pillow', fps=20)
    Display inline in Jupyter with: from IPython.display import HTML; HTML(anim.to_jshtml())

    Parameters
    ----------
    snapshot_monitor : SnapshotMonitor
    grid             : FDTDGrid
    interval_ms      : int   frame interval in milliseconds
    """
    snaps = snapshot_monitor.snapshots
    times = snapshot_monitor.snap_times
    if not snaps:
        raise ValueError("SnapshotMonitor has no recorded snapshots.")

    vmax = max(np.max(np.abs(s)) for s in snaps)
    if vmax < 1e-30:
        vmax = 1.0

    Nx, Ny = grid.Nx, grid.Ny
    extent = [0, Nx * grid.dx * 1e3, 0, Ny * grid.dy * 1e3]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(snaps[0].T, origin='lower', extent=extent,
                   cmap='RdBu_r', aspect='equal',
                   vmin=-vmax, vmax=vmax, animated=True)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    title = ax.set_title('')

    def _update(frame):
        im.set_data(snaps[frame].T)
        title.set_text(f'{snapshot_monitor.component} — '
                       f't = {times[frame]*1e9:.3f} ns  (frame {frame}/{len(snaps)-1})')
        return im, title

    anim = animation.FuncAnimation(
        fig, _update, frames=len(snaps),
        interval=interval_ms, blit=True
    )
    plt.tight_layout()
    return anim


def plot_monitor_time_series(monitor, dt: float, ax=None):
    """
    Plot a FieldMonitor or MagnitudeMonitor time series.

    X-axis: time in nanoseconds. Y-axis: field value or magnitude in SI units.
    Labels with component name and monitor location.

    Parameters
    ----------
    monitor : FieldMonitor or MagnitudeMonitor
    dt      : float   grid timestep (for label only; monitor already stores times)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    t_ns = np.array(monitor.times) * 1e9
    vals = np.array(monitor.values)

    # Determine label
    if hasattr(monitor, 'component'):
        label = f"{monitor.component} at ({monitor.i},{monitor.j},{monitor.k})"
        ylabel = 'Field value (V/m or A/m)'
    else:
        label = f"|{monitor.field}| at ({monitor.i},{monitor.j},{monitor.k})"
        ylabel = '|Field| magnitude (V/m or A/m)'

    ax.plot(t_ns, vals, lw=1.2, label=label)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(ylabel)
    ax.set_title('Field Monitor Time Series')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_energy(monitor, dt: float, ax=None):
    """
    Plot total energy vs time on a log Y-axis.

    Flat = lossless interior; decaying = PML absorbing outgoing waves.
    A rising curve indicates numerical instability — simulation must be stopped.

    Parameters
    ----------
    monitor : EnergyMonitor
    dt      : float   grid timestep (for label only)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    t_ns = np.array(monitor.times) * 1e9
    vals = np.array(monitor.values)

    # Avoid log(0) issues
    vals = np.where(vals > 0, vals, np.nan)

    ax.semilogy(t_ns, vals, lw=1.2, color='steelblue', label='Total EM energy')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Total Electromagnetic Energy vs Time\n'
                 '(decaying = PML absorbing; rising = instability!)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    return fig, ax


# ======================================================================= #
# Internal helper
# ======================================================================= #

def _draw_pml_overlay(ax, grid: FDTDGrid, d_pml: int, units: str = 'mm'):
    """
    Shade the PML region as a semi-transparent border overlay.

    Called internally by plot_grid_xy and plot_materials_xy.
    """
    Nx, Ny = grid.Nx, grid.Ny
    dx, dy = grid.dx, grid.dy
    scale = 1e3 if units == 'mm' else 1.0   # m → mm

    Lx = Nx * dx * scale
    Ly = Ny * dy * scale
    d_x = d_pml * dx * scale
    d_y = d_pml * dy * scale

    pml_color = (0.4, 0.7, 0.9, 0.25)  # light blue, semi-transparent
    edge_kw = dict(linewidth=1.2, edgecolor='steelblue', linestyle='--')

    # 4 rectangular slabs (may overlap at corners — that's fine)
    rects = [
        Rectangle((0,      0),      d_x,  Ly),   # x-low
        Rectangle((Lx-d_x, 0),      d_x,  Ly),   # x-high
        Rectangle((0,      0),      Lx,   d_y),   # y-low
        Rectangle((0,      Ly-d_y), Lx,   d_y),   # y-high
    ]
    for rect in rects:
        rect.set_facecolor(pml_color)
        rect.set_linewidth(edge_kw['linewidth'])
        rect.set_edgecolor(edge_kw['edgecolor'])
        rect.set_linestyle(edge_kw['linestyle'])
        rect.set_zorder(4)
        ax.add_patch(rect)

    # Label one corner
    ax.text(d_x / 2, Ly / 2,
            f'PML\n{d_pml} cells', ha='center', va='center',
            fontsize=7, color='steelblue', rotation=90, zorder=5)
    ax.text(Lx - d_x / 2, Ly / 2,
            f'PML\n{d_pml} cells', ha='center', va='center',
            fontsize=7, color='steelblue', rotation=90, zorder=5)
