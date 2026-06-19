"""
viz.py — All visualisation functions. No plotting in any other module.

Functions split into two groups:

INFRASTRUCTURE (no physics required):
    plot_grid_xy()         — Yee cell grid with staggered E/H positions
    plot_materials_xy()    — 2D colour map of eps/mu + PML overlay + PEC hatch
    plot_source_waveform() — Gaussian pulse time function

FIELD DIAGNOSTICS (2D / single slice):
    plot_field_snapshot()  — single 2D field snapshot
    animate_snapshots()    — animation of SnapshotMonitor data
    plot_monitor_time_series() — FieldMonitor or MagnitudeMonitor time series
    plot_energy()          — total energy vs time (log scale)

FIELD DIAGNOSTICS (full 3D):
    plot_field_slices_3d()    — orthogonal XY/XZ/YZ slice triptych
    animate_field_slices_3d() — multi-plane time animation (general)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from wavesim.grid import FDTDGrid


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
        ax.axvline(i * dx, color='lightgray', lw=0.5, zorder=1)
    for j in range(0, Ny + 1, step):
        ax.axhline(j * dy, color='lightgray', lw=0.5, zorder=1)

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
            x0, y0 = i * dx, j * dy
            hx, hy = dx, dy

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
        _draw_pml_overlay(ax, grid, cpml.d_pml)

    ax.set_xlim(0, Nx * dx)
    ax.set_ylim(0, Ny * dy)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Yee Grid — XY plane\n'
                 f'Nx={Nx}, Ny={Ny}, dx={dx:.4g} m, dy={dy:.4g} m\n'
                 f'Domain: {Nx*dx:.4g} m × {Ny*dy:.4g} m')
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

    # Physical extent for imshow (metres)
    extent = [0, Nx * dx, 0, Ny * dy]

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
        _draw_pml_overlay(ax, grid, cpml.d_pml)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Material map: {component} (k=0 slice)\n'
                 f'Domain: {Nx*dx:.4g} m × {Ny*dy:.4g} m')

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
    extent = [0, Nx * grid.dx, 0, Ny * grid.dy]

    vmax = np.max(np.abs(snapshot_array))
    if vmax < 1e-30:
        vmax = 1.0

    im = ax.imshow(snapshot_array.T, origin='lower', extent=extent,
                   cmap='RdBu_r', aspect='equal',
                   vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f'{component} (V/m or A/m)', fontsize=9)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
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
    extent = [0, Nx * grid.dx, 0, Ny * grid.dy]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(snaps[0].T, origin='lower', extent=extent,
                   cmap='RdBu_r', aspect='equal',
                   vmin=-vmax, vmax=vmax, animated=True)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
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

    # Determine label (monitor location is stored and shown in metres)
    pos_m = f"({monitor.x:.4g}, {monitor.y:.4g}, {monitor.z:.4g}) m"
    if hasattr(monitor, 'component'):
        label = f"{monitor.component} at {pos_m}"
        ylabel = 'Field value (V/m or A/m)'
    else:
        label = f"|{monitor.field}| at {pos_m}"
        ylabel = '|Field| magnitude (V/m or A/m)'

    ax.plot(t_ns, vals, lw=1.2, label=label)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(ylabel)
    ax.set_title('Field Monitor Time Series')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


# ======================================================================= #
# 3D FIELD VISUALISATIONS
# ======================================================================= #
#
# The helpers above assume a single XY (k) slice — fine for the Nz=1 era and
# for one transverse plane of a 3D run. The two functions below are the genuine
# 3D workhorses: an orthogonal-slice triptych through a full (Nx,Ny,Nz) array,
# and a general multi-plane time animator. Both accept either a field-component
# name (resolved against the grid) or a raw 3D NumPy array (e.g. a |E| envelope),
# so derived quantities plot through the same path as raw components.

def _as_3d_array(data, grid: FDTDGrid) -> np.ndarray:
    """Resolve `data` to a 3D array: a component name -> grid array, else asarray."""
    if isinstance(data, str):
        return getattr(grid, data)
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D array, got shape {arr.shape}")
    return arr


def plot_field_slices_3d(data, grid: FDTDGrid, component: str = '',
                         x: float = None, y: float = None, z: float = None,
                         cmap: str = None, symmetric: bool = None,
                         fig=None, axes=None):
    """
    Orthogonal-slice triptych (XY, XZ, YZ) through a 3D field.

    Draws three panels sharing one colour scale, with crosshairs marking where
    the other two cut planes intersect each view. This is the canonical way to
    inspect a full 3D run; the existing 2D helpers only see one k-slice.

    Parameters
    ----------
    data : str or np.ndarray
        A field-component name ('Ex'..'Hz') resolved against `grid`, or a raw
        (Nx,Ny,Nz) array such as a |E| envelope.
    component : str
        Label for the colour bar / titles (defaults to `data` when it is a name).
    x, y, z : float, optional
        Cut positions in metres for the YZ, XZ, XY planes, each snapped to the
        nearest cell. Default to the domain centre.
    cmap : str, optional
        Colormap. Defaults to 'RdBu_r' for signed data, 'inferno' otherwise.
    symmetric : bool, optional
        Force a zero-centred (diverging) scale. Auto-detected from the sign of
        the data when omitted.
    fig : matplotlib Figure, optional
        Figure to draw into (a fresh 1x3 row of axes is created on it). Ignored
        when `axes` is given.
    axes : tuple of 3 Axes, optional
        Pre-existing (ax_xy, ax_xz, ax_yz) to draw into — use this to embed the
        triptych in a larger multi-panel figure. The caller owns the suptitle.

    Returns
    -------
    (fig, (ax_xy, ax_xz, ax_yz))
    """
    arr = _as_3d_array(data, grid)
    label = component or (data if isinstance(data, str) else 'field')
    Nx, Ny, Nz = arr.shape
    # Cut planes are given in metres -> snap to cell indices (centre by default).
    i = Nx // 2 if x is None else grid.axis_index('x', x)
    j = Ny // 2 if y is None else grid.axis_index('y', y)
    k = Nz // 2 if z is None else grid.axis_index('z', z)

    vmax = float(np.max(np.abs(arr)))
    if vmax < 1e-30:
        vmax = 1.0
    if symmetric is None:
        symmetric = bool(np.any(arr < 0.0))
    if cmap is None:
        cmap = 'RdBu_r' if symmetric else 'inferno'
    vmin = -vmax if symmetric else 0.0

    Lx, Ly, Lz = Nx * grid.dx, Ny * grid.dy, Nz * grid.dz
    xi, yj = i * grid.dx, j * grid.dy
    zk = k * grid.dz

    own_fig = axes is None
    if own_fig:
        if fig is None:
            fig = plt.figure(figsize=(15, 4.6))
        ax_xy = fig.add_subplot(1, 3, 1)
        ax_xz = fig.add_subplot(1, 3, 2)
        ax_yz = fig.add_subplot(1, 3, 3)
    else:
        ax_xy, ax_xz, ax_yz = axes
        fig = ax_xy.figure
    cross_kw = dict(color='limegreen', lw=0.8, ls='--', alpha=0.8)

    # XY plane at k (transverse cross-section: keep it physically square)
    im = ax_xy.imshow(arr[:, :, k].T, origin='lower', extent=[0, Lx, 0, Ly],
                      cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax_xy.axvline(xi, **cross_kw); ax_xy.axhline(yj, **cross_kw)
    ax_xy.set_xlabel('x (m)'); ax_xy.set_ylabel('y (m)')
    ax_xy.set_title(f'XY plane  (k={k}, z={zk:.4g} m)')

    # XZ plane at j (z horizontal — usually the long axis)
    ax_xz.imshow(arr[:, j, :], origin='lower', extent=[0, Lz, 0, Lx],
                 cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax_xz.axvline(zk, **cross_kw); ax_xz.axhline(xi, **cross_kw)
    ax_xz.set_xlabel('z (m)'); ax_xz.set_ylabel('x (m)')
    ax_xz.set_title(f'XZ plane  (j={j}, y={yj:.4g} m)')

    # YZ plane at i (z horizontal)
    ax_yz.imshow(arr[i, :, :], origin='lower', extent=[0, Lz, 0, Ly],
                 cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax_yz.axvline(zk, **cross_kw); ax_yz.axhline(yj, **cross_kw)
    ax_yz.set_xlabel('z (m)'); ax_yz.set_ylabel('y (m)')
    ax_yz.set_title(f'YZ plane  (i={i}, x={xi:.4g} m)')

    cbar = fig.colorbar(im, ax=[ax_xy, ax_xz, ax_yz], pad=0.02, fraction=0.04)
    cbar.set_label(label, fontsize=10)
    if own_fig:
        fig.suptitle(f'{label} — orthogonal slices', fontsize=13)
    return fig, (ax_xy, ax_xz, ax_yz)


def animate_field_slices_3d(panels, times=None, interval_ms: int = 60,
                            suptitle: str = ''):
    """
    Animate one or more oriented 2D-plane time series side by side.

    A general multi-panel imshow animator: each panel is a pre-oriented sequence
    of 2D frames (already arranged for origin='lower' imshow) plus its physical
    extent and labels. This generalises `animate_snapshots` (single XY plane) to
    arbitrary orthogonal cuts of a 3D run — e.g. an XZ propagation view next to a
    transverse |E| pattern.

    Parameters
    ----------
    panels : list of dict, each with keys
        frames    : list of 2D np.ndarray   (required; already oriented)
        extent    : [x0, x1, y0, y1] in m   (required)
        xlabel, ylabel, title : str
        cmap      : str   (default 'RdBu_r')
        symmetric : bool  (default True -> vmin=-vmax; else 0..vmax)
        aspect    : 'equal' | 'auto'        (default 'auto')
        vlines    : list of (pos_m, color)  (optional vertical markers, metres)
        hlines    : list of (pos_m, color)  (optional horizontal markers, metres)
    times : sequence, optional
        Per-frame time in seconds; shown (in ns) in the suptitle.
    interval_ms : int
        Frame interval passed to FuncAnimation.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Save with:  anim.save('out.gif', writer='pillow', fps=18)
    """
    if not panels:
        raise ValueError("animate_field_slices_3d needs at least one panel.")
    nframes = min(len(p['frames']) for p in panels)
    if nframes == 0:
        raise ValueError("a panel has no frames.")

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(6.5 * len(panels), 4.6), squeeze=False)
    axes = axes[0]

    ims = []
    for ax, p in zip(axes, panels):
        frames = p['frames']
        sym = p.get('symmetric', True)
        vmax = max((float(np.max(np.abs(f))) for f in frames), default=1e-30)
        if vmax < 1e-30:
            vmax = 1.0
        vmin = -vmax if sym else 0.0
        im = ax.imshow(frames[0], origin='lower', extent=p['extent'],
                       cmap=p.get('cmap', 'RdBu_r'),
                       vmin=vmin, vmax=vmax,
                       aspect=p.get('aspect', 'auto'), animated=True)
        for pos, col in p.get('vlines', []):
            ax.axvline(pos, color=col, ls=':', lw=1)
        for pos, col in p.get('hlines', []):
            ax.axhline(pos, color=col, ls=':', lw=1)
        ax.set_xlabel(p.get('xlabel', '')); ax.set_ylabel(p.get('ylabel', ''))
        ax.set_title(p.get('title', ''))
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        ims.append(im)

    sup = fig.suptitle('')

    def _update(fr):
        for im, p in zip(ims, panels):
            im.set_data(p['frames'][fr])
        txt = suptitle
        if times is not None and fr < len(times):
            txt = (f'{suptitle}   ' if suptitle else '') + \
                  f't = {times[fr]*1e9:.3f} ns  (frame {fr+1}/{nframes})'
        sup.set_text(txt)
        return (*ims, sup)

    anim = animation.FuncAnimation(fig, _update, frames=nframes,
                                   interval=interval_ms, blit=False)
    plt.tight_layout()
    return anim


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

def _draw_pml_overlay(ax, grid: FDTDGrid, d_pml: int):
    """
    Shade the PML region as a semi-transparent border overlay (metres).

    Called internally by plot_grid_xy and plot_materials_xy.
    """
    Nx, Ny = grid.Nx, grid.Ny
    dx, dy = grid.dx, grid.dy

    Lx = Nx * dx
    Ly = Ny * dy
    d_x = d_pml * dx
    d_y = d_pml * dy

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
