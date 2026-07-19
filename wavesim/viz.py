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
    plot_monitor_time_series() — FieldProbe time series (component or |E|/|H|)
    plot_voltage_current() — VoltageMonitor / CurrentMonitor time series
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
    # Node coordinates are the true (possibly non-uniform) cell boundaries; on a
    # uniform grid grid.x[i] == i*dx exactly.
    xn, yn = grid.x, grid.y

    # Draw cell grid lines
    step = 1
    for i in range(0, Nx + 1, step):
        ax.axvline(xn[i], color='lightgray', lw=0.5, zorder=1)
    for j in range(0, Ny + 1, step):
        ax.axhline(yn[j], color='lightgray', lw=0.5, zorder=1)

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
            x0, y0 = xn[i], yn[j]
            hx, hy = grid.dxp[i], grid.dyp[j]

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

    ax.set_xlim(xn[0], xn[-1])
    ax.set_ylim(yn[0], yn[-1])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Yee Grid — XY plane\n'
                 f'Nx={Nx}, Ny={Ny}, dx∈[{grid.dxp.min():.4g}, {grid.dxp.max():.4g}] m\n'
                 f'Domain: {xn[-1]-xn[0]:.4g} m × {yn[-1]-yn[0]:.4g} m')
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
    # Node coordinates are the true cell boundaries — pcolormesh renders each
    # cell at its physical width, so a non-uniform (rectilinear) grid is drawn
    # correctly; on a uniform grid it matches the old imshow.
    xn, yn = grid.x, grid.y

    arr = getattr(grid, component)[:, :, 0]   # 2D slice at k=0

    im = ax.pcolormesh(xn, yn, arr.T, cmap='plasma', shading='flat',
                       vmin=arr.min(), vmax=max(arr.max(), arr.min() + 1e-10))
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(component, fontsize=10)

    # PEC overlay (a masked mesh so cells sit on the true rectilinear boundaries)
    if grid.pec_mask is not None:
        pec_2d = grid.pec_mask[:, :, 0]
        pec_overlay = np.ma.masked_where(~pec_2d.T, np.ones(pec_2d.T.shape))
        ax.pcolormesh(xn, yn, pec_overlay, shading='flat',
                      cmap=matplotlib.colors.ListedColormap([(0.2, 0.2, 0.2)]),
                      vmin=0, vmax=1, alpha=0.6, zorder=3)
        pec_patch = mpatches.Patch(color='dimgray', alpha=0.6, label='PEC')
        ax.legend(handles=[pec_patch], loc='upper right', fontsize=9)

    # PML overlay
    if cpml is not None:
        _draw_pml_overlay(ax, grid, cpml.d_pml)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Material map: {component} (k=0 slice)\n'
                 f'Domain: {xn[-1]-xn[0]:.4g} m × {yn[-1]-yn[0]:.4g} m')

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

    xn, yn = grid.x, grid.y            # true cell boundaries (non-uniform aware)

    vmax = np.max(np.abs(snapshot_array))
    if vmax < 1e-30:
        vmax = 1.0

    im = ax.pcolormesh(xn, yn, snapshot_array.T, cmap='RdBu_r',
                       shading='flat', vmin=-vmax, vmax=vmax)
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f'{component} (V/m or A/m)', fontsize=9)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'{component} snapshot — timestep {timestep}\n'
                 f't = {timestep * grid.dt * 1e9:.3f} ns')
    plt.tight_layout()
    return fig, ax


def animate_snapshots(snapshot_monitor, grid: FDTDGrid, interval_ms: int = 50,
                      log: bool = False, linthresh: float = None,
                      contour: bool = False, n_contours: int = 8):
    """
    Animate a sequence of field snapshots from a SnapshotMonitor.

    Returns a matplotlib FuncAnimation object.
    Save with:  anim.save('out.gif', writer='pillow', fps=20)
    Display inline in Jupyter with: from IPython.display import HTML; HTML(anim.to_jshtml())

    Parameters
    ----------
    snapshot_monitor : SnapshotMonitor
    grid             : FDTDGrid
    interval_ms      : int    frame interval in milliseconds
    log              : bool   if True, use logarithmic colour scaling. The field
                              is signed, so a symmetric-log (SymLogNorm) scale is
                              used: linear within +/- `linthresh`, log beyond.
    linthresh        : float  linear-region half-width for log scaling. Defaults
                              to vmax/1000 (covers ~3 decades of dynamic range).
    contour          : bool   if True, overlay contour lines on the field. Level
                              spacing follows `log`: linearly spaced when log is
                              False, log-spaced (symmetric about zero) when True.
    n_contours       : int    number of contour levels per sign.
    """
    snaps = snapshot_monitor.snapshots
    times = snapshot_monitor.snap_times
    if not snaps:
        raise ValueError("SnapshotMonitor has no recorded snapshots.")

    vmax = max(np.max(np.abs(s)) for s in snaps)
    if vmax < 1e-30:
        vmax = 1.0

    if log:
        if linthresh is None:
            linthresh = vmax / 1e3
        norm = matplotlib.colors.SymLogNorm(linthresh=linthresh,
                                            vmin=-vmax, vmax=vmax)
        imshow_kw = dict(norm=norm)
    else:
        imshow_kw = dict(vmin=-vmax, vmax=vmax)

    # Snapshots are collocated to cell centres and cropped to N-1 cells per
    # in-plane axis (see SnapshotMonitor), so take the extent from the nodes
    # bounding the cells actually present rather than the whole domain.
    nx, ny = snaps[0].shape
    extent = [grid.x[0], grid.x[nx], grid.y[0], grid.y[ny]]

    # Contour levels: log-spaced (symmetric about zero) or linearly spaced.
    if contour:
        if log:
            pos = np.logspace(np.log10(linthresh), np.log10(vmax), n_contours)
            levels = np.concatenate([-pos[::-1], pos])
        else:
            levels = np.linspace(-vmax, vmax, 2 * n_contours + 1)
        # contour() needs coordinate vectors matching the transposed data (ny, nx)
        # — the frames sit on cell centres, so these are exactly the coordinates.
        xc = grid.xc[:nx]
        yc = grid.yc[:ny]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(snaps[0].T, origin='lower', extent=extent,
                   cmap='RdBu_r', aspect='equal',
                   animated=True, **imshow_kw)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    title = ax.set_title('')

    cs_holder = [None]

    def _draw_contours(frame):
        cs_holder[0] = ax.contour(xc, yc, snaps[frame].T, levels=levels,
                                  colors='k', linewidths=0.5, alpha=0.6)

    if contour:
        _draw_contours(0)

    def _update(frame):
        im.set_data(snaps[frame].T)
        title.set_text(f'{snapshot_monitor.component} — '
                       f't = {times[frame]*1e9:.3f} ns  (frame {frame}/{len(snaps)-1})')
        if contour:
            if cs_holder[0] is not None:
                cs_holder[0].remove()
            _draw_contours(frame)
        return im, title

    # Contours are redrawn each frame, so blitting can't reliably track them.
    anim = animation.FuncAnimation(
        fig, _update, frames=len(snaps),
        interval=interval_ms, blit=not contour
    )
    plt.tight_layout()
    return anim


def plot_monitor_time_series(monitor, dt: float, ax=None):
    """
    Plot a FieldProbe time series.

    X-axis: time in nanoseconds. Y-axis: field value or magnitude in SI units.
    Labels with component name and monitor location.

    Parameters
    ----------
    monitor : FieldProbe
        ``component`` is a single component ('Ex'..'Hz') or a magnitude ('|E|'/'|H|').
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
    label = f"{monitor.component} at {pos_m}"
    if monitor.component in ('|E|', '|H|'):
        ylabel = '|Field| magnitude (V/m or A/m)'
    else:
        ylabel = 'Field value (V/m or A/m)'

    ax.plot(t_ns, vals, lw=1.2, label=label)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(ylabel)
    ax.set_title('Field Monitor Time Series')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_voltage_current(monitors, ax=None):
    """
    1D time-series plot of VoltageMonitor / CurrentMonitor data.

    Accepts a single monitor or a list. Voltages plot against the left axis
    (V), currents against the right axis (A) — a mixed list shares one time
    axis with twin y-axes, so a port's V(t) and I(t) overlay naturally.

    Parameters
    ----------
    monitors : VoltageMonitor | CurrentMonitor | list of them
    ax       : matplotlib Axes, optional (the voltage/left axis)

    Returns
    -------
    (fig, ax) — ax is the left (voltage) axis; the current axis, if created,
    is available as ``ax.right_ax``.
    """
    from wavesim.monitors import VoltageMonitor, CurrentMonitor

    if not isinstance(monitors, (list, tuple)):
        monitors = [monitors]
    if not monitors:
        raise ValueError("plot_voltage_current needs at least one monitor.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    v_mons = [m for m in monitors if isinstance(m, VoltageMonitor)]
    i_mons = [m for m in monitors if isinstance(m, CurrentMonitor)]
    if len(v_mons) + len(i_mons) != len(monitors):
        bad = [type(m).__name__ for m in monitors
               if not isinstance(m, (VoltageMonitor, CurrentMonitor))]
        raise TypeError(f"Expected VoltageMonitor/CurrentMonitor, got {bad}")

    lines = []
    for n, mon in enumerate(v_mons):
        t_ns = np.array(mon.times) * 1e9
        lines += ax.plot(t_ns, mon.values, lw=1.2, color=f'C{n}',
                         label=f'V{n if len(v_mons) > 1 else ""}(t)')
    ax.set_xlabel('Time (ns)')
    if v_mons:
        ax.set_ylabel('Voltage (V)')

    if i_mons:
        if v_mons:                       # mixed -> currents on a twin axis
            ax_i = ax.twinx()
            ax.right_ax = ax_i
        else:
            ax_i = ax
        for n, mon in enumerate(i_mons):
            t_ns = np.array(mon.times) * 1e9
            lines += ax_i.plot(t_ns, mon.values, lw=1.2, ls='--',
                               color=f'C{len(v_mons) + n}',
                               label=f'I{n if len(i_mons) > 1 else ""}(t)')
        ax_i.set_ylabel('Current (A)')

    ax.set_title('Voltage / Current Monitors')
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


# ======================================================================= #
# TEM MODE VISUALISATION
# ======================================================================= #

def plot_tem_mode(mode, ax=None, n_levels: int = 20, quiver_step: int = None):
    """
    Plot a solved TEM mode: potential contours + transverse E field arrows.

    Draws the electrostatic potential ``phi`` as filled contours, overlays the
    transverse E field as a quiver, and outlines the PEC conductors. Per-unit-
    length parameters (Z0, eps_eff) are shown in the title when available.

    Parameters
    ----------
    mode : wavesim.mode_solver.TEMMode
    ax   : matplotlib Axes, optional
    n_levels    : int   number of filled potential contour levels
    quiver_step : int   draw an E arrow every this many cells (auto if None)

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    phi = mode.phi
    Na, Nb = phi.shape
    a_name, b_name = mode.transverse_axes

    # Transverse node coordinates (true cell boundaries) → correct extents and
    # cell-center sample positions on a non-uniform mesh. Fall back to a uniform
    # da/db ruler for legacy modes that carry no node arrays.
    a_nodes = (mode.a_nodes if mode.a_nodes is not None
               else np.arange(Na + 1) * mode.da)
    b_nodes = (mode.b_nodes if mode.b_nodes is not None
               else np.arange(Nb + 1) * mode.db)
    La, Lb = a_nodes[-1] - a_nodes[0], b_nodes[-1] - b_nodes[0]
    a0, b0 = a_nodes[0], b_nodes[0]

    # Filled potential contours (transpose for origin='lower' imshow/contour).
    xa = 0.5 * (a_nodes[:-1] + a_nodes[1:])     # cell centres along axis a
    yb = 0.5 * (b_nodes[:-1] + b_nodes[1:])
    cf = ax.contourf(xa, yb, phi.T, levels=n_levels, cmap='RdBu_r')
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('potential φ (V)', fontsize=10)

    # Transverse E quiver (the two stored E components).
    (Ea_name, Ea), (Eb_name, Eb) = list(mode.E.items())
    if quiver_step is None:
        quiver_step = max(1, min(Na, Nb) // 25)
    s = quiver_step
    AX, BY = np.meshgrid(xa[::s], yb[::s], indexing='xy')
    ax.quiver(AX, BY, Ea.T[::s, ::s], Eb.T[::s, ::s],
              color='k', alpha=0.7, scale_units='xy', pivot='mid')

    # PEC conductor outline.
    if mode.pec is not None and mode.pec.any():
        ax.contour(xa, yb, mode.pec.T.astype(float), levels=[0.5],
                   colors='dimgray', linewidths=1.5)

    ax.set_aspect('equal')
    ax.set_xlabel(f'{a_name} (m)')
    ax.set_ylabel(f'{b_name} (m)')
    ax.set_xlim(a0, a0 + La); ax.set_ylim(b0, b0 + Lb)
    title = (f'TEM mode (conductor {mode.conductor_id}) — '
             f'{mode.normal}-propagation')
    if mode.impedance is not None:
        title += f'\nZ₀ = {mode.impedance:.2f} Ω,  ε_eff = {mode.eps_eff:.3f}'
    ax.set_title(title)
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

    # Physical extents and cut positions from the true node coordinates (each
    # equals N*ds / i*ds on a uniform grid, but tracks a graded mesh correctly).
    Lx, Ly, Lz = grid.x[-1], grid.y[-1], grid.z[-1]
    xi, yj = grid.x[i], grid.y[j]
    zk = grid.z[k]

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

    # XY plane at k (transverse cross-section: keep it physically square).
    # pcolormesh on the node coordinates renders a graded mesh at true widths.
    im = ax_xy.pcolormesh(grid.x, grid.y, arr[:, :, k].T, shading='flat',
                          cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xy.set_aspect('equal')
    ax_xy.axvline(xi, **cross_kw); ax_xy.axhline(yj, **cross_kw)
    ax_xy.set_xlabel('x (m)'); ax_xy.set_ylabel('y (m)')
    ax_xy.set_title(f'XY plane  (k={k}, z={zk:.4g} m)')

    # XZ plane at j (z horizontal — usually the long axis)
    ax_xz.pcolormesh(grid.z, grid.x, arr[:, j, :], shading='flat',
                     cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xz.set_aspect('auto')
    ax_xz.axvline(zk, **cross_kw); ax_xz.axhline(xi, **cross_kw)
    ax_xz.set_xlabel('z (m)'); ax_xz.set_ylabel('x (m)')
    ax_xz.set_title(f'XZ plane  (j={j}, y={yj:.4g} m)')

    # YZ plane at i (z horizontal)
    ax_yz.pcolormesh(grid.z, grid.y, arr[i, :, :], shading='flat',
                     cmap=cmap, vmin=vmin, vmax=vmax)
    ax_yz.set_aspect('auto')
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
    xn, yn = grid.x, grid.y

    # Domain span and PML-shell thickness from the true node coordinates. The
    # non-uniform rehaul keeps the outer d_pml cells uniform, so this equals
    # d_pml*dx on a uniform grid and the real shell width otherwise.
    x0, y0 = xn[0], yn[0]
    Lx = xn[-1] - x0
    Ly = yn[-1] - y0
    d_x = xn[d_pml] - x0
    d_y = yn[d_pml] - y0

    pml_color = (0.4, 0.7, 0.9, 0.25)  # light blue, semi-transparent
    edge_kw = dict(linewidth=1.2, edgecolor='steelblue', linestyle='--')

    # 4 rectangular slabs (may overlap at corners — that's fine)
    rects = [
        Rectangle((x0,           y0),           d_x,  Ly),   # x-low
        Rectangle((x0 + Lx-d_x,  y0),           d_x,  Ly),   # x-high
        Rectangle((x0,           y0),           Lx,   d_y),   # y-low
        Rectangle((x0,           y0 + Ly-d_y),  Lx,   d_y),   # y-high
    ]
    for rect in rects:
        rect.set_facecolor(pml_color)
        rect.set_linewidth(edge_kw['linewidth'])
        rect.set_edgecolor(edge_kw['edgecolor'])
        rect.set_linestyle(edge_kw['linestyle'])
        rect.set_zorder(4)
        ax.add_patch(rect)

    # Label one corner
    ax.text(x0 + d_x / 2, y0 + Ly / 2,
            f'PML\n{d_pml} cells', ha='center', va='center',
            fontsize=7, color='steelblue', rotation=90, zorder=5)
    ax.text(x0 + Lx - d_x / 2, y0 + Ly / 2,
            f'PML\n{d_pml} cells', ha='center', va='center',
            fontsize=7, color='steelblue', rotation=90, zorder=5)
