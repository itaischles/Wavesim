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
    plot_poynting()        — single power-flow (Poynting) frame: flow arrows + magnitude
    animate_poynting()     — animation of PoyntingMonitor data
    plot_monitor_time_series() — FieldProbe time series (component or |E|/|H|)
    plot_voltage_current() — VoltageMonitor / CurrentMonitor time series
    plot_energy()          — total energy vs time (log scale)

FIELD DIAGNOSTICS (full 3D):
    plot_field_slices_3d()    — orthogonal XY/XZ/YZ slice triptych
    animate_field_slices_3d() — multi-plane time animation (general)

ELECTROSTATIC SOLUTIONS (no time stepping):
    plot_electrostatic_slice() — one quantity (φ, |E|, a component) on one plane
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
    2D colour map of a material array (eps, mu or sigma) in the XY plane.

    Shows cell boundaries as thin grid lines. Annotates with colour bar
    and physical dimensions.

    If cpml is provided, overlays the PML region as a shaded border.

    PEC cells (grid.pec_mask) are marked with a distinct hatch pattern.

    Parameters
    ----------
    component : str
        One of: 'eps_x', 'eps_y', 'eps_z', 'mu_x', 'mu_y', 'mu_z',
        'sigma_x', 'sigma_y', 'sigma_z'. The sigma arrays exist only on a lossy
        grid (see :mod:`wavesim.loss`).
    """
    if component.startswith('sigma') and getattr(grid, component, None) is None:
        raise ValueError(
            f"{component!r}: this grid carries no conductivity, so there is "
            f"nothing to plot. Place a lossy material (sigma=... on set_box / "
            f"set_dielectric) or plot an 'eps_*' / 'mu_*' component instead.")

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


# ======================================================================= #
# POYNTING (power-flow) VISUALISATION
# ======================================================================= #
#
# A PoyntingMonitor frame is (Na, Nb, 3): the two in-plane axes of the slice
# followed by the Cartesian (Sx, Sy, Sz) at each cell centre. The helpers below
# split that into the two in-plane components (drawn as flow arrows) and the
# through-plane component, and render power flux in W/m². They mirror
# plot_field_snapshot / animate_snapshots for the E/H SnapshotMonitor.

_AXES = ('x', 'y', 'z')


def _poynting_split(frame: np.ndarray, normal: str):
    """Split a (Na, Nb, 3) Poynting frame into in-plane (Sa, Sb) and normal Sn.

    Returns (Sa, Sb, Sn, a_ax, b_ax) where a_ax/b_ax are the Cartesian axis
    indices (0/1/2) of the two in-plane directions, in the same order as the
    frame's two spatial axes (increasing axis index — the order
    :func:`_collocate_slice` produces).
    """
    frame = np.asarray(frame)
    n_ax = _AXES.index(normal)
    a_ax, b_ax = (k for k in range(3) if k != n_ax)
    return frame[..., a_ax], frame[..., b_ax], frame[..., n_ax], a_ax, b_ax


def _poynting_background(Sa, Sb, Sn, background: str):
    """The scalar field drawn under the flow arrows, plus (is_signed, label)."""
    if background == 'normal':
        return Sn, True, 'S·n̂ (W/m²)'
    if background == 'magnitude':
        return np.sqrt(Sa**2 + Sb**2 + Sn**2), False, '|S| (W/m²)'
    if background == 'inplane':
        return np.hypot(Sa, Sb), False, '|S_∥| (W/m²)'
    raise ValueError("background must be 'inplane', 'magnitude' or 'normal', "
                     f"got {background!r}")


def plot_poynting(frame: np.ndarray, grid: FDTDGrid, normal: str = 'z',
                  time: float = None, ax=None, quiver_step: int = None,
                  background: str = 'inplane', cmap: str = None,
                  normalize: bool = False):
    """
    Draw one Poynting (power-flow) frame: a scalar background + in-plane arrows.

    The in-plane components of S are shown as a quiver (which way power flows in
    the slice) over a colour map of a scalar derived from S (how strong the flow
    is). This is the still-frame companion to :func:`plot_field_snapshot`.

    Parameters
    ----------
    frame : np.ndarray, shape (Na, Nb, 3)
        A single frame from a :class:`~wavesim.monitors.PoyntingMonitor`
        (``monitor.snapshots[k]``): the in-plane axes followed by (Sx, Sy, Sz).
    grid : FDTDGrid
    normal : str
        Slice normal ('x'/'y'/'z'); must match the monitor's ``normal``.
    time : float, optional
        Frame time in seconds (``monitor.snap_times[k]``) — shown in the title.
    quiver_step : int, optional
        Draw an arrow every this many cells (auto from the frame size if None).
    background : str
        Scalar drawn under the arrows: 'inplane' (in-plane magnitude |S_∥|,
        default), 'magnitude' (full |S|), or 'normal' (through-plane S·n̂, drawn
        on a zero-centred diverging scale).
    cmap : str, optional
        Colormap. Defaults to 'RdBu_r' for the signed 'normal' background,
        'inferno' otherwise.
    normalize : bool
        If True, draw all arrows the same length (direction only) — the
        background then carries all magnitude information.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    Sa, Sb, Sn, a_ax, b_ax = _poynting_split(frame, normal)
    na, nb = Sa.shape

    # Frames are collocated to cell centres and cropped to N-1 cells per in-plane
    # axis, so use the nodes bounding those cells (pcolormesh) and the cell-centre
    # coordinates for the arrows — both non-uniform-grid correct.
    nodes = (grid.x, grid.y, grid.z)
    centres = (grid.xc, grid.yc, grid.zc)
    a_nodes, b_nodes = nodes[a_ax][:na + 1], nodes[b_ax][:nb + 1]
    a_cent, b_cent = centres[a_ax][:na], centres[b_ax][:nb]

    bg, signed, cbar_label = _poynting_background(Sa, Sb, Sn, background)
    if cmap is None:
        cmap = 'RdBu_r' if signed else 'inferno'
    vmax = float(np.max(np.abs(bg)))
    if vmax < 1e-30:
        vmax = 1.0
    vmin = -vmax if signed else 0.0

    im = ax.pcolormesh(a_nodes, b_nodes, bg.T, cmap=cmap, shading='flat',
                       vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)

    if quiver_step is None:
        quiver_step = max(1, min(na, nb) // 25)
    s = quiver_step
    AX, BY = np.meshgrid(a_cent[::s], b_cent[::s], indexing='xy')
    U, V = Sa.T[::s, ::s], Sb.T[::s, ::s]
    if normalize:
        norm = np.hypot(U, V)
        norm[norm == 0.0] = 1.0
        U, V = U / norm, V / norm
    ax.quiver(AX, BY, U, V, color='k', alpha=0.8, pivot='mid')

    ax.set_aspect('equal')
    ax.set_xlabel(f'{_AXES[a_ax]} (m)')
    ax.set_ylabel(f'{_AXES[b_ax]} (m)')
    title = f'Power flow  S = E × H  ({normal}-normal slice)'
    if time is not None:
        title += f'\nt = {time * 1e9:.3f} ns'
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def animate_poynting(poynting_monitor, grid: FDTDGrid, interval_ms: int = 60,
                     quiver_step: int = None, background: str = 'inplane',
                     cmap: str = None, normalize: bool = True):
    """
    Animate a :class:`~wavesim.monitors.PoyntingMonitor`: flowing power over time.

    The in-plane power flow is drawn as a quiver over a colour map of a scalar
    derived from S; both share one colour scale across the whole run so frames
    are comparable. This is the power-flow companion to
    :func:`animate_snapshots`.

    Returns a matplotlib FuncAnimation. Save with
    ``anim.save('out.gif', writer='pillow', fps=18)``.

    Parameters
    ----------
    poynting_monitor : PoyntingMonitor
    grid : FDTDGrid
    interval_ms : int
        Frame interval in milliseconds.
    quiver_step, background, cmap :
        As in :func:`plot_poynting`.
    normalize : bool
        If True (default here), arrows show direction only (unit length) so their
        motion is legible frame-to-frame; magnitude is read from the background.
    """
    snaps = poynting_monitor.snapshots
    times = poynting_monitor.snap_times
    if not snaps:
        raise ValueError("PoyntingMonitor has no recorded snapshots.")

    normal = getattr(poynting_monitor, 'normal', 'z')

    # Global colour scale so the background is comparable across frames.
    bgs = []
    for fr in snaps:
        Sa, Sb, Sn, _, _ = _poynting_split(fr, normal)
        bg, signed, cbar_label = _poynting_background(Sa, Sb, Sn, background)
        bgs.append(bg)
    if cmap is None:
        cmap = 'RdBu_r' if signed else 'inferno'
    vmax = max((float(np.max(np.abs(b))) for b in bgs), default=1e-30)
    if vmax < 1e-30:
        vmax = 1.0
    vmin = -vmax if signed else 0.0

    Sa0, Sb0, Sn0, a_ax, b_ax = _poynting_split(snaps[0], normal)
    na, nb = Sa0.shape
    nodes = (grid.x, grid.y, grid.z)
    centres = (grid.xc, grid.yc, grid.zc)
    a_nodes, b_nodes = nodes[a_ax][:na + 1], nodes[b_ax][:nb + 1]
    a_cent, b_cent = centres[a_ax][:na], centres[b_ax][:nb]

    if quiver_step is None:
        quiver_step = max(1, min(na, nb) // 25)
    s = quiver_step
    AX, BY = np.meshgrid(a_cent[::s], b_cent[::s], indexing='xy')

    def _uv(frame):
        Sa, Sb, _, _, _ = _poynting_split(frame, normal)
        U, V = Sa.T[::s, ::s], Sb.T[::s, ::s]
        if normalize:
            n = np.hypot(U, V)
            n[n == 0.0] = 1.0
            U, V = U / n, V / n
        return U, V

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(a_nodes, b_nodes, bgs[0].T, cmap=cmap, shading='flat',
                       vmin=vmin, vmax=vmax, animated=True)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=9)
    U0, V0 = _uv(snaps[0])
    q = ax.quiver(AX, BY, U0, V0, color='k', alpha=0.8, pivot='mid')
    ax.set_aspect('equal')
    ax.set_xlabel(f'{_AXES[a_ax]} (m)')
    ax.set_ylabel(f'{_AXES[b_ax]} (m)')
    title = ax.set_title('')

    def _update(fr):
        # QuadMesh.set_array wants the C values matching shading='flat'.
        im.set_array(bgs[fr].T.ravel())
        U, V = _uv(snaps[fr])
        q.set_UVC(U, V)
        title.set_text(f'Power flow  S = E × H  ({normal}-normal) — '
                       f't = {times[fr]*1e9:.3f} ns  (frame {fr}/{len(snaps)-1})')
        return im, q, title

    anim = animation.FuncAnimation(fig, _update, frames=len(snaps),
                                   interval=interval_ms, blit=False)
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

    # Transverse node coordinates → correct extents on a non-uniform mesh. Fall
    # back to a uniform da/db ruler for legacy modes that carry no node arrays.
    a_nodes = (mode.a_nodes if mode.a_nodes is not None
               else np.arange(Na + 1) * mode.da)
    b_nodes = (mode.b_nodes if mode.b_nodes is not None
               else np.arange(Nb + 1) * mode.db)
    La, Lb = a_nodes[-1] - a_nodes[0], b_nodes[-1] - b_nodes[0]
    a0, b0 = a_nodes[0], b_nodes[0]

    # Sample positions: φ and the PEC mask are **node**-indexed -- ``phi[i, j]``
    # is the potential *at* node ``(a_nodes[i], b_nodes[j])``, one primary width
    # from its neighbour (see ``TEMMode.E``) -- so they are drawn on the nodes,
    # dropping the last of each, which the (Na, Nb) profiles never represent.
    # Averaging the nodes into cell centres, as this did, is one constant
    # half-cell shift on a uniform mesh and so invisible; on a graded mesh the
    # two rulers are not a rigid shift, and a symmetric cross-section comes out
    # visibly skewed (a coax pin drawn a whole cell off its own axis, the outer
    # bore bursting through one corner of the window).
    #
    # ``Ea[i]`` lives on the *edge* from node i to i+1, so the quiver below is
    # half a cell out along its own axis. That is left alone: the arrows are
    # already strided, and averaging edges onto nodes would smear the exact zero
    # that marks the metal.
    xa = a_nodes[:Na]
    yb = b_nodes[:Nb]
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


# ======================================================================= #
# ELECTROSTATIC SOLUTIONS
# ======================================================================= #
#
# One plane, one quantity — the same way you look at a SnapshotMonitor frame,
# because it is the same question asked of a solution rather than of a run:
# pick a cut, pick what to see on it. ``normal``/``position`` follow that class
# exactly ('z' -> XY, 'y' -> XZ, 'x' -> YZ, position in metres along the
# normal), so the two do not need separate conventions remembered.
#
# Where the samples sit is the one thing that is *not* shared with it. A
# snapshot collocates to cell centres; an electrostatic solution collocates to
# **nodes**, because that is where φ is defined and moving φ to please the
# other quantities would be moving the only exact thing in the picture. E and D
# come to the nodes instead (ElectrostaticSolution.E_nodes), so one coordinate
# grid still serves every quantity — which is the property that makes the plots
# comparable to each other.

# quantity name -> (how to get it from the solution, colour-bar label)
_ES_QUANTITIES = {
    'phi':  (lambda s: s.phi,            'φ (V)'),
    '|E|':  (lambda s: s.E_magnitude(),  '|E| (V/m)'),
    'Ex':   (lambda s: s.E_nodes[0],     'Ex (V/m)'),
    'Ey':   (lambda s: s.E_nodes[1],     'Ey (V/m)'),
    'Ez':   (lambda s: s.E_nodes[2],     'Ez (V/m)'),
    '|D|':  (lambda s: s.D_magnitude(),  '|D| (C/m²)'),
    'Dx':   (lambda s: s.D_nodes[0],     'Dx (C/m²)'),
    'Dy':   (lambda s: s.D_nodes[1],     'Dy (C/m²)'),
    'Dz':   (lambda s: s.D_nodes[2],     'Dz (C/m²)'),
}

# normal -> (axis sliced, the two in-plane axes, their labels), matching
# SnapshotMonitor and mode_solver: 'z' -> XY, 'y' -> XZ, 'x' -> YZ.
_ES_PLANES = {
    'z': (2, 0, 1, 'x', 'y'),
    'y': (1, 0, 2, 'x', 'z'),
    'x': (0, 1, 2, 'y', 'z'),
}


def _node_edges(nodes: np.ndarray, centres: np.ndarray, n: int) -> np.ndarray:
    """The ``n+1`` drawing boundaries for ``n`` node-centred samples.

    ``pcolormesh(shading='flat')`` wants boundaries, and for a node-centred
    array they are *not* ``grid.x``: that array holds the ``n+1`` boundaries of
    ``n`` **cells**, while these are ``n`` samples sitting *at* nodes. Drawing
    one against the other shifts the whole picture half a cell — the same
    off-by-half-a-cell trap :mod:`wavesim.parts` and :mod:`wavesim.pec` both
    document, arriving in the plotting layer.

    Node ``i`` owns the half cell either side of it, so the boundaries are the
    cell centres, closed by the two end nodes themselves. That is exactly the
    dual cell :func:`wavesim.electrostatics._node_dual` integrates over, half
    cells at the walls included, which is what puts a Neumann symmetry plane on
    the edge of the picture rather than half a cell inside it.
    """
    if n < 2:
        return nodes[:2]
    return np.concatenate([nodes[:1], centres[:n - 1], nodes[n - 1:n]])


def plot_electrostatic_slice(sol, quantity: str = 'phi', normal: str = 'z',
                             position: float = None, ax=None, cmap: str = None,
                             symmetric: bool = None, conductors: bool = True,
                             aspect: str = 'equal'):
    """
    2D colour map of one electrostatic quantity on one plane.

    The solution equivalent of :func:`plot_field_snapshot`: choose a cut plane
    and what to see on it.

    Parameters
    ----------
    sol : wavesim.electrostatics.ElectrostaticSolution
    quantity : str
        ``'phi'`` (the default), ``'|E|'``, ``'|D|'``, or a single component
        ``'Ex'``/``'Ey'``/``'Ez'``/``'Dx'``/``'Dy'``/``'Dz'``. Everything is
        drawn on the nodes, so all of them share one coordinate grid.
    normal : {'z', 'y', 'x'}
        Axis the plane is perpendicular to, as :class:`~wavesim.monitors.SnapshotMonitor`:
        ``'z'`` gives an XY plane, ``'y'`` an XZ plane, ``'x'`` a YZ plane.
    position : float, optional
        Where to cut, in metres along ``normal``, snapped to the nearest node.
        Defaults to the middle of the domain.
    ax : matplotlib Axes, optional
        Draw into an existing axes instead of a new figure. The caller then owns
        the title.
    cmap : str, optional
        Defaults to ``'RdBu_r'`` for signed data and ``'inferno'`` otherwise.
    symmetric : bool, optional
        Force a zero-centred (diverging) scale. Auto-detected from the sign of
        the plane when omitted, so a potential that goes negative is drawn
        against a diverging scale and ``|E|`` never is.
    conductors : bool
        Outline the conductors. Taken from the node mask the solve actually
        pinned, not from ``pec_mask`` — the two differ by half a cell on every
        high-side surface, so a ``pec_mask`` outline would not sit on the
        equipotential the picture shows.
    aspect : str
        Axes aspect ratio, ``'equal'`` by default so a cross-section is drawn in
        true proportion — a field picture at a distorted aspect misleads about
        the geometry of the field. When this owns the figure it is shaped to the
        plane's own extents, so an elongated cut comes out wide and short rather
        than as a sliver in a square. Pass ``'auto'`` to stretch the plane to
        fill the axes instead, which is worth it for a very long, thin domain.

    Returns
    -------
    (fig, ax)
    """
    if quantity not in _ES_QUANTITIES:
        raise ValueError(f"quantity must be one of {sorted(_ES_QUANTITIES)}, "
                         f"got {quantity!r}")
    if normal not in _ES_PLANES:
        raise ValueError(f"normal must be 'x', 'y' or 'z', got {normal!r}")

    getter, label = _ES_QUANTITIES[quantity]
    n_axis, a_axis, b_axis, a_name, b_name = _ES_PLANES[normal]

    grid = sol.grid
    nodes = (grid.x, grid.y, grid.z)
    centres = (grid.xc, grid.yc, grid.zc)
    shape = sol.phi.shape

    idx = (shape[n_axis] // 2 if position is None
           else grid.axis_index(normal, position))
    idx = min(idx, shape[n_axis] - 1)          # the N-th node is not carried
    plane = np.take(np.asarray(getter(sol)), idx, axis=n_axis)

    ae = _node_edges(nodes[a_axis], centres[a_axis], shape[a_axis])
    be = _node_edges(nodes[b_axis], centres[b_axis], shape[b_axis])

    if ax is None:
        # Shape the figure to the plane, so an equal-aspect cut through a long
        # thin domain is a wide short picture rather than a sliver adrift in a
        # square one. Bounded, because a 100:1 domain would otherwise ask for a
        # figure too flat to label.
        ratio = float(be[-1] - be[0]) / max(float(ae[-1] - ae[0]), 1e-30)
        height = min(max(7.0 * ratio, 2.6), 7.5) if aspect == 'equal' else 6.0
        fig, ax = plt.subplots(figsize=(8.0, height + 1.2))
    else:
        fig = ax.figure

    vmax = float(np.max(np.abs(plane)))
    if vmax < 1e-30:
        vmax = 1.0
    if symmetric is None:
        symmetric = bool(np.any(plane < 0.0))
    if cmap is None:
        cmap = 'RdBu_r' if symmetric else 'inferno'

    im = ax.pcolormesh(ae, be, plane.T, shading='flat', cmap=cmap,
                       vmin=-vmax if symmetric else 0.0, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(label, fontsize=10)

    if conductors:
        _draw_conductor_outline(ax, sol, n_axis, a_axis, b_axis, idx)

    ax.set_aspect(aspect)
    ax.set_xlabel(f'{a_name} (m)')
    ax.set_ylabel(f'{b_name} (m)')
    if ax.get_title() == '':
        ax.set_title(f'{label}   —   {normal} = {nodes[n_axis][idx]:.4g} m\n'
                     f'{_es_conditions(sol)}')
    plt.tight_layout()
    return fig, ax


def _draw_conductor_outline(ax, sol, n_axis: int, a_axis: int, b_axis: int,
                            idx: int, color='dimgray'):
    """Contour the solved conductor on one plane of a slice plot.

    ``sol.node_pec`` is node-centred like everything else drawn here, so the
    outline lands on the same ruler as the field and on the equipotential the
    colours show. A plane wholly inside or wholly outside metal has no boundary
    to draw and is skipped, rather than contouring a constant.
    """
    pec = sol.node_pec
    if pec is None:
        return
    plane = np.take(pec, idx, axis=n_axis)
    if not plane.any() or plane.all() or min(plane.shape) < 2:
        return
    nodes = (sol.grid.x, sol.grid.y, sol.grid.z)
    ax.contour(nodes[a_axis][:plane.shape[0]], nodes[b_axis][:plane.shape[1]],
               plane.T.astype(float), levels=[0.5], colors=color,
               linewidths=1.4)


def _es_conditions(sol) -> str:
    """The drive conditions, so a picture is self-describing."""
    driven = ", ".join(f'{n} = {v:g} V'
                       for n, v in sorted(sol.potentials.items()))
    if sol.grounded_bodies:
        driven += (", " if driven else "") + f'{sol.grounded_bodies} grounded'
    return driven or "no assigned potentials"


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
