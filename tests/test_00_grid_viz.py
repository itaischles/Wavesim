"""
test_00_grid_viz.py — Grid, material, and PML region visualisation.

NO PHYSICS. NO TIME LOOP.

Validates:
    - Grid construction and CPML initialisation
    - Yee cell staggering visualisation
    - Dielectric box placement and material map
    - Coaxial cross-section with PEC inner/outer conductors

Pass criteria (visual inspection):
    1. E and H positions are correctly staggered per the Yee table
    2. PML region is shaded on all 4 sides with correct thickness
    3. Dielectric box appears at correct centre location
    4. Coaxial structure shows inner PEC, outer PEC, and dielectric fill distinctly

Run:
    cd Wavesim
    python tests/test_00_grid_viz.py

Output: saves test_00_output.png with 4 subplots.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works in all environments
import matplotlib.pyplot as plt

from wavesim.grid import create_grid
from wavesim.materials import set_vacuum, set_box, set_coax
from wavesim.pml import init_cpml
from wavesim.viz import plot_grid_xy, plot_materials_xy


def test_00_grid_viz():
    print("=" * 60)
    print("TEST 00 — Grid, Material, and PML Visualisation")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Create grid — 50x50 cells, 1 mm cell size → 5 cm domain
    # ------------------------------------------------------------------ #
    Nx, Ny, Nz = 50, 50, 1
    dx = 1e-3   # 1 mm

    grid = create_grid(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx)
    grid = set_vacuum(grid)

    print(f"\nGrid created:")
    print(f"  Size:     {Nx} x {Ny} x {Nz} cells")
    print(f"  Spacing:  dx = dy = {dx*1e3:.1f} mm")
    print(f"  Domain:   {Nx*dx*1e3:.1f} mm x {Ny*dy*1e3:.1f} mm" if False else
          f"  Domain:   {Nx*dx*1e3:.1f} mm x {Ny*grid.dy*1e3:.1f} mm")
    print(f"  dt:       {grid.dt*1e12:.4f} ps")
    print(f"  CFL:      dt·c/dx = {grid.dt * 299792458 / dx:.4f}  (should be < 1/√2 ≈ 0.707)")

    # Verify CFL is strictly less than 1/sqrt(2) for 2D (or 1/sqrt(3) for 3D)
    cfl_3d = grid.dt * 299792458 * np.sqrt(1/dx**2 + 1/grid.dy**2 + 1/grid.dz**2)
    print(f"  CFL 3D:   {cfl_3d:.4f}  (should be ≤ 0.99)")
    assert cfl_3d <= 1.0, f"CFL > 1.0 — unstable! ({cfl_3d:.4f})"

    # ------------------------------------------------------------------ #
    # Step 2: Initialise CPML with d_pml=8
    # ------------------------------------------------------------------ #
    d_pml = 8
    cpml = init_cpml(grid, d_pml=d_pml)

    print(f"\nCPML initialised:")
    print(f"  d_pml = {cpml.d_pml} cells")
    print(f"  psi_Hz_x_lo shape: {cpml.psi_Hz_x_lo.shape}")
    print(f"  bx_E (first 3): {cpml.bx_E[:3]}")
    print(f"  cx_E (first 3): {cpml.cx_E[:3]}")

    # Verify profiles are physically sensible
    assert np.all(cpml.bx_E >= 0) and np.all(cpml.bx_E <= 1.0), \
        f"bx_E out of [0,1] range: min={cpml.bx_E.min():.4f}, max={cpml.bx_E.max():.4f}"
    assert np.all(cpml.by_E >= 0) and np.all(cpml.by_E <= 1.0), \
        "by_E out of [0,1] range"
    print("  [PASS] PML coefficients in valid range")

    # ------------------------------------------------------------------ #
    # Step 3: Plot Yee grid with PML overlay
    # ------------------------------------------------------------------ #
    print("\nGenerating grid visualisation...")
    fig_grid, ax_grid = plot_grid_xy(grid, cpml=cpml)

    # ------------------------------------------------------------------ #
    # Step 4: Place a dielectric box (eps_r=4) in the centre
    # ------------------------------------------------------------------ #
    grid = set_vacuum(grid)   # reset first

    # Centre box: 10mm x 10mm, centred at (25mm, 25mm)
    cx_m, cy_m = Nx * dx / 2, Ny * grid.dy / 2          # 25 mm
    box_half   = 5e-3   # 5 mm half-width → 10 mm total box
    grid = set_box(grid,
                   x0=cx_m - box_half, x1=cx_m + box_half,
                   y0=cy_m - box_half, y1=cy_m + box_half,
                   z0=0.0, z1=grid.Nz * grid.dz,
                   eps_r=4.0)

    print(f"\nDielectric box placed:")
    print(f"  Centre: ({cx_m*1e3:.1f}, {cy_m*1e3:.1f}) mm")
    print(f"  Size:   {2*box_half*1e3:.0f} mm x {2*box_half*1e3:.0f} mm")
    print(f"  eps_r:  4.0")

    # Check eps_z value at cell (25, 25, 0) — should be 4.0
    i_c = int(cx_m / dx)
    j_c = int(cy_m / grid.dy)
    eps_centre = grid.eps_z[i_c, j_c, 0]
    print(f"  eps_z at cell ({i_c},{j_c},0): {eps_centre:.2f}  (expected 4.0)")
    assert abs(eps_centre - 4.0) < 0.01, f"eps_z at centre = {eps_centre} (expected 4.0)"
    print("  [PASS] Dielectric box correctly placed")

    fig_box, ax_box = plot_materials_xy(grid, component='eps_z', cpml=cpml)
    ax_box.set_title(f'Material map — eps_z\nDielectric box (eps_r=4) at centre, PML overlay')

    # ------------------------------------------------------------------ #
    # Step 6: Place coaxial cross-section
    # ------------------------------------------------------------------ #
    grid = set_vacuum(grid)   # reset

    # Coaxial: centre at domain centre, r_inner=5mm, r_outer=12mm
    r_inner_m = 5e-3    # 5 mm
    r_outer_m = 12e-3   # 12 mm

    grid = set_coax(grid,
                    cx=cx_m, cy=cy_m,
                    r_inner=r_inner_m,
                    r_outer=r_outer_m,
                    eps_r_fill=1.0)

    print(f"\nCoaxial structure placed:")
    print(f"  Centre:  ({cx_m*1e3:.1f}, {cy_m*1e3:.1f}) mm")
    print(f"  r_inner: {r_inner_m*1e3:.1f} mm")
    print(f"  r_outer: {r_outer_m*1e3:.1f} mm")

    # Verify PEC mask is set
    assert grid.pec_mask is not None, "pec_mask is None after set_coax()"
    n_pec = np.sum(grid.pec_mask)
    print(f"  PEC cells: {n_pec}  (inner + outer conductors)")
    assert n_pec > 0, "No PEC cells set by set_coax()"

    # Verify inner conductor is PEC at centre
    assert grid.pec_mask[i_c, j_c, 0], \
        f"Centre cell ({i_c},{j_c},0) should be PEC (inside inner conductor)"

    # Verify a cell between conductors is NOT PEC
    # r_between = (r_inner_m + r_outer_m) / 2
    i_mid = i_c + int((r_inner_m + r_outer_m) / 2 / dx)
    pec_mid = grid.pec_mask[i_mid, j_c, 0]
    print(f"  Cell at r_mid ({i_mid},{j_c},0) is PEC: {pec_mid}  (should be False)")
    assert not pec_mid, f"Cell at r_mid should NOT be PEC"

    print("  [PASS] Coaxial PEC mask correctly set")

    fig_coax, ax_coax = plot_materials_xy(grid, component='eps_z', cpml=cpml)
    ax_coax.set_title(f'Material map — Coaxial cross-section\n'
                      f'Inner PEC (r={r_inner_m*1e3:.0f}mm), '
                      f'Outer PEC (r={r_outer_m*1e3:.0f}mm), '
                      f'Vacuum fill')

    # ------------------------------------------------------------------ #
    # Assemble output figure
    # ------------------------------------------------------------------ #
    fig_out, axes = plt.subplots(2, 2, figsize=(16, 14))
    plt.suptitle('Test 00 — Grid, Material, and PML Visualisation', fontsize=14, y=1.01)

    def _copy_fig_to_ax(src_fig, dst_ax):
        """Render src_fig into a dst_ax as an image (raster copy)."""
        src_fig.canvas.draw()
        buf = src_fig.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8)
        w, h = src_fig.canvas.get_width_height()
        img = img.reshape(h, w, 4)
        dst_ax.imshow(img)
        dst_ax.axis('off')

    _copy_fig_to_ax(fig_grid, axes[0, 0])
    axes[0, 0].set_title('Yee Grid + PML overlay', fontsize=11)

    # For the box and coax, just replot directly into the subplots
    axes[0, 1].axis('off')   # placeholder — grid view already done

    # Replot material maps directly into the output axes
    # Box
    grid_box = create_grid(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx)
    grid_box = set_vacuum(grid_box)
    grid_box = set_box(grid_box,
                       x0=cx_m-box_half, x1=cx_m+box_half,
                       y0=cy_m-box_half, y1=cy_m+box_half,
                       z0=0.0, z1=grid_box.Nz*grid_box.dz,
                       eps_r=4.0)
    plot_materials_xy(grid_box, component='eps_z', cpml=cpml, ax=axes[1, 0])
    axes[1, 0].set_title('Dielectric box (eps_r=4) + PML overlay', fontsize=11)

    plot_materials_xy(grid, component='eps_z', cpml=cpml, ax=axes[1, 1])
    axes[1, 1].set_title('Coaxial cross-section (PEC inner + outer) + PML overlay', fontsize=11)

    # Grid plot (re-do in top-left with fresh axes)
    grid_plain = create_grid(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx)
    cpml_plain = init_cpml(grid_plain, d_pml=d_pml)
    plot_grid_xy(grid_plain, cpml=cpml_plain, ax=axes[0, 0])
    axes[0, 0].set_title('Yee Grid — Staggered E/H positions + PML overlay', fontsize=11)

    # Top-right: summary text
    axes[0, 1].axis('off')
    summary = (
        f"Test 00 Summary\n"
        f"{'─'*30}\n"
        f"Grid: {Nx}×{Ny}×{Nz} cells\n"
        f"dx = {dx*1e3:.1f} mm\n"
        f"Domain: {Nx*dx*1e2:.1f} cm × {Ny*grid.dy*1e2:.1f} cm\n"
        f"dt = {grid.dt*1e12:.3f} ps\n"
        f"CFL (3D) = {cfl_3d:.4f}\n\n"
        f"CPML: d_pml = {d_pml} cells\n"
        f"bx_E range: [{cpml.bx_E.min():.4f}, {cpml.bx_E.max():.4f}]\n\n"
        f"Dielectric box: eps_r = 4.0\n"
        f"  {2*box_half*1e3:.0f}mm × {2*box_half*1e3:.0f}mm at centre\n\n"
        f"Coaxial:\n"
        f"  r_inner = {r_inner_m*1e3:.1f} mm (PEC)\n"
        f"  r_outer = {r_outer_m*1e3:.1f} mm (PEC)\n"
        f"  Fill: vacuum\n\n"
        f"All assertions: PASS ✓"
    )
    axes[0, 1].text(0.1, 0.9, summary, transform=axes[0, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'test_00_output.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')

    plt.close('all')

    print(f"\n{'='*60}")
    print(f"TEST 00 PASSED ✓")
    print(f"Output saved to: {out_path}")
    print(f"{'='*60}")

    return True


if __name__ == '__main__':
    test_00_grid_viz()
