# FDTD Engine v1 — Environment Setup

Python 3.10+ required. Uses **Miniconda on Windows** with the Anaconda Prompt (or PowerShell with conda initialised). All `python` / `conda` commands below run on the Windows side — no WSL needed.

---

## 1. Create and activate a conda environment

Open **Anaconda Prompt** (or PowerShell after `conda init powershell`):

```bat
conda create -n wavesim python=3.11 -y
conda activate wavesim
```

You'll need to run `conda activate wavesim` at the start of every session, or set it as the default interpreter in VS Code (see section 5).

---

## 2. Install dependencies

```bat
conda install -n wavesim numpy matplotlib scipy pillow -y
```

All four packages are in the default Anaconda channel — no `conda-forge` needed.

**What each package is for:**

| Package | Used by |
|---|---|
| `numpy` | All field arrays, curl operators, CPML coefficients |
| `matplotlib` | All visualisation (`viz.py`), animation output |
| `scipy` | FFT analysis in test_03 (cavity resonance), test_04 (waveguide) |
| `Pillow` | Saving animated GIFs from `anim.save('out.gif', writer='pillow')` |

---

## 3. Clone the repository

```bat
git clone https://github.com/itaischles/Wavesim.git
cd Wavesim
```

The directory layout is:

```
Wavesim\
├── README.md
├── wavesim\                  ← the solver package (import name: wavesim)
│   ├── __init__.py
│   ├── constants.py
│   ├── grid.py
│   ├── materials.py
│   ├── update.py
│   ├── pml.py
│   ├── pec.py
│   ├── sources.py
│   ├── monitors.py
│   └── viz.py
├── tests\
│   ├── test_00_grid_viz.py
│   ├── test_01_source_viz.py
│   ├── test_02_free_space.py
│   ├── test_03_pec_cavity.py
│   ├── test_04_waveguide.py
│   └── test_05_coax_tem.py          ← planned, not yet implemented
└── docs\                 ← API_GUIDE.md, this file, design notes
```

---

## 4. Run the tests

Tests must be run **from the project root** and **in order** — each is a prerequisite for the next.

```bat
cd Wavesim
conda activate wavesim

python tests\test_00_grid_viz.py
python tests\test_01_source_viz.py
python tests\test_02_free_space.py
python tests\test_03_pec_cavity.py
python tests\test_04_waveguide.py
```

Tests 00–04 are implemented and passing; `test_05_coax_tem.py` is planned.

Each test prints a `PASSED` / `FAILED` summary to stdout and saves a PNG (and,
for tests 02–04, an animated GIF) next to the test script. These artifacts are
git-ignored and regenerated on each run.

---

## 5. VS Code integration

1. Open the `Wavesim` folder: **File → Open Folder**
2. Select the interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `wavesim` conda env (it will show the path ending in `\envs\wavesim\python.exe`)
3. Open any test file and press **F5** to run, or use the integrated terminal with `conda activate wavesim` already active

The tests use `matplotlib.use('Agg')` so they run without a pop-up window and save PNG output to disk. To get interactive plots while working in VS Code, comment out that line at the top of the test file.

---

## 6. Verify the installation

Run this from the project root to confirm everything is importable:

```bat
cd C:\projects\Wavesim
conda activate wavesim

python -c "
import numpy as np, matplotlib, scipy, PIL
print(f'numpy      {np.__version__}')
print(f'matplotlib {matplotlib.__version__}')
print(f'scipy      {scipy.__version__}')
print(f'Pillow     {PIL.__version__}')
from wavesim.grid import create_grid
from wavesim.pml import init_cpml
grid = create_grid(Nx=10, Ny=10, Nz=1, dx=1e-3)
cpml = init_cpml(grid, d_pml=3)
print('wavesim package: OK')
"
```

Expected output (exact versions may differ):

```
numpy      2.x.x
matplotlib 3.x.x
scipy      1.x.x
Pillow     10.x.x
wavesim package: OK
```

---

## 7. Freeze the environment (optional)

To lock exact versions for reproducibility:

```bat
conda activate wavesim
pip freeze > requirements.txt
```

To recreate on another machine:

```bat
conda create -n wavesim python=3.11 -y
conda activate wavesim
pip install -r requirements.txt
```
