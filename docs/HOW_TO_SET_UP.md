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
pip install numba             :: optional — only for the faster backend='numba'
```

The four conda packages are in the default Anaconda channel — no `conda-forge`
needed. `numba` is optional (PyPI wheel; works with the current numpy).

**What each package is for:**

| Package | Used by |
|---|---|
| `numpy` | All field arrays, curl operators, CPML coefficients |
| `matplotlib` | All visualisation (`viz.py`), animation output |
| `scipy` | FFT analysis (e.g. cavity resonance, waveguide dispersion) |
| `Pillow` | Saving animated GIFs from `anim.save('out.gif', writer='pillow')` |
| `numba` *(optional)* | The ~10–12× multithreaded `Simulation(backend='numba')` / `wavesim/backend_numba.py` |

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
└── docs\                 ← API_GUIDE.md, this file, design notes
```

---

## 4. Run a simulation

Run scripts **from the project root** (or add the repo to `sys.path`) with the
`wavesim` env active:

```bat
cd Wavesim
conda activate wavesim
python your_script.py
```

[`TUTORIAL.md`](TUTORIAL.md) and [`API_GUIDE.md`](API_GUIDE.md) contain
copy-pasteable runnable examples. Section 6 below is a quick import smoke test
that confirms the install works.

---

## 5. VS Code integration

1. Open the `Wavesim` folder: **File → Open Folder**
2. Select the interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `wavesim` conda env (it will show the path ending in `\envs\wavesim\python.exe`)
3. Open any script and press **F5** to run, or use the integrated terminal with `conda activate wavesim` already active

A script that calls `matplotlib.use('Agg')` runs without a pop-up window and saves PNG output to disk. To get interactive plots while working in VS Code, comment out that line.

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
try:
    import numba
    print(f'numba      {numba.__version__}  (threads {numba.config.NUMBA_NUM_THREADS}) — backend=numba available')
except ImportError:
    print('numba      not installed (optional — backend=numba unavailable)')
"
```

Expected output (exact versions may differ):

```
numpy      2.x.x
matplotlib 3.x.x
scipy      1.x.x
Pillow     10.x.x
wavesim package: OK
numba      0.6x.x  (threads 12) — backend=numba available
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
