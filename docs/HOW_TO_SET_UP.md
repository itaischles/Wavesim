# Wavesim — Environment Setup

Python 3.10+ required. The dependency list is short — `numpy`, `scipy`,
`matplotlib`, `pillow` (plus optional `numba`) — so **conda is not a
requirement**. Pick whichever of the two options below you prefer:

- **Option A — conda** (§1A/§2): convenient on Windows, bundles a Python
  interpreter with the environment.
- **Option B — plain Python + venv + pip** (§1B/§2): no conda at all; use a
  system/python.org install of Python 3.10+.

All commands run natively on Windows — no WSL needed. `bat`/`cmd` snippets are
shown; on macOS/Linux use the POSIX equivalents (noted inline).

---

## 1A. Option A — create and activate a conda environment

Open **Anaconda Prompt** (or PowerShell after `conda init powershell`):

```bat
conda create -n wavesim python=3.11 -y
conda activate wavesim
```

You'll need to run `conda activate wavesim` at the start of every session, or set it as the default interpreter in VS Code (see section 5).

## 1B. Option B — create and activate a venv (no conda)

Requires an existing Python 3.10+ (from [python.org](https://www.python.org/downloads/)
or your OS package manager — check with `python --version`):

```bat
python -m venv .venv
.venv\Scripts\activate           :: macOS/Linux: source .venv/bin/activate
```

As with conda, re-activate the venv (`.venv\Scripts\activate`) at the start of
every session, or point VS Code at `.venv` as its interpreter (see section 5).

---

## 2. Install dependencies

**Option A — conda:**

```bat
conda install -n wavesim numpy scipy matplotlib pillow -y
pip install numba             :: optional — only for the faster backend='numba'
```

The four conda packages are in the default Anaconda channel — no `conda-forge`
needed.

**Option B — pip (inside the activated venv):**

```bat
pip install numpy scipy matplotlib pillow
pip install numba             :: optional — only for the faster backend='numba'
```

Either way, `numba` is optional (PyPI wheel; works with the current numpy).

**What each package is for:**

| Package | Used by |
|---|---|
| `numpy` | All field arrays, curl operators, CPML coefficients |
| `scipy` | The 2D TEM mode solver (`mode_solver.py`) and FFT analysis (cavity resonance, waveguide dispersion) |
| `matplotlib` | All visualisation (`viz.py`), animation output |
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
│   ├── backend_numba.py
│   ├── sources.py
│   ├── mode_solver.py
│   ├── monitors.py
│   ├── simulation.py
│   └── viz.py
└── docs\                 ← this file
```

---

## 4. Run a simulation

Run scripts **from the project root** (or add the repo to `sys.path`) with the
environment active — `conda activate wavesim` (Option A) or the venv activate
(Option B):

```bat
cd Wavesim
conda activate wavesim           :: Option A;  Option B: .venv\Scripts\activate
python your_script.py
```

Every `wavesim/*.py` module carries a thorough docstring with copy-pasteable
runnable examples (start with `__init__.py`, `simulation.py`, `mode_solver.py`).
Section 6 below is a quick import smoke test that confirms the install works.

---

## 5. VS Code integration

1. Open the `Wavesim` folder: **File → Open Folder**
2. Select the interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose your environment: the `wavesim` conda env (path ending in `\envs\wavesim\python.exe`) for Option A, or the venv (`.venv\Scripts\python.exe`) for Option B
3. Open any script and press **F5** to run, or use the integrated terminal with the environment already active

A script that calls `matplotlib.use('Agg')` runs without a pop-up window and saves PNG output to disk. To get interactive plots while working in VS Code, comment out that line.

---

## 6. Verify the installation

Run this from the project root to confirm everything is importable:

```bat
cd C:\projects\Wavesim
conda activate wavesim           :: Option A;  Option B: .venv\Scripts\activate

python -c "
import numpy as np, matplotlib, scipy, PIL
print(f'numpy      {np.__version__}')
print(f'matplotlib {matplotlib.__version__}')
print(f'scipy      {scipy.__version__}')
print(f'Pillow     {PIL.__version__}')
import wavesim as ws
grid = ws.create_grid(Nx=10, Ny=10, Nz=1, dx=1e-3)
cpml = ws.init_cpml(grid, d_pml=3)
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

To lock exact versions for reproducibility (works for either option, since the
packages come from PyPI):

```bat
conda activate wavesim           :: Option A;  Option B: .venv\Scripts\activate
pip freeze > requirements.txt
```

To recreate on another machine — Option A (conda):

```bat
conda create -n wavesim python=3.11 -y
conda activate wavesim
pip install -r requirements.txt
```

…or Option B (venv, no conda):

```bat
python -m venv .venv
.venv\Scripts\activate           :: macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
