import os

# Must precede any numba.cuda import: the default NVIDIA binding is blocked in
# this environment (see the CUDA backend notes).
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '0')

import pytest


def cuda_available():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: full 3D runs (tens of seconds); -m 'not slow' to skip")
