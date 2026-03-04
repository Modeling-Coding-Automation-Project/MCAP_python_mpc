"""
File: nonlinear_mpc.py

"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from python_mpc.nonlinear_mpc_twice_differentiable import NonlinearMPC_TwiceDifferentiable
from python_mpc.nonlinear_mpc_optimization_engine import NonlinearMPC_OptimizationEngine
