"""
File: nonlinear_mpc_optimization_engine.py

"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_optimization'))

import inspect
import numpy as np
import sympy as sp
from dataclasses import is_dataclass

from python_mpc.mpc_common import initialize_kalman_filter_with_EKF

from external_libraries.MCAP_python_control.python_control.kalman_filter import DelayedVectorObject


class NonlinearMPC_OptimizationEngine:
    pass
