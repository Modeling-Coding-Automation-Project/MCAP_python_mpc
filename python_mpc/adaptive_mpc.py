import os
import inspect
import numpy as np
import sympy as sp
from dataclasses import is_dataclass

from mpc_utility.state_space_utility import SymbolicStateSpace
from mpc_utility.state_space_utility import StateSpaceEmbeddedIntegrator
from mpc_utility.state_space_utility import MPC_PredictionMatrices
from mpc_utility.state_space_utility import MPC_ReferenceTrajectory

from mpc_utility.linear_solver_utility import LMPC_QP_Solver

from external_libraries.MCAP_python_control.python_control.kalman_filter import ExtendedKalmanFilter
from external_libraries.MCAP_python_control.python_control.kalman_filter import DelayedVectorObject


class Adaptive_MPC_NoConstraints:
    def __init__(self, state_space: SymbolicStateSpace,
                 parameters_struct,
                 Np: int, Nc: int,
                 Weight_U: np.ndarray, Weight_Y: np.ndarray,
                 Q_kf: np.ndarray = None, R_kf: np.ndarray = None,
                 is_ref_trajectory: bool = False,
                 caller_file_name: str = None):
        pass
