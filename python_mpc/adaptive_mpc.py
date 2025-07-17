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

from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy


class Adaptive_MPC_NoConstraints:
    def __init__(self,
                 delta_time: float,
                 X: sp.Matrix, U: sp.Matrix, Y: sp.Matrix,
                 fxu: sp.Matrix, fxu_jacobian: sp.Matrix,
                 hx: sp.Matrix, hx_jacobian: sp.Matrix,
                 parameters_struct,
                 Np: int, Nc: int,
                 Weight_U: np.ndarray, Weight_Y: np.ndarray,
                 Q_kf: np.ndarray = None, R_kf: np.ndarray = None,
                 is_ref_trajectory: bool = False,
                 caller_file_name: str = None):

        # inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is X:
                variable_name = name
                break
        # Get the caller's file name
        if caller_file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]

        # create EKF object
        self.fxu_file_name = ExpressionDeploy.write_state_function_code_from_sympy(
            fxu, X, U)
        self.fxu_jacobian_file_name = \
            ExpressionDeploy.write_state_function_code_from_sympy(
                fxu_jacobian, X, U)

        self.hx_file_name = ExpressionDeploy.write_measurement_function_code_from_sympy(
            hx, X)
        self.hx_jacobian_file_name = \
            ExpressionDeploy.write_measurement_function_code_from_sympy(
                hx_jacobian, X)

        self.ekf = self.initialize_kalman_filter(
            fxu_file_name=self.fxu_file_name,
            fxu_jacobian_file_name=self.fxu_jacobian_file_name,
            hx_file_name=self.hx_file_name,
            hx_jacobian_file_name=self.hx_jacobian_file_name,
            Q_kf=Q_kf,
            R_kf=R_kf,
            parameters_struct=parameters_struct
        )

    def initialize_kalman_filter(self,
                                 fxu_file_name: str,
                                 fxu_jacobian_file_name: str,
                                 hx_file_name: str,
                                 hx_jacobian_file_name: str,
                                 Q_kf: np.ndarray,
                                 R_kf: np.ndarray,
                                 parameters_struct):

        local_vars = {}

        exec(f"from {fxu_file_name} import function as fxu_script_function",
             globals(), local_vars)
        exec(
            f"from {fxu_jacobian_file_name} import function as fxu_jacobian_script_function", globals(), local_vars)
        exec(f"from {hx_file_name} import function as hx_script_function",
             globals(), local_vars)
        exec(
            f"from {hx_jacobian_file_name} import function as hx_jacobian_script_function", globals(), local_vars)

        fxu_script_function = local_vars["fxu_script_function"]
        fxu_jacobian_script_function = local_vars["fxu_jacobian_script_function"]
        hx_script_function = local_vars["hx_script_function"]
        hx_jacobian_script_function = local_vars["hx_jacobian_script_function"]

        ekf = ExtendedKalmanFilter(
            state_transition_function=fxu_script_function,
            state_transition_jacobian=fxu_jacobian_script_function,
            measurement_function=hx_script_function,
            measurement_jacobian=hx_jacobian_script_function,
            process_noise_covariance=Q_kf,
            measurement_noise_covariance=R_kf,
            parameters_struct=parameters_struct
        )

        return ekf
