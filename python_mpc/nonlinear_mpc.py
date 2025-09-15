import os
import sys
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_optimization'))

import inspect
import numpy as np
import sympy as sp
from dataclasses import is_dataclass

from python_mpc.mpc_common import initialize_kalman_filter_with_EKF

from external_libraries.MCAP_python_control.python_control.kalman_filter import ExtendedKalmanFilter
from external_libraries.MCAP_python_control.python_control.kalman_filter import DelayedVectorObject
from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy

from external_libraries.MCAP_python_optimization.optimization_utility.sqp_matrix_utility import SQP_CostMatrices_NMPC
from external_libraries.MCAP_python_optimization.python_optimization.sqp_active_set_pcg_pls import SQP_ActiveSet_PCG_PLS

NMPC_SOLVER_MAX_ITERATION_DEFAULT = 20


class NonlinearMPC_TwiceDifferentiable:
    def __init__(
        self,
        delta_time: float,
        X: sp.Matrix,
        U: sp.Matrix,
        X_initial: np.ndarray,
        fxu: sp.Matrix,
        hx: sp.Matrix,
        parameters_struct,
        Np: int,
        Weight_U: np.ndarray,
        Weight_X: np.ndarray,
        Weight_Y: np.ndarray,
        U_min: np.ndarray = None,
        U_max: np.ndarray = None,
        Y_min: np.ndarray = None,
        Y_max: np.ndarray = None,
        Q_kf: np.ndarray = None,
        R_kf: np.ndarray = None,
        Number_of_Delay: int = 0,
        is_ref_trajectory: bool = False,
        caller_file_name: str = None
    ):
        # inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's file name
        if caller_file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]

        # Check compatibility
        if delta_time <= 0.0:
            raise ValueError("State space model must be discrete-time.")

        self.delta_time = delta_time
        self.Number_of_Delay = Number_of_Delay

        if (Np < self.Number_of_Delay):
            raise ValueError(
                "Prediction horizon Np must be greater than the number of delays.")

        if not is_dataclass(parameters_struct):
            raise ValueError(
                "parameters_struct must be a dataclass instance.")

        self.Np = Np

        self.X_symbolic = X
        self.U_symbolic = U
        self.fxu = fxu
        self.hx = hx

        self.INPUT_SIZE = U.shape[0]
        self.STATE_SIZE = X.shape[0]
        self.OUTPUT_SIZE = hx.shape[0]

        # initialize state
        self.X_inner_model = X_initial

        self.sqp_cost_matrices = self.generate_cost_matrices(
            X_symbolic=X,
            U_symbolic=U,
            fxu=fxu,
            hx=hx,
            Np=Np,
            Weight_U=Weight_U,
            Weight_X=Weight_X,
            Weight_Y=Weight_Y,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max,
        )

        self.sqp_cost_matrices.state_space_parameters = parameters_struct

        self.kalman_filter, \
            (self.fxu_script_function, self.fxu_file_name), \
            (self.hx_script_function, self.hx_file_name) \
            = initialize_kalman_filter_with_EKF(
                X_initial=X_initial,
                X=X, U=U,
                fxu=fxu,
                fxu_jacobian_X=self.sqp_cost_matrices.A_matrix,
                hx=hx,
                hx_jacobian=self.sqp_cost_matrices.C_matrix,
                Q_kf=Q_kf,
                R_kf=R_kf,
                parameters_struct=parameters_struct,
                Number_of_Delay=Number_of_Delay,
                file_name_without_ext=caller_file_name_without_ext
            )

        self.solver = SQP_ActiveSet_PCG_PLS(
            U_size=(self.INPUT_SIZE, self.Np)
        )

        self.solver.set_solver_max_iteration(NMPC_SOLVER_MAX_ITERATION_DEFAULT)

    def generate_cost_matrices(
            self,
            X_symbolic: sp.Matrix,
            U_symbolic: sp.Matrix,
            fxu: sp.Matrix,
            hx: sp.Matrix,
            Np: int,
            Weight_U: np.ndarray,
            Weight_X: np.ndarray,
            Weight_Y: np.ndarray,
            U_min: np.ndarray,
            U_max: np.ndarray,
            Y_min: np.ndarray,
            Y_max: np.ndarray,
    ):
        Qx = np.diag(Weight_X)
        Qy = np.diag(Weight_Y)
        R = np.diag(Weight_U)

        sqp_cost_matrices = SQP_CostMatrices_NMPC(
            x_syms=X_symbolic,
            u_syms=U_symbolic,
            state_equation_vector=fxu,
            measurement_equation_vector=hx,
            Np=Np,
            Qx=Qx,
            Qy=Qy,
            R=R,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max,
        )

        return sqp_cost_matrices

    def set_solver_max_iteration(
            self,
            max_iteration: int
    ):
        self.solver.set_solver_max_iteration(max_iteration)
