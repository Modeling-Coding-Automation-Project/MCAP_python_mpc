import os
import sys
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_optimization'))

import inspect
import numpy as np
import sympy as sp
from dataclasses import is_dataclass

from python_mpc.mpc_common import initialize_kalman_filter_with_EKF

from external_libraries.MCAP_python_control.python_control.kalman_filter import DelayedVectorObject

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

        self.U_horizon = np.zeros((self.INPUT_SIZE, self.Np))

        self.Y_store = DelayedVectorObject(self.OUTPUT_SIZE,
                                           self.Number_of_Delay)

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
            caller_file_name=caller_file_name,
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
            caller_file_name: str,
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
            caller_file_name=caller_file_name,
        )

        return sqp_cost_matrices

    def get_solver_step_iterated_number(self):
        return self.solver.get_solver_step_iterated_number()

    def set_solver_max_iteration(
            self,
            max_iteration: int
    ):
        self.solver.set_solver_max_iteration(max_iteration)

    def set_reference_trajectory(
            self,
            reference_trajectory: np.ndarray
    ):

        if not ((reference_trajectory.shape[1] == self.Np) or
                (reference_trajectory.shape[1] == 1)):
            raise ValueError(
                "Reference vector must be either a single row vector or a Np row vectors.")

        self.sqp_cost_matrices.reference_trajectory = \
            np.zeros((self.OUTPUT_SIZE, self.Np + 1))

        if reference_trajectory.shape[1] == self.Np:
            for i in range(self.OUTPUT_SIZE):
                for j in range(self.Np):
                    self.sqp_cost_matrices.reference_trajectory[i, j] = \
                        reference_trajectory[i, j]

            for i in range(self.OUTPUT_SIZE):
                self.sqp_cost_matrices.reference_trajectory[i, self.Np] = \
                    reference_trajectory[i, self.Np - 1]
        else:
            for i in range(self.OUTPUT_SIZE):
                for j in range(self.Np + 1):
                    self.sqp_cost_matrices.reference_trajectory[i, j] = \
                        reference_trajectory[i, 0]

    def calculate_this_U(self, U_horizon):

        return U_horizon[:, 0].reshape((self.INPUT_SIZE, 1))

    def compensate_X_Y_delay(self, X: np.ndarray, Y: np.ndarray):

        if self.Number_of_Delay > 0:
            Y_measured = Y

            X = self.kalman_filter.get_x_hat_without_delay()
            Y = self.kalman_filter.measurement_function(
                X, self.kalman_filter.Parameters)

            self.Y_store.push(Y)

            self.sqp_cost_matrices.set_Y_offset(
                Y_measured - self.Y_store.get())

            return X
        else:
            return X

    def update_parameters(self, parameters_struct):
        if not is_dataclass(parameters_struct):
            raise ValueError(
                "parameters_struct must be a dataclass instance.")

        self.sqp_cost_matrices.state_space_parameters = parameters_struct
        self.kalman_filter.Parameters = parameters_struct

    def update_manipulation(
            self,
            reference: np.ndarray,
            Y: np.ndarray
    ):
        U_latest = self.calculate_this_U(self.U_horizon)

        self.kalman_filter.predict_and_update(
            U_latest, Y)
        X = self.kalman_filter.x_hat

        X_compensated = self.compensate_X_Y_delay(X, Y)

        self.set_reference_trajectory(reference)

        self.U_horizon = self.solver.solve(
            U_initial=self.U_horizon,
            cost_and_gradient_function=self.sqp_cost_matrices.compute_cost_and_gradient,
            hvp_function=self.sqp_cost_matrices.hvp_analytic,
            X_initial=X_compensated,
            U_min_matrix=self.sqp_cost_matrices.U_min_matrix,
            U_max_matrix=self.sqp_cost_matrices.U_max_matrix,
        )

        U_latest = self.calculate_this_U(self.U_horizon)

        return U_latest
