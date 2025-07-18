import os
import inspect
import numpy as np
import sympy as sp
import control
from dataclasses import is_dataclass, fields, make_dataclass

from mpc_utility.state_space_utility import SymbolicStateSpace
from mpc_utility.state_space_utility import StateSpaceEmbeddedIntegrator
from mpc_utility.state_space_utility import MPC_PredictionMatrices
from mpc_utility.state_space_utility import MPC_ReferenceTrajectory

from mpc_utility.linear_solver_utility import LMPC_QP_Solver
from mpc_utility.state_space_utility_deploy import Adaptive_MPC_StateSpaceInitializer

from external_libraries.MCAP_python_control.python_control.kalman_filter import ExtendedKalmanFilter
from external_libraries.MCAP_python_control.python_control.kalman_filter import DelayedVectorObject

from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy


class AdaptiveMPC_NoConstraints:
    def __init__(self,
                 delta_time: float,
                 X: sp.Matrix, U: sp.Matrix, Y: sp.Matrix,
                 X_initial: np.ndarray,
                 fxu: sp.Matrix, fxu_jacobian_X: sp.Matrix,
                 fxu_jacobian_U: sp.Matrix,
                 hx: sp.Matrix, hx_jacobian: sp.Matrix,
                 parameters_struct,
                 Np: int, Nc: int,
                 Weight_U: np.ndarray, Weight_Y: np.ndarray,
                 Q_kf: np.ndarray = None, R_kf: np.ndarray = None,
                 Number_of_Delay: int = 0,
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

        self.parameters_struct, \
            self.parameters_X_U_struct = self.create_parameters_struct(
                parameters_struct, X, U)

        # X, U symbolic
        self.X_symbolic = X
        self.U_symbolic = U

        # initialize state
        self.X_inner_model = X_initial

        self.AUGMENTED_INPUT_SIZE = U.shape[0]
        self.AUGMENTED_STATE_SIZE = X.shape[0] + Y.shape[0]
        self.AUGMENTED_OUTPUT_SIZE = Y.shape[0]

        self.U_latest = np.zeros(
            (self.AUGMENTED_INPUT_SIZE, 1))

        # create EKF object
        self.kalman_filter, self.fxu_file_name, self.fxu_jacobian_X_file_name, \
            self.hx_file_name, self.hx_jacobian_file_name \
            = self.initialize_kalman_filter(
                X=X, U=U, Y=Y,
                fxu=fxu, fxu_jacobian_X=fxu_jacobian_X,
                hx=hx, hx_jacobian=hx_jacobian,
                Q_kf=Q_kf,
                R_kf=R_kf,
                parameters_struct=parameters_struct,
                file_name_without_ext=caller_file_name_without_ext
            )

        # state space initialization
        self.fxu_jacobian_U_script_function, \
            self.fxu_jacobian_U_file_name = \
            self.generate_fxu_jacobian_U_function(
                fxu_jacobian_U, X, U, caller_file_name_without_ext)

        self.state_space_initializer = Adaptive_MPC_StateSpaceInitializer(
            fxu_function=self.kalman_filter.state_function,
            fxu_jacobian_X_function=self.kalman_filter.state_function_jacobian,
            fxu_jacobian_U_function=self.fxu_jacobian_U_script_function,
            hx_function=self.kalman_filter.measurement_function,
            hx_jacobian_function=self.kalman_filter.measurement_function_jacobian,
            caller_file_name_without_ext=caller_file_name_without_ext
        )

        # Embedded Integrator
        self.augmented_ss = self._generate_state_space_embedded_integrator(
            fxu_jacobian_X=fxu_jacobian_X,
            fxu_jacobian_U=fxu_jacobian_U,
            hx_jacobian=hx_jacobian
        )

        self.state_space_initializer.generate_initial_embedded_integrator(
            parameters_X_U_struct=self.parameters_X_U_struct,
            state_space=self.augmented_ss)

        if Nc > Np:
            raise ValueError("Nc must be less than or equal to Np.")
        self.Np = Np
        self.Nc = Nc

        self.Weight_U_Nc = self.update_weight(Weight_U)

        self.state_space_initializer.generate_prediction_matrices_phi_f(
            Np=Np,
            Nc=Nc,
            state_space=self.augmented_ss)

        self.prediction_matrices = self._create_prediction_matrices()

        self.solver_factor = np.zeros(
            (self.AUGMENTED_INPUT_SIZE * self.Nc,
             self.AUGMENTED_OUTPUT_SIZE * self.Np))
        # self.update_solver_factor(
        #     self.prediction_matrices.Phi_ndarray, self.Weight_U_Nc)

        self.Y_store = DelayedVectorObject(self.AUGMENTED_OUTPUT_SIZE,
                                           self.Number_of_Delay)

        self.is_ref_trajectory = is_ref_trajectory

    def create_parameters_struct(self, parameters, X, U):
        parameters_struct = parameters

        # merge parameters and X, U
        free_symbols = set()
        for mat in [X, U]:
            free_symbols.update(mat.free_symbols)
        symbol_names = [str(s) for s in free_symbols]

        param_names = [k for k in vars(
            type(parameters_struct)) if not k.startswith('__')]

        existing_fields = [(f.name, f.type, f)
                           for f in fields(parameters_struct)]
        new_fields = [(name, float, 0.0)
                      for name in symbol_names if name not in param_names]

        PXU_Struct = make_dataclass(
            "PXU_Struct", existing_fields + new_fields)
        parameters_X_U_struct = PXU_Struct(**vars(parameters_struct))

        return parameters_struct, parameters_X_U_struct

    def initialize_kalman_filter(self,
                                 X: sp.Matrix, U: sp.Matrix, Y: sp.Matrix,
                                 fxu: sp.Matrix, fxu_jacobian_X: sp.Matrix,
                                 hx: sp.Matrix, hx_jacobian: sp.Matrix,
                                 Q_kf: np.ndarray,
                                 R_kf: np.ndarray,
                                 parameters_struct,
                                 file_name_without_ext: str):

        fxu_file_name = ExpressionDeploy.write_state_function_code_from_sympy(
            fxu, X, U, file_name_without_ext)
        fxu_jacobian_X_file_name = \
            ExpressionDeploy.write_state_function_code_from_sympy(
                fxu_jacobian_X, X, U, file_name_without_ext)

        hx_file_name = ExpressionDeploy.write_measurement_function_code_from_sympy(
            hx, X, file_name_without_ext)
        hx_jacobian_file_name = \
            ExpressionDeploy.write_measurement_function_code_from_sympy(
                hx_jacobian, X, file_name_without_ext)

        local_vars = {}

        exec(f"from {fxu_file_name} import function as fxu_script_function",
             globals(), local_vars)
        exec(
            f"from {fxu_jacobian_X_file_name} import function as fxu_jacobian_script_function", globals(), local_vars)
        exec(f"from {hx_file_name} import function as hx_script_function",
             globals(), local_vars)
        exec(
            f"from {hx_jacobian_file_name} import function as hx_jacobian_script_function", globals(), local_vars)

        fxu_script_function = local_vars["fxu_script_function"]
        fxu_jacobian_script_function = local_vars["fxu_jacobian_script_function"]
        hx_script_function = local_vars["hx_script_function"]
        hx_jacobian_script_function = local_vars["hx_jacobian_script_function"]

        kalman_filter = ExtendedKalmanFilter(
            state_function=fxu_script_function,
            state_function_jacobian=fxu_jacobian_script_function,
            measurement_function=hx_script_function,
            measurement_function_jacobian=hx_jacobian_script_function,
            Q=Q_kf,
            R=R_kf,
            Parameters=parameters_struct
        )

        return kalman_filter, fxu_file_name, fxu_jacobian_X_file_name, \
            hx_file_name, hx_jacobian_file_name

    def generate_fxu_jacobian_U_function(
            self, fxu_jacobian_U: sp.Matrix,
            X: sp.Matrix, U: sp.Matrix,
            file_name_without_ext: str):
        fxu_jacobian_U_file_name = ExpressionDeploy.write_state_function_code_from_sympy(
            fxu_jacobian_U, X, U, file_name_without_ext)

        local_vars = {}

        exec(f"from {fxu_jacobian_U_file_name} import function as fxu_jacobian_U_script_function",
             globals(), local_vars)
        fxu_jacobian_U_script_function = local_vars["fxu_jacobian_U_script_function"]

        return fxu_jacobian_U_script_function, fxu_jacobian_U_file_name

    def _generate_state_space_embedded_integrator(
            self,
            fxu_jacobian_X: sp.Matrix,
            fxu_jacobian_U: sp.Matrix,
            hx_jacobian: sp.Matrix
    ) -> StateSpaceEmbeddedIntegrator:

        A = sp.Matrix(fxu_jacobian_X)
        B = sp.Matrix(fxu_jacobian_U)
        C = sp.Matrix(hx_jacobian)

        state_space = SymbolicStateSpace(
            A, B, C,
            delta_time=self.delta_time,
            Number_of_Delay=self.Number_of_Delay)

        augmented_ss = StateSpaceEmbeddedIntegrator(state_space)

        # Check dimensions
        if self.AUGMENTED_INPUT_SIZE != augmented_ss.B.shape[1]:
            raise ValueError(
                "the augmented state space input must have the same size of state_space.B.")
        if self.AUGMENTED_INPUT_SIZE != state_space.B.shape[1]:
            raise ValueError(
                "the augmented state space input must have the same size of state_space.B.")

        if self.AUGMENTED_OUTPUT_SIZE != augmented_ss.C.shape[0]:
            raise ValueError(
                "the augmented state space state must have the same size of state_space.A.")
        if self.AUGMENTED_OUTPUT_SIZE != state_space.C.shape[0]:
            raise ValueError(
                "the augmented state space output must have the same size of state_space.C.")

        return augmented_ss

    def update_weight(self, Weight: np.ndarray):

        return np.diag(np.tile(Weight, (self.Nc, 1)).flatten())

    def _create_prediction_matrices(self) -> MPC_PredictionMatrices:

        prediction_matrices = MPC_PredictionMatrices(
            Np=self.Np,
            Nc=self.Nc,
            INPUT_SIZE=self.AUGMENTED_INPUT_SIZE,
            STATE_SIZE=self.AUGMENTED_STATE_SIZE,
            OUTPUT_SIZE=self.AUGMENTED_OUTPUT_SIZE)

        self.state_space_initializer.generate_Adaptive_MPC_Phi_F_Updater()

        prediction_matrices.Phi_F_updater_function = \
            self.state_space_initializer.Adaptive_MPC_Phi_F_updater_function

        prediction_matrices.update_Phi_F_adaptive_runtime(
            parameters_struct=self.parameters_struct,
            parameters_X_U_struct=self.parameters_X_U_struct,
            X_symbolic=self.X_symbolic, U_symbolic=self.U_symbolic,
            X_ndarray=self.X_inner_model, U_ndarray=self.U_latest)

        return prediction_matrices

    def update_solver_factor(self, Phi: np.ndarray, Weight_U_Nc: np.ndarray):
        if (Phi.shape[1] != Weight_U_Nc.shape[0]) or (Phi.shape[1] != Weight_U_Nc.shape[1]):
            raise ValueError("Weight must have compatible dimensions.")

        self.solver_factor = np.linalg.solve(Phi.T @ Phi + Weight_U_Nc, Phi.T)
