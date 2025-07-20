""" File: adaptive_mpc.py

This module implements Adaptive Model Predictive Control (MPC)
algorithms for nonlinear systems using symbolic state-space models.
It provides the AdaptiveMPC_NoConstraints class,
which supports state estimation via an Extended Kalman Filter (EKF),
reference trajectory tracking, and adaptive prediction matrix updates.
The controller is designed for discrete-time systems,
handles parameter adaptation, and can compensate for output delays.
Key features include symbolic model deployment,
embedded integrator support, and flexible weighting for control
and output objectives.
"""
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
    """
    Adaptive Model Predictive Control (MPC) class without input/output constraints.
    This class implements an adaptive MPC controller for discrete-time state-space models,
    supporting state estimation via an Extended Kalman Filter (EKF), embedded integrator augmentation,
    and runtime adaptation of prediction matrices. It is designed for use with symbolic models
    (SymPy) and supports reference trajectory tracking, delay compensation, and parameter adaptation.
    """

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

        # remember X, U, A, B, C symbolic matrices
        self.X_symbolic = X
        self.U_symbolic = U
        self.A_symbolic = fxu_jacobian_X
        self.B_symbolic = fxu_jacobian_U
        self.C_symbolic = hx_jacobian

        # initialize state
        self.X_inner_model = X_initial

        self.AUGMENTED_INPUT_SIZE = U.shape[0]
        self.AUGMENTED_STATE_SIZE = X.shape[0] + Y.shape[0]
        self.AUGMENTED_OUTPUT_SIZE = Y.shape[0]

        self.U_latest = np.zeros(
            (self.AUGMENTED_INPUT_SIZE, 1))

        # create EKF object
        self.kalman_filter, \
            (self.fxu_script_function, self.fxu_file_name), \
            (self.hx_script_function, self.hx_file_name) \
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
        (self.A_symbolic_script_function, self.A_symbolic_file_name), \
            (self.B_symbolic_script_function, self.B_symbolic_file_name), \
            (self.C_symbolic_script_function, self.C_symbolic_file_name), \
            self.C_symbolic = \
            self.generate_function_file(
            fxu_jacobian_X=self.A_symbolic,
            fxu_jacobian_U=self.B_symbolic,
            hx_jacobian=self.C_symbolic,
            X=X, U=U, Weight_Y=Weight_Y,
            file_name_without_ext=caller_file_name_without_ext)

        self.state_space_initializer = Adaptive_MPC_StateSpaceInitializer(
            fxu_function=self.fxu_script_function,
            fxu_jacobian_X_function=self.A_symbolic_script_function,
            fxu_jacobian_U_function=self.B_symbolic_script_function,
            hx_function=self.hx_script_function,
            hx_jacobian_function=self.C_symbolic_script_function,
            caller_file_name_without_ext=caller_file_name_without_ext
        )

        # Embedded Integrator
        self.augmented_ss = self._generate_state_space_embedded_integrator(
            fxu_jacobian_X=self.A_symbolic,
            fxu_jacobian_U=self.B_symbolic,
            hx_jacobian=self.C_symbolic
        )

        self.state_space_initializer.generate_initial_embedded_integrator(
            parameters_struct=parameters_struct, X=X, U=U,
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
        self.update_solver_factor(
            self.prediction_matrices.Phi_ndarray, self.Weight_U_Nc)

        self.Y_store = DelayedVectorObject(self.AUGMENTED_OUTPUT_SIZE,
                                           self.Number_of_Delay)

        self.is_ref_trajectory = is_ref_trajectory

    def initialize_kalman_filter(self,
                                 X: sp.Matrix, U: sp.Matrix, Y: sp.Matrix,
                                 fxu: sp.Matrix, fxu_jacobian_X: sp.Matrix,
                                 hx: sp.Matrix, hx_jacobian: sp.Matrix,
                                 Q_kf: np.ndarray,
                                 R_kf: np.ndarray,
                                 parameters_struct,
                                 file_name_without_ext: str):
        """
        Initializes an Extended Kalman Filter (EKF) using symbolic model functions and their Jacobians.

        This method generates Python code for the state and measurement functions (and their Jacobians)
        from SymPy expressions, dynamically imports them, and constructs an EKF instance with the provided
        noise covariances and parameters.

        Args:
            X (sp.Matrix): Symbolic state vector.
            U (sp.Matrix): Symbolic input vector.
            Y (sp.Matrix): Symbolic measurement vector.
            fxu (sp.Matrix): Symbolic state transition function f(x, u).
            fxu_jacobian_X (sp.Matrix): Jacobian of the state transition function with respect to X.
            hx (sp.Matrix): Symbolic measurement function h(x).
            hx_jacobian (sp.Matrix): Jacobian of the measurement function with respect to X.
            Q_kf (np.ndarray): Process noise covariance matrix.
            R_kf (np.ndarray): Measurement noise covariance matrix.
            parameters_struct: Additional parameters required for the filter.
            file_name_without_ext (str): Base filename for generated function scripts.

        Returns:
            tuple: (
                kalman_filter (ExtendedKalmanFilter): Initialized EKF instance,
                fxu_file_name (str): Filename of the generated state function script,
                fxu_jacobian_X_file_name (str): Filename of the generated state function Jacobian script,
                hx_file_name (str): Filename of the generated measurement function script,
                hx_jacobian_file_name (str): Filename of the generated measurement function Jacobian script
        """

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
        kalman_filter.x_hat = self.X_inner_model

        return kalman_filter, \
            (fxu_script_function, fxu_file_name), \
            (hx_script_function, hx_file_name)

    def generate_function_file(
            self,
            fxu_jacobian_X: sp.Matrix,
            fxu_jacobian_U: sp.Matrix,
            hx_jacobian: sp.Matrix,
            X: sp.Matrix, U: sp.Matrix,
            Weight_Y: np.ndarray,
            file_name_without_ext: str):
        """
        Generates Python function files from symbolic Jacobian matrices
          and returns references to the generated functions and their file names.

        This method takes symbolic Jacobians of the system dynamics and measurement functions, generates corresponding Python code files,
        imports the generated functions,
          and returns them along with their file names and the weighted measurement Jacobian.

        Args:
            fxu_jacobian_X (sp.Matrix): Symbolic Jacobian of the system dynamics
              with respect to state variables.
            fxu_jacobian_U (sp.Matrix): Symbolic Jacobian of the system dynamics
              with respect to input variables.
            hx_jacobian (sp.Matrix): Symbolic Jacobian of the measurement function
              with respect to state variables.
            X (sp.Matrix): Symbolic state variable vector.
            U (sp.Matrix): Symbolic input variable vector.
            Weight_Y (np.ndarray): Weighting matrix for the measurement function.
            file_name_without_ext (str): Base name for the generated Python files (without extension).

        Returns:
            tuple:
                - (function, str): Tuple containing the imported function
                  for fxu_jacobian_X and its file name.
                - (function, str): Tuple containing the imported function
                  for fxu_jacobian_U and its file name.
                - (function, str): Tuple containing the imported function
                  for hx_jacobian and its file name.
                - sp.Matrix: Weighted measurement Jacobian matrix.
        """

        file_name_without_ext_to_write = f"{file_name_without_ext}_adaptive_mpc"

        fxu_jacobian_X_file_name = ExpressionDeploy.write_state_function_code_from_sympy(
            fxu_jacobian_X, X, U, file_name_without_ext_to_write)

        fxu_jacobian_U_file_name = ExpressionDeploy.write_state_function_code_from_sympy(
            fxu_jacobian_U, X, U, file_name_without_ext_to_write)

        hx_jacobian = sp.Matrix(np.diag(Weight_Y)) * hx_jacobian

        hx_jacobian_file_name = ExpressionDeploy.write_measurement_function_code_from_sympy(
            hx_jacobian, X, file_name_without_ext_to_write)

        local_vars = {}

        exec(f"from {fxu_jacobian_X_file_name} import function as fxu_jacobian_X_script_function",
             globals(), local_vars)
        exec(f"from {fxu_jacobian_U_file_name} import function as fxu_jacobian_U_script_function",
             globals(), local_vars)
        exec(f"from {hx_jacobian_file_name} import function as hx_jacobian_script_function",
             globals(), local_vars)

        fxu_jacobian_X_script_function = local_vars["fxu_jacobian_X_script_function"]
        fxu_jacobian_U_script_function = local_vars["fxu_jacobian_U_script_function"]
        hx_jacobian_script_function = local_vars["hx_jacobian_script_function"]

        return (fxu_jacobian_X_script_function, fxu_jacobian_X_file_name), \
            (fxu_jacobian_U_script_function, fxu_jacobian_U_file_name), \
            (hx_jacobian_script_function, hx_jacobian_file_name), \
            hx_jacobian

    def _generate_state_space_embedded_integrator(
            self,
            fxu_jacobian_X: sp.Matrix,
            fxu_jacobian_U: sp.Matrix,
            hx_jacobian: sp.Matrix
    ) -> StateSpaceEmbeddedIntegrator:
        """
        Generates an augmented state-space model with an embedded integrator.

        This method constructs a symbolic state-space representation using the provided Jacobian matrices
        for the system dynamics and output equations. It then augments the state-space model with an embedded
        integrator and performs dimension checks to ensure consistency with the expected augmented input and output sizes.

        Args:
            fxu_jacobian_X (sp.Matrix): Jacobian matrix of the system dynamics with respect to the state variables.
            fxu_jacobian_U (sp.Matrix): Jacobian matrix of the system dynamics with respect to the input variables.
            hx_jacobian (sp.Matrix): Jacobian matrix of the output equation with respect to the state variables.

        Returns:
            StateSpaceEmbeddedIntegrator: Augmented state-space model with embedded integrator.

        Raises:
            ValueError: If the dimensions of the augmented input or output do not match the expected sizes.
        """

        A = fxu_jacobian_X
        B = fxu_jacobian_U
        C = hx_jacobian

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
        """
        Updates and returns the weight matrix for the MPC controller.

        Parameters
        ----------
        Weight : np.ndarray
            A 1D array of weights to be applied to the control inputs.

        Returns
        -------
        np.ndarray
            A diagonal matrix of shape (Nc * len(Weight), Nc * len(Weight)),
            where the weights are repeated Nc times and placed along the diagonal.
        """

        return np.diag(np.tile(Weight, (self.Nc, 1)).flatten())

    def _create_prediction_matrices(self) -> MPC_PredictionMatrices:
        """
        Creates and initializes the prediction matrices
          required for adaptive Model Predictive Control (MPC).

        This method constructs an instance of `MPC_PredictionMatrices`
          using the controller's prediction and control horizons,
        as well as the augmented input, state, and output sizes.
          It then generates and assigns the adaptive Phi_F updater
        function from the state space initializer.
          The prediction matrices are updated at runtime using the latest parameters,
        symbolic variables, and state/input arrays.

        Returns:
            MPC_PredictionMatrices: An object containing the initialized
              and updated prediction matrices for adaptive MPC.
        """

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
            parameters_struct=self.kalman_filter.Parameters,
            X_ndarray=self.X_inner_model,
            U_ndarray=self.U_latest)

        return prediction_matrices

    def create_reference_trajectory(
            self,
            reference_trajectory: np.ndarray,
            Y: np.ndarray):
        """
        Creates a reference trajectory for the MPC controller by computing the difference 
        between the provided reference trajectory and the current output Y.

        Parameters
        ----------
        reference_trajectory : np.ndarray
            The desired reference trajectory, either as a single row vector or a matrix 
            with Np row vectors.
        Y : np.ndarray
            The current output vector.

        Returns
        -------
        trajectory : MPC_ReferenceTrajectory
            The processed reference trajectory object for MPC.

        Raises
        ------
        ValueError
            If the reference_trajectory does not have the correct shape (must be either 
            a single row vector or Np row vectors).
        """

        if self.is_ref_trajectory:
            if not ((reference_trajectory.shape[1] == self.Np) or
                    (reference_trajectory.shape[1] == 1)):
                raise ValueError(
                    "Reference vector must be either a single row vector or a Np row vectors.")

        reference_trajectory_dif = np.zeros_like(reference_trajectory)
        for i in range(reference_trajectory.shape[0]):
            for j in range(reference_trajectory.shape[1]):
                reference_trajectory_dif[i, j] = \
                    reference_trajectory[i, j] - Y[i, 0]

        trajectory = MPC_ReferenceTrajectory(reference_trajectory_dif, self.Np)

        return trajectory

    def update_solver_factor(self, Phi: np.ndarray, Weight_U_Nc: np.ndarray):
        """
        Updates the solver factor used in the MPC optimization
          by solving a linear system.
        Parameters
        ----------
        Phi : np.ndarray
            The regressor matrix, typically of shape (N, Nc),
              where N is the number of samples and Nc is the control horizon.
        Weight_U_Nc : np.ndarray
            The weighting matrix for the control input,
              expected to be square and of shape (Nc, Nc).
        Raises
        ------
        ValueError
            If the dimensions of `Phi` and `Weight_U_Nc` are not compatible.
        Notes
        -----
        The solver factor is computed as the solution to the linear system:
            (Phi.T @ Phi + Weight_U_Nc) * X = Phi.T
        and stored in `self.solver_factor`.
        """

        if (Phi.shape[1] != Weight_U_Nc.shape[0]) or (Phi.shape[1] != Weight_U_Nc.shape[1]):
            raise ValueError("Weight must have compatible dimensions.")

        self.solver_factor = np.linalg.solve(Phi.T @ Phi + Weight_U_Nc, Phi.T)

    def solve(self, reference_trajectory: MPC_ReferenceTrajectory,
              X_augmented: np.ndarray):
        """
        Solves for the optimal control input increments (delta_U)
          based on the provided reference trajectory and augmented state.
        Args:
            reference_trajectory (MPC_ReferenceTrajectory): The reference trajectory
              object containing desired future states.
            X_augmented (np.ndarray): The current augmented state vector.
        Returns:
            np.ndarray: The computed optimal control input increments (delta_U).
        """

        # (Phi^T * Phi + Weight)^-1 * Phi^T * (Trajectory - Fx)
        delta_U = self.solver_factor @ reference_trajectory.calculate_dif(
            self.prediction_matrices.F_ndarray @ X_augmented)

        return delta_U

    def calculate_this_U(self, U, delta_U):
        """
        Updates the input matrix U by adding the corresponding elements from delta_U.
        Parameters:
            U (numpy.ndarray): The current input matrix.
            delta_U (numpy.ndarray): The change to be applied to the input matrix.
        Returns:
            numpy.ndarray: The updated input matrix after applying delta_U.
        Notes:
            Only the first AUGMENTED_INPUT_SIZE rows of delta_U are added to U.
        """

        U = U + \
            delta_U[:self.AUGMENTED_INPUT_SIZE, :]

        return U

    def compensate_X_Y_delay(self, X: np.ndarray, Y: np.ndarray):
        """
        Compensates for delay in the X and Y signals using a Kalman filter and stored Y values.
        Parameters
        ----------
        X : np.ndarray
            The current state estimate.
        Y : np.ndarray
            The current output measurement.
        Returns
        -------
        X : np.ndarray
            The compensated state estimate.
        Y : np.ndarray
            The compensated output measurement, adjusted for delay if applicable.
        Notes
        -----
        If `Number_of_Delay` is greater than zero, the function uses the Kalman filter to estimate
        the state and output without delay, stores the output, and compensates for the delay by
        adding the difference between the measured and stored output. Otherwise, it simply stores
        the output and returns the original state and output.
        """

        if self.Number_of_Delay > 0:
            Y_measured = Y

            X = self.kalman_filter.get_x_hat_without_delay()
            Y = self.kalman_filter.C @ X

            self.Y_store.push(Y)
            Y_diff = Y_measured - self.Y_store.get()

            return X, (Y + Y_diff)
        else:
            # This is for calculating y difference
            self.Y_store.push(Y)

            return X, Y

    def update_parameters(self, parameters_struct):
        """
        Updates the parameters of the Kalman filter with the provided parameters structure.

        Args:
            parameters_struct: A structure or object containing the parameters to update
                the Kalman filter with. The expected format should match the requirements
                of the Kalman filter's Parameters attribute.

        Returns:
            None
        """

        self.kalman_filter.Parameters = parameters_struct

    def update_manipulation(self, reference: np.ndarray, Y: np.ndarray):
        """
        Updates the control manipulation based on the provided reference trajectory and measured output.

        This method performs the following steps:
        1. Updates the Kalman filter with the latest control input and measured output.
        2. Compensates for delays in the state and output.
        3. Updates the adaptive prediction matrices using the current state and control input.
        4. Updates the solver factorization for the MPC optimization problem.
        5. Computes the augmented state vector using the compensated state and output.
        6. Generates the reference trajectory for the MPC.
        7. Solves the MPC optimization problem to obtain the control increment.
        8. Updates the latest control input and internal model state.

        Args:
            reference (np.ndarray): The desired reference trajectory for the output.
            Y (np.ndarray): The current measured output.

        Returns:
            np.ndarray: The updated control input to be applied.
        """

        self.kalman_filter.predict_and_update(
            self.U_latest, Y)
        X = self.kalman_filter.x_hat
        X_compensated, Y_compensated = self.compensate_X_Y_delay(X, Y)

        self.prediction_matrices.update_Phi_F_adaptive_runtime(
            parameters_struct=self.kalman_filter.Parameters,
            X_ndarray=X_compensated,
            U_ndarray=self.U_latest)

        self.update_solver_factor(
            self.prediction_matrices.Phi_ndarray, self.Weight_U_Nc)

        delta_X = X_compensated - self.X_inner_model
        delta_Y = Y_compensated - self.Y_store.get()
        X_augmented = np.vstack((delta_X, delta_Y))

        reference_trajectory = self.create_reference_trajectory(
            reference, self.Y_store.get())

        delta_U = self.solve(reference_trajectory, X_augmented)

        self.U_latest = self.calculate_this_U(self.U_latest, delta_U)

        self.X_inner_model = X_compensated

        return self.U_latest
