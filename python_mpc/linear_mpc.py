"""
File: linear_mpc.py

This module implements Linear Model Predictive Control (MPC) algorithms
for discrete-time linear time-invariant (LTI) systems,
with and without constraints.
It provides classes for unconstrained MPC (LTI_MPC_NoConstraints)
and constrained MPC (LTI_MPC),
supporting state estimation via a Kalman filter,
reference trajectory tracking,
and quadratic programming-based constraint handling.
"""
import os
import sys
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_optimization'))

import inspect
import numpy as np
import sympy as sp
from dataclasses import is_dataclass

from mpc_utility.state_space_utility import SymbolicStateSpace
from mpc_utility.state_space_utility import StateSpaceEmbeddedIntegrator
from mpc_utility.state_space_utility import MPC_PredictionMatrices
from mpc_utility.state_space_utility import MPC_ReferenceTrajectory
from mpc_utility.state_space_utility_deploy import LTV_MPC_StateSpaceInitializer
from mpc_utility.linear_solver_utility import LMPC_QP_Solver
from mpc_utility.linear_solver_utility import symbolic_to_numeric_matrix
from mpc_utility.linear_solver_utility import create_sparse_available
from external_libraries.MCAP_python_control.python_control.kalman_filter import LinearKalmanFilter
from external_libraries.MCAP_python_control.python_control.kalman_filter import DelayedVectorObject

USE_QR_DECOMPOSITION_FOR_SOLVER_FACTOR = True

# Common Functions


def create_reference_trajectory(is_ref_trajectory: bool, Np: int,
                                reference_trajectory: np.ndarray):
    """
    Creates a reference trajectory object for Model Predictive Control (MPC).
    Parameters:
        is_ref_trajectory (bool): Flag indicating whether a reference trajectory is provided.
        Np (int): Prediction horizon length.
        reference_trajectory (np.ndarray): The reference trajectory array.
            Should have either one column (single reference) or Np columns (reference for each step in the horizon).
    Returns:
        MPC_ReferenceTrajectory: An object representing the reference trajectory for the MPC controller.
    Raises:
        ValueError: If is_ref_trajectory is True and reference_trajectory does not have either 1 or Np columns.
    """
    if is_ref_trajectory:
        if not ((reference_trajectory.shape[1] == Np) or
                (reference_trajectory.shape[1] == 1)):
            raise ValueError(
                "Reference vector must be either a single row vector or a Np row vectors.")

    trajectory = MPC_ReferenceTrajectory(reference_trajectory, Np)

    return trajectory


def update_solver_factor(Phi: np.ndarray, Weight_U_Nc: np.ndarray):
    """
    Updates the solver factor matrix for a linear MPC problem
      using QR decomposition for improved numerical stability.

    This function computes a solver factor that can be used to
      efficiently solve least-squares problems of the form:
        min ||Phi * x - y||^2 + x.T * Weight_U_Nc * x
    by augmenting the system and applying QR decomposition.

    Args:
        Phi (np.ndarray): The prediction matrix of shape (N, M).
        Weight_U_Nc (np.ndarray): The control weighting matrix of shape (M, M).

    Returns:
        np.ndarray: The computed solver factor matrix.

    Raises:
        ValueError: If the dimensions of `Phi` and `Weight_U_Nc` are not compatible.
    """

    if (Phi.shape[1] != Weight_U_Nc.shape[0]) or (Phi.shape[1] != Weight_U_Nc.shape[1]):
        raise ValueError("Weight must have compatible dimensions.")

    if USE_QR_DECOMPOSITION_FOR_SOLVER_FACTOR:
        # solve with QR decomposition for better numerical stability
        A_augmented = np.vstack((Phi, np.sqrt(Weight_U_Nc)))
        Y_augmented = np.vstack(
            (np.eye(Phi.shape[0]), np.zeros((Phi.shape[1], Phi.shape[0]))))

        Q, R = np.linalg.qr(A_augmented)

        solver_factor = np.linalg.solve(R, Q.T @ Y_augmented)
    else:
        solver_factor = np.linalg.solve(Phi.T @ Phi + Weight_U_Nc, Phi.T)

    return solver_factor


def update_solver_factor_SparseAvailable(
        Phi_SparseAvailable: sp.SparseMatrix):
    """
    Updates the solver factor for a sparse matrix representation.

    This function computes an intermediate matrix by multiplying the inverse of the product
    of the transpose of the input sparse matrix and itself with the transpose of the input matrix.
    It then creates and returns a solver factor using this intermediate result.

    Args:
        Phi_SparseAvailable (sp.SparseMatrix): The input sparse matrix
          for which the solver factor is to be updated.

    Returns:
        solver_factor_SparseAvailable: The updated solver factor
          based on the input sparse matrix.

    Note:
        - The function assumes the existence of `create_sparse_available`
          and that `sp` provides the necessary sparse matrix operations.
        - The implementation currently uses a placeholder for the inverse
          calculation (`sp.ones`),
            which may need to be replaced with the actual inverse computation.
    """

    Phi_T_Phi_inv = sp.ones(
        Phi_SparseAvailable.shape[1], Phi_SparseAvailable.shape[1])

    solver_factor_SparseAvailable = create_sparse_available(
        Phi_T_Phi_inv * Phi_SparseAvailable.T)

    return solver_factor_SparseAvailable


def solve_LMPC_No_Constraints(solver_factor: np.ndarray, F_ndarray: np.ndarray,
                              reference_trajectory: MPC_ReferenceTrajectory,
                              X_augmented: np.ndarray):
    """
    Solves for the optimal control input increment (delta_U)
     in a linear MPC no constraints problem.

    Parameters:
        solver_factor (np.ndarray): Precomputed solver factor,
        typically (Phi^T * Phi + Weight)^-1 * Phi^T.
        F_ndarray (np.ndarray): System prediction matrix (F).
        reference_trajectory (MPC_ReferenceTrajectory): Object providing the reference trajectory
          and a method to compute the difference from predicted trajectory.
        X_augmented (np.ndarray): Current augmented state vector.

    Returns:
        np.ndarray: Optimal control input increment (delta_U) for the current time step.
    """
    # (Phi^T * Phi + Weight)^-1 * Phi^T * (Trajectory - Fx)
    delta_U = solver_factor @ reference_trajectory.calculate_dif(
        F_ndarray @ X_augmented)

    return delta_U


def calculate_this_U(AUGMENTED_INPUT_SIZE: int, U: np.ndarray, delta_U: np.ndarray):
    """
    Updates the input matrix U by adding the first AUGMENTED_INPUT_SIZE rows of delta_U.

    Args:
        AUGMENTED_INPUT_SIZE (int): The number of rows from delta_U to add to U.
        U (np.ndarray): The current input matrix to be updated.
        delta_U (np.ndarray): The matrix containing input increments.

    Returns:
        np.ndarray: The updated input matrix U.
    """
    U = U + \
        delta_U[:AUGMENTED_INPUT_SIZE, :]

    return U


def compensate_X_Y_delay(kalman_filter: LinearKalmanFilter,
                         Number_of_Delay: int,
                         Y_store: DelayedVectorObject,
                         X_in: np.ndarray, Y_in: np.ndarray):
    """
    Compensates for measurement delay in X and Y signals
      using a Kalman filter and a delayed vector store.

    Args:
        kalman_filter (LinearKalmanFilter): The Kalman filter object used to estimate the state.
        Number_of_Delay (int): The number of delay steps to compensate for.
        Y_store (DelayedVectorObject): An object that stores delayed Y vectors for compensation.
        X_in (np.ndarray): The input state vector without delay compensation.
        Y_in (np.ndarray): The input output vector without delay compensation.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - The compensated state vector (X).
            - The compensated output vector (Y).
    """
    if Number_of_Delay > 0:
        Y_measured = Y_in

        X = kalman_filter.get_x_hat_without_delay()
        Y = kalman_filter.C @ X

        Y_store.push(Y)
        Y_diff = Y_measured - Y_store.get()

        return X, (Y + Y_diff)
    else:
        return X_in, Y_in


class LTI_MPC_NoConstraints:
    """
    Linear Model Predictive Control (MPC) without constraints.
    This class implements a linear MPC controller for discrete-time LTI systems.
    It uses a Kalman filter for state estimation and allows for reference trajectory tracking.
    """

    def __init__(self, state_space: SymbolicStateSpace, Np: int, Nc: int,
                 Weight_U: np.ndarray, Weight_Y: np.ndarray,
                 Q_kf: np.ndarray = None, R_kf: np.ndarray = None,
                 is_ref_trajectory=False):
        # Check compatibility
        if state_space.delta_time <= 0.0:
            raise ValueError("State space model must be discrete-time.")

        self.Number_of_Delay = state_space.Number_of_Delay

        if (Np < self.Number_of_Delay):
            raise ValueError(
                "Prediction horizon Np must be greater than the number of delays.")

        self.kalman_filter = self.initialize_kalman_filter(
            state_space, Q_kf, R_kf)

        self.augmented_ss = StateSpaceEmbeddedIntegrator(state_space)

        self.AUGMENTED_STATE_SIZE = self.augmented_ss.A.shape[0]

        self.AUGMENTED_INPUT_SIZE = self.augmented_ss.B.shape[1]
        if self.AUGMENTED_INPUT_SIZE != state_space.B.shape[1]:
            raise ValueError(
                "the augmented state space input must have the same size of state_space.B.")
        self.AUGMENTED_OUTPUT_SIZE = self.augmented_ss.C.shape[0]
        if self.AUGMENTED_OUTPUT_SIZE != state_space.C.shape[0]:
            raise ValueError(
                "the augmented state space output must have the same size of state_space.C.")

        self.X_inner_model = np.zeros(
            (state_space.A.shape[0], 1))
        self.U_latest = np.zeros(
            (self.AUGMENTED_INPUT_SIZE, 1))

        if Nc > Np:
            raise ValueError("Nc must be less than or equal to Np.")
        self.Np = Np
        self.Nc = Nc

        self.Weight_U_Nc = self.update_weight(Weight_U)

        self.prediction_matrices: MPC_PredictionMatrices \
            = self.create_prediction_matrices(
                Weight_Y)

        self.solver_factor = np.zeros(
            (self.AUGMENTED_INPUT_SIZE * self.Nc,
             self.AUGMENTED_OUTPUT_SIZE * self.Np))
        self.solver_factor_SparseAvailable = sp.zeros(
            self.AUGMENTED_INPUT_SIZE * self.Nc,
            self.AUGMENTED_OUTPUT_SIZE * self.Np)

        self.update_solver_factor(
            self.prediction_matrices.Phi_ndarray, self.Weight_U_Nc)
        self.update_solver_factor_SparseAvailable(
            self.prediction_matrices.Phi_SparseAvailable)

        self.Y_store = DelayedVectorObject(self.AUGMENTED_OUTPUT_SIZE,
                                           self.Number_of_Delay)

        self.is_ref_trajectory = is_ref_trajectory

    def initialize_kalman_filter(self, state_space: SymbolicStateSpace,
                                 Q_kf: np.ndarray, R_kf: np.ndarray) -> LinearKalmanFilter:
        """
        Initializes the Kalman filter for state estimation.
        Args:
            state_space (SymbolicStateSpace): The symbolic state space model.
            Q_kf (np.ndarray): Process noise covariance matrix.
            R_kf (np.ndarray): Measurement noise covariance matrix.
        Returns:
            LinearKalmanFilter: An instance of the Kalman filter.
        """
        if Q_kf is None:
            Q_kf = np.eye(state_space.A.shape[0])
        if R_kf is None:
            R_kf = np.eye(state_space.C.shape[0])

        lkf = LinearKalmanFilter(
            A=symbolic_to_numeric_matrix(state_space.A),
            B=symbolic_to_numeric_matrix(state_space.B),
            C=symbolic_to_numeric_matrix(state_space.C),
            Q=Q_kf, R=R_kf,
            Number_of_Delay=state_space.Number_of_Delay)

        lkf.converge_G()

        return lkf

    def update_weight(self, Weight: np.ndarray):
        """
        Updates the weight matrix for the control input.
        Args:
            Weight (np.ndarray): The weight matrix for the control input.
        Returns:
            np.ndarray: A diagonal matrix with the weight values repeated for each control input.
        """
        return np.diag(np.tile(Weight, (self.Nc, 1)).flatten())

    def create_prediction_matrices(self, Weight_Y: np.ndarray) -> MPC_PredictionMatrices:
        """
        Creates the prediction matrices for the MPC controller.
        Args:
            Weight_Y (np.ndarray): The weight matrix for the output.
        Returns:
            MPC_PredictionMatrices: An instance containing the prediction matrices.
        """
        prediction_matrices = MPC_PredictionMatrices(
            Np=self.Np,
            Nc=self.Nc,
            INPUT_SIZE=self.AUGMENTED_INPUT_SIZE,
            STATE_SIZE=self.AUGMENTED_STATE_SIZE,
            OUTPUT_SIZE=self.AUGMENTED_OUTPUT_SIZE)

        if (0 != len(self.augmented_ss.A.free_symbols)) or \
                (0 != len(self.augmented_ss.B.free_symbols)) or \
                (0 != len(self.augmented_ss.C.free_symbols)):
            raise ValueError("State space model must be numeric.")

        prediction_matrices.substitute_numeric(
            self.augmented_ss.A, self.augmented_ss.B, Weight_Y * self.augmented_ss.C)

        return prediction_matrices

    def create_reference_trajectory(self, reference_trajectory: np.ndarray):
        return create_reference_trajectory(
            self.is_ref_trajectory, self.Np, reference_trajectory)

    def update_solver_factor(self, Phi: np.ndarray, Weight_U_Nc: np.ndarray):
        self.solver_factor = update_solver_factor(
            Phi, Weight_U_Nc)

    def update_solver_factor_SparseAvailable(
            self,
            Phi_SparseAvailable: sp.Matrix):
        self.solver_factor_SparseAvailable = update_solver_factor_SparseAvailable(
            Phi_SparseAvailable)

    def solve(self, reference_trajectory: MPC_ReferenceTrajectory,
              X_augmented: np.ndarray):
        return solve_LMPC_No_Constraints(
            self.solver_factor, self.prediction_matrices.F_ndarray,
            reference_trajectory, X_augmented)

    def calculate_this_U(self, U, delta_U):
        return calculate_this_U(
            self.AUGMENTED_INPUT_SIZE, U, delta_U)

    def compensate_X_Y_delay(self, X: np.ndarray, Y: np.ndarray):
        return compensate_X_Y_delay(
            self.kalman_filter, self.Number_of_Delay,
            self.Y_store, X, Y)

    def update(self, reference: np.ndarray, Y: np.ndarray):
        """
        Updates the MPC controller with the latest reference and output measurements.
        Args:
            reference (np.ndarray): The reference trajectory, which can be a single row vector or multiple row vectors.
            Y (np.ndarray): The output measurement vector.
        Returns:
            np.ndarray: The updated control input U.
        """
        self.kalman_filter.predict_and_update_with_fixed_G(
            self.U_latest, Y)
        X = self.kalman_filter.x_hat
        X_compensated, Y_compensated = self.compensate_X_Y_delay(X, Y)

        delta_X = X_compensated - self.X_inner_model
        X_augmented = np.vstack((delta_X, Y_compensated))

        reference_trajectory = self.create_reference_trajectory(reference)

        delta_U = self.solve(reference_trajectory, X_augmented)

        self.U_latest = self.calculate_this_U(self.U_latest, delta_U)

        self.X_inner_model = X_compensated

        return self.U_latest


class LTI_MPC(LTI_MPC_NoConstraints):
    """
    Linear Model Predictive Control (MPC) with constraints.
    This class extends the LTI_MPC_NoConstraints class to include constraints on the control input and output.
    It uses a quadratic programming solver to handle the constraints during the optimization process.
    """

    def __init__(self, state_space: SymbolicStateSpace, Np: int, Nc: int,
                 Weight_U: np.ndarray, Weight_Y: np.ndarray,
                 Q_kf: np.ndarray = None, R_kf: np.ndarray = None,
                 is_ref_trajectory: bool = False,
                 delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                 U_min: np.ndarray = None, U_max: np.ndarray = None,
                 Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        super().__init__(state_space, Np, Nc, Weight_U, Weight_Y,
                         Q_kf, R_kf, is_ref_trajectory)

        delta_U_Nc = np.zeros((self.AUGMENTED_INPUT_SIZE * self.Nc, 1))

        self.qp_solver = LMPC_QP_Solver(
            number_of_variables=self.AUGMENTED_INPUT_SIZE * self.Nc,
            output_size=self.AUGMENTED_OUTPUT_SIZE,
            U=self.U_latest,
            X_augmented=np.vstack(
                (self.X_inner_model, self.Y_store.get())),
            Phi=self.prediction_matrices.Phi_ndarray,
            F=self.prediction_matrices.F_ndarray,
            Weight_U_Nc=self.Weight_U_Nc,
            delta_U_Nc=delta_U_Nc,
            delta_U_min=delta_U_min, delta_U_max=delta_U_max,
            U_min=U_min, U_max=U_max,
            Y_min=Y_min, Y_max=Y_max)

    def solve(self, reference_trajectory: MPC_ReferenceTrajectory,
              X_augmented: np.ndarray):
        """
        Solves the MPC optimization problem with constraints to compute the control input.
        Args:
            reference_trajectory (MPC_ReferenceTrajectory): The reference trajectory for the MPC controller.
            X_augmented (np.ndarray): The augmented state vector, which includes the state and output.
        Returns:
            np.ndarray: The computed control input delta_U.
        """
        self.qp_solver.update_constraints(
            U=self.U_latest,
            X_augmented=X_augmented,
            Phi=self.prediction_matrices.Phi_ndarray,
            F=self.prediction_matrices.F_ndarray)

        delta_U = self.qp_solver.solve(
            Phi=self.prediction_matrices.Phi_ndarray,
            F=self.prediction_matrices.F_ndarray,
            reference_trajectory=reference_trajectory,
            X_augmented=X_augmented)

        return delta_U


class LTV_MPC_NoConstraints:
    """
    LTV_MPC_NoConstraints implements a Linear Time-Varying
      Model Predictive Controller (MPC) without explicit input/output constraints.
    This controller is designed for discrete-time,
    symbolic state-space models and supports runtime parameter updates,
    Kalman filtering for state estimation,
    and embedded integrator augmentation for offset-free tracking.
    It is suitable for applications requiring predictive control
    with time-varying system dynamics and no hard constraints
    on control actions or outputs.

    Usage:
        Instantiate with a symbolic state-space model, parameter dataclass,
          prediction and control horizons, and weighting matrices.
        Call update_manipulation(reference, Y) at each control step
          to obtain the next control input.
    """

    def __init__(self, state_space: SymbolicStateSpace,
                 parameters_struct,
                 Np: int, Nc: int,
                 Weight_U: np.ndarray, Weight_Y: np.ndarray,
                 Q_kf: np.ndarray = None, R_kf: np.ndarray = None,
                 is_ref_trajectory: bool = False,
                 caller_file_name: str = None):

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
        if state_space.delta_time <= 0.0:
            raise ValueError("State space model must be discrete-time.")

        self.Number_of_Delay = state_space.Number_of_Delay

        if (Np < self.Number_of_Delay):
            raise ValueError(
                "Prediction horizon Np must be greater than the number of delays.")

        if not is_dataclass(parameters_struct):
            raise ValueError(
                "parameters_struct must be a dataclass instance.")

        self.parameters_struct = parameters_struct

        self.state_space_initializer = LTV_MPC_StateSpaceInitializer(
            caller_file_name_without_ext)

        self.kalman_filter = self.initialize_kalman_filter(
            state_space, self.parameters_struct, Q_kf, R_kf)

        self.augmented_ss = StateSpaceEmbeddedIntegrator(state_space)
        self.augmented_ss.C = sp.Matrix(Weight_Y) * self.augmented_ss.C

        self.AUGMENTED_STATE_SIZE = self.augmented_ss.A.shape[0]

        self.AUGMENTED_INPUT_SIZE = self.augmented_ss.B.shape[1]
        if self.AUGMENTED_INPUT_SIZE != state_space.B.shape[1]:
            raise ValueError(
                "the augmented state space input must have the same size of state_space.B.")
        self.AUGMENTED_OUTPUT_SIZE = self.augmented_ss.C.shape[0]
        if self.AUGMENTED_OUTPUT_SIZE != state_space.C.shape[0]:
            raise ValueError(
                "the augmented state space output must have the same size of state_space.C.")

        self.state_space_initializer.generate_initial_embedded_integrator(
            parameters_struct=self.parameters_struct,
            state_space=self.augmented_ss)

        self.X_inner_model = np.zeros(
            (state_space.A.shape[0], 1))
        self.U_latest = np.zeros(
            (self.AUGMENTED_INPUT_SIZE, 1))

        if Nc > Np:
            raise ValueError("Nc must be less than or equal to Np.")
        self.Np = Np
        self.Nc = Nc

        self.Weight_U_Nc = self.update_weight(Weight_U)

        self.state_space_initializer.generate_prediction_matrices_phi_f(
            Np=Np,
            Nc=Nc,
            state_space=self.augmented_ss)

        self.prediction_matrices = self.create_prediction_matrices()

        self.solver_factor = np.zeros(
            (self.AUGMENTED_INPUT_SIZE * self.Nc,
             self.AUGMENTED_OUTPUT_SIZE * self.Np))
        self.solver_factor_SparseAvailable = sp.zeros(
            self.AUGMENTED_INPUT_SIZE * self.Nc,
            self.AUGMENTED_OUTPUT_SIZE * self.Np)

        self.update_solver_factor(
            self.prediction_matrices.Phi_ndarray, self.Weight_U_Nc)
        self.update_solver_factor_SparseAvailable(
            self.prediction_matrices.Phi_SparseAvailable)

        self.Y_store = DelayedVectorObject(self.AUGMENTED_OUTPUT_SIZE,
                                           self.Number_of_Delay)

        self.is_ref_trajectory = is_ref_trajectory

    def initialize_kalman_filter(self, state_space: SymbolicStateSpace,
                                 parameters_struct,
                                 Q_kf: np.ndarray, R_kf: np.ndarray) -> LinearKalmanFilter:
        """
        Initializes and returns a LinearKalmanFilter object
          using the provided symbolic state space, parameters, and noise covariances.
        Args:
            state_space (SymbolicStateSpace): The symbolic state-space representation
              containing system matrices (A, B, C) and delay information.
            parameters_struct: Structure containing parameters required for
              initializing the state-space matrices.
            Q_kf (np.ndarray): Process noise covariance matrix. If None,
              an identity matrix of appropriate size is used.
            R_kf (np.ndarray): Measurement noise covariance matrix. If None,
              an identity matrix of appropriate size is used.
        Returns:
            LinearKalmanFilter: An initialized and converged
            Kalman filter object for the given system.
        """
        if Q_kf is None:
            Q_kf = np.eye(state_space.A.shape[0])
        if R_kf is None:
            R_kf = np.eye(state_space.C.shape[0])

        A, B, C, _ = self.state_space_initializer.get_generate_initial_MPC_StateSpace(
            parameters_struct, state_space.A, state_space.B, state_space.C)

        lkf = LinearKalmanFilter(
            A=A,
            B=B,
            C=C,
            Q=Q_kf, R=R_kf,
            Number_of_Delay=state_space.Number_of_Delay)

        lkf.converge_G()

        return lkf

    def update_weight(self, Weight: np.ndarray):

        return np.diag(np.tile(Weight, (self.Nc, 1)).flatten())

    def create_prediction_matrices(self) -> MPC_PredictionMatrices:
        """
        Creates and initializes the prediction matrices
          required for Model Predictive Control (MPC).

        This method constructs an instance of `MPC_PredictionMatrices`
          using the controller's prediction and control horizons,
        as well as the sizes of the augmented input, state,
          and output vectors. It verifies that the augmented state-space
        matrices (A, B, C) are symbolic, raising a ValueError if they are not.
          The method then generates the updater function
        for the time-varying prediction matrices (Phi and F)
          and assigns it to the prediction matrices object. Finally, it
        updates the Phi and F matrices at runtime
          using the provided parameters structure.

        Returns:
            MPC_PredictionMatrices: An initialized prediction matrices
              object with updated Phi and F matrices.

        Raises:
            ValueError: If the augmented state-space matrices are not symbolic.
        """

        prediction_matrices = MPC_PredictionMatrices(
            Np=self.Np,
            Nc=self.Nc,
            INPUT_SIZE=self.AUGMENTED_INPUT_SIZE,
            STATE_SIZE=self.AUGMENTED_STATE_SIZE,
            OUTPUT_SIZE=self.AUGMENTED_OUTPUT_SIZE)

        if (0 == len(self.augmented_ss.A.free_symbols)) and \
                (0 == len(self.augmented_ss.B.free_symbols)) and \
                (0 == len(self.augmented_ss.C.free_symbols)):
            raise ValueError("State space model must be symbolic.")

        self.state_space_initializer.generate_LTV_MPC_Phi_F_Updater()

        prediction_matrices.Phi_F_updater_function = \
            self.state_space_initializer.LTV_MPC_Phi_F_updater_function

        prediction_matrices.create_build_SparseAvailable(
            self.augmented_ss.A,
            self.augmented_ss.B,
            self.augmented_ss.C)

        prediction_matrices.update_Phi_F_runtime(
            parameters_struct=self.parameters_struct)

        return prediction_matrices

    def create_reference_trajectory(self, reference_trajectory: np.ndarray):
        return create_reference_trajectory(
            self.is_ref_trajectory, self.Np, reference_trajectory)

    def update_solver_factor(self, Phi: np.ndarray, Weight_U_Nc: np.ndarray):
        self.solver_factor = update_solver_factor(
            Phi, Weight_U_Nc)

    def update_solver_factor_SparseAvailable(
            self,
            Phi_SparseAvailable: sp.Matrix):
        self.solver_factor_SparseAvailable = update_solver_factor_SparseAvailable(
            Phi_SparseAvailable)

    def solve(self, reference_trajectory: MPC_ReferenceTrajectory,
              X_augmented: np.ndarray):
        return solve_LMPC_No_Constraints(
            self.solver_factor, self.prediction_matrices.F_ndarray,
            reference_trajectory, X_augmented)

    def calculate_this_U(self, U, delta_U):
        return calculate_this_U(
            self.AUGMENTED_INPUT_SIZE, U, delta_U)

    def compensate_X_Y_delay(self, X: np.ndarray, Y: np.ndarray):
        return compensate_X_Y_delay(
            kalman_filter=self.kalman_filter,
            Number_of_Delay=self.Number_of_Delay,
            Y_store=self.Y_store,
            X_in=X,
            Y_in=Y)

    def update_parameters(self, parameters_struct):
        """
        Updates the internal parameters of the MPC controller at runtime.

        This method updates the Kalman filter's state-space matrices (A, B, C),
        the prediction matrices, and the solver factorization based on the provided
        parameters structure.

        Args:
            parameters_struct: An object or dictionary containing the updated
                parameters required for the state-space model and prediction matrices.

        Side Effects:
            - Modifies the Kalman filter's A, B, and C matrices.
            - Updates the prediction matrices used for MPC.
            - Recomputes the solver factorization with the new prediction matrices
              and control weights.
        """
        self.kalman_filter.A, \
            self.kalman_filter.B, \
            self.kalman_filter.C, _ = \
            self.state_space_initializer.update_mpc_state_space_runtime(
                parameters_struct)

        self.prediction_matrices.update_Phi_F_runtime(
            parameters_struct)

        self.update_solver_factor(
            self.prediction_matrices.Phi_ndarray, self.Weight_U_Nc)

    def update_manipulation(self, reference: np.ndarray, Y: np.ndarray):
        """
        Updates the control manipulation input based on the current reference and measured output.

        This method performs the following steps:
        1. Updates the internal Kalman filter state using the latest control input and measured output.
        2. Compensates for any delays in the state and output.
        3. Computes the difference between the compensated state and the internal model state.
        4. Augments the state with the compensated output.
        5. Generates a reference trajectory for the controller.
        6. Solves the control optimization problem to obtain the change in control input.
        7. Updates the latest control input and the internal model state.

        Args:
            reference (np.ndarray): The desired reference trajectory or setpoint for the system.
            Y (np.ndarray): The latest measured output from the system.

        Returns:
            np.ndarray: The updated control input to be applied to the system.
        """
        self.kalman_filter.predict_and_update(
            self.U_latest, Y)
        X = self.kalman_filter.x_hat
        X_compensated, Y_compensated = self.compensate_X_Y_delay(X, Y)

        delta_X = X_compensated - self.X_inner_model
        X_augmented = np.vstack((delta_X, Y_compensated))

        reference_trajectory = self.create_reference_trajectory(reference)

        delta_U = self.solve(reference_trajectory, X_augmented)

        self.U_latest = self.calculate_this_U(self.U_latest, delta_U)

        self.X_inner_model = X_compensated

        return self.U_latest


class LTV_MPC(LTV_MPC_NoConstraints):
    """
    Linear Time-Varying Model Predictive Controller (LTV_MPC) with constraints.

    This class extends `LTV_MPC_NoConstraints` to provide a constrained MPC implementation
    for linear time-varying systems. It supports input, output, and input increment constraints,
    and uses a quadratic programming (QP) solver for optimal control input calculation.
    """

    def __init__(self, state_space: SymbolicStateSpace,
                 parameters_struct, Np: int, Nc: int,
                 Weight_U: np.ndarray, Weight_Y: np.ndarray,
                 Q_kf: np.ndarray = None, R_kf: np.ndarray = None,
                 is_ref_trajectory: bool = False,
                 delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                 U_min: np.ndarray = None, U_max: np.ndarray = None,
                 Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        # % inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is state_space:
                variable_name = name
                break
        # Get the caller's file name
        caller_file_full_path = frame.f_code.co_filename
        caller_file_name = os.path.basename(caller_file_full_path)
        caller_file_name_without_ext = os.path.splitext(caller_file_name)[
            0]

        super().__init__(state_space, parameters_struct, Np, Nc, Weight_U, Weight_Y,
                         Q_kf, R_kf, is_ref_trajectory, caller_file_name)

        delta_U_Nc = np.zeros((self.AUGMENTED_INPUT_SIZE * self.Nc, 1))

        self.qp_solver = LMPC_QP_Solver(
            number_of_variables=self.AUGMENTED_INPUT_SIZE * self.Nc,
            output_size=self.AUGMENTED_OUTPUT_SIZE,
            U=self.U_latest,
            X_augmented=np.vstack(
                (self.X_inner_model, self.Y_store.get())),
            Phi=self.prediction_matrices.Phi_ndarray,
            F=self.prediction_matrices.F_ndarray,
            Weight_U_Nc=self.Weight_U_Nc,
            delta_U_Nc=delta_U_Nc,
            delta_U_min=delta_U_min, delta_U_max=delta_U_max,
            U_min=U_min, U_max=U_max,
            Y_min=Y_min, Y_max=Y_max)

    def solve(self, reference_trajectory: MPC_ReferenceTrajectory,
              X_augmented: np.ndarray):
        """
        Solves the linear MPC optimization problem
          for the given reference trajectory and augmented state.

        This method updates the QP solver's constraints based on
          the current control input and state,
        then solves the quadratic program to compute
          the optimal control input increment.

        Args:
            reference_trajectory (MPC_ReferenceTrajectory): The desired reference trajectory
              for the MPC to track.
            X_augmented (np.ndarray): The current augmented state vector.

        Returns:
            np.ndarray: The optimal change in control input (delta_U)
              computed by the QP solver.
        """

        self.qp_solver.update_constraints(
            U=self.U_latest,
            X_augmented=X_augmented,
            Phi=self.prediction_matrices.Phi_ndarray,
            F=self.prediction_matrices.F_ndarray)

        delta_U = self.qp_solver.solve(
            Phi=self.prediction_matrices.Phi_ndarray,
            F=self.prediction_matrices.F_ndarray,
            reference_trajectory=reference_trajectory,
            X_augmented=X_augmented)

        return delta_U
