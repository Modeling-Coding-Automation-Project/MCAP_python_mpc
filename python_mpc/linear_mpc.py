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
import numpy as np

from mpc_utility.state_space_utility import SymbolicStateSpace
from mpc_utility.state_space_utility import StateSpaceEmbeddedIntegrator
from mpc_utility.state_space_utility import MPC_PredictionMatrices
from mpc_utility.state_space_utility import MPC_ReferenceTrajectory
from mpc_utility.linear_solver_utility import LTI_MPC_QP_Solver
from mpc_utility.linear_solver_utility import symbolic_to_numeric_matrix
from external_libraries.MCAP_python_control.python_control.kalman_filter import LinearKalmanFilter
from external_libraries.MCAP_python_control.python_control.kalman_filter import DelayedVectorObject


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

        self.prediction_matrices = self.create_prediction_matrices(Weight_Y)

        self.solver_factor = np.zeros(
            (self.AUGMENTED_INPUT_SIZE * self.Nc,
             self.AUGMENTED_OUTPUT_SIZE * self.Np))
        self.update_solver_factor(
            self.prediction_matrices.Phi_numeric, self.Weight_U_Nc)

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
        """
        Creates a reference trajectory for the MPC controller.
        Args:
            reference_trajectory (np.ndarray): The reference trajectory, which can be a single row vector or multiple row vectors.
        Returns:
            MPC_ReferenceTrajectory: An instance containing the reference trajectory.
        """
        if self.is_ref_trajectory:
            if not ((reference_trajectory.shape[1] == self.Np) or
                    (reference_trajectory.shape[1] == 1)):
                raise ValueError(
                    "Reference vector must be either a single row vector or a Np row vectors.")

        trajectory = MPC_ReferenceTrajectory(reference_trajectory, self.Np)

        return trajectory

    def update_solver_factor(self, Phi: np.ndarray, Weight_U_Nc: np.ndarray):
        """
        Updates the solver factor for the MPC controller.
        Args:
            Phi (np.ndarray): The prediction matrix Phi.
            Weight_U_Nc (np.ndarray): The weight matrix for the control input.
        Returns:
            None
        """
        if (Phi.shape[1] != Weight_U_Nc.shape[0]) or (Phi.shape[1] != Weight_U_Nc.shape[1]):
            raise ValueError("Weight must have compatible dimensions.")

        self.solver_factor = np.linalg.solve(Phi.T @ Phi + Weight_U_Nc, Phi.T)

    def solve(self, reference_trajectory: MPC_ReferenceTrajectory,
              X_augmented: np.ndarray):
        """
        Solves the MPC optimization problem to compute the control input.
        Args:
            reference_trajectory (MPC_ReferenceTrajectory): The reference trajectory for the MPC controller.
            X_augmented (np.ndarray): The augmented state vector, which includes the state and output.
        Returns:
            np.ndarray: The computed control input delta_U.
        """
        # (Phi^T * Phi + Weight)^-1 * Phi^T * (Trajectory - Fx)
        delta_U = self.solver_factor @ reference_trajectory.calculate_dif(
            self.prediction_matrices.F_numeric @ X_augmented)

        return delta_U

    def calculate_this_U(self, U, delta_U):
        """
        Calculates the new control input U based on the previous input and the computed delta_U.
        Args:
            U (np.ndarray): The previous control input.
            delta_U (np.ndarray): The computed change in control input.
        Returns:
            np.ndarray: The updated control input U.
        """
        U = U + \
            delta_U[:self.AUGMENTED_INPUT_SIZE, :]

        return U

    def compensate_X_Y_delay(self, X: np.ndarray, Y: np.ndarray):
        """
        Compensates for delays in the state and output vectors.
        Args:
            X (np.ndarray): The state vector.
            Y (np.ndarray): The output vector.
        Returns:
            tuple: A tuple containing the compensated state vector and output vector.
        """
        if self.Number_of_Delay > 0:
            Y_measured = Y

            X = self.kalman_filter.get_x_hat_without_delay()
            Y = self.kalman_filter.C @ X

            self.Y_store.push(Y)
            Y_diff = Y_measured - self.Y_store.get()

            return X, (Y + Y_diff)
        else:
            return X, Y

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

        delta_U_Nc = np.zeros((self.solver_factor.shape[0], 1))

        self.qp_solver = LTI_MPC_QP_Solver(
            number_of_variables=self.AUGMENTED_INPUT_SIZE * self.Nc,
            output_size=self.AUGMENTED_OUTPUT_SIZE,
            U=self.U_latest,
            X_augmented=np.vstack(
                (self.X_inner_model, self.Y_store.get())),
            Phi=self.prediction_matrices.Phi_numeric,
            F=self.prediction_matrices.F_numeric,
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
            Phi=self.prediction_matrices.Phi_numeric,
            F=self.prediction_matrices.F_numeric)

        delta_U = self.qp_solver.solve(
            Phi=self.prediction_matrices.Phi_numeric,
            F=self.prediction_matrices.F_numeric,
            reference_trajectory=reference_trajectory,
            X_augmented=X_augmented)

        return delta_U
