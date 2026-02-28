"""
File: nonlinear_mpc_optimization_engine.py

NonlinearMPC_OptimizationEngine implements a nonlinear model predictive control
(NMPC) algorithm using the ALM/PM (Augmented Lagrangian Method / Penalty Method)
optimization engine, which internally uses PANOC as the inner solver.
It utilizes symbolic computation for system dynamics and measurement equations,
and employs an Extended Kalman Filter (EKF) for state estimation.

The ALM/PM optimizer is used as the unified solver for both constrained and
unconstrained problems:
- When output constraints are present (n1 > 0), the full ALM outer loop is
  activated with Lagrange multiplier updates and penalty parameter adjustment.
- When no output constraints exist (n1 = 0), the ALM outer loop reduces to a
  single PANOC solve with no augmentation overhead, since the augmented cost
  psi(u; xi) = f(u) and the exit criterion is immediately satisfied.

Unlike NonlinearMPC_TwiceDifferentiable, this implementation does not require
second-order derivative (Hessian) computations, as PANOC uses L-BFGS
quasi-Newton approximations and only needs first-order gradient information.
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

from external_libraries.MCAP_python_optimization.optimization_utility.optimization_engine_matrix_utility import OptimizationEngine_CostMatrices
from external_libraries.MCAP_python_optimization.python_optimization.panoc import (
    PANOC_Cache,
)
from external_libraries.MCAP_python_optimization.python_optimization.alm_pm_optimizer import (
    ALM_Factory,
    ALM_Problem,
    ALM_Cache,
    ALM_PM_Optimizer,
    BoxProjectionOperator,
    BallProjectionOperator,
)

NMPC_SOLVER_MAX_ITERATION_DEFAULT = 20
PANOC_TOLERANCE_DEFAULT = 1e-4
PANOC_LBFGS_MEMORY_DEFAULT = 5

ALM_MAX_OUTER_ITERATIONS_DEFAULT = 10
ALM_EPSILON_TOLERANCE_DEFAULT = 1e-4
ALM_DELTA_TOLERANCE_DEFAULT = 1e-3
ALM_INITIAL_INNER_TOLERANCE_DEFAULT = 1e-2
ALM_INITIAL_PENALTY_DEFAULT = 10.0


class SolverConfiguration:
    """
    Configuration wrapper for the PANOC/ALM optimization engine solver.

    Provides the same interface as SQP_ActiveSet_PCG_PLS for solver configuration,
    allowing drop-in replacement in user code.
    """

    def __init__(self, max_iteration: int = NMPC_SOLVER_MAX_ITERATION_DEFAULT):
        self._max_iteration = max_iteration
        self._last_iteration_count = 0

    def set_solver_max_iteration(self, max_iteration: int):
        """
        Set the maximum number of iterations for the solver.
        Parameters
        ----------
        max_iteration : int
            The maximum number of iterations the solver is allowed to perform.
        """
        self._max_iteration = max_iteration

    def get_solver_step_iterated_number(self):
        """
        Returns the number of iterations performed by the solver in the last step.
        Returns
        -------
        int
            The number of iterations performed in the last solver step.
        """
        return self._last_iteration_count


class NonlinearMPC_OptimizationEngine:
    """
    NonlinearMPC_OptimizationEngine implements a nonlinear model predictive control
    (NMPC) algorithm using the ALM/PM optimization engine (which internally uses
    PANOC as the inner solver).
    It utilizes symbolic computation for system dynamics and measurement equations,
    and employs an Extended Kalman Filter (EKF) for state estimation.
    The ALM/PM optimizer is used as the unified solver regardless of whether output
    constraints are present. When no output constraints exist, ALM reduces to a
    single PANOC solve with no augmentation overhead.

    Args:
        delta_time (float): Sampling time interval of the discrete-time system.
        X (sp.Matrix): Symbolic state vector.
        U (sp.Matrix): Symbolic input vector.
        X_initial (np.ndarray): Initial state estimate.
        fxu (sp.Matrix): Symbolic state transition function f(x, u).
        hx (sp.Matrix): Symbolic measurement function h(x).
        parameters_struct: Additional parameters required for the state
            and measurement functions.
        Np (int): Prediction horizon length.
        Weight_U (np.ndarray, optional): Weighting matrix for input
            in the cost function. Defaults to zero matrix.
        Weight_X (np.ndarray, optional): Weighting matrix for state
            in the cost function. Defaults to zero matrix.
        Weight_Y (np.ndarray, optional): Weighting matrix for output
            in the cost function. Defaults to zero matrix.
        U_min (np.ndarray, optional): Minimum input constraints.
            Defaults to None.
        U_max (np.ndarray, optional): Maximum input constraints.
            Defaults to None.
        Y_min (np.ndarray, optional): Minimum output constraints.
            Defaults to None.
        Y_max (np.ndarray, optional): Maximum output constraints.
            Defaults to None.
        Q_kf (np.ndarray, optional): Process noise covariance for EKF.
            Defaults to identity matrix.
        R_kf (np.ndarray, optional): Measurement noise covariance for EKF.
            Defaults to identity matrix.
        Number_of_Delay (int, optional): Number of delay steps in the system.
            Defaults to 0.
        caller_file_name (str, optional): Filename of the caller script
            for naming generated files. Defaults to None.
    """

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
        Weight_U: np.ndarray = None,
        Weight_X: np.ndarray = None,
        Weight_Y: np.ndarray = None,
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

        # initialize weights
        if Weight_U is None:
            Weight_U = np.zeros((self.INPUT_SIZE,))
        if Weight_X is None:
            Weight_X = np.zeros((self.STATE_SIZE,))
        if Weight_Y is None:
            Weight_Y = np.zeros((self.OUTPUT_SIZE,))

        self.cost_matrices = self.generate_cost_matrices(
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

        self.cost_matrices.state_space_parameters = parameters_struct

        self.kalman_filter, \
            (self.fxu_script_function, self.fxu_file_name), \
            (self.hx_script_function, self.hx_file_name) \
            = initialize_kalman_filter_with_EKF(
                X_initial=X_initial,
                X=X, U=U,
                fxu=fxu,
                fxu_jacobian_X=self.cost_matrices.state_jacobian_x,
                hx=hx,
                hx_jacobian=self.cost_matrices.measurement_jacobian_x,
                Q_kf=Q_kf,
                R_kf=R_kf,
                parameters_struct=parameters_struct,
                Number_of_Delay=Number_of_Delay,
                file_name_without_ext=caller_file_name_without_ext
            )

        # Setup solver configuration
        self.solver_configuration = SolverConfiguration(
            NMPC_SOLVER_MAX_ITERATION_DEFAULT)

        # Determine if output constraints are present
        self._has_output_constraints = (Y_min is not None or Y_max is not None)

        # Problem size for PANOC: nu * Np
        problem_size = self.INPUT_SIZE * self.Np

        # Create PANOC cache (inner solver for ALM)
        self._panoc_cache = PANOC_Cache(
            problem_size=problem_size,
            tolerance=PANOC_TOLERANCE_DEFAULT,
            lbfgs_memory=PANOC_LBFGS_MEMORY_DEFAULT,
        )

        # Always use ALM_PM_Optimizer as the unified solver.
        # When output constraints are present, n1 > 0 activates the full
        # ALM outer loop with Lagrange multiplier updates.
        # When no output constraints, n1 = 0 reduces ALM to a single
        # PANOC solve (psi = f, d_psi = df, exit after 1 outer iteration).
        if self._has_output_constraints:
            n1 = self.OUTPUT_SIZE * (self.Np + 1)
        else:
            n1 = 0

        self._alm_n1 = n1

        # ALM cache (wraps PANOC cache)
        self._alm_cache = ALM_Cache(self._panoc_cache, n1=n1)

        U_min_flat = self.cost_matrices.U_min_matrix.reshape((-1, 1))
        U_max_flat = self.cost_matrices.U_max_matrix.reshape((-1, 1))

        if self._has_output_constraints:
            # Output constraint box projection: [Y_min, Y_max]
            Y_min_flat = self.cost_matrices.Y_min_matrix.reshape((-1, 1))
            Y_max_flat = self.cost_matrices.Y_max_matrix.reshape((-1, 1))
            self._set_c_project = BoxProjectionOperator(
                lower=Y_min_flat, upper=Y_max_flat).project

            # Lagrange multiplier projection (ball of large radius)
            self._set_y_project = BallProjectionOperator(
                center=None, radius=1e12).project

            # ALM factory: constructs augmented cost psi(u; xi) and its gradient
            self._alm_factory = ALM_Factory(
                f=self.cost_matrices.compute_cost,
                df=self.cost_matrices.compute_gradient,
                mapping_f1=self.cost_matrices.compute_output_mapping,
                jacobian_f1_trans=self.cost_matrices.compute_output_jacobian_trans,
                set_c_project=self._set_c_project,
                n1=n1,
            )

            # ALM problem definition
            self._alm_problem = ALM_Problem(
                parametric_cost=self._alm_factory.psi,
                parametric_gradient=self._alm_factory.d_psi,
                u_min=U_min_flat,
                u_max=U_max_flat,
                mapping_f1=self.cost_matrices.compute_output_mapping,
                set_c_project=self._set_c_project,
                set_y_project=self._set_y_project,
                n1=n1,
            )
        else:
            # No output constraints: ALM_Factory wraps raw cost/gradient directly.
            # With n1=0, psi(u; xi) = f(u) and d_psi(u; xi) = df(u).
            self._alm_factory = ALM_Factory(
                f=self.cost_matrices.compute_cost,
                df=self.cost_matrices.compute_gradient,
                n1=0,
            )

            self._alm_problem = ALM_Problem(
                parametric_cost=self._alm_factory.psi,
                parametric_gradient=self._alm_factory.d_psi,
                u_min=U_min_flat,
                u_max=U_max_flat,
                n1=0,
            )

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
        """
        Generates cost matrices for nonlinear model predictive control (NMPC).
        Constructs diagonal weighting matrices for states, outputs,
        and inputs, and initializes an OptimizationEngine_CostMatrices object
        with the provided symbolic variables, system equations, prediction horizon,
        constraints, and weights.

        Args:
            X_symbolic (sp.Matrix): Symbolic state variables.
            U_symbolic (sp.Matrix): Symbolic input variables.
            fxu (sp.Matrix): Symbolic state transition equation vector.
            hx (sp.Matrix): Symbolic measurement/output equation vector.
            Np (int): Prediction horizon.
            Weight_U (np.ndarray): Weights for input variables.
            Weight_X (np.ndarray): Weights for state variables.
            Weight_Y (np.ndarray): Weights for output variables.
            U_min (np.ndarray): Minimum input constraints.
            U_max (np.ndarray): Maximum input constraints.
            Y_min (np.ndarray): Minimum output constraints.
            Y_max (np.ndarray): Maximum output constraints.
            caller_file_name (str): Name of the calling file for reference.
        Returns:
            OptimizationEngine_CostMatrices: An object containing the generated cost
            matrices and constraints for NMPC.
        """
        Qx = np.diag(Weight_X)
        Qy = np.diag(Weight_Y)
        R = np.diag(Weight_U)

        cost_matrices = OptimizationEngine_CostMatrices(
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

        return cost_matrices

    def get_solver_step_iterated_number(self):
        """
        Returns the number of iterations performed by the solver in the current step.

        Returns:
            int: The number of iterations performed by the solver in the current step.
        """
        return self.solver_configuration.get_solver_step_iterated_number()

    def set_solver_max_iteration(
            self,
            max_iteration: int
    ):
        """
        Set the maximum number of iterations for the solver.
        Parameters
        ----------
        max_iteration : int
            The maximum number of iterations the solver is allowed to perform.
        Returns
        -------
        None
        """
        self.solver_configuration.set_solver_max_iteration(max_iteration)

    def set_reference_trajectory(
            self,
            reference_trajectory: np.ndarray
    ):
        """
        Sets the reference trajectory for the nonlinear MPC controller.

        Parameters
        ----------
        reference_trajectory : np.ndarray
            Reference trajectory array of shape (OUTPUT_SIZE, Np) or (OUTPUT_SIZE, 1).
            - If shape is (OUTPUT_SIZE, Np): uses the provided trajectory
              for each prediction step,
                and extends the last value to Np+1.
            - If shape is (OUTPUT_SIZE, 1): uses the single reference value
              for all prediction steps.

        Raises
        ------
        ValueError
            If the reference_trajectory does not have shape (OUTPUT_SIZE, Np)
              or (OUTPUT_SIZE, 1).
        """
        if not ((reference_trajectory.shape[1] == self.Np) or
                (reference_trajectory.shape[1] == 1)):
            raise ValueError(
                "Reference vector must be either a single row vector or a Np row vectors.")

        self.cost_matrices.reference_trajectory = \
            np.zeros((self.OUTPUT_SIZE, self.Np + 1))

        if reference_trajectory.shape[1] == self.Np:
            for i in range(self.OUTPUT_SIZE):
                for j in range(self.Np):
                    self.cost_matrices.reference_trajectory[i, j] = \
                        reference_trajectory[i, j]

            for i in range(self.OUTPUT_SIZE):
                self.cost_matrices.reference_trajectory[i, self.Np] = \
                    reference_trajectory[i, self.Np - 1]
        else:
            for i in range(self.OUTPUT_SIZE):
                for j in range(self.Np + 1):
                    self.cost_matrices.reference_trajectory[i, j] = \
                        reference_trajectory[i, 0]

    def calculate_this_U(self, U_horizon: np.ndarray) -> np.ndarray:
        """
        Extracts and reshapes the first column of the input horizon matrix.

        Parameters
        ----------
        U_horizon : np.ndarray
            A 2D array representing the input horizon,
              where each column corresponds to an input vector at a time step.

        Returns
        -------
        np.ndarray
            A 2D array of shape (self.INPUT_SIZE, 1) containing the
              input vector for the first time step.
        """
        return U_horizon[:, 0].reshape((self.INPUT_SIZE, 1))

    def compensate_X_Y_delay(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compensates for measurement delay in the X and Y signals
        using a Kalman filter and updates internal state.

        Parameters
        ----------
        X : np.ndarray
            The current state estimate (possibly delayed).
        Y : np.ndarray
            The current measurement (possibly delayed).

        Returns
        -------
        np.ndarray
            The compensated state estimate after delay correction.
        """
        if self.Number_of_Delay > 0:
            Y_measured = Y

            X = self.kalman_filter.get_x_hat_without_delay()
            Y = self.kalman_filter.measurement_function(
                X, self.kalman_filter.Parameters)

            self.Y_store.push(Y)

            self.cost_matrices.set_Y_offset(
                Y_measured - self.Y_store.get())

            return X
        else:
            return X

    def update_parameters(self, parameters_struct):
        """
        Updates the internal parameters of the nonlinear MPC controller.
        Args:
            parameters_struct: A dataclass instance containing the updated parameters.
        Raises:
            ValueError: If parameters_struct is not a dataclass instance.
        """
        if not is_dataclass(parameters_struct):
            raise ValueError(
                "parameters_struct must be a dataclass instance.")

        self.cost_matrices.state_space_parameters = parameters_struct
        self.kalman_filter.Parameters = parameters_struct

    def update_manipulation(
            self,
            reference: np.ndarray,
            Y: np.ndarray
    ):
        """
        Updates the manipulation input for the nonlinear MPC controller.
        This method performs the following steps:
        1. Calculates the latest control input based on the current control horizon.
        2. Updates the internal Kalman filter state estimate using the
          latest control input and measurement Y.
        3. Compensates the state estimate for any delay between state (X)
          and measurement (Y).
        4. Sets the reference trajectory for the MPC optimization.
        5. Solves the MPC optimization problem using ALM/PM (which uses PANOC
          as the inner solver) to update the control horizon.
        6. Calculates and returns the latest control input after optimization.

        Args:
            reference (np.ndarray): The reference trajectory or setpoint
              for the MPC controller.
            Y (np.ndarray): The latest measurement vector.
        Returns:
            np.ndarray: The latest calculated control input to be applied.
        """
        U_latest = self.calculate_this_U(self.U_horizon)

        self.kalman_filter.predict_and_update(
            U_latest, Y)
        X = self.kalman_filter.x_hat

        X_compensated = self.compensate_X_Y_delay(X, Y)

        self.set_reference_trajectory(reference)

        self.cost_matrices.X_initial = X_compensated

        u_flat = self.U_horizon.reshape((-1, 1))

        alm_optimizer = ALM_PM_Optimizer(
            alm_cache=self._alm_cache,
            alm_problem=self._alm_problem,
            max_outer_iterations=ALM_MAX_OUTER_ITERATIONS_DEFAULT,
            max_inner_iterations=self.solver_configuration._max_iteration,
            epsilon_tolerance=ALM_EPSILON_TOLERANCE_DEFAULT,
            delta_tolerance=ALM_DELTA_TOLERANCE_DEFAULT,
            initial_inner_tolerance=ALM_INITIAL_INNER_TOLERANCE_DEFAULT,
            initial_penalty=ALM_INITIAL_PENALTY_DEFAULT,
        )

        status = alm_optimizer.solve(u_flat)
        self.solver_configuration._last_iteration_count = status.num_inner_iterations

        self.U_horizon = u_flat.reshape((self.INPUT_SIZE, self.Np)).copy()

        U_latest = self.calculate_this_U(self.U_horizon)

        return U_latest
