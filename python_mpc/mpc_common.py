import numpy as np
import sympy as sp
from dataclasses import is_dataclass

from external_libraries.MCAP_python_control.python_control.kalman_filter import ExtendedKalmanFilter

from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy


def initialize_kalman_filter_with_EKF(
        X_initial: np.ndarray,
        X: sp.Matrix, U: sp.Matrix,
        fxu: sp.Matrix, fxu_jacobian_X: sp.Matrix,
        hx: sp.Matrix, hx_jacobian: sp.Matrix,
        Q_kf: np.ndarray,
        R_kf: np.ndarray,
        parameters_struct,
        Number_of_Delay: int,
        file_name_without_ext: str
):
    """
    Initializes an Extended Kalman Filter (EKF) using symbolic model functions and their Jacobians.

    This method generates Python code for the state and measurement functions (and their Jacobians)
    from SymPy expressions, dynamically imports them, and constructs an EKF instance with the provided
    noise covariances and parameters.

    Args:
        X_initial (np.ndarray): Initial state estimate.
        X (sp.Matrix): Symbolic state vector.
        U (sp.Matrix): Symbolic input vector.
        fxu (sp.Matrix): Symbolic state transition function f(x, u).
        fxu_jacobian_X (sp.Matrix): Jacobian of the state transition function with respect to X.
        hx (sp.Matrix): Symbolic measurement function h(x).
        hx_jacobian (sp.Matrix): Jacobian of the measurement function with respect to X.
        Q_kf (np.ndarray): Process noise covariance matrix.
        R_kf (np.ndarray): Measurement noise covariance matrix.
        parameters_struct: Additional parameters required for the filter.
        Number_of_Delay (int): Number of delay steps in the system.
        file_name_without_ext (str): Base filename for generated function scripts.

    Returns:
        tuple: (
            kalman_filter (ExtendedKalmanFilter): Initialized EKF instance,
            fxu_file_name (str): Filename of the generated state function script,
            fxu_jacobian_X_file_name (str): Filename of the generated state function Jacobian script,
            hx_file_name (str): Filename of the generated measurement function script,
            hx_jacobian_file_name (str): Filename of the generated measurement function Jacobian script
    """
    if not is_dataclass(parameters_struct):
        raise ValueError(
            "parameters_struct must be a dataclass instance.")

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
        Parameters=parameters_struct,
        Number_of_Delay=Number_of_Delay
    )
    kalman_filter.x_hat = X_initial

    return kalman_filter, \
        (fxu_script_function, fxu_file_name), \
        (hx_script_function, hx_file_name)
