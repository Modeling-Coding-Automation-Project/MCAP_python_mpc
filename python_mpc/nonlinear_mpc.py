import os
import sys
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_optimization'))

import inspect
import numpy as np
import sympy as sp
from dataclasses import is_dataclass

from external_libraries.MCAP_python_control.python_control.kalman_filter import ExtendedKalmanFilter
from external_libraries.MCAP_python_control.python_control.kalman_filter import DelayedVectorObject


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
        Weight_U: np.ndarray, Weight_Y: np.ndarray,
        Q_kf: np.ndarray = None, R_kf: np.ndarray = None,
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

        self.X_symbolic = X
        self.U_symbolic = U
        self.fxu = fxu
        self.hx = hx

        # create EKF object
        self.kalman_filter, \
            (self.fxu_script_function, self.fxu_file_name), \
            (self.hx_script_function, self.hx_file_name) \
            = self.initialize_kalman_filter(
                X=X, U=U,
                fxu=fxu,
                hx=hx,
                Q_kf=Q_kf,
                R_kf=R_kf,
                parameters_struct=parameters_struct,
                file_name_without_ext=caller_file_name_without_ext
            )

    def initialize_kalman_filter(
        self,
        X: sp.Matrix,
        U: sp.Matrix,
        fxu: sp.Matrix,
        hx: sp.Matrix,
        Q_kf: np.ndarray,
        R_kf: np.ndarray,
        parameters_struct,
        file_name_without_ext: str
    ):
        pass
