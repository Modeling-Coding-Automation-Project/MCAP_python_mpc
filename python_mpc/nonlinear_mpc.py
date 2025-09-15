import os
import inspect
import numpy as np
import sympy as sp
from dataclasses import is_dataclass


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
        Np: int, Nc: int,
        Weight_U: np.ndarray, Weight_Y: np.ndarray,
        Q_kf: np.ndarray = None, R_kf: np.ndarray = None,
        Number_of_Delay: int = 0,
        is_ref_trajectory: bool = False,
        caller_file_name: str = None
    ):
        # inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        for name, value in caller_locals.items():
            if value is X:
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

        if not is_dataclass(parameters_struct):
            raise ValueError(
                "parameters_struct must be a dataclass instance.")
