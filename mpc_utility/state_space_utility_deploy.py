import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp
import inspect
import ast
import astor

from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy

MPC_STATE_SPACE_UPDATER_FILE_NAME = "mpc_state_space_updater.py"

ABCD_UPDATER_CLASS_NAME = "ABCD_Updater"
A_UPDATER_FUNCTION_NAME = "update_A"
B_UPDATER_FUNCTION_NAME = "update_B"
C_UPDATER_FUNCTION_NAME = "update_C"
D_UPDATER_FUNCTION_NAME = "update_D"


class StateSpaceUpdaterDeploy:
    @staticmethod
    def create_ABCD_update_code(
            A: sp.Matrix = None,
            B: sp.Matrix = None,
            C: sp.Matrix = None,
            D: sp.Matrix = None,
            class_name: str = ""):

        if class_name == "":
            raise ValueError(
                "class_name must be provided to create the state space updater code.")

        code_text = ""
        code_text += "from typing import Tuple\n"
        code_text += "import numpy as np\n\n\n"

        code_text += "class " + class_name + ":\n\n"

        if A is not None:
            code_text += "    @staticmethod\n    "
            function_code = StateSpaceUpdaterDeploy.create_Matrix_update_code(
                A, A_UPDATER_FUNCTION_NAME)
            function_code = function_code.replace("\n", "\n    ")
            code_text += function_code

        if B is not None:
            code_text += "@staticmethod\n    "
            function_code = StateSpaceUpdaterDeploy.create_Matrix_update_code(
                B, B_UPDATER_FUNCTION_NAME)
            function_code = function_code.replace("\n", "\n    ")
            code_text += function_code

        if C is not None:
            code_text += "@staticmethod\n    "
            function_code = StateSpaceUpdaterDeploy.create_Matrix_update_code(
                C, C_UPDATER_FUNCTION_NAME)
            function_code = function_code.replace("\n", "\n    ")
            code_text += function_code

        if D is not None:
            code_text += "@staticmethod\n    "
            function_code = StateSpaceUpdaterDeploy.create_Matrix_update_code(
                D, D_UPDATER_FUNCTION_NAME)
            function_code = function_code.replace("\n", "\n    ")
            code_text += function_code

        return code_text

    @staticmethod
    def create_Matrix_update_code(sym_object: sp.Matrix, function_name: str):
        code_text, _ = ExpressionDeploy.create_sympy_code(
            sym_object)

        code_text = code_text.replace(
            "def sympy_function(", "def " + function_name + "(")

        return code_text
