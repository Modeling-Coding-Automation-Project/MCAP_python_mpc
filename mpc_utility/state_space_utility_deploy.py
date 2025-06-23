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

A_UPDATER_CLASS_NAME = "A_Updater"
B_UPDATER_CLASS_NAME = "B_Updater"
C_UPDATER_CLASS_NAME = "C_Updater"
D_UPDATER_CLASS_NAME = "D_Updater"

SYMPY_FUNCTION_NAME = "sympy_function"


class StateSpaceUpdaterDeploy:
    @staticmethod
    def create_Matrix_update_code(sym_object: sp.Matrix, function_name: str) -> str:
        code_text, _ = ExpressionDeploy.create_sympy_code(
            sym_object)

        return code_text

    @staticmethod
    def write_param_names_argument(param_names: list) -> str:
        code_text = ""
        for i, param_name in enumerate(param_names):
            code_text += param_name
            if i < len(param_names) - 1:
                code_text += ", "

        return code_text

    @staticmethod
    def write_update_return_code(class_name: str, param_names: list) -> str:
        code_text = "return " + class_name + "." + \
            SYMPY_FUNCTION_NAME + "("

        for i, param_name in enumerate(param_names):
            code_text += f"{param_name}={param_name}"
            if i < len(param_names) - 1:
                code_text += ", "

        code_text += ")"

        return code_text

    @staticmethod
    def write_matrix_class_code(updater_class_name: str,
                                function_name: str,
                                sym_object: sp.Matrix,
                                param_names: list) -> str:

        code_text = ""

        code_text += "class " + updater_class_name + ":\n\n"
        code_text += "    @staticmethod\n    "
        code_text += "def update("

        code_text += StateSpaceUpdaterDeploy.write_param_names_argument(
            param_names)
        code_text += "):\n"

        code_text += "        " + StateSpaceUpdaterDeploy.write_update_return_code(
            updater_class_name, param_names) + "\n\n"

        code_text += "    @staticmethod\n    "
        function_code = StateSpaceUpdaterDeploy.create_Matrix_update_code(
            sym_object, function_name)
        function_code = function_code.replace("\n", "\n    ")
        code_text += function_code
        code_text += "\n\n"

        return code_text

    @staticmethod
    def create_ABCD_update_code(
            argument_struct,
            A: sp.Matrix = None,
            B: sp.Matrix = None,
            C: sp.Matrix = None,
            D: sp.Matrix = None,
            class_name: str = ""):

        if class_name == "":
            raise ValueError(
                "class_name must be provided to create the state space updater code.")

        param_names = [k for k in vars(
            type(argument_struct)) if not k.startswith('__')]

        code_text = ""
        code_text += "from typing import Tuple\n"
        code_text += "import numpy as np\n\n\n"

        # A class
        if A is not None:
            code_text += StateSpaceUpdaterDeploy.write_matrix_class_code(
                updater_class_name=A_UPDATER_CLASS_NAME,
                function_name=A_UPDATER_FUNCTION_NAME,
                sym_object=A,
                param_names=param_names)

        # B class
        if B is not None:
            code_text += StateSpaceUpdaterDeploy.write_matrix_class_code(
                updater_class_name=B_UPDATER_CLASS_NAME,
                function_name=B_UPDATER_FUNCTION_NAME,
                sym_object=B,
                param_names=param_names)

        # C class
        if C is not None:
            code_text += StateSpaceUpdaterDeploy.write_matrix_class_code(
                updater_class_name=C_UPDATER_CLASS_NAME,
                function_name=C_UPDATER_FUNCTION_NAME,
                sym_object=C,
                param_names=param_names)

        # D class
        if D is not None:
            code_text += StateSpaceUpdaterDeploy.write_matrix_class_code(
                updater_class_name=D_UPDATER_CLASS_NAME,
                function_name=D_UPDATER_FUNCTION_NAME,
                sym_object=D,
                param_names=param_names)

        # ABCD class
        code_text += "class " + class_name + ":\n\n"
        code_text += "    @staticmethod\n"
        code_text += "    def update(parameters):\n\n"

        for param_name in param_names:
            code_text += f"        {param_name} = parameters.{param_name}\n"

        code_text += "\n"
        if A is not None:
            code_text += f"        A = {A_UPDATER_CLASS_NAME}.update" + \
                f"({StateSpaceUpdaterDeploy.write_param_names_argument(param_names)})\n\n"
        else:
            code_text += "        A = None\n\n"

        if B is not None:
            code_text += f"        B = {B_UPDATER_CLASS_NAME}.update" + \
                f"({StateSpaceUpdaterDeploy.write_param_names_argument(param_names)})\n\n"
        else:
            code_text += "        B = None\n\n"

        if C is not None:
            code_text += f"        C = {C_UPDATER_CLASS_NAME}.update" + \
                f"({StateSpaceUpdaterDeploy.write_param_names_argument(param_names)})\n\n"
        else:
            code_text += "        C = None\n\n"

        if D is not None:
            code_text += f"        D = {D_UPDATER_CLASS_NAME}.update" + \
                f"({StateSpaceUpdaterDeploy.write_param_names_argument(param_names)})\n\n"
        else:
            code_text += "        D = None\n\n"

        code_text += "        return A, B, C, D\n\n"

        return code_text

    @staticmethod
    def create_write_ABCD_update_code(
            argument_struct,
            A: sp.Matrix = None,
            B: sp.Matrix = None,
            C: sp.Matrix = None,
            D: sp.Matrix = None,
            class_name: str = ABCD_UPDATER_CLASS_NAME,
            file_name: str = MPC_STATE_SPACE_UPDATER_FILE_NAME):

        code_text = StateSpaceUpdaterDeploy.create_ABCD_update_code(
            argument_struct=argument_struct,
            A=A, B=B, C=C, D=D, class_name=class_name)

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(code_text)
