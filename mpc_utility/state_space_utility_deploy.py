import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp
import inspect
import ast
import astor

from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy
from mpc_utility.state_space_utility import StateSpaceEmbeddedIntegrator

MPC_STATE_SPACE_UPDATER_FILE_NAME = "mpc_state_space_updater.py"
MPC_EMBEDDED_INTEGRATOR_UPDATER_FILE_NAME = "mpc_embedded_integrator_state_space_updater.py"
PREDICTION_MATRICES_PHI_F_UPDATER_FILE_NAME = "prediction_matrices_phi_f_updater.py"

MPC_STATESPACE_UPDATER_CLASS_NAME = "MPC_StateSpace_Updater"
EMBEDDED_INTEGRATOR_UPDATER_CLASS_NAME = "EmbeddedIntegrator_Updater"
PREDICTION_MATRICES_PHI_F_UPDATER_CLASS_NAME = "PredictionMatricesPhiF_Updater"

A_UPDATER_FUNCTION_NAME = "update_A"
B_UPDATER_FUNCTION_NAME = "update_B"
C_UPDATER_FUNCTION_NAME = "update_C"
D_UPDATER_FUNCTION_NAME = "update_D"
PHI_F_UPDATER_FUNCTION_NAME = "update_Phi_F"

A_UPDATER_CLASS_NAME = "A_Updater"
B_UPDATER_CLASS_NAME = "B_Updater"
C_UPDATER_CLASS_NAME = "C_Updater"
D_UPDATER_CLASS_NAME = "D_Updater"

SYMPY_FUNCTION_NAME = "sympy_function"


class StateSpaceUpdaterDeploy:
    @staticmethod
    def create_Matrix_update_code(sym_object: sp.Matrix, function_name: str) -> str:
        code_text, arguments_text = ExpressionDeploy.create_sympy_code(
            sym_object)

        return code_text, arguments_text

    @staticmethod
    def write_param_names_argument(param_names: list) -> str:
        code_text = ""
        for i, param_name in enumerate(param_names):
            code_text += param_name
            if i < len(param_names) - 1:
                code_text += ", "

        return code_text

    @staticmethod
    def write_update_return_code(class_name: str, arguments_text: str) -> str:
        code_text = "return " + class_name + "." + \
            SYMPY_FUNCTION_NAME + "("

        code_text += arguments_text

        code_text += ")"

        return code_text

    @staticmethod
    def write_matrix_class_code(updater_class_name: str,
                                function_name: str,
                                sym_object: sp.Matrix,
                                param_names: list) -> str:

        function_code, arguments_text = \
            StateSpaceUpdaterDeploy.create_Matrix_update_code(
                sym_object, function_name)

        code_text = ""

        code_text += "class " + updater_class_name + ":\n\n"
        code_text += "    @staticmethod\n    "
        code_text += "def update("

        code_text += StateSpaceUpdaterDeploy.write_param_names_argument(
            param_names)
        code_text += "):\n"

        code_text += "        " + StateSpaceUpdaterDeploy.write_update_return_code(
            updater_class_name, arguments_text) + "\n\n"

        code_text += "    @staticmethod\n    "
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
            class_name: str = MPC_STATESPACE_UPDATER_CLASS_NAME,
            file_name: str = MPC_STATE_SPACE_UPDATER_FILE_NAME):

        code_text = StateSpaceUpdaterDeploy.create_ABCD_update_code(
            argument_struct=argument_struct,
            A=A, B=B, C=C, D=D, class_name=class_name)

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(code_text)


class LTV_MPC_StateSpaceInitializer:
    def __init__(self):
        self.ABCD_sympy_function_generated = False
        self.embedded_integrator_ABC_function_generated = False
        self.Phi_F_function_generated = False

    def get_generate_initial_MPC_StateSpace(self, parameters_struct,
                                            A: sp.Matrix = None, B: sp.Matrix = None,
                                            C: sp.Matrix = None, D: sp.Matrix = None,
                                            file_name: str = MPC_STATE_SPACE_UPDATER_FILE_NAME):
        StateSpaceUpdaterDeploy.create_write_ABCD_update_code(
            argument_struct=parameters_struct,
            A=A, B=B, C=C, D=D, class_name=MPC_STATESPACE_UPDATER_CLASS_NAME,
            file_name=file_name)

        local_vars = {"parameters_struct": parameters_struct}

        file_name_no_extension = os.path.splitext(file_name)[0]

        exe_code = (
            f"from {file_name_no_extension} import " +
            MPC_STATESPACE_UPDATER_CLASS_NAME + "\n"
            "A_numeric, B_numeric, C_numeric, D_numeric = " +
            MPC_STATESPACE_UPDATER_CLASS_NAME + ".update(parameters_struct)\n"
        )

        exec(exe_code, globals(), local_vars)

        A_numeric = local_vars["A_numeric"]
        B_numeric = local_vars["B_numeric"]
        C_numeric = local_vars["C_numeric"]
        D_numeric = local_vars["D_numeric"]

        self.ABCD_sympy_function_generated = True

        return A_numeric, B_numeric, C_numeric, D_numeric

    def generate_initial_embedded_integrator(
            self, parameters_struct,
            state_space: StateSpaceEmbeddedIntegrator = None,
            file_name: str = MPC_EMBEDDED_INTEGRATOR_UPDATER_FILE_NAME):

        if state_space is None:
            raise ValueError("State space must be provided.")
        if not isinstance(state_space, StateSpaceEmbeddedIntegrator):
            raise TypeError(
                "State space must be an instance of StateSpaceEmbeddedIntegrator.")

        StateSpaceUpdaterDeploy.create_write_ABCD_update_code(
            argument_struct=parameters_struct,
            A=state_space.A, B=state_space.B,
            C=state_space.C, D=None,
            class_name=EMBEDDED_INTEGRATOR_UPDATER_CLASS_NAME,
            file_name=file_name)

        self.embedded_integrator_ABC_function_generated = True

    def generate_prediction_matrices_phi_f(
            self, Np: int, Nc: int,
            state_space: StateSpaceEmbeddedIntegrator = None,
            file_name: str = PREDICTION_MATRICES_PHI_F_UPDATER_FILE_NAME):

        if state_space is None:
            raise ValueError("State space must be provided.")
        if not isinstance(state_space, StateSpaceEmbeddedIntegrator):
            raise TypeError(
                "State space must be an instance of StateSpaceEmbeddedIntegrator.")

        Phi_shape = (Np * state_space.C.shape[0], Nc * state_space.B.shape[1])
        F_shape = (Np * state_space.C.shape[0], state_space.A.shape[1])

        code_text = ""
        code_text += "from typing import Tuple\n"
        code_text += "import numpy as np\n\n\n"

        code_text += "class " + PREDICTION_MATRICES_PHI_F_UPDATER_CLASS_NAME + ":\n\n"
        code_text += "    @staticmethod\n"
        code_text += "    def " + PHI_F_UPDATER_FUNCTION_NAME + \
            "(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> " + \
            f"Tuple[Tuple[{Phi_shape[0]}, {Phi_shape[1]}], Tuple[{F_shape[0]}, {F_shape[1]}]]:\n\n"

        code_text += "        Phi = np.zeros(" + str(Phi_shape) + ")\n"
        code_text += "        F = np.zeros(" + str(F_shape) + ")\n\n"

        # Create intermediate variables for C @ A and C @ B
        for i in range(Np):
            if i == 0:
                code_text += f"        C_A_1 = C @ A\n"
            else:
                code_text += f"        C_A_{i + 1} = C_A_{i} @ A\n"

        code_text += "\n"

        for i in range(Np):
            if i == 0:
                code_text += f"        C_A_0_B = C @ B\n"
            else:
                code_text += f"        C_A_{i}_B = C_A_{i} @ B\n"

        code_text += "\n"

        # substitute Phi
        for i in range(Np):
            for j in range(Nc):
                row_index = i - j
                if row_index >= 0:
                    for k in range(state_space.C.shape[0]):
                        for l in range(state_space.B.shape[1]):
                            code_text += \
                                f"        Phi[{i * state_space.C.shape[0] + k}, {j * state_space.B.shape[1] + l}] = " + \
                                f"C_A_{row_index}_B[{k}, {l}]\n"

            code_text += "\n"
        code_text += "\n"

        # substitute F
        for i in range(Np):
            for k in range(state_space.C.shape[0]):
                for l in range(state_space.A.shape[1]):
                    code_text += \
                        f"        F[{i * state_space.C.shape[0] + k}, {l}] = C_A_{i}_B[{k}, {l}]\n"
            code_text += "\n"
        code_text += "\n"

        code_text += "        return Phi, F\n\n"

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(code_text)

        self.Phi_F_function_generated = True
