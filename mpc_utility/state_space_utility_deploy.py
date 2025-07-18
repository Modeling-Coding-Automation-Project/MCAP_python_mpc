import os
import sys
sys.path.append(os.getcwd())

import sympy as sp
import importlib

from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy
from mpc_utility.state_space_utility import StateSpaceEmbeddedIntegrator

MPC_STATE_SPACE_UPDATER_FILE_NAME = "mpc_state_space_updater.py"
EMBEDDED_INTEGRATOR_UPDATER_FILE_NAME = "mpc_embedded_integrator_state_space_updater.py"
PREDICTION_MATRICES_PHI_F_UPDATER_FILE_NAME = "prediction_matrices_phi_f_updater.py"
LTV_MPC_PHI_F_UPDATER_FILE_NAME = "ltv_mpc_phi_f_updater.py"

MPC_STATE_SPACE_UPDATER_FILE_NAME_NO_EXTENSION = \
    MPC_STATE_SPACE_UPDATER_FILE_NAME.split(".")[0]
EMBEDDED_INTEGRATOR_UPDATER_FILE_NAME_NO_EXTENSION = \
    EMBEDDED_INTEGRATOR_UPDATER_FILE_NAME.split(".")[0]
PREDICTION_MATRICES_PHI_F_UPDATER_FILE_NAME_NO_EXTENSION = \
    PREDICTION_MATRICES_PHI_F_UPDATER_FILE_NAME.split(".")[0]
LTV_MPC_PHI_F_UPDATER_FILE_NAME_NO_EXTENSION = \
    LTV_MPC_PHI_F_UPDATER_FILE_NAME.split(".")[0]

MPC_STATE_SPACE_UPDATER_CLASS_NAME = "MPC_StateSpace_Updater"
EMBEDDED_INTEGRATOR_UPDATER_CLASS_NAME = "EmbeddedIntegrator_Updater"
PREDICTION_MATRICES_PHI_F_UPDATER_CLASS_NAME = "PredictionMatricesPhiF_Updater"
LTV_MPC_PHI_F_UPDATER_CLASS_NAME = "LTV_MPC_Phi_F_Updater"

A_UPDATER_FUNCTION_NAME = "update_A"
B_UPDATER_FUNCTION_NAME = "update_B"
C_UPDATER_FUNCTION_NAME = "update_C"
D_UPDATER_FUNCTION_NAME = "update_D"
MPC_STATE_SPACE_UPDATER_FUNCTION_NAME = "update"
EMBEDDED_INTEGRATOR_UPDATER_FUNCTION_NAME = "update"
PREDICTION_MATRICES_PHI_F_UPDATER_FUNCTION_NAME = "update"
LTV_MPC_PHI_F_UPDATER_FUNCTION_NAME = "update"

A_UPDATER_CLASS_NAME = "A_Updater"
B_UPDATER_CLASS_NAME = "B_Updater"
C_UPDATER_CLASS_NAME = "C_Updater"
D_UPDATER_CLASS_NAME = "D_Updater"

SYMPY_FUNCTION_NAME = "sympy_function"


class StateSpaceUpdaterDeploy:
    """
    A utility class for generating Python source code to update
      and evaluate symbolic state-space matrices (A, B, C, D) using SymPy.
    This class provides static methods to automate the creation of code
      for updating matrices based on symbolic expressions and parameter structures,
    as well as writing the generated code to files.
    """

    @staticmethod
    def create_Matrix_update_code(sym_object: sp.Matrix, function_name: str) -> str:
        """
        Generates Python code and argument text for updating a SymPy matrix.

        This function takes a SymPy Matrix object and a function name, then uses
        ExpressionDeploy.create_sympy_code to generate the corresponding Python code
        and a string representing the function arguments.

        Args:
            sym_object (sp.Matrix): The SymPy Matrix object to generate code for.
            function_name (str): The name to use for the generated function.

        Returns:
            Tuple[str, str]: A tuple containing the generated code as a string and the
            arguments text as a string.
        """
        code_text, arguments_text = ExpressionDeploy.create_sympy_code(
            sym_object)

        return code_text, arguments_text

    @staticmethod
    def write_param_names_argument(param_names: list) -> str:
        """
        Generates a comma-separated string of parameter names from a list.

        Args:
            param_names (list): A list of parameter name strings.

        Returns:
            str: A single string with parameter names separated by commas.
        """
        code_text = ""
        for i, param_name in enumerate(param_names):
            code_text += param_name
            if i < len(param_names) - 1:
                code_text += ", "

        return code_text

    @staticmethod
    def write_update_return_code(class_name: str, arguments_text: str) -> str:
        """
        Generates a string of Python code that returns the result of calling a specified class method with given arguments.

        Args:
            class_name (str): The name of the class whose method will be called.
            arguments_text (str): The arguments to pass to the method, formatted as a string.

        Returns:
            str: A string representing the Python code to return the result of the method call.
        """
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
        """
        Generates Python source code for a class that contains static methods to update and return a symbolic matrix.

        Args:
            updater_class_name (str): The name of the class to be generated.
            function_name (str): The name of the function that will update the matrix.
            sym_object (sp.Matrix): The symbolic matrix object (from sympy) to be updated.
            param_names (list): List of parameter names to be used as function arguments.

        Returns:
            str: The generated Python source code as a string, defining the class and its static methods.
        """
        function_code, arguments_text = \
            StateSpaceUpdaterDeploy.create_Matrix_update_code(
                sym_object, function_name)

        code_text = ""

        code_text += "class " + updater_class_name + ":\n\n"
        code_text += "    @staticmethod\n    "
        code_text += f"def {MPC_STATE_SPACE_UPDATER_FUNCTION_NAME}("

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
        """
        Generates Python source code for classes and a main updater class
        that compute the state-space matrices (A, B, C, D)
        based on symbolic matrix expressions and a provided parameter structure.

        Args:
            argument_struct: An object or structure containing the parameters
              required for evaluating the symbolic matrices.
            A (sp.Matrix, optional): SymPy matrix representing the symbolic expression
              for the A matrix. Defaults to None.
            B (sp.Matrix, optional): SymPy matrix representing the symbolic expression
              for the B matrix. Defaults to None.
            C (sp.Matrix, optional): SymPy matrix representing the symbolic expression
              for the C matrix. Defaults to None.
            D (sp.Matrix, optional): SymPy matrix representing the symbolic expression
              for the D matrix. Defaults to None.
            class_name (str): Name of the main updater class to be generated.
              Must be provided.

        Returns:
            str: The generated Python source code as a string, containing:
                - Import statements
                - One class per non-None matrix (A, B, C, D) for updating the respective matrix
                - A main updater class with a static method that computes
                    and returns the matrices (A, B, C, D)
                    given a parameter structure

        Raises:
            ValueError: If `class_name` is not provided (empty string).
        """
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
        code_text += f"    def {MPC_STATE_SPACE_UPDATER_FUNCTION_NAME}(parameters):\n\n"

        for param_name in param_names:
            code_text += f"        {param_name} = parameters.{param_name}\n"

        code_text += "\n"
        if A is not None:
            code_text += f"        A = {A_UPDATER_CLASS_NAME}.{MPC_STATE_SPACE_UPDATER_FUNCTION_NAME}" + \
                f"({StateSpaceUpdaterDeploy.write_param_names_argument(param_names)})\n\n"
        else:
            code_text += "        A = None\n\n"

        if B is not None:
            code_text += f"        B = {B_UPDATER_CLASS_NAME}.{MPC_STATE_SPACE_UPDATER_FUNCTION_NAME}" + \
                f"({StateSpaceUpdaterDeploy.write_param_names_argument(param_names)})\n\n"
        else:
            code_text += "        B = None\n\n"

        if C is not None:
            code_text += f"        C = {C_UPDATER_CLASS_NAME}.{MPC_STATE_SPACE_UPDATER_FUNCTION_NAME}" + \
                f"({StateSpaceUpdaterDeploy.write_param_names_argument(param_names)})\n\n"
        else:
            code_text += "        C = None\n\n"

        if D is not None:
            code_text += f"        D = {D_UPDATER_CLASS_NAME}.{MPC_STATE_SPACE_UPDATER_FUNCTION_NAME}" + \
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
            class_name: str = MPC_STATE_SPACE_UPDATER_CLASS_NAME,
            file_name: str = MPC_STATE_SPACE_UPDATER_FILE_NAME):
        """
        Generates Python code to update state-space matrices (A, B, C, D)
          and writes it to a specified file.

        This function calls `StateSpaceUpdaterDeploy.create_ABCD_update_code`
          to generate the code for updating
        the state-space matrices based on the provided arguments,
          then writes the generated code to a file.

        Args:
            argument_struct: A structure containing arguments required for code generation.
            A (sp.Matrix, optional): The state matrix A. Defaults to None.
            B (sp.Matrix, optional): The input matrix B. Defaults to None.
            C (sp.Matrix, optional): The output matrix C. Defaults to None.
            D (sp.Matrix, optional): The feedthrough matrix D. Defaults to None.
            class_name (str, optional): The name of the class to be generated.
              Defaults to MPC_STATE_SPACE_UPDATER_CLASS_NAME.
            file_name (str, optional): The name of the file to write the generated code to.
              Defaults to MPC_STATE_SPACE_UPDATER_FILE_NAME.

        Returns:
            None
        """

        code_text = StateSpaceUpdaterDeploy.create_ABCD_update_code(
            argument_struct=argument_struct,
            A=A, B=B, C=C, D=D, class_name=class_name)

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(code_text)


class LTV_MPC_StateSpaceInitializer:
    """
    A utility class for generating and managing state-space matrices and prediction matrices
    for Linear Time-Varying Model Predictive Control (LTV-MPC) systems.
    This class provides methods to:
        - Dynamically generate Python code for updating state-space matrices (A, B, C, D)
          based on user-provided parameters.
        - Write and import updater modules for state-space and embedded integrator models.
        - Generate and manage prediction matrices (Phi, F) for MPC horizons.
        - Compose updaters for LTV-MPC prediction matrices using embedded integrator and
          prediction matrix updaters.
    """

    def __init__(self, caller_file_name_without_ext: str = None):

        self.file_name_suffix = ""
        if caller_file_name_without_ext is not None:
            self.file_name_suffix = caller_file_name_without_ext + "_"

        self.mpc_state_space_updater_file_name = ""
        self.embedded_integrator_updater_file_name = ""
        self.prediction_matrices_phi_f_updater_file_name = ""
        self.LTV_MPC_Phi_F_updater_file_name = ""

        self.mpc_state_space_updater_generated = False
        self.embedded_integrator_ABC_function_generated = False
        self.Phi_F_function_generated = False
        self.LTV_MPC_Phi_F_function_generated = False

        self.mpc_state_space_updater_function = None
        self.LTV_MPC_Phi_F_updater_function = None

    def get_generate_initial_MPC_StateSpace(self, parameters_struct,
                                            A: sp.Matrix = None, B: sp.Matrix = None,
                                            C: sp.Matrix = None, D: sp.Matrix = None,
                                            file_name: str = MPC_STATE_SPACE_UPDATER_FILE_NAME):
        """
        Generates and initializes the MPC (Model Predictive Control) state-space matrices (A, B, C, D)
        by dynamically creating and executing an updater module based on the provided parameters.

        This method:
            1. Generates Python code to update the state-space matrices using the provided parameters.
            2. Writes the generated code to a specified file.
            3. Dynamically imports and executes the updater to obtain numeric matrices.
            4. Stores references to the updater class and function for later use.

        Args:
            parameters_struct: A structure containing parameters required for state-space matrix generation.
            A (sp.Matrix, optional): Symbolic or numeric A matrix. Defaults to None.
            B (sp.Matrix, optional): Symbolic or numeric B matrix. Defaults to None.
            C (sp.Matrix, optional): Symbolic or numeric C matrix. Defaults to None.
            D (sp.Matrix, optional): Symbolic or numeric D matrix. Defaults to None.
            file_name (str, optional): The file name for the generated updater module.
              Defaults to MPC_STATE_SPACE_UPDATER_FILE_NAME.

        Returns:
            tuple: A tuple containing the numeric state-space matrices
              (A_numeric, B_numeric, C_numeric, D_numeric).

        Raises:
            ImportError: If the generated updater module cannot be imported.
            AttributeError: If the updater class or function cannot be found in the generated module.
            Exception: For any errors during code generation or execution.
        """
        file_name = self.file_name_suffix + file_name

        StateSpaceUpdaterDeploy.create_write_ABCD_update_code(
            argument_struct=parameters_struct,
            A=A, B=B, C=C, D=D, class_name=MPC_STATE_SPACE_UPDATER_CLASS_NAME,
            file_name=file_name)

        local_vars = {"parameters_struct": parameters_struct}

        file_name_no_extension = os.path.splitext(file_name)[0]

        exe_code = (
            f"from {file_name_no_extension} import " +
            MPC_STATE_SPACE_UPDATER_CLASS_NAME + "\n"
            "A_numeric, B_numeric, C_numeric, D_numeric = " +
            MPC_STATE_SPACE_UPDATER_CLASS_NAME + ".update(parameters_struct)\n"
        )

        exec(exe_code, globals(), local_vars)

        A_numeric = local_vars["A_numeric"]
        B_numeric = local_vars["B_numeric"]
        C_numeric = local_vars["C_numeric"]
        D_numeric = local_vars["D_numeric"]

        self.mpc_state_space_updater_file_name = file_name
        self.mpc_state_space_updater_generated = True

        module_name = os.path.splitext(os.path.basename(file_name))[0]
        module = importlib.import_module(module_name)

        state_space_updater = getattr(
            module, MPC_STATE_SPACE_UPDATER_CLASS_NAME)

        self.mpc_state_space_updater_function = getattr(
            state_space_updater, MPC_STATE_SPACE_UPDATER_FUNCTION_NAME)

        return A_numeric, B_numeric, C_numeric, D_numeric

    def update_mpc_state_space_runtime(self, parameters_struct):
        """
        Updates the MPC (Model Predictive Control) state space
          at runtime using the provided parameters.

        Args:
            parameters_struct (Any): A structure containing the parameters
              required to update the MPC state space.

        Returns:
            Any: The result of the MPC state space updater function,
              which may vary depending on the implementation.

        Note:
            This method delegates the update operation to the
              `mpc_state_space_updater_function` attribute.
        """
        return self.mpc_state_space_updater_function(parameters_struct)

    def generate_initial_embedded_integrator(
            self, parameters_struct,
            state_space: StateSpaceEmbeddedIntegrator = None,
            file_name: str = EMBEDDED_INTEGRATOR_UPDATER_FILE_NAME):
        """
        Generates and writes the code for the initial embedded integrator's ABC update function.

        This method validates the provided state space object and uses the StateSpaceUpdaterDeploy
        utility to create and write the ABCD update code for an embedded integrator. The generated
        code is written to the specified file.

        Args:
            parameters_struct: A structure containing the parameters required for code generation.
            state_space (StateSpaceEmbeddedIntegrator, optional): The state space object representing
                the embedded integrator. Must be an instance of StateSpaceEmbeddedIntegrator.
            file_name (str, optional): The name of the file to which the generated code will be written.
                Defaults to EMBEDDED_INTEGRATOR_UPDATER_FILE_NAME.

        Raises:
            ValueError: If the state_space argument is not provided.
            TypeError: If the state_space is not an instance of StateSpaceEmbeddedIntegrator.

        Side Effects:
            Writes the generated ABC update code to the specified file.
            Sets self.embedded_integrator_ABC_function_generated to True.
        """
        file_name = self.file_name_suffix + file_name

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

        self.embedded_integrator_updater_file_name = file_name
        self.embedded_integrator_ABC_function_generated = True

    def generate_prediction_matrices_phi_f(
            self, Np: int, Nc: int,
            state_space: StateSpaceEmbeddedIntegrator = None,
            file_name: str = PREDICTION_MATRICES_PHI_F_UPDATER_FILE_NAME):
        """
        Generates and writes Python code for computing the prediction matrices
          Phi and F used in Model Predictive Control (MPC).

        This method dynamically generates a Python class and function that,
          given state-space matrices (A, B, C), computes the prediction matrices
            Phi and F for specified prediction (Np) and control (Nc) horizons.
        The generated code is written to a file for later use.

        Args:
            Np (int): Prediction horizon (number of future steps to predict).
            Nc (int): Control horizon (number of future control moves to optimize).
            state_space (StateSpaceEmbeddedIntegrator, optional): State-space model
              containing matrices A, B, and C.
                Must be an instance of StateSpaceEmbeddedIntegrator. Defaults to None.
            file_name (str, optional): Path to the file where the generated code will be saved.
                Defaults to PREDICTION_MATRICES_PHI_F_UPDATER_FILE_NAME.

        Raises:
            ValueError: If state_space is not provided.
            TypeError: If state_space is not an instance of StateSpaceEmbeddedIntegrator.

        Side Effects:
            Writes a Python file containing a class and function
              for computing Phi and F matrices.
            Sets self.Phi_F_function_generated to True upon successful code generation.
        """
        file_name = self.file_name_suffix + file_name

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

        # class for calculate Phi and F
        code_text += "class " + PREDICTION_MATRICES_PHI_F_UPDATER_CLASS_NAME + ":\n\n"
        code_text += "    @staticmethod\n"
        code_text += "    def " + PREDICTION_MATRICES_PHI_F_UPDATER_FUNCTION_NAME + \
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
                        f"        F[{i * state_space.C.shape[0] + k}, {l}] = C_A_{i + 1}[{k}, {l}]\n"
            code_text += "\n"
        code_text += "\n"

        code_text += "        return Phi, F\n\n"

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(code_text)

        self.prediction_matrices_phi_f_updater_file_name = file_name
        self.Phi_F_function_generated = True

    def generate_LTV_MPC_Phi_F_Updater(
            self, file_name: str = LTV_MPC_PHI_F_UPDATER_FILE_NAME):
        """
        Dynamically generates and writes a Python module that defines a class for updating
        the prediction matrices Phi and F for a Linear Time-Varying Model
          Predictive Control (LTV-MPC) system.

        The generated class contains a static method that:
            1. Calls an embedded integrator updater to obtain system matrices
              (A, B, C, _).
            2. Uses these matrices to compute the prediction matrices
              Phi and F via another updater.
            3. Returns Phi and F as numpy arrays.

        After writing the module to the specified file,
          this function dynamically imports the module,
        retrieves the updater class and its static method,
          and assigns the method to an instance variable
        for later use.

        Args:
            file_name (str, optional): The name of the file to write the
              generated updater class to.
                Defaults to LTV_MPC_PHI_F_UPDATER_FILE_NAME.

        Side Effects:
            - Writes a Python file containing the updater class.
            - Dynamically imports the generated module and sets instance variables:
                - self.LTV_MPC_Phi_F_updater_function: The static method
                  for updating Phi and F.
                - self.LTV_MPC_Phi_F_function_generated: Flag indicating
                  the function was generated.

        Raises:
            Any exceptions raised by file I/O or dynamic import operations.
        """
        file_name = self.file_name_suffix + file_name

        code_text = ""
        code_text += "from typing import Tuple\n"
        code_text += "import numpy as np\n\n"

        file_name_no_extension = os.path.splitext(
            self.embedded_integrator_updater_file_name)[0]
        code_text += "from " + file_name_no_extension + " import " + \
            EMBEDDED_INTEGRATOR_UPDATER_CLASS_NAME + "\n"

        file_name_no_extension = os.path.splitext(
            self.prediction_matrices_phi_f_updater_file_name)[0]

        code_text += "from " + file_name_no_extension + " import " + \
            PREDICTION_MATRICES_PHI_F_UPDATER_CLASS_NAME + "\n\n"

        # class for update Phi and F from parameters
        code_text += "class " + LTV_MPC_PHI_F_UPDATER_CLASS_NAME + ":\n\n"
        code_text += "    @staticmethod\n"
        code_text += "    def " + LTV_MPC_PHI_F_UPDATER_FUNCTION_NAME + \
            "(parameters_struct) -> Tuple[np.ndarray, np.ndarray]:\n\n"

        code_text += "        A, B, C, _ = " + \
            EMBEDDED_INTEGRATOR_UPDATER_CLASS_NAME + \
            "." + EMBEDDED_INTEGRATOR_UPDATER_FUNCTION_NAME + \
            "(parameters_struct)\n\n"

        code_text += "        Phi, F = " + \
            PREDICTION_MATRICES_PHI_F_UPDATER_CLASS_NAME + \
            "." + PREDICTION_MATRICES_PHI_F_UPDATER_FUNCTION_NAME + \
            "(A, B, C)\n\n"

        code_text += "        return Phi, F\n\n"

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(code_text)

        module_name = os.path.splitext(os.path.basename(file_name))[0]
        module = importlib.import_module(module_name)

        LTV_MPC_Phi_F_Updater = getattr(
            module, LTV_MPC_PHI_F_UPDATER_CLASS_NAME)

        self.LTV_MPC_Phi_F_updater_function = getattr(
            LTV_MPC_Phi_F_Updater, LTV_MPC_PHI_F_UPDATER_FUNCTION_NAME)

        self.LTV_MPC_Phi_F_updater_file_name = file_name
        self.LTV_MPC_Phi_F_function_generated = True


class Adaptive_MPC_StateSpaceInitializer:
    def __init__(self, fxu_function,
                 fxu_jacobian_X_function,
                 fxu_jacobian_U_function,
                 hx_function,
                 hx_jacobian_function):

        self.fxu_function = fxu_function
        self.fxu_jacobian_X_function = fxu_jacobian_X_function
        self.fxu_jacobian_U_function = fxu_jacobian_U_function
        self.hx_function = hx_function
        self.hx_jacobian_function = hx_jacobian_function
