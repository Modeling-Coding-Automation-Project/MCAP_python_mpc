"""
File: state_space_utility.py

This module provides utility classes and functions for symbolic
and numeric manipulation of state-space models,
particularly for Model Predictive Control (MPC) applications.
It leverages sympy for symbolic computation and numpy for numerical operations,
enabling the construction, augmentation, and conversion of state-space representations,
as well as the generation of prediction matrices for MPC.
"""
import numpy as np
import sympy as sp
import copy


def symbolic_to_numeric_matrix(symbolic_matrix: sp.Matrix) -> np.ndarray:
    """
    Convert a symbolic sympy matrix to a numeric numpy matrix.
    Args:
        symbolic_matrix (sp.Matrix): A sympy matrix containing symbolic expressions.
    Returns:
        np.ndarray: A numpy array with numeric values converted from the symbolic matrix.
    """
    numeric_matrix = np.zeros(
        (symbolic_matrix.shape[0], symbolic_matrix.shape[1]), dtype=float)

    for i in range(symbolic_matrix.shape[0]):
        for j in range(symbolic_matrix.shape[1]):
            numeric_matrix[i, j] = float(symbolic_matrix[i, j])

    return numeric_matrix


class SymbolicStateSpace:
    """
    A class representing a symbolic state-space model.
    Attributes:
        A (sp.Matrix): State matrix.
        B (sp.Matrix): Input matrix.
        C (sp.Matrix): Output matrix.
        D (sp.Matrix, optional): Feedthrough matrix.
        delta_time (float): Time step for discrete systems.
        Number_of_Delay (int): Number of delays in the system.
    """

    def __init__(self, A: sp.Matrix, B: sp.Matrix, C: sp.Matrix,
                 D: sp.Matrix = None, delta_time=0.0, Number_of_Delay=0):
        self.delta_time = delta_time
        self.STATE_SIZE = A.shape[0]
        self.INPUT_SIZE = B.shape[1]
        self.OUTPUT_SIZE = C.shape[0]

        if not isinstance(A, sp.MatrixBase):
            self.A = sp.Matrix(A)
        else:
            self.A = A

        if not isinstance(B, sp.MatrixBase):
            self.B = sp.Matrix(B)
        else:
            self.B = B

        if not isinstance(C, sp.MatrixBase):
            self.C = sp.Matrix(C)
        else:
            self.C = C

        if D is not None:
            if not isinstance(D, sp.MatrixBase):
                self.D = sp.Matrix(D)
            else:
                self.D = D

        self.Number_of_Delay = Number_of_Delay


class StateSpaceEmbeddedIntegrator:
    """
    A class that augments a state-space model with an embedded integrator.
    This class takes a symbolic state-space model and constructs an augmented model
    that includes the state, input, and output matrices, along with the necessary
    transformations to handle the output as an integral of the state.
    Attributes:
        delta_time (float): Time step for discrete systems.
        INPUT_SIZE (int): Number of inputs in the system.
        STATE_SIZE (int): Number of states in the system.
        OUTPUT_SIZE (int): Number of outputs in the system.
        A (sp.Matrix): Augmented state matrix.
        B (sp.Matrix): Augmented input matrix.
        C (sp.Matrix): Augmented output matrix.
    """

    def __init__(self, state_space: SymbolicStateSpace):
        if not isinstance(state_space.A, sp.MatrixBase):
            raise ValueError(
                "A must be of type sympy matrix.")
        if not isinstance(state_space.B, sp.MatrixBase):
            raise ValueError(
                "B must be of type sympy matrix.")
        if not isinstance(state_space.C, sp.MatrixBase):
            raise ValueError(
                "C must be of type sympy matrix.")

        self.delta_time = state_space.delta_time

        self.INPUT_SIZE = state_space.INPUT_SIZE
        self.STATE_SIZE = state_space.STATE_SIZE
        self.OUTPUT_SIZE = state_space.OUTPUT_SIZE

        self.A = sp.Matrix(self.STATE_SIZE + self.OUTPUT_SIZE,
                           self.STATE_SIZE + self.OUTPUT_SIZE,
                           lambda i, j: 0.0)
        self.B = sp.Matrix(self.STATE_SIZE + self.OUTPUT_SIZE,
                           self.INPUT_SIZE,
                           lambda i, j: 0.0)
        self.C = sp.Matrix(self.OUTPUT_SIZE,
                           self.STATE_SIZE + self.OUTPUT_SIZE,
                           lambda i, j: 0.0)

        self.construct_augmented_model(
            state_space.A, state_space.B, state_space.C)

    def construct_augmented_model(self, A_original: sp.Matrix,
                                  B_original: sp.Matrix, C_original: sp.Matrix):
        """
        Constructs the augmented state-space model with an embedded integrator.
        Args:
            A_original (sp.Matrix): Original state matrix.
            B_original (sp.Matrix): Original input matrix.
            C_original (sp.Matrix): Original output matrix.
        """

        o_xy_T = sp.Matrix(self.STATE_SIZE,
                           self.OUTPUT_SIZE, lambda i, j: 0.0)
        o_xy = sp.Matrix(self.OUTPUT_SIZE,
                         self.STATE_SIZE, lambda i, j: 0.0)
        I_yy = sp.eye(self.OUTPUT_SIZE)

        A_upper: sp.Matrix = A_original.row_join(o_xy_T)
        A_lower = (C_original * A_original).row_join(I_yy)
        self.A = A_upper.col_join(A_lower)

        B_upper = B_original
        B_lower = C_original * B_original
        self.B = B_upper.col_join(B_lower)

        self.C = o_xy.row_join(I_yy)


class MPC_PredictionMatrices:
    """
    A class to generate prediction matrices for Model Predictive Control (MPC).
    This class constructs the F and Phi matrices based on the state-space model
    and the specified prediction horizon (Np) and control horizon (Nc).

    Attributes:
        Np (int): Prediction horizon.
        Nc (int): Control horizon.
        INPUT_SIZE (int): Number of inputs in the system.
        STATE_SIZE (int): Number of states in the system.
        OUTPUT_SIZE (int): Number of outputs in the system.
    """

    def __init__(self, Np, Nc, INPUT_SIZE, STATE_SIZE, OUTPUT_SIZE):
        self.INPUT_SIZE = INPUT_SIZE
        self.STATE_SIZE = STATE_SIZE
        self.OUTPUT_SIZE = OUTPUT_SIZE

        self.Np = Np
        self.Nc = Nc

        self.A_symbolic = None
        self.B_symbolic = None
        self.C_symbolic = None

        self.A_SparseAvailable = None
        self.B_SparseAvailable = None
        self.C_SparseAvailable = None

        self.A_numeric_expression = None
        self.B_numeric_expression = None
        self.C_numeric_expression = None

        self.initialize_ABC()

        self.F_SparseAvailable = None
        self.Phi_SparseAvailable = None

        self.F_numeric_expression = None
        self.Phi_numeric_expression = None

        self.F_ndarray = None
        self.Phi_ndarray = None

        self.ABC_values = {}

        self.Phi_F_updater_function = None

    def initialize_ABC(self):
        """
        Initializes the symbolic matrices A, B, and C with zeros.
        This method sets up the symbolic matrices with the appropriate dimensions
        based on the state, input, and output sizes.
        """

        self.A_numeric_expression = sp.Matrix(self.STATE_SIZE, self.STATE_SIZE,
                                              lambda i, j: 0.0)
        self.B_numeric_expression = sp.Matrix(self.STATE_SIZE, self.INPUT_SIZE,
                                              lambda i, j: 0.0)
        self.C_numeric_expression = sp.Matrix(self.OUTPUT_SIZE, self.STATE_SIZE,
                                              lambda i, j: 0.0)

    def substitute_ABC_symbolic(self, A: sp.Matrix, B: sp.Matrix, C: sp.Matrix):
        """
        Substitutes the symbolic state-space matrices A, B, and C.

        Parameters:
            A (sp.Matrix): The symbolic state matrix (A) as a SymPy matrix.
            B (sp.Matrix): The symbolic input matrix (B) as a SymPy matrix.
            C (sp.Matrix): The symbolic output matrix (C) as a SymPy matrix.

        Raises:
            ValueError: If any of A, B, or C is not a SymPy matrix.

        Sets:
            self.A_symbolic: The symbolic state matrix.
            self.B_symbolic: The symbolic input matrix.
            self.C_symbolic: The symbolic output matrix.
        """

        if not isinstance(A, sp.MatrixBase):
            raise ValueError("A must be a sympy matrix.")
        if not isinstance(B, sp.MatrixBase):
            raise ValueError("B must be a sympy matrix.")
        if not isinstance(C, sp.MatrixBase):
            raise ValueError("C must be a sympy matrix.")

        self.A_symbolic = A
        self.B_symbolic = B
        self.C_symbolic = C

    def substitute_numeric_matrix_expression(self, mat: np.ndarray) -> sp.Matrix:
        """
        Substitutes numeric values into a symbolic matrix.

        Args:
            mat (np.ndarray): The numeric matrix to substitute.

        Returns:
            sp.Matrix: The symbolic matrix with substituted values.
        """
        if not isinstance(mat, np.ndarray):
            raise ValueError("Input must be a numpy ndarray.")

        symbolic_mat = sp.Matrix(mat.shape[0], mat.shape[1],
                                 lambda i, j: sp.symbols(f'm{i+1}{j+1}'))
        subs_dict = {}
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                symbol = sp.symbols(f'm{i+1}{j+1}')
                subs_dict[symbol] = mat[i, j]

        return symbolic_mat.subs(subs_dict)

    def substitute_ABC_numeric_expression(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        """
        Substitutes numeric values into the symbolic matrices A, B, and C.
        Args:
            A (np.ndarray): State matrix.
            B (np.ndarray): Input matrix.
            C (np.ndarray): Output matrix.
        """

        self.A_numeric_expression = self.substitute_numeric_matrix_expression(
            A)
        self.B_numeric_expression = self.substitute_numeric_matrix_expression(
            B)
        self.C_numeric_expression = self.substitute_numeric_matrix_expression(
            C)

        self._exponential_A_list = self._generate_exponential_A_list(
            self.A_numeric_expression)

    def substitute_numeric(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> tuple:
        """
        Substitutes numeric values into the symbolic matrices A, B, and C,
        and builds the F and Phi matrices.
        Args:
            A (np.ndarray): State matrix.
            B (np.ndarray): Input matrix.
            C (np.ndarray): Output matrix.
        """
        if isinstance(A, sp.MatrixBase):
            A = symbolic_to_numeric_matrix(A)
        else:
            if not isinstance(A, np.ndarray):
                raise ValueError("A must be a numpy ndarray or sympy matrix.")
        if isinstance(B, sp.MatrixBase):
            B = symbolic_to_numeric_matrix(B)
        else:
            if not isinstance(B, np.ndarray):
                raise ValueError("B must be a numpy ndarray or sympy matrix.")
        if isinstance(C, sp.MatrixBase):
            C = symbolic_to_numeric_matrix(C)
        else:
            if not isinstance(C, np.ndarray):
                raise ValueError("C must be a numpy ndarray or sympy matrix.")

        self.substitute_ABC_numeric_expression(A, B, C)

        self.build_matrices_numeric_expression(
            self.B_numeric_expression, self.C_numeric_expression)

        self.F_ndarray = symbolic_to_numeric_matrix(
            self.F_numeric_expression)
        self.Phi_ndarray = symbolic_to_numeric_matrix(
            self.Phi_numeric_expression)

    def build_matrices_numeric_expression(
            self, B: sp.Matrix, C: sp.Matrix) -> tuple:
        """
        Builds the F and Phi matrices based on the symbolic state-space model.
        Args:
            B (sp.Matrix): Input matrix.
            C (sp.Matrix): Output matrix.
        """
        self.F_numeric_expression = self._build_F_expression(C)
        self.Phi_numeric_expression = self._build_Phi_expression(B, C)

    def update_Phi_F_runtime(self, parameters_struct):
        """
        Updates the Phi and F matrices at runtime using the provided parameters.
        This method calls the `Phi_F_updater_function` (if defined) with the given
        `parameters_struct` to compute updated values for the Phi and F matrices.
        The resulting matrices are then stored in `self.Phi_ndarray` and `self.F_ndarray`.
        Args:
            parameters_struct: A structure or object containing parameters required
                by the `Phi_F_updater_function` to compute the updated matrices.
        Returns:
            None
        """
        if self.Phi_F_updater_function is not None:
            Phi, F = self.Phi_F_updater_function(
                parameters_struct=parameters_struct)

            self.Phi_ndarray = Phi
            self.F_ndarray = F

    def update_Phi_F_adaptive_runtime(
            self, parameters_struct,
            X_ndarray: np.ndarray, U_ndarray: np.ndarray):

        if self.Phi_F_updater_function is not None:
            Phi, F = self.Phi_F_updater_function(
                X=X_ndarray, U=U_ndarray,
                parameters_struct=parameters_struct)

            self.Phi_ndarray = Phi
            self.F_ndarray = F

    def _generate_exponential_A_list(self, A: sp.Matrix):
        """
        Generates a list of matrix powers of A up to Np.

        Args:
            A (sp.Matrix): The square matrix to be exponentiated.

        Returns:
            list: A list where the i-th element is A raised to the (i+1)-th power,
              i.e., [A, A^2, ..., A^Np].

        Notes:
            - Assumes self.Np is defined and is a positive integer.
            - Uses matrix multiplication to compute powers of A.
        """

        exponential_A_list = []

        for i in range(self.Np):
            if i == 0:
                exponential_A_list.append(A)
            else:
                exponential_A_list.append(exponential_A_list[i - 1] * A)

        return exponential_A_list

    def _build_F_expression(self, C: sp.Matrix) -> sp.Matrix:
        """
        Constructs the F expression matrix used in state-space model predictive control.

        Args:
            C (sp.Matrix): The output matrix of the state-space system.

        Returns:
            sp.Matrix: The F expression matrix of shape (OUTPUT_SIZE * Np, STATE_SIZE),
            where each block row corresponds to C multiplied by the state transition matrix
            raised to increasing powers.

        Notes:
            - self.OUTPUT_SIZE: Number of outputs in the system.
            - self.Np: Prediction horizon.
            - self.STATE_SIZE: Number of states in the system.
            - self._exponential_A_list: List of powers of the state transition matrix A,
              precomputed for each prediction step.
        """

        F_expression = sp.zeros(self.OUTPUT_SIZE * self.Np, self.STATE_SIZE)

        for i in range(self.Np):
            # C A^{i+1}
            F = C * self._exponential_A_list[i]

            F_expression[i * self.OUTPUT_SIZE:(i + 1) *
                         self.OUTPUT_SIZE, :] = F

        return F_expression

    def _build_Phi_expression(self, B: sp.Matrix, C: sp.Matrix) -> sp.Matrix:
        """
        Constructs the Phi expression matrix used in Model Predictive Control (MPC)
          for state-space models.

        The Phi matrix maps the sequence of future control inputs
          to the predicted outputs over the prediction horizon.
        It is built using the system input matrix `B`, output matrix `C`,
          and the precomputed list of powers of the
        system matrix `A` (stored in `self._exponential_A_list`).

        Args:
            B (sp.Matrix): The input matrix of the state-space model.
            C (sp.Matrix): The output matrix of the state-space model.

        Returns:
            sp.Matrix: The constructed Phi expression matrix of size
                       (OUTPUT_SIZE * Np, INPUT_SIZE * Nc),
                       where Np is the prediction horizon
                       and Nc is the control horizon.
        """

        Phi_expression = sp.zeros(self.OUTPUT_SIZE * self.Np,
                                  self.INPUT_SIZE * self.Nc)

        for i in range(self.Nc):
            for j in range(i, self.Np):
                exponent = j - i
                if exponent == 0:
                    blok = C * B
                else:
                    blok = C * \
                        self._exponential_A_list[exponent - 1] * B

                r0, c0 = j * self.OUTPUT_SIZE, i * self.INPUT_SIZE

                Phi_expression[r0:r0 + self.OUTPUT_SIZE,
                               c0:c0 + self.INPUT_SIZE] = blok

        return Phi_expression


class MPC_ReferenceTrajectory:
    """
    A class to handle the reference trajectory for Model Predictive Control (MPC).
    This class manages the reference vector, which can either be a single row vector
    or multiple row vectors, and provides a method to calculate the difference
    between the reference vector and the predicted state.
    Attributes:
        reference_vector (np.ndarray): The reference trajectory vector.
        Np (int): Prediction horizon.
        OUTPUT_SIZE (int): Number of outputs in the system.
        follow_flag (bool): Indicates whether the reference vector has multiple rows.
    """

    def __init__(self, reference_vector: np.ndarray, Np: int):
        if reference_vector.shape[1] == Np:
            self.follow_flag = True
        elif reference_vector.shape[1] == 1:
            self.follow_flag = False
        else:
            raise ValueError(
                "Reference vector must be either a single row vector or a Np row vectors.")

        self.reference_vector = reference_vector

        self.Np = Np
        self.OUTPUT_SIZE = reference_vector.shape[0]

    def calculate_dif(self, Fx: np.ndarray) -> np.ndarray:
        """
        Calculates the difference between the reference vector and the predicted state.
        Args:
            Fx (np.ndarray): The predicted state vector.
        Returns:
            np.ndarray: The difference vector, which is the reference vector minus the predicted state.
        """
        dif = np.zeros((self.Np * self.OUTPUT_SIZE, 1))

        if self.follow_flag:
            for i in range(self.Np):
                for j in range(self.OUTPUT_SIZE):
                    dif[i * self.OUTPUT_SIZE + j, :] = \
                        self.reference_vector[j, i] - \
                        Fx[i * self.OUTPUT_SIZE + j, :]
        else:
            for i in range(self.Np):
                for j in range(self.OUTPUT_SIZE):
                    dif[i * self.OUTPUT_SIZE + j, :] = \
                        self.reference_vector[j, :] - \
                        Fx[i * self.OUTPUT_SIZE + j, :]

        return dif
