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


def create_sparse_available(mat: sp.Matrix):
    """
    Converts a given SymPy matrix into a sparse binary matrix
      indicating the positions of nonzero elements.

    Parameters
    ----------
    mat : sympy.Matrix
        The input SymPy matrix to be analyzed.

    Returns
    -------
    sympy.SparseMatrix
        A sparse matrix of the same shape as `mat`,
            where each entry is 1 if the corresponding entry in `mat` is nonzero,
            and 0 otherwise.
    """

    if not isinstance(mat, sp.MatrixBase):
        raise ValueError("Input must be a sympy matrix.")

    numeric_matrix = np.zeros(
        (mat.shape[0], mat.shape[1]), dtype=int)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if float(mat[i, j]) != 0.0:
                numeric_matrix[i, j] = 1

    return sp.SparseMatrix(numeric_matrix)


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

    def substitute_ABC_numeric_expression(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        """
        Substitutes the provided numeric matrices A, B, and C into their respective symbolic expressions,
        and updates the corresponding numeric expressions and sparse availability flags.

        Args:
            A (np.ndarray): The numeric matrix to substitute for A.
            B (np.ndarray): The numeric matrix to substitute for B.
            C (np.ndarray): The numeric matrix to substitute for C.

        Side Effects:
            - Updates self.A_numeric_expression, self.B_numeric_expression, and self.C_numeric_expression
              with the substituted numeric expressions.
            - Updates self.A_SparseAvailable, self.B_SparseAvailable, and self.C_SparseAvailable
              to indicate the availability of sparse representations for the substituted matrices.
        """

        self.A_numeric_expression = self.substitute_numeric_matrix_expression(
            A)
        self.B_numeric_expression = self.substitute_numeric_matrix_expression(
            B)
        self.C_numeric_expression = self.substitute_numeric_matrix_expression(
            C)

        self.A_SparseAvailable = create_sparse_available(
            self.A_numeric_expression)
        self.B_SparseAvailable = create_sparse_available(
            self.B_numeric_expression)
        self.C_SparseAvailable = create_sparse_available(
            self.C_numeric_expression)

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
            self.A_numeric_expression, self.B_numeric_expression,
            self.C_numeric_expression)

        self.F_ndarray = symbolic_to_numeric_matrix(
            self.F_numeric_expression)
        self.Phi_ndarray = symbolic_to_numeric_matrix(
            self.Phi_numeric_expression)

    def build_matrices_numeric_expression(
            self, A: sp.Matrix, B: sp.Matrix, C: sp.Matrix) -> tuple:
        """
        Builds the F and Phi matrices based on the symbolic state-space model.
        Args:
            A (sp.Matrix): State matrix.
            B (sp.Matrix): Input matrix.
            C (sp.Matrix): Output matrix.
        """
        self.F_numeric_expression, self.F_SparseAvailable \
            = self.build_F_expression(
                A, C, self.A_SparseAvailable, self.C_SparseAvailable)
        self.Phi_numeric_expression, self.Phi_SparseAvailable \
            = self.build_Phi_expression(
                A, B, C,
                self.A_SparseAvailable, self.B_SparseAvailable, self.C_SparseAvailable)

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
        """
        Updates the state-space matrices Phi and F at runtime using an adaptive updater function.

        This method checks if a Phi_F_updater_function is defined. If so, it calls this function
        with the provided state and input arrays, as well as a parameters structure, to compute
        updated Phi and F matrices. The resulting matrices are then assigned to the instance
        attributes `Phi_ndarray` and `F_ndarray`.

        Args:
            parameters_struct: An object or structure containing parameters required by the updater function.
            X_ndarray (np.ndarray): The current state array.
            U_ndarray (np.ndarray): The current input array.

        Returns:
            None
        """

        if self.Phi_F_updater_function is not None:
            Phi, F = self.Phi_F_updater_function(
                X=X_ndarray, U=U_ndarray,
                parameters_struct=parameters_struct)

            self.Phi_ndarray = Phi
            self.F_ndarray = F

    def build_F_expression(
            self,
            A: sp.Matrix, C: sp.Matrix,
            A_SparseAvailable: sp.Matrix, C_SparseAvailable: sp.Matrix) -> sp.Matrix:
        """
        Constructs the F expression and its sparsity pattern
          for a given state-space system over a prediction horizon.

        Args:
            A (sp.Matrix): The state transition matrix of the system.
            C (sp.Matrix): The output matrix of the system.
            A_SparseAvailable (sp.Matrix): The sparsity pattern matrix corresponding to A.
            C_SparseAvailable (sp.Matrix): The sparsity pattern matrix corresponding to C.

        Returns:
            Tuple[sp.Matrix, sp.Matrix]:
                - F_expression: The block matrix stacking C * A^(i+1) for i in range(Np),
                  representing the predicted outputs over the horizon.
                - F_SparseAvailable: The corresponding sparsity pattern matrix for F_expression.

        Notes:
            - self.Np: The prediction horizon length.
            - self.OUTPUT_SIZE: The number of outputs.
            - self.STATE_SIZE: The number of states.
            - The method uses symbolic matrices from sympy (sp).
            - The sparsity pattern is processed using create_sparse_available.
        """

        F_expression = sp.zeros(self.OUTPUT_SIZE * self.Np, self.STATE_SIZE)
        F_SparseAvailable = sp.zeros(
            self.OUTPUT_SIZE * self.Np, self.STATE_SIZE)

        for i in range(self.Np):
            F = C * A**(i + 1)
            F_SA = C_SparseAvailable * A_SparseAvailable**(i + 1)

            F_expression[i * self.OUTPUT_SIZE:(i + 1) *
                         self.OUTPUT_SIZE, :] = F
            F_SparseAvailable[i * self.OUTPUT_SIZE:(i + 1) *
                              self.OUTPUT_SIZE, :] = F_SA

        F_SparseAvailable = create_sparse_available(F_SparseAvailable)

        return F_expression, F_SparseAvailable

    def build_Phi_expression(
            self, A: sp.Matrix, B: sp.Matrix, C: sp.Matrix,
            A_SparseAvailable: sp.Matrix,
            B_SparseAvailable: sp.Matrix,
            C_SparseAvailable: sp.Matrix) -> sp.Matrix:
        """
        Constructs the Phi expression and its corresponding sparse availability matrix for a state-space model
        over a prediction horizon.

        This method builds the block-lower-triangular Phi matrix used in Model Predictive Control (MPC),
        which relates future control moves to predicted outputs over the prediction horizon. It also constructs
        a corresponding matrix indicating the sparsity pattern of available elements.

        Args:
            A (sp.Matrix): State transition matrix of the system.
            B (sp.Matrix): Input matrix of the system.
            C (sp.Matrix): Output matrix of the system.
            A_SparseAvailable (sp.Matrix): Binary matrix indicating available (nonzero) elements in A.
            B_SparseAvailable (sp.Matrix): Binary matrix indicating available (nonzero) elements in B.
            C_SparseAvailable (sp.Matrix): Binary matrix indicating available (nonzero) elements in C.

        Returns:
            Tuple[sp.Matrix, sp.Matrix]:
                - Phi_expression: The constructed Phi matrix of size (OUTPUT_SIZE * Np, INPUT_SIZE * Nc).
                - Phi_SparseAvailable: The corresponding sparse availability matrix, indicating which elements
                    in Phi_expression are available (nonzero).
        """

        Phi_expression = sp.zeros(self.OUTPUT_SIZE * self.Np,
                                  self.INPUT_SIZE * self.Nc)
        Phi_SparseAvailable = sp.zeros(self.OUTPUT_SIZE * self.Np,
                                       self.INPUT_SIZE * self.Nc)

        for i in range(self.Nc):
            for j in range(i, self.Np):
                exponent = j - i
                if exponent == 0:
                    blok = C * B
                    blok_SA = C_SparseAvailable * B_SparseAvailable
                else:
                    blok = C * \
                        A**exponent * B
                    blok_SA = C_SparseAvailable * \
                        A_SparseAvailable**exponent * B_SparseAvailable

                r0, c0 = j * self.OUTPUT_SIZE, i * self.INPUT_SIZE

                Phi_expression[r0:r0 + self.OUTPUT_SIZE,
                               c0:c0 + self.INPUT_SIZE] = blok
                Phi_SparseAvailable[r0:r0 + self.OUTPUT_SIZE,
                                    c0:c0 + self.INPUT_SIZE] = blok_SA

        Phi_SparseAvailable = create_sparse_available(Phi_SparseAvailable)

        return Phi_expression, Phi_SparseAvailable


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
