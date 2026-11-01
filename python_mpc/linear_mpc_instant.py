"""
File: linear_mpc_instant.py

This module implements an Instant Model Predictive Control (iMPC) algorithm
for discrete-time linear time-invariant (LTI) systems with input constraints.
The iMPC method performs a fixed number of sub-stepped primal-dual gradient
flow updates per control step using a semi-implicit Euler discretization,
enabling real-time MPC without iterative QP solves.
Equality constraints (system dynamics and control horizon) are enforced via
null-space projection. Inequality constraints (input bounds) are handled
through a primal-dual update with a non-negativity safeguard on the dual
variable.

Reference:
    The algorithm is based on the iMPC framework using a continuous-time
    primal-dual optimization flow with sub-stepping for numerical stability.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_optimization'))

import numpy as np

from mpc_utility.state_space_utility import SymbolicStateSpace
from mpc_utility.linear_solver_utility import symbolic_to_numeric_matrix
from external_libraries.MCAP_python_control.python_control.kalman_filter import LinearKalmanFilter
from external_libraries.MCAP_python_control.python_control.kalman_filter import DelayedVectorObject

REFERENCE_CHANGED_TOL = 1e-6
ZETA_DEFAULT = 1000.0
SAFE_GP_EPS = 1e-30
NEAR_ZERO_RELATIVE_LIMIT_DEFAULT = 1e-10


def round_off_near_zero(matrix: np.ndarray, relative_limit: float = NEAR_ZERO_RELATIVE_LIMIT_DEFAULT) -> np.ndarray:
    """
    Round off values in the matrix that are near zero relative to the maximum absolute value in the matrix.

    Args:
        matrix: Input matrix to be processed.
        relative_limit: Relative threshold for rounding off near-zero values.

    Returns:
        A new matrix with near-zero values rounded off to zero.
    """
    max_abs_value = np.max(np.abs(matrix))
    if max_abs_value == 0:
        return matrix  # All values are already zero

    threshold = relative_limit * max_abs_value
    rounded_matrix = np.where(np.abs(matrix) < threshold, 0.0, matrix)
    return rounded_matrix


def avoid_zero_divide(in_value: float, division_min: float) -> float:
    """
    Avoid zero division by enforcing a minimum absolute value on the input.
    """
    if in_value < division_min:
        if in_value >= 0:
            return division_min
        elif in_value > -division_min:
            return -division_min

    return in_value


def _build_block_diagonal(block: np.ndarray, n: int) -> np.ndarray:
    """
    Build a block-diagonal matrix by repeating `block` n times.
    """
    size_r, size_c = block.shape
    result = np.zeros((size_r * n, size_c * n))
    for i in range(n):
        result[i * size_r:(i + 1) * size_r,
               i * size_c:(i + 1) * size_c] = block
    return result


class InstantMPC_LTI:
    """
    Instant Model Predictive Control (iMPC) for discrete-time LTI systems.

    This class implements an iMPC algorithm that performs a fixed number of
    sub-stepped primal-dual gradient flow updates per control step using a
    semi-implicit Euler discretization. The method is designed for discrete-time
    linear systems with input constraints. Equality constraints (system dynamics,
    control horizon) are enforced via null-space projection; inequality constraints
    (input bounds) are handled through a primal-dual update with a non-negativity
    safeguard on the dual variable. The iMPC approach provides real-time MPC
    capabilities without iterative QP solves at each step.
    """

    def __init__(
            self,
            state_space: SymbolicStateSpace,
            Np: int,
            Nc: int,
            Weight_U: np.ndarray,
            Weight_Y: np.ndarray,
            Q_kf: np.ndarray = None,
            R_kf: np.ndarray = None,
            is_reference_trajectory: bool = False,
            delta_U_min: np.ndarray = None,
            delta_U_max: np.ndarray = None,
            U_min: np.ndarray = None,
            U_max: np.ndarray = None,
            zeta: float = ZETA_DEFAULT,
            near_zero_relative_limit: float = NEAR_ZERO_RELATIVE_LIMIT_DEFAULT
    ):
        # ---- Validate ----
        if state_space.delta_time <= 0.0:
            raise ValueError("State space model must be discrete-time.")

        self.Number_of_Delay = state_space.Number_of_Delay
        if Np < self.Number_of_Delay:
            raise ValueError(
                "Prediction horizon Np must be greater than the number of delays.")

        if Nc > Np:
            raise ValueError(
                "Control horizon Nc must be less than or equal to prediction horizon Np.")

        # ---- Dimensions ----
        dt = state_space.delta_time
        Nx = state_space.STATE_SIZE
        Nu = state_space.INPUT_SIZE
        Ny = state_space.OUTPUT_SIZE

        self.dt = dt
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.Np = Np
        self.Nc = Nc
        self.is_reference_trajectory = is_reference_trajectory
        self.near_zero_relative_limit = near_zero_relative_limit

        # ---- Discrete-time plant matrices (Ad, Bd, C) ----
        Ad = np.array(symbolic_to_numeric_matrix(state_space.A), dtype=float)
        Bd = np.array(symbolic_to_numeric_matrix(state_space.B), dtype=float)
        C = np.array(symbolic_to_numeric_matrix(state_space.C), dtype=float)
        self.Ad = Ad
        self.Bd = Bd
        self.C = C

        # ---- Prediction model ----
        Ah = Ad.copy()
        Bh_mat = Bd.copy()

        # ---- delta_U formulation: augment state with embedded output integrator ----
        # Augmented state z = [delta_x; y] where delta_x = x_k - x_{k-1}, y = C @ x_k.
        # This matches the LTI_MPC augmented model and provides integral action.
        Ns = Nx + Ny  # augmented shooting state dimension
        Az = np.zeros((Ns, Ns))
        Az[:Nx, :Nx] = Ah
        Az[Nx:, :Nx] = C @ Ah
        Az[Nx:, Nx:] = np.eye(Ny)
        Bz = np.zeros((Ns, Nu))
        Bz[:Nx, :] = Bh_mat
        Bz[Nx:, :] = C @ Bh_mat
        self.Ah = Az
        self.Bh = Bz
        self.Ns = Ns

        # ---- Sequence sizes ----
        NU = Np * Nu
        NX = Np * Ns
        Nw = NU + NX
        self.NU = NU
        self.NX = NX
        self.Nw = Nw

        # ---- Objective: f(w) = 1/2 w^T P w ----
        Qk = np.diag(Weight_Y.flatten()) if Weight_Y.ndim == 1 else Weight_Y
        Rk = np.diag(Weight_U.flatten()) if Weight_U.ndim == 1 else Weight_U
        # Output = y part of augmented state: Cz = [0, I_Ny]
        Cz = np.zeros((Ny, Ns))
        Cz[:, Nx:] = np.eye(Ny)
        # Match LTI_MPC: effective Q = W_Y^T W_Y (Weight_Y applied
        # to both Phi and residual in least-squares formulation).
        WCz = Qk @ Cz
        Qk_state = WCz.T @ WCz
        Qf = _build_block_diagonal(Qk_state, Np)
        Rf = _build_block_diagonal(Rk, Np)
        P = np.block([
            [Rf, np.zeros((NU, NX))],
            [np.zeros((NX, NU)), Qf]
        ])
        # Per-step linear gradient map for reference trajectory tracking:
        self._WCzT_Qk = np.zeros((Ns, Ny))
        if self.is_reference_trajectory:
            # gradient of -(dr_k)^T Qk WCz z_k w.r.t. z_k = -WCz^T Qk dr_k
            self._WCzT_Qk = WCz.T @ Qk  # shape (Ns, Ny)

        # ---- Inequality constraints: g(w; x) <= 0 ----
        self._use_u_ub = (U_max is not None)
        self._use_u_lb = (U_min is not None)
        self._use_du_ub = (delta_U_max is not None)
        self._use_du_lb = (delta_U_min is not None)
        if U_max is not None:
            self.U_max = U_max
        if U_min is not None:
            self.U_min = U_min
        if delta_U_max is not None:
            self.delta_U_max = delta_U_max
        if delta_U_min is not None:
            self.delta_U_min = delta_U_min

        self.is_no_constraints = not (
            self._use_u_ub or self._use_u_lb or self._use_du_ub or self._use_du_lb)

        Ag_rows = []
        bg_parts = []
        # delta_U bounds on first NU elements of w
        if self._use_du_ub:
            Ag_rows.append(np.hstack([np.eye(NU), np.zeros((NU, NX))]))
            bg_parts.append(np.tile(delta_U_max, (Np, 1)))
        if self._use_du_lb:
            Ag_rows.append(np.hstack([-np.eye(NU), np.zeros((NU, NX))]))
            bg_parts.append(np.tile(-delta_U_min, (Np, 1)))
        # Absolute U bounds via cumulative sum on delta_U:
        # u_k = U_latest + Sigma_{j=0}^{k} delta_u_j  <= U_max
        # -> L_cum @ delta_U_vec <= U_max - U_latest (tiled)
        L_cum = np.zeros((NU, NU))
        for k in range(Np):
            for j in range(k + 1):
                L_cum[k * Nu:(k + 1) * Nu,
                      j * Nu:(j + 1) * Nu] = np.eye(Nu)
        if self._use_u_ub:
            Ag_rows.append(np.hstack([L_cum, np.zeros((NU, NX))]))
            bg_parts.append(np.tile(U_max, (Np, 1)))  # updated per step
        if self._use_u_lb:
            Ag_rows.append(np.hstack([-L_cum, np.zeros((NU, NX))]))
            bg_parts.append(np.tile(-U_min, (Np, 1)))  # updated per step

        if not self.is_no_constraints:
            self._Ag = np.vstack(Ag_rows)
            self._bg = np.vstack(bg_parts)
        else:
            self._Ag = np.zeros((1, Nw))
            self._bg = np.zeros((1, 1))
        Nmu = self._bg.shape[0]
        self.Nmu = Nmu

        # ---- Equality constraints: C_eq * w + D_eq * z0 = 0 ----
        Ah_s = self.Ah  # Az in delta_U mode, Ah in absolute mode
        Bh_s = self.Bh  # Bz in delta_U mode, Bh in absolute mode
        Cx = np.zeros((NX, NX))
        for i in range(Np):
            Cx[i * Ns:(i + 1) * Ns, i * Ns:(i + 1) * Ns] = -np.eye(Ns)
            if i > 0:
                Cx[i * Ns:(i + 1) * Ns, (i - 1) * Ns:i * Ns] = Ah_s
        Cu = _build_block_diagonal(Bh_s, Np)
        C_eq = np.hstack([Cu, Cx])
        D_eq = np.vstack([Ah_s, np.zeros(((Np - 1) * Ns, Ns))])
        # ---- Control horizon: delta_u_k = 0 for k >= Nc ----
        if Nc < Np:
            n_nc = (Np - Nc) * Nu
            C_nc = np.zeros((n_nc, Nw))
            for k in range(Nc, Np):
                for j in range(Nu):
                    C_nc[(k - Nc) * Nu + j, k * Nu + j] = 1.0
            C_eq = np.vstack([C_eq, C_nc])
            D_eq = np.vstack([D_eq, np.zeros((n_nc, Ns))])

        # ---- iMPC parameters ----

        # Sub-step size for numerical stability of the Euler integration.
        # With semi-implicit treatment of the projected Hessian, the remaining
        # cross-coupling terms require dtzeta_sub to be moderate (~ 1.0).
        dtzeta = dt * zeta
        dtzeta_sub_max = 1.0
        N_sub = max(1, int(np.ceil(dtzeta / dtzeta_sub_max)))
        self._N_sub = N_sub
        self._dtzeta_sub = dtzeta / N_sub

        # ---- Projection matrices ----
        Iw = np.eye(Nw)
        # C_pinv = C_eq.T @ inv(C_eq @ C_eq.T) is the right pseudoinverse of C_eq.
        # K and L share this common factor, so compute it once via solve (more stable than inv).
        C_pinv = np.linalg.solve(C_eq @ C_eq.T, C_eq).T
        K = Iw - C_pinv @ C_eq
        L = -C_pinv @ D_eq
        self._K = round_off_near_zero(
            K, relative_limit=self.near_zero_relative_limit)
        self._L = round_off_near_zero(
            L, relative_limit=self.near_zero_relative_limit)

        # Precompute projected Hessian and its semi-implicit inverse.
        P_K = round_off_near_zero(
            K @ P @ K, relative_limit=self.near_zero_relative_limit)
        self._q_K_matrix = round_off_near_zero(
            K @ P @ L, relative_limit=self.near_zero_relative_limit)  # maps xt to gradient offset

        # Solve the unconstrained QP via KKT system to obtain a linear map
        # w_unc = _w_unc_map @ xt, used to initialize w each step.
        # KKT: [P, C_eq^T; C_eq, 0] [w; lm] = [0; -D_eq] xt
        n_eq = C_eq.shape[0]
        eps_reg = 1e-10
        P_reg = P + eps_reg * Iw
        KKT = np.block([
            [P_reg, C_eq.T],
            [C_eq, np.zeros((n_eq, n_eq))]
        ])
        rhs_mat = np.vstack([np.zeros((Nw, Ns)), -D_eq])
        sol = np.linalg.solve(KKT, rhs_mat)
        self._w_unc_map = round_off_near_zero(
            sol[:Nw, :], relative_limit=self.near_zero_relative_limit)
        self._w_unc_traj_map = np.zeros((Nw, Nw))

        if self.is_reference_trajectory:
            # Warm-start correction map for trajectory linear term:
            # w_unc_traj = _w_unc_map @ xt + _w_unc_traj_map @ q_trajectory
            rhs_traj = np.vstack([np.eye(Nw), np.zeros((n_eq, Nw))])
            sol_traj = np.linalg.solve(KKT, rhs_traj)
            self._w_unc_traj_map = round_off_near_zero(
                sol_traj[:Nw, :], relative_limit=self.near_zero_relative_limit)

        M = Iw + self._dtzeta_sub * P_K
        self._M_sub_inv = round_off_near_zero(np.linalg.inv(
            M), relative_limit=self.near_zero_relative_limit)

        # Ag * K transposed, for fast gradient computation
        self._AgKT = round_off_near_zero(
            # shape (Nw, Nmu)
            (self._Ag @ K).T, relative_limit=self.near_zero_relative_limit)

        # ---- iMPC state initialization ----
        self._w = np.zeros((Nw, 1))
        self._mu = np.zeros((Nmu, 1))

        # ---- Kalman filter for state estimation ----
        self._Kalman_filter = self._initialize_kalman_filter(
            state_space, Q_kf, R_kf)

        # ---- Internal state ----
        self.X_inner_model = np.zeros((Nx, 1))
        self.U_latest = np.zeros((Nu, 1))

        self.Y_store = DelayedVectorObject(Ny, self.Number_of_Delay)

        # for delta_U reference change detection
        self._ref_prev = np.zeros((Ny, 1))

    def _initialize_kalman_filter(
            self,
            state_space: SymbolicStateSpace,
            Q_kf: np.ndarray,
            R_kf: np.ndarray
    ) -> LinearKalmanFilter:
        """
        Initialize a linear Kalman filter for state estimation.
        """
        Nx = state_space.STATE_SIZE
        Ny = state_space.OUTPUT_SIZE

        if Q_kf is None:
            Q_kf = np.eye(Nx)
        if R_kf is None:
            R_kf = np.eye(Ny)

        lkf = LinearKalmanFilter(
            A=symbolic_to_numeric_matrix(state_space.A),
            B=symbolic_to_numeric_matrix(state_space.B),
            C=symbolic_to_numeric_matrix(state_space.C),
            Q=Q_kf, R=R_kf,
            Number_of_Delay=state_space.Number_of_Delay)

        lkf.converge_G()
        return lkf

    def _compensate_X_Y_delay(self, X: np.ndarray, Y: np.ndarray):
        """
        Compensate for measurement delay using the Kalman filter.
        """
        if self.Number_of_Delay > 0:
            Y_measured = Y
            X_comp = self._Kalman_filter.get_x_hat_without_delay()
            Y_comp = self.C @ X_comp
            self.Y_store.push(Y_comp)
            Y_diff = Y_measured - self.Y_store.get()
            return X_comp, (Y_comp + Y_diff)
        else:
            return X, Y

    def update(self, reference: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Run one iMPC step: compute control input given reference and output measurement.

        This method matches the interface of ``LTI_MPC.update(reference, Y)``.

        Args:
            reference: Reference output vector (Ny, 1) or trajectory.
            Y: Measured output vector (Ny, 1).

        Returns:
            Control input U (Nu, 1).
        """
        # ---- State estimation via Kalman filter ----
        self._Kalman_filter.predict_and_update_with_fixed_G(
            self.U_latest, Y)
        X = self._Kalman_filter.x_hat
        X_compensated, _ = self._compensate_X_Y_delay(X, Y)

        x = X_compensated

        # ---- Compute reference offset ----
        ref_raw = np.array(reference)
        if self.is_reference_trajectory:
            if ref_raw.ndim == 1:
                ref_raw = ref_raw.reshape(-1, 1)
            if ref_raw.shape[1] == self.Np:
                reference_trajectory = ref_raw
            elif ref_raw.shape[1] == 1:
                reference_trajectory = np.tile(ref_raw, (1, self.Np))
            else:
                raise ValueError(
                    "For trajectory mode, reference must have shape "
                    "(Ny, 1) or (Ny, Np).")
            ref = reference_trajectory[:, 0:1]
        else:
            ref = ref_raw.reshape(-1, 1)
            reference_trajectory = None

        # delta_U mode: augmented initial condition z0 = [delta_x; y - ref]
        dx = x - self.X_inner_model
        y_dev = self.C @ x - ref
        xt = np.vstack([dx, y_dev])

        # Reset dual variable on reference change (compare base reference)
        if np.max(np.abs(self._ref_prev - ref)
                  ) > REFERENCE_CHANGED_TOL:
            self._w = np.zeros((self.Nw, 1))
            self._mu = np.zeros((self.Nmu, 1))
            self._ref_prev = ref.copy()

        # Update bg for absolute U bounds (depends on current U_latest)
        bg_parts = []
        if self._use_du_ub:
            bg_parts.append(np.tile(self.delta_U_max, (self.Np, 1)))
        if self._use_du_lb:
            bg_parts.append(np.tile(-self.delta_U_min, (self.Np, 1)))
        if self._use_u_ub:
            bg_parts.append(
                np.tile(self.U_max - self.U_latest, (self.Np, 1)))
        if self._use_u_lb:
            bg_parts.append(
                np.tile(self.U_latest - self.U_min, (self.Np, 1)))
        if not self.is_no_constraints:
            self._bg = np.vstack(bg_parts)

        # ---- iMPC algorithm with sub-stepping ----
        w = self._w
        mu = self._mu

        Nu = self.Nu
        dtzeta_sub = self._dtzeta_sub
        N_sub = self._N_sub
        bg = self._bg

        # Precompute projected gradient offset (constant across sub-steps)
        q_K = self._q_K_matrix @ xt

        # Trajectory linear correction: for step k in X_stack (0-indexed),
        # gradient of -dr_k^T Qk WCz z_{k} w.r.t. z_{k} = -WCz^T Qk dr_k,
        # where dr_k = reference_trajectory[:, k] - ref (reference deviation at step k).
        if reference_trajectory is not None:
            q_trajectory = np.zeros((self.Nw, 1))
            for k in range(self.Np):
                dr_k = reference_trajectory[:, k:k + 1] - ref
                q_trajectory[self.NU + k * self.Ns: self.NU + (k + 1) * self.Ns] += \
                    self._WCzT_Qk @ dr_k
            K_q_trajectory = self._K @ q_trajectory
            q_K_eff = q_K - K_q_trajectory
        else:
            q_trajectory = None
            q_K_eff = q_K

        # Initialize w from unconstrained QP solution each step
        w = self._w_unc_map @ xt
        if q_trajectory is not None:
            w = w + self._w_unc_traj_map @ q_trajectory

        # Precompute L@xt (constant across sub-steps)
        Lxt = self._L @ xt

        for _sub in range(N_sub):
            # ---- Projection and inequality constraints ----
            # Dense projection (C_eq includes Nc constraints)
            wproj = self._K @ w + Lxt

            mu_times_eta = np.zeros_like(mu)
            mu_new = mu.copy()

            if not self.is_no_constraints:
                g = self._Ag @ wproj - bg
                gp = np.maximum(0.0, g)
                for i in range(mu.shape[0]):
                    if mu[i, 0] > 0:
                        gp[i, 0] = g[i, 0]

                # ---- Mu non-negativity constraint (eta) ----
                dtzeta_gp = dtzeta_sub * gp
                eta = np.ones_like(mu)
                mask = ((mu + dtzeta_gp) < 0).flatten()
                if np.any(mask):
                    safe_gp = dtzeta_gp.copy()

                    for i in range(safe_gp.shape[0]):
                        safe_gp[i, 0] = avoid_zero_divide(
                            safe_gp[i, 0], SAFE_GP_EPS)

                    eta[mask] = (-mu[mask]) / safe_gp[mask]

                mu_new = mu + dtzeta_gp * eta
                mu_new = np.maximum(0.0, mu_new)
                mu_times_eta = mu * eta

            # ---- Semi-implicit w update ----
            # Projected gradient with precomputed Hessian inverse:
            # (I + dt*K*P*K) w_new = w - dt*(K*P*L*xt - K*q_trajectory + AgKT*mu*eta)
            rhs = w - dtzeta_sub * (q_K_eff + self._AgKT @ mu_times_eta)
            w_new = self._M_sub_inv @ rhs
            # Project back onto null space of C_eq
            w_new = self._K @ w_new

            w = w_new
            mu = mu_new

        # ---- Compute output from optimized w ----
        wproj = self._K @ w + Lxt
        u = self.U_latest + wproj[:Nu]

        # ---- Store updated iMPC state ----
        self._w = w
        self._mu = mu

        # ---- Update internal model state ----
        self.U_latest = u
        self.X_inner_model = X_compensated

        return self.U_latest
