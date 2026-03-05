"""
File: kinematic_bicycle_model_parking_op_en.py

This script implements and simulates a nonlinear Model Predictive Control (MPC)
system for a kinematic bicycle model using the PANOC/ALM optimization engine,
demonstrating a parking maneuver scenario.
The vehicle dynamics are symbolically derived using SymPy,
including the state-space and measurement models and their Jacobians.
The simulation runs a closed-loop control scenario,
where the MPC tracks a reference trajectory for vehicle position and
orientation while respecting input constraints.
The reference trajectory simulates a parking maneuver.
The code also visualizes the results using a custom plotter,
allowing analysis of the controller's performance over time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import sympy as sp
from dataclasses import dataclass

from python_mpc.nonlinear_mpc import NonlinearMPC_OptimizationEngine

from sample.simulation_manager.visualize.simulation_plotter_dash import SimulationPlotterDash


def create_plant_model():
    wheel_base = sp.Symbol('wheel_base', real=True,
                           positive=True, nonzero=True)
    delta_time = sp.Symbol('delta_time', real=True,
                           positive=True, nonzero=True)

    px, py, q0, q3 = sp.symbols('px py q0 q3', real=True)
    v, delta = sp.symbols('v delta', real=True)

    X = sp.Matrix([px, py, q0, q3])
    U = sp.Matrix([v, delta])

    dtheta = (v / wheel_base) * sp.tan(delta)
    half = sp.Rational(1, 2)
    dq0 = sp.cos(dtheta * delta_time * half)
    dq3 = sp.sin(dtheta * delta_time * half)

    q0_next = q0 * dq0 - q3 * dq3
    q3_next = q0 * dq3 + q3 * dq0

    f = sp.Matrix([
        px + delta_time * v * (2 * q0**2 - 1),
        py + delta_time * v * (2 * q3 * q0),
        q0_next,
        q3_next
    ])
    fxu = sp.simplify(f)

    Y = sp.Matrix([px, py, q0, q3])
    hx = Y

    return fxu, hx, X, U


@dataclass
class Parameters:
    wheel_base: float = 2.8   # [m]
    delta_time: float = 0.025  # [s]


def create_reference(delta_time: float, simulation_time: float):
    """Generate a parking maneuver reference trajectory using bicycle kinematics.

    Phase 0-2 s : v =  5 m/s, delta =   0 deg (straight forward)
    Phase 2-4 s : v =  5 m/s, delta =  25 deg (forward turning)
    Phase 4-6 s : v =  0 m/s, delta =   0 deg (stop)
    Phase 6-8 s : v = -5 m/s, delta = -25 deg (reverse turning)
    Phase 8-10 s: v =  0 m/s, delta =   0 deg (stop / parked)

    Returns:
        times           : (N, 1) time array [s]
        px_sequence     : (N, 1) x position [m]
        py_sequence     : (N, 1) y position [m]
        q0_sequence     : (N, 1) quaternion real part
        q3_sequence     : (N, 1) quaternion z-component
    """
    wheel_base = 2.8  # [m]
    n = round(simulation_time / delta_time)

    time = np.arange(n) * delta_time

    px_seq = np.zeros((n, 1))
    py_seq = np.zeros((n, 1))
    q0_seq = np.zeros((n, 1))
    q3_seq = np.zeros((n, 1))

    # Initial orientation: yaw = 0  →  q0 = cos(0/2) = 1, q3 = sin(0/2) = 0
    q0_seq[0, 0] = 1.0
    q3_seq[0, 0] = 0.0

    for i in range(1, n):
        t_prev = time[i - 1]

        if t_prev < 2.0:
            v_ref = 5.0
            delta_ref = 0.0
        elif t_prev < 4.0:
            v_ref = 5.0
            delta_ref = np.deg2rad(25.0)
        elif t_prev < 6.0:
            v_ref = 0.0
            delta_ref = 0.0
        elif t_prev < 8.0:
            v_ref = -5.0
            delta_ref = -np.deg2rad(25.0)
        else:
            v_ref = 0.0
            delta_ref = 0.0

        q0p = q0_seq[i - 1, 0]
        q3p = q3_seq[i - 1, 0]

        # Position update (consistent with bicycle model state equations)
        px_seq[i, 0] = px_seq[i - 1, 0] + delta_time * v_ref * (2 * q0p**2 - 1)
        py_seq[i, 0] = py_seq[i - 1, 0] + delta_time * v_ref * (2 * q3p * q0p)

        # Quaternion update
        dtheta = (v_ref / wheel_base) * np.tan(delta_ref)
        dq0 = np.cos(dtheta * delta_time * 0.5)
        dq3 = np.sin(dtheta * delta_time * 0.5)

        q0_next = q0p * dq0 - q3p * dq3
        q3_next = q0p * dq3 + q3p * dq0

        # Normalize to avoid numerical drift
        q_norm = np.sqrt(q0_next**2 + q3_next**2)
        q0_seq[i, 0] = q0_next / q_norm
        q3_seq[i, 0] = q3_next / q_norm

    return time.reshape(-1, 1), px_seq, py_seq, q0_seq, q3_seq


def main():
    # simulation setup
    simulation_time = 10.0
    delta_time = 0.025
    Number_of_Delay = 0

    fxu, hx, x_syms, u_syms = create_plant_model()

    OUTPUT_SIZE = hx.shape[0]

    # Prediction horizon
    Np = 16

    # define parameters
    state_space_parameters = Parameters()

    # input bounds
    U_min = np.array([[-5.0], [-np.deg2rad(30)]])
    U_max = np.array([[10.0], [np.deg2rad(30)]])

    # weights
    Weight_U = np.array([0.05, 0.05])
    Weight_Y = np.array([2.0, 2.0, 1.0, 1.0])

    Q_ekf = np.diag([1.0, 1.0, 1.0, 1.0])
    R_ekf = np.diag([1.0, 1.0, 1.0, 1.0])

    # Reference: parking maneuver trajectory
    times, px_reference_path, py_reference_path, q0_reference_path, q3_reference_path = \
        create_reference(delta_time=delta_time,
                         simulation_time=simulation_time)

    # Nonlinear MPC object
    X_initial = np.array([[px_reference_path[0, 0]],
                          [py_reference_path[0, 0]],
                          [q0_reference_path[0, 0]],
                          [q3_reference_path[0, 0]]])

    nmpc = NonlinearMPC_OptimizationEngine(
        delta_time=state_space_parameters.delta_time,
        X=x_syms,
        U=u_syms,
        X_initial=X_initial,
        fxu=fxu,
        hx=hx,
        parameters_struct=state_space_parameters,
        Np=Np,
        Weight_U=Weight_U,
        Weight_Y=Weight_Y,
        U_min=U_min,
        U_max=U_max,
        Q_kf=Q_ekf,
        R_kf=R_ekf,
        Number_of_Delay=Number_of_Delay,
    )

    x_true = X_initial
    u = np.array([[0.0], [0.0]])

    nmpc.set_solver_max_iteration(
        outer_max_iterations=10,
        inner_max_iterations=15
    )

    plotter = SimulationPlotterDash()

    y_measured = np.array([[0.0], [0.0], [0.0], [0.0]])
    y_store = [y_measured] * (Number_of_Delay + 1)
    delay_index = 0

    # simulation
    for i in range(round(simulation_time / delta_time)):
        # system response
        if i > 0:
            u = np.copy(u_from_mpc)

        x_true = nmpc.kalman_filter.state_function(
            x_true, u, state_space_parameters)

        q_norm = np.sqrt(x_true[2, 0]**2 + x_true[3, 0]**2)
        x_true[2, 0] = x_true[2, 0] / q_norm
        x_true[3, 0] = x_true[3, 0] / q_norm

        y_store[delay_index] = nmpc.kalman_filter.measurement_function(
            x_true, state_space_parameters)

        # system delay
        delay_index += 1
        if delay_index > Number_of_Delay:
            delay_index = 0

        y_measured = y_store[delay_index]

        # Reference for NMPC
        reference_path = np.zeros((OUTPUT_SIZE, Np))
        for j in range(Np):
            index = i + j
            if index >= px_reference_path.shape[0]:
                index = px_reference_path.shape[0] - 1

            reference_path[0, j] = px_reference_path[index, 0]
            reference_path[1, j] = py_reference_path[index, 0]
            reference_path[2, j] = q0_reference_path[index, 0]
            reference_path[3, j] = q3_reference_path[index, 0]

        u_from_mpc = nmpc.update_manipulation(reference_path, y_measured)

        # monitoring
        outer_solver_iteration, inner_solver_iteration = \
            nmpc.get_solver_step_iterated_number()

        px_reference = reference_path[0, 0]
        py_reference = reference_path[1, 0]
        yaw_reference = 2.0 * \
            np.arctan2(reference_path[3, 0], reference_path[2, 0])
        px_measured = y_measured[0, 0]
        py_measured = y_measured[1, 0]
        yaw_measured = 2.0 * np.arctan2(y_measured[3, 0], y_measured[2, 0])

        v_cmd = u_from_mpc[0, 0]
        delta_cmd = u_from_mpc[1, 0]

        plotter.append_name(px_reference, "px_reference")
        plotter.append_name(py_reference, "py_reference")
        plotter.append_name(yaw_reference, "yaw_reference")
        plotter.append_name(px_measured, "px_measured")
        plotter.append_name(py_measured, "py_measured")
        plotter.append_name(yaw_measured, "yaw_measured")
        plotter.append_name(v_cmd, "v")
        plotter.append_name(delta_cmd, "delta")
        plotter.append_name(outer_solver_iteration, "outer_solver_iteration")
        plotter.append_name(inner_solver_iteration, "inner_solver_iteration")

    plotter.assign("px_reference", column=0, row=0, position=(0, 0),
                   x_sequence=times, label="px_reference")
    plotter.assign("px_measured", column=0, row=0, position=(0, 0),
                   x_sequence=times, label="px_measured")
    plotter.assign("py_reference", column=0, row=0, position=(1, 0),
                   x_sequence=times, label="py_reference")
    plotter.assign("py_measured", column=0, row=0, position=(1, 0),
                   x_sequence=times, label="py_measured")
    plotter.assign("yaw_reference", column=0, row=0, position=(2, 0),
                   x_sequence=times, label="yaw_reference")
    plotter.assign("yaw_measured", column=0, row=0, position=(2, 0),
                   x_sequence=times, label="yaw_measured")
    plotter.assign("v", column=0, row=0, position=(0, 1),
                   x_sequence=times, label="v")
    plotter.assign("delta", column=0, row=0, position=(1, 1),
                   x_sequence=times, label="delta")
    plotter.assign("outer_solver_iteration", column=0, row=0, position=(2, 1),
                   x_sequence=times, label="outer_solver_iteration")
    plotter.assign("inner_solver_iteration", column=0, row=0, position=(2, 1),
                   x_sequence=times, label="inner_solver_iteration")

    plotter.plot()


if __name__ == "__main__":
    main()
