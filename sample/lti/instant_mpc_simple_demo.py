"""
File: instant_mpc_simple_demo.py

Demonstration of the Instant MPC (iMPC) controller
for a 2-state continuous-time LTI plant.

The controller drives the state from x0=[0, 0] to xr=[200/3, 5]
subject to an input upper bound u <= 160.

Plant (continuous-time):
    Ac = [[-4, -0.03], [0.75, -10]]
    Bc = [[2], [0]]
"""
from __future__ import annotations

import sys
import time as time_module
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.linalg import expm

from mpc_utility.state_space_utility import SymbolicStateSpace
from python_mpc.linear_mpc_instant import InstantMPC_LTI

from sample.simulation_manager.visualize.simulation_plotter_dash import SimulationPlotterDash
from sample.simulation_manager.signal_edit.sampler import PulseGenerator


def main():
    # %% Plant (continuous-time LTI)
    Ac = np.array([[-4.0, -0.03],
                   [0.75, -10.0]])
    Bc = np.array([[2.0],
                   [0.0]])
    Nx = Ac.shape[0]
    Nu = Bc.shape[1]

    # %% Simulation settings
    te = 6.0            # simulation time [s]
    dt = 0.01     # sampling period [s]
    Nt = int(round(te / dt)) + 1
    time_arr = dt * np.arange(Nt)

    # %% Discretize plant
    Ix = np.eye(Nx)
    Ad = expm(dt * Ac)
    Bd = np.linalg.solve(Ac, (Ad - Ix)) @ Bc

    # Output: full state measurement (C = I)
    C = np.eye(Nx)
    Ny = Nx

    # %% State space model for iMPC
    state_space = SymbolicStateSpace(
        Ad, Bd, C, delta_time=dt, Number_of_Delay=0)

    # %% Reference
    xr = np.array([[200.0 / 3.0],
                   [5.0]])

    # %% Weights
    Qk = 2.0 * np.diag([1.0, 1.0])    # state weight (Weight_Y, since C = I)
    Rk = 10.0 * np.diag([1.0])          # input weight (Weight_U)

    # %% Prediction horizon
    Np = 100
    Nc = 1

    # %% Input bounds
    U_max = np.array([[160.0]])
    U_min = np.array([[-160.0]])

    # %% iMPC parameters
    zeta = 1000.0

    # Kalman filter tuning (near-transparent for full-state measurement)
    Q_kf = 1.0e4 * np.eye(Nx)
    R_kf = 1.0e-4 * np.eye(Ny)

    # %% Create iMPC controller
    impc = InstantMPC_LTI(
        state_space, Np=Np, Nc=Nc,
        Weight_U=Rk, Weight_Y=Qk,
        Q_kf=Q_kf, R_kf=R_kf,
        U_max=U_max, U_min=U_min,
        zeta=zeta)

    # %% Simulation
    X = np.zeros((Nx, 1))     # initial state
    U = np.zeros((Nu, 1))     # initial input
    # create a sign signal that flips every 2 seconds using PulseGenerator
    _, input_signal = PulseGenerator.sample_pulse(
        sampling_interval=dt,
        start_time=time_arr[0],
        period=4.0,        # full cycle: + for 2s, - for 2s
        pulse_width=50.0,   # pulse ON duration (seconds)
        pulse_amplitude=1.0,
        duration=time_arr[-1],
    )

    # map pulse {0,1} -> sign {-1, +1}
    sign_signal = 2.0 * input_signal - 1.0

    reference = sign_signal[0, 0] * xr  # initial reference

    plotter = SimulationPlotterDash()

    t_start = time_module.time()

    for i in range(Nt):
        # measurement (full state, C = I)
        Y = C @ X

        # update reference to flip sign every 2s and run controller
        reference = sign_signal[i, 0] * xr
        U = impc.update(reference, Y)

        # logging
        plotter.append_name(X, "X")
        plotter.append_name(reference, "Xr")
        plotter.append_name(U, "U")
        plotter.append_name(U_min, "U_min")
        plotter.append_name(U_max, "U_max")

        # plant update
        X = Ad @ X + Bd @ U

    t_elapsed = time_module.time() - t_start
    print(f"Calc. time    : {t_elapsed:.5f} s / {Nt:8d} steps | "
          f"{t_elapsed / Nt * 1000:.6f} ms/step")

    # %% Visualization
    # State x1 and reference
    plotter.assign("X", position=(0, 0), column=0, row=0, x_sequence=time_arr)
    plotter.assign("Xr", position=(0, 0), column=0, row=0,
                   x_sequence=time_arr, line_style='--')

    # State x2 and reference
    plotter.assign("X", position=(0, 1), column=1, row=0, x_sequence=time_arr)
    plotter.assign("Xr", position=(0, 1), column=1, row=0,
                   x_sequence=time_arr, line_style='--')

    # Input u and upper bound
    plotter.assign("U", position=(1, 0), x_sequence=time_arr)
    plotter.assign("U_max", position=(1, 0),
                   x_sequence=time_arr, line_style='--')
    plotter.assign("U_min", position=(1, 0),
                   x_sequence=time_arr, line_style='--')

    plotter.plot("iMPC Demo: 2-State LTI Plant")


if __name__ == "__main__":
    main()
