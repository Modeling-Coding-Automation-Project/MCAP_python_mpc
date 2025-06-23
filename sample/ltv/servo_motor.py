"""
File: servo_motor.py

This script demonstrates Model Predictive Control (MPC) for a DC servo motor system without constraints.
It models the servo motor as a linear time-invariant (LTI) system, discretizes the plant, sets up an MPC controller,
and simulates the closed-loop response to a pulse reference input.
The simulation results are visualized using a plotting utility.

References:
A. Bemporad and E. Mosca, "Fulfilling hard constraints in uncertain linear systems
 by reference managing," Automatica, vol. 34, no. 4, pp. 451-461, 1998.
"""

import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import sympy as sp
import control

from mpc_utility.state_space_utility import SymbolicStateSpace
from python_mpc.linear_mpc import LTV_MPC_NoConstraints

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter
from sample.simulation_manager.signal_edit.sampler import PulseGenerator

from mpc_utility.state_space_utility_deploy import StateSpaceUpdaterDeploy, ABCD_UPDATER_CLASS_NAME


class ServoMotorParameters:
    Lshaft = 1.0
    dshaft = 0.02
    shaftrho = 7850.0
    G = 81500.0 * 1.0e6

    tauam = 50.0 * 1.0e6

    Mmotor = 100.0
    Rmotor = 0.1

    Bmotor = 0.1
    R = 20.0
    Kt = 10.0

    gear = 20.0

    Bload = 25.0


def create_plant_model_ABCD():
    PI = math.pi

    Lshaft = sp.Symbol('Lshaft', real=True)
    dshaft = sp.Symbol('dshaft', real=True)
    shaftrho = sp.Symbol('shaftrho', real=True)
    G = sp.Symbol('G', real=True)

    Mmotor = sp.Symbol('Mmotor', real=True)
    Rmotor = sp.Symbol('Rmotor', real=True)
    Jmotor = 0.5 * Mmotor * Rmotor ** 2
    Bmotor = sp.Symbol('Bmotor', real=True)
    R = sp.Symbol('R', real=True)
    Kt = sp.Symbol('Kt', real=True)

    gear = sp.Symbol('gear', real=True)

    Jload = 50.0 * Jmotor
    Bload = sp.Symbol('Bload', real=True)

    Ip = PI / 32.0 * dshaft ** 4
    Kth = G * Ip / Lshaft
    Vshaft = PI * (dshaft ** 2) / 4.0 * Lshaft
    Mshaft = shaftrho * Vshaft
    Jshaft = Mshaft * 0.5 * (dshaft ** 2 / 4.0)

    JM = Jmotor
    JL = Jload + Jshaft

    A = sp.Matrix([[0.0, 1.0, 0.0, 0.0],
                  [-Kth / JL, -Bload / JL, Kth / (gear * JL), 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [Kth / (JM * gear), 0.0, -Kth / (JM * gear ** 2), -(Bmotor + Kt ** 2 / R) / JM]])
    A = sp.simplify(A)

    B = sp.Matrix([[0.0],
                  [0.0],
                  [0.0],
                  [Kt / (R * JM)]])
    B = sp.simplify(B)

    C = sp.Matrix([[1.0, 0.0, 0.0, 0.0],
                  [Kth, 0.0, -Kth / gear, 0.0]])
    C = sp.simplify(C)

    D = sp.Matrix([[0.0],
                  [0.0]])
    D = sp.simplify(D)

    return A, B, C, D


def discretize_state_space_euler(
        A: sp.Matrix, B: sp.Matrix, C: sp.Matrix, D: sp.Matrix, dt):

    Ad = sp.eye(A.shape[0]) + dt * A
    Ad = sp.simplify(Ad)

    Bd = dt * B
    Bd = sp.simplify(Bd)

    Cd = C
    Cd = sp.simplify(Cd)

    Dd = D
    Dd = sp.simplify(Dd)

    return Ad, Bd, Cd, Dd


def main():
    # %% create state-space model
    sym_A, sym_B, sym_C, sym_D = create_plant_model_ABCD()

    dt = 0.05
    Number_of_Delay = 0

    sym_Ad, sym_Bd, sym_Cd, sym_Dd = discretize_state_space_euler(
        sym_A, sym_B, sym_C, sym_D, dt)

    ideal_plant_model = SymbolicStateSpace(
        sym_Ad, sym_Bd, sym_Cd, delta_time=dt, Number_of_Delay=Number_of_Delay)

    parameters = ServoMotorParameters()

    StateSpaceUpdaterDeploy.create_write_ABCD_update_code(
        argument_struct=parameters,
        A=sym_Ad, B=sym_Bd, C=sym_Cd, class_name=ABCD_UPDATER_CLASS_NAME,
        file_name="servo_motor_plant_updater.py")

    Weight_U = np.diag([0.001])
    Weight_Y = np.diag([1.0, 0.005])

    Np = 20
    Nc = 2

    ltv_mpc = LTV_MPC_NoConstraints(ideal_plant_model, Np=Np, Nc=Nc,
                                    Weight_U=Weight_U, Weight_Y=Weight_Y)

    # # %% simulation
    # t_sim = 20.0
    # time = np.arange(0, t_sim, dt)

    # # create input signal
    # _, input_signal = PulseGenerator.sample_pulse(
    #     sampling_interval=dt,
    #     start_time=time[0],
    #     period=20.0,
    #     pulse_width=50.0,
    #     pulse_amplitude=1.0,
    #     duration=time[-1],
    # )

    # # real plant model
    # # You can change the characteristic with changing the A, B, C matrices
    # A = sys_d.A
    # B = sys_d.B
    # C = sys_d.C
    # # D = sys_d.D

    # X = np.array([[0.0],
    #               [0.0],
    #               [0.0],
    #               [0.0]])
    # Y = np.array([[0.0],
    #               [0.0]])
    # U = np.array([[0.0]])

    # plotter = SimulationPlotter()

    # y_measured = Y
    # y_store = [Y] * (Number_of_Delay + 1)
    # delay_index = 0

    # for i in range(len(time)):
    #     # system response
    #     X = A @ X + B @ U
    #     y_store[delay_index] = C @ X

    #     # system delay
    #     delay_index += 1
    #     if delay_index > Number_of_Delay:
    #         delay_index = 0

    #     y_measured = y_store[delay_index]

    #     # controller
    #     ref = np.array([[input_signal[i, 0]], [0.0]])
    #     U = lti_mpc.update(ref, y_measured)

    #     plotter.append_name(ref, "ref")
    #     plotter.append_name(U, "U")
    #     plotter.append_name(y_measured, "y_measured")
    #     plotter.append_name(X, "X")

    # plotter.assign("ref", position=(0, 0), column=0, row=0, x_sequence=time)
    # plotter.assign("y_measured", position=(0, 0),
    #                column=0, row=0, x_sequence=time)
    # plotter.assign("ref", position=(0, 1), column=1, row=0, x_sequence=time)
    # plotter.assign("y_measured", position=(0, 1),
    #                column=1, row=0, x_sequence=time)

    # plotter.assign("X", position=(1, 0), column=0, row=0, x_sequence=time)
    # plotter.assign("X", position=(1, 0), column=1, row=0, x_sequence=time)
    # plotter.assign("X", position=(1, 1), column=2, row=0, x_sequence=time)
    # plotter.assign("X", position=(2, 1), column=3, row=0, x_sequence=time)

    # plotter.assign_all("U", position=(2, 0), x_sequence=time)

    # plotter.plot("Servo Motor plant, MPC Response")


if __name__ == "__main__":
    main()
