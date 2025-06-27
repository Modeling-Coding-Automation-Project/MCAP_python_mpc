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

"""
Lshaft = 1.0;                               % シャフト長さ
dshaft = 0.02;                              % シャフト径
shaftrho = 7850;                            % シャフトの密度 (炭素鋼)
G = 81500 * 1e6;                            % 剛性率

tauam = 50 * 1e6;                           % 剪断強度

Mmotor = 100;                               % 回転子の質量
Rmotor = 0.1;                               % 回転子の半径
Jmotor = 0.5 * Mmotor * Rmotor ^ 2;         % 回転子軸に対する慣性モーメント
Bmotor = 0.1;                               % 回転子の粘性摩擦係数(A CASO)
R = 20;                                     % 接触子の抵抗
Kt = 10;                                    % モーター定数

gear = 20;                                  % ギア比

Jload = 50*Jmotor;                          % 負荷の公称慣性モーメント
Bload = 25;                                 % 負荷の公称粘性摩擦係数

Ip = pi / 32 * dshaft ^ 4;                  % シャフトの極モーメント
Kth = G * Ip / Lshaft;                      % ねじれ剛性 (トルク/角度)
Vshaft = pi * (dshaft ^ 2) / 4 * Lshaft;    % シャフトの体積
Mshaft = shaftrho * Vshaft;                 % シャフトの質量
Jshaft = Mshaft * 0.5 * (dshaft ^ 2 / 4);   % シャフトの慣性モーメント
"""


class ServoMotorParameters:
    Kt = 10.0       # Motor constant
    Bload = 25.0    # Load viscous friction coefficient


def create_plant_model_ABCD():
    PI = math.pi

    Lshaft = 1.0                        # Length of the shaft
    dshaft = 0.02                       # Diameter of the shaft
    # Density of the shaft material (carbon steel)
    shaftrho = 7850.0
    G = 81500.0 * 1.0e6                 # Shear modulus

    Mmotor = 100.0                      # Mass of the rotor
    Rmotor = 0.1                        # Radius of the rotor
    # Moment of inertia of the rotor about its axis
    Jmotor = 0.5 * Mmotor * Rmotor ** 2
    # Viscous friction coefficient of the rotor (A CASO)
    Bmotor = 0.1
    R = 20.0                            # Resistance of the contactor
    Kt = sp.Symbol('Kt', real=True)     # Motor constant

    gear = 20.0                         # Gear ratio

    Jload = 50.0 * Jmotor               # Nominal moment of inertia of the load
    # Nominal viscous friction coefficient of the load
    Bload = sp.Symbol('Bload', real=True)

    Ip = PI / 32.0 * dshaft ** 4         # Polar moment of inertia of the shaft
    Kth = G * Ip / Lshaft                # Torsional stiffness (Torque/Angle)
    Vshaft = PI * (dshaft ** 2) / 4.0 * Lshaft  # Volume of the shaft
    Mshaft = shaftrho * Vshaft           # Mass of the shaft
    # Moment of inertia of the shaft
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

    Weight_U = np.diag([0.001])
    Weight_Y = np.diag([1.0, 0.005])

    Np = 20
    Nc = 2

    ltv_mpc = LTV_MPC_NoConstraints(state_space=ideal_plant_model,
                                    parameters_struct=parameters,
                                    Np=Np, Nc=Nc,
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
