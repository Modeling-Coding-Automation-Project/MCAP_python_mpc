import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp
from dataclasses import dataclass

from python_mpc.nonlinear_mpc import NonlinearMPC_TwiceDifferentiable


def create_plant_model():
    theta, omega, u0, dt, a, b, c, d = sp.symbols(
        'theta omega u0 dt a b c d', real=True)

    theta_next = theta + dt * omega
    omega_dot = -a * sp.sin(theta) - b * omega + c * \
        sp.cos(theta) * u0 + d * (u0 ** 2)
    omega_next = omega + dt * omega_dot

    f = sp.Matrix([theta_next, omega_next])
    h = sp.Matrix([[theta]])

    x_syms = sp.Matrix([[theta], [omega]])
    u_syms = sp.Matrix([[u0]])

    return f, h, x_syms, u_syms


@dataclass
class Parameters:
    a: float = 9.81     # gravity/l over I scaling
    b: float = 0.3      # damping
    c: float = 1.2      # state-dependent control effectiveness: cos(theta)*u
    d: float = 0.10     # actuator nonlinearity: u^2
    dt: float = 0.05    # sampling time step


def main():
    # Create symbolic plant model
    f, h, x_syms, u_syms = create_plant_model()

    # system dimensions
    nx = x_syms.shape[0]
    nu = u_syms.shape[0]
    ny = h.shape[0]
    Np = 20

    # define parameters
    state_space_parameters = Parameters()

    # input bounds
    u_min = np.array([[-2.0]])
    u_max = np.array([[2.0]])

    # weights
    Weight_U = np.diag([0.05])
    Weight_X = np.diag([2.5, 0.5])
    Weight_Y = np.diag([2.5])

    Q_ekf = np.diag([1.0, 1.0])
    R_ekf = np.diag([1.0])

    # reference
    reference = np.array([[0.0]])

    # Nonlinear MPC object
    X_initial = np.array([[np.pi / 4.0], [0.0]])

    nmpc = NonlinearMPC_TwiceDifferentiable(
        delta_time=state_space_parameters.dt,
        X=x_syms,
        U=u_syms,
        X_initial=X_initial,
        fxu=f,
        hx=h,
        parameters_struct=state_space_parameters,
        Np=Np,
        Weight_U=Weight_U,
        Weight_X=Weight_X,
        Weight_Y=Weight_Y,
        U_min=u_min,
        U_max=u_max,
        Q_kf=Q_ekf,
        R_kf=R_ekf,
        Number_of_Delay=0
    )


if __name__ == "__main__":
    main()
