import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp
from dataclasses import dataclass


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

    # reference
    reference = np.array([[0.0]])
    reference_trajectory = np.tile(reference, (1, Np + 1))


if __name__ == "__main__":
    main()
