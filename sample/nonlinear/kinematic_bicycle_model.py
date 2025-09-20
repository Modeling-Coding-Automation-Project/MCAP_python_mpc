import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp
from dataclasses import dataclass

from python_mpc.nonlinear_mpc import NonlinearMPC_TwiceDifferentiable

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter


def create_plant_model():
    # --- symbols with assumptions ---
    wb = sp.Symbol('wb', real=True, positive=True, nonzero=True)
    dTime = sp.Symbol('dTime', real=True, positive=True, nonzero=True)

    px, py, q0, q3 = sp.symbols('px py q0 q3', real=True)
    v, delta = sp.symbols('v delta', real=True)

    X = sp.Matrix([px, py, q0, q3])
    U = sp.Matrix([v, delta])

    dtheta = (v / wb) * sp.tan(delta)
    half = sp.Rational(1, 2)
    dq0 = sp.cos(dtheta * dTime * half)
    dq3 = sp.sin(dtheta * dTime * half)

    q0_next = q0 * dq0 - q3 * dq3
    q3_next = q0 * dq3 + q3 * dq0

    f = sp.Matrix([
        px + dTime * v * (2 * q0**2 - 1),
        py + dTime * v * (2 * q3 * q0),
        q0_next,
        q3_next
    ])
    fxu = sp.simplify(f)

    yaw = 2 * sp.atan2(q3, q0)
    Y = sp.Matrix([px, py, yaw])
    hx = Y

    return fxu, hx, X, U


def main():
    fxu, hx, x_syms, u_syms = create_plant_model()


if __name__ == "__main__":
    main()
