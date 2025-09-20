import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp
from dataclasses import dataclass

from python_mpc.nonlinear_mpc import NonlinearMPC_TwiceDifferentiable

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter
from sample.nonlinear.support.interpolate_path import interpolate_path_csv


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

    times, xs_i, ys_i, yaws_i = interpolate_path_csv(
        input_path="./sample/nonlinear/support/office_area_RRT_path_data.csv",
        delta_time=0.1,
        total_time=60.0
    )

    plotter = SimulationPlotter()

    plotter.append_sequence_name(xs_i, "x_ref")
    plotter.append_sequence_name(ys_i, "y_ref")
    plotter.append_sequence_name(yaws_i, "yaw_ref")

    plotter.assign("x_ref", column=0, row=0, position=(0, 0),
                   x_sequence=times, label="x_ref")
    plotter.assign("y_ref", column=0, row=0, position=(1, 0),
                   x_sequence=times, label="y_ref")
    plotter.assign("yaw_ref", column=0, row=0, position=(2, 0),
                   x_sequence=times, label="yaw_ref")

    plotter.plot()


if __name__ == "__main__":
    main()
