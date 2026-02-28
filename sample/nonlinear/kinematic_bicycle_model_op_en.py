"""
File: kinematic_bicycle_model_op_en.py

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
from sample.nonlinear.support.interpolate_path import interpolate_path_csv
