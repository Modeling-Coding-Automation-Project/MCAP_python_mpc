import csv
import math
from pathlib import Path

import numpy as np


def unwrap_angle(angle_vec):
    angle_vec_wrapped = np.zeros_like(angle_vec)
    angle_vec_wrapped[0] = angle_vec[0]

    for i in range(1, len(angle_vec)):
        dif = angle_vec[i] - angle_vec_wrapped[i - 1]
        if dif > math.pi:
            dif -= 2 * math.pi
        elif dif < -math.pi:
            dif += 2 * math.pi
        angle_vec_wrapped[i] = angle_vec_wrapped[i - 1] + dif

    return angle_vec_wrapped


def wrap_angle(angle_vec):
    angle_vec_wrapped = np.zeros_like(angle_vec)
    for i in range(len(angle_vec)):
        angle_vec_wrapped[i] = angle_vec[i]

        while angle_vec_wrapped[i] > 2 * math.pi:
            angle_vec_wrapped[i] -= 2 * math.pi
        while angle_vec_wrapped[i] < -2 * math.pi:
            angle_vec_wrapped[i] += 2 * math.pi

        if angle_vec_wrapped[i] > math.pi:
            angle_vec_wrapped[i] -= 2 * math.pi
        elif angle_vec_wrapped[i] < -math.pi:
            angle_vec_wrapped[i] += 2 * math.pi

    return angle_vec_wrapped


def read_path_csv(path):
    xs = []
    ys = []
    yaws = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row['x']))
            ys.append(float(row['y']))
            yaws.append(float(row['yaw']))
    return np.array(xs), np.array(ys), np.array(yaws)


def interpolate_path(xs, ys, yaws, total_time, delta_time):
    if total_time <= 0:
        raise ValueError('total_time must be > 0')
    if delta_time <= 0:
        raise ValueError('delta_time must be > 0')

    # Parameterize original path by cumulative distance (arc-length)
    diffs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    cumdist = np.concatenate(([0.0], np.cumsum(diffs)))
    total_dist = cumdist[-1]

    # Normalize to [0,1] along the path; if all points equal, fallback to index-based param
    if total_dist == 0:
        param = np.linspace(0.0, 1.0, len(xs))
    else:
        param = cumdist / total_dist

    # Desired output times and corresponding normalized parameter values (by time fraction)
    times = np.arange(0.0, total_time + 1e-12, delta_time)
    frac = times / total_time
    frac = np.clip(frac, 0.0, 1.0)

    # Interpolate x and y as functions of fraction
    xs_interp = np.interp(frac, param, xs)
    ys_interp = np.interp(frac, param, ys)

    # Handle yaw (angles) by unwrapping
    yaws_unwrapped = unwrap_angle(yaws)
    yaws_interp = np.interp(frac, param, yaws_unwrapped)
    # Wrap back to [-pi, pi]
    yaws_wrapped = wrap_angle(yaws_interp)

    return times, xs_interp, ys_interp, yaws_wrapped


def interpolate_path_csv(input_path, delta_time, total_time=None):
    """
    Read a path CSV, compute a sensible total_time from the path, and
    return/interpolate the path sampled every `delta_time` seconds.

    Assumption: we set the nominal speed to 1.0 so that time is equal to
    arc-length along the path. In other words total_time = total_path_length.

    Args:
        input_path (str or Path): path to the input CSV with columns x,y,yaw
        delta_time (float): desired sampling timestep for interpolated data

    Returns:
        times, xs_i, ys_i, yaws_i (numpy arrays)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_path}')

    xs, ys, yaws = read_path_csv(input_path)

    # compute cumulative distance (arc-length)
    diffs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    cumdist = np.concatenate(([0.0], np.cumsum(diffs)))
    total_dist = float(cumdist[-1])

    # Determine total_time: use provided value if given, otherwise infer
    # from path length (assume nominal speed = 1.0 so time = distance).
    if total_time is None:
        if total_dist <= 0.0:
            total_time = 1.0
        else:
            total_time = total_dist

    times, xs_i, ys_i, yaws_i = interpolate_path(
        xs, ys, yaws, total_time, delta_time)

    return times, xs_i, ys_i, yaws_i
