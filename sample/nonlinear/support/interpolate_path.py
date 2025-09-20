import csv
import math
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

import numpy as np


def wrap_pi(a):
    """
    wrap angle to [-pi, pi]
    """
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a


def ang_shortest_delta(a0, a1):
    """
    Return the shortest angle difference Δ = a1 - a0 normalized to [-pi, pi].
    """
    return math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))


def ang_lerp(a0, a1, t):
    """
    Linear interpolation of angles (shortest rotation).
    a(t) = a0 + t * Δ where Δ is normalized to [-pi, pi].
    The return value is wrapped to [-pi, pi].
    """
    return wrap_pi(a0 + t * ang_shortest_delta(a0, a1))


def read_path_csv(path):
    """
    Read path from CSV file with columns: x, y, yaw (in radians).
    """
    xs, ys, yaws = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            yaws.append(float(row["yaw"]))
    return np.array(xs), np.array(ys), np.array(yaws)


def interpolate_path(xs, ys, yaws, total_time, delta_time):
    """
    Interpolate path (x, y, yaw) given as arrays of points.
    The output is sampled at every delta_time, and the total duration is total_time.
    Yaw is interpolated considering the shortest rotation.
    """
    if total_time <= 0:
        raise ValueError("total_time must be > 0")
    if delta_time <= 0:
        raise ValueError("delta_time must be > 0")
    n = len(xs)
    if n == 0:
        raise ValueError("empty path")
    if n == 1:
        # 単一点なら定値で返す
        times = np.arange(0.0, total_time + 1e-12, delta_time)
        return (times,
                np.full_like(times, xs[0], dtype=float),
                np.full_like(times, ys[0], dtype=float),
                np.full_like(times, wrap_pi(yaws[0]), dtype=float))

    # 弧長パラメータ（累積距離）
    diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    cumdist = np.concatenate(([0.0], np.cumsum(diffs)))
    total_dist = float(cumdist[-1])

    # 正規化パラメータ param in [0,1]
    if total_dist == 0.0:
        # すべて同一点ならインデックスで代用
        param = np.linspace(0.0, 1.0, n)
    else:
        param = cumdist / total_dist

    # 出力の時間列と、その正規化位置（0..1）
    times = np.arange(0.0, total_time + 1e-12, delta_time)
    frac = np.clip(times / total_time, 0.0, 1.0)

    # どの区間に属するか（右側が境界）→ i は 0..n-2 にクリップ
    idx = np.searchsorted(param, frac, side="right") - 1
    idx = np.clip(idx, 0, n - 2)

    # 区間内の局所重み tlocal = (s - s_i) / (s_{i+1}-s_i)
    denom = (param[idx + 1] - param[idx])
    # ゼロ長区間（重複点）対策
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    tlocal = (frac - param[idx]) / safe_denom
    tlocal = np.clip(tlocal, 0.0, 1.0)

    # x, y は通常の線形補間
    xs0, xs1 = xs[idx], xs[idx + 1]
    ys0, ys1 = ys[idx], ys[idx + 1]
    xs_interp = xs0 + (xs1 - xs0) * tlocal
    ys_interp = ys0 + (ys1 - ys0) * tlocal

    # yaw は最短角差で補間（区間ごと）
    y0, y1 = yaws[idx], yaws[idx + 1]
    # ベクトル化した角度補間
    deltas = np.vectorize(ang_shortest_delta)(y0, y1)
    yaws_interp = y0 + deltas * tlocal
    # 最後に [-pi, pi] へ wrap
    yaws_wrapped = np.vectorize(wrap_pi)(yaws_interp)

    return times, xs_interp, ys_interp, yaws_wrapped


def interpolate_path_csv(input_path, delta_time, total_time=None):
    """
    Read path from a CSV file (columns: x, y, yaw in radians) and interpolate it.
    The output is sampled at every delta_time, and the total duration is total_time.
    If total_time is None, it is set to the path length assuming a nominal speed of 1.0.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    xs, ys, yaws = read_path_csv(input_path)

    diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) **
                    2) if len(xs) > 1 else np.array([])
    total_dist = float(diffs.sum()) if diffs.size > 0 else 0.0

    if total_time is None:
        total_time = total_dist if total_dist > 0.0 else 1.0

    times, xs_interp, ys_interp, yaws_wrapped = interpolate_path(
        xs, ys, yaws, total_time, delta_time
    )

    matplotlib.use('Agg')

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Times for original points (distribute along total_time)
    n = len(xs)
    if total_time > 0:
        times_orig = np.linspace(0.0, total_time, n)
    else:
        times_orig = np.arange(n)

    # X
    axs[0].plot(times_orig, xs, 'o-', label='original', markersize=4)
    axs[0].plot(times, xs_interp, '-', label='interpolated')
    axs[0].set_ylabel('x')
    axs[0].legend()

    # Y
    axs[1].plot(times_orig, ys, 'o-', label='original', markersize=4)
    axs[1].plot(times, ys_interp, '-', label='interpolated')
    axs[1].set_ylabel('y')
    axs[1].legend()

    # Yaw (wrap for original)
    from math import pi

    def wrap(a):
        return (a + pi) % (2.0 * pi) - pi

    yaws_orig_wrapped = np.vectorize(wrap)(yaws)

    axs[2].plot(times_orig, yaws_orig_wrapped, 'o-',
                label='original', markersize=4)
    axs[2].plot(times, yaws_wrapped, '-', label='interpolated')
    axs[2].set_ylabel('yaw')
    axs[2].set_xlabel('time [s]')
    axs[2].legend()

    fig.tight_layout()
    out_path = Path(os.path.join(os.getcwd(), "check_interpolated_path.png"))
    fig.savefig(out_path)
    plt.close(fig)

    return np.array(times).reshape(-1, 1), \
        np.array(xs_interp).reshape(-1, 1), \
        np.array(ys_interp).reshape(-1, 1), \
        np.array(yaws_wrapped).reshape(-1, 1)
