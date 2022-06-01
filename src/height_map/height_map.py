import json
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

HEIGHT: int = 180
WIDTH: int = 330
RAW_DEPTH_VALS_PATH: str = (
    f"{os.path.dirname(os.path.abspath(__file__))}/data/raw_depth_vals.json"
)


def load_raw_depth_vals() -> Dict[str, float]:
    with open(RAW_DEPTH_VALS_PATH) as f:
        raw_depth_vals = json.load(f)
    return raw_depth_vals


def process_raw_depth_vals(
    raw_depth_vals: Dict[str, float]
) -> np.ndarray:
    depth_vals = np.zeros((HEIGHT, WIDTH), dtype=float)
    points = np.zeros((HEIGHT * WIDTH, 3), dtype=float)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            depth = raw_depth_vals[str((x, y))]
            depth_vals[y, x] = depth
            points[y * WIDTH + x] = x, y, depth
    return points


def least_squares(drawit: bool = True) -> np.ndarray:
    """From https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points.

    I preserved the mathematical logic and the plotting implementation from the
    given reference.
    """
    raw_depth_vals = load_raw_depth_vals()
    points = process_raw_depth_vals(raw_depth_vals)

    # Do fit
    b = points[:, 2].T
    A = np.block([points[:, 0:2], np.ones_like(points[:, 0:1])])

    # Manual solution
    fit = np.linalg.inv(A.T @ A) @ A.T @ b
    errors = b - A @ fit
    residual = np.linalg.norm(errors)

    print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("errors: \n", errors)
    print("residual:", residual)

    if drawit:
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

        # Plot raw data
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, color='b')

        # Plot plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_grid, y_grid = np.meshgrid(
            np.arange(xlim[0], xlim[1]),
            np.arange(ylim[0], ylim[1])
        )
        z_grid = fit[0] * x_grid + fit[1] * y_grid + fit[2]
        ax.plot_wireframe(x_grid, y_grid, z_grid, color='k')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    return fit


if __name__ == "__main__":
    a, b, c = least_squares()
