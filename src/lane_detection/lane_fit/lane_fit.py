"""
Cluster segments of detected lane line and fit spline lines.

Maintainers
-----------
1. @Jason-Y000 (wrote the original)
2. @thedavidchu (refactored)
"""
import os
import time
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sklearn.cluster import DBSCAN


# The image path relative to this file
IMAGE_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/test/lane-line.png"

COLOUR_MAP: Dict[int, str] = {
    0: "g",
    1: "r",
    2: "y",
    3: "b",
    4: "m",
    5: "c",
    -1: "k",
}

# Add later to catch RankWarning
# import numpy as np
# import warnings
# x = [1]
# y = [2]
#
# with warnings.catch_warnings():
#     warnings.filterwarnings('error')
#     try:
#         coefficients = np.polyfit(x, y, 2)
#     except np.RankWarning:
#         print "not enough data"


def sort_by_cluster(labels: np.ndarray, data: np.ndarray) -> Dict[int, np.ndarray]:
    clusters = {}
    for label, pt in zip(labels, data):
        clusters.setdefault(label, []).append(pt)
    return {label: np.array(pts) for label, pts in clusters.items()}


def lane_fitting(points: np.ndarray) -> List[np.ndarray]:
    """Fitting lanes to a function with a variation on the sliding windows"""
    fit_points = []
    x_width = np.max(points[:, 1]) - np.min(points[:, 1])
    y_width = np.max(points[:, 0]) - np.min(points[:, 0])

    if x_width < 15 or y_width < 15:  # Hard-coded parameter, update maybe
        return [np.array([]), np.array([])]

    total_pts = len(points)
    num_windows = 20

    slice = int(total_pts // num_windows)

    # TODO: Instead of just using arbitrary slices, use local cluster like centers
    # to choose the points to be included in the average
    for n in range(num_windows):
        start_idx = n * slice
        end_idx = min((n + 1) * slice, total_pts)

        group = points[start_idx:end_idx]
        x_avg = np.mean(group, axis=0)[1]
        y_avg = np.mean(group, axis=0)[0]

        sigma_x = np.sqrt(np.sum(np.power(group[:, 1] - x_avg, 2)) / group.shape[0])
        sigma_y = np.sqrt(np.sum(np.power(group[:, 0] - y_avg, 2)) / group.shape[0])

        if (sigma_x < 5) and (sigma_y < 5):
            fit_points.append([y_avg, x_avg])

    if len(fit_points) == 0:
        return [np.array([]), np.array([])]

    fit_points = np.array(fit_points)

    x = fit_points[:, 1]
    y = fit_points[:, 0]
    tck, u = interpolate.splprep([x, y], k=3, s=32)
    out = interpolate.splev(u, tck)
    return out


def main(drawit: bool = False, timeit: bool = False) -> Dict[int, None]:
    if timeit:
        start = time.perf_counter()

    input = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    input_norm = input / 255

    rows = np.where(input_norm == 1)[0].reshape(-1, 1)
    cols = np.where(input_norm == 1)[1].reshape(-1, 1)
    coords = np.concatenate((rows, cols), axis=1)  # (y,x) points

    if timeit:
        t0 = time.perf_counter()
        print(f"Preprocessing: {t0 - start} s")

    clustering = DBSCAN(eps=15, min_samples=30).fit(coords)
    labels = clustering.labels_

    if timeit:
        t1 = time.perf_counter()
        print(f"Clustering: {t1 - t0} s")

    clusters = sort_by_cluster(labels, coords)

    if timeit:
        t2 = time.perf_counter()
        print(f"Group Clusters: {t2 - t1} s")

    lane_lines = {label: lane_fitting(pts) for label, pts in clusters.items()}

    if timeit:
        t3 = time.perf_counter()
        print(f"Fit Lanes: {t3 - t2} s")

    if drawit:
        for label, pts in clusters.items():
            # Accessing global variables is slow!
            # By default (i.e. if there are more than 6 clusters), draw in grey
            color = COLOUR_MAP.get(label, "grey")
            plt.scatter(pts[:, 1], pts[:, 0], c=color)

        for label, line_fit in lane_lines.items():
            if label == -1:
                continue
            x, y = line_fit
            plt.plot(x, y, c="k")
        plt.gca().invert_yaxis()
        plt.show()

    if timeit:
        end = time.perf_counter()
        print(f"Total time: {end - start=} s")

    return lane_lines


if __name__ == "__main__":
    main(drawit=True, timeit=True)
