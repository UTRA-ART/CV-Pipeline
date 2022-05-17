from typing import Tuple

import cv2
import numpy as np


CALIB_IMG_PATH: str = "images/calibration.png"
TEST_IMAGE_PATH: str = "images/test.png"
TOLERANCE: int = 10


def calibrate_green(calib_img_path: str, tol: int = 5) -> Tuple[
    Tuple[int, int], Tuple[int, int], Tuple[int, int]
]:
    img = cv2.imread(calib_img_path)
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    green_patch = img_hls

    hue = green_patch[:, :, 0]
    min_hue = np.min(hue) - tol
    max_hue = np.max(hue) + tol

    lum = green_patch[:, :, 1]
    min_lum = np.min(lum) - tol
    max_lum = np.max(lum) + tol

    sat = green_patch[:, :, 2]
    min_sat = np.min(sat) - tol
    max_sat = np.max(sat) + tol

    return (min_hue, max_hue), (min_lum, max_lum), (min_sat, max_sat)


def get_ramp(
    img_hls: np.ndarray,
    hue_range: Tuple[int, int],
    lum_range: Tuple[int, int],
    sat_range: Tuple[int, int]
) -> np.ndarray:
    hue = img_hls[:, :, 0]
    min_hue, max_hue = hue_range
    within_hue = np.logical_and(min_hue <= hue, hue <= max_hue)

    lum = img_hls[:, :, 1]
    min_lum, max_lum = lum_range
    within_lum = np.logical_and(min_lum <= lum, lum <= max_lum)

    sat = img_hls[:, :, 2]
    min_sat, max_sat = sat_range
    within_sat = np.logical_and(min_sat <= sat, sat <= max_sat)

    within_ramp = np.logical_and(within_hue, within_lum)
    within_ramp = np.logical_and(within_ramp, within_sat)
    return within_ramp.astype(np.uint8) * 255


def main():
    hue_range, lum_range, sat_range = calibrate_green(CALIB_IMG_PATH, TOLERANCE)
    img = cv2.imread(TEST_IMAGE_PATH)
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    within_ramp = get_ramp(img_hls, hue_range, lum_range, sat_range)
    cv2.imshow("Pixels Belonging to the Ramp", within_ramp)

    # End of program
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
