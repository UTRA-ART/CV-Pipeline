from typing import List, Tuple

import cv2
import numpy as np


CALIB_IMG_PATH: str = "images/calibration.png"
TEST_IMAGE_PATH: str = "images/test.png"
TOLERANCE: int = 10


def calibrate_green(
    calib_img_path: str, tol: int = 5
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
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


def get_green(
    img_hls: np.ndarray,
    hue_range: Tuple[int, int],
    lum_range: Tuple[int, int],
    sat_range: Tuple[int, int],
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


def get_large_contours(
    img_bgr: np.ndarray,
    min_area: float = 30.0,
    c: int = 10,
    draw_contours: bool = False,
) -> List[np.ndarray]:
    """
    We want the
    """
    # It may be useful to inspect the greyscale colour difference between the
    # ramp and the ground. Depending on which is lighter, the user may wish to
    # use either cv2.THRESH_BINARY (ramp is lighter??) or cv2.THRESH_BINARY_INV
    # (ramp is darker??).
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(img_gray.shape)

    # Debug this file by inspecting the thresh image. This can be done with
    # `cv2.imshow("Window Name", thresh)`. A useful reference guide can be found
    # [here](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html).
    thresh = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # This _must_ be an odd number
        c,
    )
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    sorted_contours = sorted(contours, key=cv2.contourArea)

    # Sort by area
    large_contours = [c for c in sorted_contours if cv2.contourArea(c) > min_area]

    if draw_contours:
        img_bgr_copy = img_bgr.copy()
        # Drop top 10 largest contours
        cv2.drawContours(img_bgr_copy, sorted_contours[-10:], -1, (0, 255, 0), 1)
        # Draw contours larger than min_area
        cv2.drawContours(img_bgr_copy, large_contours, -1, (255, 0, 0), 1)
        cv2.drawContours(
            img_bgr_copy, large_contours, len(large_contours) - 1, (0, 0, 255), 1
        )
        cv2.imshow(
            f"Large contours, {c=} (green=top-10, blue=larger than min_area, red=largest",
            img_bgr_copy,
        )

    return large_contours


def main():
    hue_range, lum_range, sat_range = calibrate_green(CALIB_IMG_PATH, TOLERANCE)
    img = cv2.imread(TEST_IMAGE_PATH)

    # Green Image
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    within_ramp = get_green(img_hls, hue_range, lum_range, sat_range)
    cv2.imshow("Pixels Belonging to the Ramp", within_ramp)

    within_ramp_bgr = np.stack([within_ramp, within_ramp, within_ramp], axis=2)
    contours = get_large_contours(within_ramp_bgr, 1000, draw_contours=True)

    # End of program
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
