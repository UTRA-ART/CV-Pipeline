"""
Ramp Detection Algorithm
========================

Description of Functionality
----------------------------
1. Sample the colour of the main body of the ramp. The image containing the
sample is stored in $CALIB_IMG_PATH.
2. Define a tolerance, within the bounds of which we accept that a pixel matches
the shade of green of the ramp.
3. Find all pixels in the image within the tolerance of green.
4. With the image of all the matching "green" pixels, we want to cluster the
shapes (I use contours because I am lazy).
5. We take all contours larger than a given area and discard contours that are
just the entire screen (sometimes happens).
6. We approximate each contour with a convex hull and bound the contour with a
(non-rotated) rectangle
7. For each of these combinations, we take the rectangles where the
corresponding convex hull makes up at least $IOU_THRESH of the area.
8. From all of the rectangles, we find the mid-x and lowest (as appears in the
image) y point. This is returned.

Immediately Foreseeable Limitations
-----------------------------------
1. To avoid accepting a contour that takes up the entirety of the image (e.g.
the contours are "inside-out"), we specify a maximum area that the ramp can be.
If we are very close to or on the ramp, then this may cause the model to stop
detecting the ramp.
2. We are so far unable to identify rotated-but-still-rectangular shapes.
3. If there is a wide variety of colours in the ramp, then we may be too
lenient with what we accept as part of the ramp.
"""

from typing import List, Tuple

import cv2
import numpy as np


SUBSTITUTE: str = "frontview"
CALIB_IMG_PATH: str = f"images/{SUBSTITUTE}_calibration.png"
TEST_IMAGE_PATH: str = f"images/{SUBSTITUTE}.png"
GREEN_TOLERANCE: int = 10
IOU_THRESHOLD: float = 0.3
MIN_AREA_RATIO: float = 0.05
MAX_AREA_RATIO: float = 0.9  # Is this too tight of a bound?


def calibrate_green(
    calib_img_path: str, tol: int = 5
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Calibrate the correct 'green' pixels.

    Improvements
    ------------
    1. Potentially separate the various tolerances. There are 6 potential
    tolerances:
        hue_up, hue_down, lum_up, lum_down, sat_up, sat_down
    """
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
    """
    Notes
    -----
    1. I tried this function with only hue, but it did not work (because of the
    grass in the "training" images). Maybe try this again in competition? In its
    current state, I do not believe there is any advantage over just comparing
    to RGB (although I haven't tested this hypothesis). My reasoning, however,
    is that both form a cube in 3-D of acceptable colours. Maybe the tolerance
    causes it to behave differently though. I'm sure there's a mathematical way
    to express the notion that I'm going at. That I think the two 3-D shapes
    that we create are shapes that a linear transform can transform between? No,
    I think more precisely, what is within the prism in one space remains in the
    prism in the second space after the transform.
    """
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
    min_area_ratio: float = 0.0,  # Minimum theoretical value
    max_area_ratio: float = 1.0,  # Maximum theoretical value
    threshold: int = 10,
    draw_contours: bool = False,
) -> List[np.ndarray]:
    # It may be useful to inspect the greyscale colour difference between the
    # ramp and the ground. Depending on which is lighter, the user may wish to
    # use either cv2.THRESH_BINARY (ramp is lighter??) or cv2.THRESH_BINARY_INV
    # (ramp is darker??).
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = img_gray.shape
    min_area = h * w * min_area_ratio
    max_area = h * w * max_area_ratio
    # Debug this file by inspecting the thresh image. This can be done with
    # `cv2.imshow("Window Name", thresh)`. A useful reference guide can be found
    # [here](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html).
    thresh = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # This _must_ be an odd number
        threshold,
    )
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort by area
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(cv2.convexHull(c)))
    large_contours = [
        c for c in sorted_contours if min_area < cv2.contourArea(c) < max_area
    ]

    if draw_contours:
        img_bgr_copy = img_bgr.copy()
        # Draw contours larger than min_area
        cv2.drawContours(img_bgr_copy, large_contours, -1, (255, 0, 0), 1)
        cv2.imshow(
            f"Large contours, {c=} "
            + "(green=top-10, blue=larger than min_area, red=largest)",
            img_bgr_copy,
        )

    return large_contours


def main(draw_contours: bool = False):
    hue_range, lum_range, sat_range = calibrate_green(CALIB_IMG_PATH, GREEN_TOLERANCE)
    img = cv2.imread(TEST_IMAGE_PATH)

    # Collect pixels that match green
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    matches_green = get_green(img_hls, hue_range, lum_range, sat_range)

    # We are expecting a 3-D tensor, so we stack the 2-D tensors
    matches_green_bgr = np.stack([matches_green, matches_green, matches_green], axis=2)
    height, width, _ = matches_green_bgr.shape
    contours = get_large_contours(
        matches_green_bgr,
        min_area_ratio=MIN_AREA_RATIO,
        max_area_ratio=MAX_AREA_RATIO,
        draw_contours=draw_contours,
    )

    # Process the contours
    # We take the convex hull to smooth out the divets in the contours and
    # simplify the computation.
    convex_hulls, bounding_rects, bounding_rot_rects = (
        [cv2.convexHull(c) for c in contours],
        [cv2.boundingRect(c) for c in contours],
        [np.int0(cv2.boxPoints(cv2.minAreaRect(c))) for c in contours],
    )
    tight_bounding_rects, tight_bounding_rot_rects = [
        r
        for c, r in zip(convex_hulls, bounding_rects)
        if cv2.contourArea(c) >= IOU_THRESHOLD * r[2] * r[3]
    ], [
        r
        for c, r in zip(convex_hulls, bounding_rot_rects)
        if cv2.contourArea(c) >= IOU_THRESHOLD * cv2.contourArea(r)
    ]

    # Group rectangles
    if tight_bounding_rects:
        full_bounding_rect = np.array(tight_bounding_rects)
        min_x, min_y, min_w, min_h = np.min(full_bounding_rect, axis=0)
        max_x = np.max(full_bounding_rect[:, 0] + full_bounding_rect[:, 2], axis=0)
        max_y = np.max(full_bounding_rect[:, 1] + full_bounding_rect[:, 3], axis=0)
        point_of_interest = ((min_x + max_x) // 2, max_y)

    if draw_contours:
        cv2.imshow("1 - Pixels Belonging to the Ramp", matches_green)

        cv2.drawContours(matches_green_bgr, convex_hulls, -1, (0, 255, 0), 1)
        for rect in tight_bounding_rects:
            x, y, dx, dy = rect
            cv2.rectangle(matches_green_bgr, (x, y), (x + dx, y + dy), (255, 0, 0), 1)
        cv2.drawContours(
            matches_green_bgr, tight_bounding_rot_rects, -1, (0, 0, 255), 1
        )
        cv2.imshow("2 - Boxes containing parts of the ramp", matches_green_bgr)

        if tight_bounding_rects:
            cv2.circle(img, point_of_interest, 5, (0, 0, 255), -1)
            cv2.imshow("3 - Point of Interest", img)

        # End of program
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if tight_bounding_rects:
        return point_of_interest
    return None, None


if __name__ == "__main__":
    point_of_interest = main(draw_contours=True)
    print(point_of_interest)
