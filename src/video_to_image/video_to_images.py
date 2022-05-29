import os
import time
import logging

import cv2

VIDEO_PATH: str = f"{os.path.dirname(os.path.abspath(__file__))}/videos/uoft-lanes-1.mp4"
IMAGE_DIR_PATH: str = f"{os.path.dirname(os.path.abspath(__file__))}/images"

logger = logging.getLogger("video_to_images")
logging.basicConfig(
    format='[%(levelname)s] [%(asctime)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %z',
    # TODO(@thedavidchu): change to logging.INFO once you are done debugging
    level=logging.DEBUG
)


def video_stem() -> str:
    """Get the stem of the basename of the video.

    E.g. in 'alpha/beta/gamma.delta.epsilon', return 'gamma.delta'.
    """
    # This is a function because I think it is less confusing to show that we
    # are breaking up the basename into the stem and the extension explicitly.
    stem, ext = os.path.splitext(os.path.basename(VIDEO_PATH))
    return stem


def show_images(every_nth_image: int = 10, delay_ms: int = 250) -> None:
    """Debugging function to show the images"""
    vidcap = cv2.VideoCapture(VIDEO_PATH)
    success, img = vidcap.read()
    num_img = 0

    start = time.perf_counter()
    while success:
        success, img = vidcap.read()

        if num_img % every_nth_image != 0:
            num_img += 1
            continue
        print(f"Image Number: {num_img}")
        # We name all of the images the same thing so that we have a smoother
        # experience.
        cv2.imshow("Image", img)
        cv2.setWindowTitle("Image", f"Image: {num_img}")
        if cv2.waitKey(delay_ms) == ord("q"):
            cv2.destroyAllWindows()
            break
        num_img += 1
    vidcap.release()
    t0 = time.perf_counter()
    logger.info(
        f"Took {t0 - start} s to iterate over {num_img} of {int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))} frames"
    )


def save_images(every_nth_image: int = 10, overwrite: bool = False) -> None:
    """Debugging function to show every N-th image in a video."""
    vidcap = cv2.VideoCapture(VIDEO_PATH)
    success, img = vidcap.read()
    num_img = 0

    start = time.perf_counter()
    while success:
        success, img = vidcap.read()

        # Admittedly, it would be a prudent design choice to take in an offset
        # as well. If we wanted to extract every 10th image, but offset by 3,
        # we would have to write more code.
        if num_img % every_nth_image != 0:
            num_img += 1
            continue
        img_path = f"{IMAGE_DIR_PATH}/{video_stem()}.{num_img}.jpg"
        # The order is crucial. We want to save if the path does not exist.
        logger.debug(
            f"{os.path.exists(img_path)} and not {overwrite} = {os.path.exists(img_path) and not overwrite}"
        )
        if os.path.exists(img_path) and not overwrite:
            num_img += 1
            continue
        logger.info(f"Saving {img_path}")
        cv2.imwrite(img_path, img)
        if cv2.waitKey(1) == ord("q"):
            break
        num_img += 1
    vidcap.release()
    t0 = time.perf_counter()
    logger.info(
        f"Took {t0 - start} s to iterate over {num_img} of {int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))} frames"
    )


def trim_video(total_num_frames: int = 100, overwrite: bool = False) -> None:
    """My attempt at shortening the video. This doesn't work though. :("""
    vidcap = cv2.VideoCapture(VIDEO_PATH)
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    output_video_path = f"{VIDEO_PATH}/{video_stem()}.0-{total_num_frames}.mp4"
    if os.path.exists(output_video_path) and not overwrite:
        logger.debug(f"{output_video_path} already exists. Not overwriting")
        return
    fourcc = cv2.VideoWriter_fourcc(*'MJP4')
    out = cv2.VideoWriter(output_video_path, fourcc=fourcc, fps=fps, frameSize=(width, height))

    for i in range(total_num_frames):
        success, img = vidcap.read()

        if not success:
            logger.info(f"End of video at frame {i}")
            break
        out.write(img)
        if cv2.waitKey(1) == ord('q'):
            break

    vidcap.release()
    out.release()


if __name__ == "__main__":
    logger.debug(f"Video exists: {os.path.exists(VIDEO_PATH)}")
    logger.debug(f"Image output directory exists: {os.path.exists(IMAGE_DIR_PATH)}")

    # save_images()
    trim_video()