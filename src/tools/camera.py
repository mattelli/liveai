"""Camera tool: capture a frame from a webcam and save it as an image."""

from pathlib import Path

from src.common import dump_func_call

__all__ = ["CAPTURED_IMAGE", "capture_image"]

CAPTURED_IMAGE: str = "captured_image.jpg"

# Test seams (module-level so the public tool signatures stay LLM-friendly).
_VIDEO_CAPTURE_FACTORY = None
_IMAGE_WRITER = None


@dump_func_call
def capture_image(out_file: str = CAPTURED_IMAGE, camera_index: int = 0) -> str:
    """
    Captures a frame from a camera and saves it as an image file.
    Useful when user asks for taking a picture.
    """
    capture_factory = _VIDEO_CAPTURE_FACTORY
    imwrite = _IMAGE_WRITER
    if capture_factory is None or imwrite is None:
        import cv2
        capture_factory = capture_factory or cv2.VideoCapture
        imwrite = imwrite or cv2.imwrite

    camera = capture_factory(camera_index)
    if not camera.isOpened():
        return f"Webcam '{camera_index}' not available"
    ret, frame = camera.read()
    camera.release()
    if not ret:
        return "Unable to capture frame"
    if not imwrite(out_file, frame):
        return f"Unable to save image '{out_file}'"
    return str(Path(out_file).resolve())