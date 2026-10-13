# OpenCV program to detect a face in real time (OpenCV 5.x)
#
# NOTE: OpenCV 5 removed the legacy Haar cv2.CascadeClassifier from the main
# package (it moved to opencv-contrib's xobjdetect module). The built-in
# replacement is the DNN-based YuNet face detector, cv2.FaceDetectorYN,
# which is faster and more accurate than Haar cascades.
#
# The detector needs one small ONNX model next to this script:
#   face_detection_yunet_2023mar.onnx
# Download it from:
#   https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
import os

import cv2

MODEL_FILE = 'face_detection_yunet_2023mar.onnx'


def find_model(name):
    """Locate the ONNX model next to this script."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{name} not found at {path}. Download it from '
            'https://github.com/opencv/opencv_zoo/raw/main/models/'
            'face_detection_yunet/face_detection_yunet_2023mar.onnx'
        )
    return path


# Load the YuNet face detector. Input size is set per frame in the loop,
# because a camera can change or report its resolution late.
detector = cv2.FaceDetectorYN_create(
    find_model(MODEL_FILE), '', (0, 0), score_threshold=0.6
)

# capture frames from a camera
cap = cv2.VideoCapture(0)

while True:

    # reads frames from a camera
    ret, img = cap.read()
    if not ret or img is None:
        break

    # YuNet requires the input size to match the frame being processed
    detector.setInputSize((img.shape[1], img.shape[0]))

    # Detects faces. Each result row is:
    #   [x, y, w, h,
    #    right-eye x,y, left-eye x,y, nose-tip x,y,
    #    right-mouth x,y, left-mouth x,y,
    #    confidence]
    _, faces = detector.detect(img)

    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])

            # To draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # YuNet returns eye *landmarks* rather than eye bounding boxes,
            # so we mark the detected eyes (and the other keypoints) with
            # circles instead of Haar's eye rectangles.
            for i in range(4, 14, 2):
                eye_x, eye_y = int(face[i]), int(face[i + 1])
                cv2.circle(img, (eye_x, eye_y), 2, (0, 127, 255), 2)

    # Display an image in a window
    cv2.imshow('img', img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
