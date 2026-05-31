import cv2

cap = cv2.VideoCapture(0)
WAIT_TIME_MS = 10

if not cap.isOpened():
    raise RuntimeError("Unable to open webcam")

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Unable to capture a frame")
        cv2.imshow("Webcam", frame)
        cv2.waitKey(WAIT_TIME_MS)
    except KeyboardInterrupt:
        print("👋 Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
