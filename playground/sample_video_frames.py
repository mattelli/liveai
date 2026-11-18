import cv2
import pathlib

video_path = pathlib.Path.cwd() / ".videos" / "vid_4.mp4"
output_dir = pathlib.Path.cwd() / ".frames_4"
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit(f"Impossibile aprire il video: {video_path}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    filename = output_dir / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(filename, frame)
    frame_idx += 1

cap.release()
print(f"Estratti {frame_idx} frame in '{output_dir}'")
