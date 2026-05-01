import cv2
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timezone


def current_date_time() -> str:
    """Returns the current date and time. Useful when user asks for time, date, or 'what day is it?'"""
    print("calling: current_date_time")
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%A, %B %d, %Y at %I:%M %p, %Z")

def extract_news(
        url: str = "https://www.ansa.it/english/", text_length:int=5, timeout_s:int=10
    ) -> str:
    """Automatic extraction of headlines from a URL."""
    print("calling: extract_news")
    headers:list[str] = ["h1", "h2"]
    try:
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
    except Exception as e:
        return f"Error fetching URL: {e}"
    soup = BeautifulSoup(r.text, "html.parser")
    headlines = []
    for tag in soup.find_all(headers):
        text = tag.get_text(strip=True)
        if len(text) > text_length:
            headlines.append(text)
    return f"\nFrom [{url}]\n"+"\n".join(headlines)

def capture_image(camera_index:int=0, out_file:str = "captured_image.jpg") -> str:
    """Capture a frame from a camera and save it as an image file"""
    print("calling: capture_image")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam '{camera_index}' not available")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Unable to capture frame")
    if not cv2.imwrite(out_file, frame):
        raise RuntimeError("Unable to save image '{out_file}'")
    return str(Path(out_file).resolve())
    
    
if __name__ == "__main__":
    print(capture_image())