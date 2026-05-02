import cv2
import requests
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import Literal

ValidPeriod = Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
ValidInterval = Literal['1m', '2m', ' 5m', '15m', '30m', '60m', '90m', '1h', '4h', '1d', '5d', '1wk', '1mo', '3mo']


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
        return f"Webcam '{camera_index}' not available"
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Unable to capture frame"
    if not cv2.imwrite(out_file, frame):
        return f"Unable to save image '{out_file}'"
    return str(Path(out_file).resolve())

def save_financial_chart(symbol: str, interval:ValidInterval, period:ValidPeriod) -> str:
    """
    Saves a plot of the closing price over time with all dates displayed on the x-axis.
    """
    print("calling: save_financial_chart")
    
    def fetch_data(
            symbol: str,
            interval: ValidInterval = '3mo',
            period: ValidPeriod = 'max',
            start_date: str = None,  # YYYY-MM-DD
            end_date: str = None  # YYYY-MM-DD
    ) -> pd.DataFrame:
        print(f"\nFetching data for [{symbol}] in the period [{period}] at interval [{interval}] ...\n")
        data = yf.download(symbol, period=period, interval=interval, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {symbol}.")
        data.reset_index(inplace=True)
        _date = ''
        if 'Date' in data.columns:
            _date = 'Date'
        elif 'Datetime' in data.columns:
            _date = 'Datetime'
        else:
            raise ValueError("Date are missing")
        if not pd.api.types.is_datetime64_any_dtype(data[_date]):
            data[_date] = pd.to_datetime(data[_date])
        if data[_date].dt.tz is not None:
            data[_date] = data[_date].dt.tz_localize(None)
        required_cols = ['Date', 'Open', 'Close']
        target_cols = [(_date, '')] + [(rc, symbol) for rc in required_cols[1:]]
        available_cols = [col for col in target_cols if col in data.columns]
        if len(available_cols) < 3:
            raise ValueError(f"Not all the required columns {required_cols} are available.")
        selected_data = data[target_cols]
        selected_data.columns = required_cols
        return selected_data
    
    def set_axis():
        date_locator = mdates.AutoDateLocator()
        date_formatter = mdates.AutoDateFormatter(date_locator)
        plt.gca().xaxis.set_major_locator(date_locator)
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.title(f'{symbol} Price Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price', fontsize=12)
        
    def set_chart():
        plt.grid(visible=True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
    try:
        data = fetch_data(symbol, interval, period)
        plt.figure(figsize=(12, 6))
        plt.plot(
            data['Date'], data['Close'], label='Close Price', marker='o', linestyle='-', color='blue', markersize=5
        )
        set_axis()
        set_chart()
        plot_path = str(Path().cwd() / f"{symbol}.jpg")
        plt.savefig(fname=plot_path, format='jpg')
        plt.close()
        return f"Plot saved to '{plot_path}'"
    except Exception as e:
        return f"Unable to plot data for symbol '{symbol}'"
    
    
if __name__ == "__main__":
    print(save_financial_chart('STLA', "1mo", "10y"))