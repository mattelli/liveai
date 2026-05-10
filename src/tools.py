import platform
import cv2
import requests
import subprocess
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

CAPTURED_IMAGE:str = "captured_image.jpg"

class ValidYF:
    period = Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    interval = Literal['1m', '2m', ' 5m', '15m', '30m', '60m', '90m', '1h', '4h', '1d', '5d', '1wk', '1mo', '3mo']

def system_info() -> str:
    """
    Returns system information such as OS, CPU.
    Useful when user asks for system status or performance.
    """
    print("calling: system_info")
    try:
        os_info = platform.platform()
        processor_info = platform.processor()
        platform_info = platform.uname()
        return f"{platform_info.system} {platform_info.release} ({os_info}), Processor: {processor_info}"
    except Exception as e:
        return f"Error fetching system info: {e}"

def current_date_time() -> str:
    """
    Returns the current date and time. 
    Useful when user asks for time, date, or 'what day is it?'
    """
    print("calling: current_date_time")
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%A, %B %d, %Y at %I:%M %p, %Z")

def read_text_file(file: str) -> str:
    """
    Read a text file and return its contents as a string.
    Useful when user asks for content description.
    """
    print(f"calling: read_text_file({file})")
    file_path = Path(file).resolve()
    if file_path.exists() and file_path.is_file():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error: {exc!r}"
    return f"Unable to read '{file}'"

def write_text_file(file: str, content: str) -> str:
    """
    Write a text file with the given its path and string content.
    Useful when user asks for content generation, e.g. code generation.
    """
    print(f"calling: write_text_file({file})")
    dest_path = Path(file).resolve()
    if dest_path.exists and dest_path.is_file():
        return "File already exists."
    try:
        dest_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc!r}"
    return f"File created at '{dest_path.resolve()}'"

def get_working_directory() -> str:
    """
    Returns the current working directory.
    Useful when user asks for getting current directory files and directories, 
    or when user asks for launching commands.
    """
    return str(Path.cwd().resolve())

def run_command(cmd: str) -> str:
    """
    Run a command with a subprocess and wait for it to terminate. 
    It returns either the command output or error as string.
    Useful when user asks to work with files, directories, 
    or to execute commands for example: 
        git, wget, curl, cd, mkdir, mv, cp, find, grep, python, ping 
    and so on so fourth.
    """
    print(f"calling: run_command >> {cmd}")
    cmd_list = cmd.split()
    process = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path.cwd(),
        text=True,
        shell=True,
    )
    rcode = process.wait()
    stdout, stderr = process.communicate()
    if rcode != 0:
        return f"Return code [{rcode}]: {stderr}"
    return stdout

def webpage_headlines(url: str = "https://www.ansa.it/english/", text_length:int=5, timeout_s:int=10) -> str:
    """
    Automatic extraction of headlines from a URL. Returns headlines as string.
    Useful when user is asking to summarize the content of a webpage given its URL.
    """
    print(f"calling: webpage_headlines({url})")
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

def capture_image(camera_index:int=0, out_file:str=CAPTURED_IMAGE) -> str:
    """
    Captures a frame from a camera and saves it as an image file.
    Useful when user asks for taking a picture.
    """
    print(f"calling: capture_image({camera_index})")
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

def save_financial_chart(symbol: str, interval:ValidYF.interval, period:ValidYF.period) -> str:
    """
    Fetch closing price of a stock symbol within a time period with a time interval.
    The closing price are plotted in a chart that is saved as image.
    Useful when user asks for getting financial info of a financial product given its symbol. 
    """
    print(f"calling: save_financial_chart({symbol})")
    
    def fetch_data(
            symbol: str,
            interval: ValidYF.interval = '1mo',
            period: ValidYF.period = '1y',
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
    print(system_info())