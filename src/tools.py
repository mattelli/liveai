import re
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
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime, timezone
from typing import Literal

try:
    from src.common import dump_func_call
except:
    from common import dump_func_call

CAPTURED_IMAGE:str = "captured_image.jpg"

class ValidYF:
    """
    Yahoo Finance time period and time interval definitions
    """
    period = Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    interval = Literal['1m', '2m', ' 5m', '15m', '30m', '60m', '90m', '1h', '4h', '1d', '5d', '1wk', '1mo', '3mo']


def exit_gracefully():
    """
    Exits the program gracefully.
    Useful when user asks to quit or exit the application.
    """
    print("\nExiting gracefully. Bye! 👋")
    exit(0)

@dump_func_call
def system_info() -> str:
    """
    Returns system information such as OS, CPU.
    Useful when user asks for system status or performance.
    """
    try:
        os_info = platform.platform()
        processor_info = platform.processor()
        platform_info = platform.uname()
        return f"{platform_info.system} {platform_info.release} ({os_info}), Processor: {processor_info}"
    except Exception as e:
        return f"Error fetching system info: {e}"

@dump_func_call
def current_date_time() -> str:
    """
    Returns the current date and time. 
    Useful when user asks for time, date, or 'what day is it?'
    """
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%A, %B %d, %Y at %I:%M %p, %Z")

@dump_func_call
def read_text_file(file: str) -> str:
    """
    Read a text file and return its contents as a string.
    Useful when user asks for content description.
    """
    file_path = Path(file).resolve()
    if file_path.exists() and file_path.is_file():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error: {exc!r}"
    return f"Unable to read '{file}'"

@dump_func_call
def write_text_file(file: str, content: str) -> str:
    """
    Write a text file with the given its path and string content.
    Useful when user asks for content generation, e.g. code generation.
    """
    dest_path = Path(file).resolve()
    if dest_path.exists and dest_path.is_file():
        return "File already exists."
    try:
        dest_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc!r}"
    return f"File created at '{dest_path.resolve()}'"

@dump_func_call
def get_working_directory() -> str:
    """
    Returns the current working directory.
    Useful when user asks for getting current directory files and directories, 
    or when user asks for launching commands.
    """
    return str(Path.cwd().resolve())

@dump_func_call
def run_command(cmd: str) -> str:
    """
    Run a command with a subprocess and wait for it to terminate. 
    It returns either the command output or error as string.
    Useful when user asks to work with files, directories, 
    or to execute commands for example: 
        git, wget, curl, cd, mkdir, mv, cp, find, grep, python, ping 
    and so on so fourth.
    """
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

@dump_func_call
def webpage_headlines(url: str = "https://www.ansa.it/english/", relevant_text_length:int=20, timeout_s:int=10) -> str:
    """
    Automatic extraction of headlines from a URL. Returns headlines as string.
    Useful when user is asking to summarize the content of a webpage given its URL.
    """
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
        if len(text) > relevant_text_length:
            headlines.append(text)
    return f"\nFrom [{url}]\n"+"\n".join(headlines)

@dump_func_call
def web_search(query:str, n_results:int=5, timeout_s:float=10.0) -> str:
    """
    Search web pages based on the query.
    Useful when user asks for search on a topic.
    """
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--incognito")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(width=300, height=200)
    try:
        driver.get("https://duckduckgo.com/")
        wait = WebDriverWait(driver, timeout_s)
        search_bar = wait.until(EC.element_to_be_clickable((By.ID, "searchbox_input")))
        search_bar.send_keys(query)
        search_bar.submit()
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article")))
        elements = driver.find_elements(By.CSS_SELECTOR, "article")
        results = [el.text.strip() for el in elements if el.text.strip()]
        result_content = "\nSearch Results:\n"
        for i, it in enumerate(results[:n_results], 1):
            result_content += f"{i}) {it}\n"
            result_content += 50*"=" + "\n"
        return result_content
    except Exception as exc:
        return f"Error: {exc!r}"
    finally:
        driver.quit()

@dump_func_call
def capture_image(out_file:str=CAPTURED_IMAGE, camera_index:int=0, ) -> str:
    """
    Captures a frame from a camera and saves it as an image file.
    Useful when user asks for taking a picture.
    """
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

@dump_func_call
def save_financial_chart(symbol: str, interval:ValidYF.interval, period:ValidYF.period) -> str:
    """
    Fetch closing price of a stock symbol within a time period with a time interval.
    The closing price are plotted in a chart that is saved as image.
    Useful when user asks for getting financial info of a financial product given its symbol. 
    """

    def fetch_data(
            symbol: str,
            interval: ValidYF.interval = '1mo',
            period: ValidYF.period = '1y',
            start_date: str = None,  # YYYY-MM-DD
            end_date: str = None  # YYYY-MM-DD
    ) -> pd.DataFrame:
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

@dump_func_call
def search_pattern_in_files(
    pattern: str, 
    directory: str, 
    include_extensions: str="", 
    exclude_extensions: str=""
    ) -> str:
    """Search for a pattern in files within a directory recursively.
    Args:
        directory (str): The root directory to search in.
        pattern (str): The pattern to search for.
        include_extensions (str): Comma separated file extensions to include.
        exclude_extensions (str): Comma separated  file extensions to exclude.
    Returns:
        String of dict[str, list[int]]: Dictionary with file paths as keys and list of line numbers as values.
    """
    include_extensions_list = [ext.strip().lower() for ext in include_extensions.split(",")] if include_extensions else [] 
    exclude_extensions_list = [ext.strip().lower() for ext in exclude_extensions.split(",")] if exclude_extensions else [] 
    regex = re.compile(pattern)
    results:dict[str, list[int]] = {}
    search_dir = Path(directory).resolve()
    if not search_dir.exists():
        return f"'{search_dir}' does not exist!"
    if not search_dir.is_dir():
        return f"'{search_dir}' is not a directory!"
    for file_path in search_dir.rglob("*"):
        if file_path.is_dir():
            continue
        ext = file_path.suffix.lower()
        if include_extensions_list and ext not in include_extensions_list:
            continue
        if exclude_extensions_list and ext in exclude_extensions_list:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines):
                    if regex.search(line):
                        if file_path not in results:
                            results[file_path] = []
                        results[file_path].append(line_num + 1)  # Line numbers are 1-based
        except Exception as e:
            continue
    if not results:
        return f"Did not find file having pattern '{pattern}' in directory '{search_dir}' "
    formatted_res = f"Found '{len(results)}' files having pattern '{pattern}' in directory '{search_dir}' :\n"
    for path, lines in results.items():
        formatted_res += f"- in file '{str(path)}' at lines: " + ", ".join(map(str, lines))  + "\n"
    return formatted_res


if __name__ == "__main__":
    print(search_pattern_in_files("error", "./src"))
