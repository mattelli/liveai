"""Finance tools built on yfinance: data download and chart plotting."""

from pathlib import Path
from typing import Literal

from src.common import default_out_dir, dump_func_call, safe_filename

__all__ = [
    "ValidYF", "fetch_financial_data", "save_financial_chart",
    "select_financial_columns",
]

REQUIRED_FINANCIAL_COLUMNS = ("Date", "Open", "Close")


class ValidYF:
    """
    Yahoo Finance time period and time interval definitions
    """
    period = Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    interval = Literal['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '4h', '1d', '5d', '1wk', '1mo', '3mo']


def select_financial_columns(data, symbol: str):
    """Reduce a yfinance frame to ``['Date', 'Open', 'Close']``.

    Accepts both flat and MultiIndex column layouts. Returns a DataFrame, or an
    error string when the data is empty or a required column is missing.
    """
    import pandas as pd

    if data is None or data.empty:
        return f"No data found for {symbol}."
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(level[0]) for level in frame.columns]
    date_col = "Date" if "Date" in frame.columns else (
        "Datetime" if "Datetime" in frame.columns else None
    )
    if date_col is None and isinstance(frame.index, pd.DatetimeIndex):
        # yfinance keeps the date as the DataFrame index, not as a column:
        # promote it (reset_index() names it 'Date'/'Datetime', or 'index' if unnamed).
        frame = frame.reset_index()
        date_col = next(
            (c for c in ("Date", "Datetime", "index") if c in frame.columns), None
        )
    if date_col is None:
        return "Date is missing"
    frame = frame.rename(columns={date_col: "Date"})
    missing = [c for c in REQUIRED_FINANCIAL_COLUMNS if c not in frame.columns]
    if missing:
        return f"Not all the required columns {list(REQUIRED_FINANCIAL_COLUMNS)} are available."
    frame = frame[list(REQUIRED_FINANCIAL_COLUMNS)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    if getattr(frame["Date"].dt, "tz", None) is not None:
        frame["Date"] = frame["Date"].dt.tz_localize(None)
    return frame


@dump_func_call
def fetch_financial_data(
        symbol: str,
        interval: ValidYF.interval = '1mo',
        period: ValidYF.period = '1y',
        start_date: str = None,  # YYYY-MM-DD
        end_date: str = None,  # YYYY-MM-DD
        to_string: bool = True
    ) -> "pd.DataFrame | str":
    """
    Fetch closing price of a stock symbol within a time period with a time interval.
    Useful when user asks for getting financial info of a financial product given its symbol.
    """
    import yfinance as yf

    try:
        data = yf.download(symbol, period=period, interval=interval, start=start_date, end=end_date)
    except Exception as e:
        return f"Error: {e}"
    selected_data = select_financial_columns(data, symbol)
    if isinstance(selected_data, str):
        return selected_data
    if to_string:
        return selected_data.to_string()
    return selected_data


@dump_func_call
def save_financial_chart(
        symbol: str,
        interval: ValidYF.interval = '1mo',
        period: ValidYF.period = '1y',
        out_dir: str | None = None,
    ) -> str:
    """
    Fetch closing price of a stock symbol within a time period with a time interval.
    The closing price are plotted in a chart that is saved as image.
    Useful when user asks for plotting a financial product given its symbol.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

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
        data = fetch_financial_data(symbol, interval, period, to_string=False)
        if isinstance(data, str):
            return data
        out_directory = Path(out_dir) if out_dir is not None else default_out_dir()
        out_directory.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(
            data['Date'], data['Close'], label='Close Price', marker='o', linestyle='-', color='blue', markersize=5
        )
        set_axis()
        set_chart()
        plot_path = out_directory / f"{safe_filename(symbol)}.jpg"
        plt.savefig(fname=str(plot_path), format='jpg')
        plt.close()
        return f"Plot saved to '{plot_path}'"
    except Exception as e:
        return f"Error: {e}"