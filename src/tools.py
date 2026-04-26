import requests
from bs4 import BeautifulSoup
from datetime import datetime


def current_date_time() -> str:
    """Returns the current date and time. Useful when user asks for time, date, or 'what day is it?'"""
    print("called: current_date_time")
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")
