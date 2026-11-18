# requirements: pip install selenium webdriver-manager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_example(url = "https://www.ansa.it/english/"):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--incognito")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h4")))
        elements = driver.find_elements(By.CSS_SELECTOR, "h4")
        results = [el.text.strip() for el in elements if el.text.strip()]
        for i, it in enumerate(results, 1):
            print(f"{i}. {it}")
    finally:
        driver.quit()

def search_example(query, n_results:int=5, timeout_s:float=10.0):
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
        for i, it in enumerate(results[:n_results], 1):
            print(f"{i}. {it}")
            print(10*"=")
    finally:
        driver.quit()

def scrape2_example(url, text_length:int=10, timeout_s:float=10.0):

    target_elements:list[str] = ["div"]

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--incognito")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.get(url)
        wait = WebDriverWait(driver, timeout_s)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "html")))
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        print(f"\nFrom [{url}]\n")
        with open("pippo.txt", mode="w", encoding='utf-8') as f:
            for elem in target_elements:
                f.write(10*"=" + f"{elem}" + 10*"=" + "\n")
                for tag in soup.find_all(target_elements):
                    text = tag.get_text(strip=True)
                    if len(text) > text_length:
                        f.write(text+ "\n")
                        f.write(10*"*" + "\n")
        print("DONE")
    finally:
        driver.quit()

if __name__ == "__main__":
    search_example("SIFT Patent")
    # scrape2_example("https://patents.google.com/patent/CN104866851A/en")
    

