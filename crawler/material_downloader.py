import os
import time
import re
import requests
from urllib.parse import unquote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from .lms_scraper import LMS_URL, LMS_USERNAME, LMS_PASSWORD, wait_for_page_ready
from PyPDF2 import PdfReader

# ---------- Selectors ----------
POPUP_CLOSE_SELECTOR = (By.ID, "close-popup")
RESOURCE_LINK_SELECTOR = "a[href*='pluginfile.php']"
FLEXPAPER_SELECTOR = "//a[contains(@href,'/mod/flexpaper/view.php')]"
PRESENTATION_SELECTOR = "//a[contains(@href,'/mod/presentation/view.php')]"
# --------------------------------


def start_driver():
    opts = webdriver.ChromeOptions()
    # comment headless while debugging
    # opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--start-maximized")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


# Maximum allowed pages in a PDF
MAX_PAGES = 50  # change this as needed

def get_pdf_page_count(file_path):
    """
    Returns the number of pages in a PDF.
    Returns 0 if the file is not a readable PDF.
    """
    try:
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception as e:
        print(f"[!] Failed to read PDF {file_path}: {e}")
        return 0


def download_file(url, cookies, local_path, max_pages=MAX_PAGES):
    """
    Downloads a file using requests with cookies from Selenium session.
    If it's a PDF, counts the number of pages and deletes if it exceeds max_pages.
    """
    try:
        with requests.get(url, cookies=cookies, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"[+] Downloaded: {local_path}")

        # If the file is a PDF, check the number of pages
        if local_path.lower().endswith(".pdf"):
            pages = get_pdf_page_count(local_path)
            print(f"[i] PDF page count: {pages}")
            if pages > max_pages:
                os.remove(local_path)
                print(f"[!] Deleted {local_path} because it exceeds {max_pages} pages")

    except Exception as e:
        print(f"[!] Failed to download {url}: {e}")


def download_flexpaper_pdf(driver, flex_url, subject_title):
    """Extract and download PDF from FlexPaper viewer"""
    driver.get(flex_url)
    wait_for_page_ready(driver, 10)
    time.sleep(1)

    html = driver.page_source
    pdf_match = re.search(r"PDFFile\s*:\s*'([^']+)'", html)
    if pdf_match:
        pdf_url = pdf_match.group(1)
        filename = unquote(os.path.basename(pdf_url.split("?")[0]))
        folder = os.path.join("data", "materials", subject_title)
        os.makedirs(folder, exist_ok=True)
        local_path = os.path.join(folder, filename)

        cookies = {c['name']: c['value'] for c in driver.get_cookies()}

        try:
            download_file(pdf_url, cookies, local_path)
            print(f"[+] FlexPaper PDF: {filename}")
        except Exception as e:
            print(f"[!] Failed FlexPaper download: {e}")
    else:
        print("[!] No PDF URL found in FlexPaper viewer.")


def download_presentation_pdf(driver, presentation_url, subject_title):
    """Detects PDFs embedded in mod/presentation/view.php pages"""
    driver.get(presentation_url)
    wait_for_page_ready(driver, 10)
    time.sleep(1)

    html = driver.page_source
    # find the <object data="pluginfile.php...pdf">
    pdf_match = re.search(r'<object[^>]+data="([^"]*pluginfile\.php[^"]+\.pdf)"', html, re.IGNORECASE)
    if not pdf_match:
        # fallback: maybe inside a <a href="pluginfile.php">
        pdf_match = re.search(r'<a[^>]+href="([^"]*pluginfile\.php[^"]+\.pdf)"', html, re.IGNORECASE)

    if pdf_match:
        pdf_url = pdf_match.group(1)
        filename = unquote(os.path.basename(pdf_url.split("?")[0]))
        folder = os.path.join("data", "materials", subject_title)
        os.makedirs(folder, exist_ok=True)
        local_path = os.path.join(folder, filename)

        cookies = {c['name']: c['value'] for c in driver.get_cookies()}

        try:
            download_file(pdf_url, cookies, local_path)
            print(f"[+] Presentation PDF downloaded: {filename}")
            print(f"[✓] Saved: {local_path}")
        except Exception as e:
            print(f"[!] Failed Presentation download {pdf_url}: {e}")
    else:
        print("[!] No embedded PDF found in presentation page.")


def download_materials():
    os.makedirs("data/materials", exist_ok=True)
    driver = start_driver()
    wait = WebDriverWait(driver, 15)

    try:
        driver.get(LMS_URL)
        time.sleep(1)

        # --- LOGIN ---
        wait.until(EC.presence_of_element_located((By.NAME, "username"))).send_keys(LMS_USERNAME)
        wait.until(EC.element_to_be_clickable((By.ID, "loginbtn"))).click()
        wait.until(EC.presence_of_element_located((By.NAME, "password"))).send_keys(LMS_PASSWORD)
        wait.until(EC.element_to_be_clickable((By.ID, "loginbtn"))).click()
        time.sleep(2)

        # close popup if present
        try:
            wait.until(EC.element_to_be_clickable(POPUP_CLOSE_SELECTOR)).click()
            print("[✓] Popup closed")
        except Exception:
            print("[✓] No popup or already closed")

        import json
        with open("data/raw_lms_data.json", "r", encoding="utf-8") as f:
            subjects = json.load(f)

        for subj in subjects:
            title = subj.get("title", "Unknown Subject")
            url = subj.get("url")
            print(f"\n[*] Visiting subject: {title}")
            driver.get(url)
            wait_for_page_ready(driver, 5)
            time.sleep(1)

            # collect all resources, flexpapers, and presentations
            resources = driver.find_elements(By.CSS_SELECTOR, RESOURCE_LINK_SELECTOR)
            resource_hrefs = [r.get_attribute("href") for r in resources if r.get_attribute("href")]

            flex_links = driver.find_elements(By.XPATH, FLEXPAPER_SELECTOR)
            flex_hrefs = [a.get_attribute("href") for a in flex_links if a.get_attribute("href")]

            pres_links = driver.find_elements(By.XPATH, PRESENTATION_SELECTOR)
            pres_hrefs = [a.get_attribute("href") for a in pres_links if a.get_attribute("href")]

            if not (resource_hrefs or flex_hrefs or pres_hrefs):
                print("[DEBUG] No downloadable links found.")
                continue

            cookies = {c['name']: c['value'] for c in driver.get_cookies()}

            # --- Normal resources ---
            for href in resource_hrefs:
                try:
                    filename = unquote(os.path.basename(href.split("?")[0]))
                    folder = os.path.join("data", "materials", title)
                    os.makedirs(folder, exist_ok=True)
                    local_path = os.path.join(folder, filename)
                    download_file(href, cookies, local_path)
                    print(f"[+] Downloaded: {filename}")
                except Exception as e:
                    print(f"[!] Failed resource: {e}")

            # --- FlexPaper ---
            for href in flex_hrefs:
                try:
                    download_flexpaper_pdf(driver, href, title)
                except Exception as e:
                    print(f"[!] FlexPaper error: {e}")

            # --- Presentation plugin PDFs ---
            for href in pres_hrefs:
                try:
                    download_presentation_pdf(driver, href, title)
                except Exception as e:
                    print(f"[!] Presentation error: {e}")

        print("\n[✓] Finished downloading all materials")

    except Exception as e:
        print("[!] Error in material downloader:", e)

    finally:
        try:
            driver.quit()
        except Exception:
            pass
