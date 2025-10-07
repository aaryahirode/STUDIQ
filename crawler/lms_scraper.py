# crawler/lms_scraper.py
import os
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

load_dotenv()

LMS_URL = os.getenv("LMS_URL")
LMS_USERNAME = os.getenv("LMS_USERNAME")
LMS_PASSWORD = os.getenv("LMS_PASSWORD")

# ---------- SELECTORS ----------
LOGIN_USERNAME_SELECTOR = (By.NAME, "username")
LOGIN_PASSWORD_SELECTOR = (By.NAME, "password")
NEXT_BUTTON_SELECTOR = (By.ID, "loginbtn")
LOGIN_BUTTON_SELECTOR = (By.ID, "loginbtn")
POPUP_CLOSE_SELECTOR = (By.ID, "close-popup")
TABLE_SELECTOR = (By.ID, "tbl-subject")
NEXT_PAGE_SELECTOR = (By.LINK_TEXT, "Next")
LAUNCH_LINK_SELECTOR = "a.launchbutton"  # selector for launch links on the dashboard
# --------------------------------


def start_driver():
    opts = webdriver.ChromeOptions()
    # comment out the next line while debugging if you want to see the UI
    # opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--start-maximized")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


def wait_for_page_ready(driver, timeout=10):
    """Wait until document.readyState == 'complete'"""
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def scrape_lms():
    os.makedirs("data", exist_ok=True)
    driver = start_driver()
    wait = WebDriverWait(driver, 25)

    driver.get(LMS_URL)
    time.sleep(1)

    try:
        # --- LOGIN ---
        print("[*] Entering username...")
        wait.until(EC.presence_of_element_located(LOGIN_USERNAME_SELECTOR)).send_keys(LMS_USERNAME)

        print("[*] Clicking Next button...")
        wait.until(EC.element_to_be_clickable(NEXT_BUTTON_SELECTOR)).click()

        print("[*] Waiting for password field...")
        wait.until(EC.presence_of_element_located(LOGIN_PASSWORD_SELECTOR)).send_keys(LMS_PASSWORD)

        print("[*] Clicking Login button...")
        wait.until(EC.element_to_be_clickable(LOGIN_BUTTON_SELECTOR)).click()

        # wait for dashboard
        print("[*] Waiting for dashboard to load...")
        # wait until either the subject table or a launch link appears
        wait.until(
            lambda d: (len(d.find_elements(By.CSS_SELECTOR, LAUNCH_LINK_SELECTOR)) > 0)
            or d.find_elements(*TABLE_SELECTOR)
        )
        time.sleep(2)
        wait_for_page_ready(driver, timeout=10)
        print("[✓] Dashboard initial content present")

        # --- close popup if present ---
        try:
            wait.until(EC.element_to_be_clickable(POPUP_CLOSE_SELECTOR)).click()
            print("[✓] Popup closed")
            time.sleep(1)
        except Exception:
            print("[!] No popup (or couldn't close) — continuing")

        # --- iterate pages and subjects ---
        subjects = []
        seen_links = set()
        page_index = 1
        max_pages = 60  # safety cap

        while page_index <= max_pages:
            print(f"\n[*] Scraping dashboard page {page_index} ...")

            # Wait for launch links to be present on the page (if none appear, try to parse table anchors)
            try:
                wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, LAUNCH_LINK_SELECTOR)) > 0, timeout=10)
            except Exception:
                # no launch buttons detected - attempt to continue but warn
                print("[!] No launch links found on this page (yet)")

            # collect current launch elements and their hrefs
            launch_elems = driver.find_elements(By.CSS_SELECTOR, LAUNCH_LINK_SELECTOR)
            if not launch_elems:
                # fallback: find anchors inside table whose href contains 'course/view.php'
                launch_elems = driver.find_elements(By.CSS_SELECTOR, "table#tbl-subject a[href*='course/view.php']")
            current_links = [el.get_attribute("href") for el in launch_elems if el.get_attribute("href")]

            print(f"[DEBUG] Found {len(current_links)} launch links on this page")

            new_found = 0

            for el in launch_elems:
                try:
                    href = el.get_attribute("href")
                    if not href or href in seen_links:
                        continue

                    # extract subject name by climbing up to ancestor td and finding the h4
                    try:
                        title_elem = el.find_element(By.XPATH, "ancestor::td//h4")
                        subject_title = title_elem.text.strip()
                    except Exception:
                        # fallback: find nearest h4 in document (less ideal)
                        try:
                            subject_title = driver.find_element(By.CSS_SELECTOR, "h4.cfullname").text.strip()
                        except Exception:
                            subject_title = "Unknown Subject"

                    # optional: instructor extraction (may fail for some structures)
                    try:
                        instructor_elem = el.find_element(By.XPATH, "ancestor::td//div[contains(@class,'istruinfocontainer')]//div")
                        instructor = instructor_elem.text.strip()
                    except Exception:
                        instructor = "N/A"

                    print(f"[+] New subject found: {subject_title} -> {href}")

                    # record it
                    subjects.append({"title": subject_title, "instructor": instructor, "url": href})
                    seen_links.add(href)
                    new_found += 1

                    # open in new tab, wait for it to load, then close
                    driver.execute_script("window.open(arguments[0]);", href)
                    time.sleep(1)
                    tabs = driver.window_handles
                    driver.switch_to.window(tabs[-1])
                    try:
                        wait_for_page_ready(driver, timeout=10)
                        time.sleep(1)  # let dynamic content settle
                        print(f"    [✓] Opened subject tab for '{subject_title}' (url: {href})")
                    except Exception:
                        print(f"    [!] Subject tab opened but page did not reach 'complete' state quickly")
                    # close subject tab and return
                    driver.close()
                    driver.switch_to.window(tabs[0])
                    time.sleep(0.5)

                except Exception as e:
                    print(f"[!] Error processing launch element: {e}")
                    # ensure we are on main tab
                    if len(driver.window_handles) > 1:
                        try:
                            driver.close()
                        except Exception:
                            pass
                        driver.switch_to.window(driver.window_handles[0])
                    continue

            if new_found == 0:
                print("[*] No new unique subjects found on this page.")

            # Attempt to go to next page
            try:
                # capture current set of seen_links to detect change after navigation
                before_links = set(seen_links)
                next_btn = driver.find_element(*NEXT_PAGE_SELECTOR)
                if not next_btn.is_displayed() or not next_btn.is_enabled():
                    print("[*] Next button not clickable or not visible. Ending pagination.")
                    break

                print("[*] Clicking Next to load the next page...")
                driver.execute_script("arguments[0].click();", next_btn)

                # wait for the page content to change (either new launch links appear or page source differs)
                WebDriverWait(driver, 12).until(lambda d: set([x.get_attribute("href") for x in d.find_elements(By.CSS_SELECTOR, LAUNCH_LINK_SELECTOR) if x.get_attribute("href")]) - before_links != set() or d.page_source != d.page_source)
                # note: the page_source != page_source check is a no-op but the first condition will usually trigger
                time.sleep(1.5)
                page_index += 1
                continue

            except Exception:
                # no Next or navigation didn't change content
                print("[✓] No more pages or unable to navigate further.")
                break

        # save gathered subjects
        out_path = "data/raw_lms_data.json"
        with open(out_path, "w", encoding="utf-8") as wf:
            json.dump(subjects, wf, indent=2, ensure_ascii=False)

        print(f"\n[✓] Finished. Unique subjects found: {len(subjects)}")
        print(f"[+] Saved to {out_path}")

    except Exception as e:
        print("[!] Top-level error:", e)

    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    scrape_lms()
