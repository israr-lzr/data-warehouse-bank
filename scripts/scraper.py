import time
import re
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Banques principales
banks = [
    "Attijariwafa Bank",
    "Banque Populaire",
    "BMCE Bank",
    "Société Générale Maroc",
    "Crédit Agricole du Maroc",
    "CIH Bank"
]

# Villes à cibler
cities = ["Casablanca", "Rabat", "Fès", "Marrakech", "Tanger", "Agadir", "Meknès", "Oujda"]

# Paramètres
max_reviews_per_branch = 100
max_branches_per_search = 30

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def scroll_page(driver, n_scrolls=10):
    for _ in range(n_scrolls):
        driver.execute_script("window.scrollBy(0,1000)")
        time.sleep(1.5)

def scrape_place_reviews(driver, bank_name, place_url):
    reviews = []
    try:
        driver.get(place_url)
        time.sleep(3)

        branch_name = "Unknown Branch"
        location = "Unknown Location"

        for sel in ["h1.DUwDvf", "h1.fontHeadlineLarge", "h1"]:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            if elems:
                branch_name = elems[0].text
                break

        for sel in ["button[data-item-id='address']", "div[aria-label*='address']"]:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            if elems:
                location = elems[0].text
                break

        for sel in ["button[aria-label*='review']", "a[href*='#reviews']"]:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            if elems:
                try:
                    elems[0].click()
                    time.sleep(2)
                except:
                    pass
                break

        review_elements = []
        for sel in ["div.jftiEf", "div[class*='review']"]:
            review_elements = driver.find_elements(By.CSS_SELECTOR, sel)
            if review_elements:
                break

        for i, element in enumerate(review_elements):
            if i >= max_reviews_per_branch:
                break
            try:
                rating = 0
                for sel in ["span[aria-label*='star']"]:
                    elems = element.find_elements(By.CSS_SELECTOR, sel)
                    if elems:
                        match = re.search(r'(\d+(\.\d+)?)', elems[0].get_attribute("aria-label"))
                        if match:
                            rating = float(match.group(1))
                            break

                review_text = ""
                for sel in ["span.wiI7pd", "span[class*='review-full-text']"]:
                    elems = element.find_elements(By.CSS_SELECTOR, sel)
                    if elems:
                        review_text = elems[0].text
                        break

                date = "Unknown Date"
                for sel in ["span.rsqaWe", "span[class*='date']"]:
                    elems = element.find_elements(By.CSS_SELECTOR, sel)
                    if elems:
                        date = elems[0].text
                        break

                if rating or review_text:
                    reviews.append({
                        "bank_name": bank_name,
                        "branch_name": branch_name,
                        "branch_location": location,
                        "review_text": review_text or "No text",
                        "rating": rating,
                        "review_date": date
                    })
            except Exception as e:
                print(f"[!] Erreur sur un avis : {e}")
                continue
    except Exception as e:
        print(f"[!] Erreur sur l'URL {place_url} : {e}")
    return reviews

def collect_bank_reviews():
    all_reviews = []
    driver = setup_driver()
    try:
        for bank in banks:
            for city in cities:
                print(f"\n🔍 Scraping {bank} - {city}")
                query = f"{bank} {city}"
                search_url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
                driver.get(search_url)
                time.sleep(3)
                scroll_page(driver, n_scrolls=10)

                place_urls = []
                elements = driver.find_elements(By.CSS_SELECTOR, "a[href^='https://www.google.com/maps/place']")
                for elem in elements:
                    href = elem.get_attribute("href")
                    if href and href not in place_urls:
                        place_urls.append(href)
                print(f"  🏢 {len(place_urls)} agences trouvées")

                for i, place_url in enumerate(place_urls[:max_branches_per_search]):
                    print(f"  📍 Agence {i+1}/{max_branches_per_search}")
                    reviews = scrape_place_reviews(driver, bank, place_url)
                    all_reviews.extend(reviews)
                    time.sleep(random.uniform(2, 4))
    finally:
        driver.quit()
    return pd.DataFrame(all_reviews)

def run_scraping_collection():
    print("🚀 Démarrage du scraping...")
    df = collect_bank_reviews()
    print(f"\n✅ Total avis collectés : {len(df)}")
    if not df.empty:
        df.to_csv("data/bank_reviews_scraping.csv", index=False)
        print("💾 Fichier sauvegardé : data/bank_reviews_scraping.csv")
    else:
        print("⚠️ Aucun avis collecté. Vérifiez les sélecteurs ou la connexion.")

if __name__ == "__main__":
    run_scraping_collection()
