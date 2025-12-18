import os
import json
import time
import dotenv
from firecrawl import Firecrawl
from bs4 import BeautifulSoup

dotenv.load_dotenv()
api_key = os.getenv("FC_API_KEY")
if not api_key:
    raise ValueError("FC_API_KEY not found in environment variables.")

app = Firecrawl(api_key)

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
ITEMS_PER_PAGE = 12
MAX_PAGES = 32

all_assessments = []

for page in range(MAX_PAGES):
    start = page * ITEMS_PER_PAGE
    page_url = f"{BASE_URL}?start={start}"

    print(f"Scraping page {page + 1} (start={start})")

    result = app.scrape(
        page_url,
        formats=["html"],
        only_main_content=False,
        timeout=120000
    )

    if not result or not hasattr(result, "html") or not result.html:
        break

    soup = BeautifulSoup(result.html, "html.parser")

    # Find correct table
    tables = soup.find_all("table")
    target_table = None

    for table in tables:
        th = table.find("th")
        if th and "Individual Test Solutions" in th.get_text(strip=True):
            target_table = table
            break

    if not target_table:
        break

    rows = target_table.find_all("tr")[1:]
    if not rows:
        break

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        name_tag = cols[0].find("a")
        if not name_tag:
            continue

        name = name_tag.get_text(strip=True)
        url = name_tag.get("href")

        # Safety filter
        if "solution" in name.lower():
            continue

        remote_testing = bool(cols[1].find("span", class_="check"))
        adaptive_irt = bool(cols[2].find("span", class_="check"))

        test_type = [
            span.get_text(strip=True)
            for span in cols[3].find_all("span", class_="product-catalogue__key")
            if span.get_text(strip=True)
        ]


        all_assessments.append({
            "name": name,
            "url": url,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "test_type": test_type
        })

    time.sleep(1)

# Deduplicate
unique = {(a["name"], a["url"]): a for a in all_assessments}
final_assessments = list(unique.values())

print(f"Total Individual Test Solutions: {len(final_assessments)}")

os.makedirs("data/raw", exist_ok=True)
with open("data/raw/shl_individual_test_solutions.json", "w", encoding="utf-8") as f:
    json.dump(final_assessments, f, indent=2, ensure_ascii=False)
