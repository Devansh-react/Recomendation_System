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

test_url = "https://www.shl.com/products/product-catalog/view/net-mvc-new/"
result = app.scrape(test_url, formats=["html"], only_main_content=False, timeout=60000)

with open("scout_output.html", "w", encoding="utf-8") as f:
    f.write(result.html)

soup = BeautifulSoup(result.html, "html.parser")
# dump anything that looks like body/description content, minus nav/footer
for tag in soup.find_all(["p", "div"], class_=True):
    text = tag.get_text(strip=True)
    if len(text) > 80:  # filter noise, keep substantial text blocks
        print(f"[{tag.get('class')}] {text[:200]}\n---")