import os
import requests
from bs4 import BeautifulSoup

os.makedirs("data/scripts", exist_ok=True)

base_url = "https://fangj.github.io/friends/"
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".html")]

for link in links:
    url = base_url + link
    filename = link.replace("season/", "")
    ep_response = requests.get(url)
    with open(f"data/scripts/{filename}", "w", encoding="utf-8") as f:
        f.write(ep_response.text)
    print(f"Downloaded {filename}")

print(f"Done. Total: {len(links)} episodes")