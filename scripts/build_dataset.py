"""Build the mini-dataset: ~50 landmarks (image + caption) from Wikipedia.

Usage: python scripts/build_dataset.py   (run from the project root)
Produces: data/images/<id>.jpg + data/metadata.jsonl

Captions are in English because CLIP is trained on English text: using Italian
would degrade retrieval.
"""

import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Wikimedia requires an identifiable User-Agent.
HEADERS = {"User-Agent": "agentic-mm-rag-prototype/0.1 (summer school project)"}
MAX_SIDE = 800  # conservative cap: sufficient for CLIP (224px) and the LMM

LANDMARKS = [
    "Eiffel Tower", "Colosseum", "Taj Mahal", "Great Wall of China", "Machu Picchu",
    "Christ the Redeemer (statue)", "Sagrada Família", "Neuschwanstein Castle", "Mount Rushmore",
    "Statue of Liberty", "Big Ben", "Tower Bridge", "Stonehenge", "Parthenon",
    "Leaning Tower of Pisa", "Milan Cathedral", "Saint Basil's Cathedral", "Hagia Sophia",
    "Sultan Ahmed Mosque", "Petra", "Angkor Wat", "Borobudur", "Himeji Castle",
    "Kinkaku-ji", "Tokyo Tower", "Sydney Opera House", "Golden Gate Bridge",
    "Empire State Building", "Chrysler Building", "Burj Khalifa", "Petronas Towers",
    "Marina Bay Sands", "CN Tower", "Space Needle", "Brandenburg Gate",
    "Cologne Cathedral", "Charles Bridge", "Alhambra", "Mosque–Cathedral of Córdoba",
    "Belém Tower", "Château de Chambord", "Mont-Saint-Michel", "Pont du Gard",
    "Sacré-Cœur, Paris", "Palace of Versailles", "Windsor Castle", "Edinburgh Castle",
    "Giant's Causeway", "Chichen Itza", "Teotihuacan", "Moai", "Uluru",
    "Hallgrímskirkja", "Atomium", "Peleș Castle", "Bran Castle", "Meteora",
    "Trevi Fountain", "Casa Batlló", "Guggenheim Museum Bilbao",
]


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=\.)\s+", text)
    return " ".join(parts[:n])


def fetch_item(title: str) -> tuple[str, Image.Image] | None:
    """Return (caption, image), or None if the page is unusable."""
    resp = requests.get(SUMMARY_API + title.replace(" ", "_"), headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if "thumbnail" not in data or not data.get("extract"):
        return None

    # Wikimedia only serves widths that are already rendered (requesting others
    # returns 400), so use the URL as-is. SVGs are logos/coats of arms, not
    # photos of the landmark, and are therefore useless for visual retrieval.
    url = data["thumbnail"]["source"]
    if ".svg" in url.lower():
        return None

    img_resp = requests.get(url, headers=HEADERS, timeout=60)
    if img_resp.status_code != 200:
        return None

    image = Image.open(BytesIO(img_resp.content)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return first_sentences(data["extract"]), image


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    images_dir = ROOT / cfg.data.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = ROOT / cfg.data.metadata

    # Wikimedia downloads fail sporadically: by keeping already downloaded
    # items, rerunning the script fetches only those that are missing.
    already = {}
    if metadata_path.exists():
        already = {json.loads(line)["id"]: json.loads(line) for line in metadata_path.open()}

    records = []
    for title in LANDMARKS:
        item_id = slugify(title)
        if item_id in already and (ROOT / already[item_id]["image"]).exists():
            records.append(already[item_id])
            continue

        time.sleep(1.5)  # without a pause, Wikimedia returns 429 after a few downloads
        item = fetch_item(title)
        if item is None:
            print(f"[skip] {title}")
            continue
        caption, image = item
        image_path = images_dir / f"{item_id}.jpg"
        image.save(image_path, "JPEG", quality=90)
        records.append({"id": item_id, "image": str(image_path.relative_to(ROOT)), "caption": caption})
        print(f"[ok]   {item_id}")

    with metadata_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\n{len(records)} items saved to {metadata_path}")


if __name__ == "__main__":
    main()
