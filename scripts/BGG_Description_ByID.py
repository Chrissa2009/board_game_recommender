import requests
import time
import xml.etree.ElementTree as ET
import pandas as pd
import html
import re
import os
import random

# === Load Game IDs from CSV ===
INPUT_CSV = "missing_game_ids.csv"  # CSV must have one column 'id'
df_ids = pd.read_csv(INPUT_CSV)
game_ids = [str(x) for x in df_ids['id'].dropna().astype(int).tolist()]
print(f"Loaded {len(game_ids)} game IDs from {INPUT_CSV}")

# === Clean text ===
def clean_text(text):
    """Unescape HTML entities, fix mojibake, remove control chars."""
    if not text:
        return ""
    text = html.unescape(text)
    try:
        if any(x in text for x in ["Ã", "â", "€", "¢", "œ", "‚", "„"]):
            text = text.encode("latin1").decode("utf-8")
    except Exception:
        pass
    text = text.replace("&#10;", "\n").replace("\r", "").strip()
    text = re.sub(r"^[\x00-\x1F\x7F-\x9F]+", "", text)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Robust XML fetch ===
def fetch_batch(game_ids, max_retries=5):
    ids_param = ",".join(game_ids)
    url = f"https://boardgamegeek.com/xmlapi2/thing?id={ids_param}&stats=1"
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=30)
            if not r.text.strip().startswith("<"):
                raise ValueError("Response not XML — possibly HTML or empty.")
            root = ET.fromstring(r.content)
            if root.tag == "message":
                print("BGG API busy — waiting 5s...")
                time.sleep(5)
                continue
            return root
        except (ET.ParseError, ValueError) as e:
            print(f"Parse error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(5 + random.uniform(0, 3))
    raise RuntimeError(f"Failed to fetch valid XML after {max_retries} attempts.")

# === Fetch and save ===
OUTPUT_CSV = "missing_bgg_games_data.csv"
games = []
batch_size = 20
save_every = 1000

if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

for i in range(0, len(game_ids), batch_size):
    batch = game_ids[i:i + batch_size]
    print(f"Fetching batch {i // batch_size + 1} of {len(game_ids) // batch_size + 1}: {len(batch)} games")

    try:
        root = fetch_batch(batch)
    except RuntimeError as e:
        print(f"Skipping batch {batch}: {e}")
        continue

    for item in root.findall("item"):
        year_node = item.find("yearpublished")
        year_published = int(year_node.get("value")) if year_node is not None else None
        if year_published and year_published > 2021:
            continue

        primary_name = None
        for n in item.findall("name"):
            if n.get("type") == "primary":
                primary_name = clean_text(n.get("value"))
                break

        description = clean_text(item.findtext("description"))

        mechanics = [
            clean_text(link.get("value"))
            for link in item.findall("link")
            if link.get("type") == "boardgamemechanic"
        ]
        categories = [
            clean_text(link.get("value"))
            for link in item.findall("link")
            if link.get("type") == "boardgamecategory"
        ]
        gametypes = []
        stats = item.find("statistics/ratings")
        if stats is not None:
            rank_elements = stats.findall("ranks/rank")
            gametypes = [
                clean_text(r.get("friendlyname").replace(" Rank", ""))
                for r in rank_elements
                if r.get("name") != "boardgame" and r.get("friendlyname")
            ]

        games.append({
            "id": item.get("id"),
            "name": primary_name,
            "description": description,
            "mechanics": "; ".join(mechanics),
            "boardgamecategory": "; ".join(categories),
            "gametype": "; ".join(gametypes),
        })

        # Save every 1000 games
        if len(games) >= save_every:
            df = pd.DataFrame(games)
            mode = "a" if os.path.exists(OUTPUT_CSV) else "w"
            header = not os.path.exists(OUTPUT_CSV)
            df.to_csv(OUTPUT_CSV, mode=mode, index=False, encoding="utf-8-sig", header=header)
            print(f"Saved {len(games)} games to {OUTPUT_CSV}")
            games = []

    time.sleep(5 + random.uniform(0, 2))

# Save remaining
if games:
    df = pd.DataFrame(games)
    mode = "a" if os.path.exists(OUTPUT_CSV) else "w"
    header = not os.path.exists(OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, mode=mode, index=False, encoding="utf-8-sig", header=header)
    print(f"Saved remaining {len(games)} games to {OUTPUT_CSV}")
