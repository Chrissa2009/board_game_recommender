import requests
from bs4 import BeautifulSoup
import time
import xml.etree.ElementTree as ET
import pandas as pd

### BGG
url = "https://boardgamegeek.com/browse/boardgame"
resp = requests.get(url)
soup = BeautifulSoup(resp.text, "html.parser")

game_links = soup.select("tr[id^='row_'] a.primary")
game_ids = [link["href"].split("/")[2] for link in game_links]

### API
def fetch_batch(game_ids):
    ids_param = ",".join(game_ids)
    url = f"https://boardgamegeek.com/xmlapi2/thing?id={ids_param}&stats=1"
    while True:
        r = requests.get(url)
        root = ET.fromstring(r.content)
        if root.tag == "message":  # waiting message from BGG API
            time.sleep(5)
            continue
        return root

games = []
batch_size = 20
for i in range(0, len(game_ids), batch_size):
    batch = game_ids[i:i + batch_size]
    root = fetch_batch(batch)




    for item in root.findall("item"):

        ### LIMIT To games published 2021 or earlier
        year_published_node = item.find("yearpublished")
        year_published = int(year_published_node.get("value")) if year_published_node is not None else None
        if year_published is not None and year_published > 2021:
            continue

        
        # Primary name
        primary_name = None
        for n in item.findall("name"):
            if n.get("type") == "primary":
                primary_name = n.get("value")
                break

        # Description
        #description = item.findtext("description")

        # Ratings and rank
        stats = item.find("statistics/ratings")
        avg_rating = stats.find("average").get("value") if stats is not None else None
        rank_node = stats.find("ranks/rank[@name='boardgame']") if stats is not None else None
        rank = rank_node.get("value") if rank_node is not None else None

        # Mechanics
        mechanics = [
            link.get("value")
            for link in item.findall("link")
            if link.get("type") == "boardgamemechanic"
        ]

        # Categories
        categories = [
            link.get("value")
            for link in item.findall("link")
            if link.get("type") == "boardgamecategory"
        ]

        # Game Types (remove ' Rank')
        gametypes = []
        if stats is not None:
            rank_elements = stats.findall("ranks/rank")
            gametypes = [
                r.get("friendlyname").replace(" Rank", "")
                for r in rank_elements
                if r.get("name") != "boardgame" and r.get("friendlyname")
            ]

        # Average Best number of players (decimal)
        best_numplayers = None
        poll = item.find("poll[@name='suggested_numplayers']")
        if poll is not None:
            total_votes = 0
            weighted_sum = 0
            for results in poll.findall("results"):
                numplayers_str = results.get("numplayers")
                try:
                    numplayers = int(numplayers_str)
                except ValueError:
                    continue  # skip "Unknown"
                best_vote = results.find("result[@value='Best']")
                if best_vote is not None:
                    votes = int(best_vote.get("numvotes"))
                    total_votes += votes
                    weighted_sum += numplayers * votes
            if total_votes > 0:
                best_numplayers = round(weighted_sum / total_votes, 2)

        # Other descriptive fields
        playingtime = item.find("playingtime").get("value") if item.find("playingtime") is not None else None
        minplaytime = item.find("minplaytime").get("value") if item.find("minplaytime") is not None else None
        maxplaytime = item.find("maxplaytime").get("value") if item.find("maxplaytime") is not None else None
        image = item.findtext("image")
        thumbnail = item.findtext("thumbnail")

        games.append({
            "id": item.get("id"),
            #"name": primary_name,
            #"description": description,
            #"avg_rating": avg_rating,
            #"rank": rank,
            "mechanics": "; ".join(mechanics),
            "boardgamecategory": "; ".join(categories),
            "gametype": "; ".join(gametypes),
            "playingtime": playingtime,
            "minplaytime": minplaytime,
            "maxplaytime": maxplaytime,
            "best_numplayers": best_numplayers,
            "image": image,
            "thumbnail": thumbnail
        })

    time.sleep(5)

# Save results
df = pd.DataFrame(games)
df.to_csv("bgg_data.csv", index=False)
print(df.head())
