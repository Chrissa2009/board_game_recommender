import io
import numpy as np
import pandas as pd
from openai import OpenAI

import streamlit as st

client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# Load game data
games_df = pd.read_csv("./data/games_master_data.csv", encoding="utf-8-sig")

desc_df = pd.read_csv("./data/game_descriptions.csv", encoding="utf-8-sig").rename(
    columns={"bgg_id": "bgg_id", "full_description": "Description"}
)

# Merge datasets on bgg_id
merged_df = pd.merge(
    games_df.drop(columns=["Description"], errors="ignore"),
    desc_df[["bgg_id", "Description"]],
    on="bgg_id",
    how="inner"
)

# Extract all category columns automatically
games_df["simple_game_categories"] = games_df["simple_game_categories"].fillna("").astype(str)
all_categories = (
    games_df["simple_game_categories"]
    .str.split(";")
    .explode()
    .str.strip()
    .dropna()
    .unique()
)
category_columns = sorted(all_categories.tolist())

def get_llm_scores(user_description: str, min_players: int, category: str):
    """
    Generate LLM-based relevance scores for candidate games based on the user description.
    Returns a 1D NumPy array of scores aligned with games_df order.
    """
    if category not in category_columns:
        raise ValueError(f"Invalid category '{category}'. Available categories: {category_columns}")

    # Filter dataset by player count and simple category
    filtered_df = merged_df[
        (merged_df["players_min"] <= min_players)
        & (merged_df["players_max"] >= min_players)
        & (merged_df["simple_game_categories"].str.contains(fr"\b{category}\b", case=False, na=False))
    ]

    if filtered_df.empty:
        return np.zeros(len(games_df))

    # Limit to top 20 by rating for token efficiency
    candidate_games = filtered_df.sort_values("avg_rating", ascending=False).head(20)

    # Prepare text for LLM input
    descriptions = "\n\n".join([
        f"Name: {row['name']}\nYear: {row['year_published']}\nDescription: {row['description']}"
        for _, row in candidate_games.iterrows()
    ])

    prompt = f"""
    The user described their ideal board game as follows:
    "{user_description}"

    You are given a list of candidate board games.
    For each game, assign a relevance score between 0 and 1 that reflects how well it matches the user's description.
    Respond *only* in CSV format with two columns: Name, LLM_Score.
    Example:
    Name,LLM_Score
    Game A,0.92
    Game B,0.74

    Games:
    {descriptions}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert board game recommender that outputs structured data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    csv_output = response.choices[0].message.content.strip()
    csv_output = "\n".join(line for line in csv_output.splitlines() if not line.strip().startswith("```")).strip()

    # Convert CSV text to DataFrame with resilient parsing
    try:
        llm_scores_df = pd.read_csv(io.StringIO(csv_output))
    except Exception:
        # Manual fallback: strip formatting and ensure 2 columns
        lines = [line for line in csv_output.splitlines() if "," in line]
        if not lines:
            return np.zeros(len(games_df))

        # Clean commas within quoted names and trim whitespace
        clean_lines = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) > 2:
                # Merge all but the last part as the game name
                name = ",".join(parts[:-1]).strip('" ')
                score = parts[-1].strip()
                clean_lines.append([name, score])
            elif len(parts) == 2:
                clean_lines.append(parts)

        # Use explicit header and create DataFrame
        header = ["name", "llm_score"]
        data_rows = [row for row in clean_lines if row[0].lower() != "name"]
        llm_scores_df = pd.DataFrame(data_rows, columns=header)

    # Standardize column names
    llm_scores_df.columns = [col.strip().replace(" ", "_") for col in llm_scores_df.columns]
    if "llm_score" not in llm_scores_df.columns and len(llm_scores_df.columns) >= 2:
        llm_scores_df.rename(columns={llm_scores_df.columns[1]: "llm_score"}, inplace=True)
    if "name" not in llm_scores_df.columns and len(llm_scores_df.columns) >= 1:
        llm_scores_df.rename(columns={llm_scores_df.columns[0]: "name"}, inplace=True)

    # Clean numeric column
    llm_scores_df["llm_score"] = (
        llm_scores_df["llm_score"].astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0]
    )
    llm_scores_df["llm_score"] = pd.to_numeric(llm_scores_df["llm_score"], errors="coerce").clip(0, 1)
    llm_scores_df.dropna(subset=["llm_score"], inplace=True)

    # Match to candidate games
    candidate_lookup = candidate_games[["bgg_id", "name"]].drop_duplicates()
    llm_scores_df = candidate_lookup.merge(llm_scores_df, on="name", how="right")
    llm_scores_df.dropna(subset=["bgg_id"], inplace=True)

    # Fill scores for all games
    full_scores = np.zeros(len(games_df))
    score_map = dict(zip(llm_scores_df["bgg_id"], llm_scores_df["llm_score"]))
    for idx, row in games_df.iterrows():
        bgg_id = row["bgg_id"] if "bgg_id" in games_df.columns else row["bgg_id"]
        if bgg_id in score_map:
            full_scores[idx] = score_map[bgg_id]

    return full_scores

if __name__ == "__main__":
    scores = get_llm_scores(
        user_description="I love cooperative adventure games with fantasy storytelling.",
        min_players=4,
        category="Abstract / Strategy"
    )
    print("LLM Scores:", scores)
    print("LLM Scores Length:", len(scores))
