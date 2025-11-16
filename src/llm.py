import io
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def semicolon_to_list(value: Any) -> list:
    """Normalize semicolon-delimited strings (or lists) into clean lists."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return []

# Load game data
games_df = pd.read_csv("./data/games_master_data.csv", encoding="utf-8-sig")

for col in [
    "game_categories",
    "simple_game_categories",
    "game_mechanics",
    "simple_game_mechanics",
    "game_types",
]:
    if col in games_df.columns:
        games_df[col] = games_df[col].apply(semicolon_to_list)

if "game_categories" in games_df.columns and "simple_game_categories" in games_df.columns:
    games_df.drop(columns=["game_categories"], inplace=True)
if "game_mechanics" in games_df.columns and "simple_game_mechanics" in games_df.columns:
    games_df.drop(columns=["game_mechanics"], inplace=True)

games_df.rename(
    columns={
        "simple_game_categories": "game_categories",
        "simple_game_mechanics": "game_mechanics",
    },
    inplace=True,
)

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
category_source = games_df["game_categories"] if "game_categories" in games_df.columns else []
all_categories = sorted(
    {
        cat.strip()
        for cats in category_source
        for cat in (cats if isinstance(cats, list) else [])
        if isinstance(cat, str) and cat.strip()
    }
)
category_columns = all_categories


def apply_attribute_filters(df: pd.DataFrame, attributes: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Apply the same attribute masks used by the ensemble to the LLM candidate pool."""
    if not attributes:
        return df

    mask = pd.Series(True, index=df.index)

    def multi_label_mask(column: str):
        selected = attributes.get(column, [])
        if column not in df.columns or not selected:
            return None
        selected_clean = {
            s.strip().lower() for s in selected if isinstance(s, str) and s.strip()
        }
        if not selected_clean:
            return None

        def matches(values):
            if isinstance(values, (list, tuple)):
                return any(
                    isinstance(v, str) and v.strip().lower() in selected_clean
                    for v in values
                )
            if isinstance(values, str):
                return values.strip().lower() in selected_clean
            return False

        return df[column].apply(matches).fillna(False)

    for attr_column in ["game_categories", "game_mechanics", "game_types"]:
        column_mask = multi_label_mask(attr_column)
        if column_mask is not None:
            mask &= column_mask

    def range_mask(column: str, value_range):
        if column not in df.columns:
            return None
        if (
            isinstance(value_range, (list, tuple))
            and len(value_range) == 2
            and any(value_range)
        ):
            min_val, max_val = value_range
            series = df[column]
            return series.between(min_val, max_val, inclusive="both").fillna(False)
        return None

    weight_mask = range_mask("game_weight", attributes.get("game_weight"))
    if weight_mask is not None:
        mask &= weight_mask

    year_mask = range_mask("year_published", attributes.get("year_published"))
    if year_mask is not None:
        mask &= year_mask

    rating_values = attributes.get("min_rating")
    if (
        "avg_rating" in df.columns
        and isinstance(rating_values, (list, tuple))
        and rating_values
    ):
        min_rating = rating_values[0]
        rating_mask = (df["avg_rating"] >= min_rating).fillna(False)
        mask &= rating_mask

    players_range = attributes.get("players")
    if (
        isinstance(players_range, (list, tuple))
        and len(players_range) == 2
        and "players_min" in df.columns
        and "players_max" in df.columns
    ):
        p_min, p_max = players_range
        players_mask = (df["players_max"] >= p_min) & (df["players_min"] <= p_max)
        mask &= players_mask.fillna(False)

    play_time_range = attributes.get("play_time")
    if (
        isinstance(play_time_range, (list, tuple))
        and len(play_time_range) == 2
        and play_time_range
        and "time_min" in df.columns
        and "time_max" in df.columns
    ):
        t_min, t_max = play_time_range
        time_mask = (df["time_max"] >= t_min) & (df["time_min"] <= t_max)
        mask &= time_mask.fillna(False)

    return df[mask]

def get_llm_scores(
    user_description: str,
    attributes: Optional[Dict[str, Any]] = None,
    top_k: int = 300,
):
    """
    Generate LLM-based relevance scores for candidate games based on the user description.
    The candidate pool is filtered with the same attribute masks used downstream so that
    the LLM signal survives the final ensemble filtering.
    """
    attributes = attributes or {}
    filtered_df = apply_attribute_filters(merged_df, attributes)

    if filtered_df.empty:
        return np.zeros(len(games_df))

    # Limit to top games by rating for token efficiency
    candidate_games = filtered_df.sort_values("avg_rating", ascending=False).head(top_k)

    # Prepare text for LLM input
    descriptions = "\n\n".join([
        f"Name: {row['name']}\nYear: {row['year_published']}\nDescription: {row.get('description', '')}"
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
        attributes={
            "players": [4, 4],
            "game_categories": ["Abstract / Strategy"],
        },
    )
    print("LLM Scores:", scores)
    print("LLM Scores Length:", len(scores))
