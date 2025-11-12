# app.py
import json
from typing import Optional
import streamlit as st
import pandas as pd
from openai import OpenAI
from model_ensemble import ensemble_scores

# ========= COLOR PALETTE =========
BACKGROUND_COLOR = "#12241C"         # Dark green for main background
BACKGROUND_SECONDARY = "#F5F5E6"     # Light gray for sidebar/user input area
BACKGROUND_INPUT = "#FFFFFF"         # White input box background
FONT_PRIMARY = "#FFF8E7"             # Off-white text on dark background
FONT_SECONDARY = "#1C1C1C"           # Dark text for light backgrounds
FONT_TERTIARY = "#C5CBB5"            # Accent text color
CARD_COLOR = "#12241C"                # Card background matches main background
BORDER_COLOR = "rgba(0, 0, 0, 0.1)"  # Soft divider/border line
CARD_BORDER_COLOR = "#1A3629"        # Card border accent
SLIDER_NOTCH_COLOR = "#A4B465"          # Muted green for slider accents
SLIDER_ACTIVE_COLOR = "#626F47"         # Darker green for active slider track
BUTTON_COLOR = "#A4B465"               # Muted green for buttons
PLACEHOLDER_TEXT = "rgba(60, 60, 60, 0.6)"  # Placeholder gray

st.set_page_config(page_title="Board Game Recommender", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
n_games = 8 

# ========= CUSTOM CSS =========
CUSTOM_STYLE = f"""
<style>
/* ==== MAIN PAGE ==== */
html, body, .stApp, section.main {{
  background-color: {BACKGROUND_COLOR} !important;
  color: {FONT_PRIMARY} !important;
}}

section.main h1, section.main h2, section.main h3, section.main h4, section.main h5, section.main h6,
section.main p, section.main span, section.main label {{
  color: {FONT_PRIMARY} !important;
}}

section.main > div.block-container {{
  max-width: 1100px !important;
  margin: 0 auto !important;
  padding-left: 3rem !important;
  padding-right: 3rem !important;
}}

/* ==== SIDEBAR ==== */
[data-testid="stSidebar"] {{
  background-color: {BACKGROUND_SECONDARY} !important;
  color: {FONT_SECONDARY} !important;
  padding: 1.5rem !important;
  border-right: 1px solid {BORDER_COLOR};
}}

[data-testid="stSidebar"] * {{
  background-color: transparent !important;
  box-shadow: none !important;
  border: none !important;
  color: {FONT_SECONDARY} !important;
  text-shadow: none !important;
}}

/* ==== INPUT ELEMENTS ==== */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"],
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] {{
  background-color: {BACKGROUND_INPUT} !important;
  color: {FONT_SECONDARY} !important;
  border-radius: 6px !important;
  border: 1px solid {BORDER_COLOR} !important;
  min-height: 2.5rem !important;
}}

[data-testid="stSidebar"] textarea {{
  background-color: {BACKGROUND_INPUT} !important;
  color: {FONT_SECONDARY} !important;
  border-radius: 6px !important;
  border: 1px solid {BORDER_COLOR} !important;
  box-shadow: none !important;
  padding: 0.75rem !important;
  min-height: 100px !important;
}}

/* Default tag styling */
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] {{
  background-color: {BACKGROUND_INPUT} !important;
  color: {FONT_SECONDARY} !important;
  border: none !important;
  box-shadow: none !important;
}}

/* Highlight tags for the 'Liked Board Games' widget */
[data-testid="stSidebar"] div[aria-label="Liked Board Games"] div[data-baseweb="tag"] {{
  background-color: #2e8b57 !important;
  color: #ffffff !important;
  font-weight: 600 !important;
}}
[data-testid="stSidebar"] div[aria-label="Liked Board Games"] div[data-baseweb="tag"] p {{
  color: #ffffff !important;
}}


[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {{
  color: {PLACEHOLDER_TEXT} !important;
}}

[data-testid="stSidebar"] [data-baseweb="select"] * {{
  color: {FONT_SECONDARY} !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{
  fill: {FONT_SECONDARY} !important;
}}

/* ==== SLIDERS ==== */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider div[data-testid="stTickLabel"],
[data-testid="stSidebar"] .stSlider div[data-testid="stTickValue"],
[data-testid="stSidebar"] .stSlider div[data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider div[data-testid="stTickBarMax"] {{
  color: {FONT_SECONDARY} !important;
}}

[data-testid="stSidebar"] .range-display {{
  font-size: 0.9rem;
  color: {FONT_SECONDARY};
  margin-top: 0.25rem;
  margin-bottom: 0.9rem;
  font-weight: 600;
}}

[data-testid="stSidebar"] .range-display span {{
  color: {SLIDER_ACTIVE_COLOR};
}}

[data-testid="stSidebar"] .range-display {{
  font-size: 0.9rem;
  color: {FONT_SECONDARY};
  margin-top: 0.25rem;
  margin-bottom: 0.9rem;
  font-weight: 600;
}}

[data-testid="stSidebar"] .range-display span {{
  color: {SLIDER_ACTIVE_COLOR};
}}

/* Sidebar primary button */
[data-testid="stSidebar"] .stButton button {{
  background-color: {BUTTON_COLOR} !important;
  color: {BACKGROUND_INPUT} !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 999px !important;
  padding: 0.5rem 1.75rem !important;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}}

[data-testid="stSidebar"] .stButton button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
}}

/* Add visible round notch (thumb handle) */
[data-testid="stSidebar"] .stSlider [role="slider"] {{
  -webkit-appearance: none;
  appearance: none;
  position: relative;
  background: none !important;
  border: none !important;
}}

[data-testid="stSidebar"] .stSlider [role="slider"]::after {{
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  width: 12px;
  height: 12px;
  transform: translate(-50%, -50%);
  border-radius: 50%;
  background-color: {SLIDER_NOTCH_COLOR};
  box-shadow: 0 0 3px rgba(0, 0, 0, 0.4);
  cursor: pointer;
}}

"""

st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)

# ========= APP CONTENT WRAPPER =========
st.markdown("""
<div class="content-wrapper" style="
    max-width: 1200px;
    margin: 0 auto;
    padding: 3rem;
">
""", unsafe_allow_html=True)

# ========= HERO SECTION =========
st.markdown(
    """
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto 3rem auto;
        width: 100%;
        gap: 2rem;
    ">
        <div style="flex: 1.5;">
            <h1 style="
                font-size: 5.2rem;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: 0.75rem;
            ">
                Find Your Next Favorite Board Game
            </h1>
            <p style="
                font-size: 1.2rem;
                color: #B8B8B8;
                max-width: 1000px;
                margin-top: 0.5rem;
            ">
                Tell us what you're looking for, and we'll recommend the perfect game from the vast BoardGame Geek library.
            </p>
        </div>
        <div style="flex: 1; text-align: right;">
            <img 
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCw0bpbYJkXhukFpr2ysykCg5MJ__D-CquLfxP2OZHpWwUjcZwUJDlcmC1zjScQqMN0xiv1fTcKGD5hPVo49tpEJ_mhUqV2_Z6h0oRazlqRYp7jZh5jOwiFwtLJh0Iku-MAFcqgbIqCSHyVX2Z5sP2UjGeXKWdisZgxvE0V3ltunAxVZfSyNom9QteaG9xFwe0tGEUuMD0FsAN_hqM2b5VY9oHP5JQ2Dy9YJVQ9kByL-8q5Eh2lgh2Qak8ezcNQxH533o9bZ-d2S0E"
                style="
                    width: 90%;
                    max-width: 380px;
                    border-radius: 12px;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
                "
            />
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_csv("./data/games.csv")

@st.cache_data
def load_mechanics():
    mechanics_df = pd.read_csv("./data/game_mechanics.csv", header=None, names=["mechanic"])
    return mechanics_df["mechanic"].dropna().sort_values().tolist()

@st.cache_data
def load_categories():
    categories_df = pd.read_csv("./data/game_categories.csv", header=None, names=["category"])
    return categories_df["category"].dropna().sort_values().tolist()

@st.cache_data
def load_game_types():
    game_types_df = pd.read_csv("./data/game_types.csv", header=None, names=["type"])
    return game_types_df["type"].dropna().sort_values().tolist()

@st.cache_data
def load_master_assets():
    cols = [
        "bgg_id",
        "thumbnail",
        "image",
        "ImagePath",
        "bgg_link",
        "players_min",
        "players_max",
        "time_min",
        "time_max",
        "time_avg",
    ]
    master_df = pd.read_csv("./data/games_master_data.csv", usecols=cols)
    master_df["bgg_id"] = pd.to_numeric(master_df["bgg_id"], errors="coerce").astype("Int64")
    master_df.dropna(subset=["bgg_id"], inplace=True)
    master_df["asset_url"] = (
        master_df["thumbnail"]
        .fillna(master_df["ImagePath"])
        .fillna(master_df["image"])
    )
    master_df.dropna(subset=["asset_url"], inplace=True)
    master_df = master_df.drop_duplicates("bgg_id")
    return master_df.set_index("bgg_id")

games_df = load_data()
games_lookup = games_df.set_index("BGGId")
mechanics_options = load_mechanics()
categories_options = load_categories()
game_type_options = load_game_types()
master_assets = load_master_assets()
DEFAULT_THUMBNAIL = "https://images.pexels.com/photos/411207/pexels-photo-411207.jpeg?auto=compress&cs=tinysrgb&h=320&w=320"
CARD_GRID_STYLE = f"""
<style>
.game-grid {{
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    margin-top: 1.5rem;
}}
.game-card {{
    background-color: {CARD_COLOR};
    color: {FONT_SECONDARY};
    border-radius: 16px;
    border: 0.5px solid {CARD_BORDER_COLOR};
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
    display: flex;
    align-items: stretch;
    min-height: 220px;
}}
.game-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}}
.game-image-wrapper {{
    width: 220px;
    height: 220px;
    flex-shrink: 0;
    overflow: hidden;
    border-top-left-radius: 16px;
    border-bottom-left-radius: 16px;
}}
.game-image-wrapper img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
}}
.game-content {{
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}
.game-title {{
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    color: {FONT_PRIMARY};
}}
.game-meta {{
    font-size: 1rem;
    color: #B8B8B8;
    display: flex;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.6rem;
}}
.game-meta .star-icon {{
    color: #ffcc00;
    font-size: 1.2rem;
}}
.game-meta .rating-value {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {FONT_PRIMARY};
}}
.game-meta-secondary {{
    font-size: 0.95rem;
    color: {FONT_SECONDARY};
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}}
.game-meta-secondary .meta-item {{
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-weight: 600;
}}
.game-meta-secondary .clock-icon,
.game-meta-secondary .cog-icon,
.game-meta-secondary .player-icon {{
    color: #ffcc00;
    font-size: 1rem;
}}
.game-meta-secondary .meta-value {{
    color: {FONT_TERTIARY};
}}
.game-desc {{
    font-size: 0.95rem;
    color: #CCC;
    margin-bottom: 0.8rem;
}}
.game-link {{
    color: {FONT_TERTIARY};
    font-weight: 600;
    text-decoration: none;
}}
.game-link:hover {{
    text-decoration: none;
}}
.game-insight {{
    font-size: 0.95rem;
    color: {FONT_PRIMARY};
    margin-bottom: 0.6rem;
    line-height: 1.35;
}}
</style>
"""

if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None
if "recommendation_reason" not in st.session_state:
    st.session_state["recommendation_reason"] = None
if "search_context" not in st.session_state:
    st.session_state["search_context"] = {}
if "game_insights" not in st.session_state:
    st.session_state["game_insights"] = {}


def generate_recommendation_reason(context: dict, recommendations: pd.DataFrame) -> Optional[str]:
    if recommendations is None or recommendations.empty:
        return None

    top_games = recommendations.head(5)[
        ["name", "game_categories", "game_mechanics", "recommender_score", "players_min", "players_max"]
    ]

    payload = {
        "user_preferences": context,
        "top_recommendations": top_games.to_dict(orient="records"),
    }
    prompt = (
        "You are chatting inside a board game recommender app. "
        "Using the structured data below, talk directly to the user in a friendly tone "
        "and explain in 2-3 sentences why these games should click for them. "
        "Call out specific mechanics, themes, or player counts that match their preferences, "
        "and keep the explanation casual and encouraging.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, upbeat friend who explains recommendations directly to the user. "
                        "Be conversational and positive, avoid sounding corporate."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def generate_game_insight(game_info: dict, context: dict) -> Optional[str]:
    payload = {
        "user_preferences": context,
        "game": game_info,
    }
    prompt = (
        "You are a friendly board game concierge responding to a specific user request. "
        "Write ONE lively sentence (max 35 words) that ties this game's traits to the given preferences. "
        "Explicitly reference at least one overlap (mechanics, categories, player count, play time, or the user's description). "
        "Avoid repeating the exact game name or sounding generic.\n\n"
        f"{json.dumps(payload, indent=2, default=str)}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert board game sommelier. "
                        "Be vivid, concise, and upbeat."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None

# ========== SIDEBAR ==========
st.sidebar.header("Your Preferences")

# --- CF inputs ---
liked_games = st.sidebar.multiselect(
    "Liked Board Games",
    options=sorted(games_df["Name"].dropna().tolist())
)

disliked_games = st.sidebar.multiselect(
    "Exclude from Recommendation",
    options=sorted(games_df["Name"].dropna().tolist())
)

# --- Filter inputs ---
default_year_range = (2000, 2021)
if "year_range" not in st.session_state:
    st.session_state["year_range"] = default_year_range

st.sidebar.slider(
    "Year Published",
    1990,
    2021,
    value=st.session_state["year_range"],
    key="year_range_slider"
)
st.session_state["year_range"] = st.session_state["year_range_slider"]
year_range = st.session_state["year_range"]
rating_min = st.sidebar.slider(
    "Minimum Rating", 1.0, 10.0, 6.5, step=0.5, format="%.1f"
)

# --- CBF inputs ---
min_players, max_players = st.sidebar.slider("Player Count Range", 1, 20, (2, 4))
play_time = st.sidebar.selectbox(
    "Play Time",
    ["Any", "<30 mins", "30-60 mins", "60-90 mins", "90-120 mins", ">120 mins"],
)
complexity = st.sidebar.slider("Weight", 1.0, 5.0, (2.3, 3.6), 0.1, format="%.1f")
mechanics = st.sidebar.multiselect("Game Mechanics", mechanics_options)
categories = st.sidebar.multiselect("Game Category / Theme", categories_options)
game_type = st.sidebar.multiselect("Game Type", game_type_options)

# --- LLM input ---
description = st.sidebar.text_area("Describe the kind of board game you enjoy",
                                   placeholder="Example: I like strategic games with some luck and engine building mechanics.")

# ========= RUN RECOMMENDER ==========
st.sidebar.markdown("### Get Recommendations (Choose a Model)")

# Define model parameter sets
model_configs = {
    "A": {"alpha": 0.50, "beta": 0.33},
    "B": {"alpha": 0.67, "beta": 0.25},
    "C": {"alpha": 0.33, "beta": 0.25},
    "D": {"alpha": 0.50, "beta": 0.50},
}

# Display four buttons side-by-side
col1, col2 = st.sidebar.columns(2)
col3, col4 = st.sidebar.columns(2)

buttons = {
    "A": col1.button("Get Recommendations A"),
    "B": col2.button("Get Recommendations B"),
    "C": col3.button("Get Recommendations C"),
    "D": col4.button("Get Recommendations D"),
}

# Determine which button was pressed
selected_model = next((key for key, pressed in buttons.items() if pressed), None)

if selected_model:
    alpha = model_configs[selected_model]["alpha"]
    beta = model_configs[selected_model]["beta"]

    # Build attribute dictionary from sidebar selections
    attributes = {
        "game_categories": categories,
        "game_mechanics": mechanics,
        "game_types": game_type,
        "game_weight": list(complexity),
        "players": [min_players, max_players],
        "year_published": list(year_range),
        "min_rating": [rating_min],
    }

    # Map play time text to numeric range
    play_time_map = {
        "Any" : [],
        "<30 mins": [0, 30],
        "30-60 mins": [30, 60],
        "60-90 mins": [60, 90],
        "90-120 mins": [90, 120],
        ">120 mins": [120, 9999],
    }
    attributes["play_time"] = play_time_map.get(play_time, [0, 9999])

    with st.spinner(f"Generating recommendations (Model {selected_model}: α={alpha}, β={beta})..."):
        recommendations = ensemble_scores(
            liked_games=[] if not liked_games else games_df.loc[games_df["Name"].isin(liked_games), "BGGId"].tolist(),
            disliked_games=[] if not disliked_games else games_df.loc[games_df["Name"].isin(disliked_games), "BGGId"].tolist(),
            exclude_games=[],
            attributes=attributes,
            description=description,
            n_recommendations=n_games,
            alpha=alpha,
            beta=beta,
        )

    if not isinstance(recommendations, pd.DataFrame):
        recommendations = pd.DataFrame()

    st.session_state["recommendations"] = recommendations
    st.session_state["recommendation_reason"] = None
    st.session_state["search_context"] = {
        "liked_games": liked_games,
        "disliked_games": disliked_games,
        "description": description,
        "attributes": attributes,
        "model_used": selected_model,
    }


recommendations_df = st.session_state.get("recommendations")
st.markdown(CARD_GRID_STYLE, unsafe_allow_html=True)
st.markdown("## Recommended Games for You")
if recommendations_df is None:
    st.info("Use the sidebar to set your preferences and generate recommendations.")
elif isinstance(recommendations_df, pd.DataFrame) and recommendations_df.empty:
    st.warning("No recommendations found. Try adjusting your filters or description.")
elif isinstance(recommendations_df, pd.DataFrame):
    recommendations_df = recommendations_df.reset_index(drop=True)
    recommendations_df = recommendations_df.merge(
        master_assets, left_on="bgg_id", right_index=True, how="left", suffixes=("", "_asset")
    )

    cards = ['<div class="game-grid">']
    for _, row in recommendations_df.head(n_games).iterrows():
        image_url = row.get("asset_url") or DEFAULT_THUMBNAIL
        title = str(row["n_rank"]) + ".  " + str(row["name"])
        score = row.get("recommender_score", 0)
        desc = f"Hybrid Score: {score:.3f}"
        bgg_link = row.get("bgg_link")
        if pd.isna(bgg_link) or not str(bgg_link).strip():
            bgg_id = row.get("bgg_id")
            if pd.notna(bgg_id):
                bgg_link = f"https://boardgamegeek.com/boardgame/{int(bgg_id)}"
            else:
                bgg_link = "https://boardgamegeek.com/"

        avg_rating = row.get("avg_rating")
        rating_display = f"{avg_rating:.1f}" if pd.notna(avg_rating) else "N/A"
        game_weight = row.get("game_weight")
        weight_display = str(f"{game_weight:.2f}") + " / 5" if pd.notna(game_weight) else "N/A"

        def _valid(val):
            return pd.notna(val) and val > 0

        def derive_playtime(source):
            min_pt = (
                source.get("time_min_asset")
                or source.get("ComMinPlaytime")
                or source.get("time_min")
            )
            max_pt = (
                source.get("time_max_asset")
                or source.get("ComMaxPlaytime")
                or source.get("time_max")
            )
            avg_pt = (
                source.get("time_avg_asset")
                or source.get("MfgPlaytime")
                or source.get("time_avg")
            )

            if _valid(min_pt) and _valid(max_pt):
                min_val = int(min_pt)
                max_val = int(max_pt)
                if min_val == max_val:
                    return f"{min_val} mins"
                return f"{min_val}-{max_val} mins"
            if _valid(avg_pt):
                return f"{int(avg_pt)} mins"
            if _valid(min_pt):
                return f"{int(min_pt)} mins"
            if _valid(max_pt):
                return f"{int(max_pt)} mins"
            return None

        play_time_display = derive_playtime(row) or "N/A"

        def derive_players(source):
            min_players = (
                source.get("players_min_asset")
                or source.get("players_min")
                or source.get("MinPlayers")
            )
            max_players = (
                source.get("players_max_asset")
                or source.get("players_max")
                or source.get("MaxPlayers")
            )
            if pd.notna(min_players) and pd.notna(max_players):
                min_val = int(min_players)
                max_val = int(max_players)
                if min_val == max_val:
                    return f"{min_val}"
                return f"{min_val}-{max_val}"
            if pd.notna(min_players):
                return f"{int(min_players)}"
            if pd.notna(max_players):
                return f"{int(max_players)}"
            return None

        players_display = derive_players(row) or "N/A"

        bgg_id = row.get("bgg_id")
        if pd.notna(bgg_id) and bgg_id in games_lookup.index:
            details = games_lookup.loc[bgg_id]
            if isinstance(details, pd.DataFrame):
                details = details.iloc[0]
            play_time_display = derive_playtime(details) or play_time_display
            players_display = derive_players(details) or players_display

        insight_key = int(bgg_id) if pd.notna(bgg_id) else title
        insight_text = st.session_state["game_insights"].get(insight_key)
        if insight_text is None:
            game_payload = {
                "name": title,
                "avg_rating": rating_display,
                "game_weight": weight_display,
                "categories": row.get("game_categories", []),
                "mechanics": row.get("game_mechanics", []),
                "hybrid_score": score,
                "play_time": play_time_display,
                "players": players_display,
            }
            context = st.session_state.get("search_context", {})
            insight_text = generate_game_insight(game_payload, context) or ""
            st.session_state["game_insights"][insight_key] = insight_text

        cards.append(
            f'<div class="game-card">'
            f'  <div class="game-image-wrapper">'
            f'    <img src="{image_url}" alt="{title}">'
            f'  </div>'
            f'  <div class="game-content">'
            f'    <div class="game-title">{title}</div>'
            f'    <div class="game-meta"><span class="star-icon">&#9733;</span> <span class="rating-value">{rating_display}</span></div>'
            f'    <div class="game-meta-secondary">'
            f'      <span class="meta-item"><span class="clock-icon">&#128337;</span>'
            f'        <span class="meta-value">{play_time_display}</span></span>'
            f'      <span class="meta-item"><span class="player-icon">&#128101;</span>'
            f'        <span class="meta-value">{players_display}</span></span>'
            f'      <span class="meta-item"><span class="cog-icon">&#9881;</span>'
            f'        <span class="meta-value">{weight_display}</span></span>'
            f'    </div>'
            f'    <div class="game-insight">{insight_text or "We think this will be a great fit!"}</div>'
            f'    <div class="game-desc">{desc}</div>'
            f'    <a href="{bgg_link}" '
            f'       class="game-link" target="_blank">View on BGG &rarr;</a>'
            f'  </div>'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)

else:
    st.warning("Unable to display recommendations. Please try running the search again.")

st.markdown("</div>", unsafe_allow_html=True)
