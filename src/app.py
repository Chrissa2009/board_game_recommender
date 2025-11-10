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
FONT_PRIMARY = "#F5F5E6"             # Off-white text on dark background
FONT_SECONDARY = "#1C1C1C"           # Dark text for light backgrounds
BORDER_COLOR = "rgba(0, 0, 0, 0.1)"  # Soft divider/border line
SLIDER_NOTCH_COLOR = "#A4B465"          # Muted green for slider accents
SLIDER_ACTIVE_COLOR = "#626F47"         # Darker green for active slider track
BUTTON_COLOR = "#A4B465"               # Muted green for buttons
PLACEHOLDER_TEXT = "rgba(60, 60, 60, 0.6)"  # Placeholder gray

st.set_page_config(page_title="Board Game Recommender", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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

[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] {{
  background-color: {BACKGROUND_INPUT} !important;
  color: {FONT_SECONDARY} !important;
  border: none !important;
  box-shadow: none !important;
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
    cols = ["bgg_id", "name", "thumbnail", "image", "ImagePath", "bgg_link"]
    master_df = pd.read_csv("./data/games_master_data.csv", usecols=cols)
    master_df["bgg_id"] = pd.to_numeric(master_df["bgg_id"], errors="coerce").astype("Int64")
    master_df.dropna(subset=["bgg_id"], inplace=True)
    master_df["asset_url"] = (
        master_df["thumbnail"]
        .fillna(master_df["ImagePath"])
        .fillna(master_df["image"])
    )
    master_df.dropna(subset=["asset_url"], inplace=True)
    master_df = master_df[["bgg_id", "asset_url", "bgg_link"]].drop_duplicates("bgg_id")
    return master_df.set_index("bgg_id")

games_df = load_data()
mechanics_options = load_mechanics()
categories_options = load_categories()
game_type_options = load_game_types()
master_assets = load_master_assets()
DEFAULT_THUMBNAIL = "https://images.pexels.com/photos/411207/pexels-photo-411207.jpeg?auto=compress&cs=tinysrgb&h=320&w=320"
CARD_GRID_STYLE = """
<style>
.game-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1.5rem;margin-top:1.5rem}
.game-card{background-color:#1A2B22;border-radius:16px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,.25);transition:transform .2s, box-shadow .2s}
.game-card:hover{transform:translateY(-4px);box-shadow:0 12px 32px rgba(0,0,0,.35)}
.game-image{width:100%;height:220px;object-fit:cover}
.game-content{padding:1rem 1.25rem;color:#F5F5E6}
.game-title{font-size:1.15rem;font-weight:700;margin-bottom:.4rem}
.game-meta{font-size:.9rem;color:#B8B8B8;display:flex;gap:1rem;align-items:center;margin-bottom:.6rem}
.game-desc{font-size:.95rem;color:#CCC;margin-bottom:.8rem}
.game-link{color:#A4B465;font-weight:600;text-decoration:none}
.game-link:hover{text-decoration:underline}
</style>
"""

if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None
if "recommendation_reason" not in st.session_state:
    st.session_state["recommendation_reason"] = None
if "search_context" not in st.session_state:
    st.session_state["search_context"] = {}


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
year_range = st.sidebar.slider("Year Published", 1990, 2021, (2000, 2021))
rating_min = st.sidebar.slider(
    "Minimum Rating", 1.0, 10.0, 6.5, step=0.5, format="%.1f"
)

# --- CBF inputs ---
min_players, max_players = st.sidebar.slider("Player Count Range", 1, 20, (2, 4))
play_time = st.sidebar.selectbox(
    "Play Time",
    ["Any", "<30 mins", "30-60 mins", "60-90 mins", "90-120 mins", ">120 mins"],
)
complexity = st.sidebar.slider("Complexity", 1.0, 5.0, (2.3, 3.6), 0.1, format="%.1f")
mechanics = st.sidebar.multiselect("Game Mechanics", mechanics_options)
categories = st.sidebar.multiselect("Game Category", categories_options)
game_type = st.sidebar.multiselect("Game Type", game_type_options)

# --- LLM input ---
description = st.sidebar.text_area("Describe the kind of board game you enjoy",
                                   placeholder="Example: I like strategic games with some luck and engine building mechanics.")

# ========= RUN RECOMMENDER ==========
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
            n_recommendations=20,
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
        master_assets, left_on="bgg_id", right_index=True, how="left"
    )

    cards = ['<div class="game-grid">']
    for _, row in recommendations_df.iterrows():
        image_url = row.get("asset_url") or DEFAULT_THUMBNAIL
        title = str(row["name"])
        score = row.get("recommender_score", 0)
        desc = f"Hybrid Score: {score:.3f}"

        cards.append(
            f'<div class="game-card">'
            f'  <img src="{image_url}" class="game-image" alt="{title}">'
            f'  <div class="game-content">'
            f'    <div class="game-title">{title}</div>'
            f'    <div class="game-meta">&#9733; {row.get("avg_rating", "N/A")}</div>'
            f'    <div class="game-desc">{desc}</div>'
            f'    <a href="{row.get("bgg_link", "https://boardgamegeek.com/")}" '
            f'       class="game-link" target="_blank">View on BGG &rarr;</a>'
            f'  </div>'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)

    st.subheader("Why we think you'll like these games", anchor=None)

    explanation_text = st.session_state.get("recommendation_reason")
    if explanation_text is None:
        with st.spinner(""):
            explanation = generate_recommendation_reason(
                st.session_state.get("search_context", {}),
                recommendations_df,
            )
            st.session_state["recommendation_reason"] = explanation
            explanation_text = explanation

    if explanation_text:
        st.write(explanation_text)
    else:
        st.info("We couldn't generate a personalized explanation this time.")
else:
    st.warning("Unable to display recommendations. Please try running the search again.")

st.markdown("</div>", unsafe_allow_html=True)
