# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
PLACEHOLDER_TEXT = "rgba(60, 60, 60, 0.6)"  # Placeholder gray

st.set_page_config(page_title="Board Game Recommender", layout="wide")

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
  max-width: 1600px !important;
  margin: 0 auto !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
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

# ========= HERO SECTION =========
st.markdown(
    """
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem auto 3rem auto;
        max-width: 1600px;
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

# ========= APP CONTENT =========
st.markdown("""
<div class="content-wrapper" style="
    max-width: 1600px;
    margin: 0 auto;
    padding: 2rem;
">
""", unsafe_allow_html=True)

st.markdown("---")

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_csv("../data/games.csv")

@st.cache_data
def load_mechanics():
    mechanics_df = pd.read_csv("../data/game_mechanics.csv", header=None, names=["mechanic"])
    return mechanics_df["mechanic"].dropna().sort_values().tolist()

@st.cache_data
def load_categories():
    categories_df = pd.read_csv("../data/game_categories.csv", header=None, names=["category"])
    return categories_df["category"].dropna().sort_values().tolist()

@st.cache_data
def load_game_types():
    game_types_df = pd.read_csv("../data/game_types.csv", header=None, names=["type"])
    return game_types_df["type"].dropna().sort_values().tolist()

@st.cache_data
def load_master_assets():
    cols = ["bgg_id", "name", "thumbnail", "image", "ImagePath", "bgg_link"]
    master_df = pd.read_csv("../data/games_master_data.csv", usecols=cols)
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

# ========== SIDEBAR ==========
st.sidebar.header("Your Preferences")

# --- CF inputs ---
liked_games = st.sidebar.multiselect("Liked Board Games", games_df["Name"].unique())
disliked_games = st.sidebar.multiselect("Disliked Board Games", games_df["Name"].unique())

# --- Filter inputs ---
year_range = st.sidebar.slider("Year Published", 1990, 2021, (2000, 2021))
rating_min, rating_max = st.sidebar.slider(
    "Rating Range", 1.0, 10.0, (6.0, 9.5), step=1.0, format="%d"
)

# --- CBF inputs ---
min_players, max_players = st.sidebar.slider("Player Count Range", 1, 20, (2, 4))
play_time = st.sidebar.selectbox("Play Time", ["<30 mins", "30–60 mins", "60–90 mins", "90–120 mins", ">120 mins"])
complexity = st.sidebar.slider("Complexity", 1.0, 5.0, (2.3, 3.6), 0.1)
mechanics = st.sidebar.multiselect("Game Mechanics", mechanics_options)
categories = st.sidebar.multiselect("Game Category", categories_options)
game_type = st.sidebar.multiselect("Game Type", game_type_options)

# --- LLM input ---
description = st.sidebar.text_area("Describe the kind of board game you enjoy",
                                   placeholder="Example: I like strategic games with some luck and engine building mechanics.")

# ========== FILTER DISPLAY ==========
filtered_df = games_df.copy()

if "YearPublished" in filtered_df.columns:
    filtered_df = filtered_df.query("YearPublished >= @year_range[0] and YearPublished <= @year_range[1]")
if "AvgRating" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["AvgRating"] >= rating_min)
        & (filtered_df["AvgRating"] <= rating_max)
    ]
if {"MinPlayers", "MaxPlayers"} <= set(filtered_df.columns):
    filtered_df = filtered_df.query("MinPlayers <= @max_players and MaxPlayers >= @min_players")

st.subheader("Games Matching Your Basic Filters")
st.write(f"{len(filtered_df)} games found.")

if not filtered_df.empty:
    if "Category" in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(7, 3))
        filtered_df["Category"].value_counts().head(10).sort_values().plot(
            kind="barh", ax=ax, color="white"
        )
        ax.set_facecolor(BACKGROUND_COLOR)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_title("Top Game Categories", color="white")
        ax.set_xlabel("Count", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        st.pyplot(fig)

    # --- Highlight Top Matches ---
    highlight_df = filtered_df.head(8)[["BGGId", "Name", "AvgRating"]].copy()
    if not highlight_df.empty:
        highlight_df = highlight_df.merge(
            master_assets, left_on="BGGId", right_index=True, how="left"
        )

    # Card styles
    st.markdown("""
    <style>
    .game-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1.5rem;margin-top:1.5rem}
    .game-card{background-color:#1A2B22;border-radius:16px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,.25);transition:transform .2s, box-shadow .2s}
    .game-card:hover{transform:translateY(-4px);box-shadow:0 12px 32px rgba(0,0,0,.35)}
    .game-image{width:100%;height:200px;object-fit:cover}
    .game-content{padding:1rem 1.25rem;color:#F5F5E6}
    .game-title{font-size:1.2rem;font-weight:700;margin-bottom:.4rem}
    .game-meta{font-size:.9rem;color:#B8B8B8;display:flex;gap:1rem;align-items:center;margin-bottom:.6rem}
    .game-desc{font-size:.95rem;color:#CCC;margin-bottom:.8rem}
    .game-link{color:#A4B465;font-weight:600;text-decoration:none}
    .game-link:hover{text-decoration:underline}
    </style>
    """, unsafe_allow_html=True)

    # Build grid HTML (no leading whitespace)
    cards = ['<div class="game-grid">']
    for _, row in highlight_df.iterrows():
        image_url = (row.get("asset_url") or DEFAULT_THUMBNAIL)
        title = str(row["Name"])
        rating_val = row.get("AvgRating")
        rating_text = f"{float(rating_val):.2f}" if pd.notna(rating_val) else "N/A"
        desc = f"Average Rating: {rating_text}"

        cards.append(
            f'<div class="game-card">'
            f'  <img src="{image_url}" class="game-image" alt="{title}">'
            f'  <div class="game-content">'
            f'    <div class="game-title">{title}</div>'
            f'    <div class="game-meta">⭐ {rating_text}</div>'
            f'    <div class="game-desc">{desc}</div>'
            f'    <a href="https://boardgamegeek.com/" class="game-link" target="_blank">View on BGG →</a>'
            f'  </div>'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
