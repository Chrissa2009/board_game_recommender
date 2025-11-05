# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from recommender_openai import recommend_games, category_columns

# --- APP CONFIG ---
st.set_page_config(page_title="Board Game Recommender", layout="wide")
st.title("Board Game Recommender")
st.markdown("Find the perfect board game using AI and real data.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/libbykirk/Downloads/games_500csv.csv")
    return df

games_df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("User Preferences")
st.sidebar.subheader("Please select at least one")

# 1. Liked Board Games
if "name" in games_df.columns:
    game_names = sorted(games_df["name"].dropna().unique())
    liked_games = st.sidebar.multiselect(
        "Search and select liked board games",
        options=game_names,
        placeholder="Type to search liked games...",
    )
else:
    liked_games = []

# 2. Disliked Board Games
if "name" in games_df.columns:
    disliked_games = st.sidebar.multiselect(
        "Search and select disliked board games",
        options=game_names,
        placeholder="Type to search disliked games...",
    )
else:
    disliked_games = []

# 3. Playtime
if "time_avg" in games_df.columns:
    playtime_buckets = {
        "< 30 minutes": (0, 30),
        "30 – 60 minutes": (30, 60),
        "60 – 90 minutes": (60, 90),
        "90 – 120 minutes": (90, 120),
        "> 120 minutes": (120, float("inf")),
    }
    selected_playtime = st.sidebar.selectbox(
        "Select average playtime",
        options=["All"] + list(playtime_buckets.keys())
    )
else:
    selected_playtime = "All"

# 4. Number of Players
if "players_min" in games_df.columns and "players_max" in games_df.columns:
    min_players = int(games_df["players_min"].min())
    max_players = int(games_df["players_max"].max())
    player_range = st.sidebar.slider(
        "Number of players",
        min_value=min_players,
        max_value=max_players,
        value=(min_players, max_players),
        step=1,
    )
else:
    player_range = (1, 20)

# 5. Game Categories
all_categories = set()
for cats in games_df.get("game_categories", []):
    if pd.notna(cats):
        all_categories.update([c.strip() for c in cats.split(";")])
all_categories = sorted(list(all_categories))
selected_category = st.sidebar.selectbox(
    "Choose a category",
    options=["All"] + all_categories
)

# 6. Year Published
if "year_published" in games_df.columns:
    min_year = int('1990')
    max_year = int(games_df["year_published"].max())
    year_range = st.sidebar.slider(
        "Year Published Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
else:
    year_range = None

# 7. Mechanics
if "game_mechanics" in games_df.columns:
    all_mechanics = set()
    for mech in games_df["game_mechanics"]:
        if pd.notna(mech):
            all_mechanics.update([m.strip() for m in mech.split(";")])
    all_mechanics = sorted(list(all_mechanics))
    selected_mechanic = st.sidebar.selectbox(
        "Select mechanic",
        options=["All"] + all_mechanics
    )
else:
    selected_mechanic = "All"

# 8. Complexity
if "game_weight" in games_df.columns:
    complexity_range = st.sidebar.slider(
        "Game Complexity (1 = simple, 5 = complex)",
        min_value=1.0,
        max_value=5.0,
        value=(1.0, 5.0),
        step=0.2
    )
else:
    complexity_range = (1.0, 5.0)

# 9. Free Text Filter
user_text_filter = st.sidebar.text_input(
    "Describe what you're looking for in a game:",
    placeholder="e.g. cooperative card game with strategy and storytelling"
)

# 10. Number of AI recommendations
top_n = st.sidebar.slider("Number of AI recommendations", 1, 10, 3)

filters_selected = bool(
    liked_games
    or disliked_games
    or selected_category != "All"
    or selected_playtime != "All"
    or player_range != (min_players, max_players)
    or selected_mechanic != "All"
    or complexity_range != (1.0, 5.0)
    or user_text_filter
    or year_range != (min_year, max_year)
)

# --- SUBMIT BUTTON ---
submit = st.sidebar.button("Apply Inputs", disabled=not filters_selected)

if submit:
# --- FILTER DATA ---
    filtered_df = games_df.copy()

# Filter liked/disliked games
    if liked_games:
        filtered_df = filtered_df[filtered_df["name"].isin(liked_games)]
    if disliked_games:
        filtered_df = filtered_df[~filtered_df["name"].isin(disliked_games)]

# Filter category
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["game_categories"].str.contains(selected_category, na=False)]

# Filter playtime
    if selected_playtime != "All":
        min_t, max_t = playtime_buckets[selected_playtime]
        filtered_df = filtered_df[(filtered_df["time_avg"] >= min_t) & (filtered_df["time_avg"] <= max_t)]

# Filter number of players
    filtered_df = filtered_df[
        (filtered_df["players_min"] <= player_range[1]) &
        (filtered_df["players_max"] >= player_range[0])
]

# Filter year published
    if year_range:
        filtered_df = filtered_df[
            (filtered_df["year_published"] >= year_range[0]) &
            (filtered_df["year_published"] <= year_range[1])
    ]

# Filter mechanics
    if selected_mechanic != "All":
        filtered_df = filtered_df[filtered_df["game_mechanics"].str.contains(selected_mechanic, na=False)]

# Filter complexity
    filtered_df = filtered_df[
        (filtered_df["game_weight"] >= complexity_range[0]) &
        (filtered_df["game_weight"] <= complexity_range[1])
]

# Filter free text
    if user_text_filter:
        filtered_df = filtered_df[
            filtered_df["description"].str.contains(user_text_filter, case=False, na=False)
        ]

# --- SHOW LOCAL STATS ---
st.subheader("Game Overview")
#st.write(f"Found {len(filtered_df)} games matching your preferences.")

if len(filtered_df) == 0:
    st.write("Input Needed")
else:
    # Table of filtered games
    #st.dataframe(filtered_df[["name", "year_published", "players_min", "players_max", "game_weight"]].head(10))
    top_games = filtered_df.head(10)
    col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1]) 
    col1.write("**Thumbnail**") 
    col2.write("**Name**") 
    col3.write("**Year**") 
    col4.write("**Min Players**") 
    col5.write("**Max Players**") 
    col6.write("**Complexity**") 
    
    for _, row in top_games.iterrows(): 
        col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
        # Only show image if URL exists
        if pd.notna(row.get("thumbnail")) and row["thumbnail"] != "":
            col1.image(row["thumbnail"], width=80) 
        else: col1.write("N/A") 
        col2.write(row["name"])
        col3.write(row["year_published"])
        col4.write(row["players_min"])
        col5.write(row["players_max"])
        col6.write(row["game_weight"])  

# --- OPENAI RECOMMENDATION SECTION ---
st.markdown("---")
st.subheader("AI Recommendations")
if st.button("Get AI Recommendations"):
    with st.spinner("Asking the AI for recommendations..."):
        recommendations = recommend_games(liked_games, selected_category, top_n)
        st.markdown("Recommended Games:")
        st.write(recommendations)
