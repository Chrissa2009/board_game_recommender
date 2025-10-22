import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Board Game Recommender",
    page_icon="🎲",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main .block-container {
        background-color: #f3f4f6;
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] {
        background-color: #374151;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider label {
        color: white !important;
    }
    div[data-baseweb="select"] > div {
        max-height: 250px !important;
        overflow-y: auto !important;
    }
    </style>
""", unsafe_allow_html=True)

# Dropdown and filter options
popular_games = [
    "Wingspan", "Terraforming Mars", "Everdell", "Scythe", "Catan", "Azul",
    "7 Wonders", "Dominion", "Pandemic", "Gloomhaven", "Ticket to Ride",
    "Codenames", "Root", "Ark Nova", "Splendor", "Brass: Birmingham",
    "Carcassonne", "Dune: Imperium", "Spirit Island", "Concordia"
]

mechanics_list = [
    "Worker Placement", "Deck Building", "Set Collection", "Area Control",
    "Engine Building", "Tile Placement", "Card Drafting", "Negotiation",
    "Hand Management", "Network Building"
]

game_types = ["Family", "Strategy", "Party", "Cooperative", "Thematic", "Abstract"]

game_categories = [
    "Adventure", "Exploration", "Fantasy", "Fighting", "Miniatures",
    "Science Fiction", "Space Exploration", "Civilization", "Environmental"
]

playtimes = ["Any", "15-30 min", "30-60 min", "60-120 min", "120+ min"]

# --- Main layout ---
st.title("🎲 Board Game Recommender")
st.markdown("Discover your next favorite game based on your preferences and description.")
st.markdown("---")

# --- Sidebar for user inputs ---
with st.sidebar:
    st.header("Your Preferences")
    
    liked_games = st.multiselect(
        "👍 Liked Board Games",
        options=popular_games,
        default=["Wingspan", "Terraforming Mars"],
        help="You can select as many games as you want."
    )

    disliked_games = st.multiselect(
        "👎 Disliked Board Games",
        options=popular_games,
        help="You can select as many games as you want."
    )

    # Game Weight (renamed from complexity)
    game_weight = st.slider("Game Weight", 1.0, 5.0, (1.5, 4.0), 0.1)

    # Number of Players (as a range slider, with "8+" label for 8)
    num_players_range = st.slider(
        "Number of Players",
        1,
        8,
        (2, 4),
        1,
        format="%d"
    )
    # Display as “8+” if upper bound selected
    if num_players_range[1] == 8:
        st.caption(f"Preferred player range: {num_players_range[0]} – 8+ players")
    else:
        st.caption(f"Preferred player range: {num_players_range[0]} – {num_players_range[1]} players")

    selected_mechanics = st.multiselect("Preferred Mechanics", options=mechanics_list)
    play_time = st.selectbox("Play Time", options=playtimes)
    selected_game_types = st.multiselect("Game Type", options=game_types)

    # New: Game Category
    selected_game_categories = st.multiselect("Game Category", options=game_categories)

    description = st.text_area("Board Game Description (free-form)", placeholder="Describe the type of game you enjoy...")

    st.markdown("---")
    st.caption("💡 Adjust filters to refine recommendations.")

# --- Footer ---
st.markdown("---")
st.caption("Built with Streamlit • Data from BoardGameGeek")