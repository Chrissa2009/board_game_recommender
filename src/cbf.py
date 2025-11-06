import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

# load precomputed CBF data
base_dir = os.path.dirname(os.path.abspath(__file__))
cbf_path = os.path.join(base_dir, "..", "data", "precomputed_CBF.pkl")
with open(cbf_path, "rb") as f:
    _cbf_data = pickle.load(f)

games_df = _cbf_data["games_df"]
mlb_game_categories = _cbf_data["mlb_game_categories"]
mlb_game_mechanics = _cbf_data["mlb_game_mechanics"]
mlb_game_types = _cbf_data["mlb_game_types"]
scaler = _cbf_data["scaler"]
weighted_features = _cbf_data["weighted_features"]  # use this as the feature matrix

# get mean value
def mean_or_default(value, default):
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
        return np.mean(value)
    elif isinstance(value, (int, float)):
        return value
    return default

# get CBF scores
def get_cbf_scores(attributes: dict):

    n_games = games_df.shape[0]

    # Build query vectors
    cat_vec = mlb_game_categories.transform(
        [attributes.get('game_categories', [])]
    ) if 'game_categories' in attributes else np.zeros((1, len(mlb_game_categories.classes_)))

    mech_vec = mlb_game_mechanics.transform(
        [attributes.get('game_mechanics', [])]
    ) if 'game_mechanics' in attributes else np.zeros((1, len(mlb_game_mechanics.classes_)))

    type_vec = mlb_game_types.transform(
        [attributes.get('game_types', [])]
    ) if 'game_types' in attributes else np.zeros((1, len(mlb_game_types.classes_)))

    # Numeric features
    game_weight_avg = mean_or_default(attributes.get('game_weight'), 2.5)
    players_avg = mean_or_default(attributes.get('players'), 3)
    play_time_avg = mean_or_default(attributes.get('play_time'), 90)

    numeric_vec = np.array([[game_weight_avg, players_avg, play_time_avg]])
    numeric_vec_scaled = scaler.transform(numeric_vec)

    # Combine feature vector (match weighted_features)
    query_vector = np.hstack([
        cat_vec * 1.5,
        mech_vec * 2.0,
        type_vec * 1.0,
        numeric_vec_scaled * 0.5
    ])

    # compute similarity
    cbf_scores = cosine_similarity(query_vector, weighted_features).flatten()

    # normalize
    if cbf_scores.max() > cbf_scores.min():
        cbf_scores_norm = (cbf_scores - cbf_scores.min()) / (cbf_scores.max() - cbf_scores.min())
    else:
        cbf_scores_norm = np.zeros_like(cbf_scores)

    return cbf_scores_norm

if __name__ == "__main__":
    import pandas as pd

    # Example user input
    sample_input = {
        "game_categories": ["Abstract / Strategy", "Animals / Nature"],
        "game_mechanics": ["Team Play"],
        "game_types": ["Customizable"],
        "game_weight": [2.8],
        "players": [4],
        "play_time": [90],
    }

    # Compute scores
    scores = get_cbf_scores(sample_input)
    print ("CBF Scores: ", scores)
    print("CBF Scores Length: ", len(scores))