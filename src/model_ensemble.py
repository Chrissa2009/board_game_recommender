import pandas as pd
import numpy as np
import warnings
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cbf import get_cbf_scores
from cf import get_cf_scores
from llm import get_llm_scores, category_columns

warnings.filterwarnings('ignore')

### Load games into games_df
games_file = "../data/games_master_data.csv"

def semicolon_to_list(value):
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, list):  # prevent double conversion
        return value
    return [item.strip() for item in str(value).split(';') if item.strip()]

games_df = pd.read_csv(
    games_file,
    usecols=['bgg_id',
             'name',
             'description',
             'image',
             'thumbnail',
             'bgg_link',
             'avg_rating',
             'bgg_rating',
             'users_rated',
             'game_weight',
             'players_min',
             'players_max',
             'players_best',
             'time_min',
             'time_max',
             'time_avg',
             'simple_game_mechanics',
             'simple_game_categories',
             'game_types',
             'year_published'],
    
    converters={'simple_game_mechanics': semicolon_to_list,
                'simple_game_categories': semicolon_to_list,
                'game_types': semicolon_to_list},
    
    dtype={'bgg_id':        'int64',
           'avg_rating':    'float64',
           'bgg_rating':    'float64',
           'users_rated':   'int64',
           'game_weight':   'float64',
           'players_best':  'float64',
           'players_min':   'int64',
           'players_max':   'int64',
           'players_best':  'float64',
           'time_min':      'int64',
           'time_max':      'int64',
           'time_avg':      'int64'})

games_df.rename(columns={'simple_game_categories': 'game_categories', 'simple_game_mechanics': 'game_mechanics'}, inplace=True)

games_df = games_df.set_index("bgg_id", drop=False)
n_games = games_df.shape[0]

### get ensemble score
def ensemble_scores(liked_games=None,
                    disliked_games=None,
                    exclude_games=None,
                    attributes=None,
                    description=None,
                    alpha: float = 0.5,
                    beta: float = 0.33,
                    n_recommendations: int = 5) -> pd.DataFrame:
    """
:    Ensemble CF, CBF, and LLM models using a hybrid weighting formula and filter

    Parameters
    ----------
    iked_games - list/array of bgg_ids (integers)
    disliked_games  - list/array of bgg_ids (integers)
    exclude_games  - list/array of bgg_ids (integers)
    description - string for llm
    attributes - dictionary
        attributes = {
        'game_types': ['Abstract Game','Family Game'] # list of game types
        'game_categories': ['Science Fiction / Space'], # list of game categories
        ''game_mechanics':['Area Control', 'Turn Order', 'Worker Placement'], # list of mechanics
        'game_weight': [1.5, 3.5],  # list of min and max weight
        'players': [2,5], # list of min and max players
        'play_time': [30,90], # list of min and max play time in minutes
        'min_rating':[6.0], # single value list of min rating
        'year_published':[2010,2025] # list of min, max year published

    Returns: pandas datafram of top-n games and these colums

    'bgg_id', 'name', 'avg_rating', 'game_categories',
    'game_mechanics', 'game_weight', 'game_types',
    'year_published', 'players_min', 'players_max'
    'recommender_score', 'cf_score_component', 'cbf_score_component', 'llm_score_component'
    
    
    -------
    pd.DataFrame
        Combined recommendations with composite score.
    """
    # get cf_scores
    # cf_scores = get_cf_scores(liked_games, disliked_games)
    cf_scores = np.zeros(n_games)
    
    # get cbf_scores
    cbf_scores = get_cbf_scores(attributes=attributes)

    # get llm_scores
    min_players = 1
    if attributes and 'players' in attributes:
        players = attributes['players']
        if isinstance(players, (list, tuple)) and len(players) > 0:
            min_players = players[0]

    category = category_columns[0]
    if attributes and 'game_categories' in attributes:
        cats = attributes['game_categories']
        if isinstance(cats, (list, tuple)) and len(cats) > 0 and cats[0]:
            category = cats[0]

    llm_scores = get_llm_scores(
        user_description=description or "",
        min_players=min_players,
        category=category,
    )
    
    # convert and validate input
    cf_scores = np.array(cf_scores)
    cbf_scores = np.array(cbf_scores)
    llm_scores = np.array(llm_scores)

    # handle zero-score cases
    cf_zero = np.all(cf_scores == 0)
    cbf_zero = np.all(cbf_scores == 0)
    llm_zero = np.all(llm_scores == 0)

    # weight if one or two vectors are zero
    if cf_zero and cbf_zero:
        beta = 1.0  # rely entirely on LLM
    elif llm_zero:
        beta = 0.0  # rely entirely on CF/CBF

    if cf_zero and not cbf_zero:
        alpha = 0.0
    elif cbf_zero and not cf_zero:
        alpha = 1.0

    # compute hybrid components
    cf_component = cf_scores * alpha
    cbf_component = cbf_scores * (1 - alpha)
    combined_cf_cbf = (cf_component + cbf_component) * (1 - beta)
    llm_component = llm_scores * beta
    hybrid_scores = combined_cf_cbf + llm_component

    # start with hybrid scores
    final_scores = hybrid_scores.copy()

    # if empty attributes
    liked_games = liked_games or []
    disliked_games = disliked_games or []
    exclude_games = exclude_games or []
    attributes = attributes or {}


    # --- Apply exclusion filters ---
    for gid in liked_games + disliked_games + exclude_games:
        if gid in games_df.index:
            idx = games_df.index.get_loc(gid)
            final_scores[idx] = 0
    
    # --- Apply attribute filters ---
    if attributes:
        # Multi-label attributes
        for attr_name in ['game_categories', 'game_mechanics', 'game_types']:
            selected = attributes.get(attr_name, [])
            if selected and any(isinstance(s, str) and s.strip() for s in selected):
                selected_clean = [s.strip().lower() for s in selected if isinstance(s, str) and s.strip()]
                mask = games_df[attr_name].apply(
                    lambda ga: isinstance(ga, list) and len(ga) > 0 and 
                               any(isinstance(a, str) and a.strip().lower() in selected_clean for a in ga)
                )
                final_scores[~mask.values] = 0

        # Numeric attributes
        if 'game_weight' in attributes:
            weight_range = attributes['game_weight']
            if isinstance(weight_range, (list, tuple)) and len(weight_range) == 2:
                w_min, w_max = weight_range
                mask = (games_df['game_weight'] >= w_min) & (games_df['game_weight'] <= w_max)
                final_scores[~mask.values] = 0

        if 'players' in attributes:
            players_range = attributes['players']
            if isinstance(players_range, (list, tuple)) and len(players_range) == 2:
                p_min, p_max = players_range
                mask = (games_df['players_max'] >= p_min) & (games_df['players_min'] <= p_max)
                final_scores[~mask.values] = 0

        if 'play_time' in attributes:
            time_range = attributes['play_time']
            if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
                t_min, t_max = time_range
                mask = (games_df['time_max'] >= t_min) & (games_df['time_min'] <= t_max)
                final_scores[~mask.values] = 0

        if 'year_published' in attributes:
            year_range = attributes['year_published']
            if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
                y_min, y_max = year_range
                mask = (games_df['year_published'] >= y_min) & (games_df['year_published'] <= y_max)
                final_scores[~mask.values] = 0

        if 'min_rating' in attributes:
            min_rating = attributes['min_rating']
            if isinstance(min_rating, (list, tuple)) and len(min_rating) > 0:
                min_rating = min_rating[0]
                mask = (games_df['avg_rating'] >= min_rating)
                final_scores[~mask.values] = 0
                    
            # Select top N recommendations ---
            valid_idx = np.where(final_scores >= 0.01)[0]
            if len(valid_idx) == 0:
                return pd.DataFrame(), np.array([]), np.array([]), np.array([]), np.array([])

    top_n_idx = valid_idx[np.argsort(final_scores[valid_idx])[::-1][:n_recommendations]]

    recommendations = games_df.iloc[top_n_idx][[
        'bgg_id', 'name', 'avg_rating', 'game_categories',
        'game_mechanics', 'game_weight', 'game_types',
        'year_published', 'players_min', 'players_max'
    ]].copy()

    recommendations['recommender_score'] = final_scores[top_n_idx].round(4)
    recommendations['cf_score_component'] = cf_component[top_n_idx].round(4)
    recommendations['cbf_score_component'] = cbf_component[top_n_idx].round(4)
    recommendations['llm_score_component'] = llm_component[top_n_idx].round(4)

    return recommendations

### Show recommendationsget_hybrid_recommendations
###
def display_recommendations(liked_games,
                            disliked_games,
                            exclude_games,
                            attributes,
                            description,
                            n_recommendations=5,
                            alpha=0.5,
                            beta=0.33):
    
    recommendations = ensemble_scores(liked_games, disliked_games, exclude_games,
                                               attributes=attributes, description=description,
                                               n_recommendations=n_recommendations,
                                               alpha=alpha, beta=beta)

    #print(recommendations)
    
    # --- Helper: get names from IDs ---
    def get_game_names(id_list):
        if not id_list:
            return "None"
        names = [games_df.loc[g]["name"] for g in id_list if g in games_df.index]
        return ", ".join(names) if names else "None"

    # --- Display search criteria ---
    print("=" * 120)
    print("Find games based on...")
    print(f"  Liking:    {get_game_names(liked_games)}")
    print(f"  Disliking: {get_game_names(disliked_games)}")
    print(f"  Excluding: {get_game_names(exclude_games)}")

    for key, values in (attributes or {}).items():
        if values:
            print(f"  {key}: {', '.join(str(v) for v in values)}")
    print("=" * 120)

    # --- Handle no results ---
    if recommendations is None or recommendations.empty:
        print("\nNo recommendations found.")
        return

    # --- Show results ---
    print("\nRecommendations:\n")
    for _, row in recommendations.iterrows():
        bgg_id = row["bgg_id"]
        score = row["recommender_score"]

        if bgg_id not in games_df.index:
            print(f"Game ID {bgg_id} not found in games_df.")
            continue

        game = games_df.loc[bgg_id]

        print(f"*** {bgg_id} {game['name']:<35} Recommender score: {score:.4f}")
        print(f"    User Rating: {game.get('avg_rating', 'N/A'):.2f}")
        print(f"    Categories: {', '.join(game.get('game_categories', []))}")
        print(f"    Game Types: {', '.join(game.get('game_types', []))}")
        print(f"    Mechanics:  {', '.join(game.get('game_mechanics', []))}")
        print(f"    Year: {int(game.get('year_published', 0))} "
              f"| Players: {int(game.get('players_min', 0))}–{int(game.get('players_max', 0))}\n")

### Run Recommender
###

# Example 1
liked_games = [163412, 230802]
disliked_games = [38929, 285533]
exclude_games = [184477]
description = ''
attributes = {'game_types': ['Abstract Game', 'Family Game'],
              'game_categories': ['Abstract / Strategy', 'Puzzle / Logic'],
              'game_weight': [1.5, 2.8],
              'players': [2,5],
              'play_time': [],
              'min_rating':[7.5],
              'year_published':[1999,2025]}

display_recommendations(liked_games, disliked_games, exclude_games, attributes, description, n_recommendations=5, alpha=0.5, beta=0.33)

# Example 2
liked_games = [167791]
disliked_games = []
description = ''
exclude_games = [175155, 221194]
attributes = {'game_types': ['Strategy Game'],
              'game_categories': ['Science Fiction / Space'],
              'game_mechanics':['Dice Rolling', 'Set Collection'],
              'game_weight': [2.0, 3.9],
              'players': [],
              'play_time': [60,180],
              'min_rating':[7.5],
              'year_published':[1999,2025]}

display_recommendations(liked_games, disliked_games, exclude_games, attributes, description, n_recommendations=5, alpha=0.5, beta=0.33)

# Example 3
liked_games = [31260, 13]
disliked_games = []
description = ''
exclude_games = [58421]
attributes = {'game_types': ['Strategy Game','Family Game'],
              'game_categories': ['Animals / Nature'],
              'game_mechanics':['Dice Rolling', 'Set Collection','Drafting'],
              'game_weight': [2.0, 3.9],
              'players': [2,5],
              'play_time': [60,180],
              'min_rating':[7.5],
              'year_published':[1999,2025]}

display_recommendations(liked_games, disliked_games, exclude_games, attributes, description, n_recommendations=5, alpha=0.5, beta=0.33)

# Example 4
liked_games = [303954]
disliked_games = []
exclude_games = []
description = 'historical games preferably with economics and trading'
attributes = {'game_types': ['Strategy Game','Family Game'],
              'game_categories': ['Historical Eras','Educational'],
              'game_mechanics':['Simulation', 'Dice Rolling', 'Worker Placement','Set Collection','Drafting'],
              'game_weight': [3.0, 4.0],
              'players': [3,7],
              'play_time': [60,300],
              'min_rating':[7.5],
              'year_published':[2000,2025]}

display_recommendations(liked_games, disliked_games, exclude_games, attributes, description, n_recommendations=5, alpha=0.5, beta=0.33)