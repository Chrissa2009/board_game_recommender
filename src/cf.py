"""
cf_model.py
Collaborative Filtering model for board game recommendations.
This module computes similarity based on user ratings.
"""

import pandas as pd
import numpy as np

def fold_in_implicit_user(V, liked_items, alpha=5, lambda_=0.03):
    """
    Compute a new user vector given items they've liked (implicit feedback).
    """
    liked_items = np.array(liked_items, dtype=int).flatten()
    V_i = V[liked_items]
    print(liked_items)
    # confidence weights
    C_i = 1 + alpha * np.ones(len(liked_items), dtype=np.float32)
    
    A = V_i.T @ (C_i[:, None] * V_i) + lambda_ * np.eye(V.shape[1])
    b = V_i.T @ (C_i * np.ones(len(liked_items), dtype=np.float32))
    
    u_new = np.linalg.solve(A, b)
    print("DEBUG V shape:", V.shape)
    print("DEBUG liked_items dtype:", liked_items.dtype)
    print("DEBUG liked_items:", liked_items)
    return u_new

def get_cf_scores(
    ratings: np.ndarray = np.array([]),
    V = None
    #games_path: str = "../data/games.csv",
):
    """
    Compute CF-based recommendation scores based on pre-computed item embedding matrix V and a vector of movie IDs of user likes

    Parameters
    ----------
    ratings : array
        array of indices of liked items
    V : matrix
        item embedding matrix used to predict CF scores

    Returns
    -------
    scores
        array of ratings for each board game
    """

    #load board game embeddings if it wasn't passed in
    if V is None:
        data = np.load('../data/V_final_quantized.npz')
        V = data["V_q"].astype(np.float32) / 127 * data["scale"]
    
    # calculte user embeddings based on inputted likes 
    u = fold_in_implicit_user(V,liked_items=ratings, alpha=5, lambda_=0.3)

    #calculate scores
    scores = V.dot(u)

    #normalize between 0 and 1 
    scores = (scores - min(scores)) / (max(scores) - min(scores))

    # returns array of scores per movie
    return scores

if __name__ == "__main__":
    # Example usage
    example_ratings = np.array([10, 50, 200])
    scores = get_cf_scores(ratings=example_ratings)
    print("CF Scores:", scores)
    print("CF Scores Length:", len(scores))
    