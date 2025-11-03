from typing import Union
import numpy as np

def get_hybrid_scores(
    cf_scores: Union[np.ndarray, list],
    cbf_scores: Union[np.ndarray, list],
    llm_scores: Union[np.ndarray, list],
    alpha: float = 0.5,
    beta: float = 0.33
) -> np.ndarray:
    cf_scores = np.array(cf_scores)
    cbf_scores = np.array(cbf_scores)
    llm_scores = np.array(llm_scores)
    
    hybrid_scores = ((cf_scores * alpha) + (cbf_scores * (1-alpha))) * (1-beta) + (llm_scores * beta)
    return hybrid_scores
