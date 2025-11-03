from typing import Union, Dict
import numpy as np

def get_hybrid_scores(
    cf_scores: Union[np.ndarray, list],
    cbf_scores: Union[np.ndarray, list],
    llm_scores: Union[np.ndarray, list],
    alpha: float = 0.5,
    beta: float = 0.33
) -> Dict[str, np.ndarray]:
    # Convert inputs to numpy arrays
    cf_scores = np.array(cf_scores)
    cbf_scores = np.array(cbf_scores)
    llm_scores = np.array(llm_scores)

    # Compute weighted components
    cf_component = cf_scores * alpha
    cbf_component = cbf_scores * (1 - alpha)
    combined_cf_cbf = (cf_component + cbf_component) * (1 - beta)
    llm_component = llm_scores * beta

    # Final hybrid score
    hybrid_scores = combined_cf_cbf + llm_component

    # Return all components
    return hybrid_scores,cf_component, cbf_component,llm_component
