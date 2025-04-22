from typing import List, Tuple
import numpy as np

def format_word_weights(coef, feature_names, topk_indices) -> List[Tuple[str, float]]:
    # Find indices of non-zero coefficients
    nonzero_coef_indices = np.flatnonzero(coef)

    # Get (feature name, coefficient) pairs
    nonzero_feature_weights = [(feature_names[i], coef[i]) for i in nonzero_coef_indices]

    # Sort by absolute value
    sorted_feature_weights = sorted(nonzero_feature_weights, key=lambda x: abs(x[1]), reverse=True)

    return sorted_feature_weights

# TODO add a method for printing out the results
# 1. Show prediction for data instance, and probability of each class
# 2. Show topk word feature weight in descending order
