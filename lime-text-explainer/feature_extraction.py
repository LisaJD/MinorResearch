import numpy as np
from scipy.sparse import csr_matrix

def get_topk_indices(xdash: np.ndarray, k: int) -> np.ndarray:
    # TODO deal with all 0 matrices
    topk_local_indices = np.argsort(xdash.data)[-k:]
    return xdash.indices[topk_local_indices]