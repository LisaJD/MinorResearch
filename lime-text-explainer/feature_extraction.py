import numpy as np
from scipy.sparse import csr_matrix

def get_topk_indices(xdash: csr_matrix, k: int) -> np.ndarray:
    """Restrict the number of features returned by explainer to K

            Args:
            xdash (csr_matrix): The tf-idf vectorized form of x
            k (int): The number of features to retain from original datapoint x's full feature set
        
        Returns:
            array of ints: The topk indices from xdash
    """
    topk_local_indices = np.argsort(xdash.data)[-k:]
    return xdash.indices[topk_local_indices]