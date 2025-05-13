from scipy.sparse import csr_matrix
import numpy as np

def reconstruct_tfidf_vector(z_binary: np.ndarray, xdash: csr_matrix, topk_indices: np.ndarray) -> np.ndarray:
    """Reconstruct the tf-idf vectorized form for perturbed sample z from its interpretable representation

            Args:
            z_binary (np.ndarray): The interpretable representation of perturbed samples z, an array of 0s and 1s
                indicating absence/presence of each word
            x_dash (csr_matrix): The tf-idf vectorized form of x
        
        Returns:
            array of floats: The coefficients of the trained surrogate k-lasso model.
    """
    x_dense = xdash.toarray().flatten()
    z_reconstructed = np.zeros(xdash.shape[1])
    z_reconstructed[topk_indices] = z_binary[topk_indices] * x_dense[topk_indices]
    return z_reconstructed