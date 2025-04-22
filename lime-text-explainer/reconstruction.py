from scipy.sparse import csr_matrix
import numpy as np

def reconstruct_tfidf_vector(z_binary: np.ndarray, xdash: csr_matrix, topk_indices: np.ndarray) -> np.ndarray:
    x_dense = xdash.toarray().flatten()
    # z_binary = z_binary.toarray().flatten()
    z_reconstructed = np.zeros(xdash.shape[1])
    # z_binary is size 10, 
    z_reconstructed[topk_indices] = z_binary[topk_indices] * x_dense[topk_indices]
    return z_reconstructed