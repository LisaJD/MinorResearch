import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple
import perturbation, reconstruction, similarity

def sample_around(x_sparse: csr_matrix) -> pd.DataFrame:
            # Convert sparse matrix to dense array (1D)
            x_dense = x_sparse.toarray().flatten()

            # Create perturbation: if the original value is non-zero, randomly choose 0 or 1; else, keep 0
            # z is binary 0 or 1 to indicate absence choosing or not choosing a word 
            z_binary = np.where(x_dense != 0, np.random.randint(0, 2, size=x_dense.shape), 0)

            return z_binary


def generate_perturbation_set(self, xdash: csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each instance xdash, generate num_samples set of three:
            1. z_binary, a perturbed sample point in representable binary format,
            2. z_dense, a reconstruction of the z binary interpretable format in tfidf vectorized form
            3. Sample weights 

        Args:
            xdash (float): The tf-idf vectorized form of x
        
        Returns:
            array of floats: The coefficients of the trained surrogate k-lasso model.
        """
        Z_binary = [] 
        Z_dense = [] 
        sample_weights = []

        for i in range(self.config.num_samples):
            if i%100==0:
                print(f"Completed {i} samples")

            z_binary = perturbation.sample_around(xdash)
            z_dense = reconstruction.reconstruct_tfidf_vector(z_binary, xdash, self.topk_indices)
            weight = similarity.similarity_kernel(self.distance_func, xdash.toarray().flatten(), z_dense, self.config.width)

            Z_binary.append(z_binary)
            Z_dense.append(z_dense)
            sample_weights.append(weight)
        
        return Z_binary, Z_dense, sample_weights