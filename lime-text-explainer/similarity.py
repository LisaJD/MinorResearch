from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


def similarity_kernel(distance_func, x: np.ndarray, z: np.ndarray, width: float = 0.75) -> float:
    """Calculate the similarity between x and z, to be used as sample weights for surrogate model

        Args:
            distance_func (function): distance function to use eg. cosine distance
            x (np.ndarray): The tf-idf vectorized form of the instance 
                x to be explained
            z (np.ndarray): The interpretable form of the perturbed instance z
        
        Returns:
            float: 
    """
    width = width * np.sqrt(np.count_nonzero(x))
    return np.exp(np.negative((distance_func(x,z)**2)) / width**2)

def cosine_distance(x: np.ndarray, z: np.ndarray) -> float:
    """Calculate the cosine distance between A, the tf-idf vectorized form of the instance 
     x to be explained 

        Args:
            x (np.ndarray): The tf-idf vectorized form of the instance 
                x to be explained
            z (np.ndarray): The interpretable form of the perturbed instance z
        
        Returns:
            float: The cosine distance between x and z.
    """
    norm_x = np.linalg.norm(x)
    norm_z = np.linalg.norm(z)
    if norm_x == 0 or norm_z == 0:
        print("Norm was 0")
        return 1.0  # max distance if either is all zeros
    cosine_similarity = np.dot(x, z) / (norm_x * norm_z)
    return 1 - cosine_similarity