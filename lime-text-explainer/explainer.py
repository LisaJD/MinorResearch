import numpy as np
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
import explanation, reconstruction, perturbation, similarity, feature_extraction

@dataclass
class LimeConfig:
    num_samples: int = 5000
    num_features_to_explain: int = 10
    alpha: float = 0.001
    k: int = 1000
    width: int = 25

class LimeExplainer:

    def __init__(self, distance_func, config: LimeConfig):
        self.distance_func = distance_func
        self.config = config
        self.k_lasso = make_pipeline(
            Lasso(alpha=config.alpha, max_iter=10000, tol=1e-4)
        )
    
    def generate_explanation(self, xdash, classifier, vectorizer):
        """Generate a LIME explanation for a given instance x 

        Args:
            x (string): The raw value of x, ie. a paragraph of text.
            xdash (float): The tf-idf vectorized form of x
        
        Returns:
            array of floats: The coefficients of the trained surrogate k-lasso model.
        """
        self.topk_indices = feature_extraction.get_topk_indices(xdash, self.config.k)
        Z_binary, Z_dense, sample_weights = perturbation.generate_perturbation_set(self, xdash)

        y_pred = classifier.predict_proba(Z_dense)[:, 1]
        self.k_lasso.fit(Z_binary, y_pred, lasso__sample_weight=sample_weights)

        coef = self.k_lasso.named_steps['lasso'].coef_
        print(coef[coef != 0])
        feature_names = vectorizer.get_feature_names_out()

        return explanation.format_word_weights(coef, feature_names, self.topk_indices)
    
