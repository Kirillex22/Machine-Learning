import numpy as np

class myPCA:
    def __init__(self, n_comp):
        self.n_comp = n_comp
    def fit_transform(self, X):
        X_cov = np.cov(X.T)
        eigen_pares = np.column_stack(np.linalg.eig(X_cov))
        pares_sorted_desc = eigen_pares[eigen_pares[:, 1]. argsort ()[::-1]]
        eigen_pares = eigen_pares[:self.n_comp] 
        eigen_vectors = np.delete(eigen_pares, 0, axis=1)            
        return np.dot(X, np.column_stack(eigen_vectors))