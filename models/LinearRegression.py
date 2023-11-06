import numpy as np
import Metrics

class MyLinearRegression:
       
    def __grad(self, k, X):
        
        coef = 0 
        for i in range(X.shape[0]):
            coef += X[i, k]
            
        return coef
    
    def __func(self, X):
        
        return (X*self.w).sum(axis = 1)
        
    def __init__(self, num_iter, learning_rate):
    
        self.w = np.array()
        self.num_iter = num_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        
        grad_coefs = []
        
        for i in range(X.shape[1]):
            grad_coefs.append(__grad(i, X))  

        grad = np.array(grad_coefs)
        
        self.w = ones(X.shape[1])         
        
        for i in range(0, num_iter):
            
            self.w = self.w - self.learning_rate*L_grad

    def predict(self, X):
        return __func(X)
        