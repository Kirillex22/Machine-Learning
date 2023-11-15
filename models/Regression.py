import numpy as np

class MyLinearRegression:
         
    def __grad(self, X, y):
        m = self.w.shape[0]
        delta = self.__func(X) - y
        grad = np.empty(X.shape[1])
        for i in range(m):
            dw = 2*np.mean(delta*self.w[i])
            grad[i] = dw
            
        return grad
              
    def __func(self, X):
        return (X*self.w).sum(axis = 1)
        
    def __init__(self, num_iter, learning_rate):
        self.w = np.array([])
        self.num_iter = num_iter
        self.learning_rate = learning_rate

    def fit(self, X, y): 
        self.w = np.ones(X.shape[1])
        
        for i in range(self.num_iter):
            grad = self.__grad(X,y)
            self.w -= self.learning_rate*grad    
                      
    def predict(self, X):
        return self.__func(X)
        