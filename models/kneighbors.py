import numpy as np
from statistics import mode

class kNN:

    def Neighbors(self, point):
        f = lambda x: x[0]
        distances = []
        for i in range(self.x.shape[0]):
                distance = self.Euclid(point, self.x[i])
                distances.append([distance, self.y[i]])
                                  
        distances.sort(key=f)
                                  
        return distances
        
    def Euclid(self, x_1, x_2):
        distance = np.sqrt(np.sum((x_2 - x_1)**2))
  
        return distance

    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.x = X_train
        self.y = y_train

    def predict(self, X_test):
        result = []
        for point in X_test:
            neighbors = self.Neighbors(point)
            neighbors = neighbors[:self.k]
            classes = []
            for neighbor in neighbors:
                classes.append(neighbor[1])
                
            result.append(mode(classes))
            
        return np.array(result)
            
            
            
        
        
        