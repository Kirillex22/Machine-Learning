import numpy as np

class Kmeans:
       
    def euclid(self, x_1, x_2):
        return np.sqrt(np.sum((x_2 - x_1)**2))

    def __init__(self, k = 1, max_iter = 10000, eps = 1e-6):
        self.k = k
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, X):
        self.predict = np.empty(X.shape[0])
        self.labels = np.arange(0, self.k, 1)
        self.centers = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for j in range(self.max_iter):
            last_predict = self.predict
            self.update_predicts(X)
            
            if np.all(last_predict == self.predict):
                break
                
            last_centers = self.centers
            self.update_centers(X)

            if (np.mean((last_centers - self.centers)**2) < eps):
                break        


    def update_predicts(self, X):
        for i in range(X.shape[0]):
            clusters = [self.euclid(X[i], centr) for centr in self.centers]
            cluster = clusters.index(min(clusters))
            self.predict[i] = cluster

    def update_centers(self, X):
        for label in self.labels:
            indices = [i for i, x in enumerate(self.predict) if x==label]
            self.centers[label] = np.mean(X[indices])
            
            


