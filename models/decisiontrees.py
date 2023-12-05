import numpy as np
import math
from scipy.stats import mode

class MyDecisionTreeClassifier:
    
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.current_depth = 0
        

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model = self.tree(X, y)
        
            
    def tree(self, X, y):
        if(self.current_depth >= self.max_depth)|(self.X.shape[0] <= self.min_samples_split):
            return {'leaf': 'leaf' ,'predict': mode(self.y)[0]}          

        optimal_split = self.splitter() 
        self.current_depth += 1

        left_indexes = self.X.iloc[:, optimal_split['index']] < optimal_split['value']
        right_indexes = self.X.iloc[:, optimal_split['index']] >= optimal_split['value']
        
        left_X, left_y = self.X[left_indexes], self.y[left_indexes]
        right_X, right_y = self.X[right_indexes], self.y[right_indexes]        

        model = {
            'index': optimal_split['index'],
            'value': optimal_split['value'],
            'gini': optimal_split['gini'],
            'left_child': self.tree(left_X, left_y),
            'right_child': self.tree(right_X, right_y)
        }
        
        return model
        

    def splitter(self):
        optimal_split = {'gini': math.inf, 'index': 0, 'value': 0}
        
        for col_index in range(self.X.shape[1]):
            split_values = np.unique(self.X.iloc[:, col_index])
            
            for split_value in split_values:
                gini = self.gini(
                    self.y[self.X.iloc[:, col_index] < split_value], 
                    self.y[self.X.iloc[:, col_index] >= split_value]
                )

                if gini < optimal_split['gini']:
                    optimal_split['gini'], optimal_split['index'], optimal_split['value'] = gini, col_index, split_value
        
        return optimal_split


    def gini(self, y_1, y_2):
        f = lambda y: np.sum((np.bincount(y) / len(y))**2)
        gini = (len(y_1) * f(y_1) + len(y_2) * f(y_2)) / (len(y_1) + len(y_2))
        
        return gini

    def predict(self, X_test):
        predicts = []
        for x in X_test:
            predicts.append(self.finder(x, self.model))
                      
        return np.array(predicts)
   
        
    def finder(self, x, node):  
        if 'leaf' in node:
            return node['predict'] 
            
        elif float(x[node['index']]) <= node['value']:
            return self.finder(x, node['left_child'])
            
        else:
            return self.finder(x, node['right_child'])
        
            
        
        
        

        





   