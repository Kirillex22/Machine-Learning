import numpy as np

class MLP:  
     def __init__(self, *args, layers, loss = 'mse', epochs):
        self.epochs = epochs
        self.layers = layers     
        #if loss == 'classification':
            #self.metric = self.gini
        #elif loss == 'mse':
            #self.metric = self.mse

    def sig(self, x):
         return 1/(1 + np.exp(-x))

    def tanh(self, x):
        return (exp(x)-exp(-x))/(exp(x)+exp(-x))

    def relu(self, x):
        return max(0.0, x)

    def dsig(self, x):
        return self.sig(x)*(1 - self.sig(x))

    def dtanh(self, x):
        return 1 - tanh(x)**2

    def drelu(self, x):
        return 1 if relu(x) > 0 else 0

    def dw(self, x):
        
        

    def f_prop(self):
        for i in range(len(self.layers) - 1):
            output = 
        
        
            
               

    def fit(self, X, y):
        self.X = X
        self.y = y
        
        for i in range(self.epochs):
            
            

            
class Layer:
    def __init__(self, *args, num_neurons, activation):
        self.num_neurons = num_neurons
        self.activation = activation


        

        
    
    
            