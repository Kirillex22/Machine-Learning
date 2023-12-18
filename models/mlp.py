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
        

    def f_prop(self):
        for i in range(len(self.layers) - 1):
            output = 
        
        
            
               

    def fit(self, X, y):
        self.X = X
        self.y = y
        
        for i in range(self.epochs):
            
            

            
class Layer:
    def __init__(self, *args, num_neurons, func, next_layer_dim):
        self.num_neurons = num_neurons
        self.func = func
        self.neurons = np.empty(num_neurons)
        for i in range(num_neurons):
            self.neurons[i] = Neuron(next_layer_dim, 1)

    def activate(self):
        self.values = self.func(np.array([neuron.v for neuron.v in self.neurons]))

    def podumat(self):
        self.activate()
        return np.array[self.values[i]*self.neurons[i].axons_weights for i in range(self.values)]    

class Neuron:
    def __init__(self, num_axons, v):
        self.v = v
        self.num_axons = num_axons
        self.axons_weights = np.empty(num_axons)
        

        
    
    
            