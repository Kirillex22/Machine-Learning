import numpy as np

class MLP:  
     def __init__(self, *args, loss = 'mse', epochs):
        self.epochs = epochs
        
        if loss == 'gini':
            self.metric = self.gini
        elif loss == 'mse':
            self.metric = self.mse

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
        

    def f_prop(self, i, input):
        self.layers[i].activate()
        self.f_prop(i+1)
        
        
            
            
            

    def fill(self, layers):
        #self.input = layers[0]
        #self.hiddens = layers[1:len(layers) - 1]
        #self.output = layers[len(layers) - 1]   
        self.layers = layers
        


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.layers[0].
        
        for i in range(self.epochs):
            
            

            
class Layer:
    def __init__(self, *args, num_neurons, func, next_layer_dimencity):
        self.num_neurons = num_neurons
        self.func = func
        self.neurons = np.empty(num_neurons)
        for i in range(num_neurons):
            self.neurons[i] = Neuron(next_layer_dimencity)

    def activate(self):
        self.activated_w = self.func(self.weights)

class Neuron:
    def __init__(self, num_axons):
        self.num_axons = num_axons
        self.weights = np.empty(num_axons)
        

        
    
    
            