import numpy as np

class MLP:
    def __init__(self, *args, epochs, layers, loss = 'mse', learning_rate = 1e-3, logging = True):

        self.epochs = epochs
        self.layers = layers
        self.learning_rate = learning_rate
        self.logging = logging
        self.weights = []
        self.biases = []

        if loss == 'mse':
            self.loss = self.mse
            self.dloss = self.dmse

        elif loss == 'cross_entropy':
            self.loss = self.cross_entropy
            self.dloss = self.dcross_entropy   

        for i in range(len(self.layers) - 1):

            self.weights.append(
                np.random.randn(
                    self.layers[i].dim, 
                    self.layers[i+1].dim
                    )
                )

            self.biases.append(
                np.random.randn(1, self.layers[i+1].dim)
                )


    def propagate(self, layer_index, wb_index, x):
        if self.logging:
            print(f'x {x}')
            print(f'w {self.weights[wb_index]}')
            print(f'b {self.biases[wb_index]} \n')
        self.layers[layer_index].raw_values = np.dot(x, self.weights[wb_index]) + self.biases[wb_index]
        return self.layers[layer_index].raw_values


    def f_prop(self, x):
        res = x #входной вектор
        for i in range(1, len(self.layers)): #начинаем со второго слоя: значения во текущем = преобразования от значения в предыдущем 
            res = self.propagate(i, i-1, res) #значения преобразования wx+b сохранены в экземпляре Layer под индексом i
            res = self.layers[i].activate(res) #активированное значение также сохранено в экземпляре Layer
            
        return res
        

    def b_prop(self, predict, y):
        grad_w = []
        grad_b = []
        k = len(self.layers) - 1

        for i in reversed(range(k+1)):  
            if i == k:            
                dE_dt = self.dloss(predict, y)
                dE_dw = np.dot(self.layers[i-1].activated_values.T, dE_dt)

            elif i == 0:
                continue

            else:
                dE_dt = dE_dh*self.layers[i].diff(self.layers[i].raw_values)
                dE_dw = np.dot(self.layers[i-1].activated_values.T, dE_dt)

            grad_w.append(dE_dw)
            grad_b.append(dE_dt)

            dE_dh = np.dot(dE_dt, self.weights[i-1].T) 

        return list(reversed(grad_w)), list(reversed(grad_b))


    def fit(self, X, y):

        for epoch in range(self.epochs):
            if self.logging:
                print(f'epoch: {epoch+1}')
            for j in range(len(X)):
                self.layers[0].activated_values = np.array([X[j]])
                predict = self.f_prop(
                    self.layers[0].activated_values
                    )

                if self.logging:
                    print(f'error: {self.cross_entropy(predict, y[j])}')
                
                grad_w, grad_b = self.b_prop(predict, y[j])

                k = len(self.layers)
                for i in range(k-1):
                    self.weights[i] -= grad_w[i]*self.learning_rate
                    self.biases[i] -= grad_b[i]*self.learning_rate
                

    def predict(self, X):
        out = np.zeros(len(X))
        for j in range(len(X)):
            out[j] = self.f_prop(X[j])
        return out

    
    def cross_entropy(self, pred, real): #предполагается softmax(pred) на вход
        return (-1)*np.sum(real*np.log(pred))


    def mse(self, pred, real):
        return np.mean((real-pred)**2)


    def dcross_entropy(self, pred, real): #предполагается softmax(pred) на вход
        return pred - real


    def dmse(self, pred, real):
        return (-0.5)*np.mean(real - pred)


class Layer:
    def __init__(self, *args, dim, act_func = None):
        
        self.dim = dim
        self.raw_values = None

        if act_func == 'relu':
            self.act_func = self.relu
            self.diff = self.drelu
        
        elif act_func == 'sig':
            self.act_func = self.sig
            self.diff = self.dsig
        
        elif act_func == 'tanh':
            self.act_func = self.tanh
            self.diff = self.dtanh

        elif act_func == 'softmax':
            self.act_func = self.softmax 
            self.diff = self.dsoftmax

        elif act_func == 'no_function':
            self.act_func = lambda x: x
            self.diff = lambda x: x


    def activate(self, x):
        self.activated_values = self.act_func(x)
        return self.activated_values

    
    def sig(self, x):
         return 1/(1 + np.exp(-x))


    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


    def relu(self, x):
        return np.maximum(0.0, x)


    def dsig(self, x):
        return self.sig(x)*(1 - self.sig(x))


    def dtanh(self, x):
        return 1 - self.tanh(x)**2


    def drelu(self, x):
        return np.where(x > 0, 1, 0)


    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))


    def dsoftmax(self, x):
        return self.softmax(x)*(1 - self.softmax(x))