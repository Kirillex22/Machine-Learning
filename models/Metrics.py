import numpy as np
from math import sqrt
   
def mean_abs_err(real: np.ndarray, predicted: np.ndarray) -> float:

   m = predicted.shape[0]
   sum = 0

   for i in range(0,m):
       y_pred = predicted[i]
       y_real = real[i]
       sum += abs(y_real - y_pred)

   result = sum/m

   return result


def mean_squared_err(real: np.ndarray, predicted: np.ndarray) -> float:
    
    m = predicted.shape[0]
    sum = 0
    
    for i in range(0,m):
        y_pred = predicted[i]
        y_real = real[i]
        sum += (y_real - y_pred)**2
    
    result = sum/m
    
    return result
    

def root_mse(real: np.ndarray, predicted: np.ndarray) -> float:
    
    result = sqrt(mean_squared_err(predicted, real))
        
    return result


def mean_abs_perc_err(real: np.ndarray, predicted: np.ndarray) -> float:
    
    m = predicted.shape[0]
    sum = 0
    
    for i in range(0,m):
        y_pred = predicted[i]
        y_real = real[i]
        if y_real == 0:
            continue
        sum += abs(1 - y_pred/y_real)
    
    result = sum/m
    
    return result
    

def r_squared(real: np.ndarray, predicted: np.ndarray) -> float:
    
    m = predicted.shape[0]
    sum = 0
    y_mean = real.mean()
    mse = mean_squared_err(real, predicted)
    
    for i in range(0,m):
        y_real = real[i]
        sum += (y_real - y_mean)**2
    
    sum /= m

    result = 1 - mse/sum
    
    return result
                 
        

    
            

        
    