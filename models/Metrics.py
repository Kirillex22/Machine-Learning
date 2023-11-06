import numpy as np
   
def mean_abs_err(real: np.ndarray, predicted: np.ndarray) -> float:

    result = np.mean(abs(real - predicted))

    return result


def mean_squared_err(real: np.ndarray, predicted: np.ndarray) -> float:

    result = np.mean((real - predicted)**2)
    
    return result
    

def root_mse(real: np.ndarray, predicted: np.ndarray) -> float:
    
    result = np.sqrt(mean_squared_err(predicted, real))
        
    return result


def mean_abs_perc_err(real: np.ndarray, predicted: np.ndarray) -> float:

    result = np.mean(abs((real - predicted)/real))
    
    return result
    

def r_squared(real: np.ndarray, predicted: np.ndarray) -> float:

    y_mean = real.mean()    
    mse = mean_squared_err(real, predicted)
    sum = np.mean((real - y_mean)**2)

    result = 1 - mse/sum
    
    return result
                 
        

    
            

        
    