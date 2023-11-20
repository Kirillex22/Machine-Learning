import numpy as np

def myBinaryConfusionMatrix(real, predicted):

    con_mtx = np.matrix('0 0; 0 0')
    size = len(predicted) 

    for i in range(size):
        case = [real[i], predicted[i]]
        
        if (case == [0,0]):
            con_mtx[0,0] +=1
        elif (case == [0,1]):
            con_mtx[0,1] +=1
        elif (case == [1,0]):
            con_mtx[1,0] +=1
        else:
            con_mtx[1,1] +=1
            
    return con_mtx   


def myClassificationReport(real, predicted):
    con_mtx = myBinaryConfusionMatrix(real, predicted)
    precs = myPrecision(real, predicted, con_mtx)
    recs = myRecall(real, predicted, con_mtx)
    f1s = myF1Score(real, predicted, con_mtx)
    print(f'accuracy: {round(acc, 2)}')
    print(f'           {0}      {1}')
    print(f'precision: {round(precs[0], 2)} {round(precs[1], 2)}')
    print(f'recall: {round(recs[0], 2)} {round(recs[1], 2)}')
    print(f'f1_score: {round(f1s[0], 2)} {round(f1s[1], 2)}')


def myAccuracy(real, predicted, con_mtx):   
     size = len(predicted)
     acc = (con_mtx[0,0] + con_mtx[1,1])/size
    
     return acc


def myPrecision(real, predicted, con_mtx):  
    result = [con_mtx[0,0]/(con_mtx[0,0]+con_mtx[1,0]), con_mtx[1,1]/(con_mtx[1,1]+con_mtx[0,1])]
    
    return result


def myRecall(real, predicted, con_mtx):  
    result = [con_mtx[0,0]/(con_mtx[0,0]+con_mtx[0,1]), con_mtx[1,1]/(con_mtx[1,1]+con_mtx[1,0])]
    
    return result


def myF1Score(real, predicted, con_mtx):
    precs = myPrecision(real, predicted, con_mtx)
    recs = myRecall(real, predicted, con_mtx)

    result = [(2*recs[0]*precs[0])/(recs[0]+precs[0]), (2*recs[1]*precs[1])/(recs[1]+precs[1])]
    
    return result
    