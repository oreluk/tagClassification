import time
import os
import numpy as np
import re
import xgboost as xgb

# Import Data

t = time.time() #tic

def loadX():
    xt = open('xCombined.csv','r').read()
    xt = xt.split('\n')
    xt = xt[0:-1]
    xT = []
    for item in xt:
        ii = item.split(',')
        jj = [(float(val)) for val in ii]
        xT.append(jj)

    xT = np.asarray(xT)

    pf = open('powerFeatures.csv', 'r').read()
    pf = pf.split('\n')
    powerFeatures = []
    for item in pf:
        powerFeatures.append(item.split(','))

    powerFeatures = powerFeatures[1:-1]
    xTr = np.concatenate([xT, powerFeatures], axis=1)
    return(xTr)

def loadY():
    yt = open('yL.csv','r').read()
    yt = yt.split('\n')
    yt = yt[1:-1]
    yTr = []
    for item in yt:
        ii = item.split(',')
        jj = [int(float(val)) for val in ii]
        yTr.append(jj)
    yTr = np.asarray(yTr)
    return(yTr)

# xgb

xTr = loadX()
yTrALL = loadY()

elapsed = time.time() - t #toc
print("The total time elasped was:" + str(elapsed))

from sklearn import cross_validation
from sklearn.metrics import f1_score
cv = cross_validation.KFold(len(xTr), n_folds=5, shuffle=True)
