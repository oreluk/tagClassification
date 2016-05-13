import time
import os
import numpy as np
import re
import xgboost as xgb

# Import Data

t = time.time() #tic

def loadX():
    xc = open('xCombined.csv','r').read()
    xc = xc.split('\n')
    xc = xc[0:-1]
    xCombined = []
    for item in xc:
        ii = item.split(',')
        jj = [(float(val)) for val in ii]
        xCombined.append(jj)

    xCombined = np.asarray(xCombined)

    pf = open('powerFeatures.csv', 'r').read()
    pf = pf.split('\n')
    powerFeatures = []
    for item in pf:
        powerFeatures.append(item.split(','))

    powerFeatures = powerFeatures[1:-1]

    xTr = np.concatenate([xCombined, powerFeatures], axis=1)

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

xTr = loadX()
yTrALL = loadY()

elapsed = time.time() - t #toc
print("The total time elasped was:" + str(elapsed))

num_round = 2000
param = {'bst:max_depth':10, 'bst:gamma':1.15, 'bst:eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'error' }
param['nthread'] = 8
plst = param.items()

for ii in range(0,len(yTrALL[0])):
    yTr = yTrALL[:, ii]
    dtrain = xgb.DMatrix(xTr, label=yTr)
    dtest = xgb.DMatrix(xTr, label=yTr)
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    bst = xgb.train( plst, dtrain, num_round, evallist )
    modelName = 'xgboost_tag' + str(ii) + '.model'
    bst.save_model(modelName)
