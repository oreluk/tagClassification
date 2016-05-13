import time
import os
import numpy as np
import re
import xgboost as xgb

def loadX():
    xc = open('KAGGLE-xCombined.csv','r').read()
    xc = xc.split('\n')
    xc = xc[0:-1]
    xCombined = []
    for item in xc:
        ii = item.split(' ')
        xCombined.append(ii)

    xCombined = np.asarray(xCombined)

    pf = open('KAGGLE-powerFeatures.csv', 'r').read()
    pf = pf.split('\n')
    powerFeatures = []
    for item in pf:
        powerFeatures.append(item.split(','))

    powerFeatures = powerFeatures[1:-1]
    xT = np.concatenate([xCombined, powerFeatures], axis=1)
    return(xT)

pp=[]
xT = loadX()
dtest = xgb.DMatrix(xT)
for xx in range(0,5):
    bst = xgb.Booster({'nthread':8}) #init model
    loadFile = "models/xgboost_tag" + str(xx) + ".model"
    bst.load_model(loadFile) # load data
    ypred = bst.predict(dtest)
    res  = [(i, ypred[i]) for i in xrange(len(ypred))]
    pp.append(res)

np.savetxt('xgbProb', pp, fmt="%s")
