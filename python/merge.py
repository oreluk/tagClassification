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

xTest = loadX()
xT = np.array(xTest, dtype=float)
weight = 0.35
dtest = xgb.DMatrix(xTest)


for xx in range(0,5):
    # logistic
    string = 'logisticRegression-BEST-tag' + str(xx) + '.model'
    lrModel = joblib.load(string)
    lrProbs = lrModel.predict_proba(xT)

    # XGB
    bst = xgb.Booster({'nthread':8}) #init model
    loadFile = "models/xgboost_tag" + str(xx) + ".model"
    bst.load_model(loadFile) # load data
    ypred = bst.predict(dtest)
    res  = [(i, ypred[i]) for i in xrange(len(ypred))]

    #logistic

    predLabel = []

    for i in range(0,len(res)):
        p = lrProbs[i][1]*(weight) + res[i][1]*(1-weight)
        if p >= 0.5:
            predLabel.append(1)
        else:
            predLabel.append(0)
    predLabel = np.asarray(predLabel)
    string = 'kaggle/mergeLR-BLEND-' + str(xx) +'.csv'
    np.savetxt(string, predLabel, delimiter=",", fmt="%s")
