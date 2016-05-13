import time
import os
import numpy as np
import re

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

from sklearn.externals import joblib

pp = []
weight = 0.3
for xx in range(0,5):
    string = 'rfModel-100-50-tag' + str(xx) + '.csv'
    rfModel = joblib.load(string)
    rfProbs = rfModel.predict_proba(xTest)
    pp.append(rfProbs)

    predLabel = []
    for i in range(0,len(res)):
        if res[i][1] > 0.35:
            predLabel.append(1)
        else:
            predLabel.append(0)
    predLabel = np.asarray(predLabel)
    string = 'kaggle/xgb-shortThreshold' + str(xx) +'.csv'
    np.savetxt(string, predLabel, delimiter=",", fmt="%s")
np.savetxt('rfProbs', pp, fmt="%s")
