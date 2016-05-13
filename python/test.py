import time
import os
import numpy as np
import re
import xgboost as xgb

# Import Data

t = time.time() #tic

def loadX():
    pf = open('powerFeatures.csv', 'r').read()
    pf = pf.split('\n')
    powerFeatures = []
    for item in pf:
        powerFeatures.append(item.split(','))

    powerFeatures = powerFeatures[1:-1]
    powerFeatures = np.asarray(powerFeatures)
    return(powerFeatures)

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

from sklearn.ensemble import RandomForestClassifier
cv = cross_validation.KFold(len(xTr), n_folds=5, shuffle=True)

rf = RandomForestClassifier(n_estimators=100, criterion='gini', \
    max_depth=50, min_samples_split=2, max_features='sqrt', \
    bootstrap=True, oob_score=False, n_jobs=-1, verbose=1)
for ii in range(0,len(yTrALL[0])):
    score = []
    yTr = yTrALL[:, ii]
    for traincv, testcv in cv:
        probs = rf.fit(xTr[traincv], yTr[traincv]).predict_proba(xTr[testcv])
        predLabel = []
        for i in range(0,len(probs)):
            if probs[i][0] < probs[i][1]:
                predLabel.append(1)
            else:
                predLabel.append(0)
        predLabel = np.asarray(predLabel)
        cc = sum(predLabel == yTr[testcv]) / float(len(probs))
        score.append(cc)
        print('Completed Fold.')
    print(score)
