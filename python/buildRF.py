# Final Project - Competition 1 - Team 4
# Stat154 Spring 2015
# Jim Oreluk

import time
import os
import numpy as np
import re

#####################################################
###               Import Data                     ###
#####################################################

t = time.time() #tic

def loadX():
    pf = open('powerFeatures2.csv', 'r').read()
    pf = pf.split('\n')
    powerFeatures = []
    for item in pf:
        powerFeatures.append(item.split(','))

    powerFeatures = powerFeatures[1:-1]
    powerFeatures = np.asarray(powerFeatures)
    return(powerFeatures)

def loadY():
    yt = open('yTrain.csv','r').read()
    yt = yt.split('\n')
    yTr = yt[0:-1]
    yTr = np.asarray(yTr)
    return(yTr)

#####################################################
###               Random Forest                   ###
#####################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

xTr = loadX()
yTr = loadY()

elapsed = time.time() - t #toc
print("The total time elasped was:" + str(elapsed))

cv = cross_validation.KFold(len(xTr), n_folds=5, shuffle=True)
rf = RandomForestClassifier(n_estimators=1000, criterion='gini', \
    max_depth=None, min_samples_split=2, max_features='sqrt', \
    bootstrap=True, oob_score=False, n_jobs=-1, verbose=1)

score = []
for traincv, testcv in cv:
    probs = rf.fit(xTr[traincv], yTr[traincv]).predict_proba(xTr[testcv])
    predLabel = []
    for i in range(0,len(probs)):
        if probs[i][0] < probs[i][1]:
            predLabel.append('1')
        else:
            predLabel.append('0')
    predLabel = np.asarray(predLabel)
    cc = sum(predLabel == yTr[testcv]) / float(len(probs))
    score.append(cc)
    print('Completed Fold.')

np.savetxt('cvScore.csv', score, fmt="%s")

# Export Model

from sklearn.externals import joblib
fit = rf.fit(xTr,yTr)
joblib.dump(fit, 'rfModel.pkl', compress=9)
