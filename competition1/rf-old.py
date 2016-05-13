# Final Project - Competition 1 - Team 4
# Stat154 Spring 2015
# Wonhee Lee, Nathan Yong Jun Lee, Wenyu Li, Jim Oreluk

import time
import os
import numpy as np
import re

#####################################################
###               Import Data                     ###
#####################################################

os.chdir('C:/Users/Jim/Dropbox/classes/stat154/finalProject')

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


    return(xCombined)

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

cv = cross_validation.KFold(len(xTr), n_folds=10, shuffle=True)
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

joblib.dump(fit, 'my_model.pkl', compress=9)
