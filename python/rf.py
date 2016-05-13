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


# Random Forest

elapsed = time.time() - t #toc
print("The total time elasped was:" + str(elapsed))

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

rf = RandomForestClassifier(n_estimators=1000, criterion='gini', \
    max_depth=50, min_samples_split=2, max_features='sqrt', \
    bootstrap=True, oob_score=False, n_jobs=-1, verbose=1)

for ii in range(0,len(yTrALL[0])):
    yTr = yTrALL[:, ii]
    fit = rf.fit(xTr, yTr)
    string = 'rfModel-1000-50-tag' + str(ii) + '.csv'
    joblib.dump(fit, string, compress=9)


# save / import model
#import pickle
#fit = rf.fit(xTr,yTr)
#pickle.dump( fit, open( "rfModel-full-log2", "wb" ) )

# rr = open('rfModel-full', 'rb')
# rfModel = pickle.loads(rr)
# pred = rfModel.predict(xTest)
