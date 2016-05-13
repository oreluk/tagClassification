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

numRounds = [2000]
numDepth = [10]
oldError = [100,100,100,100,100]
bestParms = []
error = []

for depth in numDepth:
    for nr in numRounds:
        print('Parameters Are Depth: ' + str(depth) + ' and numRounds: ' + str(nr))
        #for ii in range(0,len(yTrALL[0])):
        for ii in range(1,2):
            yTr = yTrALL[:, ii]
            score = []
            fScore = []
            for traincv, testcv in cv:
                num_round = nr
                param = {'bst:max_depth':depth,'bst:eta':0.1, 'bst:gamma':1.15, 'bst:subsample':0.9, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'error' }
                param['nthread'] = 8
                plst = param.items()
                dtrain = xgb.DMatrix( xTr[traincv], label=yTr[traincv])
                dtest = xgb.DMatrix(xTr[testcv], label=yTr[testcv])
                evallist  = [(dtest,'eval'), (dtrain,'train')]
                bst = xgb.train( plst, dtrain, num_round, evallist )
                # Prediction
                ypred = bst.predict(dtest)
                res  = [(i, ypred[i]) for i in xrange(len(ypred))]
                predLabel = []
                for i in range(0,len(res)):
                    if res[i][1] > thres:
                        predLabel.append(1)
                    else:
                        predLabel.append(0)
                predLabel = np.asarray(predLabel)
                cc = sum(predLabel == yTr[testcv])/float(len(yTr[testcv]))
                score.append(cc)
                fScore.append(f1_score(yTr[testcv], predLabel, average='macro'))
            ss = sum(score)/float(5)
            er = 1-ss
            error.insert(ii, er)
            if error[ii] < oldError[ii]:
                oldError[ii] = error[ii]
                q = [ii, 'rounds', nr, 'depth', depth, 'cvError', er, 'f1Score', fScore]
                bestParms.append(q)
            print('Tag' + str(ii)+ 'Completed, moving on to next tag. Score: ' + str(ss))
            print('------------------------\n------------------------')

np.savetxt('bestParms.csv', bestParms, delimiter=",", fmt="%s")
