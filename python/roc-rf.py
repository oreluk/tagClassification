import time
import os
import numpy as np
import re


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
    xBOTH = np.concatenate([xCombined, powerFeatures], axis=1)
    return(xCombined, powerFeatures, xBOTH)

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


# Random Forest

xCombined, powerFeatures, xBOTH = loadX()
yTrALL = loadY()



elapsed = time.time() - t #toc
print("The total time elasped was:" + str(elapsed))

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc

cv = cross_validation.KFold(len(xCombined), n_folds=10, shuffle=True)
rf = RandomForestClassifier(n_estimators=100, criterion='gini', \
    max_depth=50, min_samples_split=2, max_features='sqrt', \
    bootstrap=True, oob_score=False, n_jobs=-1, verbose=1)

def plotROC(inputPROBS, inputYTRUE):
    fpr, tpr, thresholds = roc_curve(inputYTRUE, inputPROBS, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return(roc_auc)

x = np.asarray(xCombined)
for ii in range(0,len(yTrALL[0])):
    score = []
    yTr = yTrALL[:, ii]
    for traincv, testcv in cv:
        probs = rf.fit(x[traincv], yTr[traincv]).predict_proba(x[testcv])
        string = 'rf_probs_WORD_' + str(ii)
        np.savetxt(string + '.csv', probs, delimiter=",", fmt="%s")
    roc_auc, fpr, tpr = plotROC(probs[:,1], yTr[testcv])
    print "Area under the ROC curve : %f" % roc_auc
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    if ii == 0:
        pl.title('Word Matrix ROC Curve - R Tag')
    elif ii == 1:
        pl.title('Word Matrix ROC Curve - Stat Tag')
    elif ii == 2:
        pl.title('Word Matrix ROC Curve - ML Tag')
    elif ii == 3:
        pl.title('Word Matrix ROC Curve - Math Tag')
    elif ii == 4:
        pl.title('Word Matrix ROC Curve - Numpy Tag')
    pl.legend(loc="lower right")
    pl.savefig(string + '.png', bbox_inches='tight')
    pl.show()


x = np.asarray(powerFeatures)
for ii in range(0,len(yTrALL[0])):
    score = []
    yTr = yTrALL[:, ii]
    for traincv, testcv in cv:
        probs = rf.fit(x[traincv], yTr[traincv]).predict_proba(x[testcv])
        string = 'rf_probs_PF_' + str(ii)
        np.savetxt(string + '.csv', probs, delimiter=",", fmt="%s")
    roc_auc, fpr, tpr = plotROC(probs[:,1], yTr[testcv])
    print "Area under the ROC curve : %f" % roc_auc
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    if ii == 0:
        pl.title('Power Feature ROC Curve - R Tag')
    elif ii == 1:
        pl.title('Power Feature ROC Curve - Stat Tag')
    elif ii == 2:
        pl.title('Power Feature ROC Curve - ML Tag')
    elif ii == 3:
        pl.title('Power Feature ROC Curve - Math Tag')
    elif ii == 4:
        pl.title('Power Feature ROC Curve - Numpy Tag')
    pl.legend(loc="lower right")
    pl.savefig(string + '.png', bbox_inches='tight')
    pl.show()

x = np.asarray(xBOTH)
for ii in range(0,len(yTrALL[0])):
    score = []
    yTr = yTrALL[:, ii]
    for traincv, testcv in cv:
        probs = rf.fit(x[traincv], yTr[traincv]).predict_proba(x[testcv])
        string = 'rf_probs_COMBINED_' + str(ii)
        np.savetxt(string + '.csv', probs, delimiter=",", fmt="%s")
    roc_auc, fpr, tpr = plotROC(probs[:,1], yTr[testcv])
    print "Area under the ROC curve : %f" % roc_auc
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    if ii == 0:
        pl.title('Combined Matrix ROC Curve - R Tag')
    elif ii == 1:
        pl.title('Combined Matrix ROC Curve - Stat Tag')
    elif ii == 2:
        pl.title('Combined Matrix ROC Curve - ML Tag')
    elif ii == 3:
        pl.title('Combined Matrix ROC Curve - Math Tag')
    elif ii == 4:
        pl.title('Combined Matrix ROC Curve - Numpy Tag')
    pl.legend(loc="lower right")
    pl.savefig(string + '.png', bbox_inches='tight')
    pl.show()
