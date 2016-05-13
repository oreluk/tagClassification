import time
import os
import numpy as np
import re
from sklearn.externals import joblib

xTr = np.array(xTr, dtype=float)
yTrALL = np.array(yTrALL, dtype=float)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=2000, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)

for ii in range(0,len(yTrALL[0])):
    yTr = yTrALL[:,ii]
    fit = lr.fit(xTr, yTr)
    string = 'logisticRegression-BEST-tag' + str(ii) + '.model'
    joblib.dump(fit, string, compress=9)
    print('TAG COMPLETE')



print('FINISHED WITH BUILDING MODELS')
