import time
import os
import numpy as np
import re

xTr = np.array(xTr, dtype=float)
yTrALL = np.array(yTrALL, dtype=float)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=2000, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)

for ii in range(0,len(yTrALL[0])):
    yTr = yTrALL[:,ii]
    score = cross_validation.cross_val_score(lr, xTr, yTr, cv = 5)
    cvAvg = sum(score)/float(5)
    print('Tag Number: ' + str(ii) + '  With the score of: ' +str(cvAvg))
