import os
import numpy as np

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

xT = loadX()

from sklearn.externals import joblib
print('loading model...')
rfModel = joblib.load('models/rfModel-1000-50-tag0.pkl')

predict = rfModel.predict(xT)
results = []
for n, res in enumerate(predict):
    c = [n, res]
    results.append(c)

print('saving results.')
np.savetxt('kaggle/rf_RTAG.csv', results, delimiter=',', fmt="%s")
