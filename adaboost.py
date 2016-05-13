from sklearn.ensemble import AdaBoostClassifier


clf = AdaBoostClassifier(n_estimators=500, learning_rate=0.1, algorithm='SAMME.R')

score = []
yTr = yTrALL[:, 1]
for traincv, testcv in cv:
    probs = clf.fit(xTr[traincv], yTr[traincv]).predict_proba(xTr[testcv])
    j = ['Tag ', ii, 'probs ', probs]
    score.append(j)
    predLabel = []
    for i in range(0,len(probs)):
        if probs[i][1] > 0.5:
            predLabel.append(1)
        else:
            predLabel.append(0)
    predLabel = np.asarray(predLabel)
    cc = sum(predLabel == yTr[testcv])/float(len(yTr[testcv]))
    print(cc)
