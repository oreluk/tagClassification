from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, criterion='gini', \
    max_depth=50, min_samples_split=2, max_features='sqrt', \
    bootstrap=True, oob_score=False, n_jobs=-1, verbose=1)


score = []
for ii in range(0,len(yTrALL[0])):
    yTr = yTrALL[:, ii]
    for traincv, testcv in cv:
        probs = rf.fit(xTr[traincv], yTr[traincv]).predict_proba(xTr[testcv])
        j = ['Tag ', ii, 'probs ', probs]
        score.append(j)

np.savetxt('rfProbs.csv', score, fmt="%s")
