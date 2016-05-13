
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, criterion='gini', \
    max_depth=50, min_samples_split=2, max_features='sqrt', \
    bootstrap=True, oob_score=False, n_jobs=-1, verbose=1)

numRounds = [2000]
numDepth = [10]
oldError = [100,100,100,100,100]
bestParms = []
error = []

weight = 0.8

for depth in numDepth:
    for nr in numRounds:
        print('Parameters Are Depth: ' + str(depth) + ' and numRounds: ' + str(nr))
        #for ii in range(0,len(yTrALL[0])):
        for ii in range(0,len(yTrALL[0])):
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
                probs = rf.fit(xTr[traincv], yTr[traincv]).predict_proba(xTr[testcv])

                # Weight Average Probabilites of xgb & RF

                predLabel = []
                for i in range(0,len(probs)):
                    pp = probs[i][1]*(1-weight) + res[i][1]*(weight)
                    if pp > 0.5:
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

print(bestParms)
