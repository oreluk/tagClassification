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
