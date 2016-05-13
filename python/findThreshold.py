cOld = 0
thres = np.linspace(0.0001, 0.7, 15000)
thresholdResults = []
for jj in thres:
    predLabel = []
    for i in range(0,len(res)):
        if res[i][1] > jj:
            predLabel.append(1)
        else:
            predLabel.append(0)
    predLabel = np.asarray(predLabel)
    cc = sum(predLabel == yT)/float(len(yT))
    if cc > cOld:
        print('Threshold: ' + str(jj) + 'Error: ' + str(cc))
        cOld = cc
        c = ['Threshold: ', jj, 'Error: ', cc]
        thresholdResults.append(c)

thresholdResults = np.asarray(thresholdResults)
np.savetxt('thresholdResults-.csv', thresholdResults, fmt="%s")
