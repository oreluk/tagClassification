# Final Project - Competition 1 (in class)
# Stat154 Spring 2015
# Wonhee Lee, Nathan Yong Jun Lee, Wenyu Li, Jim Oreluk
# Created: 15/04/25
# Purpose: Takes csv file, scrubs clean code, counts words which were
#           considered 'good' by training data.
#
#           Saves xTest.csv/powerFeatures.csv (same #columns of original data)


import time
import os
import numpy as np
import re
import csv


t = time.time() #tic

xData = []
with open('competition1/Xtest1.csv', 'rb') as csvfile:
    xTestReader = csv.reader(csvfile, delimiter=',')
    for row in xTestReader:
        xData.append(row)

xData = xData[1:]  # removes the [id, title, body] header.

#####################################################e
###     Scrub Code Clean -- Create Dictionary     ###
#####################################################

dd = open('common-english-words.txt','r')
commonWords = dd.read()
commonWords = commonWords.split(',')

dd = open('python_packages.txt', 'r')
pythonPackages = dd.read()
pythonPackages = pythonPackages.split('\n')
pythonPackages = pythonPackages[0:-1]
pythonPackages = [item.lower() for item in pythonPackages]

dd = open('r_packages.txt', 'r')
rPackages = dd.read()
rPackages = rPackages.split('\n')
rPackages = rPackages[0:-1]
rPackages = [item.lower() for item in rPackages]

dd = open('statwords.txt', 'r')
statPack = dd.read()
statPack = statPack.split('\n')
statPack = statPack[0:-1]
statPack = [item.lower() for item in statPack]

wordDictionary = {}
codeCount = np.empty([len(xData), 3])   # 27425-by-1  [Matches found only in Body] (as expected)
pCount = np.empty([len(xData), 3])      # 27425-by-1  [Matches found only in Body]
titleWordCount = np.empty([len(xData), 3])
pyLoopCount = np.empty([len(xData), 3])
rLoopCount = np.empty([len(xData), 3])
lowIndexCount = np.empty([len(xData), 3])
libraryCount = np.empty([len(xData), 3])
dataFrame = np.empty([len(xData), 3])
syntaxCount = np.empty([len(xData), 3])
importCount = np.empty([len(xData), 3])
packagePythonCount = np.empty([len(xData), 3])
packageRCount = np.empty([len(xData), 3])
curlyCount = np.empty([len(xData), 3])
errorPythonCount = np.empty([len(xData), 3])
errorRCount = np.empty([len(xData), 3])
statCount = np.empty([len(xData), 3])
latexCount = np.empty([len(xData), 3])

for i in range(0,len(xData)):
    for j in range(1,3):
        xData[i][j] = xData[i][j].lower()
        # Power Features
        if j == 1:
            titleWordCount[i,j] =len(xData[i][j].split(" "))
        if j == 2:
            codeCount[i,j] = xData[i][j].count('<code>')
            pCount[i,j] = xData[i][j].count('<p>')

        pyLoopCount[i,j] = len(re.findall('(?s)for.+?:', xData[i][j])) \
        + len(re.findall('(?s)if.+?:', xData[i][j])) \
        + len(re.findall('(?s)for.+?:', xData[i][j]))

        lowIndexCount[i,j] = len(re.findall('\[0\]', xData[i][j]))
        rLoopCount[i,j] = len(re.findall('(?s)for.+?\{', xData[i][j])) \
            + len(re.findall('(?s)if.+?\{', xData[i][j])) \
            + len(re.findall('(?s)while.+?\{', xData[i][j]))

        libraryCount[i,j] = len(re.findall('library\(', xData[i][j])) \
            + len(re.findall('require\(', xData[i][j]))

        dataFrame[i,j] = len(re.findall('data\.frame', xData[i][j]))
        syntaxCount[i,j] = len(re.findall('(?s)\.*?\(\)', xData[i][j]))
        importCount[i,j] = len(re.findall('import', xData[i][j]))

        pc = 0
        for pp, pat in enumerate(pythonPackages):
            currentCount = len(re.findall(pat, xData[i][j]))
            pc = pc + currentCount
        packagePythonCount[i,j] = pc
        rc = 0
        for rr, pat in enumerate(rPackages):
            currentCount = len(re.findall(pat, xData[i][j]))
            rc = rc + currentCount
        packageRCount[i,j] = rc

        sc = 0
        for gg, pat in enumerate(statPack):
            currentCount = len(re.findall(pat, xData[i][j]))
            sc = sc + currentCount
        statCount[i,j] = sc

        latexCount[i,j] = len(re.findall('\$', xData[i][j]))
        curlyCount[i,j] = len(re.findall('\{', xData[i][j]))
        errorPythonCount[i,j] = len(re.findall('Traceback', xData[i][j])) + len(re.findall('[^\s]Error', xData[i][j])) + len(re.findall('not defined', xData[i][j]))
        errorRCount[i,j] = len(re.findall('Error in', xData[i][j])) + len(re.findall('not found', xData[i][j]))

        # Delete Tagged Content, Tags, and non-alphabet characters
        xData[i][j] = re.sub('(?s)<pre>.+?</pre>', ' ', xData[i][j])
        xData[i][j] = re.sub('(?s)<code>.+?</code>', ' ', xData[i][j])
        xData[i][j] = re.sub('(?s)$.+?$', ' ', xData[i][j])
        xData[i][j] = re.sub('<p>', ' ', xData[i][j])
        xData[i][j] = re.sub('<hr>', ' ', xData[i][j])
        xData[i][j] = re.sub('</hr>', ' ', xData[i][j])
        xData[i][j] = re.sub('</p>', ' ', xData[i][j])
        xData[i][j] = re.sub('<ul>|<li>|</li>|</ul>|<ol>|</ol>', ' ', xData[i][j])
        xData[i][j] = re.sub(',|\n|~|`|[0-9]|!|@|#|$|%|/|\+|&|\*|\_|-|=|\?|\(|\)|\{|\}|\[|\]|\<|\>|\:|\;', ' ', xData[i][j])
        xData[i][j] = xData[i][j].replace('"', '')
        xData[i][j] = xData[i][j].replace('\'', ' ')
        xData[i][j] = xData[i][j].replace('.', ' ')
        xData[i][j] = xData[i][j].replace('^', ' ')
        xData[i][j] = xData[i][j].replace('\$', ' ')

        # Delete Common Words
        for kk in range(0,len(commonWords)):
            xData[i][j] = re.sub('\\b'+commonWords[kk]+'\\b', ' ', xData[i][j].lower())

        # Create Dictionary
        words = re.findall(r"[\w']+|[.,!?;]", xData[i][j])
        for word in words:
            if word in wordDictionary:
                wordDictionary[word] = wordDictionary[word] + 1
            else:
                wordDictionary[word] = 1

    # Print Progress
    if (i % 100) == 0:
        per = round(float(i)/len(xData) * 100, 3)
        string = "Scrub Progress: " + str(per) + "%."
        print(string)

# Process Power Features for Export
codeCount = np.sum(codeCount, axis=1)
codeCount = np.reshape(codeCount, [len(codeCount), 1])

pCount = np.sum(pCount, axis=1)
pCount = np.reshape(pCount, [len(pCount), 1])

titleWordCount = np.sum(titleWordCount, axis=1)
titleWordCount = np.reshape(titleWordCount, [len(titleWordCount), 1])

pyLoopCount = np.sum(pyLoopCount, axis=1)
pyLoopCount = np.reshape(pyLoopCount, [len(pyLoopCount), 1])

rLoopCount = np.sum(rLoopCount, axis=1)
rLoopCount = np.reshape(rLoopCount, [len(rLoopCount), 1])

lowIndexCount = np.sum(lowIndexCount, axis=1)
lowIndexCount = np.reshape(lowIndexCount, [len(lowIndexCount), 1])

libraryCount = np.sum(libraryCount, axis=1)
libraryCount = np.reshape(libraryCount, [len(libraryCount), 1])

dataFrame = np.sum(dataFrame, axis=1)
dataFrame = np.reshape(dataFrame, [len(dataFrame), 1])

syntaxCount = np.sum(syntaxCount, axis=1)
syntaxCount = np.reshape(syntaxCount, [len(syntaxCount), 1])

importCount = np.sum(importCount, axis=1)
importCount = np.reshape(importCount, [len(importCount), 1])

packagePythonCount = np.sum(packagePythonCount, axis=1)
packagePythonCount = np.reshape(packagePythonCount, [len(packagePythonCount), 1])

packageRCount = np.sum(packageRCount, axis=1)
packageRCount = np.reshape(packageRCount, [len(packageRCount), 1])

curlyCount = np.sum(curlyCount, axis=1)
curlyCount = np.reshape(curlyCount, [len(curlyCount), 1])

errorPythonCount = np.sum(errorPythonCount, axis=1)
errorPythonCount = np.reshape(errorPythonCount, [len(errorPythonCount), 1])

errorRCount = np.sum(errorRCount, axis=1)
errorRCount = np.reshape(errorRCount, [len(errorRCount), 1])

statCount = np.sum(statCount, axis=1)
statCount = np.reshape(statCount, [len(statCount), 1])

latexCount = np.sum(latexCount, axis=1)
latexCount = np.reshape(latexCount, [len(latexCount), 1])

powerLabels = ["PF: <code> Count", "PF: <p> Count", "PF: Words in Title", \
    "PF: Python Unique Loops", "PF: Index of [0] Count", "PF: R Unique Loops", \
    "PF: library() or require()", "PF: data.frame", "PF: syntax", "PF: import", \
    "PF: package python", "PF: package r", "PF: Curly Bracket Count", \
    "PF: Error Message Python", "PF: Error Message R", "PF: Statistic Terms", \
    "PF: Latex Symbols"]

powerCombined = np.concatenate([codeCount, pCount, titleWordCount, pyLoopCount, \
lowIndexCount, rLoopCount, libraryCount, dataFrame, syntaxCount, importCount, \
packagePythonCount, packageRCount, curlyCount, errorPythonCount, errorRCount, \
statCount, latexCount], axis=1)

powerCombined = np.vstack((powerLabels, powerCombined))

np.savetxt('competition1/xExtra-pf.csv', powerCombined, delimiter=",", fmt="%s")
#####################################################
###        Create Data Matrix of Counts           ###
#####################################################

colNames = open('colNames.csv', 'r').read()
colNames = colNames.split('\n')
colNames = colNames[0:-1]

# Feature Matrix
xCombined = np.empty([len(xData), len(colNames)])

for i, row in enumerate(xData):
    for j, p in enumerate(colNames):
        xCombined[i, j] = row[1].split(" ").count(p) + row[2].split(" ").count(p)
    if (i % 50) == 0:
        b = round(float(i)/len(xData) * 100, 2)
        string = "Data Transfered to xTrain: " + str(b) + "%."
        print(string)


for i, row in enumerate(xCombined):
    tot = sum(row)
    xCombined[i, :] = xCombined[i, :]/float(tot)

elapsed = time.time() - t #toc
print("The total time elasped was:" + str(elapsed))

np.savetxt('competition1/xExtra1.csv', xCombined, fmt="%s")


####N#################################################
###                 Prediction                    ###
#####################################################

print(xCombined.shape)

from sklearn.externals import joblib
rfModel = joblib.load('rfModel.pkl')

predict = rfModel.predict(xCombined)

# Need to do: export result to CSV file provided...
resultHeader = ["Id", "Tags"]

results = []
for n, res in enumerate(predict):
    c = [n, res]
    results.append(c)

results = np.vstack((resultHeader, results))

np.savetxt('competition1/competition1-Team4.csv', results, delimiter=',', fmt="%s")
