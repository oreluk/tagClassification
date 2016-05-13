# Final Project - Clean Kaggle Code
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
with open('XtestKaggle.csv', 'rb') as csvfile:
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
codeCount = np.empty([len(xData), 1])   # 27425-by-1  [Matches found only in Body] (as expected)
pCount = np.empty([len(xData), 1])      # 27425-by-1  [Matches found only in Body]
titleWordCount = np.empty([len(xData), 1])
pyLoopCount = np.empty([len(xData), 1])
rLoopCount = np.empty([len(xData), 1])
lowIndexCount = np.empty([len(xData), 1])
libraryCount = np.empty([len(xData), 1])
dataFrame = np.empty([len(xData), 1])
syntaxCount = np.empty([len(xData), 1])
importCount = np.empty([len(xData), 1])
packagePythonCount = np.empty([len(xData), 1])
packageRCount = np.empty([len(xData), 1])
curlyCount = np.empty([len(xData), 1])
errorPythonCount = np.empty([len(xData), 1])
errorRCount = np.empty([len(xData), 1])
statCount = np.empty([len(xData), 1])
latexCount = np.empty([len(xData), 1])

for i in range(0,len(xData)):
    for j in range(1,3):
        xData[i][j] = xData[i][j].lower()
        # Power Features
        if j == 1:
            titleWordCount[i] =len(xData[i][j].split(" "))
        if j == 2:
            codeCount[i] = xData[i][j].count('<code>')
            pCount[i] = xData[i][j].count('<p>')

        pyLoopCount[i] = len(re.findall('(?s)for.+?:', xData[i][1])) \
        + len(re.findall('(?s)if.+?:', xData[i][1])) \
        + len(re.findall('(?s)for.+?:', xData[i][1])) + len(re.findall('(?s)for.+?:', xData[i][2])) \
        + len(re.findall('(?s)if.+?:', xData[i][2])) \
        + len(re.findall('(?s)for.+?:', xData[i][2]))

        lowIndexCount[i] = len(re.findall('\[0\]', xData[i][2])) + len(re.findall('\[0\]', xData[i][2]))
        rLoopCount[i] = len(re.findall('(?s)for.+?\{', xData[i][1])) \
            + len(re.findall('(?s)if.+?\{', xData[i][1])) \
            + len(re.findall('(?s)while.+?\{', xData[i][1])) + len(re.findall('(?s)for.+?\{', xData[i][2])) \
                + len(re.findall('(?s)if.+?\{', xData[i][2])) \
                + len(re.findall('(?s)while.+?\{', xData[i][2]))

        libraryCount[i] = len(re.findall('library\(', xData[i][1])) \
            + len(re.findall('require\(', xData[i][1])) + len(re.findall('library\(', xData[i][2])) \
                + len(re.findall('require\(', xData[i][2]))

        dataFrame[i] = len(re.findall('data\.frame', xData[i][1])) + len(re.findall('data\.frame', xData[i][2]))
        syntaxCount[i] = len(re.findall('(?s)\.*?\(\)', xData[i][1])) + len(re.findall('(?s)\.*?\(\)', xData[i][2]))
        importCount[i] = len(re.findall('import', xData[i][1])) + len(re.findall('import', xData[i][2]))

        pc = 0
        for pp, pat in enumerate(pythonPackages):
            currentCount = len(re.findall(pat, xData[i][1])) + len(re.findall(pat,xData[i][2]))
            pc = pc + currentCount
        packagePythonCount[i] = pc

        rc = 0
        for rr, pat in enumerate(rPackages):
            currentCount = len(re.findall(pat, xData[i][1])) + len(re.findall(pat, xData[i][2]))
            rc = rc + currentCount
        packageRCount[i] = rc

        sc = 0
        for gg, pat in enumerate(statPack):
            currentCount = len(re.findall(pat, xData[i][1])) + len(re.findall(pat, xData[i][2]))
            sc = sc + currentCount
        statCount[i] = sc

        latexCount[i] = len(re.findall('\$', xData[i][1])) + len(re.findall('\$', xData[i][2]))
        curlyCount[i] = len(re.findall('\{', xData[i][1])) + len(re.findall('\{', xData[i][2]))
        errorPythonCount[i] = len(re.findall('Traceback', xData[i][1])) \
        + len(re.findall('[^\s]Error', xData[i][1])) + len(re.findall('not defined', xData[i][1])) \
        + len(re.findall('Traceback', xData[i][2])) + len(re.findall('[^\s]Error', xData[i][2])) \
        + len(re.findall('not defined', xData[i][2]))

        errorRCount[i] = len(re.findall('Error in', xData[i][1])) + len(re.findall('not found', xData[i][1])) \
        + len(re.findall('Error in', xData[i][2])) + len(re.findall('not found', xData[i][2]))

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

np.savetxt('KAGGLE-powerFeatures.csv', powerCombined, delimiter=",", fmt="%s")

#####################################################
###      Create Data Matrix of Word Counts        ###
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

# Change from Word Counts to Word Frequency
for i, row in enumerate(xCombined):
    tot = sum(row)
    xCombined[i, :] = xCombined[i, :]/float(tot)

elapsed = time.time() - t #toc
print("The total time elasped was:" + str(elapsed))
np.savetxt('KAGGLE-xCombined.csv', xCombined, fmt="%s")
