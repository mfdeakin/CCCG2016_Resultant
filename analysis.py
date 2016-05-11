import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.decomposition

testNameMap = {"hardEllipsoidsSingle": ("Shifted", "Ellipsoids"),
               "packedEll": ("Packed", "Ellipsoids")}

def readData(fname, size):
    fd = open(fname, 'r')
    reader = csv.reader(fd)
    next(reader)
    data = []
    for row in reader:
        if len(row) != 6:
            continue
        number, resNum, resTime, mpNum, mpTime, correct = map(int, row)
        data.append([size,
                     resNum, resTime / 1e6,
                     mpNum, mpTime / 1e6,
                     correct, number])
    return np.array(data)

def validateFName(fname):
    testInfo = fname.split('.')
    if len(testInfo) != 5:
        return None
    testName, testSize, _, testMachine, fileType = fname.split('.')
    if fileType != "csv" or testName == "cylinders":
        return None
    try:
        testSize = int(testSize)
    except ValueError:
        return None
    return [testMachine, testName, testSize]

def createPlot(data, fname, maxPrec,
               resCoeff, resConst,
               mpCoeff, mpConst):
    numQuads = data.T[0]
    resNum = data.T[1]
    resTimes = data.T[2]
    mpNum = data.T[3]
    mpTimes = data.T[4]
    plt.suptitle(fname)
    plt.axes().set_color_cycle(["cyan", "orange"])
    plt.scatter(resNum, resTimes, c = "blue",
               label = "Resultant Method")
    plt.scatter(mpNum, mpTimes, c = "red",
                label = "Increased Precision Method")
    if not (np.isnan(resCoeff) or np.isnan(resConst) or
            np.isnan(mpCoeff) or np.isnan(mpConst)):
        lines = plt.plot([0, maxPrec], [resConst, maxPrec * resCoeff + resConst], '-',
                         [0, maxPrec], [mpConst, maxPrec * mpCoeff + mpConst], '-')
        plt.setp(lines, linewidth=4)
    plt.axes().set_xlim(left = 0,
                        right = maxPrec)
    maxTime = max(max(mpTimes), max(resTimes)) * 1.0625
    plt.axes().set_ylim(bottom = 0,
                        top = maxTime)
    plt.yticks(np.arange(0, maxTime + 1, maxTime / 10.0))
    plt.axes().yaxis.get_major_formatter().set_powerlimits((0, 3))
    print("Saving '" + fname + "'")
    plt.savefig(fname, format = "png", dpi = 300)
    plt.close()

def aggregateData():
    files = os.listdir()
    datum = {}
    for fname in files:
        dataInfo = validateFName(fname)
        if dataInfo == None:
            continue
        print(fname)
        machine, test, size = dataInfo
        if not machine in datum:
            datum[machine] = {}
        if not test in datum[machine]:
            datum[machine][test] = {}
        if not size in datum[machine][test]:
            datum[machine][test][size] = np.array([])
        testData = datum[machine][test][size]
        data = readData(fname, size)
        newShape = (testData.shape[0] + data.shape[0],
                    data.shape[1])
        datum[machine][test][size] = np.append(testData, data).reshape(newShape)
    return datum

def analyzeData(datum, plotData = False):
    analysis = {}
    for machine in datum:
        analysis[machine] = {}
        for test in datum[machine]:
            mtData = np.array([])
            for size in datum[machine][test]:
                testData = datum[machine][test][size]
                mtShape = (mtData.shape[0] + testData.shape[0],
                           testData.shape[1])
                mtData = np.append(mtData, testData)
                mtData = mtData.reshape(mtShape)
            numQuads = mtData.T[0]
            resNum = mtData.T[1]
            resTimes = mtData.T[2]
            mpNum = mtData.T[3]
            mpTimes = mtData.T[4]
            if max(numQuads) == min(numQuads):
                resIndep = np.vstack([resNum,
                                      np.ones(len(resNum))]).T
                resLSqr = list(np.linalg.lstsq(resIndep, resTimes))
                resLSqr[0] = [0, resLSqr[0][0], resLSqr[0][1]]
                mpIndep = np.vstack([mpNum,
                                     np.ones(len(mpNum))]).T
                mpLSqr = list(np.linalg.lstsq(mpIndep, mpTimes))
                mpLSqr[0] = [0, mpLSqr[0][0], mpLSqr[0][1]]
            else:
                resIndep = np.vstack([numQuads, resNum,
                                  np.ones(len(resNum))]).T
                resLSqr = np.linalg.lstsq(resIndep, resTimes)
                mpIndep = np.vstack([numQuads, mpNum,
                                     np.ones(len(mpNum))]).T
                mpLSqr = np.linalg.lstsq(mpIndep, mpTimes)
            numIncorrect = len(resNum) - np.sum(mtData.T[5])
            analysis[machine][test] = (resLSqr, mpLSqr, numIncorrect)
            print(machine, test)
            print("Resultant Least Square ((quadrics, precIncreases, constant), (residual), rank, singular values):\n", resLSqr)
            print("Repeated Squares Least Square ((quadrics, precIncreases, constant), (residual), rank, singular values):\n", mpLSqr)
            print()
            if plotData:
                adjuster = np.identity(mtData.shape[1])
                adjuster[2][0] = -resLSqr[0][0]
                adjuster[4][0] = -mpLSqr[0][0]
                sizeAdjusted = adjuster.dot(mtData.T).T
                plotFName = test + "." + machine + ".all.minPrec.png"
                createPlot(mtData, plotFName, min(max(resNum), max(mpNum)),
                           resLSqr[0][1], resLSqr[0][2],
                           mpLSqr[0][1], mpLSqr[0][2])
                plotFName = test + "." + machine + ".all.maxPrec.png"
                createPlot(mtData, plotFName, max(max(resNum), max(mpNum)),
                           resLSqr[0][1], resLSqr[0][2],
                           mpLSqr[0][1], mpLSqr[0][2])
                plotFName = test + "." + machine + ".adjusted.minPrec.png"
                createPlot(sizeAdjusted, plotFName, min(max(resNum), max(mpNum)),
                           resLSqr[0][1], resLSqr[0][2],
                           mpLSqr[0][1], mpLSqr[0][2])
                plotFName = test + "." + machine + ".adjusted.maxPrec.png"
                createPlot(sizeAdjusted, plotFName, max(max(resNum), max(mpNum)),
                           resLSqr[0][1], resLSqr[0][2],
                           mpLSqr[0][1], mpLSqr[0][2])
    return analysis

def buildTable(analysis, fname):
    tableOut = open(fname, 'w')
    header = ("\\begin{tabular}{|l|l|ll|lll|l|}\n" +
              "\\hline\n" +
              "Machine & Scene & Method & Disagreements & " +
              "ms/Quadric & ms/Comp & Const ms & " +
              "Residual ($err^2$)\\\\\n" +
              "\\hline\n")
    tableOut.write(header)
    for m in analysis:
        oneMachine = False
        for t in analysis[m]:
            resLSqr, mpLSqr, numWrong = analysis[m][t]
            rowStr = "\\hline\n"
            if oneMachine == False:
                rowStr += m.capitalize() + " "
                oneMachine = True
            rowStr += ("& {:s} & Approximate & -- & " +
                       "{:0.06f} & {:0.06f} & {:0.06f} & " +
                       "{:0.06f}\\\\\n")
            rSqr = 0
            if len(mpLSqr[1]) > 0:
                rSqr = mpLSqr[1][0]
            rowStr = rowStr.format(testNameMap[t][0], mpLSqr[0][0], mpLSqr[0][1],
                                   mpLSqr[0][2], rSqr)
            tableOut.write(rowStr)
            rowStr = ("& {:s} & Resultant & {:d} & " +
                       "{:0.06f} & {:0.06f} & {:0.06f} & " +
                       "{:0.06f}\\\\\n")
            rSqr = 0
            if len(resLSqr[1]) > 0:
                rSqr = resLSqr[1][0]
            rowStr = rowStr.format(testNameMap[t][1], int(numWrong), resLSqr[0][0], resLSqr[0][1],
                                   resLSqr[0][2], rSqr)
            tableOut.write(rowStr)
    footer = ("\\hline\n" +
              "\\end{tabular}\n")
    tableOut.write(footer)

if __name__ == "__main__":
    datum = aggregateData()
    analysis = analyzeData(datum, False)
    buildTable(analysis, "comparison_table.tex")
    
