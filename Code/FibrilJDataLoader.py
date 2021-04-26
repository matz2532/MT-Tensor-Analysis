import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

class FibrilJDataLoader (object):

    def __init__(self, table, timePointTxt="", loadFromPkl=False, dumpData=True,
                 baseFolder="FibrilJ Tensors/"):
        # ----- allows creation and saving of networks and average changes of microtubule tensors -----
        self.table = table
        self.table.iloc[:, 2] = ["".join(sampleName.split(".tif")) for sampleName in self.table.iloc[:, 2]]
        self.timePointTxt = timePointTxt
        if loadFromPkl:
            self.selectedRows = pickle.load(open(baseFolder+"selectedRows{}.pkl".format(self.timePointTxt), "rb"))
            self.contours = pickle.load(open(baseFolder+"contours{}.pkl".format(self.timePointTxt), "rb"))
        else:
            self.selectedRows = self.calcRowsToSelect(self.table)
            self.contours = self.calcAllContours()
        self.connectivityNetworks = self.calcAllConnectivityNetworks()
        self.angles = self.calcAvgChangesInAngles()
        if dumpData:
            pickle.dump(self.selectedRows, open(baseFolder+"selectedRows{}.pkl".format(self.timePointTxt), "wb"))
            pickle.dump(self.contours, open(baseFolder+"contours{}.pkl".format(self.timePointTxt), "wb"))
            pickle.dump(self.connectivityNetworks, open(baseFolder+"networks{}.pkl".format(self.timePointTxt), "wb"))
            pickle.dump(self.angles, open(baseFolder+"angles{}.pkl".format(self.timePointTxt), "wb"))

    def calcRowsToSelect(self, table, timePoint=0, getAllTimePointsData=False):
        if getAllTimePointsData:
            selectedRows = np.arange(len(table))
            allGroups = table.iloc[:, 1].to_numpy()
            allSampleNames = table.iloc[:, 2].to_numpy()
        else:
            selectedRows = []
            allGroups = []
            allSampleNames = []
            for idx in range(table.shape[0]):
                time, group, sampleName = table.iloc[idx, :3]
                if time == timePoint:
                    selectedRows.append(idx)
                    allGroups.append(group)
                    allSampleNames.append(sampleName)
        selectedRowsTable = pd.DataFrame({"selected row":selectedRows, "group":allGroups, "sample name":allSampleNames})
        return selectedRowsTable

    def calcAngleDifAndSampleNameOfLastTimePoint(self):
        allAngleDiffs = []
        lastTimePointSampleNames = []
        sampleNameAndTime = self.selectedRows.iloc[:, 2].to_numpy()
        if self.timePointTxt == " 96":
            uniqueSamples = np.unique(["-".join(sampleName.split("-")[:-1]) for sampleName in sampleNameAndTime])
        else:
            uniqueSamples = np.unique(["_".join(sampleName.split("_")[:-1]) for sampleName in sampleNameAndTime])
        for name in uniqueSamples:
            isSample = [name in sampleName for sampleName in sampleNameAndTime]
            currentSampleNames = np.unique(sampleNameAndTime[isSample])
            for i in range(len(currentSampleNames)-1):
                startSampleName = currentSampleNames[i]
                endSampleName = currentSampleNames[i+1]
                print(startSampleName, endSampleName)
                angleDiff = self.calcAngleDifFor(startSampleName, endSampleName)
                allAngleDiffs.append(angleDiff)
            lastTimePointSampleNames.append(endSampleName)
        return allAngleDiffs, lastTimePointSampleNames

    def calcAngleDifFor(self, startSampleName, endSampleName):
        isStartName = np.isin(self.selectedRows.iloc[:, 2], startSampleName)
        isEndName = np.isin(self.selectedRows.iloc[:, 2], endSampleName)
        assert np.any(isStartName), "The sampleName {} is not present in the sample names {}".format(startSampleName, np.unique(self.selectedRows.iloc[:, 2]))
        assert np.any(isEndName), "The sampleName {} is not present in the sample names {}".format(endSampleName, np.unique(self.selectedRows.iloc[:, 2]))
        idxOfStart = self.selectedRows.iloc[np.where(isStartName)[0], 0]
        idxOfEnd = self.selectedRows.iloc[np.where(isEndName)[0], 0]
        startAngle = self.table.iloc[idxOfStart, 7].to_numpy() + 90
        endAngle = self.table.iloc[idxOfEnd, 7].to_numpy() + 90
        angleDiff = endAngle - startAngle
        return angleDiff

    def removeLastTimePointFromSelectedRows(self, lastTimePointSampleNames):
        isLastTimePoint = np.isin(self.selectedRows.iloc[:, 2], lastTimePointSampleNames)
        idxOfLastTimePoints = self.selectedRows.iloc[np.where(isLastTimePoint)[0], 0]
        self.selectedRows.drop(idxOfLastTimePoints, inplace=True)

    def calcAllContours(self):
        contours = {}
        for idx in self.selectedRows.iloc[:, 0]:
            entry = self.table.iloc[idx, :]
            contour = self.calcContour(entry, contourStart=9)
            contours[idx] = contour
        return contours

    def calcContour(self, entry, contourStart):
        entry = entry.iloc[contourStart:]
        contour = []
        for i in range(len(entry)):
            if i % 2 == 0:
                x = entry[i]
                if pd.isna(x):
                    break
            else:
                y = entry[i]
                if pd.isna(y):
                    print("breaking at y")
                    break
                contour.append([x, y])
        contour = np.asarray(contour).astype(int)
        return contour

    def calcAllConnectivityNetworks(self):
        allSamples = np.unique(self.selectedRows.iloc[:, 2])
        allNetworks = []
        for sampleName in allSamples:
            self.sampleName = sampleName
            selectedContours = self.getContoursOfSample(sampleName)
            contourImage = self.createContourImage(selectedContours)
            adjacencyMatrix = self.calcAdjacencyMatrix(selectedContours, contourImage)
            network = nx.from_numpy_matrix(adjacencyMatrix)
            allNetworks.append(network)
        return allNetworks

    def getContoursOfSample(self, sampleName):
        indicesInTable = self.getIndicesInTableOf(sampleName)
        selectedContours = [self.contours[idx] for idx in indicesInTable]
        return selectedContours

    def getIndicesInTableOf(self, sampleName):
        idxOfSampleName = np.where(self.selectedRows.iloc[:, 2] == sampleName)[0]
        indicesInTable = self.selectedRows.iloc[idxOfSampleName, 0]
        return indicesInTable

    def createContourImage(self, selectedContours):
        imageShape = self.calcImageShapeFrom(selectedContours)
        contourImage = np.zeros(imageShape)
        for i, contour in enumerate(selectedContours):
            for x, y in contour:
                contourImage[x, y] = i+1
        return contourImage

    def calcImageShapeFrom(self, selectedContours):
        maxX, maxY = 0, 0
        for contour in selectedContours:
            currentMaxX, currentMaxY = np.max(contour, axis=0)
            if maxX < currentMaxX:
                maxX = currentMaxX
            if maxY < currentMaxY:
                maxY = currentMaxY
        return maxX+1, maxY+1

    def calcAdjacencyMatrix(self, selectedContours, contourImage, offset=25):
        uniqueLabels = np.unique(contourImage)[1:].astype(int)
        shape = contourImage.shape
        adjMt = np.zeros((len(uniqueLabels), len(uniqueLabels)))
        for i in range(len(selectedContours)):
            label = uniqueLabels[i]
            allFoundLabels = []
            for x, y in selectedContours[i]:
                fromX, toX = self.calcRange(x, offset, shape[0])
                fromY, toY = self.calcRange(y, offset, shape[1])
                selectedContourImage = contourImage[fromX:toX, fromY:toY]
                foundLabels = np.unique(selectedContourImage)[1:]
                foundLabels = foundLabels[np.isin(foundLabels, label, invert=True)]
                allFoundLabels.append(foundLabels)
            allFoundLabels = np.concatenate(allFoundLabels).astype(int)
            adjMt[i, allFoundLabels-1] = 1
            adjMt[allFoundLabels-1, i] = 1
        return adjMt

    def calcRange(self, center, offset, maxValue):
        assert center <= maxValue, "The center value shouldn't be bigger than the maximum value. {} > {}".format(center, maxValue)
        fromValue = center - offset
        toValue = center + offset
        if fromValue < 0:
            fromValue = 0
        if toValue > maxValue:
            toValue = maxValue
        return fromValue, toValue

    def calcAvgChangesInAngles(self):
        self.table.iloc[:, 2] = [filename[:-6] for filename in self.table.iloc[:, 2]]
        allSamples = np.unique(self.table.iloc[:, 2])
        allAvgAngles = []
        for sampleName in allSamples:
            indicesInTable = np.where(np.isin(self.table.iloc[:, 2], sampleName))[0]
            angles = self.table.iloc[indicesInTable, 7].to_numpy()
            angles += 90
            timePoints = self.table.iloc[indicesInTable, 0]
            avgAngles = self.calcAvgDifference(angles, timePoints)
            allAvgAngles.append(avgAngles)
        return allAvgAngles

    def calcAvgDifference(self, angles, timePoints):
        anglesOfTimePoint = []
        for timePoint in np.unique(timePoints):
            isTimePoint = np.isin(timePoints, timePoint)
            anglesOfTimePoint.append(angles[isTimePoint])
        changesOfAngles = []
        for i in range(len(anglesOfTimePoint)-1):
            changes = np.abs(anglesOfTimePoint[i+1] - anglesOfTimePoint[i])
            changes[changes>90] -= 90
            changesOfAngles.append(changes)
        avgChanges = np.mean(changesOfAngles, axis=0)
        return avgChanges

    def extractAngles(self, selectedRows):
        allSamples = np.unique(selectedRows.iloc[:, 2])
        allAngles = []
        for sampleName in allSamples:
            indicesInTable = self.getIndicesInTableOf(sampleName).to_numpy()
            angles = self.table.iloc[indicesInTable, 7].to_numpy()
            angles += 90
            allAngles.append(angles)
        return allAngles

    def PlotContours(self):
        key = list(self.contours.keys())[0]
        contour = self.contours[key]
        maxY = np.max(contour[:, 1])+1
        maxX = np.max(contour[:, 0])+1
        image = np.zeros((maxX, maxY))
        for x,y in contour:
            image[x, y] = 255
        plt.imshow(image)
        plt.show()

    def GetContours(self):
        return self.contours

    def convertToDict(self, loadData=True, selectedRows=None, contours=None,
                      connectivityNetworks=None, angles=None, timePointTxt="",
                      baseFolder="FibrilJ Tensors/"):
        if loadData:
            selectedRows = pickle.load(open(baseFolder+"selectedRows{}.pkl".format(timePointTxt), "rb"))
            contours = pickle.load(open(baseFolder+"contours{}.pkl".format(timePointTxt), "rb"))
            connectivityNetworks = pickle.load(open(baseFolder+"networks{}.pkl".format(timePointTxt), "rb"))
            angles = pickle.load(open(baseFolder+"angles{}.pkl".format(timePointTxt), "rb"))
        uniqueSampleNames = np.unique(selectedRows.iloc[:, 2])
        tensorsOfAllSamples = {}
        dataOfAllSamples = {}
        allSamples = []
        for i, sampleName in enumerate(uniqueSampleNames):
            sampleName = "".join(sampleName.split(".tif"))
            allSamples.append(sampleName)
            dataOfAllSamples[sampleName] = {}
            network = connectivityNetworks[i]
            dataOfAllSamples[sampleName]["network"] = network
            dataOfAllSamples[sampleName]["geometryTable"] = "Dummy"
            sampleMeasureDict = {}
            for j, node in enumerate(list(network.nodes())):
                sampleMeasureDict[node] = {"angle":angles[i][j]}
            tensorsOfAllSamples[sampleName] = sampleMeasureDict
        print(allSamples)
        pickle.dump(tensorsOfAllSamples, open(baseFolder+"tensorsOfAllSamples{}.pkl".format(timePointTxt), "wb"))
        pickle.dump(dataOfAllSamples, open(baseFolder+"dataOfAllSamples{}.pkl".format(timePointTxt), "wb"))

def main():
    # ----- create data of all samples (networks and average tensor changes of intervall) for time point (48h or 96h) -----
    folder = "FibrilJ Tensors/"
    for is48hTimePoint in [False, True]:
        if is48hTimePoint:
            filename = folder + "MT_Tensor_Orientation_48h.xlsx"
            timePointTxt = " 48"
        else:
            filename = folder + "MT_Tensor_Orientation_96h.xlsx"
            timePointTxt = " 96"
        baseFolder = "FibrilJ Tensors/AverageAngleChanges/"
        table = pd.read_excel(filename)
        myFibrilJDataLoader = FibrilJDataLoader(table, timePointTxt=timePointTxt,
                                                baseFolder=baseFolder)
        myFibrilJDataLoader.convertToDict(timePointTxt=timePointTxt, baseFolder=baseFolder)

if __name__ == '__main__':
    main()
