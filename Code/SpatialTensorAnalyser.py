import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sc
import sys

from pathlib import Path
from PermutationTester import PermutationTester
from SpatialAutoCorrelationCalculator import SpatialAutoCorrelationCalculator

class SpatialTensorAnalyser (object):

    def __init__(self, baseFolder, searchAllFolders=True, sep=",", skipFooter=4,
                 onlySelectCellsAtLeastNeighbours=0, timePointTxt=""):
        self.baseFolder = baseFolder
        self.searchAllFolders = searchAllFolders
        self.sep = sep
        self.skipFooter = skipFooter
        self.onlySelectCellsAtLeastNeighbours = onlySelectCellsAtLeastNeighbours
        self.timePointTxt = timePointTxt
        assert isinstance(self.onlySelectCellsAtLeastNeighbours, int), "The onlySelectCellsAtLeastNeighbours is no int. {} != 0".format(type(onlySelectCellsAtLeastNeighbours))
        self.allowedMethodNames = ("Assortativity", "Moran's I", "Gerry's C", "Getis-Ord General G", "Anselin Local Morans I", "Getis-Ord Gi Star")
        self.loacalMethodNames = ("Anselin Local Morans I", "Getis-Ord Gi Star")
        self.permutationTester = PermutationTester(testAllPermutations=True)

    def ApplySpatialCorMethod(self, measure, method, groups,
                              filenameToSave=None,
                              printOut=False):
        allSampleCorrelations = {}
        self.measure = measure
        assert method in self.allowedMethodNames, "The method {} is not exisiting in the allowed method names. {} not in {}".format(self.measure, self.measure, self.allowedMethodNames)
        isLocalMethod = method in self.loacalMethodNames
        for sampleName in self.tensorsOfAllSamples.keys():
            self.sampleName = sampleName
            geometryTable = self.dataOfAllSamples[sampleName]["geometryTable"]
            correlation = self.ApplySpatialCorrelation(sampleName, measure, method)
            allSampleCorrelations[sampleName] = correlation
        if not groups is None and not isLocalMethod:
            sortedSampleNames, sortedCorrelations = self.SortDictValuesByGroupKeys(allSampleCorrelations, groups)
            if printOut:
                for i, groupName in enumerate(groups):
                    print("group:", groupName)
                    for sampleName, sampleCorrelation in zip(sortedSampleNames[i], sortedCorrelations[i]):
                        if isinstance(sampleCorrelation, (list, tuple, np.ndarray)) :
                            print("sample:", sampleName, "with cor of", sampleCorrelation[0], "p-value:", sampleCorrelation[1])
                        else:
                            print("sample:", sampleName, "with cor of", sampleCorrelation)
            pValueTxt = self.calcPValueTxt(sortedCorrelations, groups)
            if len(sortedCorrelations[0].shape) == 1:
                columns = ["sample name", "r", ""]
            else:
                columns = ["sample name", "r", "p-value"]
            table = self.convertToDf(sortedSampleNames, sortedCorrelations, columns)
            meanTable = self.createMeanTable(sortedCorrelations, groups)
            if filenameToSave:
                table.to_csv(filenameToSave, index=False)
                meanTable.to_csv(filenameToSave, index=False, mode="a")
                file = open(filenameToSave, "a")
                file.write(pValueTxt)
                file.close()
            return table, meanTable
        return allSampleCorrelations

    def ApplySpatialCorrelation(self, sampleName, measure, method):
        self.currentSampleName = sampleName
        network = self.getDataOfSample(sampleName, "network")
        tensorMeasure = self.getTensorMeasureOfSample(sampleName, measure)
        self.removedNotOccuringNodes(network, list(tensorMeasure.keys()))
        selectedCells = self.getCellsWithAtLeastNeighbours(network)
        selectedValues = self.calcMeasureOf(tensorMeasure, selectedCells)
        if method == self.allowedMethodNames[0]:
            correlation = self.calcAssortativityOf(tensorMeasure, network, selectedCells)
        elif method == self.allowedMethodNames[1]:
            correlation = SpatialAutoCorrelationCalculator(network,
                                        selectedValues,
                                        method=method,
                                        selectedCells=selectedCells).GetMoransI()
        elif method == self.allowedMethodNames[2]:
            correlation = SpatialAutoCorrelationCalculator(network,
                                        selectedValues,
                                        method=method,
                                        selectedCells=selectedCells).GetGearysC()
        elif method == self.allowedMethodNames[3]:
            correlation = SpatialAutoCorrelationCalculator(network,
                                        selectedValues,
                                        method=method,
                                        selectedCells=selectedCells).GetGetisOrdGeneralG()
        elif method == self.allowedMethodNames[4]:
            correlation = SpatialAutoCorrelationCalculator(network,
                                        selectedValues,
                                        method=method,
                                        selectedCells=selectedCells).GetAnselinLocalMoransI()
        elif method == self.allowedMethodNames[5]:
            correlation = SpatialAutoCorrelationCalculator(network,
                                        selectedValues,
                                        method=method,
                                        selectedCells=selectedCells).GetGetisOrdGiStar()
        return correlation

    def getTensorMeasureOfSample(self, sampleName, measureName):
        assert sampleName in self.tensorsOfAllSamples, "The sample {} was not found. Only the samples {} are available.".format(sampleName, self.tensorsOfAllSamples.keys())
        tensorMeasure = {}
        tensorsOfSample = self.tensorsOfAllSamples[sampleName]
        assert measureName in tensorsOfSample[list(tensorsOfSample.keys())[0]], "The measureName {} was not found. Only the measureNames {} are available.".format(measureName, tensorsOfSample[list(tensorsOfSample.keys())[0]].keys())
        for cellLabel, allMeasures in tensorsOfSample.items():
            tensorMeasure[cellLabel] = allMeasures[measureName]
        return tensorMeasure

    def getDataOfSample(self, sampleName, dataName):
        assert sampleName in self.dataOfAllSamples, "The sample {} was not found. Only the samples {} are available.".format(sampleName, self.dataOfAllSamples.keys())
        dataOfSample = self.dataOfAllSamples[sampleName]
        assert dataName in dataOfSample, "The data {} was not found. Only the data {} are available.".format(dataName, dataOfSample.keys())
        return dataOfSample[dataName]

    def removedNotOccuringNodes(self, network, cellsOccuring):
        allNodes = np.asarray(list(network.nodes()))
        cellsOccuring = np.asarray(cellsOccuring).astype(int)
        isNodeMissing = np.isin(allNodes, cellsOccuring, invert=True)
        nodesToRemove = allNodes[isNodeMissing]
        network.remove_nodes_from(nodesToRemove)

    def getCellsWithAtLeastNeighbours(self, network):
        selectedCells = []
        degrees = nx.degree(network)
        for cell, degree in degrees:
            if degree >= self.onlySelectCellsAtLeastNeighbours:
                selectedCells.append(cell)
        return selectedCells

    def calcMeasureOf(self, tensorMeasure, selectedCells):
        selectedValues = np.zeros(len(selectedCells))
        for i, cell in enumerate(selectedCells):
            selectedValues[i] = tensorMeasure[cell]
        return selectedValues

    def calcAssortativityOf(self, tensorMeasure, network, selectedCells, saveAssortativity=False):
        selectedValues = self.calcMeasureOf(tensorMeasure, selectedCells)
        allAvgValues = np.zeros(len(selectedCells))
        for i, cell in enumerate(selectedCells):
            neighbours = network.neighbors(cell)
            try:
                allAvgValues[i] = self.calcAvgDictValueOfEntries(tensorMeasure, neighbours) # len(list(neighbours))#
            except:
                nx.draw(network, with_labels=True, pos=nx.spring_layout(network) )
                plt.show()
                sys.exit()
        if saveAssortativity:
            table = pd.DataFrame(np.asarray([selectedCells, selectedValues, allAvgValues]), index=["Label", "value", "avg neighbour value"])
            table = table.T
            filename = "Results/FibrilJ/Assortivity/assortativity of {}.csv".format(self.currentSampleName)
            table.to_csv(filename, index=False)
        pearsonr = sc.stats.stats.pearsonr(selectedValues, allAvgValues)
        return pearsonr

    def calcAvgDictValueOfEntries(self, dictionary, entries):
        summedValue = 0
        nrOfValidEntries = 0
        for e in entries:
            if e in dictionary:
                summedValue += dictionary[e]
                nrOfValidEntries += 1
        return summedValue / nrOfValidEntries

    def SortDictValuesByGroupKeys(self, dictionary, groupingKeys):
        sortedSampleNames = []
        sortedDictValues = []
        for groupKey in groupingKeys:
            sampleNames = []
            dictValues = []
            for key, value in dictionary.items():
                if groupKey in key:
                    sampleNames.append(key)
                    dictValues.append(value)
            sortedSampleNames.append(sampleNames)
            sortedDictValues.append(np.asarray(dictValues))
        return sortedSampleNames, sortedDictValues

    def calcPValueTxt(self, sortedCorrelations, groups):
        pValueTxt = ""
        nrOfOtherGroups = len(groups)-1
        pValues = np.zeros(nrOfOtherGroups)
        for i in range(nrOfOtherGroups):
            group1 = sortedCorrelations[0]
            group2 = sortedCorrelations[i+1]
            if len(group1.shape) > 1:
                group1 = group1[:, 0]
                group2 = group2[:, 0]
            pValue = self.permutationTester.CalcPermutationTest(group1, group2)
            currentPValueTxt = "{} vs {} with p-value of {}\n".format(groups[0], groups[i+1], pValue)
            pValueTxt += currentPValueTxt
        return pValueTxt

    def convertToDf(self, sortedSampleNames, sortedDictValues, columns=None):
        table = []
        sortedSampleNames = np.concatenate(sortedSampleNames)
        sortedDictValues = np.concatenate(sortedDictValues)
        for i in range(len(sortedSampleNames)):
            row = [sortedSampleNames[i]]
            if isinstance(sortedDictValues[i], (list, tuple, np.ndarray)):
                values = np.round(list(sortedDictValues[i]), 5)
            else:
                values = [np.round(sortedDictValues[i], 5), np.NaN]
            row.extend(values)
            table.append(row)
        table = pd.DataFrame(table)
        if not columns is None:
            table.columns = columns
        return table

    def createMeanTable(self, sortedCorrelations, groups):
        columns = ["group", "mean", "sd"]
        meanTable = []
        for i, group in enumerate(groups):
            correlationOfGroup = sortedCorrelations[i]
            if len(correlationOfGroup.shape) > 1:
                correlationOfGroup = correlationOfGroup[:, 0]
            meanValue = np.mean(correlationOfGroup)
            sdValue = np.std(correlationOfGroup)
            meanTableEntry = [group, np.round(meanValue, 5), np.round(sdValue, 5)]
            meanTable.append(meanTableEntry)
        meanTable = pd.DataFrame(meanTable, columns=columns)
        return meanTable

    def GetSampleNames(self):
        return list(self.tensorsOfAllSamples.keys())

    def SetDataOfAllSamples(self, dataOfAllSamples):
        self.dataOfAllSamples = dataOfAllSamples

    def SetTensorsOfAllSamples(self, tensorsOfAllSamples):
        self.tensorsOfAllSamples = tensorsOfAllSamples

def main():
    baseFolder = "FibrilJ Tensors/AverageAngleChanges/"
    groups = ["WT", "clasp", "ktn"]
    method = "Getis-Ord General G"
    measure = "angle"
    resultsFolder = "Results/FibrilJ Tensors/Spatial Autocorrelation Tables/{}/AverageAngleChanges/".format(measure)
    Path(resultsFolder).mkdir(parents=True, exist_ok=True)
    for timePointTxt in [" 48", " 96"]:
        filenameToSave = "{}{} {}{}.csv".format(resultsFolder, measure, method, timePointTxt)
        print("filenameToSave", filenameToSave)
        analyser = SpatialTensorAnalyser(baseFolder)
        tensorsOfAllSamples = pickle.load(open(baseFolder+"tensorsOfAllSamples{}.pkl".format(timePointTxt), "rb"))
        dataOfAllSamples = pickle.load(open(baseFolder+"dataOfAllSamples{}.pkl".format(timePointTxt), "rb"))
        analyser.SetDataOfAllSamples(dataOfAllSamples)
        analyser.SetTensorsOfAllSamples(tensorsOfAllSamples)
        allAssortativities = analyser.ApplySpatialCorMethod(measure=measure, method=method,
                                                            groups=groups, printOut=False,
                                                            filenameToSave=filenameToSave)

if __name__ == '__main__':
    main()
