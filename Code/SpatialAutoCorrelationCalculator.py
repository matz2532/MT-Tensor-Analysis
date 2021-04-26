import networkx as nx
import numpy as np
import scipy as sc
import sys

class SpatialAutoCorrelationCalculator (object):

    def __init__(self, network, value, method=None,
                 selectedCells=None):
        self.network = network
        if not selectedCells is None:
            if len(selectedCells) != len(list(self.network.nodes())):
                self.network = self.network.subgraph(selectedCells)
        self.weightMatrix = nx.to_numpy_matrix(self.network)
        self.value = np.asarray(value)
        self.setNeededValues()
        self.methodNames = ("Moran's I", "Gerry's C", "Getis-Ord General G",
                            "Anselin Local Morans I", "Getis-Ord Gi Star")
        self.moransI = None
        self.gearysC = None
        self.getisOrdGeneralG = None
        self.getisOrdGiStar = None
        if method is None:
            self.moransI = self.calcMoransI()
            self.gearysC = self.calcGearysC()
            self.getisOrdGeneralG = self.calcGetisOrdGeneralG()
            self.anselinLocalMoransI = self.calcAnselinLocalMoransI()
            self.getisOrdGiStar = self.calcGetisOrdGiStar()
        elif method == self.methodNames[0]:
            self.moransI = self.calcMoransI()
        elif method == self.methodNames[1]:
            self.gearysC = self.calcGearysC()
        elif method == self.methodNames[2]:
            self.getisOrdGeneralG = self.calcGetisOrdGeneralG()
        elif method == self.methodNames[3]:
            self.anselinLocalMoransI = self.calcAnselinLocalMoransI()
        elif method == self.methodNames[4]:
            self.getisOrdGiStar = self.calcGetisOrdGiStar()
        else:
            print("The method {} does not exist".format(method))

    def setNeededValues(self):
        self.meanValue = np.mean(self.value)
        self.valueDifFromMean = self.value - self.meanValue
        self.numberOfValues = len(self.value)
        self.sumOfWeightMatrix = np.sum(self.weightMatrix)
        self.summedVariance = np.sum((self.value - self.meanValue)**2)

    def calcMoransI(self):
        self.weightedCovariance = self.summedWeightedDeviationsOfAllIAndJ()
        moransI = (self.numberOfValues * self.weightedCovariance) / (self.sumOfWeightMatrix * self.summedVariance)
        expectedMoaransI = self.calcExpectedMoransI()
        pValue = self.calcPValue(moransI, expectedMoaransI)
        return moransI, pValue

    def summedWeightedDeviationsOfAllIAndJ(self):
        weightedCov = 0
        for i in range(self.weightMatrix.shape[0]):
            for j in range(self.weightMatrix.shape[1]):
                weightedCov += self.weightMatrix[i, j] * self.valueDifFromMean[i] * self.valueDifFromMean[j]
        return weightedCov

    def calcExpectedMoransI(self):
        expectedMoaransI = -1 / ( self.numberOfValues - 1 )
        return expectedMoaransI

    def calcPValue(self, value, expectedValue, method="Moran's I"):
        if method == "Moran's I":
            A, B, C = self.calcABCOfMoransI()
        elif method == "Geary's C":
            A, B, C = self.calcABCOfGearysC()
        elif method == "Getis-Ord General G":
            A, B, C = self.calcABCOfGeneralG()
        else:
            print("The method {} does not exist.".format(method))
            sys.exit()
        ESquaredValue = (A + B) / C
        var = ESquaredValue - expectedValue**2
        zScore = (value - expectedValue) / np.sqrt(var)
        pValue = self.convertZScoreToPValue(zScore)
        return pValue

    def calcABCOfMoransI(self):
        n = self.numberOfValues
        nSquared = n**2
        S0Squared = self.sumOfWeightMatrix
        S1 = self.calcS1()
        S2 = self.calcS2()
        D = np.sum(self.valueDifFromMean**4) / ( np.sum(self.valueDifFromMean**2) ** 2 )
        A = n * ( (nSquared - 3*n + 3) * S1 - n * S2 + 3 * S0Squared )
        B = D * ( (nSquared - n) * S1 - 2 * n * S2 + 6 * S0Squared )
        C = (n - 1) * (n - 2) * (n - 3) * S0Squared
        return A, B, C

    def calcS1(self):
        S1 = 0
        for i in range(self.weightMatrix.shape[0]):
            for j in range(self.weightMatrix.shape[1]):
                S1 += (self.weightMatrix[i, j] + self.weightMatrix[j, i]) ** 2
        S1 *= 0.5
        return S1

    def calcS2(self):
        S2 = 0
        for i in range(self.weightMatrix.shape[0]):
            summedRowI = np.sum(self.weightMatrix[i, :])
            summedColI = np.sum(self.weightMatrix[:, i])
            S2 += (summedRowI + summedColI) ** 2
        return S2

    def calcABCOfGearysC(self):
        S0 = self.sumOfWeightMatrix
        S1 = self.calcS1()
        S2 = self.calcS2()
        n = self.numberOfValues
        D = np.sum(self.valueDifFromMean**4) / ( np.sum(self.valueDifFromMean**2) ** 2 )
        nSquared = np.sqrt(n)
        # under normality?
        # A = (2*S1+S2)*(n-1)
        # B = -4*S0**2
        # C = 2*(n+1)*S0**2
        # under random
        A = (n-1)*S1*(nSquared-3*n+3-D*(n-1)) + S0**2*(nSquared-3-D*(n-1)**2)
        B = - 1 * (0.25*(n-1)*S2*(nSquared+3*n-6-D*(nSquared-n+2)))
        C = n*(n-2)*(n-3)*S0**2
        return A, B, C

    def calcABCOfGeneralG(self):
        summedDiagonalWeights = 0
        sumOfWeightWithOutIEqualJ = 0
        for i in range(self.weightMatrix.shape[0]):
            for j in range(self.weightMatrix.shape[1]):
                if i != j:
                    sumOfWeightWithOutIEqualJ += self.weightMatrix[i, j]
        W = sumOfWeightWithOutIEqualJ
        S1 = self.calcS1()
        S2 = self.calcS2()
        n = self.numberOfValues
        D0 = (n**2 - 3*n + 3)*S1 - n*S2 + 3 * W**2
        D1 = - ( (n**2 - n)*S1 - 2*n*S2 + 6 * W**2)
        D2 = - (2*n*S1 - (n+3)*S2 + 6 * W**2)
        D3 = 4 * (n-1) * S1 - 2 * (n+1) * S2 + 8 * W**2
        D4 = S1 - S2 + W**2
        summedSquaredValues = np.sum(self.value**2)
        squaredSummedValues = np.sum(self.value)**2
        A = D0*np.sum(self.value**2)**2 + D1*np.sum(self.value**4) + D2*np.sum(self.value)**2*np.sum(self.value**2)
        B = D3*np.sum(self.value)*np.sum(self.value**3) + D4*np.sum(self.value)**4
        C = (np.sum(self.value)**2 - np.sum(self.value**2))**2 * n*(n-1)*(n-2)*(n-3)
        return A, B, C

    def convertZScoreToPValue(self, zScore):
        return 0.5 * (1 + sc.special.erf(zScore / np.sqrt(2)))

    def calcGearysC(self):
        weightedSummedDifferences = self.calcWeightedSummedDifferences()
        C = (self.numberOfValues - 1) * weightedSummedDifferences / (2 * self.sumOfWeightMatrix * self.summedVariance)
        expectedGearysC = 1
        pValue = self.calcPValue(C, expectedGearysC, method="Geary's C")
        return C, pValue

    def calcWeightedSummedDifferences(self):
        weightedSummedDifferences = 0
        for i in range(self.weightMatrix.shape[0]):
            for j in range(self.weightMatrix.shape[1]):
                weightedSummedDifferences += self.weightMatrix[i, j] * (self.value[i] - self.value[j])**2
        return weightedSummedDifferences

    def calcGetisOrdGeneralG(self):
        sumWeightedMultipliedValues = self.calcSumWeightedMultipliedValues()
        sumMultipliedValues = self.calcSumMultipliedValues()
        generalG = sumWeightedMultipliedValues / sumMultipliedValues
        expectedGeneralG = self.calcExpectedGeneralG()
        pValue = self.calcPValue(generalG, expectedGeneralG, method="Getis-Ord General G")
        return generalG, pValue

    def calcSumWeightedMultipliedValues(self):
        sumWeightedMultipliedValues = 0
        for i in range(self.weightMatrix.shape[0]):
            for j in range(self.weightMatrix.shape[1]):
                if i != j:
                    sumWeightedMultipliedValues += self.weightMatrix[i, j] * self.value[i] * self.value[j]
        return sumWeightedMultipliedValues

    def calcSumMultipliedValues(self):
        sumMultipliedValues = 0
        for i in range(self.weightMatrix.shape[0]):
            for j in range(self.weightMatrix.shape[1]):
                if i != j:
                    sumMultipliedValues += self.value[i] * self.value[j]
        return sumMultipliedValues

    def calcExpectedGeneralG(self):
        sumOfWeightWithOutIEqualJ = 0
        for i in range(self.weightMatrix.shape[0]):
            for j in range(self.weightMatrix.shape[1]):
                if i != j:
                    sumOfWeightWithOutIEqualJ += self.weightMatrix[i, j]
        return sumOfWeightWithOutIEqualJ / (self.numberOfValues * (self.numberOfValues-1))

    def calcAnselinLocalMoransI(self):
        allLocalIs = np.zeros(self.numberOfValues)
        allZScores = np.zeros(self.numberOfValues)
        allPValues = np.zeros(self.numberOfValues)
        for i in range(self.numberOfValues):
            localI = self.calcLocalMoransI(i)
            allLocalIs[i] = localI
            zScore = self.calcLocalMoransIZScore(i, localI)
            pValue = self.convertZScoreToPValue(zScore)
            allZScores[i] = zScore
            allPValues[i] = pValue
        return allLocalIs, allZScores, allPValues

    def calcLocalMoransI(self, i):
        summedWeightedDifOfMean = self.calcSummedWeightedDifOfMean(i)
        SiSquared = self.calcSummedSquaredDifOfMean(i) / (self.numberOfValues - 1)
        Ii = (self.value[i]-self.meanValue) * summedWeightedDifOfMean / SiSquared
        return Ii

    def calcSummedSquaredDifOfMean(self, i):
        summedSquaredDifOfMean = 0
        for j in range(self.numberOfValues):
            if i != j:
                summedSquaredDifOfMean += (self.value[j] - self.meanValue)**2
        return summedSquaredDifOfMean

    def calcSummedWeightedDifOfMean(self, i):
        summedWeightedDifOfMean = 0
        for j in range(self.numberOfValues):
            if i != j:
                summedWeightedDifOfMean += self.weightMatrix[i, j] * (self.value[j] - self.meanValue)
        return summedWeightedDifOfMean

    def calcLocalMoransIZScore(self, i, localI):
        expectedLocalI = - (np.sum(self.weightMatrix[i, :]) - self.weightMatrix[i, i]) / (self.numberOfValues - 1)
        localMoransVar = self.calcLocalMoransVar(i, expectedLocalI)
        zScore = (localI - expectedLocalI) / np.sqrt(localMoransVar)
        return zScore

    def calcLocalMoransVar(self, i, expectedLocalI):
        summedSquaredWeights = self.calcSummedSquaredWeights(i)
        n = self.numberOfValues
        b2i = self.calcB2i(i)
        weightThing = self.calcWeightThing(i)
        A = (n-b2i) * summedSquaredWeights / (n - 1)
        B = (2*b2i - n) * weightThing / ((n-1) * (n-2))
        expectedsquaredLocalIi = A - B
        localMoransVar = expectedsquaredLocalIi - expectedLocalI**2
        return localMoransVar

    def calcSummedSquaredWeights(self, i):
        summedSquaredWeights = 0
        for j in range(self.numberOfValues):
            if i != j:
                summedSquaredWeights += self.weightMatrix[i, j]**2
        return summedSquaredWeights

    def calcB2i(self, i):
        numerator = 0
        denominator = 0
        for j in range(self.numberOfValues):
            if i != j:
                squaredDefOfMean = (self.value[j]-self.meanValue)**2
                numerator += squaredDefOfMean
                denominator += squaredDefOfMean**2
        denominator = denominator**2
        b2i = numerator/denominator
        return b2i

    def calcWeightThing(self, i):
        weightThing = 0
        for k in range(self.numberOfValues):
            if k != i:
                for h in range(self.numberOfValues):
                    if h != i:
                        weightThing += self.weightMatrix[i, k]*self.weightMatrix[i, h]
        return weightThing

    def calcGetisOrdGiStar(self):
        allGStars = np.zeros(self.numberOfValues)
        for i in range(self.numberOfValues):
            giStar = self.calcGiStar(i)
            allGStars[i] = giStar
        pValues = np.asarray([self.convertZScoreToPValue(zScore) for zScore in allGStars])
        return allGStars, pValues

    def calcGiStar(self, i):
        flattenedWeightOfRow = self.weightMatrix[i, :].flatten()
        summedWeightedByRowValue = np.sum(np.multiply(flattenedWeightOfRow, self.value))
        summedRowWeight = np.sum(self.weightMatrix[i, :])
        squaredRowOfWeight = np.multiply(flattenedWeightOfRow, flattenedWeightOfRow)
        summedsquaredRowWeight = np.sum(squaredRowOfWeight)
        S = np.sqrt(np.sum(self.value**2)/self.numberOfValues - self.meanValue**2)
        weightNormFactor = np.sqrt( (self.numberOfValues*summedsquaredRowWeight - summedRowWeight**2) /  (self.numberOfValues-1) )
        giStar = (summedWeightedByRowValue - self.meanValue*summedRowWeight) / (S * weightNormFactor)
        return giStar

    def GetMoransI(self):
        return self.moransI

    def GetGearysC(self):
        return self.gearysC

    def GetGetisOrdGeneralG(self):
        return self.getisOrdGeneralG

    def GetAnselinLocalMoransI(self):
        return self.anselinLocalMoransI

    def GetGetisOrdGiStar(self):
        return self.getisOrdGiStar

if __name__ == '__main__':
    main()
