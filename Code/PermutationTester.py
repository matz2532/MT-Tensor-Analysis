import itertools
import numpy as np
import scipy.stats as st

class PermutationTester (object):

    def __init__(self, group1=None, group2=None, testAllPermutations=True):
        # ----- simple permutation tester, ATTENTION: don't use testAllPermutations=True, when more than a total of 18 values are used -----
        self.group1 = group1
        self.group2 = group2
        self.testAllPermutations = testAllPermutations
        if not self.group1 is None and not self.group2 is None:
            self.pValue = self.CalcPermutationTest()
        else:
            self.pValue = None
            self.extremerCases = None
            self.realDifOfMeans = None
            self.expectedHypothesis = None
            self.numberOfPermutations = 0

    def CalcPermutationTest(self, group1=None, group2=None, testAllPermutations=None, sizeThreshold=11):
        if not group1 is None:
            self.group1 = group1
        if not group2 is None:
            self.group2 = group2
        assert not self.group1 is None, "The group 1 is still None."
        assert not self.group2 is None, "The group 2 is still None."
        if testAllPermutations is None:
            testAllPermutations = self.testAllPermutations
        self.sizeGroup1 = len(self.group1)
        self.sizeGroup2 = len(self.group2)
        if self.sizeGroup1 + self.sizeGroup2 > sizeThreshold:
            print("applied students t-test as total number of samples is to big. {} > {}".format(self.sizeGroup1 + self.sizeGroup2, sizeThreshold))
            pValue = st.ttest_ind(self.group1, self.group2)[1]
        else:
            self.realDifOfMeans = self.calcDifOfMeansBetween(self.group1, self.group2)
            if testAllPermutations:
                pValue = self.calcPermutationTestOnAllPermutations()
        return pValue

    def calcDifOfMeansBetween(self, group1, group2):
        return np.mean(group1) - np.mean(group2)

    def calcPermutationTestOnAllPermutations(self, excludeRealPermutation=False):
        concatenatedGroups = np.concatenate([self.group1, self.group2])
        isRealDifNegative = self.realDifOfMeans < 0
        self.expectedHypothesis = "bigger or equal" if isRealDifNegative else "smaller or equal"
        self.isFirst = True
        self.extremerCases = 0
        self.numberOfPermutations = 0
        for permutation in itertools.permutations(concatenatedGroups):
            if not (self.isFirst and excludeRealPermutation):
                group1 = permutation[:self.sizeGroup1]
                group2 = permutation[self.sizeGroup1:]
                difOfMeans = self.calcDifOfMeansBetween(group1, group2)
                if isRealDifNegative:
                    if difOfMeans < self.realDifOfMeans:
                        self.extremerCases += 1
                else:
                    if difOfMeans > self.realDifOfMeans:
                        self.extremerCases += 1
                self.numberOfPermutations += 1
            if self.isFirst:
                self.isFirst = False
        pValue = self.extremerCases/self.numberOfPermutations
        return pValue

    def GetPValue(self):
        return self.pValue

    def GetDifOfMeanBetweenGroups(self):
        return self.realDifOfMeans

    def GetExtremerCases(self):
        return self.extremerCases

    def GetNumberOfPermutations(self):
        return self.numberOfPermutations

    def GetExpectedHypothesis(self):
        return self.expectedHypothesis

def main():
    possibleScenarios = ["WT", "clasp", "ktn"]
    firstScenario = possibleScenarios[0]
    secondScenario = possibleScenarios[0]
    testBetweenTimePoints = True
    test48hTimePoint = True # only used in case of: testBetweenTimePoints = False
    nameGroupes = np.asarray(["WT", "WT", "WT", "clasp-1", "clasp-1", "clasp-1", "clasp-1", "clasp-1", "clasp-1", "ktn 1-2", "ktn 1-2", "ktn 1-2", "ktn 1-2"])
    generalG48h = [0.15734, 0.15779, 0.15434, 0.12895, 0.10213, 0.13908, 0.14682, 0.13577, 0.16244, 0.13412, 0.10334, 0.11713, 0.08635]
    generalG96h = [0.25241, 0.19924, 0.24688, 0.19368, 0.33321, 0.2247, 0.33536, 0.17259, 0.29013, 0.24626, 0.16594, 0.16419, 0.18472]
    generalG48h = np.asarray(generalG48h)
    generalG96h = np.asarray(generalG96h)
    if testBetweenTimePoints:
        group1 = generalG48h[nameGroupes==firstScenario]
        group2 = generalG96h[nameGroupes==secondScenario]
    else:
        if test48hTimePoint:
            group1 = generalG48h[nameGroupes==firstScenario]
            group2 = generalG48h[nameGroupes==secondScenario]
        else:
            group1 = generalG96h[nameGroupes==firstScenario]
            group2 = generalG96h[nameGroupes==secondScenario]
    myPermutationTester = PermutationTester(group1, group2)
    pValue = myPermutationTester.GetPValue()
    expectedHypothesis = myPermutationTester.GetExpectedHypothesis()
    difOfMeanBetweenGroups = myPermutationTester.GetDifOfMeanBetweenGroups()
    print(expectedHypothesis)
    print(difOfMeanBetweenGroups)
    print(pValue)

if __name__ == '__main__':
    main()
