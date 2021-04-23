import itertools
import numpy as np
import scipy.stats as st

class PermutationTester (object):

    def __init__(self, group1=None, group2=None, testAllPermutations=True):
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

def testBetweenAllGroups(valuesA, valuesB, groupNames):
    # print("testing between time points")
    valuesA = np.asarray(valuesA)
    valuesB = np.asarray(valuesB)
    assert len(valuesA) == len(valuesB), "Values from a and b don't have the same length. {} != {}".format(len(valuesA), len(valuesB))
    assert len(valuesA) == len(groupNames), "Values from a and the group names don't have the same length. {} != {}".format(len(valuesA),len(groupNames))
    uniqueGroup = np.unique(groupNames)
    nrOfGroups = len(uniqueGroup)
    for i in range(nrOfGroups):
        isGroup = np.isin(groupNames, uniqueGroup[i])
        valuesOfGroupA = valuesA[isGroup]
        valuesOfGroupB = valuesB[isGroup]
        if uniqueGroup[i] == "WT" or uniqueGroup[i] == "ktn 1-2":
            pass
            # pValue = np.round(PermutationTester(valuesOfGroupA, valuesOfGroupB).GetPValue(), 5)
            # print("{} {} (Permutation Test)".format(uniqueGroup[i], pValue))
        else:
            pValue = np.round(st.ttest_rel(valuesOfGroupA, valuesOfGroupB)[1], 5)
            print("{} {} (Paired t-Test)".format(uniqueGroup[i], pValue))

def mainTestGeneralG():
    wtGroup = ["WT"]*3
    claspGroup = ["clasp-1"]*6
    ktnGroup = ["ktn1-2"]*4
    groups = np.concatenate([wtGroup, claspGroup, ktnGroup])
    print(groups)
    diffOfGeneralG48To96 = [0.09507000000000002, 0.04144999999999999, 0.09253999999999998, 0.06472999999999998, 0.23108, 0.08562, 0.18853999999999999, 0.03681999999999999, 0.12769, 0.11214000000000002, 0.0626, 0.047060000000000005, 0.09837]
    for i in diffOfGeneralG48To96:
        print(i)

    print(np.round(np.mean(diffOfGeneralG48To96),5))
    print(np.round(np.std(diffOfGeneralG48To96),5))
    diffOfGeneralG48To96 = np.asarray(diffOfGeneralG48To96)
    uniqueGroupNames = np.unique(groups)
    nrOfUniqueGroups = len(uniqueGroupNames)
    pValue = np.round(st.ttest_ind(diffOfGeneralG48To96[groups=="clasp-1"], diffOfGeneralG48To96[groups=="ktn1-2"])[1], 5)
    print(" mean clasp", np.mean(diffOfGeneralG48To96[groups=="clasp-1"]))
    print(" mean ktn", np.mean(diffOfGeneralG48To96[groups=="ktn1-2"]))
    print("{} vs {} {} (Students t-Test)".format("clasp-1", "ktn1-2", pValue))
    for i in range(nrOfUniqueGroups):
        groupName1 = uniqueGroupNames[i]
        groupValues1 = diffOfGeneralG48To96[groups==groupName1]
        print(np.round(np.mean(groupValues1),5), np.round(np.std(groupValues1),5))
        # for j in range(i+1, nrOfUniqueGroups):
        #     groupName2 = uniqueGroupNames[j]
        #     groupValues2 = diffOfGeneralG48To96[groups==groupName2]
        #     myPermutationTester = PermutationTester(groupValues1, groupValues2)
        #     pValue = myPermutationTester.GetPValue()
        #     pValue = np.round(pValue, 5)
        #     print("{} vs {} {} (Permutation test)".format(groupName1, groupName2, pValue))

def main():
    import sys
    nameGroupes = np.asarray(["WT", "WT", "WT", "clasp-1", "clasp-1", "clasp-1", "clasp-1", "clasp-1", "clasp-1", "ktn 1-2", "ktn 1-2", "ktn 1-2", "ktn 1-2"])
    # assortativity48h = [0.03252, 0.03417, -0.31437, 0.3007, -0.21848, -0.55554, -0.06799, 0.42144, -0.18837, 0.04679, -0.01255, -0.42024, 0.19466]
    # assortativity96h = [-0.54924, -0.32995, -0.39764, -0.41646, -0.04932, -0.43788, -0.69624, 0.22621, -0.61895, -0.60791, 0.11014, -0.44243, 0.05615]
    # testBetweenAllGroups(assortativity48h, assortativity96h, nameGroupes)
    # moransI48h = [0.01791, -0.02308, -0.12266, 0.17327, -0.17325, -0.30827, -0.06495, 0.21634, -0.06393, 0.06448, 0.01175, -0.19662, 0.14585]
    # moransI96h = [-0.24277, -0.09501, -0.29551, -0.22859, -0.07227, -0.22043, -0.26036, 0.181, -0.15989, -0.29418, 0.01078, -0.22163, 0.13411]
    # testBetweenAllGroups(moransI48h, moransI96h, nameGroupes)
    # gearysC48h = [0.99868, 1.10357, 0.94796, 0.85405, 1.24379, 1.22112, 0.98526, 0.70222, 1.08488, 0.96115, 0.92633, 1.14646, 0.87915]
    # gearysC96h = [1.16192, 0.89513, 1.09315, 1.16225, 1.03724, 1.12954, 1.028, 0.96146, 0.81426, 1.16174, 0.77051, 1.14783, 0.87697]
    # testBetweenAllGroups(gearysC48h, gearysC96h, nameGroupes)
    # generalG48h = [0.15181, 0.13954, 0.18146, 0.15479, 0.10468, 0.15204, 0.17363, 0.1636, 0.20382, 0.16291, 0.09996, 0.13286, 0.10671]
    # generalG96h = [0.26182, 0.22227, 0.19876, 0.18554, 0.23872, 0.21661, 0.33291, 0.19592, 0.42223, 0.24624, 0.16173, 0.19679, 0.28291]
    # testBetweenAllGroups(generalG48h, generalG96h, nameGroupes)
    # sys.exit()
    # wtGroup = [-0.004542277962648819, 0.09626922631824217, 0.029273977949967576, 0.18166864366291502]
    # claspGroup = [0.17139794755936663, 0.06617869168660782, 0.15190779553534]
    generalG48h = [0.15734, 0.15779, 0.15434, 0.12895, 0.10213, 0.13908, 0.14682, 0.13577, 0.16244, 0.13412, 0.10334, 0.11713, 0.08635]
    generalG96h = [0.25241, 0.19924, 0.24688, 0.19368, 0.33321, 0.2247, 0.33536, 0.17259, 0.29013, 0.24626, 0.16594, 0.16419, 0.18472]
    generalG48h = np.asarray(generalG48h)
    generalG96h = np.asarray(generalG96h)
    group1 = generalG96h[nameGroupes=="clasp-1"]
    group2 = generalG96h[nameGroupes=="ktn 1-2"]
    myPermutationTester = PermutationTester(group1, group2)
    pValue = myPermutationTester.GetPValue()
    expectedHypothesis = myPermutationTester.GetExpectedHypothesis()
    difOfMeanBetweenGroups = myPermutationTester.GetDifOfMeanBetweenGroups()
    print(expectedHypothesis)
    print(difOfMeanBetweenGroups)
    print(pValue)

if __name__ == '__main__':
    # mainTestGeneralG()
    main()
