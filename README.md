# MT-Tensor-Analysis
Python script to analyze spatial correlation (Getis-Ord General G) of average microtubule (MT) changes in cotelydon pavement cells of Arabidopsis thaliana as used by Eng et al., Current Biology, 2021

These scripts were used to analyze spatial correlation of the average MT (mCHERRY-TUA5) changes between 15 second time steps over a 15 minute time intervall in 48h and 96h old cotyledons pavement cells from three different scenarios (wild-type, ktn1-2, and clasp-1).
In the FibrilJDataLoader.py script, the MT tensor input file ("FibrilJ Tensor/OVERVIEW_LOG_48h_CROP.xlsx" and "FibrilJ Tensor/OVERVIEW_LOG_96h.xlsx") as well as the corresponding adjacency lists (suffix: "_cell neighborhood 2D.csv") are used as input:
 - calculation of average MT change for each scenario and time point in the 15 minute time intervall
 - creation of the cellular connectivity network from the adjacency list for each tissue (as no cell division occured the same network was used for both time points)

In the SpatialAutoCorrelationCalculator.py script, the Getis-Ord General G spatial correlation is calculated using the network and average MT changes values as input.

In the PermutationTester.py script, the values of two groups are statistically tested applying permutation test between two groups.

In the SpatialTensorAnalyser.py script, the average MT changes as well as the networks (from FibrilJDataLoader.py) are used to group the same scenarios spatial correlation (SpatialAutoCorrelationCalculator.py), calculate their differences between the scenarios (PermutationTester.py), and save the results under "Results/FibrilJ Tensors/Spatial Autocorrelation Tables/angle/angle Getis-Ord General G 48.csv" or "angle Getis-Ord General G 96.csv".

In case of question, please do not hesitate to contact:
sampathkumar@mpimp-golm.mpg.de, t.matz@mpimp-golm.mpg.de, reng@mpipz.mpg.de
