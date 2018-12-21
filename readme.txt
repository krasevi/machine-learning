Group project#62. Machine learning with Python.

**About**
This is a student project of the university of St. Gallen of the course Advanced Programming.
The goal of the project was to create a supervised learning classifier by using existing datasets. 
This file was initially based on the tutorial https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
It is enhanced by different plots and visualizations using the seaborn library, the possibility to choose different datasets as input,
and a PDF generator to save the output and results of the classification in a pdf.

**Pre-requisites**
The program works with Python3
In order to run it, the following libraries need to be installed:
Numpy, Matplotlib, Seaborn, Sklearn, FPDF

**Instructions**
1. start machinelearning_vX.X.py
2. You might add your own .csv dataset to the folder following the instructions (see /data/readme.txt)
3. input the name of an existing dataset in the /data folder (e.g. iris.csv or iris)
4. choose the evaluation fraction that splits the dataset into training and test data (e.g. 0.3)
5. access results in the output folder (see /output/readme.txt)

**Files**
Code: /machinelearinig_vX.X.py
PDF part of the code(has been integrated): /PDF.py
Input datasets in csv: /data
Output folder containing all the plots and the PDF reports: /output


**Description**
Data classification with supervised learning based on a predefined dataset including a comparison of different algorithms. 
5 datasets available: Iris, Ecoli, Glass, Yeast and Leaves. Further datasets can be added but have to be cleansed (see /data.readme)

In the beginning, all necessary libraries such as Numpy, Matplotlib, Pandas, Seaborn, Sklearn and FPDF were installed 
The next step was retrieving datasets. We chose several datasets from The UCI Machine Learning Repository: Iris flowers, Ecoli and Leaves.
Three of the datasets are multivariate and are based on real attributes. Similar attributes make these datasets applicable for classification 
tasks while difference in samples' scale brings in value to our small research.
To prepare the data for the research it has to be cleansed and brought to a common format: 
the first line represents name vectors while first row is class vectors. All data used is numerical.

After loading the datasets, plots are being generated in order to get initial feeling of the dataset.
We created several discriptive statistical graphs: boxplot and scatter plot matrices.

Afterwards, the dataset is being randomly split in two parts. One for classification purposes and one for validating if the classification has been correct.
Hence, it is possible to evaluate which of the supervised learning algorithms that will be used afterwards performed with the highest accuracy.
, LinearDiscriminantAnalysis, KNeighborsClassifier, DecisionTreeClassifier, GaussianNB and SVC.
By using these algorithms and the dataset can be classified. A comparison of the classified data with the evaluation data is the basis to calculate the 
model accuracy. Finally, the accuracy was calculated and visualized by a confusion matrix. The confusion matrix shows on the x axis the identified classes
and on the y axis the correct class. In the best case, the identified class matches the evaluation class.

As an ouput, the program generates a PDF that contains all parts of the classification including the plots of the dataset, the accuracy of the different models
and the confusion matrix.

**Sources**
tutorial:https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
leaf: http://archive.ics.uci.edu/ml/datasets/Leaf
iris: http://archive.ics.uci.edu/ml/datasets/Iris
ecoli: http://archive.ics.uci.edu/ml/datasets/Ecoli
yeast: http://archive.ics.uci.edu/ml/datasets/Yeast
glass: http://archive.ics.uci.edu/ml/datasets/Glass+Identification


