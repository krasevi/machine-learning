Grop project#62. Machine learning with Python.

Data classification with supervised learning based on a predefined data set uncluding a comparison of different algorithms. Case of 3 Datasets: Iris flowers, Ecoli and Leaves.

Assignment tasks:

1) Prepare for coding by installing libraries and downloading data;
2) Understand the data, test algorithms and visualise results;
3) Sum up the results in a PDF-file.

Project development discription:

Before start, all necessary libraries such as Numpy, Matplotlyb.pyplot, Pandas, Seaborn, Sklearn and FPDF were installed. The next step was retrieving datasets. We chose several datasets from The UCI Machine Learning Repository: Iris flowers, Ecoli and Leaves, which have several charasteristics in common. For example, three of them are multivariate and are based on real attributes. Similar attributes make these datasets applicable for classification tasks while difference in samples' scale brings in value to our small research.
In order to have data, prepared for the research ("clean" data), we restructured it in accordance with the following order: first line represents name vectors while first row is class vectors. All data used is numerical.
After loading the datasets, we moved to getting the initial feeling of the datasets. We created several discriptive statistical graphs: boxplot and scatter plot matrices. After some preparations such as random data splitting using the sklearn function train_test_split, spot checking of the algorithms and appending them into a list, we evaluated Supervised Learning Classification Methods in turn.
The final stage of our project was drawing a confusion matrix to visualise the results of the evaluation.

1. start machinelearning_vX.X.py
2. You might add your own .csv dataset to the folder following the instructions 
3. input the name of an existing dataset in the /data folder (e.g. iris.csv or iris)
4. choose the evaluation fraction that splits the dataset into training and test data (e.g. 0.3)
5. access results in the output folder - descriptions available in the folder
