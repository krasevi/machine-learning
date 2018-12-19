Grop project#62. Machine learning with Python.

Data classification with supervised learning based on a predefined data set uncluding a comparison of different algorithms. Case of 3 Datasets: Iris flowers, Ecoli and Leaves.

Assignment tasks:

1) Prepare for coding by installing libraries and finding data;
2) Test algorithms, visualise results and define the best algorithm;
3) Sum up the results and make suggestions for further work.

Project development discription:

Before start, all necessary libraries such as SciPy, Sklearn and FPDF were installed. The next step was retrieving datasets. We chose several datasets from The UCI Machine Learning Repository: Iris flowers, Ecoli and Leaves, which have several charasteristics in common. For example, three of them are multivariate and are based on real attributes. Similar attributes makes these datasets applicable for conducting classification tasks.
In order to have clean data, prepared for the research, we restructured it in accordance with the following order: first line is name vectors while first row is class vectors. All data used is numerical.
After loading the dataset and defining input and output paths, we moved to getting the initial feeling of the datasets. That means creation of several discriptive statistical graphs. First is a boxplot, based on the minimum, first quartile, median, third quartile, and maximum of data defined with every algorithm. The second and third explanatory graphs are scatter plot matrices, drawn with matplotlib and seaborn libraries. After some preparations such as random data splitting using the sklearn function train_test_split, spot checking of the algorithms and appending them into a list, we evaluated Supervised Learning Classification Methods in turn.
The final stage of our project was drawing a confusion matrix to visualise the results of the evaluation.

1. start machinelearning_vX.X.py
2. You might add your own .csv dataset to the folder following the instructions 
3. input the name of an existing dataset in the /data folder (e.g. iris.csv or iris)
4. choose the evaluation fraction that splits the dataset into training and test data (e.g. 0.3)
5. access results in the output folder - descriptions available in the folder
