# Machine learning
# An introduction into supervised learning respectively classification methods
# including a comparison of different algorithms

# This file was initially based on the tutorial:
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# it is enhanced by nice seaborn plots :)
# the possibility to work with different datasets that can be chosen
# additional output data and confusion matrix to compare the algorithms
# pdf printing method of the results

# ______________________________________________________________________________
# 1. Preparation

# Import libraries
# import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
# from pandas.plotting import scatter_matrix

# import sklearn
from sklearn import model_selection
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from fpdf import FPDF

# datasets have been taken from:

# url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# url="http://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
# add further urls***

# in order to have clean data and to make them interchangable, data have
# been re-ordered with the stucture:
# - first line contains names vector
# - first row contains class vector
# - only numerical data is applicable

# ______________________________________________________________________________
# 2. Loading the dataset

# Load dataset
# Check whether file exists and define input path

while True:
	# Enter the inputdata filename (e.g. iris.csv)
	filename = input("Enter dataset filename\n> ")
	my_file = Path("data/"+filename)
	if my_file.is_file():
    	# file exists
		input_path="data/"+filename
		break
	else:
 		# maybe .csv ending was being forgotten (e.g. iris)
		my_file = Path("data/"+filename+".csv")
		if my_file.is_file():
	    	# file exists
			input_path="data/"+filename+".csv"
			break
		else:
			print("""error file does not exist in program path
			\nplease enter a valid [.csv] file such as 'iris.csv'!\n""")

# define output path
output_path="output/"+filename.split('.csv')[0]+".pdf"


# Enter evaluation fraction of the dataset
validationSize = float(input("Enter the evaluation fraction of the data as floating point number\n> "))

dataset = pd.read_csv(input_path, delimiter=';')
names = list(dataset.columns)
numVars = len(names)-1

# split datasets to separate values from class
array = dataset.values
data = array[:, 1:numVars]
clas = array[:, 0]

class_names = np.unique(np.array(clas))

# ______________________________________________________________________________
# 3. Get a feeling for the data set

# shape of the dataset
print(dataset.shape)

# sneak-peak into dataset
print(pd.DataFrame(dataset).head(5))

# sneak peak of the first few lines
print(dataset.head(20))

# show the descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# Show boxplot of dataset
# prepare plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(linestyle='dotted')
ax = sns.boxplot(data=dataset, palette="Set2")
plt.savefig("output/boxplot.png")

# Show histograms of dataset (not working)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.grid(linestyle='dotted')
# ax = sns.distplot(a=data)

# Show scatter plot matrix with matplotlib (disabled)
# scatter_matrix(dataset)
# plt.savefig("output/scatterplot1.png")

# Show Scatter plot matrix with seaborn
# plt.figure(figsize=(10, 10))
sns.pairplot(dataset, diag_kind='hist', hue='class')
plt.savefig("output/scatterplot.png")

# ______________________________________________________________________________
# 4. Preparing the evaluation

# randomly split data using the sklearn function train_test_split
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(data, clas, test_size=validationSize)

# Test options and evaluation metric
scoring = 'accuracy'

# Spot Check Algorithms
# append Machine Learning algorithms in a list
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# ______________________________________________________________________________
# 5. Evaluate the different Supervised Learning Classification Methods

# evaluate each model in turn
results = []
model_names = []
predictions = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	model_names.append(name)
	model.fit(X_train, Y_train)
	predictions.append(model.predict(X_validation))
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(model_names)
plt.savefig("output/comparison.png")


# draw confusion matrix for all the different Algorithms
fig = plt.figure()
i = 0
# loop for number of vertical plots
for y in range(2):
	# loop for number of horizontal plots
	for x in range(3):
		i=(y*3)+x
		mat = confusion_matrix(Y_validation, predictions[i])
		ax = fig.add_subplot(2,3,i+1,sharex=ax,sharey=ax)
		ax.grid(linestyle='dotted')
		ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='BuGn_r', xticklabels=class_names, yticklabels=class_names)
		ax.set_title(model_names[i]+" (k="+str(len(class_names))+")")
		ax.set_xlabel("validations")
		ax.set_ylabel("predictions")
		# make inner labels invisible in order to view the data easier
		if y == 0:
			plt.setp(ax.get_xticklabels(), visible=False)
			ax.set_xlabel("")
		if x != 0:
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.set_ylabel("")
		i+=1

plt.tight_layout()
plt.savefig("output/confusion_matrix.png")

# ______________________________________________________________________________
# 6. Print All the results into a PDF document

# print PDF

# define texts for the PDF
text_1 = "Grop project#62. Machine learning with Python."
text_2 = "Assignment overview"
text_3 = "Data classification with supervised learning based on a predefined data set uncluding a comparison of different algorithms. Case of 3 Datasets: Iris flowers, Ecoli and "
text_4 = "Assignment tasks:"
text_5 = "Project development discription"
text_6 = "Before starting, all necessary libraries such as SciPy, Sklearn and FPDF were installed. The next step was retrieving datasets. We chose several datasets from The UCI Machine Learning Repository: Iris flowers, Ecoli and , which have several charasteristics in common. For example, three of them are multivariate and are based on real attributes. Similar attributes makes these datasets applicable for conducting classification tasks. In order to have clean data, prepared for the research, we restructured it in accordance with the following order: first line is name vectors while first row is class vectors. All data used is numerical."
text_7 = "After loading the dataset and defining input and output paths, we moved to getting the initial feeling of the datasets. That means creation of several discriptive statistical graphs."
text_8 = "First is a boxplot, based on the minimum, first quartile, median, third quartile, and maximum of every algorithm. As it can be seen, "
text_9 = "The second and third explanatory graphs are scatter plot matrices, drawn with matplotlib and seaborn.  "
text_10 = "After some preparations such as random data splitting using the sklearn function train_test_split, spot checking of the algorithms and appending them into a list, we evaluated the different Supervised Learning Classification Methods in turn. The comparison of the results can be seen on the picture below:"
text_11 = "The final stage of our project was drawing a confusion matrix for all the different algorithms."
text_12 = "Results showed, that..."

# add texts and plots to the pdf

from fpdf import FPDF
pdf = FPDF(format='A4')
pdf.set_margins(20,20,10)
pdf.add_page()
pdf.set_font("Times", "B", size=14)
pdf.cell(0, 10, txt=text_1, align="C", ln=1)
pdf.set_font("Times","I", size=12)
pdf.cell(0, 7, txt=text_2, ln=2)
pdf.set_font("Times", size=12)
pdf.multi_cell(0, 7, txt=text_3)
pdf.set_font("Times","I", size=12)
pdf.cell(0, 7, txt=text_4)
pdf.set_font("Times", size=12)
pdf.set_xy(70.00, 57.00)
pdf.cell(0,0,'1) Prepare for coding by installing libraries and finding data',(pdf.get_x(), pdf.get_y()))
pdf.set_xy(70.00, 64.00)
pdf.cell(0,0,'2) Test algorithms, visualise results and define the best algorithm',(pdf.get_x(), pdf.get_y()))
pdf.set_xy(70.00, 71.00)
pdf.cell(0,0,'3) Test algorithms, visualise results and define the best algorithm',(pdf.get_x(), pdf.get_y()), ln=1)
pdf.set_xy(70.00, 78.00)
pdf.cell(0,0,'4) Sum up the results and make suggestions for further work',(pdf.get_x(), pdf.get_y()), ln=1)
pdf.set_font("Times","I", size=12)
pdf.cell(0, 10, txt=text_5, ln=1)
pdf.set_font("Times", size=12)
pdf.multi_cell(0, 7, txt=text_6)
pdf.multi_cell(0, 7, txt=text_7)

pdf.set_font("Times", size=12)
pdf.multi_cell(0, 7, txt=text_8)
#pdf.rect(60, 170, 100, 60)
pdf.image("output/boxplot.png", 60, 170, 100, 60)

pdf.add_page()
pdf.set_margins(20,20,10)
pdf.set_font("Times", size=12)
pdf.multi_cell(0, 7, txt=text_9)
#pdf.rect(20, 30, 80, 60)
pdf.image("output/scatterplot.png", 20, 30, 80, 60)
#pdf.rect(110, 30, 80, 60)
#pdf.image("output/scatterplot1.png", 110, 30, 80, 60)
pdf.set_xy(20.00, 100.00)
pdf.multi_cell(0,7,text_10,(pdf.get_x(), pdf.get_y()))
pdf.image("output/scatterplot.png", 20, 30, 80, 60)
#pdf.rect(60, 125, 100, 60)
pdf.image("output/comparison.png", 60, 130, 100, 60)

pdf.set_xy(20.00, 190.00)
pdf.multi_cell(0,7,text_11,(pdf.get_x(), pdf.get_y()))
#pdf.rect(60, 200, 100, 60)
pdf.image("output/confusion_matrix.png", 60, 210, 100, 60)
pdf.set_xy(20.00, 265.00)
pdf.multi_cell(0,7,text_12,(pdf.get_x(), pdf.get_y()))

# save pdf to output folder
pdf.output(output_path)
