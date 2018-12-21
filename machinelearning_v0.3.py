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

dataset = pd.read_csv(input_path, delimiter=';',encoding='cp1252')
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

#make graph ticklables readable through rotation and fontsize
if numVars > 5:
	rotate=30
	fsize=8
elif numVars > 10:
	rotate=90
	fsize=6
else:
	rotate = 0
	fsize = 9

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(linestyle='dotted')
ax = sns.boxplot(data=dataset, palette="Set2")
plt.xticks(fontsize=fsize, rotation=rotate)
plt.tight_layout()
plt.savefig("output/boxplot.png")

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
messages = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	model_names.append(name)
	model.fit(X_train, Y_train)
	predictions.append(model.predict(X_validation))
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	messages.append(msg)
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
from fpdf import FPDF
# define texts for the PDF
title_1 = "Supervised Learning Classification of "+ filename + " dataset"
subtitle_1 = "Content:"
text_1 = "The following report shows a comparison of different supervised learning algorithms that classify the data of the " + filename + " dataset."
bullet_1 = " 1) Get an overview of dataset"
bullet_2 = " 2) Compare Classification algorithms"
#bullet_3 = " 3) Evaluate different Algorithms"
#bullet_4 = " 4) Show results"

subtitle_2 = "1) Get an overview of the dataset:"
text_2 = "The boxplot diagram shows an overview of the different features within the dataset and its attributes. On the X-Axis the different features of the"+filename+"dataset are represented. The Y-Axis shows how the attributes of the different attributes of the features. The boxplot is based on the minimum, first quartile, median, third quartile, and maximum."
text_3 = "The second graph is a scatterplot that shows how the feature attributes are distributed for the different classes. Classification algorithms should show better results if clusters can be recognized within the scatterplot."

subtitle_3 = "2) Compare classification algorithms"
text_4 = "Now that we got a feeling of how the dataset is distributed, a random split for training and evaluation data has been conducted. The following boxplot diagram shows the comparison of the different supervised learning algorithms:"
text_5 = "Finally, the confusion matrix shows a comparison of the predictions by the algorithms with the test data (Y-Axis) and the validation through the actual class labels (X-Axis). In the optimum all data is distributed on a diagonal line from the top left corner to the bottom right corner. This means that the predictions match the validation."
text_6 = "The accuracies of the different algorithms are as follows:"

bullet_5 = messages[0]
bullet_6 = messages[1]
bullet_7 = messages[2]
bullet_8 = messages[3]
bullet_9 = messages[4]
bullet_10 = messages[5]

# add texts and plots to the pdf
pdf = FPDF(format='A4')

pdf.add_page()
pdf.set_margins(20,20,10)

# document title
pdf.set_font("Times", "B", size=14)
pdf.cell(0, 10, txt=title_1, align="C", ln=1)

# document content
pdf.set_font("Times","I", size=12)
pdf.cell(0, 7, txt=subtitle_1, ln=2)

# bullet points of content
pdf.set_font("Times", size=12)
pdf.set_xy(70.00, 40.00)
pdf.cell(0,0,bullet_1,(pdf.get_x(), pdf.get_y()), ln=1)
pdf.set_xy(70.00, 47.00)
pdf.cell(0,0,bullet_2,(pdf.get_x(), pdf.get_y()), ln=1)
# pdf.set_xy(30.00, 52.00)
# pdf.cell(0,0,bullet_3,(pdf.get_x(), pdf.get_y()), ln=1)
# pdf.set_xy(30.00, 59.00)
# pdf.cell(0,0,bullet_4,(pdf.get_x(), pdf.get_y()), ln=1)

# introduction section
pdf.set_font("Times", size=12)
pdf.set_xy(20.00, 52.00)
pdf.multi_cell(0, 7, text_1,(pdf.get_x(), pdf.get_y()))

# boxplot description
pdf.set_font("Times","I", size=12)
pdf.cell(200, 7, txt=subtitle_2, ln=2)

pdf.set_font("Times", size=12)
pdf.set_xy(20.00, 75.00)
pdf.multi_cell(0, 7, text_2, (pdf.get_x(), pdf.get_y()))

#boxplot diagram

pdf.image("output/boxplot.png", 30, 110, 150, 150)

# new page
pdf.add_page()
pdf.set_margins(20,20,10)


# scatterplot and description

pdf.set_font("Times", size=12)
pdf.multi_cell(0, 7, txt=text_3)
pdf.image("output/scatterplot.png", 20, 70, 160, 160)

# new page
pdf.add_page()
pdf.set_margins(20,20,10)

#comparison of the different algorithms

pdf.set_font("Times","I", size=12)
pdf.cell(0, 7, txt=subtitle_3, ln=2)

pdf.set_font("Times", size=12)
pdf.set_xy(20.00, 30.00)
pdf.multi_cell(0,7,text_4,(pdf.get_x(), pdf.get_y()))
pdf.image("output/comparison.png", 60, 55, 100, 60)


# list of the accuracies of the different algorithms
pdf.set_xy(20.00, 120.00)
pdf.multi_cell(0,7,text_6,(pdf.get_x(), pdf.get_y()))
pdf.set_xy(70.00, 134.00)
pdf.cell(0,0,bullet_5,(pdf.get_x(), pdf.get_y()))
pdf.set_xy(70.00, 141.00)
pdf.cell(0,0,bullet_6,(pdf.get_x(), pdf.get_y()))
pdf.set_xy(70.00, 148.00)
pdf.cell(0,0,bullet_7,(pdf.get_x(), pdf.get_y()), ln=1)
pdf.set_xy(70.00, 155.00)
pdf.cell(0,0,bullet_8,(pdf.get_x(), pdf.get_y()), ln=1)
pdf.set_xy(70.00, 162.00)
pdf.cell(0,0,bullet_9,(pdf.get_x(), pdf.get_y()), ln=1)
pdf.set_xy(70.00, 170.00)
pdf.cell(0,0,bullet_10,(pdf.get_x(), pdf.get_y()), ln=1)

# draw confusion matrix

# only draw confusion matrix if not too much variables
if numVars > 9:
	# save pdf to output folder
	pdf.output(output_path)
else:
	pdf.set_xy(20.00, 175.00)
	pdf.multi_cell(0,7,text_5,(pdf.get_x(), pdf.get_y()))
	pdf.image("output/confusion_matrix.png", 40, 220, 120, 72)
	# save pdf to output folder
	pdf.output(output_path)
