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
if numVars > 15:
	# save pdf to output folder
	pdf.output(output_path)
else:
	pdf.set_xy(20.00, 175.00)
	pdf.multi_cell(0,7,text_5,(pdf.get_x(), pdf.get_y()))
	pdf.image("output/confusion_matrix.png", 40, 220, 120, 72)
	# save pdf to output folder
	pdf.output(output_path)
