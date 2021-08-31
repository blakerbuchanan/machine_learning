import csv
import sys
import numpy as np
import time
from numpy import linalg as LA

'''
	To run binary logistic regression on the resulting formatted data, run
	python3 lr.py formatted_train.tsv formatted_valid.tsv formatted_test.tsv dict.txt smalltrain_out.labels smalltest_out.labels smallmetrics_out.txt 60
'''
#-------------------------------------------------------------#
#--------------- BEGIN FUNCTION DEFINITIONS ------------------#
#-------------------------------------------------------------#

# Efficient computation of dot product (working)
def effDotProduct(theta,x):

	S = 0.0
	# Use thetas corresponding to the indices given in the feature
	# vector to compute the dot product
	for i in range(len(x)):
		xi = int(x[i][0]) # index of ith (as indexed in this loop) word in dictionary
		xv = float(x[i][1]) # value of the ith (as indexed in this loop) word in feature vector
		S = S + theta[xi]*float(xv)

	return S

# Compute the negative conditional log-likelihood for data x, y, and features theta
def getLogLikelihood(x,y,theta):
	J = 0
	for i in range(len(x[0])):
		d = effDotProduct(theta,x[i])
		J = J + (-y[i]*d + np.log(1+np.exp(d)))

	return J

# SGD Update
def singleSGDStep(theta,eta,xDataSingle,yDataSingle):
	
	G = np.zeros(len(theta))

	# Compute gradient
	d = effDotProduct(theta,xDataSingle)

	# This is the logistic regression part
	R = float(np.exp(d))/float((1.0 + np.exp(d)))

	for j in xDataSingle:
		G[j[0]] = j[1] * ( yDataSingle - R )
		theta[j[0]] = theta[j[0]] + eta*G[j[0]]

	return theta

def SGD(xdata,ydata,theta0):

	# Initialize theta
	theta = theta0

	# Define step size for SGD
	eta = 0.1

	# Do SGD on the data
	for i in range(len(xdata)):
		theta = singleSGDStep(theta,eta,xdata[i],ydata[i])

	return theta

def getRawData(file):
	with open(file) as tsvfile:
			reader = csv.reader(tsvfile, delimiter='\t')
			data = []
			for row in reader:
			 data.append(row)

	return data

if __name__ == "__main__":
	#---------------------------------------------#
	#--------------- BEGIN MAIN ------------------#
	#---------------------------------------------#
	
	# Get inputs from command line
	formattedTrain = sys.argv[1]
	formattedValid = sys.argv[2]
	formattedTest = sys.argv[3]
	dictInput = sys.argv[4]
	trainOutLabels = sys.argv[5]
	testOutLabels = sys.argv[6]
	metricsOut = sys.argv[7]
	numOfEpochs = sys.argv[8]

	# Format the data for stochastic gradient descent
	trainData = getRawData(formattedTrain)
	validData = getRawData(formattedValid)
	testData = getRawData(formattedTest)

	yTrainData = []
	xTrainData = []
	dictList = getRawData(dictInput)
	dictListSize = len(dictList)
	theta0 = np.zeros(dictListSize + 1)

	# Adjust feature data a bit more...
	for i in range(len(trainData)):
		yTrainData.append(int(trainData[i][0]))
		xi = []
		xi.append([0,1])

		# minus one because I don't want the label
		for j in range(len(trainData[i][1:len(trainData[i])])-1):
			# plus one because I don't want the label
			xi_current = str(trainData[i][j+1]).split(':')
			xi_current = [int(entry) for entry in xi_current]
			# xi_current = np.array(xi_current)
			xi.append(xi_current)

		xTrainData.append(xi)

	xTrainData = np.array(xTrainData)

	# Initialize theta
	theta = theta0

	#-----------------------------------------#
	# BEGIN STOCHASTIC GRADIENT DESCENT
	#-----------------------------------------#

	# Do SGD for numOfEpochs
	for n in range(int(numOfEpochs)):
		# Full SGD through the training data
		theta = SGD(xTrainData,yTrainData,theta)

	#-----------------------------------------#
	# BEGIN PREDICTION ON TRAINING DATA
	#-----------------------------------------#

	trainLabelFile = open(trainOutLabels,"w")
	metricsOutFile = open(metricsOut,"w")

	trainCount = 0
	trainTotal = len(trainData)

	for i in range(len(trainData)):

		prob1 = 1.0/(1.0 + np.exp(-effDotProduct(theta,xTrainData[i])))
		prob0 = 1.0 - prob1

		if prob1 > prob0:
			prediction = 1
		else:
			prediction = 0

		if yTrainData[i] == prediction:
			trainCount = trainCount + 1
		
		trainLabelFile.write(str(prediction))
		trainLabelFile.write(str('\n'))

	trainAccuracy = float(trainCount) / float(trainTotal)
	trainError = 1 - trainAccuracy
	metricsOutFile.write(str('error(train): ') + str(trainError))

	trainLabelFile.close()

	#-----------------------------------------#
	# BEGIN PREDICTION ON TEST DATA
	#-----------------------------------------#
	xTestData = []
	yTestData = []
	testCount = 0
	testTotal = len(testData)

	# Adjust feature data a bit more...
	for i in range(len(testData)):
		yTestData.append(int(testData[i][0]))
		xi = []
		xi.append([0,1])

		# minus one because I don't want the label
		for j in range(len(testData[i][1:len(testData[i])])-1):
			# plus one because I don't want the label
			xi_current = str(testData[i][j+1]).split(':')
			xi_current = [int(entry) for entry in xi_current]
			# xi_current = np.array(xi_current)
			xi.append(xi_current)

		xTestData.append(xi)

	xTestData = np.array(xTestData)

	testLabelFile = open(testOutLabels,"w")

	for i in range(len(testData)):

		prob1 = 1.0/(1.0 + np.exp(-effDotProduct(theta,xTestData[i])))
		prob0 = 1.0 - prob1

		if prob1 > prob0:
			prediction = 1
		else:
			prediction = 0

		if yTestData[i] == prediction:
			testCount = testCount + 1

		testLabelFile.write(str(prediction))
		testLabelFile.write(str('\n'))

	testAccuracy = float(testCount) / float(testTotal)
	testError = 1 - testAccuracy

	metricsOutFile.write(str('\n'))
	metricsOutFile.write(str('error(test): ') + str(testError))

	trainLabelFile.close()
	metricsOutFile.close()


