import csv
import sys
import numpy as np
import time
from copy import copy

class NeuralNetworkData:
	def __init__(self):
		self.x = np.array([]) # input data
		self.a = np.array([]) # linear combination of weights at hidden layer
		self.z = np.array([]) # activation at hidden layer (node values for hidden layer)
		self.b = np.array([]) # linear combination of weights at output layer
		self.yhat = np.array([]) # activation at output layer (softmax)
		self.J = 0 # cross-entropy loss

def SGD(trainingData, testData,inputs):

	# M = SIZE OF TRAINING DATA VECTOR x
	# D = SIZE OF HIDDEN LAYER VECTOR z
	# K = NUMBER OF POSSIBLE CLASSES
	init_flag = int(inputs[7])
	num_epoch = int(inputs[5])
	learning_rate = float(inputs[-1])
	metrics_out = inputs[4]

	M = np.size(trainingData[0,:])-1; # -1 because of label element
	D = int(inputs[6])
	K = 10
	N = float(np.size(trainingData[:,0]))
	NT = float(np.size(testData[:,0]))

	# Determine weight initialization
	if init_flag == 1:
		# Initialize all weights to random
		alpha = np.random.uniform(-0.1,0.1,(D,M+1)) # M+1 due to bias
		beta = np.random.uniform(-0.1,0.1,(K,D+1)) # D+1 due to bias
		alpha[:,0] = 0
		beta[:,0] = 0

	if init_flag == 2:
		# Initialize all weights to zero
		alpha = np.zeros((D,M+1),dtype=float) # M+1 due to bias
		beta = np.zeros((K,D+1),dtype=float) # D+1 due to bias

	metricsOut = open(metrics_out,'w')

	for e in range(num_epoch):
		
		for data in copy(trainingData):
			# Organize data
			xdata = data
			oneHotVar = xdata[0]
			xdata[0] = 1 # Add in bias of 1
			ydata = getOneHotVec(oneHotVar, K)
			
			# Compute neural network layers
			nn = NNForward(xdata,ydata,alpha,beta)

			# Compute gradients via backpropagation
			[g_alpha,g_beta] = NNBackward(xdata,ydata,alpha,beta,nn)

			# g_alpha_check = finiteDiffCheckAlpha(xdata,ydata,alpha,beta)
			# g_beta_check = finiteDiffCheckBeta(xdata,ydata,alpha,beta)

			# Update parameters
			alpha = alpha - learning_rate*g_alpha
			beta = beta - learning_rate*g_beta
			

		JD = meanCrossEntropy(copy(trainingData),alpha,beta,N,K)
		JDT = meanCrossEntropy(copy(testData),alpha,beta,NT,K)

		metricsOut.write('epoch=')
		metricsOut.write(str(e+1))
		metricsOut.write(' crossentropy(train): ')
		metricsOut.write(str(JD))
		metricsOut.write(str('\n'))
		
		metricsOut.write('epoch=')
		metricsOut.write(str(e+1))
		metricsOut.write(' crossentropy(test): ')
		metricsOut.write(str(JDT))
		metricsOut.write(str('\n'))

	errorTrain = predict(copy(trainingData),alpha,beta,N,K)[1]
	errorTest = predict(copy(testData),alpha,beta,NT,K)[1]

	metricsOut.write('error (train): ')
	metricsOut.write(str(errorTrain))
	metricsOut.write(str('\n'))
	metricsOut.write('error (test): ')
	metricsOut.write(str(errorTest))
	metricsOut.write(str('\n'))

	metricsOut.close()

	return(alpha,beta)

# Make a forward pass through the neural network
def NNForward(x,y,alpha,beta):

	nn_data = NeuralNetworkData() # intermediate quantities

	nn_data.a = np.dot(alpha,x[:,None]) # Compute linear combination at input
	nn_data.z = sigma(nn_data.a) # Compute sigmoid activation at hidden layer nodes
	nn_data.z = np.insert(nn_data.z,0,1) # Insert bias term of z_0 = 1
	nn_data.b = np.dot(beta,nn_data.z[:,None]) # Compute linear combination at hidden layer
	nn_data.yhat = softmax(nn_data.b) # Compute softmax activation at output
	nn_data.J = crossEntropy(nn_data.yhat,y) # Compute cross-entropy loss

	return nn_data

# Do backpropagation on the neural network
def NNBackward(x,y,alpha,beta,nn_data):

	# nn_data has x, a, z, b, yhat, and J
	gyhat = -np.divide(y[:,None],nn_data.yhat)
	gb = nn_data.yhat - y[:,None]
	gbeta = np.dot(gb, nn_data.z[:,None].T)
	gz = np.dot(beta.T, gb)
	gz = np.delete(gz, 0, 0)
	nn_data.z = np.delete(nn_data.z, 0, 0)
	ga1 = nn_data.z[:,None]*(1 - nn_data.z[:,None])
	ga = gz*ga1
	galpha = np.dot(ga, x[:,None].T)

	return galpha, gbeta

def crossEntropy(yhat,y):
	J = -np.sum(y[:,None]*np.log(yhat))
	return J

def sigma(a):
	Z = 1/(1 + np.exp(-a))
	return Z

def softmax(b):
	Z = np.sum(np.exp(b))
	yhat = np.exp(b)/Z
	return yhat

def getRawData(file):
	with open(file) as tsvfile:
  		reader = csv.reader(tsvfile, delimiter=',')
  		data= []
  		for row in reader:
			data.append(row)

	return data

def getOneHotVec(label, K):
	oneHot = np.zeros(K)
	oneHot[int(label)] = 1
	return oneHot

def finiteDiffCheckAlpha(x,y,alpha,beta):
	
	epsilon = 1e-5
	
	grad = np.zeros(np.shape(alpha))

	for m in range(len(alpha[:,0])):

		d = np.zeros(len(alpha[:,0]))
		d[m] = 1
		d = d.reshape(np.size(d),1)
		nnf = NNForward(x, y, alpha + epsilon * d, beta)
		nnb = NNForward(x, y, alpha - epsilon * d, beta)
		v = (nnf.J - nnb.J) / (2*epsilon)

	return(grad)

def finiteDiffCheckBeta(x,y,alpha,beta):
	
	epsilon = 1e-5
	
	grad = np.zeros(np.shape(beta))

	for m in range(len(beta[:,0])):

		d = np.zeros(len(beta[:,0]))
		d[m] = 1
		d = d.reshape(np.size(d),1)
		nnf = NNForward(x, y, alpha, beta + epsilon*d)
		nnb = NNForward(x, y, alpha, beta - epsilon*d)
		v = (nnf.J - nnb.J) / (2*epsilon)
		# print(v)
		# grad[m,:] = v

	return(grad)

def meanCrossEntropy(dataset,alpha,beta,N,K):

	J = 0
	for data in copy(dataset):
		# Organize data
		xdata = data
		oneHotVar = xdata[0]
		xdata[0] = 1
		# xdata = xdata.reshape(np.size(xdata),1)
		ydata = getOneHotVec(oneHotVar, K)
		
		# Compute neural network layers
		nn1 = NNForward(xdata,ydata,alpha,beta)
		J = J + nn1.J
	
	JD = J / N # + (0.01/2)*np.linalg.norm(alpha) + (0.01/2)*np.linalg.norm(beta)

	return(JD)

def predict(dataset, alpha, beta, N, K):

	c = 0
	predictions = []
	for data in copy(dataset):
		# Organize data
		xdata = data
		oneHotVar = int(xdata[0])
		xdata[0] = 1
		ydata = getOneHotVec(oneHotVar, K)
			
		# Compute the neural network prediction
		nn = NNForward(xdata,ydata,alpha,beta)

		prediction = np.argmax(nn.yhat)

		predictions.append(prediction)

		if prediction == oneHotVar:
			c = c + 1

	error = 1.0 - c/N

	return(predictions, error)


if __name__ == "__main__":
	# Get all command line inputs
	train_input = sys.argv[1]
	test_input = sys.argv[2]
	train_out = sys.argv[3]
	test_out = sys.argv[4]
	metrics_out = sys.argv[5]
	num_epoch = sys.argv[6]
	hidden_units = sys.argv[7]
	init_flag = sys.argv[8]
	learning_rate = sys.argv[9]

	inputs = [train_input,test_input,train_out,test_out,metrics_out,num_epoch,hidden_units,init_flag,learning_rate];

	# Process raw data
	train_data = getRawData(train_input)
	train_data = np.array(train_data,dtype=float)

	test_data = getRawData(test_input)
	test_data = np.array(test_data,dtype=float)

	# Train using stochastic gradient descent
	[alpha_train,beta_train] = SGD(train_data,test_data,inputs)

	K = 10
	N = float(np.size(train_data[:,0]))
	NT = float(np.size(test_data[:,0]))

	# Make predictions on the training data
	predictions = predict(train_data,alpha_train,beta_train,N,K)[0]

	trainOut = open(train_out,'w')

	for i in predictions:
		trainOut.write(str(i))
		trainOut.write(str('\n'))

	trainOut.close()

	# Make predictions on the test data
	predictionsTest = predict(test_data,alpha_train,beta_train,NT,K)[0]

	testOut = open(test_out,'w')

	for i in predictionsTest:
		testOut.write(str(i))
		testOut.write(str('\n'))

	testOut.close()