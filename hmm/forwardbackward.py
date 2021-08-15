# This file will implement both a brute force method and 
# the forward-backward algorithm for computing the relevant
# probabilities to solve a part-of-speech tagging problem (HMM)

# Execute this code using the following command
# python forwardbackward.py toytrain.txt toy_index_to_word.txt toy_index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt predicted.txt metrics.txt
# python forwardbackward.py trainwords.txt index_to_word.txt index_to_tag.txt hmmprior_full_1000.txt hmmemit_full_1000.txt hmmtrans_full_1000.txt predicted.txt metrics.txt

import numpy as np
import csv
import sys
import itertools

def getRawData(file):
	with open(file) as csvfile:
  		reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
  		data = []
  		for row in reader:
			 data.append(row)
	return data

def genDict(file):
	d = dict()
	data = np.loadtxt(file,dtype='string')

	for i in range(len(data)):
			d[data[i]] = i
	
	return d

def splitData(data):
	ydata = []
	xdata = []
	
	for sentence in data:
		xi = []
		yi = []
		for word in sentence:
			temp = word.split('_')
			xi.append(temp)

		xdata.append(xi)

	return xdata

def stringToIndex(data,dic,parameterInd):

	stringToIndexData = data

	for sentInd in range(len(data)):
		for wordAndTagInd in range(len(data[sentInd])):
			wordAndTag = data[sentInd][wordAndTagInd]
			stringToIndexData[sentInd][wordAndTagInd][parameterInd] = dic.get(wordAndTag[parameterInd])
	
	return(stringToIndexData)

def separateData(data):
	words = []
	tags = []

	for sentence in data:
		sentJustWords = []
		sentJustTags = []

		for wordTagPair in sentence:
			sentJustWords.append(wordTagPair[0])
			sentJustTags.append(wordTagPair[1])

		words.append(sentJustWords)
		tags.append(sentJustTags)

	return words, tags

# Reimplement this so that transition and emission matrices are in log space (avoid underflow).
def getAlphaMatrix(initProb, transProb, emitProb, indexToWord, indexToTag, testData):

	numOfTags = len(indexToTag.values())
	numOfWords = len(testData)

	# t = word index
	# j = tag index
	# k = inner tag index

	alphaMatrix = np.zeros([numOfTags,numOfWords],dtype=float)

	# Forward Algorithm
	for t in range(numOfWords):
		# print(t)
		for j in range(numOfTags):
			if t==0:
				x_0 = testData[t][0]

				alphaMatrix[j,t] = initProb[j]*emitProb[j,x_0]
			else:
				x_t = testData[t][0]
				for k in range(numOfTags):

					alphaMatrix[j,t] = alphaMatrix[j,t] + emitProb[j,x_t]*alphaMatrix[k,t-1]*transProb[k,j]
	
	return alphaMatrix

# Reimplement this so that transition and emission matrices are in log space (avoid underflow).
def getBetaMatrix(initProb, transProb, emitProb, indexToWord, indexToTag, testData):

	numOfTags = len(indexToTag.values())
	# numOfWords = len(indexToWord.values())
	numOfWords = len(testData)
	# t = word index
	# j = tag index
	# k = inner tag index

	betaMatrix = np.zeros([numOfTags,numOfWords],dtype=float)

	
	betaMatrix[:,numOfWords-1] = 1

	# Backward Algorithm [::-1]
	for t in range(numOfWords)[::-1]:
		for j in range(numOfTags):
			if t < (numOfWords-1):
				x_tplus1 = testData[t+1][0]
				for k in range(numOfTags):

					betaMatrix[j,t] = betaMatrix[j,t] + emitProb[k,x_tplus1]*betaMatrix[k,t+1]*transProb[j,k]

			else:
				betaMatrix[j,t] = 1.0
				

	return betaMatrix


if __name__ == "__main__":
	test_input = sys.argv[1]
	index_to_word = sys.argv[2]
	index_to_tag = sys.argv[3]
	hmmprior = sys.argv[4]
	hmmemit = sys.argv[5]
	hmmtrans = sys.argv[6]
	predicted = sys.argv[7]
	metrics = sys.argv[8]

	testInput = getRawData(test_input)

	indexToWord = genDict(index_to_word)
	indexToTag = genDict(index_to_tag)

	# Split data into distinguishable word-tag pairs (without delimiter)
	newTestData = splitData(testInput)
	testData = newTestData[0]

	# Index the hidden Markov state by 0 and the observation (tag) by 1
	stateIndex = 0
	obsIndex = 1

	# Map all words to indices as given in the index_to_word and index_to_tag files
	wordToIndex = stringToIndex(newTestData,indexToWord,stateIndex)
	testDataToIndex = stringToIndex(wordToIndex,indexToTag,obsIndex)

	wordsAndTags = separateData(testDataToIndex)
	wordsTestIndex = wordsAndTags[0][0]
	tagsTestIndex = wordsAndTags[1][0]
	numOfTrainEx = len(newTestData)

	# Load in the initialization probabilities
	initProb = np.loadtxt(hmmprior)

	# Load in the transition probabilities
	transProb = np.loadtxt(hmmtrans)

	# Load in the emission probabilities
	emitProb = np.loadtxt(hmmemit)

	# Do prediction on the test data and compute the average log likelihood
	predictedLabels = []
	logLikelihood = []

	for testSentence in testDataToIndex:

		alpha = getAlphaMatrix(initProb, transProb, emitProb, indexToWord, indexToTag, testSentence)
		beta = getBetaMatrix(initProb, transProb, emitProb, indexToWord, indexToTag, testSentence)
		alphaSum = np.log(np.sum(alpha[:,-1]))
		logLikelihood.append(alphaSum)

		wordTagPairs = []
		for t in range(len(testSentence)):
			predictionProb = alpha[:,t]*beta[:,t]
			word = testSentence[t][0]
			y_that = np.argmax(predictionProb)
			tag = y_that
			wordTagPairs.append([word,tag])

		predictedLabels.append(wordTagPairs)

	avgLogLikelihood = np.average(logLikelihood)

	count = 0
	total = 0

	for i in range(len(predictedLabels)):
		for j in range(len(predictedLabels[i])):
			total = total + 1
			if predictedLabels[i][j][1] == testDataToIndex[i][j][1]:
				count = count + 1


	accuracy = float(count) / float(total)

	# Convert numerical assignments back to string assignments for printing
	backToWords = dict((value,key) for key,value in indexToWord.iteritems())
	backToTags = dict((value,key) for key,value in indexToTag.iteritems())

	predictedLabelsOut = []

	for sentence in predictedLabels:
		newSentence = []
		for wordTagPair in sentence:
			word = wordTagPair[0]
			tag = wordTagPair[1]
			word = backToWords.get(word)
			tag = backToTags.get(tag)

			newSentence.append([word,tag])

		predictedLabelsOut.append(newSentence)

	# Write the predicted labels to a text file
	predictedOut = open(predicted,'w')

	for sentence in predictedLabelsOut:
		sentLength = len(sentence)

		for wordTagPairInd in range(sentLength):

			predictedOut.write(str(sentence[wordTagPairInd][0]))
			predictedOut.write(str('_'))
			predictedOut.write(str(sentence[wordTagPairInd][1]))

			if wordTagPairInd != sentLength-1:
				predictedOut.write(str(' '))

		predictedOut.write(str('\n'))

	predictedOut.close()


	# Write the metrics to a text file
	metricsOut = open(metrics,'w')

	metricsOut.write(str('Average Log-Likelihood: '))
	metricsOut.write('{0:11f}'.format(avgLogLikelihood))
	metricsOut.write(str('\n'))
	metricsOut.write(str('Accuracy: %0.12f') % accuracy)

	metricsOut.close()

		








