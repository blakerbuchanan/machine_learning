# This function reads in data and computes hidden
# Markov model parameters associated with the training data
 
# Execute this code with the following command
# Toy data set: python learnhmm.py toytrain.txt toy_index_to_word.txt toy_index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt
# 1000-word data set: python learnhmm.py trainwords.txt index_to_word.txt index_to_tag.txt hmmprior_full_1000.txt hmmemit_full_1000.txt hmmtrans_full_1000.txt

import numpy as np
import csv
import sys

def getRawData(file):
	with open(file) as csvfile:
  		reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
  		data = []
  		for row in reader:
			 data.append(row)

	return data

def stringToIndex(data,dic,parameterInd):
	
	stringToIndexData = data

	for sentence in data:
		indSent = data.index(sentence)
		for wordAndTag in sentence:
			indWord = sentence.index(wordAndTag)
			stringToIndexData[indSent][indWord][parameterInd] = dic.get(wordAndTag[parameterInd])

	return stringToIndexData

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
			# yi.append(temp[1])

		xdata.append(xi)
		# ydata.append(yi)

	return xdata

# This function computes the probabilities given a particular input data set and a condition
def calcProb(data, attr, cond):

	newData = data
	newData[0] = list(3*np.ones(len(data[0])))
	# print(len(newData))

	newData = np.array(newData,dtype='int')
	newData = newData[:,attr]
	newData = list(newData)

	n1 = float(newData.count(cond))
	N = float(len(newData)-1)

	if N != 0.0:
		prob = n1 / N
	else:
		prob = 0.0

	return prob

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

# Computes the initialization probabilities, also denoted
# as \pi in the assignment write-up
def getInitProb(data, dictUse, obsIndex):

	# The data input to this function is already parsed and changed
	# to the index notation
	numOfTags = len(dictUse.values())
	pi = np.zeros([numOfTags,1],dtype=float)

	firstWordIndex = 0
	firstWords = []

	for sentence in data:
		firstWordInd = sentence[firstWordIndex][obsIndex]
		pi[firstWordInd] = pi[firstWordInd] + 1

	# Add pseudocounts to count vector
	for k in range(len(pi)):
		pi[k] = pi[k] + 1

	Z = np.sum(pi)

	for i in range(len(pi)):
		pi[i] = float(pi[i]) / float(Z)

	return pi

# Computes transition probabilities (A)
def getTransProb(data, dictUse, obsIndex):

	#	   k ------------
	#	 j a11 a12 ... a1k
	#	 | a21 a22 ... a2k
	#	 |  .			.
	#	 |  .			.
	#	 | aj1 aj2 ... ajk

	numOfTags = len(dictUse.values())
	aMatrix = np.zeros([numOfTags,numOfTags],dtype=float)


	for sentence in data:
		sentLength = len(sentence)

		for wordInd in range(sentLength-1):
			# Get y_t = k
			obs_t = sentence[wordInd+1]

			# Get y_(t-1) = j
			obs_tminus1 = sentence[wordInd]

			# Check the number of times s_k follows s_j (s_j followed by s_k) in the training data set
			aMatrix[obs_tminus1,obs_t] = aMatrix[obs_tminus1,obs_t] + 1

	# Add the pseudocounts to the A matrix
	aMatrix = aMatrix + 1

	# Normalize the count matrix to get a probabilities distribution over columns
	for row in range(numOfTags):
		Z = float(sum(aMatrix[row,:]))
		aMatrix[row,:] = aMatrix[row,:] / Z
		
	return aMatrix

# Computes the emission probabilities, also known
# as the B matrix in the assignment write-up
def getEmissionProb(data, dictTags, dictWords):

	#	   k ------------
	#	 j b11 b12 ... b1k
	#	 | b21 b22 ... b2k
	#	 |  .			.
	#	 |  .			.
	#	 | bj1 bj2 ... bjk

	# k = word index
	# j = tag index

	numOfTags = len(dictTags.values())
	numOfWords = len(dictWords.values())

	bMatrix = np.zeros([numOfTags,numOfWords],dtype=float)

	for sentence in data:
		sentLength = len(sentence)

		for word in sentence:
			# Get x_t = k
			x_t = word[0]

			# Get y_t = j
			y_t = word[1]

			# Check the number of times s_j is associated with word k in the training data set
			bMatrix[y_t,x_t] = bMatrix[y_t,x_t] + 1
			
	# Add the pseudocounts to the matrix
	bMatrix = bMatrix + 1

	# Normalize the count matrix to get a probabilities distribution over columns
	for row in range(numOfTags):
		Z = float(sum(bMatrix[row,:]))
		bMatrix[row,:] = bMatrix[row,:] / Z

	return bMatrix


if __name__ == "__main__":
	# Take inputs for file reading and writing
	train_input = sys.argv[1]
	index_to_word = sys.argv[2]
	index_to_tag = sys.argv[3]
	hmmprior = sys.argv[4]
	hmmemit = sys.argv[5]
	hmmtrans = sys.argv[6]

	# Convert training data and generate dictionaries for the word index and tag index
	trainData = getRawData(train_input)
	indexToWord = genDict(index_to_word)
	indexToTag = genDict(index_to_tag)

	# Split data into distinguishable word-tag pairs (without delimiter)
	newTrainData = splitData(trainData)
	# n = 10000
	# newTrainData = newTrainData[0:n]

	# Index the hidden Markov state by 0 and the observation (tag) by 1
	stateIndex = 0
	obsIndex = 1

	# Map all words to indices as given in the indexToWord and indexToTag files
	wordToIndex = stringToIndex(newTrainData,indexToWord,stateIndex)
	dataToIndex = stringToIndex(wordToIndex,indexToTag,obsIndex)

	# Generate separate lists containing sentences and tags
	wordsAndTags = separateData(dataToIndex)
	wordsIndex = wordsAndTags[0]
	tagsIndex = wordsAndTags[1]

	# ---------------------------------------------------------------- #
	# COMPUTE INITIAL PROBABILITIES AND SAVE TO TEXT FILE

	initProb = getInitProb(dataToIndex,indexToTag,obsIndex)

	# Save initial probabilities in a text file
	np.savetxt(str(hmmprior), initProb, delimiter='\n', fmt='%.18e')

	# ---------------------------------------------------------------- #

	# ---------------------------------------------------------------- #
	# COMPUTE TRANSITION PROBABILITIES AND SAVE TO TEXT FILE

	transProb = getTransProb(tagsIndex, indexToTag, obsIndex)
	numOfRows = np.shape(transProb)[0]

	transProb = np.matrix(transProb)

	with open(str(hmmtrans),'wb') as f:
		for line in transProb:
			np.savetxt(f,line,fmt='%.18e')

	# ---------------------------------------------------------------- #

	# ---------------------------------------------------------------- #
	# COMPUTE EMISSION PROBABILITIES AND SAVE TO TEXT FILE

	emisProb = getEmissionProb(dataToIndex, indexToTag, indexToWord)
	emisProb = np.matrix(emisProb)

	with open(str(hmmemit),'wb') as f:
		for line in emisProb:
			np.savetxt(f,line,fmt='%.18e')

	# ---------------------------------------------------------------- #

