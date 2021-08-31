import csv
import sys
import numpy as np
import time

'''
To output formatted training, validation, and test data, run
python feature.py smalltrain_data.tsv smallvalid_data.tsv smalltest_data.tsv dict.txt formatted_train.tsv formatted_valid.tsv formatted_test.tsv 1

'''

def getRawData(file):
	with open(file) as tsvfile:
  		reader = csv.reader(tsvfile, delimiter='\t')
  		data = []
  		for row in reader:
			  data.append(row)
	return(data)

def getDict(dictionary):
  with open(dictionary) as tsvfile:
      reader = csv.reader(tsvfile, delimiter=' ')
      data = []
      for row in reader:
        data.append(row)
  return(data)

def formatData(data,inDict,feature_flag,threshold):

  if feature_flag == 2:
    label = data[0]
    inDictArray = np.array(inDict)
    listOfWords = inDictArray[:,0].tolist()

    formatList = [label]

    trimBagWords = []

    for i in data:
      
      c = data.count(i)
      
      if i in listOfWords and c < threshold:
        if i not in trimBagWords:
          trimBagWords.append(i)
          iLoc = listOfWords.index(i)
          value = 1
        
          formatList.append(str(inDict[iLoc][1]) + str(":") + str(value))

  else:
    label = data[0]
    inDictArray = np.array(inDict)
    listOfDictWords = inDictArray[:,0].tolist()
    
    bagOfWords = []

    formatList = [label]

    for i in data:
      # print(i)

      if i in listOfDictWords:
        
        if i not in bagOfWords:
          iLoc = listOfDictWords.index(i)
          bagOfWords.append(i)
          formatList.append(str(inDict[iLoc][1]) + str(":") + str(1))
      # else:
      #     formatList.append(str(inDict[iLoc][1]) + str(":") + str(0))

  return(formatList)

if __name__ == "__main__":
  # startTime = time.time()
  threshold = 4

  trainData = sys.argv[1]
  validData = sys.argv[2]
  testData = sys.argv[3]
  dictInput = sys.argv[4]
  formattedTrain = sys.argv[5]
  formattedValid = sys.argv[6]
  formattedTest = sys.argv[7]
  featureFlag = sys.argv[8]
  featureFlag = int(featureFlag)

  inTrainData = getRawData(trainData)
  inDict = getDict(dictInput)

  formattedTrainFile = open(formattedTrain,'w')
  s = '\t'

  for i in range(len(inTrainData)):
    inTrainDataWords = inTrainData[i][1].split()
    newTrainData = [inTrainData[i][0]] + inTrainDataWords
    formattedTrainData = formatData(newTrainData,inDict,featureFlag,threshold)
    formattedTrainDataJoined = s.join(formattedTrainData)
    formattedTrainFile.write(formattedTrainDataJoined)
    formattedTrainFile.write('\n')

  inValidData = getRawData(validData)
  formattedValidFile = open(formattedValid,'w')

  for i in range(len(inValidData)):
    inValidDataWords = inValidData[i][1].split()
    newValidData = [inValidData[i][0]] + inValidDataWords
    formattedValidData = formatData(newValidData,inDict,featureFlag,threshold)
    formattedValidDataJoined = s.join(formattedValidData)
    formattedValidFile.write(formattedValidDataJoined)
    formattedValidFile.write('\n')

  inTestData = getRawData(testData)
  formattedTestFile = open(formattedTest,'w')

  for i in range(len(inTestData)):
    inTestDataWords = inTestData[i][1].split()
    newTestData = [inTestData[i][0]] + inTestDataWords
    formattedTestData = formatData(newTestData,inDict,featureFlag,threshold)
    formattedTestDataJoined = s.join(formattedTestData)
    formattedTestFile.write(formattedTestDataJoined)
    formattedTestFile.write('\n')

  # print(time.time()-startTime)