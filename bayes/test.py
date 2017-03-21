#-*- coding: UTF-8 -*-

import numpy as np

from sklearn.naive_bayes import MultinomialNB
import bayesian

trainPath = "/opt/app/highlevel/training/data/TRAIN"
testPath = "/opt/app/highlevel/training/data/TEST"
stopwordsPath = "/home/hldev/Work/PYproject/bayes/stopwords.txt"

stopWords_set = bayesian.getStopwords(stopwordsPath)
vocabulary_list = bayesian.getVocabularyList(trainPath, stopWords_set)
train_list, trainTarget_list = bayesian.getTrainVector(trainPath, vocabulary_list)
test_list, testTarget_list = bayesian.getTestVector(testPath, vocabulary_list)


train_vector = np.array(train_list)
trainTarget_vector = np.array(trainTarget_list)
test_vector = np.array(test_list)
testTarget_vector = np.array(testTarget_list)

clf = MultinomialNB().fit(train_vector, trainTarget_vector)
bayesian_predict = clf.predict(test_vector)
wrongNumber = (testTarget_vector != bayesian_predict).sum()
rightRate = 1 - (float(wrongNumber) / test_vector.shape[0])

print
print("Right rate out of a total %d is : %f" % (test_vector.shape[0], rightRate))