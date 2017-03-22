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

conditional_prob = clf.__dict__.get("feature_log_prob_")
prior_prob = clf.__dict__.get("class_log_prior_")
negativeCp_list = list(conditional_prob[0])
positiveCp_list = list(conditional_prob[1])
negativePrior = prior_prob[0]
positivePrior = prior_prob[1]

negativeKeywords, positiveKeywords = bayesian.getKeywords(negativeCp_list, positiveCp_list, vocabulary_list)

print
for k in range(30):
    print negativeKeywords[k][0],
print
for m in range(30):
    print positiveKeywords[m][0],

print
print("Right rate out of a total %d is : %f" % (test_vector.shape[0], rightRate))