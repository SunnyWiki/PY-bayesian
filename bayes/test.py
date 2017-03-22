#-*- coding: UTF-8 -*-

import numpy as np
from sklearn.naive_bayes import MultinomialNB
import bayesian

trainPath = "/opt/app/highlevel/training/data/TRAIN"
testPath = "/opt/app/highlevel/training/data/TEST"
stopwordsPath = "/home/hldev/Work/PYproject/bayes/stopwords.txt"
negativeModelPath = "/opt/app/highlevel/training/model/pyNegativeModel.txt"
positiveModelPath = "/opt/app/highlevel/training/model/pyPositiveModel.txt"

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

negativeModel, positiveModel = bayesian.getKeywords(negativeCp_list, positiveCp_list, vocabulary_list)

# bayesian.writeModel(negativeModelPath, negativeModel)
# bayesian.writeModel(positiveModelPath, positiveModel)

negativeModel_dict = bayesian.readModel(negativeModelPath)
positiveModel_dict = bayesian.readModel(positiveModelPath)

negModel_list = []
posModel_list = []

for key in negativeModel_dict:
    negModel_list.append((key, float(negativeModel_dict[key])))
for key in positiveModel_dict:
    negModel_list.append((key, float(positiveModel_dict[key])))

negModel_list = sorted(negModel_list, key=lambda pair: pair[1], reverse=True)
posModel_list = sorted(posModel_list, key=lambda pair: pair[1], reverse=True)


print("Right rate out of a total %d is : %f" % (test_vector.shape[0], rightRate))