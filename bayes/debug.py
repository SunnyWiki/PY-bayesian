#-*- coding: UTF-8 -*-

import os
import jieba
import numpy as np
from sklearn.naive_bayes import MultinomialNB


# from sklearn import datasets
# iris = datasets.load_iris()
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# print("Number of mislabeled points out of a total %d points : %d"% (iris.data.shape[0],(iris.target != y_pred).sum()))


trainPath = "/opt/app/highlevel/training/data/TRAIN"
testPath = "/opt/app/highlevel/training/data/TEST"
stopwordsPath = "/home/hldev/Work/PYproject/bayes/stopwords.txt"

categoryDir = os.listdir(trainPath)
testCatDir = os.listdir(testPath)
vocab_set = set([])
vocab_list = []

stopwords_list = []
stopwords_set = set([])
stopwords_raw = [line.strip() for line in open(stopwordsPath).readlines()]

for s in stopwords_raw:
    stopwords_list.append(s.decode('gbk'))
stopwords_set = set(stopwords_list)

for cat in categoryDir:
    if cat == "负面":
        negativePaths = os.listdir(trainPath + "/" + cat)
    if cat == "正面":
        positivePaths = os.listdir(trainPath + "/" + cat)

for text in negativePaths:
    f = open(trainPath + "/负面/" + text, "r")
    try:
        content = f.read()
        words_list = jieba.cut(content)
        print(", ".join(words_list))
        for word in words_list:
            if word not in stopwords_set:
                vocab_set.add(word)
    finally:
        f.close()

for text in positivePaths:
    f = open(trainPath + "/正面/" + text, "r")
    try:
        content = f.read()
        words_list = jieba.cut(content)
        for word in words_list:
            if word not in stopwords_set:
                vocab_set.add(word)
    finally:
        f.close()

vocab_list = list(vocab_set)

train_vector = []
trainTarget_vector = []
for text in negativePaths:
    content_vector = [0] * len(vocab_list)
    f = open(trainPath + "/负面/" + text, "r")
    try:
        content = f.read()
        words_list = jieba.cut(content)
        for word in words_list:
            if word in vocab_list:
                content_vector[vocab_list.index(word)] = content_vector[vocab_list.index(word)] + 1
        train_vector.append(content_vector)
        trainTarget_vector.append(int(0))
    finally:
        f.close()

for text in positivePaths:
    content_vector = [0] * len(vocab_list)
    f = open(trainPath + "/正面/" + text, "r")
    try:
        content = f.read()
        words_list = jieba.cut(content)
        for word in words_list:
            if word in vocab_list:
                content_vector[vocab_list.index(word)] = content_vector[vocab_list.index(word)] + 1
        train_vector.append(content_vector)
        trainTarget_vector.append(int(1))
    finally:
        f.close()

test_vector = []
testTarget_vector = []
for cat in testCatDir:
    if cat == "负面":
        t_negativePaths = os.listdir(testPath + "/" + cat)
    if cat == "正面":
        t_positivePaths = os.listdir(testPath + "/" + cat)

for text in t_negativePaths:
    content_vector = [0] * len(vocab_list)
    f = open(testPath + "/负面/" + text, "r")
    try:
        content = f.read()
        words_list = jieba.cut(content)
        print(", ".join(words_list))
        for word in words_list:
            if word in vocab_list:
                content_vector[vocab_list.index(word)] = content_vector[vocab_list.index(word)] + 1
        test_vector.append(content_vector)
        testTarget_vector.append(int(0))
    finally:
        f.close()

for text in t_positivePaths:
    content_vector = [0] * len(vocab_list)
    f = open(testPath + "/正面/" + text, "r")
    try:
        content = f.read()
        words_list = jieba.cut(content)
        for word in words_list:
            if word in vocab_list:
                content_vector[vocab_list.index(word)] = content_vector[vocab_list.index(word)] + 1
        test_vector.append(content_vector)
        testTarget_vector.append(int(1))
    finally:
        f.close()

train_vector = np.array(train_vector)
trainTarget_vector = np.array(trainTarget_vector)
test_vector = np.array(test_vector)
testTarget_vector = np.array(testTarget_vector)

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
negativeKeywords_list = []
positiveKeywords_list = []

for i in range(len(vocab_list)):
    negativeKeywords_list.append((vocab_list[i], negativeCp_list[i]))
    positiveKeywords_list.append((vocab_list[i], positiveCp_list[i]))
negativeKeywords_list = sorted(negativeKeywords_list, key=lambda pair: pair[1], reverse=True)
positiveKeywords_list = sorted(positiveKeywords_list, key=lambda pair: pair[1], reverse=True)



print("Right rate out of a total %d is : %f" % (test_vector.shape[0], rightRate))












