#-*- coding: UTF-8 -*-


import os
import jieba



def getStopwords(path):
    stopwords_list = []
    stopwords_raw = [line.strip() for line in open(path).readlines()]
    for s in stopwords_raw:
        stopwords_list.append(s.decode('gbk'))
    stopwords_set = set(stopwords_list)
    return stopwords_set


def getVocabularyList(path, stopWords):
    trainDir = os.listdir(path)
    vocab_set = set([])
    for cat in trainDir:
        if cat == "负面":
            negativePaths = os.listdir(path + "/" + cat)
        if cat == "正面":
            positivePaths = os.listdir(path + "/" + cat)
    for text in negativePaths:
        f = open(path + "/负面/" + text, "r")
        try:
            content = f.read()
            words_list = list(jieba.cut(content))
            # print(path + "/负面/" + text)
            # print(", ".join(words_list))
            for word in words_list:
                if word not in stopWords:
                    vocab_set.add(word)
        finally:
            f.close()
    for text in positivePaths:
        f = open(path + "/正面/" + text, "r")
        try:
            content = f.read()
            words_list = list(jieba.cut(content))
            # print(path + "/正面/" + text)
            # print(", ".join(words_list))
            for word in words_list:
                if word not in stopWords:
                    vocab_set.add(word)
        finally:
            f.close()
    vocab_list = list(vocab_set)
    return vocab_list


def getTrainVector(path, vocab_list):
    trainCatDir = os.listdir(path)
    train_vector = []
    trainTarget_vector = []
    for cat in trainCatDir:
        if cat == "负面":
            negativePaths = os.listdir(path + "/" + cat)
        if cat == "正面":
            positivePaths = os.listdir(path + "/" + cat)
    for text in negativePaths:
        content_vector = [0] * len(vocab_list)
        f = open(path + "/负面/" + text, "r")
        try:
            content = f.read()
            words_list = list(jieba.cut(content))
            for word in words_list:
                if word in vocab_list:
                    content_vector[vocab_list.index(word)] = content_vector[vocab_list.index(word)] + 1
            train_vector.append(content_vector)
            trainTarget_vector.append(int(0))
        finally:
            f.close()
    for text in positivePaths:
        content_vector = [0] * len(vocab_list)
        f = open(path + "/正面/" + text, "r")
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
    return train_vector, trainTarget_vector


def getTestVector(path, vocab_list):
    testCatDir = os.listdir(path)
    test_vector = []
    testTarget_vector = []
    for cat in testCatDir:
        if cat == "负面":
            t_negativePaths = os.listdir(path + "/" + cat)
        if cat == "正面":
            t_positivePaths = os.listdir(path + "/" + cat)
    for text in t_negativePaths:
        content_vector = [0] * len(vocab_list)
        f = open(path + "/负面/" + text, "r")
        try:
            content = f.read()
            words_list = list(jieba.cut(content))
            print (path + "/负面/" + text)
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
        f = open(path + "/正面/" + text, "r")
        try:
            content = f.read()
            words_list = list(jieba.cut(content))
            print (path + "/正面/" + text)
            print(", ".join(words_list))
            for word in words_list:
                if word in vocab_list:
                    content_vector[vocab_list.index(word)] = content_vector[vocab_list.index(word)] + 1
            test_vector.append(content_vector)
            testTarget_vector.append(int(1))
        finally:
            f.close()
    return test_vector, testTarget_vector











