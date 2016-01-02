#!/usr/bin/env python
# coding: utf-8 

__author__ = 'liangliang'

import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier


class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0):
        """
        :param kernel: kernel type, it can be linear, intersection, chi_square and Hellinger, default='linear'
        :param C: penalty parameter C of the error term, default=1.0
        :return: None
        """

        self.kernel = kernel
        if kernel == 'intersection' or kernel == 'chi_square':
            self.svm = svm.SVC(kernel='precomputed')
        else:
            #self.svm = svm.LinearSVC(C=C)
            self.svm = svm.SVC(kernel='linear')
        self.trainData = None

    def chiSquareKernel(self, M, N):
        """
        :param M: numpy array of shape [n_samples, n_features] or numpy array of shape [n_features]
        :param N: numpy array of shape [n_samples, n_features]
        :return: kernel matrix
        """

        if len(M.shape) == 1:
            result = np.sum(2*(M*N / (M+N+1e-9)), axis=1)
        else:
            result = np.zeros((M.shape[0], N.shape[0]))
            for i, elm in enumerate(M):
                result[i] = np.sum(2*(elm*N / (elm+N+1e-9)), axis=1)
        return result

    def intersectionKernel(self, M, N):
        """
        :param M: numpy array of shape [n_samples, n_features] or numpy array of shape [n_features]
        :param N: numpy array of shape [n_samples, n_features]
        :return: kernel matrix
        """

        if len(M.shape) == 1:
            result = np.sum(np.minimum(M, N), axis=1)
        else:
            result = np.zeros((M.shape[0], N.shape[0]))
            for i, elm in enumerate(M):
                result[i] = np.sum(np.minimum(elm, N), axis=1)
        return result

    def train(self, trainData, trainLabel):
        """
        :param trainData: numpy array of shape [n_samples, n_features]
        :param trainLabel: numpy array of shape [n_samples]
        :return: None
        """
        if self.kernel == 'intersection':
            kernelMatrix = self.intersectionKernel(trainData, trainData)
            self.svm.fit(kernelMatrix, trainLabel)
            self.trainData = trainData
        elif self.kernel == 'chi_square':
            kernelMatrix = self.chiSquareKernel(trainData, trainData)
            self.svm.fit(kernelMatrix, trainLabel)
            self.trainData = trainData
        elif self.kernel == 'Hellinger':
            trainData = np.sqrt(trainData)
            self.svm.fit(trainData, trainLabel)
        else:
            self.svm.fit(trainData, trainLabel)

    def predict(self, testData):
        """
        :param testData: numpy array of shape [n_samples, n_features]
        :return: predict labels
        """
        if self.kernel == 'intersection':
            kernelMatrix = self.intersectionKernel(testData, self.trainData)
            return self.svm.predict(kernelMatrix)
        elif self.kernel == 'chi_square':
            kernelMatrix = self.chiSquareKernel(testData, self.trainData)
            return self.svm.predict(kernelMatrix)
        elif self.kernel == 'Hellinger':
            testData = np.sqrt(testData)
            return self.svm.predict(testData)
        else:
            return self.svm.predict(testData)
        

class SVMClassifier_OVR:
    def __init__(self, kernel='linear', C=1.0):
        """
        :param kernel: kernel type, it can be linear, intersection, chi_square and Hellinger, default='linear'
        :param C: penalty parameter C of the error term, default=1.0
        :return: None
        """

        self.kernel = kernel
        if kernel == 'intersection' or kernel == 'chi_square':
            self.svm = OneVsRestClassifier(svm.SVC(kernel='precomputed'))
        else:
            #self.svm = svm.LinearSVC(C=C)
            self.svm = OneVsRestClassifier(svm.SVC(kernel='linear'))
        self.trainData = None

    def chiSquareKernel(self, M, N):
        """
        :param M: numpy array of shape [n_samples, n_features] or numpy array of shape [n_features]
        :param N: numpy array of shape [n_samples, n_features]
        :return: kernel matrix
        """

        if len(M.shape) == 1:
            result = np.sum(2*(M*N / (M+N+1e-9)), axis=1)
        else:
            result = np.zeros((M.shape[0], N.shape[0]))
            for i, elm in enumerate(M):
                result[i] = np.sum(2*(elm*N / (elm+N+1e-9)), axis=1)
        return result

    def intersectionKernel(self, M, N):
        """
        :param M: numpy array of shape [n_samples, n_features] or numpy array of shape [n_features]
        :param N: numpy array of shape [n_samples, n_features]
        :return: kernel matrix
        """

        if len(M.shape) == 1:
            result = np.sum(np.minimum(M, N), axis=1)
        else:
            result = np.zeros((M.shape[0], N.shape[0]))
            for i, elm in enumerate(M):
                result[i] = np.sum(np.minimum(elm, N), axis=1)
        return result

    def train(self, trainData, trainLabel):
        """
        :param trainData: numpy array of shape [n_samples, n_features]
        :param trainLabel: numpy array of shape [n_samples]
        :return: None
        """
        if self.kernel == 'intersection':
            kernelMatrix = self.intersectionKernel(trainData, trainData)
            self.svm.fit(kernelMatrix, trainLabel)
            self.trainData = trainData
        elif self.kernel == 'chi_square':
            kernelMatrix = self.chiSquareKernel(trainData, trainData)
            self.svm.fit(kernelMatrix, trainLabel)
            self.trainData = trainData
        elif self.kernel == 'Hellinger':
            trainData = np.sqrt(trainData)
            self.svm.fit(trainData, trainLabel)
        else:
            self.svm.fit(trainData, trainLabel)

    def predict(self, testData):
        """
        :param testData: numpy array of shape [n_samples, n_features]
        :return: predict labels
        """
        if self.kernel == 'intersection':
            kernelMatrix = self.intersectionKernel(testData, self.trainData)
            return self.svm.predict(kernelMatrix)
        elif self.kernel == 'chi_square':
            kernelMatrix = self.chiSquareKernel(testData, self.trainData)
            return self.svm.predict(kernelMatrix)
        elif self.kernel == 'Hellinger':
            testData = np.sqrt(testData)
            return self.svm.predict(testData)
        else:
            return self.svm.predict(testData)
    
    def accuracy_topk(self, testData, testLabel, k=1):
        """
        :param testData: numpy array of shape [n_samples, n_features]
        :param testLabel: numpy array of shape [n_samples]
        :return: accuracy
        """
        if self.kernel == 'intersection':
            kernelMatrix = self.intersectionKernel(testData, self.trainData)
            pred = self.svm.decision_function(kernelMatrix)
        elif self.kernel == 'chi_square':
            kernelMatrix = self.chiSquareKernel(testData, self.trainData)
            pred = self.svm.decision_function(kernelMatrix)
        elif self.kernel == 'Hellinger':
            testData = np.sqrt(testData)
            pred = self.svm.decision_function(testData)
        else:
            pred = self.svm.decision_function(testData)
        top_k = pred.argsort()[:, -1:-k-1:-1]
        return top_k
        
if __name__ == '__main__':
    from cPickle import load
    from sklearn import preprocessing

    # load data
    trainData = np.array(load(open('./test/trainData.pkl', 'rb')))
    trainLabel = load(open('./test/trainLabel.pkl', 'rb'))
    testData = np.array(load(open('./test/testData.pkl', 'rb')))
    testLabel = load(open('./test/testLabel.pkl', 'rb'))
    
    # Linear kernel
    classifier = SVMClassifier_OVR(kernel='linear')
    classifier.train(trainData, trainLabel)
    top_k = classifier.accuracy_topk(testData, testLabel, 3)
    le = preprocessing.LabelEncoder()
    label = le.fit(testLabel)
    print sum(1.0 * (label.inverse_transform(top_k)[:, 0] == testLabel))
    

    # Linear kernel
    classifier = SVMClassifier(kernel='linear')
    classifier.train(trainData, trainLabel)
    result = classifier.predict(testData)
    correct = sum(1.0 * (result == testLabel))
    accuracy = correct / len(testLabel)
    print "SVM (" + classifier.kernel + "): " + str(accuracy) + " (" + str(int(correct)) + "/" + str(len(testLabel)) + ")"
# 
#     # Hellinger kernel
#     classifier = SVMClassifier(kernel='Hellinger')
#     classifier.train(trainData, trainLabel)
#     result = classifier.predict(testData)
#     correct = sum(1.0 * (result == testLabel))
#     accuracy = correct / len(testLabel)
#     print "SVM (" + classifier.kernel + "): " + str(accuracy) + " (" + str(int(correct)) + "/" + str(len(testLabel)) + ")"
# 
#     # Intersection kernel
#     classifier = SVMClassifier(kernel='intersection')
#     classifier.train(trainData, trainLabel)
#     result = classifier.predict(testData)
#     correct = sum(1.0 * (result == testLabel))
#     accuracy = correct / len(testLabel)
#     print "SVM (" + classifier.kernel + "): " + str(accuracy) + " (" + str(int(correct)) + "/" + str(len(testLabel)) + ")"
# 
#     # Chi square kernel
#     classifier = SVMClassifier(kernel='chi_square')
#     classifier.train(trainData, trainLabel)
#     result = classifier.predict(testData)
#     correct = sum(1.0 * (result == testLabel))
#     accuracy = correct / len(testLabel)
#     print "SVM (" + classifier.kernel + "): " + str(accuracy) + " (" + str(int(correct)) + "/" + str(len(testLabel)) + ")"