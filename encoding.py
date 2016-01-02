#!/usr/bin/env python
# coding: utf-8 

__author__ = 'liangliang'

import numpy as np


class HardHistogramEncoder:
    def compute(self, features, visualWords):
        """
        :param features: numpy array of shape [n_samples, n_features]
        :param visualWords: numpy array of shape [n_visual_words, n_features]
        :return: numpy array of shape [n_samples, n_visual_words]
        """
        n_samples = features.shape[0]
        n_visual_words = visualWords.shape[0]
        features_features = np.sum(features * features, axis=1)[:, np.newaxis]
        visualWords_visualWords = np.sum(visualWords * visualWords, axis=1)[np.newaxis, :]
        D = np.repeat(features_features, n_visual_words, axis=1) - 2 * features.dot(visualWords.T) + np.repeat(visualWords_visualWords, n_samples, axis=0)
        featureCodes = np.zeros((n_samples, n_visual_words))
        featureCodes[range(n_samples), D.argmin(axis=1)] = 1
        return featureCodes

class SoftHistogramEncoder:
    def __init__(self, sigma=1):
        self.sigma = sigma
        
    def compute(self, features, visualWords):
        """
        :param features: numpy array of shape [n_samples, n_features]
        :param visualWords: numpy array of shape [n_visual_words, n_features]
        :return: numpy array of shape [n_samples, n_visual_words]
        """
        n_samples = features.shape[0]
        n_visual_words = visualWords.shape[0]
        features_features = np.sum(features * features, axis=1)[:, np.newaxis]
        visualWords_visualWords = np.sum(visualWords * visualWords, axis=1)[np.newaxis, :]
        D = np.repeat(features_features, n_visual_words, axis=1) - 2 * features.dot(visualWords.T) + np.repeat(visualWords_visualWords, n_samples, axis=0)
        featureCodes = np.exp(-0.5*self.sigma*D)
        return featureCodes

class LLCEncoder:
    def __init__(self, knn=5, beta=1e-4):
        """
        :param knn: number of nearest neighboring
        :param beta: regularization to improve condition
        """
        self.knn = knn
        self.beta = beta
        
    def compute(self, features, visualWords):
        """
        :param features: numpy array of shape [n_samples, n_features]
        :param visualWords: numpy array of shape [n_visual_words, n_features]
        :return: numpy array of shape [n_samples, n_visual_words]
        """
        n_samples = features.shape[0]
        n_visual_words = visualWords.shape[0]
        features_features = np.sum(features * features, axis=1)[:, np.newaxis]
        visualWords_visualWords = np.sum(visualWords * visualWords, axis=1)[np.newaxis, :]
        D = np.repeat(features_features, n_visual_words, axis=1) - 2 * features.dot(visualWords.T) + np.repeat(visualWords_visualWords, n_samples, axis=0)
        idx = D.argsort()[:, :self.knn]
        
        # llc approximation coding
        featureCodes = np.zeros((n_samples, n_visual_words))
        for i in range(n_samples):
            z = visualWords[idx[i]] - np.repeat(features[i][np.newaxis, :], self.knn, axis=0)
            C = z.dot(z.T)
            tc = np.trace(C)
            if tc < 0:
                tc = 2
            C = C + self.beta * tc * np.eye(self.knn)
            try:
                w = np.linalg.solve(C, np.ones(self.knn))
                w = w / np.sum(w)
                featureCodes[i, idx[i]] = w
            except:
                print 'Error'
            
        return featureCodes
           
if __name__ == '__main__':
    from PIL import Image
    from feature import DenseSIFTFeatureDescriptors
    from visualDictionary import VisualDictionary

    img = np.array(Image.open('./test/1.jpg').convert('L'), dtype=np.float32)
    featureDescriptors = DenseSIFTFeatureDescriptors(16, 8)
    features, frames = featureDescriptors.compute(img)

    visualDictionary = VisualDictionary(500)
    visualWords = visualDictionary.buildByKMeans(features)

    encoder = SoftHistogramEncoder()
    featureCodes = encoder.compute(features[0].reshape(1, -1), visualWords)
    print featureCodes
