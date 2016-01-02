#!/usr/bin/env python
# coding: utf-8 

__author__ = 'liangliang'

from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GMM


class VisualDictionary:
    """
    Attributes:
        visualWords: visual words in dictionary, a size*numFeature array
        size: the number of visual words in dictionary
    """

    def __init__(self, size):
        """
        :param size: the number of visual words in dictionary
        """
        self.size = size

    def buildByKMeans(self, features):
        """
        :param features: numpy array of shape [n_samples, n_features]
        :return: numpy array of shape [n_visual_words, n_features]
        """
        kmeans = MiniBatchKMeans(n_clusters=self.size, init_size=3*self.size)
        kmeans.fit(features)
        return kmeans.cluster_centers_
    
    def buildByGMM(self, features):
        """
        :param features: numpy array of shape [n_samples, n_features]
        :return: [weight, mean, covariance]
            weight: the mixing weights for each mixture component, numpy array of shape [n_visual_words,]
            mean: mean parameters for each mixture component, numpy array of shape [n_visual_words, n_features]
            covariance: covariance parameters for each mixture component, numpy array of shape [n_visual_words, n_features]
        """
        gmm = GMM(n_components=self.size)
        gmm.fit(features)
        return [gmm.weights_, gmm.means_, gmm.covars_]

if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    from feature import DenseSIFTFeatureDescriptors

    img = np.array(Image.open('./test/1.jpg').convert('L'), dtype=np.float32)
    featureDescriptors = DenseSIFTFeatureDescriptors(16, 8)
    features, frames = featureDescriptors.compute(img)

    visualDictionary = VisualDictionary(20)
    visualWords = visualDictionary.buildByKMeans(features)
    print visualWords.shape