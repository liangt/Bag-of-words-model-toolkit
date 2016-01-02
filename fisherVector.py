'''
Created on Dec 17, 2015

@author: liangliang
'''

import numpy as np
from sklearn.mixture import GMM


class FisherSuperEncoder:
    def __init__(self, dictionary_size=256, pca_dimension=80, patch_size=32, stride=4):
        """
        :param dictionary_size: size of dictionary
        :param pca_dimension: dimensionality of feature
        :param patch_size: patch size
        :param stride: spatial stride
        """
        self.dictionary_size = dictionary_size
        self.pca_dimension = pca_dimension
        self.patch_size = patch_size
        self.stride = stride
    
    def build_dictionary(self, features):
        """
        :param features: numpy array of shape [n_samples, n_features]
        """
        # compute mean and covariance matrix for the PCA
        pca_mean = features.mean(axis=0)
        features = features - pca_mean
        cov = np.dot(features.T, features)
        
        # compute PCA matrix and keep only pca_dimension dimensions
        eigvals, eigvecs = np.linalg.eig(cov)
        perm = eigvals.argsort()
        pca_transform = eigvecs[:, perm[-self.pca_dimension:]]
        
        # transform sample with PCA
        features = np.dot(features, pca_transform)
        
        # train GMM
        gmm = GMM(n_components=self.dictionary_size)
        gmm.fit(features)
        
        self.pca_mean = pca_mean
        self.pca_transform = pca_transform
        self.weights = gmm.weights_
        self.means = gmm.means_
        self.covariance = np.sqrt(1 / gmm.covars_)
        self.gmm = gmm
    
    def fisher_vector(self, features):
        """
        :param features: numpy array of shape [n_samples, n_features]
        :return: fisher vector of image
        """
    
        features = features - self.pca_mean
        features = np.dot(features, self.pca_transform)
        
        sqrt2 = np.sqrt(2)

        featureU = np.zeros((self.dictionary_size, self.pca_dimension))
        featureV = np.zeros((self.dictionary_size, self.pca_dimension))
        featureNum = features.shape[0]
        for feature in features:
            prob = self.gmm.predict_proba(feature[np.newaxis, :]).flatten()
            for i in range(self.dictionary_size):
                dif = feature - self.means[i]
                wt = np.sqrt(self.weights[i])
                featureU[i] += prob[i] * np.diag(self.covariance[i]).dot(dif) / wt
                featureV[i] += prob[i] * (np.diag(dif).dot(np.diag(self.covariance[i])).dot(dif) - 1) / (sqrt2 * wt)
                                                    
        featureU /= featureNum
        featureV /= featureNum
        feature_code = np.hstack((featureU.flatten(), featureV.flatten()))

        # normalize fisher code
        feature_code /= np.linalg.norm(feature_code)
        
        return feature_code
    
    def super_vector(self, features):
        """
        :param features: numpy array of shape [n_samples, n_features]
        :return: fisher vector of image
        """
    
        features = features - self.pca_mean
        features = np.dot(features, self.pca_transform)

        featureU = np.zeros((self.dictionary_size, self.pca_dimension))
        featureS = np.zeros(self.dictionary_size)
        featureNum = features.shape[0]
        for feature in features:
            prob = self.gmm.predict_proba(feature[np.newaxis, :]).flatten()
            featureS += prob
            for i in range(self.dictionary_size):
                featureU[i] += prob[i] * (feature - self.means[i])
                                                    
        featureS /= featureNum
        featureU /= featureS[:, np.newaxis]
        feature_code = np.hstack((featureS, featureU.flatten()))

        # normalize fisher code
        feature_code /= np.linalg.norm(feature_code)
        
        return feature_code
    
    def fisher_super_vector(self, features):
        """
        :param features: numpy array of shape [n_samples, n_features]
        :return: fisher and super vector of image
        """
    
        features = features - self.pca_mean
        features = np.dot(features, self.pca_transform)
        
        sqrt2 = np.sqrt(2)

        featureU_fv = np.zeros((self.dictionary_size, self.pca_dimension))
        featureV_fv = np.zeros((self.dictionary_size, self.pca_dimension))
        featureU_sv = np.zeros((self.dictionary_size, self.pca_dimension))
        featureS_sv = np.zeros(self.dictionary_size)
        featureNum = features.shape[0]
        for feature in features:
            prob = self.gmm.predict_proba(feature[np.newaxis, :]).flatten()
            featureS_sv += prob
            for i in range(self.dictionary_size):
                dif = feature - self.means[i]
                wt = np.sqrt(self.weights[i])
                featureU_fv[i] += prob[i] * np.diag(self.covariance[i]).dot(dif) / wt
                featureV_fv[i] += prob[i] * (np.diag(dif).dot(np.diag(self.covariance[i])).dot(dif) - 1) / (sqrt2 * wt)
                featureU_sv[i] += prob[i] * (feature - self.means[i])
                                                    
        featureU_fv /= featureNum
        featureV_fv /= featureNum
        feature_code_fv = np.hstack((featureU_fv.flatten(), featureV_fv.flatten()))
        
        featureS_sv /= featureNum
        featureU_sv /= featureS_sv[:, np.newaxis]
        feature_code_sv = np.hstack((featureS_sv, featureU_sv.flatten()))

        # normalize fisher code
        feature_code_fv /= np.linalg.norm(feature_code_fv)
        feature_code_sv /= np.linalg.norm(feature_code_sv)
        
        return [feature_code_fv, feature_code_sv]
    
    
if __name__ == '__main__':
    encoder = FisherSuperEncoder(dictionary_size=5, pca_dimension=3)
    features = np.random.rand(20, 3)
    encoder.build_dictionary(features)
    feature = np.random.rand(10, 3)
    fv, sv = encoder.fisher_super_vector(feature)
    print fv.shape, sv.shape