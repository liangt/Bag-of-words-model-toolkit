#!/usr/bin/env python
# coding: utf-8 

__author__ = 'liangliang'

import numpy as np


class Pooling:
    def __init__(self, size=200, level=2, norm='L2', sub_norm='L1'):
        """
        :param size: size of visual dictionary, default value is 200
        :param level: level of spatial pyramid, default value is 2; when level<0, set level=0; when level>2, set level=2
        :param norm: normalization method of entire histogram
        :param sub_norm: normalization method of each spatial region histogram
        """
        self.size = size
        if level < 0:
            self.level = 0
        elif level > 2:
            self.level = 2
        else:
            self.level = level
        self.norm = norm
        self.sub_norm = sub_norm

    def sumPooling(self, featureCodes, frames, width, height):
        """
        :param featureCodes: numpy array of shape n_samples or [n_samples, n_visual_words]
        :param frames: numpy array of shape  [n_samples, 2]
        :param width: width of image
        :param height: height of image
        :return: numpy array of shape n_visual_words * (1 + 4 + ... + 4^level)
        """
        num = 2 ** self.level
        w = width / num
        h = height / num

        histogramOfLevelTwo = np.zeros((16, self.size))
        for featureCode, frame in zip(featureCodes, frames):
            idx_x = int(frame[0] / w)
            idx_y = int(frame[1] / h)
            idx = idx_y * num + idx_x
            histogramOfLevelTwo[idx] += featureCode

        histogramOfLevelOne = np.zeros((4, self.size))
        histogramOfLevelOne[0] = histogramOfLevelTwo[0] + histogramOfLevelTwo[1] + \
                                 histogramOfLevelTwo[4] + histogramOfLevelTwo[5]
        histogramOfLevelOne[1] = histogramOfLevelTwo[2] + histogramOfLevelTwo[3] + \
                                 histogramOfLevelTwo[6] + histogramOfLevelTwo[7]
        histogramOfLevelOne[2] = histogramOfLevelTwo[8] + histogramOfLevelTwo[9] + \
                                 histogramOfLevelTwo[12] + histogramOfLevelTwo[13]
        histogramOfLevelOne[3] = histogramOfLevelTwo[10] + histogramOfLevelTwo[11] + \
                                 histogramOfLevelTwo[14] + histogramOfLevelTwo[15]

        histogramOfLevelZero = histogramOfLevelOne[0] + histogramOfLevelOne[1] + \
                               histogramOfLevelOne[2] + histogramOfLevelOne[3]

        # normalize each spatial region histogram
        if self.sub_norm == 'L2':
            histogramOfLevelZero /= np.linalg.norm(histogramOfLevelZero) + 1e-8
            histogramOfLevelOne /= np.linalg.norm(histogramOfLevelOne, axis=1).reshape((histogramOfLevelOne.shape[0],1)) + 1e-8
            histogramOfLevelTwo /= np.linalg.norm(histogramOfLevelTwo, axis=1).reshape((histogramOfLevelTwo.shape[0],1)) + 1e-8
        else:
            histogramOfLevelZero /= histogramOfLevelZero.max() + 1e-8
            histogramOfLevelOne /= histogramOfLevelOne.max(axis=1).reshape((histogramOfLevelOne.shape[0], 1)) + 1e-8
            histogramOfLevelTwo /= histogramOfLevelTwo.max(axis=1).reshape((histogramOfLevelTwo.shape[0], 1)) + 1e-8

        # stack each spatial region histogram
        if self.level == 0:
            result = histogramOfLevelZero
        elif self.level == 1:
            result = np.concatenate((histogramOfLevelZero.flatten() * 0.5, histogramOfLevelOne.flatten() * 0.5))
        else:  # self.level == 2
            result = np.concatenate((histogramOfLevelZero.flatten() * 0.25, histogramOfLevelOne.flatten() * 0.25,
                                     histogramOfLevelTwo.flatten() * 0.5))

        # normalize entire histogram
        if self.norm == 'L1':
            result /= result.max()
        else:
            result /= np.linalg.norm(result)

        return result
    
    def sumPoolingAllLevel(self, featureCodes, frames, width, height):
        """
        :param featureCodes: numpy array of shape n_samples or [n_samples, n_visual_words]
        :param frames: numpy array of shape  [n_samples, 2]
        :param width: width of image
        :param height: height of image
        :return: list of all three level features [level0, level1, level2]
        """
        num = 4
        w = width / num
        h = height / num

        histogramOfLevelTwo = np.zeros((16, self.size))
        for featureCode, frame in zip(featureCodes, frames):
            idx_x = int(frame[0] / w)
            idx_y = int(frame[1] / h)
            idx = idx_y * num + idx_x
            histogramOfLevelTwo[idx] += featureCode

        histogramOfLevelOne = np.zeros((4, self.size))
        histogramOfLevelOne[0] = histogramOfLevelTwo[0] + histogramOfLevelTwo[1] + \
                                 histogramOfLevelTwo[4] + histogramOfLevelTwo[5]
        histogramOfLevelOne[1] = histogramOfLevelTwo[2] + histogramOfLevelTwo[3] + \
                                 histogramOfLevelTwo[6] + histogramOfLevelTwo[7]
        histogramOfLevelOne[2] = histogramOfLevelTwo[8] + histogramOfLevelTwo[9] + \
                                 histogramOfLevelTwo[12] + histogramOfLevelTwo[13]
        histogramOfLevelOne[3] = histogramOfLevelTwo[10] + histogramOfLevelTwo[11] + \
                                 histogramOfLevelTwo[14] + histogramOfLevelTwo[15]

        histogramOfLevelZero = histogramOfLevelOne[0] + histogramOfLevelOne[1] + \
                               histogramOfLevelOne[2] + histogramOfLevelOne[3]

        # normalize each spatial region histogram
        if self.sub_norm == 'L2':
            histogramOfLevelZero /= np.linalg.norm(histogramOfLevelZero) + 1e-8
            histogramOfLevelOne /= np.linalg.norm(histogramOfLevelOne, axis=1).reshape((histogramOfLevelOne.shape[0],1)) + 1e-8
            histogramOfLevelTwo /= np.linalg.norm(histogramOfLevelTwo, axis=1).reshape((histogramOfLevelTwo.shape[0],1)) + 1e-8
        else:
            histogramOfLevelZero /= histogramOfLevelZero.max() + 1e-8
            histogramOfLevelOne /= histogramOfLevelOne.max(axis=1).reshape((histogramOfLevelOne.shape[0], 1)) + 1e-8
            histogramOfLevelTwo /= histogramOfLevelTwo.max(axis=1).reshape((histogramOfLevelTwo.shape[0], 1)) + 1e-8

        # normalize entire histogram
        if self.norm == 'L1':
            level2 = np.concatenate((histogramOfLevelZero.flatten() * 0.25, histogramOfLevelOne.flatten() * 0.25,
                                         histogramOfLevelTwo.flatten() * 0.5))
            level2 /= level2.max()
            level1 = np.concatenate((histogramOfLevelZero.flatten() * 0.5, histogramOfLevelOne.flatten() * 0.5))
            level1 /= level1.max()
            histogramOfLevelZero /= histogramOfLevelZero.max()
        else:
            level2 = np.concatenate((histogramOfLevelZero.flatten() * 0.25, histogramOfLevelOne.flatten() * 0.25,
                                         histogramOfLevelTwo.flatten() * 0.5))
            level2 /= np.linalg.norm(level2)
            level1 = np.concatenate((histogramOfLevelZero.flatten() * 0.5, histogramOfLevelOne.flatten() * 0.5))
            level1 /= np.linalg.norm(level1)
            histogramOfLevelZero /= np.linalg.norm(histogramOfLevelZero)

        return [histogramOfLevelZero, level1, level2]
    
    def maxPooling(self, featureCodes, frames, width, height):
        """
        :param featureCodes: numpy array of shape n_samples or [n_samples, n_visual_words]
        :param frames: numpy array of shape  [n_samples, 2]
        :param width: width of image
        :param height: height of image
        :return: numpy array of shape n_visual_words * (1 + 4 + ... + 4^level)
        """
        num = 2 ** self.level
        w = width / num
        h = height / num

        histogramOfLevelTwo = np.zeros((16, self.size))
        for featureCode, frame in zip(featureCodes, frames):
            idx_x = int(frame[0] / w)
            idx_y = int(frame[1] / h)
            idx = idx_y * num + idx_x
            histogramOfLevelTwo[idx] = np.maximum(histogramOfLevelTwo[idx], featureCode)

        histogramOfLevelOne = np.zeros((4, self.size))
        histogramOfLevelOne[0] = np.array([histogramOfLevelTwo[0], histogramOfLevelTwo[1], histogramOfLevelTwo[4], histogramOfLevelTwo[5]]).max(axis=0)
        histogramOfLevelOne[1] = np.array([histogramOfLevelTwo[2], histogramOfLevelTwo[3], histogramOfLevelTwo[6], histogramOfLevelTwo[7]]).max(axis=0)
        histogramOfLevelOne[2] = np.array([histogramOfLevelTwo[8], histogramOfLevelTwo[9], histogramOfLevelTwo[12], histogramOfLevelTwo[13]]).max(axis=0)
        histogramOfLevelOne[3] = np.array([histogramOfLevelTwo[10], histogramOfLevelTwo[11], histogramOfLevelTwo[14], histogramOfLevelTwo[15]]).max(axis=0)

        histogramOfLevelZero = histogramOfLevelOne.max(axis=0)

        # normalize each spatial region histogram
        if self.sub_norm == 'L2':
            histogramOfLevelZero /= np.linalg.norm(histogramOfLevelZero) + 1e-8
            histogramOfLevelOne /= np.linalg.norm(histogramOfLevelOne, axis=1).reshape((histogramOfLevelOne.shape[0],1)) + 1e-8
            histogramOfLevelTwo /= np.linalg.norm(histogramOfLevelTwo, axis=1).reshape((histogramOfLevelTwo.shape[0],1)) + 1e-8
        else:
            histogramOfLevelZero /= histogramOfLevelZero.max() + 1e-8
            histogramOfLevelOne /= histogramOfLevelOne.max(axis=1).reshape((histogramOfLevelOne.shape[0], 1)) + 1e-8
            histogramOfLevelTwo /= histogramOfLevelTwo.max(axis=1).reshape((histogramOfLevelTwo.shape[0], 1)) + 1e-8

        # stack each spatial region histogram
        if self.level == 0:
            result = histogramOfLevelZero
        elif self.level == 1:
            result = np.concatenate((histogramOfLevelZero.flatten() * 0.5, histogramOfLevelOne.flatten() * 0.5))
        else:  # self.level == 2
            result = np.concatenate((histogramOfLevelZero.flatten() * 0.25, histogramOfLevelOne.flatten() * 0.25,
                                     histogramOfLevelTwo.flatten() * 0.5))

        # normalize entire histogram
        if self.norm == 'L1':
            result /= result.max()
        else:
            result /= np.linalg.norm(result)

        return result
    
    def maxPoolingAllLevel(self, featureCodes, frames, width, height):
        """
        :param featureCodes: numpy array of shape n_samples or [n_samples, n_visual_words]
        :param frames: numpy array of shape  [n_samples, 2]
        :param width: width of image
        :param height: height of image
        :return: list of all three level features [level0, level1, level2]
        """
        num = 4
        w = width / num
        h = height / num

        histogramOfLevelTwo = np.zeros((16, self.size))
        for featureCode, frame in zip(featureCodes, frames):
            idx_x = int(frame[0] / w)
            idx_y = int(frame[1] / h)
            idx = idx_y * num + idx_x
            histogramOfLevelTwo[idx] = np.maximum(histogramOfLevelTwo[idx], featureCode)

        histogramOfLevelOne = np.zeros((4, self.size))
        histogramOfLevelOne[0] = np.array([histogramOfLevelTwo[0], histogramOfLevelTwo[1], histogramOfLevelTwo[4], histogramOfLevelTwo[5]]).max(axis=0)
        histogramOfLevelOne[1] = np.array([histogramOfLevelTwo[2], histogramOfLevelTwo[3], histogramOfLevelTwo[6], histogramOfLevelTwo[7]]).max(axis=0)
        histogramOfLevelOne[2] = np.array([histogramOfLevelTwo[8], histogramOfLevelTwo[9], histogramOfLevelTwo[12], histogramOfLevelTwo[13]]).max(axis=0)
        histogramOfLevelOne[3] = np.array([histogramOfLevelTwo[10], histogramOfLevelTwo[11], histogramOfLevelTwo[14], histogramOfLevelTwo[15]]).max(axis=0)

        histogramOfLevelZero = histogramOfLevelOne.max(axis=0)

        # normalize each spatial region histogram
        if self.sub_norm == 'L2':
            histogramOfLevelZero /= np.linalg.norm(histogramOfLevelZero) + 1e-8
            histogramOfLevelOne /= np.linalg.norm(histogramOfLevelOne, axis=1).reshape((histogramOfLevelOne.shape[0],1)) + 1e-8
            histogramOfLevelTwo /= np.linalg.norm(histogramOfLevelTwo, axis=1).reshape((histogramOfLevelTwo.shape[0],1)) + 1e-8
        else:
            histogramOfLevelZero /= histogramOfLevelZero.max() + 1e-8
            histogramOfLevelOne /= histogramOfLevelOne.max(axis=1).reshape((histogramOfLevelOne.shape[0], 1)) + 1e-8
            histogramOfLevelTwo /= histogramOfLevelTwo.max(axis=1).reshape((histogramOfLevelTwo.shape[0], 1)) + 1e-8

        # normalize entire histogram
        if self.norm == 'L1':
            level2 = np.concatenate((histogramOfLevelZero.flatten() * 0.25, histogramOfLevelOne.flatten() * 0.25,
                                         histogramOfLevelTwo.flatten() * 0.5))
            level2 /= level2.max()
            level1 = np.concatenate((histogramOfLevelZero.flatten() * 0.5, histogramOfLevelOne.flatten() * 0.5))
            level1 /= level1.max()
            histogramOfLevelZero /= histogramOfLevelZero.max()
        else:
            level2 = np.concatenate((histogramOfLevelZero.flatten() * 0.25, histogramOfLevelOne.flatten() * 0.25,
                                         histogramOfLevelTwo.flatten() * 0.5))
            level2 /= np.linalg.norm(level2)
            level1 = np.concatenate((histogramOfLevelZero.flatten() * 0.5, histogramOfLevelOne.flatten() * 0.5))
            level1 /= np.linalg.norm(level1)
            histogramOfLevelZero /= np.linalg.norm(histogramOfLevelZero)

        return [histogramOfLevelZero, level1, level2]

if __name__ == '__main__':
    from PIL import Image
    from encoding import HardHistogramEncoder
    from feature import DenseSIFTFeatureDescriptors
    from visualDictionary import VisualDictionary

    img = np.array(Image.open('./test/1.jpg').convert('L'), dtype=np.float32)
    featureDescriptors = DenseSIFTFeatureDescriptors(16, 8)
    features, frames = featureDescriptors.compute(img)

    visualDictionary = VisualDictionary(20)
    visualWords = visualDictionary.buildByKMeans(features)

    encoder = HardHistogramEncoder()
    featureCodes = encoder.compute(features, visualWords)

    pooling = Pooling(20, 2)
    feature = pooling.maxPooling(featureCodes, frames, img.shape[1], img.shape[0])

    print feature.shape
