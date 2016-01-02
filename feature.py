#!/usr/bin/env python
# coding: utf-8 

__author__ = 'liangliang'

import ctypes as ct
import numpy as np


# Dynamic allocation through callbacks of ctypes
# Allocate NumPy arrays if and when we need a buffer for C code to operate on
# By allocating your buffers as NumPy arrays, the Python garbage collector can take care of this
# Here use callable object instead of function
class Allocator:
    CFUNCTYPE = ct.CFUNCTYPE(ct.c_long, ct.c_int, ct.POINTER(ct.c_int))

    def __init__(self):
        self.allocated_arrays = []

    def __call__(self, dim, shape):
        x = np.zeros(shape[:dim], 'f4')
        self.allocated_arrays.append(x)
        return x.ctypes.data_as(ct.c_void_p).value

    def getcFunc(self):
        return self.CFUNCTYPE(self)

    cFunc = property(getcFunc)


class DenseSIFTFeatureDescriptors:
    def __init__(self, binSize, step, sqrt=False):
        """
        :param binSize: width of the SIFT spatial bins
        :param step: spatial stride
        :param sqrt: whether use RootSIFT
        """
        self.binSize = binSize
        self.step = step
        self.sqrt = sqrt

    def compute(self, img):
        """
        :param img: gray image, a height*width array of float32
        :return (features, frames): features is a numpy array of shape [n_samples, 128], frames is a numpy array of shape  [n_samples, 2]
        """
        dsiftlib = ct.CDLL('./lib/libdsift.so')  # Loading dynamic link library

        # Specifying the required return type and argument types
        alloc = Allocator()
        dsiftlib.denseSIFT.restype = None
        dsiftlib.denseSIFT.argtypes = [ct.POINTER(ct.c_float), ct.c_int, ct.c_int, ct.c_int, ct.c_int, Allocator.CFUNCTYPE]

        dsiftlib.denseSIFT(img.ctypes.data_as(ct.POINTER(ct.c_float)), img.shape[1], img.shape[0], self.step, self.binSize, alloc.cFunc)

        if self.sqrt:
            alloc.allocated_arrays[0] = np.sqrt(alloc.allocated_arrays[0] / alloc.allocated_arrays[0].max())

        return alloc.allocated_arrays

class RGSIFTFeatureDescriptors:
    def __init__(self, binSize, step, sqrt=False):
        """
        :param binSize: width of the SIFT spatial bins
        :param step: spatial stride
        :param sqrt: whether use RootSIFT
        """
        self.binSize = binSize
        self.step = step
        self.sqrt = sqrt

    def compute(self, img):
        """
        :param img: RGB image, a height*width*3 array of float32
        :return (features, frames): features is a numpy array of shape [n_samples, 256], frames is a numpy array of shape  [n_samples, 2]
        """
        featureDescriptors = DenseSIFTFeatureDescriptors(self.binSize, self.step, self.sqrt)
        
        s = np.sum(img, axis=2)
        img /= s[:, :, np.newaxis] + 1e-8
        
        r_feature, frame = featureDescriptors.compute(img[:, :, 0])
        g_feature, frame = featureDescriptors.compute(img[:, :, 1])
        
        feature = np.hstack((r_feature, g_feature))
        
        return (feature, frame)

class OpponentSIFTFeatureDescriptors:
    def __init__(self, binSize, step, sqrt=False):
        """
        :param binSize: width of the SIFT spatial bins
        :param step: spatial stride
        :param sqrt: whether use RootSIFT
        """
        self.binSize = binSize
        self.step = step
        self.sqrt = sqrt

    def compute(self, img):
        """
        :param img: RGB image, a height*width*3 array of float32
        :return (features, frames): features is a numpy array of shape [n_samples, 384], frames is a numpy array of shape  [n_samples, 2]
        """
        featureDescriptors = DenseSIFTFeatureDescriptors(self.binSize, self.step, self.sqrt)
        
        o1 = (img[:, :, 0] - img[:, :, 1]) / np.sqrt(2)
        o2 = (img[:, :, 0] + img[:, :, 1] - 2 * img[:, :, 2]) / np.sqrt(6)
        o3 = np.sum(img, axis=2) / np.sqrt(3)
        
        o1_feature, frame = featureDescriptors.compute(o1)
        o2_feature, frame = featureDescriptors.compute(o2)
        o3_feature, frame = featureDescriptors.compute(o3)
        
        feature = np.hstack((o1_feature, o2_feature, o3_feature))
        
        return (feature, frame)  
    
class WSIFTFeatureDescriptors:
    def __init__(self, binSize, step, sqrt=False):
        """
        :param binSize: width of the SIFT spatial bins
        :param step: spatial stride
        :param sqrt: whether use RootSIFT
        """
        self.binSize = binSize
        self.step = step
        self.sqrt = sqrt

    def compute(self, img):
        """
        :param img: RGB image, a height*width*3 array of float32
        :return (features, frames): features is a numpy array of shape [n_samples, 256], frames is a numpy array of shape  [n_samples, 2]
        """
        featureDescriptors = DenseSIFTFeatureDescriptors(self.binSize, self.step, self.sqrt)
        
        o3 = np.sum(img, axis=2) / np.sqrt(3) + 1e-8
        o1 = (img[:, :, 0] - img[:, :, 1]) / np.sqrt(2) / o3
        o2 = (img[:, :, 0] + img[:, :, 1] - 2 * img[:, :, 2]) / np.sqrt(6) / o3
        
        
        
        o1_feature, frame = featureDescriptors.compute(o1)
        o2_feature, frame = featureDescriptors.compute(o2)
        
        feature = np.hstack((o1_feature, o2_feature))
        
        return (feature, frame)        
    
if __name__ == '__main__':
    from PIL import Image

    img = np.array(Image.open('./test/1.jpg'), dtype=np.float32)
    featureDescriptors = RGSIFTFeatureDescriptors(16, 8)
    features, frames = featureDescriptors.compute(img)
    print features.shape, frames.shape
    print features[:10, :]
    print frames[:10, :]