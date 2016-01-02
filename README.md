# Bag-of-words-model-toolkit
BoW model toolkit implemented by python

The toolkit implements the bag-of-words model in computer vision. The toolkit contains the following modules:

1. Feature Extract
2. Dictionary Generate
3. Encoding
4. Pooling
5. classify

### Prerequest
1. [VLFeat](http://www.vlfeat.org/)
   VLFeat library is used to compile dense SIFT descriptor. See [README](lib/README.md) for more information.
2. [Numpy](http://www.numpy.org/)
3. [scikit-learn](http://scikit-learn.org/stable/)

### Modules
1. Feature Extract
   In this toolkit, I only implement dense sampling strategy. The implemented feature discriptors are SIFT and SIFT-based color discriptors, such as rgSIFT, OpponentSIFT and WSIFT.
2. Dictionary Generate
   The visual dictionary is built by clustering algorithms, such as k-means and GMM. Because GMM is more time-consuming and memory-consuming, it is only used in Fisher vector encoding and Super vector encoding.
3. Encoding
   The implemented encoding methods are:
   * Hard Histogram
   * Soft Histogram
   * Locality-constrained Linear Coding (LLC) 
   * Fisher vector encoding
   * Super vector encoding
4. Pooling
   I implement two pooling methods, average pooling and max pooling, and implement the SPM
   model. For convenience, I only implement 0, 1, or 2 level SPM.
5. classify
   The classifier use in this toolkit is SVM and the kernel can be:
   * Linear kernel
   * Hellinger kernel
   * Intersection kernel
   * Chi-square kernel
