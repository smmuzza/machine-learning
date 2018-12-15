# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:35:27 2018

@author: shane


--- Feature Scaling ---

x' = (x - xmin) / (xmax - xmin)

normalize the feature by the range of values, transforming to between 0-1

disadvantage, outliers may mess up the scalingarr[i]

can do it in single line of code in sklearn

min_max_scalar

https://scikit-learn.org/stable/modules/preprocessing.html

>>> from sklearn import preprocessing
>>> import numpy as np
>>> X_train = np.array([[ 1., -1.,  2.],
...                     [ 2.,  0.,  0.],
...                     [ 0.,  1., -1.]])
>>> X_scaled = preprocessing.scale(X_train)

>>> X_scaled                                          
array([[ 0.  ..., -1.22...,  1.33...],
       [ 1.22...,  0.  ..., -0.26...],
       [-1.22...,  1.22..., -1.06...]])


SVM and Kmeans need rescaling, since they have multiple related dimensions

Decision trees do no, just making a cut in one dimension
and the cut in another. Same for linear regression, since the features all have
their own cooefficent which will adjust during training anyway, making feature
scaling unnessecary. 

"""

