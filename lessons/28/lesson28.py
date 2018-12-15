# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 21:01:37 2018

@author: shane

Random Projection and ICA

Multiply by a random matrix to reduce the dimensions

from d dimensions to k dimensions

-- JOHNSON-LINDENSTRAUSS LEMMA --
a dataset of N points in high dimensional eucildian space can be mapped onto
a lower dimension in a way that preserves the distance between the points to
a large degree

Faster than PCA, especially in higher dimensional data

-- Random Projection in SKLEARN -- 
from sklearn import random_projection
rp = random_projection.SparseRandomProjection()
new_X = rp.fit_transform(X)

-- Independent Component Analysis (ICA) -- 

Cocktail problem, piano, chello, TV

Wow, pretty cool blind source seperation results!

Assumes components are statistically independent, and non-Gaussian

ICA needs as many observations as the original signals we are trying to separate

"""

