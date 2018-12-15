# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:41:30 2018

@author: shane

-------

=== Principle Component Analysis (PCA) ===

Dimensionality reduction.

sqrt(2) for the length of vectors

projection onto the direction of maximal variance minimises information less,
the sum of the distance of the projections is minimal

find first and second priciple component

-- Review / Summary --
1. systemized way to transform input features into priciple components
2. use principle components as new features
3. PCs are directions in data that maximise variance (minimise function loss) 
when you project or compress down onto them

-- PCA in SKLearn --
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
The function doPCA should take data as an argument.

-- When to use PCA --
1. latent features driving the patterns in data
2. dimensionality reduction
   2.1 visualize high level data
   2.2 reduce noise
   2.3 make other algos work better because of fewer inputs 
   (regression, classification, eigenfaces)
   
-- PCA for facial recognition --
1. high dimensionality, many pixels
2. faces have general patterns that can be captured


https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html



   


"""

