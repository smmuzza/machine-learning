# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:46:21 2018

@author: shane

---

Lesson 21: Clustering and Dimensionality Reduction

---

K-MEANS

1. assign: find the best point squared distance to each to cluster centers
2. optimize: find the best cluster center based on minimum squared distance to the points

https://www.naftaliharris.com/blog/visualizing-k-means-clustering/

How to solve the issue of not knowing how many initial clusters, and where to 
intially place them?

https://scikit-learn.org/stable/modules/clustering.html

As a hill climbing algorithm, it is very dependent on where the initial clusters
are placed. Therefore, for repeated runs on the same data, it will not produce
deterministic results unless the initial clustrer priors are the same.

Depending on the initial guesses, it is possible to get stuck in a local minimum.

As such, one is forced to run the algorithm multiple times.

"""

