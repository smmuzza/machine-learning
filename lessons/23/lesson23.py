# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:50:29 2018

@author: shane

LESSON 21: HIERARCHICAL AND DENSITY BASED CLUSTERING

K means considerations. There are many types of clusters where it does poorly.

I tries to find mostly spherical or hyper-spherical results.

K-means has limitations, especially for less spherical shapes.

HIERARCHICAL

-- Single link clustering --

When making the third link do you use the distance
between the first or the second point? Use the closest point.

Can make a Dendrogram to make a single cluster for 1 cluster. Cut the tree
at a different height to yeild different numbers of clusters.

Tends to produce long clusers, not always desirable.

Also sometimes a single cluster can eat up almost the whole data set.

Linkage Dendrograms can be insightful.

-- Agglomerative Clustering --

1. Complete Link Clustering

2. Average Link Clustering

3. WARD'S Method


Maximum or complete-linkage clustering 	max { d ( a , b ) : a ∈ A , b ∈ B } . 
Minimum or single-linkage clustering 	min { d ( a , b ) : a ∈ A , b ∈ B } . 
Mean or average linkage clustering, or UPGMA
Ward's minimum variance criterion minimizes the total within-cluster variance.

DBSCAN - Density Basesd Spacial Clustering Applications with Noise


HIERARCHICAL Pros
HIERARCHICAL graphs are informative, visual
potent when underlying structure also has heirarchies, like biology

HIERARCHICAL Cons
Sensitive to outliers
Computationally extensive



DBSCAN - Density Clustering

Epsilon minimum distance, plus number of points to be considered a cluster

Pro - dont need to specify the number of clusters
Pro - flexibility in shape
Pro - robust with noise and outliers

Con - border points from two clusters are the points that find them first
Con - finding clusters of varying densities

Use H DBSCAN



"""

