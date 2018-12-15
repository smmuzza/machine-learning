# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 23:15:15 2018

@author: shane

Gaussian Mixture Models

Can retrieve original 2 Gaussians from a 2D MV guassian data set

GMM clustering with expectation maximization

Step 1. init, can initialize the Gaussians with k means also

Step 2. Soft cluster, get the liklihood pdf score normalized against all clusters, 
results in a probability of 0-1 

Step 3. maximization, reestimate the params of the gaussian

Step 4. evaluate log liklihood

USING SKLEARN 

from sklearn import datasets, mixture
X = datasets.load_iris().data[:10]
gmm = mixture.GaussianMixture(n_components=3)
gmm.fit()
clustering = gmm.predict()


--PROS--
soft clustering, sample membership to multiple clusters
cluster shapoe flexibility

--CONS--
sensitive to initialization values
possible convergence to local optimum
slow convergence rates


** CLUSTER ANALYSIS PROCESS **

1. feature selection and extraction
2. clustering algo selection and tuning, and a distance score
euclidian
cosine distance
pearsons measure
3. cluster validation
4. results intepretation (usually needs domain expertise)


** CLUSTER VALIDATION **

External indices: when we have labels. 

Rand Index

RI = A+b/(n choose 2)

a = pairs in same cluster in label C and result K
b = pairs in different cluster in label C and result K
n = number of samples/points

Adjusted rand index
ARI = RI - ExpectedIndex / (max(RI) - ExpectedIndex)


Internal indices:

** Silhouette coefficient **
Si = bi - ai / max(ai*bi) -- average to do whole data set

a = average distance to other sample in the same cluster
b = average distance to samples in the closest neighboring cluster

 (only appropriate for certain clustering algos, 
like k means an single link). Good scores not always good, depending on the 
shape of the actaul true data. NOT GOOD FOR DBSCAN (use DBCV).    
can run clustering with different values, i.e. for K in k means


  

    
Relative indices:

Compactness
Seperability



"""

