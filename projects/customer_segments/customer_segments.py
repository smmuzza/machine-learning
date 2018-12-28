# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:13:47 2018

@author: shane
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")
    
# Display a description of the dataset
display(data.describe())

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [1,12,75]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset:")
display(samples)

import seaborn as sns
samples_bar = samples.append(data.describe().loc['mean'])
samples_bar = samples_bar.append(data.describe().loc['std'])
samples_bar.index = indices + ['mean'] + ['std']
_ = samples_bar.plot(kind='bar', figsize=(14,6), title='data with mean and std dev bar plot')

import seaborn as sns
residual = samples - data.describe().loc['mean']
resid_sq = residual*residual
variance = data.describe().loc['std'] * data.describe().loc['std']
chi_sq_score = resid_sq / variance
chi_sq_score.index = indices
_ = chi_sq_score.plot(kind='bar', figsize=(14,6), title='chi sq scores')
display(chi_sq_score)


# --- Implementations: Feature Relevance ---

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
#  	Fresh 	Milk 	Grocery 	Frozen 	Detergents_Paper 	Delicatessen
featureToDrop = 'Grocery'
new_data = data.drop([featureToDrop], axis=1)

# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
from sklearn.model_selection import train_test_split
X = new_data
y = data.loc[:,featureToDrop]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print('score is: ', score)

# Produce a scatter matrix for each pair of features in the data
corr = data.corr()
import seaborn as sns
#ax = sns.heatmap(corr)
#pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# --- Implementation: Feature Scaling ---
# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
#pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Display the log-transformed sample data
display(log_samples)


# --- Implementations: Outlier Detection ---

# For each feature find the data points with extreme high or low values
featureOutliers = []
outliers  = []
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 # 3.0
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    featureOutliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(featureOutliers)
    
    # append the outlier indices to a list
    for i in featureOutliers.index:
        print('outlier index: ', i)
        outliers.append(i)
    
# OPTIONAL: Select the indices for data points you wish to remove
import collections
repeated_outliers=[]
counter=collections.Counter(outliers)
print(counter)
for el in counter:
    if counter[el]>1:
        repeated_outliers.append(el)
        
print("\nFollowing records are outliers for more than one feature:", list(set(repeated_outliers)))

# Remove the outliers, if any were specified
outliers = repeated_outliers
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# --- Implementations: PCA ---

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

print(pca.explained_variance_ratio_)  
print('1st and 2nd priciple component variance sum: ', pca.explained_variance_ratio_[0] + 
      pca.explained_variance_ratio_[1]) 
print('1-4 priciple component variance sum: ', pca.explained_variance_ratio_[0] + 
      pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2] + 
      pca.explained_variance_ratio_[3]) 

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# --- Implementations: Dimensionality Reduction ---

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)
pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# Create a biplot
vs.biplot(good_data, reduced_data, pca)

# ---  Implementation: Creating Clusters ---

# TODO: Apply your clustering algorithm of choice to the reduced data
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score

# loop through different number of initial clusters and score them
for k in range(2, 15):
    clusterer = KMeans(n_clusters=k)

    # TODO: Predict the cluster for each data point
    preds = clusterer.fit_predict(reduced_data) 

    # TODO: Find the cluster centers
    centers = clusterer.cluster_centers_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    print('K-Means silhouette_score for k: ', k, ' =', score)
    
# loop through different number of initial clusters and score them
from sklearn import datasets, mixture
for k in range(2, 15):
    clusterer = mixture.GaussianMixture(n_components=k)
    clusterer.fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data) 

    # TODO: Find the cluster centers
    centers = clusterer.means_ 

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    print('GMM silhouette_score for k: ', k, ' =', score)
    

k = 2    
clusterer = KMeans(n_clusters=k, random_state=11).fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data) 

# TODO: Find the cluster centers
centers = clusterer.cluster_centers_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data, preds)
print('K-Means silhouette_score for k: ', k, ' =', score)   
    
# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)   


# --- Implementation: Data Recovery ---

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

# Display a description of the dataset
display(data.describe())

# Display the predictions
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)
    
# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)
 