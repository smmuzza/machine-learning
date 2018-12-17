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
indices = [1,5,12]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset:")
display(samples)


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
    step = 1.5
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    featureOutliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(featureOutliers)
    
    # append the outlier indices to a list
    for i in featureOutliers.index:
        print('outlier index: ', i)
        outliers.append(i)
    
# OPTIONAL: Select the indices for data points you wish to remove

# Remove the outliers, if any were specified
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