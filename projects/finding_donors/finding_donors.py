# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:30:09 2018

@author: shane
"""

print("\nRUNNING FINDING DONORS PROJECT...\n")

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=5))

# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
greater50k = 0
lessOrEqual50k = 0
incomeData = data['income']
for i in incomeData:
    if i == '>50K':
        greater50k += 1
    elif i == '<=50K':
        lessOrEqual50k +=1

# error checking for counting, should catch corrupt strings
assert (greater50k + lessOrEqual50k == n_records), "num 50k <= and > don't match total num records!"
    
n_greater_50k = greater50k

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = lessOrEqual50k

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = greater50k/n_records *100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))

"""
"""

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
#vs.distribution(data)

"""
"""

# Log-transform the skewed features
# https://becominghuman.ai/how-to-deal-with-skewed-dataset-in-machine-learning-afd2928011cc
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
#vs.distribution(features_log_transformed, transformed = True)

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))

"""
"""

# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
# pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
# Set records with "<=50K" to 0 and records with ">50K" to 1
income = []
for i in income_raw:
    if i == '>50K':
        income.append(1)
    elif i == '<=50K':
        income.append(0)

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print(encoded)

"""
"""

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


"""
It is always important to consider the naive prediction for your data, 
to help establish a benchmark for whether a model is performing well. 

That been said, using that prediction would be pointless: If we predicted all 
people made less than $50,000, CharityML would identify no one as donors.
"""

'''
TP = np.sum(income) # Counting the ones as this is the naive case. 
Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''

print("\nNaive Model Evaluation...\n")

# TODO: Calculate accuracy, precision and recall
TP = np.sum(income)
TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
FP = len(income) - TP - TN - FN # even though TN and FN is 0 in this case, the formula FP = income.count() - TP is incorrect
print('TP:', TP)
print('FP:', FP)
print('TN:', TN)
print('FN:', FN)

# accuracy = totalCorrect / totalPredictions 
# recall = TP / (TP+FN) -- row 1 in confusion matrix
# precision = TP / (TP+FP) -- col 1 in confusion matrix
accuracy = (TP+TN) / len(income)
recall = TP / (TP + FN)
precision = TP / (TP + FP)

# F1 score
F1 = 2 * precision * recall / (precision + recall)
print('F1:', F1)

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore = (1 + beta**2) * precision * recall / ((beta)**2 * precision + recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

"""
Implementation - Creating a Training and Predicting Pipeline
"""

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score 

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train[:300]) # predict train size already 300
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train[:300], average='binary', beta=0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, average='binary', beta=0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

"""
Implementation: Initial Model Evaluation
"""

print("\nInitial model evaluation...\n")

# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
 
# TODO: Initialize the three models
clf_A = GaussianNB()
clf_Bbackup = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=5, max_depth=8, random_state=11), n_estimators=100, learning_rate=1.0, random_state=7)
clf_B = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6, random_state=27), n_estimators=100, learning_rate=1.0, random_state=7)
clf_C = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=11)
clf_Cbackup = SVC(kernel='poly', degree=2, C=0.1) # SVC(kernel='rbf', gamma=27), SVC(kernel='poly', degree=10, C=0.1)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(round(0.1*len(y_train)))
samples_1 = int(round(0.01*len(y_train)))

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

"""
Implementation: Model Tuning
"""

# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# TODO: Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=11)

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'random_state': [11], 'n_estimators': [50, 100, 200], 'max_depth': [8, 16, 24],'min_samples_leaf': [1, 4],'min_samples_split': [2, 4]}

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, average='binary', beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
print("fitting grid search object to data...\n")
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
print("getting best estimator...\n")
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and 
print("predicting for original model and optimized model...\n")
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

print("... and explore what parameters ended up being used in the new model:\n", best_clf)

"""
Feature Importance
"""

# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import RandomForestClassifier

# TODO: Train the supervised model on the training set using .fit(X_train, y_train)
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=11)
model.fit(X_train, y_train)

# TODO: Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:7]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:7]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("\nFinal Model trained on reduced data (top 7 features)\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
