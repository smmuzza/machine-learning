# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:41:35 2018

@author: shane

LESSON 8 - MODEL SELECTION

"""

"""
Gozilla with a flyswatter
A fly with a bazooka

Overfitting - High Variance.
Does well in the training set, but memorizing instead of
noticng characterics, so does well in the test set
Great on the training test, bad on the test set.
Like memorizing the book word by word

Underfitting - High Bias
Model is no complex enough.
Bad on training set and test set.
Like coming to an exam completely unprepared

Good model. Good on training and test data.

"""

"""
MODEL COMPLEXITY GRAPH

graph with model and test errors as lines
number of errors on y axis
number of models of the x axis, i.e. poly 1, 2, 3

"""

trainError = 0
testError = 2

"""
CROSS VALIDATION

Low testing and training errors is the model to pick

Cross validation set will be used for choices about the model

Test set will be used for final perfroamce evaluation

"""

"""
Recycle data with K-Fold cross validation
k buckets of data, train k times, average results to 
get a final model

kf = KFold(12,3, shuffle=true), shuffle to randomize data

"""

"""
Learning Curves
High bias - not  enough complexity 
High variance - too much complexity

Cross validation error and training error should converge
for a good model, complex enough without overfitting

Model that overfits doesn't do well on cross validation, even after
a lot of training data, no convergence

For a high bias model, the error remains large even as training 
data is added, implying that the model in not complex enough 

"""

# Import, read, and split data
import pandas as pd
data = pd.read_csv('data.csv')
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(55)

### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import utils

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

### Logistic Regression
estimator = LogisticRegression()
utils.draw_learning_curves(X, y, estimator, 100)

### Decision Tree
estimator = GradientBoostingClassifier()
utils.draw_learning_curves(X, y, estimator, 100)

### Support Vector Machine
estimator = SVC(kernel='rbf', gamma=1000)
utils.draw_learning_curves(X, y, estimator, 100)

"""
RESULT: 
    Logistic Regression under fits, doesnt do well on training or validation
    Gradient Boosting Classifier decision tree is just right
    SVM overfits, doesnt do well on validation set
"""

"""
MACHINE LEARNING PROCESS
1. train a bunch of models with data
2. use cross validation set to pick the best model
3. test the selected model with test data to make sure it is good

Hyper-parameter like depth in the Neural Network

GRID SEARCH

Grid-search is performed by simply picking a list of values for each parameter,
 and trying out all possible combinations of these values. This might look 
 methodical and exhaustive. But in truth even a random search of the parameter
 space can be MUCH more effective than a grid search!

"""
