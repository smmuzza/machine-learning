# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 23:14:16 2018

@author: shane

DECISION TREES

Cutting the data with multiple lines at each branch in the tree.

ENTROPY

The more homogeneous a set, the less entropy

The less deltas or possible configurations, the less entropy

Knowledge is the opposite of entropy.

Low knowledge = high entropy and visa vera

Turn the product into sum. Logirithm can help.
log(ab) = log(a) + log(b)

-Log2 since information theory in bits.

Entropy of 4 ball, 3 red, 1 blue config, 
related to prob of selecting starting position
log2(0.75) + log2(0.75) + log2(0.75) + log2(0.25)
= 0.415 + 0.415 + 0.415 + 2

General formula for entropy for balls of 2 colors
Ent = -m/(m+n) * log2(m/m+n) - n/(m+n) * log2(n/m+n)

For multiclass state, can generalise to
Entropy = -sum[i=1,n] * pi * log2(pi)
where pi is the probability of each state

"""

import numpy as np
#4 ball, 3 red, 1 blue entropy
numMicroStates = 4
Pwin = 0.75*0.75*0.75*0.25
negLog2Pwin = -np.log2(0.75) - np.log2(0.75) \
          - np.log2(0.75) - np.log2(0.25) 
entropy = negLog2Pwin / numMicroStates
print(- np.log2(Pwin)/4 )

"""
If we have a bucket with eight red balls,
 three blue balls, and two yellow balls, 
 what is the entropy of the set of balls? 
 Input your answer to at least three decimal places.
"""
totalBalls = 8 + 3 + 2
Pred = 8 / totalBalls
Pblue = 3 / totalBalls
Pyellow = 2 / totalBalls
entropyRed = -Pred * np.log2(Pred)
entropyBlue = -Pblue * np.log2(Pblue)
entropyYellow = -Pyellow * np.log2(Pyellow) 
Entropy = entropyRed + entropyBlue + entropyYellow
print("entropy of set of balls, 3 types: ", Entropy)

"""
INFORMATION GAIN

Informatio gain is the change in entropy

Find the entropy of the parent node, and the avg
entropy of the children

InfoGain = Ent(Parent) - Avg(sum(Ent(Children)))

Example, parent with 2 children
InfoGain = Ent(Parent) - (m/(m+n)Ent(C1) + n/(m+n)Ent(C2))

MAXIMISING INFORMATION GAIN

Split by decisions that maximise info gain

RANDOM FORESTS

Decision trees tend to overfit a lot.

Solution: Build random trees, then pick the prediction
that appears the most.

"""

"""
HYPERPARAMETERS FOR DECISION TREES

MAX DEPTH

Largest length between root and leaf. 
A tree of max length k can have at most 2^k leaves.

MIN SAMPLES PER LEAF

When splitting a node, can have the issue of all the
samples going into 1 leaf except for 1, i.e. out of 
100, 99 go to a single leaf. This means the decision
is pretty inefficient, so good to avoid (little change
in entropy). Can be an int or a float.

If it is an int, then it is the number of samples.
If is is a float, it will be considered a percentage.

MIN NUMBER OF SAMPLES PER SPLIT                                         

Same as min samples per split but applied per node 
instead of per leaf.

MAX NUMBER OF FEATURES

Limit the number of features looked for in each split 
instead of running the whole data set on each split.

This number should be large enough to find a goood 
feature, but not too large as to be too slow.

Large depth very often causes overfitting, since a 
tree that is too deep, can memorize the data. Small 
depth can result in a very simple model, which may 
cause underfitting.

Small minimum samples per leaf may result in leaves 
with very few samples, which results in the model 
memorizing the data, or in other words, overfitting.
 Large minimum samples may result in the tree not 
 having enough flexibility to get built, and may 
 result in underfitting.

max_depth: The maximum number of levels in the tree.
min_samples_leaf: The minimum number of samples allowed in a leaf.
min_samples_split: The minimum number of samples required to split an internal node.
max_features : The number of features to consider when looking for the best split.


"""

# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# play with hyperparameters such as max_depth and min_samples_leaf, and 
# try to find the simplest possible model, i.e., the least likely one to overfit!

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 1)
#model = DecisionTreeClassifier()

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)


"""
TITANIC SURVIVAL WITH DECISION TREES
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Pretty display for notebooks
#%matplotlib inline

# Set a random seed
import random
random.seed(42)

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
display(full_data.head())

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
features_raw = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(features_raw.head())

features = pd.get_dummies(features_raw)
features = features.fillna(0.0)
display(features.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)

# Import the classifier from sklearn
from sklearn.tree import DecisionTreeClassifier

# TODO: Define the classifier, and fit it to the data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)

# TODO: Train the model
model2 = DecisionTreeClassifier(max_depth = 8,\
                                min_samples_split = 6, \
                                min_samples_leaf = 6)
model2.fit(X_train, y_train)

# TODO: Make predictions
# Making predictions
y_train_pred = model2.predict(X_train)
y_test_pred = model2.predict(X_test)

# TODO: Calculate the accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)

