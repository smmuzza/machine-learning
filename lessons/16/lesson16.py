# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:01:29 2018

@author: shane

EMSEMBLE METHODS

BOOSTINMG
 
Combining the opinions of friends.

Combining the weak learnings to create a strong learner.

All we need is that they do slightly better than random change in order
to see an improvement.

Pick a fully random subset of data and train multiple models.

Then make the models vote.

BOOSTING with ADABOOST

One of the most popular methods. Created in 1996.

Weighting the data:

Make the different learners focus on the parts where other learners do
poorly, by selecting data for learners based on poor performance of 
other models. 

Weighting the models:
      

"""

import numpy as np
#Accuracy = (TP+TN)/(TP+TN+FP+FN) 
acc1 =  7/8
weight1 = np.log(7/1)
weight1 = np.log(acc1/(1-acc1))

acc2 =  4/8
weight2 = np.log(4/4)
weight2 = np.log(acc2/(1-acc2))

acc3 =  2/6
weight3 = np.log(2/6)
weight3 = np.log(acc3/(1-acc3))

"""
ln(8/0) = inf
ln(0/8) = -inf

COMBINING MODELS

3 models 
weight = (+0.84, +1.3, +1.84), positive region
weight = (-0.84, -1.3, -1.84), subtract the weight

add the positive and neg weights to each region

then say that if the sum is positive then the combined model 
will predict positive, and if the sum is negative then the 
combined model will predict negative

>>> from sklearn.ensemble import AdaBoostClassifier
>>> model = AdaBoostClassifier()
>>> model.fit(x_train, y_train)
>>> model.predict(x_test)

COMMON HYPERPARAMETERS
base_estimator: The model utilized for the weak learners 
(Warning: Don't forget to import the model that you decide to use 
for the weak learner).
n_estimators: The maximum number of weak learners used.

>>> from sklearn.tree import DecisionTreeClassifier
>>> model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)

"""