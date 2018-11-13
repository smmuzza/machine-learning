# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:36:25 2018

@author: shane

SUPPORT VECTOR MACHINES

Perceptron as as an algorithm which minimizes an error function.

Punishing wrongly classified points on the wrong side of the line.

CLASSIFCATION ERROR

with margin, so
wx + b = -1
wx + b = 0
wx + b = 1

Use boundary margin lines for the starting point of 0 error.

MARGIN ERROR 

Want a large margin of error

Large margin gives small error, small margin gives large error

E = sq(W)
Margin = 2 / sqrt(W)
where W is the weights of the linear function w1x1 + ... + wnxn + b = 0

"""

"""
(Optional) Margin Error Calculation

on paper...
"""

"""
SVM error = classification_error + margin_error

classification_error = sum dist of mis-classified samples
margin_error = twice the norm value of the weights W for +/- margin

THE C PARAMETER

Perfer margin or accuracy? The C parameter provides this.

SVM error = C * classification_error + margin_error

Large C classifies well but has large margin
Small C has a large margin but also more classification errors

POLY KERNEL #1

Kernel trick. 

Example, 2D problem, go from line to plane

y1 = x*x
y2 = 4

How to bring y = x*x back to the line and find the boundary there?

4 = x*x

This means mapping the data to a higher poly space, then using that
higher poly space to find a cutting line for the data, and then mapping
the data back into the original data space.

solve for x = 2 and x = -2, now we have two boundaries for the line

POLY KERNEL #2

Pull the samples apart by using a circle, or by putting the samples into
a building.

The sample or the building method?

They are actually the same method, the kernel method!

x+y
or xy
or x*x + y*y

which one can cut the data
 x*x + y*y
 
 but can make as table to show that when the points are fed into the
 equation, the red and blue points can be cut by a line 
 
Then the question becomes, how many dimensions does one need in order
to split the data?

z = x*x + y*y

paraboloid

the planes of the building intercect with the paraboloid (expanding
cone, parabola in 3 dimensions)

therefore the circle and the building method are the same

2 to 5 DIMENSIONS

(x,y) 2d to (x,y,xx,xy,yy) 5d

poly kernel is xy, plus their products xx xy yy, and all functions
therefore, like a circle and a polynomial

can expand this to a 3d poly (or more), and get
x, y, xx, xy, yy, xxx, xxy, xyy, yyy 

the degree of a poly is a hyperparameter that we can tune to get the
best possible model

RBF KERNEL

Radial Basis Functions

Mountains on top of each point.

Cut the mountain with a plane or line or equivalent.

Large gamma or small gamma. Large gives a narrow mountain, a small
small gamma gives a flat mountain.

Normal distribution. Gamma is part of that, associated with the width
of the norm dist in an inverse way.

"""

# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(kernel='poly', degree=10, C=0.1)
model = SVC(kernel='rbf', gamma=27)

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)

plt.scatter(data[:,0], data[:,1], c=y)