# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 22:07:44 2018

@author: shane
"""

"""
classification: spam or not
regression: how much
"""

"""
linear regression, moving a line by changing the parameters

absolute trick, want the line to come closer to the point

y = w1 * x + w2 (i.e. y=mx + c)
y = (w1+p) * x + (w2 + 1)

if point above the line
y = (w1 + p * alpha) * x + (w2 + alpha)

if point below the line
y = (w1 - p * alpha) * x + (w2 - alpha)

where p is the distance from the y axis, and alpha is the learning rate
"""

"""
square trick

add a vertical distance instead of just a horizontal distance in the 
absolute trick

y = (w1 - p * (q-q') * alpha) * x + (w2 - (q-q') * alpha)

also takes care of points that are under the line without needing
to have two rules like the absolute trick

can get away with a smaller learning rate and the line will converge faster

"""

"""
gradient descent

1. draw a line and find the error
2. move line and recompute the error

take the gradient of the error function wrt weights,
the negative of this gradient will be where the error decreases the most,
so this can be used to take steps towards the minimum or at least 
a place with small error

w_i -> w_i - (alpha) * d/dw_i (Error)

"""

"""
MEAN ABSOLUTE ERROR

E = 1/m * sum[i=1, i=m](abs(y - y_hat))
where y_hat is the mean and M is the number of samples


MEAN SQUARED ERROR

make a square with the point and the line. the difference in the y position
of the point and the intercept of y on the line

E = 1 / 2m * sum[i=1, i=m](pow(y - y_hat, 2))


MINIMISING ERROR FUNCTIONS
when we minimise the MAE we are using a gradient descent step
the gradient descent step is the same thing as the absolute trick

when minimise the sq error the gradient descent step is the same thing
are the square trick


MSE vs total squared
Therefore, since derivatives are linear functions, the gradient of T is
also m times the gradient of M

However, the gradient descent step consists of subtracting the gradient of 
the error times the learning rate α \alpha α. Therefore, choosing 
between the mean squared error and the total squared error really just 
amounts to picking a different learning rate.

In real life, we'll have algorithms that will help us determine a good 
learning rate to work with. Therefore, if we use the mean error or the 
total error, the algorithm will just end up picking a different 
learning rate.

Batch vs Stochastic Gradient Descent
More specifically, the squared (or absolute) trick, when applied to 
a point, gives us some values to add to the weights of the model. 
We can add these values, update our weights, and then apply the squared 
(or absolute) trick on the next point. Or we can calculate these values
 for all the points, add them, and then update the weights with the sum
 of these values.

The best way to do linear regression, is to split your data into many 
small batches. Each batch, with roughly the same number of points. 
Then, use each batch to update your weights. This is still called 
mini-batch gradient descent.

"""

get_ipython().run_line_magic('matplotlib', 'inline')

# TODO: Add import statements
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('bmiandlifeexpectancy.csv')
 
# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
model = LinearRegression()
x_values = np.array(bmi_life_data[['BMI']])
y_values = np.array(bmi_life_data[['Life expectancy']])
model.fit(x_values, y_values)
bmi_life_model = model

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)

"""
HIGHER DIMENSIONS
3D = fitting a plane

N  dimensional space, creates a N-1 dimension hyperplane

y_hat = w1*x1 + w2*x2 + ... + wn-1*xn-1 + wn

"""

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the data from the boston house-prices dataset 
boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']

# Make and fit the linear regression model
# TODO: Fit the model and assign it to the model variable
model = LinearRegression()
model.fit(x, y)

# Make a prediction using the model
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
# TODO: Predict housing price for the sample_house
prediction = model.predict(sample_house)
print(prediction)

import matplotlib.pyplot as plt
col = x[:,0]
print(col)
plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)
plt.scatter(x[:,2], y)

"""
CLOSED FORM SOLUTION

Can solve mathematically for a system with N equations and N unknowns,
linear algrebra can do this, but as N gets big it means a big matrix inversion,
which is expensive, and something to avoid, so not generally practical.

2-DIMENSIONAL CASE
let data = x1, ..., xn
let labels = y1, ..., yn
let weights = w1 and w2

define MSE = E(w1,w2) = 1/(2m) * sum[i=1,i=m]((yhat - y)^2)

need to minimise the MSE, ignore 1/m as a normalization constant, 
sub yhat in, therefore

E(w1,w2) = 1/2 * sum[i=1,i=m]((w1*xi + w2 - y)^2)

to minimise find the derivatives wrt w1 and w2 and set them to 0

dE/dw1 = 0 = w1*sum(xi^2) + w2*sum(xi) - sum(xi*yi)
dE/dw2 = 0 = w1*sum(xi) + w2*sum(1) - sum(yi)

solving the simulaneous equations for w1 and w2

N-DIMENSIONAL CASE

Matrix math

E(W) = 1/m * ((XW)' - y')(XW - y)

where ((XW)' - y')(XW - y) is equivalent of a sum

expanding E(W) = W'X'XW - (XW)'y - y'(XW) + y'y

since (XW)'y = y'(XW), the inner product of two vectors, therefore
E(W) = W'X'XW - 2 * (XW)'y + y'y

to minimise find the derivative, using the chain rule...
dE/dW = 2X'XW - 2 * (X)'y

[need to brush up on linear algebra and calc on it]

0 = 2X'XW - 2 * (X)'y
X'XW = X'y
W = inv(X'X)X'y

however this solution is expensive for large matrix sizes       

"""

"""
LINEAR REGRESSION WARNINGS

- works best when the data is linear
- sensitive to outliers

POLYNOMIAL REGRESSION

yhat = w1*x3 + w2*x2 + w3*x + w4 
same as linear regression, just minimise the weights 

REGULARIZATION

The model takes an error and minimises it, which makes the model trained to
fit the data set perfectly instead of generalizing

take the complexity into account when calculating the error scores

L1 regularization: take the abs coefficients (weights) and add them to the error

L2 regularization: add the sq of the coeffecients and add them to the error

Complex model gets punished more, are we punishing too much or too little?

Requires low error so complexity ok, or requires simplicity and speed to ok
with some errors?

Lambda. Use this to multiply the complexity penalty by. How much to punish.

Which Lambda and which regularization, L1 and L2

L2 usually faster unless the data is sparse than L1 is better

L1 gives feature selection. L2 no feature selection.

L1 for sparse outputs. L2 for non-sparse outputs.

"""
