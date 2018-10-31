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



"""
