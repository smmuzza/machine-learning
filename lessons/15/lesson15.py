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

POLY KERNEL

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

"""

