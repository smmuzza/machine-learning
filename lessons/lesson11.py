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

y = w1 * x + w2
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

"""