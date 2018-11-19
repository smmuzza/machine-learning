# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:59:47 2018

@author: shane

Lesson 7

"""

"""
1. 2. confusion matrix, actual in cols, predicted in rows
TP, FN; 
FP, TN

3. accuracy = totalCorrect / totalPredictions 

When will this not work? All transactions are good in credit card fraud.
because it never gets the fruadulant transactions

FP v.s. FN, which is worse depends on the application

High recall for medical diagnosis, 
--> FP mistakes are ok (further testing)
= TP / (TP+FN) -- row 1 in confusion matrix

High precicsion for spam detection, 
--> FN mistakes are ok (dont send real email to spam)
= TP / (TP+FP) -- col 1 in confusion matrix

"""

acc = 11/14
print("acc: ", acc)

precision = 6/8
print("precision: ", precision)

recall = 6/7
print("precision: ", recall)

"""
F1 score
a probabilitic normalized metric
combine probabilities by multiplying them

Harmonic mean = 2xy/(x+y) = F1 score

as opposed to 

arithmetic mean = (x+y)/2

Note that the harmonic mean is always lower than the arithmetic mean

The harmonic mean can be expressed as the reciprocal of the arithmetic mean

"""

precision = 55.6
recall = 83.3
F1 = 2 * precision * recall / (precision + recall)
print("F1: ", F1)

"""
F-beta score
weighted F1 score
(1 + N^2) * (x * y)/((N^2)x+y) = F-beta

(1 + beta^2) * (x * y)/((beta^2)precision + recall) = F-beta

For the spaceship model,
 we can't really afford any malfunctioning parts, and 
 it's ok if we overcheck some of the parts that are working well. 
 Therefore, this is a high recall model, so we associate it with 
 beta = 2.

For the notifications model, 
since it's free to send them, we won't get harmed too much 
if we send them to more people than we need to. But we also 
shouldn't overdo it, since it will annoy the users. We also 
would like to find as many interested users as we can. Thus, 
this is a model which should have a decent precision and a 
decent recall. Beta = 1 should work here.

For the Promotional Material model, 
since it costs us to send the material, we really don't want 
to send it to many people that won't be interested. Thus, 
this is a high precision model. Thus, beta = 0.5 will work here.

"""

"""

ROC Curve - Reciever Operating Characteristic
prefect split, area under curve is 1
good split, area under curve is high, maybe 0.8
random curve, random, so area under curve is 0.5

Y axis = true positive rate
X axis = false positive rate


"""

"""
REGRESSIONS METRICS

1. mean absolute error, error from the simple average value
2. mean squared error, error from linear regression
3. R2 score = 1 - MSE (linear regression fit error) / MAE (simple) 

Bad model, R2 score is close to 0, as MSE is large
Good Model, R2 score is close to 1, as MSE is close to 0


"""