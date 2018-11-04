# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:06:11 2018

@author: shane

CLASSIFICATION ALGORITHM

Class instead of value (regression)

Basis for deep learning

Linear boundary, acceptance at a uni

w = w1, w2
x = x1, x2
y = label 0 or 1

yhat = prediction, 0 or 1

3D data, fitting a 2D plane boundary

Given the table in the video above, what would the dimensions be for 
input features (x), the weights (W), and the bias (b) to satisfy (Wx + b)?
ans: W:(1*n), x:(n*1), b:1*1 -- since b is a constant

"""

"""
PERCEPTRON

The building block of nueral networks, just the encoding of a function into 
a graph.

test(7)/graph(6) -> node(-18) -> output

"""

import pandas as pd

# AND PERCEPTRON
print('\nAND PERCEPTRON\n')
# Set weight1, weight2, and bias
weight1 = 0.5
weight2 = 0.5
bias = -1.0

# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))

# OR PERCEPTRON
print('\nOR PERCEPTRON\n')

# Set weight1, weight2, and bias
weight1 = 0.5
weight2 = 0.5
bias = -0.5

# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, True, True, True]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))


# NOT PERCEPTRON
print('\nNOT PERCEPTRON\n')

# TODO: Set weight1, weight2, and bias
weight1 = 0.0
weight2 = -0.5
bias = 0.25

# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))

"""
XOR perception: requires a multi layer neural network

x1 -> NAND (AND -> NOT)
                          -> AND -> XOR
x2 -> OR

"""

"""
PERCEPTRON TRICK

In real life, though, we can't be building these perceptrons ourselves. 
The idea is that we give them the result, and they build themselves.

pos >= 0 = 3x1+ 4x2 - 10

EXAMPLE

3x1+ 4x2 - 10 = 0
A(4,5) - come closer! +1 for bias unit
= 3-4 + 4-5 -10+1 =  -1 -1 - 9 
learning rate = 0.1, so subtract since above the line
= 3-0.4 + 4-0.5 -10-0.1 =  2.6 +3.5 -10.1 -- this line is actually closer to A

B(1,1)
learning rate 0.1, so add since below the line
= 3.1x1 + 4.1x2 - 9.9

QUESTION

- 3x1+ 4x2 - 10 = 0
- learning rate 0.1
- how many times to apply learning rate to make 1,1 correctly classified

based score = 3+4-10 = -3

if I do it 1
3.1 + 4.1 -9.9 = -2.7

if I do it 2
3.1 + 4.1 -9.9 = -2.7

"""

# make a quick script to find when the learning is enough
learning_rate = 0.1
x1 = 3
x2 = 4
b = -10
p = [1,1]
yhat = x1*p[0] + x2*p[1] + b
print("solve point classification with for loop")
for i in range(20):
    x1 = x1 + learning_rate
    x2 = x2 + learning_rate
    b = b + learning_rate
    yhat = x1*p[0] + x2*p[1] + b
    print("attempt %s: x1*p[0] + x2*p[1] + b = %s" % (i, yhat))
    

x1 = 3
x2 = 4
b = -10
p = [1,1]
yhat = x1*p[0] + x2*p[1] + b
i = 0
print("solve point classification with while loop")

while(yhat < 0.0):
    x1 = x1 + learning_rate
    x2 = x2 + learning_rate
    b = b + learning_rate
    yhat = x1*p[0] + x2*p[1] + b
    print("attempt %s: x1*p[0] + x2*p[1] + b = %s" % (i, yhat))  
    i += 1

"""
Perceptron Algorithm

1. start with random weights w1, ..., wn, b
2. for every misclassified point x1, ..., xn:
    2.1 if prediction = 0
        for i=1 to n
            wi += a*xi 
        b += a
    2.2 if prediction = 1
        for i=1 to n
            wi -= a*xi 
        b -= a

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)
data = np.array(pd.read_csv('data.csv'))
#plt.scatter(x[:,0], y)
X = data[:,[0,1]]
y = data[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
#        if y[i]-y_hat == 1: # real answer 1
#            W[0] += X[i][0]*learn_rate
#            W[1] += X[i][1]*learn_rate
#            b += learn_rate
#        elif y[i]-y_hat == -1: # real answer 0
#            W[0] -= X[i][0]*learn_rate
#            W[1] -= X[i][1]*learn_rate
#            b -= learn_rate
        # faster solution replacing the if    
        W[0] += (y[i]-y_hat) * X[i][0]*learn_rate
        W[1] += (y[i]-y_hat) * X[i][1]*learn_rate
        b += (y[i]-y_hat) * learn_rate
            
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

trainPerceptronAlgorithm(X, y)
