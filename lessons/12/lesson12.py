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

    
