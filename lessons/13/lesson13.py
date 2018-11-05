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