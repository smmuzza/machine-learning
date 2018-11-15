# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:30:09 2018

@author: shane
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=5))

# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
greater50k = 0
lessOrEqual50k = 0
incomeData = data['income']
for i in incomeData:
    if i == '>50K':
        greater50k += 1
    elif i == '<=50K':
        lessOrEqual50k +=1

# error checking for counting, should catch corrupt strings
assert (greater50k + lessOrEqual50k == n_records), "num 50k <= and > don't match total num records!"
    
n_greater_50k = greater50k

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = lessOrEqual50k

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = greater50k/n_records *100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)

