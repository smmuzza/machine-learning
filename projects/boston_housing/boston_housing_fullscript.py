# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:41:54 2018

@author: shane
"""

get_ipython().run_line_magic('matplotlib', 'inline')

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

"""
Use the header names of the first row to col data column-wise, as per previous lessons
#X = np.array(data[['x1', 'x2']])
#y = np.array(data['y'])
"""
print(data['MEDV'][:10]) # compare vs the actual data in variable explorer for verification

# Minimum price of the data
minimum_price = data['MEDV'].min()

# Maximum price of the data
maximum_price = data['MEDV'].max()

# Mean price of the data
mean_price = data['MEDV'].mean()

# Median price of the data
median_price = data['MEDV'].median()

# Standard deviation of prices of the data
std_price = data['MEDV'].std()

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))

"""
QUESTION 1:

RM: The average number of rooms in the neighborhood should indicate a higher value of the home, since it is more likely to be a house. Furthermore, a house with many rooms is more likely to be expensive, as each room to rent is worth real market value. As a ballpark, one would expect each room, especially bedrooms (which I we don't have data labels on specifically), to add something like 30% to the value of the house. So in short, it will increase the value, but be how much depends on the definition of room and how said rooms are labelled in the data set. Higher RM should correlate with higher home prices.

LSTAT: Economically the more poor, the lower the value of the real estate, as the poor simply cannot afford to live there. One likely exception to this can be certain areas where a particular percentage of the housing is labelled affordable housing. In areas with affordable housing, there may be a legally mandated percentage of lower class workers living there cheaply on subsidised housing, and therefore the statistics in these areas may be misleading, as the value of the property may be high though the tentent is there on under-market rate rents. In such situations the LSAT may not be as an effective a predictor as it is in other areas. In general, low LSTAT should indicate higther median home prices.

PTRATIO: I am assuming that this is referring to parent to teach ratios at public schools. These are a fairly good indicator of wealth, as richer school districts can afford to have more teachs per student, which is desirable as it is strongly correlated to things like children's grades and admission to top universities. As such, a lower RTRATIO should yield higher median home prices.

The below plots (median home price on x axis) show roughly linear trend lines for PTRATIO and RM, though PTRATIO has a large variance and some outliers to the trend. LSTAT seems to be bi-modal, made up of roughly 2 primary distributions, one above 600k, and one below 600k median price.
"""

# plot the data to get some idea of the trend lines
plt.scatter(data['MEDV'], data['RM'])
plt.scatter(data['MEDV'], data['LSTAT'])
plt.scatter(data['MEDV'], data['PTRATIO'])
plt.legend()

# Import 'r2_score'
# REF MATH: https://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination
# note that the math is for 2 variables, yet we only define a single variable in our example
# REF CODE: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html 
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true ,y_predict)
    
    # Return the score
    return score

# Calculate the performance of this model (with dummy values)
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))

"""
QUESTION 2:
A R2 score of 0.923 is pretty good.

That means that the more than 92% of the variance in the dependent variable that is predictable from the independent variable. In our case greater than 92 percent of the true variable variance is predicted by our estimation (where our estimation of the true underlying state is based on measurements somehow related to the real state).

However, for this example, the size of the sample arrays are small (5), hence variance as a statistic is not very accurate. A sample size greater than 100 would inspire more confidence.

"""

# Import 'train_test_split'
from sklearn.model_selection import train_test_split

# Shuffle and split the data into training and testing subsets
X = np.array(data[['RM', 'LSTAT', 'PTRATIO']])
y = np.array(data['MEDV'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Success
print("Training and testing split was successful.")

"""
QUESTION 3:
Splitting the data set into a testing an training set is essential.

Overfitting describes a model that fits perfectly to examples it has seen, but fits poorly to examples it has never seen before. This was a big problem in the early days of Neural Networks, before techniques like regularization, and test-train splits, were common place. These overfitted neural networks were seen as "brittle", and made experts pessimistic about their prospects. 

To limit the chance of overfit, we want to, among other things, have a special section of data that is never used for training. This is called a test set. Since the model has never seen this data before, if it can work well on the test set then it is a strong indication that we have a good model that is well trained. As such, performance against the test set it used to ensure we have not overfitted.

Underfitting is usually the result of having not enough data to train the appropriate model (hence the advantage of things like data augmentation), or that the model is too simple, too complex (such as fitting a high order polynomial to a linear funtion), or in-appropriate in others ways so that it cannot well shape or fit itself to the data provided. Therefore, the model will not perform well even on the training set on which is has attempted to learn.

If the model is "just-right" is should neither under-fit nor over-fit, and hence perform well on both the test set and the training set.
"""

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)

vs.ModelComplexity(X_train, y_train)

"""

QUESTION 4:

"Choose one of the graphs above and state the maximum depth for the model. "
--> Max depth = 3

"What happens to the score of the training curve as more training points are added? What about the testing curve?"
--> For max depth = 3, the training curve and testing curve both converge to around 0.8

"Would having more training points benefit the model? "
--> Probably not. After around 300 both the training and test curves seems to plateau 

QUESTION 5:

"When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance?"
--> High bias. Since the training score is not good and does not increase as more points are added, it is suggestive of suffering from high bias. 

"How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?"
--> In this case the there is a large gap between the good training score and the poor test score. This suggests overfitting as the model cannot generalise well to the data in the test set. 

QUESTION 6:

"Which maximum depth do you think results in a model that best generalizes to unseen data? "
--> 4

"What intuition lead you to this answer?"
--> The answer above for Question-5 "Bias-Variance Tradeoff", also aligns with the complexity curve, where the validation score peaks at a depth of around 4 (hence poor complexity performance for model depth 1), and variance (lighter green area around the dark green mean) increases as the depth increases (hence large variance for model depth 10). 

In model depth 3 we have good convergence of both training and test curves, and in model depth 6 we have some divergence of test and training curves, but better performance on the test set. The plots of training-test learning curves, combined with the complexity graph, indicate a sweet-spot trade off between test set performance and training set performance somewhere around model depth 4.
"""

"""

QUESTION 7:

The "grid" represents the search space of the model, akin to degrees of freedom or ways in which them model can move. A model based on these "hyper-parameters" are the things that have to be trained and tested against. For example, a hyper parameter of a decision tree is depth, for a neural network can be the number of hidden layers, and for a polynomial the degree of the polynomial.

To test how well these parameters are working, the model can be executed with these parameters in place, and its predictive power compared against the results of a test set as ground truth. The results of the model prediction against the test set are often compared staticially with things like recall, precision, F1 (harmonic mean based), F-Beta score (weighted F1 score), ROC cuves, and R2 scores.

QUESTION 8:

"What is the k-fold cross-validation training technique?" 
--> This refers to the approach of (ideally randomizing then) breaking up the data into k buckets, training k times, and averaging results to get a final model. From the k buckets/subsamples, a single bucket/subsample is retained as validation data for testing, while the remaining k − 1 buckets/subsamples for training. This techinique is referred to as a type of "out of order sampling" or "rotation estimation".

Because the results of all k-1 trained models are averaged, the chance that overfitting is reduced as it is unlikley that an over-fitted model will perform well on a test set and all of the training sets. In other words, to perform well on many different training sets and the final test set the model has have general predictive power than is independent of one particular data set.

It is important to randomize both data and the training process. This helps to remove any chance of that the data or the training process and its ordering will introduce bias, "leak" knoweldge of the test set into the model and causing it to overfit. When adjusting a model without the k buckets in k-fold cross-validation, there is the risk that the training process can become corrupted as we keep training and tweaking and tuning until we do well on the test set, indirectly imparting knowledge of the test set into the model. This increases the of overfitting, and means that the model may no longer generalize well to data outside of the test set.

In addition to, or instead of, k-fold cross-validation, yet another part of the dataset can be held out as "validation set". After training, the model is compared against the validation set, then finally against the test set, and has to perform well against both. However, only the validation set performance can be used for model selection and choice of hyper-parameters. The test set must be left isolated from model selection, least it become corrupted as described above.
"""

# IMPLEMENTATION: FITTING A MODEL
# The 'max_depth' parameter can be thought of as how many questions 
# the decision tree algorithm is allowed to ask about the data before making a prediction

