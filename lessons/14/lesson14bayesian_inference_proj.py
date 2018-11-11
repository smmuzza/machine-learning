# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 09:52:56 2018

@author: shane
"""

import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])

# Output printing out first 5 rows
df.head()

'''
Solution
'''
df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
df.head() # returns (rows, columns)

'''
Solution:
'''
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
print(documents)
for i in documents:
    # TODO
    lower_case_documents.append(i.lower())
print(lower_case_documents)

'''
Solution:
'''
sans_punctuation_documents = []
import string

for i in lower_case_documents:
    # TODO
    sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
    
print(sans_punctuation_documents)

'''
Solution:
'''
preprocessed_documents = []
for i in sans_punctuation_documents:
    # TODO
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)

'''
Solution
'''
frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)
pprint.pprint(frequency_list)



'''
Solution
'''
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words='english')

'''
Practice node:
Print the 'count_vector' object which is an instance of 'CountVectorizer()'
'''
print(count_vector)

'''
Solution:
'''
count_vector.fit(documents)
count_vector.get_feature_names()

'''
Solution
'''
doc_array = count_vector.transform(documents).toarray()
print(doc_array)

'''
Solution
'''
frequency_matrix = pd.DataFrame(doc_array, 
                                columns = count_vector.get_feature_names())
print(frequency_matrix)


'''
Solution

NOTE: sklearn.cross_validation will be deprecated soon to sklearn.model_selection 
'''
# split into training and testing sets
# USE from sklearn.model_selection import train_test_split to avoid seeing deprecation warning.
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

'''
[Practice Node]

The code for this segment is in 2 parts. Firstly, we are learning a vocabulary dictionary for the training data 
and then transforming the data into a document-term matrix; secondly, for the testing data we are only 
transforming the data into a document-term matrix.

This is similar to the process we followed in Step 2.3

We will provide the transformed data to students in the variables 'training_data' and 'testing_data'.
'''

'''
Solution
'''
# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

"""
P(D) is the probability of a person having Diabetes. 
It's value is 0.01 or in other words, 1% of the general population 
has diabetes(Disclaimer: these values are assumptions and are not 
reflective of any medical study).

P(Pos) is the probability of getting a positive test result.

P(Neg) is the probability of getting a negative test result.

P(Pos|D) is the probability of getting a positive result on a test 
done for detecting diabetes, given that you have diabetes. 
This has a value 0.9. In other words the test is correct 90% of 
the time. This is also called the Sensitivity or True Positive Rate.

P(Neg|~D) is the probability of getting a negative result on a test 
done for detecting diabetes, given that you do not have diabetes. 
This also has a value of 0.9 and is therefore correct, 90% of the time.
 This is also called the Specificity or True Negative Rate.

Putting our values into the formula for Bayes theorem we get:

`P(D|Pos) = P(D) * P(Pos|D) / P(Pos)`
OR
`P(D|Pos) * P(Pos) = P(D) * P(Pos|D) `
OR
`P(D|Pos) / P(Pos|D) = P(D) / P(Pos) `

"""

'''
Instructions:
Calculate probability of getting a positive test result, P(Pos)
'''

'''
Solution (skeleton code will be provided)
'''
# P(D)
p_diabetes = 0.01

# P(~D)
p_no_diabetes = 0.99

# Sensitivity or P(Pos|D)
p_pos_diabetes = 0.9

# Specificity or P(Neg|~D)
p_neg_no_diabetes = 0.9

# P(Pos)
p_pos = (p_diabetes * p_pos_diabetes) + (p_no_diabetes * (1 - p_neg_no_diabetes))
print('The probability of getting a positive test result P(Pos) is: {}',format(p_pos))
Sensitivity = p_pos_diabetes
Specificity = p_neg_no_diabetes
p_pos = (p_diabetes * Sensitivity) + (p_no_diabetes * (1 - Specificity)) # TODO
print('The probability of getting a positive test result P(Pos) is: {}',format(p_pos))

'''
Solution
'''
# P(D|Pos)
p_diabetes_pos = (p_diabetes * p_pos_diabetes) / p_pos # TODO
print('Probability of an individual having diabetes, given that that individual got a positive test result is:\
',format(p_diabetes_pos)) 

'''
Solution
'''
# P(Pos|~D)
p_pos_no_diabetes = 0.1

# P(~D|Pos)
p_no_diabetes_pos = p_no_diabetes * p_pos_no_diabetes / p_pos # TODO
print ('Probability of an individual not having diabetes, given that that individual got a positive test result is:'\
,p_no_diabetes_pos)

"""
Naive Bayes' is an extension of Bayes' theorem that assumes 
that all the features are independent of each other.
"""

'''
Instructions: Compute the probability of the words 'freedom' and 
'immigration' being said in a speech, or
P(F,I).

The first step is multiplying the probabilities of Jill Stein giving 
a speech with her individual 
probabilities of saying the words 'freedom' and 'immigration'. 
Store this in a variable called p_j_text

The second step is multiplying the probabilities of Gary Johnson 
giving a speech with his individual 
probabilities of saying the words 'freedom' and 'immigration'. 
Store this in a variable called p_g_text

The third step is to add both of these probabilities and you will
 get P(F,I).
'''

'''
Solution: Step 1
'''
# P(J)
p_j = 0.5

# P(F/J)
p_j_f = 0.1

# P(I/J)
p_j_i = 0.1

p_j_text = p_j * p_j_f * p_j_i
print(p_j_text)

'''
Solution: Step 2
'''
# P(G)
p_g = 0.5

# P(F/G)
p_g_f = 0.7

# P(I/G)
p_g_i = 0.2

p_g_text = p_g * p_g_f * p_g_i
print(p_g_text)

'''
Solution: Step 3: Compute P(F,I) and store in p_f_i
'''
p_f_i = p_j_text + p_g_text
print('Probability of words freedom and immigration being said are: ', format(p_f_i))

'''
Solution
'''
p_j_fi = p_j_text / p_f_i
print('The probability of Jill Stein saying the words Freedom and Immigration: ', format(p_j_fi))

'''
Solution
'''
p_g_fi = p_g * (p_g_f * p_g_i) / p_f_i
print('The probability of Gary Johnson saying the words Freedom and Immigration: ', format(p_g_fi))

'''
Instructions:

We have loaded the training data into the variable 'training_data' and the testing data into the 
variable 'testing_data'.

Import the MultinomialNB classifier and fit the training data into the classifier using fit(). Name your classifier
'naive_bayes'. You will be training the classifier using 'training_data' and y_train' from our split earlier. 
'''

'''
Solution
'''
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB() # TODO
naive_bayes.fit(training_data, y_train)

'''
Instructions:
Now that our algorithm has been trained using the training data set we can now make some predictions on the test data
stored in 'testing_data' using predict(). Save your predictions into the 'predictions' variable.
'''

'''
Solution
'''
predictions = naive_bayes.predict(testing_data)        

'''
** Accuracy ** measures how often the classifier makes the correct 
prediction. It’s the ratio of the number of correct predictions to 
the total number of predictions (the number of test data points).

** Precision ** tells us what proportion of messages we classified 
as spam, actually were spam. It is a ratio of true positives(words 
classified as spam, and which are actually spam) to all positives(all 
words classified as spam, irrespective of whether that was the 
correct classification), in other words it is the ratio of

[True Positives/(True Positives + False Positives)]

** Recall(sensitivity)** tells us what proportion of messages that 
actually were spam were classified by us as spam. It is a ratio of 
true positives(words classified as spam, and which are actually spam)
 to all the words that were actually spam, in other words it is the 
 ratio of

[True Positives/(True Positives + False Negatives)]

Instructions:
Compute the accuracy, precision, recall and F1 scores of your model using your test data 'y_test' and the predictions
you made earlier stored in the 'predictions' variable.
'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
        