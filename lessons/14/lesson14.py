# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:35:57 2018

@author: shane

Naive Bayes

BAYES THEOREM

Person has red sweater.
2/5 Alex
3/5 Brenda

Initially 50/50 prior, since we have no info
The 60/40 guess based on the red sweater is the posterior

Switching from what is known, to what is inferred.

Known: P(A) P(R|A)
Inferred: P(A|R)

Basically do a probability tree, then remove the events that
cannot occur or not of interest, then normalise the remaining
probabilities by the sum of the remaining probabilities so 
that they still add up to 1 after removing events.

Udacity has a really good, intuitive explaination of Baye's
theorem, the best I have seen to date.

P(A|R) = P(A)P(R|A) / norm
P(B|R) = P(B)P(R|B) / norm
where norm = P(A)P(R|A) + P(B)P(R|B)

QUIZ: FALSE POSITIVE

99 percent accuracty test
1/10,000 people are sick

P(easy|spam) = 1/3
P(money|spam) = 2/3
P(easy|ham) = 1/5
P(easy|spam) = 1/5

Go from what is known, to what is inferred.

NAIVE BAYES - CONDITIONAL PROBABILITY

Assume probabilities are independent. Makes algo simple and 
fast.

P(spam|easy,money) proportial:
    P(easy|spam)P(money|spam)P(spam)= 1/12

P(ham|easy,money) proportial:
    P(easy|ham)P(money|ham)P(spam) = 1/40

"""


