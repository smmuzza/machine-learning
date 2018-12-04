# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:22:40 2018

@author: shane

LESSON 22: CLustering Mini-Project

There are several ways of choosing the number of clusters, k. 
We'll look at a simple one called "the elbow method". 
The elbow method works by plotting the ascending values of k versus the
 total error calculated using that k.
 
 More on Collaborative Filtering

    This is a simplistic recommendation engine that shows the most basic idea of "collaborative filtering". There are many heuristics and methods to improve it. The Netflix Prize tried to push the envelope in this area by offering a prize of US$1,000,000 to the recommendation algorithm that shows the most improvement over Netflix's own recommendation algorithm.
    That prize was granted in 2009 to a team called "BellKor's Pragmatic Chaos". This paper shows their approach which employed an ensemble of a large number of methods.
    Netflix did not end up using this $1,000,000 algorithm because their switch to streaming gave them a dataset that's much larger than just movie ratings -- what searches did the user make? What other movies did the user sample in this session? Did they start watching a movie then stop and switch to a different movie? These new data points offered a lot more clues than the ratings alone.

Take it Further

    This notebook showed user-level recommendations. We can actually use the almost exact code to do item-level recommendations. These are recommendations like Amazon's "Customers who bought (or viewed or liked) this item also bought (or viewed or liked)". These would be recommendations we can show on each movie's page in our app. To do this, we simple transpose the dataset to be in the shape of Movies X Users, and then cluster the movies (rather than the users) based on the correlation of their ratings.
    We used the smallest of the datasets Movie Lens puts out. It has 100,000 ratings. If you want to dig deeper in movie rating exploration, you can look at their Full dataset containing 24 million ratings.

 
 TODO copy CODE HERE!

"""

