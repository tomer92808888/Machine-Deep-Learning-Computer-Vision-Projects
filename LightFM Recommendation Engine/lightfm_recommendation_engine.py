# -*- coding: utf-8 -*-
"""LightFM Recommendation Engine

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-AGrG7hUy-eXTrYR7jMjyhrftEl9T1S7

# Using LightFM for Recommendations

Check out [LightFM here](https://lyst.github.io/lightfm/docs/index.html) and view it's [documentation here](http://lyst.github.io/lightfm/docs/home.html) 

LightFM is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.

It also makes it possible to incorporate both item and user metadata into the traditional matrix factorization algorithms. It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

The details of the approach are described in the LightFM paper, available on [arXiv](http://arxiv.org/abs/1507.08439).
"""

# Install lightFM, takes around 15 seconds
!pip install lightfm

"""The first step is to get the Movielens data. This is a classic small recommender dataset, consisting of around 950 users, 1700 movies, and 100,000 ratings. The ratings are on a scale from 1 to 5, but we’ll all treat them as implicit positive feedback in this example.

Fortunately, this is one of the functions provided by LightFM itself.
"""

# Import our modules
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Use one of LightFM's inbuild datasets, setting the minimum rating to return at over 4.0
data = fetch_movielens(min_rating = 4.0)
data

# Get our key and value from our dataset
# By printing it, we see it's comprised of a data segments containing test, train, item_features, item_feature_labels & item_labels 
for key, value in data.items():
    print(key, type(value), value.shape)

# What type of data are we working with? coo_matrix
type(data['train'])

# Each row represents a user, and each column an item. 
# We use .tocsr() to view it as a Compressed Sparse Row format, it's an inbuilt function in the coo_matrix object
m1 = data['train'].tocsr()

print(m1[0,0])
print(m1[0,1])

"""**coo_matrix - A sparse matrix in COOrdinate format - Intended Usage:**

- COO is a fast format for constructing sparse matrices
- Once a matrix has been constructed, convert to CSR or CSC format for fast arithmetic and matrix vector operations
- By default when converting to CSR or CSC format, duplicate (i,j) entries will be summed together.  This facilitates efficient construction of finite element matrices and the like. (see example)
"""

print(repr(data['train'])) # rept() is used in debugging to get a string representation of object
print(repr(data['test']))

"""# Let's now create and train our model

**Four loss functions are available:**

- **logistic**: useful when both positive (1) and negative (-1) interactions are present.
- **BPR**: Bayesian Personalised Ranking pairwise loss. Maximises the prediction difference between a positive example and a randomly chosen negative example. Useful when only positive interactions are present and optimising ROC AUC is desired.
- **WARP**: Weighted Approximate-Rank Pairwise loss. Maximises the rank of positive examples by repeatedly sampling negative examples until rank violating one is found. Useful when only positive interactions are present and optimising the top of the recommendation list (precision@k) is desired.
- **k-OS WARP**: k-th order statistic loss. A modification of WARP that uses the k-th positive example for any given user as a basis for pairwise updates.

**Two learning rate schedules are available:**
- adagrad
- adadelta
"""

# Creat our model object from LightFM
# We specify the loss type to be WARP (Weighted Approximate-Rank Pairwise )
model = LightFM(loss = 'warp')

# Extract our training and test datasets
train = data['train']
test = data['test']

# Fit our model over 10 epochs
model.fit(train, epochs=10)

"""# Performance Evaluation

We use Precision and AUC to avaluate our model performance.

**The ROC AUC metric for a model**: the probability that a randomly chosen positive example has a higher score than a randomly chosen negative example. A perfect score is 1.0.

**The precision at k metric for a model**: the fraction of known positives in the first k positions of the ranked list of results. A perfect score is 1.0.
"""

# Evaluate it's performance
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

"""We got 
# Let's see what movies are recommended for some users
"""

def sample_recommendation(model, data, user_ids):
    '''uses model, data and a list of users ideas and outputs the recommended movies along with known positives for each user'''
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        scores = model.predict(user_id, np.arange(n_items))

        top_items = data['item_labels'][np.argsort(-scores)]
      
        print("User %s" % user_id)
        print("Known positives:")
        
        # Print the first 3 known positives
        for x in known_positives[:3]:
            print("%s" % x)
        
        # Print the first 3 recommended movies
        print("Recommended:")
        for x in top_items[:3]:
            print("%s" % x)
        print("\n")

# Testing on users 6, 125 and 336
sample_recommendation(model, data, [6, 125, 336])