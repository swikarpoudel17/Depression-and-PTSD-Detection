#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: spoudel
"""

import pickle
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


## Loading train and validation data
def load_data(train_path, val_path):
    with open(train_path, 'rb') as handle:
        tweets_train = pickle.load(handle)
    with open(val_path, 'rb') as handle:
        tweets_val = pickle.load(handle)
    
    ## Removing a dataset which has condition as condition
    ## To be safe, from both train and val set
    tweets_train = [x for x in tweets_train if x[0]['condition'] != 'condition']
    tweets_val = [x for x in tweets_val if x[0]['condition'] != 'condition']
    
    tweets_train = [x for x in tweets_train if len(x) >= 10]
    tweets_val = [x for x in tweets_val if len(x) >= 10]
    
    ## Extracting conditions separately for each user
    conditions = [u[0]['condition'] for u in tweets_train]
    val_conditions = [u[0]['condition'] for u in tweets_val]
    
    return tweets_train, tweets_val, conditions, val_conditions

## Loading test data only
def load_test_data(test_path):
    with open(test_path, 'rb') as handle:
        tweets_test = pickle.load(handle)
    
    ## Removing a dataset which has condition as condition
    tweets_test = [x for x in tweets_test if x[0]['condition'] != 'condition']
    
    ## Extracting conditions separately for each user
    test_conditions = [u[0]['condition'] for u in tweets_test]
    
    return (tweets_test, test_conditions)


## Encoding labels to represent in integer format
def encode_labels(conditions, val_conditions, device):
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(conditions)
    conditions_labels = torch.tensor(integer_labels, dtype=torch.long).to(device)
    
    integer_labels = label_encoder.fit_transform(val_conditions)
    val_conditions_labels = torch.tensor(integer_labels, dtype=torch.long).to(device)
    
    return conditions_labels, val_conditions_labels


## Encoding labels to represent in integer format
def encode_test_labels(test_conditions, device):
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(test_conditions)
    test_conditions_labels = torch.tensor(integer_labels, dtype=torch.long).to(device)    
    return test_conditions_labels


## Computing class weights since there are difference in the number of samples between control, depression and PTSD
def compute_class_weights(conditions):
    unique_conditions, class_sizes = np.unique(conditions, return_counts=True)
    total_samples = len(conditions)
    class_weights = [total_samples / size for size in class_sizes]
    class_weights_normalized = np.array(class_weights) / sum(class_weights)
    return np.float32(class_weights_normalized)
