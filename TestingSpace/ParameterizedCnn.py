#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:35:45 2019

@author: jcafaro

This script is designed to fit and test a parameterized CNN, where filters are 
constrained to be mixture of gaussians (defined by 2 gaussians defined by cov
matricies and amplitudes)
"""

# %% import modules

import numpy as np
import tensorflow as tf

# %% 

# filter model - mixture of gaussians
def MixGuass(Fshape,amp,cov):
    # Fshape is 1x2, amp is 1xN, cov is 2x2xN, where N is number of filters
    if np.shape(Fshape)==(0,1) ;
    
    for i in range(np.shape(amp))
    Gauss1 = np.random.multivariate_normal([filterWidth/2,filterWidth/2],[VarP1,CovP1;CovP1,VarP1])
    
    

# Create flat input vector
x_fc = tf.placeholder(tf.float32,[None, 784])

# Create weight matrix variable
W = tf.Variable(tf.truncated_normal([784,10],stddev=0.1))

# Create bias variable
b = tf.Variable(tf.zeros([10]))

# Apply fully connected layer
y_preact = tf.matmul(x_fc, W)+b
y = tf.nn.relu(y_preact)


