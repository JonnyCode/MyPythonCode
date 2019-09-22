# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% import modules
import tensorflow as tf
import numpy as np


# %% import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# %% plot eg from mnist training set
import matplotlib.pyplot as plt 
l = int(np.sqrt(mnist.train.images.shape[1]))
EgImage = np.reshape(mnist.train.images[1,:],(l,l))
plt.imshow(EgImage, cmap='gray')


# %% linear regression with keras

from keras.models import Sequential

model = Sequential()

from keras.layers import Dense

model.add(Dense(units=10, activation='softmax', input_dim=mnist.train.images.shape[1]))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(mnist.train.images, mnist.train.labels, epochs=10, batch_size=32) #trains

score = model.evaluate(mnist.test.images, mnist.test.labels, batch_size=32) # test
print("fraction correct =",  score[1])

# %% MLP with keras
# Image (784 dimensions) -> fully connected layer (500 hidden units) -> nonlinearity (ReLU) -> 
# fully connected layer (100 hidden units) -> nonlinearity (ReLU) -> (2x2 max pool) -> 
#fully connected (256 hidden units) -> nonlinearity (ReLU) -> fully connected (10 hidden units) -> softmax

model = Sequential()
model.add(Dense(units=500, activation='relu', input_dim=mnist.train.images.shape[1]))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(mnist.train.images, mnist.train.labels, epochs=10, batch_size=32) #trains

score = model.evaluate(mnist.test.images, mnist.test.labels, batch_size=32) # test
print("fraction correct =",  score[1])

# %% MLP with keras and early stopping on validation set

import keras

model = Sequential()
model.add(Dense(units=500, activation='relu', input_dim=mnist.train.images.shape[1]))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(monitor='categorical_crossentropy',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')

model.fit(mnist.train.images, mnist.train.labels, callbacks=es, validation_split=0.2, epochs=10, batch_size=32) #trains

score = model.evaluate(mnist.test.images, mnist.test.labels, batch_size=32) # test
print("fraction correct =",  score[1])



