import os
import re
import sys
import numpy as np
import pandas as pd

import pyaudio
import wave

import tensorflow as tf
import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Define the optimization methods.
sgd = tflearn.optimizers.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=100)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
rmsprop = tflearn.optimizers.RMSProp(learning_rate=0.03, decay=0.999)
adam = tflearn.optimizers.Adam(learning_rate=0.001, beta1=0.99)

def createModel(nbClasses=8, imageSize1=128, imageSize2=431):
    
    print("[.] Creating DNN model")
    
    convnet = input_data(shape=[None, imageSize1, imageSize2, 1], name='input')

    convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 512, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='elu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, nbClasses, activation='softmax')
    convnet = regression(convnet, optimizer=rmsprop, loss='categorical_crossentropy')
    
    music_model = tflearn.DNN(convnet)
    
    print("    Model created.")

    return music_model


train_X = np.load("train_x.npy")
train_y = np.load("train_y.npy")
test_X = np.load("test_x.npy")
test_y = np.load("test_y.npy")
NUM_CLASSES = 8

input_size1 = train_X.shape[1]
input_size2 = train_X.shape[2]

train_X = train_X.reshape(-1, input_size1, input_size2, 1)
test_X = test_X.reshape(-1, input_size1, input_size2, 1)

model = createModel(NUM_CLASSES, input_size1, input_size2)

print("[.] Load the weights")
model.load('content_music_recommender.tfl')
print("    Weights loaded.")

accuracy=model.evaluate(test_X,test_y)
print("Accuracy:", accuracy)

prediction = model.predict(test_X)

print (tf.confusion_matrix(test_y, prediction, num_classes=NUM_CLASSES))


