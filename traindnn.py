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

import librosa
import librosa.display
import IPython.display

import matplotlib.pyplot as plt
import matplotlib.style as ms

ms.use('seaborn-muted')

def create_feature(train_df):
    rows_to_add_X = []
    rows_to_add_y = []
    print("Creating dataset")
    for feature_file, target in zip(train_df['musicFeaturesFileName'], train_df['styleID']):
        file_to_load = features_files_dir + feature_file        
        feat = np.load(file_to_load)
        for i in range(len(feat)):
            rows_to_add_X.append(feat[i])
            rows_to_add_y.append(target)
    return np.array(rows_to_add_X), np.array(rows_to_add_y)


def shuffle_data(X, y, num_of_classes = 8, test_size=0.2):
    print ("prepare dataset")
    from sklearn.utils import shuffle
    X_new, y_new = shuffle(X, y, random_state=0)

    #after that, split dataset, 20% to test set
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1, test_size=test_size)

    # then re-generate y to one-hot vector
    from keras.utils import to_categorical
    train_y = to_categorical(train_y, num_classes=num_of_classes)
    test_y = to_categorical(test_y, num_classes=num_of_classes)
    
    return train_X, test_X, train_y, test_y


def extract_test_feature(fn, offset=5, duration=20):
    sound_clip, sr = librosa.load(fn, sr=11025, offset=offset, duration=duration)
    melspec = librosa.feature.melspectrogram(y=sound_clip, sr=sr, \
                            n_fft=1024, hop_length=512, n_mels=128)
    return melspec


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


# music style mapping
style_Mapping = ["rock", "metal", "pop", "punk", "folk", "hop", "black", "country"]

def create_dataset():

    #the path where the extracted music feature files will be saved
    features_files_dir = './music/MusicFeatures/'
    #the path of the file of music's other infomation
    music_info_dir = './music/MusicInfo/'
    # read in data using pandas
    train_df = pd.read_csv (music_info_dir + "IntegratedMusicInfo.csv")
    #target values
    train_df['styleID'].unique()


    # class count
    NUM_CLASSES = len(train_df['styleID'].unique())

    X, y = create_feature(train_df)

    train_X, test_X, train_y, test_y = shuffle_data(X, y, NUM_CLASSES)

    # save to file
	np.save("train_x.npy", train_X)
	np.save("train_y.npy", train_y)
	np.save("test_x.npy", test_X)
	np.save("test_y.npy", test_y)

    return train_X, test_X, train_y, test_y, NUM_CLASSES


train_X = np.load("train_x.npy")
train_y = np.load("train_y.npy")
test_X = np.load("test_x.npy")
test_y = np.load("test_y.npy")
NUM_CLASSES = 8

#train_X, test_X, train_y, test_y, NUM_CLASSES = create_dataset()

input_size1 = train_X.shape[1]
input_size2 = train_X.shape[2]

train_X = train_X.reshape(-1, input_size1, input_size2, 1)
test_X = test_X.reshape(-1, input_size1, input_size2, 1)

model = createModel(NUM_CLASSES, input_size1, input_size2)

print("[.] Load the weights")
model.load('content_music_recommender_demo.tfl')
print("    Weights loaded.")


print("[.] Model is training")
model.fit(train_X, train_y, n_epoch=1, batch_size=32, shuffle=True, \
          validation_set=0.2, snapshot_step=5, show_metric=True)

print("    Model trained.")


print("[.] Save the weights")
model.save('content_music_recommender_demo.tfl')
print("    Weights saved.")




