#!/usr/bin/env python3
# python 3 script

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def get_cnn(k, n_gen, embedding_size):

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(k, n_gen, embedding_size)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(tf.keras.layers.Dense(6, activation='relu', use_bias=False))

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        print(model.summary())
        return model


class CNN():

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def __init__(self):
        super(CNN, self).__init__()
        self.input = tf.keras.layers.InputLayer(input_shape=(d1,d2))
        self.conv1d = tf.keras.layers.Conv1D(256, 3, activation='relu')
        self.local1d = tf.keras.layers.LocallyConnected1D(128, 3, activation='relu')
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(4, activation='relu', use_bias=False)

    def call(self, inputs):
        x = self.input(inputs)
        x = self.conv1d(x)
        x = self.local1d(x)
        x = self.conv1d(x)
        x = self.local1d(x)
        x = self.pooling(x)
        x = self.dropout(x)

        return self.dense(x)

    def compile(self):
        self.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return None
