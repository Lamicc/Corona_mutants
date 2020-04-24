#!/usr/bin/env python3
# python 3 script

import tensorflow as tf
from tensorflow.keras.layers import *

class CNN(tf.keras.Model):

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
