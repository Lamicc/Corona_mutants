#!/usr/bin/env python3
# python 3 script

import time
import h5py
import numpy as np
import tensorflow as tf
from models import *

def load_data(file):
    with h5py.File(file, 'r') as hf:
        X = hf['kmer'][:]
        Y = hf['label'][:]

    return X, Y


train_X,train_Y = load_data("train_2sam.h5")
val_X, val_Y = load_data("test_1.h5")


model = get_cnn(11,20,6)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = "log_dir", histogram_freq=1)
model.fit(train_X,train_Y, epochs=30, batch_size= 128,
                    validation_data=(val_X, val_Y),callbacks = [tensorboard_callback])
