#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
import gc

from load_fasta import *

# Usage: ./aux_input_rnn.py fasta_file felations_file outdir
fasta_file, relations_file, outdir = sys.argv[1:]

# Create character to index and index to character mappings
vocab = [' ','A','T','G','C','^','$']
char2ind = {u:i for i,u in enumerate(vocab)}
ind2char = np.array(vocab)

# Function to create a vector of integers from sequence
def seq2int(seq):
    return np.array([char2ind[c] for c in seq])

# Function to create a one-hot encoding of integers
def onehot(integers, d):
    return np.eye(d)[integers]

# Load FASTA and one-hot encode it
sequences = fasta_to_dict(fasta_file)

# Load ancestor-descendant relations
relations = pd.read_csv(relations_file, sep="\t")

n = 16 # Process n nucleotides at a time
d_enc = 4 + 1 # Input dimensionality for encoder
d_dec = len(vocab) # Input dimensionality for decoder

# Construct model
lstm_units = 128

# The encoder takes an ancestral sequence in onehot format
encoder_in = tf.keras.Input(
    shape=(n, d_enc), dtype='float32', name='encoder_input'
)

# An encoder LSTM predicts the next character in the ancestral sequence
x, h, c = tf.keras.layers.LSTM(
    lstm_units, return_sequences=True, stateful=False,
    recurrent_initializer='glorot_uniform',
    name='encoder', return_state=True
)(encoder_in)

# # The encoder LSTM outputs softmax values for loss calculation
# encoder_out = tf.keras.layers.Dense(
#     d_enc, activation='softmax', name='encoder_softmax'
# )(x)

# A time proxy (tree distance) is injected into the context vectors
aux_in = tf.keras.Input(shape=(1,), name='aux_input')
h = tf.keras.layers.concatenate([h, aux_in], name='aux_injection_h')
c = tf.keras.layers.concatenate([c, aux_in], name='aux_injection_c')

# The time proxy is mixed into the context vectors through two dense networks
h = tf.keras.layers.Dense(
    lstm_units+1, kernel_initializer='glorot_normal', name='injection_h_reshape_1'
)(h)
h = tf.keras.layers.Dropout(0.2)(h)
h = tf.keras.layers.Dense(
    lstm_units, kernel_initializer='glorot_normal', name='injection_h_reshape_2'
)(h)

c = tf.keras.layers.Dense(
    lstm_units+1, kernel_initializer='glorot_normal', name='injection_c_reshape_1'
)(c)
c = tf.keras.layers.Dropout(0.2)(c)
c = tf.keras.layers.Dense(
    lstm_units, kernel_initializer='glorot_normal', name='injection_c_reshape_2'
)(c)

# The decoder takes a descendant sequence formatted as integers
decoder_in = tf.keras.Input(shape=(n+2,d_dec), dtype='float32', name='decoder_input')

# A decoder LSTM predicts the next character in the descendant sequence
x, _, _ = tf.keras.layers.LSTM(
    lstm_units, return_sequences=True, stateful=False,
    recurrent_initializer='glorot_uniform',
    name='decoder', return_state=True
)([decoder_in, h, c])

# The decoder LSTM outputs softmax values for loss calculation and prediction
decoder_out = tf.keras.layers.Dense(
    d_dec, activation='softmax', name='decoder_softmax'
)(x)

# Assemble the model from the specified parts
model = tf.keras.Model(
    inputs=[encoder_in, decoder_in, aux_in], outputs=[decoder_out]
)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()

# Train the model using a small number of relations at a time to fit RAM
m = 64

# The genome allows for a certain number of overlapping sentences
n_sentences = len(sequences['NODE_0000000']) - n

# Randomly shuffle the relations indices
relation_indices = list(relations.index.values)
np.random.shuffle(relation_indices)

# Function to train on subset of relations
def train_on_subset(model, M):
    # Determine which relations of count 'm' are to be trained upon
    wanted_relations = relation_indices[(M*m):(M*m+m)]
    # Determine the number of relations (fewer in last round)
    N = len(wanted_relations)
    # Initialize training data arrays
    train_aux = np.zeros((N * n_sentences, 1))
    train_enc_in = np.zeros((N * n_sentences, n, d_enc), dtype='float32')
    train_dec_in = np.zeros((N * n_sentences, n+2, d_dec), dtype='float32')
    train_dec_out = np.zeros((N * n_sentences, n+2, d_dec), dtype='float32')
    # Initialize relation counter to zero
    j = 0
    # Iterate over relations indices 'i'
    for i in wanted_relations:
        # Extract information about the j:th relation
        g, d, A, D = relations.iloc[i,:].T.values
        # Every part of the genome will have the same distance value
        train_aux[j:(j+1)*n_sentences,] = np.array([d])
        # Iterate over sentences 'k' with length 'n' in the genome
        for k in range(n_sentences):
            # Make deep copy of sequences in order to manipulate them
            A_seq = deepcopy(sequences[A])[(k*n):(k*n+n+1)]
            D_seq = deepcopy(sequences[D])[(k*n):(k*n+n+1)]
            # Remove gaps
            A_seq = A_seq.replace('-','')
            D_seq = D_seq.replace('-','')
            # Add start and end tokens to decoder input
            D_seq = '^' + D_seq + '$'
            # Pad sequences to desired length
            A_seq += ' '*(n - len(A_seq) + 1)
            D_seq += ' '*(n + 2 - len(D_seq) + 1)
            # Turn sequences into integer vectors
            A_seq = seq2int(A_seq)
            D_seq = seq2int(D_seq)
            # Store as encoder and decoder inputs
            train_enc_in[j*n_sentences+k, :, :] = onehot(A_seq[:-1], d_enc)
            train_dec_in[j*n_sentences+k, :, :] = onehot(D_seq[:-1], d_dec)
            # Store as decoder output
            train_dec_out[j*n_sentences+k, :, :] = onehot(D_seq[1:], d_dec)
        j += 1
        print(
            "Formatting data: " + str(np.round(j/m*100, 1)) + "%",
            end="\r", flush=True
        )
    # Fit the model using training data
    _ = model.fit(
        [train_enc_in, train_dec_in, train_aux],
        [train_dec_out],
        epochs=1, batch_size=256, shuffle=True
    )
    return model

# Iterate over sets of relations
for M in range(int(np.ceil(relations.shape[0] / m))):
    if not M % 10:
        model.save(outdir + "/aux_input_rnn." + str(M) + ".model")
    model = train_on_subset(model, M)
    _ = gc.collect()

# Save final trained model
model.save(outdir + "/aux_input_rnn." + str(M) + ".model")
