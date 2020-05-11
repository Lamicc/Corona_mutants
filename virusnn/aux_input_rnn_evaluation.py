import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
import gc

from load_fasta import *

# Create character to index and index to character mappings
vocab = [' ','A','T','G','C','^','$']
char2ind = {u:i for i,u in enumerate(vocab)}
ind2char = np.array(vocab)

# Make predictions
def mutate(sequence, distance, model):
    # Turn the sequence into the expected encoder input
    encoder_input = np.eye(d_enc)[seq2int(sequence + ' '*(n - len(sequence)))]
    encoder_input = tf.expand_dims(encoder_input, 0)
    # Turn the distance into the expected auxiliary input
    auxiliary_input = np.array([distance])
    auxiliary_input = tf.expand_dims(auxiliary_input, 0)
    # Prepare an initial decoder input
    decoder_input = np.eye(d_dec)[np.array([char2ind["^"]] * (n+2))]
    decoder_input = tf.expand_dims(decoder_input, 0)
    # Evaluate
    for t in range(decoder_input.shape[1]):
        pred = model.predict([encoder_input, decoder_input, auxiliary_input])
        p = np.array(pred[-1][t])
        character = np.random.choice(range(0, d_dec), 1, p = p)
        decoder_input = np.array(decoder_input[0])
        if t < decoder_input.shape[0]-1:
            decoder_input[t+1] = onehot(character, d_dec)
            decoder_input = tf.expand_dims(decoder_input, 0)
    return ''.join(ind2char[np.argmax(decoder_input, axis=1)])

# Interesting data and location?
i = 736903
g, d, A, D = relations.iloc[i,:].T.values
start_pos = 4500
end_pos = 4516

sequence = sequences[A][start_pos:end_pos].replace('-','')
