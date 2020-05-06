import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy

from load_fasta import *

# Define infiles
fasta_file = ""
relations_file = ""

# Load FASTA and one-hot encode it
sequences = fasta_to_dict(fasta_file)

# Load ancestor-descendant relations
relations = pd.read_csv(relations_file, sep="\t")

# Create character to index and index to character mappings
vocab = [' ','A','T','G','C','^','$']
char2ind = {u:i for i,u in enumerate(vocab)}
ind2char = np.array(vocab)

# Function to create a vector of integers from sequence
def seq2int(seq):
    return np.array([char2ind[c] for c in seq])

n = 128 # Process 128 nucleotides at a time
d_enc = 4 + 1 # Input dimensionality for encoder
d_dec = len(char2ind) # Input dimensionality for decoder

# Prepare training data
N = 500
n_sentences =  39178 // (n+1)
train_aux = np.zeros((N * n_sentences, 2))
train_enc_in = np.zeros((N * n_sentences, n), dtype='int32')
train_dec_in = np.zeros((N * n_sentences, n+2), dtype='int32')
train_enc_out = np.zeros((N * n_sentences, n), dtype='int32')
train_dec_out = np.zeros((N * n_sentences, n+2), dtype='int32')
j = 0
for i in np.random.randint(0, np.shape(relations)[0], size=N):
    g, d, A, D = relations.iloc[i,:].T.values
    train_aux[j:(j+1)*n_sentences,] = np.array([g, d])
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
        A_seq = A_seq + ' '*(n - len(A_seq) + 1)
        D_seq = D_seq + ' '*(n + 2 - len(D_seq) + 1)
        # Turn sequences into integer vectors
        A_seq = seq2int(A_seq)
        D_seq = seq2int(D_seq)
        # Store as encoder and decoder inputs
        train_enc_in[j*n_sentences+k,] = A_seq[:-1]
        train_dec_in[j*n_sentences+k,] = D_seq[:-1]
        # Store as encoder and decoder outputs
        train_enc_out[j*n_sentences+k,] = A_seq[1:]
        train_dec_out[j*n_sentences+k,] = D_seq[1:]
    j += 1

# Construct model
encoder_in = tf.keras.Input(shape=(n,), dtype='int32', name='encoder_input')

x = tf.keras.layers.Embedding(
    output_dim=d_enc, input_dim=d_enc, input_length=n,
    name='encoder_embedding'
)(encoder_in)

x, h, c = tf.keras.layers.LSTM(
    n, return_sequences=True, stateful=False,
    recurrent_initializer='glorot_uniform',
    name='encoder', return_state=True
)(x)

encoder_out = tf.keras.layers.Dense(
    n, activation='softmax', name='encoder_softmax'
)(x)

# Inject generations/distance into hidden state
aux_in = tf.keras.Input(shape=(2,), name='aux_input')
h = tf.keras.layers.concatenate([h, aux_in], name='aux_injection')

# Reshape to match expected LSTM initial state
h = tf.keras.layers.Dense(
    n+2, kernel_initializer='glorot_normal', name='injection_reshape_1',
    activation = 'sigmoid'
)(h)
h = tf.keras.layers.Dense(
    n, kernel_initializer='glorot_normal', name='injection_reshape_2',
    activation = 'sigmoid'
)(h)

decoder_in = tf.keras.Input(shape=(n+2,), dtype='int32', name='decoder_input')

x = tf.keras.layers.Embedding(
    output_dim=d_dec, input_dim=d_dec, input_length=n+2,
    name='decoder_embedding'
)(decoder_in)

x = tf.keras.layers.LSTM(
    n, return_sequences=True, stateful=False,
    recurrent_initializer='glorot_uniform',
    name='decoder'
)([x, h, c])

decoder_out = tf.keras.layers.Dense(
    n+2, activation='softmax', name='decoder_softmax'
)(x)

model = tf.keras.Model(
    inputs=[encoder_in, decoder_in, aux_in], outputs=[encoder_out, decoder_out]
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(
    [train_enc_in, train_dec_in, train_aux],
    [train_enc_out, train_dec_out],
    epochs=10, batch_size=128, shuffle=True
)
