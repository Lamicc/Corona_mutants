import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy

from load_fasta import *

# Define infiles
fasta_file = "/ssd/johannes/fdd3424/results/2020-05-06/gisaid_cov2020.train_50.augur_seq.ali.fasta.gz"
relations_file = "/ssd/johannes/fdd3424/results/2020-05-07/gisaid_cov2020.train_50.relations.tab.gz"

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

# The encoder takes an ancestral sequence formatted as integers
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


# Prepare training data
close = np.array(relations[relations["Generations"] == 1].index)
far = np.array(relations[relations["Distance"] > 0.08].index)
wanted_relations = np.unique(np.concatenate([close, far]))
np.random.shuffle(wanted_relations)
wanted_relations = wanted_relations[:1000]
N = len(wanted_relations)
gs = 0
ge = len(sequences['NODE_0000000']) // n
genome_sentence_range = range(gs, ge)
n_sentences = len(genome_sentence_range)
train_aux = np.zeros((N * n_sentences, 1))
train_enc_in = np.zeros((N * n_sentences, n, d_enc), dtype='float32')
train_dec_in = np.zeros((N * n_sentences, n+2, d_dec), dtype='float32')
train_enc_out = np.zeros((N * n_sentences, n, d_enc), dtype='float32')
train_dec_out = np.zeros((N * n_sentences, n+2, d_dec), dtype='float32')
j = 0
# for i in np.random.randint(0, np.shape(relations)[0], size=N):
# for i in range(N):
for i in wanted_relations:
    g, d, A, D = relations.iloc[i,:].T.values
    train_aux[j:(j+1)*n_sentences,] = np.array([d])
    for k in genome_sentence_range:
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
        train_enc_in[j*n_sentences+k-gs, :, :] = onehot(A_seq[:-1], d_enc)
        train_dec_in[j*n_sentences+k-gs, :, :] = onehot(D_seq[:-1], d_dec)
        # Store as encoder and decoder outputs
        train_enc_out[j*n_sentences+k-gs, :, :] = onehot(A_seq[1:], d_enc)
        train_dec_out[j*n_sentences+k-gs, :, :] = onehot(D_seq[1:], d_dec)
    j += 1


# Fit the model using training data
model.fit(
    [train_enc_in, train_dec_in, train_aux],
    [train_dec_out],
    epochs=20, batch_size=256, shuffle=True
)

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
            decoder_input[t+1] = np.eye(d_dec)[character]
            decoder_input = tf.expand_dims(decoder_input, 0)
    return ''.join(ind2char[np.argmax(decoder_input, axis=1)])

# Interesting data and location?
i = 736903
g, d, A, D = relations.iloc[i,:].T.values
start_pos = 4500
end_pos = 4516

sequence = sequences[A][start_pos:end_pos].replace('-','')
