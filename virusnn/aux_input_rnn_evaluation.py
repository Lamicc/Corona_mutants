import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
import gc

from load_fasta import *

# Function to create a vector of integers from sequence
def seq2int(seq):
    return np.array([char2ind[c] for c in seq])

# Function to create a one-hot encoding of integers
def onehot(integers, d):
    return np.eye(d)[integers]

# Define infiles
hotspots_file = "/home/johannes/proj/crse/data/2020-05-12/gisaid_cov2020.test_50.hotspots.txt"
relations_file = "/ssd/johannes/fdd3424/results/2020-05-07/gisaid_cov2020.test_50.relations.tab.gz"
fasta_file = "/ssd/johannes/fdd3424/results/2020-05-06/gisaid_cov2020.test_50.augur_seq.ali.fasta.gz"
model_dir = "/home/johannes/proj/crse/results/2020-05-12/aux_input_rnn.370.model"

hotspots_file = "/home/johannes.aspsam/gisaid_cov2020.test_50.hotspots.txt"
relations_file = "/home/johannes.aspsam/gisaid_cov2020.test_50.relations.tab.gz"
fasta_file = "/home/johannes.aspsam/gisaid_cov2020.test_50.augur_seq.ali.fasta.gz"
model_dir = "/home/johannes.aspsam/results/2020-05-11/aux_input_rnn.370.model"

# Create character to index and index to character mappings
vocab = [' ','A','T','G','C','^','$']
char2ind = {u:i for i,u in enumerate(vocab)}
ind2char = np.array(vocab)

# Load FASTA
sequences = fasta_to_dict(fasta_file)

# Load ancestor-descendant relations
relations = pd.read_csv(relations_file, sep="\t")

# Load hotspots information
hotspots = pd.read_csv(
    hotspots_file, sep="\t", names=["pos","A","C","G","T","-","N"]
)

# Load model data
model_data = tf.saved_model.load(model_dir)

# Reconstruct model
n = 16
d_enc = 5
d_dec = 7
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

# Replace weights with those trained before
model.set_weights([x.numpy() for x in model_data.trainable_variables])

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
        # character = np.random.choice(range(0, d_dec), 1, p = p)
        character = np.argmax(p)
        decoder_input = np.array(decoder_input[0])
        if t < decoder_input.shape[0]-1:
            decoder_input[t+1] = onehot(character, d_dec)
            decoder_input = tf.expand_dims(decoder_input, 0)
    return ''.join(ind2char[np.argmax(decoder_input, axis=1)])

# Determine stretches of 16 nt with and without hotspots
input_windows = np.array(
    [list(range(i, i+n)) for i in range(len(sequences['NODE_0000000'])-n+1)]
)
hotspot_pos = np.array(
    hotspots.loc[hotspots.drop(['pos'], axis=1).max(axis=1) < 0.99]['pos']
)
hotspot_windows = np.array(
    [int(len(np.intersect1d(x, hotspot_pos)) > 0) for x in input_windows]
)

# Sample 500 relations from conserved regions and mutation-prone regions
# Make predictions
def extract_seq(seqid, window):
    return sequences[seqid][min(window):(max(window)+1)].replace('-','')

prediction_samples = pd.DataFrame(
    columns=(
        'Position', 'Hotspot', 'Distance',
        'Ancestral', 'Descendant', 'Predicted'
    )
)

m = 5000

for i in range(m):
    # Report progress
    print(str(np.round(i/m*100, 1)) + "%", end="\r", flush=True)
    # Sample a relation
    j = np.random.randint(relations.shape[0])
    # Extract information about relation
    g, d, A, D = relations.iloc[j,:].T.values
    # Sample two windows; one conserved and one hotspot
    cw = np.random.choice(np.arange(hotspot_windows.size)[hotspot_windows == 0])
    mw = np.random.choice(np.arange(hotspot_windows.size)[hotspot_windows == 1])
    c_win = input_windows[cw]
    m_win = input_windows[mw]
    # Extract sequences
    ca_seq = extract_seq(A, c_win)
    cd_seq = extract_seq(D, c_win)
    ma_seq = extract_seq(A, m_win)
    md_seq = extract_seq(D, m_win)
    # Predict sequences
    cp_seq = mutate(ca_seq, d, model).split("$")[0].replace('^','')
    mp_seq = mutate(ma_seq, d, model).split("$")[0].replace('^','')
    # Save data in dataframe
    prediction_samples.loc[2*i] = [min(c_win), 0, d, ca_seq, cd_seq, cp_seq]
    prediction_samples.loc[2*i+1] = [min(m_win), 1, d, ma_seq, md_seq, mp_seq]

# Save sampled predictions
prediction_samples.to_csv(
    "/home/johannes.aspsam/gisaid_cov2020.test_50.argmax_prediction_samples.tab",
    sep = "\t", index = False
)
