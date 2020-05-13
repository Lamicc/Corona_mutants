#!/usr/bin/env python3
# python 3 script

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from load_fasta import fasta_to_dict

base2code_dna = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5, '-':0}
def seq2code(seq):
    return [base2code_dna[x] for x in seq]

def get_ancestral_list(sample_id, tsvfile):
    df = pd.read_csv(tsvfile, delimiter = "\t")
    ndf = df.loc[df['Descendant'] == sample_id]
    return ndf.Ancestor

# 5mer x 30 generations
def get_ancestral_seq(sample_id, n_gen, seq_dict, ancestor_list):
    # sample, pos, seq, target
    df = pd.DataFrame(columns=['sample','pos','history_seq','label'])
    sample_seq = seq_dict[sample_id]
    n_pos = len(sample_seq)
    for i in range(n_pos):
        ancestral_seq = ""
        for anc in ancestor_list[:n_gen]:
            ancestral_seq = seq_dict[anc][i] + ancestral_seq
        target = seq2code(sample_seq)[i]
        ancestral_seq = seq2code(ancestral_seq)
        onehot_bases = tf.one_hot(ancestral_seq, 6)
        df = df.append({'sample': sample_id, 'pos':i,
                        'history_seq':onehot_bases, 'label':target}, ignore_index=True)
    return df


def extract_kmer_feature(sample_id, k, ancestral_df,df ):
    mid_base = int(k/2)
    for i in range(len(ancestral_df)-4):
        kmer = ancestral_df['history_seq'][i:i+k].values
        kmer_stack = tf.stack(kmer)
        target = ancestral_df.loc[ancestral_df['pos'] == (i+mid_base)]['label'].values[0]
        df = df.append({'sample': sample_id, 'pos':i+mid_base,
                        'kmer':kmer_stack, 'label':target}, ignore_index=True)
    return df


def preprocess(tsvfile, fasta_file):
    n_gen = 10
    k = 5
    sample_id = 'EPI_ISL_402121'
    ancestor_list = get_ancestral_list(sample_id, tsvfile)
    seq_dict = fasta_to_dict(fasta_file)
    ans_df = get_ancestral_seq(sample_id, n_gen, seq_dict, ancestor_list)
    df = pd.DataFrame(columns=['sample','pos','kmer','label'])
    df = extract_kmer_feature(sample_id,k, ans_df,df)
    sample = [i for i in df['sample'].values]
    kmer = [i for i in df['kmer'].values]
    pos = [i for i in df['pos'].values]
    label = [i for i in df['label'].values]

    file_name = 'test_80.h5'

    with h5py.File(file_name, 'w') as hf:
        #hf.create_dataset('sample',  data=np.stack(sample))
        hf.create_dataset('kmer',  data=np.stack(kmer))
        hf.create_dataset('pos',  data=np.stack(pos))
        hf.create_dataset('label',  data=np.stack(label))

    return None

preprocess("gisaid_cov2020.train_80.relations.tab.gz", "gisaid_cov2020.train_80.augur_seq.ali.fasta.gz")
