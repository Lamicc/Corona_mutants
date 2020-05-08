import numpy as np

def fasta_to_dict(infile):
    """Read FASTA file (can be gzip-compressed) and return dictionary."""
    # Initialize FASTA dictionary
    fasta = {}
    # Check if infile is gzip
    if infile.endswith(".gz"):
        # Open gzip infile
        import gzip
        infile = gzip.open(infile, 'rt')
    else:
        # Open regular infile
        infile = open(infile, 'rt')
    # Read the first line of FASTA infile
    line = infile.readline()
    # Iterate over lines of FASTA infile
    while line:
        # If the line starts with a ">", save the sequence ID
        if line.startswith(">"):
            # Clean up sequence ID
            seqid = line.split()[0].lstrip(">")
            # Initialize empty sequence entry in FASTA dictionary
            fasta[seqid] = ""
        # Otherwise it is a sequence line
        else:
            # Add the sequence line to the FASTA dictionary entry
            fasta[seqid] += line.strip()
        # Read the next line
        line = infile.readline()
    # Close infile
    infile.close()
    # Return FASTA dictionary after reading all lines of the infile
    return fasta

def chars_in_dict(seqdict):
    """Return ordered list of characters in dictionary of sequences."""
    # Initialize set of characters
    characters = set([])
    # Iterate over sequences
    for seqid in seqdict.keys():
        # Update the set of characters with those in the current sequence
        characters.update(set(list(seqdict[seqid])))
    # Return characters as an ordered list
    return sorted(list(characters))

def seq_one_hot(sequence, characters):
    """One-hot encode a sequence of characters. Each column is one character."""
    # Create dictionary linking characters to indices
    char_to_ind = dict(zip(characters, range(len(characters))))
    # Set up matrix of zeros
    M = np.zeros((len(characters),len(sequence)))
    # Iterate over tuples of character index and sequence index
    for i in zip([char_to_ind[c[0]] for c in sequence], range(len(sequence))):
        # Set cell corresponding to character to one
        M[i[0],i[1]] = 1
    # Return finished one-hot encoding
    return M


def marginal_seq_one_hot(fasta_dict, characters=None):
    """ Creates a marginal one-hot sequence matrix. It adds up each sample's matrix.
    Rows correspond to nucleotides, columns correspond to genomic locations. """

    print("Creating marginal of one-hot sequence matrix...")
    if characters is None:
        characters = ['A', 'C', 'G', 'T', '-', 'N']

    dict_keys = list(fasta_dict.keys())
    num_samples = len(dict_keys)

    # Initialize the matrix with the first sample
    marginal_seq = seq_one_hot(fasta_dict[dict_keys[0]], characters=characters)
    # Add the rest of the samples to the matrix
    for key_idx in range(1, num_samples):
        marginal_seq = marginal_seq + seq_one_hot(fasta_dict[dict_keys[key_idx]], characters=characters)
    return marginal_seq


if False:
    infile = 'data/2020-05-07/gisaid_cov2020.test_20.augur_seq.ali.fasta.gz'
    print("\nConverting ", infile, " into dict...")
    fasta_file = fasta_to_dict(infile)
    print("\tNumber of keys in dict: ", len(list(fasta_file.keys())))

    marginal_seq = marginal_seq_one_hot(fasta_file)
    print("\tmarginal_seq.shape: ", marginal_seq.shape)
    print(marginal_seq)

