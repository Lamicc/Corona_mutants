#!/usr/bin/env python3

import sys

# Usage: ./clean_up_seqids.py fasta_in tree_in fasta_out tree_out
fasta_in, tree_in, fasta_out, tree_out = sys.argv[1:]

with open(fasta_in, 'r') as fi, open(tree_in, 'r') as ti, \
    open(fasta_out, 'w') as fo, open(tree_out, 'w') as to:

    # Load tree text
    tree = ti.read()

    # Read in FASTA IDs while cleaning it and tree up
    for line in fi.readlines():
        if line.startswith(">"):
            # Identify ID
            header = line.lstrip(">")
            id = line.split("|")[1]
            # Replace FASTA header with ID in output FASTA
            fo.write(">" + id + "\n")
            # Replace FASTA header with ID in tree
            tree = tree.replace(header.strip(), id)
        else:
            fo.write(line)

    # Save output tree
    to.write(tree)
