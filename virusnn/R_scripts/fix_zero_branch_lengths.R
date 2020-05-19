#!/usr/bin/env Rscript

# Usage: ./fix_zero_branch_lengths.R in_tree out_tree

options(width=150)
library(phytools)

# Read command line arguments
args = commandArgs(trailingOnly=T)
in_tree = args[1]
out_tree = args[2]

# Load tree
phylo_tree = read.tree(in_tree)

# Set edges with length zero to a tenth of the minimum non-zero length
m = min(phylo_tree$edge.length[phylo_tree$edge.length > 0]) / 10
phylo_tree$edge.length[phylo_tree$edge.length == 0] = m

# Save modified tree
write.tree(phylo_tree, out_tree)
