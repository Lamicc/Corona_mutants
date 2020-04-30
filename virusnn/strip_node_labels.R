#!/usr/bin/env Rscript

# Usage: ./strip_node_labels.R in_tree out_tree

options(width=150)
library(phytools)

# Read command line arguments
args = commandArgs(trailingOnly=T)
in_tree = args[1]
out_tree = args[2]

# Load tree
phylo_tree = read.tree(in_tree)

# Strip node labels
phylo_tree$node.label = NULL

# Save modified tree
write.tree(phylo_tree, out_tree)
