#!/usr/bin/env Rscript

# Usage: ./reroot_tree.R in_tree outgroup_id out_tree

options(width=150)
library(phytools)

# Read command line arguments
args = commandArgs(trailingOnly=T)
in_tree = args[1]
outgroup_id = args[2]
out_tree = args[3]

# Load tree
phylo_tree = read.tree(in_tree)

# Reroot tree at outgroup edge
phylo_tree = reroot(phylo_tree, which(phylo_tree$tip.label == outgroup_id))

# Save rerooted tree
write.tree(phylo_tree, out_tree)
