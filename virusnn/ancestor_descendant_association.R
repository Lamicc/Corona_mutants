#!/usr/bin/env Rscript

# Usage: ./ancestor_descendant_association.R in_tree n_threads out_table

options(width=150)
library(tidyverse)
library(phytools)
library(foreach)
library(doMC)

# Read command line arguments
args = commandArgs(trailingOnly=T)
in_tree = args[1]
n_threads = args[2]
out_table = args[3]

# Register number of parallel threads
registerDoMC(n_threads)

# Load tree
phylo_tree = read.tree(in_tree)

# Define function to return node or tip label
get_label = function (n) {
  # Determine the number of tips
  m = length(phylo_tree$tip.label)
  if (n > m) {
    # If the node number exceeds the number of tips, it refers to a node
    n = n - length(phylo_tree$tip.label)
    return(phylo_tree$node.label[n])
  } else {
    # Otherwise it refers to a tip
    return(phylo_tree$tip.label[n])
  }
}

# Create table of edges, representing ancestor-descendant relations
edge_table = as_tibble(phylo_tree$edge) %>%
  # Rename columns
  rename(AncestralNode = V1, DescendantNode = V2) %>%
  # Add edge lengths
  mutate(Distance = phylo_tree$edge.length) %>%
  # Add sequence IDs
  mutate(
    Ancestor = sapply(AncestralNode, get_label),
    Descendant = sapply(DescendantNode, get_label)
  ) %>%
  # Remove superfluous information
  select(-AncestralNode, -DescendantNode)

# Get every possible ancestor-descendant relation
all_relations = bind_rows(
  foreach(descendant = edge_table$Descendant) %dopar% {
    # Start at a gap of one generation
    generations = 1
    # Get the relation between descendant and its immediate ancestor
    relation = filter(edge_table, Descendant == descendant)
    # Determine the ID of the ancestor
    ancestor = relation$Ancestor
    # Determine the distance between descendant and its immediate ancestor
    distance = relation$Distance
    # Set up a list of relations over generations
    relations = list(
      tibble(
        Generations = generations,
        Distance = distance,
        Ancestor = ancestor,
        Descendant = descendant
      )
    )
    # As long as the ancestor is not the root...
    while (ancestor != phylo_tree$node.label[1]) {
      # Get the relation of the ancestor and its ancestor
      relation = filter(edge_table, Descendant == ancestor)
      # Determine the ID of the ancestor
      ancestor = relation$Ancestor
      # Determine the distance between descendant and the earlier ancestor
      distance = distance + relation$Distance
      # Increment the number of generations
      generations = generations + 1
      # Save relation between the original descendant and the earlier ancestor
      relations[[generations]] = tibble(
        Generations = generations,
        Distance = distance,
        Ancestor = ancestor,
        Descendant = descendant
      )
    }
    return(bind_rows(relations))
  }
)

# Sort table and save it
write_tsv(arrange(all_relations, Descendant, Generations), out_table)
