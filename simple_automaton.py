#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:51:09 2020

@author: apurvabadithela
"""
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from restrict_transitions import augment_paths, min_cut_edges, remove_edges
# Constructing graph induced by automaton:
G = nx.DiGraph()
G.add_edge("n1", "n2", capacity=1.0)
G.add_edge("n2", "n3", capacity=1.0)
G.add_edge("n2", "n4", capacity=1.0)
G.add_edge("n2", "n5", capacity=1.0)
G.add_edge("n3", "n6", capacity=1.0)
G.add_edge("n4", "n6", capacity=1.0)
G.add_edge("n5", "n6", capacity=1.0)
G.add_edge("n6", "n7", capacity=1.0)

# Specifying the goal and propositions:
g ="n7"
w = "n3"
q0 = "n1"

P = augment_paths(G, [q0], [g])
MC = min_cut_edges(G, [q0], [g])
print(P)
print(MC)

C, Q0 = remove_edges(G, [[w], [g]], [q0])
print(C)


