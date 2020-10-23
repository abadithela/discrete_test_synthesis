#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 01:19:48 2020

@author: apurvabadithela
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:32:59 2020

@author: apurvabadithela
"""
# Generates plots for large number of examples:

import networkx as nx
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from restrict_transitions_complex import all_augment_paths, construct_graph, run_iterations, remove_edges, to_directed, post_process_cuts, process_history
# from restrict_transitions import augment_paths, min_cut_edges, remove_edges, to_directed, post_process_cuts, process_history
from grid_functions import construct_grid, plot_static_map, base_grid, plot_final_map, plot_augment_paths

# Parameters:
ex = 2

if ex == 1:
    M = 3 # Number of rows
    N = 3 # Number of columns
    g = "n"+str(M*N) # Goal state
    n = 2 # Number of propositions
    props = [["n7"], ["n3"]] # Propositions: [[p1], [p2]]
    props.append([g])
    
if ex == 2:
    M = 5 # Number of rows
    N = 10 # Number of columns
    g = "n"+str(M*N) # Goal state
    n = 4 # Number of propositions
    props = [["n17"],["n14"]] # , ["n3"], ["n15"]] # Propositions: [[p1], [p2]]
    # props = [["n23"]] # Propositions: [[p1], [p2]]
    props.append([g])
    
if ex == 3:
    M = 10 # Number of rows
    N = 10 # Number of columns
    g = "n"+str(M*N) # Goal state
    n = 4 # Number of propositions
    props = [["n85"], ["n33"], ["n27"], ["n69"]] # Propositions: [[p1], [p2]]
    props = [["n59"], ["n89"]] # Propositions: [[p1], [p2]]

    # props = [["n23"]] # Propositions: [[p1], [p2]]
    props.append([g])

# Constructing graphs:
G = nx.DiGraph()
nodes = ["n"+str(ii+1) for ii in range(M*N)]
G.add_nodes_from(nodes)
allow_diag = False # True means that diagonal transitions are allowed
G = construct_grid(G, M, N, allow_diag)
Gdir, aux_vertices_count = to_directed(G)

# Find test graph:
t = time.time()
C, Q0, Chist, discard, nkeep, ncount = remove_edges(G, props)

if discard==False:
    print(C)
    postC = post_process_cuts(C, Gdir)
    print(postC)
    postChist = process_history(Chist, Gdir)
else:
    postC = []
    postChist = [[] for ii in range(len(props))]
# post_Chist = process_history(Chist, Gdir)
elapsed = time.time() - t

print("Time taken: ")
print(elapsed)

# Plotting figures:
fig, ax = base_grid(M,N, props)
FIG, AX = plot_static_map(G, M, N, props, Q0, postChist)
fig_final, ax_final = plot_final_map(G,M,N,props,Q0,postC)

# Plotting with augmented paths on the final graph:
test_graph = G.remove_edges_from(postC)
Paug = all_augment_paths(G, props)
fig_aug, ax_aug = plot_augment_paths(Paug, G, M, N, props, Q0, postC)
plt.show()  # Showing plots

