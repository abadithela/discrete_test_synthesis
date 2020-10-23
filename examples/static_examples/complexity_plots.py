#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:01:37 2020

@author: apurvabadithela
"""
import networkx as nx
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from restrict_transitions_complex import all_augment_paths, construct_graph, run_iterations, remove_edges, to_directed, post_process_cuts, process_history
from grid_functions import construct_grid, plot_static_map, base_grid, plot_final_map, plot_augment_paths


# ==============================================================================================#
# Parametrized gridworld iterative solver:
#===============================================================================================#
Niter = 5 # No. of iterations for each example
M = [3, 5] # no. of rows
N = [3, 5] # No. of columns
lM = len(M)
lN = len(N)
time_arr_no_diag = [[] for iM in range(lM)] # Stores time taking to solve a problem
timed_out_num_no_diag = [[] for iM in range(lM)]# Stores no. of infeasible iterations
time_arr_diag = [[] for iM in range(lM)]  # Stores time taking to solve a problem
timed_out_num_diag = [[] for iM in range(lM)]  # Stores no. of infeasible iterations
iterations_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
iterations_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
total_iterations_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
total_iterations_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible

for ii in range(lM):
    iM = M[ii]
    iN = N[ii]
    nstates = iM*iN
    nP = int(np.floor(np.sqrt(nstates)-1)) # No. of propositions
    time_arr_iM_iN_no_diag = [0 for inP in range(nP)]
    timed_out_iM_iN_no_diag = [0 for inP in range(nP)]
    time_arr_iM_iN_diag = [0 for inP in range(nP)]
    timed_out_iM_iN_diag = [0 for inP in range(nP)]
    iteration_iM_iN_diag = [0 for inP in range(nP)]
    iteration_iM_iN_no_diag = [0 for inP in range(nP)]
    total_iteration_iM_iN_diag = [0 for inP in range(nP)]
    total_iteration_iM_iN_no_diag = [0 for inP in range(nP)]
    
    for inP in range(1, nP+1):
        time_avg, timed_out_avg, iteration_avg, total_iter_avg = run_iterations(iM, iN, inP, Niter, False)  # Nodiagonal transitions
        time_arr_iM_iN_no_diag[inP-1] = time_avg
        timed_out_iM_iN_no_diag[inP-1] = timed_out_avg
        iteration_iM_iN_no_diag[inP-1] = iteration_avg
        total_iteration_iM_iN_no_diag[inP-1] = total_iter_avg
        
        time_avg, timed_out_avg, iteration_avg, total_iter_avg = run_iterations(iM, iN, inP, Niter, True)  # Nodiagonal transitions
        time_arr_iM_iN_diag[inP-1] = time_avg
        timed_out_iM_iN_diag[inP-1] = timed_out_avg
        iteration_iM_iN_diag[inP-1] = iteration_avg
        total_iteration_iM_iN_diag[inP-1] = total_iter_avg  
        
    time_arr_no_diag[ii] = time_arr_iM_iN_no_diag.copy()
    timed_out_num_no_diag[ii] = timed_out_iM_iN_no_diag.copy()
    time_arr_diag[ii] = time_arr_iM_iN_diag.copy()
    timed_out_num_diag[ii] = timed_out_iM_iN_diag.copy()
    iterations_diag[ii] = iteration_iM_iN_diag.copy()
    iterations_no_diag[ii]= iteration_iM_iN_no_diag.copy()
    total_iterations_diag[ii] = total_iteration_iM_iN_diag.copy()
    total_iterations_no_diag[ii] = total_iteration_iM_iN_no_diag.copy()
        
# Plotting data:
# Figure 1: Gridworld size t vs. runtime with: 2 propositions and 4 propisitions
# fig, ax = plt.figure()
# y = time_arr_diag[2][2][2] # no. of props = 2
# x = 8
# ax.plot()
# plt.show()       
