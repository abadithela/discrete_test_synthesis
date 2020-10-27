#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:54:03 2020

@author: apurvabadithela
"""

# Breaking down restrict_transitions_complex into simple snippets of code:
import numpy as np
import time
import ipdb
import random
from grid_functions import construct_grid
import networkx as nx
import gurobipy as gp
import scipy.sparse as sp
from gurobipy import GRB
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
from aug_path import get_augmenting_path
from networkx.algorithms.flow.utils import build_residual_network
from milp_functions import keep_aug_paths_corrected, cut_aug_paths_corrected, cut_aug_paths_eff, cut_aug_paths, milp, keep_aug_paths, construct_milp_params, find_nkeep, augment_paths, sdistance, construct_c
# Main algorithm to remove edges:
# Inputs: directed graph G with capacities = 1 on all edges, p: struct that maps propositions with nodes
# and Q0, set of initial conditions. If none, then Q0 is found.
# Returns: C (set of edges to be restricted), Q0 (set of initial conditions to start the test run)

def remove_edges(G, props, Q0=None):
    Gf, node_q0, find_Q0 = check_Q0(G, Q0)
    Gin = Gf.copy()
    alg_fail_main = False
    C = []
    n = len(props)-1 # No. of propositions to cover + goal
    assert(n>0)
    i = n-1 # For indexing
    Chist = [] # History of cuts
    discard=False
    niterations = np.zeros((len(props), 1))
    nkeep = 0
    while i > 0 and n>=2:  # so that it's not just p1 and g
        # Setting edges to constrian pi:
        C, Chisti, discard_i, nkeep, niter_pi, alg_fail = remove_pi_edges(Gin, props, i, C) # Input Gf already has input C edges removed from it
        if alg_fail:
            alg_fail_main = True
            break
        if discard_i:
            discard = discard_i
            break
        niterations[i] = niter_pi # No. of iterations for blocking pi
        Gin = Gf.copy() # Copying the whole original graph
        Gin.remove_edges_from(C) # Removing all the edges found by remove_pi_edges
        Chist.append(Chisti) # Appending cuts for the ith node
        i -= 1 # Decreasing i
        print(C)
        
    # When i = 0, its time for constraints to be placed for p1, and if find_Q0 =1, to find the set of initial conditions
    if find_Q0:
        Q0, aug_path_nodes = pre_p1(Gin, props) # Finding predecessors of p1 (p[0]) not on augmenting paths from p1 to p2
        Gq0, node_q0, find_Q0 = check_Q0(Gin, Q0)
    else:
        Gq0 = Gin.copy()
        
    # Reduce q0 to p1 weight to 1, if it exists: Why?
    if ((node_q0, props[0][0]) in Gq0.edges):
        errant_edge = (node_q0, props[0][0])
        Gq0.remove_edge(*errant_edge)
        Gq0.add_edge(errant_edge[0], errant_edge[1], capacity=1.0)
        
    assert(find_Q0 == 0) # Now, that initial condition points have been found, this should be 0
    # Representing all augmenting paths with one node
    Gq0.add_node("q0_aug_nodes")
    for q in aug_path_nodes:
        Gq0.add_edge(q, "q0_aug_nodes") # Edge considered to have infinite capacity
    aug_nodes = "q0_aug_nodes"
    
    if(node_q0 != props[0][0]): # Don't find cut edges if pi-1 is not the same as pi.
        p0 = [[node_q0], props[0], [aug_nodes]]
        C, Chist0, discard_i, nkeep_q0, n_iter, alg_fail = remove_pi_edges_Q0(Gq0, p0, 1, C)
        if alg_fail:
            alg_fail_main = True
            
        if discard_i:
            discard = discard_i
        else:
            C = process_Q0_cuts(Gq0, aug_path_nodes, C) # processing cuts
            Chist_new0 = [] # Processed q0 nodes with clean edges only
            for Cii in Chist0:
                Cnew_ii = process_Q0_cuts(Gq0, aug_path_nodes, Cii)
                Chist_new0.append(Cnew_ii)
            Chist.append(Chist_new0)
    else:
        niterations[0] = 0 # node p1 is the same as q0
    # else:
    #     if(node_q0 != props[0][0]):
    #         p0 = [[node_q0], props[0], [aug_nodes]]
    #         C, Chist0, discard_i, nkeep_q0, n_iter, alg_fail = remove_pi_edges_Q0(Gin, p0, 1, C)
    #         if alg_fail:
    #             alg_fail_main = True
    #         if discard_i:
    #             discard = discard_i
    #         else:
    #             Chist.append(Chist0) # Appending the cuts for the last node
    #             niterations[0] = n_iter # Adding no. of iterations
    #     else:
    #         niterations[0] = 0 # node p1 is the same as q0
    if alg_fail_main:
        C = []
        Q0 = []
        Chist = []
        nkeep = []
        niterations = []
    return C, Q0, Chist, discard, nkeep, niterations, alg_fail_main


# Function to remove edges for proposition pi, and adding newly removed edges to C
def remove_pi_edges(G, props, i, C):
    n = len(props)-1
    Gi = G.copy()
    Gk = G.copy()
    Gi.remove_node(props[i][0]) 
    Chisti = []
    discard = False
    Pcut, n_paths_to_cut = cut_aug_paths_eff(Gi, props, i)    
    Pkeep, MC_keep, alg_fail = keep_aug_paths(Gk, props, i)   
    n_iterations = 0                   
    while np.sum(n_paths_to_cut) > 0:
        A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = construct_milp_params(Pcut,Pkeep, MC_keep, n)
        newC, timed_out, feas = milp(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e) # Getting new cuts from the MILP
        n_iterations+=1 # Adding no. of iterations
        if timed_out==True:
            discard = "time_out"
            break

        C = update(G, C, newC) # Updating cuts
        Chisti.append(newC)
        # Prepare for next iteration of cuts for pi:
        Gi = G.copy()
        Gk = G.copy()
        Gi.remove_node(props[i][0]) 
        Gi.remove_edges_from(C)
        Gk.remove_edges_from(C)
        Pcut, n_paths_to_cut = cut_aug_paths_eff(Gi, props, i)    
        Pkeep, MC_keep, alg_fail = keep_aug_paths(Gk, props, i)
    nkeep = find_nkeep(Pkeep)
    return C, Chisti, discard, nkeep, n_iterations, alg_fail

# Function to remove edges for proposition pi, and adding newly removed edges to C
def remove_pi_edges_Q0(G, props, i, C):
    n = len(props)-1
    Gi = G.copy()
    Gk = G.copy()
    Gi.remove_node(props[i][0]) 
    discard = False
    Pcut, n_paths_to_cut = cut_aug_paths_eff(Gi, props, i)    
    Pkeep, MC_keep, alg_fail = keep_aug_paths(Gk, props, i)   
    Chist0 = [] 
    niterations = 0                  
    while np.sum(n_paths_to_cut) > 0:
        A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = construct_milp_params(Pcut,Pkeep, MC_keep, n)
        A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = remove_q0_edges(A_cut, n_cut, A_keep, D_keep, n_keep, cost,e)
        newC, timed_out, feas = milp(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e) # Getting new cuts from the MILP
        niterations+=1 # Adding iterations
        if timed_out==True:
            discard = "time_out"
            break

        C = update(G, C, newC) # Updating cuts
        Chist0.append(newC)
        
        # Prepare for next iteration of cuts for pi:
        Gi = G.copy()
        Gk = G.copy()
        Gi.remove_node(props[i][0]) 
        Gi.remove_edges_from(C)
        Gk.remove_edges_from(C)
        Pcut, n_paths_to_cut = cut_aug_paths_eff(Gi, props, i)    
        Pkeep, MC_keep, alg_fail = keep_aug_paths(Gk, props, i)
    nkeep = find_nkeep(Pkeep)
    return C, Chist0, discard, nkeep, niterations, alg_fail

# Checks of initial condition set is provided; if not signals to the main 
# restrict_transitions function that it needs to be created
def check_Q0(G, Q0):
    Gf = G.copy()
    if Q0 is None:
        find_Q0 = 1
        node_q0 = None
    else:
        find_Q0 = 0
        assert(q0 in G.nodes for q0 in Q0)
        if len(Q0)==1:
            node_q0 = Q0[0]
        else:
            Gf.add_node("q0")
            for q0 in Q0:
                Gf.add_edge("q0", q0) # Edge considered to have infinite capacity
            node_q0 = "q0"
    return Gf, node_q0, find_Q0

# Returns nodes on all augmented paths:
def all_aug_path_nodes(G, props):
    aug_path_nodes = []
    for ii in range(len(props)-1):
        pi = props[ii]
        pi_next = props[ii+1]
        P = augment_paths(G, pi, pi_next)
        if aug_path_nodes == []:
            new_aug_path_nodes = list(set().union(*P)) # Nodes on paths P
            aug_path_nodes = new_aug_path_nodes.copy()
        else:
            new_aug_path_nodes = list(set().union(*P))
            aug_path_nodes = add_nodes(aug_path_nodes, new_aug_path_nodes)
    assert(aug_path_nodes!=[])
    return aug_path_nodes

# Function returning predecessors of p1 that are not on aug paths from p1 to p2, p2 to p3, ...
def pre_p1(G, props):
    aug_path_nodes = all_aug_path_nodes(G, props)
    # p2 = props[1].copy() # Second proposition on the graph
    # g = props[-1].copy() # Goal node on the graph
    # nodes_closer_goal = sdistance(G, p2, g)
    # aug_path_nodes= add_nodes(aug_path_nodes, nodes_closer_goal)
    G0 = G.copy()
    # Remove nodes that are closer to the goal than p2:
    p1 = props[0].copy()
    if p1[0] in aug_path_nodes:
        aug_path_nodes.remove(p1[0]) # Don't want to remove the source vertex
    else:
        print("Something fishy:")
        print(aug_path_nodes)
        print(p1[0])
    G0.remove_nodes_from(aug_path_nodes)
    ancestor_tree = nx.dfs_tree(G0.reverse(), source=p1[0]).reverse() # Finding all upstream nodes that could lead to G0 but not via the aug path nodes
    pred_p1_init = list(ancestor_tree.nodes)
    pred_p1=pred_p1_init.copy()
    # Removing auxiliary vertices:
    for v in pred_p1_init:
        if (v[0:2]=="an"):
            pred_p1.remove(v)
    assert(pred_p1 != [])
    return pred_p1, aug_path_nodes

# Function returning predecessors of p1 that are not on aug paths from p1 to p2, p2 to p3, ...
# Sat 5:56 pm: Trying to make the initial condition Q0 set better.
def pre_p1_better(G, props):
    aug_path_nodes = []
    for ii in range(len(props)-1):
        pi = props[ii]
        pi_next = props[ii+1]
        P = augment_paths(G, pi, pi_next)
        if aug_path_nodes == []:
            new_aug_path_nodes = list(set().union(*P)) # Nodes on paths P
            aug_path_nodes = new_aug_path_nodes.copy()
        else:
            new_aug_path_nodes = list(set().union(*P))
            aug_path_nodes = add_nodes(aug_path_nodes, new_aug_path_nodes)
    p2 = props[1].copy() # Second proposition on the graph
    g = props[-1].copy() # Goal node on the graph
    # nodes_closer_goal = sdistance(G, p2, g)
    # aug_path_nodes= add_nodes(aug_path_nodes, nodes_closer_goal)
    G0 = G.copy()
    # Remove nodes that are closer to the goal than p2:
    p1 = props[0].copy()
    if p1[0] in aug_path_nodes:
        aug_path_nodes.remove(p1[0]) # Don't want to remove the source vertex
    else:
        print("Something fishy:")
        print(aug_path_nodes)
        print(p1[0])
    # G0.remove_nodes_from(aug_path_nodes)
    ancestor_tree = nx.dfs_tree(G0.reverse(), source=p1[0]).reverse() # Finding all upstream nodes that could lead to G0 but not via the aug path nodes
    
    pred_p1_init = list(ancestor_tree.nodes)
    pred_p1=pred_p1_init.copy()
    # Removing auxiliary vertices:
    for v in pred_p1_init:
        if (v[0:2]=="an"):
            pred_p1.remove(v)
    for v in aug_path_nodes:
        if v in pred_p1_init:
            pred_p1.remove(v)
    assert(pred_p1 != [])
    return pred_p1

# Adding nodes from nodes_closer_goal to aug_path_nodes without repeating elements:
def add_nodes(aug_path_nodes, nodes_closer_goal):
    new_aug_path_nodes = aug_path_nodes.copy()
    for node in nodes_closer_goal:
        if node not in aug_path_nodes:
            new_aug_path_nodes.append(node)
    return new_aug_path_nodes

# Returning nodes not on path from p1 to p2:
def pre_p1_old(G, p1, p2, g):
    P = augment_paths(G, p1, p2)
    aug_path_nodes = list(set().union(*P)) # Nodes on paths P
    G0 = G.copy()
    # Remove nodes that are closer to the goal than p2:
    nodes_closer_goal = sdistance(G, p2, g)
    aug_path_nodes.remove(p1[0]) # Don't want to remove the source vertex
    aug_path_nodes = add_nodes(aug_path_nodes, nodes_closer_goal) # Finding nodes closer to goal
    G0.remove_nodes_from(aug_path_nodes)
    ancestor_tree = nx.dfs_tree(G0.reverse(), source=p1[0]).reverse() # Finding all upstream nodes that could lead to G0 but not via the aug path nodes
    pred_p1_init = list(ancestor_tree.nodes)
    pred_p1=pred_p1_init.copy()
    # Removing auxiliary vertices:
    for v in pred_p1_init:
        if (v[0:2]=="an"):
            pred_p1.remove(v)
    assert(pred_p1 != [])
    return pred_p1

# Remove q0-->v edges where v is in Q0: 
# Removing vectors associated with q0 ---> v
def remove_q0_edges(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e):
    # Find q0-->v indices in e:
    q0_idx_e = []
    for ii in range(len(e)):
        ei = e[ii]
        if(ei[0]=="q0"):
            q0_idx_e.append(ii)
    if(q0_idx_e):
        new_e = [e[ii] for ii in range(len(e)) if ii not in q0_idx_e]
        nA_cut = []
        nA_keep = []
        nD_keep = []
        nn_cut = []
        nn_keep = []
        for jj in range(len(n_cut)):
            njj = n_cut[jj]
            nnjj = njj
            new_Ai = np.array([])
            Ai = A_cut[jj].copy()
            # Converting to 2d array
            if Ai.ndim == 1:
                Ai=Ai[:,np.newaxis]
            for k in range(len(e)):
                if k not in q0_idx_e:
                    if new_Ai.size != 0:
                        new_Ai = np.column_stack((new_Ai, Ai[:,k]))
                    else:
                        new_Ai = Ai[:,k]
            # Verify there is still a path to cut:
            for r in range(njj):
                if not new_Ai[r,:].any():
                  nnjj-=1
                
            # Remove rows with zeros:
            if new_Ai.ndim>1:
                new_Ai = new_Ai[~np.all(new_Ai == 0, axis=1)]
            if nnjj != 0:
                nn_cut.append(nnjj)
                nA_cut.append(new_Ai)
        
        for jj in range(len(n_keep)):
            njj = n_keep[jj]
            nnjj = njj
            new_Ai = np.array([])
            new_Di = np.array([])
            Ai = A_keep[jj].copy()
            Di = D_keep[jj].copy()
            # Converting to 2d array
            if Ai.ndim == 1:
                Ai=Ai[:,np.newaxis]
            # Converting to 2d array
            if Di.ndim == 1:
                Di=Di[:,np.newaxis]
            for k in range(len(e)):
                if k not in q0_idx_e:
                    if new_Ai.size != 0:
                        new_Ai = np.column_stack((new_Ai, Ai[:,k]))
                    else:
                        new_Ai = Ai[:,k]
            new_Di = new_Ai @ np.ones((len(new_e),1))
                    
            # Verify there is still a path to cut:
            #for r in range(njj):
            #    if not new_Ai[r,:].any():
            #      nnjj-=1
            # Remove rows with all zeros:
            #if new_Ai.ndim>1:
            #    new_Ai = new_Ai[~np.all(new_Ai == 0, axis=1)]
            #    new_Di = new_Di[~np.all(new_Di == 0, axis=1)]
            
            if nnjj!=0:      
                nn_keep.append(nnjj)
                nA_keep.append(new_Ai)
                nD_keep.append(new_Di)
        new_cost = construct_c(nn_keep)
    else:
        nA_cut = A_cut.copy()
        new_e = e.copy()
        nn_cut = n_cut.copy()
        nA_keep = A_keep.copy()
        nD_keep = D_keep.copy()
        nn_keep = n_keep.copy()
    return nA_cut, nn_cut, nA_keep, nD_keep, nn_keep, new_cost, new_e

# Updating cuts:
def update(G, C, newC):
    for new_cut_edge in newC:
        assert(new_cut_edge not in C) # sanity check to make sure we're not cutting edges that have already been cut
    if newC:
        R = build_residual_network(G, capacity="capacity")
        for edge in newC:
            if R[edge[0]][edge[1]]['capacity'] == 1:
                C.append(edge)
    return C

# ============================================= Corrected Version ====================================== #
# Corrected versions of the remove_edges algorithm to accommodate all transitions cuts:
# Corrected version of removing edges:

# Function to remove "q0_aug_nodes" and process cuts:
def process_Q0_cuts(Gq0, aug_path_nodes, C):
    Cnew = []
    for e in C:
        if (e[1]=="q0_aug_nodes"):
            for anode in aug_path_nodes:
                candidate_edge = (e[0], anode)
                if candidate_edge in Gq0:
                    Cnew.append(candidate_edge)
        elif (e[0]=="q0_aug_nodes"):
            for anode in aug_path_nodes:
                candidate_edge = (anode, e[1])
                if candidate_edge in Gq0:
                    Cnew.append(candidate_edge)
        else:
            Cnew.append(e)
    return Cnew

def remove_edges_corrected(G, props, Q0=None):
    Gf, node_q0, find_Q0 = check_Q0(G, Q0)
    Gin = Gf.copy()
    alg_fail_main = False
    C = []
    n = len(props)-1 # No. of propositions to cover + goal
    assert(n>0)
    i = n-1 # For indexing
    Chist = [] # History of cuts
    discard=False
    niterations = np.zeros((len(props), 1))
    nkeep = 0
    if i > 0 and n>=2:  # so that it's not just p1 and g
        # Setting edges to constrian pi:
        C, Chisti, discard_i, nkeep, niter_pi, alg_fail = remove_pi_edges_corrected(Gin, props, i, C) # Input Gf already has input C edges removed from it
        if alg_fail:
            alg_fail_main = True
        if discard_i:
            discard = discard_i
        niterations[i] = niter_pi # No. of iterations for blocking pi
        Gin = Gf.copy() # Copying the whole original graph
        Gin.remove_edges_from(C) # Removing all the edges found by remove_pi_edges
        Chist.append(Chisti) # Appending cuts for the ith node
        i -= 1 # Decreasing i
        print(C)
        
    # When i = 0, its time for constraints to be placed for p1, and if find_Q0 =1, to find the set of initial conditions
    if not (alg_fail_main or discard):
        if find_Q0:
            Q0, aug_path_nodes = pre_p1(Gin, props) # Finding predecessors of p1 (p[0]) not on augmenting paths from p1 to p2
            Gq0, node_q0, find_Q0 = check_Q0(Gin, Q0)
        else:
            Gq0 = Gin.copy()
            aug_path_nodes = all_aug_path_nodes(Gin, props)
            p1 = props[0].copy()
            if p1[0] in aug_path_nodes:
                aug_path_nodes.remove(p1[0]) # Don't want to remove the source vertex
            
        # Reduce q0 to p1 weight to 1, if it exists: Why?
        if ((node_q0, props[0][0]) in Gq0.edges):
            errant_edge = (node_q0, props[0][0])
            Gq0.remove_edge(*errant_edge)
            Gq0.add_edge(errant_edge[0], errant_edge[1], capacity=1.0)
            
        assert(find_Q0 == 0) # Now, that initial condition points have been found, this should be 0
        # Representing all augmenting paths with one node
        Gq0.add_node("q0_aug_nodes")
        for q in aug_path_nodes:
            Gq0.add_edge(q, "q0_aug_nodes") # Edge considered to have infinite capacity
        aug_nodes = "q0_aug_nodes"
        
        if(node_q0 != props[0][0]): # Don't find cut edges if pi-1 is not the same as pi.
            p0 = [[node_q0], props[0], [aug_nodes]]
            C, Chist0, discard_i, nkeep_q0, n_iter, alg_fail = remove_pi_edges_Q0_corrected(Gq0, p0, 1, C)
            if alg_fail:
                alg_fail_main = True
            elif discard_i:
                discard = discard_i
                alg_fail_main = True
            else:
                C = process_Q0_cuts(Gq0, aug_path_nodes, C) # processing cuts
                Chist_new0 = [] # Processed q0 nodes with clean edges only
                for Cii in Chist0:
                    Cnew_ii = process_Q0_cuts(Gq0, aug_path_nodes, Cii)
                    Chist_new0.append(Cnew_ii)
                Chist.append(Chist_new0)
        else:
            niterations[0] = 0 # node p1 is the same as q0
    else:
        C = []
        Q0 = []
        Chist = []
        nkeep = []
        niterations = []
    return C, Q0, Chist, discard, nkeep, niterations, alg_fail_main

# Function to remove edges for proposition pi, and adding newly removed edges to C
def remove_pi_edges_corrected(G, props, i, C):
    n = len(props)-1
    Gi = G.copy()
    Gk = G.copy()
    Gi.remove_node(props[i][0]) 
    Chisti = []
    discard = False
    Pcut, n_paths_to_cut = cut_aug_paths_corrected(G, props)    
    Pkeep, MC_keep, alg_fail = keep_aug_paths_corrected(Gk, props)   
    n_iterations = 0   
    MAX_ITER = 100                
    while np.sum(n_paths_to_cut) > 0:
        A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = construct_milp_params(Pcut,Pkeep, MC_keep, n)
        newC, timed_out, feas = milp(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e) # Getting new cuts from the MILP
        n_iterations+=1 # Adding no. of iterations
        if timed_out==True:
            discard = "time_out"
            break
        if feas==True:
            discard="infeasible"
            break
        if n_iterations > MAX_ITER:
            discard = "time_out"
            break
        C = update(G, C, newC) # Updating cuts
        Chisti.append(newC)
        # Prepare for next iteration of cuts for pi:
        Gi = G.copy()
        Gk = G.copy()
        # Gi.remove_node(props[i][0]) 
        Gi.remove_edges_from(C)
        Gk.remove_edges_from(C)
        Pcut, n_paths_to_cut = cut_aug_paths_corrected(Gi, props)    
        Pkeep, MC_keep, alg_fail = keep_aug_paths_corrected(Gk, props)
    nkeep = find_nkeep(Pkeep)
    return C, Chisti, discard, nkeep, n_iterations, alg_fail

# Function to remove edges for proposition pi, and adding newly removed edges to C
def remove_pi_edges_Q0_corrected(G, props, i, C):
    n = len(props)-1
    Gi = G.copy()
    Gk = G.copy()
    Gi.remove_node(props[i][0]) 
    discard = False
    Pcut, n_paths_to_cut = cut_aug_paths_corrected(G, props)    
    Pkeep, MC_keep, alg_fail = keep_aug_paths_corrected(Gk, props)   
    Chist0 = [] 
    niterations = 0      
    MAX_ITER = 100            
    while np.sum(n_paths_to_cut) > 0:
        A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = construct_milp_params(Pcut,Pkeep, MC_keep, n)
        A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = remove_q0_edges(A_cut, n_cut, A_keep, D_keep, n_keep, cost,e)
        
        newC, timed_out, feas = milp(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e) # Getting new cuts from the MILP
        niterations+=1 # Adding iterations
        if timed_out==True:
            discard = "time_out"
            break
        if feas==True:
            discard="infeasible"
            break
        if niterations > MAX_ITER:
            discard = "time_out"
            break
        C = update(G, C, newC) # Updating cuts
        Chist0.append(newC)
        
        # Prepare for next iteration of cuts for pi:
        Gi = G.copy()
        Gk = G.copy()
        # Gi.remove_node(props[i][0]) 
        Gi.remove_edges_from(C)
        Gk.remove_edges_from(C)
        Pcut, n_paths_to_cut = cut_aug_paths_corrected(Gi, props)    
        Pkeep, MC_keep, alg_fail = keep_aug_paths_corrected(Gk, props)
    nkeep = find_nkeep(Pkeep)
    return C, Chist0, discard, nkeep, niterations, alg_fail