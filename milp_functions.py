#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:28:18 2020

@author: apurvabadithela
"""


# File contains MILP functions that are used by the restrict_transitions_simplified.py class

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


# Finding number of augmented paths in final graph:
def find_nkeep(Pkeep):
    ni = [0 for Pi in Pkeep]
    for ii in range(len(Pkeep)):
        Pi = Pkeep[ii]
        ni[ii] = len(Pi)
    nkeep=ni.copy()
    return nkeep

# Function that searches for augmented paths on a graph from source p[j<i] to pi+1 that bypasses waypoint p[i]
 # If Pcut_j is empty, there are no augmenting paths from pj to pi+1
def cut_aug_paths(G, props, i):
    Pcut = []
    n_paths_to_cut = []
    for j in range(i):
        n_paths_to_cut_j = 0
        Pcut_j = augment_paths(G, props[j], props[i+1]) # From pj to pi+1; which is index by pi
        if Pcut_j:
            n_paths_to_cut_j+=len(Pcut_j) # If there are paths to cut, update n_paths_to_cut
        Pcut.append(Pcut_j)
        n_paths_to_cut.append(n_paths_to_cut_j)
    return Pcut, n_paths_to_cut

# Corrected version: Cuts all paths from pj to pi+1
def cut_aug_paths_corrected(Gi, props):
    Pcut = []
    n_paths_to_cut = []
    N = len(props)
    for i in range(1, N-1):
        G= Gi.copy()
        G.remove_node(props[i][0])
        Pcut_new = []
        n_paths_to_cut_ji = []
        for j in range(i):
            # n_paths_to_cut_ji = []
            Pcut_j = augment_paths(G, props[j], props[i+1]) # From pj to pi+1; which is index by pi
            Pcut_j_new = Pcut_j.copy()
            if (j < i-1): # checking if any of pj+1 to pi-1 are on aug path; if so, remove them
                higher_props = [props[k][0] for k in range(j+1, i)]   
                for Pj in Pcut_j:
                    higher_prop_present = any(ph in Pj for ph in higher_props)
                    if higher_prop_present:
                        Pcut_j_new.remove(Pj)
            if Pcut_j_new:
                n_paths_to_cut_ji.append(len(Pcut_j_new)) # If there are paths to cut, update n_paths_to_cut
            
            Pcut_new.append(Pcut_j_new)
        Pcut.extend(list(reversed(Pcut_new)))
        n_paths_to_cut.extend(list(reversed(n_paths_to_cut_ji)))
    return Pcut, n_paths_to_cut

# function that searches for augmented paths and min-cut edges that need to be kept
def keep_aug_paths(G, props, i):
    Pkeep = []
    MC_keep = []
    alg_fail = False
    for jj in range(i+1):
        Pkeep_j = augment_paths(G, props[jj], props[jj+1])            
        try:
            assert(Pkeep_j != []) # Shouldn't be empty. there should always be a path from pj to pj+1; otherwise example is invalid
        except AssertionError as e:
            alg_fail = True
            break
        Pkeep.append(Pkeep_j)
        MCj = min_cut_edges(G, props[jj], props[jj+1])
        MC_keep.append(MCj)
    return Pkeep, MC_keep, alg_fail

# Function that searches for augmented paths for all indices; all at once
def keep_aug_paths_corrected(G, props):
    Pkeep = []
    MC_keep = []
    alg_fail = False
    n = len(props)-1
    for jj in range(n):
        other_props = [p[0] for p in props if (p!=props[jj] and p!=props[jj+1])] # Other props that might be present; want to get rid of
        Gjj = G.copy()
        Gjj.remove_nodes_from(other_props)
        Pkeep_j = augment_paths(Gjj, props[jj], props[jj+1]) 
        Pkeep_j_new = Pkeep_j.copy()
        # Consider only augmented paths not going through other vertices:
        # for Pj in Pkeep_j:
        #     other_prop_present = any(props[k][0] in Pj for k in range(n) if (k!=jj and k!=(jj+1)))
        #     if other_prop_present:
        #         Pkeep_j_new.remove(Pj)           
        try:
            assert(Pkeep_j != []) # Shouldn't be empty. there should always be a path from pj to pj+1; otherwise example is invalid
        except AssertionError as e:
            alg_fail = True
            break
        Pkeep.append(Pkeep_j_new)
    
    for jj in range(n-1, -1, -1):
        MCj = prepare_Gj(G, props, jj, Pkeep[jj], Pkeep) # This assumes there is always a Pkeep for every j to j+1
        MC_keep.append(MCj)
    MC_keep.reverse() # Returns MC_keep in the correct order
    return Pkeep, MC_keep, alg_fail

# Main MILP program:
def milp(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e):
    nc = len(cost)
    ne = len(e)
    epsilon = 0.5 # Factor that keeps b variables to 1
    newC = []
    try:
        # Create a new model
        m = gp.Model("milp")
        m.setParam(GRB.Param.TimeLimit, 300.0) # Setting time limit: 5 min
        # Create variables: 
        x = m.addMVar(ne, vtype=GRB.BINARY, name="x")
        b = m.addMVar(nc, vtype=GRB.BINARY, name="b")
        # m.setParam('OutputFlag', 0)  # Also dual_subproblem.params.outputflag = 0
        # m.params.threads = 4
        # Set objective: c.T*b; minimizing cuts to augmenting paths pj to pj+1
        m.setObjective(cost[:,0] @ b, GRB.MINIMIZE)
    
        # Add constraint: Aji+1 x = 1
        jcut_idx = 0
        for jj in range(len(n_cut)):
            njj = n_cut[jj]
            Aj = A_cut[jj]
            rhs = np.ones((njj,), dtype=int)
            assert(len(Aj)==njj) # Sanity check
            assert(len(Aj[0])==ne) # Sanity check
            m.addConstr(Aj @ x >= rhs, name="c1_"+str(njj))
            jcut_idx += njj    
        
        # Add constraint: Ajj+1 x <= Djj+1 b
        jkeep_idx = 0
        for jj in range(len(n_keep)):
            njj = n_keep[jj]
            Aj = A_keep[jj]
            Dj = D_keep[jj]
            rhs = np.zeros((njj,), dtype=int)
            m.addConstr(Aj @ x - np.diag(Dj[:,0]) @ b[jkeep_idx: jkeep_idx+njj] <= rhs, name="c2i_"+str(njj))
            assert(len(Aj @ np.ones((ne,), dtype=int)) == njj) # Sanity check
            assert(len(np.diag(Dj[:,0]) @ np.ones((njj,), dtype=int)) == njj) # Sanity check
            rhs = (1-epsilon)*np.ones((njj,), dtype=int)
            m.addConstr(b[jkeep_idx: jkeep_idx+njj] - Aj @ x <= rhs, name="c2ii_"+str(njj))
            ones = np.ones((njj,), dtype=int)
            # lhs = np.dot(ones, b[j_idx: j_idx+njj])
            m.addConstr(ones @ b[jkeep_idx: jkeep_idx+njj] <= njj-1)
            jkeep_idx += njj
    
        # Optimize model
        m.optimize()
        timed_out = False
        feas = False
        if(m.status==GRB.INFEASIBLE):
            feas = True
        if(m.status == GRB.TIME_LIMIT):
            timed_out = True

       # for v in m.getVars():
       #     print('%s %g' % (v.varName, v.x))
        # for v in m.getVars():
        #     print('%s %g' % (v.varName, v.b))
       # print('Obj: %g' % m.objVal)
        if feas!=True and timed_out!=True:
            xopt=np.zeros((ne,1))
            bopt = np.zeros((nc,1))
            xidx = 0
            bidx = 0
            for v in m.getVars():
                if(xidx < ne):
                    xopt[xidx] = v.x
                    xidx+=1
                else:
                    bopt[bidx] = v.x
                    bidx+=1
            
            newC = interpret_milp_output(xopt,e)
            print(newC) 
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
    
    return newC, timed_out, feas

# Interpreting MILP output:
def interpret_milp_output(x,e):
    lx = len(x)
    assert(lx == len(e))
    newC = [e[k] for k in range(lx) if x[k,0]==1]
    return newC

# Constructing the set of edges needed to be cut:
def construct_e(Pcut):
    n_cut = []
    e = [] # Edge vector
    path_indices = [] # Stores the indices of edges in e that belong to a path that needs to be cut
    # Parameters for paths that must be cut:
    for jj in range(len(Pcut)):
        Pcut_j = Pcut[jj]
        n_cut_j = len(Pcut_j) # No. of paths to cut from pj to pi+1
        path_indices_j = []    # Stores a list of edge_in_path for each path from pj to pi+1
        for path in Pcut_j:   # Finding edges in each path
            if path:
                edges_in_path = [] # Stores 
                for k in range(len(path)-1):
                    candidate_edge = (path[k], path[k+1]) # candidate edge to be cut
                    if candidate_edge not in e:
                        e.append(candidate_edge)
                    eidx = e.index(candidate_edge)
                    edges_in_path.append(eidx)
                path_indices_j.append(edges_in_path)
            else:
                n_cut_j -= 1
        path_indices.append(path_indices_j) # If there are no paths from pj to pi+1, this is an empty list
        n_cut.append(n_cut_j)
    return e, n_cut, path_indices

# Constructing A_cut:
def construct_cut_matrix(n_cut_paths, ne, cut_path_indices):
    A_cut = []
    for j in range(len(n_cut_paths)):
        cut_paths_j = cut_path_indices[j]
        row_idx = n_cut_paths[j]
        if row_idx > 0: # If there are any paths to be cut from pj to pi+1:
            A_cut_j = np.zeros((row_idx, ne))
            for jj in range(row_idx):
                cut_path_indices_j = cut_paths_j[jj]
                for eidx in cut_path_indices_j:
                    A_cut_j[jj, eidx] = 1
            A_cut.append(A_cut_j)
    return A_cut

# Constructing A_keep, D_keep and n_keep:
def construct_keep_parameters(Pkeep, MC_keep,e, ne):
    # Parameters relating to paths that must be kept (as many as possible):
    n_keep = []
    jidx = 0
    A_keep = []
    D_keep = []
    for Pkeep_j in Pkeep:
        n_keep_j = len(Pkeep_j)
        MCj = MC_keep[jidx]
        A_keep_j= None
        
        for path in Pkeep_j:
            if path:
                row = np.zeros((1,ne)) # Adding a row to A_jj+1
                for k in range(len(path)-1):
                    candidate_edge = (path[k], path[k+1])
                    if candidate_edge in e and candidate_edge in MCj: # If this is an edge that is a candidate to be cut in Pkeep and is a minimum cut edge from pj to pj+1:
                        eidx = e.index(candidate_edge)
                        row[0, eidx] = 1 
                if A_keep_j is None:
                    A_keep_j = row.copy()
                else:
                    A_keep_j = np.vstack((A_keep_j, row))
            else:
                n_keep_j -= 1
                
        if (n_keep_j>0):
            n_keep.append(n_keep_j)
            A_keep.append(A_keep_j)
            assert(A_keep_j is not None)
                
            # Creating D_keep:
            ones = np.ones((ne, 1))
            # Check dimensions match up:
            D_keep_j = A_keep_j @ ones
            
            assert(len(A_keep_j[0]) == ne) 
            assert(len(D_keep_j) == len(A_keep_j))
            D_keep.append(D_keep_j)
        jidx += 1
        
    return A_keep, D_keep, n_keep

# construct MILP parameters:
def construct_milp_params(Pcut, Pkeep, MC_keep, n):
    e, n_cut, cut_path_indices = construct_e(Pcut) # Finding paths that need to be cut
    ne = len(e) # Number of possible edges that could be cut
    n_cut_paths = np.sum(np.array(n_cut))
    ncuts = 0
    for ii in cut_path_indices:
        ncuts += len(ii)
    assert(ncuts == n_cut_paths)  # Sanity check
    A_cut = construct_cut_matrix(n_cut, ne, cut_path_indices) # Constructing first class of constraints
    A_keep, D_keep, n_keep = construct_keep_parameters(Pkeep, MC_keep,e, ne)   
    # Remove any zeros from the n_cut vector:
    ncut_new = []
    for ci in n_cut:
        if ci != 0:
            ncut_new.append(ci)
    # Sanity check assertions:
    assert(len(A_cut)==len(ncut_new))
    assert(len(A_keep)==len(n_keep))
    cost = construct_c(n_keep)      # uncomment for original cost function
    # cost = construct_c(n_keep, Pkeep) # Uncomment for modified cost
    return A_cut, ncut_new, A_keep, D_keep, n_keep, cost, e

# construct MILP parameters with a small adjustment to the cost
def construct_milp_params_corrected(Pcut, Pkeep, MC_keep, n):
    e, n_cut, cut_path_indices = construct_e(Pcut) # Finding paths that need to be cut
    ne = len(e) # Number of possible edges that could be cut
    n_cut_paths = np.sum(np.array(n_cut))
    ncuts = 0
    for ii in cut_path_indices:
        ncuts += len(ii)
    assert(ncuts == n_cut_paths)  # Sanity check
    A_cut = construct_cut_matrix(n_cut, ne, cut_path_indices) # Constructing first class of constraints
    A_keep, D_keep, n_keep = construct_keep_parameters(Pkeep, MC_keep,e, ne)   
    # Remove any zeros from the n_cut vector:
    ncut_new = []
    for ci in n_cut:
        if ci != 0:
            ncut_new.append(ci)
    # Sanity check assertions:
    assert(len(A_cut)==len(ncut_new))
    assert(len(A_keep)==len(n_keep))
    cost = construct_c(n_keep)  
    return A_cut, ncut_new, A_keep, D_keep, n_keep, cost, e

# Creating the cost vector for the MILP based on no. of paths that need to be preserved:
def construct_c(n_keep):
    n_keep_paths = int(np.sum(np.array(n_keep)))
    cost = np.ones((n_keep_paths, 1))
    idx = 0
    for n_keep_j in n_keep:
        cost[idx:idx + n_keep_j, 0] = 1.0/n_keep_j
        idx = idx + n_keep_j
    return cost

# Corrected cost vector to account for cycles:
def corrected_construct_c(n_keep, Pkeep):
    n_keep_paths = int(np.sum(np.array(n_keep)))
    cost = np.ones((n_keep_paths, 1))
    # Process nodes in each path:
    nodes_keep = []
    node_prop = [] # Nodes representing propositions
    nodes_Pj_list = [] # Keep track of list of all nodes
    for Pj in Pkeep:
        s = Pj[0][0]
        t = Pj[0][-1]
        if s not in node_prop:
            node_prop.append(s)
        if t not in node_prop:
            node_prop.append(t)
        nodes_Pj = list(set().union(*Pj)) # All nodes in Pj
        nodes_Pj_list.append(nodes_Pj) # Storing all nodes for every Pj
        nodes_keep = list(set().union(nodes_keep, nodes_Pj))
    nodes_keep = [ii for ii in nodes_keep if ii not in node_prop] # Removing nodes that are a part of node_prop
    node_count = dict()
    # Creating a dictionary of node counts:
    for nj in nodes_keep:
        node_count[nj] = 0
    for nodes_Pj in nodes_Pj_list:
        for nj in nodes_Pj:
            if nj not in node_prop:
                node_count[nj] += 1
    
    # Creating the cost vector:
    idx= 0
    for jj in range(len(Pkeep)):
        njj = n_keep[jj]  
        Pj = Pkeep[jj] # jth path
        for path in Pj:
            node_count_path = [node_count[nj] for nj in path[1:-1]]
            max_node_share = max(node_count_path)
            cost[idx] = 1.0/njj * 1.0/max_node_share
            idx+=1
    assert(idx == n_keep_paths)
    return cost

# Finding nodes closer to goal than p2
# This function might have issues; written fast.
def sdistance(G, p2, g):
    Gc = G.copy()
    dp2 = len(nx.shortest_path(Gc, source=p2[0], target=g[0]))
    ancestor_tree = nx.dfs_tree(Gc.reverse(), source=g[0], depth_limit=dp2-1).reverse() # Finding all upstream nodes that could lead to G0 but not via the aug path nodes
    nodes_closer_goal = list(ancestor_tree)
    return nodes_closer_goal

# Returns all augmented paths from source s to sink t on graph G
def augment_paths(G, start, end, unit_capacities=False):
    s = start[0] # Assuming only one state for each source and sink
    t = end[0]
    P = []
 
    # Reading augmented paths from the flow network:
    Gc = G.copy()
    R, P = get_augmenting_path(Gc,s,t)
    
    Rf = R.graph["flow_value"]
    if unit_capacities:
        assert(len(P)==Rf) # number of augmenting paths should be equal to the flow value for unit capacities
    return P

# finding all augmnting paths p1, p2, ..., pn on final graph:
def all_augment_paths(G, props):
    Paug = []
    for ii in range(len(props)-1):
        Pi = augment_paths(G, props[ii], props[ii+1], unit_capacities=True)
        Paug.append(Pi)
    return Paug

# Function that prepares the min-cut edges graph:
# G is the original graph on which Pkeep_j (aug paths from j to j+1) and Pkeep (aug paths from k-1 to k) for all k
# are found. Function prepare_Gj constructs a graph Gj in which for each augmented path p in Pj, minimum-cut edges are
# computed on the graph Gp, which is the graph for which all nodes on augmented paths from Pk-1 to Pk are removed except for those on path p
def prepare_Gj(G, props, jj, Pkeep_j, Pkeep):
    MCj = []
    exclude_nodes = []
    for Pkeep_k in Pkeep:         # All aug paths from pk to pk+1
        nodes_k = list(set().union(*Pkeep_k)) # Collecting all nodes that are in Pkeep_k
        exclude_nodes = add_elems(exclude_nodes, nodes_k) # We don't want to exclude the current node pj from the final list
    for ap in Pkeep_j:
        node_remove_Gp = [node for node in exclude_nodes if node not in ap]
        Gp = G.copy()
        Gp.remove_nodes_from(node_remove_Gp)
        MCp = min_cut_edges(Gp, props[jj], props[jj+1])
        MCj = add_elems(MCj, MCp)
    return MCj

# Helper function to the prepare_Gj function:
# Adding elements(edges/nodes) from listB to listA that are not already in listA:
def add_elems(listA, listB):
    newA = listA.copy()
    for elem in listB:
        if elem not in listA:
            newA.append(elem)
    return newA

# Finds all paths in a graph 
# Finds minimum cut edges in a graph:
def min_cut_edges(G, start, end):
    MC = []
    s = start[0] # Assuming only one state for each source and sink
    t = end[0]
    R = edmonds_karp(G, s, t)
    fstar = R.graph["flow_value"]
    
    P = augment_paths(G,start,end)
    
    # Going over edges in the augmented paths
    for pathi in P: # Iterating over paths
        for ii in range(len(pathi)-1): # Iterating over edges in each path
            candidate_edge = (pathi[ii], pathi[ii+1])
            Gtemp = G.copy()
            # Snaity check:
            assert(candidate_edge in Gtemp.edges)
            Gtemp.remove_edge(*candidate_edge)
            Gtemp_residual = edmonds_karp(Gtemp, s, t)
            flow = Gtemp_residual.graph["flow_value"]
            if (fstar-flow)>= 1:             # If there is a decrease in flow caused by removal of this edge, then it is a minimum cut edge
                MC.append(candidate_edge)
    return MC


### Efficient versions of the same original functions above. Ex: If pj to pi+1 is a cut path, then a cu
# t path from pk (<j) to pi+1 is not included if it contains pj along the way.
# The only function need to be modified is cut_aug_paths(?)
def cut_aug_paths_eff(G, props, i):
    Pcut = []
    n_paths_to_cut = []
    for j in range(i-1, -1, -1): # Decreasing order from i-1 to 1
        n_paths_to_cut_j = 0
        Pcut_j = augment_paths(G, props[j], props[i+1]) # From pj to pi+1; which is index by pi
        Pcut_j_new = Pcut_j.copy()
        if (j < i-1): # checking if any of pj+1 to pi-1 are on aug path; if so, remove them
            higher_props = [props[k][0] for k in range(j+1, i)]   
            for Pj in Pcut_j:
                higher_prop_present = any(ph in Pj for ph in higher_props)
                if higher_prop_present:
                    Pcut_j_new.remove(Pj)
        if Pcut_j_new:
            n_paths_to_cut_j+=len(Pcut_j_new) # If there are paths to cut, update n_paths_to_cut
        Pcut.append(Pcut_j_new)
        n_paths_to_cut.append(n_paths_to_cut_j)
    Pcut_new = list(reversed(Pcut))
    return Pcut_new, n_paths_to_cut
