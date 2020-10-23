#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:25:34 2020

@author: apurvabadithela
"""

# This code works! except for the initial condition part; which we will fix

import numpy as np
import networkx as nx
import gurobipy as gp
import scipy.sparse as sp
from gurobipy import GRB
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
from aug_path import get_augmenting_path
from networkx.algorithms.flow.utils import build_residual_network

# Main algorithm to remove edges:
# Inputs: directed graph G with capacities = 1 on all edges, p: struct that maps propositions with nodes
# and Q0, set of initial conditions. If none, then Q0 is found.
# Returns: C (set of edges to be restricted), Q0 (set of initial conditions to start the test run)
def remove_edges(G, props, Q0=None):
    Gf, node_q0, find_Q0 = check_Q0(G, Q0)
    Gin = Gf.copy()
    C = []
    n = len(props)-1 # No. of propositions to cover + goal
    assert(n>0)
    i = n-1 # For indexing
    Chist = [] # History of cuts
    while i > 0 and n>=2:  # so that it's not just p1 and g
        C, Chisti = remove_pi_edges(Gin, props, i, C) # Input Gf already has input C edges removed from it
        Gin = Gf.copy() # Copying the whole original graph
        Gin.remove_edges_from(C) # Removing all the edges found by remove_pi_edges
        Chist.append(Chisti) # Appending cuts for the ith node
        i -= 1 # Decreasing i
        print(C)
        
    # When i = 0, its time for constraints to be placed for p1, and if find_Q0 =1, to find the set of initial conditions
    if find_Q0:
        Q0 = pre_p1(Gin, props[0], props[1], props[-1]) # Finding predecessors of p1 (p[0]) not on augmenting paths from p1 to p2
        Gq0, node_q0, find_Q0 = check_Q0(Gin, Q0)
        # Reduce q0 to p1 weight to 1, if it exists:
        if ((node_q0, props[0][0]) in Gq0.edges):
            errant_edge = (node_q0, props[0][0])
            Gq0.remove_edge(*errant_edge)
            Gq0.add_edge(errant_edge[0], errant_edge[1], capacity=1.0)
            
        assert(find_Q0 == 0) # Now, that initial condition points have been found, this should be 0
        p0 = [[node_q0], props[0], props[1]]
        if(node_q0 != props[0][0]): # Don't find cut edges if pi-1 is not the same as pi.
            C, Chist0 = remove_pi_edges_Q0(Gq0, p0, 1, C)
            Chist.append(Chist0)
    else:
        p0 = [[node_q0], props[0], props[1]]
        assert(node_q0 != props[0])
        C, Chist0 = remove_pi_edges_Q0(Gin, p0, 1, C)
        Chist.append(Chist0) # Appending the cuts for the last node
    return C, Q0, Chist

# Function to remove edges for proposition pi, and adding newly removed edges to C
def remove_pi_edges(G, props, i, C):
    n = len(props)-1
    Gi = G.copy()
    Gk = G.copy()
    Gi.remove_node(props[i][0]) 
    count = []
    Chisti = []
    Pcut, n_paths_to_cut = cut_aug_paths(Gi, props, i)    
    Pkeep, MC_keep = keep_aug_paths(Gk, props, i, count)                      
    while np.sum(n_paths_to_cut) > 0:
        A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = construct_milp_params(Pcut,Pkeep, MC_keep, n)
        newC = milp(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e) # Getting new cuts from the MILP
        C = update(G, C, newC) # Updating cuts
        Chisti.append(newC)
        # Prepare for next iteration of cuts for pi:
        Gi = G.copy()
        Gk = G.copy()
        Gi.remove_node(props[i][0]) 
        Gi.remove_edges_from(C)
        Gk.remove_edges_from(C)
        Pcut, n_paths_to_cut = cut_aug_paths(Gi, props, i)    
        Pkeep, MC_keep = keep_aug_paths(Gk, props, i, count)
    return C, Chisti

# Function to remove edges for proposition pi, and adding newly removed edges to C
def remove_pi_edges_Q0(G, props, i, C):
    n = len(props)-1
    Gi = G.copy()
    Gk = G.copy()
    Gi.remove_node(props[i][0]) 
    count = []
    Pcut, n_paths_to_cut = cut_aug_paths(Gi, props, i)    
    Pkeep, MC_keep = keep_aug_paths(Gk, props, i, count)   
    Chist0 = []                   
    while np.sum(n_paths_to_cut) > 0:
        A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = construct_milp_params(Pcut,Pkeep, MC_keep, n)
        A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = remove_q0_edges(A_cut, n_cut, A_keep, D_keep, n_keep, cost,e)
        newC = milp(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e) # Getting new cuts from the MILP
        C = update(G, C, newC) # Updating cuts
        Chist0.append(newC)
        # Prepare for next iteration of cuts for pi:
        Gi = G.copy()
        Gk = G.copy()
        Gi.remove_node(props[i][0]) 
        Gi.remove_edges_from(C)
        Gk.remove_edges_from(C)
        Pcut, n_paths_to_cut = cut_aug_paths(Gi, props, i)    
        Pkeep, MC_keep = keep_aug_paths(Gk, props, i, count)
    return C, Chist0

# Remove q0-->v edges where v is in Q0:
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
    
# Function that searches for augmented paths on a graph from source p[j<i] to pi+1 that bypasses waypoint p[i]
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

# function that searches for augmented paths and min-cut edges that need to be kept
def keep_aug_paths(G, p, i, count):
    Pkeep = []
    MC_keep = []
    for j in range(i+1):
        Pkeep_j = augment_paths(G, p[j], p[j+1])
        Pkeep.append(Pkeep_j)
        MCj = min_cut_edges(G, p[j], p[j+1], count)
        count.append(1)
        MC_keep.append(MCj)
    return Pkeep, MC_keep

# Main MILP program:
def milp(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e):
    nc = len(cost)
    ne = len(e)
    epsilon = 0.5 # Factor that keeps b variables to 1
    newC = []
    try:
        # Create a new model
        m = gp.Model("milp")
        m.setParam(GRB.Param.TimeLimit, 600.0) # Setting time limit: 10 min
        # Create variables: 
        x = m.addMVar(ne, vtype=GRB.BINARY, name="x")
        b = m.addMVar(nc, vtype=GRB.BINARY, name="b")
    
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
            m.addConstr(Aj @ x == rhs, name="c1_"+str(njj))
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


       # for v in m.getVars():
       #     print('%s %g' % (v.varName, v.x))
        # for v in m.getVars():
        #     print('%s %g' % (v.varName, v.b))
       # print('Obj: %g' % m.objVal)
        
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
    
    return newC

# Interpreting MILP output:
def interpret_milp_output(x,e):
    lx = len(x)
    assert(lx == len(e))
    newC = [e[k] for k in range(lx) if x[k,0]==1]
    return newC

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

# Constructing the set of edges needed to be cut:
def construct_e(Pcut):
    n_cut = []
    e = [] # Edge vector
    path_indices = [] # Stores the indices of edges in e that belong to a path that needs to be cut
    # Parameters for paths that must be cut:
    for Pcut_j in Pcut:
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
                
        n_keep.append(n_keep_j)
        A_keep.append(A_keep_j)
        
        # Creating D_keep:
        ones = np.ones((ne, 1))
        D_keep_j = A_keep_j @ ones
        D_keep.append(D_keep_j)
        jidx += 1
        # Check dimensions match up:
        assert(len(D_keep_j) == len(A_keep_j))
        assert(len(A_keep_j[0]) == ne)
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
    
    cost = construct_c(n_keep)  
    return A_cut, n_cut, A_keep, D_keep, n_keep, cost, e

# Creating the cost vector for the MILP based on no. of paths that need to be preserved:
def construct_c(n_keep):
    n_keep_paths = np.sum(np.array(n_keep))
    cost = np.ones((n_keep_paths, 1))
    idx = 0
    for n_keep_j in n_keep:
        cost[idx:idx + n_keep_j, 0] = 1.0/n_keep_j
        idx = idx + n_keep_j
    return cost

# Function returning predecessors of p1 that are not on p1 to p2. 
def pre_p1(G, p1, p2, g):
    P = augment_paths(G, p1, p2)
    aug_path_nodes = list(set().union(*P)) # Nodes on paths P
    G0 = G.copy()
    # Remove nodes that are closer to the goal than p2:
    nodes_closer_goal = sdistance(G, p2, g)
    aug_path_nodes.remove(p1[0]) # Don't want to remove the source vertex
    aug_path_nodes.extend(nodes_closer_goal) # Finding nodes closer to goal
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

# Finding nodes closer to goal than p2
# This function might have issues; written fast.
def sdistance(G, p2, g):
    Gc = G.copy()
    dp2 = len(nx.shortest_path(Gc, source=p2[0], target=g[0]))
    ancestor_tree = nx.dfs_tree(Gc.reverse(), source=g[0], depth_limit=dp2-1).reverse() # Finding all upstream nodes that could lead to G0 but not via the aug path nodes
    nodes_closer_goal = list(ancestor_tree)
    return nodes_closer_goal

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

def min_cut_edges(G, start, end, count):
    MC = []
    s = start[0] # Assuming only one state for each source and sink
    t = end[0]
    R = edmonds_karp(G, s, t)
    fstar = R.graph["flow_value"]
    if(len(count)==40):
        print("debug")
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

# Function to introduce auxiliary vertices for undirected edges in G to be replaced by directed edges:
# Returns a directed graph and the number of auxiliary vertices. If G is fully undirected, aug_v_count = |G.edges|
def to_directed(G):
    Gdir = G.copy()
    aux_v_count=0
    for e in G.edges:
        u = e[0] # start
        v = e[1] # end
        # Check if reverse edge is also in the graph:
        if (e in Gdir.edges and (v,u) in Gdir.edges): 
            aux_v = "an"+str(aux_v_count)
            Gdir.remove_edge(v,u)
            Gdir.add_edge(v, aux_v, capacity=1.0)
            Gdir.add_edge(aux_v, u, capacity=1.0)
            aux_v_count+=1
    return Gdir, aux_v_count
    
# Function to post-process cuts and get rid of auxiliary vertices:
def post_process_cuts(C, Gdir):
    postC = C.copy()
    for e in C:
        if e[0][0:2]=="an":
            pred_aux = list(Gdir.predecessors(e[0])) 
            assert(len(pred_aux) == 1) # Sanity check
            pred = pred_aux[0]
            actual_edge = (pred, e[1])
            postC.append(actual_edge)
            postC.remove(e)
        if e[1][0:2]=="an":
            succ_aux = list(Gdir.successors(e[1])) 
            assert(len(succ_aux) == 1) # Sanity check
            succ = succ_aux[0]
            actual_edge = (e[0], succ)
            postC.append(actual_edge)
            postC.remove(e)
    assert(postC != [])
    return postC

# Post-processing along with histroy of cuts. This is for plotting purposes:
def process_history(Chist, Gdir):
    nC = len(Chist)
    post_Chist = Chist.copy()
    
    for ii in range(nC):
        Chisti = Chist[ii]
        for jj in range(len(Chisti)):
            post_Chist[ii][jj] = post_process_cuts(Chisti[jj], Gdir)
    return post_Chist
