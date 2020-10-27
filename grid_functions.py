#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:00:46 2020

@author: apurvabadithela
"""
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# constructing a grid graph:
# Inputs: G is a graph with nodes labeled n1 to n100.
#         M: number of rows
#         N: number of columns
#         allow_diag: True/False
def construct_grid(G, M, N, allow_diag):
    pos = lambda irow, icol: N*(irow-1) + icol
    east = lambda irow, icol: N*(irow-1) + icol + 1
    west = lambda irow, icol: N*(irow-1) + icol - 1
    north = lambda irow, icol: N*(irow-2) + icol
    south = lambda irow, icol: N*(irow) + icol
    northE = lambda irow, icol: N*(irow-2) + icol + 1
    southE = lambda irow, icol: N*(irow) + icol + 1
    northW = lambda irow, icol: N*(irow-2) + icol -1
    southW = lambda irow, icol: N*(irow) + icol - 1
    
    Gf = G.copy()
    for irow in range(1,M+1):
        for icol in range(1,N+1):
            state = pos(irow, icol)
            if allow_diag:
                successors = [east(irow, icol), west(irow,icol), north(irow, icol), south(irow,icol),
                              northE(irow,icol), southE(irow, icol), northW(irow, icol), southW(irow, icol)]
            else:
                successors = [east(irow, icol), west(irow,icol), north(irow, icol), south(irow,icol)]
            # Taking care of corners:
            if(irow == 1):
                rem_succ = [north(irow, icol), northW(irow, icol), northE(irow, icol)]
                for r in rem_succ:
                    if r in successors:
                        successors.remove(r)
            if(irow == M):
                rem_succ = [south(irow, icol), southW(irow, icol), southE(irow, icol)]
                for r in rem_succ:
                    if r in successors:
                        successors.remove(r)
            if(icol == 1):
                rem_succ = [northW(irow, icol), west(irow, icol), southW(irow, icol)]
                for r in rem_succ:
                    if r in successors:
                        successors.remove(r)
            if(icol == N):
                rem_succ = [northE(irow, icol), east(irow, icol), southE(irow, icol)]
                for r in rem_succ:
                    if r in successors:
                        successors.remove(r)
            # Make sure there are no negative vertices:
            for s in successors:
                if s <= 0:
                    print(state)
                assert(s>0)
            # Adding successors:
            for s in successors:
                u = "n"+str(state)
                v = "n"+str(s)
                Gf.add_edge(u, v, capacity=1.0)
    return Gf

# Plotting final map with all the obstacles:
def plot_final_map(G, M, N, props, Q0, C):
    fig, ax = base_grid(M,N,props, Q0)
    Obs_patches = determine_loc(M,N, C)
    p = PatchCollection(Obs_patches, color ='r', alpha=0.4)
    ax.add_collection(p)
    return fig, ax

# Plotting function to capture propositions, Q0 and transitions that have to be blocked:
def plot_static_map(G, M, N, props, Q0, Chist):
    nC = len(Chist)
    FIG = []
    AX = []
    fig, ax = base_grid(M,N,props, Q0) # If nothing, the original map gets plotted
    for ii in range(nC):
        fig, ax = base_grid(M,N,props, Q0)
        for jj in range(ii+1):
            Cii = Chist[jj]
            for Cii_set in Cii:
                Obs_patches = determine_loc(M,N, Cii_set)
                p = PatchCollection(Obs_patches, color ='r', alpha=0.2)
                ax.add_collection(p)
    FIG.append(fig)
    AX.append(ax)
    return FIG, AX

# Finding x, y location from state:
def rcloc(u, M, N):
    if(u%N == 0):
        col = N
        row = u//N
    else:
        col = u%N
        row = u//N + 1

    return row, col

# Given x,y locations of 2 vertices, finds the lower and upper wedge patches:
def wedge_angles(ru, cu, rv, cv):
    tl = 0
    tu = 360
    if(cv==cu-1 and rv==ru):
        tl = -90
        tu = 90
    if(cv==cu+1 and rv==ru):
        tl = 90
        tu = 270
    if(cv==cu and rv==ru-1):
        tl = 0
        tu = 180
    if(cv==cu and rv==ru+1):
        tl = 180         # Maybe because rows are flipped?
        tu = 360
    if(cv==cu-1 and rv==ru+1):
        tl = 270
        tu = 360
    if(cv==cu+1 and rv==ru-1): 
        tl = 90
        tu = 180
    if(cv==cu-1 and rv==ru-1):
        tl = 0
        tu = 90
    if(cv==cu+1 and rv==ru+1):
        tl = 180
        tu = 270
    return tl, tu

# Given x,y locations of 2 vertices, finds the lower and upper wedge patches:
# # Wedge angles following the normal conventions // counterclockwise rotation. row+1 is below row on grid. 
# def wedge_angles2(ru, cu, rv, cv):
#     tl = 0
#     tu = 360
#     if(cv==cu-1 and rv==ru):
#         tl = -90
#         tu = 90
#     if(cv==cu+1 and rv==ru):
#         tl = 90
#         tu = 270
#     if(cv==cu and rv==ru+1):
#         tl = 0
#         tu = 180
#     if(cv==cu and rv==ru-1):
#         tl = 180         # Maybe because rows are flipped?
#         tu = 360
#     if(cv==cu-1 and rv==ru-1):
#         tl = 270
#         tu = 360
#     if(cv==cu+1 and rv==ru+1): 
#         tl = 90
#         tu = 180
#     if(cv==cu-1 and rv==ru+1):
#         tl = 0
#         tu = 90
#     if(cv==cu+1 and rv==ru-1):
#         tl = 180
#         tu = 270
#     return tl, tu

# Finding locations to place obstacles:
def determine_loc(M,N, Cii_loc):
    Obs_patches = []
    for e in Cii_loc:
        u = int(e[0][1:]) # start vertex
        v = int(e[1][1:]) # end vertex
        ru, cu = rcloc(u, M, N)
        rv, cv = rcloc(v, M, N)
        ox = (cu + cv)/2
        oy = (ru + rv)/2
        tl, tu = wedge_angles(ru, cu, rv, cv)
        wedge = Wedge((ox, oy), .1, tl, tu)
        Obs_patches.append(wedge)
    return Obs_patches

# Base grid
def base_grid(M, N, props, Q0=None):
    colors = dict(**mcolors.CSS4_COLORS)
    
    msz = 12 # Scaling of linear parameters
    lw= 2 # Linewidth
    fsz = 14 # fontsize for text
    
    fig = plt.figure()
    ax = plt.gca()

    # Setting up grid and static obstacles:
    # Grid matrix has extra row and column to accomodate the 1-indexing of the gridworld
    grid_matrix = np.zeros((M+1, N+1))

    # Setting up gridlines
    ax.set_xlim(0.5,N+0.5)
    ax.set_ylim(M+0.5, 0.5)

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, N+1, 1), minor='True')
    ax.set_yticklabels(np.flipud(np.arange(1, M+1, 1)), minor='True')

    # Gridlines based on minor ticks
    ygrid_lines = np.flipud(np.arange(1, M+1, 1)) - 0.5
    xgrid_lines = np.arange(1, N+1, 1) - 0.5
    
    # Major ticks
    ax.set_xticks(xgrid_lines)
    ax.set_yticks(ygrid_lines)

    # Minor ticks
    ax.set_yticks(ygrid_lines+0.5, minor='True')
    ax.set_xticks(xgrid_lines+0.5, minor='True')
    ax.grid(which='major', axis='both', linestyle='-', color=colors['silver'], linewidth=lw)
    plt.setp(ax.get_xmajorticklabels(), visible=False)
    plt.setp(ax.get_ymajorticklabels(), visible=False)
    
    # Plotting goals and propositions:
    props_loc= get_locations(M,N,props)
    for ii in range(len(props)-1): # For all propositions except goal
        props_loc_ii = props_loc[ii]
        plt.text(props_loc_ii[1], props_loc_ii[0], '$p_{}$'.format(ii+1),fontsize=fsz)
    props_goal = props_loc[-1]
    plt.text(props_goal[1], props_goal[0], '$g$',fontsize=fsz) # plotting goal text
    if Q0:
        Q0_list = [[q0] for q0 in Q0] # Putting into a format acceptable by get_locations
        Q0_loc = get_locations(M,N,Q0_list)
        for ii in range(len(Q0_loc)):
            Q0_loc_ii = Q0_loc[ii]
            plt.text(Q0_loc_ii[1]-0.5, Q0_loc_ii[0], '$q_0$',fontsize=fsz)
    return fig, ax

# Getting locations of propositions:
def get_locations(M,N,props):
    nprop = len(props)
    props_loc = [[] for ii in range(nprop)]
    for ii in range(nprop):
        pi = int(props[ii][0][1:])
        if(pi%N == 0):
            ii_col = N
            ii_row = pi//N
        else:
            ii_col = pi%N
            ii_row = pi//N + 1
        props_loc[ii] = [ii_row, ii_col]
    return props_loc
    
 # Show augmented_paths on final graph:
# G is the graph without auxiliary vertices, and C are the edges cut from the graph
def plot_augment_paths(Paug, G, M, N, props, Q0, C):
    fig,ax = plot_final_map(G, M, N, props, Q0, C)
    test_graph = G.copy()
    test_graph.remove_edges_from(C)
    for ii in range(len(Paug)):
        cii = c=np.random.rand(3,)
        Pi = Paug[ii]
        for path in Pi:
            path_rc = []
            for jj in range(len(path)):
                u = int(path[jj][1:])
                # v = path[jj+1][1:]
                ru, cu = rcloc(u, M, N)
                # rv, cv = rcloc(v, M, N)
                path_rc.append([cu, ru])
            path_list = np.array(path_rc)
            ax.plot(path_list[:,0], path_list[:,1], color=cii) 
    return fig, ax

# Colormap:
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)