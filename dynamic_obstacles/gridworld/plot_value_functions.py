# This script is to visualize value functions and plot them to make sure that the dynamic test synthesis policies are correct
# 8/26/20 by Apurva Badithela

import numpy as np
import random
from random import randrange
import importlib
import itertools
import pickle
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import math
import networkx as nx
import pdb

from . import gridworld_class 
from . import Player_class
from . import Game_graph_class

def make_plots(GW, GameGraph, propositions, goal, test_config, test_matrix, k, fignum):
    figs_env, axs_env = env_value_function(GW, GameGraph, propositions, goal, test_config, test_matrix, k, fignum)
    figs_sys, axs_sys = sys_value_function(GW, GameGraph, propositions, goal, test_config, test_matrix, k, fignum+4)
    plt.show()

def env_value_function(GW, GameGraph, propositions, goal, test_config, test_matrix, k, fignum):
    FIG = []
    AX = []
    Ns = GameGraph.Ns
    Ne = GameGraph.Ne
    Val = np.zeros((Ns, Ne))
    
    Val_env = GameGraph.Val_env

    # Plotting value function:
    for key, val in Val_env.items():
        node = int(key[3:])
        ne, ns = GameGraph.state2node(Ns, Ne, node)
        Val[int(ns)-1, int(ne)-1] = val
    
    # Plots for different environment values:
    for ii in range(Ne):
        fig, ax, im = create_base_figure(GW,propositions, goal, test_config, test_matrix, ii+1, fignum + ii)
        ax = val_function(fig, ax, Val[:,ii])
        FIG.append(fig)
        AX.append(AX)
    return FIG, AX

def sys_value_function(GW, GameGraph, propositions, goal, test_config, test_matrix, k, fignum):
    FIG = []
    AX = []
    Ns = GameGraph.Ns
    Ne = GameGraph.Ne
    Val = np.zeros((Ns, Ne))
    for ns in range(Ns):
        for ne in range(Ne):
            Val[ns, ne] = float('inf')
    Val_sys = GameGraph.Val_sys

    # Plotting value function:
    for key, val in Val_sys.items():
        node = int(key[3:])
        ne, ns = GameGraph.state2node(Ns, Ne, node)
        Val[int(ns)-1, int(ne)-1] = val
    
    # Plots for different environment values:
    for ii in range(Ne):
        fig, ax, im = create_base_figure(GW,propositions, goal, test_config, test_matrix, ii+1, fignum + ii)
        ax = val_function(fig, ax, Val[:,ii])
        FIG.append(fig)
        AX.append(AX)
    return FIG, AX

def create_base_figure(GW, propositions, goal, test_config, test_matrix, ne, fignum):
    msz = 12
    fig, ax, im = GW.base_plot(fignum) # Drawing the base plot of the grid world
    fig, ax, im = test_config.base_plot(fig, ax, im, propositions, goal) # Constructing the base plot
    fig, ax, im = test_config.static_obstacle_plot(fig, ax, im, test_matrix[-1]) # Plotting static obstacles on the grid

    # Specific to the easter egg hunt
    if(ne==2):
        ax.plot(4, 6, "g*", markersize=msz)
    if(ne == 3):
        ax.plot(6, 4, "g*", markersize=msz)
    if(ne == 4):
        ax.plot(4, 6, "g*", markersize=msz)
        ax.plot(6, 4, "g*", markersize=msz)
    return fig, ax, im

def get_cell(ns, Nrows, Ncols):
    if(ns%Ncols == 0):
        x = Ncols
        y = ns//Ncols + 1
    else:
        x = ns%Ncols
        y = ns//Ncols + 1
    return x, y

def val_function(fig, ax, value):
    fsz = 8
    for ns in range(len(value)):
        x,y = get_cell(ns, 6, 6)
        if(value[ns]<float('inf')):
            ii = int(value[ns])
            v = str(ii)
        else:
            v = "inf"
        ax.text(x, y, v, fontsize=fsz)
    return ax
