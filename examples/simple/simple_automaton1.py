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
import graphviz as gvz

from dynamic_obstacles.transitions import automata_graph 

def setup_example():
    sys_states = [2,3,8,9]
    env_states = [1,4,5,6,7,10]
    goal = setup_goal()
    Ns = len(sys_states)
    Ne = len(env_states)
    static_obs = setup_static_obs()
    G = setup_graph(sys_states, env_states)
    return G, Ns, Ne, static_obs

# Plot example:
def plot_example():
    sys_states = [2,3,8,9]
    env_states = [1,4,5,6,7,10]
    G, Ns, Ne, static_obs = setup_example()
    draw_transitions(sys_states, env_states, G.edges())

# Main function to run example for value function:
def run_example():
    G, Ns, Ne, static_obs = setup_example()

    # System value function
    Game_Graph = automata_graph.GeneralGameGraph(G, Ns, Ne, static_obs)
    quant_env = 'forall'
    quant_sys = 'exists'
    win_agent = 's'
    Wsys2 = Game_Graph.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
    Val_sys = Game_Graph.get_value_function(win_agent) # System Value Function

    # Setup propositions:

    # Environment value function:

# Setting up static obstacles:
def setup_static_obs():
    static_obs = [4]
    return static_obs

# Setup graph:
def setup_graph(sys_states, env_states):
    G = nx.DiGraph()
    G.add_nodes_from(sys_states, node_shape="s",  node_color="none")
    G.add_nodes_from(env_states)
    T = setup_transitions()
    G.add_edges_from(T)
    return G

# Generate transitions: 
def setup_transitions():
    T = [(1,2), (2,1),
         (1,3), (3,1),
         (2,4), (4,2),
         (2,5), (5,2),
         (3,6), (6,3),
         (3,7), (7,3),
         (4,8), (8,4),
         (5,8), (8,5),
         (9,6), (6,9),
         (9,7), (7,9),
         (10,8), (8,10),
         (10,9), (9,10)]
    return T

# Visualize transitions via graphviz:
def draw_transitions(sys_states, env_states, edges):
    g = gvz.Digraph('simple_automaton', filename='simple1.gv', engine='neato')
    g.attr('node', shape='ellipse')
    for s in env_states:
        g.node(str(s), label=str(s))
    g.attr('node', shape='diamond')
    for s in sys_states:
        g.node(str(s), label=str(s))
    
    for e in edges:
        g.edge(str(e[0]), str(e[1]))
    g.view()

# Setting propositions:
def setup_propositions():

    return P

# Setup reach goal:
def setup_goal():
    G = [1] # Top node is the goal
    return G

# Display final value function:
def display_value_function(Game_Graph, player):
    if(player == 's'):
        V = Game_Graph.get_val_function('s')
    if(player == 'e'):
        V = Game_Graph.get_val_function('e')
    
    return fig, ax