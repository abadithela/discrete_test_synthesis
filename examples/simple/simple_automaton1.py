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
    sys_states = [2,5,6,7,8,13,14]
    env_states = [1,3,4,9,10,11,12,15]
    goal = setup_goal()
    Ns = 4
    Ne = 3
    static_obs = setup_static_obs()
    sys_nodes, env_nodes = setup_nodes()
    G = setup_graph(sys_nodes, env_nodes)
    return G, Ns, Ne, static_obs

def state2node(state, Ns, Ne, player):
    node = Ne*(state[1]-1)+state[0]
    if player == "s":
        return "v2_"+str(node)
    else:
        return "v1_"+str(node)

# Setting up nodes of a graph:
def setup_nodes():
    system_states = [[1,4], [3,3], [2,3], [3,2], [2,2], [3,1], [2,1]]
    env_states = [[1,4], [3,4], [2,4], [3,3], [3,2], [2,3], [2,2], [1,1]]
    sys_nodes = []
    env_nodes = []
    Ns = 4
    Ne = 3
    for ii in range(len(system_states)):
        node = state2node(system_states[ii], Ns, Ne, "s")
        sys_nodes.append(node)
    for ii in range(len(env_states)):
        node = state2node(env_states[ii], Ns, Ne, "s")
        env_nodes.append(node)
    return sys_nodes, env_nodes

# Plot example:
def plot_example():
    sys_states = [2,5,6,7,8,13,14]
    env_states = [1,3,4,9,10,11,12,15]
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
    goals = setup_goal()
    goal_vertices = Game_Graph.set_win_states(goals)
    Wsys2 = Game_Graph.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
    Val_sys = Game_Graph.get_value_function(win_agent) # System Value Function

    # Setup propositions:
    P = setup_propositions()

    # Environment value function:
    def set_env_goal_nodes(coverage_props_list):
        env_goal_nodes = []
        lC = len(coverage_props_list)
        for ns in range(1, Ns+1):
            for ne in range(1, Ne):
                for ii in range(lC):
                    prop_lambda_func = coverage_props_list[ii][0]
                    if (prop_lambda_func(ns, ne)):
                        env_goal_nodes.append([ne,ns])
        return env_goal_nodes

    def compute_env_winning_set(coverage_props_list):
        coverage_props = dict(coverage_props_list)
        Game_Graph.set_vertex_weight(coverage_props)
        env_goal_nodes = set_env_goal_nodes(coverage_props_list)
        goal_vertices = Game_Graph.set_win_states(env_goal_nodes)
        # Robust Pre for system winning set computation:
        quant_env = 'exists'
        quant_sys = 'exists'
        win_agent = 'e'
        Wenv = Game_Graph.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
        Val_env = Game_Graph.get_value_function(win_agent) # Environment Value Function
        return Wenv, Val_env

    Wenv, Val_env = compute_env_winning_set(P)
    pdb.set_trace()
    print(max(1,2))
# Setting up static obstacles:
def setup_static_obs():
    static_obs = [[3,3]]
    static_obs_nodes = []
    obs_state = 'e' # Only env states are blocked
    for ii in range(len(static_obs)):
        v = state2node(static_obs[ii], 4, 3, 'e')
        static_obs_nodes.append(v)
    return static_obs_nodes

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
    T = [("v1_10", "v2_10"), ("v2_10", "v1_10"),
         ("v2_10","v1_12"), ("v1_12","v2_10"),
         ("v2_10", "v1_11"), ("v1_11", "v2_10"),
         ("v2_9", "v1_12"), 
         ("v2_6", "v1_12"),
         ("v2_8", "v1_11"),
         ("v2_5", "v1_11"),
         ("v2_9", "v1_9"), ("v1_9", "v2_9"),
         ("v2_6", "v1_6"), ("v1_6", "v2_6"),
         ("v2_3","v1_9"),
         ("v2_3","v1_6"), 
         ("v2_3", "v1_1"), ("v1_1","v2_3"),
         ("v1_8","v2_8"), ("v2_8","v1_8"),
         ("v1_5", "v2_5"), ("v2_5", "v1_5"),
         ("v2_2","v1_8"),
         ("v2_2", "v1_5"), 
         ("v2_2", "v1_1"), ("v1_1", "v2_2")]
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
    P = [(lambda ns, ne: ns==4 and ne==1, 1), (lambda ns, ne: ns==2 and ne==3, 2)]
    return P

# Setup reach goal:
def setup_goal():
    G = [[1,4]] # Top node is the goal
    return G

# Display final value function:
def display_value_function(Game_Graph, player):
    if(player == 's'):
        V = Game_Graph.get_val_function('s')
    if(player == 'e'):
        V = Game_Graph.get_val_function('e')
    
    return fig, ax