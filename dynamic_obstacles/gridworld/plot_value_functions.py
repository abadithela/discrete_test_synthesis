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

def plot_env_value_function(GW, GameGraph, k, fignum):
    fig, ax, im = GW.base_plot(fignum) # Drawing the base plot of the grid world
    Val_env_k = GameGraph.value_function('e'):
    return fig, ax, im

def plot_sys_value_function(GW, GameGraph, k, fignum):
    fig, ax, im = GW.base_plot(fignum) # Drawing the base plot of the grid world
    Val_sys_k = GameGraph.value_function('s'):
    return fig, ax, im