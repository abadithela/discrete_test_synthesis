# A file to illustrate how test suites can be configured:
# File illustrating examples for game graphs:
# Remember to run the script in the directory you want to save the figures in

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
import inspect
from matplotlib import animation

from dynamic_obstacles.gridworld.gridworld_class import GridWorld as gc
from dynamic_obstacles.gridworld.Player_class import Player as pc
from dynamic_obstacles.gridworld.Game_graph_class import GameGraph as gg
from static_obstacles.gridworld.test_run_configuration import test_run_configuration as trc
from time import gmtime, strftime

def static_example():
    # Automatically set filenames:
    file_path = "examples/static_examples/"
    fname_matrix = file_path + "Data/static_obstacle_matrix_#.dat"
    fname_grid_w_prop = file_path + "Figures/initial_grid_#.png"
    fname_grid_w_static_obs = file_path + "Figures/grid_static_obs_#.png"
    fname_propositions = file_path + "Data/propositions_#.dat"
    fname_goal = file_path + "Data/goal_#.dat"
    fname_ani = file_path + "Animations/static_obs_ani_#.avi"
    fname_matrix = fname_matrix.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_propositions = fname_propositions.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_goal = fname_goal.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_ani = fname_ani.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_grid_w_prop = fname_grid_w_prop.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_grid_w_static_obs = fname_grid_w_static_obs.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

    pkl_matrix = open(fname_matrix,"wb")
    pkl_prop = open(fname_propositions, "wb")
    pkl_goal = open(fname_goal,"wb")
    # pkl_ani = open(fname_ani, "wb")

    # Setting initial variables
    M = 10 # No. of rows
    N = 10 # No. of columns
    Nprop = 3 # No. of propositions
    Nmax_prop = 3 # Maximum number of nodes in each proposition

    # Setting up goals and propositions:
    goal_row = M # Row counted starting from 1 top to bottom
    goal_col = N # Column counted starting from 1 left to right
    sys_reach = [N*(goal_row-1) + goal_col]
    nNodes = M*N
    nprop_to_cover = [random.choice(range(1,Nmax_prop+1)) for ii in range(Nprop)] # A list of numbers with each element between 1 and 3
    nNodes_to_cover = [[random.choice(range(1,nNodes+1)) for ii in range(nprop_to_cover[jj])] for jj in range(len(nprop_to_cover))] # Randomly choosing the vertices to cover

    take_user_input = input("Do you want to set grid size, proposition locations and goal? [y/n] ")
    if take_user_input == "y":
        M = input("Enter no. of rows ")
        N = input("Enter no. of columns ")
        M = int(M)
        N = int(N)

        # Setting up goals and propositions:
        goal_row = input("Enter row location of the goal (counting starts from 1 top to bottom) ") # Row counted starting from 1 top to bottom
        goal_col = input("Enter column location of the goal (counting starts from 1 left to right) ") # Column counted starting from 1 left to right
        goal_row = int(goal_row)
        goal_col = int(goal_col)
        cell2node = lambda r, c: N*(r-1) + c
        sys_reach = [cell2node(goal_row, goal_col)]
        nNodes = M*N
        rand_prop = input("Do you want to set proposition locations manually? [y/n] ")
        if rand_prop == "n":
            Nprop = int(input("How many different propositions do you want to cover? "))
            Nmax_prop = int(input("Atmost how many grid cells can each proposition have (enter a number)? "))
            nprop_to_cover = [random.choice(range(1,Nmax_prop+1)) for ii in range(Nprop)] # A list of numbers with each element between 1 and 3
            nNodes_to_cover = [[random.choice(range(1,nNodes+1)) for ii in range(nprop_to_cover[jj])] for jj in range(len(nprop_to_cover))] # Randomly choosing the vertices to cover
        else:
            Nprop = int(input("How many different propositions do you want to cover? "))
            nprop_to_cover = []
            nNodes_to_cover = []
            for ii in range(1, Nprop+1):
                Nmax_ii = int(input("How many grid cells do you want to specify for proposition " + str(ii)+" ? "))
                nprop_to_cover.append(Nmax_ii)
                grid_locs= []
                for jj in range(1, Nmax_ii+1):
                    row, col = [int(x) for x in input("Enter row and column grid location: ").split()]
                    grid_locs.append(cell2node(row, col))
                nNodes_to_cover.append(grid_locs)
    # Construct gridworld states and transitions:
    GW = gc([M,N])
    GW.grid_transitions()
    test_config = trc(GW) # Setting up a test configuration

    print("System reachability goal: ")
    print(*sys_reach)
    print("Propositions to cover in testing: ")
    for prop in nNodes_to_cover:
        print(*prop)
    pickle.dump(nNodes_to_cover, pkl_prop)
    pickle.dump(sys_reach, pkl_goal)
    pkl_prop.close()
    pkl_goal.close()

    # Add propositions:
    test_config.set_final_reach_goal(sys_reach)
    test_config.set_propositions(nNodes_to_cover)

    # Create Base plot of goal and proposition locations:
    fig_num = 1
    fig, ax, im = GW.base_plot(fig_num)
    fig, ax, im = test_config.base_plot(fig, ax, im, nNodes_to_cover, sys_reach)
    fig.savefig(fname_grid_w_prop, dpi=fig.dpi)
    fig_num+=1

    fig2, ax2, im2 = GW.base_plot(fig_num)

    # Creating static obstacles and plotting them:
    cut_transitions, static_obstacles = test_config.generate_static_obstacles(nNodes_to_cover, sys_reach)
    test_matrix = test_config.update_matrix(static_obstacles)
    # ani = test_config.animate_static(fig2, ax2, im2, test_matrix)
    fig2, ax2, im2 = test_config.base_plot(fig2, ax2, im2, nNodes_to_cover, sys_reach) # Constructing the base plot
    fig2, ax2, im2 = test_config.static_obstacle_plot(fig2, ax2, im2, test_matrix[-1]) # Plotting static obstacles on the grid
    fig.savefig(fname_grid_w_static_obs, dpi=fig.dpi) # Save figure

    # Files to save animation and test matrix:
    pickle.dump(test_matrix[-1], open(fname_matrix,"wb"))
    pkl_matrix.close()
    # writervideo = animation.FFMpegWriter(fps=60)
    # ani.save(fname_ani, writer=writervideo)
    # ani.save(fname_ani)
    plt.show()