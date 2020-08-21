# Apurva Badithela
# File to synthesize test cases for Easter Egg Hunt Grid framework shown in Easter_Egg_Hunt.png and Easter_Egg_Hunt_grid.png

import numpy as np
import random
from random import randrange
import sys
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

from base_classes.gridworld_class import GridWorld as gc
from base_classes.Player_class import Player as pc
from base_classes.Game_graph_class import GameGraph as gg
from base_classes.General_Game_Graph_class import GeneralGameGraph as ggg
from base_classes.test_run_configuration import test_run_configuration as trc
from time import gmtime, strftime

def run_easter_egg_example():
    # % ============== Configuring File Names to Save Data: ============== %
    file_path = "examples/combined_examples/Easter_Egg_Hunt/"
    fname_matrix = file_path + "static_obstacle_matrix_#.dat"
    fname_grid_w_prop = file_path + "initial_grid_#.png"
    fname_grid_w_static_obs = file_path + "grid_static_obs_#.png"
    fname_propositions = file_path + "propositions_#.dat"
    fname_goal = file_path + "goal_#.dat"
    fname_ani_or = file_path + "test_ani_or_#.avi"                # Animation for or propositions
    fname_ani_and = file_path + "test_ani_or_#.avi"                # Animation for AND propositions
    fname_matrix = fname_matrix.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_propositions = fname_propositions.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_goal = fname_goal.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_ani_or = fname_ani_or.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_ani_and = fname_ani_and.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_grid_w_prop = fname_grid_w_prop.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fname_grid_w_static_obs = fname_grid_w_static_obs.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

    pkl_matrix = open(fname_matrix,"wb")
    pkl_prop = open(fname_propositions, "wb")
    pkl_goal = open(fname_goal,"wb")

    ### ================ Configuring the static test environment ========================== % ###
    # % ================ Setting initial variables for static nodes ======================== %
    M = 6 # No. of rows
    N = 6 # No. of columns
    Nprop = 2 # No. of propositions
    Nmax_prop = 1 # Maximum number of nodes in each proposition

    # Setting up goals and propositions:
    goal_row = M # Row counted starting from 1 top to bottom
    goal_col = N # Column counted starting from 1 left to right
    sys_reach = [N*(goal_row-1) + goal_col]
    nNodes = M*N
    rand = "no"
    via = "ValueFunc"

    if rand == "yes":
        nprop_to_cover = [random.choice(range(1,Nmax_prop+1)) for ii in range(Nprop)] # A list of numbers with each element between 1 and 3
        nNodes_to_cover = [[random.choice(range(1,nNodes+1)) for ii in range(nprop_to_cover[jj])] for jj in range(len(nprop_to_cover))] # Randomly choosing the vertices to cover
    else:
        cell2node = lambda r, c: N*(r-1) + c
        nprop_to_cover = [1,1]
        if via == "ValueFunc":
            cover_cells = [[[3,6]], [[6,3]]]
        if via == "Labels":
            cover_cells = [[[2,6]], [[6,3]]]
        nNodes_to_cover = [[cell2node(cell[0],cell[1]) for cell in cover_cells_ii] for cover_cells_ii in cover_cells]
    
    # % =============== Construct gridworld states and transitions: ============= %
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

    # % =============== Synthesizing static obstacles ========================= %
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
    fig_num += 1
    # Creating static obstacles and plotting them:
    if via == "ValueFunc":
        cut_transitions, static_obstacles = test_config.generate_static_obstacles(nNodes_to_cover, sys_reach)
    if via == "Labels":
        static_obstacles = test_config.generate_static_obstacles_via_labels(nNodes_to_cover, sys_reach)

    test_matrix = test_config.update_matrix(static_obstacles)
    fig2, ax2, im2 = test_config.base_plot(fig2, ax2, im2, nNodes_to_cover, sys_reach) # Constructing the base plot
    fig2, ax2, im2 = test_config.static_obstacle_plot(fig2, ax2, im2, test_matrix[-1]) # Plotting static obstacles on the grid
    fig2.savefig(fname_grid_w_static_obs, dpi=fig.dpi) # Save figure

    # Files to save animation and test matrix:
    pickle.dump(test_matrix[-1], open(fname_matrix,"wb"))
    pkl_matrix.close()
    # plt.show()

    ### ================== End of configuring the static test environment =================== ###
    # ------------------------------------------------------------------------------------------#
    ### ================== Synthesizing Dynamic Test Strategy =============================== ###
    # ToDo: Extend it to multiple coverage goals. For now, we can only do this for one adversary
    # Environment specification comprises of environment behavior: agent can be present/absent in its cell
    # ===============Initializing dynamic agent transitions ======================= #
    # Configuring the gridworld and adding agent transitions:
    static_obstacles_cell = [GW.node2cell[s] for s in static_obstacles[0]]
    GW.add_static_obstacles(static_obstacles_cell)

    # Adding players and transitions:
    GW.grid_transitions()
    Tsys_list = [(1,[1,2,7]), (2,[1,2,3,8]), (3,[2,3,4,9]), (4,[3,4,5,10]), (5,[4,5,6,11]), (6,[5,6,12]), 
    (7,[1,7,8,13]), (8,[2,7,8,9,14]), (9,[3,8,9,10,15]), (10,[4,9,10,11,16]), (11,[5,10,11,12,17]), (12,[6,11,12,18]), 
    (13,[7,13,14, 19]), (14,[8,13,14,15,20]), (15,[9,14,15,16,21]), (16,[10,15,16, 17, 22]), (17, [11,16,17,18,23]), (18, [12,17,18,24]),
    (19,[13,19,20,25]), (20,[14,19,20,21,26]), (21, [15,20,21,22,27]), (22,[16,21,22,23,28]), (23, [17,22,23,24,29]), (24,[18,23,24,30]),
    (25, [19,25,26,31]), (26, [20, 25, 26,27, 32]), (27, [21,26,27,28,33]), (28, [22,27,28,29,34]), (29,[23,28,29,30,35]), (30, [24,29,30,36]),
    (31, [25,31,32]), (32, [26,31,32,33]), (33, [27,32,33,34]), (34, [28,33,34,35]), (35, [29,34,35,36]), (36,[30,35,36])] # System transitions
    Tsys = dict(Tsys_list)
    sys_transitions = []
    env_transitions = []
    for key, val in Tsys.items():
        sys_transitions.append(val)
    Tenv_list = [(1, [1,3,2]), (2,[2]), (3,[1,3]), (4, [4])]
    Tenv = dict(Tenv_list)
    for key, val in Tenv.items():
        env_transitions.append(val)
    unsafe = [[3, 24], [4, 24], [2, 34], [4, 34]]
    Ns = len(sys_transitions)
    Ne = len(env_transitions)
    ## --------------------------------------- Setting game graph: -------------------------------------------------- ##
    Game_Graph = ggg(static_obstacles[0], sys_transitions, env_transitions, unsafe)

    # Removing some edges:
    start1 = "v1_"+str(Game_Graph.state(Ns, Ne, 34, 1))
    end1 = "v2_"+str(Game_Graph.state(Ns, Ne, 34, 2))
    start2 = "v1_"+str(Game_Graph.state(Ns, Ne, 24, 1))
    end2 = "v2_"+str(Game_Graph.state(Ns, Ne, 24, 3))
    rem_edges = [[start1, end1], [start2, end2]]
    Game_Graph.remove_edges(rem_edges)

    # Unsafe Nodes:
    U_nodes= []
    for v in Game_Graph.U:
        p, ns, ne = Game_Graph.unpack_vertex(v)
        U_nodes.append([ne,ns])

    # Finding the system winning set and value function:
    sys_goal_nodes = [[1, 36], [2, 36], [3, 36]]
    goal_vertices = Game_Graph.set_win_states(sys_goal_nodes)

    # Robust Pre for system winning set computation:
    quant_env = 'forall'
    quant_sys = 'exists'
    win_agent = 's'
    Wsys2 = Game_Graph.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
    Val_sys = Game_Graph.get_value_function(win_agent) # System Value Function
    # pdb.set_trace()

    # Setting weights:
    # coverage_props_list = [(lambda ns, ne: ns==33 or ns==18, 1)]
    coverage_props_list = [(lambda ns, ne: ns==33 and ne==2, 2), (lambda ns, ne: ns==18, 1)]
    coverage_props = dict(coverage_props_list)
    Game_Graph.set_vertex_weight(coverage_props)

    # Finding environment winning set and value function:
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

    Wenv, Val_env = compute_env_winning_set(coverage_props_list)
    # ---------------------- This is just for debugging ------------------------------------- #
    # Looking at the nodes:
    Wenv_max = Wenv[-1]
    Wsys_max = Wsys2[-1]
    Wenv_sys_nodes = []
    Wenv_env_nodes = []
    Wsys_sys_nodes = []
    Wsys_env_nodes = []
    for v in Wenv_max[0]:
        state = int(v[3:])
        ne, ns = Game_Graph.state2node(Ns, Ne, state)
        Wenv_env_nodes.append([ne,ns])
    for v in Wenv_max[1]:
        state = int(v[3:])
        ne, ns = Game_Graph.state2node(Ns, Ne, state)
        Wenv_sys_nodes.append([ne,ns])
    for v in Wsys_max[0]:
        state = int(v[3:])
        ne, ns = Game_Graph.state2node(Ns, Ne, state)
        Wsys_env_nodes.append([ne,ns])
    for v in Wsys_max[1]:
        state = int(v[3:])
        ne, ns = Game_Graph.state2node(Ns, Ne, state)
        Wsys_sys_nodes.append([ne,ns])

    # ----------------------- Synthesis of dynamic test strategy: <>p1 or <>p2 ----------------------------- #
    # Choosing initial condition:
    ns_0 = 1
    ne_0 = 1

    q0 = "v1_"+str(Game_Graph.state(Ns, Ne, ns_0, ne_0))
    # pdb.set_trace()
    test_run_or = [q0]
    qcur = q0
    test_length = 70
    next_turn = 'e'
    NTOTAL = []
    NCORRECT = []
    cones = []
    for t in range(1, test_length):
        q = test_run_or[t-1]
        turn = next_turn
        if turn == 'e':
            qn = Game_Graph.test_policy(q, Wsys2)
            test_run_or.append(qn)
            next_turn = 's'
        if turn == 's':
            qn, flag_win, ntotal, ncorrect = Game_Graph.agent_policy(q, cones, Wsys2)
            NTOTAL.append(ntotal)
            NCORRECT.append(ncorrect)
            test_run_or.append(qn)
            next_turn = 'e'
            if flag_win: # If the game has been won stop simulation
                break

    # ----------------------- Synthesis of dynamic test strategy: <>p1 AND <>p2 ----------------------------- #
    # A function to check if a proposition has been covered by an action of the system and to recompute the value function of the graph
    # so that the next proposition is covered
    def update_coverage_props_list(coverage_props_list, qn):
        new_coverage_props_list = coverage_props_list.copy() # Default is the list of propositions with weights
        need_update = False # Default
        rem_prop_lambda = []
        p, ns, ne = Game_Graph.unpack_vertex(qn)
        lC = len(coverage_props_list)
        for ii in range(lC):
            prop_lambda_func = coverage_props_list[ii][0] # Pulling out the lambda function separate from the weights
            if(prop_lambda_func(ns, ne)):
                need_update = True
                new_coverage_props_list.remove(coverage_props_list[ii])
                rem_prop_lambda = prop_lambda_func
        return new_coverage_props_list, rem_prop_lambda, need_update

    def check_prop_coverage(qn, coverage_props_list, Wenv, Val_env):
        new_coverage_props_list, rem_prop_lambda, need_update = update_coverage_props_list(coverage_props_list, qn)
        new_Wenv = Wenv.copy()
        new_Val_env = Val_env.copy()
        if need_update:
            new_Wenv, new_Val_env = compute_env_winning_set(coverage_props_list)
        return new_coverage_props_list, new_Wenv, new_Val_env

    # Notes: The test run doesn't always return the same optimal test run. Sometimes it is bad, and it covers only one proposition. Why? It should always find a policy that works. That means, in the backward reachability, 
    # there is something wrong in how the value function is updated.
    # pdb.set_trace()
    q0 = "v1_"+str(Game_Graph.state(Ns, Ne, ns_0, ne_0))
    test_run_and = [q0]
    qcur = q0
    test_length = 70
    next_turn = 'e'
    NTOTAL = []
    NCORRECT = []
    cones = []
    for t in range(1, test_length):
        q = test_run_and[t-1]
        turn = next_turn
        if turn == 'e':
            qn = Game_Graph.test_policy(q, Wsys2)
            test_run_and.append(qn)
            next_turn = 's'
        if turn == 's':
            qn, flag_win, ntotal, ncorrect = Game_Graph.agent_policy(q, cones,  Wsys2)
            coverage_props_list, Wenv, Val_env = check_prop_coverage(qn, coverage_props_list, Wenv, Val_env) # Checks if a proposition has been covered by state qn, and updates the value function of the game graph accordingly.
            NTOTAL.append(ntotal)
            NCORRECT.append(ncorrect)
            test_run_and.append(qn)
            next_turn = 'e'
            if flag_win: # If the game has been won stop simulation
                break

    ### ================== End of Synthesizing Dynamic Test Strategy =============================== ###
    ## ------------------------ Animating test run --------------------------------------------- ###
    def get_scell(ns):
        if (ns%N == 0):
            col = N
            row = ns//N
        else:
            col = ns%N
            row = ns//N + 1
        return [row, col]

    # Make this more general
    def get_ecell(ne):
        if(ne == 1):
            row = 0
            col = 0
        if (ne == 2):
            row = 6
            col = 4
        if (ne==3):
            cell = get_scell(34)
            row = 4
            col = 6
        if (ne == 4):
            cell1 = get_scell(24)
            cell2 = get_scell(34)
            row = [4, 6]
            col = [6,4]
        return [row, col]

    # Function to get test run nodes from test run:
    def get_test_run_nodes(test_run):
        test_run_nodes = []
        test_run_cells = []
        for v in test_run:
            state = int(v[3:])
            ne, ns = Game_Graph.state2node(Ns, Ne, state)
            test_run_nodes.append([ne, ns])
            scell = get_scell(ns)
            ecell = get_ecell(ne)
            test_state = [ecell, scell]
            test_run_cells.append(test_state)
        return test_run_nodes, test_run_cells

    test_run_or_nodes, test_run_or_cells = get_test_run_nodes(test_run_or)
    test_run_and_nodes, test_run_and_cells = get_test_run_nodes(test_run_and)

    # Saving animations:
    def fig_with_obstacles(fig_num):
        fig, ax, im = GW.base_plot(fig_num)
        fig, ax, im = test_config.base_plot(fig, ax, im, nNodes_to_cover, sys_reach) # Constructing the base plot
        fig, ax, im = test_config.static_obstacle_plot(fig, ax, im, test_matrix[-1]) # Plotting static obstacles on the grid
        return fig, ax, im
    skip_transitions = ['e'] # i.e the smooth transitions part can be skipepd
    fig, ax, im = fig_with_obstacles(fig_num)
    fig_num+=1
    anim1 = GW.animate_test_run_gg(test_run_or_cells, fig, ax, skip_transitions)
    writervideo = animation.FFMpegWriter(fps=60)
    anim1.save(fname_ani_or, writer=writervideo)

    print("Printing test run")
    print(test_run_or_nodes)
    # fig, ax, im = fig_with_obstacles(fig_num)
    # fig_num+=1
    # anim2 = GW.animate_test_run_gg(test_run_and_cells, fig, ax, skip_transitions)
    # writervideo = animation.FFMpegWriter(fps=60)
    # anim2.save(fname_ani_and, writer=writervideo)
    plt.show()