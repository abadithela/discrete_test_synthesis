# % ================ Formal Methods for T&E and V&V of autonomous systems =============== #
# % ================ Apurva Badithela // 6/2/2020 =============== #
# % ================ File defining grid world environment ============= #
# This file contains code for 2 players on the grid...
# % Gridworld class:
# %     + init: Takes in size of the grid as list [M,N] where M is the number of rows and N is the number of columns
import numpy as np
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

from base_classes.gridworld_class import GridWorld as gc
from base_classes.Player_class import Player as pc
from base_classes.Game_graph_class import GameGraph as gg

from time import gmtime, strftime

# -----------------------------------------------------------------------------------------------------------------------------------------#
# % ======= Main function to test the classes ========== % #
def gridworld_example():
    # Filenames:
    file_path = "examples/dynamic_obstacles_only/"
    gen_file_avi = file_path+"Animations/dynamic_obstacle_#"
    gen_file_fig = file_path+"Figures/dynamic_obstacle_#"
    gen_file_avi.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    gen_file_fig.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    fn = gen_file_avi+".avi"
    fn_plot = gen_file_fig+".png"

    # Begin actual examples:
    ex = 2
    if ex == 1:
        M = 10
        N = 10
        # Construct gridworld states and transitions:
        GW = gc([M,N])
        # Adding players and transitions:
        GW.grid_transitions()
        obs_col  = M-3
        tester_col = obs_col+1
        tester_bump_row = math.floor(N/2)
        agent_transitions = ['all']
        tester_transitions = ['specific', tester_col, tester_bump_row]
        sys_goal_cell = [M, N]
        env_goal_cell = [[ii, tester_col] for ii in range(1,M+1) if ii!=tester_bump_row]
        extra_cells = [[tester_bump_row-1, tester_col+1], [tester_bump_row, tester_col+1], [tester_bump_row+1, tester_col+1]]
        env_goal_cell.extend(extra_cells)

    if ex == 2:
        M = 10
        N = 10
        # Construct gridworld states and transitions:
        obs_col  = M-3
        GW = gc([M,N])
        static_obs = [[ii, obs_col] for ii in range(2,N)]
        GW.add_static_obstacles(static_obs)
        # Adding players and transitions:
        GW.grid_transitions()
        tester_col = obs_col+1
        agent_transitions = ['all']
        tester_transitions = ['specific', tester_col]
        sys_goal_cell = [M, N]
        env_goal_cell = [[ii, tester_col] for ii in range(1,M+1)]

    if ex == 3:
        M = 4
        N = 4
        # Construct gridworld states and transitions:
        GW = gc([M,N])
        # Adding players and transitions:
        GW.grid_transitions()
        Tsys_list = [(1,[1,2,5]), (2,[1,2,3,6]), (3,[2,3,4,7]), (4,[3,4,8]), (5,[1,5,6,9]), (6,[2,5,6,7,10]), (7,[3,6,7,8,11]), (8,[4,7,8,12]), (9,[5,9,10,13]), (10,[6,9,10,11,14]), (11,[7,10,11,12,15]), (12,[8,11,12,16]), (13,[9,13,14]), (14,[10,13,14,15]), (15,[11,14,15,16]), (16,[12,15,16])] # System transitions
        Tsys = dict(Tsys_list)
        agent_transitions=['very_specific', Tsys]
        
        Tenv_list = [(1, [1]), (2,[3,6]), (3,[2,7]), (4, [4]), (5, [5]), (6,[2,10]), (7,[3,11]), (8, [8]), (9, [9]), (10, [6,14]), (11, [7,15]), (12, [12]), (13, [13]), (15, [11,14]), (14, [10, 15]), (16, [16])]
        
        Tenv = dict(Tenv_list)
        tester_transitions = ['very_specific', Tenv]
        sys_goal_cell = [M,N]
        env_goal_cell = [[1,2], [1,3], [2,2], [2,3], [3,2], [3,3], [4,2], [4,3]]

    # Simple runner blocker:
    if ex == 4:
        M = 5
        N = 1
        Ns = 5
        Ne = 5
        # Construct gridworld states and transitions:
        GW = gc([M,N])
        # Adding players and transitions:
        GW.grid_transitions()
        Tenv_list = [(1,[1]), (2,[3]), (3, [2,4]), (4, [3]), (5, [5])]
        Tenv = dict(Tenv_list)
        Tsys_list = [(1, [1,2,3,4]), (2, [1,2,3,5]), (3, [1,2,3,4,5]), (4, [1,3,4,5]), (5, [2,3,4,5])]
        Tsys = dict(Tsys_list)
        agent_transitions=['very_specific', Tsys]
        tester_transitions = ['very_specific', Tenv]
        sys_goal_cell = [M,N]
        env_goal_cell = [[2,1], [3,1], [4,1]]

    GW.add_player('agent', agent_transitions, 's')
    GW.add_player('tester', tester_transitions, 'e')

    # Checking player transitions have been correctly added:
    tester = GW.get_player('tester')
    tester_transitions2 = tester.get_transitions()

    # Making game graph and getting vertices and edges:
    GW.construct_graph()
    GAME = GW.get_game_graph()
    E, V = GAME.get_edges_vertices()

    # Add trace here to make sure that the edges and vertices are set correctly
    # pdb.set_trace()

    # Finding the system winning set and value function:
    goal_cells = [[env_cell, sys_goal_cell] for env_cell in env_goal_cell]
    goal_vertices = GAME.set_win_states(goal_cells)
    assert(all([elem in V for elem in goal_vertices[0]]))
    assert(all([elem in V for elem in goal_vertices[1]]))

    # Robust Pre for system winning set computation:
    quant_env = 'forall'
    quant_sys = 'exists'
    win_agent = 's'
    Wsys2 = GAME.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
    assert(Wsys2[-1]==Wsys2[-2]) # Assertion to make sure we have the right fixpoint or else we need to increase the threshold in Game_graph_class.py

    # Wsys, Wsys_env, Wsys_sys = GAME.synt_winning_set2(win_agent, goal_vertices, quant_env, quant_sys)
    Val_sys = GAME.get_value_function(win_agent)
    # Check Value function:
    Wenv_phisys, Wsys_phisys = GAME.win_cells(Wsys2)
    nmax_sys = len(Wenv_phisys[-1]) + len(Wsys_phisys[-1])

    # # Testing base plot
    # fig = 1
    # ax, im = GW.base_plot(fig)
    env_locations = [tuple(ei) for ei in env_goal_cell]
    min_fig = 1 # Minimum figure index
    # Shaded out unless figures need to be plotted to save memory
    # fig_list, im_list, ax_list = GW.plot_win_set(Wsys_phisys, env_locations, 's', min_fig)
    min_fig += len(env_locations)

    # fig2_list, im2_list, ax2_list = GW.plot_win_set(Wenv_phisys, env_locations, 's', min_fig)
    min_fig += len(env_locations)

    # # Add trace here for the winning set synthesis. Good
    # pdb.set_trace()
    # Finding the environment winning set and value function:
    # !!!! The environment and the system places need to be switched
    # IMPORTANT_NOTE: When we find the environment winning set, we need to have environment and system cells switched
    if ex == 1:
        bump_row=5
        sys_goal_cell = [[1, obs_col], [M, obs_col]]
        env_goal_cell = [[ii, tester_col] for ii in range(1,M+1) if ii!=bump_row]
        extra_cells = [[bump_row-1, tester_col+1], [bump_row, tester_col+1], [bump_row+1, tester_col+1]]
        env_goal_cell.extend(extra_cells)
    if ex == 2:
        setting = "single_goal"
        if(setting == "double_goal"):
            sys_goal_cell = [[1, obs_col], [M, obs_col]]
            env_goal_cell = [[ii, tester_col] for ii in range(1,M+1)]
            grid_pos_cover = [[10, 7], [1,7]]
            weight = [2,1]
        # Print one goal cell at a time:
        if(setting == "single_goal"):
            sys_goal_cell = [[M, obs_col]]
            env_goal_cell = [[ii, tester_col] for ii in range(1,M+1)]
            grid_pos_cover = [[10, 7]]
            weight = [2]
    goal_cells = [[env_cell, sys_cell] for env_cell in env_goal_cell for sys_cell in sys_goal_cell]
    goal_vertices = GAME.set_win_states(goal_cells)

    # Robust Pre for system winning set computation:
    quant_env = 'exists'
    quant_sys = 'exists'
    win_agent = 'e'

    # Call set vertex weight here:
    SYS_DICT = GAME.get_dict('s')
    C2N = GW.get_C2N()
    SYS_DICT_grid_to_ns = SYS_DICT[0]
    node_cover = [C2N[tuple(c)] for c in grid_pos_cover]
    state_cover = [SYS_DICT_grid_to_ns[ii] for ii in node_cover]
    coverage_props_list = []
    assert(len(state_cover)==len(weight))
    lambda_fncs = []
    for jj in range(len(state_cover)):
        lambda_fnc = lambda ns, ne: ns == state_cover[jj]
        lambda_fncs.append(lambda_fnc)
    for ii in range(len(weight)):
        coverage_props_list.append((lambda_fncs[ii], weight[ii]))
    coverage_props = dict(coverage_props_list)
    GAME.set_vertex_weight(coverage_props) # Coverage_props is a dictionary consisting of lambda functions and the weights associated with states at which those lambda functions are true
    # # Set point to check that propositions had been set correctly:
    # pdb.set_trace()
    W_env = GAME.win_reach(win_agent, goal_vertices, quant_env, quant_sys)
    assert(W_env[-1]==W_env[-2]) # Assertion to make sure we have the right fixpoint or else we need to increase the threshold in Game_graph_class.py
    Val_env = GAME.get_value_function(win_agent)

    # # Check Value function:
    # num_val_phienv = 0 # Number of states with value function that is not zero for environment specifications
    # for v in V:
    #     if Val_env[v] > 0:
    #         num_val_phienv += 1
    Wenv_phienv, Wsys_phienv = GAME.win_cells(W_env)
    nmax_env = len(Wenv_phienv[-1]) + len(Wsys_phienv[-1])
    # # Check that environment winning set is properly satisfied: Good
    # pdb.set_trace()

    # Getting the number of states with a value function:
    num_val_phienv, num_val_phisys = GAME.get_val_numbers()
    # #Plotting env_winning_set:
    # fig3_list, im3_list, ax3_list = GW.plot_win_set(Wsys_phienv, env_locations, 'e', min_fig)
    min_fig += len(env_locations)
    # fig4_list, im4_list, ax4_list = GW.plot_win_set(Wenv_phienv, env_locations, 'e', min_fig)
    min_fig += len(env_locations)
    plt.show()

    show_base_plot ="no"
    # # Creating base plot to illustate example:
    if show_base_plot == "yes":
        fignum = 100
        msz = 12
        fig, ax, im = GW.base_plot(fignum)
        env_ex = [4,8]
        sys_ex = [1,3]
        goal_ex = [M,N]
        cov_ex = sys_goal_cell.copy()
        ax.plot(env_ex[1], env_ex[0], "g*", markersize=msz) # Plotting environment
        ax.plot(sys_ex[1], sys_ex[0], "b*", markersize=msz) # Plotting system
        ax.plot(goal_ex[1], goal_ex[0], "kX", markersize=msz) # Plotting goal
        ax.plot(cov_ex[0][1], cov_ex[0][0], "k4", markersize=msz) # Plotting environment
        ax.plot(cov_ex[1][1], cov_ex[1][0], "k4", markersize=msz) # Plotting environment

    #   Printing the value functions:
    print_val_function = "no"
    if print_val_function == "yes":
        print(Val_sys)
        print(Val_env)

    # ================== Simulating a test run: =========================== #
    # Choosing an initial condition:
    q0 = GAME.choose_init(W_env, Wsys2)
    cell_q0 = GAME.get_vertex(q0)
    cones = []

    test_run = [q0]
    test_run_cells = [cell_q0]
    qcur = q0
    test_length = 70
    next_turn = 'e'
    NTOTAL = []
    NCORRECT = []

    for t in range(1, test_length):
        q = test_run[t-1]
        turn = next_turn
        if turn == 'e':
            qn = GAME.test_policy(q, Wsys2)
            cell_qn = GAME.get_vertex(qn)
            test_run.append(qn)
            test_run_cells.append(cell_qn)
            next_turn = 's'
        if turn == 's':
            qn, flag_win, ntotal, ncorrect = GAME.agent_policy(q, cones,  Wsys2)
            NTOTAL.append(ntotal)
            NCORRECT.append(ncorrect)
            cell_qn = GAME.get_vertex(qn)
            test_run.append(qn)
            test_run_cells.append(cell_qn)
            next_turn = 'e'
            if flag_win: # If the game has been won stop simulation
                break

    # pdb.set_trace() # Breakpoint to check if simulation ran correctly. Good
    # ================= Calling a function to simulate the test run ======================== #
    anim = GW.animate_test_run(test_run_cells, min_fig, fn)
    writervideo = animation.FFMpegWriter(fps=60)
    anim.save(fn, writer=writervideo)
    min_fig+=1
    # ================= Calling a function to plot how many correct actions the system had at each step of the test run =================== #
    create_plot_correct_actions = 1
    if create_plot_correct_actions:
        msz = 6
        ltest = len(NCORRECT)
        # assert(ltest == int(ltest)) # Checking that it is a whole number
        xval = [ii for ii in range(ltest)]
        fig = plt.figure(min_fig)
        ax = plt.gca()
        # Labels for major ticks
        ax.set_xticks(np.arange(0, ltest+1, 1))
        ax.set_yticks(np.arange(0, 6, 1))
        ltotal, = plt.plot(xval, NTOTAL, "r.", markersize=2*msz, label="# total actions")
        lcorrect, = plt.plot(xval, NCORRECT, "b.", markersize=msz, label="# correct actions")
        plt.legend(handles=[ltotal, lcorrect])
        plt.savefig(fn_plot)

    plt.show()