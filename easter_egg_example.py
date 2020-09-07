from examples.easter_egg_hunt import easter_egg
from dynamic_obstacles.gridworld import plot_value_functions

GW, GameGraph, propositions, goal, test_config, test_matrix = easter_egg.run_easter_egg_example()
fignum = 100
k = 1
plot_value_functions.make_plots(GW, GameGraph, propositions, goal, test_config, test_matrix, k, fignum)

