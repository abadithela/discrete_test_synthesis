#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:11:51 2020

@author: apurvabadithela
"""


import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


fig, ax = plt.subplots()

resolution = 5  # the number of vertices
N = 3
# x = np.array([0, 1, 2])
# y = np.array([0, 1, 2])
# radii = 0.1*np.array([0, 1, 2])
patches = []
# for x1, y1, r in zip(x, y, radii):
#     circle = Circle((x1, y1), r)
#     patches.append(circle)

x = np.array([0])
y = np.array([0])
radii = 0.1*np.array([0])
theta1 = 360.0*np.array([0])
theta2 = 360.0*np.array([1])
for x1, y1, r, t1, t2 in zip(x, y, radii, theta1, theta2):
    wedge = Wedge((x1, y1), r, t1, t2)
    patches.append(wedge)

# Some limiting conditions on Wedge
patches += [
    # Wedge((.3, .7), .1, 0, 360),             # Full circle
    # Wedge((.7, .8), .2, 0, 360, width=0.05),  # Full ring
    Wedge((.6, .2), .2, 0, 180),              # Full sector
    # Wedge((.8, .3), .2, 45, 90, width=0.10),  # Ring sector
]

# for i in range(N):
#     polygon = Polygon(np.random.rand(N, 2), True)
#     patches.append(polygon)

colors = 100*np.random.rand(len(patches))
p = PatchCollection(patches, alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
fig.colorbar(p, ax=ax)

plt.show()