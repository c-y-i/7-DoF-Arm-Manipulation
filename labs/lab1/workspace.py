from lib.calculateFK import FK
from core.interfaces import ArmController

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from math import pi

fk = FK()

# the dictionary below contains the data returned by calling arm.joint_limits()
limits = [
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -1.7628, 'upper': 1.7628},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -3.0718, 'upper': -0.0698},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -0.0175, 'upper': 3.7525},
    {'lower': -2.8973, 'upper': 2.8973}
 ]

# TODO: create plot(s) which visualize the reachable workspace of the Panda arm,
# accounting for the joint limits.
#
# We've included some very basic plotting commands below, but you can find
# more functionality at https://matplotlib.org/stable/index.html

# x_points, y_points, z_points = [], [], []

q_var = [0, 0, 0, -pi/2, 0, pi/2, pi/4]
result = fk.forward(q_var)
joint_pos, T0e = result[0], result[1]
x_points = joint_pos[:, 0]
y_points = joint_pos[:, 1]
z_points = joint_pos[:, 2]

# plt.plot()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# TODO: update this with real results
# ax.scatter(1,1,1) # plot the point (1,1,1)
ax.scatter(x_points, y_points, z_points, c='r', marker='o')

# Plot lines connecting the joints to represent the links between them
ax.plot(x_points, y_points, z_points, c='k', linewidth=2)

# Optionally: Plot the end-effector separately to highlight it
ax.scatter(T0e[0, 3], T0e[1, 3], T0e[2, 3], c='b', marker='x', s=100, label='End-Effector')

plt.show()
