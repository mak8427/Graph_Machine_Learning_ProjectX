import matplotlib.pylab as plt
import numpy as np

from hik.data.kitchen import Kitchen
from hik.data import PersonSequences
from hik.vis import plot_pose
from networkx.classes import nodes

dataset = "A"
# load geometry
kitchen = Kitchen.load_for_dataset(
    dataset=dataset,
    data_location="data/scenes"
)

# load poses
person_seqs = PersonSequences(
    person_path="data/poses"
)

smplx_path = "data/body_models"
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

dataset = "C"  # ["A", "B", "C", "D"]
frame = 5
kitchen.plot(ax, frame)
for person in person_seqs.get_frame(dataset, frame):
    plot_pose(ax, person["pose3d"], linewidth=1)
ax.axis('off')

import pandas as pd
nodes_data = person["pose3d"]
# Define edges (pairs of node indices)
edges = [
    (1, 0),
    (0, 2),
    (1, 4),
    (4, 7),
    (7, 10),
    (2, 5),
    (5, 8),
    (8, 11),
    (16, 18),
    (18, 20),
    (17, 19),
    (19, 21),
    (16, 13),
    (17, 14),
    (13, 12),
    (14, 12),
    (12, 15),
    (15, 23),
    (24, 25),
    (24, 26),
    (25, 27),
    (26, 28),
]


# Enable interactive mode
plt.ion()

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
ax.scatter(nodes_df['x'], nodes_df['y'], nodes_df['z'], c='b', label='Nodes')

# Plot edges
for edge in edges:
    start_node = nodes_df.iloc[edge[0]]
    end_node = nodes_df.iloc[edge[1]]
    ax.plot(
        [start_node['x'], end_node['x']],
        [start_node['y'], end_node['y']],
        [start_node['z'], end_node['z']],
        c='r'
    )

# Labels and legend
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Node and Edge Plot')
ax.legend()

# Rotation loop
for angle in range(0, 360, 2):
    ax.view_init(30, angle)  # Elevation, Azimuth
    plt.draw()
    plt.pause(0.01)

plt.ioff()  # Turn off interactive mode
plt.show()