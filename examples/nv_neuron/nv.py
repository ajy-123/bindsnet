import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_weights
from matplotlib import animation
import networkx as nx



from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.network import Network

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, NVNodes
from bindsnet.network.topology import Connection

# Manually defined parameters
dt = 0.5
device = "cpu"
time = 100



# Parameters
num_rings = 2
nodes_per_ring = 5
total_nodes = num_rings * nodes_per_ring
COUPLING_CONSTANT = 1

# Create simple Torch NN
network = Network(dt=dt)
ring = NVNodes(total_nodes)
network.add_layer(ring, name="Ring")

def generate_adjacency_matrix(num_rings, nodes_per_ring):
    total_nodes = num_rings * nodes_per_ring

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((total_nodes, total_nodes), dtype=int)

    # Fill in adjacency matrix for each ring
    for ring in range(num_rings):
        for node in range(nodes_per_ring):
            from_node = ring * nodes_per_ring + node
            to_node = ring * nodes_per_ring + (node + 1) % nodes_per_ring  # Connect to the next node in the same ring
            adjacency_matrix[from_node, to_node] = 1

            # Connect to corresponding nodes in other rings
            for other_ring in range(num_rings):
                if other_ring != ring:  # Skip the current ring (already connected)
                    to_node = other_ring * nodes_per_ring + node
                    adjacency_matrix[from_node, to_node] = COUPLING_CONSTANT

    return adjacency_matrix

# Example usage:
num_rings = 2
nodes_per_ring = 5
adjacency_matrix = generate_adjacency_matrix(num_rings, nodes_per_ring)
print(adjacency_matrix)

adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)

def visualize_adjacency_matrix(adjacency_matrix):
    # Create a graph from the adjacency matrix
    G = nx.Graph(np.array(adjacency_matrix))
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # You can choose different layout algorithms
    nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=8)
    
    # Display the graph
    plt.title("Square Lattice with Extended Nodes")
    plt.show()
    plt.savefig("examples/nv_neuron/coupling_constant_0.1/adjacency_graph.png")

visualize_adjacency_matrix(adjacency_matrix)

# Define adjacency matrix that describes a ring
C1 = Connection(source=ring, target=ring, w=adjacency_matrix_tensor)
network.add_connection(C1, source="Ring", target="Ring")
ring.in_degree = torch.mul(torch.ones(total_nodes), 2)

# Monitors for visualizing activity
v_out = Monitor(ring, ["s"], time=time, device=device)
network.add_monitor(v_out, name="v_out")
v_cap = Monitor(ring, ["v"], time=time, device=device)
network.add_monitor(v_cap, name="v_cap")
w_monitor = Monitor(C1, ["w"], time = time, device = device)
network.add_monitor(w_monitor, name = "w_monitor")


# Set initial state of the network
ring.v[0] = 0.1
ring.s[0] = 0
ring.y[0] = 0

# 
network.run(inputs = dict(), time=time)
outs = v_out.get("s")[:,0].t()
caps = v_cap.get("v")[:,0]

caps = caps.reshape(10, 10)



plot_weights(C1.w, 0, 1, im = None, figsize =  (8,3), cmap="hot_r", save="examples/nv_neuron/weight_map.png")
plot_weights(outs, 0, 1, im = None, figsize = (8,3), cmap = "hot_r", save ="examples/nv_neuron/output_map.png")
plot_weights(caps, 0, 1, im = None, figsize = (8,3), cmap = "hot_r", save ="examples/nv_neuron/capacitor_map.png")