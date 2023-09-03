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
n_neurons = 6
device = "cpu"
time = 100

# Create simple Torch NN
network = Network(dt=dt)
ring = NVNodes(n_neurons)
network.add_layer(ring, name="Ring")

# Define adjacency matrix that describes a ring
W_ring = torch.zeros(n_neurons, n_neurons)
W_ring[torch.arange(n_neurons),torch.arange(-1,n_neurons-1)%n_neurons] = 1
print(W_ring)
C1 = Connection(source=ring, target=ring, w=W_ring)
network.add_connection(C1, source="Ring", target="Ring")
ring.in_degree = torch.ones(n_neurons)

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