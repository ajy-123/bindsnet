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
import random
import math

import networkx as nx

def construct_graph(ROWS, COLS, A, B, C, D):
    num_vertices = 0
    rings = {}
    graph = nx.DiGraph()

    #Initialize the corners of the graph
    for i in range(ROWS):
        for j in range(COLS):
            rings[(i,j)] = {}

            if i == 0 and j == 0:
                t = [num_vertices, num_vertices + 1]
                num_vertices += 2
                rings[(i,j)]["T"] = t

                b = [num_vertices, num_vertices + 1]
                num_vertices += 2
                rings[(i,j)]["B"] = b
            elif i == 0:
                t = [rings[(i, j-1)]["T"][1], num_vertices + 1]
                num_vertices +=1
                rings[(i,j)]["T"] = t

                b = [rings[(i, j-1)]["T"][1], num_vertices + 1]
                num_vertices +=1
                rings[(i,j)]["B"] = b
            elif j == 0:
                rings[(i,j)]["T"] = rings[i-1, j]["B"]

                b = [num_vertices, num_vertices + 1]
                num_vertices += 2
                rings[(i,j)]["B"] = b
            else:
                rings[(i,j)]["T"] = rings[i-1, j]["B"]

                b = [rings[(i, j-1)]["B"][1], num_vertices + 1]
                num_vertices += 1
                rings[(i,j)]["B"] = B
    #Initialize A, B, C, and D Nodes in the graph
    for i in range(ROWS):
        for j in range(COLS):

            l = []
            t = []
            r = []
            b = []

            #left nodes
            if j == 0 and (i + j) % 2 == 0:
                if D > 0:
                    l.extend(range(num_vertices, num_vertices + D))
                    num_vertices += D
            
            elif j == 0 and (i + j) % 2 == 1:
                if B > 0:
                    l.extend(range(num_vertices, num_vertices + B))
                    num_vertices += B
            else:
                l = rings[(i,j - 1)]["R"]
            
            rings[(i,j)]["L"] = l

            #right nodes
            if (i + j) % 2 == 0:
                if B > 0:
                    r.extend(range(num_vertices, num_vertices + B))
                    num_vertices += B
                elif (i + j) % 2 == 1:
                    r.extend(range(num_vertices, num_vertices + D))
                    num_vertices += D

            rings[(i,j)]["R"] = r
            
            #top nodes
            if i == 0:
                t.append(rings[(i,j)]["T"][0])
                if A > 0:
                    t.extend(range(num_vertices, num_vertices + A))
                    num_vertices += A
                t.append(rings[(i,j)]["T"][1])
            else:
                t = rings[(i-1 , j)]["B"]
            
            rings[(i,j)]["T"] = t

            #bottom nodes
            if i % 2 == 0:
                b.append(rings[(i,j)]["B"][0])
                if C > 0:
                    b.extend(range(num_vertices, num_vertices + C))
                    num_vertices += C
                b.append(rings[i,j]["B"][1])
            
            else:
                b.append(rings[(i,j)]["B"][0])
                if A > 0:
                    b.extend(range(num_vertices, num_vertices + A))
                    num_vertices += A
                b.append(rings[i,j]["B"][1])
            
            rings[i,j]["B"] = b

    #CONNECT NODES FROM DICITONARY TO GRAPH
    #clockwise (i + j) mod 2 is 0: left -> top -> right -> bottom (left to right)
    #counterclockwise (i +j) mod 2 is 1: bottom -> right -> top -> left (right to left)
    for i in range(ROWS):
        for j in range(COLS):
            currRing = rings[(i, j)]
            #CLOCKWISE
            if (i + j) % 2 == 0:
                # Connect right nodes (also connecting right to bottom)
                if len(currRing["R"]) == 0:
                    graph.add_edge(currRing["T"][-1], currRing["B"][-1])
                else:
                    for k in range(len(currRing["R"])):
                        if(k == len(currRing["R"]) - 1):
                            graph.add_edge(currRing["R"][k], currRing["B"][-1])
                            continue
                        graph.add_edge(currRing["R"][k], currRing["R"][k + 1])
                
                # Connect left nodes (also connect left to top)
                if len(currRing["L"]) == 0:
                    graph.add_edge(currRing["B"][0], currRing["T"][0])
                else:
                    for k in reversed(range(len(currRing["L"] ))):
                        if(k == len(currRing["L"]) - 1):
                            graph.add_edge(currRing["L"][k], currRing["T"][0])
                            continue
                        graph.add_edge(currRing["L"][k], currRing["L"][k - 1])
                
                #Connect top nodes (also top to right)
                for k in range(len(currRing["T"])):
                    if(k == len(currRing["T"]) - 1 and len(currRing["R"]) > 0):
                        graph.add_edge(currRing["T"][k], currRing["R"][0])
                        continue
                    graph.add_edge(currRing["T"][k], currRing["T"][k+1])


                #Connect bottom nodes (also bottom to left)
                for k in reversed(range(len(currRing["B"]))):
                    if(k == 0 and len(currRing["L"]) > 0):
                        graph.add_edge(currRing["B"][k], currRing["L"][0])
                    graph.add_edge(currRing["B"][k], currRing["B"][k-1])
        
            #COUNTERCLOCKWISE
            else:
                # Connect right nodes (also connect right to top)
                if len(currRing["R"]) == 0:
                    graph.add_edge(currRing["B"][-1], currRing["T"][-1])
                else:
                    for k in reversed(range(len(currRing["R"]))):
                        if(k == len(currRing["R"])- 1):
                            graph.add_edge(currRing["R"][k], currRing["T"][-1])
                            continue
                        graph.add_edge(currRing["R"][k], currRing["R"][k - 1])

                # Connect left nodes (also connect left to bottom)
                if len(currRing["L"]) == 0:
                    graph.add_edge(currRing["T"][0], currRing["B"][0])
                else:
                    for k in range(len(currRing["L"])):
                        if(k == len(currRing["L"]) - 1):
                            graph.add_edge(currRing["L"][k], currRing["B"][0])
                            continue
                        graph.add_edge(currRing["L"][k], currRing["L"][k + 1])

                # Connect top nodes
                for k in reversed(range(len(currRing["T"]))):
                    if(k == 0 and len(currRing["L"]) > 0):
                        graph.add_edge(currRing["B"][k], currRing["L"][0])
                    graph.add_edge(currRing["L"][k], currRing["L"][k-1])

                # Connect bottom nodes
                for k in range(len(currRing["B"])):
                    if(k == len(currRing["B"]) - 1 and len(currRing["R"]) > 0):
                        graph.add_edge(currRing["B"][k], currRing["R"][0])
                        continue
                    graph.add_edge(currRing["B"][k], currRing["B"][k+1])
    print(rings)
                    
    return rings, graph

# Example usage:
ROWS = 1
COLS = 1
A = 1
B = 1
C = 1
D = 1

rings, graph = construct_graph(ROWS, COLS, A, B, C, D)



# Draw the graph
pos = nx.spring_layout(graph, seed=42)  # You can choose different layout algorithms
nx.draw(graph, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=8)

# Display the graph
plt.title("Square Lattice with Extended Nodes")
plt.show()
plt.savefig("examples/nv_neuron/example_graph.png")

