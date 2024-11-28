import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def find_cycle(matrix, preferences):
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes with weights
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            node = (i, j)
            weight = matrix[i][j]
            graph.add_node(node, weight=weight)

    # check if according to preference i want this item
    # check if item has to someone
    # if all true make arrow "I want this"
    for index, p in enumerate(preferences):
        for item in p[index+1:]:
            if item == index: continue
            for agent in range(len(matrix)):
                if agent == index: continue
                # index,index -> agent,item




    plot_graph(graph)

def plot_graph(G):
    pos = nx.spring_layout(G)  # Positions for all nodes
    labels = {node: f"{node}\n{attr['weight']}" for node, attr in G.nodes(data=True)}  # Include (i, j) and weight

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=False, node_size=700, node_color="lightblue", edge_color="gray", arrowsize=15)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    plt.title("Directed Graph from Matrix")
    plt.show()

if __name__ == '__main__':
    mat = [[0.25 ,0.25, 0.25, 0.25, 0.], [0.25, 0.25 ,0.25, 0.,   0.25], [0. ,  0.25, 0.25, 0.25, 0.25], [0.25, 0. ,  0.25, 0.25, 0.25],
     [0.25 ,0.25 ,0. ,  0.25 ,0.25]]
    pref = [[1, 3, 4, 0, 2], [1, 0, 4, 2, 3], [3, 1, 4, 0, 2], [0, 2, 1, 4, 3], [3, 4, 0, 2, 1]]
    find_cycle(mat, pref)