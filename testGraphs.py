import networkx as nx
from matplotlib import pyplot as plt




def find_cycle_and_adjust(matrix, preferences):
    # Create a directed graph
    graph = nx.DiGraph()
    add_node_to_graph(matrix,graph)

    while True:
        re_add_edges_to_graph(matrix, graph, preferences)
        try:
            adjust_matrix_if_cycle(matrix, graph, preferences)
            # plot_graph(graph)
        except:
            break


def add_node_to_graph(matrix,graph):
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            node = (i, j)
            weight = matrix[i][j]
            graph.add_node(node, weight=weight)

def re_add_edges_to_graph(matrix, graph, preferences):
    graph.clear_edges()
    # check if according to preference i want this item
    # check if item has to someone
    # if all true make arrow "I want this"
    for me, p in enumerate(preferences):
        # reverse the list because i want the higher from me
        for item_index, item in enumerate(p[::-1]):
            for him in range(len(matrix)):
                for other_item in p[::-1][item_index + 1:]:
                    if him == me: continue
                    # index,p[index] -> agent,item
                    if matrix[him][other_item] <= 0: continue
                    if matrix[me][item] >= 1 : continue
                    graph.add_weighted_edges_from([((me, item), (him, other_item), matrix[him][other_item])])

def adjust_matrix_if_cycle(matrix, graph, preferences):
    try:
        # This will return the first cycle it finds
        cycle = nx.find_cycle(graph, orientation='original')
        # print("Cycle found:", cycle)
    except nx.NetworkXNoCycle:
        print("Cycle not found:")
        raise ValueError("no Cycle")
    cycle = cycle[0][:-1]
    min_weight = min(graph[u][v]['weight'] for u, v in zip(cycle, cycle[1:] + cycle[:1]))
    for (u_agent, u_item), (v_agent, v_item) in zip(cycle, cycle[1:] + cycle[:1]):
        matrix[v_agent][v_item] -= min_weight
        matrix[u_agent][v_item] += min_weight



    return

def plot_graph(G):
    # Define positions for the nodes based on their (i, j) coordinates
    pos = {(i, j): (j, -i) for i, j in G.nodes()}  # x=j, y=-i to mimic a top-left matrix layout

    # Create labels that include node name and its weight
    labels = {
        (i, j): f"{(i, j)}\n{attr['weight']}"
        for (i, j), attr in G.nodes(data=True)
    }

    # Draw the graph
    plt.figure(figsize=(8, 6))  # Adjust the size as needed
    nx.draw(G, pos, with_labels=False, node_size=700, node_color="lightblue", edge_color="gray", arrowsize=15)
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    # Draw edge labels, which include the weight of the edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Directed Graph in Matrix Layout with Weights")
    plt.show()

if __name__ == '__main__':
    mat = [[0.33333333 ,0.33333333 ,0.33333333], [0.33333333 ,0.33333333, 0.33333333], [0.33333333 ,0.33333333 ,0.33333333]]
    pref = [[1, 0, 2], [1, 2, 0], [0, 2, 1]]
    # mat = [[0.25 ,0.25, 0.25, 0.25, 0.], [0.25, 0.25 ,0.25, 0.,   0.25], [0. ,  0.25, 0.25, 0.25, 0.25], [0.25, 0. ,  0.25, 0.25, 0.25],
    #  [0.25 ,0.25 ,0. ,  0.25 ,0.25]]
    # pref = [[1, 3, 4, 0, 2], [1, 0, 4, 2, 3], [3, 1, 4, 0, 2], [0, 2, 1, 4, 3], [3, 4, 0, 2, 1]]
    find_cycle_and_adjust(mat, pref)