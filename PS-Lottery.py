import copy
import math
import random
from time import perf_counter

import networkx as nx
import numpy
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment


def bikroft(matrix: numpy.array) -> list:
    """
    Consider any random allocation with n agents and n
    items in which each agent gets one unit of items. Birkhoff’s algorithm can
    decompose such a random allocation (which can be represented by a bistochastic
    matrix) into a convex combination of at most n2 −n+1 deterministic allocations
    (represented by permutation matrices)
    :param matrix: square bistochastic
    :return P: deterministic allocations represented by permutation matrices with probability for each matrix
    """
    # check is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("must be a square matrix")
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    if not np.allclose(row_sums, 1) or not np.allclose(col_sums, 1) or not np.all(matrix >= 0):
        print(matrix)
        raise ValueError("must be a bistochastic matrix")

    P = []
    M0 = np.zeros((matrix.shape[0], matrix.shape[1]))

    while not np.allclose(matrix, M0):
        # Replace zeros with a large value
        modified_matrix = np.where(matrix == 0, np.Inf, matrix)
        # Apply linear_sum_assignment to find the optimal assignment
        row_indices, col_indices = linear_sum_assignment(modified_matrix)
        # Find the edges corresponding to the matched pairs
        edges = [(row, col) for row, col in zip(row_indices, col_indices) if matrix[row, col] != 0]

        scalar = min(matrix[x][y] for x, y in edges)

        P_tmp = np.zeros_like(matrix)
        for x, y in edges:
            P_tmp[x, y] = 1
        P.append((scalar, P_tmp))
        matrix = matrix - P_tmp * scalar
    return P


def PS(agents, objects, preferences):
    """
    Probabilistic Serial (PS) Algorithm. The PS rule takes as input the strict
    ordinal preferences of agents over items as well as the available amounts of each
    of the items. Agents start eating their most preferred item at unit speed until
    the item is consumed. They continue eating their most preferred items until
    all the items are consumed. The outcome is a random allocation in which each
    agent’s probability of getting an item is the fraction of the item that she ate.

    :param agents: vector of agents to allocate object to them :param objects: vector of objects to allocate to agent
    :param preferences: ordinal preferences of agents to object represent as matrix of element that can be
    int = object or set with object that agent do really preference between them . list of agents for each list of
    objects in the order he wants
    :param objects:
    :return: p the random assignment
    """
    num_agents = len(agents)
    num_objects = len(objects)

    stage = 0
    O = [objects]
    T = [[0] * num_objects]
    Tmin = [0]
    P = np.zeros((num_agents, num_objects))
    preferences_tmp = copy.deepcopy(preferences)

    # run until no objects remain
    while any(O[stage]):
        # read to be eaten object
        MaxN = set()
        for agent_pref in preferences_tmp:
            while len(agent_pref) > 0 and agent_pref[0] not in O[stage]:
                agent_pref.pop(0)
            if len(agent_pref) > 0:
                MaxN.add(agent_pref[0])

        # how much each object is been eaten by one agent. the min is the one that most people eat so end first
        # or how much time take to eat the object
        T.append([0] * num_objects)
        for o in MaxN:
            T[stage + 1][o] = ((1 - np.sum(P[:, o])) /
                               sum([1 for pref in preferences_tmp if len(pref) > 0 and pref[0] == o]))
        Tmin.append(min([o for i, o in enumerate(T[stage + 1]) if i in MaxN]))

        # Update P of the objects been eaten
        for a in agents:
            for o in objects:
                if o in MaxN and len(preferences_tmp[a]) > 0 and preferences_tmp[a][0] == o:
                    P[a][o] = P[a][o] + Tmin[stage + 1]
        # Remove the eaten objects
        O.append([o for o in O[stage] if o not in [x for x in MaxN if T[stage + 1][x] == Tmin[stage + 1]]])
        stage += 1
    return P


def cut_capacity(G, S):
    capacity = 0
    for u, v in G.edges():
        if u in S and v not in S:
            capacity += G[u][v]['capacity']
    return capacity


def EPS(agents, objects, preferences):
    n = len(agents)
    k = 0
    m = len(objects)
    P = np.zeros((n, m))

    G = nx.DiGraph()
    S = 0
    T = m + n + 1
    G.add_node(S)
    G.add_node(T)
    agents = [x + 1 for x in agents]
    objects = [x + n + 1 for x in objects]
    G.add_nodes_from(agents)
    G.add_nodes_from(objects)
    A = copy.deepcopy(objects)
    preferences_tmp = copy.deepcopy(preferences)
    preferences_tmp = [
        [set([pref]) if not isinstance(pref, set) else pref for pref in agent_pref]
        if isinstance(agent_pref, list) else agent_pref for agent_pref in preferences_tmp]
    C = [[0] * n]
    H = [preferences_tmp[x - 1][0] for x in agents if len(preferences_tmp[x - 1]) > 0]

    G.add_edges_from([(u, T, {'capacity': 1}) for u in A])
    G.add_edges_from([(u, v, {'capacity': np.Infinity}) for u in agents for v in A if (v - n - 1) in H[u - 1]])

    L_original = 1e-10
    G.add_edges_from([(S, u, {'capacity': C[k][u - 1] + L_original}) for u in agents])

    while A:

        L = L_original  # (nx.cut_size(G, {T}, range(0, 10), weight='capacity')) / len(agents)-0.1
        for u in G.successors(S):
            G[S][u]['capacity'] = C[k][u - 1] + L
        value, cut = nx.minimum_cut(G, S, T)
        low = L
        high = 1 - 1e-10
        while low <= high - 1e-10:
            mid = (low + high) / 2
            G_TMP = G
            for u in G_TMP.successors(S):
                G_TMP[S][u]['capacity'] = C[k][u - 1] + mid
            value, cut = nx.minimum_cut(G_TMP, S, T)
            if cut[0] == {0}:
                low = mid
                if low <= high - 1e-10:
                    low = high
                    break
            else:
                high = mid
                if low <= high - 1e-10:
                    low = mid
                    break

        L = low
        for u in G.successors(0):
            G[0][u]['capacity'] = C[k][u - 1] + L
        flow, flow_dict = nx.maximum_flow(G, S, T)
        for x in agents:
            if x not in cut[0]:
                continue
            for u in flow_dict[x]:
                if u not in cut[0]:
                    continue
                P[x - 1][u - n - 1] = flow_dict[x][u]

        # Remove all edges that start with nodes in agents
        for agent in agents:
            edges_to_remove = [(agent, v) for u, v in G.edges if u == agent]
            G.remove_edges_from(edges_to_remove)

        in_cut = [x for x in agents + A if x in cut[0]]

        # remove eaten object
        objects_to_remove = {x - n - 1 for x in cut[0]}
        for agent_pref in preferences_tmp:
            for pref in agent_pref:
                # if isinstance(pref, set):
                pref -= objects_to_remove
        preferences_tmp = [[pref for pref in agent_pref if pref] for agent_pref in preferences_tmp]

        H = [preferences_tmp[x - 1][0] if len(preferences_tmp[x - 1]) > 0 else [] for x in agents]
        G.add_edges_from(
            [(u, v, {'capacity': np.Infinity}) for u in agents for v in A if
             v - n - 1 in H[u - 1]])

        C.append([0 if x in in_cut else C[k][x - 1] + L for x in agents])
        G.remove_nodes_from([a for a in agents if preferences_tmp[a - 1] == []])
        G.remove_nodes_from([o for o in A if o in in_cut])
        A = [x for x in A if x not in in_cut]

        k += 1

    # pos = nx.circular_layout(G)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'capacity'))
    # nx.draw(G, pos, **draw_options)
    # plt.show()
    return P


def break_lexicographically(arr_with_sets):
    flattened = []
    for item in arr_with_sets:
        if isinstance(item, set):
            flattened.extend(sorted(list(item)))
        else:
            flattened.append(item)
    return flattened


def PS_Lottery(agents, objects, preferences, use_EPS=False):
    """
    Algorithm that utilize the PS algorithm and Bikroft algorithm to make Simultaneously Achieving Ex-ante and
    Ex-post Fairness with no need for number of agents and object to be equal. and if using ehr EPS algorithm,
    it will achieve that even with not strictly prefrence
    :param agents: get number of agents
    :param objects: get number of object
    :param preferences: for each agent need order list corresponding to the most preferred objects (index) to the least,
    it can have set of index if he preferred them the same
    :param use_EPS: difficult false, if true run EPS, if false handel ties with lexicographically order.
    :return: list of deterministic allocation matrix and scalar that represent the chance for this allocation
    """
    # assignment
    n = len(agents)
    m = len(objects)
    c = math.ceil(m / n)
    # 1 make dummy object
    dummy = list(range(n + 1, n + n * c - m + 1))
    # 2 add the dummy to the real
    objects_dummy = objects + dummy
    # 4 new preference
    preferences_dummy = [pref + dummy for pref in preferences]
    # 5 run (E)PS then split to presenters
    if use_EPS:
        P = EPS(agents, objects_dummy, preferences_dummy)
    else:
        preferences_dummy = [break_lexicographically(pref) for pref in preferences_dummy]
        P = PS(agents, objects_dummy, preferences_dummy)
    preferences_dummy = [break_lexicographically(pref) for pref in preferences_dummy]
    # split to presenters
    extP = np.zeros((n * c, n * c))
    for i, pref in enumerate(preferences_dummy):
        ate = 0
        agent = i
        for o in pref:
            if ate + P[i][o] <= 1:
                extP[agent][o] = P[i][o]
                ate += P[i][o]
            else:
                extP[agent][o] = 1 - ate
                agent = agent + n
                extP[agent][o] = P[i][o] - (1 - ate)
                ate = P[i][o] - (1 - ate)
    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    result = []
    for item in bikroft(extP):
        remove_dummy = np.delete(item[1], dummy, axis=1)
        stack_agents = np.array(np.sum([remove_dummy[row::n] for row in range(n)], axis=1))
        result.append((item[0], stack_agents))
    return result


def print_array(array):
    for row in array:
        print("    " + " ".join(str(int(cell)) for cell in row))


def print_PS_Lottery(result):
    print("=" * 40)
    for item in result:
        print(f"Probability: {item[0]}")
        print("Matrix:")
        print_array(item[1])
        print()


def measure_time(func, *args, **kwargs):
    """
    Measure the time taken to run function.
    Taken from the lecture
    :param func: the function to run
    :param args: the arguments to pass to the function
    :param kwargs: the keyword arguments to pass to the function
    :return: the time taken to run the function
    """
    start = perf_counter()
    func(*args, **kwargs)
    end = perf_counter()
    return end - start


def rnd_pref(agents, objects):
    objects = list(range(objects))
    prefrence = []
    for _ in range(agents):
        np.random.shuffle(objects)
        prefrence.append(objects.copy())
    return prefrence


def compare_solution_methods():
    """
    Compare the time taken to solve linear equations with the root function and with np.linalg.solve.
    we take samples is from 1 to 1000 with jumps of 10, each run 10 time and compute the average.
    after getting the times plot them together in one graph.

    """
    sizes = range(7, 15, 1)
    iteration = 1
    PS_times = []
    EPS_times = []
    for i, size in enumerate(sizes):
        print("*" * 20 + str(i) + "*" * 20)
        PS_times_for_average = []
        EPS_times_for_average = []
        for _ in range(iteration):
            objects = np.random.randint(size - 5, size + 5)
            prefrence = rnd_pref(size, objects)
            PS_times_for_average.append(measure_time(PS_Lottery, list(range(size)), list(range(objects)), prefrence,use_EPS=False))
            EPS_times_for_average.append(measure_time(PS_Lottery, list(range(size)), list(range(objects)), prefrence,use_EPS=True))

        PS_times.append(np.average(PS_times_for_average))
        EPS_times.append(np.average(EPS_times_for_average))
    plt.plot(sizes, PS_times, 'g-', label='PS')
    plt.plot(sizes, EPS_times, 'b-', label='EPS')
    plt.xlabel('Size')
    plt.ylabel('time in sec')
    plt.legend()
    plt.savefig("compere.png")  # after you plot the graphs, save them to a file and upload it separately.
    plt.show()  # this should show the plot on your screen


if __name__ == '__main__':
    compare_solution_methods()
