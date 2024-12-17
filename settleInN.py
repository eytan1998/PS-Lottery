import concurrent
import copy
import math
import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.algorithms.matching import is_perfect_matching
from scipy.optimize import linear_sum_assignment

import testGraphs
import withEnvy
from testGraphs import find_cycle_and_adjust


def bikroft_graph(matrix: np.array) -> list:
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
    if (
            not np.allclose(row_sums, 1)
            or not np.allclose(col_sums, 1)
            or not np.all(matrix >= 0)
    ):
        print(matrix)
        raise ValueError("must be a bistochastic matrix")

        # Create a bipartite graph
    B = nx.Graph()

    # Assume len(preferences) is the number of nodes in one partition (agents),
    # and len(alloc[0]) is the number of nodes in the other partition (items).
    n = matrix.shape[0]
    agents = range(n)
    items = range(n, n * 2)

    # Add nodes
    B.add_nodes_from(agents, bipartite=0)
    B.add_nodes_from(items, bipartite=1)

    P = []
    M0 = np.zeros((matrix.shape[0], matrix.shape[1]))



    while not np.allclose(matrix, M0):
        B.clear_edges()
        # Add edges according to preferences and current allocation
        for agent in range(n):
            for item in range(n):
                if matrix[agent][item] > 0:
                    B.add_edge(agent, item + n)
        graph_matching = nx.bipartite.maximum_matching(B, top_nodes=agents)


        # Find the edges corresponding to the matched pairs
        edges = [(x[0],x[1]-n) for x in graph_matching.items()][:n]

        scalar = min(matrix[x][y] for x, y in edges)

        P_tmp = np.zeros_like(matrix)
        for x, y in edges:
            P_tmp[x, y] = 1
        P.append((scalar, P_tmp))
        matrix = matrix - P_tmp * scalar
    return P
def bikroft(matrix: np.array) -> list:
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
    if (
            not np.allclose(row_sums, 1)
            or not np.allclose(col_sums, 1)
            or not np.all(matrix >= 0)
    ):
        print(matrix)
        raise ValueError("must be a bistochastic matrix")

    P = []
    M0 = np.zeros((matrix.shape[0], matrix.shape[1]))

    while not np.allclose(matrix, M0):
        # Replace zeros with a large value
        modified_matrix = np.where(matrix == 0, np.inf, matrix)
        # Apply linear_sum_assignment to find the optimal assignment
        row_indices, col_indices = linear_sum_assignment(modified_matrix)
        # Find the edges corresponding to the matched pairs
        edges = [
            (row, col)
            for row, col in zip(row_indices, col_indices)
            if matrix[row, col] != 0
        ]

        scalar = min(matrix[x][y] for x, y in edges)

        P_tmp = np.zeros_like(matrix)
        for x, y in edges:
            P_tmp[x, y] = 1
        P.append((scalar, P_tmp))
        matrix = matrix - P_tmp * scalar
    return P


def round_robin_envy(agents, objects, preferences, alloc, use_optimizer=False):
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
    # n = m
    m = len(objects)
    if n != m:
        raise ValueError("not NxN")

    # 3) allocation
    alloc_tmp = copy.deepcopy(alloc)

    # Iterate over each column
    for col in range(n):

        # Count the number of zeros in the column
        num_zeros = np.sum(alloc_tmp[:, col] == 0)

        # Avoid division by zero if there are no zeros in the column
        if num_zeros > 0:
            # Replace zeros with 1/(number of zeros in the column)
            alloc_tmp[:, col] = np.where(alloc_tmp[:, col] == 0, 1 / num_zeros, alloc_tmp[:, col])
    P = alloc_tmp - alloc
    if use_optimizer:
        find_cycle_and_adjust(P, preferences)
    # split to presenters

    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    result = []
    for item in bikroft(P):
        remove_dummy = np.delete(item[1], [], axis=1)
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))

    return result, P


def matching_one_by_one(agents, objects, preferences, alloc):
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
    # n = m
    m = len(objects)
    if n != m:
        raise ValueError("not NxN")

    # 3) allocation
    alloc_tmp = copy.deepcopy(alloc)

    num_agents = len(agents)
    num_objects = len(objects)

    P = np.zeros((num_agents, num_objects))
    preferences_tmp = copy.deepcopy(preferences)
    for agent in range(n):
        what_i_ate = [item for item in preferences[agent] if alloc_tmp[agent][item] == 1]
        preferences_tmp[agent] = [item for item in preferences_tmp[agent] if item not in what_i_ate]

    # Create a bipartite graph
    B = nx.Graph()

    # Assume len(preferences) is the number of nodes in one partition (agents),
    # and len(alloc[0]) is the number of nodes in the other partition (items).
    n = len(preferences)
    agents = range(n)
    items = range(n, n * 2)

    # Add nodes
    B.add_nodes_from(agents, bipartite=0)
    B.add_nodes_from(items, bipartite=1)

    # Add edges according to preferences and current allocation
    is_it_perfect_matching = False

    while not is_it_perfect_matching:
        for agent, pref in enumerate(preferences_tmp):
            B.add_edge(agent, pref[0] + n)
            pref.pop(0)
            # testGraphs.plot_graph_bipartite(B, n)

            # Use NetworkX to find a matching
            matching_graph = nx.bipartite.maximum_matching(B, top_nodes=agents)
            is_it_perfect_matching = nx.is_perfect_matching(B, matching_graph)
            if is_it_perfect_matching:
                continue

    for a, i in matching_graph.items():
        if a >= n:
            break
        P[a][i - n] = 1

    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    result = []
    for item in bikroft(P):
        remove_dummy = np.delete(item[1], [], axis=1)
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))

    return result, P


def matching_all(agents, objects, preferences, alloc):
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
    # n = m
    m = len(objects)
    if n != m:
        raise ValueError("not NxN")

    # 3) allocation
    alloc_tmp = copy.deepcopy(alloc)

    num_agents = len(agents)
    num_objects = len(objects)

    P = np.zeros((num_agents, num_objects))
    preferences_tmp = copy.deepcopy(preferences)
    for agent in range(n):
        what_i_ate = [item for item in preferences[agent] if alloc_tmp[agent][item] == 1]
        preferences_tmp[agent] = [item for item in preferences_tmp[agent] if item not in what_i_ate]

    # Create a bipartite graph
    B = nx.Graph()

    # Assume len(preferences) is the number of nodes in one partition (agents),
    # and len(alloc[0]) is the number of nodes in the other partition (items).
    n = len(preferences)
    agents = range(n)
    items = range(n, n * 2)

    # Add nodes
    B.add_nodes_from(agents, bipartite=0)
    B.add_nodes_from(items, bipartite=1)

    # Add edges according to preferences and current allocation
    is_it_perfect_matching = False

    while not is_it_perfect_matching:
        for agent, pref in enumerate(preferences_tmp):
            B.add_edge(agent, pref[0] + n)
            pref.pop(0)
        # testGraphs.plot_graph_bipartite(B, n)

        # Use NetworkX to find a matching
        matching = nx.bipartite.maximum_matching(B, top_nodes=agents)

        is_it_perfect_matching = nx.is_perfect_matching(B, matching)

    for a, i in matching.items():
        if a >= n:
            break
        P[a][i - n] = 1

    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    result = []
    for item in bikroft(P):
        remove_dummy = np.delete(item[1], [], axis=1)
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))

    return result, P


def matching(agents, objects, preferences, alloc):
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
    # n = m
    m = len(objects)
    if n != m:
        raise ValueError("not NxN")

    # 3) allocation
    alloc_tmp = copy.deepcopy(alloc)

    num_agents = len(agents)
    num_objects = len(objects)

    P = np.zeros((num_agents, num_objects))
    preferences_tmp = copy.deepcopy(preferences)
    for agent in range(n):
        what_i_ate = [item for item in preferences[agent] if alloc_tmp[agent][item] == 1]
        preferences_tmp[agent] = [item for item in preferences_tmp[agent] if item not in what_i_ate]

    # Create a bipartite graph
    B = nx.Graph()

    # Assume len(preferences) is the number of nodes in one partition (agents),
    # and len(alloc[0]) is the number of nodes in the other partition (items).
    n = len(preferences)
    agents = range(n)
    items = range(n, n * 2)

    # Add nodes
    B.add_nodes_from(agents, bipartite=0)
    B.add_nodes_from(items, bipartite=1)

    # Add edges according to preferences and current allocation
    is_it_perfect_matching = False
    # get the envy level the most envy number + and the index of him
    envy_level = sort_envy_levels(getEnvyLevel(alloc_tmp, preferences))

    # add in this order:
    # 1. what has - stage
    # 2. preference order
    while not is_it_perfect_matching:
        for agent_set in envy_level:
            for agent in agent_set:
                 pref = preferences_tmp[agent]
                 item = pref.pop(0)
                 B.add_edge(agent, item + n)

            # Use NetworkX to find a matching
            graph_matching = nx.bipartite.maximum_matching(B, top_nodes=agents)
            is_it_perfect_matching = nx.is_perfect_matching(B, graph_matching)
            if is_it_perfect_matching:
                break

    for a, i in graph_matching.items():
        if a >= n:
            break
        P[a][i - n] = 1

    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    result = []
    for item in bikroft(P):
        remove_dummy = np.delete(item[1], [], axis=1)
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))

    return result, P

def getEnvyLevel(mat, pref ):
    envy_level = [(0, np.inf) for _ in range(len(mat[0]))]
    for me in range(len(mat)):
        for other in range(len(mat)):
            if me == other:
                continue
            my_sum = is_sum = 0
            for i, item in enumerate(pref[me]):
                my_sum += mat[me][item]
                is_sum += mat[other][item]
                if is_sum > my_sum:
                    if envy_level[me][0] > is_sum - my_sum:
                        continue
                    if envy_level[me][0] < is_sum - my_sum  :
                        envy_level[me] = (is_sum - my_sum, i)
                    elif envy_level[me][1] > i:
                        envy_level[me] = (is_sum - my_sum, i)
    return envy_level

def sort_envy_levels(envy_level):
    # Sort primarily by descending envy and secondarily by ascending index
    envy_level_and_indexes = [(index,(envy,ind)) for index,(envy,ind) in enumerate(envy_level)]
    sorted_envy_level = sorted(envy_level_and_indexes, key=lambda x: (-x[1][0], x[1][1]))

    envy_level_index = []
    last_envy = (-1,-1)
    for (index,(envy,ind)) in sorted_envy_level:
        if last_envy == (envy,ind):
            last_set = envy_level_index.pop()
            last_set.add(index)
            envy_level_index.append(last_set)
        else:
            envy_level_index.append({index})
            last_envy = (envy,ind)

    return envy_level_index

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
            T[stage + 1][o] = (1 - np.sum(P[:, o])) / sum(
                [1 for pref in preferences_tmp if len(pref) > 0 and pref[0] == o]
            )
        Tmin.append(min([o for i, o in enumerate(T[stage + 1]) if i in MaxN]))

        # Update P of the objects been eaten
        for a in agents:
            for o in objects:
                if (
                    o in MaxN
                    and len(preferences_tmp[a]) > 0
                    and preferences_tmp[a][0] == o
                ):
                    P[a][o] = P[a][o] + Tmin[stage + 1]
        # Remove the eaten objects
        O.append(
            [
                o
                for o in O[stage]
                if o not in [x for x in MaxN if T[stage + 1][x] == Tmin[stage + 1]]
            ]
        )
        stage += 1
    return P

def break_lexicographically(arr_with_sets):
    flattened = []
    for item in arr_with_sets:
        if isinstance(item, set):
            flattened.extend(sorted(list(item)))
        else:
            flattened.append(item)
    return flattened

def update_preferences(original_pref, allocaion, day, leap=1):

    original_m = len(original_pref)
    preferences_days = []
    for pref in original_pref:
        new_pref = []
        for index,p in enumerate(pref):
                new_pref.append(p)
                for d in range(1, day):
                    new_pref.append(p + (original_m * d))
        preferences_days.append(new_pref)


    for item in range(original_m):
        for ag in range(original_m):
            items_list_remain = list(range(item, (day-1) * original_m, original_m))
            for _ in range(int(day-1-allocaion[ag][item])):
                preferences_days[ag].remove(items_list_remain.pop())
    return preferences_days

def PS_lottery_extended(agents, objects, preferences, alloc, day):
    """
    Algorithm that utilize the PS algorithm and Bikroft algorithm to make Simultaneously Achieving Ex-ante and
    Ex-post Fairness with no need for number of agents and object to be equal. and if using ehr EPS algorithm,
    it will achieve that even with not strictly prefrence
    :param alloc:
    :param day:
    :param agents: get number of agents
    :param objects: get number of object
    :param preferences: for each agent need order list corresponding to the most preferred objects (index) to the least,
    it can have set of index if he preferred them the same
    :param use_EPS: difficult false, if true run EPS, if false handel ties with lexicographically order.
    :return: list of deterministic allocation matrix and scalar that represent the chance for this allocation
    """
    # assignment
    n = len(agents)
    original_m = len(objects)

    object_days = list(range(original_m * day))
    preferences_days = []
    for pref in preferences:
        new_pref = []
        for p in pref:
            new_pref.append(p)
            for d in range(1,day):
                new_pref.append(p+(original_m * d))
        preferences_days.append(new_pref)
    m = len(object_days)
    c = math.ceil(m / n)
    # 1 make dummy object
    dummy = list(range(m, m + (n * c - m)))
    # 2 add the dummy to the real
    objects_dummy = object_days + dummy
    # 4 new preference
    preferences_dummy = preferences_days
    if day > 1:
        preferences_dummy = update_preferences(preferences,alloc,day)

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
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))

    # Collapse columns of P to original_m columns
    P_collapsed = np.zeros((P.shape[0], original_m))
    for col in range(original_m):
        # Sum original column and the respective offset columns
        P_collapsed[:, col] = P[:, col::original_m].sum(axis=1)

    R_collapsed = []
    for r in result:
        # if not from the allocation remove dont add
        newR = np.zeros((r[1].shape[0], original_m))
        for col in range(original_m):
            newR[:, col] = r[1][:, col::original_m].sum(axis=1)
        if not np.all(newR >= alloc): continue
        R_collapsed.append((r[0],newR))
    if not R_collapsed:
        print('KHASDK')
    return R_collapsed, P_collapsed
def PS_lottery_filter(agents, objects, preferences, alloc, day):
    """
    Algorithm that utilize the PS algorithm and Bikroft algorithm to make Simultaneously Achieving Ex-ante and
    Ex-post Fairness with no need for number of agents and object to be equal. and if using ehr EPS algorithm,
    it will achieve that even with not strictly prefrence
    :param alloc:
    :param day:
    :param agents: get number of agents
    :param objects: get number of object
    :param preferences: for each agent need order list corresponding to the most preferred objects (index) to the least,
    it can have set of index if he preferred them the same
    :param use_EPS: difficult false, if true run EPS, if false handel ties with lexicographically order.
    :return: list of deterministic allocation matrix and scalar that represent the chance for this allocation
    """
    # assignment
    n = len(agents)
    original_m = len(objects)

    object_days = list(range(original_m * day))
    preferences_days = []
    for pref in preferences:
        new_pref = []
        for p in pref:
            new_pref.append(p)
            for d in range(1,day):
                new_pref.append(p+(original_m * d))
        preferences_days.append(new_pref)
    m = len(object_days)
    c = math.ceil(m / n)
    # 1 make dummy object
    dummy = list(range(m, m + (n * c - m)))
    # 2 add the dummy to the real
    objects_dummy = object_days + dummy
    # 4 new preference
    preferences_dummy = [pref + dummy for pref in preferences_days]

    preferences_dummy = [
    break_lexicographically(pref) for pref in preferences_dummy]
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
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))

    # Collapse columns of P to original_m columns
    P_collapsed = np.zeros((P.shape[0], original_m))
    for col in range(original_m):
        # Sum original column and the respective offset columns
        P_collapsed[:, col] = P[:, col::original_m].sum(axis=1)

    R_collapsed = []
    for r in result:
        # if not from the allocation remove dont add
        newR = np.zeros((r[1].shape[0], original_m))
        for col in range(original_m):
            newR[:, col] = r[1][:, col::original_m].sum(axis=1)
        if not np.all(newR >= alloc): continue
        R_collapsed.append((r[0],newR))
    if not R_collapsed:
        print('KHASDK')
    return R_collapsed, P_collapsed

def ps_on_left(agents, objects, preferences, alloc):
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
    # n = m
    m = len(objects)
    if n != m:
        raise ValueError("not NxN")

    # 3) allocation
    alloc_tmp = copy.deepcopy(alloc)

    num_agents = len(agents)
    num_objects = len(objects)

    P = np.zeros((num_agents, num_objects))
    preferences_tmp = copy.deepcopy(preferences)
    for agent in range(n):
        what_i_ate = [item for item in preferences[agent] if alloc_tmp[agent][item] == 1]
        preferences_tmp[agent] = [item for item in preferences_tmp[agent] if item not in what_i_ate]
        if what_i_ate:
            preferences_tmp[agent].extend(what_i_ate)

    ate = [0 for _ in agents]
    left_to_eat = [1 for _ in objects]

    threshhold = 1

    while sum(left_to_eat) > 0:
        who_can_eat = [agents[i] for i in range(len(agents)) if ate[i] < threshhold]
        if who_can_eat == []:
            break

        # min of
        # how much left from the wanted item to eat / how much wanting to eat him
        # ???? the dist to next envy
        what_i_want_eat = [
            -1 if preferences_tmp[a] == [] else preferences_tmp[a][0] for a in agents
        ]
        until_end_of_item = [left_to_eat[o] /
                             (1 if len([x for x in who_can_eat if what_i_want_eat[x] == o]) == 0
                              else len([x for x in who_can_eat if what_i_want_eat[x] == o]))
                             for o in objects if left_to_eat[o] > 0] + [np.inf]
        how_much_to_eat = min([min(until_end_of_item)
                                  , min(threshhold - ate[a] for a in who_can_eat)])

        # update P
        for a in who_can_eat:
            P[a][what_i_want_eat[a]] += how_much_to_eat
            alloc_tmp[a][what_i_want_eat[a]] += how_much_to_eat

            # remove the eaten from who can eat, and the sum from the delay of who didnt eat
        for a in who_can_eat:
            ate[a] += how_much_to_eat

        for a in who_can_eat:
            left_to_eat[what_i_want_eat[a]] -= how_much_to_eat
        # TODO np.round
        left_to_eat = [np.round(a, 16) for a in left_to_eat]

        # update the next to eat(tmp_prefrence)
        for i in range(len(preferences_tmp)):
            preferences_tmp[i] = [x for x in preferences_tmp[i] if left_to_eat[x] > 0]

    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    result = []
    for item in bikroft(P):
        remove_dummy = np.delete(item[1], [], axis=1)
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))

    return result, P


def round_robin(agents, objects, preferences, alloc):
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
    # n = m
    m = len(objects)
    if n != m:
        raise ValueError("not NxN")

    # 3) allocation
    alloc_tmp = copy.deepcopy(alloc)

    # remove from preference what i already got
    preferences_tmp = copy.deepcopy(preferences)
    for agent in range(n):
        what_i_ate = [item for item in preferences[agent] if alloc_tmp[agent][item] == 1]
        preferences_tmp[agent] = [item for item in preferences_tmp[agent] if item not in what_i_ate]

    P = np.zeros((n, n))

    for agent in range(n):
        what_i_want_to_eat = preferences_tmp[agent][0]
        P[agent][what_i_want_to_eat] = 1
        for p in preferences_tmp:
            try:
                p.remove(what_i_want_to_eat)
            except:
                continue

    # Iterate over each agen (round robin)

    # split to presenters

    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    result = []
    for item in bikroft(P):
        remove_dummy = np.delete(item[1], [], axis=1)
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))

    return result, P


def same_luck(agents, objects, preferences, alloc, use_optimizer=False):
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
    # n = m
    m = len(objects)
    if n != m:
        raise ValueError("not NxN")

    # 3) allocation
    alloc_tmp = copy.deepcopy(alloc)

    # Iterate over each column
    for col in range(n):

        # Count the number of zeros in the column
        num_zeros = np.sum(alloc_tmp[:, col] == 0)

        # Avoid division by zero if there are no zeros in the column
        if num_zeros > 0:
            # Replace zeros with 1/(number of zeros in the column)
            alloc_tmp[:, col] = np.where(alloc_tmp[:, col] == 0, 1 / num_zeros, alloc_tmp[:, col])
    P = alloc_tmp - alloc
    if use_optimizer:
        find_cycle_and_adjust(P, preferences)
    # split to presenters

    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    result = []
    bik = bikroft(P)
    for item in bik:
        remove_dummy = np.delete(item[1], [], axis=1)
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))

    return result, P


def print_PS_Lottery(result, r, c, prefe, alloc):
    print("================================================")
    index = 0
    for item in result:
        newMat = alloc + item[1]
        print(f"Probability: {item[0]}")
        print(f"Matrix No {index}:")
        print_mat(newMat, r, c)
        print(f"EF? {whatEF(newMat, prefe)}")
        index += 1


def print_mat(matrix, row, col):
    row_names = list(row)
    column_names = list(col)

    # Ensure the length of row_names and column_names match the dimensions of the matrix
    assert (
            len(row_names) == matrix.shape[0]
    ), "Row names length must match number of rows in the matrix."
    assert (
            len(column_names) == matrix.shape[1]
    ), "Column names length must match number of columns in the matrix."

    # Print table header
    header = " | ".join(f"{col:10}" for col in column_names)
    print(f"{'':10} | {header}")
    print("-" * (len(header) + 12))  # Adjust based on column width

    # Print table rows
    for row_name, row_data in zip(row_names, matrix):
        row_str = "\t".join(f"{item:10}" for item in row_data)
        print(f"{row_name:10} | {row_str}")


def compute_envy_for_me(me, mat, pref) -> int:
    def compute_for_other(other):
        nonlocal max_envy
        if me == other:
            return 0
        my_sum = 0
        is_sum = 0
        local_max_envy = 0
        for item in pref[me]:
            my_sum += mat[me][item]
            is_sum += mat[other][item]
            if is_sum > my_sum and local_max_envy < is_sum - my_sum:
                local_max_envy = is_sum - my_sum
        return local_max_envy

    max_envy = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(compute_for_other, range(len(mat)))

    # Collect the maximum envy value from all threads
    for result in results:
        if result > max_envy:
            max_envy = result

    return max_envy


def whatEF(mat, pref) -> int:
    max_envy = 0
    n = len(mat)

    # Create a pool of processes
    with multiprocessing.Pool() as pool:
        # Use starmap to pass multiple arguments to each task
        results = pool.starmap(compute_envy_for_me, [(me, mat, pref) for me in range(n)])

    # Find the maximum envy from the results
    max_envy = max(results)

    return max_envy


def generate_random_preferences(agents, items):
    preferences = []
    for agent in agents:
        # Shuffle items to create a random preference list for each agent
        shuffled_items = random.sample(items, len(items))
        preferences.append(shuffled_items)
    return preferences


# def sol_with_optimize(num, agent, item, original_pref):
#     max_envy = -1
#     allocation = np.zeros((len(agent), len(item)))
#     for _ in range(num // 2):
#
#         result = same_luck(
#             agents=agent, objects=item, preferences=original_pref, alloc=allocation, use_optimizer=True)
#         tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
#         if tmp_max_envy > max_envy:
#             max_envy = tmp_max_envy
#
#         new_results = result[0]
#         user_input = random.randint(0, len(new_results) - 1)
#         mat = new_results[int(user_input)][1]
#         allocation += mat
#     return max_envy
#

def sol_matching_one_by_one(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    for _ in range(num // 2):
        result = matching_one_by_one(
            agents=agent, objects=item, preferences=original_pref, alloc=allocation)
        with ThreadPoolExecutor() as executor:
            # Use list comprehension to submit tasks for each r in result[0]
            futures = [executor.submit(whatEF, allocation + r[1], original_pref) for r in result[0]]

            # Get results and compute the maximum
            tmp_max_envy = max(future.result() for future in futures)
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation += mat
    return max_envy


def sol_matching_all(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    for _ in range(num // 2):
        result = matching_all(
            agents=agent, objects=item, preferences=original_pref, alloc=allocation)
        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation += mat
    return max_envy


def sol_matching(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    for _ in range(num // 2):
        result = matching(
            agents=agent, objects=item, preferences=original_pref, alloc=allocation)
        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation += mat
    return max_envy


def sol_same_luck(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    for _ in range(num // 2):
        result = same_luck(
            agents=agent, objects=item, preferences=original_pref, alloc=allocation)
        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0][:2]])
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation += mat
    return max_envy

def sol_ps_filter(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    original_pref = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    for day in range(1,num * 2):

        result = PS_lottery_filter(agents=agent, objects=item, preferences=original_pref, alloc=allocation,day=day)

        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation = mat
    return max_envy

def sol_ps_extended(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    original_pref = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    for day in range(1,num * 2):

        result = PS_lottery_extended(agents=agent, objects=item, preferences=original_pref, alloc=allocation,day=day)

        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation = mat
    return max_envy

def sol_ps_on_left(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    for _ in range(num * 2):
        try:
            result = ps_on_left(
                agents=agent, objects=item, preferences=original_pref, alloc=allocation)
        except:
            continue
        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation += mat
    return max_envy


def sol_round_robin(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    for _ in range(num * 2):
        result = round_robin(
            agents=agent, objects=item, preferences=original_pref, alloc=allocation)
        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation += mat
    return max_envy


def compare_solution_methods():
    nums = range(3,304,40  )
    iteration = 1
    first = []
    second = []

    for i, num in enumerate(nums):
        print(num)
        average_matching = []
        average_same_luck = []

        agent = list(range(num))
        item = list(range(num))

        for _ in range(iteration):
            original_pref = generate_random_preferences(agents=agent, items=item)
            # average_same_luck.append(sol_same_luck(num, agent, item, original_pref)/num)
            average_matching.append(sol_matching(num, agent, item, original_pref)/num)
        # first.append(np.average(average_same_luck))
        # print(f'{num}: {np.average(average_same_luck)}')
        second.append(np.average(average_matching))
        print(f'{num}: {np.average(average_matching)}')
    #     print(f'{num}:{first}\n{average_same_luck}')
    # print(list(first))
    # pre_calculated_average_same_luck = ([1.0, 2.0, 2.0, 2.7, 2.6, 3.6, 3.6, 3.7, 4.1, 4.4, 4.7, 5.0, 5.1, 5.7, 5.1, 5.9, 6.1]# 3-19
    #                                     +[6.3, 6.3, 6.3, 6.9, 6.9, 7.4, 7.3, 7.5, 7.8, 8.0] # 20 -29
    #                                     +[7.6 +8.2+8.4+ 8.6 + 9+ 8.5 + 8.9+ 8.9+ 8.9 + 9.5] # 30 -39
    #                                     +[9.4+10+10 +10+ 10.3 ])
    # plt.plot(nums, first, 'g-*')
    plt.plot(nums, second, 'r-*')
    plt.xlabel('number of agents and objects')
    plt.ylabel('max envy')
    plt.grid(True)
    # plt.axhline(y=.1, color='black', linestyle='--')  # Black dashed line at y=0
    plt.savefig("compare.png")
    # plt.legend(["This is my legend"], fontsize="x-large")
    plt.show()  # this should show the plot on your screen


if __name__ == "__main__":
    compare_solution_methods()
