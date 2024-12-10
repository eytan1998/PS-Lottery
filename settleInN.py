import copy
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
    for item in bikroft(P):
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
    max_envy = 0
    for other in range(len(mat)):
        if me == other:
            continue
        my_sum = 0
        is_sum = 0
        for item in pref[me]:
            my_sum += mat[me][item]
            is_sum += mat[other][item]
            if is_sum > my_sum and max_envy < is_sum - my_sum:
                max_envy = is_sum - my_sum
    return max_envy


def whatEF(mat, pref) -> int:
    max_envy = 0
    n = len(mat)

    # Use ThreadPoolExecutor to parallelize computation
    with ThreadPoolExecutor() as executor:
        # Submit a job for each `me` index
        futures = [executor.submit(compute_envy_for_me, me, mat, pref) for me in range(n)]

        # Get results and find the maximum envy
        for future in futures:
            max_envy = max(max_envy, future.result())

    return max_envy


def generate_random_preferences(agents, items):
    preferences = []
    for agent in agents:
        # Shuffle items to create a random preference list for each agent
        shuffled_items = random.sample(items, len(items))
        preferences.append(shuffled_items)
    return preferences


def sol_with_optimize(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    for _ in range(num // 2):

        result = same_luck(
            agents=agent, objects=item, preferences=original_pref, alloc=allocation, use_optimizer=True)
        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation += mat
    return max_envy


def sol_matching_one_by_one(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    for _ in range(num // 2):
        result = matching_one_by_one(
            agents=agent, objects=item, preferences=original_pref, alloc=allocation)
        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
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
        tmp_max_envy = max([whatEF(allocation + r[1], original_pref) for r in result[0]])
        if tmp_max_envy > max_envy:
            max_envy = tmp_max_envy

        new_results = result[0]
        user_input = random.randint(0, len(new_results) - 1)
        mat = new_results[int(user_input)][1]
        allocation += mat
    return max_envy


def sol_ps_on_left(num, agent, item, original_pref):
    max_envy = -1
    allocation = np.zeros((len(agent), len(item)))
    for _ in range(num * 2):
        try:
            result = withEnvy.Envy_Lottery(
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
    nums = range(3, 30 )
    iteration = 10
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

            average_same_luck.append(sol_matching_all(num, agent, item, original_pref))
            average_matching.append(sol_matching(num, agent, item, original_pref))
        first.append(max(average_same_luck))
        second.append(max(average_matching))

    plt.plot(nums, first, 'g-*')
    plt.plot(nums, second, 'r-*')
    plt.xlabel('number of agents and objects')
    plt.ylabel('max envy')
    plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='--')  # Black dashed line at y=0
    plt.savefig("compare.png")
    # plt.legend(["This is my legend"], fontsize="x-large")
    plt.show()  # this should show the plot on your screen


if __name__ == "__main__":
    compare_solution_methods()
