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


def cut_capacity(G, S):
    capacity = 0
    for u, v in G.edges():
        if u in S and v not in S:
            capacity += G[u][v]["capacity"]
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
        if isinstance(agent_pref, list)
        else agent_pref
        for agent_pref in preferences_tmp
    ]
    C = [[0] * n]
    H = [preferences_tmp[x - 1][0] for x in agents if len(preferences_tmp[x - 1]) > 0]

    G.add_edges_from([(u, T, {"capacity": 1}) for u in A])
    G.add_edges_from(
        [
            (u, v, {"capacity": np.inf})
            for u in agents
            for v in A
            if (v - n - 1) in H[u - 1]
        ]
    )

    L_original = 1e-10
    G.add_edges_from([(S, u, {"capacity": C[k][u - 1] + L_original}) for u in agents])

    while A:
        L = L_original  # (nx.cut_size(G, {T}, range(0, 10), weight='capacity')) / len(agents)-0.1
        for u in G.successors(S):
            G[S][u]["capacity"] = C[k][u - 1] + L
        value, cut = nx.minimum_cut(G, S, T)
        low = L
        high = 1 - 1e-10
        while low <= high - 1e-10:
            mid = (low + high) / 2
            G_TMP = G
            for u in G_TMP.successors(S):
                G_TMP[S][u]["capacity"] = C[k][u - 1] + mid
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
            G[0][u]["capacity"] = C[k][u - 1] + L
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
        preferences_tmp = [
            [pref for pref in agent_pref if pref] for agent_pref in preferences_tmp
        ]

        H = [
            preferences_tmp[x - 1][0] if len(preferences_tmp[x - 1]) > 0 else []
            for x in agents
        ]
        G.add_edges_from(
            [
                (u, v, {"capacity": np.inf})
                for u in agents
                for v in A
                if v - n - 1 in H[u - 1]
            ]
        )

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
    dummy = list(range(m, m + (n * c - m)))
    # 2 add the dummy to the real
    objects_dummy = objects + dummy
    # 4 new preference
    preferences_dummy = [pref + dummy for pref in preferences]
    # 5 run (E)PS then split to presenters
    if use_EPS:
        P = EPS(agents, objects_dummy, preferences_dummy)
    else:
        preferences_dummy = [
            break_lexicographically(pref) for pref in preferences_dummy
        ]
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
    return result


def print_array(array):
    for row in array:
        print("    " + " ".join(str(int(cell)) for cell in row))


# def print_PS_Lottery(result):
#     print("=" * 40)
#     for item in result:
#         print(f"Probability: {item[0]}")
#         print("Matrix:")
#         print_array(item[1])
#         print()


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


def combine_columns(matrix, n):
    # Convert to a NumPy array if it's not already
    matrix = np.array(matrix)

    # Calculate the number of new columns
    num_rows, num_cols = matrix.shape
    new_cols_count =  n  # ceil(num_cols / n)

    # Initialize an empty array for the combined columns
    combined_matrix = np.zeros((num_rows, new_cols_count))

    # Combine columns
    for i in range(num_cols):
        col_index = i % n
        combined_matrix[:, col_index] += matrix[:, i]

    return combined_matrix

def print_mat(matrix, row, col):
    row_names = list(row)
    column_names = list(col)

    matrix = combine_columns(matrix=matrix,n= len(col))

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


def isEF(mat, pref) -> bool:
    for me in range(len(mat)):
        for other in range(len(mat)):
            if me == other:
                continue
            my_sum = 0
            is_sum = 0
            for item in pref[me]:
                my_sum += sum(
                    [
                        mat[me][theItem]
                        for theItem in range(item, len(mat[0]), len(pref) + 1)
                    ]
                )
                is_sum += sum(
                    [
                        mat[other][theItem]
                        for theItem in range(item, len(mat[0]), len(pref) + 1)
                    ]
                )
                if is_sum - 0.00001 > my_sum:
                    return False
    return True


def isEF1(mat, pref):
    for me in range(len(mat)):
        for other in range(len(mat)):
            if me == other:
                continue
            my_sum = 0
            is_sum = -1
            for item in pref[me]:
                my_sum += sum(
                    [
                        mat[me][theItem]
                        for theItem in range(item, len(mat[0]), len(pref) + 1)
                    ]
                )
                is_sum += sum(
                    [
                        mat[other][theItem]
                        for theItem in range(item, len(mat[0]), len(pref) + 1)
                    ]
                )
                if is_sum - 0.00000000001 > my_sum:
                    return (is_sum,my_sum)
    return True

def isEF2(mat, pref) -> bool:
    for me in range(len(mat)):
        for other in range(len(mat)):
            if me == other:
                continue
            my_sum = 0
            is_sum = -2
            for item in pref[me]:
                my_sum += sum(
                    [
                        mat[me][theItem]
                        for theItem in range(item, len(mat[0]), len(pref) + 1)
                    ]
                )
                is_sum += sum(
                    [
                        mat[other][theItem]
                        for theItem in range(item, len(mat[0]), len(pref) + 1)
                    ]
                )
                if is_sum - 0.00000000001 > my_sum:
                    print (f"{is_sum,my_sum}")
                    raise ValueError("ex-post ef2")
    return True

def print_PS_Lottery(result, r, c, prefe):
    print("=" * 40)
    index = 0
    for item in result:
        print(f"Probability: {item[0]}")
        print(f"Matrix No {index}:")
        print_mat(item[1], r, c)
        print(f"Is ex-post EF? {isEF(item[1],prefe)}")
        print(f'Is ex-post EF-1? {isEF1(item[1],prefe)}')
        print(f'Is ex-post EF-2? {isEF2(item[1],prefe)}')
       
        print()
        index+=1


def update(originalPref, allocaion, day, leap=1):
    modul = len(originalPref[0])
    newL = [[l[i] + (modul * day) for i in range(len(l))] for l in originalPref]
    newLL = []
    for i in range(len(newL)):
        leap = 0
        tmp = []
        first = newL[i][0]
        allocation_befor_first = [x for x in allocaion[i] if first % modul == x % modul]
        tmp += allocation_befor_first
        tmp += [first]
        tmp += [x for x in allocaion[i] if x not in allocation_befor_first]
        tmp += [x for x in newL[i] if x != first]
        newLL.append(tmp)
    return newLL

def generate_random_preferences(agents, items):
    preferences = []
    for agent in agents:
        # Shuffle items to create a random preference list for each agent
        shuffled_items = random.sample(items, len(items))
        preferences.append(shuffled_items)
    return preferences


if __name__ == "__main__":
    # compare_solution_methods()
    day = 0
    num = 3
    agent = list(range(num))
    item = list(range(num))
    originalPref = generate_random_preferences(agents=agent, items=item)
    # agent = [0, 1, 2]
    # item = [0, 1, 2, 3]
    # originalPref = [[0, 1, 2, 3], [0, 2, 1, 3], [3, 1, 0, 2]]
    alloction = [[] for _ in agent]
    while True:
        print(f'=======================================\n                   day {day}              \n===============================')
        thisDayItem = list(range((day+1)*len(item)))
        pred = update(originalPref=originalPref, allocaion=alloction, day=day)
        

        ps_mat = PS(agent, thisDayItem, pred)
        print_mat(ps_mat, row=agent, col=item)
        print(f"Is ex-ante EF? {isEF(ps_mat,pred)}")
        print(
            f'Is ex-ante EF-1? {True if isEF1(ps_mat,pred) else "=========================================="}'
        )

        result = PS_Lottery(agent, thisDayItem, pred)
        print_PS_Lottery(result=result, r=agent, c=item, prefe=originalPref)

        print("What you like to do? \nq) quit the program \n0-inf) enter the number of the matrix you want to allocate(default)")
        # user_input = input()
        user_input = 0

        if user_input == "q":
            break
        if int(user_input) >= len(result):
            break
        # update allocation
        # get last 4 item
        mat = result[int(user_input)][1]
        newAllc = []
        for ag in agent:
            newAgPref = []
            extendPref = [x + day * len(item) for x in originalPref[ag]]
            for it in extendPref:
                newAgPref += [x for x in alloction[ag] if x != it and x % len(item) == it % len(item)]
                if mat[ag][it] == 0:
                    continue
                # all the item the was prev to this
                newAgPref.append(it)
                # add the item to alloc in order
            alloction[ag] = newAgPref
        day += 1


# it = [0,1,2,3]
# pred = [[0,1,2,3], [0, 2,1,3], [3,1,0,2]]
# [[1],[0,2],[3]]
# =============
# it = [0,1,2,3,4,5,6,7]
# pred = [[4, 1, 5, 6, 7], [0, 4, 2, 6, 5, 7], [3, 7, 5, 4, 6]]
# [[4,1,6],[0,2],[3,7,5]]
# =============
# it = [0,1,2,3,4,5,6,7,8,9,10,11]
# pred = [[4, 8, 1, 6, 9, 10, 11], [0, 8, 2, 10, 9, 11], [3, 7, 11, 5, 9, 8, 10]]
# [[4,8,1,6],[0,9,2,10],[3,7,11,5]]
# =============
# it = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# pred = [[4, 8, 12, 1, 6, 13, 14, 15], [0, 12, 9, 2, 10, 14, 13, 15], [3, 7, 11, 15, 5, 13, 12, 14]]
# [[4,8,1,13,6],[0,12,9,2,10],[3,7,11,15,5,14]]
# =============
