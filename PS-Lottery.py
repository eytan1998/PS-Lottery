import copy
import math

import numpy
import numpy as np
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

    :param agents: vector of agents to allocate object to them
    :param objects: vector of objects to allocate to agent
    :param preferences: ordinal preferences of agents to object
    represent as list of list. list of agents for each list of objects in the order he want
    :return: p the random assignment
    """
    num_agents = len(agents)
    num_objects = len(objects)

    stage = 0
    O = [objects]
    T = [[0] * num_objects]
    Tmin = [0]
    P = [np.zeros((num_agents, num_objects))]
    preferences_tmp = copy.deepcopy(preferences)

    while O[stage]:
        MaxN = set()
        for agent_pref in preferences_tmp:
            while len(agent_pref) > 0 and agent_pref[0] not in O[stage]:
                agent_pref.pop(0)
            if len(agent_pref) > 0:
                MaxN.add(agent_pref[0])

        T.append([0] * num_objects)

        for o in MaxN:
            T[stage + 1][o] = ((1 - np.sum(P[stage][:, o])) /
                               sum([1 for pref in preferences_tmp if len(pref) > 0 and pref[0] == o]))
        Tmin.append(min([value for value in T[stage + 1] if value > 0]))

        P.append(np.copy(P[stage]))
        for a in agents:
            for o in objects:
                if o in MaxN and len(preferences_tmp[a]) > 0 and preferences_tmp[a][0] == o:
                    P[stage + 1][a][o] = P[stage][a][o] + Tmin[stage + 1]
        O.append([o for o in O[stage] if o not in [x for x in MaxN if T[stage + 1][x] == Tmin[stage + 1]]])
        stage += 1

    return P[stage]


def EPS():
    pass


def PS_Lottery(agents, objects, preferences):
    # assignment
    n = len(agents)
    m = len(objects)
    c = math.ceil(m / n)
    # 1 make dummy object
    dummy = list(range(n + 1, n + n * c - m + 1))
    # 2 add the dummy to the real
    objectsD = objects + dummy
    # 4 new preference TODO add support to not strictly order
    preferencesDc = [pref + dummy for pref in preferences]
    # 5 instead of run PS fore each presenter of the agent
    # i run like they can eat more than one and then split to presenter
    # TODO add possibility to run EPS
    P = PS(agents, objectsD, preferencesDc)
    extP = np.zeros((n * c, n * c))
    for i, pref in enumerate(preferencesDc):
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
        stack_agents = np.array(np.sum([remove_dummy[row::n] for row in range(n)],axis=1))
        result.append((item[0], stack_agents))
    return result

if __name__ == '__main__':

    print(PS_Lottery(list(range(2)), list(range(4)), [[0, 1, 2, 3],[0, 2, 1, 3]]))
