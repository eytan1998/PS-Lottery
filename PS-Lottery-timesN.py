import copy
import math
import random

import numpy as np
from scipy.optimize import linear_sum_assignment


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


def MPS_Lottery(agents, objects, preferences, alloc):
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
    c = 1
    # 1 make dummy object
    dummy = list(range(m, m + (n * c - m)))
    dummy = []
    # 2 add the dummy to the real
    objects_dummy = objects + dummy
    # 4 new preference
    preferences_dummy = [pref + dummy for pref in preferences]
    # 5 run Eats
    P = np.zeros((n, len(objects_dummy)))
    P = np.zeros((n, n))

    # for each agent i want to know

    # 1) ate
    ate = [0 for _ in agents]
    # 2) marked [[0,1],[2,3],[4,1]]
    marked = [
        True if sum(al) < max(sum(al) for al in alloc) else False for al in alloc
    ]
    # 3) alloction
    alloc_tmp = copy.deepcopy(alloc)
    alloc_tmp = np.hstack((alloc_tmp, np.zeros((n, n * c - m))))
    # 4) prefrence
    preferences_tmp = copy.deepcopy(preferences_dummy)
    # 5)freshhold
    threshhold = math.floor(m / n)
    # 6) eaten
    left_to_eat = [1 for _ in objects_dummy]

    ro = 0
    # eat until the cant eat
    while True:
        if ro > 1000:
            raise ValueError("asd")
        ro += 1
        # first who can eat when in non break
        who_can_eat = [agents[i] for i in range(len(agents)) if round(ate[i],8) < threshhold]
        if not who_can_eat:
            break

        what_i_want_eat = [
            -1 if preferences_tmp[a] == [] else preferences_tmp[a][0] for a in agents
        ]

        # get the min of all alloction and reduce from all
        min_of_all = min(alloc_tmp[a][what_i_want_eat[a]] for a in who_can_eat)
        for ag in agents:
            alloc_tmp[ag][what_i_want_eat[ag]] -= min_of_all

        # choose who eats == 0
        have_from_my_item = [alloc_tmp[a][what_i_want_eat[a]] for a in agents]
        who_will_eat = [a for a in who_can_eat if have_from_my_item[a] <= 0]
        the_next_in_line = min([x for x in have_from_my_item if x > 0] + [np.inf])
        # calculate how much can eat min(diff from threshhold, the next in line,item eaten)
        until_end_of_item = [left_to_eat[o] /
                             (1 if len([x for x in who_will_eat if what_i_want_eat[x] == o]) == 0
                              else len([x for x in who_will_eat if what_i_want_eat[x] == o]))
                             for o in objects_dummy if left_to_eat[o] > 0] + [np.inf]
        how_much_to_eat = min(
            min(threshhold - ate[a] for a in who_will_eat),
            # the_next_in_line / len(who_will_eat),
            the_next_in_line,
            min(until_end_of_item)
        )
        # update P
        for a in who_will_eat:
            P[a][what_i_want_eat[a]] += how_much_to_eat
        # remove the eaten from who can eat, and the sum from the delay of who didnt eat
        for a in who_will_eat:
            ate[a] += how_much_to_eat

        for a in who_will_eat:
            left_to_eat[what_i_want_eat[a]] -= how_much_to_eat
        left_to_eat = [np.round(a, 8) for a in left_to_eat]

        how_much_been_eaten = len(who_will_eat) * how_much_to_eat
        for a in [a for a in agents if a not in who_will_eat]:
            alloc_tmp[a][what_i_want_eat[a]] -= how_much_been_eaten

        # move allocation from this to next
        for a,p in enumerate(preferences_tmp):
            carryToNext = 0
            for i in p:
                alloc_tmp[a][i] += carryToNext
                if left_to_eat[i] <= 0:
                    carryToNext += alloc_tmp[a][i]


        # update the next to eat(tmp_preference)
        for i in range(len(preferences_tmp)):
            preferences_tmp[i] = [x for x in preferences_tmp[i] if left_to_eat[x] > 0]

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
    if len(dummy) > 0:
        P = P[:, :-len(dummy)]
    return result, P


def print_PS_Lottery(result, r, c, prefe, alloc):
    print("================================================")
    index = 0
    for item in result:
        newMat = alloc + item[1]
        print(f"Probability: {item[0]}")
        print(f"Matrix No {index}:")
        print_mat(newMat, r, c)
        print(f"Is ex-post EF? {isEF(newMat, prefe)}")
        print(f"Is ex-post EF-1? {isEF1(newMat, prefe)}")
        print(f"Is ex-post EF-2? {isEF2(newMat, prefe)}")

        print()
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


def isEF(mat, pref) -> bool:
    for me in range(len(mat)):
        for other in range(len(mat)):
            if me == other:
                continue
            my_sum = 0
            is_sum = 0
            for item in pref[me]:
                my_sum += mat[me][item]
                is_sum += mat[other][item]
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
                my_sum += mat[me][item]
                is_sum += mat[other][item]
                if is_sum - 0.00000000001 > my_sum:
                    # print(f'I {me} envy {other} with {is_sum - my_sum} with the item {item} my pref is {pref[me]}')
                    # print(pref)
                    # raise ValueError("ex-post ef2")
                    return False
    return True


def isEF2(mat, pref) -> bool:
    for me in range(len(mat)):
        for other in range(len(mat)):
            if me == other:
                continue
            my_sum = 0
            is_sum = -2
            for item in pref[me]:
                my_sum += mat[me][item]
                is_sum += mat[other][item]
                if is_sum - 0.00000000001 > my_sum:
                    print(f'I {me} envy {other} with {is_sum - my_sum} with the item {item} my pref is {pref[me]}')
                    print(pref)
                    raise ValueError("ex-post ef2")
                    # return (is_sum, my_sum)

    return True


def generate_random_preferences(agents, items):
    preferences = []
    for agent in agents:
        # Shuffle items to create a random preference list for each agent
        shuffled_items = random.sample(items, len(items))
        preferences.append(shuffled_items)
    return preferences


if __name__ == "__main__":
    # compare_solution_methods()
    num_of_test = 1000000
    test_per_sit = 100
    max_agent = 3
    max_object = 3
    for _ in range(num_of_test):
        day = 0
        # agent = [0, 1, 2]
        # item = [0, 1, 2]
        # originalPref = [[1, 0, 2], [0, 1, 2], [1, 0, 2]]
        # num = random.randint(2, max_agent)
        num = 3
        agent = list(range(num))
        item = list(range(num))
        originalPref = generate_random_preferences(agents=agent,items=item)
        alloction = np.zeros((len(agent), len(item)))
        for _ in range(test_per_sit):
            print(
                f"================================================\n                   day {day}              \n================================================"
            )

            #   given allocation and original pref give allocation
            try:
                result = MPS_Lottery(
                    agents=agent, objects=item, preferences=originalPref, alloc=alloction)

            except:
                break

            print_PS_Lottery(
                result=[(1, result[1])],
                r=agent,
                c=item,
                prefe=originalPref,
                alloc=alloction,
            )
            print_PS_Lottery(
                result=result[0], r=agent, c=item, prefe=originalPref, alloc=alloction
            )
            print(
                "What you like to do? \nq) quit the program \n0-inf) enter the number of the matrix you want to allocate(default)"
            )
            user_input = random.randint(0, len(result[0])-1)
            # user_input = -1
            # while user_input < 0 or user_input >= len(result[0]):
            #     user_input = input()
            #     if user_input == "q":
            #         sys.exit()
            #     user_input = int(user_input)
            # update allocation
            # get last 4 item
            mat = result[0][int(user_input)][1]
            alloction += mat
            day += 1
