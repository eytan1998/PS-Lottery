import copy
import random
from time import sleep

import numpy as np
from matplotlib import pyplot as plt
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
    # n = m
    m = len(objects)
    if n != m:
        raise ValueError("not NxN")

    # 3) alloction
    alloc_tmp = copy.deepcopy(alloc)
    # alloc_tmp = np.hstack((alloc_tmp, np.zeros((n, n))))
    # 4) prefrence
    preferences_tmp = copy.deepcopy(preferences)

    P = np.zeros((n, n))

    # Iterate over each column
    for col in range(n):
        # Count the number of zeros in the column
        num_zeros = np.sum(alloc_tmp[:, col] == 0)

        # Avoid division by zero if there are no zeros in the column
        if num_zeros > 0:
            # Replace zeros with 1/(number of zeros in the column)
            alloc_tmp[:, col] = np.where(alloc_tmp[:, col] == 0, 1 / num_zeros, alloc_tmp[:, col])
    P = alloc_tmp- alloc
    # split to presenters
    extP = np.zeros((n, n))
    for i, pref in enumerate(preferences_tmp):
        ate = 0
        age = i
        for o in pref:
            if ate + P[i][o] <= 1:
                extP[age][o] = P[i][o]
                ate += P[i][o]
            else:
                extP[age][o] = 1 - ate
                age = age + n
                extP[age][o] = P[i][o] - (1 - ate)
                ate = P[i][o] - (1 - ate)
    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    result = []
    for item in bikroft(extP):
        remove_dummy = np.delete(item[1], [], axis=1)
        stack_agents = np.array(
            np.sum([remove_dummy[row::n] for row in range(n)], axis=1)
        )
        result.append((item[0], stack_agents))
    # if len(dummy) > 0:
    #     P = P[:, :-len(dummy)]
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


def whatEF(mat, pref) -> int:
    max_envy =0
    for me in range(len(mat)):
        for other in range(len(mat)):
            if me == other:
                continue
            my_sum = 0
            is_sum = 0
            for item in pref[me]:
                my_sum += mat[me][item]
                is_sum += mat[other][item]
                if is_sum > my_sum and max_envy < is_sum-my_sum:
                    max_envy = is_sum - my_sum
    return max_envy


def generate_random_preferences(agents, items):
    preferences = []
    for agent in agents:
        # Shuffle items to create a random preference list for each agent
        shuffled_items = random.sample(items, len(items))
        preferences.append(shuffled_items)
    return preferences


if __name__ == "__main__":
    # compare_solution_methods()
    test_per_sit = 10

    nums = range(2,60)
    iteration = 1
    envys = []
    for i, num in enumerate(nums):
        print(num)
        envy_for_average = []
        for _ in range(iteration):
            max_envy = -1
            day = 0
            agent = list(range(num))
            item = list(range(num))


            originalPref = generate_random_preferences(agents=agent,items=item)
            alloction = np.zeros((len(agent), len(item)))
            for _ in range(test_per_sit):
                if np.all(alloction == 1):
                    alloction = np.zeros((len(agent), len(item)))
                # print(f"================================================\n                   day {day}              \n================================================")

                #   given allocation and original pref give allocation
                try:
                    result = MPS_Lottery(
                        agents=agent, objects=item, preferences=originalPref, alloc=alloction)
                except:
                    break

                user_input = random.randint(0, len(result[0]) - 1)

                # print_PS_Lottery(
                #     result=[(1, result[1])],
                #     r=agent,
                #     c=item,
                #     prefe=originalPref,
                #     alloc=alloction,
                # )
                # print(f'User input: {user_input}')

                # print_PS_Lottery(
                #     result=result[0], r=agent, c=item, prefe=originalPref, alloc=alloction
                # )
                tmpMaxEnvy = max([whatEF(alloction+ r[1],originalPref) for r in result[0]])
                if tmpMaxEnvy > max_envy:
                    max_envy = tmpMaxEnvy
                 # minEnvy = min([whatEF(alloction+ r[1],originalPref) for r in result[0]])
                # if minEnvy > 1:
                #     raise ValueError(f'EF{minEnvy}')
                # newResults = [r for r in result[0] if whatEF(alloction+r[1],originalPref) == minEnvy]
                # maxProb = max([r[0] for r in newResults])
                # newResults = [r for r in newResults if r[0] == maxProb]
                newResults = result[0]
                # print(
                #     "What you like to do? \nq) quit the program \n0-inf) enter the number of the matrix you want to allocate(default)"
                # )
                user_input = random.randint(0, len(newResults)-1)
                # print(f'User input: {user_input}')
                # user_input = -1
                # while user_input < 0 or user_input >= len(result[0]):
                #     user_input = input()
                #     if user_input == "q":
                #         sys.exit()
                #     user_input = int(user_input)
                # update allocation
                # get last 4 item
                mat = newResults[int(user_input)][1]
                alloction += mat

                day += 1
            if max_envy< 0:
                continue
            envy_for_average.append(max_envy)
        envys.append(np.average(envy_for_average))

    plt.plot(nums, envys, 'g-o')
    plt.xlabel('number of agents and objects')
    plt.ylabel('max envy')
    plt.grid(True)
    # plt.axhline(y=0, color='black', linestyle='--')  # Black dashed line at y=0
    plt.legend()
    plt.savefig("compere.png")  # after you plot the graphs, save them to a file and upload it separately.
    plt.show()  # this should show the plot on your screen
