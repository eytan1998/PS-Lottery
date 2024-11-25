import copy
import math
import random
import networkx as nx

import numpy as np
from matplotlib import pyplot as plt
from networkx.drawing import bipartite_layout
from scipy.optimize import linear_sum_assignment


def Envy_Lottery(agents, objects, preferences, alloc):
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

    # Create a bipartite graph
    B = nx.Graph()

    # Add nodes with the bipartite attribute
    top_nodes = [i for i in range(n)]
    bottom_nodes = [chr(i+ord('A')) for i in range(m)]
    B.add_nodes_from(top_nodes, bipartite=0)
    B.add_nodes_from(bottom_nodes, bipartite=1)

    # Create a matrix representation
    matrix_size = (len(top_nodes), len(bottom_nodes))
    matching_matrix = np.zeros(matrix_size, dtype=int)

    alloc_tmp = copy.deepcopy(alloc)
    preferences_tmp = copy.deepcopy(preferences)

    while True:
        # add v for the one with the most envy

        # get the envy level the most envy number + and the index of him
        envy_level = getEnvyLevel(alloc_tmp, preferences)
        who_will_eat = getTheWhoMostEnvy(envy_level, agents)

        for a in who_will_eat:
            if not preferences_tmp[a]:
                continue
            B.add_edge(a, chr(preferences_tmp[a][0]+ord('A')))
            alloc_tmp[a][preferences_tmp[a][0]] += 1
            preferences_tmp[a].pop(0)

        # # Draw the bipartite graph
        # pos = nx.bipartite_layout(B, top_nodes)  # Position nodes in bipartite layout
        # nx.draw(B, pos, with_labels=True, node_color=['skyblue' if n in top_nodes else 'lightgreen' for n in B.nodes()])
        # plt.title("Bipartite Graph")
        # plt.show()

        # Check for a perfect matching
        matching = nx.bipartite.maximum_matching(B,top_nodes)
        # Extract the matching pairs
        perfect_matching = {u: v for u, v in matching.items() if u in top_nodes}
        if len(perfect_matching) == len(top_nodes):
            for u, v in perfect_matching.items():
                i = top_nodes.index(u)
                j = bottom_nodes.index(v)
                matching_matrix[i, j] = 1  # Mark the matching pair
            break



    return matching_matrix, matching_matrix


def getTheWhoMostEnvy(envy_level,ag):
    max_envy = max(t[0] if i in ag else -1 for i,t in enumerate(envy_level))
    candidates = [(index, t) for index, t in enumerate(envy_level) if t[0] == max_envy and index in ag]
    min_index = min(t[1] for index, t in candidates)
    return [index for index, t in candidates if t[1] == min_index]


def getEnvyLevel(mat, pref ) -> bool:
    envy_level = [(0, np.inf) for _ in range(len(mat[0]))]
    for me in range(len(mat)):
        for other in range(len(mat)):
            if me == other:
                continue
            my_sum = is_sum = 0
            for i, item in enumerate(pref[me]):
                my_sum += mat[me][item]
                is_sum += mat[other][item]
                if is_sum - 0.00001 > my_sum:
                    envy_level[me] = (is_sum - my_sum, i)
    return envy_level




def print_PS_Lottery(res, r, c, prefe, alloc):
    print("================================================")
    index = 0
    for item in res:
        new_mat = alloc + item[1]
        print(f"Probability: {item[0]}")
        print(f"Matrix No {index}:")
        print_mat(new_mat, r, c)
        print(f"Is ex-post EF? {isEF(new_mat, prefe)}")
        print(f"Is ex-post EF-1? {isEF1(new_mat, prefe)}")
        print(f"Is ex-post EF-2? {isEF2(new_mat, prefe)}")

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
                    print(f'I {me} envy {other} with {is_sum - my_sum} with the item {item} my pref is {pref[me]}')
                    print(pref)
                    raise ValueError("ex-post ef1")
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

    return True


def generate_random_preferences(agents, items):
    preferences = []
    for agent in agents:
        # Shuffle items to create a random preference list for each agent
        shuffled_items = random.sample(items, len(items))
        preferences.append(shuffled_items)
    return preferences


if __name__ == "__main__":
    num_of_test = 1000
    test_per_sit = 100
    max_agent = 3
    max_object = 5
    for _ in range(num_of_test):
        day = 0
        agent = [0, 1, 2]
        items = [0, 1, 2]
        originalPref = [[1, 0, 2], [0, 1, 2], [1, 0, 2]]
        # agent = list(range(random.randint(2, max_agent)))
        # agent = items = list(range(random.randint(2, max_object)))
        # originalPref = generate_random_preferences(agents=agent,items=items)
        alloction = np.zeros((len(agent), len(items)))
        for _ in range(test_per_sit):
            print(
                f"================================================\n                   day {day}              \n================================================"
            )

            #   given allocation and original pref give allocation
            result = Envy_Lottery(
                agents=agent, objects=items, preferences=originalPref, alloc=alloction
            )
            print_PS_Lottery(
                res=[(1, result[1])],
                r=agent,
                c=items,
                prefe=originalPref,
                alloc=alloction,
            )
            print_PS_Lottery(
                res=result[0], r=agent, c=items, prefe=originalPref, alloc=alloction
            )
            print(
                "What you like to do? \nq) quit the program \n0-inf) enter the number of the matrix you want to allocate(default)"
            )
            user_input = 0
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
