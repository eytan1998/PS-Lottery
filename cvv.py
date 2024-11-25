import cvxpy as cp
import numpy as np


def calculate_envy(mVars, mat, pref):
    num_individuals = mVars.shape[0]
    envy_levels = cp.Variable(num_individuals)

    # Envy constraints
    constraints = []
    for me in range(num_individuals):
        for other in range(num_individuals):
            if me == other:
                continue
            # Calculate utility sums
            my_sum = cp.sum(cp.multiply(mVars[me, :], mat[me, :]))
            other_sum = cp.sum(cp.multiply(mVars[other, :], mat[other, :]))
            envy_condition = other_sum - my_sum

            # We want to track the positive envy level
            constraints.append(envy_levels[me] >= cp.pos(envy_condition))

    # Total envy to minimize
    total_envy = cp.sum(envy_levels)
    return total_envy, constraints


def envy_freeness_problem(mat, pref):
    num_individuals = len(mat)
    num_items = len(mat[0])

    # Decision variable: allocation matrix
    mVars = cp.Variable((num_individuals, num_items), boolean=True)

    # Constraints: each item must be assigned to exactly one individual
    constraints = [cp.sum(mVars, axis=0) == 1]

    # Calculate total envy and additional constraints
    total_envy, envy_constraints = calculate_envy(mVars, mat, pref)

    # Combine constraints
    constraints += envy_constraints

    # Formulate the problem
    problem = cp.Problem(cp.Minimize(total_envy), constraints)

    # Solve the problem
    problem.solve()

    return mVars.value, problem.value  # Return the allocation matrix and the minimized envy


# Example usage
mat = np.array([[3, 2, 1], [2, 3, 1], [2, 3, 1]])  # Example payoffs
pref = [[0, 1, 2], [1, 0, 2], [1, 0, 2]]  # Example preferences
allocation, minimized_envy = envy_freeness_problem(mat, pref)

print("Allocation matrix:\n", allocation)
print("Minimized envy:", minimized_envy)
