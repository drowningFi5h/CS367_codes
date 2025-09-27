import random
import time
import numpy as np


# Problem Utility Functions and utilities

# clause is satisfied if at least one literal is True
def is_clause_ok(clause , assign) :
    for literal in clause :
        v_id = abs(literal)
        v_val = assign.get(v_id)

        if (literal > 0 and v_val is True) or \
                (literal < 0 and v_val is False) :
            return True
    return False


# counts the number of unsatisfied clauses and returns that count as cost
def find_cost(formula , assignment) :
    u_count = 0
    for clause in formula :
        if not is_clause_ok(clause , assignment) :
            u_count += 1
    return u_count


# creating a random assignment of n_vars variables
def start_random(n_vars) :
    return {i : random.choice([ True , False ]) for i in range(1 , n_vars + 1)}


# A simple 3-SAT problem generator
def make_problem_instance(k , m , n) :
    formula = [ ]
    variables = list(range(1 , n + 1))

    for _ in range(m) :
        if n < k :
            print("Error: n must be >= k")
            return [ ]
        clause_vars = random.sample(variables , k)

        clause = [ ]
        for var in clause_vars :
            # Randomly choose a variable or negation
            literal = var if random.choice([ True , False ]) else -var
            clause.append(literal)

        formula.append(clause)

    return formula


# Heuristic Score Functions

# Number of unsatisfied clauses
def H1_score(formula , assignment) :
    return find_cost(formula , assignment)


# GSAT-style Gain (Maximize)
def H2_gain(formula , assignment , var_to_flip) :
    orig_u = find_cost(formula , assignment)

    temp_assign = assignment.copy()
    temp_assign[ var_to_flip ] = not assignment[ var_to_flip ]

    new_u = find_cost(formula , temp_assign)

    # Gain = (Old Unsatisfied) - (New Unsatisfied)
    gain = orig_u - new_u
    return gain


# Neighborhood Definitions

# Flip a single random variable
def N1_rand_flip(assignment , n_vars) :
    neighbor = assignment.copy()
    f_var = random.choice(list(assignment.keys()))
    neighbor[ f_var ] = not assignment[ f_var ]
    return neighbor


if __name__ == '__main__' :
    K = 3
    M = 5  # clauses
    N = 4  # variables

    print("H2 Gain and N1 Neighborhood")
    # Simple test formula: (x1 or x2 or x3) and (not x1 or x2 or not x3) and (not x2 or not x3 or x1)
    test_formula = [[1, 2, 3], [-1, 2, -3], [-2, -3, 1]]

    # Initial assignment: {1: F, 2: T, 3: T} (Cost = 2)
    test_assign = {1: False, 2: True, 3: True}
    initial_cost = H1_score(test_formula, test_assign)

    print(f"1. Test Formula: {test_formula}")
    print(f"2. Initial Assignment: {test_assign} (Cost: {initial_cost})")

    var_to_test = 1
    gain = H2_gain(test_formula, test_assign, var_to_test)
    print(f"3. H2 Gain for flipping var {var_to_test} (x{var_to_test}): {gain}")

    neighbor = N1_rand_flip(test_assign, N)
    print(f"4. N1 Neighbor generated: {neighbor}")

