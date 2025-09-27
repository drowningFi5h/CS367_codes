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


if __name__ == '__main__' :
    K = 3
    M = 5  # clauses
    N = 4  # variables
    print("--- Test 1: Problem Generation and H1 Cost ---")

    # Problem instance
    test_formula = make_problem_instance(K , M , N)
    print(f"1. Generated 3-SAT Formula (M={M}, N={N}): {test_formula}")

    # Random assignment
    test_assign = start_random(N)
    print(f"2. Random Assignment: {test_assign}")

    # Initial cost (H1 score)
    initial_cost = H1_score(test_formula , test_assign)
    print(f"3. Initial H1 Cost (Unsatisfied Clauses): {initial_cost} / {M}")

    if initial_cost == 0 :
        print("-> Assignment unexpectedly satisfied the formula!")
    else :
        print("-> Core functions (generation/cost) appear to be working.")
