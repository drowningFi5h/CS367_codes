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


# Hill-Climbing Algorithms
# Steepest-Ascent Hill-Climbing (always moves to the best N1 neighbor)
def climb_hill(formula , n_vars , h_func , max_steps=5000) :
    current_assign = start_random(n_vars)

    for steps in range(1 , max_steps + 1) :
        c_cost = h_func(formula , current_assign)
        if c_cost == 0 :
            return current_assign , steps , True  # Solved

        best_n_assign = None
        best_n_cost = c_cost

        # Explore the N1 neighborhood (single flip)
        for var_to_flip in range(1 , n_vars + 1) :
            neighbor = current_assign.copy()
            neighbor[ var_to_flip ] = not current_assign[ var_to_flip ]

            n_cost = h_func(formula , neighbor)

            if n_cost < best_n_cost :
                best_n_cost = n_cost
                best_n_assign = neighbor

        if best_n_assign is None :
            return current_assign , steps , False  # Stuck

        current_assign = best_n_assign

    return current_assign , max_steps , False  # Max steps reached


# Special Hill-Climbing using H2 Gain maximization
def climb_hill_h2(f , n_v , max_s) :
    assign = start_random(n_v)
    for s in range(1 , max_s + 1) :
        if find_cost(f , assign) == 0 : return assign , s , True

        best_gain , best_flip_var = -float('inf') , None

        for var_to_flip in range(1 , n_v + 1) :
            gain = H2_gain(f , assign , var_to_flip)
            if gain > best_gain :
                best_gain = gain
                best_flip_var = var_to_flip

        if best_gain <= 0 : return assign , s , False

        assign[ best_flip_var ] = not assign[ best_flip_var ]
    return assign , max_s , False


# Local Beam Search
# Keeps track of 'beam_width' states, explores their neighbors, and selects the best '
def search_beam(formula , n_vars , beam_width , h_func , max_steps=5000) :
    current_beam = [ start_random(n_vars) for _ in range(beam_width) ]

    for steps in range(1 , max_steps + 1) :
        for assign in current_beam :
            if h_func(formula , assign) == 0 :
                return assign , steps , True

        next_cands = [ ]

        # Generate N1 neighbors for all states in the beam
        for assign in current_beam :
            for flip_var in range(1 , n_vars + 1) :
                neighbor = assign.copy()
                neighbor[ flip_var ] = not assign[ flip_var ]

                cost = h_func(formula , neighbor)
                next_cands.append((cost , neighbor))

        # Select top 'beam_width' candidates
        next_cands.sort(key=lambda x : x[ 0 ])
        current_beam = [ c[ 1 ] for c in next_cands[ :beam_width ] ]

        if not current_beam :
            return None , steps , False

    return current_beam[ 0 ] , max_steps , False  # Max steps reached


if __name__ == '__main__' :
    K = 3
    M = 5  # clauses
    N = 4  # variables

    print("Beam Search")

    # Generate a small, likely solvable problem
    test_formula = make_problem_instance(K , M , N)

    BEAM_W = 3
    start_time_bs = time.time()
    _ , steps_bs , solved_bs = search_beam(test_formula , N , BEAM_W , H1_score , max_steps=20)
    time_bs = time.time() - start_time_bs

    print(f"1. Formula: {test_formula}")
    print(f"2. Beam Search (B={BEAM_W}, H1): Solved: {solved_bs}, Steps: {steps_bs}, Time: {time_bs:.4f}s")

    BEAM_W_HC = 1
    _ , steps_bs_hc , solved_bs_hc = search_beam(test_formula , N , BEAM_W_HC , H1_score , max_steps=20)
    print(f"3. Beam Search (B={BEAM_W_HC}, H1): Solved: {solved_bs_hc}, Steps: {steps_bs_hc}")
