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


# Variable Neighborhood Search
def N1_rand_flip(assignment , n_vars) :
    neighbor = assignment.copy()
    f_var = random.randint(1 , n_vars)
    neighbor[ f_var ] = not assignment[ f_var ]
    return neighbor


def N2_two_flip(assignment , n_vars) :
    neighbor = assignment.copy()
    vars_to_flip = random.sample(range(1 , n_vars + 1) , min(2 , n_vars))

    for var in vars_to_flip :
        neighbor[ var ] = not assignment[ var ]
    return neighbor


def N3_tri_flip(assignment , n_vars) :
    neighbor = assignment.copy()
    vars_to_flip = random.sample(range(1 , n_vars + 1) , min(3 , n_vars))

    for var in vars_to_flip :
        neighbor[ var ] = not assignment[ var ]
    return neighbor


# Variable Neighborhood Descent
def descend_neighborhoods(formula , n_vars , neighborhood_funcs , h_func , max_steps=5000) :
    current_assign = start_random(n_vars)

    for steps in range(1 , max_steps + 1) :
        c_cost = h_func(formula , current_assign)
        if c_cost == 0 :
            return current_assign , steps , True

        k_idx = 0  # Start with N1

        while k_idx < len(neighborhood_funcs) :
            N_k = neighborhood_funcs[ k_idx ]

            best_n_k = None
            best_c_k = c_cost

            # Sample N_k neighborhood
            num_samples = n_vars * (k_idx + 1)

            for _ in range(num_samples) :
                neighbor = N_k(current_assign , n_vars)
                neighbor_cost = h_func(formula , neighbor)

                if neighbor_cost < best_c_k :
                    best_c_k = neighbor_cost
                    best_n_k = neighbor

            # VND Move Strategy
            if best_c_k < c_cost :
                current_assign = best_n_k
                c_cost = best_c_k
                k_idx = 0  # Reset to N1
            else :
                k_idx += 1  # Try the next neighborhood

        if k_idx == len(neighborhood_funcs) :
            return current_assign , steps , False

    return current_assign , max_steps , False  # Max steps reached


# Comparative Analysis Function
def run_full_comparison(k, m, n, num_runs, max_steps):

    # Make a random SAT problem
    formula = make_problem_instance(k, m, n)
    if not formula:
        print("No formula generated.")
        return

    results = {
        'HC (H1)': [],
        'BS3 (H1)': [],
        'BS4 (H1)': [],
        'VND (H1)': [],
        'HC (H2)': [],
    }

    print(f"\nTesting {num_runs} runs on {k}-SAT with {m} clauses and {n} vars")

    start = time.time()
    for i in range(num_runs):
        if i % 10 == 0 and i > 0:
            print(f"Run {i}")

        # Hill Climbing (H1)
        _, steps, solved = climb_hill(formula, n, H1_score, max_steps)
        results['HC (H1)'].append((solved, steps))

        # Beam Search width 3
        _, steps, solved = search_beam(formula, n, 3, H1_score, max_steps)
        results['BS3 (H1)'].append((solved, steps))

        # Beam Search width 4
        _, steps, solved = search_beam(formula, n, 4, H1_score, max_steps)
        results['BS4 (H1)'].append((solved, steps))

        # VND
        _, steps, solved = descend_neighborhoods(formula, n, [N1_rand_flip, N2_two_flip, N3_tri_flip], H1_score, max_steps)
        results['VND (H1)'].append((solved, steps))

        # Hill Climbing (H2)
        _, steps, solved = climb_hill_h2(formula, n, max_steps)
        results['HC (H2)'].append((solved, steps))

    print("\nDone! Time:", round(time.time() - start, 2), "s")
    print("Results:")
    for algo in results:
        solved = sum(1 for ok, _ in results[algo] if ok)
        avg_steps = np.mean([s for ok, s in results[algo] if ok]) if solved else max_steps
        print(f"{algo}: solved {solved}/{num_runs}, avg steps: {avg_steps:.2f}")

    return results


if __name__ == '__main__' :
    K = 3
    M = 5  # clauses
    N = 4  # variables

    K_CLAUSE_LENGTH = 3
    MAX_STEPS = 5000
    NUM_RUNS = 10

    # Under-Constrained (Easy)
    M1_CLAUSES = 15
    N1_VARS = 10

    print("TEST SET 1: UNDER-CONSTRAINED (Ratio = 1.5)")
    run_full_comparison(K_CLAUSE_LENGTH , M1_CLAUSES , N1_VARS , NUM_RUNS , MAX_STEPS)

    # Near Criticality (Hardest)
    M2_CLAUSES = 43
    N2_VARS = 10

    print("TEST SET 2: NEAR CRITICALITY (Ratio = 4.3)")
    run_full_comparison(K_CLAUSE_LENGTH , M2_CLAUSES , N2_VARS , NUM_RUNS , MAX_STEPS)

    # Over-Constrained (Easy)
    M3_CLAUSES = 70
    N3_VARS = 10
    print("TEST SET 3: OVER-CONSTRAINED (Ratio = 7.0)")
    run_full_comparison(K_CLAUSE_LENGTH , M3_CLAUSES , N3_VARS , NUM_RUNS , MAX_STEPS)