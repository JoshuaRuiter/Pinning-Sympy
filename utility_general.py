# Various general utility functions related to matrices

import sympy as sp
import itertools
from copy import deepcopy

def is_diagonal(my_matrix):
    # Return true if matrix is diagonal'    
    rows, cols = my_matrix.shape
    return all(my_matrix[i,j] == 0 for i in range(rows) for j in range(cols) if i != j)

def vector_variable(letter, length):
    return sp.Matrix(sp.symarray(letter, length))

def indent_multiline(s, prefix="\t"):
    return "\n".join(prefix + line for line in s.splitlines())

def pretty_map(lhs, rhs, arrow='->'):
    lhs_lines = sp.pretty(lhs).splitlines()
    rhs_lines = sp.pretty(rhs).splitlines()

    # Heights
    h_lhs = len(lhs_lines)
    h_rhs = len(rhs_lines)
    h_max = max(h_lhs, h_rhs)

    # Pad top to center vertically
    lhs_pad_top = (h_max - h_lhs) // 2
    rhs_pad_top = (h_max - h_rhs) // 2
    lhs_lines = [''] * lhs_pad_top + lhs_lines + [''] * (h_max - h_lhs - lhs_pad_top)
    rhs_lines = [''] * rhs_pad_top + rhs_lines + [''] * (h_max - h_rhs - rhs_pad_top)

    # Determine where arrow goes (middle line)
    arrow_line = h_max // 2

    # Width of LHS for alignment
    max_lhs_width = max(len(line) for line in lhs_lines)

    result = ""
    for i, (l, r) in enumerate(zip(lhs_lines, rhs_lines)):
        if i == arrow_line:
            result = result + (f"{l.ljust(max_lhs_width)} {arrow} {r}\n")
        else:
            # maintain same spacing as arrow width
            result = result + (f"{l.ljust(max_lhs_width)} {' ' * len(arrow)} {r}\n")
    result = result[:-1] # chop off the very last newline character
    return result

def brute_force_vanishing_solutions_sparse(vanishing_conditions,
                                           variable_candidate_dict,
                                           min_nonzero=0,
                                           stop_after_solution=False,
                                           display = False):
    """
    Brute-force solver for vanishing conditions with sparsity ordering.

    Parameters
    ----------
    vanishing_conditions : list
        SymPy expressions assumed to vanish (== 0)
    variable_candidate_dict : dict
        {symbol: [candidate values]}
    min_nonzero : int
        Minimum total number of nonzero variables
    stop_after_solution : bool
        Stop after first solution is found

    Returns
    -------
    list of dict
        Solutions in SymPy solve-style format
    """

    solutions = []

    # Normalize equations
    vanishing_conditions = [sp.sympify(eq) for eq in vanishing_conditions]
    variables = list(variable_candidate_dict.keys())

    # Partition variables
    zero_allowed = [v for v in variables if 0 in variable_candidate_dict[v]]
    zero_forbidden = [v for v in variables if 0 not in variable_candidate_dict[v]]

    # Nonzero candidates for every variable
    nonzero_candidates = {
        v: [c for c in variable_candidate_dict[v] if c != 0]
        for v in variables
    }

    # Sanity checks
    for v in zero_forbidden:
        if not nonzero_candidates[v]:
            raise ValueError(f"Variable {v} has no valid nonzero candidates")

    F = len(zero_forbidden)      # forced nonzeros
    m = len(zero_allowed)        # optional nonzeros

    if min_nonzero > F + m:
        raise ValueError(
            f"min_nonzero={min_nonzero} exceeds total variables={F + m}"
        )

    # Required nonzeros among zero-allowed variables
    k_min = max(0, min_nonzero - F)
    k_max = m

    if display:
        print("\n" + "=" * 60 + "\nAttempting sparse brute force solution")
        print("Variables:", variables)
        print("Candidate dictionary:",variable_candidate_dict)
        print("Variables that can be zero:", zero_allowed)
        print("Variables that can't be zero':", zero_forbidden)
        print("Minimum nonzero variables:", min_nonzero)
        print("Vanishing conditions:")
        for e in vanishing_conditions:
            sp.pprint(e)
        tried = 0

    # k = number of nonzero variables chosen among zero-allowed ones
    for k in range(k_min, k_max + 1):
        
        if display: print(f"\nTrying assignments with {F + k} total nonzeros")

        for support in itertools.combinations(zero_allowed, k):

            # Initialize assignment
            assignment = {v: 0 for v in zero_allowed if v not in support}

            # Candidate lists
            nz_lists = [nonzero_candidates[v] for v in support]
            forced_lists = [nonzero_candidates[v] for v in zero_forbidden]

            for values in itertools.product(*nz_lists, *forced_lists):
                split = len(support)
                assignment.update(dict(zip(support, values[:split])))
                assignment.update(dict(zip(zero_forbidden, values[split:])))

                if display:
                    tried += 1
                    if tried % 10000 == 0:
                        print("Tried:", tried)

                if all(eq.subs(assignment) == 0 for eq in vanishing_conditions):
                    if display: print(f"SOLUTION FOUND (total nonzeros = {F + k})")
                    solutions.append(assignment.copy())

                    if stop_after_solution:
                        if display: print(f"Total candidates tried: {tried}\n" + "="*60 + "\n")
                        return solutions

    if display: print(f"\nTotal candidates tried: {tried}" + "=" * 60 + "\n")
    return solutions

def brute_force_vanishing_solutions_exact_pairs(vanishing_conditions,
                                                variable_candidate_dict,
                                                variable_pairs,
                                                min_nonzero=0,
                                                stop_after_solution=False,
                                                display = False):
    
    # This pair-based brute forcing method seemed useful for the case of
    # brute-forcing in the scenario of variables x_i + p_e*y_i,
    # but turned out not to be as useful as expected. I have kept it around
    # because it might turn out to be necessary later.
    
    """
    Brute-force solver enforcing exactly one nonzero per variable pair.

    Each pair (x, y) satisfies: exactly one of x, y is nonzero
    """

    solutions = []
    vanishing_conditions = [sp.sympify(eq) for eq in vanishing_conditions]
    num_pairs = len(list(variable_pairs))

    # Since exactly one nonzero per pair:
    total_nonzero = num_pairs
    if min_nonzero > total_nonzero:
        raise ValueError(
            f"min_nonzero={min_nonzero} exceeds total nonzeros={total_nonzero}"
        )

    # Precompute nonzero candidates
    nonzero_candidates = {
        v: [c for c in variable_candidate_dict[v] if c != 0]
        for v in variable_candidate_dict
    }

    # Validate pairs
    for x, y in variable_pairs:
        if not nonzero_candidates[x] and not nonzero_candidates[y]:
            raise ValueError(
                f"Neither variable in pair ({x}, {y}) has nonzero candidates"
            )
        
    if display:
        print("\n" + "=" * 60 + "\nAttempting paired brute force solution")
        print("Variable pairs:", list(variable_pairs))
        print("Exactly one nonzero per pair")
        print("Total nonzero variables:", total_nonzero)
        print("Vanishing conditions:")
        for e in vanishing_conditions:
            sp.pprint(e)
        tried = 0

    # For each pair: 0 = choose x, 1 = choose y
    side_choices = []
    for x, y in variable_pairs:
        choices = []
        if nonzero_candidates[x]:
            choices.append(0)
        if nonzero_candidates[y]:
            choices.append(1)
        side_choices.append(choices)

    # Iterate over side selections
    for sides in itertools.product(*side_choices):

        assignment = {}
        value_vars = []
        value_lists = []

        for (x, y), side in zip(variable_pairs, sides):
            if side == 0:
                assignment[y] = 0
                value_vars.append(x)
                value_lists.append(nonzero_candidates[x])
            else:
                assignment[x] = 0
                value_vars.append(y)
                value_lists.append(nonzero_candidates[y])

        # Iterate over actual nonzero values
        for values in itertools.product(*value_lists):
            assignment.update(dict(zip(value_vars, values)))

            if display:
                tried += 1
                if tried % 10000 == 0: print("Tried:", tried)

            if all(eq.subs(assignment) == 0 for eq in vanishing_conditions):
                if display: print("SOLUTION FOUND")
                solutions.append(assignment.copy())

                if stop_after_solution:
                    if display: print(f"Total candidates tried: {tried}" + "=" * 60 + "\n")
                    return solutions

    if display: print(f"Total candidates tried: {tried}" + "=" * 60 + "\n")
    return solutions

def find_zero_vars(expr, candidate_vars, generic_vars):
    zero_vars = set()
    if expr.is_zero: return zero_vars
    expr = sp.factor(sp.simplify(expr))
    
    for var in candidate_vars:
        
        if not expr.has(var): continue
        quotient = sp.simplify(expr / var)

        # Quotient must involve only generic variables
        if quotient.free_symbols - generic_vars: continue

        # Quotient must not be identically zero
        if quotient.is_zero: continue

        zero_vars.add(var)

    return zero_vars
            
def prune_singletons(matrix, variable_candidate_dict):
    """
    Return a pruned copy of variable_candidate_dict where any variable that is
    the sole remaining nonzero entry in a row or column has 0 removed
    from its candidate list.
    """

    pruned = deepcopy(variable_candidate_dict)
    n, m = matrix.shape
    assert n == m

    def prune_var(var):
        if var in pruned:
            pruned[var] = [v for v in pruned[var] if v != 0]

    # --- Check rows ---
    for i in range(n):
        vars_in_row = []
        for j in range(n):
            entry = matrix[i,j]
            if entry != 0 and entry in pruned:
                vars_in_row.append(entry)

        if len(vars_in_row) == 1:
            prune_var(vars_in_row[0])

    # --- Check columns ---
    for j in range(n):
        vars_in_col = []
        for i in range(n):
            entry = matrix[i,j]
            if entry != 0 and entry in pruned:
                vars_in_col.append(entry)

        if len(vars_in_col) == 1:
            prune_var(vars_in_col[0])

    return pruned