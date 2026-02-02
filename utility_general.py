# Various general utility functions related to matrices

import sympy as sp
import multiprocessing as mp
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

def has_structural_contradiction(eqs, nonzero_vars):
    for eq in eqs:
        eq = sp.together(sp.simplify(eq))

        # Case 1: nonzero constant = 0
        if eq.is_number and eq != 0:
            return True

        # Case 2: equation forces a nonzero variable to be zero
        if eq.is_Symbol and eq in nonzero_vars:
            return True

        # Case 3: rational equation with constant numerator
        num, den = eq.as_numer_denom()

        if num.is_number and num != 0:
            # check whether denominator involves only invertible vars
            den_syms = den.free_symbols
            if den_syms and den_syms.issubset(nonzero_vars):
                return True

    return False

def _solve_worker(eqs, vars_to_solve_for, q):
    try:
        sol = sp.solve(eqs, vars_to_solve_for, dict=True, simplify=False)
        q.put(sol)
    except Exception:
        q.put(None)

def solve_with_timeout(eqs, vars_to_solve_for, timeout):
    q = mp.Queue()
    p = mp.Process(target=_solve_worker, args=(eqs, vars_to_solve_for, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError
    return q.get()

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