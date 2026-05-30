# Various general utility functions related to matrices

import sympy as sp
import random
import multiprocessing as mp
from copy import deepcopy

def is_diagonal(my_matrix):
    # Return true if matrix is diagonal'    
    rows, cols = my_matrix.shape
    return all(my_matrix[i,j] == 0 for i in range(rows) for j in range(cols) if i != j)

def vector_variable(letter, length):
    return sp.Matrix(sp.symarray(letter, length))

def random_int_vector(length, lower_bound, upper_bound, nonzero = True):
    elements = []
    for _ in range(length):
        val = random.randint(lower_bound, upper_bound)
        
        # If nonzero is flagged, redraw if we hit 0
        if nonzero:
            while val == 0:
                val = random.randint(lower_bound, upper_bound)
                
        elements.append(val)
        
    return sp.Matrix(elements)


def entry_to_mask(val):
    # 1. Fast track: explicit structural zero/nonzero
    if val.is_zero is True: return 0
    if val.is_zero is False: return 1
        
    # 2. Medium track: Quick numerical evaluation to catch non-zero constants
    # (e.g., expressions with sqrt(d) that evaluate to a distinct float)
    try:
        # Chop eliminates tiny floating-point noise around zero
        num_val = val.evalf(chop=True)
        if num_val == 0:
            return 0
        if num_val.is_number and num_val != 0:
            return 1
    except (TypeError, ValueError):
        pass

    # 3. Slow track: Fallback to full algebraic simplification
    return 0 if val.simplify().is_zero else 1

def nonzero_pattern_match(A, B):
    """
    Compares the nonzero patterns of two matrices by building and comparing binary masks.
    Optimized for cases where the matrices are highly likely to match.
    """
    if A.shape != B.shape: return False
    mask_A = A.applyfunc(entry_to_mask)
    mask_B = B.applyfunc(entry_to_mask)
    return mask_A == mask_B

def randomize_symbolic_matrix(matrix_expr, 
                              lower_bound = -5, 
                              upper_bound = 5,
                              ignore_variables = None, 
                              nonzero = True):
    """
    Takes a matrix expression containing free variables and returns a concrete, 
    fully randomized numerical matrix by substituting integers within the specified range.
    
    Parameters:
    -----------
    matrix_expr : sp.Matrix
        The symbolic matrix expression to randomize.
    ignore_variables : iterable of sp.Symbol, optional
        Symbols present in the matrix that should remain symbolic and not be substituted.
    lower_bound : int, optional
        Minimum integer value for random assignment.
    upper_bound : int, optional
        Maximum integer value for random assignment.
    nonzero : bool, optional
        If True, ensures no variables are substituted with 0.
    """
    # Normalize ignore_variables to a set for fast, unified lookups
    if ignore_variables is None:
        ignored = set()
    elif isinstance(ignore_variables, (str, sp.Symbol)):
        ignored = {ignore_variables}
    else:
        ignored = set(ignore_variables)
        
    # Isolate variables that are not explicitly ignored
    symbols_to_randomize = matrix_expr.free_symbols - ignored
    
    substitution_dict = {}
    for sym in symbols_to_randomize:
        val = random.randint(lower_bound, upper_bound)
        if nonzero:
            while val == 0:
                val = random.randint(lower_bound, upper_bound)
        substitution_dict[sym] = sp.Integer(val)
        
    return matrix_expr.subs(substitution_dict)

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
        sol = sp.solve(eqs, 
                       vars_to_solve_for, 
                       dict = True, 
                       simplify = False)
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