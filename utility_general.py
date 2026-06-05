# Various general utility functions related to matrices

import sympy as sp
import operator
# import random
import multiprocessing as mp
from copy import deepcopy
from tabulate import tabulate
# from wcwidth import wcswidth

def is_diagonal(my_matrix):
    # Return true if matrix is diagonal'    
    rows, cols = my_matrix.shape
    return all(my_matrix[i,j] == 0 for i in range(rows) for j in range(cols) if i != j)

def vector_variable(letter, length):
    return sp.Matrix(sp.symarray(letter, length))

def format_table(table, headers):
    """
    Formats math expressions within a table to utilize clean Unicode symbols,
    manually constructs a rigid fancy_grid layout, and cleanly isolates fraction
    components from SymPy matrix wrappers to prevent mangled stacked blocks.
    """
    import re

    subscript_map = {"_0": "₀", "_1": "₁", "_2": "₂", "_3": "₃", "_4": "₄",
                     "_5": "₅", "_6": "₆", "_7": "₇", "_8": "₈", "_9": "₉"}
    exponent_map = {"**2": "²", "**3": "³", "**4": "⁴"}

    def get_visual_width(text):
        return len(text)

    def visual_ljust(text, width):
        return text + " " * (width - get_visual_width(text))

    # --- Phase 1: Process and clean text symbols ---
    processed_rows = []
    for row in table:
        new_row = [str(item) for item in row]
        raw_str = new_row[-1]

        # 1. Clean up outer SymPy Matrix wrappers and structural brackets globally
        if 'Matrix' in raw_str:
            raw_str = re.sub(r'Matrix\(\[?', '', raw_str)
            raw_str = re.sub(r'\]?\)', '', raw_str)
        raw_str = raw_str.replace('[', '').replace(']', '')

        # 2. Convert power representations to clean unicode superscripts
        for ascii_exp, uni_exp in exponent_map.items():
            raw_str = raw_str.replace(ascii_exp, uni_exp)

        # 3. Convert subscripts cleanly
        for ascii_sub, uni_sub in subscript_map.items():
            raw_str = raw_str.replace(ascii_sub, uni_sub)

        # 4. Replace multiplication with dots safely
        raw_str = raw_str.replace('* ', '⋅').replace('*', '⋅')

        # 5. Handle vertical stacked fractions cleanly without carrying stray brackets
        if '/' in raw_str:
            parts = raw_str.split('/')
            numerator = parts[0].strip()
            denominator = parts[1].strip()
            
            # Extract explicit signs (+/-) from the front of the numerator to keep the fraction centered
            prefix = ""
            sign_match = re.match(r"^([-\s+]+)", numerator)
            if sign_match:
                prefix = sign_match.group(1)
                numerator = numerator[len(prefix):].strip()

            w = max(get_visual_width(numerator), get_visual_width(denominator))
            
            num_pad = (w - get_visual_width(numerator)) // 2
            denom_pad = (w - get_visual_width(denominator)) // 2
            
            num_line = " " * num_pad + numerator + " " * (w - get_visual_width(numerator) - num_pad)
            bar_line = '-' * w
            denom_line = " " * denom_pad + denominator + " " * (w - get_visual_width(denominator) - denom_pad)
            
            padding = " " * get_visual_width(prefix)
            raw_str = f"{prefix}{num_line}\n{padding}{bar_line}\n{padding}{denom_line}"
        else:
            raw_str = raw_str.strip()

        new_row[-1] = raw_str
        processed_rows.append(new_row)

    # Convert headers to strings and establish absolute dimensions
    str_headers = [str(h) for h in headers]
    num_cols = len(str_headers)

    # --- Phase 2: Compute uniform column widths using visual tracking ---
    col_widths = [get_visual_width(h) for h in str_headers]
    
    for row in processed_rows:
        for col_idx in range(num_cols):
            cell_lines = row[col_idx].split('\n')
            max_line_len = max(get_visual_width(line) for line in cell_lines) if cell_lines else 0
            if max_line_len > col_widths[col_idx]:
                col_widths[col_idx] = max_line_len

    # --- Phase 3: Construct the rigid Unicode fancy_grid ---
    top_border = "╒" + "╤".join("═" * (w + 2) for w in col_widths) + "╕"
    mid_header = "╞" + "╪".join("═" * (w + 2) for w in col_widths) + "╡"
    row_sep = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    bot_border = "╘" + "╧".join("═" * (w + 2) for w in col_widths) + "╛"

    def render_row_block(cells):
        split_cells = [cell.split('\n') for cell in cells]
        max_lines = max(len(lines) for lines in split_cells)
        
        row_output_lines = []
        for line_idx in range(max_lines):
            line_parts = []
            for col_idx in range(num_cols):
                cell_lines = split_cells[col_idx]
                text_segment = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                padded_text = visual_ljust(text_segment, col_widths[col_idx])
                line_parts.append(f" {padded_text} ")
            row_output_lines.append("│" + "│".join(line_parts) + "│")
            
        return "\n".join(row_output_lines)

    # Assemble final layout output
    output_lines = [top_border]
    output_lines.append(render_row_block(str_headers))
    output_lines.append(mid_header)
    
    for r_idx, row in enumerate(processed_rows):
        output_lines.append(render_row_block(row))
        if r_idx < len(processed_rows) - 1:
            output_lines.append(row_sep)
            
    output_lines.append(bot_border)
    return "\n".join(output_lines)

def format_table_OLD(table, headers):
    """
    Formats math expressions within a table to utilize clean Unicode symbols,
    constructs multi-line fraction boxes cleanly, stabilizes line lengths
    using visual character widths to prevent tabulate border clipping, 
    and returns a fancy_grid tabulate string.
    """
    import re
    import unicodedata

    subscript_map = {"_0": "₀", "_1": "₁", "_2": "₂", "_3": "₃", "_4": "₄",
                     "_5": "₅", "_6": "₆", "_7": "₇", "_8": "₈", "_9": "₉"}
    exponent_map = {"**2": "²", "**3": "³", "**4": "⁴"}

    def get_visual_width(text):
        """Calculates accurate visual printing width for strings containing unicode symbols."""
        return sum(2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1 for c in text)

    def visual_ljust(text, width):
        """Pads a string out to a specific visual width taking unicode spacing into account."""
        current_width = get_visual_width(text)
        return text + " " * (width - current_width)

    formatted_table = []

    for row in table:
        new_row = list(row)
        raw_str = str(new_row[-1])

        # 1. Convert power representations to clean unicode superscripts FIRST
        for ascii_exp, uni_exp in exponent_map.items():
            raw_str = raw_str.replace(ascii_exp, uni_exp)

        # 2. Apply variable transformations (fixes the "- u_0" spacing bug)
        for ascii_sub, uni_sub in subscript_map.items():
            pattern = rf"(\b\w){ascii_sub}([²³⁴])?"
            # Avoid inserting a hard space if the variable immediately follows a negative sign
            def replace_sub(m):
                var = m.group(1)
                exp = m.group(2) if m.group(2) else ''
                # Look backwards in the raw string up to the match index to check for a minus
                match_start = m.start()
                if match_start > 0 and raw_str[match_start - 1] in ('-', '+', '[', '(', ' '):
                    return f"{var}{uni_sub}{exp}"
                return f" {var}{uni_sub}{exp}"
                
            raw_str = re.sub(pattern, replace_sub, raw_str)

        # 3. Replace multiplication with dots
        prod_pattern = r"([₀-₉])([²³⁴])?\*\s*(\w)([₀-₉])([²³⁴])?"
        def replace_prod(m):
            p1 = m.group(2) if m.group(2) else ''
            p2 = m.group(5) if m.group(5) else ''
            return f"{m.group(1)}{p1}⋅{m.group(3)}{m.group(4)}{p2}"
        
        raw_str = re.sub(prod_pattern, replace_prod, raw_str)
        raw_str = raw_str.replace('* ', '⋅').replace('*', '⋅')

        # 4. Reconstruct fraction layout cleanly if an explicit division is present
        if ' / ' in raw_str or '/' in raw_str:
            parts = raw_str.split('/')
            numerator = parts[0].strip()
            denominator = parts[1].strip()

            if numerator.startswith('Matrix(['):
                numerator = numerator.replace('Matrix([', '').replace('])', '')
            if numerator.startswith('[') and denominator.endswith(']'):
                numerator = numerator[1:]
                denominator = denominator[:-1]

            prefix = ""
            sign_match = re.match(r"^([-\s+]+)", numerator)
            if sign_match:
                prefix = sign_match.group(1)
                numerator = numerator[len(prefix):].strip()

            width = max(len(numerator), len(denominator))

            num_line = numerator.center(width)
            bar_line = '-' * width
            denom_line = denominator.center(width)

            padding = " " * len(prefix)
            coeff_for_printing = (
                f"{prefix}{num_line}\n"
                f"{padding}{bar_line}\n"
                f"{padding}{denom_line}"
            )
        else:
            if raw_str.startswith('Matrix('):
                raw_str = raw_str.replace('Matrix(', '').rstrip(')')
            coeff_for_printing = raw_str

        # 5. Fix right-edge border alignment using exact visual widths
        lines = coeff_for_printing.split('\n')
        max_visual_len = max(get_visual_width(line) for line in lines) if lines else 0
        padded_coeff = '\n'.join(visual_ljust(line, max_visual_len) for line in lines)

        new_row[-1] = padded_coeff
        formatted_table.append(new_row)

    return tabulate(formatted_table, headers=headers, tablefmt="fancy_grid")

### DEPRECATED
# def random_int_vector(length, lower_bound, upper_bound, nonzero = True):
#     elements = []
#     for _ in range(length):
#         val = random.randint(lower_bound, upper_bound)
#         # If nonzero is flagged, redraw if we hit 0
#         if nonzero:
#             while val == 0:
#                 val = random.randint(lower_bound, upper_bound)
#         elements.append(val)
#     return sp.Matrix(elements)

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

def compare_nonzero_pattern(A, B, op = operator.eq):
    if A.shape != B.shape: return False
    mask_A = A.applyfunc(entry_to_mask)
    mask_B = B.applyfunc(entry_to_mask)
    
    # Standard: op(mask_A, mask_B) checks mask_A == mask_B
    # Covering: op(mask_A, mask_B) checks mask_A <= mask_B (equivalent to B >= A)
    return all(op(a, b) for a, b in zip(mask_A, mask_B))

### DEPRECATED
# def compute_order(my_function, my_input, limit):
#     original_input = my_input
#     x = my_input
#     i = 1
#     while i <= limit:
#         x = my_function(x)
#         if x == original_input:
#             return i
#         print("\noriginal input = ")
#         sp.pprint(original_input)
#         print("i =", i)
#         print("f^i(x) =")
#         sp.pprint(x)
#         i = i + 1
#     assert False, "Order exceeds limit"

### DEPRECATED
# def randomize_symbolic_matrix(matrix_expr, 
#                               lower_bound = -5, 
#                               upper_bound = 5,
#                               ignore_variables = None, 
#                               nonzero = True):
#     """
#     Takes a matrix expression containing free variables and returns a concrete, 
#     fully randomized numerical matrix by substituting integers within the specified range.
    
#     Parameters:
#     -----------
#     matrix_expr : sp.Matrix
#         The symbolic matrix expression to randomize.
#     ignore_variables : iterable of sp.Symbol, optional
#         Symbols present in the matrix that should remain symbolic and not be substituted.
#     lower_bound : int, optional
#         Minimum integer value for random assignment.
#     upper_bound : int, optional
#         Maximum integer value for random assignment.
#     nonzero : bool, optional
#         If True, ensures no variables are substituted with 0.
#     """
#     # Normalize ignore_variables to a set for fast, unified lookups
#     if ignore_variables is None:
#         ignored = set()
#     elif isinstance(ignore_variables, (str, sp.Symbol)):
#         ignored = {ignore_variables}
#     else:
#         ignored = set(ignore_variables)
        
#     # Isolate variables that are not explicitly ignored
#     symbols_to_randomize = matrix_expr.free_symbols - ignored
    
#     substitution_dict = {}
#     for sym in symbols_to_randomize:
#         val = random.randint(lower_bound, upper_bound)
#         if nonzero:
#             while val == 0:
#                 val = random.randint(lower_bound, upper_bound)
#         substitution_dict[sym] = sp.Integer(val)
        
#     return matrix_expr.subs(substitution_dict)

def indent_multiline(s, prefix="\t"):
    return "\n".join(prefix + line for line in s.splitlines())

def pretty_map(lhs, rhs, arrow='->', use_unicode=True):
    # Pass the use_unicode flag down to SymPy's pretty printer
    lhs_lines = sp.pretty(lhs, use_unicode=use_unicode).splitlines()
    rhs_lines = sp.pretty(rhs, use_unicode=use_unicode).splitlines()

    # Optional: Fallback to an ASCII arrow if unicode is disabled 
    # and the user passed a unicode arrow by default.
    if not use_unicode and arrow == '→':
        arrow = '->'

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
    max_lhs_width = max(len(line) for line in lhs_lines) if lhs_lines else 0

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