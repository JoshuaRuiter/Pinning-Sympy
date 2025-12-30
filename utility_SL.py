import sympy as sp
from utility_general import is_diagonal, generate_character_list, determine_roots

def is_group_element_SL(matrix_to_test, form = None):
    return matrix_to_test.det() == 1

def is_torus_element_SL(matrix_to_test):
    return is_diagonal(matrix_to_test) and matrix_to_test.det() == 1

def generic_torus_element_SL(matrix_size, rank, form, letter = 't'):
    # Output a torus element of the usual diagonal subgroup of SL
    assert(matrix_size == rank + 1)
    vec_t = sp.symarray(letter,matrix_size)
    t = sp.diag(*vec_t)
    t[matrix_size-1,matrix_size-1] = t[matrix_size-1,matrix_size-1]/sp.prod(vec_t)
    return t

def trivial_characters_SL(matrix_size, rank = None):
    return [[1]*matrix_size]

def is_lie_algebra_element_SL(matrix_to_test, form = None):
    return matrix_to_test.trace() == 0

def generic_lie_algebra_element_SL(matrix_size, rank = None, form = None, letter = 'x'):
    X = sp.Matrix(sp.symarray(letter, (matrix_size, matrix_size)))
    trace = X.trace()
    bottom_right_entry = X[matrix_size -1, matrix_size - 1]
    X[matrix_size - 1, matrix_size - 1] = bottom_right_entry - trace
    return X

def determine_SL_roots(n):
    # Calculate roots and root spaces for SL_n
    
    # Generating a list of character vectors to try as roots
        # For special linear groups, the roots are of the form [0...,0,1,0,...0,-1,0,...0]
        # so we don't need an upper bound greater than 1
    list_of_characters = generate_character_list(character_length = n,
                                                 upper_bound = 1,
                                                 padded_zeros = 0)
    
    # Setting up the generic Lie algebra element
    x = sp.Matrix(sp.MatrixSymbol('x',n,n))
    lie_algebra_condition = sp.trace(x) # the condition is really trace(x)=0 but
                                     # the "=0" part is assumed by the Sympy solver
    
    # Setting up the generic torus element
    torus_variables = list(sp.symbols('s:'+str(n-1)))
    prod = 1
    for var in torus_variables:
        prod = prod*var
    torus_variables.append(1/prod)
    s = sp.Matrix(sp.diag(torus_variables))
    
    # Doing the calculations
    return determine_roots(generic_torus_element = s,
                           generic_lie_algebra_element = x,
                           lie_algebra_condition = lie_algebra_condition,
                           list_of_characters = list_of_characters,
                           variables_to_solve_for = x)