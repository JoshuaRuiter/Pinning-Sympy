import sympy as sp
from utility_general import is_diagonal

def is_group_element_SL(matrix_to_test, form = None):
    return matrix_to_test.det() == 1

def is_torus_element_SL(matrix_to_test):
    return (is_diagonal(matrix_to_test) and 
            matrix_to_test.det() == 1)

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