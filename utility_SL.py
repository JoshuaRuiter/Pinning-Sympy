import sympy as sp
from utility_general import is_diagonal

def is_group_element_SL(matrix_to_test, form = None):
    return matrix_to_test.det() == 1

def is_torus_element_SL(matrix_to_test):
    return is_diagonal(matrix_to_test) and matrix_to_test.det() == 1

def generic_torus_element_SL(matrix_size, rank, form, my_vec):
    # Output a torus element of the usual diagonal subgroup of SL
    assert(matrix_size == rank + 1)
    assert(len(my_vec) == rank)
    t = sp.zeros(matrix_size)
    for i in range(rank):
        t[i,i] = my_vec[i]
    t[matrix_size-1,matrix_size-1] = 1/sp.prod(my_vec)
    return t

def is_lie_algebra_element_SL(matrix_to_test, form = None):
    return matrix_to_test.trace() == 0

def generic_lie_algebra_element_SL(matrix_size, rank = None, form = None, letters = 'x'):
    letter = letters[0]
    X = sp.Matrix(sp.symarray(letter, (matrix_size, matrix_size)))
    trace = X.trace()
    bottom_right_entry = X[matrix_size -1, matrix_size - 1]
    X[matrix_size - 1, matrix_size - 1] = bottom_right_entry - trace
    return X