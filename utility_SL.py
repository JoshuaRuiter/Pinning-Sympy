import sympy as sp
import numpy as np
from utility_general import is_diagonal

def is_group_element_SL(matrix_to_test, form = None):
    return matrix_to_test.det() == 1

def is_torus_element_SL(matrix_to_test, rank = None):
    return (is_diagonal(matrix_to_test) and 
            matrix_to_test.det() == 1)

def generic_torus_element_SL(matrix_size, rank = None, letter = 't'):
    # Output a torus element of the usual diagonal subgroup of SL
    if rank is None:
        rank = matrix_size - 1
    else: 
        assert matrix_size == rank + 1, "Rank of diagonal torus in SL_n is n-1"
    vec_t = sp.symarray(letter,matrix_size)
    t = sp.diag(*vec_t)
    t[matrix_size-1,matrix_size-1] = t[matrix_size-1,matrix_size-1]/sp.prod(vec_t)
    return t

def character_entries_SL(matrix_size, rank = None):
    return [1]*matrix_size

def trivial_characters_SL(matrix_size, rank = None):
    trivial_characters = [[1] * matrix_size]
    matrix_with_trivial_character_columns = np.array(np.stack(trivial_characters, axis=1))
    return matrix_with_trivial_character_columns

def is_lie_algebra_element_SL(matrix_to_test, form = None):
    return matrix_to_test.trace() == 0

def generic_lie_algebra_element_SL(matrix_size, rank = None, form = None, letter = 'x'):
    X = sp.Matrix(sp.symarray(letter, (matrix_size, matrix_size)))
    trace = X.trace()
    bottom_right_entry = X[matrix_size -1, matrix_size - 1]
    X[matrix_size - 1, matrix_size - 1] = bottom_right_entry - trace
    return X