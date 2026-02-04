import sympy as sp
import numpy as np
from utility_general import is_diagonal

def group_constraints_SL(matrix_to_test, form = None):
    return [matrix_to_test.det() - 1]

def lie_algebra_constraints_SL(matrix_to_test, form = None):
    return [matrix_to_test.trace()]
    
def is_torus_element_SL(matrix_to_test, rank = None):
    return (is_diagonal(matrix_to_test) and 
            matrix_to_test.det() == 1)

def generic_torus_element_SL(matrix_size, rank = None, letter = 't'):
    # Output a torus element of the usual diagonal subgroup of SL
    if rank is not None:  assert matrix_size == rank + 1, "Rank of diagonal torus in SL_n is n-1"
    v = sp.symarray(letter, matrix_size, nonzero = True)
    return sp.diag(*v[:-1], v[-1] / sp.prod(v))

def character_entries_SL(matrix_size, rank = None):
    # This encodes the fact that characters of the diagonal torus 
    # for SL_n allows nonzero entries in all components of the vector
    # Technically, it would be possible to do this with always
    # having a zero in the last entry, but then the root system
    # would look way less symmetric
    return [1]*matrix_size

def trivial_characters_SL(matrix_size, rank = None):
    # The vector of all ones is a trivial character 
    # because the determinant is required to be 1
    return np.ones((matrix_size, 1), dtype=int)

def generic_lie_algebra_element_SL(matrix_size, rank = None, form = None, letter = 'x'):
    X = sp.Matrix(sp.symarray(letter, (matrix_size, matrix_size)))
    X[matrix_size-1, matrix_size-1] -= X.trace()
    return X
    
    # OLD VERSION
    # X = sp.Matrix(sp.symarray(letter, (matrix_size, matrix_size)))
    # trace = X.trace()
    # bottom_right_entry = X[matrix_size -1, matrix_size - 1]
    # X[matrix_size - 1, matrix_size - 1] = bottom_right_entry - trace
    # assert X.trace().is_zero
    # return X