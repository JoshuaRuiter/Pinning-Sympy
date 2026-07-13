# This file contains useful helper functions related to special linear groups
# In particular, these are all in service of constructing an instance of a 
# pinned_group object to represent a special linear group.

# EXAMPLE USE CASE: constructing a 3x3 special linear pinned_group object
# n = 3
# T = split_torus(rank = n-1,
#                 is_element = is_torus_element_SL,
#                 generic_element = generic_torus_element_SL,
#                 trivial_character_matrix = trivial_characters_SL(n),
#                 nontrivial_character_entries = character_entries_SL(n))
# SL_n = pinned_group(name_string = f"SL(n={n})",
#                     matrix_size = n,
#                     form = None,
#                     group_constraints = group_constraints_SL,
#                     maximal_split_torus = T,
#                     lie_algebra_constraints = lie_algebra_constraints_SL,
#                     generic_lie_algebra_element = generic_lie_algebra_element_SL,
#                     non_variables = None)
# SL_n.fit_pinning(display = True)
# SL_n.validate_pinning(display = True)

import sympy as sp
import numpy as np
from utility_general import is_diagonal

group_constraints_SL_string = "$\operatorname{det}(X)=1$"
lie_algebra_constraints_SL_string = "$\operatorname{tr}(X)=0$"

def group_constraints_SL(matrix_to_test, form = None):
    # The requirement for a matrix X to be an element of a special linear group is det(X) = 1    
    # The pinned_group class expects this information in the form of a vanishing condition,
    # so this method effectively returns the equation "det(X)-1=0"
    return [matrix_to_test.det() - 1]

def lie_algebra_constraints_SL(matrix_to_test, form = None):
    # The requirements for a matrix X to be an element of a special linear Lie algebra is tr(X) = 0    
    # The pinned_group class expects this information in the form of a vanishing condition,
    # so this method effectively returns the equation "tr(X)=0"
    return [matrix_to_test.trace()]

def is_torus_element_SL(matrix_to_test, rank = None):
    # The requirement for a matrix to be an element of the usual torus subgroup of SL_n 
    # is just that the matrix is diagonal (and the determinant is 1)
    return (is_diagonal(matrix_to_test) and 
            matrix_to_test.det() == 1)

def generic_torus_element_SL(matrix_size, rank = None, letter = 't'):
    # This outputs a "generic" element of the diagonal torus in SL_n
    # Give a letter like 't', this creates the matrix
    # diag(t_1, t_2, ..., t_{n-1}, 1/(t_1*t_2*...*t_{n-1}))
    if rank is not None: assert matrix_size == rank + 1, "Rank of diagonal torus in SL_n is n-1"
    v = sp.symarray(letter, matrix_size, nonzero = True)
    return sp.diag(*v[:-1], v[-1] / sp.prod(v))

def character_entries_SL(matrix_size, rank = None):
    # When computing the root system for a special linear group,
    # this tells the solver to use character which can have nonzero
    # entries in all components.
    
    # Technically, it would be possible to find characters always
    # having a zero in the last entry, but then the root system
    # would look much less symmetric.
    return [1]*matrix_size

def trivial_characters_SL(matrix_size, rank = None):
    # The vector of all ones is a trivial character 
    # because the determinant is required to be 1
    return np.ones((matrix_size, 1), dtype=int)

def generic_lie_algebra_element_SL(matrix_size, rank = None, form = None, letter = 'x'):
    # This outputs a "generic" element of the Lie algebra sl_n
    # Since the only requirement is that the trace be zero, 
    # this just makes a matrix with entries x_{ij}
    # except the bottom right corner is changed so that the trace is zero,
    # i.e. the bottom right corner is the negative of the sum of the other diagonal entries    
    X = sp.Matrix(sp.symarray(letter, (matrix_size, matrix_size)))
    X[matrix_size-1, matrix_size-1] -= X.trace()
    return X