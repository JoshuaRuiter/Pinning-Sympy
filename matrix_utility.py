# Various general utility functions related to matrices

import sympy as sp
from sympy import shape, prod

def is_diagonal(my_matrix):
    # Return true if matrix is diagonal
    my_shape = shape(my_matrix)
    rows = my_shape[0]
    columns = my_shape[1]
    for i in range(rows):
        for j in range(columns):
            if i != j:
                if my_matrix[i,j] != 0:
                    return False
    return True

def evaluate_character(character,torus_element):
    # Evaluate a character at a particular torus element
    # Character needs to be in the form of a vector like [1,0,0]
    my_shape = shape(torus_element)
    assert(my_shape[0]==my_shape[1])        # torus_element should be a square matrix
    assert(my_shape[0]==len(character))     # torus_element size should match character length
    assert(is_diagonal(torus_element))      # torus element should be diagonal
    return_value = 1;
    for i in range(len(character)):
        return_value = return_value * (torus_element[i,i]**character[i])
    return return_value

def generic_torus_element_SL(matrix_size, rank, form, my_vec):
    # Output a torus element of the usual diagonal subgroup of SL
    assert(matrix_size == rank + 1)
    assert(len(my_vec) == rank)
    t = sp.zeros(matrix_size)
    for i in range(rank):
        t[i,i] = my_vec[i]
    t[matrix_size-1,matrix_size-1] = 1/prod(my_vec)
    return t

def is_torus_element_SO(matrix_to_test, root_system, form):
    root_system_rank = len(root_system.simple_roots())
    
    if shape(matrix_to_test != (matrix_size,matrix_size)):
        return False
    
    for i in range(root_system_rank):
        if (matrix_to_test[i,i]*matrix_to_test[root_system_rank+i,root_system_rank+i]!=1):
            return False
    for j in range(matrix_size - 2*root_system_rank):
        if (matrix_to_test[2*root_system_rank+j,2*root_system_rank+j]!=1):
            return False
    return True