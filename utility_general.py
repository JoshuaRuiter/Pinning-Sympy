# Various general utility functions related to matrices

import sympy as sp

def is_diagonal(my_matrix):
    # Return true if matrix is diagonal'    
    rows, cols = my_matrix.shape
    return all(my_matrix[i,j] == 0 for i in range(rows) for j in range(cols) if i != j)

def vector_variable(letter, length):
    return sp.Matrix(sp.symarray(letter, length))

