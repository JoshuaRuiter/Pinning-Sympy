# Various general-utility functions related to matrices

from sympy import shape

def is_diagonal(my_matrix):
    # Return true if matrix is diagonal
    my_shape = shape(my_matrix)
    rows= my_shape[0]
    columns = my_shape[1]
    for i in range(rows):
        for j in range(columns):
            if i != j:
                if my_matrix[i,j] != 0:
                    return False
    return True            
    
    