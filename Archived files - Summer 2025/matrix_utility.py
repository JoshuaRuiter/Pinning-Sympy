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
    
def evaluate_character(character,torus_element):
    # Evaluate a character at a particular torus element
    # Character needs to be in the form of a vector like [1,0,0]
    # Torus element should be a diagonal matrix
    
    my_shape = shape(torus_element)
    
    # torus_element should be a square matrix
    assert(my_shape[0]==my_shape[1])
    
    # torus_element size should match character length
    assert(my_shape[0]==len(character))
    
    # torus element should be diagonal
    assert(is_diagonal(torus_element))
    
    return_value = 1;
    for i in range(len(character)):
        return_value = return_value * torus_element[i,i]**character[i]
    
    return return_value

