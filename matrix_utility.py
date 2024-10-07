# Various general-utility functions related to matrices

from sympy import shape
from numpy import array

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

def integer_linear_combos(root_list,alpha,beta):
    # Return a list of all positive integer linear combinations
    # of two roots alpha and beta within a list of roots
    
    assert(alpha in root_list)
    assert(beta in root_list)
    
    # combos is a dictionary, where keys are tuples (i,j)
    # and values are roots of the form i*alpha+j*beta
    combos = {}
    
    if not(root_sum(alpha,beta) in root_list):
        # If alpha+beta is not a root, there are no integer linear combos
        # and we return an empty list
        return combos
    else:
        combos[(1,1)] = root_sum(alpha,beta)
    
    # Run a loop where each iteration, we try adding alpha and beta
    # to each existing combo
    while True:
        new_combos = increment_combos(root_list,combos,alpha,beta)
        if len(combos) == len(new_combos):
            break;
        combos = new_combos
    
    return combos

def increment_combos(root_list,old_combos,alpha,beta):
    new_combos = old_combos.copy() # A shallow copy
    for key in old_combos:
        i = key[0]
        j = key[1]
        old_root = old_combos[key]
        if root_sum(old_root, alpha) in root_list:
            new_combos[(i+1,j)] = root_sum(old_root,alpha)
        if root_sum(old_root,beta) in root_list:
            new_combos[(i,j+1)] = root_sum(old_root,beta)
    return new_combos
    
def root_sum(list_1, list_2):
    return list(array(list_1) + array(list_2))

def scale_root(constant, root):
    return list(constant*array(root))
    