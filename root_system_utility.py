from numpy import array, dot

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

def reflect_root(alpha,beta):
    # Given two roots alpha and beta, compute the reflection
    # of beta across the hyperplane perpendicular to alpha
    coeff = int(-2*dot(alpha,beta)/dot(alpha,alpha))
    return root_sum(beta,scale_root(coeff,alpha))