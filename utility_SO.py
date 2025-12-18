import sympy as sp

def is_group_element_SO(matrix_to_test, form):
    X = matrix_to_test
    B = form.matrix
    return (X.T*B*X == B and X.det()==1)

def is_torus_element_SO(matrix_to_test, matrix_size, rank, form):
    n = matrix_size
    q = rank
    if sp.shape(matrix_to_test) != (n,n):
        return False
    for i in range(q):
        if (matrix_to_test[i,i]*matrix_to_test[q+i,q+i]!=1):
            return False
    for j in range(n - 2*q):
        if (matrix_to_test[2*q+j,2*q+j]!=1):
            return False
    return True

def generic_torus_element_SO(matrix_size, rank, form, vec_t):
    n = matrix_size
    q = rank
    assert(q == len(vec_t))
    my_matrix = sp.zeros(n)
    for i in range(q):
        my_matrix[i,i] = vec_t[i]
        my_matrix[q+i,q+i] = 1/vec_t[i]
    for j in range(n - 2*q):
        my_matrix[2*q + j,2*q+j] = 1
    return my_matrix

def is_lie_algebra_element_SO(matrix_to_test,form):
    X = matrix_to_test
    B = form.matrix        
    return (X.T*B == -B*X)


def generic_lie_algebra_element_SO(matrix_size, letter = 'x'):
    return None # PLACEHOLDER, INCOMPLETE