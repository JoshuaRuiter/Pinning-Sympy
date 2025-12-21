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

def generic_lie_algebra_element_SO(matrix_size, rank, form, letters = 'x'):
    n = matrix_size
    q = rank
    c = form.anisotropic_vector
    letter = letters[0]
    X = sp.Matrix(sp.symarray(letter, (n,n)))
    
    # X must be a block matrix of the form
    # [X_11, X_12, X_13
    #  X_21, X_22, X_23
    #  X_31, X_32, X_33]
    # subject to the constraints that
    # X_22 = -(X_11).T
    # X_12 = -(X_12).T
    # X_21 = -(X_21).T
    # X_13 = -(X_32).T * C
    # X_23 = -(X_31).T * C
    # X_33 = -C**(-1) * (X_33).T * C
    
    # set the (2,2) block to be the negative transpose of the (1,1) block
    for i in range(q):
        for j in range(q):
            X[i+q,j+q] = -X[j,i]
            
    # make the (1,2) and (2,1) blocks be their own negative transpose
    for i in range(q):
        # the diagonal must be zero
        X[i,i+q] = 0 
        X[i+q,i] = 0
        for j in range(i):
            X[j,i+q] = -X[i,j+q]
            X[j+q,i] = -X[i+q,j]
            
    # set up the (1,3) and (2,3) blocks
    for i in range(q):
        for j in range(n-2*q):
            X[i, j+2*q] = -c[j]*X[j+2*q, i+q] 
            X[i+q, j+2*q] = -c[j]*X[j+2*q, i] 
    
    # set up the (3,3) block
    for i in range(n-2*q):
        X[i+2*q,i+2*q] = 0
        for j in range(i):
            X[i+2*q, j+2*q] = -c[j]/c[i] *X[j+2*q,i+2*q]
    
    return X