import sympy as sp
import numpy as np

def group_constraints_SO(matrix_to_test, form):
    """
    Returns a list of SymPy equations enforcing:
        1. X^T*B*X = B
        2. det(X)) = 1
    X : n x n SymPy Matrix
    B : n x n SymPy Matrix representing the form
    """
    X = matrix_to_test
    n = X.shape[0]
    B = form.matrix
    M = X.T * B * X - B
    eqs = []
    for i in range(n):
        for j in range(n):
            expr = M[i,j]
            if not expr.is_zero:
                eqs.append(expr)
    det_expr = X.det() - 1
    if not det_expr.is_zero: eqs.append(det_expr)
    return eqs
    
def lie_algebra_constraints_SO(matrix_to_test, form):
    X = matrix_to_test
    n = X.shape[0]
    B = form.matrix
    M = X.T*B + B*X
    eqs = []
    for i in range(n):
        for j in range(n):
            expr = M[i,j]
            if not expr.is_zero:
                eqs.append(expr)
    tr_expr = X.trace()
    if not tr_expr.is_zero: eqs.append(tr_expr)
    return eqs

def is_lie_algebra_element_SO(matrix_to_test, form):
    X = matrix_to_test
    B = form.matrix
    return (X.T*B).equals(-B*X) and sp.simplify(X.trace()) == 0

def is_torus_element_SO(matrix_to_test, rank):
    n = matrix_to_test.shape[0]
    q = rank
    for i in range(q):
        if (matrix_to_test[i,i]*matrix_to_test[q+i,q+i]!=1):
            return False
    for j in range(n - 2*q):
        if (matrix_to_test[2*q+j,2*q+j]!=1):
            return False
    return True

def generic_torus_element_SO(matrix_size, rank, letter = 't'):
    # Output a 'generic' element of the diagonal torus subgroup of 
    # the special orthogonal group, which has the form
    # diag(t_1, ..., t_q, t_1^(-1), ..., t_q^(-1), 1, ..., 1)
    # where q = rank
    vec_t = sp.symarray(letter,rank,nonzero=True)
    t = sp.eye(matrix_size)
    for i in range(rank):
        t[i,i] = vec_t[i]
        t[rank+i,rank+i] = 1/vec_t[i]
    return t

def character_entries_SO(matrix_size, rank):
    return [1]*rank + [0]*(matrix_size - rank)

def trivial_characters_SO(matrix_size, rank):
    trivial_characters = [np.array([1 if j == i or j == i + rank else 0 for j in range(matrix_size)])for i in range(rank)]
    if not (rank == 1 and matrix_size == 2): trivial_characters.append([1] * matrix_size)
    matrix_with_trivial_character_columns = np.array(np.stack(trivial_characters, axis=1))
    return matrix_with_trivial_character_columns

def generic_lie_algebra_element_SO(matrix_size, rank, form, letter = 'x'):
    n = matrix_size
    q = rank
    m = n - 2*q
    assert form is not None
    c = form.anisotropic_vector
    assert len(c) == m
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
    # and trace(X) = 0, which is equivalent to trace(X_33) = 0
    # because tr(X_22) = - tr(X_11)
    
    # set the (2,2) block to be the negative transpose of the (1,1) block
    for i in range(q):
        for j in range(q):
            X[i+q,j+q] = -X[j,i]
            
    # (1,2) and (2,1) blocks are equal to their own negative transpose,
    # i.e. skew-symmetric
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

    assert is_lie_algebra_element_SO(X, form)
    return X