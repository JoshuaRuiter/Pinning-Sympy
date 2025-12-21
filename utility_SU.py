from utility_general import is_diagonal, custom_conjugate, custom_real_part, custom_imag_part
import sympy as sp

def is_group_element_SU(matrix_to_test, form):
    # Return true if matrix_to_test is in the special unitary group
    # associated to form.
    # The condition for this is (conj(X^T))*H*X = H, where
        # X = matrix_to_test
        # H = form.matrix
    X = matrix_to_test
    H = form.matrix
    X_conjugate = custom_conjugate(X, form.primitive_element)   
    return (X_conjugate.T*H*X == H and X.det() == 1)

def is_torus_element_SU(matrix_to_test, rank, form):
    # Return true if matrix_to_test is an element of the diagonal torus of
    # the special unitary group.
    
    # All elements of this torus have the form
    # diag(t_1, ..., t_q, t_1^(-1), ..., t_q^(-1), 1, ..., 1)
        # where t_1, ..., t_q are 'purely real' if the form is hermitian,
        # and 'purely imaginary' if the form is skew-hermitian
    T = matrix_to_test
    if not(is_diagonal(T)):
        return False
    q = rank
    for i in range(q):
        if T[i,i]*T[q+i,q+i] != 1:
            return False
    return True

def generic_torus_element_SU(matrix_size, rank, form, vec_t):
    # Output a 'generic' element of the diagonal torus subgroup of the special unitary group
    # Given a vecgtor vec_t = [t_1, t_2, ..., t_q] of length q = root_system.rank,
    # the assciated torus element to return is
    # diag(t_1, ..., t_q, t_1^(-1), ..., t_q^(-1), 1, ..., 1)
    assert(len(vec_t) == rank)
    
    # Note that t_1, ..., t_q should be 'purely real' if the form is hermitian, and 
    # 'purely imaginary' if the form is skew-hermitian
    # What this means is that we should include an extra factor of form.primitive_element
    # with each t_i in the skew-hermitian case
    
    T = sp.eye(matrix_size)
    for i in range(rank):
        T[i,i] = vec_t[i]
        T[i+rank,i+rank] = 1/vec_t[i]
    return T

def is_lie_algebra_element_SU(matrix_to_test, form):
    # Return true of matrix_to_test is an element of the special unitary group
    # The condition is
        # conj(X^T)*H + H*X = 0, where
        # X = matrix_to_test
        # H = matrix associated to the (skew-)hermitian form
    X = matrix_to_test
    H = form.matrix
    X_conjugate = custom_conjugate(X, form.primitive_element)    
    return (X_conjugate.T*H == -H*X)

def generic_lie_algebra_element_SU(matrix_size, rank, form, letters = ('x','y')):
    n = matrix_size
    q = rank
    c = form.anisotropic_vector
    p_e = form.primitive_element
    eps = form.epsilon
    A = sp.Matrix(sp.symarray(letters[0], (n,n)))
    B = sp.Matrix(sp.symarray(letters[1], (n,n)))
    X = A + p_e*B
    
    # X must be a block matrix of the form
    # [X_11, X_12, X_13
    #  X_21, X_22, X_23
    #  X_31, X_32, X_33]
    # subject to the constraints that
    # X_22 = -(X_11)*
    # X_12 = -eps(X_12)*
    # X_21 = -eps(X_21)*
    # X_13 = -eps(X_32)* C
    # X_23 = -(X_31)* C
    # X_33 = -C^(-1) (X_33)* C
    
    # set the (2,2) block to be the negative conjugate transpose of the (1,1) block
    for i in range(q):
        for j in range(q):
            X[i+q,j+q] = -custom_conjugate(X[j,i], p_e)
    
    # make the (1,2) and (2,1) blocks be their own negative conjugate transpose
    for i in range(q):
        # unlike with SO, the diagonal entries of these blocks need not be zero
        # if eps=1, the diagonal entries must be "purely real"
        # if eps=-1, the diagonal entries must be "purely imaginary"
        if eps == 1:
            X[i,i+q] = custom_imag_part(X[i,i+q], p_e) * p_e
            X[i+q,i] = custom_imag_part(X[i+q,i], p_e) * p_e
        elif eps == -1:
            X[i,i+q] = -custom_real_part(X[i,i+q], p_e)
            X[i+q,i] = -custom_real_part(X[i+q,i], p_e)
            
        # the loop below takes care of the the non-diagonal entries
        for j in range(i):
            X[j,i+q] = -eps * custom_conjugate(X[i,j+q], p_e)
            X[j+q,i] = -eps * custom_conjugate(X[i+q,j], p_e)

    # set up the (1,3) and (2,3) blocks
    for i in range(q):
        for j in range(n-2*q):
            X[i, j+2*q] = -eps*c[j]*custom_conjugate(X[j+2*q, i+q], p_e)
            X[i+q, j+2*q] = -c[j]*custom_conjugate(X[j+2*q, i], p_e)
    
    # set up the (3,3) block
    for i in range(n-2*q):
        # unlike with SO, the diagonal entries of the (3,3) block need not be zero,
        # but they must be "purely imaginary"
        X[i+2*q,i+2*q] = custom_imag_part(X[i+2*q,i+2*q], p_e)* p_e
        for j in range(i):
            X[i+2*q, j+2*q] = -c[j]/c[i] * custom_conjugate(X[j+2*q,i+2*q], p_e)

    return X