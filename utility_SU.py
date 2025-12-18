from utility_general import is_diagonal, custom_conjugate
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
    return (X_conjugate.T*H*X == H)

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
    q = rank
    assert(len(vec_t) == q)
    
    # Note that t_1, ..., t_q should be 'purely real' if the form is hermitian, and 
    # 'purely imaginary' if the form is skew-hermitian
    # What this means is that we should include an extra factor of form.primitive_element
    # with each t_i in the skew-hermitian case
    if form.epsilon == 1:
        extra_factor = 1
    elif form.epsilon == -1:
        extra_factor = form.primitive_element
    else:
        # Should be impossible
        raise Exception('Invalid value of epsilon for a (skew-)hermitian form.')
    
    T = sp.eye(matrix_size)
    for i in range(q):
        T[i,i] = vec_t[i]*extra_factor
        T[i+q,i+q] = (1/vec_t[i])*extra_factor
        
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
    return (X_conjugate.T * H == -H*X)

def generic_lie_algebra_element_SU(matrix_size, letter = 'x'):
    return None # PLACEHOLDER, INCOMPLETE