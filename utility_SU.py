from utility_general import is_diagonal
import sympy as sp
import numpy as np

def custom_conjugate(my_expression, primitive_element):
    # Conjugate an expression with entries in the quadratic field extension k(sqrt(d))
    # by replacing sqrt(d) with -sqrt(d)
    
    # This checks that expression is something valid to substitute into
    # i.e. not an integer or something, and then does the subsitution
    if hasattr(my_expression, 'has') and my_expression.has(primitive_element):
        return my_expression.subs(primitive_element, -primitive_element)
    else:
        # If the expression is an integer or such type, then 
        # just return the expression
        return my_expression

def custom_real_part(my_expression, primitive_element):
    # Return the "real part" of an element of a quadratic field extension k(p_e)
    # by replacing p_e with zero
    if hasattr(my_expression, 'has') and my_expression.has(primitive_element):
        return my_expression.subs(primitive_element, 0)
    else:
        # If the expression is an integer or such type, then 
        # just return the expression
        return my_expression

def custom_imag_part(my_expression, primitive_element):
    # Return the "real part" of an element of a quadratic field extension k(p_e)
    # by extracing the coefficient on p_e
    if hasattr(my_expression, 'has') and my_expression.has(primitive_element):
        return my_expression.coeff(primitive_element)
    else:
        # If the expression is an integer or such type, then 
        # just return the expression
        return 0
    
def group_constraints_SU(matrix_to_test, form):
    """
    Returns a list of SymPy equations enforcing:
        1. (X conjugate transpose)*H*X = H
        2. det(X)) = 1
    X : n x n SymPy Matrix
    H : n x n SymPy Matrix representing the form
    """
    X = matrix_to_test
    X_conjugate = custom_conjugate(X, form.primitive_element)
    n = X.shape[0]
    H = form.matrix
    M = X_conjugate.T * H * X - H
    eqs = []
    for i in range(n):
        for j in range(n):
            expr = M[i,j]
            if not expr.is_zero:
                eqs.append(expr)
    det_expr = X.det() - 1
    if not det_expr.is_zero: eqs.append(det_expr)
    return eqs

def lie_algebra_constraints_SU(matrix_to_test, form):
    X = matrix_to_test
    X_conjugate = custom_conjugate(X, form.primitive_element)
    n = X.shape[0]
    H = form.matrix
    M = X_conjugate.T*H + H*X
    eqs = []
    for i in range(n):
        for j in range(n):
            expr = M[i,j]
            if not expr.is_zero:
                eqs.append(expr)
    tr_expr = X.trace()
    if not tr_expr.is_zero: eqs.append(tr_expr)
    return eqs

def is_torus_element_SU(matrix_to_test, rank):
    # Return true if matrix_to_test is an element of the diagonal torus of
    # the special unitary group.
    
    # All elements of this torus have the form
    # diag(t_1, ..., t_q, t_1^(-1), ..., t_q^(-1), 1, ..., 1)
        # where t_1, ..., t_q are 'purely real' if the form is hermitian,
        # and 'purely imaginary' if the form is skew-hermitian
    if not(is_diagonal(matrix_to_test)): return False
    for i in range(rank):
        if matrix_to_test[i,i]*matrix_to_test[rank+i,rank+i] != 1: return False
    return True

def generic_torus_element_SU(matrix_size, rank, letter = 't'):
    # Output a 'generic' element of the diagonal torus subgroup of 
    # the special unitary group, which has the form
    # diag(t_1, ..., t_q, t_1^(-1), ..., t_q^(-1), 1, ..., 1) =
    #       [T     0        0]
    #       [0     T^(-1)   0]
    #       [0     0        I]
    # where q = is the rank (of SU_n_q, and also the rank of the torus)
    #   and also q is the Witt index of the associated (skew)-Hermitian form
    vec_t = sp.symarray(letter,rank, nonzero = True)
    t = sp.eye(matrix_size)
    for i in range(rank):
        t[i,i] = vec_t[i]
        t[rank+i,rank+i] = 1/vec_t[i]
    return t

def character_entries_SU(matrix_size, rank):
    return [1]*rank + [0]*(matrix_size - rank)

def trivial_characters_SU(matrix_size, rank):
    trivial_characters = [np.array([1 if j == i or j == i + rank else 0 for j in range(matrix_size)])for i in range(rank)]
    if not (rank == 1 and matrix_size == 2): trivial_characters.append([1] * matrix_size)
    matrix_with_trivial_character_columns = np.array(np.stack(trivial_characters, axis=1))
    return matrix_with_trivial_character_columns

def is_lie_algebra_element_SU(matrix_to_test, form):
    # Return true of matrix_to_test is an element of the special unitary group
    # The condition is
        # conj(X^T)*H + H*X = 0, where
        # X = matrix_to_test
        # H = matrix associated to the (skew-)hermitian form
    X = matrix_to_test
    H = form.matrix
    X_conjugate = custom_conjugate(X, form.primitive_element)
    
    ##########################################
    # print("\nH =")
    # sp.pprint(H)
    # print("\nX =")
    # sp.pprint(X)
    # print("\ntrace(X) =")
    # sp.pprint(X.trace())
    # print("\nX_conjugate.T*H + H*X =")
    # sp.pprint(X_conjugate.T*H + H*X)
    ##########################################
    
    return (X_conjugate.T*H).equals(-H*X) and sp.simplify(X.trace()) == 0

def generic_lie_algebra_element_SU(matrix_size, rank, form, letter = 'x'):
    n = matrix_size
    q = rank    
    assert form is not None
    c = form.anisotropic_vector
    p_e = form.primitive_element
    eps = form.epsilon
    X_real = sp.Matrix(sp.symarray(letter + '_r', (n,n)))
    X_imag = sp.Matrix(sp.symarray(letter + '_i', (n,n)))
    X = X_real + p_e*X_imag
    
    # X must be a block matrix of the form
    # [X_11, X_12, X_13
    #  X_21, X_22, X_23
    #  X_31, X_32, X_33]
    # subject to the constraints that
    # X_11 is arbitrary
    # X_22 = -(X_11)*
    # X_12 = -eps(X_12)*
    # X_21 = -eps(X_21)*
    # X_13 = -eps(X_32)* C
    # X_23 = -(X_31)* C
    # X_33 = -C^(-1) (X_33)* C
    # trace(X) = 0
    
    # In genera, the trace condition is equivalent to
    # trace(X_11) + trace(X_22) + trace(X_33) = 0
    # Since X_22 = -conjugate(X_11.T),
    # we have trace(X_22) = -conjugate(trace(X_11))
    # so in general,
    # trace(X_11) - conjugate(trace(X_11)) + trace(X_33) = 0
        
    # set up the (3,3) block first, because in the case n > 2*q,
    # we need to adjust the trace of X_11 in a way that depends
    # on X_33
    for i in range(n-2*q):
        # unlike with SO, the diagonal entries of the (3,3) block need not be zero,
        # but they must be "purely imaginary"
        X[i+2*q,i+2*q] = custom_imag_part(X[i+2*q,i+2*q], p_e)* p_e
        for j in range(i):
            X[i+2*q, j+2*q] = -c[j]/c[i] * custom_conjugate(X[j+2*q,i+2*q], p_e)
    
    # In the case n=2q (a quasi-split group),
    # the condition trace(X) = 0 is equivalent to
    # trace(X_11) - conjugate(trace(X_11)) = 0
    # or equivalently trace(X_11) = conjugate(trace(X_11))
    # which is to say trace(X_11) must be 'purely real'
    if n == 2*q:
        # In this case, we just modify the bottom right entry of the X_11 block
        # so that trace(X_11) is 'purely real'
        # This along with the requirement X_22 = -conjugate(X_11.T) 
        # will ensure that trace(X) = trace(X_11) + trace(X_22) = 0
        X[q-1,q-1] = (custom_real_part(X[q-1,q-1],p_e) 
                      - p_e*sum(custom_imag_part(X[i,i],p_e) for i in range(q-1)))
    else:
        assert n > 2*q
        trace_X_33 = sum(X[2*q + i, 2*q+ i] for i in range(n-2*q))
        X[q-1,q-1] = (custom_real_part(X[q-1,q-1],p_e) 
                      - p_e*sum(custom_imag_part(X[i,i],p_e) for i in range(q-1))
                      - trace_X_33/2)

    # Set the (2,2) block to be the negative conjugate 
    # transpose of the (1,1) block i.e. X_22 = -X_11^*
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
    
    ##########################################
    # print("\nX=")
    # sp.pprint(X)
    # print("\ntrace(X) =")
    # sp.pprint(sp.simplify(X.trace()))
    # trace_X_11 = sp.simplify(sum(X[i,i] for i in range(q)))
    # print("\nTrace(X_11) =")
    # sp.pprint(trace_X_11)
    # H = form.matrix
    # X_conjugate = custom_conjugate(X,p_e)
    # print("\nX_conjugate.T*H + H*X =")
    # sp.pprint(X_conjugate.T*H + H*X)
    ##########################################
    
    assert is_lie_algebra_element_SU(X, form)
    return X