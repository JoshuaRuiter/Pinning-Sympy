# This file contains useful helper functions related to special orthogonal groups
# In particular, these are all in service of constructing an instance of a 
# pinned_group object to represent a special orthogonal group.

# EXAMPLE USE CASE: constructing a 4x4 special orthogonal pinned_group object of rank q=2
# q = 2
# n = 4
# anisotropic_vec = vector_variable('c',n-2*q)
# NIF = nondegenerate_isotropic_form(dimension = n,
#                                    witt_index = q,
#                                    anisotropic_vector = anisotropic_vec,
#                                    epsilon = None,
#                                    primitive_element = None)
# T = split_torus(rank = q,
#                 is_element = is_torus_element_SO,
#                 generic_element = generic_torus_element_SO,
#                 trivial_character_matrix = trivial_characters_SO(n,q),
#                 nontrivial_character_entries = character_entries_SO(n,q))
# SO_n_q = pinned_group(name_string = f"SO(n={n}, q={q})",
#                       matrix_size = n,
#                       form = NIF, 
#                       group_constraints = group_constraints_SO,
#                       maximal_split_torus = T,
#                       lie_algebra_constraints = lie_algebra_constraints_SO,
#                       generic_lie_algebra_element = generic_lie_algebra_element_SO,
#                       non_variables = None)
# SO_n_q.fit_pinning(display = True)
# SO_n_q.validate_pinning(display = True)

import sympy as sp
import numpy as np
from utility_general import is_diagonal

group_constraints_SO_string = "$X^T B X = B$ and $\operatorname{det}(X)=1$"
lie_algebra_constraints_SO_string = "$X^T B + BX = 0$ and $\operatorname{tr}(X)=0$"

def group_constraints_SO(matrix_to_test, form):
    # The requirements for a matrix X to be an element of the special orthogonal group with form matrix B are
    # 1) (X^T) * B * X = B
    # 2) det(X) = 1
    # The pinned_group class expects this information in the form of vanishing conditions,
    # so this method effectively returns the equations "X^T*B*X-B=0" and "det(X)-1=0"
    # However, the equations are split up matrix-entry-wise for various reasons
    X = matrix_to_test
    B = form.matrix
    M = X.T * B * X - B
    eqs = [e for e in M if not e.is_zero]
    d = X.det() - 1
    return eqs + ([] if d.is_zero else [d])

def lie_algebra_constraints_SO(matrix_to_test, form):
    # The requirements for a matrix X to be an element of the special orthogonal Lie algebra with form matrix B are
    # 1) (X^T) * B + B * X = 0
    # 2) tr(X) = 0
    # The pinned_group class expects this information in the form of vanishing conditions,
    # so this method effectively returns the equations "X^T*B + BX=0" and "tr(X)=0"
    # However, the equations are split up matrix-entry-wise for various reasons
    X = matrix_to_test
    M = X.T * form.matrix + form.matrix * X
    eqs = [e for e in M if not e.is_zero]
    t = X.trace()
    return eqs + ([] if t.is_zero else [t])
    
def is_torus_element_SO(matrix_to_test, rank):
    # The requirements for a matrix X to be an element of the diagonal torus subgroup of SO_n,q are
    # 1) X is diagonal
    # 2) det(X) = 1
    # 3) The first q diagonal entries are the inverses of the next q diagonal entries, respectively
    # 4) The last n-2*q entries are 1
    
    # In other words, X has the form
    # diag(t_1, ..., t_q, t_1^(-1), ..., t_q^(-1), 1, ..., 1)
    if not is_diagonal(matrix_to_test): return False
    X = matrix_to_test
    n = X.shape[0]
    q = rank
    return (
        all(X[i, i] * X[q + i, q + i] == 1 for i in range(q)) and
        all(X[2*q + j, 2*q + j] == 1 for j in range(n - 2*q))
    )

def generic_torus_element_SO(matrix_size, rank, letter = 't'):
    # Output a 'generic' element of the diagonal torus subgroup of 
    # the special orthogonal group, which has the form
    # diag(t_1, ..., t_q, t_1^(-1), ..., t_q^(-1), 1, ..., 1)
    # where q = rank
    n = matrix_size
    q = rank
    v = sp.symarray(letter, rank, nonzero=True)
    return sp.diag(*v, *(1/v), *([1] * (n - 2*q)))

def character_entries_SO(matrix_size, rank):
    # When computing the root system for a special orthogonal group,
    # this tells the solver to only use characters which have
    # nonzero entries in the first q=rank components.
    # This is possible because a torus element is determined by
    # its first q diagonal entries.
    return [1]*rank + [0]*(matrix_size - rank)

def trivial_characters_SO(matrix_size, rank):
    # Since elements of the diagonal subgroup have the form
    # diag(t_1, ..., t_q, t_1^(-1), ..., t_q^(-1), 1, ..., 1)
    # the character (1, 0, ..., 0, 1, 0, ...)
    # where the two 1's appear at positions 1 and q+1 
    # is the trivial group homomorphism on the torus
    # This encodes all of the trivial characters determined by the
    # dependence of the i and q+i entries of the torus,
    # as well as the character of all 1's arising from the det=1 condition
    trivial_characters = [np.array([1 if j == i or j == i + rank else 0 for j in range(matrix_size)])for i in range(rank)]
    if not (rank == 1 and matrix_size == 2): trivial_characters.append([1] * matrix_size)
    return np.array(np.stack(trivial_characters, axis=1))

def generic_lie_algebra_element_SO(matrix_size, rank, form, letter = 'x'):
    n = matrix_size
    q = rank
    m = n - 2*q
    assert form is not None
    c = form.anisotropic_vector
    assert len(c) == m
    X = sp.Matrix(sp.symarray(letter, (n,n)))
    
    # An element X of the special orthogonal Lie algebra is a block matrix of the form
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
            
    return X