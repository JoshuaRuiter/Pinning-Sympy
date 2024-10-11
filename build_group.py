from sympy import zeros, eye, prod, sqrt, symbols
from sympy.liealgebras.root_system import RootSystem
from pprint import pprint
from pinned_group import pinned_group
from matrix_utility import is_diagonal
from root_system_utility import root_sum, scale_root
from nondegenerate_isotropic_form import nondegenerate_isotropic_form
from numpy import shape
import math

def group_builder_tests():
    # Run some tests
    
    ###########################
    ## SPECIAL LINEAR GROUPS ##
    ###########################
    for n in (2,3,4):
        SL_n = build_special_linear_group(n)
        SL_n.run_tests()
        
    # Some split special orthogonal groups (n=2q or n=2q+1)
    for q in (2,3):
        for n in (2*q,2*q+1):
            SO_n_q = build_special_orthogonal_group(n,q)
            SO_n_q.display_root_spaces()
            SO_n_q.run_tests()
    
    # Some non-split special orthogonal groups 
        # SO_n_q is quasi-split if n=2q+2, and
        # neither split nor quasi-split if n>2+2q, 
        # but the behavior seems to be basically the same in these two cases
    for q in (1,2):
        for n in (2*q+2,2*q+3,2*q+4):
            SO_n_q = build_special_orthogonal_group(n,q)
            SO_n_q.run_tests()
    
    ###############################################
    ## SPECIAL UNITARY GROUPS NOT IMPLEMENTED YET
    ###############################################
    # Some quasi-split special unitary groups (n=2q)
    # for q in (2,3):
    #     n=2*q
    #     SU_n_q = build_special_unitary_group(n,q)
    #     SU_n_q.run_tests()
    
    ###############################################
    ## SPECIAL UNITARY GROUPS NOT IMPLEMENTED YET
    ###############################################
    # Some non-quasi-split special unitary groups (n>2q)
    # for q in (2,3):
    #     for n in (2*q+1,2*q+2):
    #         SU_n_q = build_special_unitary_group(n,q)
    #         SU_n_q.run_tests()
    
def build_special_linear_group(matrix_size):
    # Build a pinned_group object representing the special linear group SL_n
    
    name_string = "special linear group of size " + str(matrix_size)
    root_system_rank = matrix_size-1
    root_system = RootSystem("A"+str(root_system_rank))
    
    # Special linear groups don't have an associated bilinear or hermitian form,
    # so we use a nondegenerate_isotropic_form with nonsense values
    # dimension -1 and epsilon=2 as a placeholder object
    form = nondegenerate_isotropic_form(-1,0,100,0,0)
    
    def is_lie_algebra_element_SL(my_matrix,form):
        return my_matrix.trace()==0
    
    def is_group_element_SL(my_matrix,form):
        return my_matrix.det()==1
    
    def is_torus_element_SL(my_matrix,form):
        return (my_matrix.det()==1 and 
                is_diagonal(my_matrix))
    
    def root_space_dimension_SL(matrix_size,root_system,root):
        # All root spaces in SL_n have dimension 1
        return 1
    
    def root_space_map_SL(matrix_size,root_system,form,root,my_input):
        # Given a root alpha for the A_n root system,
        # output an element of the associated root space.
        # That is, output the matrix with my_input in
        # one entry, and otherwise zero entries.
        # The root should be a vector of length n,
        # with two nonzero entries, 1 and -1.
        assert(len(root)==matrix_size)
        assert(sum(root)==0)
        i = root.index(1)
        j = root.index(-1)
        my_output = zeros(matrix_size)
        my_output[i,j] = my_input
        return my_output

    def root_subgroup_map_SL(matrix_size,root_system,form,root,my_input):
        # Given a root alpha for the A_n root system,
        # output an element of the associated root subgroup.
        # That is, output the matrix with my_input in one off-diagonal
        # entry, 1's the diagonal, and zeros elsewhere.
        return eye(matrix_size) + root_space_map_SL(matrix_size,root_system,form,root,my_input)
    
    def torus_element_map_SL(matrix_size,root_system,my_vec):
        # Output a torus element of the usual diagonal subgroup of SL
        # using a vector of rank my_vec
        # the length of my_vec must match the root_system_rank
        root_system_rank = len(root_system.simple_roots())
        assert(len(my_vec) == root_system_rank)
        assert(matrix_size == root_system_rank+1)
        t = zeros(matrix_size)
        for i in range(root_system_rank):
            t[i,i] = my_vec[i]
        t[matrix_size-1,matrix_size-1] = 1/prod(my_vec)
        return t

    def commutator_coefficient_map_SL(matrix_size,root_system,form,alpha,beta,p,q,u,v):
        # Return the coefficient arising from taking the commutator of transvection matrices
        root_list = list(root_system.all_roots().values())
        if not(root_sum(alpha,beta) in root_list):
            return 0
        else:
            i = alpha.index(1)
            j = alpha.index(-1)
            k = beta.index(1)
            l = beta.index(-1)
            assert((j==k and i!=l) or 
                   (j!=k and i==l))
            if j==k:
                return u*v
            else: # so i==l
                return -u*v
            
    def weyl_group_element_map_SL(matrix_size,root_system,form,alpha,u):
        return (root_subgroup_map_SL(matrix_size,root_system,form,alpha,u)*
                root_subgroup_map_SL(matrix_size,root_system,form,scale_root(-1,alpha),-1/u)*
                root_subgroup_map_SL(matrix_size,root_system,form,alpha,u))
    
    def weyl_group_coefficient_map_SL(matrix_size,root_system,form,alpha,beta,v):
        i = alpha.index(1)
        # j = alpha.index(-1)
        k = beta.index(1)
        l = beta.index(-1)
        if i==k or i==l:
            return -v
        else:
            return v
    
    return pinned_group(name_string,
                        matrix_size,
                        form,
                        root_system,
                        is_lie_algebra_element_SL,
                        is_group_element_SL,
                        is_torus_element_SL,
                        root_space_dimension_SL,
                        root_space_map_SL,
                        root_subgroup_map_SL,
                        torus_element_map_SL,
                        commutator_coefficient_map_SL,
                        weyl_group_element_map_SL,
                        weyl_group_coefficient_map_SL)

def build_special_orthogonal_group(matrix_size, root_system_rank):
    # Build a pinned_group object representing a special orthogonal group SO_{n,q}(k,B)
    #   n is a positive integer, at least 4 to be interesting
    #       n is the size of the matrices
    #   k is a field
    #   B is the matrix of a nondegenerate symmetric bilinear form b of Witt index q
    #   SO_{n,q}(k,B) is the set of (n x n) matrices X with entries from k satisfying 
    #           (X^t)*B*X=B and det(X)=1 (where X^t is the transpose)
    #       Using the theory of bilinear forms we may assume that B is an (n x n) 
    #       block matrix of the form 
    #                   [0 I 0]
    #                   [I 0 0]
    #                   [0 0 C]
    #       where I is the (q x q) identity matrix and C is (n-2q x n-2q) and diagonal (and invertible). 
    #       For example, in the case n=4 and q=1, B has the form
    #                   [0 ,1 ,0 ,0 ]
    #                   [1 ,0 ,0 ,0 ]
    #                   [0 ,0 ,c1,0 ]
    #                   [0 ,0 ,0 ,c2]
    #       where c1, c2 are any nonzero elements of the field k.
    #
    #       In general, q<=n/2. If q=n/2 or q=(n-1)/2, then SO_n(k,B) is split and this is understood.
    #       If q=n/2-1, then SO_n(k,B) is quasi-split, this is an important case to understand.
    #       We ignore the case q=0, because in this case the group is not isotropic.
    
    name_string = ("special orthogonal group of size " + str(matrix_size) + 
                    " with Witt index " + str(root_system_rank))
    root_system = RootSystem("B"+str(root_system_rank))
    anisotropic_variables = symbols('c:'+str(matrix_size-2*root_system_rank))
    form = nondegenerate_isotropic_form(matrix_size,root_system_rank,0,anisotropic_variables,0)
    
    def is_lie_algebra_element_SO(my_matrix,form):
        X = my_matrix
        B = form.matrix
        
        print("\n")
        pprint(X)
        pprint(B)
        pprint(X.T)
        pprint(X.T*B)
        pprint(-X*B)
        
        return (X.T*B == -X*B)
        
    def is_group_element_SO(my_matrix,form):
        X = my_matrix
        B = form.matrix
        return (X.transpose * B * X == B)
    
    def is_torus_element_SO(matrix_size,root_system,matrix_to_test):
        root_system_rank = len(root_system.simple_roots())
        
        if shape(matrix_to_test != (matrix_size,matrix_size)):
            return False
        
        for i in range(root_system_rank):
            if (matrix_to_test[i,i]*matrix_to_test[root_system_rank+i,root_system_rank+i]!=1):
                return False
        for j in range(matrix_size - 2*root_system_rank):
            if (matrix_to_test[2*root_system_rank+j,2*root_system_rank+j]!=1):
                return False
        return True
            
    def root_space_dimension_SO(matrix_size,root_system,root):
        # The root system associated with a special orthogonal group SO_n_q is
        # type B. In type B, there are two kinds of roots: 
            # Long roots of the form (0,...,0,1,0,...,0,-1,0,...,0), and
            # Short roots of the form (0,...,0,1,0,...,0) or (0,...,0,-1,0,...,0)
        # The root space associated to a long root is 1-dimensional,
        # the root space associated to a short root has dimension n-2q,
        # where n is the size of the matrices and q is the Witt index
        my_sum = abs(sum(root))
        if my_sum == 0 or my_sum == 2:
            return 1
        elif my_sum == 1:
            root_system_rank = len(root_system.simple_roots())
            return matrix_size - 2*root_system_rank
        else:
            raise Exception('Unexpected root for special orthogonal group.')
        
    def root_space_map_SO(matrix_size, root_system, form, alpha, v):
        # Return a generic element of the root space in the special orthogogal Lie algebra
        # associated to the root alpha, with input v
        
        n = matrix_size
        q = len(root_system.simple_roots()) # The rank of the root system
        output_matrix = zeros(matrix_size)
        vec_C = form.anisotropic_vector
        
        assert(n >= 2*q)
        assert(len(v) == root_space_dimension_SO(n,root_system,alpha))
        # v must be a vector of length 1 if alpha is a long root,
        # and v must be a vectof of length n-2q if alpha is a short root
        
        sum_type = sum(alpha) 
            # 0,2, or -2 for long roots
            # 1 or -1 for short roots
        
        if sum_type == 0:
            # alpha is a long root, of the form
            # (0,..,0,1,0,...,0,-1,0,...,0)
            # In this case, v should just be a "scalar" (really a vector of length 1)
            assert(len(v) == 1)
            
            i = alpha.index(1)
            j = alpha.index(-1)
            output_matrix[i,j] = v[0]
            output_matrix[q+j,q+i] = -v[0]
            
        elif sum_type == 2:
            # alpha is a long root, of the form
            # (0,..,0,1,0,...,0,1,0,...,0)
            # In this case, v should just be a "scalar" (really a vector of length 1)
            assert(len(v) == 1)
            
            # This gets the locations of both 1's
            my_indices = [k for k,val in enumerate(alpha) if val==1]
            i = my_indices[0]
            j = my_indices[1]
            output_matrix[i,q+j] = v[0]
            output_matrix[j,q+i] = -v[0]
            
        elif sum_type == -2:
            # alpha is a long root, of the form
            # (0,..,0,-1,0,...,0,-1,0,...,0)
            # In this case, v should just be a "scalar" (really a vector of length 1)
            assert(len(v) == 1)
            
            # This gets the locations of both 1's
            my_indices = [k for k,val in enumerate(alpha) if val==-1]
            i = my_indices[0]
            j = my_indices[1]
            output_matrix[i+q,j] = v[0]
            output_matrix[j+q,i] = -v[0]
            
        elif sum_type == 1:
            # alpha is a short root, of the form
            # (0,...0,1,0,...,0)
            # In this case, v should be a vector of length n-2q
            assert(len(v) == n-2*q)
            
            i = alpha.index(1)
            for s in range(n-2*q):
                output_matrix[i,2*q+s] = -vec_C[s]*v[s]
                output_matrix[2*q+s,i] = v[s]
        
        elif sum_type == -1:
            # alpha is a short root, of the form
            # (0,...0,-1,0,...,0)
            # In this case, v should be a vector of length n-2q
            assert(len(v) == n-2*q)
        
            i = alpha.index(-1)
            for s in range(n-2*q):
                output_matrix[q+i,2*q+s] = -vec_C[s]*v[s]
                output_matrix[2*q+s,i] = v[s]
        
        else:
            raise Exception('Unexpected root in special orthogonal group')
            
        return output_matrix
        
    def root_subgroup_map_SO(matrix_size, root_system, form, alpha, v):
        # In general, this is just the matrix exponential of
        # root_space_map_SO of the same inputs
        # In this case, we know that the 3rd power is zero,
        # so we just go out to the second power
        X_v = root_space_map_SO(matrix_size, root_system, form, alpha, v)
        return eye(matrix_size) + X_v + 1/2*X_v**2
        
    def torus_element_map_SO(matrix_size, root_system, vec_t):
        root_system_rank = len(root_system.simple_roots())
        assert(root_system_rank == len(vec_t))
        my_matrix = zeros(matrix_size)
        for i in range(root_system_rank):
            my_matrix[i,i] = vec_t[i]
            my_matrix[root_system_rank+i,root_system_rank+i] = 1/vec_t[i]
        for j in range(matrix_size - 2*root_system_rank):
            my_matrix[2*root_system_rank + j,2*root_system_rank+j] = 1
        return my_matrix
        
    def commutator_coefficient_map_SO():
        # PLACEHOLDER
        x=0
        
    def weyl_group_element_map_SO():
        # PLACEHOLDER
        x=0
        
    def weyl_group_coefficient_map_SO():
        # PLACEHOLDER
        x=0
    
    return pinned_group(name_string,
                        matrix_size,
                        form,
                        root_system,
                        is_lie_algebra_element_SO,
                        is_group_element_SO,
                        is_torus_element_SO,
                        root_space_dimension_SO,
                        root_space_map_SO,
                        root_subgroup_map_SO,
                        torus_element_map_SO,
                        commutator_coefficient_map_SO,
                        weyl_group_element_map_SO,
                        weyl_group_coefficient_map_SO)

    def is_lie_algebra_element_SO(my_matrix):
        # check it is skew-symmetric; M.T should return the transpose of M
        # (Joshua) This is not quite right, needs to check that (X^t)*B*X=B and det(X)=1
        #           where B is the associated form matrix
        return my_matrix.T == -1*my_matrix

    def is_group_element_SO(my_matrix):
        return (my_matrix.T*my_matrix== eye(matrix_size)) and (my_matrix.det()==1)
    
def build_special_unitary_group(matrix_size,root_system_rank,epsilon):
    # Build a pinned_group object representing the special unitary group SU_{n,q}(k,H)
    #   n is a positive integer, at least 4 to be interesting
    #   k is a field
    #       L/k is a quadratic field extension with primitive element sqrt(d)
    #       The nontrivial Galois automorphism of L is an involution (which fixes k), 
    #       denoted sig or conj. It behaves very much like complex conjugation, 
    #       i.e. conj(sqrt(d)) = -sqrt(d)
    #   H is the matrix of a nondegenerate hermitian or skew-hermitian form of Witt index q
    #       (on an L-vector space V of dimension n)
    #       eps = 1 or -1, and determines skew-ness: H is hermitian if eps=1, skew-hermitian of eps=-1
    #   SU_{n,q}(k,H) is the set of (n x n) matrices X (with entries in L) 
    #       satsifying X^h*B*X=B and det(X)=1 (where X^h is the conjugate transpose, i.e. X^h = conj(X^t))
    #   By general theory of hermitian forms we may assume that H is an (n x n) block matrix of the form
    #                   [0      I       0]
    #                   [eps*I  0       0]
    #                   [0      0       C]
    #       where C is diagonal and satisfies conj(C)=eps*C. 
    #           i.e. f eps=1, then C has "purely real" entries from k, 
    #           and if eps=-1 then C has "purely imaginary" entries from k*sqrt(d)
    #
    #   In general, q<=n/2 because Witt index can never exceed half the dimension.
    #       If q=n/2 (n=2q), the group is quasi-split and studied in my thesis.
    #       So we are a bit more interested in the case where q<n/2 (equivalently n>2q)
    #       We ignore the case q=0, because in this case the group is not isotropic so it behaves quite differently.
    
    name_string = ("special unitary group of size " + str(matrix_size) + 
                    " with Witt index " + str(root_system_rank) + 
                    " and epsilon = " + str(epsilon))
    root_system = RootSystem("C"+str(root_system_rank))
    anisotropic_variables = symbols('c:'+str(matrix_size-2*root_system_rank))
    d = symbols('d')
    primitive_element = sqrt(d)
    form = nondegenerate_isotropic_form(matrix_size,root_system_rank,epsilon,
                                        anisotropic_variables,primitive_element)
    
    def is_lie_algebra_element_SU():
        # PLACEHOLDER
        x=0
        
    def is_group_element_SU():
        # PLACEHOLDER
        x=0
    
    def is_torus_element_SU():
        # PLACEHOLDER
        x=0
        
    def root_space_dimension_SU():
        # PLACEHOLDER
        x=0
        
    def root_space_map_SU():
        # PLACEHOLDER
        x=0
        
    def root_subgroup_map_SU():
        # PLACEHOLDER
        x=0
        
    def torus_element_map_SU():
        # PLACEHOLDER
        x=0
        
    def commutator_coefficient_map_SU():
        # PLACEHOLDER
        x=0
        
    def weyl_group_element_map_SU():
        # PLACEHOLDER
        x=0
        
    def weyl_group_coefficient_map_SU():
        # PLACEHOLDER
        x=0
        
    return pinned_group(name_string,
                        matrix_size,
                        form,
                        root_system,
                        is_lie_algebra_element_SU,
                        is_group_element_SU,
                        is_torus_element_SU,
                        root_space_dimension_SU,
                        root_space_map_SU,
                        root_subgroup_map_SU,
                        torus_element_map_SU,
                        commutator_coefficient_map_SU,
                        weyl_group_element_map_SU,
                        weyl_group_coefficient_map_SU)

if __name__ == "__main__":
    group_builder_tests()
