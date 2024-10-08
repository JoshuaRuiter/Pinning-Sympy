from sympy import zeros, eye, prod
from sympy.liealgebras.root_system import RootSystem
from pprint import pprint
from pinned_group import pinned_group
from matrix_utility import is_diagonal
from root_system_utility import root_sum, scale_root
import math

def group_builder_tests():
    # Run some tests
    
    for n in (2,3,4):
        SL_n = build_special_linear_group(n)
        SL_n.run_tests()

def build_special_linear_group(matrix_size):
    # Build a pinned_group object representing the special linear group SL_n
    
    name_string = "special linear group of size " + str(matrix_size)
    root_system_rank = matrix_size-1
    root_system = RootSystem("A"+str(root_system_rank))
    form_matrix = zeros(matrix_size) # Not needed for special linear groups
    
    def is_lie_algebra_element_SL(my_matrix):
        return my_matrix.trace()==0
    
    def is_group_element_SL(my_matrix):
        return my_matrix.det()==1
    
    def is_torus_element_SL(my_matrix):
        return (my_matrix.det()==1 and 
                is_diagonal(my_matrix))
    
    def root_space_dimension_SL(root):
        # All root spaces in SL_n have dimension 1
        return 1
    
    def root_space_map_SL(matrix_size,root_system,form_matrix,root,my_input):
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

    def root_subgroup_map_SL(matrix_size,root_system,form_matrix,root,my_input):
        # Given a root alpha for the A_n root system,
        # output an element of the associated root subgroup.
        # That is, output the matrix with my_input in one off-diagonal
        # entry, 1's the diagonal, and zeros elsewhere.
        return eye(matrix_size) + root_space_map_SL(matrix_size,root_system,form_matrix,root,my_input)
    
    def torus_element_map_SL(matrix_size,root_system_rank,my_vec):
        # Output a torus element of the usual diagonal subgroup of SL
        # using a vector of rank my_vec
        # the length of my_vec must match the root_system_rank
        assert(len(my_vec)==root_system_rank)
        t = zeros(root_system_rank+1)
        for i in range(root_system_rank):
            t[i,i] = my_vec[i]
        t[matrix_size-1,matrix_size-1] = 1/prod(my_vec)
        return t

    def commutator_coefficient_map_SL(matrix_size,root_system,form_matrix,alpha,beta,p,q,u,v):
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
            
    def weyl_group_element_map_SL(matrix_size,root_system,form_matrix,alpha,u):
        return (root_subgroup_map_SL(matrix_size,root_system,form_matrix,alpha,u)*
                root_subgroup_map_SL(matrix_size,root_system,form_matrix,scale_root(-1,alpha),-1/u)*
                root_subgroup_map_SL(matrix_size,root_system,form_matrix,alpha,u))
    
    def weyl_group_coefficient_map_SL(matrix_size,root_system,form_matrix,alpha,beta,v):
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
                        form_matrix,
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
    
    
def build_special_orthogonal_group(matrix_size):
    # Build a pinned_group object representing the special orthogonal group SO_n
    
    name_string = "special orthogonal group of size " + str(matrix_size)
    root_system_rank = math.floor(matrix_size/2)-1
    root_system = RootSystem("A"+str(root_system_rank))
    form_matrix = zeros(matrix_size)
    
    
def build_special_unitary_group(size):
    # INCOMPLETE
    x=0

if __name__ == "__main__":
    group_builder_tests()
