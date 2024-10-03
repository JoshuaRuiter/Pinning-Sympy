from pinned_group import pinned_group
from sympy.liealgebras.root_system import RootSystem
from matrix_utility import is_diagonal

def group_builder_tests():
    # Run some tests
    
    SL_2 = build_special_linear_group(2)
    SL_2.run_tests()

def build_special_linear_group(matrix_size):
    # Build a pinned_group object representing the special linear group SL_n
    
    name_string = "special linear group of size " + str(matrix_size)
    root_system_rank = matrix_size-1
    root_system = RootSystem("A"+str(root_system_rank))
    
    def is_lie_algebra_element(my_matrix):
        return my_matrix.trace()==0
    
    def is_group_element(my_matrix):
        return my_matrix.det()==1
    
    def is_torus_element(my_matrix):
        return (my_matrix.det()==1 and 
                is_diagonal(my_matrix))
    
    def root_space_dimension(root):
        return 1
    
    return pinned_group(name_string,
                        matrix_size,
                        root_system,
                        is_lie_algebra_element,
                        is_group_element,
                        is_torus_element,
                        root_space_dimension)
    
    
def build_special_orthogonal_group(size):
    # INCOMPLETE
    x=0
    
def build_special_unitary_group(size):
    # INCOMPLETE
    x=0

if __name__ == "__main__":
    group_builder_tests()