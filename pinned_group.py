from sympy import symbols, symarray, Matrix
from pprint import pprint
from matrix_utility import evaluate_character

class pinned_group:
    
    def __init__(self, 
                 name_string, 
                 matrix_size,
                 form_matrix,
                 root_system,
                 is_lie_algebra_element,
                 is_group_element,
                 is_torus_element,
                 root_space_dimension,
                 root_space_map,
                 root_subgroup_map,
                 torus_element_map):
        # Build a pinned group from scratch by providing all inputs
        
        self.name_string = name_string
        self.matrix_size = matrix_size
        self.form_matrix = form_matrix
        
        self.root_system = root_system
        self.root_system_rank = len(root_system.simple_roots())
        self.root_list = list(root_system.all_roots().values())
        
        self.is_lie_algebra_element = is_lie_algebra_element
        self.is_group_element = is_group_element
        self.is_torus_element = is_torus_element
    
        self.root_space_dimension = root_space_dimension
        self.root_space_map = root_space_map
        self.root_subgroup_map = root_subgroup_map
        self.torus_element_map = torus_element_map
                
        # WeylGroupMap
        # HomDefectCoefficientMap
        # CommutatorCoefficientMap
        # WeylGroupCoefficientMap
    
    def run_tests(self):
        print("\nRunning tests to verify a pinning of the " + self.name_string + "...")
        self.test_basics()
        self.test_root_space_maps_are_almost_homomorphisms()
        self.test_torus_conjugation()
        
        print("All tests complete.")
        
    def test_basics(self):
        print("\tRunning basic tests...")
    
        print("\t\tChecking root spaces belong to the Lie algebra...",end='')
        for root in self.root_list:
            u = symbols('u')
            X = self.root_space_map(self.matrix_size,self.root_system,self.form_matrix,root,u)
            assert(self.is_lie_algebra_element(X))
        print("done.")
        
        print("\t\tChecking root subgroups belong to the group...",end='')
        for root in self.root_list:
            u = symbols('u')
            X_u = self.root_subgroup_map(self.matrix_size,self.root_system,self.form_matrix,root,u)
            assert(self.is_group_element(X_u))   
        print("done.")
        
        print("\tBasic tests passed.")
        
    def test_root_space_maps_are_almost_homomorphisms(self):
        print("\tChecking root space spaces are (almost) homomorphisms...",end='')
        for root in self.root_list:
            u,v = symbols('u v ')
            X_u = self.root_space_map(self.matrix_size,self.root_system,self.form_matrix,root,u)
            X_v = self.root_space_map(self.matrix_size,self.root_system,self.form_matrix,root,v)
            X_u_plus_v = self.root_space_map(self.matrix_size,self.root_system,self.form_matrix,root,u+v)
            assert(X_u+X_v==X_u_plus_v)
        print("done.")
        
    def test_torus_conjugation(self):
        print("\tChecking torus conjugation formula...",end='')
        
        vec_t = Matrix(symarray('t',self.root_system_rank))
        t = self.torus_element_map(self.matrix_size,self.root_system_rank,vec_t)
        
        for alpha in self.root_list:
            alpha_of_t = evaluate_character(alpha,t)
            u = symbols('u')
            
            # Torus conjugation on the Lie algebra/root spaces
            X_u = self.root_space_map(self.matrix_size,self.root_system,self.form_matrix,alpha,u)
            LHS1 = t*X_u*t**(-1)
            RHS1 = self.root_space_map(self.matrix_size,self.root_system,self.form_matrix,alpha,alpha_of_t*u)
            assert(LHS1==RHS1)
            
            # Torus conjugation on the group/root subgroups
            x_u = self.root_subgroup_map(self.matrix_size,self.root_system,self.form_matrix,alpha,u)
            LHS2 = t*x_u*t**(-1)
            RHS2 = self.root_subgroup_map(self.matrix_size,self.root_system,self.form_matrix,alpha,alpha_of_t*u)

            assert(LHS2==RHS2)
        print("done.")