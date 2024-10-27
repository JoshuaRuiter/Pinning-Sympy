from sympy import symbols, symarray, Matrix, eye, zeros, simplify
import numpy as np
from pprint import pprint
from matrix_utility import evaluate_character

class pinned_group:
    
    def __init__(self, 
                 name_string, 
                 matrix_size,
                 form,
                 root_system,
                 is_lie_algebra_element,
                 is_group_element,
                 is_torus_element,
                 root_space_dimension,
                 root_space_map,
                 root_subgroup_map,
                 torus_element_map,
                 commutator_coefficient_map,
                 weyl_group_element_map,
                 weyl_group_coefficient_map):
        
        # Build a pinned group from scratch by providing all inputs
        
        self.name_string = name_string
        self.matrix_size = matrix_size
        self.form = form
        # self.form_matrix = self.form.matrix # I want to get rid of this,
        #                                     # but it requires changing a lot of
        #                                     # things in the tests below,
        #                                     # do this eventually
        #                                     # Basically, I think we should only pass self.form
        #                                     # as an argument, not the matrix itself
        
        self.root_system = root_system
        self.root_system_rank = root_system.rank
        self.root_list = root_system.root_list
        
        self.is_lie_algebra_element = is_lie_algebra_element
        self.is_group_element = is_group_element
        self.is_torus_element = is_torus_element
    
        self.root_space_dimension = root_space_dimension
        self.root_space_map = root_space_map
        self.root_subgroup_map = root_subgroup_map
        self.torus_element_map = torus_element_map
        
        self.commutator_coefficient_map = commutator_coefficient_map
        self.weyl_group_element_map = weyl_group_element_map
        self.weyl_group_coefficient_map = weyl_group_coefficient_map
        
        # HomDefectCoefficientMap
    
    def run_tests(self):
        print("\nRunning tests to verify a pinning of the " + self.name_string + "...")
        self.test_basics()
        self.test_root_space_maps_are_almost_homomorphisms()
        self.test_torus_conjugation()
        self.test_commutator_formula()
        self.test_weyl_group()   
        print("All tests complete.")
        
    def test_basics(self):
        print("\tRunning basic tests...")
    
        print("\t\tChecking root spaces belong to the Lie algebra...",end='')
        for root in self.root_list:
            dim = self.root_space_dimension(self.matrix_size,self.root_system,root)
            u = symarray('u',dim)
            X = self.root_space_map(self.matrix_size,self.root_system,self.form,root,u)
            assert(self.is_lie_algebra_element(X,self.form))
        print("done.")
        
        print("\t\tChecking root subgroups belong to the group...",end='')
        for root in self.root_list:
            dim = self.root_space_dimension(self.matrix_size,self.root_system,root)
            u = symarray('u',dim)
            X_u = self.root_subgroup_map(self.matrix_size,self.root_system,self.form,root,u)
            assert(self.is_group_element(X_u,self.form))   
        print("done.")

    def test_root_space_maps_are_almost_homomorphisms(self):
        print("\tChecking root space spaces are (almost) homomorphisms...",end='')
        for root in self.root_list:
            dim = self.root_space_dimension(self.matrix_size,self.root_system,root)
            u = symarray('u',dim)
            v = symarray('v',dim)
            X_u = self.root_space_map(self.matrix_size,self.root_system,self.form,root,u)
            X_v = self.root_space_map(self.matrix_size,self.root_system,self.form,root,v)
            X_u_plus_v = self.root_space_map(self.matrix_size,self.root_system,self.form,root,u+v)
            assert(X_u+X_v==X_u_plus_v)
        print("done.")
        
    def test_torus_conjugation(self):
        print("\tChecking torus conjugation formula...",end='')
        
        vec_t = Matrix(symarray('t',self.root_system_rank))
        t = self.torus_element_map(self.matrix_size,self.root_system,self.form,vec_t)
        
        for alpha in self.root_list:
            dim = self.root_space_dimension(self.matrix_size,self.root_system,alpha)
            u = symarray('u',dim)
            alpha_of_t = evaluate_character(alpha,t)
            
            # Torus conjugation on the Lie algebra/root spaces
            X_u = self.root_space_map(self.matrix_size,self.root_system,self.form,alpha,u)
            LHS1 = t*X_u*t**(-1)
            RHS1 = self.root_space_map(self.matrix_size,self.root_system,self.form,alpha,alpha_of_t*u)
            assert(LHS1==RHS1)
            
            # Torus conjugation on the group/root subgroups
            x_u = self.root_subgroup_map(self.matrix_size,self.root_system,self.form,alpha,u)
            LHS2 = t*x_u*t**(-1)
            RHS2 = self.root_subgroup_map(self.matrix_size,self.root_system,self.form,alpha,alpha_of_t*u)

            assert(LHS2==RHS2)
        print("done.")
        
    def test_commutator_formula(self):
        print("\tChecking commutator formula...",end='')

        for alpha in self.root_list:
            dim_V_alpha = self.root_space_dimension(self.matrix_size,self.root_system,alpha)
            u = symarray('u',dim_V_alpha)
            
            x_alpha_u = self.root_subgroup_map(self.matrix_size,self.root_system,self.form,alpha,u)
            for beta in self.root_list:
                # Commutator formula only applies when the two roots
                # not scalar multiples of each other
                                
                if not(self.root_system.is_proportional(alpha,beta)):
                    dim_V_beta = self.root_space_dimension(self.matrix_size,self.root_system,beta)
                    v = symarray('v',dim_V_beta)
                    x_beta_v = self.root_subgroup_map(self.matrix_size,self.root_system,self.form,beta,v)
                    LHS = x_alpha_u*x_beta_v*(x_alpha_u**(-1))*(x_beta_v**(-1))
                    
                    # This gets a list of all positive integer linear combinations of alpha and beta
                    # that are in the root system. 
                    # It is formatted as a dictionary where keys are tuples (i,j) and the value 
                    # associated to a key (i,j) is the root i*alpha+j*beta
                    linear_combos = self.root_system.integer_linear_combos(alpha,beta)
                    
                    # The right hand side is a product over positive integer lienar combinations of alpha and beta
                    # with coefficients depending on some function N(alpha,beta,i,j,u,v)
                    RHS = eye(self.matrix_size)
                    for key in linear_combos:
                        i = key[0]
                        j = key[1]
                        root = linear_combos[key]
                        
                        # Check that i*alpha+j*beta = root
                        assert(np.all(i*alpha+j*beta == root))

                        # Compute the commutator coefficient that should arise
                        N = self.commutator_coefficient_map(self.matrix_size,self.root_system,self.form,alpha,beta,i,j,u,v);
                        my_sum = alpha+beta
                        RHS = RHS * self.root_subgroup_map(self.matrix_size,self.root_system,self.form,my_sum,N)
                    assert(LHS==RHS)
            
        print("done.")
        
    def test_weyl_group(self):
        print("\tRunning tests related to Weyl group elements...")
        u,v = symbols('u v ')
        
        print("\t\tChecking Weyl group elements normalize the torus...",end='')
        vec_t = Matrix(symarray('t',self.root_system_rank))
        t = self.torus_element_map(self.matrix_size,self.root_system,self.form,vec_t)
        for alpha in self.root_list:
            w_alpha_u = self.weyl_group_element_map(self.matrix_size,self.root_system,self.form,alpha,u)
            conjugation = w_alpha_u * t * (w_alpha_u**(-1))
            assert(self.is_torus_element(conjugation,self.root_system,self.form))
        print("done.")
        
        print("\t\tChecking Weyl group conjugation formula...",end='')
        for alpha in self.root_list:
            w_alpha_1 = self.weyl_group_element_map(self.matrix_size,self.root_system,self.form,alpha,1)
            for beta in self.root_list:
                x_beta_v = self.root_subgroup_map(self.matrix_size,self.root_system,self.form,beta,v)
                LHS = w_alpha_1 * x_beta_v * (w_alpha_1**(-1))
                
                reflected_root = self.root_system.reflect_root(alpha,beta)
                weyl_group_coeff = self.weyl_group_coefficient_map(self.matrix_size,self.root_system,self.form,alpha,beta,v)
                RHS = self.root_subgroup_map(self.matrix_size,self.root_system,self.form,reflected_root,weyl_group_coeff)
                
                assert(simplify(LHS-RHS)==zeros(self.matrix_size))
                
        print("done.")
        
    def display_root_spaces(self):
        for alpha in self.root_list:            
            dim = self.root_space_dimension(self.matrix_size,self.root_system,alpha)
            u = symarray('u',dim)
            X_alpha_u = self.root_space_map(self.matrix_size,self.root_system,self.form,alpha,u)
            print("\nRoot: ")
            print(alpha)            
            print("\nGeneric element of root space: " )
            pprint(X_alpha_u)