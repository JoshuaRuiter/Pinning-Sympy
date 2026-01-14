# pinned_group is a custom class for storing all the data of a pinned algebraic group.

# There are two broad categories of features:
#   1. Computational tools
#   2. Verification tools

# The computational tools take the bare bones needed to describe a matrix group, 
# then work out all the details of the root system, Lie algebra, root spaces,
# root subgroups, Weyl group elements, homomorphism defect coefficients, 
# commutator coefficients, and Weyl group conjugation coefficients.
# These details are then stored internally in the group object.

# The verification tools take the fitted data above and run a battery of tests
# to verify all the properties of a pinning.

import sympy as sp
import numpy as np
from utility_general import (determine_roots,
                             generate_character_list, 
                             parse_root_pairs, 
                             reduce_character_list,
                             vector_variable,
                             evaluate_character)
from root_system import root_system
from utility_roots import visualize_graph

class pinned_group:
    
    def __init__(self,
                 name_string,
                 matrix_size,
                 rank,
                 form,
                 is_group_element,
                 is_torus_element,
                 generic_torus_element,
                 trivial_characters,
                 is_lie_algebra_element,
                 generic_lie_algebra_element,
                 non_variables = None):
        
        self.name_string = name_string
        self.matrix_size = matrix_size
        self.rank = rank
        self.form = form
        self.is_group_element = is_group_element
        self.is_torus_element = is_torus_element
        self.generic_torus_element = generic_torus_element
        self.trivial_characters = trivial_characters
        self.is_lie_algebra_element = is_lie_algebra_element
        self.generic_lie_algebra_element = generic_lie_algebra_element
        self.non_variables = non_variables
                
        # These get set by running .fit_pinning()
        self.root_system = None
        self.root_space_dimension = None
        self.root_space_map = None
        self.root_subgroup_map = None
        self.homomorphism_defect_map = None
        self.commutator_coefficient_map = None
        self.weyl_element_map = None
        self.weyl_conjugation_coefficient_map = None
        
    def fit_pinning(self, display):
        # work out all the computational details related to Lie algebra, roots, etc.
        
        ###############################################################
        # TEMPORARILTY DISABLED
        ###############################################################
        # if display:
        #     print('\n' + '-' * 100)
        #     print(f"Fitting a pinning for {self.name_string}")
        #     if self.form is not None:
        #         print("\nForm matrix:")
        #         sp.pprint(self.form.matrix)
                
        #     x = self.generic_lie_algebra_element(matrix_size = self.matrix_size,
        #                                          rank = self.rank,
        #                                          form = self.form,
        #                                          letter = 'x')
        #     print("\nGeneric Lie algebra element:")            
        #     sp.pprint(x)
                
        #     t = self.generic_torus_element(matrix_size = self.matrix_size,
        #                                    rank = self.rank,
        #                                    form = self.form,
        #                                    letter = 't')
        #     print("\nGeneric torus element:")
        #     sp.pprint(t)
            
        #     print("\nTrivial characters:")
        #     for c in self.trivial_characters:
        #         sp.pprint(c)
        
        self.fit_root_system(display)
        self.fit_root_spaces(display)
        self.fit_root_subgroup_maps(display)
        self.fit_homomorphism_defect_coefficients(display)
        self.fit_commutator_coefficients(display)
        self.fit_weyl_group(display)
        
        if display: print(f"Pinning fitting for {self.name_string} is complete.")
        print('-' * 100)

    def fit_root_system(self, display = True):
        t = self.generic_torus_element(matrix_size = self.matrix_size,
                                       rank = self.rank,
                                       form = self.form,
                                       letter = 't')
        x = self.generic_lie_algebra_element(matrix_size = self.matrix_size,
                                             rank = self.rank,
                                             form = self.form,
                                             letter = 'x')
        x_vars = x.free_symbols
        if self.non_variables is not None: x_vars = x_vars - self.non_variables
        full_char_list = generate_character_list(character_length = self.matrix_size,
                                            upper_bound = 2,  
                                            padded_zeros = self.matrix_size - 2*self.rank)
        reduced_char_list = reduce_character_list(vector_list = full_char_list,
                                                  quotient_vectors = self.trivial_characters)
        
        if display:
            print("\nCandidate characters:",len(full_char_list))                
            print("Candidates after quotienting by trivial characters:",len(reduced_char_list))
                
        self.roots_and_root_spaces = determine_roots(generic_torus_element = t,
                                                generic_lie_algebra_element = x,
                                                list_of_characters = reduced_char_list,
                                                variables_to_solve_for = x_vars)
        root_list = parse_root_pairs(root_info = self.roots_and_root_spaces,
                                     what_to_get = 'roots')
        self.root_system = root_system(root_list)
        
        if display:
            print("Root system:",self.root_system.name_string)
            print("Number of roots:",len(root_list))
            if self.root_system.is_irreducible:
                print("Dynkin diagram:",visualize_graph(self.root_system.dynkin_graph))
    
    def fit_root_spaces(self, display = True):
        
        #######################################################
        # TO DO
        #######################################################
        # I want to convert this into a dictionary, so that 
        # root_space_dim can be a dictionary lookup,
        # rather than a O(n) linear search
        self.root_space_dimension_list = []
        for r, x in self.roots_and_root_spaces:
            x_vars = x.free_symbols
            if self.non_variables is not None:
                x_vars = x_vars - self.non_variables
            if self.form is not None:
                if self.form.primitive_element is not None: 
                    x_vars = x_vars - {self.form.primitive_element}
                if self.form.anisotropic_vector is not None:
                    x_vars = x_vars - self.form.anisotropic_vector.free_symbols
            self.root_space_dimension_list.append([r, len(x_vars)])

        #######################################################
        # TO DO
        #######################################################
        # I want to convert root_space_dimension_list 
        # into a dictionary, so that this can be a 
        # dictionary key lookup, rather than a linear
        # list search
        def root_space_dim(root):
            for r, d in self.root_space_dimension_list:
                if np.array_equal(r, root):
                    return d
        self.root_space_dimension = root_space_dim
        
        def root_sp_map(root, u):
            assert(self.root_system.is_root(root))
            dim = self.root_space_dimension(root)
            assert(len(u) == dim)
            for r, x in self.roots_and_root_spaces:
                if np.array_equal(r, root):
                    x_vars = x.free_symbols
                    if self.non_variables is not None: 
                        x_vars = x_vars - self.non_variables
                    if self.form is not None:
                        if self.form.primitive_element is not None: 
                            x_vars = x_vars - {self.form.primitive_element}
                        if self.form.anisotropic_vector is not None: 
                            x_vars = x_vars - self.form.anisotropic_vector.free_symbols
                    assert(len(x_vars) == len(u))
                    x_vars = list(x_vars)
                    for i in range(dim):
                        x = x.subs(x_vars[i],u[i])
                    return sp.simplify(x)
        self.root_space_map = root_sp_map
    
    def fit_root_subgroup_maps(self, display = True):
        def root_subgp_map(root, u):
            assert(self.root_system.is_root(root))
            dim = self.root_space_dimension(root)
            assert(len(u) == dim)
            x = self.root_space_map(root, u)
            if any(x**3): raise ValueError("Unexpected: The 3rd power of a root space element is a nonzero matrix.")
            return sp.simplify(np.eye(self.matrix_size, dtype = int) + x + x*x/2)
        self.root_subgroup_map = root_subgp_map
        
        
        #########################################################
        # TEMPORARILY DISABLED
        #########################################################
        # if display:
        #     print(f"\nRoots, root spaces, and root subgroups for {self.name_string}:")
        #     for r, x in self.roots_and_root_spaces:
        #         dim = self.root_space_dimension(r)
        #         print("\tRoot:",r)
        #         print("\tRoot space dimension:",dim)
        #         print("\tRoot space:")
        #         sp.pprint(x)
        #         print("\tRoot subgroup:")
        #         u = vector_variable(letter = 'u', length = dim)
        #         X_alpha_u = self.root_subgroup_map(r, u)
        #         sp.pprint(X_alpha_u)
        #         print()
        
    def fit_homomorphism_defect_coefficients(self, display = True):
        # For multipliable roots, the root subgroup maps are not quite homomorphisms
        # Instead, there is a general formula
        # X_alpha(u)*X_alpha(v) = X_alpha(u+v) * product over i>1 of X_i*alpha(some function of u and v)
        # The general product is kinda dumb because I am pretty sure i can only be 2
        # That function of u and v is what I call the "homomorphism defect coefficient"
        
        self.homomorphism_defect_coefficient_list = []
        for alpha in self.root_system.root_list:
            if self.root_system.is_multipliable_root(alpha):
                    d_alpha = self.root_space_dimension(alpha)
                    d_2alpha = self.root_space_dimension(2*alpha)
                    u = vector_variable(letter = 'u', length = d_alpha)
                    v = vector_variable(letter = 'v', length = d_alpha)
                    w = vector_variable(letter = 'w', length = d_2alpha)
                    X_alpha_u = self.root_subgroup_map(alpha, u)
                    X_alpha_v = self.root_subgroup_map(alpha, v)
                    X_2alpha_w = self.root_subgroup_map(2*alpha, w)
                    X_alpha_sum = self.root_subgroup_map(alpha, u+v)
                    vanishing_expression = sp.simplify(X_alpha_u * X_alpha_v - X_alpha_sum * X_2alpha_w)
                    variables_to_solve_for = w.free_symbols
                    solutions_list = sp.solve(vanishing_expression,variables_to_solve_for,dict=True)
                    assert(len(solutions_list) == 1)
                    solutions_dict = solutions_list[0]
                    solutions_dict_keys = list(solutions_dict.values())
                    assert(len(solutions_dict_keys) == 1)
                    w_solution = solutions_dict_keys[0]
                    self.homomorphism_defect_coefficient_list.append([alpha, u, v, w_solution])
                    
        if display:
            if len(self.homomorphism_defect_coefficient_list) == 0:
                print("\nNo multipliable roots, so no homomorphism defect coefficients.")
            else:
                print(f"\nHomomorphism defect coefficients for {self.name_string}:")
                for c in self.homomorphism_defect_coefficient_list:
                    print("\nRoot:",c[0])
                    print("Homomorphism defect coefficient:",c[-1])

        def hom_defect_coeff(root, u, v):
            assert(self.root_system.is_root(root))
            if not self.root_system.is_multipliable_root(root):
                return [1]
            else:
                for alpha, u_prime, v_prime, w_solution in self.homomorphism_defect_coefficient_list:
                    if np.array_equal(alpha, root):
                        # Found the right root
                        hdc = w_solution
                        assert(len(u) == len(u_prime))
                        assert(len(v) == len(v_prime))
                        for i in range(len(u)):
                            hdc = hdc.subs(u_prime[i], u[i])
                        for j in range(len(v)):
                            hdc = hdc.subs(v_prime[j], v[j])
                        return [hdc]
                    
        self.homomorphism_defect_map = hom_defect_coeff
        
    def fit_commutator_coefficients(self, display = True):
        
        self.commutator_coefficient_list = []
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            u = vector_variable(letter = 'u', length = d_alpha)
            x_alpha_u = self.root_subgroup_map(alpha, u)
            for beta in self.root_system.root_list:
                if  self.root_system.is_root(alpha + beta) and not(self.root_system.is_proportional(alpha,beta)):
                    # compute the commutator coefficient and store it in a big list
                    
                    # compute the true commutator / left hand side of the equation
                    d_beta = self.root_space_dimension(beta)
                    v = vector_variable(letter = 'v', length = d_beta)
                    x_beta_v = self.root_subgroup_map(beta, v)
                    LHS = x_alpha_u*x_beta_v*(x_alpha_u**(-1))*(x_beta_v**(-1))
                    
                    # This gets a list of all positive integer linear combinations of
                    # alpha and beta that are in the root system. 
                    # It is formatted as a dictionary where keys are tuples (i,j) and the value 
                    # associated to a key (i,j) is the root i*alpha+j*beta
                    linear_combos = self.root_system.integer_linear_combos(alpha,beta)
                    
                    # The right hand side of the commutator formula is a product over 
                    # positive integer linear combinations of alpha and beta
                    # with coefficients depending on some function N(alpha,beta,i,j,u,v)
                    RHS = np.eye(self.matrix_size, dtype = int)
                    vars_to_solve_for = set()
                    new_coeff_lines = []
                    
                    # Assemble the right hand side of the commutator formula,
                    # which is the product over all (i,j) pairs of positive integers
                    # such that i*alpha+j*beta is a root, of factors 
                    # X_(i*alpha+j*beta) ( N_ij(u,v) )
                    # where N_ij is some function of u and v, called the commutator coefficient
                    # Note that N_ij depends crucially on alpha and beta as well
                    for key in linear_combos:
                        i = key[0]
                        j = key[1]
                        root = linear_combos[key]
                        assert(np.array_equal(i*alpha + j*beta, root))
                        d_ij = self.root_space_dimension(root)
                        N_ij = vector_variable(letter = 'N' + str(i) + str(j), length = d_ij)
                        vars_to_solve_for = vars_to_solve_for.union(set(N_ij))
                        factor_ij = self.root_subgroup_map(root, N_ij)
                        RHS = RHS * factor_ij
                        new_coeff_lines.append([alpha, beta, i, j, u, v, N_ij])

                    # Now solve for all of the N_ij
                    vanishing_expression = sp.simplify(LHS - RHS)
                    solutions_list = sp.solve(vanishing_expression, vars_to_solve_for, dict=True)
                    solutions_dict = solutions_list[0]
                    assert(len(solutions_list) == 1)
                    assert(len(solutions_dict) == len(vars_to_solve_for))
                    
                    # Extract the solutions and store them in new_coeff_lines
                    # which is then transferred to self.commutator_coefficient_list
                    for index, line in enumerate(new_coeff_lines):
                        coeff_vec = line[-1] # extract the coefficients from the last entry of the new line
                        for var in vars_to_solve_for:
                            coeff_vec = coeff_vec.subs(var, solutions_dict[var])
                        line[-1] = coeff_vec # replace with the commutator coefficients that we solved for
                        self.commutator_coefficient_list.append(line)
                    
                    
        def ccm(alpha, beta, i, j, u, v):
            if not self.root_system.is_root(alpha + beta):
                return [0]
            elif self.root_system.is_proportional(alpha, beta):
                raise ValueError("Commutator formula does not apply to proportional root pairs.")
            else:
                # look up the coefficient in the big stored list
                for (alpha_prime, beta_prime, i_prime, j_prime, 
                     u_prime, v_prime, coeff) in self.commutator_coefficient_list:
                    if (np.array_equal(alpha, alpha_prime) and
                        np.array_equal(beta, beta_prime) and
                        i == i_prime and
                        j == j_prime):
                            assert(len(u) == len(u_prime))
                            assert(len(v) == len(v_prime))
                            for i in range(len(u)):
                                coeff = coeff.subs(u_prime[i], u[i])
                            for j in range(len(v)):
                                coeff = coeff.subs(v_prime[j], v[j])
                            return coeff
                raise ValueError("Commutator coefficient should exist, but could not be located.")
                        
        self.commutator_coefficient_map = ccm
        
        #########################################################
        # TEMPORARILY DISABLED
        #########################################################
        # if display:
        #     if len(self.commutator_coefficient_list) == 0:
        #         print("\nNo pairs of summable roots, so no commutator coefficients.")
        #     else:
        #         print(f"\nCommutator coefficients for {self.name_string}:\n")
        #         for root1 in self.root_system.root_list:
        #             d_1 = self.root_space_dimension(root1)
        #             u = vector_variable(letter = 'u', length = d_1)
        #             for root2 in self.root_system.root_list:
        #                 if self.root_system.is_root(root1 + root2) and not self.root_system.is_proportional(root1, root2):
        #                     d_2 = self.root_space_dimension(root2)
        #                     v = vector_variable(letter = 'v', length = d_2)
        #                     linear_combos = self.root_system.integer_linear_combos(root1,root2)
        #                     print("\tRoot 1:",root1)
        #                     print("\tRoot 2:",root2)
        #                     for key in linear_combos:
        #                         i = key[0]
        #                         j = key[1]
        #                         root = linear_combos[key]
        #                         assert(np.array_equal(i*root1 + j*root2, root))
        #                         coeff = self.commutator_coefficient_map(root1, root2, i, j, u, v)
        #                         print("\t\t(i,j):",(i,j))
        #                         print("\t\ti*(Root 1) + j*(Root 2):",root)
        #                         print("\t\tCommutator coefficient:",coeff,"\n")
                                
    def fit_weyl_group(self, display = True):
        # TO DO:
        # fit weyl group elements
        # fit weyl group conjugation coefficients
        
        self.weyl_element_list = {}
        # TO DO
        # Populate a dictionary of Weyl elemennts
        # Keys should be roots (as tuples?)
        # Values should be group elements (matrices)
        for alpha in self.root_system.root_list:
            
            d_alpha = self.root_space_dimension(alpha)
            u = vector_variable(letter = 'u', length = d_alpha)
            x_alpha_u = self.root_subgroup_map(alpha, u)
            
            ##########################
            print()
            print("-" * 50)
            print("alpha =",alpha) 
            print("\nx_alpha_u =")
            sp.pprint(x_alpha_u)
            ##########################
            
            # I want an element w_alpha which belongs to the subgroup
            # generated by elements of the form x_{i*alpha}(u) where
            # i = +/-1 (and potentially +/-2 in the case that alpha is a multipliable root)
            # and w_alpha should have the following properties:
            #   -w_alpha should be in the group, obviously
            #   -more specifically, w_alpha should be in the subgroup generated by
            #       elemens x_alpha(u), x_{-alpha}(u), (and potentially x_{+/-2alpha}(u))
            #   -normalize the torus
            #   -square to elemenets that centralize the torus
            #   -for any other root beta, conjugating x_beta(u) by w_alpha should 
            #       end up with x_{s_alpha(beta)} ( ??? )
            #       where s_alpha(beta) is the reflection of beta 
            #       across the hyperplane perpendicular to alpha
        
            # Maybe the right approach is to use the conjugation formula to get
            # a bunch of equations (one for each root beta)
            # and solve those?
            
            # Initialize a fully symbolic (n by n) matrix
            w_alpha = sp.Matrix(sp.symarray('w', (self.matrix_size, self.matrix_size)))
            det = sp.det(w_alpha)
            vars_to_solve_for = w_alpha.free_symbols

            # Make an equation for each other root beta, 
            # coming from conjugating x_beta(vector of 1's)
            vanishing_conditions = [det - 1]
            i = 0
            for beta in self.root_system.root_list:
                i = i + 1
                d_beta = self.root_space_dimension(beta)
                u = vector_variable(letter = 'u' + str(i), length = d_beta)
                x_beta_u = self.root_subgroup_map(beta, u)
                
                gamma = self.root_system.reflect_root(alpha, beta)
                d_gamma = self.root_space_dimension(gamma)
                assert(d_beta == d_gamma)
                
                #c_i = sp.symbols('c' + str(i))
                # v_gamma = vector_variable(letter = 'v' + str(i), length = d_gamma)
                #x_gamma_v = self.root_subgroup_map(gamma, v_gamma)
                #x_gamma_cu = self.root_subgroup_map(gamma, c_i * u)
                x_gamma_cu = self.root_subgroup_map(gamma, -u)
                
                #vars_to_solve_for = vars_to_solve_for.union({c_i})
                
                LHS = sp.simplify(w_alpha * x_beta_u)
                #RHS = sp.simplify(x_gamma_v * w_alpha)
                RHS = sp.simplify(x_gamma_cu * w_alpha)
                
                condition = sp.simplify(LHS-RHS)
                vanishing_conditions.append(condition)
                
                #########################################
                print("-" * 40)
                print("\nbeta =",beta)
                #print("\nx_beta_one =")
                #sp.pprint(x_beta_one)
                print("\nx_beta_u = ")
                sp.pprint(x_beta_u)
                print("\ngamma (reflected root) =",gamma)
                print("\nx_gamma_cu =")
                sp.pprint(x_gamma_cu)
                #print("\nx_gamma_v =")
                #sp.pprint(x_gamma_v)

                #print("\nRather than solve w_alpha * x_beta_u * w_alpha^(-1) = x_gamma_v")
                #print("we will solve w_alpha * x_beta_u = x_gamma_v * w_alpha")
                #print("which should be faster since it doesn't involve a matrix inverse")
                
                print("\nw_alpha * x_beta_u =")
                sp.pprint(LHS)
                print("\nx_gamma_cu * w_alpha =")
                sp.pprint(RHS)
                #########################################
            
            print("-" * 40)
            print("\nw_alpha =")
            sp.pprint(w_alpha)
            i = 0
            for c in vanishing_conditions:
                i = i + 1
                print(f"\nVanishing condition {i}:")
                sp.pprint(c)
            
            # I have no idea what to expect from this solve
            solutions_list = sp.solve(vanishing_conditions,vars_to_solve_for,dict=True)
            
            ##########################
            print("\nSolutions to the combined set of equations:")
            sp.pprint(solutions_list)
            print("-" * 50)
            ##########################
            
            # Do some kind of substitution for w_alpha based on the solutions
            # INCOMPLETE
            
            # Store the new weyl element
            self.weyl_element_list[tuple(alpha)] = w_alpha
        
        def wem(alpha):
            return self.weyl_element_list[tuple(alpha)]
        
        self.weyl_element_map = wem
        
        if display:
            print("\nWeyl elements and Weyl conjugation coefficients:\n")
            for alpha in self.root_system.root_list:
                w_alpha = self.weyl_element_map(alpha)
                print("Root:",alpha)
                print("Weyl element:")
                sp.pprint(w_alpha)
                print()
        
    def validate_pinning(self, display = True):
        # run tests to verify the pinning, run only after fitting
        if display:
            print()
            print('=' * 100)
            print(f"Running tests to validate pinning of {self.name_string}.")
            
        #########################################################
        # TEMPORARILY DISABLED
        # #########################################################
        # self.validate_basics(display)
        # self.root_system.verify_root_system_axioms(display)
        # self.validate_root_space_maps(display)
        # self.validate_root_subgroup_maps(display)
        # self.validate_torus_conjugation_formula(display)
        # self.validate_commutator_formula(display)
        self.validate_weyl_group_properties(display)
        
        if display:
            print()
            print(f"Pinning validation tests for {self.name_string} complete.")
            print('=' * 100) 
        
    def validate_basics(self, display = True):
        # test that the generic torus element is in the group
        if display: print("\nChecking that a generic torus element is in the group... ", end="")
        s = self.generic_torus_element(matrix_size = self.matrix_size,
                                       rank = self.rank,
                                       form = self.form,
                                       letter = 's')
        assert(self.is_group_element(matrix_to_test = s, form = self.form))
        if display: print("done.")
        
        # tests for is_in_lie_algebra
        if display: print("Checking that a generic Lie algebra element is in the Lie algebra... ", end="")
        A = self.generic_lie_algebra_element(matrix_size = self.matrix_size, 
                                                     rank = self.rank, 
                                                     form = self.form,
                                                     letter = 'a')
        assert(self.is_lie_algebra_element(matrix_to_test = A,form = self.form))        
        if display: print("done.")        
        
    def validate_root_space_maps(self, display = True):
        print("\nVerifying properties of root spaces...")    
        # verify that root spaces belong to the Lie algebra
        if display: print("\tChecking that root spaces are in the Lie algebra... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            X_alpha_a = self.root_space_map(alpha, a)
            assert(self.is_lie_algebra_element(X_alpha_a, self.form))
        if display: print("done.")
        
        if display: print("\tChecking that roots of equal norm have same dimensional root spaces...",end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            for beta in self.root_system.root_list:
                d_beta = self.root_space_dimension(beta)
                if np.dot(alpha, alpha) == np.dot(beta,beta):
                    assert(d_alpha == d_beta)
        if display: print("done.")
        
        if display: print("\tChecking that root space maps are additive... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            b = vector_variable(letter = 'b', length = d_alpha)
            X_alpha_a = self.root_space_map(alpha,a)
            X_alpha_b = self.root_space_map(alpha,b)
            LHS = X_alpha_a + X_alpha_b
            RHS = self.root_space_map(alpha,a+b)
            assert(LHS.equals(RHS))
        if display: print("done.")
        
        if display: print("Finished verifying properties of root spaces.")
        
    def validate_root_subgroup_maps(self, display = True):
        if display: print("\nVerifying properties of root subgroup maps...")
        
        if display: print("\tChecking that root subgroup maps are in the group... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            X_alpha_a = self.root_subgroup_map(alpha, a)
            assert(self.is_group_element(X_alpha_a, self.form))
        if display: print("done")
    
        if display: print("\tChecking that root subgroup maps are pseudo-homomorphisms... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            b = vector_variable(letter = 'b', length = d_alpha)
            X_alpha_a = self.root_subgroup_map(alpha,a)
            X_alpha_b = self.root_subgroup_map(alpha,b)
            X_alpha_sum = self.root_subgroup_map(alpha,a+b)
            LHS = X_alpha_a * X_alpha_b
            extra_factor = np.eye(self.matrix_size, dtype = int)
            if self.root_system.is_root(2*alpha):
                coeff = self.homomorphism_defect_map(alpha, a, b)
                assert(len(coeff) == self.root_space_dimension(2*alpha))
                extra_factor = self.root_subgroup_map(2*alpha, coeff)
            if not self.root_system.is_multipliable_root(alpha):
                assert(np.array_equal(extra_factor, np.eye(self.matrix_size, dtype = int)))
            RHS = X_alpha_sum * extra_factor
            assert(LHS.equals(RHS))
        if display: print("done.")
    
        if display: print("Root subgroup map verifications complete.")
    
    def validate_torus_conjugation_formula(self, display = True):
        if display: print("\nChecking torus conjugation formulas... ", end="")
        s = self.generic_torus_element(matrix_size = self.matrix_size,
                                       rank = self.rank,
                                       form = self.form,
                                       letter = 's')
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            alpha_of_s = evaluate_character(alpha,s)
            
            # Torus conjugation on the Lie algebra/root spaces
            X_alpha_a = self.root_space_map(alpha,a)
            LHS1 = s*X_alpha_a*s**(-1)
            RHS1 = self.root_space_map(alpha,alpha_of_s*a)
            assert(LHS1.equals(RHS1))
            
            # Torus conjugation on the group/root subgroups
            x_alpha_a = self.root_subgroup_map(alpha, a)
            LHS2 = s*x_alpha_a*s**(-1)
            RHS2 = self.root_subgroup_map(alpha,alpha_of_s*a)
            assert(LHS2.equals(RHS2))
        if display: print("done.")
    
    def validate_commutator_formula(self, display = True):
        if display: print("\nVerifying commutator properties...")
        if display: print("\tChecking commutator formulas... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            x_alpha_a = self.root_subgroup_map(alpha, a)
            for beta in self.root_system.root_list:
                # Commutator formula only applies when the two roots are not scalar multiples
                if not(self.root_system.is_proportional(alpha,beta)):
                    d_beta = self.root_space_dimension(beta)
                    b = vector_variable(letter = 'b', length = d_beta)
                    x_beta_b = self.root_subgroup_map(beta, b)
                    LHS = x_alpha_a*x_beta_b*(x_alpha_a**(-1))*(x_beta_b**(-1))
                    
                    # This gets a list of all positive integer linear combinations of alpha and beta
                    # that are in the root system. 
                    # It is formatted as a dictionary where keys are tuples (i,j) and the value 
                    # associated to a key (i,j) is the root i*alpha+j*beta
                    linear_combos = self.root_system.integer_linear_combos(alpha,beta)
                    
                    # The right hand side of the commutator formula is a product over 
                    # positive integer linear combinations of alpha and beta
                    # with coefficients depending on some function N(alpha,beta,i,j,u,v)
                    RHS = np.eye(self.matrix_size, dtype = int)
                    for key in linear_combos:
                        i = key[0]
                        j = key[1]
                        root = linear_combos[key]
                        
                        # Check that i*alpha+j*beta = root
                        assert(np.array_equal(i*alpha + j*beta, root))
                        assert(np.all(i*alpha+j*beta == root))

                        # Compute the commutator coefficient that should arise,
                        # then multiply by the new factor
                        coeff = self.commutator_coefficient_map(alpha,beta,i,j,a,b)
                        new_factor = self.root_subgroup_map(root, coeff)
                        RHS = RHS * new_factor
                        
                    assert(LHS.equals(RHS))
        if display: print("done.")

        if display: print("\tVerifying that swapping order of roots negates the coefficient...", end = "")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            for beta in self.root_system.root_list:
                d_beta = self.root_space_dimension(beta)
                b = vector_variable(letter = 'b', length = d_beta)
                if self.root_system.is_root(alpha + beta) and not self.root_system.is_proportional(alpha,beta):
                    linear_combos = self.root_system.integer_linear_combos(alpha,beta)#############
                    for key in linear_combos:
                        i = key[0]
                        j = key[1]
                        coeff_1 = self.commutator_coefficient_map(alpha,beta,i,j,a,b)
                        coeff_2 = self.commutator_coefficient_map(beta,alpha,j,i,b,a)
                        assert(coeff_1 == -coeff_2)
        if display: print("done.")
        
        if display: print("Done verifying properties of commutators.")
        
    def validate_weyl_group_properties(self, display = True):
        if display: print("\nVerifying properties of the Weyl group...")
        
        if display: print("\tChecking Weyl elements are in the group... ", end="")
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            assert(self.is_group_element(w_alpha, self.form))
        if display: print("done.")
        
        if display: print("\tChecking Weyl elements belong to the appropriate generated subgroup... ", end="")
        # INCOMPLETE
        # I don't know how to check this, probably too hard
        if display: print("TEST NOT IMPLEMENTED")
        
        if display: print("\tChecking Weyl elements normalize the torus... ", end="")
        t = self.generic_torus_element(matrix_size = self.matrix_size,
                                       rank = self.rank,
                                       form = self.form,
                                       letter = 't')
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            conjugation = w_alpha * t * (w_alpha.inv())
            assert(self.is_torus_element(matrix_to_test = conjugation, 
                                         matrix_size = self.matrix_size, 
                                         rank = self.rank, 
                                         form = self.form))
        if display: print("done.")
        
        if display: print("\tChecking squared Weyl elements centralize the torus... ", end="")
        t = self.generic_torus_element(matrix_size = self.matrix_size,
                                       rank = self.rank,
                                       form = self.form,
                                       letter = 't')
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            w_alpha_squared = w_alpha * w_alpha
            conjugation = sp.simplify( w_alpha_squared * t * (w_alpha_squared.inv()) )
            assert(conjugation.equals(t))
        if display: print("done.")
        
        if display: print("\tChecking Weyl element conjugation formula... ", end="")
        # INCOMPLETE
        if display: print("TEST NOT YET IMPLEMENTED.")
        
        if display: print("\tChecking Weyl group conjugation coefficients square to 1... ", end="")
        # INCOMPLETE
        if display: print("TEST NOT YET IMPLEMENTED.")
        
        if display: print("Weyl group verifications complete.")