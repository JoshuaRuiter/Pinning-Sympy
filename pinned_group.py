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
import copy
from utility_general import vector_variable
from utility_roots import (generate_character_list, 
                           reduce_character_list, 
                           determine_roots, 
                           visualize_graph, 
                           evaluate_character,
                           evaluate_cocharacter,
                           generic_kernel_element)
from root_system import root_system

class pinned_group:

    def __init__(self,
                 name_string,
                 matrix_size,
                 form,
                 is_group_element,
                 maximal_split_torus,
                 is_lie_algebra_element,
                 generic_lie_algebra_element,
                 non_variables = None):
        
        self.name_string = name_string
        self.matrix_size = matrix_size
        self.form = form
        self.non_variables = non_variables
        self.torus = maximal_split_torus
        self.rank = self.torus.rank
        
        self.is_group_element = lambda matrix_to_test : is_group_element(matrix_to_test, self.form)
        self.is_torus_element = lambda matrix_to_test : self.torus.is_element(matrix_to_test, self.rank)
        self.generic_torus_element = lambda letter : self.torus.generic_element(self.matrix_size, 
                                                                                self.rank, 
                                                                                letter)
        self.is_lie_algebra_element = lambda matrix_to_test : is_lie_algebra_element(matrix_to_test, self.form)
        self.generic_lie_algebra_element = lambda letter : generic_lie_algebra_element(self.matrix_size, 
                                                                                       self.rank, 
                                                                                       self.form, 
                                                                                       letter)

        # These get set by running .fit_pinning()
        self.root_system = None
        self.root_space_dimension_dict = None
        self.root_space_dimension = None
        self.root_space_dict = None
        self.root_space_map = None
        self.root_subgroup_map = None
        self.homomorphism_defect_coefficient_dict = None
        self.homomorphism_defect_map = None
        self.commutator_coefficient_dict = None
        self.commutator_coefficient_map = None
        self.weyl_element_map = None
        self.weyl_conjugation_coefficient_map = None
    
    def fit_pinning(self, display = True):
        # work out all the computational details related to Lie algebra, roots, etc.
        
        if display:
            print("\n" + '-' * 100 + "\n")
            print(f"Fitting a pinning for {self.name_string}")
            if self.form is not None:
                print("\nForm matrix:")
                sp.pprint(self.form.matrix)
                
            x = self.generic_lie_algebra_element('x')
            print("\nGeneric Lie algebra element:")            
            sp.pprint(x)
                
            t = self.generic_torus_element('t')
            print("\nGeneric torus element:")
            sp.pprint(t)
            
            print("\nTrivial characters:")
            sp.pprint(sp.Matrix(self.torus.trivial_character_matrix))

        self.fit_root_system(display)
        self.fit_root_spaces(display)
        self.fit_root_subgroup_maps(display)
        self.fit_weyl_group(display)
        self.fit_homomorphism_defect_coefficients(display)
        self.fit_commutator_coefficients(display)

        if display: 
            print(f"\nPinning fitting for {self.name_string} is complete")
            print("\n" + '-' * 100 + "\n")
            
    def fit_root_system(self, display = True):
        t = self.generic_torus_element('t')
        x = self.generic_lie_algebra_element('x')
        x_vars = x.free_symbols
        if self.non_variables is not None: x_vars = x_vars - self.non_variables
        full_char_list = generate_character_list(self.torus.nontrivial_character_entries,
                                                 upper_bound = 2)
        
        reduced_char_list = reduce_character_list(vector_list = full_char_list,
                                                  lattice_matrix = self.torus.trivial_character_matrix)
        
        if display:
            print("\nCandidate characters:",len(full_char_list))                
            print("Candidates after quotienting by trivial characters:",len(reduced_char_list))
        
        self.root_space_dict = determine_roots(generic_torus_element = t,
                                                generic_lie_algebra_element = x,
                                                list_of_characters = reduced_char_list,
                                                variables_to_solve_for = x_vars,
                                                time_updates = False)
        root_list = list(self.root_space_dict.keys())
        self.root_system = root_system(root_list,self.torus.trivial_character_matrix)
        
        if display:
            print("Root system:",self.root_system.name_string)
            print("Number of roots:",len(root_list))
            if self.root_system.is_irreducible:
                print("Dynkin diagram:",visualize_graph(self.root_system.dynkin_graph))
         
    def fit_root_spaces(self, display = True):
        self.root_space_dimension_dict = {}
        for r, x in self.root_space_dict.items():
            x_vars = self.without_non_variables(list(x.free_symbols))
            self.root_space_dimension_dict[r] = len(x_vars)

        def root_space_dim(alpha):
            #is_root, alpha_equiv = self.root_system.is_root(alpha, return_equivalent = True)
            # assert is_root, "Cannot get root space dimension for non-root"
            # return self.root_space_dimension_dict[tuple(alpha_equiv)]
            assert self.root_system.is_root(alpha), "Cannot get root space dimension for non-root"
            return self.root_space_dimension_dict[alpha]
        self.root_space_dimension = root_space_dim
        
        def root_sp_map(alpha, u):
            # is_root, alpha_equiv = self.root_system.is_root(alpha, return_equivalent = True)
            # assert is_root, "Cannot get root space for non-root"
            # dim = self.root_space_dimension(alpha_equiv)
            # assert len(u) == dim, "Wrong length of input vector to root space map"
            # x = self.root_space_dict[tuple(alpha_equiv)]
            assert self.root_system.is_root(alpha), "Cannot get root space for non-root"
            dim = self.root_space_dimension(alpha)
            assert len(u) == dim, "Wrong size input vector to root space map"
            x = self.root_space_dict[alpha]
            x_vars = self.without_non_variables(list(x.free_symbols))
            assert len(x_vars) == len(u), "Mismatched number of variables to substitute in root space map"
            for i in range(dim):
                x = x.subs(x_vars[i],u[i])
            return sp.simplify(x)
        self.root_space_map = root_sp_map
    
    def without_non_variables(self, list_of_variables):
        new_list = copy.deepcopy(list_of_variables)
        if self.non_variables is not None:
            for v in self.non_variables:
                if v in new_list: new_list.remove(v)
        if self.form is not None:
            if self.form.primitive_element is not None:
                if self.form.primitive_element in new_list: new_list.remove(self.form.primitive_element)
            if self.form.anisotropic_vector is not None:
                for v in self.form.anisotropic_vector.free_symbols:
                    if v in new_list: new_list.remove(v)
        return new_list
    
    def fit_root_subgroup_maps(self, display = True):
        def root_subgp_map(alpha, u):
            # is_root, alpha_equiv = self.root_system.is_root(alpha, return_equivalent = True)
            # assert is_root, "Cannot get root space dimension for non-root"
            # dim = self.root_space_dimension(alpha_equiv)
            # assert len(u) == dim, "Wrong length of input vector to root subgroup map"
            # x = self.root_space_map(alpha_equiv, u)
            assert self.root_system.is_root(alpha), "Cannot get root space dimension for non-root"
            dim = self.root_space_dimension(alpha)
            assert len(u) == dim, "Wrong length of input vector to root subgroup map"
            x = self.root_space_map(alpha, u)
            assert not any(x**3), "Unexpected: The 3rd power of a root space element is a nonzero matrix."
            return sp.simplify(np.eye(self.matrix_size, dtype = int) + x + x*x/2)
        self.root_subgroup_map = root_subgp_map
        
        ###############################################
        # MOVE TO FIT_WEYL_GROUPS ONCE THAT IS WORKING
        ###############################################
        if display:
            print(f"\nRoot spaces and root subgroups for {self.name_string}:")
            for r in self.root_space_dict:
                x = self.root_space_dict[r]
                dim = self.root_space_dimension(r)
                print("\n\tRoot:",r)
                print("\tCoroot:",self.root_system.coroot_dict[r])
                print("\tRoot space dimension:",dim)
                print("\tRoot space:")
                sp.pprint(x)
                print("\tRoot subgroup:")
                u = vector_variable(letter = 'u', length = dim)
                X_alpha_u = self.root_subgroup_map(r, u)
                sp.pprint(X_alpha_u)

    def fit_homomorphism_defect_coefficients(self, display = True):
        # For multipliable roots, the root subgroup maps are not quite homomorphisms
        # Instead, there is a general formula
        # X_alpha(u)*X_alpha(v) = X_alpha(u+v) * product over i>1 of X_i*alpha(some function of u and v)
        # The general product is kinda dumb because I am pretty sure i can only be 2
        # That function of u and v is what I call the "homomorphism defect coefficient"
        
        self.homomorphism_defect_coefficient_dict = {}
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
                    assert len(solutions_list) == 1, "Unexpected: more than one solution for a homomorphism defect coefficient"
                    solutions_dict = solutions_list[0]
                    solutions_dict_keys = list(solutions_dict.values())
                    assert len(solutions_dict_keys) == 1, "Unexpected: more than one homomorphism defect coefficient"
                    w_solution = solutions_dict_keys[0]
                    self.homomorphism_defect_coefficient_dict[alpha] = [u, v, w_solution]
                    
        if display:
            if len(self.homomorphism_defect_coefficient_dict) == 0: 
                print("\nNo multipliable roots, so no homomorphism defect coefficients.")
            else:
                print(f"Homomorphism defect coefficients for {self.name_string}:")
                for alpha in self.homomorphism_defect_coefficient_dict:
                    print("\nRoot:",alpha)
                    print("Homomorphism defect coefficient:",self.homomorphism_defect_coefficient_dict[alpha][2])

        def hom_defect_coeff(alpha, u, v):
            is_root, alpha_equiv = self.root_system.is_root(alpha, with_equivalent = True)
            assert is_root, "Cannot get homomorphism defect coefficient for non-root"
            if not self.root_system.is_multipliable_root(alpha): return [0]
            else:
                dict_entry = self.homomorphism_defect_coefficient_dict[alpha_equiv]
                u_prime = dict_entry[0]
                v_prime = dict_entry[1]
                hdc = dict_entry[2]
                assert len(u) == len(u_prime), "Wrong length vector input for homomorphism defect coefficient 1st input"
                assert len(v) == len(v_prime), "Wrong length vector input for homomorphism defect coefficient 2nd input"
                for i in range(len(u)):
                    hdc = hdc.subs(u_prime[i], u[i])
                for j in range(len(v)):
                    hdc = hdc.subs(v_prime[j], v[j])
                return [hdc]
        self.homomorphism_defect_map = hom_defect_coeff

    def fit_commutator_coefficients(self, display = True):
        self.commutator_coefficient_dict = {}
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            u = vector_variable(letter = 'u', length = d_alpha)
            x_alpha_u = self.root_subgroup_map(alpha, u)
            for beta in self.root_system.root_list:
                if  (self.root_system.is_root(alpha + beta) 
                     and not(self.root_system.is_proportional(alpha,beta,with_ratio=False))):
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
                    vars_to_solve_for = set()
                    new_coeff_dict = {}
                    
                    # Assemble the right hand side of the commutator formula,
                    # which is the product over all (i,j) pairs of positive integers
                    # such that i*alpha+j*beta is a root.
                    # For each such pair (i,j), the RHS product has factor 
                    # X_(i*alpha+j*beta) ( N_ij(u,v) )
                    # where N_ij is some function of u and v, called the commutator coefficient
                    # Note that N_ij depends crucially on alpha and beta as well
                    RHS = np.eye(self.matrix_size, dtype = int)
                    for key in linear_combos:
                        i = key[0]
                        j = key[1]
                        root = linear_combos[key]
                        assert root.equals(i*alpha + j*beta), "Error in linear combos"
                        d_ij = self.root_space_dimension(root)
                        N_ij = vector_variable(letter = 'N' + str(i) + str(j), length = d_ij)
                        vars_to_solve_for = vars_to_solve_for.union(set(N_ij))
                        RHS = RHS * self.root_subgroup_map(root, N_ij)
                        new_coeff_dict[(alpha, beta, i,j)] = (u, v, N_ij)
                        
                    # Now solve for all of the N_ij
                    vanishing_expression = sp.simplify(LHS - RHS)
                    solutions_list = sp.solve(vanishing_expression, vars_to_solve_for, dict=True)
                    solutions_dict = solutions_list[0]
                    assert len(solutions_list) == 1, "Unexpected: more than one solution for commutator coefficient"
                    assert len(solutions_dict) == len(vars_to_solve_for), "Unexpected: unable to solve for all commutator coefficients"

                    # plug in solutions for N_ij variables,
                    # and transfer to the permanent storage dictionary
                    for key, value in new_coeff_dict.items():
                        # key is a quadruple (alpha, beta, i, j)
                        # value is a triple (u, v, N_ij)
                        assert len(key) == 4, "Wrong length of key in commutator coefficient dictioanry"
                        assert len(value) == 3, "Wrong length of value in commutator coefficient dictionary"
                        alpha = key[0]
                        beta = key[1]
                        i = key[2]
                        j = key[3]
                        coeff_vec = value[2]
                        for var in vars_to_solve_for:
                            coeff_vec = coeff_vec.subs(var, solutions_dict[var])
                        self.commutator_coefficient_dict[(alpha, beta, i, j)] = (u, v, coeff_vec)

        def ccm(alpha, beta, i, j, u, v):
            if not self.root_system.is_root(alpha + beta): return [0]
            assert not self.root_system.is_proportional(alpha, beta, with_ratio = False), \
                "Commutator formula does not apply to proportional root pairs"

            u_prime, v_prime, coeff = self.commutator_coefficient_dict[(alpha, beta, i, j)]
            assert len(u) == len(u_prime), "Wrong length vector input for commutator coefficient 1st input"
            assert len(v) == len(v_prime), "Wrong length vector input for commutator coefficient 1st input"
            for i in range(len(u)):
                coeff = coeff.subs(u_prime[i], u[i])
            for j in range(len(v)):
                coeff = coeff.subs(v_prime[j], v[j])
            return sp.simplify(coeff)
        self.commutator_coefficient_map = ccm

        if display:
            if len(self.commutator_coefficient_dict) == 0:
                print("\nNo pairs of summable roots, so no commutator coefficients.")
            else:
                print(f"\nCommutator coefficients for {self.name_string}:\n")
                for root1 in self.root_system.root_list:
                    d_1 = self.root_space_dimension(root1)
                    u = vector_variable(letter = 'u', length = d_1)
                    for root2 in self.root_system.root_list:
                        if (self.root_system.is_root(root1 + root2)
                            and not self.root_system.is_proportional(root1, root2, with_ratio = False)):
                            d_2 = self.root_space_dimension(root2)
                            v = vector_variable(letter = 'v', length = d_2)
                            linear_combos = self.root_system.integer_linear_combos(root1,root2)
                            print("\tRoot 1:",root1)
                            print("\tRoot 2:",root2)
                            for key in linear_combos:
                                i = key[0]
                                j = key[1]
                                root = linear_combos[key]
                                
                                ################################################
                                # print("\n\nroot1 =",root1)
                                # print("root2 =",root2)
                                # print("linear combos:",linear_combos)
                                # print("(i,j) =",(i,j))
                                # print("combo =",root)
                                # print("i*root1 + j*root2 =",i*root1+j*root2)
                                ################################################
                                
                                assert(root.equals(i*root1 + j*root2)), "Error with linear combos"
                                coeff = self.commutator_coefficient_map(root1, root2, i, j, u, v)
                                print("\t\t(i,j):",(i,j))
                                print("\t\ti*(Root 1) + j*(Root 2):",root)
                                if len(coeff) == 1:
                                    print("\t\tCommutator coefficient:",coeff[0])
                                else:
                                    print("\t\tCommutator coefficients:",",".join([str(c) for c in coeff]))
                                print()

    def fit_weyl_group(self, display = True):
        
        # Fit reflections s_alpha of the torus,
        # and elements w_alpha in the normalizer of the torus
        def torus_refl_map(alpha, t):
            assert self.root_system.is_root(alpha), "Can only perform torus reflection with a root from the root system"
            assert self.is_torus_element(t), "Can only perform torus reflection on a torus element"
            alpha_of_t = evaluate_character(alpha, t)
            alpha_of_t_inverse = alpha_of_t**(-1)
            alpha_check = self.root_system.coroot_dict[alpha]
            alpha_check_of_alpha_of_t_inverse = evaluate_cocharacter(alpha_check, alpha_of_t_inverse)
            assert self.is_torus_element(alpha_check_of_alpha_of_t_inverse), \
                "Cocharacter must return torus element"
            return t*alpha_check_of_alpha_of_t_inverse
        self.torus_reflection_map = torus_refl_map
        
        # Populate a dictionary of Weyl elements
        # Keys are roots (as tuples)
        # Values are group elements (matrices)
        # INCOMPLETE

    def validate_pinning(self, display = True):
        # run tests to verify the pinning, run only after fitting
        
        if display:
            print("\n" + '-' * 100 + "\n")
            print(f"Running tests to validate pinning of {self.name_string}")
            
            self.validate_basics(display)
            self.root_system.verify_root_system_axioms(display)
            self.validate_root_space_maps(display)
            self.validate_root_subgroup_maps(display)
            self.validate_torus_conjugation_formula(display)
            self.validate_commutator_formula(display)
            self.validate_weyl_group_properties(display)
            
        if display:
            print(f"\nPinning validation tests for {self.name_string} complete")
            print("\n" + '-' * 100 + "\n")
             
    def validate_basics(self, display = True):
        # test that the generic torus element is in the group
        if display: print("\nChecking that a generic torus element is in the group... ", end="")
        s = self.generic_torus_element('s')
        assert self.is_group_element(s), "Generic torus element is not a group element"
        assert self.is_torus_element(s), "Generic torus element is not a torus element"
        if display: print("done.")
        
        if display: print("Checking that cocharacters map into the torus...",end="")
        u = sp.symbols('u')
        for alpha in self.root_system.root_list:
            alpha_check = self.root_system.coroot_dict[alpha]
            
            # print("\n\nalpha =",alpha)
            # print("\nalpha_check =",alpha_check)
            
            alpha_check_of_u = evaluate_cocharacter(alpha_check, u)
            assert self.is_torus_element(alpha_check_of_u), \
                "Cocharacter alpha_check={alpha_check} does not output torus elements"
        if display: print("done.")
        
        # tests for is_in_lie_algebra
        if display: print("Checking that a generic Lie algebra element is in the Lie algebra... ", end="")
        A = self.generic_lie_algebra_element('a')
        assert self.is_lie_algebra_element(A), "Generic lie algebra element is not a Lie algebra element"
        if display: print("done.")      
        
    def validate_root_space_maps(self, display = True):
        print("\nVerifying properties of root spaces...")    
        if display: print("\tChecking that root spaces are in the Lie algebra... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            X_alpha_a = self.root_space_map(alpha, a)
            assert self.is_lie_algebra_element(X_alpha_a), \
                f"Root space for alpha = {alpha} is not in the Lie algebra"
        if display: print("done.")
        
        if display: print("\tChecking that roots of equal norm have same dimensional root spaces...",end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            for beta in self.root_system.root_list:
                d_beta = self.root_space_dimension(beta)
                if alpha.norm() == beta.norm():
                    assert d_alpha == d_beta, \
                        f"Roots alpha = {alpha} and beta = {beta} have" + \
                        "equal norm have different dimensions of root space: " \
                        f"d_alpha = {d_alpha} and d_beta = {d_beta}"
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
            assert LHS.equals(RHS), f"Root space map for alpha = {alpha} fails additivity"
        if display: print("done.")
        
        if display: print("\tChecking that negative roots have transpose root spaces...", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            d_n_alpha = self.root_space_dimension(-1*alpha)
            assert d_alpha == d_n_alpha, f"Roots alpha = {alpha} and -alpha = {-alpha} " + \
                                            "have different root space dimensions"
            a = vector_variable(letter = 'a', length = d_alpha)
            b = vector_variable(letter = 'b', length = d_n_alpha)
            X_alpha_a = self.root_space_map(alpha, a)
            X_n_alpha_b = self.root_space_map(-1*alpha, b)
            vanishing_condition = X_alpha_a - X_n_alpha_b
            vars_to_solve_for = a.free_symbols.union(b.free_symbols)
            solutions_list = sp.solve(vanishing_condition, vars_to_solve_for, dict=True)
            assert len(solutions_list) >= 1, "Roots alpha = {alpha} and -alpha = {-alpha} have " + \
                                                "have root spaces which are not transposes"
        if display: print("done.")
        
        if display: print("Finished verifying properties of root spaces.")
        
    def validate_root_subgroup_maps(self, display = True):
        if display: print("\nVerifying properties of root subgroup maps...")
        
        if display: print("\tChecking that root subgroup maps are in the group... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            X_alpha_a = self.root_subgroup_map(alpha, a)
            assert self.is_group_element(X_alpha_a), \
                f"Output of root subgroup map for alpha = {alpha} is not a group element"
        if display: print("done")
    
        if display: 
            pseudo = "pseudo-" if not self.root_system.is_reduced else ""
            print(f"\tChecking that root subgroup maps are {pseudo}homomorphisms... ", end="")
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
                assert len(coeff) == self.root_space_dimension(2*alpha), \
                    f"Homomorphism defect coefficient for alpha = {alpha} has wrong length"
                extra_factor = self.root_subgroup_map(2*alpha, coeff)
            if not self.root_system.is_multipliable_root(alpha):
                assert np.array_equal(extra_factor, np.eye(self.matrix_size, dtype = int)), \
                    f"Non-multipliable root alpha = {alpha} has non-trivial homomrophism defect coefficient"
            RHS = X_alpha_sum * extra_factor
            assert LHS.equals(RHS), "Homomorphism defect coefficient for " + \
                                    f"alpha = {alpha} does not satisfy its defining formula"
        if display: print("done.")
        
        if display: print("Root subgroup map verifications complete.")
    
    def validate_torus_conjugation_formula(self, display = True):
        if display: print("\nChecking torus conjugation formulas... ", end="")
        s = self.generic_torus_element('s')
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            alpha_of_s = evaluate_character(alpha,s)
            
            # Torus conjugation on the Lie algebra/root spaces
            X_alpha_a = self.root_space_map(alpha,a)
            LHS1 = s*X_alpha_a*s**(-1)
            RHS1 = self.root_space_map(alpha,alpha_of_s*a)
            assert LHS1.equals(RHS1), \
                "Torus conjugation does not act as character evaluation " + \
                    f"on a root space for alpha = {alpha}"
            
            # Torus conjugation on the group/root subgroups
            x_alpha_a = self.root_subgroup_map(alpha, a)
            LHS2 = s*x_alpha_a*s**(-1)
            RHS2 = self.root_subgroup_map(alpha,alpha_of_s*a)
            assert LHS2.equals(RHS2), \
                "Torus conjugation does not act as character evaluation " + \
                    f"on a root subgroup for alpha = {alpha}"
        if display: print("done.")
        
    def validate_commutator_formula(self, display = True):
        if display: print("\nVerifying commutator properties...")
        if display: print("\tChecking commutator formulas... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            x_alpha_a = self.root_subgroup_map(alpha, a)
            for beta in self.root_system.root_list:
                # Commutator formula only applies when the two roots are not proportional
                if not(self.root_system.is_proportional(alpha,beta,with_ratio=False)):
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
                        assert root.equals(i*alpha + j*beta), \
                            f"Error with linear combos: alpha = {alpha}, beta = {beta}, (i,j) = {(i,j)}"

                        # Compute the commutator coefficient that should arise,
                        # then multiply by the new factor
                        coeff = self.commutator_coefficient_map(alpha,beta,i,j,a,b)
                        new_factor = self.root_subgroup_map(root, coeff)
                        RHS = RHS * new_factor
                        
                    assert LHS.equals(RHS), f"Commutator formula failed for alpha = {alpha}, beta = {beta}"
        if display: print("done.")

        if display: print("\tVerifying that swapping order of roots negates the coefficient...", end = "")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            for beta in self.root_system.root_list:
                d_beta = self.root_space_dimension(beta)
                b = vector_variable(letter = 'b', length = d_beta)
                if (self.root_system.is_root(alpha + beta)
                    and not self.root_system.is_proportional(alpha,beta,with_ratio=False)):
                    linear_combos = self.root_system.integer_linear_combos(alpha,beta)
                    for key in linear_combos:
                        i = key[0]
                        j = key[1]
                        coeff_1 = sp.simplify(self.commutator_coefficient_map(alpha,beta,i,j,a,b))
                        coeff_2 = sp.simplify(self.commutator_coefficient_map(beta,alpha,j,i,b,a))
                        assert coeff_1.equals(-coeff_2), \
                            f"Swapped roots do not have negative commutator coefficient" + \
                                f"alpha = {alpha}, beta = {beta}"
        if display: print("done.")
        
        if display: print("Done verifying properties of commutators.")

    def validate_weyl_group_properties(self, display = True):
        if display: print("\nVerifying properties of the Weyl group...")
        t = self.generic_torus_element('t')
    
        if display: print("\tChecking that s_alpha outputs torus elements...",end="")
        for alpha in self.root_system.root_list:
            s_alpha_of_t = self.torus_reflection_map(alpha, t)
            
            # print("\n\nalpha = ",alpha)
            # print("\ns_alpha(t) =")
            # sp.pprint(s_alpha_of_t)
            
            assert self.is_torus_element(s_alpha_of_t), \
                f"Torus reflection by s_alpha (with alpha = {alpha}) does not land in the torus"
        if display: print("done.")
    
        if display: print("\tChecking that s_alpha pointwise fixes ker(alpha)...", end="")
        for alpha in self.root_system.root_list:
            k = generic_kernel_element(alpha, t)
            assert self.is_torus_element(k)
            s_alpha_of_k = self.torus_reflection_map(alpha, k)
            assert k.equals(s_alpha_of_k), f"Torus reflection does not fix ker({alpha})"
        if display: print("done.")
    
        if display: print("\tChecking that s_alpha inverts T/ker(alpha)...", end="")
        for alpha in self.root_system.root_list:
            # The map s_alpha pointwise fixes ker(alpha),
            # so it induces a map on the quotient T/ker(alpha).
            # This induced map must be inversion, i.e. for t in T,
            # s_alpha(t) * ker(alpha) = t^(-1) * ker(alpha)
            # as cosets. Equivalently,
            # s_alpha(t) * t is in ker(alpha)
            # so we can check this computationally 
            # by checking that alpha( s_alpha(t) * t ) = 1
            s_alpha_of_t = self.torus_reflection_map(alpha, t)
            alpha_of_s_times_t = evaluate_character(alpha, s_alpha_of_t * t)
            assert alpha_of_s_times_t == 1, \
                f"The map s_alpha does not properly invert T/ker(alpha) for alpha={alpha}"
        if display: print("done.")
        
        if display: print("\tChecking that s_alpha^2 is the identity on the torus...",end="")
        for alpha in self.root_system.root_list:
            s_alpha_of_t = self.torus_reflection_map(alpha,t)
            s_alpha_squared_of_t = self.torus_reflection_map(alpha, s_alpha_of_t)
            assert t.equals(s_alpha_squared_of_t), f"s_alpha^2 is not identity for alpha={alpha}"
        if display: print("done.")
    
        if display: print("\tChecking Weyl elements are in the group... ", end="")
        # for alpha in self.root_system.root_list:
        #     w_alpha = self.weyl_element_map(alpha)
        #     assert self.is_group_element(w_alpha, self.form), "Weyl element is not a group element"
        # if display: print("done.")
        if display: print("TEST SKIPPED.")
        
        if display: print("\tChecking Weyl elements normalize the torus... ", end="")
        # for alpha in self.root_system.root_list:
        #     w_alpha = self.weyl_element_map(alpha)
        #     conjugation = sp.simplify(w_alpha * t * (w_alpha.inv()))
        #     assert self.is_torus_element(conjugation, self.matrix_size, self.rank, self.form), \
        #         "Weyl element does not noramlize the torus"
        # if display: print("done.")
        if display: print("TEST SKIPPED.")
        
        if display: print("\tChecking squared Weyl elements centralize the torus... ", end="")
        # for alpha in self.root_system.root_list:
        #     w_alpha = self.weyl_element_map(alpha)
        #     w_alpha_squared = sp.simplify(w_alpha * w_alpha)
        #     LHS = sp.simplify(w_alpha_squared*t)
        #     RHS = sp.simplify(t*w_alpha_squared)
        #     assert LHS.equals(RHS), "Squared weyl element does not centralize the torus"
        # if display: print("done.")
        if display: print("TEST SKIPPED.")
        
        if display: print("\tChecking Weyl element conjugation formula... ", end="")
        # INCOMPLETE
        if display: print("TEST NOT YET IMPLEMENTED.")
        
        if display: print("\tChecking Weyl group conjugation coefficients square to 1... ", end="")
        # INCOMPLETE
        if display: print("TEST NOT YET IMPLEMENTED.")
        
        if display: print("Weyl group verifications complete.")