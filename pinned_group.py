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
import itertools
from utility_general import (vector_variable, 
                             pretty_map,
                             find_zero_vars, 
                             indent_multiline,
                             brute_force_vanishing_solutions_sparse,
                             prune_singletons,
                             solve_with_timeout,
                             has_structural_contradiction)
from utility_roots import (generate_character_list, 
                           reduce_character_list, 
                           determine_roots, 
                           visualize_graph, 
                           evaluate_character,
                           evaluate_cocharacter,
                           generic_kernel_element)
from root_system import root_system
from utility_SU import custom_real_part

class pinned_group:

    def __init__(self,
                 name_string,
                 matrix_size,
                 form,
                 group_constraints,
                 maximal_split_torus,
                 lie_algebra_constraints,
                 generic_lie_algebra_element,
                 non_variables = None):
        
        self.name_string = name_string
        self.matrix_size = matrix_size
        self.form = form
        self.group_constraints = group_constraints
        self.lie_algebra_constraints = lie_algebra_constraints
        self.non_variables = non_variables
        self.torus = maximal_split_torus
        self.rank = self.torus.rank
        
        def in_group(matrix_to_test):
            eqs = self.group_constraints(matrix_to_test, self.form)
            return all(sp.simplify(e) == 0 for e in eqs)
        self.is_group_element = in_group
        
        def in_lie_algebra(matrix_to_test):
            eqs = self.lie_algebra_constraints(matrix_to_test, self.form)
            return all(sp.simplify(e) == 0 for e in eqs)
        self.is_lie_algebra_element = in_lie_algebra
        
        self.is_torus_element = lambda matrix_to_test : \
            self.torus.is_element(matrix_to_test, self.rank)
        self.generic_torus_element = lambda letter : \
            self.torus.generic_element(self.matrix_size, self.rank, letter)
        self.generic_lie_algebra_element = lambda letter : \
            generic_lie_algebra_element(self.matrix_size, self.rank, self.form, letter)

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
            
            print("\nTrivial characters (rows):")
            sp.pprint(sp.Matrix(self.torus.trivial_character_matrix).T)

        self.fit_root_system(display)
        self.fit_root_spaces(display)
        self.fit_root_subgroup_maps(display)
        self.fit_homomorphism_defect_coefficients(display)
        self.fit_commutator_coefficients(display)
        self.fit_weyl_group_elements(display)
        self.fit_weyl_conjugation_coefficients(display)
        self.fit_coroot_torus_elements(display)
    
        if display:
            print("Fitting complete")
            # self.display_pinning_info()
            print("\n" + '-' * 100 + "\n")
    
    def display_pinning_info(self):
        
        print(f"\nPinning information for {self.name_string}:")
        
        print("\nRoot system:",self.root_system.name_string)
        print("Number of roots:",len(self.root_system.root_list))
        if self.root_system.is_irreducible:
            print("Dynkin diagram:",visualize_graph(self.root_system.dynkin_graph))
        
        t = self.generic_torus_element('t')
        if self.root_system.is_reduced:
            print("\nRoots, coroots, root spaces, root subgroups," + 
                  f"\n\ttorus reflection maps, and Weyl elements for {self.name_string}:")
        else:
            print("\nRoots, coroots, root spaces, root subgroups, torus reflection maps, " + 
                           f"\n\tWeyl elements, and homomorphism defect coefficients for {self.name_string}:")
        for alpha in self.root_space_dict:
            dim = self.root_space_dimension(alpha)
            mult_tag = ""
            if not self.root_system.is_reduced:
                if self.root_system.is_multipliable_root(alpha):
                    mult_tag = "(multipliable)" 
                else:
                    mult_tag ="(not multipliable)"
            print("\nRoot / alpha:", alpha, mult_tag)
            print("Coroot / alpha^:", self.root_system.coroot_dict[alpha])
            print("Root space dimension / d_alpha:", dim)
            print("Root space generic element / X_alpha(u):")
            u = vector_variable(letter = 'u', length = dim)
            print(indent_multiline(sp.pretty(self.root_space_map(alpha, u))))
            print("Root subgroup generic element / x_alpha(u):")
            print(indent_multiline(sp.pretty(self.root_subgroup_map(alpha, u))))
            print("Torus reflection map / s_alpha:")
            print(indent_multiline(pretty_map(t, self.torus_reflection_map(alpha,t))))
            print("Weyl element / w_alpha:")
            print(indent_multiline(sp.pretty(self.weyl_element_map(alpha))))
            if self.root_system.is_multipliable_root(alpha):
                print("Homomorphism defect coefficient:", \
                      self.homomorphism_defect_coefficient_dict[alpha][2])

        if len(self.commutator_coefficient_dict) == 0:
            print("\nThere are no pairs of summable roots, so there are no commutator coefficients\n")
        else:
            print(f"\nCommutator coefficients for {self.name_string}:\n")
            for alpha in self.root_system.root_list:
                d_alpha = self.root_space_dimension(alpha)
                u = vector_variable(letter = 'u', length = d_alpha)
                for beta in self.root_system.root_list:
                    if (self.root_system.is_root(alpha + beta)
                        and not self.root_system.is_proportional(alpha, beta, with_ratio = False)):
                        d_beta = self.root_space_dimension(beta)
                        v = vector_variable(letter = 'v', length = d_beta)
                        linear_combos = self.root_system.integer_linear_combos(alpha,beta)
                        print("\tRoot 1 / alpha:",alpha)
                        print("\tRoot 2 / beta:",beta)
                        for key in linear_combos:
                            i = key[0]
                            j = key[1]
                            combo = linear_combos[key]
                            assert(combo.equals(i*alpha + j*beta)), "Error with linear combos"
                            coeff = self.commutator_coefficient_map(alpha, beta, i, j, u, v)
                            coeff_for_printing = sp.pretty(coeff[0]) if len(coeff) == 1 else sp.pretty(coeff.T)
                            print("\t\t(i,j):",(i,j))
                            print("\t\ti*alpha + j*beta:",combo)
                            print("\t\tCommutator coefficient:",end="")
                            if len(coeff_for_printing.splitlines()) > 1: 
                                coeff_for_printing = indent_multiline(coeff_for_printing, prefix = "\t\t\t")
                                print("\n")
                            print(coeff_for_printing)
                            print()
                            

        print(f"Weyl conjugation coefficients for {self.name_string}:\n")
        for alpha in self.root_system.root_list:
            for beta in self.root_system.root_list:
                d_beta = self.root_space_dimension(beta)
                u = vector_variable(letter = 'u', length = d_beta)
                gamma = self.root_system.reflect_root(hyperplane_root = alpha,
                                                      root_to_reflect = beta)
                d_gamma = self.root_space_dimension(gamma)
                assert d_beta == d_gamma
                phi_u = self.weyl_conjugation_coefficient_map(alpha, beta, u)
                phi_u_for_printing = sp.pretty(phi_u[0]) if len(phi_u) == 1 else sp.pretty(phi_u.T)
                print("\tRoot 1 / alpha:",alpha)
                print("\tRoot 2 / beta:",beta)
                print("\tReflection / sigma_alpha(beta):",gamma)
                print("\tWeyl conjugation coefficient: ", end="")
                if len(phi_u_for_printing.splitlines()) > 1: 
                    phi_u_for_printing = indent_multiline(phi_u_for_printing, prefix = "\t\t\t")
                    print("\n")
                print(phi_u_for_printing)
                print()
                                
        print(f"End of pinning information for {self.name_string}")
        
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
            print("\nFitting root system")
            print("\tCandidate characters:",len(full_char_list))                
            print("\tCandidates after quotient by trivial characters:",len(reduced_char_list))
        
        self.root_space_dict = determine_roots(generic_torus_element = t,
                                                generic_lie_algebra_element = x,
                                                list_of_characters = reduced_char_list,
                                                vars_to_solve_for = x_vars,
                                                time_updates = False)
        root_list = list(self.root_space_dict.keys())
        self.root_system = root_system(root_list,self.torus.trivial_character_matrix)
         
    def fit_root_spaces(self, display = True):
        
        if display: print("Fitting root spaces/root space maps")
        
        # In the case of any split group, the root spaces all have dimension 1
        # In general though, root space dimensions can be large
        
        # For example, the group SO_n_q with q>=2 and n>=2q+2 is non-split
        # with relative root system of type B, and the dimension of a root space
        # for a long root is 1, but the dimension of a 
        # root space for a short root is n-2q
        
        # For another example, the group SU_n_q with q>=2 is quasi-split if n=2q
        # and non-quasisplit if n>2q. The relative root system is type C for n=2q
        # and type BC for n>2q (a non-reduced root system with 3 root lengths)
        # In the n>2q case, the root space dimension is 
        #   1 for long roots
        #   2 for medium roots
        #   2(n-2q) for short roots
        # Note that these dimensions are all over k, not over L=k(sqrt(d))
        
        self.root_space_dimension_dict = {}
        for r, x in self.root_space_dict.items():
            x_vars = self.without_non_variables(list(x.free_symbols))
            self.root_space_dimension_dict[r] = len(x_vars)

        def root_space_dim(alpha):
            #is_root, alpha_equiv = self.root_system.is_root(alpha, return_equivalent = True)
            # assert is_root, "Cannot get root space dimension for non-root"
            # return self.root_space_dimension_dict[tuple(alpha_equiv)]
            assert self.root_system.is_root(alpha), \
                "Cannot get root space dimension for non-root"
            return self.root_space_dimension_dict[alpha]
        self.root_space_dimension = root_space_dim
        
        def root_sp_map(alpha, u):
            # is_root, alpha_equiv = self.root_system.is_root(alpha, return_equivalent = True)
            # assert is_root, "Cannot get root space for non-root"
            # dim = self.root_space_dimension(alpha_equiv)
            # assert len(u) == dim, "Wrong length of input vector to root space map"
            # x = self.root_space_dict[tuple(alpha_equiv)]
            assert self.root_system.is_root(alpha), \
                "Cannot get root space for non-root"
            dim = self.root_space_dimension(alpha)
            assert len(u) == dim, \
                "Wrong size input vector to root space map"
            x = self.root_space_dict[alpha]
            x_vars = self.without_non_variables(list(x.free_symbols))
            assert len(x_vars) == len(u), \
                "Mismatched number of variables to substitute in root space map"
            for i in range(dim):
                x = x.subs(x_vars[i],u[i])
            return sp.simplify(x)
        self.root_space_map = root_sp_map
    
    def fit_root_subgroup_maps(self, display = True):
        
        if display: print("Fitting root subgroup maps")
        
        # We compute root subgroups by matrix exponentiating the root spaces
        # This is not valid in characteristic 2, because it involves dividing by 2
        # It might also have theoretical issues in general, but I think it mostly
        # works at least in characteristic zero
        
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

    def fit_homomorphism_defect_coefficients(self, display = True):
        # For multipliable roots, the root subgroup maps are not quite homomorphisms
        # Instead, there is a general formula
        # X_alpha(u)*X_alpha(v) = X_alpha(u+v) * product over i>1 of X_i*alpha(some function of u and v)
        # The general product is kinda dumb because I am pretty sure i can only be 2
        # That function of u and v is what I call the "homomorphism defect coefficient"
        
        if display: print("Fitting homomorphism defect coefficients")
        
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
                    assert len(solutions_list) == 1, \
                        "Unexpected: more than one solution for a homomorphism defect coefficient"
                    solutions_dict = solutions_list[0]
                    solutions_dict_keys = list(solutions_dict.values())
                    assert len(solutions_dict_keys) == 1, \
                        "Unexpected: more than one homomorphism defect coefficient"
                    w_solution = solutions_dict_keys[0]
                    self.homomorphism_defect_coefficient_dict[alpha] = [u, v, w_solution]

        def hom_defect_coeff(alpha, u, v):
            is_root, alpha_equiv = self.root_system.is_root(alpha, with_equivalent = True)
            assert is_root, "Cannot get homomorphism defect coefficient for non-root"
            if not self.root_system.is_multipliable_root(alpha): return [0]
            else:
                dict_entry = self.homomorphism_defect_coefficient_dict[alpha_equiv]
                u_prime = dict_entry[0]
                v_prime = dict_entry[1]
                hdc = dict_entry[2]
                assert len(u) == len(u_prime), \
                    "Wrong length vector input for homomorphism defect coefficient 1st input"
                assert len(v) == len(v_prime), \
                    "Wrong length vector input for homomorphism defect coefficient 2nd input"
                for i in range(len(u)):
                    hdc = hdc.subs(u_prime[i], u[i])
                for j in range(len(v)):
                    hdc = hdc.subs(v_prime[j], v[j])
                return [hdc]
        self.homomorphism_defect_map = hom_defect_coeff

    def fit_commutator_coefficients(self, display = True):
        
        if display: print("Fitting commutator coefficients")
        
        # In SL_n, a commutator [x_alpha(u), x_beta(v)]
        # is nontrivial if and only if alpha+beta is a root,
        # in which case alpha+beta is the only positive integral
        # linear combination of alpha and beta which is a root
        # So in this case, the commutator formula takes the form
        # [x_alpha(u), x_beta(v)] = x_{alpha+beta}(N(u,v))
        # This N(u,v) is what is called a commutator coefficient here
        # In the case of SL_n, N(u,v)=+/-uv
        
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

    def fit_weyl_group_elements(self, display = True):
        
        if display: print("Fitting torus reflections s_alpha")
        # Fit reflections s_alpha of the torus,
        # and elements w_alpha in the normalizer of the torus
        def torus_refl_map(alpha, t):
            assert self.root_system.is_root(alpha), \
                "Can only perform torus reflection with a root from the root system"
            assert self.is_torus_element(t), \
                "Can only perform torus reflection on a torus element"
            alpha_of_t = evaluate_character(alpha, t)
            alpha_of_t_inverse = alpha_of_t**(-1)
            alpha_check = self.root_system.coroot_dict[alpha]
            alpha_check_of_alpha_of_t_inverse = evaluate_cocharacter(alpha_check, alpha_of_t_inverse)
            assert self.is_torus_element(alpha_check_of_alpha_of_t_inverse), \
                "Cocharacter must return torus element"
            return t*alpha_check_of_alpha_of_t_inverse
        self.torus_reflection_map = torus_refl_map
        
        if display: print("Fitting Weyl elements w_alpha")
        
        # Populate a dictionary of Weyl elements
        # Keys are roots (as tuples)
        # Values are group elements (matrices)
        self.weyl_element_list = {}
        t = self.generic_torus_element('t')
        s = self.generic_torus_element('s')
        
        # For the group SL_n, 
        # w_alpha = x_alpha(1) * x_{-alpha}(-1) * x_alpha(1)
        # up to multiplication by a diagonal matrix
        
        for alpha in self.root_system.root_list:
            
            # The process for computing w_alpha is:
            #   1. Initialize a fully symbolic n x n matrix with entries w_ij
            #   2. Use the equation w_alpha * x_beta(u) * w_alpha^(-1) = x_{sigma_alpha(beta)} (v)
            #       (for beta ranging over the set of roots) to eliminate some variables w_ij
            #   3. Use the equation w_alpha * t * w_alpha^(-1) = s (w_alpha normalizes the torus)
            #       to eliminate some more entries of w_alpha
            #   4. If the group is a special unitary group of a Hermitian form on a quadratic
            #       field extension L / k where L = k(p_e), replace w_ij with x_ij + p_e*y_ij 
            #       where p_e is the primitive element of the field extension
            #   5. At this point, w_ij should be a pretty sparse matrix. We expect each entry of w_alpha
            #       to be 0, +/-1, +/-p_e, or +/-(1/p_e), so we enumerate all of those possibilities. 
            #       We can eliminate 0 as a possibility for any variable which is alone it ins row/column.
            #   6. Finally, do a brute force solve on the remaining variables with possible values,
            #       making sure that we try variable combinations with more zeros first. This is implemented
            #       in brute_force_vanishing_solutions_sparse.
            #   6. Even with all of the pruning, there can still be a lot of combinations to try, but provided
            #       w_alpha is monomial as we expect, at least one solution should be found relatively quickly.
            #       The brute force solver has a stop_after_solution flag which is set to true. If all possible
            #       solutions for w_alpha are desired, this can be set to false, but the solving will take
            #       a very long time in this case.
            w_alpha = sp.Matrix(self.matrix_size, self.matrix_size, lambda i, j: sp.symbols(f'w_{i}_{j}'))
            w_vars = w_alpha.free_symbols
            
            generic_vars = t.free_symbols
            if self.non_variables is not None:
                generic_vars = generic_vars.union(self.non_variables)
            if self.form is not None:
                generic_vars = generic_vars.union(self.form.anisotropic_vector.free_symbols)
      
            # Use conjugation of root subgroups to eliminate matrix entries of w_alpha
            k = 0
            zero_vars = set()
            conjugation_eqs = []
            for beta in self.root_system.root_list:
                gamma = self.root_system.reflect_root(alpha, beta)
                d_gamma = self.root_space_dimension(gamma)
                d_beta = self.root_space_dimension(beta)
                assert d_beta == d_gamma
                u = vector_variable(f'u_{k}', d_beta)
                v = vector_variable(f'v_{k}', d_gamma)
                k = k + 1
                x_beta_u = self.root_subgroup_map(beta, u)
                x_gamma_v = self.root_subgroup_map(gamma, v)
                conjugation_matrix_eq = w_alpha * x_beta_u - x_gamma_v * w_alpha
    
                # Consider each entry of this matrix equation
                # to see if any can be used to eliminate a variable
                for i in range(self.matrix_size):
                    for j in range(self.matrix_size):
                        expr = conjugation_matrix_eq[i,j]
                        if not expr.is_zero:
                            conjugation_eqs.append(expr)
                            new_zero_vars = find_zero_vars(expr, w_vars, generic_vars.union(u.free_symbols))
                            if len(new_zero_vars) >= 1:
                                zero_vars = zero_vars.union(new_zero_vars)
                                w_vars = w_vars - new_zero_vars
            for var in zero_vars:
                w_alpha = w_alpha.subs(var,0)

            # Use the fact that w_alpha normalizes the torus to eliminate matrix entries of w_alpha
            zero_vars = set()
            normalizer_matrix_eq = w_alpha * t - s * w_alpha
            normalizer_eqs = []
            for i in range(self.matrix_size):
                for j in range(self.matrix_size):
                    expr = normalizer_matrix_eq[i,j]
                    if not expr.is_zero:
                        normalizer_eqs.append(expr)
                        new_zero_vars_1 = find_zero_vars(expr, w_vars, generic_vars)
                        new_zero_vars_2 = find_zero_vars(expr, w_vars, s.free_symbols)
                        if len(new_zero_vars_1) >= 1:
                            zero_vars = zero_vars.union(new_zero_vars_1)
                        if len(new_zero_vars_2) >= 1:
                            zero_vars = zero_vars.union(new_zero_vars_2)
            for var in zero_vars:
                w_alpha = w_alpha.subs(var,0)


            # Another condition: w_alpha should belong to the subgroup generated by
            # the images of x_alpha and x_{-alpha} and the torus
            # INCOMPLETE

            # If there is a quadratic field extension involved,
            # replace each nonzero entry w_ij of w_alpha
            # by x_ij + p_e * y_ij, where p_e is the primitive element
            # Simultaneously, construct the candidate values dictionary
            w_mask = w_alpha.applyfunc(lambda x: int(x != 0))
            variable_candidate_dict = {}
            if self.form is not None and self.form.primitive_element is not None:
                p_e = self.form.primitive_element
                x_mat = sp.Matrix(self.matrix_size, self.matrix_size, lambda i, j: sp.symbols(f'x_{i}_{j}'))
                y_mat = sp.Matrix(self.matrix_size, self.matrix_size, lambda i, j: sp.symbols(f'y_{i}_{j}'))
                w_alpha = x_mat + p_e * y_mat
                w_alpha = w_alpha.multiply_elementwise(w_mask)
                x_mat = x_mat.multiply_elementwise(w_mask)
                y_mat = y_mat.multiply_elementwise(w_mask)
                x_vars = sorted(x_mat.free_symbols, key=lambda s: s.name)
                y_vars = sorted(y_mat.free_symbols, key=lambda s: s.name)
                for var in x_vars:
                    variable_candidate_dict[var] = [0,1,-1]
                for var in y_vars:
                    variable_candidate_dict[var] = [0,1,-1,1/p_e**2,-1/p_e**2]
            else:
                # Non-quadratic field extension case
                for var in w_alpha.free_symbols:
                    variable_candidate_dict[var] = [0,1,-1]
                    
            # Prune zero values for matrix entries appearing as the lone entry
            # in their row or column, since we know the determinant is nonzero
            variable_candidate_dict = prune_singletons(matrix = w_alpha, 
                                                        variable_candidate_dict = variable_candidate_dict)


            # Solve the equations for group membership, using a brute force solver
            # This solver just tries all possible combinations of variables in the candidate
            # dictionary. The "sparse" aspect means that it starts by trying combinations
            # with more zeros first, because we expect w_alpha to be a monomial matrix
            # meaning that most of the variables will be zero.
            group_eqs = self.group_constraints(w_alpha, self.form)
            # solutions_list = brute_force_vanishing_solutions_sparse(group_eqs, 
            #                                                         variable_candidate_dict,
            #                                                         min_nonzero = self.matrix_size,
            #                                                         stop_after_solution = True,
            #                                                         display = False)
        
            solutions_list = brute_force_vanishing_solutions_sparse(group_eqs, 
                                                                    variable_candidate_dict,
                                                                    min_nonzero = self.matrix_size,
                                                                    stop_after_solution = False,
                                                                    display = True)
            
            assert len(solutions_list) >= 1, f"No solutions for w_alpha for alpha = {alpha} from the group {self.name_string}"
            # assert len(solutions_list) == 1, "More solutions obtained than expected"
            
            ############################
            print("Solutions found:", len(solutions_list))
            for d in solutions_list:
                sp.pprint(w_alpha.subs(d))
                print()
            ############################
            
            # In general, there are going to be multiple solutions for w_alpha
            # However, the stop_after_solution = True flag in the solver
            # means that the solver should return only one solution.
            # Finding all the solutions would take way more time.
            solutions_dict = solutions_list[0]
            w_alpha = w_alpha.subs(solutions_dict)
            self.weyl_element_list[alpha] = w_alpha
            
        def wem(alpha):
            return self.weyl_element_list[alpha]
        self.weyl_element_map = wem

    def fit_weyl_conjugation_coefficients(self, display = True):
        # given a weyl element w_alpha,
        # and a root beta, 
        # find the coefficient/function phi so that
        # w_alpha * x_beta(u) * w_alpha^(-1) = x_{sigma_alpha(beta)} ( phi(u) )
        
        # For the group SL_n, phi(u) = +/-u
        # For quasi-split special unitary groups SU_n_q with n=2q, phi(u) should be +/-u or +/-conjugate(u)
        
        # keys are (alpha, beta)
        # values are (u, phi(u))
        self.weyl_conjugation_coefficient_dict = {}
        
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            w_alpha_inverse = w_alpha**(-1)
            for beta in self.root_system.root_list:
                d_beta = self.root_space_dimension(beta)
                u = vector_variable('u', d_beta)
                x_beta_u = self.root_subgroup_map(beta, u)
                LHS = w_alpha*x_beta_u*w_alpha_inverse
                
                gamma = self.root_system.reflect_root(hyperplane_root = alpha,
                                                      root_to_reflect = beta)
                d_gamma = self.root_space_dimension(gamma)
                assert d_gamma == d_beta, "Reflected roots should have same root space dimension"
                v = vector_variable('v',d_gamma)
                x_gamma_v = self.root_subgroup_map(gamma, v)
                RHS = x_gamma_v
                
                vars_to_solve_for = v.free_symbols
                solutions_list = sp.solve(LHS-RHS, vars_to_solve_for, dict = True)
                assert len(solutions_list) == 1, "Unexpected number of solutions for Weyl conjugation coefficient"
                solutions_dict = solutions_list[0]
                
                phi_u = v.subs(solutions_dict)
                self.weyl_conjugation_coefficient_dict[(alpha, beta)] = (u, phi_u)
        
        def wccm(alpha, beta, u):
            key = (alpha, beta)
            val = self.weyl_conjugation_coefficient_dict[key]
            input_vars = val[0]
            output = val[1]
            assert len(u) == len(input_vars)
            for i, var in enumerate(input_vars):
                output = output.subs(var, u[i])
            return output
        self.weyl_conjugation_coefficient_map = wccm

    def fit_coroot_torus_elements(self, display = True):
        # Fit the elemnents h_alpha(t)
        # INCOMPLETE
        x = 0

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

    def belongs_to_generated_subgroup(self, matrix_to_test, 
                                      generating_roots,
                                      min_word_length,
                                      max_word_length,
                                      include_torus = True,
                                      solve_timeout = 10.0):
        # Test if a matrix belongs to the subgroup generated by elements
        # x_alpha(u) where alpha ranges over generating_roots, along with the torus. 
        # This is tested by checking products up to max_word_length.
        # The torus factor is not counted for the purposes of word length.
        
        M = matrix_to_test
        
        for alpha in generating_roots:
            assert self.root_system.is_root(alpha)
        assert self.is_group_element(M)
    
        n = M.rows
    
        # generic torus element
        # (not counted towards word length)
        if include_torus:
            t = self.generic_torus_element(letter='t')
            t_vars = list(t.free_symbols)
            nonzero_vars = t.free_symbols
        else:
            nonzero_vars = set()
    
        generator_dict = {}
        i = 0
        for alpha in generating_roots:
            d_alpha = self.root_space_dimension(alpha)
            u = vector_variable(f"u{i}",d_alpha)
            x_alpha_u = self.root_subgroup_map(alpha,u)         
            
            if self.form is not None and self.form.primitive_element is not None:
                p_e = self.form.primitive_element
                
                if not M.has(p_e) and x_alpha_u.has(p_e):
                    
                    print("Extracting real part")
                    print("x_alpha_u =")
                    sp.pprint(x_alpha_u)
                    print("x_alpha_u, real part =")
                    custom_real_part(x_alpha_u, p_e, is_matrix = True)
                    
                    x_alpha_u = custom_real_part(x_alpha_u, p_e, is_matrix = True)
                
            generator_dict[alpha] = x_alpha_u
            i += 1
    
        print("\n\nAttempting to find a word equal to M")
        print("Group:",self.name_string)
        print("M =")
        sp.pprint(M)
        print("Generating roots:",generating_roots)
        print("Generator matrices:")
        for alpha in generating_roots:
            sp.pprint(generator_dict[alpha])
            print()
        print("Min word length:",min_word_length)
        print("Max word length:",max_word_length)
        print("Include torus?",include_torus)
    
        for k in range(min_word_length, max_word_length + 1):
            for word in itertools.product(generating_roots, repeat=k):
                
                # skip words in which the same root is used twice consecutively
                if any(word[i] == word[i+1] for i in range(len(word)-1)):
                    continue
                
                expr = sp.eye(n)
                vars_to_solve_for = set()
    
                for alpha in word:
                    x_alpha = generator_dict[alpha]
                    expr = expr * x_alpha
                    vars_to_solve_for |= self.without_non_variables(x_alpha.free_symbols)    
            
                # allow an arbitrary torus factor
                if include_torus: 
                    expr = t * expr
                    vars_to_solve_for = vars_to_solve_for.union(t_vars)
    
                eqs = []
                for i in range(n):
                    for j in range(n):
                        if expr[i, j] != M[i, j]:
                            eqs.append(expr[i, j] - M[i, j])

                print("\nNext attempted word:", word)
                print("Word length:",k)
                print("Variables:",vars_to_solve_for)
                print("Word:")
                if include_torus:
                    sp.pprint(t)
                    print()
                for alpha in word:
                    sp.pprint(generator_dict[alpha])
                    print()
                print("\nWord, multiplied out:")
                sp.pprint(sp.simplify(expr))
                print("\nVanishing conditions:")
                for eq in eqs:
                    sp.pprint(sp.simplify(eq))
                    print()

                if has_structural_contradiction(eqs, nonzero_vars):
                    print("Structural contradiction detected, no attempt to solve was made")
                    continue
                
                print("No structural contradiction detected, attempting to solve")
                
                try:
                    solutions = solve_with_timeout(eqs,list(vars_to_solve_for),timeout = solve_timeout)
                except TimeoutError:
                    print("Solver timed out and gave up\n")
                    continue
                except (sp.PolynomialError,
                        NotImplementedError,
                        ValueError,
                        ZeroDivisionError):
                    print("Error encountered while solving\n")
                    continue

                print("\nSolutions found:")
                print(solutions)
                
                if solutions:
                    return True
    
        print("\nAll word attempts failed, no solutions found\n")
    
        return False
        
    def validate_pinning(self, display = True):
        # run tests to verify the pinning, run only after fitting
        
        if display:
            print("\n" + '-' * 100 + "\n")
            print(f"Running tests to validate pinning of {self.name_string}")
            
            # self.validate_basics(display)
            # self.root_system.verify_root_system_axioms(display)
            # self.validate_root_space_maps(display)
            # self.validate_root_subgroup_maps(display)
            # self.validate_commutator_formula(display)
            self.validate_weyl_group_properties(display)
            # self.validate_coroot_torus_elements(display)
            
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
        
        if display: print("Checking the Lie algebra is additively closed... ", end="")
        A = self.generic_lie_algebra_element('a')
        B = self.generic_lie_algebra_element('b')
        assert self.is_lie_algebra_element(A+B), "Lie algebra is not additively closed"
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
        
        if display: print("\tChecking torus conjugation formulas... ", end="")
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
    
        if display: print("\tChecking torus conjugation formulas... ", end="")
        s = self.generic_torus_element('s')
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            alpha_of_s = evaluate_character(alpha,s)
            
            # Torus conjugation on the group/root subgroups
            x_alpha_a = self.root_subgroup_map(alpha, a)
            LHS2 = s*x_alpha_a*s**(-1)
            RHS2 = self.root_subgroup_map(alpha,alpha_of_s*a)
            assert LHS2.equals(RHS2), \
                "Torus conjugation does not act as character evaluation " + \
                    f"on a root subgroup for alpha = {alpha}"
        if display: print("done.")
    
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
        
        if display: print("\tChecking that x_alpha(u) belongs to its own root subgroup...", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable('a',d_alpha)
            x_alpha_a = self.root_subgroup_map(alpha,a)
            assert self.belongs_to_generated_subgroup(matrix_to_test = x_alpha_a, 
                                                      generating_roots = [alpha], 
                                                      min_word_length = 1,
                                                      max_word_length = 1,
                                                      include_torus = False)
        if display: print("done.")

        if display: print("Root subgroup map verifications complete.")
    
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

        if display: print("\tChecking that swapping root order negates the coefficient...", end = "")
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
                            "Swapped roots do not have negative commutator coefficient" + \
                                f"alpha = {alpha}, beta = {beta}"
        if display: print("done.")
        
        if display: print("Done verifying properties of commutators.")

    def validate_weyl_group_properties(self, display = True):
        if display: print("\nVerifying properties of the Weyl group...")
        t = self.generic_torus_element('t')
    
        if display: print("\tChecking that s_alpha outputs torus elements...",end="")
        for alpha in self.root_system.root_list:
            s_alpha_of_t = self.torus_reflection_map(alpha, t)
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
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            assert self.is_group_element(w_alpha), "Weyl element is not a group element"
        if display: print("done.")
        
        if display: print("\tChecking Weyl elements normalize the torus... ", end="")
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            conjugation = sp.simplify(w_alpha * t * (w_alpha.inv()))
            assert self.is_torus_element(conjugation), \
                "Weyl element does not noramlize the torus"
        if display: print("done.")
        
        if display: print("\tChecking squared Weyl elements centralize the torus... ", end="")
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            w_alpha_squared = sp.simplify(w_alpha * w_alpha)
            LHS = sp.simplify(w_alpha_squared*t)
            RHS = sp.simplify(t*w_alpha_squared)
            assert LHS.equals(RHS), "Squared weyl element does not centralize the torus"
        if display: print("done.")
        
        if display: print("\tChecking Weyl elements belong to appropriately generated subgroup...", end="")
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            
            # d_alpha = self.root_space_dimension(alpha)
            # a = vector_variable('a',d_alpha)
            # x_alpha_a = self.root_subgroup_map(alpha,a)
            # b = vector_variable('b',d_alpha)
            # x_m_alpha_b = self.root_subgroup_map((-1)*alpha,b)
            # print("\n\nalpha =",alpha)
            # print("-alpha =",(-1)*alpha)
            # print("\nx_alpha(a) =")
            # sp.pprint(x_alpha_a)
            # print("\nx_-alpha(b) =")
            # sp.pprint(x_m_alpha_b)
            # print("\nw_alpha =")
            # sp.pprint(w_alpha)
            
            assert self.belongs_to_generated_subgroup(matrix_to_test = w_alpha, 
                                                      generating_roots = [alpha, (-1)*alpha], 
                                                      min_word_length = 3,
                                                      max_word_length = 3,
                                                      include_torus = True)
        # if display: print("TEST SKIPPED")
        if display: print("done.")
        
        if display: print("\tChecking Weyl element conjugation formula... ", end="")
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            w_alpha_inverse = w_alpha.inv()
            for beta in self.root_system.root_list:
                gamma = self.root_system.reflect_root(hyperplane_root = alpha, root_to_reflect = beta)
                d_beta = self.root_space_dimension(beta)
                d_gamma = self.root_space_dimension(gamma)
                assert d_beta == d_gamma, "Reflected roots have mismatched dimensions"
                a = vector_variable('a',d_beta)
                b = vector_variable('b',d_gamma)
                x_beta_a = self.root_subgroup_map(beta,a)
                x_gamma_b = self.root_subgroup_map(gamma,b)
                LHS = w_alpha * x_beta_a * (w_alpha_inverse)
                
                # Flexible test, just check if there is a solution
                sols = sp.solve(LHS-x_gamma_b, b.free_symbols, dict=True)
                assert len(sols) >= 1, f"Weyl element for alpha = {alpha} doesn't properly" + \
                                        f"conjugate the root subgroup U_beta where beta = {beta}"
                                        
                # Rigid test, verify that the solution is as expected
                phi_a = self.weyl_conjugation_coefficient_map(alpha, beta, a)
                if d_gamma > 1:
                    assert len(phi_a) == d_gamma
                x_gamma_phi_a = self.root_subgroup_map(gamma, phi_a)
                RHS = x_gamma_phi_a
                assert LHS.equals(RHS), "Weyl conjugation coefficient is incorrect"
                
                # # Check that phi is self-inverse
                # phi_phi_a = self.weyl_conjugation_coefficient_map(alpha, beta, phi_a)
                
                # print("\n\nalpha =",alpha)
                # print("beta =",beta)
                # print("sigma_alpha(beta) =",gamma)
                # print("\nw_alpha =")
                # sp.pprint(w_alpha)
                # print("\nphi(a) =",phi_a)
                # print("\nx_beta(a) =")
                # sp.pprint(x_beta_a)
                # print("\nx_gamma(phi(a)) =")
                # sp.pprint(x_gamma_phi_a)
                # print("\nphi(phi(a)) =",phi_phi_a)
                
                # assert a.equals(phi_phi_a), "Weyl conjugation map should have order 2"
        if display: print("done.")
        
        if display: print("Weyl group verifications complete.")
        
    def validate_coroot_torus_elements(self, display = True):
        # INCOMPLETE
        print("\nCOROOT TORUS ELEMENT TESTS NOT IMPLEMENTED")
        