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
import operator
import copy
import itertools
from utility_general import (vector_variable, 
                             compare_nonzero_pattern,
                             format_table,
                             pretty_map,
                             find_zero_vars, 
                             #indent_multiline,
                             #solve_with_timeout,
                             #has_structural_contradiction,
                             prune_singletons)
from utility_roots import (generate_character_list, 
                           reduce_character_list, 
                           determine_roots, 
                           visualize_graph, 
                           evaluate_character,
                           generic_kernel_element,
                           evaluate_cocharacter)
from root_system import root_system
from functools import reduce
from operator import mul
from tabulate import tabulate

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

        # These get set by running fit_pinning
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
        self.torus_coroot_map = None
    
    def fit_pinning(self, display = True):
        # Work out all the computational details related to Lie algebra, roots, etc.
        
        if display:
            print("\n" + '-' * 100 + "\n")
            print(f"Fitting a pinning for {self.name_string}")
            if self.form is not None:
                print("\nForm matrix:")
                sp.pprint(self.form.matrix)
                
            print("\nGeneric Lie algebra element:")            
            sp.pprint(self.generic_lie_algebra_element('x'))

            print("\nGeneric torus element:")
            sp.pprint(self.generic_torus_element('t'))
            
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
            self.display_pinning_info()
            print("\n" + '-' * 100 + "\n")
    
    def display_pinning_info(self):
        print(f"\nPinning information for {self.name_string}:")
        print("\nRoot system:",self.root_system.name_string)
        print("Number of roots:",len(self.root_system.root_list))
        if self.root_system.is_irreducible:
            print("Dynkin diagram:",visualize_graph(self.root_system.dynkin_graph))
        
        print(f"\nRoots and associated data for {self.name_string}:")
        print(self.get_roots_table())
        
        if len(self.commutator_coefficient_dict) == 0:
            print("\nThere are no pairs of summable roots, so there are no commutator coefficients")
        else:
            print(f"\nCommutator coefficients for {self.name_string}:")
            print(self.get_commutator_table())
        
        print(f"\nWeyl conjugation coefficients for {self.name_string}:")
        print(self.get_weyl_conjugation_table())

        print(f"\nEnd of pinning information for {self.name_string}")
    
    def get_roots_table(self):
        # Compile a text table for info on roots, coroots, etc.
        import re
        
        t = self.generic_torus_element('t')
        tt = sp.symbols('t')
        table = []
        
        for alpha in self.root_space_dict:
            alpha_check = self.root_system.coroot_dict[alpha]
            d_alpha = self.root_space_dimension(alpha)
            u = vector_variable(letter = 'u', length = d_alpha)
            X_alpha_u = sp.pretty(self.root_space_map(alpha, u), use_unicode=False)
            x_alpha_u = sp.pretty(self.root_subgroup_map(alpha, u), use_unicode=False)
            s_alpha = pretty_map(t, self.torus_reflection_map(alpha, t), use_unicode=False)
            w_alpha = sp.pretty(self.weyl_element_map(alpha), use_unicode=False)
            h_alpha_t = sp.pretty(self.coroot_torus_element_map(alpha, tt), use_unicode=False)
    
            if self.root_system.is_reduced:
                table.append([alpha, alpha_check, d_alpha, X_alpha_u, x_alpha_u, s_alpha, w_alpha, h_alpha_t])
            else:
                # A non-reduced root system has multipliable and non-multipliable roots
                # Also, for multipliable roots, there is a feature that I call the
                # "homomorphism defect coefficient"
                mult = self.root_system.is_multipliable_root(alpha)
                hdc = self.homomorphism_defect_coefficient_dict[alpha][2] if mult else "NA"
                table.append([alpha, mult, alpha_check, d_alpha, X_alpha_u, x_alpha_u, s_alpha, w_alpha, h_alpha_t, hdc])
    
        if self.root_system.is_reduced:
            headers = ["α", "α^", "d_α", "X_α(u)", "x_α(u)", "s_α", "w_α", "h_α(t)"]
        else:
            headers = ["α", "Multipliable", "α^", "d_α", "X_α(u)", "x_α(u)", "s_α", "w_α", "h_α(t)", "Hom defect"]

        subscript_map = {"_0": "₀", "_1": "₁", "_2": "₂", "_3": "₃", "_4": "₄",
                         "_5": "₅", "_6": "₆", "_7": "₇", "_8": "₈", "_9": "₉"}
        cleaned_table = []
    
    
        for row in table:
            cleaned_row = []
            for cell in row:
                cell_str = str(cell)
                
                # Replace square brackets [ and ] with |
                cell_str = cell_str.replace('[', '|').replace(']', '|')
                
                # 1. Substitute subscripts: '_x' (2 chars) -> 'ₓ ' (1 char + 1 space = 2 chars)
                # This guarantees that the string length does not change.
                for ascii_sub, uni_sub in subscript_map.items():
                    pattern = rf"(\w){ascii_sub}(\^\d+)?"
                    cell_str = re.sub(
                        pattern, 
                        lambda m: f"{m.group(1)}{uni_sub} {m.group(2) if m.group(2) else ''}", 
                        cell_str
                    )
                    
                # 2. Substitute multiplication asterisks with dots
                # Matches '*' and strips any trailing space to keep width at exactly 1
                cell_str = re.sub(r"\*\s?", "⋅", cell_str)
                    
                cleaned_row.append(cell_str)
            cleaned_table.append(cleaned_row)
    
        return tabulate(cleaned_table, headers=headers, tablefmt="fancy_grid")

    def get_commutator_table(self):
        assert(len(self.commutator_coefficient_dict) > 0)
        
        table = []

        for (alpha, beta) in self.root_system.summable_non_proportional_pairs:
            d_alpha = self.root_space_dimension(alpha)
            d_beta = self.root_space_dimension(beta)
            u = vector_variable(letter = 'u', length = d_alpha)
            v = vector_variable(letter = 'v', length = d_beta)
            linear_combos = self.root_system.integer_linear_combos(alpha, beta)
            
            for key in linear_combos:
                i = key[0]
                j = key[1]
                combo = linear_combos[key]  # combo = i*alpha + j*beta
                coeff = self.commutator_coefficient_map(alpha, beta, i, j, u, v)
                
                # Extract expression; pass object directly or string conversion inside format_table
                raw_expr = coeff[0] if len(coeff) == 1 else coeff.T
                
                table.append([alpha, beta, i, j, combo, raw_expr])
                
        headers = ["α", "β", "i", "j", "iα + jβ", "N_ij^αβ(u,v)"]
        return format_table(table, headers)    
    
    def get_weyl_conjugation_table(self):
        table = []
        
        for alpha in self.root_system.root_list:
            for beta in self.root_system.root_list:
                d_beta = self.root_space_dimension(beta)
                u = vector_variable(letter = 'u', length = d_beta)
                gamma = self.root_system.reflect_root(hyperplane_root = alpha,
                                                      root_to_reflect = beta)
                d_gamma = self.root_space_dimension(gamma)
                assert d_beta == d_gamma
                phi_u = self.weyl_conjugation_coefficient_map(alpha, beta, u)
                
                # Extract the expression object directly so format_table can stringify it
                raw_expr = phi_u[0] if len(phi_u) == 1 else phi_u.T
                
                table.append([alpha, beta, gamma, raw_expr])
                
        headers = ["\nα", "\nβ", "\nγ=σ_α(β)", "Weyl \nconjugation \ncoefficient"]
        
        return format_table(table, headers)
    
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
            print("\nFitting root system (Φ)")
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
        
        if display: print("Fitting root spaces/root space maps (X_α)")
        
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
            assert self.root_system.is_root(alpha), \
                "Cannot get root space dimension for non-root"
            return self.root_space_dimension_dict[alpha]
        self.root_space_dimension = root_space_dim
        
        def root_sp_map(alpha, u):
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
    
    def fit_root_subgroup_maps(self, display = True):
        
        if display: print("Fitting root subgroup maps (x_α)")
        
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
        # The general product is a bit silly because i can only be 1 or 2
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
        
        if display: print("Fitting commutator coefficients (N_ij^αβ(u,v))")
        
        # In SL_n, a commutator [x_alpha(u), x_beta(v)]
        # is nontrivial if and only if alpha+beta is a root,
        # in which case alpha+beta is the only positive integral
        # linear combination of alpha and beta which is a root
        # So in this case, the commutator formula takes the form
        # [x_alpha(u), x_beta(v)] = x_{alpha+beta}(N(u,v))
        # This N(u,v) is what is called a commutator coefficient here
        # In the case of SL_n, N(u,v)=+/-uv
        
        # More generally, [x_alpha(u), x_beta(v)] is a product over
        # positive integer linear combinations of alpha and beta
        # that are roots, of x_{i*alpha+j*beta} (N_ij(u,v))
        
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
        
        #############################################
        display_extra = False  # toggle for debugging
        #############################################
        
        if display: print("Fitting torus reflections (s_α)")
        # Fit reflections s_alpha of the torus,
        # and elements w_alpha in the normalizer of the torus
        def torus_refl_map(alpha, t):
            assert self.root_system.is_root(alpha), "Can only perform torus reflection with a root from the root system"
            assert self.is_torus_element(t), "Can only perform torus reflection on a torus element"
            alpha_of_t = evaluate_character(alpha, t)
            alpha_of_t_inverse = alpha_of_t**(-1)
            alpha_check = self.root_system.coroot_dict[alpha]
            alpha_check_of_alpha_of_t_inverse = evaluate_cocharacter(alpha_check, alpha_of_t_inverse)
            assert self.is_torus_element(alpha_check_of_alpha_of_t_inverse), "Cocharacter must return torus element"
            return t*alpha_check_of_alpha_of_t_inverse
        self.torus_reflection_map = torus_refl_map
        
        if display: print("Fitting Weyl elements (w_α)")
        
        # Populate a dictionary of Weyl elements
        # Keys are roots (as tuples)
        # Values are group elements (matrices)
        self.weyl_element_list = {}
        t = self.generic_torus_element('t')
        s = self.generic_torus_element('s')
        
        # EXAMPLE: For the group SL_n, 
        # w_alpha = x_alpha(1) * x_{-alpha}(-1) * x_alpha(1)
        # up to multiplication by a diagonal matrix
        
        # The process for computing w_alpha is:
        #   1. Initialize a fully symbolic n x n matrix with entries w_ij
        #   2. Use the equation w_alpha * x_beta(u) * w_alpha^(-1) = x_{sigma_alpha(beta)} (v)
        #       for beta ranging over the set of roots
        #       to eliminate some variables w_ij
        #   3. Use the equation w_alpha * t * w_alpha^(-1) = s 
        #       that is, w_alpha normalizes the torus
        #       to eliminate some more entries of w_alpha
        #   4. If the group is a special unitary group of a Hermitian form 
        #       on a quadratic field extension L / k where L = k(p_e), 
        #       then replace w_ij with x_ij + p_e*y_ij 
        #       where p_e is the primitive element of the field extension
        #   5. At this point, w_ij should be a pretty sparse matrix
        #       We expect each entry of w_alpha to be 0, +/-1, +/-p_e, or +/-(1/p_e), 
        #       so we enumerate all of those possibilities. 
        #       We can eliminate 0 as a possibility for any variable which is alone its row/column.
        #   6. Finally, do a brute force solve on the remaining variables with possible values,
        #       trying all possible combinations with exactly n nonzero variables 
        #       (where n is the size of the matrices).
        #   7. After a solution is found via brute force, 
        #       test if it pattern matches the subgroup generated by x_alpha(u) and x_{-alpha}(u)
        #       If yes, stop searching.
        #       If not, keep looking for other candidates. 
        
        for alpha in self.root_system.root_list:

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
            variable_candidate_dict = prune_singletons(w_alpha, variable_candidate_dict)
            
            # Next we solve for w_alpha using the group equations as vanishing conditions,
            # using a brute force method which just tries all possible combinations of variables
            # in the candidate dictionary. We expect the solution to have a certain number
            # of nonzero variables, so we only look for solutiosn with a fixed number of nonzero variables.
            
            # BEGIN BRUTE FORCE SOLVER
            group_eqs = self.group_constraints(w_alpha, self.form)
            vanishing_conditions = group_eqs
            min_nonzero = self.matrix_size
            max_nonzero = self.matrix_size
            variables = list(variable_candidate_dict.keys())

            # Partition variables into those where zero is a candidate, and those where it is not
            zero_allowed = [v for v in variables if 0 in variable_candidate_dict[v]]
            zero_forbidden = [v for v in variables if 0 not in variable_candidate_dict[v]]

            # Nonzero candidates for every variable
            nonzero_candidates = {
                v: [c for c in variable_candidate_dict[v] if c != 0]
                for v in variables
            }

            # Sanity check
            for v in zero_forbidden:
                if not nonzero_candidates[v]: raise ValueError(f"Variable {v} has no valid nonzero candidates")

            F = len(zero_forbidden)      # forced nonzeros
            m = len(zero_allowed)        # optional nonzeros
            
            if max_nonzero < F: raise ValueError(f"max_nonzero={self.matrix_size} is less than the number of forced nonzeros={F}")
            if max_nonzero > F + m: max_nonzero = F + m

            if display_extra:
                print("\n\n" + "=" * 60 + "\nAttempting brute force solution for w_alpha")
                print("alpha =",alpha)
                print("Variables:", variables)
                print("Candidate dictionary:",variable_candidate_dict)
                print("Variables that can be zero:", zero_allowed)
                print("Variables that can't be zero:", zero_forbidden)
                print("Vanishing conditions:")
                for e in vanishing_conditions:
                    sp.pprint(e)
                tried = 0

            # k = number of nonzero variables chosen among zero-allowed ones
            k_min = max(0, min_nonzero - F)
            k_max = m if max_nonzero is None else min(m, max_nonzero - F)
            solution_found = False
            for k in range(k_min, k_max + 1):
                if solution_found: break
                if display_extra: 
                    print(f"\nTrying combinations with {k} nonzero variables")
                    forced_factor = reduce(mul,(len(nonzero_candidates[v]) for v in zero_forbidden),1)
                    support_factor = sum(reduce(mul, (len(nonzero_candidates[v]) for v in support), 1)
                        for support in itertools.combinations(zero_allowed, k))
                    print("Combinations to try:",support_factor * forced_factor)
                for support in itertools.combinations(zero_allowed, k):
                    if solution_found: break
                    assignment = {v: 0 for v in zero_allowed if v not in support}
                    nz_lists = [nonzero_candidates[v] for v in support]
                    forced_lists = [nonzero_candidates[v] for v in zero_forbidden]
                    for values in itertools.product(*nz_lists, *forced_lists):
                        if solution_found: break
                        split = len(support)
                        assignment.update(dict(zip(support, values[:split])))
                        assignment.update(dict(zip(zero_forbidden, values[split:])))
                        if display_extra:
                            tried += 1
                            if tried % 10000 == 0:
                                print("Combinations tried:", tried)
                        if all(eq.subs(assignment) == 0 for eq in vanishing_conditions):
                            if display_extra: 
                                print("\nBrute force solver found a solution for w_alpha, checking it pattern matches the expected subgroup")
                            sol = w_alpha.subs(assignment)
                            pattern_match = self.compare_subgroup_pattern(matrix_to_test = sol,
                                                                          generating_roots = [alpha, -alpha], 
                                                                          min_word_length = 3,
                                                                          max_word_length = 3,
                                                                          op = operator.le,
                                                                          display = display_extra)
                            if pattern_match:
                                solution_found = True
                                w_alpha = sol
                                if display_extra: 
                                    print("\n\tFound a solution for w_alpha belonging to the right subgroup")
                                    print("\talpha = ",alpha)
                                    print("\tw_alpha =")
                                    sp.pprint(w_alpha)
                                break
                            else:
                                if display_extra: 
                                    print("Candidate does not belong to the expected subgroup")
        
            assert solution_found, f"No solutions for w_alpha for alpha = {alpha} from the group {self.name_string}"
            
            if display_extra: 
                print("\nBrute force solver complete, solution found.")
                print("Group:",self.name_string)
                print("alpha =",alpha)
                print("w_alpha = ")
                sp.pprint(w_alpha)
                print("Total combinations tried:",tried)
                print("=" * 60 + "\n")
            # END OF BRUTE FORCE SOLVER
            
            self.weyl_element_list[alpha] = w_alpha
            
        def wem(alpha):
            return self.weyl_element_list[alpha]
        self.weyl_element_map = wem

    def fit_weyl_conjugation_coefficients(self, display = True):
        # Given a weyl element w_alpha and a root beta, 
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
        # These elements are defined by the following properties:
        #   1. h_alpha(t) should be an element of the torus
        #   2. For each alpha, conjugating the root subgropu U_alpha by h_alpha_t should correspond to "scaling by t"
        #       That is, h_alpha(t) * x_alpha(u) * h_alpha(t)^(-1)
        #       should be equal to x_alpha(t * u)
        #       Since h_alpha(t) is a torus element and in general, conjugating x_alpha(u) by a torus element s
        #       follows the equation
        #       s * x_alpha(u) * s^(-1) = x_alpha( alpha(s) * u )
        #       the equation involving h is just a special case of this and we just need to satisfy
        #       alpha( h_alpha(t) ) = t
        
        # EXAMPLE:
        # For classical split groups like SL_n and SO_n, h_alpha(t) can be written as a product
        # of Weyl group elements, i.e. h_alpha(t) = w_alpha(t) * w_alpha(-1)
        # However, this relies on a generalization of the Weyl elements which is not implemented here
        # (we don't have w_alpha(t), just things like w_alpha(1))
        
        if display: print("Fitting coroot torus elements (h_α)")
        
        # Populate a dictionary of coroot torus elements
        # Keys are roots (as tuples)
        # Values are group elements (matrices)
        self.coroot_torus_element_list = {}
        
        # Strategy for computing h_alpha(t):
        #   1. Instantiate a generic torus element h
        #   2. Write down the equation h * x_alpha(u) * h^(-1) = x_alpha(t * u), just rearranged into
        #       h * x_alpha(u) = x_alpha(t*u) * h for slight computation improvement
        #   3. Solve for entries of h
        
        t = sp.symbols('t')
        for alpha in self.root_system.root_list:
            
            h = self.generic_torus_element('h')
            vars_to_solve_for = self.without_non_variables(h.free_symbols)

            d_alpha = self.root_space_dimension(alpha)
            u = vector_variable(letter = 'u', length = d_alpha)
            x_alpha_u = self.root_subgroup_map(alpha, u)
            x_alpha_tu = self.root_subgroup_map(alpha, t*u)
        
            LHS = h * x_alpha_u
            RHS = x_alpha_tu * h
            vanishing_expression = sp.simplify(LHS - RHS)
            solutions_list = sp.solve(vanishing_expression, vars_to_solve_for, dict=True)
            solutions_dict = solutions_list[0]
            h_sol_1 = sp.simplify(h.subs(solutions_dict))
            
            # For any remaning variables in h_sol that are not t, just set them to 1
            h_sol_2 = h_sol_1.subs({sym: 1 for sym in h_sol_1.free_symbols if sym != t})
            
            ##############################################################
            #### FOR DEBUGGING
            # print("\nalpha = ")
            # sp.pprint(alpha)
            # print("\nh = ")
            # sp.pprint(h)
            # print("\nVariables to solve for:", vars_to_solve_for)
            # print("\nx_alpha(u) = ")
            # sp.pprint(x_alpha_u)
            # print("\nLHS = h * x_alpha(u) =")
            # sp.pprint(LHS)
            # print("\nRHS = x_alpha(tu) * h = ")
            # sp.pprint(RHS)
            # print("\nVanishing conditions = LHS - RHS = ")
            # sp.pprint(vanishing_expression)
            # print("\nSolutions list: ", solutions_list)
            # print("\nSolutions dict: ", solutions_dict)
            # print("\nSolution for h_alpha(t), may have free parameters:")
            # sp.pprint(h_sol_1)
            # print("\nSolution for h_alpha(t) with free parameters set to 1:")
            # sp.pprint(h_sol_2)
            ##############################################################
            
            assert len(solutions_list) >= 1, f"No solutions for h_alpha(t), alpha = {alpha}"
            
            # plug in solutions and transfer to the storage dictionary
            self.coroot_torus_element_list[alpha] = h_sol_2

        def cte(alpha, t):
            # t should be a scalar
            assert self.root_system.is_root(alpha), "Cannot get h_alpha(t) if alpha is not a root"
            h_alpha_var = self.coroot_torus_element_list[alpha]
            h_vars = self.without_non_variables(list(h_alpha_var.free_symbols))
            
            #############################################
            ### FOR DEBUGGING
            # print("\nalpha = ")
            # sp.pprint(alpha)
            # print("\nt = ", t)
            # print("\nh_alpha(t) = ")
            # sp.pprint(h_alpha_var)
            # print("\nVariables in h_alpha(t):", h_vars)
            #############################################
            
            assert(len(h_vars) == 1)
            my_var = next(iter(h_vars))
            h_alpha_t = h_alpha_var.subs(my_var, t)
            return h_alpha_t
        self.coroot_torus_element_map = cte

    def without_non_variables(self, list_of_variables):
        # Sometimes when solving, we need to ignore some variables
        # such as the primitive element p_e = sqrt(d)
        # or the variables c1, c2, ... in the bottom right corner
        # of the form matrix
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

    def compare_subgroup_pattern(self,
                                 matrix_to_test,
                                 generating_roots,
                                 min_word_length,
                                 max_word_length,
                                 op = operator.eq,
                                 display = False):
        # Given a matrix M=matrix_to_test,
        # do a sort of sanity check on whether M might belong
        # to the subgroup generated by elements x_alpha(u)
        # where alpha ranges over generating roots
        # This is tested by checking words of length in a certain range,
        # and just comparing the nonzero patterns of M and those words
        
        M = matrix_to_test
        n = M .rows
        eye_n = sp.eye(n)
        
        assert self.is_group_element(M), "Matrix does not belong to the group"
        for alpha in generating_roots:
            assert self.root_system.is_root(alpha), "List contains non-roots"
        
        if display:
            print("\n\tGroup:", self.name_string)
            print("\tFinding a word with nonzero pattern that matches M=")
            sp.pprint(M)
            print(f"\tWord length range: {min_word_length} to {max_word_length}")
            print("\tGenerating roots:", generating_roots)
        
        
        for k in range(min_word_length, max_word_length + 1):
            for word_roots in itertools.product(generating_roots, repeat=k):
                
                # skip words in which the same root is used twice consecutively
                if any(word_roots[i] == word_roots[i+1] for i in range(len(word_roots)-1)):
                    continue
                    
                # Build the matrix expression for the word
                # Each factor gets its own unique, indexed symbolic variable vector
                # the unique indices aren't really necessary, but whatever
                word_expr = eye_n
                for idx, alpha in enumerate(word_roots):
                    d_alpha = self.root_space_dimension(alpha)
                    u = vector_variable(f"u{idx}", d_alpha)
                    x_alpha_u = self.root_subgroup_map(alpha, u)
                    word_expr = word_expr * x_alpha_u
                
                if display:
                    print(f"\n\tChecking root word pattern: {word_roots}")
                    print("Test word: ")
                    sp.pprint(word_expr)

                # Verify pattern match
                if compare_nonzero_pattern(M, word_expr, op = op):
                    if display:
                        print(f"\tFound pattern match at length {k} with word: {word_roots}")
                    return True

        if display:
            print("\tAll word pattern checks failed, no structural match found.")
        return False

    def validate_pinning(self, display = True):
        # Run tests to verify the pinning
        # Run this only after fitting
        
        if display:
            print("\n" + '-' * 100 + "\n")
            print(f"Running tests to validate pinning of {self.name_string}")
            
            self.validate_basics(display)
            self.root_system.verify_root_system_axioms(display)
            self.validate_root_space_maps(display)
            self.validate_root_subgroup_maps(display)
            self.validate_commutator_formula(display)
            self.validate_weyl_group_properties(display)
            self.validate_coroot_torus_elements(display)
            
        if display:
            print(f"\nPinning validation tests for {self.name_string} complete")
            print("\n" + '-' * 100 + "\n")
             
    def validate_basics(self, display = True):
        s = self.generic_torus_element('s')
        u = sp.symbols('u')
        A = self.generic_lie_algebra_element('a')
        B = self.generic_lie_algebra_element('b')
        
        if display: print("\nVerifying properties of the torus... ")
        
        if display: print("\tChecking that a generic torus element is in the group... ", end="")
        assert self.is_group_element(s), "Generic torus element is not a group element"
        assert self.is_torus_element(s), "Generic torus element is not a torus element"
        if display: print("done.")
        
        if display: print("\tChecking that cocharacters map into the torus... ",end="")
        for alpha in self.root_system.root_list:
            alpha_check = self.root_system.coroot_dict[alpha]
            alpha_check_of_u = evaluate_cocharacter(alpha_check, u)
            assert self.is_torus_element(alpha_check_of_u), \
                "Cocharacter alpha_check={alpha_check} does not output torus elements"
        if display: 
            print("done.")
            print("Torus verifications complete.")
            print("\nVerifying properties of the Lie algebra...")

        if display: print("\tChecking that a generic Lie algebra element is in the Lie algebra... ", end="")
        assert self.is_lie_algebra_element(A), "Generic lie algebra element is not a Lie algebra element"
        if display: print("done.")      
        
        if display: print("\tChecking the Lie algebra is additively closed... ", end="")
        assert self.is_lie_algebra_element(A+B), "Lie algebra is not additively closed"
        if display: print("done.")      
        
        if display: print("\tChecking that the Lie algebra is closed under Lie bracket... ", end="")
        assert self.is_lie_algebra_element(A*B-B*A), 'Lie algebra is not closed under brackets'
        if display: print("done.")
        
        if display: print("Lie algebra verifications complete.")
        
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
        
        if display: print("\tChecking that roots of equal norm have same dimensional root spaces... ",end="")
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
        
        if display: print("\tChecking that negative roots have transpose root spaces... ", end="")
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
        
        if display: print("Root space verifications complete.")
        
    def validate_root_subgroup_maps(self, display = True):
        if display: print("\nVerifying properties of root subgroups...")
        
        if display: print("\tChecking that root subgroup maps output elements of the group... ", end="")
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
                    f"Non-multipliable root alpha = {alpha} has non-trivial homomorphism defect coefficient"
            RHS = X_alpha_sum * extra_factor
            assert LHS.equals(RHS), "Homomorphism defect coefficient for " + \
                                    f"alpha = {alpha} does not satisfy its defining formula"
        if display: print("done.")

        if display: print("\tChecking subgroup nonzero pattern matching... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            x_alpha_a = self.root_subgroup_map(alpha,a)
            assert self.compare_subgroup_pattern(matrix_to_test = x_alpha_a,
                                                 generating_roots = [alpha],
                                                 min_word_length = 1,
                                                 max_word_length = 1,
                                                 op = operator.eq,
                                                 display = False)
        
        for (alpha, beta) in self.root_system.non_proportional_pairs:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            x = self.root_subgroup_map(alpha, a)
            d_beta = self.root_space_dimension(beta)
            b = vector_variable(letter = 'b', length = d_beta)
            y = self.root_subgroup_map(beta, b)
            assert self.compare_subgroup_pattern(matrix_to_test = x*y,
                                                 generating_roots = [alpha,beta],
                                                 min_word_length = 2,
                                                 max_word_length = 2,
                                                 op = operator.eq,
                                                 display = False)
            assert self.compare_subgroup_pattern(matrix_to_test = x*y*x,
                                                 generating_roots = [alpha,beta],
                                                 min_word_length = 3,
                                                 max_word_length = 3,
                                                 op = operator.eq,
                                                 display = False)
            assert self.compare_subgroup_pattern(matrix_to_test = x*y*x*y,
                                                 generating_roots = [alpha,beta],
                                                 min_word_length = 4,
                                                 max_word_length = 4,
                                                 op = operator.eq,
                                                 display = False)
        if display: 
            print("done.")
            print("Root subgroup map verifications complete.")
    
    def validate_commutator_formula(self, display = True):
        
        if display: 
            print("\nVerifying commutator properties...")
            print("\tChecking commutator formulas... ", end="")
        
        # Commutator formula only applies when the two roots are not proportional
        # and they are summable (i.e. alpha + beta is a root)
        for (alpha, beta) in self.root_system.summable_non_proportional_pairs:
            assert self.root_system.is_root(alpha+beta)
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            x_alpha_a = self.root_subgroup_map(alpha, a)
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
        if display: 
            print("done.")
            print("\tChecking that swapping root order negates the coefficient... ", end = "")
        
        for (alpha, beta) in self.root_system.summable_non_proportional_pairs:
            assert self.root_system.is_root(alpha+beta)
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            d_beta = self.root_space_dimension(beta)
            b = vector_variable(letter = 'b', length = d_beta)
            linear_combos = self.root_system.integer_linear_combos(alpha,beta)
            for key in linear_combos:
                i = key[0]
                j = key[1]
                coeff_1 = sp.simplify(self.commutator_coefficient_map(alpha,beta,i,j,a,b))
                coeff_2 = sp.simplify(self.commutator_coefficient_map(beta,alpha,j,i,b,a))
                assert coeff_1.equals(-coeff_2), \
                    "Swapped roots do not have negative commutator coefficient" + \
                        f"alpha = {alpha}, beta = {beta}"
        if display: 
            print("done.")
            print("Commutator verifications complete.")

    def validate_weyl_group_properties(self, display = True):
        if display: print("\nVerifying properties of the Weyl group... ")
        t = self.generic_torus_element('t')
    
        if display: print("\tChecking that s_α outputs torus elements... ",end="")
        for alpha in self.root_system.root_list:
            s_alpha_of_t = self.torus_reflection_map(alpha, t)
            assert self.is_torus_element(s_alpha_of_t), \
                f"Torus reflection by s_α (with α = {alpha}) does not land in the torus"
        if display: print("done.")
    
        if display: print("\tChecking that s_α pointwise fixes ker(α)...", end="")
        for alpha in self.root_system.root_list:
            k = generic_kernel_element(alpha, t)
            assert self.is_torus_element(k)
            s_alpha_of_k = self.torus_reflection_map(alpha, k)
            assert k.equals(s_alpha_of_k), f"Torus reflection does not fix ker({alpha})"
        if display: print("done.")
    
        if display: print("\tChecking that s_α inverts T/ker(α)... ", end="")
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
                f"The map s_α does not properly invert T/ker(α) for α={alpha}"
        if display: print("done.")
        
        if display: print("\tChecking that s_α^2 is the identity on the torus...",end="")
        for alpha in self.root_system.root_list:
            s_alpha_of_t = self.torus_reflection_map(alpha,t)
            s_alpha_squared_of_t = self.torus_reflection_map(alpha, s_alpha_of_t)
            assert t.equals(s_alpha_squared_of_t), f"s_α^2 is not identity for α={alpha}"
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
        
        if display: print("\tChecking Weyl elements pattern match the expected subgroup... ", end="")
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            generators = [alpha, -alpha]
            assert self.compare_subgroup_pattern(matrix_to_test = w_alpha,
                                                generating_roots = generators,
                                                min_word_length = 3,
                                                max_word_length = 3,
                                                op = operator.le,
                                                display = False)
        if display: print("done.")
        
        if display: print("\tChecking Weyl element conjugation formula...", end="")
        # orders = {}
        for alpha in self.root_system.root_list:
            w_alpha = self.weyl_element_map(alpha)
            w_alpha_inverse = w_alpha.inv()
            for beta in self.root_system.root_list:
                gamma = self.root_system.reflect_root(hyperplane_root = alpha, root_to_reflect = beta)
                d_beta = self.root_space_dimension(beta)
                d_gamma = self.root_space_dimension(gamma)
                assert d_beta == d_gamma, "Reflected roots have mismatched dimensions"
                a = vector_variable('a',d_beta)
                x_beta_a = self.root_subgroup_map(beta,a)
                LHS = w_alpha * x_beta_a * (w_alpha_inverse)
                phi_a = self.weyl_conjugation_coefficient_map(alpha, beta, a)
                if d_gamma > 1: assert len(phi_a) == d_gamma
                x_gamma_phi_a = self.root_subgroup_map(gamma, phi_a)
                RHS = x_gamma_phi_a
                assert LHS.equals(RHS), "Weyl conjugation coefficient is incorrect"

                # Find the order of phi
                # Usually it seems to be 2, but I think it can be higher
                # It seems like sometimes the order is not even finite?
                
                # def phi(u):
                #     return self.weyl_conjugation_coefficient_map(alpha, beta, u)
                # order_phi = compute_order(phi, a, 10)
                
                # if order_phi in orders:
                #     orders[order_phi] += 1
                # else:
                #     orders[order_phi] = 1
                # assert order_phi <= 10, "Order of phi is larger than expected"
                
                # phi_phi_a = self.weyl_conjugation_coefficient_map(alpha, beta, phi_a)
                
                # print("\n\nalpha =",alpha)
                # print("beta =",beta)
                # print("sigma_alpha(beta) =",gamma)
                # print("\nw_alpha =")
                # sp.pprint(w_alpha)
                # print("\na = ")
                # sp.pprint(a)
                # print("phi(a) =", phi_a)
                # sp.pprint(phi_a)
                # print("Order of phi =", order_phi)
                # print("\nx_beta(a) =")
                # sp.pprint(x_beta_a)
                # print("\nx_gamma(phi(a)) =")
                # sp.pprint(x_gamma_phi_a)
                # print("\nphi(phi(a)) =",phi_phi_a)
        # if display: print("done. \n\t\tOrders found: ", orders)
        
        if display: print("\nWeyl group verifications complete.")
                
    def validate_coroot_torus_elements(self, display = True):
        
        if display: print("\nVerifying properties of torus coroot elements... ")
        t = sp.symbols('t')
        
        if display: print("\tVerifying they belong to the torus... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            x_alpha_a = self.root_subgroup_map(alpha, a)
            h_alpha_t = self.coroot_torus_element_map(alpha, t)
            assert self.is_torus_element(h_alpha_t), "h_alpha(t) is not in the torus"
        if display: print("\tdone.")
            
        if display: print("\tVerifying character evaluations... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            x_alpha_a = self.root_subgroup_map(alpha, a)
            h_alpha_t = self.coroot_torus_element_map(alpha, t)
            alpha_of_h_alpha_t = evaluate_character(alpha, h_alpha_t)
            assert alpha_of_h_alpha_t == t, "alpha(h_alpha(t)) is not t as expected"            
        if display: print("\tdone.")
        
        if display: print("\tVerifying conjugation by coroot torus elements... ", end="")
        for alpha in self.root_system.root_list:
            d_alpha = self.root_space_dimension(alpha)
            a = vector_variable(letter = 'a', length = d_alpha)
            x_alpha_a = self.root_subgroup_map(alpha, a)
            h_alpha_t = self.coroot_torus_element_map(alpha, t)
            h_alpha_t_inverse = h_alpha_t**(-1)
            LHS = h_alpha_t * x_alpha_a * h_alpha_t_inverse
            RHS = self.root_subgroup_map(alpha , t*a)
            assert LHS.equals(RHS), \
                "Torus coroot element conjugation does not scale as expected " + \
                    f"on a root subgroup for alpha = {alpha}"
        if display: print("done.")
        
        if display: print("Torus coroot verifications complete.")