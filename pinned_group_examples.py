# This file is used to run the primary battery of tests utilizing all of the 
# important tools related to pinned groups.

# The main battery of tests does the following for each of a list of classical groups:
#   1. Create a split_torus object
#   2. Create a pinned_group object
#   3. Fit a pinning for that pinned_group object
#   4. Check that the Dynkin type is as expected
#   5. Run a battery of tests to validate the pinning

# The groups considered here are:
#   Special linear group
#   Special orthogonal group associated to a nondegenerate isotropic bilinear form
#   Special unitary group associated to a nondegenerate isotropic hermitian or skew-hermitian form

import sympy as sp
import time
from pinned_group import pinned_group
from nondegenerate_isotropic_form import nondegenerate_isotropic_form
from split_torus import split_torus
from utility_general import vector_variable
from utility_SL import (group_constraints_SL,
                        is_torus_element_SL,
                        generic_torus_element_SL,
                        trivial_characters_SL,
                        character_entries_SL,
                        lie_algebra_constraints_SL,
                        generic_lie_algebra_element_SL)
from utility_SO import (group_constraints_SO,
                        is_torus_element_SO,
                        generic_torus_element_SO,
                        trivial_characters_SO,
                        character_entries_SO,
                        lie_algebra_constraints_SO,
                        generic_lie_algebra_element_SO)
from utility_SU import (group_constraints_SU,
                        is_torus_element_SU,
                        generic_torus_element_SU, 
                        trivial_characters_SU,
                        character_entries_SU,
                        lie_algebra_constraints_SU,
                        generic_lie_algebra_element_SU)

def main():
    to_do_list = ("TO DO LIST:" + "\n\t" + 
                  "Implement a Latex output file instead of console output" + "\n\t" +
                  "Make some tools or tables for visualizing equations/identites" + "\n\t" +
                  "Improve speed/optimization in a number of places. Top candidates:" + "\n\t\t" + 
                      "fit_weyl_elements, weyl group nonzero pattern matching" + "\n\t" +
                  "Find a way to implement belongs_to_generated_subgroup in a" + "\n\t\t" +
                      "computationally feasible way, even if with random numerical stuff" +
                  "Implement some kind of quadratic field extension class")
    print(to_do_list)
    
    print("\nDemonstrating usage of pinned group class")
    start_time = time.perf_counter()
    sp.init_printing(wrap_line=False)
    eps_values = [-1,1] # should only include +/-1
    
    #############################################
    ### Edit these to change which tests are run
    ### A "full test" is 2<=n<=6 and 1<=q<=3
    ### Full test takes over an hour to run
    n_min = 1
    n_max = 6
    q_min = 1
    q_max = 3
    #############################################
    
    #####################################################################
    ### Comment these out temporarily to shorten tests
    n_max_SL = 3 # SL_4 and beyond take a long time to compute roots
    run_SL_tests(n_min, min(n_max, n_max_SL))
    run_SO_split_tests(n_min, n_max, q_min, q_max)
    run_SO_nonsplit_tests(n_min, n_max, q_min, q_max)
    run_SU_quasisplit_tests(n_min, n_max, q_min, q_max, eps_values)
    run_SU_nonquasisplit_tests(n_min, n_max, q_min, q_max, eps_values)
    #####################################################################
    
    print("\nAll tests complete.")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Time to run tests: {round(execution_time/60, 1)} minutes")
    
    print("\n" + to_do_list)

def run_SL_tests(n_min, n_max):
    print("\n" + '=' * 100 + "\n")
    print("Running calculations and verifications for special linear groups")
    n_min = max(n_min, 2) # n=1 doesn't make sense, SL_1 is just the trivial group
    for n in range(n_min, n_max + 1):
        T = split_torus(matrix_size = n,
                        rank = n-1,
                        is_element = is_torus_element_SL,
                        generic_element = generic_torus_element_SL,
                        trivial_character_matrix = trivial_characters_SL(n),
                        nontrivial_character_entries = character_entries_SL(n))
        SL_n = pinned_group(name_string = f"SL(n={n})",
                            matrix_size = n,
                            form = None,
                            group_constraints = group_constraints_SL,
                            maximal_split_torus = T,
                            lie_algebra_constraints = lie_algebra_constraints_SL,
                            generic_lie_algebra_element = generic_lie_algebra_element_SL,
                            non_variables = None)
        SL_n.fit_pinning(display = True)
        assert SL_n.root_system.dynkin_type == 'A', \
            f"SL(n={n}) should have type A but " + \
            f"computations gave type {SL_n.root_system.dynkin_type}"
        SL_n.validate_pinning(display = True)
    print("\nDone with special linear groups")
    print("\n" + '=' * 100 + "\n")

def run_SO_split_tests(n_min, n_max, q_min, q_max):
    ###################################################
    ## SPLIT SPECIAL ORTHOGONAL GROUPS (n=2q or n=2q+1) 
    ###################################################
    print("\n" + '=' * 100 + "\n")
    print("Running calculations and verifications for split special orthogonal groups")
    q_min = max(q_min, 2) # doesn't make sense if q=1, there are no roots
    for q in range(q_min, q_max + 1):
        n_range = [n for n in (2*q, 2*q+1) if n_min <= n and n <= n_max]
        for n in n_range:
            anisotropic_vec = vector_variable('c',n-2*q)
            NIF = nondegenerate_isotropic_form(dimension = n,
                                                witt_index = q,
                                                anisotropic_vector = anisotropic_vec,
                                                epsilon = None,
                                                primitive_element = None)
            T = split_torus(matrix_size = n,
                            rank = q,
                            is_element = is_torus_element_SO,
                            generic_element = generic_torus_element_SO,
                            trivial_character_matrix = trivial_characters_SO(n,q),
                            nontrivial_character_entries = character_entries_SO(n,q))
            SO_n_q = pinned_group(name_string = f"SO(n={n}, q={q})",
                                matrix_size = n,
                                form = NIF, 
                                group_constraints = group_constraints_SO,
                                maximal_split_torus = T,
                                lie_algebra_constraints = lie_algebra_constraints_SO,
                                generic_lie_algebra_element = generic_lie_algebra_element_SO,
                                non_variables = None)
            SO_n_q.fit_pinning(display = True)
            if n==2*q:
                if q==2:
                    expected_type = ['A','A']
                elif q==3:
                    expected_type = 'A'
                else:
                    expected_type = 'D'
            else:
                expected_type = 'B'
            assert SO_n_q.root_system.dynkin_type == expected_type, \
                    f"SO(n={n}, q={q}) is type {expected_type} but computations " + \
                    f"gave type {SO_n_q.root_system.dynkin_type}"
                    
            SO_n_q.validate_pinning(display = True)
    print("\nDone with split special orthogonal groups")
    print("\n" + '=' * 100 + "\n")

def run_SO_nonsplit_tests(n_min, n_max, q_min, q_max):
    #############################################################################
    ## NON-SPLIT SPECIAL ORTHOGONAL GROUPS 
        ## SO_n_q is quasi-split if n=2q+2, and
        ## neither split nor quasi-split if n>2+2q, 
        ## but the behavior seems to be basically the same in these two cases
    #############################################################################
    print("\n" + '=' * 100 + "\n")
    print("Running calculations and verifications for non-split special orthogonal groups")
    for q in range(q_min, q_max + 1):
        n_min = max(2*q+2, n_min) # only non-split if n >= 2q+2
        for n in range(n_min, n_max + 1):
            anisotropic_vec = vector_variable('c',n-2*q)
            NIF = nondegenerate_isotropic_form(dimension = n,
                                                witt_index = q,
                                                anisotropic_vector = anisotropic_vec,
                                                epsilon = None,
                                                primitive_element = None)
            T = split_torus(matrix_size = n,
                            rank = q,
                            is_element = is_torus_element_SO,
                            generic_element = generic_torus_element_SO,
                            trivial_character_matrix = trivial_characters_SO(n,q),
                            nontrivial_character_entries = character_entries_SO(n,q))            
            SO_n_q = pinned_group(name_string = f"SO(n={n}, q={q})",
                                matrix_size = n,
                                form = NIF, 
                                group_constraints = group_constraints_SO,
                                maximal_split_torus = T,
                                lie_algebra_constraints = lie_algebra_constraints_SO,
                                generic_lie_algebra_element = generic_lie_algebra_element_SO,
                                non_variables = None)
            SO_n_q.fit_pinning(display = True)
            if q == 1:
                expected_type = 'A'
            else:
                expected_type = 'B'
            assert SO_n_q.root_system.dynkin_type == expected_type, \
                    f"SO(n={n}, q={q}) is type {expected_type} but computations " + \
                    f"gave type {SO_n_q.root_system.dynkin_type}"
            SO_n_q.validate_pinning(display = True)
    print("Done with non-split special orthogonal groups")
    print("\n" + '=' * 100 + "\n")
    
def run_SU_quasisplit_tests(n_min, n_max, q_min, q_max, eps_values):
    ############################################################
    ## QUASI-SPLIT SPECIAL UNITARY GROUPS (n=2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\n" + '=' * 100 + "\n")
    print("Running calculations and verifications for quasi-split special unitary groups")
    d = sp.symbols('d', nonzero = True)
    p_e = sp.sqrt(d)
    for q in range(q_min, q_max + 1):
        n=2*q
        if n_min <= n and n <= n_max:
            for eps in eps_values:
                anisotropic_vec = vector_variable('c',n-2*q)
                if eps == -1:
                    anisotropic_vec = anisotropic_vec * p_e
                NIF = nondegenerate_isotropic_form(dimension = n,
                                                    witt_index = q,
                                                    anisotropic_vector = anisotropic_vec,
                                                    epsilon = eps,
                                                    primitive_element = p_e)
                T = split_torus(matrix_size = n,
                                rank = q,
                                is_element = is_torus_element_SU,
                                generic_element = generic_torus_element_SU,
                                trivial_character_matrix = trivial_characters_SU(n,q),
                                nontrivial_character_entries = character_entries_SU(n,q))     
                SU_n_q = pinned_group(name_string = f"SU(n={n}, q={q}, eps={eps})",
                                      matrix_size = n,
                                      form = NIF, 
                                      group_constraints = group_constraints_SU,
                                      maximal_split_torus = T,
                                      lie_algebra_constraints = lie_algebra_constraints_SU,
                                      generic_lie_algebra_element = generic_lie_algebra_element_SU,
                                      non_variables = {d})
                SU_n_q.fit_pinning(display = True)
                if q == 1:
                    expected_type = 'A'
                elif q == 2:
                    expected_type = 'B'
                else:
                    expected_type = 'C'
                assert SU_n_q.root_system.dynkin_type == expected_type, \
                        f"SU(n={n}, q={q}, eps={eps}) is type {expected_type} but computations " + \
                        f"gave type {SU_n_q.root_system.dynkin_type}"
                SU_n_q.validate_pinning(display = True)
    print("Done with quasi-split special unitary groups")
    print("\n" + '=' * 100 + "\n")

def run_SU_nonquasisplit_tests(n_min, n_max, q_min, q_max, eps_values):
    ############################################################
    ## NON-QUASI-SPLIT SPECIAL UNITARY GROUPS (n>2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\n" + '=' * 100 + "\n")
    print("Running calculations and verifications for non-(quasi-split) special unitary groups")
    d = sp.symbols('d', nonzero = True)
    p_e = sp.sqrt(d)
    q_min = max(q_min, 2) # doesn't make sense if q=1, there are no roots
    for q in range(q_min , q_max+1):
        n_min = max(2*q+1, n_min) # only non-quasi-split if n>=2*q+1
        for n in range(n_min, n_max + 1):
            for eps in eps_values:
                anisotropic_vec = vector_variable('c',n-2*q)
                if eps == -1:
                    anisotropic_vec = anisotropic_vec * p_e
                NIF = nondegenerate_isotropic_form(dimension = n,
                                                    witt_index = q,
                                                    anisotropic_vector = anisotropic_vec,
                                                    epsilon = eps,
                                                    primitive_element = p_e)
                T = split_torus(matrix_size = n,
                                rank = q,
                                is_element = is_torus_element_SU,
                                generic_element = generic_torus_element_SU,
                                trivial_character_matrix = trivial_characters_SU(n,q),
                                nontrivial_character_entries = character_entries_SU(n,q))     
                SU_n_q = pinned_group(name_string = f"SU(n={n}, q={q}, eps={eps})",
                                      matrix_size = n,
                                      form = NIF, 
                                      group_constraints = group_constraints_SU,
                                      maximal_split_torus = T,
                                      lie_algebra_constraints = lie_algebra_constraints_SU,
                                      generic_lie_algebra_element = generic_lie_algebra_element_SU,
                                      non_variables = {d})
                SU_n_q.fit_pinning(display = True)
                assert SU_n_q.root_system.dynkin_type == 'BC', \
                    f"SU(n={n}, q={q}, eps={eps}) is type BC but computations " + \
                    f"gave type {SU_n_q.root_system.dynkin_type}"
                SU_n_q.validate_pinning(display = True)
    print("\nDone with non-(quasi-split) special unitary groups")
    print("\n" + '=' * 100 + "\n")

if __name__ == "__main__":
    main()