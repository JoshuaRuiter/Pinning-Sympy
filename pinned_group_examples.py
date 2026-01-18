# Use pinned_group class to work out some examples.

from utility_SL import (is_group_element_SL,
                        is_torus_element_SL,
                        generic_torus_element_SL,
                        trivial_characters_SL,
                        is_lie_algebra_element_SL, 
                        generic_lie_algebra_element_SL)
from utility_SO import (is_group_element_SO,
                        is_torus_element_SO,
                        generic_torus_element_SO,
                        trivial_characters_SO,
                        is_lie_algebra_element_SO, 
                        generic_lie_algebra_element_SO)
from utility_SU import (is_group_element_SU,
                        is_torus_element_SU,
                        generic_torus_element_SU, 
                        trivial_characters_SU,
                        is_lie_algebra_element_SU, 
                        generic_lie_algebra_element_SU)
from pinned_group import pinned_group
from nondegenerate_isotropic_form import nondegenerate_isotropic_form
import sympy as sp


def main():
    to_do_list = ("To do list:" + "\n\t" + 
                  "Fix root system stuff - SL(3) is coming out to B2 root system!!!" + "\n\t" +
                  "Convert everything to new bidict format for root lists, with orthogonal complement coordinates" + "\n\t" +
                  "Fix issue in root space maps which convert set to list and assume order is preserved" + "\n\t" +
                  "Check that angle bracket testing is doing the proper thing with new root system conventions" + "\n\t" +
                  "Fix root system reflection method" + "\n\t" +
                  "Fix dynkin type classifier to use quotient images of roots" + "\n\t" +
                  "Fix linear combos to work with new root system conventions" + "\n\t" +
                  "Switch to manual tuple operations rather than numpy conversions" + "\n\t" +
                  "Fix root_system.irreducible_components" + "\n\t" +
                  "Fix root system tests to be based on quotient projections, not original roots" + "\n\t" +
                  "Do I need to fix linear combos to work with new convention?" + "\n\t" +
                  "Set up so that generic_torus_element and such don't require redundant inputs " + 
                      "when used within pinned_group" + "\n\t" +
                  "Compute Weyl group elements" + "\n\t" +
                  "Run tests to validate properties of Weyl group elements" + "\n\t" +
                  "Maybe I need an implementation of the Jacobson-Morozov theorem," +
                      "i.e. an algorithm that generates sl2-triples" + "\n\t" +
                  "Implement capability for printing to file instead of to console" + "\n\t" +
                  "Add functionality to root_system class to construct standard " + 
                      "models of root systems based on given Dynkin type" + "\n\t" +
                  "Add documentation, including Readme on Github")
    print(to_do_list)
    
    print("\nDemonstrating usage of the pinned_group class")
    sp.init_printing(wrap_line=False)
    run_SL_tests()
    run_SO_split_tests()
    run_SO_nonsplit_tests()
    run_SU_quasisplit_tests()
    run_SU_nonquasisplit_tests()
    print("\nAll tests complete.")
    
    print(to_do_list)
    
def run_SL_tests():
    print("\n" + '≡' * 100)
    print("\nRunning calculations and verifications for special linear groups")
    for n in (2,3,4):
        SL_n = pinned_group(name_string = f"SL(n={n})",
                            matrix_size = n,
                            rank = n-1,
                            form = None,
                            is_group_element = is_group_element_SL,
                            is_torus_element = is_torus_element_SL,
                            generic_torus_element = generic_torus_element_SL,
                            trivial_character_matrix = trivial_characters_SL(n),
                            is_lie_algebra_element = is_lie_algebra_element_SL,
                            generic_lie_algebra_element = generic_lie_algebra_element_SL,
                            non_variables = None)
        SL_n.fit_pinning(display = True)
        SL_n.validate_pinning(display = True)
    print("Done with special linear groups.\n")
    print('≡' * 100 + "\n")

def run_SO_split_tests():
    ###################################################
    ## SPLIT SPECIAL ORTHOGONAL GROUPS (n=2q or n=2q+1) 
    ###################################################
    print("\n" + '≡' * 100)
    print("\nRunning calculations and verifications for split special orthogonal groups")
    for q in (2,3):
        for n in (2*q,2*q+1):
            anisotropic_vec = sp.Matrix(sp.symarray('c',n-2*q))
            NIF = nondegenerate_isotropic_form(dimension = n,
                                                witt_index = q,
                                                anisotropic_vector = anisotropic_vec,
                                                epsilon = None,
                                                primitive_element = None)
            SO_n_q = pinned_group(name_string = f"SO(n={n}, q={q})",
                                matrix_size = n,
                                rank = q,
                                form = NIF, 
                                is_group_element = is_group_element_SO,
                                is_torus_element = is_torus_element_SO,
                                generic_torus_element = generic_torus_element_SO,
                                trivial_character_matrix = trivial_characters_SO(n,q), 
                                is_lie_algebra_element = is_lie_algebra_element_SO,
                                generic_lie_algebra_element = generic_lie_algebra_element_SO,
                                non_variables = None)
            SO_n_q.fit_pinning(display = True)     
            SO_n_q.validate_pinning(display = True)
    print("Done with split special orthogonal groups.\n")
    print('≡' * 100 + "\n")

def run_SO_nonsplit_tests():
    #############################################################################
    ## NON-SPLIT SPECIAL ORTHOGONAL GROUPS 
        ## SO_n_q is quasi-split if n=2q+2, and
        ## neither split nor quasi-split if n>2+2q, 
        ## but the behavior seems to be basically the same in these two cases
    #############################################################################
    print("\n" + '≡' * 100)
    print("\nRunning calculations and verifications for non-split special orthogonal groups")
    for q in (1,2):
        for n in (2*q+2,2*q+3,2*q+4):
            anisotropic_vec = sp.Matrix(sp.symarray('c',n-2*q))
            NIF = nondegenerate_isotropic_form(dimension = n,
                                                witt_index = q,
                                                anisotropic_vector = anisotropic_vec,
                                                epsilon = None,
                                                primitive_element = None)
            SO_n_q = pinned_group(name_string = f"SO(n={n}, q={q})",
                                  matrix_size = n,
                                  rank = q,
                                  form = NIF, 
                                  is_group_element = is_group_element_SO, 
                                  is_torus_element = is_torus_element_SO,
                                  generic_torus_element = generic_torus_element_SO,
                                  trivial_character_matrix = trivial_characters_SO(n,q),
                                  is_lie_algebra_element = is_lie_algebra_element_SO,
                                  generic_lie_algebra_element = generic_lie_algebra_element_SO,
                                  non_variables = None)
            SO_n_q.fit_pinning(display = True)
            SO_n_q.validate_pinning(display = True)
    print("Done with non-split special orthogonal groups.\n")
    print('≡' * 100 + "\n")
    
def run_SU_quasisplit_tests():
    ############################################################
    ## QUASI-SPLIT SPECIAL UNITARY GROUPS (n=2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\n" + '≡' * 100)
    print("\nRunning calculations and verifications for quasi-split special unitary groups")
    d = sp.symbols('d')
    p_e = sp.sqrt(d)
    for q in (2,3):
        for eps in (1,-1):
            n=2*q
            anisotropic_vec = sp.Matrix(sp.symarray('c',n-2*q))
            if eps == -1:
                anisotropic_vec = anisotropic_vec * p_e
            NIF = nondegenerate_isotropic_form(dimension = n,
                                               witt_index = q,
                                               anisotropic_vector = anisotropic_vec,
                                               epsilon = eps,
                                               primitive_element = p_e)
            SU_n_q = pinned_group(name_string = f"SU(n={n}, q={q}, eps={eps})",
                                  matrix_size = n,
                                  rank = q,
                                  form = NIF, 
                                  is_group_element = is_group_element_SU, 
                                  is_torus_element = is_torus_element_SU,
                                  generic_torus_element = generic_torus_element_SU,
                                  trivial_character_matrix = trivial_characters_SU(n,q),
                                  is_lie_algebra_element = is_lie_algebra_element_SU,
                                  generic_lie_algebra_element = generic_lie_algebra_element_SU,
                                  non_variables = {d})
            SU_n_q.fit_pinning(display = True)   
            SU_n_q.validate_pinning(display = True)
    print("Done with quasi-split special unitary groups.\n")
    print('≡' * 100 + "\n")

def run_SU_nonquasisplit_tests():
    ############################################################
    ## NON-QUASI-SPLIT SPECIAL UNITARY GROUPS (n>2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\n" + '≡' * 100)
    print("\nRunning calculations and verifications for non-(quasi-split) special unitary groups")
    d = sp.symbols('d')
    p_e = sp.sqrt(d)
    for q in (2,3):
        for n in (2*q+1,2*q+2):
            for eps in (1,-1):
                anisotropic_vec = sp.Matrix(sp.symarray('c',n-2*q))
                if eps == -1:
                    anisotropic_vec = anisotropic_vec * p_e
                NIF = nondegenerate_isotropic_form(dimension = n,
                                                    witt_index = q,
                                                    anisotropic_vector = anisotropic_vec,
                                                    epsilon = eps,
                                                    primitive_element = p_e)
                SU_n_q = pinned_group(name_string = f"SU(n={n}, q={q}, eps={eps})",
                                      matrix_size = n,
                                      rank = q,
                                      form = NIF, 
                                      is_group_element = is_group_element_SU, 
                                      is_torus_element = is_torus_element_SU,
                                      generic_torus_element = generic_torus_element_SU,
                                      trivial_character_matrix = trivial_characters_SU(n,q),
                                      is_lie_algebra_element = is_lie_algebra_element_SU,
                                      generic_lie_algebra_element = generic_lie_algebra_element_SU,
                                      non_variables = {d}) 
                SU_n_q.fit_pinning(display = True)
                SU_n_q.validate_pinning(display = True)
    print("Done with non-(quasi-split) special unitary groups.\n")
    print('≡' * 100 + "\n")
        
if __name__ == "__main__":
    main()