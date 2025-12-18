# Use pinned_group class to work out some examples.

from utility_general import is_diagonal, custom_conjugate
from utility_general import formula_files, main_var_1, main_var_2, anisotropic_var, primitive_element
from utility_SL import (is_group_element_SL, 
                        is_torus_element_SL, 
                        generic_torus_element_SL, 
                        is_lie_algebra_element_SL, 
                        generic_lie_algebra_element_SL)
from utility_SO import (is_group_element_SO, 
                        is_torus_element_SO, 
                        generic_torus_element_SO, 
                        is_lie_algebra_element_SO, 
                        generic_lie_algebra_element_SO)
from utility_SU import (is_group_element_SU, 
                        is_torus_element_SU, 
                        generic_torus_element_SU, 
                        is_lie_algebra_element_SU, 
                        generic_lie_algebra_element_SU)
from pinned_group import pinned_group
from nondegenerate_isotropic_form import nondegenerate_isotropic_form
import sympy as sp
import pickle

def main():
    print("Demonstrating usage of the pinned_group class")
    sp.init_printing(wrap_line=False)
    
    #########################
    ## SPECIAL LINEAR GROUPS              
    #########################
    print("\nRunning calculations and verifications for special linear groups")
    for n in (2,3,4):
        SL_n = pinned_group(name_string = f"SL(n={n})",
                            matrix_size = n,
                            rank = n-1,
                            form = None,
                            is_group_element = is_group_element_SL,
                            is_torus_element = is_torus_element_SL,
                            generic_torus_element = generic_torus_element_SL,
                            is_lie_algebra_element = is_lie_algebra_element_SL,
                            generic_lie_algebra_element = generic_lie_algebra_element_SL)
        SL_n.fit_pinning(display = True)
        
        print("\nRunning tests to verify the results of calculated pinning information")        
        SL_n.verify_pinning(display = True)
    print("Done with special linear groups.\n")

    ###################################################
    ## SPLIT SPECIAL ORTHOGONAL GROUPS (n=2q or n=2q+1) 
    ###################################################
    print("\nRunning calculations and verifications for split special orthogonal groups")
    for q in (2,3):
        for n in (2*q,2*q+1):
            anisotropic_vec = sp.Matrix(sp.symarray(anisotropic_var,n-2*q))
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
                                is_lie_algebra_element = is_lie_algebra_element_SO,
                                generic_lie_algebra_element = generic_lie_algebra_element_SO)
            SO_n_q.fit_pinning(display = True)
            
            print("\nRunning tests to verify the results of calculated pinning information")        
            SO_n_q.verify_pinning(display = True)
        print("Done with split special orthogonal groups.\n")
        
    #############################################################################
    ## NON-SPLIT SPECIAL ORTHOGONAL GROUPS 
        ## SO_n_q is quasi-split if n=2q+2, and
        ## neither split nor quasi-split if n>2+2q, 
        ## but the behavior seems to be basically the same in these two cases
    #############################################################################
    print("\nRunning calculations and verifications for non-split special orthogonal groups")
    for q in (1,2):
        for n in (2*q+2,2*q+3,2*q+4):
            anisotropic_vec = sp.Matrix(sp.symarray(anisotropic_var,n-2*q))
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
                                  is_lie_algebra_element = is_lie_algebra_element_SO,
                                  generic_lie_algebra_element = generic_lie_algebra_element_SO)
            SO_n_q.fit_pinning(display = True)
            
            print("\nRunning tests to verify the results of calculated pinning information")        
            SO_n_q.verify_pinning(display = True)
    print("Done with non-split special orthogonal groups.\n")

    ############################################################
    ## QUASI-SPLIT SPECIAL UNITARY GROUPS (n=2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\nRunning calculations and verifications for quasi-split special unitary groups")
    d = sp.symbols('d')
    p_e = sp.sqrt(d)
    for eps in (1,-1):
        for q in (2,3):
            n=2*q
            anisotropic_variables = sp.symbols(f"c:{n-2*q}")
            NIF = nondegenerate_isotropic_form(dimension = n,
                                                witt_index = q,
                                                anisotropic_vector = anisotropic_variables,
                                                epsilon = eps,
                                                primitive_element = p_e)
            
            SU_n_q = pinned_group(name_string = f"SU(n={n}, q={q}, eps={eps})",
                                  matrix_size = n,
                                  rank = q,
                                  form = NIF, 
                                  is_group_element = is_group_element_SU, 
                                  is_torus_element = is_torus_element_SU,
                                  generic_torus_element = generic_torus_element_SU,
                                  is_lie_algebra_element = is_lie_algebra_element_SU,
                                  generic_lie_algebra_element = generic_lie_algebra_element_SU)
            SU_n_q.fit_pinning(display = True)
            
            print("\nRunning tests to verify the results of calculated pinning information")        
            SU_n_q.verify_pinning(display = True)
    print("Done with quasi-split special unitary groups.\n")
    
    ############################################################
    ## NON-QUASI-SPLIT SPECIAL UNITARY GROUPS (n>2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\nRunning calculations and verifications for non-(quasi-split) special unitary groups")
    d = sp.symbols('d')
    p_e = sp.sqrt(d)
    for eps in (1,-1):
        for q in (2,3):
            for n in (2*q+1,2*q+2):
                anisotropic_variables = sp.symbols(f"c:{n-2*q}")
                NIF = nondegenerate_isotropic_form(dimension = n,
                                                    witt_index = q,
                                                    anisotropic_vector = anisotropic_variables,
                                                    epsilon = eps,
                                                    primitive_element = p_e)
                SU_n_q = pinned_group(name_string = f"SU(n={n}, q={q}, eps={eps})",
                                      matrix_size = n,
                                      rank = q,
                                      form = NIF, 
                                      is_group_element = is_group_element_SU, 
                                      is_torus_element = is_torus_element_SU,
                                      generic_torus_element = generic_torus_element_SU,
                                      is_lie_algebra_element = is_lie_algebra_element_SU,
                                      generic_lie_algebra_element = generic_lie_algebra_element_SU)
                SU_n_q.fit_pinning(display = True)
                
                print("\nRunning tests to verify the results of calculated pinning information")        
                SU_n_q.verify_pinning(display = True)
    print("Done with non-(quasi-split) special unitary groups.\n")

    print("\nAll demonstrations complete.")
    
        
if __name__ == "__main__":
    main()