# Compute and store data on a list of classical groups. For each group,
#   1. Create a split_torus object
#   2. Create a pinned_group object
#   3. Fit a pinning for that pinned_group object
#   4. Store the pinned_group object

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
                        group_constraints_SL_string,
                        is_torus_element_SL,
                        generic_torus_element_SL,
                        trivial_characters_SL,
                        character_entries_SL,
                        lie_algebra_constraints_SL,
                        lie_algebra_constraints_SL_string,
                        generic_lie_algebra_element_SL)
from utility_SO import (group_constraints_SO,
                        group_constraints_SO_string,
                        is_torus_element_SO,
                        generic_torus_element_SO,
                        trivial_characters_SO,
                        character_entries_SO,
                        lie_algebra_constraints_SO,
                        lie_algebra_constraints_SO_string,
                        generic_lie_algebra_element_SO)
from utility_SU import (group_constraints_SU,
                        group_constraints_SU_string,
                        is_torus_element_SU,
                        generic_torus_element_SU, 
                        trivial_characters_SU,
                        character_entries_SU,
                        lie_algebra_constraints_SU,
                        lie_algebra_constraints_SU_string,
                        generic_lie_algebra_element_SU)

def main():
    
    ##################
    overwrite = False 
    ##################
    
    ########################
    n_min = 1
    n_max = 5
    q_min = 1
    q_max = 3
    eps_values = [-1,1]
    ########################
    
    print(f"\nBuilding pinned groups with overwrite = {overwrite}")
    start_time = time.perf_counter()
    sp.init_printing(wrap_line = False)
    
    #######################################################################################
    build_and_store_SL(n_min, n_max, overwrite)
    build_and_store_SO_split(n_min, n_max, q_min, q_max, overwrite)
    build_and_store_SO_nonsplit(n_min, n_max, q_min, q_max, overwrite)
    build_and_store_SU_quasisplit(n_min, n_max, q_min, q_max, eps_values, overwrite)
    build_and_store_SU_nonquasisplit(n_min, n_max, q_min, q_max, eps_values, overwrite)
    #######################################################################################
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\nAll constructions complete, total time: {round(execution_time/60, 1)} minutes")

def write_group(G, overwrite):
    if G.file_exists():
        print(f"File already exists for {G.name_string}")
        if not overwrite:
            print(f"Skipping {G.name_string}")
        else:
            print(f"Creating a new file to overwrite {G.name_string}")
            G.fit_pinning(display = False)
            G.write_to_file(overwrite)
    else:
        print(f"File does not yet exist for {G.name_string}")
        G.fit_pinning(display = False)
        G.write_to_file(overwrite)

def build_and_store_SL(n_min, n_max, overwrite):
    print("\nSpecial linear groups")
    n_min = max(n_min, 2) # n=1 doesn't make sense, SL_1 is just the trivial group
    for n in range(n_min, n_max + 1):
        T = split_torus(matrix_size = n,
                        rank = n-1,
                        is_element = is_torus_element_SL,
                        generic_element = generic_torus_element_SL,
                        trivial_character_matrix = trivial_characters_SL(n),
                        nontrivial_character_entries = character_entries_SL(n))
        SL_n = pinned_group(name_string = f"SL-n{n}",
                            matrix_size = n,
                            form = None,
                            group_constraints = group_constraints_SL,
                            group_constraints_string = group_constraints_SL_string,
                            maximal_split_torus = T,
                            lie_algebra_constraints = lie_algebra_constraints_SL,
                            lie_algebra_constraints_string = lie_algebra_constraints_SL_string,
                            generic_lie_algebra_element = generic_lie_algebra_element_SL,
                            non_variables = None)
        write_group(SL_n, overwrite)

def build_and_store_SO_split(n_min, n_max, q_min, q_max, overwrite):
    ###################################################
    ## SPLIT SPECIAL ORTHOGONAL GROUPS (n=2q or n=2q+1) 
    ###################################################
    print("\nSplit special orthogonal groups")
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
            SO_n_q = pinned_group(name_string = f"SO-n{n}-q{q}",
                                matrix_size = n,
                                form = NIF, 
                                group_constraints = group_constraints_SO,
                                group_constraints_string = group_constraints_SO_string,
                                maximal_split_torus = T,
                                lie_algebra_constraints = lie_algebra_constraints_SO,
                                lie_algebra_constraints_string = lie_algebra_constraints_SO_string,
                                generic_lie_algebra_element = generic_lie_algebra_element_SO,
                                non_variables = None)
            write_group(SO_n_q, overwrite)

def build_and_store_SO_nonsplit(n_min, n_max, q_min, q_max, overwrite):
    #############################################################################
    ## NON-SPLIT SPECIAL ORTHOGONAL GROUPS 
    ## SO_n_q is quasi-split if n=2q+2, and
    ## neither split nor quasi-split if n>2+2q, 
    ## but the behavior seems to be basically the same in these two cases
    #############################################################################
    print("\nNon-split special orthogonal groups")
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
            SO_n_q = pinned_group(name_string = f"SO-n{n}-q{q}",
                                matrix_size = n,
                                form = NIF, 
                                group_constraints = group_constraints_SO,
                                group_constraints_string = group_constraints_SO_string,
                                maximal_split_torus = T,
                                lie_algebra_constraints = lie_algebra_constraints_SO,
                                lie_algebra_constraints_string = lie_algebra_constraints_SO_string,
                                generic_lie_algebra_element = generic_lie_algebra_element_SO,
                                non_variables = None)
            write_group(SO_n_q, overwrite)
    
def build_and_store_SU_quasisplit(n_min, n_max, q_min, q_max, eps_values, overwrite):
    ############################################################
    ## QUASI-SPLIT SPECIAL UNITARY GROUPS (n=2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\nQuasi-split special unitary groups")
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
                eps_str = "1" if eps > 0 else "minus1"
                SU_n_q = pinned_group(name_string = f"SU-n{n}-q{q}-eps-{eps_str}",
                                      matrix_size = n,
                                      form = NIF, 
                                      group_constraints = group_constraints_SU,
                                      group_constraints_string = group_constraints_SU_string,
                                      maximal_split_torus = T,
                                      lie_algebra_constraints = lie_algebra_constraints_SU,
                                      lie_algebra_constraints_string = lie_algebra_constraints_SU_string,
                                      generic_lie_algebra_element = generic_lie_algebra_element_SU,
                                      non_variables = {d})
                write_group(SU_n_q, overwrite)
                
def build_and_store_SU_nonquasisplit(n_min, n_max, q_min, q_max, eps_values, overwrite):
    ############################################################
    ## NON-QUASI-SPLIT SPECIAL UNITARY GROUPS (n>2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\nNon-(quasi-split) special unitary groups")
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
                eps_str = "1" if eps > 0 else "minus1"
                SU_n_q = pinned_group(name_string = f"SU-n{n}-q{q}-eps-{eps_str}",
                                      matrix_size = n,
                                      form = NIF, 
                                      group_constraints = group_constraints_SU,
                                      group_constraints_string = group_constraints_SU_string,
                                      maximal_split_torus = T,
                                      lie_algebra_constraints = lie_algebra_constraints_SU,
                                      lie_algebra_constraints_string = lie_algebra_constraints_SU_string,
                                      generic_lie_algebra_element = generic_lie_algebra_element_SU,
                                      non_variables = {d})
                write_group(SU_n_q, overwrite)

if __name__ == "__main__":
    main()