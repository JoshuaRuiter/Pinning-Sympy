# import sympy as sp
from pinned_group import pinned_group
# from nondegenerate_isotropic_form import nondegenerate_isotropic_form
from split_torus import split_torus
# from utility_general import vector_variable
from utility_SL import (group_constraints_SL,
                        is_torus_element_SL,
                        generic_torus_element_SL,
                        trivial_characters_SL,
                        character_entries_SL,
                        lie_algebra_constraints_SL,
                        generic_lie_algebra_element_SL)
import os
import dill

n = 3
T = split_torus(matrix_size = n,
                rank = n-1,
                is_element = is_torus_element_SL,
                generic_element = generic_torus_element_SL,
                trivial_character_matrix = trivial_characters_SL(n),
                nontrivial_character_entries = character_entries_SL(n))


print(T)

filename = f"SL_{n}.pkl"
if os.path.exists(filename):
    print(f"Loading pre-computed pinned group from {filename}...")
    with open(filename, 'rb') as f:
        SL_n = dill.load(f)
else:
    print("Computing pinned group (this may take a while)...")
    SL_n = pinned_group(name_string = f"SL(n={n})",
                    matrix_size = n,
                    form = None,
                    group_constraints = group_constraints_SL,
                    maximal_split_torus = T,
                    lie_algebra_constraints = lie_algebra_constraints_SL,
                    generic_lie_algebra_element = generic_lie_algebra_element_SL,
                    non_variables = None)
    
    # Run the expensive fitting process
    SL_n.fit_pinning(display = True)
    
    print(f"Saving computed group to {filename}...")
    with open(filename, 'wb') as f:
        dill.dump(SL_n, f) # Use dill here too


print(SL_n.root_system)

alpha = SL_n.root_system.root_list[0]
