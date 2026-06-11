import sympy as sp
import numpy as np
from utility_general import vector_variable, pretty_map
from nondegenerate_isotropic_form import nondegenerate_isotropic_form
from split_torus import split_torus
from pinned_group import pinned_group
from utility_SU import (group_constraints_SU,
                        is_torus_element_SU,
                        generic_torus_element_SU, 
                        trivial_characters_SU,
                        character_entries_SU,
                        lie_algebra_constraints_SU,
                        generic_lie_algebra_element_SU)
from utility_roots import visualize_graph
import pandas as pd
import os
import dill

n = 5
q = 2
d = sp.symbols('d', nonzero = True)
p_e = sp.sqrt(d)
eps = 1
anisotropic_vec = vector_variable('c',n-2*q)
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

filename = f"SU_{n}_{q}_{eps}.pkl"
if os.path.exists(filename):
    print(f"Loading pre-computed pinned group from {filename}...")
    with open(filename, 'rb') as f:
        SU_n_q = dill.load(f)
else:
    print("Computing pinned group (this may take a while)...")
    SU_n_q = pinned_group(
        name_string=f"SU(n={n}, q={q}, eps={eps})",
        matrix_size=n,
        form=NIF, 
        group_constraints=group_constraints_SU,
        maximal_split_torus=T,
        lie_algebra_constraints=lie_algebra_constraints_SU,
        generic_lie_algebra_element=generic_lie_algebra_element_SU,
        non_variables={d})
    
    # Run the expensive fitting process
    SU_n_q.fit_pinning(display=False)
    
    print(f"Saving computed group to {filename}...")
    with open(filename, 'wb') as f:
        dill.dump(SU_n_q, f) # Use dill here too

print("Form matrix:")
sp.pprint(NIF.matrix)
print()
print(T)

BC_q = SU_n_q.root_system

print("\nRoot system:",SU_n_q.root_system.name_string)
print("Dynkin diagram:",visualize_graph(BC_q.dynkin_graph))

long_roots = [alpha for alpha in BC_q.root_list if alpha.dot(alpha) == 4]
medium_roots = [alpha for alpha in BC_q.root_list if alpha.dot(alpha) == 2]
short_roots = [alpha for alpha in BC_q.root_list if alpha.dot(alpha) == 1]

# print("Number of roots:",len(BC_q.root_list))
# print("\tLong roots:")
# for alpha in long_roots:
#     print("\t\t",alpha)
# print("\tMedium roots:")
# for alpha in medium_roots:
#     print("\t\t",alpha)
# print("\tShort roots:")
# for alpha in short_roots:
#     print("\t\t",alpha)

t = SU_n_q.generic_torus_element('t')
tt = sp.symbols('t')
root_table = []
for alpha in SU_n_q.root_space_dict:
    sq_length = alpha.dot(alpha)
    alpha_check = SU_n_q.root_system.coroot_dict[alpha]
    d_alpha = SU_n_q.root_space_dimension(alpha)
    u = vector_variable(letter = 'u', length = d_alpha)
    X_alpha_u = sp.pretty(SU_n_q.root_space_map(alpha, u))
    x_alpha_u = sp.pretty(SU_n_q.root_subgroup_map(alpha, u))
    s_alpha = pretty_map(t, SU_n_q.torus_reflection_map(alpha, t))
    w_alpha = sp.pretty(SU_n_q.weyl_element_map(alpha))
    h_alpha_t = sp.pretty(SU_n_q.coroot_torus_element_map(alpha, tt))
    mult = SU_n_q.root_system.is_multipliable_root(alpha)
    hdc = SU_n_q.homomorphism_defect_coefficient_dict[alpha][2] if mult else "NA"
    root_table.append([alpha, sq_length, mult, alpha_check, d_alpha, X_alpha_u, x_alpha_u, s_alpha, w_alpha, h_alpha_t, hdc])

root_headers = ["alpha", "sq_length", "multipliable", "alpha_check", "d_alpha", "X_alpha(u)", "x_alpha(u)", "s_alpha", "w_alpha", "h_alpha(t)", "Hom defect"]
root_df = pd.DataFrame(root_table, columns = root_headers)

long_root_df = root_df[root_df["sq_length"] == 4]
medium_root_df = root_df[root_df["sq_length"] == 2]
short_root_df = root_df[root_df["sq_length"] == 1]

columns_to_print = ['alpha', 'alpha_check', 'multipliable', 'sq_length', 'd_alpha']

print("\nLong roots")
print(long_root_df[columns_to_print].to_string(index=False))

print("\nMedium roots")
print(medium_root_df[columns_to_print].to_string(index=False))

print("\nShort roots")
print(short_root_df[columns_to_print].to_string(index=False))

print(f"\nRoots and associated data for {SU_n_q.name_string}:")
print(SU_n_q.get_roots_table())

print(f"\nCommutator coefficients for {SU_n_q.name_string}:")
print(SU_n_q.get_commutator_table())

# print(f"\nWeyl conjugation coefficients for {SU_n_q.name_string}:")
# print(SU_n_q.get_weyl_conjugation_table())


