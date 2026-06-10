import sympy as sp

from utility_general import vector_variable
from utility_roots import evaluate_character, evaluate_cocharacter


# This module is intentionally separate from pinned_group.py.
#
# It is a demo/experimental path for constructing Weyl elements using the
# rank-one root-subgroup formula instead of the brute-force Weyl search in the
# core class. Keeping it here lets us test the mathematical shortcut without
# changing the original fitting pipeline.
#
# The method is:
#   1. Try the direct formula w_alpha = X_alpha(1) X_{-alpha}(-1) X_alpha(1).
#   2. Accept it only for one-dimensional root spaces, where the input "1" is
#      canonical.
#   3. Validate the resulting matrix by exact symbolic checks.
#
# Optimization here is deliberately modest:
#   - It avoids the full symbolic matrix/brute-force search when the direct
#     rank-one formula applies.
#   - During validation, repeated root data for fixed (alpha, beta) is stored
#     in a small local dictionary. This is memoization, not backtracking: it
#     does not change the search space, it only avoids recomputing reflection
#     and root-space dimension data.


def construct_weyl_element_from_root_subgroups(G, alpha):
    # Standard rank-one Weyl representative. This is expected to work
    # for one-dimensional root spaces, such as the split SL_n roots.
    assert G.root_system.is_root(alpha), \
        "Cannot construct a Weyl element from a non-root"
    if G.root_space_dimension(alpha) != 1:
        return None
    if G.root_space_dimension(-alpha) != 1:
        return None

    one = sp.Matrix([1])
    minus_one = sp.Matrix([-1])
    return sp.simplify(
        G.root_subgroup_map(alpha, one)
        * G.root_subgroup_map(-alpha, minus_one)
        * G.root_subgroup_map(alpha, one)
    )


def is_valid_weyl_element_candidate(G, alpha, w_alpha):
    # Validation is intentionally exact and group-level. The shortcut is only
    # useful if the constructed matrix satisfies the same properties expected
    # from the brute-force Weyl element:
    #   - w_alpha is in G,
    #   - w_alpha normalizes the torus,
    #   - w_alpha sends each root subgroup U_beta to U_{sigma_alpha(beta)}.
    assert G.root_system.is_root(alpha), \
        "Cannot validate a Weyl element candidate for a non-root"
    if not G.is_group_element(w_alpha):
        return False

    t = G.generic_torus_element('t')
    if not G.is_torus_element(sp.simplify(w_alpha * t * w_alpha.inv())):
        return False

    w_alpha_inverse = w_alpha.inv()
    root_data_cache = {}
    for beta in G.root_system.root_list:
        # Cache root data inside this validation pass. These values are purely
        # determined by the root system and do not depend on the candidate
        # matrix entries.
        key = (alpha, beta)
        if key not in root_data_cache:
            gamma = G.root_system.reflect_root(alpha, beta)
            root_data_cache[key] = (
                gamma,
                G.root_space_dimension(beta),
                G.root_space_dimension(gamma),
            )
        gamma, d_beta, d_gamma = root_data_cache[key]
        if d_beta != d_gamma:
            return False

        u = vector_variable('u', d_beta)
        v = vector_variable('v', d_gamma)
        x_beta_u = G.root_subgroup_map(beta, u)
        x_gamma_v = G.root_subgroup_map(gamma, v)
        lhs = sp.simplify(w_alpha * x_beta_u * w_alpha_inverse)
        try:
            sols = sp.solve(lhs - x_gamma_v, v.free_symbols, dict=True)
        except Exception:
            return False
        if len(sols) == 0:
            return False

    return True


def fit_weyl_group_elements_from_root_subgroups(G, display=True):
    # Experimental alternative Weyl-element fitting method.
    # This lives outside pinned_group.py so it can be tested without changing
    # the core fitting path.
    if display:
        print("Fitting torus reflections (s_α)")

    def torus_refl_map(alpha, t):
        assert G.root_system.is_root(alpha), "Can only perform torus reflection with a root from the root system"
        assert G.is_torus_element(t), "Can only perform torus reflection on a torus element"
        alpha_of_t = evaluate_character(alpha, t)
        alpha_of_t_inverse = alpha_of_t**(-1)
        alpha_check = G.root_system.coroot_dict[alpha]
        alpha_check_of_alpha_of_t_inverse = evaluate_cocharacter(alpha_check, alpha_of_t_inverse)
        assert G.is_torus_element(alpha_check_of_alpha_of_t_inverse), "Cocharacter must return torus element"
        return t * alpha_check_of_alpha_of_t_inverse

    G.torus_reflection_map = torus_refl_map

    if display:
        print("Fitting Weyl elements (w_α) from root subgroups")
    G.weyl_element_list = {}
    for alpha in G.root_system.root_list:
        w_alpha = construct_weyl_element_from_root_subgroups(G, alpha)
        assert w_alpha is not None, \
            f"Direct root-subgroup Weyl construction does not apply to alpha = {alpha}"
        assert is_valid_weyl_element_candidate(G, alpha, w_alpha), \
            f"Direct root-subgroup Weyl construction failed validation for alpha = {alpha}"
        G.weyl_element_list[alpha] = w_alpha

    def wem(alpha):
        return G.weyl_element_list[alpha]

    G.weyl_element_map = wem
