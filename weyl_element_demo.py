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


def root_subgroup_formula_applies(G, alpha):
    """Return whether the direct rank-one formula is defined.

    Input:
        G: a partially fitted pinned_group with root spaces and root subgroups.
        alpha: a root of G.root_system.
    Output:
        True if both U_alpha and U_-alpha are one-dimensional; otherwise False.
    """
    assert G.root_system.is_root(alpha), \
        "Cannot construct a Weyl element from a non-root"
    return (
        G.root_space_dimension(alpha) == 1
        and G.root_space_dimension(-alpha) == 1
    )


def construct_weyl_element_from_root_subgroups(G, alpha):
    """Construct a Weyl representative using the root-subgroup formula.

    Input:
        G: a partially fitted pinned_group with root subgroup maps available.
        alpha: a root of G.root_system.
    Output:
        The matrix x_alpha(1) x_-alpha(-1) x_alpha(1), or None if the
        direct formula is not canonical for this root.
    """
    if not root_subgroup_formula_applies(G, alpha):
        return None

    one = sp.Matrix([1])
    minus_one = sp.Matrix([-1])
    return sp.simplify(
        G.root_subgroup_map(alpha, one)
        * G.root_subgroup_map(-alpha, minus_one)
        * G.root_subgroup_map(alpha, one)
    )


def validate_weyl_candidate_is_group_element(G, w_alpha):
    """Check that a candidate Weyl element lies in the matrix group.

    Input:
        G: a pinned_group.
        w_alpha: a matrix candidate for one Weyl element.
    Output:
        True if w_alpha satisfies G's group equations; otherwise False.
    """
    return G.is_group_element(w_alpha)


def validate_weyl_candidate_normalizes_torus(G, w_alpha):
    """Check that a candidate Weyl element normalizes the split torus.

    Input:
        G: a pinned_group with a generic torus element available.
        w_alpha: a matrix candidate for one Weyl element.
    Output:
        True if w_alpha * t * w_alpha^-1 is a torus element for generic t.
    """
    t = G.generic_torus_element('t')
    conjugation = sp.simplify(w_alpha * t * w_alpha.inv())
    return G.is_torus_element(conjugation)


def get_reflected_root_data(G, alpha, beta, root_data_cache=None):
    """Return root-system data used in one Weyl conjugation check.

    Input:
        G: a pinned_group with root-space dimensions available.
        alpha: the root defining the Weyl reflection.
        beta: the root being reflected.
        root_data_cache: optional dict for memoizing data inside one pass.
    Output:
        A tuple (gamma, d_beta, d_gamma), where gamma = sigma_alpha(beta).
    """
    key = (alpha, beta)
    if root_data_cache is not None and key in root_data_cache:
        return root_data_cache[key]

    gamma = G.root_system.reflect_root(alpha, beta)
    root_data = (
        gamma,
        G.root_space_dimension(beta),
        G.root_space_dimension(gamma),
    )
    if root_data_cache is not None:
        root_data_cache[key] = root_data
    return root_data


def solve_weyl_conjugation_for_beta(G, alpha, beta, w_alpha, root_data_cache=None):
    """Solve the root-subgroup conjugation equation for one beta.

    Input:
        G: a pinned_group with root subgroup maps available.
        alpha: the root defining the candidate Weyl element.
        beta: the root subgroup to conjugate.
        w_alpha: the matrix candidate for the alpha Weyl element.
        root_data_cache: optional dict shared across beta checks.
    Output:
        A solution dict for v in
        w_alpha * x_beta(u) * w_alpha^-1 = x_sigma_alpha(beta)(v),
        or None if the equation cannot be solved.
    """
    gamma, d_beta, d_gamma = get_reflected_root_data(
        G, alpha, beta, root_data_cache
    )
    if d_beta != d_gamma:
        return None

    u = vector_variable('u', d_beta)
    v = vector_variable('v', d_gamma)
    x_beta_u = G.root_subgroup_map(beta, u)
    x_gamma_v = G.root_subgroup_map(gamma, v)
    lhs = sp.simplify(w_alpha * x_beta_u * w_alpha.inv())

    try:
        sols = sp.solve(lhs - x_gamma_v, v.free_symbols, dict=True)
    except Exception:
        return None
    if len(sols) == 0:
        return None
    return sols[0]


def validate_weyl_candidate_conjugates_root_subgroups(G, alpha, w_alpha):
    """Check all root-subgroup conjugation equations for one candidate.

    Input:
        G: a pinned_group with root subgroup maps available.
        alpha: the root defining the candidate Weyl element.
        w_alpha: the matrix candidate for the alpha Weyl element.
    Output:
        True if every root subgroup U_beta is sent to
        U_sigma_alpha(beta); otherwise False.
    """
    root_data_cache = {}
    for beta in G.root_system.root_list:
        sol = solve_weyl_conjugation_for_beta(
            G, alpha, beta, w_alpha, root_data_cache
        )
        if sol is None:
            return False
    return True


def is_valid_weyl_element_candidate(G, alpha, w_alpha):
    """Run the shared validation checks for one candidate Weyl element.

    Input:
        G: a pinned_group with torus and root subgroup data available.
        alpha: the root defining the candidate Weyl element.
        w_alpha: the matrix candidate to validate.
    Output:
        True if w_alpha is in G, normalizes the torus, and conjugates each
        U_beta into U_sigma_alpha(beta); otherwise False.
    """
    assert G.root_system.is_root(alpha), \
        "Cannot validate a Weyl element candidate for a non-root"
    return (
        validate_weyl_candidate_is_group_element(G, w_alpha)
        and validate_weyl_candidate_normalizes_torus(G, w_alpha)
        and validate_weyl_candidate_conjugates_root_subgroups(G, alpha, w_alpha)
    )


def construct_validated_weyl_element_from_root_subgroups(G, alpha):
    """Construct and validate one root-subgroup Weyl element.

    Input:
        G: a partially fitted pinned_group.
        alpha: a root of G.root_system.
    Output:
        (w_alpha, None) on success, or (None, failure_reason) on failure.
    """
    w_alpha = construct_weyl_element_from_root_subgroups(G, alpha)
    if w_alpha is None:
        return None, (
            "direct root-subgroup formula requires one-dimensional "
            f"opposite root spaces; skipped alpha = {alpha}"
        )
    if not is_valid_weyl_element_candidate(G, alpha, w_alpha):
        return None, (
            "direct root-subgroup Weyl construction failed validation "
            f"for alpha = {alpha}"
        )
    return w_alpha, None


def fit_torus_reflections_from_coroots(G, display=True):
    """Install the torus reflection map s_alpha on G.

    Input:
        G: a partially fitted pinned_group with coroots available.
        display: whether to print progress.
    Output:
        None. Mutates G by assigning G.torus_reflection_map.
    """
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


def construct_validated_weyl_elements_from_root_subgroups(G):
    """Construct and validate root-subgroup Weyl elements for every root.

    Input:
        G: a partially fitted pinned_group.
    Output:
        (weyl_elements, None) on success, where weyl_elements maps roots to
        matrices; otherwise (None, failure_reason). This function does not
        mutate G.
    """
    weyl_elements = {}
    for alpha in G.root_system.root_list:
        w_alpha, failure_reason = construct_validated_weyl_element_from_root_subgroups(
            G, alpha
        )
        if failure_reason is not None:
            return None, failure_reason
        weyl_elements[alpha] = w_alpha
    return weyl_elements, None


def install_weyl_element_map(G, weyl_elements):
    """Install a completed Weyl element dictionary on G.

    Input:
        G: a pinned_group to mutate.
        weyl_elements: dict mapping each root alpha to its matrix w_alpha.
    Output:
        None. Mutates G by assigning G.weyl_element_list and G.weyl_element_map.
    """
    G.weyl_element_list = weyl_elements

    def wem(alpha):
        return G.weyl_element_list[alpha]

    G.weyl_element_map = wem


def try_fit_weyl_group_elements_from_root_subgroups(G, display=True):
    """Try the root-subgroup Weyl method without raising on method failure.

    Input:
        G: a partially fitted pinned_group.
        display: whether to print progress and failure reasons.
    Output:
        True if Weyl elements were constructed and installed; False if this
        method does not apply or fails validation.
    """
    fit_torus_reflections_from_coroots(G, display)
    if display:
        print("Trying Weyl elements (w_α) from root subgroups")

    weyl_elements, failure_reason = construct_validated_weyl_elements_from_root_subgroups(G)
    if failure_reason is not None:
        if display:
            print(f"Root-subgroup Weyl fast path skipped: {failure_reason}")
        return False

    install_weyl_element_map(G, weyl_elements)
    return True


def fit_weyl_group_elements_from_root_subgroups(G, display=True):
    """Fit Weyl elements using the strict root-subgroup demo method.

    Input:
        G: a partially fitted pinned_group.
        display: whether to print progress.
    Output:
        None. Mutates G by installing torus reflections and Weyl elements.
        Raises AssertionError if the root-subgroup method cannot handle G.
    """
    fit_torus_reflections_from_coroots(G, display)
    if display:
        print("Fitting Weyl elements (w_α) from root subgroups")

    weyl_elements, failure_reason = construct_validated_weyl_elements_from_root_subgroups(G)
    assert failure_reason is None, failure_reason
    install_weyl_element_map(G, weyl_elements)
