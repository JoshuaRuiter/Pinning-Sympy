from contextlib import contextmanager
import time

import sympy as sp

from utility_general import vector_variable
from utility_roots import evaluate_character, evaluate_cocharacter


# This module is intentionally separate from pinned_group.py.
#
# It is a test/demo path for constructing Weyl elements using the rank-one
# root-subgroup formula instead of the brute-force Weyl search in the core
# class. The original brute-force support/value enumeration still lives in
# pinned_group.py; a factored external copy of that algorithm lives in
# weyl_element_factored.py.
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


def new_weyl_demo_counters():
    """Create counters for the demo Weyl construction path.

    Input:
        None.
    Output:
        A dictionary whose values are integer counters, plus a failure_reason
        field that records the first method failure.
    """
    return {
        "roots_attempted": 0,
        "formula_applicable": 0,
        "candidates_constructed": 0,
        "validation_attempts": 0,
        "group_element_checks": 0,
        "group_element_failures": 0,
        "torus_normalizer_checks": 0,
        "torus_normalizer_failures": 0,
        "conjugation_checks": 0,
        "conjugation_failures": 0,
        "roots_validated": 0,
        "roots_installed": 0,
        "time_total_seconds": 0.0,
        "time_torus_reflections_seconds": 0.0,
        "time_formula_applicability_seconds": 0.0,
        "time_candidate_construction_seconds": 0.0,
        "time_validation_total_seconds": 0.0,
        "time_group_element_checks_seconds": 0.0,
        "time_torus_normalizer_checks_seconds": 0.0,
        "time_conjugation_checks_seconds": 0.0,
        "time_install_seconds": 0.0,
        "time_brute_force_fallback_seconds": 0.0,
        "failure_reason": None,
    }


def increment_counter(counters, key, amount=1):
    """Increment one counter if a counters dictionary was provided.

    Input:
        counters: a dict from new_weyl_demo_counters(), or None.
        key: counter name.
        amount: integer amount to add.
    Output:
        None. Mutates counters when counters is not None.
    """
    if counters is not None:
        counters[key] += amount


def add_elapsed_time(counters, key, elapsed_seconds):
    """Add elapsed time to a timing counter if counters were provided.

    Input:
        counters: a dict from new_weyl_demo_counters(), or None.
        key: timing counter name.
        elapsed_seconds: float number of seconds to add.
    Output:
        None. Mutates counters when counters is not None.
    """
    if counters is not None:
        counters[key] += elapsed_seconds


def record_weyl_demo_brute_force_fallback_time(G, elapsed_seconds):
    """Record brute-force fallback time on G's demo counters when available.

    Input:
        G: a pinned_group that may have G.weyl_demo_counters.
        elapsed_seconds: float number of seconds spent in brute-force fallback.
    Output:
        None. Mutates G.weyl_demo_counters when present.
    """
    counters = getattr(G, "weyl_demo_counters", None)
    add_elapsed_time(counters, "time_brute_force_fallback_seconds", elapsed_seconds)


@contextmanager
def timed_stage(counters, key):
    """Measure one stage and add its elapsed time to counters.

    Input:
        counters: a dict from new_weyl_demo_counters(), or None.
        key: timing counter name.
    Output:
        Context manager that records elapsed wall-clock time.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        add_elapsed_time(counters, key, time.perf_counter() - start)


def record_failure(counters, failure_reason):
    """Record the first failure reason in a counters dictionary.

    Input:
        counters: a dict from new_weyl_demo_counters(), or None.
        failure_reason: string describing why the method failed.
    Output:
        None. Mutates counters when counters is not None.
    """
    if counters is not None and counters["failure_reason"] is None:
        counters["failure_reason"] = failure_reason


def print_weyl_demo_counters(counters):
    """Print a short summary of demo Weyl construction counters.

    Input:
        counters: a dict from new_weyl_demo_counters().
    Output:
        None. Prints to stdout.
    """
    print("Root-subgroup Weyl demo counters:")
    for key in (
        "roots_attempted",
        "formula_applicable",
        "candidates_constructed",
        "validation_attempts",
        "group_element_checks",
        "group_element_failures",
        "torus_normalizer_checks",
        "torus_normalizer_failures",
        "conjugation_checks",
        "conjugation_failures",
        "roots_validated",
        "roots_installed",
    ):
        print(f"\t{key}: {counters[key]}")
    if counters["failure_reason"] is not None:
        print(f"\tfailure_reason: {counters['failure_reason']}")
    print("Root-subgroup Weyl demo timing:")
    for key in (
        "time_total_seconds",
        "time_torus_reflections_seconds",
        "time_formula_applicability_seconds",
        "time_candidate_construction_seconds",
        "time_validation_total_seconds",
        "time_group_element_checks_seconds",
        "time_torus_normalizer_checks_seconds",
        "time_conjugation_checks_seconds",
        "time_install_seconds",
        "time_brute_force_fallback_seconds",
    ):
        print(f"\t{key}: {counters[key]:.4f}")


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


def construct_weyl_element_from_root_subgroups(G, alpha, counters=None):
    """Construct a Weyl representative using the root-subgroup formula.

    Input:
        G: a partially fitted pinned_group with root subgroup maps available.
        alpha: a root of G.root_system.
        counters: optional demo counter dictionary to update.
    Output:
        The matrix x_alpha(1) x_-alpha(-1) x_alpha(1), or None if the
        direct formula is not canonical for this root.
    """
    with timed_stage(counters, "time_formula_applicability_seconds"):
        formula_applies = root_subgroup_formula_applies(G, alpha)
    if not formula_applies:
        return None

    increment_counter(counters, "formula_applicable")
    one = sp.Matrix([1])
    minus_one = sp.Matrix([-1])
    with timed_stage(counters, "time_candidate_construction_seconds"):
        w_alpha = sp.simplify(
            G.root_subgroup_map(alpha, one)
            * G.root_subgroup_map(-alpha, minus_one)
            * G.root_subgroup_map(alpha, one)
        )
    increment_counter(counters, "candidates_constructed")
    return w_alpha


def validate_weyl_candidate_is_group_element(G, w_alpha, counters=None):
    """Check that a candidate Weyl element lies in the matrix group.

    Input:
        G: a pinned_group.
        w_alpha: a matrix candidate for one Weyl element.
        counters: optional demo counter dictionary to update.
    Output:
        True if w_alpha satisfies G's group equations; otherwise False.
    """
    increment_counter(counters, "group_element_checks")
    with timed_stage(counters, "time_group_element_checks_seconds"):
        is_group_element = G.is_group_element(w_alpha)
    if not is_group_element:
        increment_counter(counters, "group_element_failures")
    return is_group_element


def validate_weyl_candidate_normalizes_torus(G, w_alpha, counters=None):
    """Check that a candidate Weyl element normalizes the split torus.

    Input:
        G: a pinned_group with a generic torus element available.
        w_alpha: a matrix candidate for one Weyl element.
        counters: optional demo counter dictionary to update.
    Output:
        True if w_alpha * t * w_alpha^-1 is a torus element for generic t.
    """
    increment_counter(counters, "torus_normalizer_checks")
    with timed_stage(counters, "time_torus_normalizer_checks_seconds"):
        t = G.generic_torus_element('t')
        conjugation = sp.simplify(w_alpha * t * w_alpha.inv())
        normalizes_torus = G.is_torus_element(conjugation)
    if not normalizes_torus:
        increment_counter(counters, "torus_normalizer_failures")
    return normalizes_torus


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


def solve_weyl_conjugation_for_beta(G, alpha, beta, w_alpha,
                                    root_data_cache=None, counters=None):
    """Solve the root-subgroup conjugation equation for one beta.

    Input:
        G: a pinned_group with root subgroup maps available.
        alpha: the root defining the candidate Weyl element.
        beta: the root subgroup to conjugate.
        w_alpha: the matrix candidate for the alpha Weyl element.
        root_data_cache: optional dict shared across beta checks.
        counters: optional demo counter dictionary to update.
    Output:
        A solution dict for v in
        w_alpha * x_beta(u) * w_alpha^-1 = x_sigma_alpha(beta)(v),
        or None if the equation cannot be solved.
    """
    increment_counter(counters, "conjugation_checks")
    with timed_stage(counters, "time_conjugation_checks_seconds"):
        gamma, d_beta, d_gamma = get_reflected_root_data(
            G, alpha, beta, root_data_cache
        )
        if d_beta != d_gamma:
            increment_counter(counters, "conjugation_failures")
            return None

        u = vector_variable('u', d_beta)
        v = vector_variable('v', d_gamma)
        x_beta_u = G.root_subgroup_map(beta, u)
        x_gamma_v = G.root_subgroup_map(gamma, v)
        lhs = sp.simplify(w_alpha * x_beta_u * w_alpha.inv())

        try:
            sols = sp.solve(lhs - x_gamma_v, v.free_symbols, dict=True)
        except Exception:
            increment_counter(counters, "conjugation_failures")
            return None
        if len(sols) == 0:
            increment_counter(counters, "conjugation_failures")
            return None
        return sols[0]


def validate_weyl_candidate_conjugates_root_subgroups(G, alpha, w_alpha, counters=None):
    """Check all root-subgroup conjugation equations for one candidate.

    Input:
        G: a pinned_group with root subgroup maps available.
        alpha: the root defining the candidate Weyl element.
        w_alpha: the matrix candidate for the alpha Weyl element.
        counters: optional demo counter dictionary to update.
    Output:
        True if every root subgroup U_beta is sent to
        U_sigma_alpha(beta); otherwise False.
    """
    root_data_cache = {}
    for beta in G.root_system.root_list:
        sol = solve_weyl_conjugation_for_beta(
            G, alpha, beta, w_alpha, root_data_cache, counters
        )
        if sol is None:
            return False
    return True


def is_valid_weyl_element_candidate(G, alpha, w_alpha, counters=None):
    """Run the shared validation checks for one candidate Weyl element.

    Input:
        G: a pinned_group with torus and root subgroup data available.
        alpha: the root defining the candidate Weyl element.
        w_alpha: the matrix candidate to validate.
        counters: optional demo counter dictionary to update.
    Output:
        True if w_alpha is in G, normalizes the torus, and conjugates each
        U_beta into U_sigma_alpha(beta); otherwise False.
    """
    assert G.root_system.is_root(alpha), \
        "Cannot validate a Weyl element candidate for a non-root"
    increment_counter(counters, "validation_attempts")
    with timed_stage(counters, "time_validation_total_seconds"):
        if not validate_weyl_candidate_is_group_element(G, w_alpha, counters):
            return False
        if not validate_weyl_candidate_normalizes_torus(G, w_alpha, counters):
            return False
        if not validate_weyl_candidate_conjugates_root_subgroups(G, alpha, w_alpha, counters):
            return False
        increment_counter(counters, "roots_validated")
        return True


def construct_validated_weyl_element_from_root_subgroups(G, alpha, counters=None):
    """Construct and validate one root-subgroup Weyl element.

    Input:
        G: a partially fitted pinned_group.
        alpha: a root of G.root_system.
        counters: optional demo counter dictionary to update.
    Output:
        (w_alpha, None) on success, or (None, failure_reason) on failure.
    """
    increment_counter(counters, "roots_attempted")
    w_alpha = construct_weyl_element_from_root_subgroups(G, alpha, counters)
    if w_alpha is None:
        failure_reason = (
            "direct root-subgroup formula requires one-dimensional "
            f"opposite root spaces; skipped alpha = {alpha}"
        )
        record_failure(counters, failure_reason)
        return None, failure_reason
    if not is_valid_weyl_element_candidate(G, alpha, w_alpha, counters):
        failure_reason = (
            "direct root-subgroup Weyl construction failed validation "
            f"for alpha = {alpha}"
        )
        record_failure(counters, failure_reason)
        return None, failure_reason
    return w_alpha, None


def fit_torus_reflections_from_coroots(G, display=True, counters=None):
    """Install the torus reflection map s_alpha on G.

    Input:
        G: a partially fitted pinned_group with coroots available.
        display: whether to print progress.
    Output:
        None. Mutates G by assigning G.torus_reflection_map.
    """
    if display:
        print("Fitting torus reflections (s_α)")

    with timed_stage(counters, "time_torus_reflections_seconds"):
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


def construct_validated_weyl_elements_from_root_subgroups(G, counters=None):
    """Construct and validate root-subgroup Weyl elements for every root.

    Input:
        G: a partially fitted pinned_group.
        counters: optional demo counter dictionary to update.
    Output:
        (weyl_elements, None) on success, where weyl_elements maps roots to
        matrices; otherwise (None, failure_reason). This function does not
        mutate G.
    """
    weyl_elements = {}
    for alpha in G.root_system.root_list:
        w_alpha, failure_reason = construct_validated_weyl_element_from_root_subgroups(
            G, alpha, counters
        )
        if failure_reason is not None:
            return None, failure_reason
        weyl_elements[alpha] = w_alpha
    return weyl_elements, None


def install_weyl_element_map(G, weyl_elements, counters=None):
    """Install a completed Weyl element dictionary on G.

    Input:
        G: a pinned_group to mutate.
        weyl_elements: dict mapping each root alpha to its matrix w_alpha.
        counters: optional demo counter dictionary to update.
    Output:
        None. Mutates G by assigning G.weyl_element_list and G.weyl_element_map.
    """
    with timed_stage(counters, "time_install_seconds"):
        G.weyl_element_list = weyl_elements
        increment_counter(counters, "roots_installed", len(weyl_elements))

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
    counters = new_weyl_demo_counters()
    G.weyl_demo_counters = counters
    with timed_stage(counters, "time_total_seconds"):
        fit_torus_reflections_from_coroots(G, display, counters)
        if display:
            print("Trying Weyl elements (w_α) from root subgroups")

        weyl_elements, failure_reason = construct_validated_weyl_elements_from_root_subgroups(
            G, counters
        )
        if failure_reason is not None:
            if display:
                print(f"Root-subgroup Weyl fast path skipped: {failure_reason}")
            success = False
        else:
            install_weyl_element_map(G, weyl_elements, counters)
            success = True

    if display:
        print_weyl_demo_counters(counters)
    if not success:
        return False
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
    counters = new_weyl_demo_counters()
    G.weyl_demo_counters = counters
    with timed_stage(counters, "time_total_seconds"):
        fit_torus_reflections_from_coroots(G, display, counters)
        if display:
            print("Fitting Weyl elements (w_α) from root subgroups")

        weyl_elements, failure_reason = construct_validated_weyl_elements_from_root_subgroups(
            G, counters
        )
        if failure_reason is None:
            install_weyl_element_map(G, weyl_elements, counters)
    if failure_reason is not None:
        if display:
            print_weyl_demo_counters(counters)
        assert failure_reason is None, failure_reason
    if display:
        print_weyl_demo_counters(counters)
