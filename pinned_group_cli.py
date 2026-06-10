import argparse
import time

import sympy as sp

from nondegenerate_isotropic_form import nondegenerate_isotropic_form
from pinned_group import pinned_group
from split_torus import split_torus
from utility_general import vector_variable
from utility_SL import (character_entries_SL,
                        generic_lie_algebra_element_SL,
                        generic_torus_element_SL,
                        group_constraints_SL,
                        is_torus_element_SL,
                        lie_algebra_constraints_SL,
                        trivial_characters_SL)
from utility_SO import (character_entries_SO,
                        generic_lie_algebra_element_SO,
                        generic_torus_element_SO,
                        group_constraints_SO,
                        is_torus_element_SO,
                        lie_algebra_constraints_SO,
                        trivial_characters_SO)
from utility_SU import (character_entries_SU,
                        generic_lie_algebra_element_SU,
                        generic_torus_element_SU,
                        group_constraints_SU,
                        is_torus_element_SU,
                        lie_algebra_constraints_SU,
                        trivial_characters_SU)
from weyl_element_demo import fit_weyl_group_elements_from_root_subgroups


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run symbolic pinning examples without editing pinned_group_examples.py."
    )
    parser.add_argument(
        "suite",
        nargs="?",
        default="sl",
        choices=["sl", "so-split", "so-nonsplit", "su-quasisplit", "su-nonquasisplit", "all"],
        help="Example suite to run. Default: sl.",
    )
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=3)
    parser.add_argument("--q-min", type=int, default=1)
    parser.add_argument("--q-max", type=int, default=3)
    parser.add_argument(
        "--eps",
        type=int,
        nargs="+",
        default=[-1, 1],
        choices=[-1, 1],
        help="Unitary epsilon values to test. Default: -1 1.",
    )
    parser.add_argument(
        "--weyl-method",
        choices=["brute", "root-subgroups"],
        default="brute",
        help="Weyl element construction method. root-subgroups is experimental/demo only. Default: brute.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def fit_group(G, display=True, validate=True, weyl_method="brute"):
    if weyl_method == "brute":
        G.fit_pinning(display=display)
    elif weyl_method == "root-subgroups":
        if display:
            print("\n" + "-" * 100 + "\n")
            print(f"Fitting a pinning for {G.name_string}")
        G.fit_root_system(display)
        G.fit_root_spaces(display)
        G.fit_root_subgroup_maps(display)
        G.fit_homomorphism_defect_coefficients(display)
        G.fit_commutator_coefficients(display)
        fit_weyl_group_elements_from_root_subgroups(G, display)
        G.fit_weyl_conjugation_coefficients(display)
        G.fit_coroot_torus_elements(display)
        if display:
            print("Fitting complete")
            G.display_pinning_info()
            print("\n" + "-" * 100 + "\n")
    else:
        raise ValueError(f"Unknown Weyl method: {weyl_method}")

    if validate:
        G.validate_pinning(display=display)


def run_SL_tests(n_min, n_max, display=True, validate=True, weyl_method="brute"):
    print("\n" + "=" * 100 + "\n")
    print("Running calculations and verifications for special linear groups")
    n_min = max(n_min, 2)
    for n in range(n_min, n_max + 1):
        T = split_torus(matrix_size=n,
                        rank=n - 1,
                        is_element=is_torus_element_SL,
                        generic_element=generic_torus_element_SL,
                        trivial_character_matrix=trivial_characters_SL(n),
                        nontrivial_character_entries=character_entries_SL(n))
        SL_n = pinned_group(name_string=f"SL(n={n})",
                            matrix_size=n,
                            form=None,
                            group_constraints=group_constraints_SL,
                            maximal_split_torus=T,
                            lie_algebra_constraints=lie_algebra_constraints_SL,
                            generic_lie_algebra_element=generic_lie_algebra_element_SL,
                            non_variables=None)
        fit_group(SL_n, display, validate, weyl_method)
        assert SL_n.root_system.dynkin_type == "A", \
            f"SL(n={n}) should have type A but computations gave type {SL_n.root_system.dynkin_type}"
    print("\nDone with special linear groups")
    print("\n" + "=" * 100 + "\n")


def run_SO_split_tests(n_min, n_max, q_min, q_max, display=True, validate=True, weyl_method="brute"):
    print("\n" + "=" * 100 + "\n")
    print("Running calculations and verifications for split special orthogonal groups")
    q_min = max(q_min, 2)
    for q in range(q_min, q_max + 1):
        n_range = [n for n in (2 * q, 2 * q + 1) if n_min <= n and n <= n_max]
        for n in n_range:
            anisotropic_vec = vector_variable("c", n - 2 * q)
            NIF = nondegenerate_isotropic_form(dimension=n,
                                                witt_index=q,
                                                anisotropic_vector=anisotropic_vec,
                                                epsilon=None,
                                                primitive_element=None)
            T = split_torus(matrix_size=n,
                            rank=q,
                            is_element=is_torus_element_SO,
                            generic_element=generic_torus_element_SO,
                            trivial_character_matrix=trivial_characters_SO(n, q),
                            nontrivial_character_entries=character_entries_SO(n, q))
            SO_n_q = pinned_group(name_string=f"SO(n={n}, q={q})",
                                  matrix_size=n,
                                  form=NIF,
                                  group_constraints=group_constraints_SO,
                                  maximal_split_torus=T,
                                  lie_algebra_constraints=lie_algebra_constraints_SO,
                                  generic_lie_algebra_element=generic_lie_algebra_element_SO,
                                  non_variables=None)
            fit_group(SO_n_q, display, validate, weyl_method)
    print("\nDone with split special orthogonal groups")
    print("\n" + "=" * 100 + "\n")


def run_SO_nonsplit_tests(n_min, n_max, q_min, q_max, display=True, validate=True, weyl_method="brute"):
    print("\n" + "=" * 100 + "\n")
    print("Running calculations and verifications for non-split special orthogonal groups")
    for q in range(q_min, q_max + 1):
        n_min = max(2 * q + 2, n_min)
        for n in range(n_min, n_max + 1):
            anisotropic_vec = vector_variable("c", n - 2 * q)
            NIF = nondegenerate_isotropic_form(dimension=n,
                                                witt_index=q,
                                                anisotropic_vector=anisotropic_vec,
                                                epsilon=None,
                                                primitive_element=None)
            T = split_torus(matrix_size=n,
                            rank=q,
                            is_element=is_torus_element_SO,
                            generic_element=generic_torus_element_SO,
                            trivial_character_matrix=trivial_characters_SO(n, q),
                            nontrivial_character_entries=character_entries_SO(n, q))
            SO_n_q = pinned_group(name_string=f"SO(n={n}, q={q})",
                                  matrix_size=n,
                                  form=NIF,
                                  group_constraints=group_constraints_SO,
                                  maximal_split_torus=T,
                                  lie_algebra_constraints=lie_algebra_constraints_SO,
                                  generic_lie_algebra_element=generic_lie_algebra_element_SO,
                                  non_variables=None)
            fit_group(SO_n_q, display, validate, weyl_method)
    print("\nDone with non-split special orthogonal groups")
    print("\n" + "=" * 100 + "\n")


def run_SU_quasisplit_tests(n_min, n_max, q_min, q_max, eps_values, display=True, validate=True, weyl_method="brute"):
    print("\n" + "=" * 100 + "\n")
    print("Running calculations and verifications for quasi-split special unitary groups")
    d = sp.symbols("d", nonzero=True)
    p_e = sp.sqrt(d)
    for q in range(q_min, q_max + 1):
        n = 2 * q
        if n_min <= n and n <= n_max:
            for eps in eps_values:
                anisotropic_vec = vector_variable("c", n - 2 * q)
                if eps == -1:
                    anisotropic_vec = anisotropic_vec * p_e
                NIF = nondegenerate_isotropic_form(dimension=n,
                                                    witt_index=q,
                                                    anisotropic_vector=anisotropic_vec,
                                                    epsilon=eps,
                                                    primitive_element=p_e)
                T = split_torus(matrix_size=n,
                                rank=q,
                                is_element=is_torus_element_SU,
                                generic_element=generic_torus_element_SU,
                                trivial_character_matrix=trivial_characters_SU(n, q),
                                nontrivial_character_entries=character_entries_SU(n, q))
                SU_n_q = pinned_group(name_string=f"SU(n={n}, q={q}, eps={eps})",
                                      matrix_size=n,
                                      form=NIF,
                                      group_constraints=group_constraints_SU,
                                      maximal_split_torus=T,
                                      lie_algebra_constraints=lie_algebra_constraints_SU,
                                      generic_lie_algebra_element=generic_lie_algebra_element_SU,
                                      non_variables={d})
                fit_group(SU_n_q, display, validate, weyl_method)
    print("\nDone with quasi-split special unitary groups")
    print("\n" + "=" * 100 + "\n")


def run_SU_nonquasisplit_tests(n_min, n_max, q_min, q_max, eps_values, display=True, validate=True, weyl_method="brute"):
    print("\n" + "=" * 100 + "\n")
    print("Running calculations and verifications for non-(quasi-split) special unitary groups")
    d = sp.symbols("d", nonzero=True)
    p_e = sp.sqrt(d)
    q_min = max(q_min, 2)
    for q in range(q_min, q_max + 1):
        n_min = max(2 * q + 1, n_min)
        for n in range(n_min, n_max + 1):
            for eps in eps_values:
                anisotropic_vec = vector_variable("c", n - 2 * q)
                if eps == -1:
                    anisotropic_vec = anisotropic_vec * p_e
                NIF = nondegenerate_isotropic_form(dimension=n,
                                                    witt_index=q,
                                                    anisotropic_vector=anisotropic_vec,
                                                    epsilon=eps,
                                                    primitive_element=p_e)
                T = split_torus(matrix_size=n,
                                rank=q,
                                is_element=is_torus_element_SU,
                                generic_element=generic_torus_element_SU,
                                trivial_character_matrix=trivial_characters_SU(n, q),
                                nontrivial_character_entries=character_entries_SU(n, q))
                SU_n_q = pinned_group(name_string=f"SU(n={n}, q={q}, eps={eps})",
                                      matrix_size=n,
                                      form=NIF,
                                      group_constraints=group_constraints_SU,
                                      maximal_split_torus=T,
                                      lie_algebra_constraints=lie_algebra_constraints_SU,
                                      generic_lie_algebra_element=generic_lie_algebra_element_SU,
                                      non_variables={d})
                fit_group(SU_n_q, display, validate, weyl_method)
    print("\nDone with non-(quasi-split) special unitary groups")
    print("\n" + "=" * 100 + "\n")


def main():
    args = parse_args()
    sp.init_printing(wrap_line=False)
    display = not args.quiet
    validate = not args.skip_validation
    start_time = time.perf_counter()

    if args.suite in ("sl", "all"):
        run_SL_tests(args.n_min, args.n_max, display, validate, args.weyl_method)
    if args.suite in ("so-split", "all"):
        run_SO_split_tests(args.n_min, args.n_max, args.q_min, args.q_max, display, validate, args.weyl_method)
    if args.suite in ("so-nonsplit", "all"):
        run_SO_nonsplit_tests(args.n_min, args.n_max, args.q_min, args.q_max, display, validate, args.weyl_method)
    if args.suite in ("su-quasisplit", "all"):
        run_SU_quasisplit_tests(args.n_min, args.n_max, args.q_min, args.q_max, args.eps, display, validate, args.weyl_method)
    if args.suite in ("su-nonquasisplit", "all"):
        run_SU_nonquasisplit_tests(args.n_min, args.n_max, args.q_min, args.q_max, args.eps, display, validate, args.weyl_method)

    print("\nAll requested examples complete.")
    print(f"Time to run examples: {round((time.perf_counter() - start_time) / 60, 1)} minutes")


if __name__ == "__main__":
    main()
