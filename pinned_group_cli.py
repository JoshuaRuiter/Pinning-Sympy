import argparse
import sys
import time
from pathlib import Path

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
from weyl_element_demo import (
    fit_weyl_group_elements_from_root_subgroups,
    record_weyl_demo_brute_force_fallback_time,
    try_fit_weyl_group_elements_from_root_subgroups,
)


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
        choices=["brute", "root-subgroups", "auto"],
        default="brute",
        help=(
            "Weyl element construction method. root-subgroups is strict "
            "experimental/demo mode; auto tries root-subgroups first and "
            "falls back to brute. Default: brute."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--output",
        choices=["text", "latex"],
        default="text",
        help="Output format. text keeps the current terminal output. latex writes a .tex document. Default: text.",
    )
    parser.add_argument(
        "--out",
        help="Output file for --output latex. If omitted, LaTeX is printed to stdout.",
    )
    return parser.parse_args()


def latex_text(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def latex_math(obj):
    if isinstance(obj, bool):
        return r"\mathrm{" + str(obj) + "}"
    if obj is None:
        return r"\mathrm{None}"
    if isinstance(obj, str):
        return r"\mathrm{" + latex_text(obj) + "}"
    if isinstance(obj, tuple):
        return r"\left(" + ", ".join(latex_math(item) for item in obj) + r"\right)"
    if isinstance(obj, list):
        return r"\left[" + ", ".join(latex_math(item) for item in obj) + r"\right]"
    return sp.latex(obj)


def latex_display(lhs, rhs):
    return "\\[\n" + lhs + " = " + rhs + "\n\\]\n"


def pinned_group_to_latex(G):
    lines = [
        r"\section{" + latex_text(G.name_string) + "}",
        "",
        r"\subsection{Root system}",
        latex_display(r"\Phi", latex_math(G.root_system.name_string)),
        latex_display(r"|\Phi|", latex_math(len(G.root_system.root_list))),
    ]

    if G.root_system.is_irreducible:
        lines.append(latex_display(r"\mathrm{Dynkin\ type}", latex_math(G.root_system.dynkin_type)))

    lines.extend([
        "",
        r"\subsection{Roots and associated data}",
    ])

    t = G.generic_torus_element("t")
    scalar_t = sp.symbols("t")

    for alpha in G.root_space_dict:
        d_alpha = G.root_space_dimension(alpha)
        u = vector_variable(letter="u", length=d_alpha)
        alpha_latex = latex_math(alpha)

        lines.extend([
            "",
            r"\subsubsection*{$\alpha = " + alpha_latex + r"$}",
            latex_display(r"\alpha^\vee", latex_math(G.root_system.coroot_dict[alpha])),
            latex_display(r"d_\alpha", latex_math(d_alpha)),
            latex_display(r"X_\alpha(u)", latex_math(G.root_space_map(alpha, u))),
            latex_display(r"x_\alpha(u)", latex_math(G.root_subgroup_map(alpha, u))),
            latex_display(r"s_\alpha(t)", latex_math(G.torus_reflection_map(alpha, t))),
            latex_display(r"w_\alpha", latex_math(G.weyl_element_map(alpha))),
            latex_display(r"h_\alpha(t)", latex_math(G.coroot_torus_element_map(alpha, scalar_t))),
        ])

        if not G.root_system.is_reduced:
            multipliable = G.root_system.is_multipliable_root(alpha)
            lines.append(latex_display(r"\mathrm{multipliable}", latex_math(multipliable)))
            if multipliable:
                hom_defect = G.homomorphism_defect_coefficient_dict[alpha][2]
                lines.append(latex_display(r"\mathrm{homomorphism\ defect}", latex_math(hom_defect)))

    lines.extend([
        "",
        r"\subsection{Commutator coefficients}",
    ])

    if len(G.commutator_coefficient_dict) == 0:
        lines.append(r"No pairs of roots are summable, so there are no commutator coefficients.")
    else:
        for alpha, beta in G.root_system.summable_non_proportional_pairs:
            d_alpha = G.root_space_dimension(alpha)
            d_beta = G.root_space_dimension(beta)
            u = vector_variable(letter="u", length=d_alpha)
            v = vector_variable(letter="v", length=d_beta)
            linear_combos = G.root_system.integer_linear_combos(alpha, beta)

            lines.extend([
                "",
                r"\subsubsection*{$\alpha = " + latex_math(alpha) + r",\quad \beta = " + latex_math(beta) + r"$}",
            ])

            for key in linear_combos:
                i, j = key
                combo = linear_combos[key]
                coeff = G.commutator_coefficient_map(alpha, beta, i, j, u, v)
                raw_expr = coeff[0] if len(coeff) == 1 else coeff.T

                lines.append(latex_display(r"i\alpha + j\beta", latex_math(combo)))
                lines.append(latex_display(
                    r"N_{" + latex_math(i) + "," + latex_math(j) + r"}^{\alpha,\beta}(u,v)",
                    latex_math(raw_expr),
                ))

    lines.extend([
        "",
        r"\subsection{Weyl conjugation coefficients}",
    ])

    for alpha in G.root_system.root_list:
        for beta in G.root_system.root_list:
            d_beta = G.root_space_dimension(beta)
            u = vector_variable(letter="u", length=d_beta)
            gamma = G.root_system.reflect_root(hyperplane_root=alpha,
                                               root_to_reflect=beta)
            phi_u = G.weyl_conjugation_coefficient_map(alpha, beta, u)
            raw_expr = phi_u[0] if len(phi_u) == 1 else phi_u.T

            lines.extend([
                "",
                r"\subsubsection*{$\alpha = " + latex_math(alpha) + r",\quad \beta = " + latex_math(beta) + r"$}",
                latex_display(r"\sigma_\alpha(\beta)", latex_math(gamma)),
                latex_display(r"\varphi_{\alpha,\beta}(u)", latex_math(raw_expr)),
            ])

    return "\n".join(lines)


def latex_document(groups):
    sections = [pinned_group_to_latex(G) for G in groups]
    return "\n".join([
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\allowdisplaybreaks",
        r"\begin{document}",
        "",
        r"\title{Symbolic Pinning Output}",
        r"\maketitle",
        "",
        "\n\n".join(sections),
        "",
        r"\end{document}",
        "",
    ])


def fit_group_with_demo_weyl_method(G, display=True, weyl_method="root-subgroups"):
    if display:
        print("\n" + "-" * 100 + "\n")
        print(f"Fitting a pinning for {G.name_string}")
    G.fit_root_system(display)
    G.fit_root_spaces(display)
    G.fit_root_subgroup_maps(display)
    G.fit_homomorphism_defect_coefficients(display)
    G.fit_commutator_coefficients(display)

    if weyl_method == "root-subgroups":
        fit_weyl_group_elements_from_root_subgroups(G, display)
    elif weyl_method == "auto":
        used_fast_path = try_fit_weyl_group_elements_from_root_subgroups(G, display)
        if not used_fast_path:
            if display:
                print("Falling back to brute-force Weyl element search")
            brute_force_start = time.perf_counter()
            G.fit_weyl_group_elements(display)
            brute_force_elapsed = time.perf_counter() - brute_force_start
            record_weyl_demo_brute_force_fallback_time(G, brute_force_elapsed)
            if display:
                print(f"Brute-force Weyl fallback time: {brute_force_elapsed:.4f} seconds")
    else:
        raise ValueError(f"Unknown demo Weyl method: {weyl_method}")

    G.fit_weyl_conjugation_coefficients(display)
    G.fit_coroot_torus_elements(display)
    if display:
        print("Fitting complete")
        G.display_pinning_info()
        print("\n" + "-" * 100 + "\n")


def fit_group(G, display=True, validate=True, weyl_method="brute"):
    if weyl_method == "brute":
        G.fit_pinning(display=display)
    elif weyl_method in ("root-subgroups", "auto"):
        if display:
            print(f"Using demo Weyl method: {weyl_method}")
        fit_group_with_demo_weyl_method(G, display, weyl_method)
    else:
        raise ValueError(f"Unknown Weyl method: {weyl_method}")

    if validate:
        G.validate_pinning(display=display)

    return G


def run_SL_tests(n_min, n_max, display=True, validate=True, weyl_method="brute"):
    groups = []
    if display:
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
        groups.append(fit_group(SL_n, display, validate, weyl_method))
        assert SL_n.root_system.dynkin_type == "A", \
            f"SL(n={n}) should have type A but computations gave type {SL_n.root_system.dynkin_type}"
    if display:
        print("\nDone with special linear groups")
        print("\n" + "=" * 100 + "\n")
    return groups


def run_SO_split_tests(n_min, n_max, q_min, q_max, display=True, validate=True, weyl_method="brute"):
    groups = []
    if display:
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
            groups.append(fit_group(SO_n_q, display, validate, weyl_method))
    if display:
        print("\nDone with split special orthogonal groups")
        print("\n" + "=" * 100 + "\n")
    return groups


def run_SO_nonsplit_tests(n_min, n_max, q_min, q_max, display=True, validate=True, weyl_method="brute"):
    groups = []
    if display:
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
            groups.append(fit_group(SO_n_q, display, validate, weyl_method))
    if display:
        print("\nDone with non-split special orthogonal groups")
        print("\n" + "=" * 100 + "\n")
    return groups


def run_SU_quasisplit_tests(n_min, n_max, q_min, q_max, eps_values, display=True, validate=True, weyl_method="brute"):
    groups = []
    if display:
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
                groups.append(fit_group(SU_n_q, display, validate, weyl_method))
    if display:
        print("\nDone with quasi-split special unitary groups")
        print("\n" + "=" * 100 + "\n")
    return groups


def run_SU_nonquasisplit_tests(n_min, n_max, q_min, q_max, eps_values, display=True, validate=True, weyl_method="brute"):
    groups = []
    if display:
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
                groups.append(fit_group(SU_n_q, display, validate, weyl_method))
    if display:
        print("\nDone with non-(quasi-split) special unitary groups")
        print("\n" + "=" * 100 + "\n")
    return groups


def main():
    args = parse_args()
    sp.init_printing(wrap_line=False)
    display = not args.quiet and args.output == "text"
    validate = not args.skip_validation
    start_time = time.perf_counter()
    groups = []

    if args.suite in ("sl", "all"):
        groups.extend(run_SL_tests(args.n_min, args.n_max, display, validate, args.weyl_method))
    if args.suite in ("so-split", "all"):
        groups.extend(run_SO_split_tests(args.n_min, args.n_max, args.q_min, args.q_max, display, validate, args.weyl_method))
    if args.suite in ("so-nonsplit", "all"):
        groups.extend(run_SO_nonsplit_tests(args.n_min, args.n_max, args.q_min, args.q_max, display, validate, args.weyl_method))
    if args.suite in ("su-quasisplit", "all"):
        groups.extend(run_SU_quasisplit_tests(args.n_min, args.n_max, args.q_min, args.q_max, args.eps, display, validate, args.weyl_method))
    if args.suite in ("su-nonquasisplit", "all"):
        groups.extend(run_SU_nonquasisplit_tests(args.n_min, args.n_max, args.q_min, args.q_max, args.eps, display, validate, args.weyl_method))

    if args.output == "latex":
        output = latex_document(groups)
        if args.out:
            Path(args.out).write_text(output)
            print(f"Wrote LaTeX output to {args.out}")
        else:
            print(output)

    if display or args.output == "latex":
        status_stream = sys.stderr if args.output == "latex" and not args.out else sys.stdout
        print("\nAll requested examples complete.", file=status_stream)
        print(f"Time to run examples: {round((time.perf_counter() - start_time) / 60, 1)} minutes", file=status_stream)


if __name__ == "__main__":
    main()
