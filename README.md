# Pinning-Sympy

A Python project for symbolic computation with pinned algebraic groups, root systems, and classical matrix group structures.

## What it does

`Pinning-Sympy` provides:
- a `pinned_group` class for storing and computing pinned group data,
- symbolic construction of root systems, Lie algebras, root spaces, and Weyl group elements,
- computation and verification of commutator coefficients and conjugation formulas,
- support for classical groups: special linear, special orthogonal, and special unitary

## Installation

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

Run the example driver to exercise the library and see how pinned groups are built:

```bash
python pinned_group_examples.py
```

By default, the example script runs tests for special linear groups. There are additional commented examples for orthogonal and unitary groups.

There is also an experimental command-line runner that lets you choose examples without editing `pinned_group_examples.py`:

```bash
python pinned_group_cli.py --help
python pinned_group_cli.py sl --n-max 3
python pinned_group_cli.py so-split --n-max 5 --q-max 2
python pinned_group_cli.py sl --n-max 3 --weyl-method root-subgroups
python pinned_group_cli.py sl --n-max 3 --weyl-method auto
python pinned_group_cli.py sl --n-max 3 --output latex --out output.tex
```

The `root-subgroups` Weyl method is a strict demo/experimental method for the
direct rank-one formula. The `auto` method tries that demo method first and
falls back to the default brute-force Weyl search when the direct formula does
not apply or fails validation.

For quick interactive experiments, `examples_for_chat.py` loads the saved
`SL_3.pkl` pinned group when available, or computes and saves it if the file is
missing:

```bash
python examples_for_chat.py
```

## Project layout

- `pinned_group.py` - core class and algorithms for pinned algebraic groups.
- `pinned_group_examples.py` - example usage and verification routines.
- `pinned_group_cli.py` - optional command-line runner for selecting example suites and Weyl methods.
- `weyl_element_demo.py` - experimental Weyl element construction helpers kept outside the core class.
- `examples_for_chat.py` - small interactive example that loads or builds a cached `SL_3` pinned group.
- `root_system.py` - root system and reflection utilities.
- `split_torus.py` - split torus construction and helpers.
- `nondegenerate_isotropic_form.py` - anisotropic form and quadratic extension utilities.
- `utility_general.py` - general symbolic utilities, solver helpers, and pruning logic.
- `utility_roots.py` - root-character list generation and reduction tools.
- `utility_SL.py`, `utility_SO.py`, `utility_SU.py` - group-specific constraint definitions and torus generators.
- `Notes document/` - longer mathematical notes and bibliography used during development.
- `temp_workspace.py` - scratch workspace for ad hoc calculations.
- `SL_3.pkl` - saved pinned group data used by `examples_for_chat.py`.
- `SU_5_2_1.pkl` - saved exploratory data for a unitary example.

## Dependencies

- `sympy`
- `numpy`
- `tabulate`

These are the only external dependencies required by the repository.

The cached example in `examples_for_chat.py` also uses `dill` for reading and
writing `.pkl` files. Install it separately if you want to run that script:

```bash
pip install dill
```

## Current development notes

The project is still in experimental development. Current focus areas include:
- update the notes document,
- improving performance and reducing brute-force search in the Weyl element solver,
- keeping experimental Weyl construction and validation helpers outside the core class while they are being tested,
- implementing a tool for visualizing equations/identities,
- implement a belongs_to_generated_subgroup method for pinned_group in a computational feasible way,
- extending support to more group types and non-split torus structures,
- moving `root_system` toward a `root_datum` design.

## Notes

This repository is intended for exploration and experimentation with pinned groups and root data. It is not yet packaged for broader distribution.
