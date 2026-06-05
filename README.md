# Pinning-Sympy

A Python project for symbolic computation with pinned algebraic groups, root systems, and classical matrix group structures.

## What it does

`Pinning-Sympy` provides:
- a `pinned_group` class for storing and computing pinned group data,
- symbolic construction of root systems, Lie algebras, root spaces, and Weyl group elements,
- computation and verification of commutator coefficients and conjugation formulas,
- support for classical groups such as `SL`, `SO`, and `SU`.

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

By default, the example script runs tests for special linear groups. There are additional commented examples for `SO` and `SU` cases.

## Project layout

- `pinned_group.py` - core class and algorithms for pinned algebraic groups.
- `pinned_group_examples.py` - example usage and verification routines.
- `root_system.py` - root system and reflection utilities.
- `split_torus.py` - split torus construction and helpers.
- `nondegenerate_isotropic_form.py` - anisotropic form and quadratic extension utilities.
- `utility_general.py` - general symbolic utilities, solver helpers, and pruning logic.
- `utility_roots.py` - root-character list generation and reduction tools.
- `utility_SL.py`, `utility_SO.py`, `utility_SU.py` - group-specific constraint definitions and torus generators.

## Dependencies

- `sympy`
- `numpy`
- `tabulate`

These are the only external dependencies required by the repository.

## Current development notes

The project is still in experimental development. Current focus areas include:
- improving performance and reducing brute-force search in the Weyl element solver,
- fixing formatting and symbolic printing for subscripted variables,
- resolving parenthesis and coefficient issues in commutator tables,
- extending support to more group types and non-split torus structures,
- moving `root_system` toward a `root_datum` design.

## Notes

This repository is intended for exploration and experimentation with pinned groups and root data. It is not yet packaged for broader distribution.
