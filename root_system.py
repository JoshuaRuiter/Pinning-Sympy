# This is a custom class to model root systems of Lie algebras.
# There is a built-in class implemented by Sympy, but it lacks functionality needed for pinnings of groups.

# Note that these root systems are not assumed to be reduced, i.e. they do not necessarily
# satisfy the assumption that the only multiple of a root in the root system are +1 and -1.
# Dropping this axiom leads to "non-reduced root systems," but all such root systems are
# of type BC, which is essentially the union of the type B and type C root systems 
# (given a certain realization/embedding of these root systems).
# In particular, type BC includes some roots where twice the root is another root
# (so for those longer roots, half that root is a root).

import sympy as sp
import numpy as np
import itertools
from utility_general import vector_variable
from utility_roots import (vector, 
                           in_column_span, 
                           connected_components, 
                           directed_dynkin_graphs,
                           visualize_graph)
from tabulate import tabulate

valid_dynkin_types = ('A', 'B','C','BC','D','E','F','G')

class root_system:
    
    # A model of a root system object
    
    # All roots are stored as vector type objects
    # vector is an extension of tuple, defined in utility_roots
    # root_list stores a list of representing roots as vectors
    
    # Roots are considered up to equivalence by a given lattice L
    # The lattice is stored in the form of a matrix whose columns generate the lattice,
    # and the columns must be equal in length/dimension to the roots
    
    # Root vectors are viewed as representatives of classes in the quotient lattice Z^n/L
    # i.e. two vectors are the same if their difference is in L.
    
    def __init__(self, root_list, lattice_matrix = None):
        
        # check that all roots are vectors
        for r in root_list: assert type(r) == vector, \
            "Root system can only be constructed using vectors"
        self.root_list = root_list
        
        # check that all roots are vectors have same length/dimension, 
        # which must also match the number of rows in the lattice matrix
        first_vec_length = len(self.root_list[0])
        for alpha in self.root_list: 
            assert len(alpha) == first_vec_length, \
                "Root system can't have vectors of different lengths"
        self.vector_length = first_vec_length
        
        if lattice_matrix is None:
            # make the lattice_matrix a single column of zeros if there isn't one
            self.lattice_matrix = sp.zeros(self.vector_length, 1)
        else:
            self.lattice_matrix = lattice_matrix
        
        assert self.vector_length == self.lattice_matrix.shape[0], \
            "Lattice matrix has wrong number of rows"

        # Determine the irreducible components
        component_root_lists = determine_irreducible_components(self.root_list)
        self.is_irreducible = (len(component_root_lists) == 1)
        
        # Make an ordered list of root lengths,
        # then determine if all roots are the same length, 
        # i.e. whether the root system is simply laced
        self.root_norms = sorted({r.norm() for r in self.root_list})
        self.is_simply_laced = (len(set(self.root_norms)) == 1)
        
        if self.is_irreducible:
            self.determine_properties()
        else:
            self.components = [root_system(c, self.lattice_matrix) for c in component_root_lists]
            self.is_reduced = all([c.is_reduced for c in self.components])
            self.dynkin_type = [c.dynkin_type for c in self.components]
            self.rank = [c.rank for c in self.components]
            self.name_strings = [c.name_string for c in self.components]
            self.name_string = "_x_".join(self.name_strings)

        # Set up a dictionary of coroots
        self.coroot_dict = {r : self.compute_coroot(r) for r in self.root_list}
        
        # Set up a list of non-proportional un-ordered pairs
        self.non_proportional_pairs = [
            (alpha, beta) for i, alpha in enumerate(self.root_list)
            for beta in self.root_list[i+1:]
            if not self.is_proportional(alpha, beta)
        ]
        
        # Set up a list of ordered pairs whose sum is also a root
        self.summable_non_proportional_pairs = [
            (alpha, beta) for alpha in self.root_list for beta in self.root_list
            if self.is_root(alpha + beta) and not self.is_proportional(alpha, beta)
        ]
        
    def determine_properties(self):
        # Determine and populate various internal variables of the root system, including:
        #   -sorted list of root lengths
        #   -simply laced or not
        #   -reduced vs nonreduced
        #   -choose a set of positive roots
        #   -choose a set of simple roots
        #   -compute the Cartan matrix with a choice of simple roots
        #   -compute the underlying graph of the Dynkin diagram from the Cartan matrix
        #   -connectedness
        #   -determine the Dynkin type from various other info above
        
        # Key assumption: the root system is irreducible
        # Some of these properties and computations still apply to reducible root systems,
        # but particularly the Dynkin type classification only makes sense for
        # irreducible root systems.
        assert(self.is_irreducible)
        
        # Check for non-reduced-ness,
        # i.e. if there are any pairs of roots that
        # are proportional in a ratio other than +/-1
        self.is_reduced = True
        for i, alpha in enumerate(self.root_list):
            for beta in self.root_list[i+1:]:
                proportional, ratio = self.is_proportional(alpha, beta, with_ratio = True)
                if proportional and abs(ratio) != 1:
                    self.is_reduced = False
                    self.dynkin_type = 'BC'
                    break
            if not self.is_reduced:
                break
        
        # Make choices of positive and simple roots
        # Note that there are many equally valid choices of positive roots,
        # and this is only one such valid choice.
        # The implementation here is non-deterministic. 
        # Specifically, it chooses a random vector and then assigns all 
        # roots on one side of the associated perpendicular hyperplane
        # to be positive. Because of this randomness, a seed argument
        # is included so that you can get a consistent (though still arbitrary)
        # choice of positive roots.
        self.positive_roots = choose_positive_roots(self.root_list, max_tries=100,seed=0)
        self.simple_roots = choose_simple_roots(self.positive_roots)
        
        # Check that the number of simple roots is the same as the
        # rank of the matrix spanned by the list of all roots
        matrix_of_roots = sp.Matrix(self.root_list)
        matrix_rank = matrix_of_roots.rank()
        assert len(self.simple_roots) == matrix_rank, "Rank computations mismatch"
        self.rank = matrix_rank
        
        # Build the Cartan matrix from the choice of simple roots
        self.cartan_matrix = build_cartan_matrix(self.simple_roots)
        
        # Build the (directed, weighted) Dynkin diagram/graph
        self.dynkin_graph = build_directed_dynkin_graph(self.simple_roots)
        
        # Double check irreducibility
        # Irreducibility is equivalent to the Dynkin graph being connected.
        assert len(connected_components(self.dynkin_graph)) == 1, "Irreducibility computations mismatch"
        
        # Determine the Dynkin type of each component using the Dynkin diagram
        # If the root system is non-reduced, the Dynkin type must be 'BC',
        # and this should already have been set above.
        if self.is_reduced:
            self.dynkin_type, r = determine_dynkin_type(self.dynkin_graph)
            assert self.rank == r, "Rank computations mismatch"
        else:
            assert self.dynkin_type == 'BC', "Non-reduced root system must be type BC"
        self.name_string = self.dynkin_type + str(self.rank)
        
    def is_root(self, vector_to_test, with_equivalent = False):
        # Return true if vector_to_test is an element of the root system. 
        # This is slightly more complicated than just comparing it to each root in a list,
        # because these root systems might live in a quotient space
        # by an integer lattice, rather than just a Euclidean space.
        
        assert isinstance(vector_to_test, vector), \
            "root_system.is_root expects vector input"
        assert len(vector_to_test) == self.vector_length, \
            "root_system.is_root received vector of unexpected length"
        
        # First check if the vector is already in the list, ignoring equivalence
        if vector_to_test in self.root_list:
            return (True, vector_to_test) if with_equivalent else True
        
        # The vector_to_test is not one of the representatives in root_list
        # so now we have to check if it is equivalent to any of them
        for r in self.root_list:
            if in_column_span(vector_to_test - r, self.lattice_matrix):
                return (True, r) if with_equivalent else True

        return (False, None) if with_equivalent else False

    def compute_coroot(self, alpha):
        # Given a root alpha (as vector of integers)
        # return a cocharacter vector alpha_check which must satisfy:
        #    <alpha, alpha_check> = 2
        
        assert isinstance(alpha, vector), "Character must be a vector"
        assert self.is_root(alpha), "Can only find cocharacter of a root"
        assert self.lattice_matrix is not None, "Root system must have a lattice matrix"
        
        # For E_6, E_7, and E_8, the easiest way to write the roots involves 
        # half-integers, so we don't insist on integers in that case
        if self.dynkin_type == 'E':
            assert all((2 * a) == int(2 * a) for a in alpha), \
                "Character must have integer or half-integer entries for Type E"
        else:
            assert all(isinstance(a,int) for a in alpha), "Character must have integer entries"
        
        n = self.vector_length
        L = sp.Matrix(self.lattice_matrix)
        x = vector_variable('x', n)
        x_vars = x.free_symbols
        
        # dot product of alpha with alpha_check must be 2
        eq1 = sum(alpha[i] * x[i] for i in range(n)) - 2
        
        # dot product of alpha check with any trivial character must be zero
        # The columns of L are the trivial characters,
        # and this system has one entry for each column of L
        eq2 = (x.T * L)
        
        # Solve the equations
        solutions_list = sp.solve([eq1, eq2], x_vars, dict=True)
        assert len(solutions_list) >= 1, "No solution found for cocharacter equation"
        assert len(solutions_list) == 1, \
            "Not sure what to do with multiple solutions to cocharacter equations"
        solutions_dict = solutions_list[0]
        
        x_general_solution = x.subs(solutions_dict)

        # If there are no free variables remaining, we're done
        if len(x_general_solution.free_symbols) == 0:
            if self.dynkin_type == 'E':
                # In type E, allow half-integers
                assert all((2 * a) == int(2 * a) for a in x_general_solution)
                return vector([float(a) if a % 1 != 0 else int(a) for a in x_general_solution])
            else:
                # Outside type E, only allow integers
                # Convert to vector and return
                assert all(a == int(a) for a in x_general_solution)
                return vector([int(a) for a in x_general_solution])
        
        # If there are free variables remaining, there is more to do
        # In this case, the general solution has the form
        # x = z + Y*Z^k
        # where z is a fixed vector, 
        # and Y is an integer matrix
        free_vars = x_general_solution.free_symbols
        assert len(free_vars) >= 1
        z = x_general_solution.subs({u : 0 for u in free_vars})
        Y_cols = []
        for u in free_vars:
            Y_cols.append((x_general_solution - z).subs(u,1).subs(
                {t : 0 for t in free_vars if t != u}
            ))
        Y = sp.Matrix.hstack(*Y_cols)
        n, k = Y.shape
        assert n == len(alpha) == len(x)
        
        # Now to find our cocharacter, we choose
        # the vector of the form x = z + Y*u of minimal Euclidean norm
        # This is accomplished by minimizing a quadratic form
        G = Y.T * Y
        b = -Y.T * z
        u0 = G.LUsolve(b) # value of u that minimizes norm(z+Y*u)
        u0_round = [int(round(ui)) for ui in u0]
        assert len(u0) == len(u0_round) == k
        
        # Compile a list of candidate minimizers
        # by looping over vectors with entries 0, 1, -1
        candidates = []
        for delta in itertools.product([-1,0,1], repeat = len(u0_round)):
            u = sp.Matrix([u0_round[i] + delta[i] for i in range(len(u0_round))])
            candidates.append(z+Y*u)
    
        # Choose the candidate with minimal norm
        def norm_sq(v): return v.dot(v)
        x_min = min(candidates, key = norm_sq)
        
        # Convert SymPy objects to standard Python numbers ---
        x_min_clean = [float(a) for a in x_min]
        
        # For Type E, allow half-integers. Otherwise, strictly enforce integers.
        if self.dynkin_type == 'E':
            # For type E, allow integers and half-integers
            assert all((2 * a) == int(2 * a) for a in x_min_clean)
            return vector([a if a % 1 != 0 else int(a) for a in x_min_clean])
        else:
            # For types other than E, only allow integers
            assert all(a == int(a) for a in x_min_clean)
            return vector([int(a) for a in x_min_clean])

    def is_same_root(self, alpha, beta):
        # Compare roots, up to equivalence by the integer lattice
        assert type(alpha) == type(beta) == vector, "is_same_root requires vectors"
        return alpha.equals(beta) or in_column_span(alpha - beta, self.lattice_matrix)

    def is_proportional(self, alpha, beta, with_ratio = False, max_denominator = 2):
        # Return true if alpha and beta are proportional roots
        # In most root systems, this only happens if alpha = beta or alpha = -beta,
        # but in the type BC root system it is possible to have alpha = 2*beta
        
        # Because this proportionality is modulo the lattice L,
        # testing whether alpha and beta are proportional is equivalent
        # to asking whether there exists a rational number k such that
        # k*alpha - beta is in the integer span of L
        # Equivalently,
        # p*alpha - q*beta is in the integer span of L for some integers p and q
        # This second formulation has the advantage of only involving integers
        
        # --- FAST GEOMETRIC FILTER ---
        # If two vectors are proportional, their dot product squared equals the product of their norms:
        # (alpha . beta)^2 == (alpha . alpha) * (beta . beta)
        # We do this using exact integer/rational arithmetic to avoid float issues.
        dot_prod = alpha.dot(beta)
        if dot_prod * dot_prod != alpha.dot(alpha) * beta.dot(beta):
            return (False, None) if with_ratio else False

        # If they passed the geometric filter, they ARE collinear.
        # Now we just find the ratio and see if it fits your lattice criteria.
        zero_vec = vector([0] * len(alpha))
        
        # Prepare candidate vectors for all small denominators
        candidates = []
        candidate_ratios = []
        for q in range(1, max_denominator+1):
            for p in range(-max_denominator, max_denominator+1):
                # This fast check catches almost everything if they are collinear
                if zero_vec.equals(p*alpha - q*beta):
                    return (True, p/q) if with_ratio else True
                candidates.append(p*alpha - q*beta)
                candidate_ratios.append(p / q)
            
        # Loop through candidates ONLY if the fast zero_vec check didn't catch it
        for candidate, ratio in zip(candidates, candidate_ratios):
            if in_column_span(candidate, self.lattice_matrix):
                return (True, ratio) if with_ratio else True
                
        return (False, None) if with_ratio else False

    def reflect_root(self, hyperplane_root, root_to_reflect):
        # Return the reflection of beta across the hyperplane perpendicular to alpha
        alpha = hyperplane_root
        beta = root_to_reflect
        numerator = 2*alpha.dot(beta)
        denominator = alpha.dot(alpha)
        assert numerator % denominator == 0
        return beta - (numerator // denominator) * alpha

    def is_multipliable_root(self, vector_to_test):
        # A root alpha is multipliable if 2*alpha is also a root
        # In a reduced root system, there are no multipliable roots,
        # but the BC root system has multipliable roots.
        return self.is_root(vector_to_test) and self.is_root(2*vector_to_test)

    def integer_linear_combos(self, alpha, beta):
        # Given two roots alpha and beta,
        # a linear combination of them is an expression i*alpha+j*beta
        # where i and j are positive integers
        # and i*alpha + j*beta is also a root
        
        # Because of various theoretical considerations, 
        # if (i,j) is a pair such that i*alpha + j*beta is a root, 
        # then for any pair (i', j') with 0<i'<i and 0>=<j'<j 
        # it must be that (i')*alpha + (j')*beta is also a root
        
        # Therefore, to find the set of all pairs (i,j) such that i*alpha+j*beta
        # is a root, we start with the pair (1,1) and successively test
        # incremented versions of this, i.e. if (1,1) is a valid linear combination,
        # then we try (1,2) and (2,1).
        # If (1,2) is a valid pair, then try (2,2) and (1,3).
        # If (2,1) is not a valid pair, there is no need to try (3,1) for example.
        
        # The data from this calculation is returned in the form of a dictionary
        # with keys that are tuple pairs (i,j)
        # and the associated value is the root (vector object) i*alpha+j*beta
        
        assert self.is_root(alpha) and self.is_root(beta), "Can't compute integer linear combos with non-roots"
    
        combos = {}
        start = alpha + beta
        if not self.is_root(start):
            return combos
    
        combos[(1, 1)] = start
        queue = [(1, 1, start)]  # store (i, j, root) tuples
    
        while queue:
            i, j, root = queue.pop(0)
    
            # Try adding alpha
            next_root = root + alpha
            key = (i + 1, j)
            if self.is_root(next_root) and key not in combos:
                combos[key] = next_root
                queue.append((i + 1, j, next_root))
    
            # Try adding beta
            next_root = root + beta
            key = (i, j + 1)
            if self.is_root(next_root) and key not in combos:
                combos[key] = next_root
                queue.append((i, j + 1, next_root))
    
        return combos

    def __repr__(self):
        s = f"Root system of type {self.name_string}"
        if self.is_irreducible:
            s += f"\nDynkin diagram: {visualize_graph(self.dynkin_graph)}"
            s += f"\nCartan matrix: \n{self.cartan_matrix}"
        else:
            s += "\nDynkin diagrams of components: "
            s += "\t".join([visualize_graph(c.dynkin_graph) for c in self.components])
        s += f"\nReduced: {self.is_reduced}"
        s += f"\nSimply laced: {self.is_simply_laced}"
        s += f"\nNumber of roots: {len(self.root_list)}"
        table = []
        for alpha in self.root_list:
            alpha_check = self.coroot_dict[alpha]
            table.append([alpha, alpha_check])
        headers = ("Root", "Coroot")
        s += "\n" + tabulate(table, headers)
        return s

    def _get_tikz_dynkin_tex(self, subgraph):
        # Assert that we are passing a single, valid irreducible graph slice
        degrees = {v: len(neighbors) for v, neighbors in subgraph.items()}
        fork_nodes = [v for v, deg in degrees.items() if deg >= 3]
        
        # An irreducible system has at most one fork node (Type D, E)
        assert len(fork_nodes) <= 1, "Passed subgraph is not irreducible (multiple branching junctions found)"
        
        node_positions = {}

        # --- Branched component layout (D, E) ---
        if fork_nodes:
            fork = fork_nodes[0]
            
            def get_path_from_neighbor(start_node):
                path = [start_node]
                visited = {fork, start_node}
                curr = start_node
                while True:
                    next_nodes = [n for n in subgraph[curr] if n not in visited]
                    if not next_nodes: 
                        break
                    curr = next_nodes[0]
                    path.append(curr)
                    visited.add(curr)
                return path

            branches = [get_path_from_neighbor(n) for n in subgraph[fork].keys()]
            branches.sort(key=len)
            
            vertical_branch = branches[0]
            left_branch = branches[1]
            right_branch = branches[2]
            left_branch.reverse()
            
            backbone = left_branch + [fork] + right_branch
            
            for x_idx, node in enumerate(backbone):
                node_positions[node] = (x_idx * 1.5, 0.0)
                    
            fork_x, _ = node_positions[fork]
            for y_idx, node in enumerate(vertical_branch):
                node_positions[node] = (fork_x, (y_idx + 1) * 1.2)

        # --- Linear component layout (A, B, C, F, G, or isolated points) ---
        else:
            leaf_nodes = [v for v, deg in degrees.items() if deg == 1]
            start = min(leaf_nodes) if leaf_nodes else min(subgraph)
            
            visited = {start}
            order = [start]
            current = start
            while True:
                unvisited = [u for u in subgraph[current] if u not in visited]
                if not unvisited: 
                    break
                next_node = unvisited[0]
                order.append(next_node)
                visited.add(next_node)
                current = next_node
                
            for x_idx, node in enumerate(order):
                node_positions[node] = (x_idx * 1.5, 0.0)

        # --- BUILD ENVIRONMENT STRING ---
        comp_tex = []
        comp_tex.append(r"\begin{tikzpicture}[")
        comp_tex.append(r"    node/.style={circle, draw, fill=black!5, minimum size=6mm, inner sep=0pt},")
        comp_tex.append(r"    arrow/.style={-latex, thick},")
        comp_tex.append(r"    doublearrow/.style={double, double distance=2pt, -latex, thick},")
        comp_tex.append(r"    triplearrow/.style={double, double distance=4pt, -latex, thick},")
        comp_tex.append(r"    plain/.style={thick},")
        comp_tex.append(r"    baseline=(current bounding box.center)")
        comp_tex.append(r"]")

        # Render nodes
        for node, (x, y) in node_positions.items():
            comp_tex.append(f"    \\node[node] ({node}) at ({x:.2f}, {y:.2f}) {{${node}$}};")

        # Render edges
        drawn_edges = set()
        for u in subgraph:
            for v in subgraph[u]:
                if (u, v) in drawn_edges or (v, u) in drawn_edges:
                    continue
                
                mult_u_to_v = subgraph[u].get(v, 1)
                mult_v_to_u = subgraph[v].get(u, 1)
                
                if mult_u_to_v > mult_v_to_u:
                    if mult_u_to_v == 2:
                        comp_tex.append(f"    \\draw[doublearrow] ({u}) -- ({v});")
                    elif mult_u_to_v == 3:
                        comp_tex.append(f"    \\draw[plain] ({u}) -- ({v});")
                        comp_tex.append(f"    \\draw[doublearrow] ({u}) -- ({v});")
                    else:
                        comp_tex.append(f"    \\draw[arrow] ({u}) -- node[above] {{{mult_u_to_v}}} ({v});")
                elif mult_u_to_v < mult_v_to_u:
                    if mult_v_to_u == 2:
                        comp_tex.append(f"    \\draw[doublearrow] ({v}) -- ({u});")
                    elif mult_v_to_u == 3:
                        comp_tex.append(f"    \\draw[plain] ({v}) -- ({u});")
                        comp_tex.append(f"    \\draw[doublearrow] ({v}) -- ({u});")
                    else:
                        comp_tex.append(f"    \\draw[arrow] ({v}) -- node[above] {{{mult_v_to_u}}} ({u});")
                else:
                    comp_tex.append(f"    \\draw[plain] ({u}) -- ({v});")
                
                drawn_edges.add((u, v))

        comp_tex.append(r"\end{tikzpicture}")
        return "\n".join(comp_tex)

    def to_tex(self):
        # Generate a formatted LaTeX string for the base properties, matrices, and roots table
        
        # 1. Base summary properties in a table
        tex = "\\subsection*{Summary Properties}\n"
        tex += r"\begin{tabular}{|l|l|}" + "\n"
        tex += r"    \hline" + "\n"
        
        safe_name_string = self.name_string.replace("_", "\\_")
        tex += f"    Dynkin type & \\texttt{{{safe_name_string}}} \\\\\n"
        tex += r"    \hline" + "\n"
        tex += f"    Reduced & {self.is_reduced} \\\\\n"
        tex += r"    \hline" + "\n"
        tex += f"    Simply laced & {self.is_simply_laced} \\\\\n"
        tex += r"    \hline" + "\n"
        tex += f"    Number of roots & {len(self.root_list)} \\\\\n"
        tex += r"    \hline" + "\n"
        tex += "\\end{tabular}\n\n"

        # =====================================================================
        # DYNKIN DIAGRAM SECTION (Loops over connected components)
        # =====================================================================
        tex += "\\subsection{Dynkin diagram}\n"
        
        # Gather subgraphs cleanly based on whether the system is reducible or not
        subgraphs = []
        if self.is_irreducible:
            # If the top-level system has the attribute, use it directly
            if hasattr(self, 'dynkin_graph'):
                subgraphs.append(self.dynkin_graph)
            else:
                # Fallback if an irreducible system hasn't calculated it yet
                subgraphs.append(build_directed_dynkin_graph(self.simple_roots))
        else:
            # Reducible system: extract the graph from each individual component
            for comp in self.components:
                if hasattr(comp, 'dynkin_graph'):
                    subgraphs.append(comp.dynkin_graph)
                else:
                    subgraphs.append(build_directed_dynkin_graph(comp.simple_roots))

        # Generate a standalone tikzpicture for each isolated component subgraph
        tikz_diagrams = []
        for subgraph in subgraphs:
            tikz_str = self._get_tikz_dynkin_tex(subgraph)
            tikz_diagrams.append(tikz_str)

        if tikz_diagrams:
            tex += "\\begin{center}\n"
            # Separate components with a horizontal gap (\quad) side-by-side
            tex += "\n\\quad\n".join(tikz_diagrams) + "\n"
            tex += "\\end{center}\n\n"

        # 2. Structural Details (Cartan Matrix / Components)
        if self.is_irreducible:
            cartan_latex = sp.latex(sp.Matrix(self.cartan_matrix))
            tex += "\\subsection{Cartan matrix}\n"
            tex += f"\\[\n{cartan_latex}\n\\]\n\n"
        else:
            tex += "\\subsection{Irreducible components}\n"
            tex += "This is a reducible root system with the following components:\n\n"
            tex += r"\noindent \begin{tabular}{|c|c|}" + "\n"
            tex += r"    \hline" + "\n"
            tex += "    \\textbf{Dynkin type} & \\textbf{Rank} \\\\\n"
            tex += r"    \hline" + "\n"
            for c in self.components:
                safe_comp_name = c.name_string.replace("_", "\\_")
                tex += f"    \\texttt{{{safe_comp_name}}} & {c.rank} \\\\\n"
            tex += "\\end{tabular}\n\n"

        # 3. Roots, coroots, and root properties table
        tex += "\\subsection{Roots and coroots}\n"
        tex += "\\begin{center}\n"
        tex += r"\begin{tabular}{|l|l|c|c|c|}" + "\n"
        tex += r"    \hline" + "\n"
        tex += "    \\textbf{Root} & \\textbf{Coroot} & \\textbf{Simple} & \\textbf{Sign} & \\textbf{Multipliable} \\\\\n"
        tex += r"    \hline" + "\n"
        
        for alpha in self.root_list:
            alpha_coroot = self.coroot_dict[alpha]
            alpha_latex = f"${sp.latex(alpha)}$"
            coroot_latex = f"${sp.latex(alpha_coroot)}$"
            
            if self.is_irreducible:
                target_simple = self.simple_roots
                target_positive = self.positive_roots
            else:
                comp = next((c for c in self.components if any(alpha.equals(r) for r in c.root_list)), None)
                target_simple = comp.simple_roots if comp else []
                target_positive = comp.positive_roots if comp else []

            is_simple = any(alpha.equals(s) for s in target_simple)
            is_positive = any(alpha.equals(p) for p in target_positive)
            is_multipliable = self.is_multipliable_root(alpha)
            
            simple_str = "Yes" if is_simple else "No"
            sign_str = "$+$" if is_positive else "$-$"
            multipliable_str = "Yes" if is_multipliable else "No"
            
            tex += f"    {alpha_latex} & {coroot_latex} & {simple_str} & {sign_str} & {multipliable_str} \\\\\n"
            tex += r"    \hline" + "\n"
            
        tex += "\\end{tabular}\n"
        tex += "\\end{center}\n"

        # Generate the formatted LaTeX string strictly for the combinations section
        pair_rows = []
        for idx, alpha in enumerate(self.root_list):
            for beta in self.root_list[idx + 1:]:
                combos = self.integer_linear_combos(alpha, beta)
                if combos:
                    alpha_lat = sp.latex(alpha)
                    beta_lat = sp.latex(beta)
                    
                    sorted_pairs = sorted(combos.keys())
                    pairs_str = ", ".join(f"({i},{j})" for (i, j) in sorted_pairs)
                    
                    pair_rows.append(f"    ${alpha_lat}$ & ${beta_lat}$ & ${pairs_str}$ \\\\\n")

        combo_table = ""
        if pair_rows:
            combo_table += "\\begin{center}\n"
            combo_table += r"\begin{tabular}{|l|l|c|}" + "\n"
            combo_table += r"    \hline" + "\n"
            combo_table += "    $\\alpha$ & $\\beta$ & Pairs $(i,j)$ \\\\\n"
            combo_table += r"    \hline" + "\n"
            for row in pair_rows:
                combo_table += row
                combo_table += r"    \hline" + "\n"
            combo_table += "\\end{tabular}\n"
            combo_table += "\\end{center}\n"
        else:
            combo_table += "\\noindent No pairs of roots generate positive integer linear combinations.\n"

        return tex, combo_table

    def verify_root_system_axioms(self, display = True):
        # A list of tests to verify the root system axioms
        # for a given root_system object.
        
        # These properties are:
        #   1. The zero vector is not a root.
        #   2. For any root alpha, -alpha is also a root, 
        #           and the only scalar multiples of alpha that are allowed as roots are +/-alpha, +/-2*alpha, +/-(1/2)*alpha.
        #           NOTE: The classical definition of root system includes a stronger axiom,
        #                   which is that the only scalar multiples of a root alpha that are
        #                   roots are alpha and -alpha, but this class allows for non-reduced
        #                   root systems which weaken this axiom.
        #   3. For any pair of roots alpha and beta, the reflection of alpha 
        #           across the hyperplane perpendicular to beta is a root.
        #   4. For any pair of roots alpha and beta, the "angle bracket" <alpha,beta> is an integer.
        #           <alpha, beta> = 2 * dot(alpha,beta) / dot(beta,beta)
        #   5. For any root alpha, the dot product of alpha with its coroot alpha^ is 2.
        
        if display: print(f'\nVerifying root system axioms for the {self.name_string} root system.')
        
        if display: print('\tChecking that zero is not a root... ',end='')
        zero_vector = vector([0] * self.vector_length)
        assert not(self.is_root(zero_vector)), "Zero vector should not be a root"
        if display: print('done.')
        
        if display: print('\tChecking that the negation of a root is a root... ',end='')
        for alpha in self.root_list:
            assert self.is_root(-1*alpha, with_equivalent = False), \
                "Negative of root is not a root"
        if display: print('done.')
        
        if display: print('\tChecking that a reflection of a root is a root... ',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                assert self.is_root(self.reflect_root(alpha,beta)), \
                "Reflection of a root is not a root but should be"
        if display: print('done.')
        
        if display: print('\tChecking that the angle bracket of two roots is an integer... ',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                angle_bracket = 2* alpha.dot(beta) / beta.dot(beta)
                assert angle_bracket == int(angle_bracket), \
                    "Angle bracket of roots should be an integer"
        if display: print('done.')
        
        if display: print('\tChecking that the dot product of a root with its coroot is 2... ',end='')
        for alpha in self.root_list:
            alpha_check = self.coroot_dict[alpha]
            angle_bracket = alpha.dot(alpha_check)
            assert angle_bracket == 2, "Dot product of a root with its coroot is not 2"
        if display: print('done.')
        
        if display: print('\tChecking that ratios between proportional roots are +/-1, +/-2, or +/-0.5... ',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                proportional, ratio = self.is_proportional(alpha,beta,with_ratio=True)
                if proportional: assert ratio in (1.0,-1.0,2.0,-2.0,0.5,-0.5), "Invalid ratio between roots"
        if display: print('done.')
        
        if display: print('Root system axiom checks completed.')

def construct_root_list_from_dynkin_type(dynkin_type, rank):
    # Construct a "standard" vector representation/model
    # of a root system with specified Dynkin type and rank
    
    # The valid Dynkin types are A through G, plus the BC non-reduced root system.
    # The pinned group side of this code base only supports groups of types
    # A, B, C, BC, and some of type D, but the root system class supports types E, F, and G.
    # Types E, F, and G are called the "exceptional" root systems, as they do not belong
    # to an infinite family (like A, B, C, BC, and D)
    
    assert dynkin_type in ['A','B','C','BC','D','E','F','G'], "Invalid Dynkin type"
    assert rank >= 1, "Invalid rank"
    
    root_list = [] 
    vector_length = None
    
    if dynkin_type == 'A':
        # We represent A_q as a set of vectors in R^(q+1), as 
        # vectors with two nonzero entries, one entry is +1, and another entry is -1
        vector_length = rank + 1
        root_list = [
            vector([1 if i == p else -1 if i == q else 0 for i in range(vector_length)])
            for p in range(vector_length)
            for q in range(vector_length)
            if p != q
        ]
    
    elif dynkin_type in ['B','C','BC','D']:
        # We represent B_q, C_q, D_q and BC_q as sets of vectors in R^q
        # There are three types of vectors:
        #   1) vectors with one nonzero entry of +/-1
        #   2) vectors with one nonzero entry of +/-2
        #   3) vectors with two nonzero entries of +/-1, with any combination of signs
        # D_q contains (1)
        # B_q contains (1) and (2)
        # C_q contains (2) and (3)
        # BC_q contains (1), (2), and (3)
        vector_length = rank
        two_nonzero_entries = [
            vector([ (a if i == p else b if i == q else 0) for i in range(vector_length) ])
            for p in range(vector_length)
            for q in range(p+1, vector_length)
            for a in (1, -1)
            for b in (1, -1)
        ]
        one_nonzero_entry_1 = [
            vector([s if i == p else 0 for i in range(vector_length)])
            for p in range(vector_length)
            for s in (1, -1)
        ]
        one_nonzero_entry_2 = [
            vector([s if i == p else 0 for i in range(vector_length)])
            for p in range(vector_length)
            for s in (2, -2)
        ]
        if dynkin_type == 'B':
            root_list = two_nonzero_entries + one_nonzero_entry_1
        elif dynkin_type == 'C':
            root_list = two_nonzero_entries + one_nonzero_entry_2
        elif dynkin_type == 'D':
            if rank == 1:
                root_list = one_nonzero_entry_1
            else:
                root_list = two_nonzero_entries
        else:
            # type BC
            root_list = two_nonzero_entries + one_nonzero_entry_1 + one_nonzero_entry_2
        
    elif dynkin_type == 'E':
        assert rank in (6,7,8)
        vector_length = 8
        
        # Following this page: https://en.wikipedia.org/wiki/E8_(mathematics)#E8_root_system
        # we realize E_8 as the set of vectors in R^8 with squared length equal to 2, 
        # such that all coordinates are integers or half-integers,
        # and the sum of all coordinates is even
        # Then we realize E_7 and E_6 as subsets of this
        
        # More explicitly, this consists of two subsets of vectors:
        #   1) A copy of D_8, i.e. all vectors with two nonzero entries of +/-1 in any combination of signs (112 roots))
        #   2) All vectors with 8 entries of +/-1/2 with an even number of negative signs (240 roots)

        two_nonzero_entries = [
            vector([ (a if i == p else b if i == q else 0) for i in range(vector_length) ])
            for p in range(vector_length)
            for q in range(p+1, vector_length)
            for a in (1, -1)
            for b in (1, -1)
        ]
        assert len(two_nonzero_entries) == 112
        
        half_vectors_sum_even =  [
            vector(v)
            for v in itertools.product((1/2, -1/2), repeat=8)
            if sum(v) % 2 == 0 
            # sum of coordinates being even is equivalent 
            # to having an even number of minus signs
        ]
        assert len(half_vectors_sum_even) == 128
        
        E_8_root_list = two_nonzero_entries + half_vectors_sum_even
        
        if rank == 8:
            root_list = E_8_root_list
        elif rank == 7:
            # We realize E_7 as the subset of E_8 consisting of vectors orthogonal to (0,0,0,0,0,0,1,1)
            E_7_root_list = [v for v in E_8_root_list if v[-2] + v[-1] == 0]
            root_list = E_7_root_list
        if rank == 6:
            # We realize E_6 as the subset of E_7 consisting of vectors orthogonal to (0,0,0,0,0,1,1,0)
            E_7_root_list = [v for v in E_8_root_list if v[-2] + v[-1] == 0]
            E_6_root_list = [v for v in E_7_root_list if v[-3] + v[-2] == 0]
            root_list = E_6_root_list
        
    elif dynkin_type == 'F':
        # We represent F_4 as a set of vectors in R^4
        #   -vectors with one nonzero entry of +/-2
        #   -vectors with two nonzero entries of +/-1, in any combination of signs
        #   -vectors with four nonzero entries of +/-1, in any combination of signs
        assert rank == 4
        vector_length = 4
        one_nonzero_entry_2 = [
            vector([s if i == p else 0 for i in range(vector_length)])
            for p in range(vector_length)
            for s in (2, -2)
        ]
        two_nonzero_entries = [
            vector([ (a if i == p else b if i == q else 0) for i in range(vector_length) ])
            for p in range(vector_length)
            for q in range(p+1, vector_length)
            for a in (1, -1)
            for b in (1, -1)
        ]
        four_nonzero_entries = [vector(v) for v in __import__('itertools').product((1, -1), repeat=vector_length)]
        root_list = one_nonzero_entry_2 + two_nonzero_entries + four_nonzero_entries
        
    elif dynkin_type == 'G':
        # We represent G_2 as a set of vectors in R^3
        # Specifically, as vectors in which there is one entry of 0, 2, or -2
        # and the other two components are 1 or -1, 
        # and the sum of all entries is zero
        assert rank == 2
        vector_length = 3
        allowed_patterns = [
            (2, -1, -1),
            (-2, 1, 1),
            (0, 1, -1),
        ]
        root_list = [
            vector(v)
            for pattern in allowed_patterns
            for v in set(itertools.permutations(pattern))
        ]
        
    else:
        assert False, f'There is no root system of Dynkin type {dynkin_type}'
    
    return root_list

def construct_root_system_from_dynkin_type(dynkin_type, rank):
    return root_system(root_list = construct_root_list_from_dynkin_type(dynkin_type, rank),
                       lattice_matrix = None)

def determine_irreducible_components(roots):
    # Determine the irreducible components of a root system.
    # Each component is a maximal set of roots that are connected via nonzero inner products.
    # Returns a list of lists, where each inner list is an irreducible component.

    unvisited = set(range(len(roots)))  # indices of roots not yet assigned to a component
    components = []

    while unvisited:
        # Start a new component from an arbitrary unvisited root
        start = unvisited.pop()
        component_indices = {start}
        stack = [start]

        while stack:
            root_index = stack.pop()
            # Find neighbors among unvisited roots
            neighbors = {j for j in unvisited if roots[root_index].dot(roots[j]) != 0}
            stack.extend(neighbors)
            component_indices.update(neighbors)
            unvisited -= neighbors  # mark neighbors as visited

        # Collect the actual roots for this component
        components.append([roots[i] for i in component_indices])

    return components

def choose_positive_roots(root_list, max_tries = 100, seed = 0):
    # Procedure: choose a hyperplane not containing any of the roots
    # then choose one of the sides of that hyperplane.
    # All roots on that side of the hyperplane are positive.
    
    # Note that this algorithm is non-deterministic 
    # except that a seed is used to get consistent results.
    # In general, there are many valid choices of positive and negative roots
    # for a root system. Any hyperplane that does not contain one of the roots
    # can be used to divide it up in this way. On the theoretical side,
    # the way to enumerate the different choices of positive roots
    # has to do with something called Weyl chambers and a transitive
    # action of the Weyl group. This implementation is much more 
    # quick and dirty, and just picks a random way to divide things up.
    
    vector_length = len(root_list[0])
    
    # Use a numpy random generator object so that
    # results of choosing positive roots are consistent
    # between different runs of the code
    rng = np.random.default_rng(seed)
    for t in range(max_tries):
        random_vec = vector(rng.random(vector_length))          # Generate a random vector
        dot_products = [random_vec.dot(r) for r in root_list]   # Compute dot product with every root
        if 0 in dot_products:
            # If the random vector is perpendicular to any roots, that causes a problem
            # This is very rare probabalistically, but it can theoretically happen
            # If it does happen, just choose a new random vector
            continue
        else:
            # Choose the positive roots to be roots with positive dot product with the random vector
            positive_roots = [r for (r, val) in zip(root_list, dot_products) if val > 0]
            return positive_roots
    raise RuntimeError("Failed to find a vector not proportional to a root.")

def choose_simple_roots(positive_roots):
    # A root is simple if it cannot be written as the sum of two other positive roots.

    simple_roots = []
    for alpha in positive_roots:
        is_sum = False
        for beta in positive_roots:
            if alpha.equals(beta): 
                continue
            # If (alpha - beta) is also a positive root, 
            # then alpha = beta + (alpha - beta) is not simple.
            if any((alpha - beta).equals(gamma) for gamma in positive_roots):
                is_sum = True
                break
        if not is_sum:
            simple_roots.append(alpha)
            
    return simple_roots

def build_cartan_matrix(simple_roots):
    # The cartan matrix of a root system is a matrix encoding
    # essentially all of the important structure of the root system
    # see https://en.wikipedia.org/wiki/Cartan_matrix
    # The (i,j) entry of the Cartan matrix is the angle bracket <alpha_i, alpha_j>
    # where (alpha_1, alpha_2, ... alpha_n) is a list of simple roots.
    
    # Part of the reason that the choice of positive and simple roots
    # can be randomized is that due to various theoretical considerations,
    # the Cartan matrix is independent of all of those choices.
    
    # Note that the Dynkin type / Dynkin diagram of a root system
    # can be fully recovered from the Cartan matrix.
    
    rank = len(simple_roots)
    A = np.zeros((rank, rank), dtype=int)
    for i, alpha in enumerate(simple_roots):
        for j, beta in enumerate(simple_roots):
            A[i, j] = int(round(2 * alpha.dot(beta) / beta.dot(beta)))
    return A

def build_directed_dynkin_graph(simple_roots):
    
    # See https://en.wikipedia.org/wiki/Dynkin_diagram for info on Dynkin diagrams.
    # The Dynkin diagram is a graphical combinatorial object
    # storing exactly the same information as the Cartan matrix,
    # which is to say it stores all of the important information
    # about a set of simple roots and the angles between them.
    
    cartan_matrix = build_cartan_matrix(simple_roots)
    assert len(simple_roots) == cartan_matrix.shape[0], \
        "Cartan matrix dimensions do not match number of simple roots"
    rank = len(simple_roots)
    graph = {i: {} for i in range(rank)}
    for i, alpha_i in enumerate(simple_roots):
        len_i = alpha_i.norm()
        for j, alpha_j in enumerate(simple_roots):
            if i == j:
                continue
            len_j = alpha_j.norm()
            a_ij = cartan_matrix[i, j]
            a_ji = cartan_matrix[j, i]
            if a_ij != 0:
                mult = max(abs(a_ij), abs(a_ji))
                if len_i > len_j: # i -> j
                    graph[i][j] = mult
                    graph[j][i] = 1
                elif len_i < len_j: # j -> i
                    graph[i][j] = 1
                    graph[j][i] = mult
                else: # equal length, symmetric single edge
                    graph[i][j] = 1
                    graph[j][i] = 1
    return graph

def determine_dynkin_type(dynkin_graph):
    # Determine the dynkin type of an irreducible, reduced root system from its
    # (weighted, directed) Dynkin graph
    
    # If the root system is not reduced, then this method will not help.
    # In that case, you can't actually tell apart types B and BC from the Dynkin graph anyway,
    # so it would be impossible to classify.
    
    my_rank = len(dynkin_graph)
    
    if my_rank == 1: 
        # This needs to be checked before computing other properties,
        # because other calculations fail to work in this case
        return ('A', 1)

    # Compute some useful characteristics of the graph
    degree_dict = {v: len(dynkin_graph[v]) for v in dynkin_graph}
    degrees = list(degree_dict.values())
    edge_multiplicities = sorted(mult for v in dynkin_graph for mult in dynkin_graph[v].values())
    is_simply_laced = (max(edge_multiplicities, default = -1) == 1)
    nodes_with_multiple_leaf_neighbors = [
        v for v in dynkin_graph
        if sum(1 for u in dynkin_graph[v] if len(dynkin_graph[u]) == 1) > 1
    ]
    leaf_nodes = [v for v in dynkin_graph if len(dynkin_graph[v]) == 1]
    single_edge_leaf_nodes = []
    
    for v in leaf_nodes:
        simple_edges = True
        outgoing_edges = sum(dynkin_graph[v].values())
        if outgoing_edges > 1:
            simple_edges = False
            break
        for u in dynkin_graph:
            if v in dynkin_graph[u] and dynkin_graph[u][v] > 1:
                simple_edges = False
                break
        if simple_edges:
            single_edge_leaf_nodes.append(v)
    num_single_edge_leaf_nodes = len(single_edge_leaf_nodes)
    double_edge_to_leaf = False
    for v in leaf_nodes:
        for u in dynkin_graph:
            if v in dynkin_graph[u] and dynkin_graph[u][v] > 1:
                double_edge_to_leaf = True

    # Sort into Dynkin type based on characteristics
    if is_simply_laced:
        if 3 not in degrees: # no forks, so type A
            my_type = 'A'
        elif len(nodes_with_multiple_leaf_neighbors) >= 1:
            # D has a node with two leaf neighbors, E does not
            my_type = 'D'
            assert(my_rank >= 4)
        else:
            my_type = 'E'
            assert(my_rank in (6,7,8))
    else: # non-simply-laced case
        if 3 in edge_multiplicities:
            my_type = 'G'
            assert(my_rank == 2)
        elif num_single_edge_leaf_nodes == 2:
            my_type = 'F'
            assert(my_rank == 4)
        else:
            if double_edge_to_leaf:
                my_type = 'B'
                assert(my_rank >= 2)
            else:
                my_type = 'C'
                assert(my_rank >= 3)

    return my_type, my_rank

def expected_number_of_roots(dynkin_type, rank):
    # Return the number of roots belonging to the root system of given type and rank
    # This is mostly just used for validation/testing.
    assert dynkin_type in valid_dynkin_types, "Invalid Dynkin type"
    if dynkin_type == 'A':
        return rank*(rank+1)
    elif dynkin_type in ('B','C'):
        return 2*(rank**2)
    elif dynkin_type == 'BC':
        return 2*(rank**2) + 2*rank
    elif dynkin_type == 'D':
        if rank == 1: 
            return 2
        else:
            return 2*rank*(rank-1)
    elif dynkin_type == 'E':
        assert rank in (6,7,8)
        if rank == 6: return 72
        if rank == 7: return 126
        if rank == 8: return 240
    elif dynkin_type == 'F':
        assert rank == 4
        return 48
    elif dynkin_type == 'G':
        assert rank == 2
        return 12
    assert False, "Invalid Dynkin type"

def test_dynkin_constructor(upper_bound = 5):
    # Use the constructor for a variety of root systems and verify the root system axioms
    print("Testing constructor for root systems from Dynkin type...")
    
    # Type A
    for q in range(1,upper_bound + 1):
        print(f"\nConstructing A{q}",end="")
        A_q = construct_root_system_from_dynkin_type('A',q)
        print(f"\nDynkin diagram of type A{q}:", visualize_graph(A_q.dynkin_graph),end="")
        assert len(A_q.root_list) == expected_number_of_roots('A',q), \
            f"Expected {expected_number_of_roots('A',q)} for A_{q} but constructor produced {len(A_q.root_list)}"
        assert A_q.dynkin_type == 'A'
        assert A_q.rank == q
        assert A_q.is_reduced
        assert A_q.is_irreducible
        assert A_q.is_simply_laced
        A_q.verify_root_system_axioms()
    
    # Type B
    for q in range(1,upper_bound + 1):
        print(f"\nConstructing B{q}",end="")
        B_q = construct_root_system_from_dynkin_type('B',q)
        print(f"\nDynkin diagram of type B{q}:", visualize_graph(B_q.dynkin_graph),end="")
        assert len(B_q.root_list) == expected_number_of_roots('B',q), \
            f"Expected {expected_number_of_roots('B',q)} for B_{q} but constructor produced {len(B_q.root_list)}"
        if q == 1:
            # B_1 and A_1 are isomorphic, and the traditional label is A_1
            assert B_q.dynkin_type == 'A'
            assert B_q.is_simply_laced
        else:
            assert B_q.dynkin_type == 'B'
            assert not B_q.is_simply_laced
        assert B_q.rank == q
        assert B_q.is_reduced
        assert B_q.is_irreducible
        B_q.verify_root_system_axioms()
        
    # Type C
    for q in range(1,upper_bound + 1):
        print(f"\nConstructing C{q}",end="")
        C_q = construct_root_system_from_dynkin_type('C',q)
        print(f"\nDynkin diagram of type C{q}:", visualize_graph(C_q.dynkin_graph),end="")
        assert len(C_q.root_list) == expected_number_of_roots('C',q), \
            f"Expected {expected_number_of_roots('C',q)} for C_{q} but constructor produced {len(C_q.root_list)}"
        if q == 1:
            # C_1 and A_1 are isomorphic, and the traditional label is A_1
            assert C_q.dynkin_type == 'A'
            assert C_q.is_simply_laced
        elif q == 2:
            # C_2 and B_2 are isomorphic, and the traditional label is B_2
            assert C_q.dynkin_type == 'B'
            assert not C_q.is_simply_laced
        else:
            assert C_q.dynkin_type == 'C'
            assert not C_q.is_simply_laced
        assert C_q.rank == q
        assert C_q.is_reduced
        assert C_q.is_irreducible
        C_q.verify_root_system_axioms()
        
    # Type BC
    for q in range(1,upper_bound + 1):
        print(f"\nConstructing BC{q}",end="")
        BC_q = construct_root_system_from_dynkin_type('BC',q)
        print(f"\nDynkin diagram of type BC{q}:", visualize_graph(BC_q.dynkin_graph),end="")
        assert len(BC_q.root_list) == expected_number_of_roots('BC',q), \
            f"Expected {expected_number_of_roots('BC',q)} for BC_{q} but constructor produced {len(BC_q.root_list)}"
        assert BC_q.dynkin_type == 'BC'
        assert BC_q.rank == q
        assert not BC_q.is_reduced
        assert not BC_q.is_simply_laced
        assert BC_q.is_irreducible
        BC_q.verify_root_system_axioms()
    
    # Type D
    for q in range(1,upper_bound + 1):
        print(f"\nConstructing D{q}",end="")
        D_q = construct_root_system_from_dynkin_type('D',q)
        if q != 2: print(f"\nDynkin diagram of type D{q}:", visualize_graph(D_q.dynkin_graph),end="")
        assert len(D_q.root_list) == expected_number_of_roots('D',q), \
            f"Expected {expected_number_of_roots('D',q)} for D_{q} but constructor produced {len(D_q.root_list)}"
        if q in (1,3):
            # A_1 and D_1 are isomorphic, as are A_3 and D_3
            assert D_q.dynkin_type == 'A'
            assert D_q.rank == q
            assert D_q.is_irreducible
        elif q == 2:
            # D_2 is isomorphic to A_1 x A_1
            assert D_q.dynkin_type == ['A','A']
            assert D_q.rank == [1,1]
            assert not D_q.is_irreducible
        else:
            assert D_q.dynkin_type == 'D'
            assert D_q.is_irreducible
        assert D_q.is_simply_laced
        assert D_q.is_reduced
        D_q.verify_root_system_axioms()
    
    # Type F
    print("\nConstructing F4",end="")
    F_4 = construct_root_system_from_dynkin_type('F',4)
    print("\nDynkin diagram of type F4:", visualize_graph(F_4.dynkin_graph),end="")
    assert len(F_4.root_list) == expected_number_of_roots('F',4), \
        f"Expected {expected_number_of_roots('F',4)} for F_4 but constructor produced {len(F_4.root_list)}"
    assert F_4.dynkin_type == 'F'
    assert F_4.rank == 4
    assert not F_4.is_simply_laced
    assert F_4.is_reduced
    assert F_4.is_irreducible
    F_4.verify_root_system_axioms()
    
    # Type G
    print("\nConstructing G2",end="")
    G_2 = construct_root_system_from_dynkin_type('G',2)
    print("\nDynkin diagram of type G2:", visualize_graph(G_2.dynkin_graph),end="")
    assert len(G_2.root_list) == expected_number_of_roots('G',2), \
        f"Expected {expected_number_of_roots('G',2)} for G_2 but constructor produced {len(G_2.root_list)}"
    assert G_2.dynkin_type == 'G'
    assert G_2.rank == 2
    assert not G_2.is_simply_laced
    assert G_2.is_reduced
    assert G_2.is_irreducible
    G_2.verify_root_system_axioms()
    
    # Type E    
    for q in (6,7,8):
        print(f"\nConstructing E{q}",end="")
        E_q = construct_root_system_from_dynkin_type('E',q)
        print(f"\nDynkin diagram of type E{q}:", visualize_graph(E_q.dynkin_graph),end="")
        assert len(E_q.root_list) == expected_number_of_roots('E',q), \
            f"Expected {expected_number_of_roots('E',q)} for E_{q} but constructor produced {len(E_q.root_list)}"
        assert E_q.dynkin_type == 'E'
        assert E_q.rank == q
        assert E_q.is_simply_laced
        assert E_q.is_reduced
        assert E_q.is_irreducible
        E_q.verify_root_system_axioms()
        
    print("\nAll tests passed for root system constructor.")

def test_dynkin_classifier():
    # Run a battery of tests to verify that the dynkin type classifier works
    print("Testing Dynkin type classifier...")
    for name in directed_dynkin_graphs:
        print("\nName:",name)
        true_type = name[0]
        true_rank = int(name[-1])
        print("\tTrue type:",true_type)
        print("\tTrue rank:",true_rank)
        graph = directed_dynkin_graphs[name]
        print("\tGraph visualization:", visualize_graph(graph))
        print("\tGraph as dictionary:",graph)
        calculated_type, calculated_rank = determine_dynkin_type(graph)
        print("\tCalculated type:",calculated_type)
        print("\tCalculated rank:",calculated_rank)
        assert true_type == calculated_type
        assert true_rank == calculated_rank
    print("\nAll tests passed for Dynkin classifier.")

def test_root_system_print(upper_bound = 5):
    
    # Types A, B, C, BC, D
    for q in range(1, upper_bound):
        A_q = construct_root_system_from_dynkin_type('A',q)
        print(A_q)
        print()        
        
        B_q = construct_root_system_from_dynkin_type('B',q)
        print(B_q)
        print()        
        
        C_q = construct_root_system_from_dynkin_type('C',q)
        print(C_q)
        print()   
        
        BC_q = construct_root_system_from_dynkin_type('BC',q)
        print(BC_q)
        print()
        
        D_q = construct_root_system_from_dynkin_type('D',q)
        print(D_q)
        print()
    
    # Type F
    F_4 = construct_root_system_from_dynkin_type('F',4)
    print(F_4)
    print()
    
    # Type G
    G_2 = construct_root_system_from_dynkin_type('G',2)
    print(G_2)
    print()
    
    # Type E    
    for q in (6,7,8):
        E_q = construct_root_system_from_dynkin_type('E',q)
        print(E_q)
        print()

def run_tests():
    test_dynkin_classifier()
    print()
    test_dynkin_constructor(upper_bound = 5)
    print()
    test_root_system_print(upper_bound = 5)
    
if __name__ == "__main__":
    run_tests()
    