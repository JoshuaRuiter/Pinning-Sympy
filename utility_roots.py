# Various helper methods related to roots, characters, and root systems

import itertools
import numpy as np
import sympy as sp
import time
from utility_general import is_diagonal

# Low rank examples of Dynkin graphs, used for testing purposes
# Each graph is a dictionary of dictionaries.
# Top-level keys are integer indices 0, 1, 2, 3, ... representing nodes.
# Top-level values are dictionaries {node1: edges1, node2 : edges 2}
# Inner-level keys are also integer indices representing nodes
# Inner-level values are the number of edges between the outer-level key and the inner-level key.

# For example, the Dynkin diagram of type A2 is {0: {1: 1}, 1: {0: 1}},
# which represents a graph with two nodes (0 and 1) and one edge between them.

# To store information about directed edges, we use the following convention: 
# given a graph dictionary g, and nodes i and j,
# g[i][j] and g[j][i] together store information about the weighted directed edge between i and j
# If g[i][j] = g[j][i] = 1, then there is a single undirected edge
# If g[i][j] = n > 1, then there is a weighted edge of multiplicity n pointing from node i to node j
    # In this case, we make the convention that g[j][i] = 1
    # In terms of root systems, this indicates that the node i represents a longer root than the root represented by node j.

# For example, the Dynkin diagram of type B2 is {0: {1: 2}, 1: {0: 1}}
# which indicates that there are two nodes, 0 and 1,
# and g[0][1] = 2 means that there is an edge of weight 2 pointing from node 0 to node 1
# We also have g[1][0] = 1 as set by convention.

directed_dynkin_graphs = {
    
    # ----- Type A_n -----
    "A1": {0: {}},
    "A2": {0: {1: 1}, 1: {0: 1}},
    "A3": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1}},
    "A4": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 1}},

    # ----- Type B_n -----
    "B2": {0: {1: 2}, 1: {0: 1}},
    "B3": {0: {1: 1}, 1: {0: 1, 2: 2}, 2: {1: 1}},
    "B4": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 2}, 3: {2: 1}},

    # ----- Type C_n -----    
    "C3": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 2}},
    "C4": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 2}},

    # ----- Type D_n -----
    "D4": {0: {1: 1}, 1: {0: 1, 2: 1, 3: 1}, 2: {1: 1}, 3: {1: 1}},
    "D5": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1, 4: 1}, 3: {2: 1}, 4: {2: 1}},
    "D6": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 1, 4: 1, 5:1}, 4: {3: 1}, 5: {3: 1}},
    "D7": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 1, 4: 1}, 
                      4: {3: 1, 5: 1, 6: 1}, 5: {4: 1}, 6: {4: 1}},

    # ----- Exceptional types -----
    "E6": {0: {1: 1}, 1: {0: 1, 3: 1}, 2: {3: 1}, 3: {1: 1, 2: 1, 4: 1}, 4: {3: 1, 5: 1}, 5: {4: 1}},
    "E7": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 1, 4: 1, 6: 1}, 
                      4: {3: 1, 5: 1}, 5: {4: 1}, 6: {3: 1}},
    "E8": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 1, 4: 1, 7: 1}, 
                      4: {3: 1, 5: 1}, 5: {4: 1, 6: 1}, 6: {5: 1}, 7: {3: 1}},

    "F4": {0: {1: 1}, 1: {0: 1, 2: 2}, 2: {1: 1, 3: 1}, 3: {2: 1}},
    "G2": {0: {1: 3}, 1: {0: 1}},
}

# vector is basically a light wrapper for the tuple class,
# which allows for entry-wise addition and scalar multiplication
# in the usual manner of vectors.
# At some point I tried using Numpy arrays for this kind of thing, 
# but Numpy arrays are not hashable so they cannot be used as keys
# for dictionary objects. Since this vector type is an extension
# of tuple, it is hashable and thus can be used as keys for dictionaries.
class vector(tuple):
    def __new__(cls, iterable):
        values = tuple(iterable)
        for x in values:
            if not isinstance(x, (int, float, complex)):
                raise TypeError(f"vector entries must be int, float, or complex, got {type(x)}")
        return super().__new__(cls, iterable)

    # Addition (entry-wise)
    def __add__(self, other):
        assert isinstance(other, vector), "Invalid vector addition"
        assert len(self) == len(other), "Can't add vectors of different lengths"
        result = vector(tuple(a+b for a, b in zip(self, other)))
        assert len(result) == len(self), "Addition changed vector length"
        return result

    # Subtraction (entry-wise)
    def __sub__(self, other):
        assert isinstance(other, vector), "Invalid vector subtraction"
        assert len(self) == len(other), "Can't subtract vectors of different lengths"
        result = vector(tuple(a-b for a, b in zip(self, other)))
        assert len(result) == len(self), "Subtraction changed vector length"
        return result

    # Scalar multiplication
    def __mul__(self, scalar):
        assert isinstance(scalar, (int, float, complex)), "Invalid scalar type"
        result = vector(tuple(a * scalar for a in self))
        assert len(result) == len(self), "Scalar multplication is changed vector length"
        return result
    
    __rmul__ = __mul__  # allow scalar * vector

    # Negation (entry-wise)
    def __neg__(self):
        return (-1)*self

    # Equality (entry-wise)
    def equals(self, other):
        assert isinstance(other, vector), "Invalid vector comparison"
        return (len(self) == len(other)
                and all(a == b for a, b in zip(self, other)))
    
    # Dot product
    def dot(self, other):
        if not isinstance(other, vector):
            raise TypeError("Dot product requires another Vector")
        return sum(a * b for a, b in zip(self, other))

    # Norm (length)
    def norm(self):
        return np.sqrt(self.dot(self))

    # String representation
    def __repr__(self):
        return f"{tuple(self)}"
    
    @property
    def as_column(self):
        return np.array(self).reshape(-1,1)
    
    @property
    def as_row(self):
        return np.array(self).reshape(1,-1)

def in_column_span(v, M):
    # Return true if vector is in the integer column span of M
    # This is tested by augmenting M with v as an additional column,
    # then comparing the ranks of M and the augmented matrix (M|v)
    # If the rank changed, then v is not in the column span of M,
    # and if the rank did not change, v is in the column span of M
    M = np.asarray(M, dtype = int)
    rank_M = np.linalg.matrix_rank(M)
    augmented = np.hstack([M,v.as_column]) # augment M with v as an additional vector
    rank_augmented = np.linalg.matrix_rank(augmented)
    return rank_augmented == rank_M

def evaluate_character(alpha,torus_element):
    # Evaluate a character at a particular torus element
    # More concretely, given a diagonal matrix
    # t = diag(t_1, t_2, ..., t_n)
    # and a character (integer tuple) alpha = (a_1, a_2, ..., a_n)
    # return the product
    # t_1^(a_1) * (t_2)^(a_2) * ... * (t_n)^(a_n)
    
    # Validate the inputs
    assert isinstance(alpha, vector), "Characters must be represented as vectors"
    assert is_diagonal(torus_element), "Cannot evaluate character on non-diagonal matrix" 
    rows, cols = sp.shape(torus_element)
    assert rows == cols == len(alpha), "Character length mismatch" 
    
    # Compute the character evaluation
    return sp.prod(torus_element[i,i]**alpha[i] for i in range(len(alpha)))

def evaluate_cocharacter(cocharacter,scalar):
    # Evaluate a cocharacter at a scalar
    # More concretely, given a scalar x 
    # and a cocharacter (integer tuple) c = (c_1, c_2, ..., c_n)
    # return the matrix
    # diag(x^(c_1), x^(c_2), ..., x^(c_n))
    assert isinstance(cocharacter, vector), "Cocharacters must be represented as vectors"
    diag_elements = [scalar**int(exp) for exp in cocharacter]
    return sp.diag(*diag_elements)

def generic_kernel_element(alpha, generic_torus_element):
    # Return a generic element of the kernel of a character alpha,
    # using a generic torus element t as a starting point
    # More concretely, solve the equation alpha(t) = 1
    # for entries of t, then return a substituted version of t
    # which is a solution to that equation
    
    t = generic_torus_element
    
    assert isinstance(alpha, vector), "Characters and roots must be represented by vectors"
    assert all(isinstance(a, int) for a in alpha), "Character components must be integers"
    assert is_diagonal(t), "Torus element must be diagonal"
    
    alpha_of_t = evaluate_character(alpha, t)
    
    # If already in the kernel, just return the generic torus element
    if sp.simplify(alpha_of_t - 1) == 0: return t

    t_vars = t.free_symbols
    solutions_list = sp.solve(alpha_of_t - 1, t_vars, dict = True)
    assert len(solutions_list) >= 1, "No solution found for kernel equation"
    solutions_dict = solutions_list[0]
    return t.subs(solutions_dict)

def determine_irreducible_components(roots):
    # Determine the irreducible components of a root system
    # That is, find subsets of roots which are mutually orthogonal
    
    n = len(roots)
    visited = [False]*n
    components = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp_indices = []
        while stack:
            k = stack.pop()
            if visited[k]:
                continue
            visited[k] = True
            comp_indices.append(k)
            for j in range(n):
                # Check for orthogonality
                if not visited[j] and np.dot(roots[k], roots[j]) != 0:
                    stack.append(j)
        components.append([roots[k] for k in comp_indices])
    return components

def connected_components(graph):
    # Given a graph g represented as
    #     {v: {u: w, ...}, ...}
    # return a list of connected components, each in the same format.

    visited = set()
    components = []

    for start in graph:
        if start in visited:
            continue

        # Depth-first search to collect one component
        stack = [start]
        component_vertices = set()

        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            component_vertices.add(v)
            stack.extend(graph[v].keys())

        # Build the induced subgraph
        component = {
            v: {u: graph[v][u] for u in graph[v] if u in component_vertices}
            for v in component_vertices
        }
        
        components.append(component)
    return components

def generate_character_list(nonzero_entries, upper_bound):
    # Generate a potential list of characters (vectors)
    # This just returns a list of all vectors of certain length with nonzero entries
    # in allowed positions, with all possible integer values between
    # -upper_bound and upper_bound
    # EXAMPLE: if nonzero_entries = [1,1,0] and upper_bound = 2,
    # this returns this list of vectors
    # (-2,-2,0), (-2,-1,0), (-2,0,0), (-2,1,0), (-2,2,0)
    # (-1,-2,0), (-1,-1,0), (-1,0,0), (-1,1,0), (-1,2,0)
    # (0,-2,0),  (0,-1,0),  (0,0,0),  (0,1,0),  (0,2,0)
    # (1,-2,0),  (1,-1,0),  (1,0,0),  (1,1,0),  (1,2,0)
    # (2,-2,0),  (2,-1,0),  (2,0,0),  (2,1,0),  (2,2,0)
    
    values = []
    for allow in nonzero_entries:
        if allow:
            values.append(range(-upper_bound, upper_bound + 1))
        else:
            values.append((0,))
    return [vector(v) for v in itertools.product(*values)]

def reduce_character_list(vector_list, lattice_matrix):
    # Take a list of vectors, and return a sub-list
    # consisting of only vectors which are not pairwise equivalent
    # under quotienting by a lattice generated by the columns of lattice_matrix
    
    if len(lattice_matrix) == 0: return vector_list # If no lattice matrix, everything is distinct
    
    V = vector_list
    M = sp.Matrix(lattice_matrix).transpose()   # Matrix whose rows span the lattice
    null_basis = M.nullspace()
    
    def max_entry(v): return max(abs(x) for x in v)
    def support_size(v): return sum(1 for x in v if x != 0)
    def length_sq(v): return sum(x*x for x in v)
    def support_pattern(v): return tuple(1 if x != 0 else 0 for x in v)
    def abs_nullspace_dot_sum(v):
        if not null_basis: return 0
        v_vec = sp.Matrix(v)
        return sum(abs(v_vec.dot(q)) for q in null_basis)
    
    # Comparison key, higher is preferred
    def priority_key(v):
        return (
            -abs_nullspace_dot_sum(v),  # smaller sum preferred, zero if orthogonal to all lattice generators
            -max_entry(v),              # smaller maximum entry preferred
            -support_size(v),           # fewer nonzeros is better
            -length_sq(v),              # shorter vector preferred
            support_pattern(v)          # earlier nonzeros preferred
        )
    
    # If nullspace is trivial, everything is equivalent, just return the best vector by priority key
    if not null_basis: return [max(V, key=priority_key)]
    
    # Matrix whose rows are a basis for the null space of W_mat
    Q_mat = sp.Matrix.vstack(*[q.T for q in null_basis])
    
    best = {}
    for v in V:
        key = tuple(Q_mat * sp.Matrix(v))
        if key not in best or priority_key(v) > priority_key(best[key]): best[key] = v
    return list(best.values())

def determine_roots(generic_torus_element,
                    generic_lie_algebra_element,
                    list_of_characters,
                    vars_to_solve_for,
                    time_updates = False):
    
    # Caculate roots and root spaces.
    # Return in a dictionary format, where keys are roots (as tuples)
    # and the value is the generic element of the root space
    
    # The key equation is
    # t * X * t^(-1) = alpha(t) * X
    # The goal is to find alpha's that solve this equation,
    # where t and X are arbitrary/generic elements of the 
    # torus and Lie algebra respectively
    
    root_space_dict = {}
    t = generic_torus_element
    x = sp.Matrix(generic_lie_algebra_element)
    LHS = sp.simplify(t*x*t**(-1))
    
    if time_updates:
        print("\nComputing roots...")
        n = len(list_of_characters)
        print("Testing " + str(n) + " candidate characters.")
        i = 0
        t0 = time.time()
    
    for alpha in list_of_characters:
        
        if time_updates:
            i = i + 1
            t1 = time.time()
            if i % 100 == 0:
                print("\tTesting candidate", i)
                print("\tRoots found so far:", len(root_space_dict))
                elapsed = t1-t0
                avg = elapsed/i
                remaining = (n-i)*avg
                print("\tTime elapsed:", int(elapsed), "seconds")
                print("\tAverage time per candidate:", round(avg,2), "seconds")
                print("\tEstimated time remaining:", int(remaining), "seconds")
            
        alpha_of_t = evaluate_character(alpha,t)
        
        if alpha_of_t == 1: 
            # If the character evaluates to trivial on a generic torus element, 
            # then it is definitely not a root
            continue 
        
        RHS = alpha_of_t*x
        my_equation = sp.simplify(LHS-RHS)
        
        solutions_list = sp.solve(my_equation, vars_to_solve_for, dict=True)
        assert(len(solutions_list) == 1)
        solutions_dict = solutions_list[0]
        solutions_list = sp.solve(my_equation,vars_to_solve_for,dict=True)
        assert(len(solutions_list) == 1)
        solutions_dict = solutions_list[0]
        
        if solutions_dict:
            all_zero = all(var in solutions_dict and solutions_dict[var] == 0 for var in vars_to_solve_for)
            if not(all_zero): # For nonzero characters with a solution, add as a root
                generic_root_space_element = x.subs(solutions_dict)
                if not generic_root_space_element.is_zero_matrix:
                    root_space_dict[alpha] = sp.simplify(generic_root_space_element)
                    
    return root_space_dict

def visualize_graph(graph):
    # Build an ASCII string representation of a Dynkin diagram graph
    
    if not graph:
        return ""

    def get_edge_str(u, v):
        mult_u_to_v = graph[u].get(v, 1)
        mult_v_to_u = graph[v].get(u, 1)
        if mult_u_to_v > mult_v_to_u:
            if mult_u_to_v == 2: return "=>="
            if mult_u_to_v == 3: return "≡>≡"
            return f"-{mult_u_to_v}>"
        elif mult_u_to_v < mult_v_to_u:
            if mult_v_to_u == 2: return "=<="
            if mult_v_to_u == 3: return "≡<≡"
            return f"<-{mult_v_to_u}"
        return "---"

    degrees = {v: len(neighbors) for v, neighbors in graph.items()}
    fork_nodes = [v for v, deg in degrees.items() if deg >= 3]

    # --- BRANCHED CASE (Dynkin types D and E) ---
    if fork_nodes:
        fork = fork_nodes[0]
        
        # Helper to trace a linear path starting from a neighbor of the fork
        def get_path_from_neighbor(start_node):
            path = [start_node]
            visited = {fork, start_node}
            curr = start_node
            while True:
                next_nodes = [n for n in graph[curr] if n not in visited]
                if not next_nodes:
                    break
                curr = next_nodes[0]
                path.append(curr)
                visited.add(curr)
            return path

        # Get the 3 distinct paths branching out from the fork
        branches = [get_path_from_neighbor(n) for n in graph[fork].keys()]
        
        # Sort branches by length. The shortest path is always the vertical arm!
        branches.sort(key=len)
        vertical_branch = branches[0]  # e.g., [stub_node]
        left_branch = branches[1]      # e.g., [node, node...]
        right_branch = branches[2]     # e.g., [node, node...]
        
        # We like the backbone to read left-to-right cleanly. 
        # Reverse the left branch so it ends at the fork connection point.
        left_branch.reverse()
        
        # Construct the full horizontal backbone order
        # left_nodes ---> fork ---> right_nodes
        backbone_order = left_branch + [fork] + right_branch
        
        # Build the horizontal string representation
        ascii_parts = [str(backbone_order[0])]
        for i in range(1, len(backbone_order)):
            ascii_parts.append(get_edge_str(backbone_order[i-1], backbone_order[i]) + str(backbone_order[i]))
        backbone_line = "".join(ascii_parts)
        
        # Calculate exactly where the fork sits in the final string to line up the vertical arm
        fork_str = str(fork)
        padding_len = backbone_line.rfind(fork_str)
        padding = " " * padding_len
        centering_offset = " " * (len(fork_str) // 2)
        
        # Pick the vertical stub node (and any extra nodes if the branch was longer, though it's 1 for D/E)
        vertical_node = vertical_branch[0]
        
        return (
            f"\n{padding}{centering_offset}{vertical_node}"
            f"\n{padding}{centering_offset}|"
            f"\n{backbone_line}"
        )

    # --- LINEAR CASE (Dynkin types A, B, C, F, G) ---
    else:
        leaf_nodes = [v for v, deg in degrees.items() if deg == 1]
        start = min(leaf_nodes) if leaf_nodes else min(graph)

        visited = {start}
        order = [start]
        current = start
        while True:
            unvisited_neighbors = [u for u in graph[current] if u not in visited]
            if not unvisited_neighbors:
                break
            next_node = unvisited_neighbors[0]
            order.append(next_node)
            visited.add(next_node)
            current = next_node

        ascii_parts = [str(order[0])]
        for i in range(1, len(order)):
            ascii_parts.append(get_edge_str(order[i-1], order[i]) + str(order[i]))

        return "".join(ascii_parts)