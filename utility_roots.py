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

class vector(tuple):
    def __new__(cls, iterable):
        values = tuple(iterable)
        for x in values:
            if not isinstance(x, (int, float, complex)):
                raise TypeError(f"vector entries must be int, float, or complex, got {type(x)}")
        return super().__new__(cls, iterable)

    # Addition
    def __add__(self, other):
        assert isinstance(other, vector), "Invalid vector addition"
        assert len(self) == len(other), "Can't add vectors of different lengths"
        result = vector(tuple(a+b for a, b in zip(self, other)))
        assert len(result) == len(self), "Addition changed vector length"
        return result

    # Subtraction
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

    # Equality
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

    # Representation
    def __repr__(self):
        return f"{tuple(self)}"
    
    @property
    def as_column(self):
        return np.array(self).reshape(-1,1)
    
    @property
    def as_row(self):
        return np.array(self).reshape(1,-1)

def in_integer_column_span(v, M):
    # Return true if vector is in the integer column span of M
    M = np.asarray(M, dtype = int)
    rank_M = np.linalg.matrix_rank(M)
    augmented = np.hstack([M,v.as_column]) # add vector as additional column to M
    rank_augmented = np.linalg.matrix_rank(augmented)
    return rank_augmented == rank_M

def evaluate_character(alpha,torus_element):
    # Evaluate a character at a particular torus element
    
    # Validate the inputs
    assert isinstance(alpha, vector), "Characters must be represented as vectors"
    assert is_diagonal(torus_element), "Cannot evaluate character on non-diagonal matrix" 
    rows, cols = sp.shape(torus_element)
    assert rows == cols == len(alpha), "Character length mismatch" 
    
    # Compute the character evaluation
    return sp.prod(torus_element[i,i]**alpha[i] for i in range(len(alpha)))

def evaluate_cocharacter(cocharacter,scalar):
    # Evaluate a cocharacter at a scalar
    assert isinstance(cocharacter, vector), "Cocharacters must be represented as vectors"
    length = len(cocharacter)
    output = sp.eye(length, dtype=int)
    for i in range(length):
        output[i,i] = scalar**cocharacter[i]
    return output

def generic_kernel_element(alpha, t):
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
    
    # print("\n\nGenerating a generic kernel element")
    # print("\nalpha =",alpha)
    # print("\nt = ")
    # sp.pprint(t)
    # print("\nvariables in t: ",t_vars)
    # print("\nalpha(t) =",alpha_of_t)
    # print("\nSolutions to alpha(t) = 1 : ")
    # sp.pprint(solutions_list)
    # print("\nSubstituted torus element =")
    # sp.pprint(t.subs(solutions_dict))
    
    return t.subs(solutions_dict)

def determine_irreducible_components(roots):
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
                if not visited[j] and np.dot(roots[k], roots[j]) != 0:
                    stack.append(j)
        components.append([roots[k] for k in comp_indices])
    return components

def connected_components(graph):
    """
    Given a graph g represented as
        {v: {u: w, ...}, ...}
    return a list of connected components, each in the same format.
    """
    visited = set()
    components = []

    for start in graph:
        if start in visited:
            continue

        # DFS to collect one component
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
    # Possible values for each coordinate
    values = []
    for allow in nonzero_entries:
        if allow:
            values.append(range(-upper_bound, upper_bound + 1))
        else:
            values.append((0,))
    return [vector(v) for v in itertools.product(*values)]

def reduce_character_list(vector_list, lattice_matrix):
    # take a list of numpy vectors, and return a sub-list
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
    def nullspace_dot_sum(v):
        if not null_basis: return 0
        v_vec = sp.Matrix(v)
        return sum(abs(v_vec.dot(q)) for q in null_basis)
    
    # Comparison key, higher is preferred
    def priority_key(v):
        return (
            -nullspace_dot_sum(v), # smaller sum preferred, zero if orthogonal to all lattice generators
            -max_entry(v),         # smaller maximum entry preferred
            -support_size(v),      # fewer nonzeros is better
            -length_sq(v),         # shorter vector preferred
            support_pattern(v)     # earlier nonzeros preferred
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
                    variables_to_solve_for,
                    time_updates = False):
    
    # Caculate roots and root spaces
    # return in a dictionary format, where keys are roots (as tuples)
    # and the value is the generic element of the root space
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
        if alpha_of_t != 1: # ignore cases where the character is trivial
            RHS = alpha_of_t*x
            my_equation = sp.simplify(LHS-RHS)
            solutions_list = sp.solve(my_equation,variables_to_solve_for,dict=True)
            assert(len(solutions_list) == 1)
            solutions_dict = solutions_list[0]
            if len(solutions_dict) > 0 :
                all_zero = True 
                for var in variables_to_solve_for:  # check that not all variables are zero
                    if not(var in solutions_dict.keys()) or solutions_dict[var] != 0:
                        all_zero = False
                        break
                if not(all_zero): # For nonzero characters with a solution, add as a root
                    generic_root_space_element = x
                    for var, value in solutions_dict.items():
                        generic_root_space_element = generic_root_space_element.subs(var,value)
                    if not generic_root_space_element.is_zero_matrix:
                        root_space_dict[alpha] = sp.simplify(generic_root_space_element)
    return root_space_dict

def visualize_graph(graph):
    """
    ASCII Dynkin diagram for chain-like graphs with fixed 3-character edges.
    Arrows follow the adjacency dict convention (outgoing multiplicity).
    
    Args:
        graph (dict): adjacency dict {node: {neighbor: multiplicity, ...}}
    
    Returns:
        str: ASCII diagram as a string
    """
    # Step 1: start at a leaf node
    leaf_nodes = [v for v, neighbors in graph.items() if len(neighbors) == 1]
    start = min(leaf_nodes) if leaf_nodes else min(graph)

    # Step 2: traverse the chain
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

    # Step 3: build ASCII
    ascii_parts = [str(order[0])]
    for i in range(1, len(order)):
        prev = order[i-1]
        curr = order[i]

        mult_prev_to_curr = graph[prev].get(curr, 1)
        mult_curr_to_prev = graph[curr].get(prev, 1)

        # Decide edge symbol
        if mult_prev_to_curr > mult_curr_to_prev:
            # arrow from prev -> curr
            if mult_prev_to_curr == 2:
                edge = "=>="
            elif mult_prev_to_curr == 3:
                edge = "≡>≡"
            else:
                edge = f"-{mult_prev_to_curr}>"
        elif mult_prev_to_curr < mult_curr_to_prev:
            # arrow from curr -> prev
            if mult_curr_to_prev == 2:
                edge = "=<="
            elif mult_curr_to_prev == 3:
                edge = "≡<≡"
            else:
                edge = f"<-{mult_curr_to_prev}"
        else:
            # symmetric or single edge
            edge = "---"

        ascii_parts.append(edge + str(curr))

    return "".join(ascii_parts)