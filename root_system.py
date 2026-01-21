# This is a custom class to model root systems of Lie algebras.
# There is a built-in class implemented by Sympy, but it lacks functionality that I want.

# Note that these root systems are not assumed to be reduced, i.e. they do not necessarily
# satisfy the assumption that the only multiple of a root in the root system are +1 and -1.
# Dropping this axiom leads to some "non-reduced root systems," but all such root systems are
# of type BC, which is essentially the union of the type B and type C root systems 
# (given a certain realization/embedding of these root systems).
# In particular, type BC includes some roots where twice the root is another root
# (so for those longer roots, half that root is a root).

import sympy as sp
import numpy as np
import itertools
from utility_general import vector_variable
from utility_roots import (vector, 
                           in_integer_column_span, 
                           connected_components, 
                           directed_dynkin_graphs,
                           visualize_graph)

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
        
        if lattice_matrix is None:
            # make the lattice_matrix a single column of zeros if there isn't one
            self.lattice_matrix = sp.zeros((self.vector_length, 1))
        else:
            self.lattice_matrix = lattice_matrix
        
        # check that all roots are vectors have same length/dimension, 
        # which must also match the number of rows in the lattice matrix
        first_vec_length = len(self.root_list[0])
        for alpha in self.root_list: 
            assert len(alpha) == first_vec_length, \
                "Root system can't have vectors of different lengths"
        self.vector_length = first_vec_length
        assert self.vector_length ==  self.lattice_matrix.shape[0], \
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
            self.name_strings = [c.name_string for c in self.components]
            self.name_string = "_x_".join(self.name_strings)

        # Set up a dictionary of coroots
        self.coroot_dict = {r : self.compute_coroot(r) for r in self.root_list}

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
            if in_integer_column_span(vector_to_test - r, self.lattice_matrix):
                return (True, r) if with_equivalent else True

        return (False, None) if with_equivalent else False

    def compute_coroot(self, alpha):
        """
        Given a root alpha (as vector of integers)
        return a cocharacter vector alpha_check which must satisfy:
            <alpha, alpha_check> = 2  i.e. alpha dot
        """
        
        assert isinstance(alpha, vector), "Character must be a vector"
        assert self.is_root(alpha), "Can only find cocharacter of a root"
        assert all(isinstance(a,int) for a in alpha), "Character must have integer entries"
        assert self.lattice_matrix is not None, "Root system must have a lattice matrix"
        
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
        
        #############################################################
        # print("\n\nComputing a coroot")
        # print("alpha =",alpha)
        # print("alpha_check =",x)
        # print("Vanishing conditions:")
        # sp.pprint(eq1)
        # sp.pprint(eq2)
        #############################################################

        # Solve the equations
        solutions_list = sp.solve([eq1, eq2], x_vars, dict=True)
        assert len(solutions_list) >= 1, "No solution found for cocharacter equation"
        assert len(solutions_list) == 1, \
            "Not sure what to do with multiple solutions to cocharacter equations"
        solutions_dict = solutions_list[0]
        
        x_general_solution = x.subs(solutions_dict)
        
        #############################################################
        # print("General solution for coroot:")
        # sp.pprint(x_general_solution)
        # print()
        #############################################################
        
        # If there are no free variables remaining, we're done
        if len(x_general_solution.free_symbols) == 0:
            # Confirm that all entries are integers, 
            # then convert to vector and return
            assert all(a == int(a) for a in x_general_solution)
            return vector([int(a) for a in x_general_solution])
            return x_general_solution
        
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
        
        # Confirm that all entries are integers, then convert to vector and return
        assert all(a == int(a) for a in x_min)
        return vector([int(a) for a in x_min])


    def is_same_root(self, alpha, beta):
        assert type(alpha) == vector, "same_root expects vector input 1st argument"
        assert type(beta) == vector, "same_root expects vector input for 2nd argument"
        return alpha.equals(beta) or in_integer_column_span(alpha - beta, self.lattice_matrix)

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
    
        zero_vec = vector([0] * len(alpha))
        
        # Prepare candidate vectors for all small denominators
        candidates = []
        candidate_ratios = []
        for q in range(1, max_denominator+1):
            for p in range(-max_denominator, max_denominator+1):
                
                # If a candidate is zero, we don't need to
                # go through computing the rank of an augmented
                # matrix to check, so this saves a lot of time
                # in that case
                if zero_vec.equals(p*alpha - q*beta):
                    return (True, p/q) if with_ratio else True
                
                candidates.append(p*alpha - q*beta)
                candidate_ratios.append(p / q)
        for candidate, ratio in zip(candidates, candidate_ratios):
            if in_integer_column_span(candidate, self.lattice_matrix):
                return (True, ratio) if with_ratio else True
        return (False, None) if with_ratio else False

    def reflect_root(self, alpha, beta):
        return alpha - 2 * alpha.dot(beta) / beta.dot(beta) * beta

    def is_multipliable_root(self, vector_to_test):
        return self.is_root(vector_to_test) and self.is_root(2*vector_to_test)

    def integer_linear_combos(self,alpha,beta):
        # Return a list of all positive integer linear combinations
        # of two roots alpha and beta within a list of roots
        assert self.is_root(alpha) and self.is_root(beta), "Can't compute integer linear combos with non-roots"
        
        # The output is a dictionary, where keys are tuples (i,j)
        # and values are roots of the form i*alpha+j*beta
        combos = {}
        my_sum = alpha+beta
        if not(self.is_root(my_sum)):
            # If alpha+beta is not a root, there are no integer linear combos
            # and we return an empty dictionary
            return combos
        else:
            combos[(1,1)] = my_sum
        
        # Run a loop where each iteration, we try adding alpha and beta
        # to each existing combo
        while True:
            new_combos = self.increment_combos(alpha,beta,combos)
            if len(combos) == len(new_combos):
                break;
            combos = new_combos
        return combos
    
    def increment_combos(self,alpha,beta,old_combos):
        new_combos = old_combos.copy() # A shallow copy
        for key in old_combos:
            i = key[0]
            j = key[1]
            old_root = old_combos[key]
            if self.is_root(alpha + old_root):
                new_combos[(i+1,j)] = alpha + old_root
            if self.is_root(beta + old_root):
                new_combos[(i,j+1)] = beta + old_root
        return new_combos

    def verify_root_system_axioms(self, display = True):
        if display: print('\nRunning tests to verify root system axioms for the ' 
                          + self.name_string + ' root system.')
        
        if display: print('\tChecking that zero is not a root...',end='')
        zero_vector = vector([0] * self.vector_length)
        assert not(self.is_root(zero_vector)), "Zero vector should not be a root"
        if display: print('done.')
        
        if display: print('\tChecking that the negative of a root is a root...',end='')
        for alpha in self.root_list:
            assert self.is_root(-1*alpha, with_equivalent = False), \
                "Negative of root is not a root"
        if display: print('done.')
        
        if display: print('\tChecking that a reflection of a root is another root...',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                assert self.is_root(self.reflect_root(alpha,beta)), \
                "Reflection of a root is not a root but should be"
        if display: print('done.')
        
        if display: print('\tChecking that the angle bracket of two roots is an integer...',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                angle_bracket = 2* alpha.dot(beta) / beta.dot(beta)
                assert angle_bracket == int(angle_bracket), \
                "Angle bracket of roots should be an integer"
        if display: print('done.')
        
        if display: print('\tChecking that the dot product of a root with its coroot is 2...',end='')
        for alpha in self.root_list:
            alpha_check = self.coroot_dict[alpha]
            angle_bracket = alpha.dot(alpha_check)
            assert angle_bracket == 2, "Dot product of a root with its coroot is not 2"
        if display: print('done.')
        
        if display: print('\tChecking that ratios between proportional roots are +/-1, +/-2, or +/-0.5...',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                proportional, ratio = self.is_proportional(alpha,beta,with_ratio=True)
                if proportional: assert ratio in (1.0,-1.0,2.0,-2.0,0.5,-0.5), "Invalid ratio between roots"
        if display: print('done.')
        
        if display: print('Root system axiom checks completed.')

def determine_irreducible_components(roots):
    # Build separate lists for each of the irreducible components, 
    # if there are multiple
    n = len(roots)
    visited = [False]*n
    components = []
    for i in range(n):
        if visited[i]: continue
        stack = [i]
        comp_indices = []
        while stack:
            k = stack.pop()
            if visited[k]: continue
            visited[k] = True
            comp_indices.append(k)
            for j in range(n):
                if not visited[j] and roots[k].dot(roots[j]) != 0: stack.append(j)
        components.append([roots[k] for k in comp_indices])
    return components

def choose_positive_roots(root_list, max_tries = 100, seed = 0):
    # Procedure: choose a hyperplane not containing any of the roots
    # then choose one of the sides of that hyperplane.
    # All roots on that side of the hyperplane are positive.
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
    simple_roots = []
    for alpha in positive_roots:
        is_simple = True
        for beta in positive_roots:
            if alpha.equals(beta): continue
            if alpha - beta in positive_roots:
                is_simple = False
                break
        if is_simple:
            simple_roots.append(alpha)
    return simple_roots

def build_cartan_matrix(simple_roots):
    rank = len(simple_roots)
    A = np.zeros((rank, rank), dtype=int)
    for i, alpha in enumerate(simple_roots):
        for j, beta in enumerate(simple_roots):
            A[i, j] = int(round(2 * alpha.dot(beta) / beta.dot(beta)))
    return A

def build_directed_dynkin_graph(simple_roots):
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

def test_dynkin_classifier():
    # Run a battery of tests to verify that the dynkin type classifier works
    print("Testing Dynkin type classifier on a pre-populated "+
          "list of irreducible reduced root systems...")
    for name in directed_dynkin_graphs:
        print("\nName:",name)
        true_type = name[0]
        true_rank = int(name[-1])
        print("\tTrue type:",true_type)
        print("\tTrue rank:",true_rank)
        graph = directed_dynkin_graphs[name]
        print("\tGraph visualization:", visualize_graph(graph))
        print("\tGraph as dictionary:",graph)
        calculated_type, calculated_rank = root_system.determine_dynkin_type(graph)
        print("\tDetermined type:",calculated_type)
        print("\tDetermined rank:",calculated_rank)
        assert true_type == calculated_type
        assert true_rank == calculated_rank
    print("\nAll tests passed.")