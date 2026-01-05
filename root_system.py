# This is a custom class to model root systems of Lie algebras.
# There is a built-in class implemented by Sympy, but it lacks some functionality.

# Note that these root systems are not assumed to be reduced, i.e. they do not necessarily
# satisfy the assumption that the only multiple of a root in the root system are +1 and -1.
# Dropping this axiom leads to some "non-reduced root systems," but all such root systems are
# of type BC, which is essentially the union of the type B and type C root systems 
# (given a certain realization/embedding of these root systems).
# In particular, type BC includes some roots where twice the root is another root
# (so for those longer roots, half that root is a root).

import numpy as np
import sympy as sp
from utility_roots import connected_components

class root_system:
    
    def __init__(self, root_list, full_init =  True):
        self.root_list = root_list
        
        # check that all roots are vectors have same length/dimension
        vector_length = len(self.root_list[0])
        for alpha in self.root_list: 
            assert(len(alpha) == vector_length)
        self.vector_length = vector_length
        
        # Determine the irreducible components
        component_root_lists = root_system.determine_irreducible_components(self.root_list)
        self.is_irreducible = (len(component_root_lists) == 1)
        
        if not self.is_irreducible:
            self.components = [root_system(c, full_init = True) for c in component_root_lists]
        
        if full_init:
            if self.is_irreducible:
                self.determine_properties()
            else:
                self.name_string = ""
                for c in self.components:
                    c.determine_properties()
                    self.name_string = self.name_string + "_x_" + c.name_string
                self.name_string = self.name_string[3:]
        
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
        assert(self.is_irreducible)
        
        # make an ordered list of root lengths,
        # then determine if all roots are the same length, i.e. whether the root system is simply laced
        self.root_lengths = sorted({np.dot(r, r) for r in self.root_list})
        self.is_simply_laced = (len(set(self.root_lengths)) == 1)
        
        # determining reduced vs non-reduced
        self.is_reduced = True
        for alpha in self.root_list:
            for beta in self.root_list:
                if self.is_proportional(alpha,beta):
                    mask = (beta != 0)
                    ratio = (alpha[mask]/beta[mask])[0]
                    if ratio not in (1.0,-1.0): self.is_reduced = False
                    break        
            else:
                continue # only executed if the inner loop did NOT break
            break # only executed if the inner loop DID break
        
        # Make choices of positive and simple roots
        self.positive_roots = root_system.choose_positive_roots(self.root_list,
                                                                max_tries = 100,
                                                                seed = 0)
        self.simple_roots = root_system.choose_simple_roots(self.positive_roots)
        
        # check that the number of simple roots is the same as the
        # rank of the matrix spanned by the list of all roots
        assert(len(self.simple_roots) == sp.Matrix(self.root_list).rank())
        self.rank = len(self.simple_roots)
        
        # Build the Cartan matrix from the choice of simple roots
        self.cartan_matrix = root_system.build_cartan_matrix(self.simple_roots)
        
        # Build the (directed, weighted) Dynkin diagram/graph
        self.dynkin_graph = root_system.build_directed_dynkin_graph(self.simple_roots)
        
        # Check again for irreducibility
        # Could remove this, it's just a verification
        assert(len(connected_components(self.dynkin_graph)) == 1)
        
        # Determine the Dynkin type of each component using the Dynkin diagram
        self.dynkin_type, r = root_system.determine_dynkin_type(self.dynkin_graph)
        assert(self.rank == r)
        
        self.name_string = self.dynkin_type + str(self.rank)
    
    @staticmethod
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
    
    @staticmethod
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
            random_vec = rng.random(vector_length) # Generate a random vector
            # Compute the dot product of that vector with every root
            dot_products = [np.dot(random_vec, r) for r in root_list]
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
        
    @staticmethod
    def choose_simple_roots(positive_roots):
        simple_roots = []
        for alpha in positive_roots:
            is_simple = True
            for beta in positive_roots:
                if not np.any(beta) or np.allclose(beta, alpha): continue
                gamma = alpha - beta
                if any(np.array_equal(root,gamma) for root in positive_roots):
                    is_simple = False
                    break
            if is_simple:
                simple_roots.append(alpha)
        return simple_roots
        
    @staticmethod
    def build_cartan_matrix(simple_roots):
        rank = len(simple_roots)
        A = np.zeros((rank, rank), dtype=int)
        for i, alpha in enumerate(simple_roots):
            for j, beta in enumerate(simple_roots):
                A[i, j] = int(round(2 * np.dot(alpha, beta) / np.dot(beta, beta)))
        return A
    
    @staticmethod
    def build_directed_dynkin_graph(simple_roots):
        cartan_matrix = root_system.build_cartan_matrix(simple_roots)
        rank = cartan_matrix.shape[0] 
        graph = {i: {} for i in range(rank)} 
        for i in range(rank):
            for j in range(rank):
                if i == j:
                    continue
                a_ij = cartan_matrix[i, j]
                a_ji = cartan_matrix[j, i]
                if a_ij != 0:
                    len_i = np.dot(simple_roots[i], simple_roots[i])
                    len_j = np.dot(simple_roots[j], simple_roots[j])
                    mult = max(abs(a_ij), abs(a_ji))
                    if len_i > len_j:
                        # i -> j
                        graph[i][j] = mult
                        graph[j][i] = 1
                    elif len_i < len_j:
                        # j -> i
                        graph[i][j] = 1
                        graph[j][i] = mult
                    else:
                        # equal length, symmetric single edge
                        graph[i][j] = 1
                        graph[j][i] = 1
        return graph

    @staticmethod
    def determine_dynkin_type(dynkin_graph):
        # Determine the dynkin type of an irreducible root system from its
        # (weighted, directed) Dynkin graph
        
        my_rank = len(dynkin_graph)
        
        if my_rank == 1: 
            # only one node
            # this needs to be checked before computing other properties,
            # because it causes some degeneracy and things fail to calculate
            return ('A', 1)
    
        degree_dict = {v: len(dynkin_graph[v]) for v in dynkin_graph}
        #print("\t(Node : Degree) dictionary:",degree_dict)
        
        degrees = list(degree_dict.values())
        #print("\tDegrees:",degrees)
        
        edge_multiplicities = sorted(mult for v in dynkin_graph for mult in dynkin_graph[v].values())
        #print("\tEdge multiplicities:",edge_multiplicities)
            
        is_simply_laced = (max(edge_multiplicities, default = -1) == 1)
        #print("\tIs simply laced:", is_simply_laced)
        
        nodes_with_multiple_leaf_neighbors = [
            v for v in dynkin_graph
            if sum(1 for u in dynkin_graph[v] if len(dynkin_graph[u]) == 1) > 1
        ]
        #print("\tNodes with multiple leaf neighbors:",nodes_with_multiple_leaf_neighbors)
        
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
        #print("\tLeaf nodes:", leaf_nodes)
        #print("\tSingle edge leaf nodes:", single_edge_leaf_nodes)
        #print("\tNumber of single edge leaf nodes:", num_single_edge_leaf_nodes)
    
        double_edge_to_leaf = False
        for v in leaf_nodes:
            for u in dynkin_graph:
                if v in dynkin_graph[u] and dynkin_graph[u][v] > 1:
                    double_edge_to_leaf = True
        #print("\tIs there a double edge going to a leaf?",double_edge_to_leaf)
    
        if is_simply_laced:
            if 3 not in degrees:
                # no forks, so type A
                my_type = 'A'
            elif len(nodes_with_multiple_leaf_neighbors) >= 1:
                # D has a node with two leaf neighbors, E does not
                my_type = 'D'
                assert(my_rank >= 4)
            else:
                my_type = 'E'
                assert(my_rank in (6,7,8))
        else:
            if 3 in edge_multiplicities:
                my_type = 'G'
                assert(my_rank == 2)
            elif num_single_edge_leaf_nodes == 2:
                my_type = 'F'
                assert(my_rank == 4)
            else:
                ###########################################################
                # something to detect type BC needs to be here I think
                ###########################################################
                if double_edge_to_leaf:
                    my_type = 'B'
                    assert(my_rank >= 2)
                else:
                    my_type = 'C'
                    assert(my_rank >= 3)
    
        return my_type, my_rank
    
    def is_root(self,vector_to_test):
        # Return true if vector_to_test is a root
        return any(np.array_equal(root,vector_to_test) for root in self.root_list)
    
    def reflect_root(self,alpha,beta):
        # Given two roots alpha and beta, compute the reflection of beta across
        # the hyperplane perpendicular to alpha.
        # In usual notation, this is mathematically written as sigma_alpha(beta)
        
        # print("\nalpha=",alpha)
        # print("beta=",beta)
        # print("reflection=",beta - 2*np.dot(alpha,beta)/np.dot(alpha,alpha) * alpha)
        # print("\nAll roots:")
        # for r in self.root_list:
        #     sp.pprint(r)
        
        return beta - 2*np.dot(alpha,beta)/np.dot(alpha,alpha) * alpha
    
    def is_proportional(self,alpha,beta):
        # Return true if alpha and beta are proportional roots
        # In most root systems, this only happens if alpha = beta or alpha = -beta,
        # but in the type BC root system it is possible to have alpha = 2*beta
        alpha_mask = (alpha != 0)
        beta_mask = (beta != 0)
        ratios = (alpha[beta_mask] / beta[beta_mask]) # Not a typo, I promise. Need to use the same mask for both.
        return np.array_equal(alpha_mask,beta_mask) and np.all(ratios == ratios[0])
    
    def integer_linear_combos(self,alpha,beta):
        # Return a list of all positive integer linear combinations
        # of two roots alpha and beta within a list of roots
        assert(self.is_root(alpha))
        assert(self.is_root(beta))
        
        # The output is a dictionary, where keys are tuples (i,j)
        # and values are roots of the form i*alpha+j*beta
        combos = {}
        if not(self.is_root(alpha+beta)):
            # If alpha+beta is not a root, there are no integer linear combos
            # and we return an empty list
            return combos
        else:
            combos[(1,1)] = alpha+beta
        
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
            if self.is_root(alpha+old_root):
                new_combos[(i+1,j)] = old_root+alpha
            if self.is_root(beta+old_root):
                new_combos[(i,j+1)] = old_root+beta
        return new_combos
    
    def verify_root_system_axioms(self):
        print('\nRunning tests to verify root system axioms for the ' + self.name_string + ' root system.')
        
        print('\tChecking that zero is not a root...',end='')
        zero_vector = np.zeros((1,self.vector_length),dtype=int)
        assert(not(self.is_root(zero_vector)))
        print('passed.')
        
        print('\tChecking that the negative of a root is a root...',end='')
        for alpha in self.root_list:
            assert(self.is_root(-alpha))
        print('passed.')
        
        print('\tChecking that a reflection of a root is another root...',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                assert(self.is_root(self.reflect_root(alpha,beta)))
        print('passed.')
        
        print('\tChecking that the angle bracket of two roots is an integer...',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                angle_bracket = 2*np.dot(alpha,beta)/np.dot(alpha,alpha)
                assert(angle_bracket == int(angle_bracket))
        print('passed.')
        
        print('\tChecking that only multiples of root that are roots are +/-1 or +/-2 or +/-0.5...',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                if self.is_proportional(alpha,beta):
                    mask = (beta != 0)
                    ratio = (alpha[mask]/beta[mask])[0]
                    assert(ratio in (1.0,-1.0,2.0,-2.0,0.5,-0.5))
        print('passed.')
        print('Root system axiom checks completed.')