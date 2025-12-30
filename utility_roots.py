import numpy as np
import sympy as sp

def is_zero(v, tol=1e-9):
    return np.linalg.norm(v) < tol

def vector_in_list(v, L, tol=1e-9):
    """Check whether vector v is in list L up to tolerance."""
    return any(is_zero(v - w, tol) for w in L)

def root_lengths(roots):
    """Return sorted list of distinct squared root lengths."""
    lengths = {np.dot(r, r) for r in roots}
    return sorted(lengths)

def positive_roots(roots, max_tries=100):
    """
    Choose a generic linear functional and split roots
    into positive and negative roots.
    """
    roots = [np.array(r, dtype=float) for r in roots]
    dim = len(roots[0])
    for _ in range(max_tries):
        v = np.random.randn(dim)
        vals = [np.dot(v, r) for r in roots]
        if all(abs(val) > 1e-9 for val in vals):
            return [r for r, val in zip(roots, vals) if val > 0]
    raise RuntimeError("Failed to find a generic linear functional")
    
def simple_roots(roots):
    """
    Given a root system (possibly non-reduced),
    return a list of simple roots.
    """
    pos = positive_roots(roots)
    simples = []
    for alpha in pos:
        is_simple = True
        for beta in pos:
            if is_zero(beta):
                continue
            if np.allclose(beta, alpha):
                continue
            gamma = alpha - beta
            if vector_in_list(gamma, pos):
                is_simple = False
                break
        if is_simple:
            simples.append(alpha)
    return simples

def cartan_matrix(simple_roots):
    """
    Compute the Cartan matrix from a list of simple roots.
    """
    rank = len(simple_roots)
    A = np.zeros((rank, rank), dtype=int)
    for i, alpha in enumerate(simple_roots):
        for j, beta in enumerate(simple_roots):
            A[i, j] = int(round(2 * np.dot(alpha, beta) / np.dot(beta, beta)))
    return A

def dynkin_graph(cartan_matrix):
    """
    Build Dynkin graph from Cartan matrix.
    Returns adjacency list with edge multiplicities.
    """
    rank = cartan_matrix.shape[0]
    graph = {i: {} for i in range(rank)}
    for i in range(rank):
        for j in range(i+1, rank):
            if cartan_matrix[i, j] != 0:
                mult = max(abs(cartan_matrix[i, j]), abs(cartan_matrix[j, i]))
                graph[i][j] = mult
                graph[j][i] = mult
    return graph

def degrees(graph):
    return {v: len(graph[v]) for v in graph}

def edge_multiplicities(graph):
    return sorted(mult for v in graph for mult in graph[v].values())

def is_simply_laced(graph):
    return all(mult == 1 for v in graph for mult in graph[v].values())

def is_disconnected(graph):
    """
    Check if an undirected graph is disconnected.
    
    graph: dict of the form {node: {neighbor: multiplicity, ...}, ...}
    Returns True if the graph has more than one connected component.
    """
    if not graph:
        return True
    visited = set()
    nodes = list(graph.keys())
    def dfs(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
    # Start DFS from the first node
    dfs(nodes[0])
    # If some node wasn't visited, graph is disconnected
    return len(visited) < len(nodes)

def determine_dynkin_type(roots):
    simples = simple_roots(roots)
    C = cartan_matrix(simples)
    rank = C.shape[0]
    graph = dynkin_graph(C)
    deg_seq = sorted(len(graph[v]) for v in graph)
    mults = edge_multiplicities(graph)
    
    print("\nRoots:")
    sp.pprint(roots)
    print("\nSimple roots:")
    sp.pprint(simples)
    print("\nCartan matrix:")
    sp.pprint(C)
    print("\nRank:", rank)
    print("\nGraph:", graph)
    print("\nDegree sequence:",deg_seq)
    print("\nEdge multiplicities:",mults)
    
    print("TO DO: break up disconnected diagrams into components, then classify the connected components")
    
    if max(mults, default=1) == 3: 
        # Recognize type G by the presence of a triple edge
        dynkin_type = 'G'
        assert(rank == 2)
        
    elif not is_simply_laced(graph):
        if deg_seq == [1,2,2,1]:
            # Recognize type F by the degree sequence
            dynkin_type = 'F'
        else:
            # type B, C, or BC
            root_lengths = {np.dot(r, r) for r in roots}
            if len(root_lengths) >= 3:
                # recognize type BC by the presence of three different root lengths
                dynkin_type = 'BC'
            elif rank == 2:
                # B2 and C2 are isomorphic, we call it B2 by pure convention
                dynkin_type = 'B'
            else:
                # type B or C
                # B has 1 short root in the list of simple roots
                # C has 1 long root in the list of simple roots    
                simple_root_lengths = {np.dot(r, r) for r in simples}
                min_len = min(simple_root_lengths)
                num_short = sum(l == min_len for l in simple_root_lengths)
                if num_short == 1:
                    dynkin_type = 'B'
                elif num_short == len(simples) - 1:
                    dynkin_type = 'C'
                else:
                    raise ValueError("Not a reduced B/C root system")
    else: 
        # simpy laced, types ADE
        assert(is_simply_laced(graph))
        if rank == 1: dynkin_type = 'A'
        elif rank == 2:
            if max(deg_seq) == 0: dynkin_type = 'D'
            else: dynkin_type = 'A'
        elif rank == 3:
            if max(deg_seq) == 2: dynkin_type = 'A'
            else: dynkin_type = 'D'    
        elif deg_seq.count(1) == 2 and deg_seq.count(2) == rank - 2: dynkin_type = 'A'
        elif deg_seq.count(3) == 1 and deg_seq.count(1) == 3: dynkin_type = 'D'
        else:
            dynkin_type = 'E'
            assert rank in (6, 7, 8)

    return dynkin_type, rank
