import numpy as np
import sympy as sp

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

    # ----- Non-reduced -----    
    "BC2": {0: {1: 2}, 1: {0: 1}},
    "BC3": {0: {1: 1}, 1: {0: 1, 2: 2}, 2: {1: 1}},
    "BC4": {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 2}, 3: {2: 1}}
}

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

