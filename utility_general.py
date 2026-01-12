# Various general utility functions related to matrices

import sympy as sp
import numpy as np
import itertools
import time

def is_diagonal(my_matrix):
    # Return true if matrix is diagonal'    
    rows, cols = my_matrix.shape
    return all(my_matrix[i,j] == 0 for i in range(rows) for j in range(cols) if i != j)

def vector_variable(letter, length):
    return sp.Matrix(sp.symarray(letter, length))

def evaluate_character(character,torus_element):
    # Evaluate a character at a particular torus element
    # Character needs to be in the form of a vector like [1,0,0]
    my_shape = sp.shape(torus_element)
    assert(my_shape[0]==my_shape[1])        # torus_element should be a square matrix
    assert(my_shape[0]==len(character))     # torus_element size should match character length
    assert(is_diagonal(torus_element))      # torus element should be diagonal
    return_value = 1;
    for i in range(len(character)):
        return_value = return_value * (torus_element[i,i]**character[i])
    return return_value

def is_zero_expr(expr):
    # return zero if expression is zero, regardless of matrix or scalar quantity
    return expr.equals(0) if not hasattr(expr, 'shape') else expr.equals(sp.zeros(*expr.shape))

def matrix_sub(expr, matrix_to_replace, matrix_to_substitute):
    return expr.subs(dict(zip(matrix_to_replace, matrix_to_substitute)))

def generate_character_list(character_length, upper_bound, padded_zeros):
    # Return a list of all possible integer vectors of length character_length
    #   with entries ranging from -upper_bound to +upper_bound
    # The last few digits are all zeros, so the number of entries that 
    #   can vary are just character_length-padded_zeros 
    padded_zeros = max(padded_zeros, 0) # needs to be at least zero
    return [np.array(c + (0,) * padded_zeros, dtype=int) 
            for c in itertools.product(range(-upper_bound, upper_bound + 1),
                                       repeat=character_length - padded_zeros)]

def reduce_character_list(vector_list, quotient_vectors):
    # take a list of numpy vectors, and return a sub-list
    # consisting of only vectors which are not pairwise equivalent
    # under quotienting by list of quotient vectors
    V = vector_list
    W = quotient_vectors
        
    if not W: return V # If W is empty, everything is distinct
    
    W_mat = sp.Matrix(W)    # Matrix whose rows span W
    Q = W_mat.nullspace()   # Basis for the null space of W_mat
    
    def support_size(v): return sum(1 for x in v if x != 0)
    def support_pattern(v): return tuple(1 if x != 0 else 0 for x in v)
    def length_sq(v): return sum(x*x for x in v)
    
    # Comparison key, higher is preferred
    def priority_key(v):
        return (
            -support_size(v),     # fewer nonzeros is better
            -length_sq(v),        # shorter vector preferred
            support_pattern(v)    # earlier nonzeros preferred
        )
    
    # If nullspace is trivial, everything is equivalent, just return the best vector by priority key
    if not Q: return [max(V, key=priority_key)]
    
    # Matrix whose rows are a basis for the null space of W_mat
    Q_mat = sp.Matrix.vstack(*[q.T for q in Q]) 
    
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
    # return in a list of pairs format, where the first entry is the root,
    # and the second entry is the generic element of the root space
    roots_and_root_spaces = []
    t = generic_torus_element
    x = generic_lie_algebra_element
    LHS = t*x*t**(-1)
    
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
                print("\tRoots found so far:", len(roots_and_root_spaces))
                elapsed = t1-t0
                avg = elapsed/i
                remaining = (n-i)*avg
                print("\tTime elapsed:", int(elapsed), "seconds")
                print("\tAverage time per root:", round(avg,2), "seconds")
                print("\tEstimated time remaining:", int(remaining), "seconds")
            
        alpha_of_t = evaluate_character(alpha,t)
        if alpha_of_t != 1: # ignore cases where the character is trivial
            RHS = alpha_of_t*x
            my_equation = LHS-RHS
            solutions_list = sp.solve(my_equation,variables_to_solve_for,dict=True)
            if len(solutions_list) == 1:
                solutions_dict = solutions_list[0]
                if len(solutions_dict)>0: # check that not all variables are zero
                    all_zero = True
                    for var in variables_to_solve_for:
                        if not(var in solutions_dict.keys()) or solutions_dict[var] != 0:
                            all_zero = False
                            break
                    if not(all_zero): # For nonzero characters with a solution, add as a root
                        generic_root_space_element = sp.Matrix(x)
                        for var,value in solutions_dict.items():
                            generic_root_space_element = generic_root_space_element.subs(var,value)
                        if not generic_root_space_element.is_zero_matrix:
                            generic_root_space_element = sp.simplify(generic_root_space_element)
                            roots_and_root_spaces.append([alpha, generic_root_space_element])
            else: # I don't think this is possible, but if it happens I want to know
                raise Exception("An unexpected error occured with the length of the solution set.")
    return roots_and_root_spaces

def parse_root_pairs(root_info, what_to_get):
    # Parse a list of pairs of the form
    #   [(root_1,root_space_1), (root_2, root_space_2), ...]
    # into just the roots,
    # or just the root spaces
    if what_to_get == 'roots': return [pair[0] for pair in root_info]
    elif what_to_get == 'root_spaces': return [pair[1] for pair in root_info]
    else: raise ValueError("Can only parse root pairs into roots or root_spaces. Attempted input was",what_to_get)