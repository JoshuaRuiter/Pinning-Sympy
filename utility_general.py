# Various general utility functions related to matrices

import sympy as sp
import numpy as np
import itertools

def is_diagonal(my_matrix):
    # Return true if matrix is diagonal'    
    rows, cols = my_matrix.shape
    return all(my_matrix[i,j] == 0 for i in range(rows) for j in range(cols) if i != j)

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
    subs_dict = dict(zip(matrix_to_replace, matrix_to_substitute))
    return expr.subs(subs_dict)

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
    Q = W_mat.nullspace()   # Basis for the then null space of W_mat
    
    def length_sq(v): return sum(x*x for x in v)
    #def positivity_score(v): return sum(1 for x in v if x > 0)
    def nonzero_pattern(v):
     # tuple of 1/0 indicating nonzero entries
     return tuple(1 if x != 0 else 0 for x in v)
        
    # If nullspace is trivial, everything is equivalent, just return the shortest vector
    if not Q: return [max(V,key=lambda v: (-length_sq(v), nonzero_pattern(v)))]
    
    # Matrix whose rows are a basis for the null space of W_mat
    Q_mat = sp.Matrix.vstack(*[q.T for q in Q]) 
    
    best = {}
    for v in V:
           key = tuple(Q_mat * sp.Matrix(v))
           if key not in best:
               best[key] = v
               continue
           v_best = best[key]
           if (length_sq(v) < length_sq(v_best) or
                       (
                           length_sq(v) == length_sq(v_best) and
                           nonzero_pattern(v) > nonzero_pattern(v_best)
                       )
                   ):
               best[key] = v
    return list(best.values())

def determine_roots(generic_torus_element,
                    generic_lie_algebra_element,
                    list_of_characters,
                    variables_to_solve_for):
    # Caculate roots and root spaces
    # return in a list of pairs format, where the first entry is the root,
    # and the second entry is the generic element of the root space
    roots_and_root_spaces = []
    t = generic_torus_element
    x = generic_lie_algebra_element
    LHS = t*x*t**(-1)
    
    test_alpha = (1,0,0,0)
    
    for alpha in list_of_characters:
        alpha_of_t = evaluate_character(alpha,t)
        if alpha_of_t != 1: # ignore cases where the character is trivial
            RHS = alpha_of_t*x
            my_equation = LHS-RHS
            solutions_list = sp.solve(my_equation,variables_to_solve_for,dict=True)
            
            # #if np.array_equal(alpha,test_alpha):
            # if True:
            #     print("\nt=")
            #     sp.pprint(t)
            #     print("\nx=")
            #     sp.pprint(x)
            #     print("\nt*x*t^(-1)=")
            #     sp.pprint(LHS)
            #     print("\nalpha=")
            #     sp.pprint(alpha)
            #     print("\nalpha of t=")
            #     sp.pprint(alpha_of_t)
            #     print("\nalpha_of_t * x=")
            #     sp.pprint(RHS)
            #     print("\nLHS-RHS=")
            #     sp.pprint(LHS-RHS)                
            #     print("\nSolutions to LHS-RHS=0")
            #     sp.pprint(solutions_list)
            #     print("\nType of solution object:", type(solutions_list))
            #     print("\nLength of solutions list:")
            #     print(len(solutions_list))
            #     print("\nType of first solutions entry:",type(solutions_list[0]))
            #     print("\nSolutions entry-by-entry:")
            #     for s in solutions_list:
            #         sp.pprint(s)
            
            if len(solutions_list) == 1:
                solutions_dict = solutions_list[0]
                
                # #if np.array_equal(alpha,test_alpha):
                # if True:
                #     print("\nSolutions dictionary:")
                #     sp.pprint(solutions_dict)
                #     print("\nLength of solutions dictionary:")
                #     print(len(solutions_dict))
                
                if len(solutions_dict)>0: # check that not all variables are zero
                    all_zero = True
                    for var in variables_to_solve_for:
                        if not(var in solutions_dict.keys()) or solutions_dict[var] != 0:
                            all_zero = False
                            break
                        
                    
                    # #if np.array_equal(alpha,test_alpha):
                    # if True:
                    #     print("\nIs everything zero?")
                    #     print(all_zero)
                    
                    if not(all_zero): # For nonzero characters with a solution, add as a root
                        generic_root_space_element = sp.Matrix(x)
                        for var,value in solutions_dict.items():
                            generic_root_space_element = generic_root_space_element.subs(var,value)
                        if not generic_root_space_element.is_zero_matrix:
                            roots_and_root_spaces.append([alpha, generic_root_space_element])
            else: # I don't think this is possible, but if it happens I want to know
                raise Exception("An unexpected error occured with the length of the solution set.")
    return roots_and_root_spaces

def parse_root_pairs(root_info, what_to_get):
    # Parse a list of pairs of the form
    #   [(root_1,root_space_1), (root_2, root_space_2), ...]
    # into just the roots,
    # or just the root spaces
    if what_to_get == 'roots':
        return [pair[0] for pair in root_info]
    elif what_to_get == 'root_spaces':
        return [pair[1] for pair in root_info]
    else:
        raise ValueError("Can only parse root pairs into roots or root_spaces. Attempted input was",what_to_get)