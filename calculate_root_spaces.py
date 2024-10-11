# Code to determine root systems, root spaces, and root subgroups

import numpy
from sympy import (symbols, MatrixSymbol, Matrix, trace, solve, sqrt, 
                   pprint, Identity, ZeroMatrix, BlockMatrix)
from itertools import product
from matrix_utility import is_diagonal, evaluate_character

def main():
    
    # n_range = (2,3,4)
    # SL_root_calc_test(n_range)
    
    ## SO_n_q is split if n=2q or n=2q+1,
    ##   quasi-split if n=2q+2,
    ##   and neither if n>2q+2
    q_range = (2,3)
    SO_split_root_calc_test(q_range)
    
    ## SO_n_q is split if n=2q or n=2q+1,
    ##   quasi-split if n=2q+2,
    ##   and neither if n>2q+2
    q_range = (1,2)
    n_width = 2
    SO_nonsplit_root_calc_test(q_range,n_width)
    
    # ## SU_n_q is quasi-split when n=2q, and not quasi-split when n>2q
    # q_range = (2,3)
    # SU_quasisplit_root_calc_test(q_range)

    # # ## SU_n_q is quasi-split when n=2q, and not quasi-split when n>2q
    # q_range = (2,3)
    # n_cap = 1
    # SU_nonquasisplit_root_calc_test(q_range,n_cap)
    
def SL_root_calc_test(n_range):
    # demonstration of SL_n calculations
    print("Computing roots and root spaces for special linear groups...\n")
    for n in n_range:
        SL_n_roots = determine_SL_roots(n)
        print("Roots for SL" + str(n) + ":",end =" ")
        pprint(parse_root_pairs(SL_n_roots,'roots'))
        
        print("Root spaces for SL" + str(n) + ":")
        pprint(parse_root_pairs(SL_n_roots,'root_spaces'))
        print("\n")
    print("Done with computing roots and root spaces for special linear groups.\n")

def SO_split_root_calc_test(q_range):
    # demonstration of SO_n_q calculations for split groups, i.e. when n=2q or n=2q+1
    print("Computing roots and root spaces for split special orthgonal groups...\n")
    for q in q_range:
        for n in (2*q,2*q+1):
            SO_n_q_roots = determine_SO_roots(n,q)
            
            print("Roots for SO_" + str(n) + "_" + str(q) + ":",end =" ")
            pprint(parse_root_pairs(SO_n_q_roots,'roots'))
            
            print("Root spaces for SO_" + str(n) + "_" + str(q) + ":")
            pprint(parse_root_pairs(SO_n_q_roots,'root_spaces'))
            print("\n")
    print("Done with computing roots and root spaces for split special orthogonal groups.\n")

def SO_nonsplit_root_calc_test(q_range,n_width):
    # demonstration of SO_n_q calculations for non-split groups
    print("Computing roots and root spaces for non-split special orthgonal groups...\n")
    for q in q_range:
        for n in range(2*q+2,2*q+2+n_width):
            SO_n_q_roots = determine_SO_roots(n,q)
            
            print("Roots for SO_" + str(n) + "_" + str(q) + ":",end =" ")
            pprint(parse_root_pairs(SO_n_q_roots,'roots'))
            
            print("Root spaces for SO_" + str(n) + "_" + str(q) + ":")
            pprint(parse_root_pairs(SO_n_q_roots,'root_spaces'))
            print("\n")
    print("Done with computing roots and root spaces for non-split special orthogonal groups.\n")

def SU_quasisplit_root_calc_test(q_range):
    # demonstration of SU_n_q calculations for quasi-split groups
    print("Computing roots and root spaces for quasisplit special unitary groups...\n")
    
    for q in q_range:
        n=2*q
        for eps in (1,-1):
            SU_n_q_roots = determine_SU_roots(n,q,eps)
            
            name_string = "SU_" + str(n) + "_" + str(q) + " with eps = " + str(eps)+":"
            
            print("Roots for " + name_string,end =" ")
            pprint(parse_root_pairs(SU_n_q_roots,'roots'))    
            
            print("\nRoot spaces for " + name_string)
            pprint(parse_root_pairs(SU_n_q_roots,'root_spaces'))
    print("Done with computing roots and root spaces for quasisplit special unitary groups.\n")

def SU_nonquasisplit_root_calc_test(q_range,n_cap):
    # demonstration of SU_n_q calculations for non-quasi-split groups
    print("Computing roots and root spaces for non-quasisplit special unitary groups...\n")
    
    for q in q_range:
        for n in range(2*q+1,2*q+1+n_cap):
            for eps in (1,-1):
                SU_n_q_roots = determine_SU_roots(n,q,eps)
                
                name_string = "SU_" + str(n) + "_" + str(q) + " with eps = " + str(eps)+":"
                
                print("Roots for " + name_string,end =" ")
                pprint(parse_root_pairs(SU_n_q_roots,'roots'))    
                
                print("\nRoot spaces for " + name_string)
                pprint(parse_root_pairs(SU_n_q_roots,'root_spaces'))
    print("Done with computing roots and root spaces for non-quasisplit special unitary groups.\n")

def determine_SU_roots(n,q,eps):
    # Calculate roots and root spaces for SU_{n,q}(k,H)
    #   n is a positive integer, at least 4 to be interesting
    #   k is a field
    #       pe is the primitive element of a quadratic field extension L/k
    #       The nontrivial Galois automorphism of L is an involution, denoted sig or conj
    #       It behaves very much like complex conjugation, i.e. conj(pe) = -pe
    #   H is the matrix of a nondegenerate hermitian or skew-hermitian form of Witt index q
    #       eps = 1 or -1, and determines skew-ness: H is hermitian if eps=1, skew-hermitian of eps=-1
    #   SU_{n,q}(k,H) is the set of (n x n) matrices X (entries in L) 
    #       satsifying X^h*B*X=B and det(X)=1 (where X^h is the conjugate transpose, i.e. X^h = conj(X^t))
    #   In general, using the theory of bilinear/hermitian forms we can always assume that H is an (n x n)
    #       block matrix of the form
    #                   [0      I       0]
    #                   [eps*I  0       0]
    #                   [0      0       C]
    #       where C is diagonal and satisfies conj(C)=eps*C. 
    #       More concretely, if eps=1, then C has entries from k, and if eps=-1 then C has "purely imaginary" entries from k, k-multiple of sqrt(d)
    #
    #   In general, q<=n/2 because Witt index can never exceed half the dimension.
    #       If q=n/2, the group is quasi-split and studied in my thesis.
    #       So we are a bit more interested in the case where q<n/2.
    #       We ignore the case q=0, because in this case the group is not isotropic.
    
    # Generating a list of character vectors to try as roots.
        # We only need characters of the form (a_1, a_2, ... a_q, 0, ... 0)
        # because of the shape of the torus elements.
    upper_bound = 2
        # Upper limit for checking for characters
        # For special unitary groups, the roots are of the two forms below:
        #       [0...,0,1,0,...0,-1,0,...0]
        #       [0...,0,+/-1,0,...0]
        #       [0...,0,+/-2,0,...0]
        # so we don't need an upper bound greater than 2
    list_of_characters = generate_character_list(n,upper_bound,padded_zeros=n-q)
    
    # Setting up the matrix H
    I_q = Identity(q)
    C_variables = symbols('c:'+str(n-2*q))
    C = Matrix(numpy.diag(C_variables))
    Z_qq = ZeroMatrix(q,q)
    Z_qd = ZeroMatrix(q,n-2*q) # d=n-2q
    Z_dq = ZeroMatrix(n-2*q,q) # d=n-2q
    H = Matrix(BlockMatrix([[Z_qq,I_q,Z_qd],
                            [eps*I_q,Z_qq,Z_qd],
                            [Z_dq,Z_dq,C]]))
    
    # Setting up the generic Lie algebra element
    x = Matrix(MatrixSymbol('x',n,n))
    y = Matrix(MatrixSymbol('y',n,n))
    variables_to_solve_for = []
    for i in range(n):
        for j in range(n):
            variables_to_solve_for.append(x[i,j])
            variables_to_solve_for.append(y[i,j])
    d = symbols('d')
    pe = sqrt(d) # primitive element
    z = x+pe*y
    z_conjugate = x-pe*y
    z_conjugate_transpose = z_conjugate.T
    lie_algebra_condition = z_conjugate_transpose*H+H*z
    
    # Setting up the generic torus element
    # The diagonal torus we will use in SU_{n,q}(k,H) is diagonal matrices of the block form
    #       [T     0        0]
    #       [0     T^(-1)   0]
    #       [0     0        I]
    #       where T=diag(s1, s2, ... s_q) is any invertible diagonal matrix with entries from k
    #       and I is the (n-2q) identity matrix
    torus_variables = symbols('s:'+str(q))
    diagonal_entries = (list(torus_variables) + 
                        list(numpy.reciprocal(torus_variables)) + 
                        list([1]*(n-2*q)))
    s = Matrix(numpy.diag(diagonal_entries))

    # Do the calculations
    return determine_roots(s,z,lie_algebra_condition,list_of_characters,variables_to_solve_for)

def determine_SO_roots(n,q):
    # Calculate roots and root spaces for SO_{n,q}(k,B)
    #   n is a positive integer, at least 4 to be interesting
    #   k is a field
    #   B is the matrix of a nondegenerate symmetric bilinear form of Witt index q
    #   SO_{n,q}(k,B) is the set of (n x n) matrices X with entries from k satisfying (X^t)*B*X=B and det(X)=1 (where X^t is the transpose)
    #       In general, using the theory of bilinear forms we can always assume that B is an (n x n) 
    #       block matrix of the form 
    #                   [0 I 0]
    #                   [I 0 0]
    #                   [0 0 C]
    #       where I is the (q x q) identity matrix and C is (n-2q x n-2q) and diagonal (and invertible). 
    #       For example, in the case n=4 and q=1, B has the form
    #                   [0,1,0,0]
    #                   [1,0,0,0]
    #                   [0,0,c1,0]
    #                   [0,0,0,c2]
    #       where c1, c2 are any nonzero elements of the field k.
    #
    #       In general, q<=n/2. If q=n/2 or q=(n-1)/2, then SO_n(k,B) is split and this is understood.
    #       If q=n/2-1, then SO_n(k,B) is quasi-split, this is an important case to understand.
    #       We ignore the case q=0, because in this case the group is not isotropic.
    
    # Generating a list of character vectors to try as roots.
        # We only need characters of the form (a_1, a_2, ... a_q, 0, ... 0)
        # because of the shape of the torus elements.
    upper_bound = 1
        # Upper limit for checking for characters
        # For special orthogonal groups, the roots are of the two forms below:
        #       [0...,0,1,0,...0,-1,0,...0]
        #       [0...,0,+/-1,0,...0]
        # so we don't need an upper bound greater than 1
    list_of_characters = generate_character_list(n,upper_bound,padded_zeros=n-q)
    
    # Setting up the matrix B
    I_q = Identity(q)
    C_variables = symbols('c:'+str(n-2*q))
    C = Matrix(numpy.diag(C_variables))
    Z_qq = ZeroMatrix(q,q)
    Z_qd = ZeroMatrix(q,n-2*q) # d=n-2q
    Z_dq = ZeroMatrix(n-2*q,q) # d=n-2q
    B = Matrix(BlockMatrix([[Z_qq,I_q,Z_qd],
                     [I_q,Z_qq,Z_qd],
                     [Z_dq,Z_dq,C]]))
    
    # Setting up the generic Lie algebra element
    x = Matrix(MatrixSymbol('x',n,n))
    x_transpose = x.T
    lie_algebra_condition = x_transpose*B+B*x
    
    # Setting up the generic torus element
    # The diagonal torus we will use in SO_{n,q}(k,B) is diagonal matrices of the block form
    #       [T     0        0]
    #       [0     T^(-1)   0]
    #       [0     0        I]
    #       where T=diag(s1, s2, ... s_q) is any invertible diagonal matrix
    #       and I is the (n-2q) identity matrix
    torus_variables = symbols('s:'+str(q))
    diagonal_entries = (list(torus_variables) + 
                        list(numpy.reciprocal(torus_variables)) + 
                        list([1]*(n-2*q)))
    s = Matrix(numpy.diag(diagonal_entries))
    
    # Do the calculations
    return determine_roots(s,x,lie_algebra_condition,list_of_characters,x)

def determine_SL_roots(n):
    # Calculate roots and root spaces for SL_n
    
    # Generating a list of character vectors to try as roots
    upper_bound = 1 # Upper limit for checking for characters
                    # For special linear groups, the roots are of the form [0...,0,1,0,...0,-1,0,...0]
                    # so we don't need an upper bound greater than 1
    list_of_characters = generate_character_list(n,upper_bound,padded_zeros=0)
    
    # Setting up the generic Lie algebra element
    x = Matrix(MatrixSymbol('x',n,n))
    lie_algebra_condition = trace(x) # the condition is really trace(x)=0 but
                                     # the "=0" part is assumed by the Sympy solver
    
    # Setting up the generic torus element
    torus_variables = list(symbols('s:'+str(n-1)))
    prod = 1
    for var in torus_variables:
        prod = prod*var
    torus_variables.append(1/prod)
    s = Matrix(numpy.diag(torus_variables))
    
    # Doing the calculations
    return determine_roots(s,x,lie_algebra_condition,list_of_characters,x)
    
def determine_roots(generic_torus_element,
                    generic_lie_algebra_element,
                    lie_algebra_condition,
                    list_of_characters,
                    variables_to_solve_for):

    # Caculate roots and root spaces
    # return in a list of pairs format, where the first entry is the root,
    # and the second entry is the generic element of the root space
    
    roots_and_root_spaces = []
    s = generic_torus_element
    x = generic_lie_algebra_element
    LHS = s*x*s**(-1)
    
    # variables_to_solve_for = []
    # for i in range(len(x)):
    #     for j in range(len(x)):
    #         variables_to_solve_for.append(x[i,j])
    
    for alpha in list_of_characters:
        alpha_of_s = evaluate_character(alpha,s)
        
        if alpha_of_s != 1: # ignore cases where the character is trivial
            RHS = alpha_of_s*Matrix(x)
            my_equations = [lie_algebra_condition,LHS-RHS]
            solutions_list = solve(my_equations,variables_to_solve_for,dict=True)
            # solutions_list = solve(my_equations,x,dict=True)
            
            # pprint(solutions_list)
        
            if len(solutions_list) == 1:
                solutions_dictionary = solutions_list[0]
                
        
                if len(solutions_dictionary)>0:
            
                    # check that not all variables are zero
                    all_zero = True
                    for variable in variables_to_solve_for:
                        if not(variable in solutions_dictionary.keys()) or solutions_dictionary[variable] != 0: 
                            all_zero = False
                            break
                    
                    # For nonzero characters with a solution, add as a root
                    if not(all_zero):
                        generic_root_space_element = Matrix(x)
                        for var,value in solutions_dictionary.items():
                            generic_root_space_element = generic_root_space_element.subs(var,value)
                        roots_and_root_spaces.append([alpha, generic_root_space_element])
                        
            else:
                # I don't think this is possible, but if it happens I want to know
                print("An unexpected error occured with the length of the solution set.")
                assert(False)
                    
    return roots_and_root_spaces


def generate_character_list(character_length, upper_bound, padded_zeros):
    # Return a list of all possible integer vectors of length character_length
    #   with entries ranging from -upper_bound to +upper_bound
    # The last few digits are all zeros, so the number of entries that 
    #   can vary are just character_length-padded_zeros 
    return [nontrivial_character+(0,)*padded_zeros for 
            nontrivial_character in product(range(-upper_bound,upper_bound+1),
                                                  repeat=character_length-padded_zeros)]

def generate_variable_names(name_string, upper_bound, dimensions):
    # generate a list of variable name strings
    # For example, generate_variable_names("A",3,2)
    # generates the list of strings ["A11","A12","A13","A21","A22","A23","A13","A23","A33"]
    variable_names = ''
    list_of_tuples = list(product(range(1,upper_bound+1),repeat=dimensions))
    for t in list_of_tuples:    
        variable_names = variable_names + name_string
        for i in t:
            variable_names = variable_names + str(i)
        variable_names = variable_names + ' '
    return variable_names[0:-1]

def parse_root_pairs(root_list, what_to_get):
    # Parse a list of pairs of the form
    #   [(root_1,root_space_1), (root_2, root_space_2), ...]
    # into just the roots,
    # or just the root spaces
        
    if what_to_get == 'roots':
        return [pair[0] for pair in root_list]
        
    elif what_to_get == 'root_spaces':
        return [pair[1] for pair in root_list]
    
    else:
        assert(false)


if __name__ == "__main__":
    main()