# Various general utility functions related to matrices

import sympy as sp
import pickle
from nondegenerate_isotropic_form import nondegenerate_isotropic_form

main_var_1 = 'x'
main_var_2 = 'y'
anisotropic_var = 'c'
primitive_element = sp.sqrt(sp.symbols('d'))

formula_path = 'formulas/'
formula_files = {
    'single_var_det_formulas' : formula_path + 'single_var_det_formulas.pkl',
    'two_var_det_formulas' : formula_path + 'two_var_det_formulas.pkl',
    'SO_group_conditions' : formula_path + 'SO_group_conditions.pkl',
    'SO_lie_conditions' : formula_path + 'SO_lie_conditions.pkl',
    'SU_group_conditions' : formula_path + 'SU_group_conditions.pkl',
    'SU_lie_conditions' : formula_path + 'SU_lie_conditions.pkl'   
    }

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

def custom_conjugate(my_expression, p_e):
    # Conjugate an expression with entries in the quadratic field extension k(sqrt(d))
    # by replacing sqrt(d) with -sqrt(d)
    return my_expression.subs(p_e, -p_e)

def custom_real_part(my_expression, p_e):
    # Return the "real part" of an element of a quadratic field extension k(p_e)
    # by replacing p_e with zero
    return my_expression.subs(p_e, 0)

def custom_imag_part(my_expression, p_e):
    # Return the "real part" of an element of a quadratic field extension k(p_e)
    # by extracing the coefficient on p_e
    return my_expression.coeff(p_e)

def generate_and_store_formulas(n_max = 6,
                                q_max = 3, 
                                var_1 = main_var_1,
                                var_2 = main_var_2,
                                a_var = anisotropic_var,
                                p_e = primitive_element):
    
    print(f"Compiling single variable determinant formulas for 1<=n<={n_max}.")
    single_var_det_formulas = {}
    file_1 = formula_files['single_var_det_formulas']
    for n in range(n_max + 1):
        print(f"\tComputing determiannt formula for n={n}. ",end="")
        X = sp.Matrix(sp.symarray(var_1, (n, n)))
        single_var_det_formulas[n] = sp.expand(sp.det(X))
        print("Done.")
    with open(file_1, 'wb') as f:
        pickle.dump(single_var_det_formulas, f)
    print("Finished compiling single variable determinant formulas.")
    print("\tFormulas written to file:",file_1)
    
    print(f"\nCompiling special orthogonal group and Lie algebra formulas for 1<=q<={q_max} and 1<=n<={n_max}.")
    SO_group_conditions = {}
    SO_lie_conditions = {}
    file_2 = formula_files['SO_group_conditions']
    file_3 = formula_files['SO_lie_conditions']
    for q in range(1, q_max + 1):
        for n in range(2*q, n_max + 1):
            print(f"\tComputing formulas for q={q} and n={n}. ", end="")
            X = sp.Matrix(sp.symarray(var_1, (n, n)))
            anisotropic_vec = sp.Matrix(sp.symarray(a_var,n-2*q))
            B = nondegenerate_isotropic_form.build_symmetric_matrix(dimension = n,
                                                                    witt_index = q,
                                                                    anisotropic_vector = anisotropic_vec)
            SO_group_conditions[(n,q)] = sp.expand(X.T*B*X - B)
            SO_lie_conditions[(n,q)] = sp.expand(X.T*B + B*X)
            print("Done.")
    with open(file_2, 'wb') as f:
        pickle.dump(SO_group_conditions, f)
    with open(file_3, 'wb') as f:
        pickle.dump(SO_lie_conditions, f)
    print("Finished compiling special orthogonal group and Lie algebra formulas.")
    print("\tGroup formulas written to file: \t\t",file_2)
    print("\tLie algebra formulas written to file: \t",file_3)
    
    print(f"\nCompiling two variable determinant formulas for 1<=n<={n_max}")
    print(f"  and special unitary group and Lie algebra formulas for 1<=q<={q_max} and 1<=n<={n_max}.")
    two_var_det_formulas = {}    
    SU_group_conditions = {}
    SU_lie_conditions = {}
    file_4 = formula_files['two_var_det_formulas']
    file_5 = formula_files['SU_group_conditions']
    file_6 = formula_files['SU_lie_conditions']
    for q in range(1, q_max + 1):
        for n in range(2*q, 4+1): # upper bound would idealy be n_max+1, but calculations for n=5 and n=6 take too long
            print(f"\tComputing two variable determinant formula for n={n}. ", end="")
            X = sp.Matrix(sp.symarray(var_1, (n, n)))
            Y = sp.Matrix(sp.symarray(var_2, (n, n))) * p_e
            Z = X + Y
            two_var_det_formulas[n] = sp.expand(sp.det(Z))
            print("Done.")
            
            Z_conjugate = custom_conjugate(Z, p_e)
            anisotropic_vec = sp.Matrix(sp.symarray(a_var,n-2*q))
            for eps in (-1,1):
                print(f"\tComputing formulas for q={q} and n={n} and epsilson={eps}. ", end="")
                H = nondegenerate_isotropic_form.build_hermitian_matrix(dimension = n,
                                                                        witt_index = q,
                                                                        epsilon = eps,
                                                                        anisotropic_vector = anisotropic_vec,
                                                                        primitive_element = p_e)
                SU_group_conditions[(n,q)] = sp.expand(Z_conjugate.T*H*Z - H)
                SU_lie_conditions[(n,q)] = sp.expand(Z_conjugate.T*Z + Z*X)
                print("Done.")
    with open(file_4, 'wb') as f:
         pickle.dump(two_var_det_formulas, f)
    with open(file_5, 'wb') as f:
        pickle.dump(SU_group_conditions, f)
    with open(file_6, 'wb') as f:
        pickle.dump(SU_lie_conditions, f)
    print("Finished compiling two variable determinant formulas")
    print("  and special unitary group and Lie algebra formulas.")
    print("\tDeterminant formulas written to file: \t",file_4)
    print("\tGroup formulas written to file: \t\t",file_5)
    print("\tLie algebra formulas written to file: \t",file_6)
    
if __name__ == "__main__":
    generate_and_store_formulas()