import numpy as np
import sympy as sp
from character_class import character_class
from utility_SL import trivial_characters_SL
from utility_SO import trivial_characters_SO
from utility_SU import trivial_characters_SU

# Test SL(n) character lattices
for n in (2,3,4):
    lattice_generators = trivial_characters_SL(n)
    lattice_matrix = sp.Matrix(np.stack(lattice_generators, axis=1))
    
    one_vec = np.array([1] * n)
    v1 = np.array([2] + [0]*(n-1))
    v2 = v1 - one_vec
    v3 = np.array([1,-1] + [0]*(n-2))
    v4 = v3 + one_vec
    v1_class = character_class(v1, lattice_matrix)
    v2_class = character_class(v2, lattice_matrix)
    v3_class = character_class(v3, lattice_matrix)
    v4_class = character_class(v4, lattice_matrix)
    test_classes = [v1_class, v2_class, v3_class, v4_class]
    
    print("-"*100)
    print("n=",n)
    print()
    
    # for c in test_classes:
    #     c.display()
    #     print()
    
    # for c in test_classes:
    #     print("-"*50)
    #     print("c:")
    #     c.display()
    #     print()
        
    #     print("2*c:")
    #     twice_c = c*2
    #     twice_c.display()
    #     print()
    
    for c1 in test_classes:
        for c2 in test_classes:

            print("-"*50)
            print("c1:")
            c1.display(show_matrices = False)
            print()
            
            print("c2:")
            c2.display(show_matrices = False)
            print()
            
            print("c1 + c2:")
            my_sum = c1 + c2
            my_sum.display(show_matrices = False)
            print()
            
            print("c1 - c2:")
            my_diff1 = c1 - c2
            my_diff1.display(show_matrices = False)
            print()
            
            print("c2 - c1:")
            my_diff2 = c2 - c1
            my_diff2.display(show_matrices = False)
            print()
            
            print("c1 == c2:",c1 == c2)
    

# # Test SO(n,q)/SU(n,q) character lattices
# for q in (2,3):
#     for n in (2*q, 2*q+1, 2*q+2, 2*q+3, 2*q+4):
#         lattice_generators = trivial_characters_SO(n,q)
#         v1 = None
#         v2 = None
#         v1_class = character_class(v1, lattice_generators)
#         v2_class = character_class(v2, lattice_generators)
#         # print("v1 reduced representative =",v1_class.reduced_representative)
#         # print("v2 reduced representative =",v2_class.reduced_representative)
#         # print("v1 == v2?",v1 == v2)
