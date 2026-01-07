# pinned_group is a custom class for storing all the data of a pinned algebraic group.

# There are two broad categories of features:
#   1. Computational tools
#   2. Verification tools

# The computational tools take the bare bones needed to describe a matrix group, 
# then work out all the details of the root system, Lie algebra, root spaces,
# root subgroups, Weyl group elements, homomorphism defect coefficients, 
# commutator coefficients, and Weyl group conjugation coefficients.
# These details are then stored internally in the group object.

# The verification tools take the fitted data above and run a battery of tests
# to verify all the properties of a pinning.

import sympy as sp
import numpy as np
from utility_general import (determine_roots, 
                             generate_character_list, 
                             parse_root_pairs, 
                             reduce_character_list)
from root_system import root_system
from utility_roots import visualize_graph

class pinned_group:
    
    def __init__(self,
                 name_string,
                 matrix_size,
                 rank,
                 form,
                 is_group_element,
                 is_torus_element,
                 generic_torus_element,
                 trivial_characters,
                 is_lie_algebra_element,
                 generic_lie_algebra_element,
                 non_variables = None):
        
        self.name_string = name_string
        self.matrix_size = matrix_size
        self.rank = rank
        self.form = form
        
        self.is_group_element = is_group_element
        self.is_torus_element = is_torus_element
        self.generic_torus_element = generic_torus_element
        self.trivial_characters = trivial_characters
        self.is_lie_algebra_element = is_lie_algebra_element
        self.generic_lie_algebra_element = generic_lie_algebra_element
        
        self.non_variables = non_variables
                
        # These get set by running .fit_pinning(), (though not yet implemented)
        self.root_system = None
        self.root_space_dimension = None
        self.root_space_map = None
        self.root_subgroup_map = None
        self.homomorphism_defect_map = None
        self.commutator_coefficient_map = None
        self.weyl_group_element_map = None
        self.weyl_group_coefficient_map = None
        
    def fit_pinning(self, display):
        # work out all the computational details related to Lie algebra, roots, etc.
        
        if display:
            print(f"\nFitting a pinning for {self.name_string}")
            if self.form is not None:
                print("Form matrix:")
                sp.pprint(self.form.matrix)
                
            x = self.generic_lie_algebra_element(matrix_size = self.matrix_size,
                                                 rank = self.rank,
                                                 form = self.form,
                                                 letter = 'x')
            print("\nGeneric Lie algebra element:")            
            sp.pprint(x)
                
            t = self.generic_torus_element(matrix_size = self.matrix_size,
                                           rank = self.rank,
                                           form = self.form,
                                           letter = 't')
            print("\nGeneric torus element:")
            sp.pprint(t)
            
            print("\nTrivial characters:")
            for c in self.trivial_characters:
                sp.pprint(c)
            
        self.fit_root_system(display = display)
        # next things to fit: root space dimensions, root spaces, root subgroup maps

    def fit_root_system(self, display = True):
        t = self.generic_torus_element(matrix_size = self.matrix_size,
                                       rank = self.rank,
                                       form = self.form,
                                       letter = 't')
        x = self.generic_lie_algebra_element(matrix_size = self.matrix_size,
                                             rank = self.rank,
                                             form = self.form,
                                             letter = 'x')
        x_vars = x.free_symbols
        if self.non_variables is not None:
            x_vars = x_vars - self.non_variables
        full_char_list = generate_character_list(character_length = self.matrix_size,
                                            upper_bound = 2, 
                                            padded_zeros = self.matrix_size - 2*self.rank)
        reduced_char_list = reduce_character_list(vector_list = full_char_list,
                                                  quotient_vectors = self.trivial_characters)
        
        if display:
            print("\nCandidate characters:",len(full_char_list))                
            print("Candidates after quotienting by trivial characters:",len(reduced_char_list))
                
        self.roots_and_root_spaces = determine_roots(generic_torus_element = t,
                                                generic_lie_algebra_element = x,
                                                list_of_characters = reduced_char_list,
                                                variables_to_solve_for = x_vars)
        root_list = parse_root_pairs(root_info = self.roots_and_root_spaces,
                                     what_to_get = 'roots')
        
        def rsd(root):
            # I am not sure if this is the right way to do things
            # for unitary groups, because of the quadratic field extension
            # Perhaps the "complex variable" inputs need to be counted
            # as only a single dimension, I'm not sure
            for r, x in self.roots_and_root_spaces:
                if np.array_equal(r, root):
                    x_vars = x.free_symbols
                    if self.non_variables is not None:
                        x_vars = x_vars - self.non_variables
                    return len(x_vars)
        self.root_space_dimension = rsd
        
        def rsm(root, u):
            # This isn't what I want, I want it to take an input variable
            # but this is a first approximation
            assert(len(u) == self.root_space_dimension(root))
            for r, x in self.roots_and_root_spaces:
                if np.array_equal(r, root):
                    return x
        self.root_space_map = rsm
        
        if display:
            print("\nRoots and root spaces:")
            for r, x in self.roots_and_root_spaces:
                print("\tRoot:",r)
                print("\tRoot space:")
                sp.pprint(x)
                print("\tRoot space dimension:",self.root_space_dimension(r))
                print()
        
        self.root_system = root_system(root_list)
        
        if display:
            print("Root system:",self.root_system.name_string)
            print("Number of roots:",len(root_list))
            if self.root_system.is_irreducible:
                print("Dynkin diagram:",visualize_graph(self.root_system.dynkin_graph))
        
    def verify_pinning(self, display = True):
        # run tests to verify the pinning
        # run only after fitting
        
        self.verify_basics()
        self.verify_root_system()
        # tests to write after that: root space dimensions, root spaces, root subgroup maps
        
        print("\nOTHER TESTS NOT YET IMPLEMENTED")
        
    def verify_basics(self, display = True):
        # test that the generic torus element is in the group
        if display: print("\nChecking that a generic torus element is in the group... ", end="")
        t = self.generic_torus_element(matrix_size = self.matrix_size,
                                       rank = self.rank,
                                       form = self.form,
                                       letter = 't')
        assert(self.is_group_element(matrix_to_test = t, form = self.form))
        if display: print("done.")
        
        # tests for is_in_lie_algebra
        if display: print("Checking that a generic Lie algebra element is in the Lie algebra... ", end="")
        A = self.generic_lie_algebra_element(matrix_size = self.matrix_size, 
                                                     rank = self.rank, 
                                                     form = self.form,
                                                     letter = 'x')
        
        assert(self.is_lie_algebra_element(matrix_to_test = A, 
                                           form = self.form))        
        if display: print("done.")
        
    def verify_root_system(self, display = True):        
        self.root_system.verify_root_system_axioms()