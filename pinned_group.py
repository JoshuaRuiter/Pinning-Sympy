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

class pinned_group:
    
    def __init__(self,
                 name_string,
                 matrix_size,
                 rank,
                 form,
                 is_group_element,
                 is_torus_element,
                 generic_torus_element,
                 is_lie_algebra_element,
                 generic_lie_algebra_element,
                 lie_algebra_condition):
        
        self.name_string = name_string
        self.matrix_size = matrix_size
        self.rank = rank
        self.form = form
        
        self.is_group_element = is_group_element
        self.is_torus_element = is_torus_element
        self.generic_torus_element = generic_torus_element
        self.is_lie_algebra_element = is_lie_algebra_element
        self.generic_lie_algebra_element = generic_lie_algebra_element
        self.lie_algebra_condition = lie_algebra_condition
                
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
            vec_t = sp.Matrix(sp.symarray('t',self.rank))
            t = self.generic_torus_element(self.matrix_size,self.rank,self.form,vec_t)
            print("Generic torus element:")
            sp.pprint(t)
            
        self.fit_root_stuff(display = True)
        # next things to fit: root system, root space dimensions, root spaces, root subgroup maps

    def fit_root_stuff(self, display = True):
        # INCOMPLETE
        x=0
        
    def verify_pinning(self, display = True):
        # run tests to verify the pinning
        # run only after fitting
        
        self.verify_basics()
        self.verify_root_stuff()
        # tests to write after that: root system, root space dimensions, root spaces, root subgroup maps
        
        print("OTHER TESTS NOT YET IMPLEMENTED")
        
    def verify_basics(self, display = True):
        # test that the generic torus element is in the group
        if display: print("\nChecking that a generic torus element is in the group... ", end="")
        vec_t = sp.Matrix(sp.symarray('t',self.rank))
        t = self.generic_torus_element(self.matrix_size,self.rank,self.form,vec_t)
        assert(self.is_group_element(matrix_to_test = t, form = self.form))
        if display: print("done.")
        
        # tests for is_in_lie_algebra
        if display: print("Checking that a generic Lie algebra element is in the Lie algebra... ", end="")
        A = self.generic_lie_algebra_element(matrix_size = self.matrix_size, 
                                             rank = self.rank, 
                                             form = self.form,
                                             letters = ('x','y'))
        
        assert(self.is_lie_algebra_element(matrix_to_test = A, 
                                           form = self.form))        
        if display: print("done.")
        
    def verify_root_stuff(self, display = True):
        x=0 # INCOMPLETE