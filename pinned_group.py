class pinned_group:
    
    def __init__(self, 
                 name_string, 
                 matrix_size,
                 root_system,
                 is_lie_algebra_element,
                 is_group_element,
                 is_torus_element,
                 root_space_dimension):
        # Build a pinned group from scratch by providing all inputs
        
        self.name_string = name_string
        self.matrix_size = matrix_size
        
        self.root_system = root_system
        
        self.is_lie_algebra_element = is_lie_algebra_element
        self.is_group_element = is_group_element
        self.is_torus_element = is_torus_element
    
        self.root_space_dimension = root_space_dimension
       
        # Form
        # RootSpaceMap
        # RootSubgroupMap
        # WeylGroupMap
        # GenericTorusElementMap
        # HomDefectCoefficientMap
        # CommutatorCoefficientMap
        # WeylGroupCoefficientMap
                
    
    def run_tests(self):
        print("\nRunning tests to verify a pinning of the " + self.name_string + "...")
        self.test_basics()
        self.test_root_space_maps_are_almost_homomorphisms()
        
        print("All tests complete.")
        
    def test_basics(self):
        print("\tRunning basic tests...")
    
        print("\t\tChecking root spaces belong to the Lie algebra...",end='')
        print("TESTS NOT WRITTEN",end=' ')
        print("done.")
        
        print("\t\tChecking root subgroups belong to the group...",end='')
        print("TESTS NOT WRITTEN",end=' ')
        print("done.")
        
        print("\tBasic tests passed.")
        
    def test_root_space_maps_are_almost_homomorphisms(self):
        print("\tChecking root space spaces are (almost) homomorphisms...",end='')
        print("TESTS NOT WRITTEN",end=' ')
        print("done.")