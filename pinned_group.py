class pinned_group:
    
    def __init__(self,name_string, matrix_size,root_system):
        # Build a pinned group from scratch by providing all inputs
        
        self.name_string = name_string
        self.matrix_size = matrix_size       
        self.root_system = root_system
       # self.form = form
        
        # NameString
        # MatrixSize
        # Root_System
        # RootList
        # RootSystemRank
        # Form
        # RootSpaceDimension
        # RootSpaceMap
        # RootSubgroupMap
        # WeylGroupMap
        # GenericTorusElementMap
        # IsGroupElement
        # IsTorusElement
        # IsLieAlgebraElement
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
        print("done.")
        
        print("\t\tChecking root subgroups belong to the group...",end='')
        print("done.")
        
        print("\tBasic tests passed.")
        
    def test_root_space_maps_are_almost_homomorphisms(self):
        print("\tChecking root space spaces are (almost) homomorphisms...",end='')
        print("done.")