# A class to package all of the information related 
# to a split torus subgroup of a pinned group.

class split_torus:
    
    def __init__(self, 
                 rank, 
                 is_element, 
                 generic_element, 
                 trivial_character_matrix, 
                 nontrivial_character_entries):
        
        assert isinstance(rank, int) and rank >= 1, "Rank of a split torus must be a positive integer"
        self.rank = rank
        
        # is_element should be a function
        # input: matrix
        # output: boolean
        self.is_element = is_element 
        
        # generic_element should be a function
        # input: string (label for the variable)
        # output: matrix
        self.generic_element = generic_element
        
        # trivial_characte_matrix should be a matrix
        # whose rows are a basis for the sublattice of trivial characters
        self.trivial_character_matrix = trivial_character_matrix
        
        # nontrivial_character_entries should be a binary tuple
        #storing the nontrivial non-redundant entries
        self.nontrivial_character_entries = nontrivial_character_entries
        