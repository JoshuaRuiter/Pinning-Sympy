class nondegenerate_isotropic_form:
    
    # A class representing a nondegenerate isotropic 
    # symmetric bilinear, hermitian, or skew-hermitian form 
    # on a vector space V. 
    
    # In the case of a symmetric bilinear form, V is a vector space over a field K (char != 2)
    # In the case of a hermitian or skew-hermitian form, V is a vector space over a field L (char != 2)
    #   which is a quadratic extension of a field K.
    
    def __init__(self,
                 dimension,
                 witt_index,
                 epsilon,
                 anisotropic_vector,
                 primitive_element,
                 name_string):
        
        self.dimension = dimension  # The dimension of the associated vector space V,
                                    # also the size of the associated matrix
        self.witt_index = witt_index
        self.epsilon = epsilon # epsilon=1 for hermitian or symmetric bilinear
                               # epsilon=-1 for skew-hermitian
        self.anisotropic_vector = anisotropic_vector    # A vector to store the diagonal entries
                                                        # of the anisotropic block of the matrix   
        self.primitive_element = primitive_element  # Only relevant for hermitian/skew-hermitian
                                                    # The primitive element of the associated field extension
        self.name_string = name_string # string, should be 'symmetric bilinear' or 'hermitian' or 'skew-hermitian'