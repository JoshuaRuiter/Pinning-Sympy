import sympy as sp

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
                 anisotropic_vector = None,
                 epsilon = None,
                 primitive_element = None):
        
        if dimension == -1:
            # Flag used to have a "non-object"
            self.matrix = [1]
        
        self.dimension = dimension  # The dimension of the associated vector space V,
                                    # also the size of the associated matrix
        self.witt_index = witt_index
        self.epsilon = epsilon  # epsilon=0 to indicate symmetric bilinear
                                # epsilon=1 to indicate hermitian
                                # epsilon=-1 to indicate skew-hermitian
        self.anisotropic_vector = anisotropic_vector    # A vector to store the diagonal entries
                                                        # of the anisotropic block of the matrix
                                                        
        # self.anisotropic_matrix = Matrix(numpy.diag(anisotropic_vector))
        
        self.primitive_element = primitive_element  # Only relevant for hermitian/skew-hermitian
                                                    # The primitive element of the associated field extension
        
        
        if epsilon is None:  # symmetric bilinear case
            self.name_string = 'symmetric bilinear'
            self.matrix = self.build_symmetric_matrix(dimension, 
                                                      witt_index, 
                                                      anisotropic_vector)
        elif epsilon == 1 or epsilon == -1: # hermitian case
            self.name_string = 'hermitian' if epsilon == 1 else "skew-hermitian"
            self.matrix = self.build_hermitian_matrix(dimension, 
                                                      witt_index, 
                                                      anisotropic_vector,
                                                      epsilon, 
                                                      primitive_element)
        else:
            raise ValueError("Unsupported epsilon")
    
    @staticmethod
    def build_symmetric_matrix(dimension,
                               witt_index,
                               anisotropic_vector):
        n = dimension
        q = witt_index
        I_q = sp.eye(q) # Previous version of this line: I_q = sp.Identity(q)
        Z_qq = sp.zeros(q)
        if n == 2*q:
            M = sp.BlockMatrix([
                [Z_qq, I_q],
                [I_q, Z_qq]
                ])
        elif n > 2*q:
            Z_qd = sp.zeros(q, n-2*q)
            Z_dq = sp.zeros(n-2*q, q)
            C = sp.diag(*anisotropic_vector)
            M = sp.BlockMatrix([
                [Z_qq, I_q, Z_qd],
                [I_q, Z_qq, Z_qd],
                [Z_dq, Z_dq, C]
            ])
        else:
            raise Exception(f"Invalid values to construct symmetric matrix, n={n}, q={q}")
        return M.as_explicit()
    
    @staticmethod
    def build_hermitian_matrix(dimension,
                               witt_index,
                               anisotropic_vector,
                               epsilon,
                               primitive_element):
        n = dimension
        q = witt_index
        I_q = sp.eye(q) # Previous version of this line: I_q = sp.Identity(q)
        Z_qq = sp.zeros(q)
        if n == 2*q:
            M = sp.BlockMatrix([
                [Z_qq, I_q],
                [epsilon*I_q, Z_qq]
            ])
        elif n > 2*q:            
            Z_qd = sp.zeros(q, n-2*q)
            Z_dq = sp.zeros(n-2*q, q)
            C = sp.diag(*anisotropic_vector)
            M = sp.BlockMatrix([
                [Z_qq, I_q, Z_qd],
                [epsilon*I_q, Z_qq, Z_qd],
                [Z_dq, Z_dq, C]
            ])
        else:
            raise Exception(f"Invalid values to construct hermitian matrix, n={n}, q={q}")
        return M.as_explicit()
    
    
