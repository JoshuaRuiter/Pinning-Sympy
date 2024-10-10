import numpy
from sympy import Matrix, pprint, Identity, ZeroMatrix, BlockMatrix

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
                 primitive_element):
        
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
        self.primitive_element = primitive_element  # Only relevant for hermitian/skew-hermitian
                                                    # The primitive element of the associated field extension
        
        if self.epsilon == 0:
            self.name_string = 'symmetric bilinear'
            self.matrix = self.build_symmetric_matrix(self.dimension,self.witt_index,self.anisotropic_vector)
            
        elif self.epsilon == 1:
            self.name_string = 'hermitian'
            self.matrix = self.build_hermitian_matrix(self.dimension,self.witt_index,self.epsilon,
                                                      self.anisotropic_vector, self.primitive_element)
            
        elif self.epsilon == -1:
            self.name_string = 'skew-hermitian'
            self.matrix = self.build_hermitian_matrix(self.dimension,self.witt_index,self.epsilon,
                                                      self.anisotropic_vector, self.primitive_element)
        elif self.epsilon == 100:
            # Nonsense value of epsilon, just flag for 
            x=0
            
        else:
            raise Exception('Invalid type flag for nondegenerate isotropic form')
    
    @staticmethod
    def build_symmetric_matrix(dimension,witt_index,anisotropic_vector):
        n = dimension
        q = witt_index
        C = Matrix(numpy.diag(anisotropic_vector))
        I_q = Identity(q)
        Z_qq = ZeroMatrix(q,q)
        Z_qd = ZeroMatrix(q,n-2*q) # d=n-2q
        Z_dq = ZeroMatrix(n-2*q,q) # d=n-2q
        B = Matrix(BlockMatrix([[Z_qq,I_q,Z_qd],
                         [I_q,Z_qq,Z_qd],
                         [Z_dq,Z_dq,C]]))
        return B
    
    @staticmethod
    def build_hermitian_matrix(dimension,
                               witt_index,
                               epsilon,
                               anisotropic_vector,
                               primitive_element):
        n = dimension
        q = witt_index
        I_q = Identity(q)
        C = Matrix(numpy.diag(anisotropic_vector))
        Z_qq = ZeroMatrix(q,q)
        Z_qd = ZeroMatrix(q,n-2*q) # d=n-2q
        Z_dq = ZeroMatrix(n-2*q,q) # d=n-2q
        H = Matrix(BlockMatrix([[Z_qq,I_q,Z_qd],
                                [epsilon*I_q,Z_qq,Z_qd],
                                [Z_dq,Z_dq,C]]))
        return H