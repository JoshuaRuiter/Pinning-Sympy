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
        
        assert dimension >= 1, "Cannot instantiate nondegenerate isotropic form with dimension < 1"
        assert witt_index >= 1, "Cannot instantiate nondegenerate isotropic form with Witt index < 1"
        assert epsilon in (None, 1, -1), "Cannot instantiate nondegenerate isotropic form: invalid value of epsilon"
        
        self.dimension = dimension  # The dimension of the associated vector space V,
                                    # also the size of the associated matrix
        self.witt_index = witt_index
        self.epsilon = epsilon  # epsilon=None to indicate symmetric bilinear
                                # epsilon=1 to indicate hermitian
                                # epsilon=-1 to indicate skew-hermitian
        self.anisotropic_vector = anisotropic_vector    # A vector to store the diagonal entries
                                                        # of the anisotropic block of the matrix
    
        self.anisotropic_matrix = None
        if anisotropic_vector is not None:
            self.anisotropic_matrix = sp.Matrix(sp.diag(*anisotropic_vector))
        
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
    
    @staticmethod
    def build_symmetric_matrix(dimension,
                               witt_index,
                               anisotropic_vector):
    
        #  Let B be the matrix of a nondegenerate symmetric bilinear form of Witt index q
        #       In general, using the theory of bilinear forms we can always assume that B is an (n x n) 
        #       block matrix of the form 
        #                   [0 I 0]
        #                   [I 0 0]
        #                   [0 0 C]
        #       where I is the (q x q) identity matrix and C is (n-2q x n-2q) and diagonal (and invertible). 

        #       In general, q<=n/2. If q=n/2 or q=(n-1)/2, then SO_n(k,B) is split.
        #       If q=n/2-1, then SO_n(k,B) is quasi-split.
        #       We ignore the case q=0, because in this case the group is not isotropic.
        
        n = dimension
        q = witt_index
        if n < 2*q:  raise ValueError(f"Invalid values to construct symmetric matrix, n={n}, q={q}")
        I = sp.eye(q)
        Z = sp.zeros(q)
        d = n - 2*q
        
        blocks = (
            [[Z, I], 
             [I, Z]] if d == 0 
            else
            [[Z, I, sp.zeros(q, d)],
             [I, Z, sp.zeros(q, d)],
             [sp.zeros(d, q), sp.zeros(d, q), sp.diag(*anisotropic_vector)]]
        )
        return sp.BlockMatrix(blocks).as_explicit()
    
    @staticmethod
    def build_hermitian_matrix(dimension,
                               witt_index,
                               anisotropic_vector,
                               epsilon,
                               primitive_element):
        
        #   Let H be the matrix of a nondegenerate hermitian or skew-hermitian form of Witt index q
        #       eps = 1 or -1, and determines skew-ness: H is hermitian if eps=1, skew-hermitian of eps=-1
        #   In general, using the theory of bilinear/hermitian forms we can always assume that H is an (n x n)
        #       block matrix of the form
        #                   [0      I       0]
        #                   [eps*I  0       0]
        #                   [0      0       C]
        #       where C is diagonal and satisfies conj(C)=eps*C. 
        #       In other words, if eps=1, then C has entries from k, and if eps=-1 then C has 
        #       "purely imaginary" entries from k(sqrt(d)), where sqrt(d) is the primitive element
        #
        #   In general, q<=n/2 because Witt index can never exceed half the dimension.
        #       If q=n/2, the group SU_n_q is quasi-split.
        #       We ignore the case q=0, because in this case the group SU_n_q is not isotropic.


        n = dimension
        q = witt_index
        if n < 2*q: raise ValueError(f"Invalid values to construct hermitian matrix, n={n}, q={q}")
        I = sp.eye(q)
        Z = sp.zeros(q)
        d = n - 2*q
        C = sp.diag(*anisotropic_vector)
        
        ################################################
        # I don't understand why this was commented out
        # It seems like it should be included
        ################################################
        # if epsilon == -1:
        #     C *= primitive_element
    
        blocks = (
            [[Z, I], [epsilon*I, Z]] if d == 0 else
            [[Z, I, sp.zeros(q, d)],
             [epsilon*I, Z, sp.zeros(q, d)],
             [sp.zeros(d, q), sp.zeros(d, q), C]]
        )
    
        return sp.BlockMatrix(blocks).as_explicit()

    
    
