# A class to package all of the information related 
# to a split torus subgroup of a pinned group.

import sympy as sp

class split_torus:
    
    def __init__(self, 
                 matrix_size,
                 rank, 
                 is_element, 
                 generic_element, 
                 trivial_character_matrix, 
                 nontrivial_character_entries):
        
        assert isinstance(matrix_size, int) and matrix_size >= 1
        self.matrix_size = matrix_size
                
        assert isinstance(rank, int) and rank >= 1, "Rank of a split torus must be a positive integer"
        self.rank = rank
        
        # is_element must be a function
        # Input: matrix (square), rank (integer)
        # Output: boolean
        # EXAMPLES:
        # def is_torus_element_SL(X, rank = None):
        #     return (is_diagonal(X) and X.det() == 1)
        # def is_torus_element_SO(X, rank):
        #     if not is_diagonal(X): return False
        #     n = X.shape[0]
        #     q = rank
        #     return (
        #         all(X[i, i] * X[q + i, q + i] == 1 for i in range(q)) and
        #         all(X[2*q + j, 2*q + j] == 1 for j in range(n - 2*q))
        #     )
        self.is_element = is_element 
        
        # generic_element must be a function
        # Input: string (label for the variable)
        # Output: matrix
        # EXAMPLES:
        # def generic_torus_element_SL(matrix_size, rank = None, letter = 't'):
        #     if rank is not None: assert matrix_size == rank + 1, "Rank of diagonal torus in SL_n is n-1"
        #     v = sp.symarray(letter, matrix_size, nonzero = True)
        #     return sp.diag(*v[:-1], v[-1] / sp.prod(v))
        # def generic_torus_element_SO(matrix_size, rank, letter = 't'):
        #     n = matrix_size
        #     q = rank
        #     v = sp.symarray(letter, rank, nonzero=True)
        #     return sp.diag(*v, *(1/v), *([1] * (n - 2*q)))
        self.generic_element = generic_element
        
        # trivial_characte_matrix must be a matrix
        # whose rows are a basis for the sublattice of trivial characters
        # EXAMPLES:
        # def trivial_characters_SL(matrix_size, rank = None):
        #     return np.ones((matrix_size, 1), dtype=int)
        # def trivial_characters_SO(matrix_size, rank):
        #     trivial_characters = [np.array([1 if j == i or j == i + rank else 0 for j in range(matrix_size)])for i in range(rank)]
        #     if not (rank == 1 and matrix_size == 2): trivial_characters.append([1] * matrix_size)
        #     return np.array(np.stack(trivial_characters, axis=1))
        self.trivial_character_matrix = trivial_character_matrix
        
        # nontrivial_character_entries must be a binary tuple
        # storing the nontrivial non-redundant entries
        # EXAMPLES:
        # def character_entries_SL(matrix_size, rank = None):
        #     return [1]*matrix_size
        # def character_entries_SO(matrix_size, rank):
        #     return [1]*rank + [0]*(matrix_size - rank)
        self.nontrivial_character_entries = nontrivial_character_entries
        
    def __repr__(self):
        element = self.generic_element(self.matrix_size, self.rank, letter = 't')
        return f"Split torus with \n\tMatrix size: {self.matrix_size} \n\tRank: {self.rank} \n\tGeneric element:\n" + sp.pretty(element)