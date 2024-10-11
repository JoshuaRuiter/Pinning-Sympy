# Custom class to model root systems of Lie algebras
# There is a built-in class implemented by Sympy, but
# it does not do some of the things that I would like.

from itertools import product

class root_system:
    
    def __init__(self,dynkin_type,rank,vector_length):
        assert(dynkin_type in ('A','B','C','BC','D','E','F','G'))
        assert(rank>0)
        self.dynkin_type = dynkin_type
        self.rank = rank
        self.vector_length = vector_length
        self.name_string = dynkin_type + str(rank)
        self.root_list = self.build_root_list()
        
    def build_root_list(self):
        # Construct root systems of various types
        if self.dynkin_type == 'A':
            # The vectors must have length at least rank+1
            assert(self.vector_length > self.rank)
            
            # There are rank*(rank+1) total roots
            product_excluding_diagonal = ((x, y) for x, y in product(range(self.rank+1), repeat=2) if x != y)
            root_list = ([[1 if index == i else (-1 if index == j else 0) 
                           for index in range(self.vector_length)] 
                         for (i,j) in product_excluding_diagonal])
            
        elif self.dynkin_type == 'B':
            # PLACEHOLDER
            x=0
            
        elif self.dynkin_type == 'C':
            # PLACEHOLDER
            x=0
        
        elif self.dynkin_type == 'BC':
            # PLACEHOLDER
            x=0
            
        elif self.dynkin_type == 'D':
            # PLACEHOLDER
            x=0
            
        elif self.dynkin_type == 'E':
            # PLACEHOLDER
            x=0
            
        elif self.dynkin_type == 'F':
            # PLACEHOLDER
            x=0
            
        elif self.dynkin_type == 'G':
            # PLACEHOLDER
            x=0
            
        else:
            raise Exception('Invalid Dynkin type, unable to construct list of roots.')
            
        return root_list
    
    def check_root_system_axioms(self):
        # PLACEHOLDER
        x=0
    
    def run_tests():
        print('\nRunning root system tests...')
        
        for n in (2,3,4):
            A_n = root_system('A',n,n+1)
            A_n.check_root_system_axioms()
            
            # ADD OTHER TYPES        
        
        print('\nRoot system tests complete.')

if __name__ == "__main__":
    root_system.run_tests()