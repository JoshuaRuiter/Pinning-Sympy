# This is a custom class to model root systems of Lie algebras.
# There is a built-in class implemented by Sympy, but it lacks some functionality.

# Note that these root systems are not assumed to be reduced, i.e. they do not necessarily
# satisfy the assumption that the only multiple of a root in the root system are +1 and -1.
# Dropping this axiom leads to some "non-reduced root systems," but all such root systems are
# of type BC, which is essentially the union of the type B and type C root systems 
# (with a certain realization/embedding of these root systems).
# In particular, type BC includes some roots where twice the root is another root
# (so for those longer roots, half that root is a root).

# Running this file will run some tests constructing the various implemented root systems
# and verifying the root system axioms.

# As of June 2, 2025, only types A, B, C, and BC are implemented. 
# Types E, F, and G are not important to my current plans/research, so I have not implemented them yet.



import numpy as np

class root_system:
    
    def __init__(self,dynkin_type,rank,vector_length):
        # Constructor method
        # Inputs: 
            # dynkin_type - a string, should be 'A','B','C','BC','D','E','F', or 'G'.
            # rank - the rank of the root system, always a positive integer.
            #           for type A, any positive integer
            #           for type B, C, or BC should be at least 2
            #           for type D, rank must be at least 4
            #           for type E, rank must be 5, 6, 7, or 8
            #           for type F, rank must be 4
            #           for type G, rank must be 2
            # vector_length - the length of the vectors to store the roots as.
            #               in general, the vector_length can be longer than necessary,
            #               and the constructor will just pad the end with zeros
            #
            #               From a theoretical perspective, the vector_length is the
            #               dimension of Euclidean space that the root space is
            #               embedded inside of.
            #
            #               There are somewhat complicated dependencies between 
            #               vector_length and rank. For example, with type A,
            #               vector_length must be at least rank+1.
            #               There are different requirements for each type.
        
        assert(dynkin_type in ('A','B','C','BC','D','E','F','G'))
        assert(rank>0)
        self.dynkin_type = dynkin_type
        self.rank = rank
        self.vector_length = vector_length
        self.name_string = dynkin_type + str(rank)
        
        # The main data is the list of roots
        # We store the roots as vectors
        self.root_list = self.build_root_list()
    
    def build_root_list(self):
        # Construct root systems of various types
        
        if self.dynkin_type == 'A':
            # The vectors must have length at least rank+1
            assert(self.vector_length >= self.rank + 1)
            
            # There are rank*(rank+1) total roots
            # They are all vectors of the form (0,...,0,1,0,....,0,-1,0,...,0)
            # of length equal to rank+1, where the 1 and -1 can come in either order
            nonzero_positions = range(self.rank+1)
            all_positions = range(self.vector_length)
            root_list = [np.array([1 if i == pos1 else -1 if i == pos2 else 0 for i in all_positions])
                         for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 != pos2]
            
        elif self.dynkin_type == 'B':
            # The vectors must have length at least rank
            assert(self.vector_length >= self.rank)
            
            # There are 2*rank^2 total roots
            # They come in two types:
                # Short roots, of the form (0,...,0,+/-1,0,....,0)
                # Long roots, of the form (0,...,0,+/-1,0,...,0,+/-1,0,...,0)
            nonzero_positions = range(self.rank)
            all_positions = range(self.vector_length)
            
            positive_short_roots = [np.array([1 if i == pos else 0 for i in all_positions]) for pos in nonzero_positions]
            negative_short_roots = [np.array([-1 if i == pos else 0 for i in all_positions]) for pos in nonzero_positions]
            positive_long_roots =  [np.array([1 if i == pos1 or i == pos2 else 0 for i in all_positions]) 
                                    for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 < pos2]
            negative_long_roots =  [np.array([-1 if i == pos1 or i == pos2 else 0 for i in all_positions])
                                    for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 < pos2]
            mixed_long_roots = [np.array([1 if i == pos1 else -1 if i == pos2 else 0 for i in all_positions])
                                for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 != pos2]
            root_list = positive_short_roots + negative_short_roots + positive_long_roots + negative_long_roots + mixed_long_roots
            
        elif self.dynkin_type == 'C':      
            # The vectors must have length at least rank
            assert(self.vector_length >= self.rank)      
        
            # There are 2*rank^2 total roots
            # They come in two types:
                # Short roots, of the form (0,...,0,+/-1,0,...,0,+/-1,0,...,0)
                # Long roots, of the form (0,...,0,+/-2,0,....,0)
            nonzero_positions = range(self.rank)
            all_positions = range(self.vector_length)
            
            positive_long_roots = [np.array([2 if i == pos else 0 for i in all_positions]) for pos in nonzero_positions]
            negative_long_roots = [np.array([-2 if i == pos else 0 for i in all_positions]) for pos in nonzero_positions]
            positive_short_roots =  [np.array([1 if i == pos1 or i == pos2 else 0 for i in all_positions]) 
                                    for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 < pos2]
            negative_short_roots =  [np.array([-1 if i == pos1 or i == pos2 else 0 for i in all_positions])
                                    for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 < pos2]
            mixed_short_roots = [np.array([1 if i == pos1 else -1 if i == pos2 else 0 for i in all_positions])
                                for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 != pos2]
            root_list = positive_long_roots + negative_long_roots + positive_short_roots + negative_short_roots + mixed_short_roots

        elif self.dynkin_type == 'BC':
            # The vectors must have length at least rank
            assert(self.vector_length >= self.rank)    
        
            # There are 2*rank*(rank+1) total roots
            # They come in three types:
                # Short roots, of the form (0,...,0,+/-1,0,....,0)
                # Medium roots, of the form (0,...,0,+/-1,0,...,0,+/-1,0,...,0)
                # Long roots, of the form (0,...,0,+/-2,0,....,0)
            nonzero_positions = range(self.rank)
            all_positions = range(self.vector_length)
            
            positive_short_roots = [np.array([1 if i == pos else 0 for i in all_positions]) for pos in nonzero_positions]
            negative_short_roots = [np.array([-1 if i == pos else 0 for i in all_positions]) for pos in nonzero_positions]
            positive_long_roots = [np.array([2 if i == pos else 0 for i in all_positions]) for pos in nonzero_positions]
            negative_long_roots = [np.array([-2 if i == pos else 0 for i in all_positions]) for pos in nonzero_positions]
            positive_medium_roots =  [np.array([1 if i == pos1 or i == pos2 else 0 for i in all_positions]) 
                                    for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 < pos2]
            negative_medium_roots =  [np.array([-1 if i == pos1 or i == pos2 else 0 for i in all_positions])
                                    for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 < pos2]
            mixed_medium_roots = [np.array([1 if i == pos1 else -1 if i == pos2 else 0 for i in all_positions])
                                for pos1 in nonzero_positions for pos2 in nonzero_positions if pos1 != pos2]
            root_list = (positive_short_roots + negative_short_roots + 
                         positive_long_roots + negative_long_roots + 
                         positive_medium_roots + negative_medium_roots +
                         mixed_medium_roots)
                
        elif self.dynkin_type == 'D':
            # PLACEHOLDER, INCOMPLETE
            # There are 2*rank*(rank-1) total roots
            x=0
            
        elif self.dynkin_type == 'E':
            # PLACEHOLDER, INCOMPLETE
            x=0
            
        elif self.dynkin_type == 'F':
            # PLACEHOLDER, INCOMPLETE
            x=0
            
        elif self.dynkin_type == 'G':
            # PLACEHOLDER, INCOMPLETE
            x=0
            
        else:
            raise Exception('Invalid Dynkin type, unable to construct list of roots.')
            
        return root_list
    
    def is_root(self,vector_to_test):
        # Return true if vector_to_test is a root
        return any(np.array_equal(root,vector_to_test) for root in self.root_list)
    
    def reflect_root(self,alpha,beta):
        # Given two roots alpha and beta, compute the reflection of beta across the hyperplane perpendicular to alpha.
        # In usual notation, this is mathematically written as sigma_alpha(beta)
        return beta - 2*np.dot(alpha,beta)/np.dot(alpha,alpha) * alpha
    
    def is_proportional(self,alpha,beta):
        # Return true if alpha and beta are proportional roots
        # In most root systems, this only happens if alpha = beta or alpha = -beta,
        # but in the type BC root system it is possible to have alpha = 2*beta
        
        alpha_mask = (alpha != 0)
        beta_mask = (beta != 0)
        ratios = (alpha[beta_mask] / beta[beta_mask]) # Not a typo, I promise. Need to use the same mask for both.
        return np.array_equal(alpha_mask,beta_mask) and np.all(ratios == ratios[0])
    
    def integer_linear_combos(self,alpha,beta):
        
        # Return a list of all positive integer linear combinations
        # of two roots alpha and beta within a list of roots
        
        assert(self.is_root(alpha))
        assert(self.is_root(beta))
        
        # The output is a dictionary, where keys are tuples (i,j)
        # and values are roots of the form i*alpha+j*beta
        combos = {}
        
        if not(self.is_root(alpha+beta)):
            # If alpha+beta is not a root, there are no integer linear combos
            # and we return an empty list
            return combos
        else:
            combos[(1,1)] = alpha+beta
        
        # Run a loop where each iteration, we try adding alpha and beta
        # to each existing combo
        while True:
            new_combos = self.increment_combos(alpha,beta,combos)
            if len(combos) == len(new_combos):
                break;
            combos = new_combos
        
        return combos

    def increment_combos(self,alpha,beta,old_combos):
        new_combos = old_combos.copy() # A shallow copy
        for key in old_combos:
            i = key[0]
            j = key[1]
            old_root = old_combos[key]
            if self.is_root(alpha+old_root):
                new_combos[(i+1,j)] = old_root+alpha
            if self.is_root(beta+old_root):
                new_combos[(i,j+1)] = old_root+beta
        return new_combos
    
    def check_root_system_axioms(self):
        print('\nRunning tests to verify root system axioms for the ' + self.name_string + ' root system.')
        
        print('\tChecking that zero is not a root...',end='')
        zero_vector = np.zeros((1,self.vector_length),dtype=int)
        assert(not(self.is_root(zero_vector)))
        print('passed.')
        
        print('\tChecking that a reflection of a root is another root...',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                assert(self.is_root(self.reflect_root(alpha,beta)))
        print('passed.')
        
        print('\tChecking that the angle bracket of two roots is an integer...',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                angle_bracket = 2*np.dot(alpha,beta)/np.dot(alpha,alpha)
                assert(angle_bracket == int(angle_bracket))
        print('passed.')
        
        print('\tChecking that only multiples of root that are roots are +/-1 or +/-2 or +/-0.5...',end='')
        for alpha in self.root_list:
            for beta in self.root_list:
                if self.is_proportional(alpha,beta):
                    mask = (beta != 0)
                    ratio = (alpha[mask]/beta[mask])[0]
                    assert(ratio in (1.0,-1.0,2.0,-2.0,0.5,-0.5))
        print('passed.')
        print('Tests completed.')
    
    def run_tests():
        print('\nRunning root system tests on types A, B, C, and BC...')
        
        for n in (2,3,4):
            A_n = root_system('A',n,n+1)
            assert(len(A_n.root_list) == n*(n+1))
            assert(len(A_n.root_list[0]) == n+1)
            A_n.check_root_system_axioms()
            
            A_n_padded = root_system('A',n,n+2)
            assert(len(A_n_padded.root_list) == n*(n+1))            
            assert(len(A_n_padded.root_list[0]) == n+2)
            A_n_padded.check_root_system_axioms()
            
            B_n = root_system('B',n,n)
            assert(len(B_n.root_list) == 2*(n**2))
            assert(len(B_n.root_list[0]) == n)
            B_n.check_root_system_axioms()
            
            B_n_padded = root_system('B',n,2*n)
            assert(len(B_n_padded.root_list) == 2*(n**2))
            assert(len(B_n_padded.root_list[0]) == 2*n)
            B_n_padded.check_root_system_axioms()
            
            C_n = root_system('C',n,n)
            assert(len(C_n.root_list) == 2*(n**2))
            C_n.check_root_system_axioms()
            
            BC_n = root_system('BC',n,n)
            assert(len(BC_n.root_list) == 2*n*(n+1))
            BC_n.check_root_system_axioms()
            
            BC_n_padded = root_system('BC',n,2*n)
            assert(len(BC_n_padded.root_list) == 2*n*(n+1))
            assert(len(BC_n_padded.root_list[0]) == 2*n)
            BC_n_padded.check_root_system_axioms()
        
        #########################################################
        # Constructors for types D, E, F, G not implemented yet #
        #########################################################
        
        # for n in (4,5):
        #     D_n = root_system('D',n,n)
        #     assert(len(D_n.root_list) == 2*n*(n-1))
        #     D_n.check_root_system_axioms()
            
        # for n in (6,7,8):
        #     E_n = root_system('E',n,8)
        #     if n == 6:
        #         assert(len(E_n.root_list) == 72)
        #     elif n == 7:
        #         assert(len(E_n.root_list) == 126)
        #     elif n == 8:
        #         assert(len(E_n.root_list) == 240)
        #     E_n.check_root_system_axioms()
        
        # F_4 = root_system('F',4,4)
        # F_4.check_root_system_axioms()
        
        # G_2 = root_system('G',2,3)
        # G_2.check_root_system_axioms()
        
        print('\nTesting integer linear combos...',end='')
        A_5 = root_system('A',5,6)
        for alpha in A_5.root_list:
            for beta in A_5.root_list:
                combos = A_5.integer_linear_combos(alpha,beta)
                assert(len(combos) <= 1)
                
        B_5 = root_system('B',5,5)
        for alpha in B_5.root_list:
            for beta in B_5.root_list:
                combos = B_5.integer_linear_combos(alpha,beta)
                assert(len(combos) <= 2)
                
        C_5 = root_system('C',5,5)
        for alpha in C_5.root_list:
            for beta in C_5.root_list:
                combos = C_5.integer_linear_combos(alpha,beta)
                assert(len(combos) <= 2)
                
        BC_5 = root_system('BC',5,5)
        for alpha in BC_5.root_list:
            for beta in BC_5.root_list:
                combos = BC_5.integer_linear_combos(alpha,beta)
                assert(len(combos) <= 3)
        print('passed.')

        print('\nRoot system tests complete (for types A, B, C, and BC).')
    

if __name__ == "__main__":
    root_system.run_tests()