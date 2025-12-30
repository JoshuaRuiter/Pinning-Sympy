# This is a custom class to model root systems of Lie algebras.
# There is a built-in class implemented by Sympy, but it lacks some functionality.

# Note that these root systems are not assumed to be reduced, i.e. they do not necessarily
# satisfy the assumption that the only multiple of a root in the root system are +1 and -1.
# Dropping this axiom leads to some "non-reduced root systems," but all such root systems are
# of type BC, which is essentially the union of the type B and type C root systems 
# (given a certain realization/embedding of these root systems).
# In particular, type BC includes some roots where twice the root is another root
# (so for those longer roots, half that root is a root).

import numpy as np
import sympy as sp
from utility_roots import determine_dynkin_type

class root_system:
    
    def __init__(self, root_list):
        self.root_list = root_list
        
        # check that all roots are vectors of the same dimension
        vector_length = len(root_list[0])
        for alpha in root_list: 
            assert(len(alpha) == vector_length)
        self.vector_length = vector_length
        
        # determining if the root system is reduced
        self.is_reduced = True
        for alpha in root_list:
            for beta in root_list:
                if self.is_proportional(alpha,beta):
                    mask = (beta != 0)
                    ratio = (alpha[mask]/beta[mask])[0]
                    if ratio not in (1.0,-1.0): self.is_reduced = False
                    break        
            else:
                continue # only executed if the inner loop did NOT break
            break # only executed if the inner loop DID break
        
        # determining if the root system is simply laced (all roots of equal length)
        self.is_simply_laced = True
        for alpha in self.root_list:
            for beta in self.root_list:
                if np.dot(alpha,alpha) != np.dot(beta,beta):
                    self.is_simply_laced = False
                    break
            else:
                continue # only executed if the inner loop did NOT break
            break # only executed if the inner loop DID break
        
        # determine the rank by taking the rank 
        # of the matrix spanned by the roots
        M = sp.Matrix(self.root_list)
        self.rank = M.rank()
        
        # determining the Dynkin type
        # This is done by arbitrarily choosing a set of simple roots,
        # computing the Cartan matrix and various other metrics,
        # and using a chain of if-then case logic
        self.dynkin_type, r = determine_dynkin_type(self.root_list)
        
        #print("r=",r)
        #print("self.rank=",self.rank)
        
        assert(r == self.rank)
        self.name_string = self.dynkin_type + '_' + str(self.rank)    
    
    def is_root(self,vector_to_test):
        # Return true if vector_to_test is a root
        return any(np.array_equal(root,vector_to_test) for root in self.root_list)
    
    def reflect_root(self,alpha,beta):
        # Given two roots alpha and beta, compute the reflection of beta across
        # the hyperplane perpendicular to alpha.
        # In usual notation, this is mathematically written as sigma_alpha(beta)
        
        # print("\nalpha=",alpha)
        # print("beta=",beta)
        # print("reflection=",beta - 2*np.dot(alpha,beta)/np.dot(alpha,alpha) * alpha)
        # print("\nAll roots:")
        # for r in self.root_list:
        #     sp.pprint(r)
        
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
    
    def verify_root_system_axioms(self):
        print('\nRunning tests to verify root system axioms for the ' + self.name_string + ' root system.')
        
        print('\tChecking that zero is not a root...',end='')
        zero_vector = np.zeros((1,self.vector_length),dtype=int)
        assert(not(self.is_root(zero_vector)))
        print('passed.')
        
        print('\tChecking that the negative of a root is a root...',end='')
        for alpha in self.root_list:
            assert(self.is_root(-alpha))
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
        print('Root system axiom checks completed.')