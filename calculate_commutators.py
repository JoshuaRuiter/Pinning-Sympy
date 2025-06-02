# This file is intended to help with calculating commutators for 
# special linear, special orthogonal, and special unitary groups.

import build_group as bg

def main():
    
    #########################################
    ## SPECIAL LINEAR GROUPS
    ########################################
    n_range = (2,3)
    SL_commutator_calculation(n_range)
    
def SL_commutator_calculation(n_range):
    # demonstration of SL_n calculations
    print("Computing commutators for special linear groups...\n")
    
    for n in n_range:
        SL_n = bg.build_special_linear_group(n)
        calculate_commutators(SL_n)
        
    print("Finished computing commutators for special linear groups...\n")

def calculate_commutators(my_group):
    # INCOMPLETE
    # Code below copied from the commuator formula test
    # Needs to be changed in various ways to work here
    x=0
    
    # for alpha in self.root_list:
    #     dim_V_alpha = self.root_space_dimension(self.matrix_size,self.root_system,alpha)
    #     u = symarray('u',dim_V_alpha)
        
    #     x_alpha_u = self.root_subgroup_map(self.matrix_size,self.root_system,self.form,alpha,u)
    #     for beta in self.root_list:
    #         # Commutator formula only applies when the two roots
    #         # not scalar multiples of each other
                            
    #         if not(self.root_system.is_proportional(alpha,beta)):
    #             dim_V_beta = self.root_space_dimension(self.matrix_size,self.root_system,beta)
    #             v = symarray('v',dim_V_beta)
    #             x_beta_v = self.root_subgroup_map(self.matrix_size,self.root_system,self.form,beta,v)
    #             LHS = x_alpha_u*x_beta_v*(x_alpha_u**(-1))*(x_beta_v**(-1))
                
    #             # This gets a list of all positive integer linear combinations of alpha and beta
    #             # that are in the root system. 
    #             # It is formatted as a dictionary where keys are tuples (i,j) and the value 
    #             # associated to a key (i,j) is the root i*alpha+j*beta
    #             linear_combos = self.root_system.integer_linear_combos(alpha,beta)
                
    #             # The right hand side is a product over positive integer lienar combinations of alpha and beta
    #             # with coefficients depending on some function N(alpha,beta,i,j,u,v)
    #             RHS = eye(self.matrix_size)
    #             for key in linear_combos:
    #                 i = key[0]
    #                 j = key[1]
    #                 root = linear_combos[key]
                    
    #                 # Check that i*alpha+j*beta = root
    #                 assert(np.all(i*alpha+j*beta == root))
    
    #                 # Compute the commutator coefficient that should arise
    #                 N = self.commutator_coefficient_map(self.matrix_size,self.root_system,self.form,alpha,beta,i,j,u,v);
    #                 my_sum = alpha+beta
    #                 RHS = RHS * self.root_subgroup_map(self.matrix_size,self.root_system,self.form,my_sum,N)
    #             assert(LHS.equals(RHS))
                
    print("Done with computing roots and root spaces for special linear groups.\n")

if __name__ == "__main__":
    main()