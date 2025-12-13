from build_group import build_special_linear_group, build_special_unitary_group
import os

def main():
    
    ###########################
    ## Special linear groups
    ## Status: complete, works as intended
    ##########################
    for n in (2,3,4):
        SL_n = build_special_linear_group(n)
        SL_n.calculate_and_display_commutator_coefficients()
    
    #########################################################
    ## Quasi-split special unitary groups (n=2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ## Status: not fully implemented, doesn't account for multiple integral linear combinations
    ## and makes no attempt to extract coefficients beyond alpha+beta
    
    # for eps in (1,-1):
    #     for q in (2,3):
    eps = 1 #
    q = 2 #
    
    n=2*q
    SU_n_q = build_special_unitary_group(n,q,eps)
    SU_n_q.calculate_and_display_commutator_coefficients()
    
    # ## Some non-quasi-split special unitary groups (n>2q)
    # ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    # for eps in (1,-1):
    #     for q in (2,3):
    #         for n in (2*q+1,2*q+2):
    #             SU_n_q = build_special_unitary_group(n,q,eps)
    #             SU_n_q.calculate_and_display_commutator_coefficients()

if __name__ == "__main__":
    main()