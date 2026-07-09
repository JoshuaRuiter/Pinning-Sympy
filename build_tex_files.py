# Automated creation of .tex file from template for pinned_group objects stored in files

from pinned_group import pinned_group
import time
import sympy as sp

def main():
    
    print("\nRunning tests on groups")
    start_time = time.perf_counter()
    sp.init_printing(wrap_line=False)
    
    ########################
    overwrite = True
    compile_PDF = True
    n_min = 1
    n_max = 5
    q_min = 1
    q_max = 3
    eps_values = [-1,1]
    ########################
    
    #####################################################################
    build_tex_SL(n_min, n_max, overwrite, compile_PDF)
    build_tex_SO_split(n_min, n_max, q_min, q_max, overwrite, compile_PDF)
    build_tex_SO_nonsplit(n_min, n_max, q_min, q_max, overwrite, compile_PDF)
    build_tex_SU_quasisplit(n_min, n_max, q_min, q_max, eps_values, overwrite, compile_PDF)
    build_tex_SU_non_quasisplit(n_min, n_max, q_min, q_max, eps_values, overwrite, compile_PDF)
    #####################################################################
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\nAll tex operations complete, total time: {round(execution_time/60, 1)} minutes")

def build_tex_SL(n_min, n_max, overwrite, compile_PDF):
    print("\nBuilding .tex files for special linear groups")
    n_min = max(n_min, 2) # n=1 doesn't make sense, SL_1 is just the trivial group
    for n in range(n_min, n_max + 1):
        name_string = f"SL(n={n})"
        SL_n = pinned_group.load_from_file(name_string)
        SL_n.write_to_tex(overwrite, compile_PDF)

def build_tex_SO_split(n_min, n_max, q_min, q_max, overwrite, compile_PDF):
    ###################################################
    ## SPLIT SPECIAL ORTHOGONAL GROUPS (n=2q or n=2q+1) 
    ###################################################
    print("\nBuilding .tex files for split special orthogonal groups")
    q_min = max(q_min, 2) # doesn't make sense if q=1, there are no roots
    for q in range(q_min, q_max + 1):
        n_range = [n for n in (2*q, 2*q+1) if n_min <= n and n <= n_max]
        for n in n_range:
            name_string = f"SO(n={n}, q={q})"
            SO_n_q = pinned_group.load_from_file(name_string)
            SO_n_q.write_to_tex(overwrite, compile_PDF)

def build_tex_SO_nonsplit(n_min, n_max, q_min, q_max, overwrite, compile_PDF):
    #############################################################################
    ## NON-SPLIT SPECIAL ORTHOGONAL GROUPS 
        ## SO_n_q is quasi-split if n=2q+2, and
        ## neither split nor quasi-split if n>2+2q, 
        ## but the behavior seems to be basically the same in these two cases
    #############################################################################
    print("\nBuilding .tex files for non-split special orthogonal groups")
    for q in range(q_min, q_max + 1):
        n_min = max(2*q+2, n_min) # only non-split if n >= 2q+2
        for n in range(n_min, n_max + 1):
            name_string = f"SO(n={n}, q={q})"
            SO_n_q = pinned_group.load_from_file(name_string)
            SO_n_q.write_to_tex(overwrite, compile_PDF)
    
def build_tex_SU_quasisplit(n_min, n_max, q_min, q_max, eps_values, overwrite, compile_PDF):
    ############################################################
    ## QUASI-SPLIT SPECIAL UNITARY GROUPS (n=2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\nBuilding .tex files for quasi-split special unitary groups")
    for q in range(q_min, q_max + 1):
        n=2*q
        if n_min <= n and n <= n_max:
            for eps in eps_values:
                name_string = f"SU(n={n}, q={q}, eps={eps})"
                SU_n_q = pinned_group.load_from_file(name_string)
                SU_n_q.write_to_tex(overwrite, compile_PDF)

def build_tex_SU_non_quasisplit(n_min, n_max, q_min, q_max, eps_values, overwrite, compile_PDF):
    ############################################################
    ## NON-QUASI-SPLIT SPECIAL UNITARY GROUPS (n>2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("\nBuilding .tex files for non-(quasi-split) special unitary groups")
    q_min = max(q_min, 2) # doesn't make sense if q=1, there are no roots
    for q in range(q_min , q_max+1):
        n_min = max(2*q+1, n_min) # only non-quasi-split if n>=2*q+1
        for n in range(n_min, n_max + 1):
            for eps in eps_values:
                name_string = f"SU(n={n}, q={q}, eps={eps})"
                SU_n_q = pinned_group.load_from_file(name_string)
                SU_n_q.write_to_tex(overwrite, compile_PDF)
                
if __name__ == "__main__":
    main()