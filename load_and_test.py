# Load pinned_groups from stored files, and run validation tests on each.
# Only run after constructing store pinned_group objects using compute_and_store.

from pinned_group import pinned_group
import time
import sympy as sp

def main():
    
    print("\nRunning tests on groups")
    start_time = time.perf_counter()
    sp.init_printing(wrap_line=False)
    
    ########################
    display = True
    n_min = 1
    n_max = 6
    q_min = 1
    q_max = 3
    eps_values = [-1,1]
    ########################
    
    #####################################################################
    run_SL_tests(n_min, n_max, display)
    run_SO_split_tests(n_min, n_max, q_min, q_max, display)
    run_SO_nonsplit_tests(n_min, n_max, q_min, q_max, display)
    run_SU_quasisplit_tests(n_min, n_max, q_min, q_max, eps_values, display)
    run_SU_nonquasisplit_tests(n_min, n_max, q_min, q_max, eps_values, display)
    #####################################################################
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\nAll tests complete, total time: {round(execution_time/60, 1)} minutes")

def run_SL_tests(n_min, n_max, display):
    print("Running tests for special linear groups")
    n_min = max(n_min, 2) # n=1 doesn't make sense, SL_1 is just the trivial group
    for n in range(n_min, n_max + 1):
        name_string = f"SL-n{n}"
        SL_n = pinned_group.load_from_file(name_string)
        assert SL_n.root_system.dynkin_type == 'A', \
            f"SL(n={n}) should have type A but " + \
            f"computations gave type {SL_n.root_system.dynkin_type}"
        SL_n.validate_pinning(display = display)

def run_SO_split_tests(n_min, n_max, q_min, q_max, display):
    ###################################################
    ## SPLIT SPECIAL ORTHOGONAL GROUPS (n=2q or n=2q+1) 
    ###################################################
    print("Running tests for split special orthogonal groups")
    q_min = max(q_min, 2) # doesn't make sense if q=1, there are no roots
    for q in range(q_min, q_max + 1):
        n_range = [n for n in (2*q, 2*q+1) if n_min <= n and n <= n_max]
        for n in n_range:
            name_string = f"SO-n{n}-q{q}"
            SO_n_q = pinned_group.load_from_file(name_string)
            if n==2*q:
                if q==2:
                    expected_type = ['A','A']
                elif q==3:
                    expected_type = 'A'
                else:
                    expected_type = 'D'
            else:
                expected_type = 'B'
            assert SO_n_q.root_system.dynkin_type == expected_type, \
                    f"SO(n={n}, q={q}) is type {expected_type} but computations " + \
                    f"gave type {SO_n_q.root_system.dynkin_type}"
            SO_n_q.validate_pinning(display = display)

def run_SO_nonsplit_tests(n_min, n_max, q_min, q_max, display):
    #############################################################################
    ## NON-SPLIT SPECIAL ORTHOGONAL GROUPS 
        ## SO_n_q is quasi-split if n=2q+2, and
        ## neither split nor quasi-split if n>2+2q, 
        ## but the behavior seems to be basically the same in these two cases
    #############################################################################
    print("Running tests for non-split special orthogonal groups")
    for q in range(q_min, q_max + 1):
        n_min = max(2*q+2, n_min) # only non-split if n >= 2q+2
        for n in range(n_min, n_max + 1):
            name_string = f"SO-n{n}-q{q}"
            SO_n_q = pinned_group.load_from_file(name_string)
            if q == 1:
                expected_type = 'A'
            else:
                expected_type = 'B'
            assert SO_n_q.root_system.dynkin_type == expected_type, \
                    f"SO(n={n}, q={q}) is type {expected_type} but computations " + \
                    f"gave type {SO_n_q.root_system.dynkin_type}"
            SO_n_q.validate_pinning(display = display)
    
def run_SU_quasisplit_tests(n_min, n_max, q_min, q_max, eps_values, display):
    ############################################################
    ## QUASI-SPLIT SPECIAL UNITARY GROUPS (n=2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("Running tests for quasi-split special unitary groups")
    for q in range(q_min, q_max + 1):
        n=2*q
        if n_min <= n and n <= n_max:
            for eps in eps_values:
                eps_str = "1" if eps > 0 else "minus1"
                name_string = f"SU-n{n}-q{q}-eps-{eps_str}"
                SU_n_q = pinned_group.load_from_file(name_string)
                if q == 1:
                    expected_type = 'A'
                elif q == 2:
                    expected_type = 'B'
                else:
                    expected_type = 'C'
                assert SU_n_q.root_system.dynkin_type == expected_type, \
                        f"SU(n={n}, q={q}, eps={eps}) is type {expected_type} but computations " + \
                        f"gave type {SU_n_q.root_system.dynkin_type}"
                SU_n_q.validate_pinning(display = display)

def run_SU_nonquasisplit_tests(n_min, n_max, q_min, q_max, eps_values, display):
    ############################################################
    ## NON-QUASI-SPLIT SPECIAL UNITARY GROUPS (n>2q)
    ## eps=1 is Hermitian, eps=-1 is skew-Hermitian
    ###########################################################
    print("Running tests for non-(quasi-split) special unitary groups")
    q_min = max(q_min, 2) # doesn't make sense if q=1, there are no roots
    for q in range(q_min , q_max+1):
        n_min = max(2*q+1, n_min) # only non-quasi-split if n>=2*q+1
        for n in range(n_min, n_max + 1):
            for eps in eps_values:
                eps_str = "1" if eps > 0 else "minus1"
                name_string = f"SU-n{n}-q{q}-eps-{eps_str}"
                SU_n_q = pinned_group.load_from_file(name_string)
                assert SU_n_q.root_system.dynkin_type == 'BC', \
                    f"SU(n={n}, q={q}, eps={eps}) is type BC but computations " + \
                    f"gave type {SU_n_q.root_system.dynkin_type}"
                SU_n_q.validate_pinning(display = display)
                
if __name__ == "__main__":
    main()