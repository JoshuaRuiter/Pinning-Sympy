from pinned_group import pinned_group
from sympy.liealgebras.root_system import RootSystem

def build_group(dynkin_type,rank):
    # Build a pinned group from just the Dynkin diagram type and rank
    # Current goal is to support types A, B, and C
    
    # dynkin_type should be a string, "A", "B", "C"
    assert(dynkin_type == "A" or dynkin_type=="B" or dynkin_type=="C")
    
    # rank should be a positive integer
    assert(rank >= 1)
    
    if dynkin_type == "A":
        matrix_size = rank+1
        name_string = "special linear group of size " + str(matrix_size)
        root_system = RootSystem("A"+str(rank))
        
        my_group = pinned_group(name_string, matrix_size,root_system)
        
    elif dynkin_type == "B":
        # INCOMPLETE
        x=0
        
    elif dynkin_type == "C":
        # INCOMPLETE
        x=0
        
    return my_group