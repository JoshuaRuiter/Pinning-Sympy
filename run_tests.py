#import pinned_group
from build_group import build_group
#import sympy
#from sympy.liealgebras.root_system import RootSystem

def main():
    run_SL_tests(1);
    run_SL_tests(2);
    run_SL_tests(3);

def run_SL_tests(n):
    SL_n = build_group("A",n+1)
    SL_n.run_tests()

if __name__ == "__main__":
    main()