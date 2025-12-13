# A class to model elements of a quadratic field extension
# DEPRECATED, no longer necessary

from sympy import symbols, sqrt

# Quadratic field extension element
class qfee: 
    
    def __init__(self,primitive_element,real,imag):
        self.primitive_element = primitive_element
        self.real = real
        self.imag = imag
    
    def __str__(q):
        # Get a string version of q
        # This is necessary for use in print statements
        return str(q.real+q.imag*q.primitive_element)
    
    def __eq__(q1,q2):
        # check if q1==q2
        return (q1.primitive_element == q2.primitive_element and
                q1.real == q2.real and
                q1.imag == q2.imag)
        
    def __add__(q1, q2):
        # Add two elements of the same quadratic field extension
        assert(q1.primitive_element == q2.primitive_element)
        return qfee(q1.primitive_element, q1.real+q2.real, q1.imag+q2.imag)
    
    def __neg__(q):
        # Negate a quadratic field extension element
        return qfee(q.primitive_element,-q.real,-q.imag)
    
    def __sub(q1,q2):
        # Perform the subtraction q1-q2
        assert(q1.primitive_element == q2.primitive_element)
        return qfee(q1.primitive_element, q1.real-q2.real, q1.imag-q2.imag)
    
    def __mul__(q1, q2):
        # Perform the multiplication q1*q2
        assert(q1.primitive_element == q2.primitive_element)
        pe = q1.primitive_element
        product_real_part = q1.real*q2.real + q1.imag*q2.imag*(pe**2)
        product_imag_part = q1.real*q2.imag + q1.imag*q2.real
        return qfee(q1.primitive_element,product_real_part,product_imag_part)
        
    def conj(q):
        # Conjugate q, i.e. replace the primitive element with its negative
        return qfee(q.primitive_element,q.real,-q.imag)

    def run_tests():
        # INCOMPLETE
        print("Testing class qfee (quadratic field extension element)...")
        
        d = symbols('d') # non-square in arbitrary field k
        pe = sqrt(d) # primitive element
        
        q1 = qfee(pe,1,1)
        q2 = qfee(pe,1,2)
        q3 = qfee(pe,2,3)
        q4 = qfee(pe,3,4)
        q5 = qfee(pe,1,0)
        q6 = qfee(pe,0,1)
        q7 = qfee(pe,2*d+1,3)
    
        # Testing addition and equality
        assert(q1+q2==q3)
        assert(q1+q3==q4)
        assert(q5+q6==q1)
        
        # Testing multiplication
        assert(q5*q1 == q1)
        assert(q5*q4 == q4)
        assert(q1*q2 == q7)
        
        # Test negation
        # INCOMPLETE
        
        # Test conjugation
        # INCOMPLETE
        
        print("All tests passed.")
        
if __name__ == "__main__":
    qfee.run_tests()
    
