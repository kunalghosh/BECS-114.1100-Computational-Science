from __future__ import division
import numpy as np

def recursive_trapezoid(n,R,f,a,b):
    '''
    Computes the integral evaluation using the recursive trapezoid rule.
    The function actually evaluates R(n,m) but since m is 0 for us to 
    prefer trapezoid rule, we don't accept that as an input.
    It accepts the function 'f' we are evaluating.
    returns R(n,0) added to R.

    The base case of this recursive this function is R(0,0) which
    is assumed to have already been inserted into the dictionary R
    that's why it is not added here.
    '''
    retVal = -1
    if (n,0) in R.keys():
        retVal = R[(n,0)]
    elif n == 0:
        retVal = 0.5 * (b-a) * (f(a) + f(b))
    else:
        h = (b-a) * (0.5 ** n)
        retVal = 0.5 * recursive_trapezoid(n-1, R, f, a, b) + h * sum([ f(a+(2*k-1)*h) for k in xrange(1,(2**(n-1))+1)])
    return retVal

def get_R(n, m, R, f, a, b):
    '''
    This function computes and returns the value of R with R(n,m) added to it.
    It expects the old R as an input and also the function we are evaluating
    '''
    retVal = -1
    if (n,m) in R.keys():
        retVal = R[(n,m)]
    elif m == 0:
        # call recursive trapezoidal function
        retVal = recursive_trapezoid(n,R,f,a,b)
    else:
        retVal = get_R(n,m-1,R,f,a,b) + (1.0/((4**m)-1)) * (get_R(n,m-1,R,f,a,b) - get_R(n-1,m-1,R,f,a,b))
    return retVal
        
def integrate(f,a,b,nrows):
    # The R "matrix" which would store the values computed from trapezoidal and
    # Romberg's algorithm
    # This is implemented as a dictionary which stores a tuple (n,m) as the key
    # and the corresponding R(n,m) evaluation as the value against the (n,m) key
    R = {}
    rows = nrows
    for i in range(rows):
        for j in range(i+1):
            R[(i,j)] = get_R(i,j,R,f,a,b)
    return R[i,j], R

def printR(R, n):
    for i in range(rows):
        for j in range(i+1):
            print("{0:.10f}".format(R[(i,j)])),
        print
    print


if __name__ == '__main__':
    rows = 9
    #---------function 
    print("Analytical Integral is ln(3) = "),
    Actual = 1.0986122886681098
    print(Actual)
    f = lambda x: 1 / (1+x)
    a = 0
    b = 2
    integral, R = integrate(f,a,b,rows)
    print "Numerical Integral is :",integral
    print("Error is {}".format(abs(Actual-integral)))
    printR(R,rows)
    #---------function 2
    print("Analytical Integral is e^1 - 1 = "),
    Actual = 1.718281828459045
    print(Actual)
    f = lambda x: np.e ** x 
    a = 0
    b = 1
    integral, R = integrate(f,a,b,rows)
    print "Numerical Integral is", integral
    print("Error is {}".format(abs(Actual-integral)))
    printR(R,rows)
    #---------function 3
    print("Analytical Integral is 2/3 = "),
    Actual = 0.6666666666666666
    print(Actual)
    f = lambda x: np.sqrt(x) 
    a = 0
    b = 1
    integral, R = integrate(f,a,b,rows)
    print "Numerical Integral is", integral
    print("Error is {}".format(abs(Actual-integral)))
    printR(R,rows)

