from __future__ import division
import doctest
import pylab as pl
import numpy as np
# a)	use newton's method to calculate roots
# 		equation = x^3 - x - 5
#		x_initial = 0.57735
#		max_iter = 50
#		print the results and explain them

## TODO : Write tests
# x^3 - x - 5 , re-written to get rid of powers
def f(x):
    '''
    >>> f(1)
    -5
    >>> f(2)
    1
    >>> f(3)
    19
    '''
    return (x * (x+1) * (x-1)) - 5

# 3 x^2 - 1 , re-written to get rid of powers
def f_dash(x):
    '''
    >>> f_dash(1)
    2
    >>> f_dash(2)
    11
    '''
    return (3 * (x + 1) * (x - 1)) + 2

def is_within_bound(val, lower, upper):
    returnVal = False
    # Checks for strict bounds
    if val >= lower and val <= upper:
        returnVal = True
    return returnVal

def get_next_approx_newton(x_n_minus_1):
    x = x_n_minus_1
    return x - (f(x) / f_dash(x))

def get_next_approx_bisection(func, b_min, b_max):
    # returns the new approx root of func
    # via bisection method, new lower and upper bounds
    mid = 0.5 * (b_min + b_max)
    new_b_min = b_min
    new_b_max = b_max
    if (func(new_b_min) * func(mid) < 0):
        new_b_max = mid
    elif (func(mid) * func(new_b_max) < 0):
        new_b_min = mid
    return mid, new_b_min, new_b_max

if __name__ == '__main__':
    doctest.testmod()
    x_n_minus_1 = 0.57735 # x_initial
    # iterate for a maximum of 50 iterations
    max_iter = 50
    output = []
    # Bounds for bisection method.
    # Assuming [-5, 5] arbitrarily and it also satisfies
    # f(-5) * f(5) < 0
    bisection_min, bisection_max = -5, 5

    # run till new root approximation and old root approximation
    # are not same upto the given precision
    tolerance = 0.0000000001

    # first approximation from newton's method to setup the while loop
    x_n = get_next_approx_newton(x_n_minus_1)
    print ["method", "New x", "f(x)"]
    print ["newton", x_n, f(x_n)]

    while abs(x_n - x_n_minus_1) > tolerance:
        x_n_minus_1 = x_n
        new_vals = []
        if is_within_bound(x_n, bisection_min, bisection_max):
            x_n = get_next_approx_newton(x_n_minus_1)
            # new_vals.append("newton")
            print ["newton", x_n, f(x_n)]
        else:
            x_n, bisection_min, bisection_max = get_next_approx_bisection(f, bisection_min, bisection_max)
            # new_vals.append("bisection")
            print ["bisection", x_n, f(x_n), "new bisec min = ", bisection_min, "new bisec max = ", bisection_max]
        new_vals.append(x_n)
        new_vals.append(f(x_n))
        output.append(new_vals)


#     for _ in xrange(max_iter):
#     	x_n = get_next_approx_newton(x_n_minus_1)
#     	output.append((x_n_minus_1, f(x_n_minus_1), x_n))
# 	x_n_minus_1 = x_n

#     for val in output:
# 	print val

    pl.figure(1)
    # plot the curve
    x = np.linspace(-5,5,1000)
    y = [f(_) for _ in x]
    pl.grid(True)
    pl.plot(x,y)
    pl.scatter(output[-1][0],0)
    # pl.scatter([v[0] for v in output],[0]*len(output), c=[np.linspace(0,1,3) for _ in range(20)])
    # pl.xlim(xmin=-5,xmax=5)
    pl.savefig("solution2b_fig.pdf")
    pl.show() 
