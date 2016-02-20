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


def get_next_appox(x_n_minus_1):
    x = x_n_minus_1
    return x - (f(x) / f_dash(x))

if __name__ == '__main__':
    doctest.testmod()
    x_n_minus_1 = 0.57735 # x_initial
    # iterate for a maximum of 50 iterations
    max_iter = 50
    output = []
    for _ in xrange(max_iter):
    	x_n = get_next_appox(x_n_minus_1)
    	output.append((x_n_minus_1, f(x_n_minus_1)))
	x_n_minus_1 = x_n

    for val in output:
	print "& {} & {} & \\\\".format(val[0], val[1])

    pl.figure(1)
    # plot the curve
    x = np.linspace(-5,5,1000)
    y = [f(_) for _ in x]
    pl.grid(True)
    pl.plot(x,y)
    pl.scatter(output[-1][0],0)
    # pl.scatter([v[0] for v in output],[0]*len(output), c=[np.linspace(0,1,3) for _ in range(20)])
    # pl.xlim(xmin=-5,xmax=5)
    pl.savefig("solution2a_fig.pdf")
    pl.show() 
