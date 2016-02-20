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
# x^3 - 1 , re-written to get rid of powers
def f(x_r,x_i):	
    '''
    >>> f(1,1)
    (-3.0, 2.0)	
    >>> f(2,1)
    (1.0, 11.0)
    >>> f(1,0)
    (0.0, 0.0)
    '''
    z = complex(x_r, x_i)
    retVal = (z * z * z) - 1
    return retVal.real, retVal.imag 

# 3 x^2 , re-written to get rid of powers
def f_dash(x_r, x_i):
    '''
    >>> f_dash(1,1)
    (0.0, 6.0)
    >>> f_dash(2,1)
    (9.0, 12.0)
    '''
    z = complex(x_r, x_i)
    retVal = 3 * z * z
    return retVal.real, retVal.imag


def get_next_appox(x_r, x_i):
    '''
    x_r : previous real value of x
    x_i : previous imag value of x
    '''
    xr = x_r
    xi = x_i
    x = complex(xr, xi)

    fx_tuple = f(xr, xi)
    fx = complex(fx_tuple[0], fx_tuple[1])
    
    fdashx_tuple = f_dash(xr, xi)
    fdashx = complex(fdashx_tuple[0], fdashx_tuple[1])

    retVal = x - (fx / fdashx)
    return retVal.real, retVal.imag

def get_magnitude(z):
    xr, xi = z
    return xr * xr + xi * xi

def run_newton_method(xr_init, xi_init, epsilon=10 ** (-12), max_iter=100):
    '''
    xr_init = initial real value of x
    xi_init = initial imag value of x
    '''
    xr_old = xr_init
    xi_old = xi_init
    xr_new, xi_new = get_next_appox(xr_init, xi_init)
    while(max_iter != 0):
        max_iter -= 1
        xr_old, xi_old = xr_new, xi_new
        xr_new, xi_new = get_next_appox(xr_old, xi_old)
        if abs(xr_old - xr_new) < epsilon and abs(xi_old - xi_new) < epsilon:
            break
        if f(xr_new, xi_new) < epsilon:
            break
    return xr_new, xi_new

def whichRoot(xr, xi):
    retVal = -1
    if xr < -0.4 and xr > -0.6 and xi  > -0.9 and xi < -0.7:
        retVal = 1
        # retVal = "r."
    elif (xr + -1 < .01 and xi < .01):
        retVal = 2
        # retVal = "g."
    elif (xr > -0.6 and xr < -0.4) and (xi > 0.8 and xi < 0.9):
        retVal = 3
        # retVal = "b."
    return retVal

if __name__ == '__main__':
    doctest.testmod()

    x_real_nums = np.linspace(-1, 1, 1000)
    x_imag_nums = np.linspace(-1, 1, 1000)

    outputs = []

    for xr in x_real_nums:
        outputs.append([])
        for xi in x_imag_nums:
            fx_r, fx_i = run_newton_method(xr, xi)
            outputs[-1].append(whichRoot(fx_r,fx_i))
    pl.figure(1)
    pl.pcolor(np.array(outputs, np.float))
    pl.draw()
    pl.savefig("figure_q3_test.pdf")
    pl.show()
