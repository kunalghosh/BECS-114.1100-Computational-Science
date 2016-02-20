from __future__ import division
# from math import sin, cos
import pylab as pl
import numpy as np
from numpy import sin,cos

def f_dash_central_diff(f,x,h):
    return (f(x + h) - f(x - h)) / (2 * h)
    
def calculate():
    x = 0.5
    h_initial = 0.5
    h = np.asarray([h_initial/(4 ** _) for _ in range(25)])

    f = sin
    f_dash = cos
    f_3dash = lambda x: -1 * cos(x)

    error_total = abs(f_dash_central_diff(f,x,h) - f_dash(x))
    error_trunc = abs((-1 * (h ** 2)/6) * f_3dash(x))
    error_round = abs(error_total - error_trunc)
    
    print(h, error_total,  error_trunc, error_round)
    pl.loglog(h, error_total, h, error_trunc, h, error_round)
    pl.legend(["$\epsilon$", "$\epsilon_t$", "$\epsilon_r$"], loc = 4)
    pl.savefig("solution1_fig.pdf")
    pl.show()
    
if __name__ == '__main__':
    calculate()
    
