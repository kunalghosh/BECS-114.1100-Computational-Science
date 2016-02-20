from __future__ import division
# from math import sin, cos
import pylab as pl
import numpy as np
from numpy import sin,cos

def f_dash_central_diff(f,x,h):
    '''
    Calculation of the first derivative 
    using the central difference formula
    f'(x) = f(x + h) - f(x - h)
            -------------------
                    2*h
    Inputs :- 
    f : The function whose first derivative is to be calculated.
        (Assuming the function just takes 1 argument)
    x : The value at which the number is to be calculated.
    h : Interval around x

    Output :-
    Returns the value of f(x) with truncation error
    ''' 
    
    return (f(x + h) - f(x - h)) / (2 * h)
    


if __name__ == '__main__':
    # The value at which the first derivative is to be calculated.
    x = 0.5

    # Interval around x 
    h = np.linspace(10 ** -8, 1, 25)


    # The function whose derivative is to be calculated
    f = sin

    # The analytical derivative of the above function
    f_dash = cos

    # Third derivative of f
    f_3dash = lambda x: -1 * cos(x)

    # The number of datapoints for which to calculate the plot
    # n = 250

    # Data points
    error_total = abs(f_dash_central_diff(f,x,h) - f_dash(x))
    error_trunc = ((-1 * (h ** 2)/6) * f_3dash(x))
    error_round = (error_total - error_trunc)
    dp = zip(h, error_total, error_trunc, error_round)

    # for _ in range(n):
    # for hi in h:
    #     # Calculate the value of the function 
    #     # error_total = abs(f_dash_central_diff(f, x, h) - f_dash(x))
    #     error_total = abs(f_dash_central_diff(f, x, hi) - f_dash(x))

    #     # Truncation error
    #     # error_trunc = abs((-1 * (h ** 2)/6) * f_3dash(x))
    #     error_trunc = abs((-1 * (hi ** 2)/6) * f_3dash(x))

    #     # Rounding error
    #     error_round = abs(error_total - error_trunc)

    #     dp.append([h, error_total, error_trunc, error_round])
    #     

    #     h = h/4

    # log_dp = np.log(dp)
    # dp = log_dp
    print(dp)
    # pl.loglog(dp[0], dp[1], dp[0], dp[2], dp[0], dp[3])
    # pl.plot(dp[0], dp[1], dp[0], dp[2], dp[0], dp[3])
    pl.loglog(dp[0], dp[2])
    pl.legend(["$\epsilon$", "$\epsilon_t$", "$\epsilon_r$"], loc = 4)
    pl.savefig("ex1.pdf")
    pl.show()
    


