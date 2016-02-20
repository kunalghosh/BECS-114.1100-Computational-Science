from __future__ import division
import numpy as np
import pylab as pl

def S(f,a,b):
    # Implement's the Simpson's method
    h = abs(b-a)
    c = (a+b)/2.0
    return (h/6.0) * (f(a) + 4*f(c) + f(b))

def adaptive_simpson(f,a,b,e,level_max):
    retVal = None
    c = (a+b)/2.0
    s1 = S(f,a,b)
    s2 = S(f,a,c) + S(f,c,b)
    err = (1/15.0)*(abs(s2 - s1))
    print "c={}".format(c)
    inpts.append(a)
    inpts.append(b)
    inpts.append(c)
    if level_max == 0 or err < e :
        # base case 
        # print " a={}, b={}, c={}, s1={}, s2={}, level={}".format(a,b,c,s1,s2,level_max)
        retVal = s2
    elif  err > e:
        # sub divide into left and right
        left = adaptive_simpson(f,a,c,e/2.0,level_max -1)
        right = adaptive_simpson(f,c,b,e/2.0,level_max -1)
        retVal = left + right
    return retVal

if __name__ == '__main__':
    # call adaptive simpson's for
    e = 0.5 * 10**(-4)
    level_max = 30
    # function 1
    f = lambda x: 4.0/(1 + x**2)
    a = 0
    b = 1
    inpts = [] # intervals
    result = adaptive_simpson(f,a,b,e,level_max)
    print("Numerical Integral by adaptive simpson's method: {}".format(result))

    xmin,xmax = int(a-2),int(b+2)+1 # for plots
    ymin,ymax = -5,5 # for plots
    x = np.linspace(xmin,xmax,1000)#floor and ceil used incase a and b are not ints
    x_interest = np.linspace(a,b,1000)
    pl.plot(x,f(x))
    pl.plot(x_interest, f(x_interest),color='red')
    # plotting axes
    pl.plot(range(xmin,xmax+1), 0*np.arange(xmin,xmax+1), color='black')
    pl.plot(0*np.arange(ymin,ymax+1), range(ymin,ymax+1), color='black')
    pl.scatter(inpts, [0 for _ in inpts], color="green")
    pl.plot()
    pl.xlabel("x")
    pl.ylabel("f(x)")
    pl.xlim(xmin,xmax)
    pl.ylim(ymin,ymax)
    pl.grid()
    pl.savefig("ex5_fig1.pdf")
    pl.show()
    
    # function 2
    f = lambda x: np.cos(2.0*x)/(np.e ** x)
    a = 0
    b = 2*np.pi
    inpts = [] # intervals
    result = adaptive_simpson(f,a,b,e,level_max)
    print("Numerical Integral by adaptive simpson's method: {}".format(result))

    xmin,xmax = int(a-2),int(b+2)+1 # for plots
    ymin,ymax = -0.3,1.3 # for plots
    x = np.linspace(xmin,xmax,1000)#floor and ceil used incase a and b are not ints
    x_interest = np.linspace(a,b,1000)
    pl.xlabel("x")
    pl.xlim(xmin,xmax)
    pl.ylim(ymin,ymax)
    pl.grid()
    pl.ylabel("f(x)")
    pl.plot(x,f(x))
    pl.plot(x_interest, f(x_interest),color='red')
    # plotting axes
    pl.plot(range(xmin,xmax+1), 0*np.arange(xmin,xmax+1), color='black')
    pl.plot(0*np.arange(int(ymin)-1,int(ymax)+1), range(int(ymin)-1,int(ymax)+1), color='black')
    pl.scatter(inpts, [0 for _ in inpts], color="green")
    pl.savefig("ex5_fig2.pdf")
    pl.show()
