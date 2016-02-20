from __future__ import division
import numpy as np
import pylab as pl

one_by_nine = 1/9.0
nine_by_511 = 9/511.0
def getIest_Imp(N):
    y = np.random.random(N)
    # Inverse function calculated manually
    x = np.power(511*y + 1, one_by_nine)
    f = 2*np.power(x,8) -1
    w = nine_by_511 * np.power(x,8) 
    Iest = np.mean(np.true_divide(f,w))
    return Iest

def getIest_Sm(N):
    # b = 2 and a = 1 Hence b-a = 1 
    # which doesn't affect the integral 
    # has been omitted from the calculations
    x = np.random.random(N)+1 # shift the rnadom values between 1 and 2
    f = 2*np.power(x,8)-1
    Iest = np.mean(f)
    return Iest

if __name__ == '__main__':
    np.random.seed(7777)
    exact = 1013/9.0
    Iest_Imp = []
    absErr_Imp = []
    Iest_Sm = []
    absErr_Sm = []
    Ns = map(lambda x: int(round(x)), np.logspace(2, 5, 50))
    for N in Ns:
        estImp = getIest_Imp(N)
        Iest_Imp.append(estImp)
        errImp = abs(exact-estImp)
        absErr_Imp.append(errImp)

        estSm = getIest_Sm(N)
        Iest_Sm.append(estSm)
        errSm = abs(exact-estSm)
        absErr_Sm.append(errSm)

        print "{} & {} & {} & {} & {} \\\\".format(N,estImp,errImp,estSm,errSm)

    pl.plot(Ns, Iest_Imp,label="Importance Sampling")
    pl.plot(Ns, Iest_Sm,label="Sample Mean")
    pl.plot(Ns, np.ones(len(Iest_Sm))*exact, label="Exact")
    pl.legend(framealpha=0.5)
    pl.xlabel("Ns")
    pl.ylabel("Integral")
    pl.savefig("Integrals.png")
    
    pl.figure()
    pl.loglog(Ns,absErr_Imp,label="Importance Sampling")
    pl.loglog(Ns,absErr_Sm,label="Sample Mean")
    pl.loglog(Ns,map(lambda x:1/(x**0.5),Ns),label="1/sqrt(N)")
    pl.legend(framealpha=0.5)
    pl.xlabel("Ns")
    pl.ylabel("Absolute Errors")
    pl.savefig("Errors.png")
    pl.show()
