from __future__ import division
import numpy as np
import pylab as pl

def mc_circle_np(n):
    '''
    We would be drawing random samples from -1 to 1
    '''
    inside = 0
    x = np.random.random(n)*2-1
    y = np.random.random(n)*2-1
    inside = np.sum((np.square(x,out=x) + np.square(y,out=y)) < 1)
    return 4 * (inside/n)

if __name__ == '__main__':
    n = 1000
    pi = 3.141592654
    pi_est = []
    err_est = []
    avg_abs_err = []
    start,end,step = 1000,31000,1000
    for N in xrange(start,end,step):
        pis = np.zeros(n)
        for i in xrange(1000):
            pis[i] = mc_circle_np(N)
        pi_est_m = np.mean(pis)
        pi_est.append(pi_est_m)
        #err_est_m = np.log(np.std(pis)/np.sqrt(N))
        err_est_m = np.log(np.std(pis)/np.sqrt(n))
        err_est.append(err_est_m)
        avg_abs_err_m = np.log(np.abs(pi_est_m - pi))
        avg_abs_err.append(avg_abs_err_m)
        print("{} & {} & {} & {} \\\\".format(N,pi_est_m,err_est_m,avg_abs_err_m))
    pl.plot(pi_est)
    pl.ylabel("Pi Estimate")
    pl.ylim((3.1,3.2))
    pl.xlabel("Iterations")
    pl.grid()
    pl.savefig("pi_est.png")

    pl.figure()
    pl.plot(err_est)
    pl.plot(avg_abs_err)
    N = np.arange(start,end,step)
    pl.plot(np.log(np.true_divide(1,np.sqrt(N))))
    pl.xlabel("Iterations")
    pl.legend(["Error Estimate","log Average Absolute Err","log 1/sqrt(N)"])
    pl.grid()
    pl.savefig("err_est.png")
    pl.show()
