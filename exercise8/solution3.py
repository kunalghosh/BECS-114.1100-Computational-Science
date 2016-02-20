from __future__ import division
import random
import math
import pylab as pl

pi    = math.pi
log   = math.log
cos   = math.cos
sin   = math.sin
sqrt  = math.sqrt
floor = math.floor

def get_gauss_random():
    x1 = random.random()
    x2 = random.random()

    y1 = sqrt(-2*log(x1))*cos(2*pi*x2)
    y2 = sqrt(-2*log(x1))*sin(2*pi*x2)

    # We don't multiply by sigma
    # implying the value is unit variance

    # Since are random numbers are uniform
    # between 0 and 1, we have zero mean.

    # Hence our distributionis zero mean and 
    # unit variance.
    return y1,y2


def get_bin(yi, ymin, ymax, Nbin):
    val = ((yi - ymin) / (ymax - ymin)) * Nbin
    return floor(val)

if __name__ == '__main__':
    seed = 7777
    random.seed(seed)
    Nbin = 100
    bins = [0] * Nbin
    ymin, ymax = -5, 5
    binVals = []
    ybins = []
    for _ in xrange(100000):
        y = get_gauss_random()
        ybins.append(y)
        #binVals.append(get_bin(yi, ymin, ymax, Nbin))
        # binNum = get_bin(yi, ymin, ymax, Nbin)
        # bins[binNum]+=1
    y1s,y2s = zip(*ybins)
    pl.figure()
    pl.grid()
    pl.hist(y1s,bins=Nbin,range=(ymin,ymax))
    #pl.hist(binVals, bins=Nbin)
    pl.xlabel("Bins")
    pl.title("Y1s")
    pl.savefig("figure3_y1s.png")
    pl.figure()
    pl.grid()
    pl.hist(y2s,bins=Nbin,range=(ymin,ymax))
    #pl.hist(binVals, bins=Nbin)
    pl.xlabel("Bins")
    pl.title("Y2s")
    pl.savefig("figure3_y2s.png")
    pl.show()
