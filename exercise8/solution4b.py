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

def get_moment(rand_nums, m):
    return sum([x**m for x in rand_nums])/len(rand_nums)

if __name__ == '__main__':
    seed = 7777
    random.seed(seed)
    N = 100
    m = 10000
    means = []
    for _ in xrange(m):
        temp = 0
        for _ in xrange(N):
            temp += random.random() ** 2
        means.append(temp/N)
    pl.hist(means, bins=100, range=(0.2,0.5))
    pl.savefig("fig4b.png")
    pl.show()

    

