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

def lcg(x):
    # Implementing lcg(a,b,m)
    # a = 128 b = 0 m = 509
    a = 128
    b = 0
    m = 509
    randVal = (a*x + b)%m
    return randVal / m # scale the rand num to be in 0,1 range

def get_moment(rand_nums, m):
    return sum([x**m for x in rand_nums])/len(rand_nums)

if __name__ == '__main__':
    seed = 7777
    random.seed(seed)
    randLcg = lcg(seed)   
    lcgs = [randLcg]
    srgl = []
    N = 10**7
    moment = [[],[],[]]
    for _ in xrange(N):
        randLcg = lcg(randLcg)
        lcgs.append(randLcg)
        srgl.append(random.random())
    for m in [1,2,3]:
        trueMoment = 1/(m+1)
        print "M = {}".format(m)
        for N in [10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            lcgMoment = get_moment(lcgs[1:N], m)
            srglMoment = get_moment(srgl[1:N], m)
            moment[m-1].append((lcgMoment, srglMoment, trueMoment))
            print("{} & {} & {} & {} \\\\".format(N,lcgMoment,srglMoment,trueMoment))

    

