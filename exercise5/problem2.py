from __future__ import division
from scipy.linalg import hilbert
import numpy as np
from pprint import pprint
import pylab as pl

def new_solve(A,b):
    A = np.asarray(A, np.float)
    b = np.asarray(b, np.float)
    # Create the scale vector max(abs(ri)) of each row ri in A
    S = np.zeros(A.shape[0], np.float)
    S = np.max(np.abs(A), axis=1)
    # Create the Index vector
    I = np.asarray(range(np.max(S.shape)))
    # iterate over as many times as there are rows (r = 0 to rmax)
    for idx in range(len(I)):
        ## print "I = {}".format(I+1)
        ## print "A = "; pprint(A)
        ## print "b = "; pprint(b)
        
        r = I[idx]
        # get the values from the idx(th) column
        rows_we_want = I[idx:]
        ## print "rows_we_want = {}".format(rows_we_want+1)
        corresponding_column_values = A[rows_we_want, idx]
        ## print "corresponding_column_values = "
        ## pprint(corresponding_column_values) 
        # divide the column values by their corresponding scale and get the index of the row with max value
        div_val = np.true_divide(np.abs(corresponding_column_values),S[rows_we_want])
        I_idx = np.argmax(div_val)  
        ## print "div_val = {} I_idx = {}".format(div_val, I_idx)
        # because the above index is in I, the actual row is
        act_idx = I[idx:][I_idx]
        ## print "act_idx = {}".format(act_idx+1)
        max_row = A[act_idx,:]
        ## print "max_row = "
        ## pprint(max_row)
        # swap current Idx with max_row's idx
        ## print "Swapping idx={} act_idx={} I_idx={}".format(idx+1, act_idx+1, I_idx+1)
        # swap the 0th idx and the new max in the sub array of I
        I[idx:][0], I[idx:][I_idx] = I[idx:][I_idx], I[idx:][0]
        # iterate over remaining rows and update them
        ## print "rem_rows = {}".format(I[idx:]+1)
        for rem_rows in I[idx+1:]:
            # Get the appropriate multiple of the pivot row. to make the remaining row's idx column a zero
            multiplier = np.true_divide(A[rem_rows][idx],max_row[idx])
            ## print "idx = {} row = {} row_idx = {} max_row_val = {} multiplier = {}".format(act_idx, rem_rows+1, A[rem_rows][idx], max_row[idx], multiplier)
            A[rem_rows,idx:] -= max_row[idx:] * multiplier
            b[rem_rows] -= b[act_idx] * multiplier
        
    # print "DONE."
    return I, A, b

def gauss(I, A, b):
    # returns the solutions to x
    x = np.zeros(I.shape)
    # because this is directly used in indexing and
    # max index of x would be len(x) -1
    len_x = len(x)-1
    # reverse I because we go in reverse I order.
    I = I[::-1]
    for count,row in enumerate(I):
        # get the row which we need to evaluate.
        weighted_sum_of_already_computed_x = 0
        for i in range(count):
            # if its the first value, we need to evaluate once.
            # for the second value, we need to evaluate twice and so on.
            col = len_x-i
            weighted_sum_of_already_computed_x += A[row, col] * x[col]
        # len(x)-count-1 because indices from 3 to 0 when len(x) = 4
        x[len_x-count] = (b[row] - weighted_sum_of_already_computed_x) / A[row,len_x-count]


    return x

def error(x,x_actual):
    diff = np.abs(x-x_actual)
    return np.sqrt(np.divide(np.sum(diff ** 2),x.shape))


if __name__ == '__main__':
    errs = []
    # A = np.asarray([[  3, -13,   9,   3],
    #     [ -6,   4,   1, -18],
    #     [  6,  -2,   2,   4],
    #     [ 12,  -8,   6,  10]], np.float)
    # b = np.asarray([-19, -34,  16,  26], np.float)
    errors = []
    n_vals = [2,3,5,8,12,15]
    # n_vals = range(2,200)
    for n in n_vals:
        A = hilbert(n)
        b = np.sum(A,axis=1)
        I,A,b = new_solve(A,b)
        x = gauss(I,A,b)
        errors.append(error(x,np.ones(x.shape)))
        print n,errors[-1],np.linalg.cond(A)
        # print n,x,errors[-1],np.linalg.cond(A)
    # pl.subplot(211)
    # pl.plot(n_vals, np.log(errors),c='b')
    # pl.grid()
    # pl.scatter(n_vals, np.log(errors),c='r',marker="o")
    # pl.xlabel("values of n")
    # pl.ylabel("LOG(RMS error)")
    # pl.subplot(212)
    pl.plot(n_vals, errors,c='b')
    pl.grid()
    pl.scatter(n_vals, errors,c='r',marker="o")
    pl.xlabel("values of n")
    pl.ylabel("RMS error")
    pl.show()
    # for n in [2,5,8,12,15]:
    #     A = hilbert(n)
    #     b = np.sum(A, axis=1) # sum each row to get column matrix.
    #     x = solve(A,b)
    #     err.append((n,get_rms_error(x)))
