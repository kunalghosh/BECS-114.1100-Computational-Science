import numpy as np
import pylab as pl

def dot_prod(x,y):
    # the lenght of arrays must be same.
    assert(len(x) == len(y))
    # get corresponding values from x and y 
    # for v in zip(x,y)
    # find their product and put the result in a list, list comprehension.
    # sum over the list
    return np.sum([v[0]*v[1] for v in zip(x,y)])

def get_next_alpha(x,qPrev):
    # Caclulating new alpha now
    alpha_num = dot_prod(x*qPrev,qPrev)
    alpha_denom = dot_prod(qPrev,qPrev)
    return np.true_divide(alpha_num, alpha_denom) 

def get_next_beta(x,qPrev,qOld):
    # Calculating new beta now
    beta_num = dot_prod(x*qPrev,qOld) # beta's numerator
    beta_denom = dot_prod(qOld,qOld) # beta's denominator
    return np.true_divide(beta_num, beta_denom) 

def get_parameters(x,y):
    m = len(x) - 1
    # qi-2
    qOld = np.ones(x.shape)

    # Initial alpha alues
    alpha = np.zeros(x.shape)
    alpha[0] = np.true_divide(dot_prod(x * qOld, qOld),dot_prod(qOld,qOld))

    # allocating space for beta
    beta = np.zeros(x.shape)

    # qi-1
    qPrev = x - alpha[0]

    # Initializing c
    c = np.zeros(x.shape)
    c[0] = np.true_divide(dot_prod(y,qOld),dot_prod(qOld,qOld))
    c[1] = np.true_divide(dot_prod(y,qPrev),dot_prod(qPrev,qPrev))

    # Rho i and i-1
    rho = np.ones(x.shape)
    rho[0] = dot_prod(y,y) - np.true_divide(np.square(dot_prod(y,qOld)),dot_prod(qOld,qOld))
    rho[1] = rho[0] - np.true_divide(np.square(dot_prod(y,qPrev)),dot_prod(qPrev,qPrev))

    # Start with degree 1
    n = 1 

    # Initializing sigma_sq
    sigma_sq = np.ones(x.shape)
    sigma_sq[0] = np.true_divide(rho[0], m)
    sigma_sq[1] = np.true_divide(rho[1], m-n) 
    while(True):
        # Calculating new beta now
        # beta_num = dot_prod(x*qPrev,qOld) # beta's numerator
        # beta_denom = dot_prod(qOld,qOld) # beta's denominator
        # beta[n] = np.true_divide(beta_num, beta_denom) 
        beta[n] = get_next_beta(x,qPrev,qOld)
        
        # Caclulating new alpha now
        # alpha_num = dot_prod(x*qPrev,qPrev)
        # alpha_denom = dot_prod(qPrev,qPrev)
        # alpha[n] = np.true_divide(alpha_num, alpha_denom) 
        alpha[n] = get_next_alpha(x,qPrev)

        # Calculating q_n+1
        qNew = x*qPrev - alpha[n]*qPrev - beta[n]*qOld

        # Calculating new C
        c[n+1] = np.true_divide(dot_prod(y,qNew),dot_prod(qNew,qNew))

        # Calculate rho_n+1
        rho[n+1] = rho[n] - np.true_divide(np.square(dot_prod(y,qNew)),dot_prod(qNew,qNew))
        # Calculate sigma_sq_n+1
        sigma_sq[n+1] = np.true_divide(rho[n+1],m-(n+1))
        # if sigma_sq_n+1 > sigma_sq_n 
        # or abs(sigma_sq_n - sigma_sq_n+1) < 0.1
        # Then stop
        if sigma_sq[n+1] > sigma_sq[n] or np.abs(sigma_sq[n] - sigma_sq[n+1]) < 0.1:
            break
        else:
            # Update n += 1
            n += 1
            # update qOld,qPrev = qPrev,qNew
            np.copyto(qOld,qPrev)
            np.copyto(qPrev,qNew)
    
    return c,alpha,beta,sigma_sq,n

def evaluate(t,alpha,beta,c,n):
    x_vals = np.asarray(t)
    retVals = []
    for x in x_vals:
        sum = 0
        # qi-2
        qOld = np.ones(1)
        sum += c[0]*qOld
        # qi-1
        qPrev = x - alpha[0]
        sum += c[1]*qPrev
        for i in range(2,n+1): # 2,n+1 because 0 and 1 are already calculated
            qNew = x*qPrev - alpha[i]*qPrev - beta[i]*qOld
            qOld = qPrev
            qPrev = qNew
            sum += c[i] * qNew
        retVals.append(sum)
    return retVals

# def printVals(i,alpha,beta,variance,c):
#     def get_format_string(i):
#         w = [2,8,8,10,6] # widths in columns
#         if isinstance(i,int):
#             new_w = map(lambda x: "0:<.{}f".format(x),w)
#             retVal = "{{}} {{}} {{}} {{}} {{}}".format(*new_w)
#         else:
#             new_w = map(lambda x: ":<{}".format(x),w)
#             retVal = "{{}} {{}} {{}} {{}} {{}}".format(*new_w)
#         return retVal
# 
#     fmt_string = get_format_string(i)
#     print fmt_string
#     print(fmt_string.format(i,alpha,beta,variance,c))

def process_and_plot(data,pl,file_name=None):
    x,y = map(np.asarray,zip(*data))
    c,alpha,beta,sigma_sq,n = get_parameters(x,y)
    
    print("\n{:>2} {:>10} {:>10} {:>10} {:>10}".format("i","alpha","beta","variance","c"))
    for i in range(n+1):
        print("{:2d} {:10f} {:10f} {:10f} {:10f}".format(i,alpha[i],beta[i],sigma_sq[i],c[i]))
        # print(i,alpha[i],beta[i],sigma_sq[i],c[i])

    pl.scatter(x,y,label="Original Data")
    xvals = np.linspace(min(x),max(x),100)
    yvals = evaluate(xvals,alpha,beta,c,n)
    pl.plot(xvals,yvals,c='r',label="Fitted Polynomial")
    pl.xlim(min(x)-5,max(x)+5)
    pl.ylim(min(y)-5,max(y)+5)
    pl.xlabel("X")
    pl.ylabel("Y")
    pl.legend()
    if file_name != None:
        pl.savefig(file_name)

def print_data(data):
    for d in data:
        print("{} & {} \\\\".format(d[0],d[1]))

if __name__ == '__main__':
    # Q1.a data
    data = [(-9.70,3.76),
            (-7.30,1.78),
            (-5.40,1.52),
            (-5.00,1.31),
            (-3.01,0.31),
            (-2.13,0.23),
            (-1.20,0.45),
            (-0.56,0.29),
            (0.00,0.00),
            (1.20,0.45),
            (4.50,0.28),
            (6.70,2.12),
            (9.90,3.91),
            (10.00,3.47),
            (12.30,5.59)]
    print_data(data)
    pl.figure()
    process_and_plot(data,pl,"Q1_fig.png") 
    # test data, almost straight line
    data_2 = [(1,4),(2,4.5),(3,3.9),(4,4.6),(4,3.7)]
    pl.figure()
    process_and_plot(data_2,pl)
    # Q1.b data
    data_3 = [(1000,0.340),(1650,0.545),(1800,0.907),(1900,1.61),(1950,2.51),(1960,3.15),(1970,3.65),(1980,4.20),(1990,5.30)]
    pl.figure()
    process_and_plot(data_3,pl,"Q1b_fig.png")
    # Q1.b test data
    test_data = [(1000,0.340),(1100,0.38),(1200,0.42),(1300,0.5),(1450,0.51),(1650,0.545),(1800,0.907),(1900,1.61),(1950,2.51),(1960,3.15),(1970,3.65),(1980,4.20),(1990,5.30)]
    print_data(test_data)
    pl.figure()
    process_and_plot(test_data,pl,"Q1b_test_fig.png")

    pl.show()
    pl.close()
