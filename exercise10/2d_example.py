from __future__ import division
from itertools import combinations
import cPickle as pickle
import sys

import pylab
import numpy as np

class PLattice:
    def __init__(self, dimension, initial_value, temperature, seed=None):
        # create an array storing the values on the lattice and set them
        # to initial_value
        if seed is not None:
            np.random.seed(seed)
            rand_vals = np.random.random(dimension)*2-1 # scale the rand numbers between -1,1
            self.lattice = np.floor(rand_vals) + np.ceil(rand_vals)
        else:
            np.random.seed(5555)
            self.lattice = initial_value * pylab.ones(dimension)

        self.dimension = dimension # dimension = (row,col)
        self.length = np.prod(self.dimension) 

        self.row_max = self.dimension[0]
        self.col_max = self.dimension[1]
        self.J = 1
        self.temp = temperature
        self.indices = np.asarray([zip(np.ones(self.col_max).astype(int)*r,np.arange(self.col_max)) for r in xrange(self.row_max)])
        l,b,h = self.indices.shape
        self.indices = self.indices.reshape(l*b,h)

        # compute the energy and magnetization of the initial configuration
        self.energy = self.compute_energy()
        self.magnetization = self.compute_magnetization()
  
    def __get_indices__(self, idx):
        # the modulus operator implements the periodic boundary
        # (may not be the most efficient way but it's ok for this...)
        # one should check that negative values of idx behave also as expected
        # idx in this case is a Tuple OR a List of Tuples
        # List of Tuples would allow us to vectorize operations.
        retVal = None
        if isinstance(idx,np.ndarray) or isinstance(idx,list):
            new_rows,new_cols = np.asarray(zip(*idx))
            new_rows = new_rows % self.row_max
            new_cols = new_cols % self.col_max
            retVal = np.asarray(zip(new_rows, new_cols))
        elif isinstance(idx,tuple):
            new_row = idx[0] % self.row_max
            new_col = idx[1] % self.col_max
            retVal = (new_row, new_col) 
        else:
            raise ValueError("Only list of Tuples or Tuples accepted")
        return retVal
    
    # see below the flip method and the flip example in the main-part on how 
    # __getitem__ and __setitem__ work
    def __get_left_idx(self,idx):
        retVal = None
        if isinstance(idx,np.ndarray) or isinstance(idx,list):
            rows,cols = np.asarray(zip(*idx))
            cols = (cols - 1) % self.col_max
            retVal = np.asarray(zip(rows,cols))
        elif isinstance(idx,tuple):
            retVal = (idx[0],(idx[1]-1) % self.col_max)
        else:
            raise ValueError("Only list of Tuples or Tuples accepted")
        return retVal

    def __get_right_idx(self,idx):
        retVal = None
        if isinstance(idx,np.ndarray) or isinstance(idx,list):
            rows,cols = np.asarray(zip(*idx))
            cols = (cols + 1) % self.col_max
            retVal = np.asarray(zip(rows,cols))
        elif isinstance(idx,tuple):
            retVal = (idx[0],(idx[1]+1) % self.col_max)
        else:
            raise ValueError("Only list of Tuples or Tuples accepted")
        return retVal

    def __get_top_idx(self,idx):
        retVal = None
        if isinstance(idx,np.ndarray) or isinstance(idx,list):
            rows,cols = np.asarray(zip(*idx))
            rows = (rows - 1) % self.row_max
            retVal = np.asarray(zip(rows,cols))
        elif isinstance(idx,tuple):
            retVal = ((idx[0]-1) % self.row_max,idx[1])
        else:
            raise ValueError("Only list of Tuples or Tuples accepted")
        return retVal

    def __get_bottom_idx(self,idx):
        retVal = None
        if isinstance(idx,np.ndarray) or isinstance(idx,list):
            rows,cols = np.asarray(zip(*idx))
            rows = (rows + 1) % self.row_max
            retVal = np.asarray(zip(rows,cols))
        elif isinstance(idx,tuple):
            retVal = ((idx[0]+1) % self.row_max,idx[1])
        else:
            raise ValueError("Only list of Tuples or Tuples accepted")
        return retVal

    def __getitem__(self, idx):
        idxes = self.__get_indices__(idx)
        retVal = None
        if isinstance(idx,np.ndarray) or isinstance(idx,list):
            retVal = self.lattice[zip(*idxes)]
        elif isinstance(idx,tuple):
            retVal = self.lattice[idxes]
        else:
            raise ValueError("Only list of Tuples or Tuples accepted")
        return retVal
  
    def __setitem__(self, idx, val):
        # same here
        self.lattice[self.__get_indices__(idx)] = val
  
    def flip(self, idx, compute_energy=True):
        # this is equal to self[idx] = -1 * self[idx]
        # self[idx] causes call to either __getitem__ or __setitem__ (see below)
        self[idx] *= -1
        if compute_energy:
            self.energy = self.compute_energy()
  
    def compute_magnetization(self):
        return np.sum(self.lattice)

    def get_energy(self):
        return self.energy
    
    def get_energy_per_site(self):
        return np.true_divide(self.get_energy(),self.length)

    def get_magnetization(self):
        return self.magnetization

    def get_magnetization_per_site(self):
        return np.true_divide(self.get_magnetization(),self.length)

    def compute_energy(self):
        # compute the energy here and return it
        # get values of the right cell and the bottom cell and do this for each cell.
        # this ensure that an i,j is not indexed twice
        right_indices = self.__get_right_idx(self.indices)
        bottom_indices = self.__get_bottom_idx(self.indices)
        right_vals = self[right_indices]
        bottom_vals = self[bottom_indices]
        cell_vals = self[self.indices]
        np.add(right_vals, bottom_vals, out=bottom_vals)
        np.multiply(bottom_vals,cell_vals,out=cell_vals)
        return np.sum(cell_vals) * -1
    
    def __is_flip_accepted(self, idx):
        retVal = None
        deltaE = 2 * self[idx] * ( self[self.__get_bottom_idx(idx)]
                                    + self[self.__get_top_idx(idx)]
                                    + self[self.__get_left_idx(idx)]
                                    + self[self.__get_right_idx(idx)] )
        if deltaE <= 0:
            retVal = True
        else:
            w = np.exp((-1 * deltaE)/self.temp) # kB = 1
            if np.random.random() < w:
                retVal = True
            else:
                retVal = False
        return retVal,deltaE

    def do_montecarlo(self):
        # we need max_row * max_col random indices  
        rand_row_indices = np.random.randint(low=0,high=self.row_max,size=self.length)
        rand_col_indices = np.random.randint(low=0,high=self.col_max,size=self.length)
        idxes = zip(rand_row_indices,rand_col_indices)
        for idx in idxes:
            result, deltaE = self.__is_flip_accepted(idx)
            if result: # Flip Accepted
                self.flip(idx,compute_energy=False)
                self.energy += deltaE
                self.magnetization += 2 * self[idx]

    def print_lattice(self):
        import pprint
        pprint.pprint(self.lattice)
    
    def get_lattice(self):
        return self.lattice

def plot_lattice(lattice,fileName):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig = plt.figure()
    x,y = lattice.shape
    X = np.arange(0, x, 1)
    Y = np.arange(0, y, 1)
    X, Y = np.meshgrid(X, Y)
    R = lattice[X,Y]
    surf = plt.imshow(R,origin='lower', aspect='auto', extent=(1,x,1,y))
    plt.savefig(fileName+".png")
    plt.close()

if __name__ == "__main__":
    # create the lattice object
    l = PLattice((32,32), -1,temperature = 2.265, seed=None)
  
    # print the energy
    print l.energy
  
    # print the values of the lattice at the left neighbor, current index and
    # right neighbor to check that periodic boundary works...

    # Code to check periodic boundary
    # for i in xrange(l.col_max):
    #     print "Col = {} -- ".format(i),
    #     print l[0,i-1], l[0,i], l[0,i+1]

    # for i in xrange(l.row_max):
    #     print "Row = {} -- ".format(i),
    #     print l[i-1,0], l[i,0], l[i+1,0]
  
    # here's how the monte carlo simulation could be implemented
    # you need to use e.g. lists to keep track of the energy etc.
    # at each iteration...
    runlength = 1000
    lattice_shape = (32,32)
    energies_per_temp = []
    magnetizations_per_temp = []
    lattices_per_temp = []
    xvals = range(1,runlength+1)
    seeds = [None,6000,7000,8000,9000,10000,11000]
    colours = ["k","r","g","b","c","m","y"]

    # For checking equlibriation
    m_old,m_new = -200,200 # Mean magnetization
    m_abs_old, m_abs_new = -200,200 # Mean abs magnetization
    
    for temp in [2.1,3.5]:
        energies = []
        magnetizations = []
        lattices = []
        pylab.figure()
        for idx,val in enumerate(zip(seeds,colours)):
            run_seed,color = val
            energy = []
            magnetization = []
            l = PLattice(lattice_shape, -1, temperature = temp, seed=run_seed)
            for i in xrange(1, runlength+1):
                # ... keep track of the interesting quantities
                # print the progress in long runs
                l.do_montecarlo()
                energy.append(l.get_energy_per_site())
                magnetization.append(l.get_magnetization_per_site())
                if i % 100 == 0:
                    # print "Temp = %f , Seed = %d , %d MCS completed." % (temp, run_seed if run_seed is not None else -1, i)
                    m_old, m_new = m_new, np.mean(magnetization[-100:]) 
                    m_abs_old, m_abs_new = m_abs_new, np.mean(np.abs(magnetization[-100:]))
                    err_m, err_abs = np.abs(m_old-m_new) , np.abs(m_abs_old - m_abs_new)
                    if err_m < 0.01:
                        print "Temp = {} Run {} Convergence after {} MCS at M = {}, M_err = {}".format(temp, idx,i,m_new,err_m)
                    if err_abs < 0.01:
                        print "Temp = {} Run {} Convergence after {} MCS at abs(M) = {}, abs(M)err = {}".format(temp,idx,i,m_abs_new,err_abs)
    
            print "Temp = {} Run {} Final Data after {} MCS at abs(M) = {}, abs(M)err = {}, abs(M) = {}, abs(M)err = {}".format(temp,idx,i,m_abs_new,err_abs,m_abs_new,err_abs)
            energies.append(energy)
            magnetizations.append(magnetization)
            lattices.append(l.get_lattice())
            plot_lattice(lattices[-1],"%d_run_%d"%(int(temp),idx))
    
            pylab.plot(xvals,energy,marker=".",c=color,label="Run %d" % idx)
            pylab.plot(xvals,magnetization,marker=".",c=color)
        pylab.legend(framealpha=0.5,loc=10)
        pylab.xlabel("Run Length")
        pylab.title("2D Ising model Temp = %f L = %d \n (Magnetization on Top, Energy below)" % (temp,lattice_shape[0]))
        pylab.savefig("energyVsmagnetization_%d.png"%int(temp))
        pylab.show()
        for idx,val in enumerate(zip(magnetizations,colours)):
            m,c = val
            pylab.plot(xvals, m, marker=".",c=c,label="Run %d" % idx)
        pylab.legend(framealpha=0.5,loc=10)
        pylab.xlabel("Run Length")
        pylab.ylabel("Magnetization")
        pylab.title("2D Ising model Temp = %f L = %d" % (temp,lattice_shape[0]))
        pylab.savefig("magnetization_%d.png"%int(temp))
        pylab.show()
        energies_per_temp.append(energies)
        magnetizations_per_temp.append(magnetizations)
        lattices_per_temp.append(lattices)
    
    with open('data.pkl', 'wb') as dat_dmp_file:
        pickle.dump([energies_per_temp, magnetizations_per_temp, lattices_per_temp], dat_dmp_file)
    
    temp = 2.265
    runlength = 50000
    xvals = range(1,runlength+1)
    pylab.figure()
    energies = []
    magnetizations = []
    lattices = []

    
    for idx,val in enumerate(zip([seeds[1]],[colours[1]])):
        run_seed,color = val
        energy = []
        magnetization = []
        l = PLattice(lattice_shape, -1, temperature=temp, seed=run_seed)
        for i in xrange(1, runlength+1):
            l.do_montecarlo()
            energy.append(l.get_energy_per_site())
            magnetization.append(l.get_magnetization_per_site())
            if i % 1000 == 0:
                print "Temp = %f , Seed = %d , %d MCS completed." % (temp, run_seed if run_seed is not None else -1, i)
        energies.append(energy)
        magnetizations.append(magnetization)
        lattices.append(l.get_lattice())
        plot_lattice(lattices[-1],"50000_lattice")
        # pylab.plot(xvals,energy,marker=".",c=color,label="Run %d" % idx)
        try:
            pylab.plot(xvals,magnetization,marker=".",c=color)
        except:
            pass

    with open('data50000.pkl', 'wb') as dat_dmp_file:
        pickle.dump([energies, magnetizations], dat_dmp_file)
        #pickle.dump(magnetizations, dat_dmp_file)

    pylab.legend(framealpha=0.5,loc=10)
    pylab.xlabel("Run Length")
    pylab.ylabel("Magnetization")
    pylab.title("2D Ising model Temp = %f L = %d" % (temp,lattice_shape[0]))
    pylab.savefig("magnetization5000_%d.png"%int(temp))
    pylab.show()

