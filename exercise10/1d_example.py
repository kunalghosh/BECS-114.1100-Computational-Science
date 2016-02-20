import pylab

# Simple 1 dimensional lattice with periodic boundary
#
# This is an incomplete example and only to give you some ideas of the
# possible ways for implementing the lattice and Monte Carlo simulation!
class PLattice:
  def __init__(self, length, initial_value):
    # create an array storing the values on the lattice and set them
    # to initial_value
    self.lattice = initial_value * pylab.ones(length)
    self.length = length
    # compute the energy of the initial configuration
    self.energy = self.compute_energy()
    # ... more variables

  # see below the flip method and the flip example in the main-part on how 
  # __getitem__ and __setitem__ work
  def __getitem__(self, idx):
    # the modulus operator implements the periodic boundary
    # (may not be the most efficient way but it's ok for this...)
    # one should check that negative values of idx behave also as expected
    return self.lattice[idx % self.length]

  def __setitem__(self, idx, val):
    # same here
    self.lattice[idx % self.length] = val

  def flip(self, idx):
    # this is equal to self[idx] = -1 * self[idx]
    # self[idx] causes call to either __getitem__ or __setitem__ (see below)
    self[idx] *= -1

  def compute_energy(self):
    # compute the energy here and return it
    return 0

  def do_montecarlo_step(self):
    # implement here and remove pass
    pass

  # ... more methods


if __name__ == "__main__":
  # create the lattice object
  l = PLattice(5, -1)

  # some simple examples of usage below:

  # print the energy
  print l.energy

  # flip the first lattice point (note that now energy is not correct and
  # should be updated accordingly...)
  # l[0] on the right hand side of the assignment calls __getitem__ with idx
  # set to 0
  # l[0] on the left hand side of the assignment calls __setitem__ with idx
  # set to 0 and val set to -1 * l[0] (thus implementing the assignment)
  # note: for 2D lattice, l[0,0] would set idx to (0,0)
  l[0] = -1 * l[0]

  # print the values of the lattice at the left neighbor, current index and
  # right neighbor to check that periodic boundary works...
  for i in xrange(l.length):
    print l[i-1], l[i], l[i+1]

  # let's flip the first lattice points back using the flip method since we
  # didn't update the energy...
  l.flip(0)

  # here's how the monte carlo simulation could be implemented
  # you need to use e.g. lists to keep track of the energy etc.
  # at each iteration...
  one_mcs = l.length
  runlength = 50000
  for i in xrange(1, runlength+1):
    for j in xrange(one_mcs): # or this loop could be inside do_montecarlo_step
      l.do_montecarlo_step()
    # ... keep track of the interesting quantities
    # print the progress in long runs
    if i % 10000 == 0:
      print "%d MCS completed." % i