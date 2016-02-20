from __future__ import division
import cPickle as pickle
import numpy as np

temp_equilibrium = {"2.1":500,"3.5":200} # for T=2.1 and T=3.5 respectively
# load the pkl file for 2.1 and 3.5 temperatures
with open("data.pkl") as f:
    [energies_per_temp, magnetizations_per_temp, lattices_per_temp] = pickle.load(f)

# for each temperature
for idx,temp in enumerate(temp_equilibrium):
    mag_sum = 0
    mag_sq_sum = 0
    mag_abs_sum = 0
    mag_abs_sq_sum = 0
    equlib_time = temp_equilibrium[temp]

    # skip the values for run in ground state. hence [1:]
    magnetizations = magnetizations_per_temp[idx][1:] 
    total_vals = 0
    for magnetization in magnetizations:
        mag_data = magnetization[equlib_time:]
        mag_sum += np.sum(mag_data) 
        total_vals += len(mag_data)
        mag_sq_sum += np.sum(np.square(mag_data))
        mag_abs_sum += np.sum(np.abs(mag_data))
        mag_abs_sq_sum += np.sum(np.square(np.abs(mag_data)))

    mean = lambda x: x/total_vals
    mag_mean = mean(mag_sum)
    mag_err = np.sqrt((1.0/(total_vals-1))*(mean(mag_sq_sum) - np.square(mean(mag_sum))))
    abs_mag_mean = mean(mag_abs_sum)
    mag_abs_err = np.sqrt((1.0/(total_vals-1))*(mean(mag_abs_sq_sum) - np.square(mean(mag_abs_sum))))

    print "For Temp = {} <m> = {} err = {} AND <|m|> = {} err = {}".format(temp,mag_mean,mag_err,abs_mag_mean,mag_abs_err)

with open("data50000.pkl") as f:
    [energies, magnetizations] = pickle.load(f)

equilibrium = 5000
mag_dat = magnetizations[0][equilibrium:]
mag_dat_sq = np.square(mag_dat)
mag_err = np.sqrt((1.0/(len(mag_dat)-1))*(np.mean(mag_dat_sq) - np.square(np.mean(mag_dat))))

abs_mag_dat = np.abs(mag_dat)
abs_mag_dat_sq = np.square(abs_mag_dat)
abs_mag_err = np.sqrt((1.0/(len(mag_dat)-1))*(np.mean(abs_mag_dat_sq) - np.square(np.mean(abs_mag_dat))))


print "For Temp = 2.265 <m> = {} err = {} AND <|m|> = {} err = {}".format(np.mean(mag_dat), mag_err, np.mean(abs_mag_dat), abs_mag_err)


