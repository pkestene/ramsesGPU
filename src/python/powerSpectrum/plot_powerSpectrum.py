"""
   Plot power spectrum.
"""

import sys

import numpy as np
import h5py

from powerSpectrum2 import powSpectrum
from powerSpectrum2 import read_ramses_gpu_data
from powerSpectrum2 import squareModulusFourier

#
# MAIN
#
if __name__ == '__main__':
    from timeit import Timer
    
    Nbins=64
    
    # default data name is density
    dataname='density'

    argc=len(sys.argv)
    if argc<=1:
        print 'not enough parameter; you must provide hdf5 filename'
        sys.exit(1)
    if argc>1:
        filename=sys.argv[1]
    if argc>2:
        dataname=sys.argv[2]

    # read HDF5 data
    data = read_ramses_gpu_data(filename,dataname)

    # compute abs(Fourier coef)^2
    data_fftabs = squareModulusFourier(data)
    print 'abs(Fourier coef)^2 computed...'

    # compute power spectrum
    freq, pow_data = powSpectrum(data_fftabs,Nbins)

    # plot spectrum
    import matplotlib.pyplot as plt
    
    plt.loglog(freq, pow_data)
    plt.show()
