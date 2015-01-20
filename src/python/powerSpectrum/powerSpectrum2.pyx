#!/usr/bin/env python

""" Compute power spectrum in a cythonic way.

    Use setup.py to build module.
    If you want to test the module, you can create an executable like this:
    * cython --embed powerSpectrum2.pyx
    * Use the gcc build command line from setup.py log, and add -lpython2.7 to link.
"""

# Standard library imports
from __future__ import division
import sys
import numpy as np
cimport numpy as np

# HDF5 package
import h5py

# NetCDF package
from scipy.io import netcdf as nc
#import Scientific.IO.NetCDF as nc

# data type
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "math.h":
    double sqrt(double data)
    float  sqrtf(float data)

#
# ###################################################
#
def read_h5(fileName,fieldName='density'):
    """Read hdf5 data from RAMSES-GPU simulation run using h5py module.

    Possible value for fieldName : density, energy,
    magnetic_field_x, magnetic_field_y, magnetic_field_z,
    momentum_x, momentum_y, momentum_z

    return numpy array of corresponding field.
    """

    f=h5py.File(fileName,'r')

    # TODO test if fieldName is in f.keys()
    data = f[fieldName].value

    f.close()

    print '%s read; returning field %s (%d,%d,%d)' % (fileName, fieldName, 
                                                      data.shape[0],
                                                      data.shape[1],
                                                      data.shape[2])
                                                          
    return data

#
# ###################################################
#
def read_nc(fileName,fieldName='rho'):
    """ReadNetCDF data from RAMSES-GPU simulation run using Scientific.IO.NetCDF module.

    Possible value for fieldName : rho, E, rho_vx, rho_vy, rho_vz,
    Bx, By, Bz

    return numpy array of corresponding field.
    """

    f=nc.NetCDFFile(fileName,'r')

    # TODO test if fieldName is in f.keys()
    data = f.variables[fieldName].getValue()

    # get total time
    totalTime = getattr(f, 'total time')

    f.close()

    print '%s read; returning field %s (%d,%d,%d)' % (fileName, fieldName,
                                                      data.shape[0],
                                                      data.shape[1],
                                                      data.shape[2])

    return data, totalTime

#
# ###################################################
#
def nc_read_totalTime(fileName):
    """ReadNetCDF data from RAMSES-GPU simulation run using Scientific.IO.NetCDF module.
    """
    f=nc.NetCDFFile(fileName,'r')
    # get total time
    totalTime = getattr(f, 'total time')

    f.close()

    return totalTime

#
# ###################################################
#
def squareModulusFourier(data):

    """Compute square modulus Fourier coefficient of array data.
    """
    # fft package
    from scipy.fftpack import fftn as fft_sc
    #from anfft import fftn as fft_an
   
    # compute fft
    data_fft = fft_sc(data)

    # abs(fft)^2
    data_fftabs =data_fft * data_fft.conjugate()

    # return real part
    return np.real(data_fftabs)

#
# ###################################################
#
def powSpectrum(np.ndarray[DTYPE_t, ndim=3] data_fftabs,
                int nBins=32):
    """
    Compute power spectrum of Nd array data using nBins bins.
    
    arguments:
    data_fftabs -- a real value data array containing square modulus Fourier coefficients
    nBins       -- number of bins (power spectrum)
    """

    assert data_fftabs.dtype == DTYPE

    #from math import sqrt
    #from libc.math cimport sqrt

    # frequency
    cdef np.ndarray[DTYPE_t , ndim=1] data_psd   = np.zeros(nBins,dtype=DTYPE)
    cdef np.ndarray[np.int_t, ndim=1] data_histo = np.zeros(nBins,dtype='int') 

    cdef int lx = data_fftabs.shape[0]
    cdef int ly = data_fftabs.shape[1]
    cdef int lz = data_fftabs.shape[2]
    
    # assume lx=ly=lz here
    # cdef DTYPE_t maxSize = sqrt( DTYPE(lx*lx+ly*ly+lz*lz) )/2.0
    cdef DTYPE_t maxSize = lx/2.0

    cdef double pi = 3.141592653589793

    # resolution
    cdef DTYPE_t dk = maxSize/(nBins-1)

    # frequency bin array
    cdef np.ndarray[DTYPE_t, ndim=1] freq = np.arange(nBins, dtype=DTYPE)*dk

    # local variable
    cdef int     i,  j,  k, index
    cdef int     kx, ky, kz, 
    cdef DTYPE_t kmod
    

    for i in range(lx):
            
        # compute kx
        if (i<lx/2):
            kx = i
        else:
            kx = i-lx
            
        for j in range(ly):
                    
            # compute ky
            if (j<ly/2):
                ky = j
            else:
                ky = j-ly
                
            for k in range(lz):
                
                # compute kz
                if (k<lz/2):
                    kz = k
                else:
                    kz = k-lz
                    
                kmod = sqrt(kx*kx+ky*ky+kz*kz)
                index = int(kmod/dk+0.5)

                if (kmod <= maxSize):
                    data_histo[index] += 1
                    data_psd[index]   += data_fftabs[i,j,k]

    # normalize psd
    for i in range(nBins):
        if data_histo[i] != 0:
            data_psd[i] /= data_histo[i]

            # use normalization specified in Kritsuk et al. arXiv1103.5525v2
            # paragraph 4.1.1
            data_psd[i] *= 4./3.*pi*( (i+1)**3-i**3 )*dk

    # print data_psd

    return freq,data_psd

if __name__ == "__main__":

    # get input hdf5 filename from argv[1]
    if (len(sys.argv) > 1):
        hdf5Filename = sys.argv[1]
    else:
        print "you must provide an HDF5 input filename."
        sys.exit("Execution failed.")

    # read HDF5 data
    data = read_h5(hdf5Filename,'density')

    data_fftabs = squareModulusFourier(data)

    freq, pow_data = powSpectrum(data_fftabs,64)

    #print freq, pow_data
    print freq.shape, pow_data.shape

    import matplotlib.pyplot as plt
    

    plt.loglog(freq, pow_data)
    plt.show()
