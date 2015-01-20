#!/usr/bin/env python

""" Compute power spectrum
"""

# Standard library imports
import sys
import numpy as np

# HDF5 package
import h5py

#
# ###################################################
#
def read_ramses_gpu_data(fileName,fieldName='density'):
    """Read hdf5 data from RAMSES-GPU simulation run using h5py module.

    Possible value for fieldName : density, energy,
    magnetic_field_x, magnetic_field_y, magnetic_field_z,
    momentum_x, momentum_y, momentum_z

    return numpy array of corresponfing field.
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
def powSpectrum(data_fftabs,nBins=32):
    """
    Compute power spectrum of Nd array data using nBins bins.
    
    arguments:
    data_fftabs -- a real value data array containing square modulus Fourier coefficients
    nBins       -- number of bins (power spectrum)
    """

    from math import sqrt

    # frequency
    data_psd   = np.zeros(nBins,dtype='float64')
    data_histo = np.zeros(nBins,dtype='int') 


    if len(data_fftabs.shape) == 2:
        lx,ly = data_fftabs.shape
        maxSize = sqrt(lx*lx+ly*ly)/2
    else:
        lx,ly,lz = data_fftabs.shape
        maxSize = sqrt(lx*lx+ly*ly+lz*lz)/2

    # resolution
    dk = maxSize/(nBins-1)

    # frequency array
    freq = np.arange(nBins)*dk

    if len(data_fftabs.shape) == 2:
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
                    
                kmod = sqrt(kx*kx+ky*ky)
                index = int(kmod/dk+0.5)
                
                data_histo[index] += 1
                data_psd[index] += data_fftabs[i,j]
            
    else:
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

                    data_histo[index] += 1
                    data_psd[index] += data_fftabs[i,j,k]

    # print data_psd, data_histo
    for i in range(nBins):
        if data_histo[i] != 0:
            data_psd[i] /= data_histo[i]
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
    data = read_ramses_gpu_data(hdf5Filename,'density')

    data_fftabs = squareModulusFourier(data)

    freq, pow_data = powSpectrum(data_fftabs,64)

    #print freq, pow_data
    print freq.shape, pow_data.shape

    import matplotlib.pyplot as plt
    

    plt.loglog(freq, pow_data)
    plt.show()
