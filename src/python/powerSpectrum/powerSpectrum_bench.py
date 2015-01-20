"""
   Mini benchmark for powerSpectrum (pure python versus cython)
"""

import sys

from powerSpectrum  import powSpectrum as powSpectrum_py
from powerSpectrum2 import powSpectrum as powSpectrum_cy

def bench_powerSpectrum_py(data,Nbins):
    freq, pow_data = powSpectrum_py(data,Nbins)
    print pow_data

def bench_powerSpectrum_cy(data,Nbins):
    freq, pow_data = powSpectrum_cy(data,Nbins)
    print pow_data

#
# MAIN
#
if __name__ == '__main__':
    from timeit import Timer
    
    nb =  1
    Nx = 100
    Ny = 100
    Nz = 100
    
    argc=len(sys.argv)
    if (argc>1):
        Nx=int(sys.argv[1])
    if (argc>2):
        Ny=int(sys.argv[2])
    if (argc>3):
        Nz=int(sys.argv[3])
        
    N  = Nx*Ny*Nz    

    Nbins = 32

    t1 = Timer("bench_powerSpectrum_py(data_fftabs,Nbins)", "import numpy as np; from __main__ import bench_powerSpectrum_py; from powerSpectrum import squareModulusFourier; data=np.arange(1.0*%d).reshape(%d,%d,%d); data_fftabs = squareModulusFourier(data); Nbins=%d" % (N,Nx,Ny,Nz,Nbins) )
    time_ps_py = t1.timeit(number=nb)/nb
    print 'power spectrum pure python (%d,%d,%d) : %f seconds' % (Nx,Ny,Nz,time_ps_py)

    t2 = Timer("bench_powerSpectrum_cy(data_fftabs,Nbins)", "import numpy as np; from __main__ import bench_powerSpectrum_cy; from powerSpectrum import squareModulusFourier; data=np.arange(1.0*%d).reshape(%d,%d,%d); data_fftabs = squareModulusFourier(data); Nbins=%d" % (N,Nx,Ny,Nz,Nbins) )
    time_ps_cy = t2.timeit(number=nb)/nb
    print 'power spectrum      cython (%d,%d,%d) : %f seconds' % (Nx,Ny,Nz,time_ps_cy)

    print 'speedup cython versus python %f' % (time_ps_py/time_ps_cy)

