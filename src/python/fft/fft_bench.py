"""
   Mini benchmark for scipy FFT and anfft FFT

   anfft (FFTW wrapper) - http://code.google.com/p/anfft/
   Example install steps: 
     1. set env variable PYTHONUSERBASE
     2. unpack anfft-0.2.tar.gz
     3. run 'python setup.py install --user' to install package in PYTHONUSERBASE
"""

from scipy.fftpack import fftn as fft_sc
from anfft import fftn as fft_an

def bench_fft_scipy(data):
    """Test scipy fft"""
    fft_sc(data)

def bench_fft_anfft(data):
    """Test  an fft"""
    fft_an(data)

if __name__ == '__main__':
    from timeit import Timer
    
    nb = 1
    Nx=200
    Ny=400
    Nz=200

    import sys
    argc=len(sys.argv)
    if argc>1:
        Nx=int(sys.argv[1])
    if argc>2:
        Ny=int(sys.argv[2])
    if argc>3:
        Nz=int(sys.argv[3])
    N  = Nx*Ny*Nz

    t = Timer("bench_fft_scipy(data)", "import numpy as np; from __main__ import bench_fft_scipy; data=np.arange(1.0*%d).reshape(%d,%d,%d)" % (N,Nx,Ny,Nz) )
    time_scipy = t.timeit(number=nb)/nb
    print 'fft scipy (%d,%d,%d) : %f seconds' % (Nx,Ny,Nz,time_scipy)
    
    t2 = Timer("bench_fft_anfft(data)", "import numpy as np; from __main__ import bench_fft_anfft; data=np.arange(1.0*%d).reshape(%d,%d,%d)" % (N,Nx,Ny,Nz) )
    time_anfft = t2.timeit(number=nb)/nb
    print 'fft anfft (%d,%d,%d) : %f seconds' % (Nx,Ny,Nz,time_anfft)
 
    print 'speedup anfft versus scipy %f' % (time_scipy/time_anfft)
