#!/usr/bin/env python

"""
Generate (fractional) Brownian field and save it in a Netcdf file.

"""

# numerical packages
import numpy as np

# Fast Fourier Transform
from scipy.fftpack import fft, ifft, fftn, ifftn

#
# Create 2D Brownian field (assume nx and ny are even)
#
# h : Holder exponent in [0, 1]
# if h == 0.5 : reguler Brownian motion
# if h != 0.5 : fractional Brownian motion
def createBrownianField2d(nx,ny,h=0.5):

    # initialize Fourier coef
    dataFFT = np.zeros((nx,ny)).astype(complex) 

    print dataFFT.flags
    print dataFFT.shape
    
    # 1st loop to fill quadrant 1 and 4
    for i in range(nx/2+1):

        # compute kx
        kx = i

        # compute i2
        if (i==0):
            i2=0
        elif (i==nx/2):
            i2=nx/2
        else:
            i2=nx-i
            
        for j in range(ny/2+1):

            # compute ky
            ky = j

            # compute j2
            if (j==0):
                j2=0
            elif (j==ny/2):
                j2=ny/2
            else:
                j2=ny-j

            kSquare = 1.0*(kx**2+ky**2)
            if kSquare>0:
                radius = np.power(kSquare, -(2*h+2)/4) * np.random.normal()
                phase = 2 * np.pi * np.random.uniform()
            else:
                radius = 1.0
                phase  = 0.0

            # fill fourier coefficient so that ifft is real (imag = 0)
            dataFFT[i ,j ] = radius*np.cos(phase) + 1j*radius*np.sin(phase)
            dataFFT[i2,j2] = radius*np.cos(phase) - 1j*radius*np.sin(phase)
            
    # make sure that Fourier coef at i=0, j=ny/2 is real
    dataFFT[0,ny/2] = np.real(dataFFT[0,ny/2]) + 1j*0

    # make sure that Fourier coef at i=nx/2, j=0 is real
    dataFFT[nx/2,0] = np.real(dataFFT[nx/2,0]) + 1j*0

    # make sure that Fourier coef at i=nx/2, j=ny/2 is real
    dataFFT[nx/2,ny/2] = np.real(dataFFT[nx/2,ny/2]) + 1j*0

    # 2nd loop to fill quadrant 2 and 3 
    for i in range(1,nx/2):

        # compute kx
        kx = i

        # compute i1,i2
        i1=nx-i
        i2=i
            
        for j in range(1,ny/2):

            # compute ky
            ky = j

            # compute j1,j2
            j1=j
            j2=ny-j

            kSquare = 1.0*(kx**2+ky**2)
            radius = np.power(kSquare, -(2*h+2)/4) * np.random.normal()
            phase = 2 * np.pi * np.random.uniform()

            # fill fourier coefficient so that ifft is real (imag = 0)
            dataFFT[i1,j1] = radius*np.cos(phase) + 1j*radius*np.sin(phase)
            dataFFT[i2,j2] = radius*np.cos(phase) - 1j*radius*np.sin(phase)

    return ifftn(dataFFT)
                    
#
# Create 3D Brownian field (assume nx, ny, nz are even)
#
# h : Holder exponent in [0, 1]
# if h == 0.5 : reguler Brownian motion
# if h != 0.5 : fractional Brownian motion
def createBrownianField3d(nx,ny,nz,h=0.5):

    # initialize Fourier coef
    dataFFT = np.zeros((nx,ny,nz)).astype(complex) 

    #print dataFFT.flags
    #print dataFFT.shape

    # first loop to fill octant 1 and 8
    for i in range(nx/2+1):

        # compute kx
        kx = i

        # compute i2
        if (i==0):
            i2=0
        elif (i==nx/2):
            i2=nx/2
        else:
            i2=nx-i

        for j in range(ny/2+1):

            # compute ky
            ky = j

            # compute j2
            if (j==0):
                j2=0
            elif (j==ny/2):
                j2=ny/2
            else:
                j2=ny-j

            for k in range(nz/2+1):

                # compute kz
                kz = k

                # compute k2
                if (k==0):
                    k2=0
                elif (k==nz/2):
                    k2=nz/2
                else:
                    k2=nz-k

                kSquare = 1.0*(kx**2+ky**2+kz**2)
                if kSquare>0:
                    radius = np.power(kSquare, -(2*h+3)/4) * np.random.normal()
                    phase = 2 * np.pi * np.random.uniform()
                else:
                    radius = 1.0
                    phase  = 0.0 

                # fill Fourier coef so that ifft is real
                dataFFT[i ,j ,k ] = radius*np.cos(phase) + 1j*radius*np.sin(phase)
                dataFFT[i2,j2,k2] = radius*np.cos(phase) - 1j*radius*np.sin(phase)

    # make sure that Fourier coef at i=nx/2 ... is real
    dataFFT[nx/2,0   ,0   ] = np.real(dataFFT[nx/2,0   ,0   ]) + 1j*0
    dataFFT[0   ,ny/2,0   ] = np.real(dataFFT[0   ,ny/2,0   ]) + 1j*0
    dataFFT[0   ,0   ,nz/2] = np.real(dataFFT[0   ,0   ,nz/2]) + 1j*0

    dataFFT[nx/2,ny/2,0   ] = np.real(dataFFT[nx/2,ny/2,0   ]) + 1j*0
    dataFFT[nx/2,0   ,nz/2] = np.real(dataFFT[nx/2,0   ,nz/2]) + 1j*0
    dataFFT[0   ,ny/2,nz/2] = np.real(dataFFT[0   ,ny/2,nz/2]) + 1j*0

    dataFFT[nx/2,ny/2,nz/2] = np.real(dataFFT[nx/2,ny/2,nz/2]) + 1j*0

    # 2nd loop to fill all other octants
    for i in range(1,nx/2):

        # compute kx
        kx = i

        for j in range(1,ny/2):

            # compute ky
            ky = j

            for k in range(1,nz/2):

                # compute kz
                kz = k

                # octant indexes
                i1 = nx-i
                i2 = i

                j1 = ny-j
                j2 = j

                k1 = nz-k
                k2 = k

                # fill Fourier coef so that ifft is real

                kSquare = 1.0*(kx**2+ky**2+kz**2)
                radius = np.power(kSquare, -(2*h+3)/4) * np.random.normal()
                phase = 2 * np.pi * np.random.uniform()

                dataFFT[i1,j2,k2] = radius*np.cos(phase) + 1j*radius*np.sin(phase)
                dataFFT[i2,j1,k1] = radius*np.cos(phase) - 1j*radius*np.sin(phase)

                radius = np.power(kSquare, -(2*h+3)/4) * np.random.normal()
                phase = 2 * np.pi * np.random.uniform()

                dataFFT[i2,j1,k2] = radius*np.cos(phase) + 1j*radius*np.sin(phase)
                dataFFT[i1,j2,k1] = radius*np.cos(phase) - 1j*radius*np.sin(phase)
                
                radius = np.power(kSquare, -(2*h+3)/4) * np.random.normal()
                phase = 2 * np.pi * np.random.uniform()

                dataFFT[i2,j2,k1] = radius*np.cos(phase) + 1j*radius*np.sin(phase)
                dataFFT[i1,j1,k2] = radius*np.cos(phase) - 1j*radius*np.sin(phase)
                
    return ifftn(dataFFT)
                    
    
#
# Save 3d data array in a netcdf file 
# only rho_vx is filled, dummy values in other fields
#
def saveNetcdf(fileName,data):

    # NetCDF package
    try:
        from ase.io.pupynere import NetCDFFile
    except ImportError:
        print "ase.io.pupynere not installed on this computer."

    # get input data shape
    nx,ny,nz = data.shape

    print 'data type '+str(data.dtype)
    
    # open file
    f = NetCDFFile(fileName,'w')

    # create netcdf dimensions
    f.createDimension('x',nx)
    f.createDimension('y',ny)
    f.createDimension('z',nz)

    rho = f.createVariable('rho','d',('x','y','z'))
    rho[:] = np.ones((nx,ny,nz))

    E = f.createVariable('E','d',('x','y','z'))
    E[:] = np.zeros((nx,ny,nz))
    
    rho_vx = f.createVariable('rho_vx','d',('x','y','z'))
    rho_vx[:] = data
    rho_vy = f.createVariable('rho_vy','d',('x','y','z'))
    rho_vy[:] = np.zeros((nx,ny,nz))
    rho_vz = f.createVariable('rho_vz','d',('x','y','z'))
    rho_vz[:] = np.zeros((nx,ny,nz))

    Bx = f.createVariable('Bx','d',('x','y','z'))
    Bx[:] = np.zeros((nx,ny,nz))
    By = f.createVariable('By','d',('x','y','z'))
    By[:] = np.zeros((nx,ny,nz))
    Bz = f.createVariable('Bz','d',('x','y','z'))
    Bz[:] = np.zeros((nx,ny,nz))

    f.sync()
    f.close()

#
# main
#
if __name__ == '__main__':

    # parse command line
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      default="test.nc",
                      help="write netcdf FILE", metavar="FILE")
    parser.add_option("-s", "--size", dest="size",
                      default=64,
                      help="linear size of 3D data", type="int")

    (options, args) = parser.parse_args()

    #print options.filename
    print 'Program run with args:\nfilename=%s\nsize=%d' % (options.filename, options.size)

    # 2d test
    #test = createBrownianField2d(64,64)
    #import matplotlib.pyplot as plt
    #plt.imshow(np.real(test))
    #plt.show()

    # 3d test
    # test = createBrownianField3d(32,32,32)
    # import matplotlib.pyplot as plt
    # plt.subplot(3,1,1)
    # plt.imshow(np.real(test[:,:,0]))
    # plt.subplot(3,1,2)
    # plt.imshow(np.real(test[:,0,:]))
    # plt.subplot(3,1,3)
    # plt.imshow(np.real(test[0,:,:]))
    # plt.show()

    # create synthetic data
    data = createBrownianField3d(64,64,64)
    
    # netcdf file save
    saveNetcdf(options.filename, np.real(data))
