"""
   Mini testbench for FFT-based Poisson solver.

   P. Kestener, June 6 2015.

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fftn
#from pyfftw.interfaces.numpy_fft import fftn

def initData(data,testCase):

    x0, y0, z0 = 0.5, 0.5, 0.5
    
    # get data dimensions
    Nx,Ny,Nz=data.shape

    Lx,Ly,Lz=1.0,1.0,1.0
    
    # sin wave
    if testCase==0:

        # essayer avec une longueur d'onde non entiere
        x = 1.0*np.arange(Nx)/Nx
        y = 1.0*np.arange(Ny)/Ny
        z = 1.0*np.arange(Nz)/Nz

        x = np.sin(2.0*np.pi*x/Lx) 
        y = np.sin(2.0*np.pi*y/Ly) 
        z = np.sin(2.0*np.pi*z/Lz) 

        x = x.reshape(Nx,1,1)
        y = y.reshape(1,Ny,1)
        z = z.reshape(1,1,Nz)

        data += x*y*z

        print("data shape %d %d %d") % ( data.shape[0] , data.shape[1] , data.shape[2] )

        exactSolution = - x*y*z / ( (2*np.pi/Lx)**2 + (2*np.pi/Ly)**2 + (2*np.pi/Lz)**2 )

        print("data shape %d %d %d") % ( exactSolution.shape[0] , exactSolution.shape[1] , exactSolution.shape[2] )
        
    # Gaussian shape
    elif testCase==1:

        x = 1.0*np.arange(Nx)/Nx - x0
        y = 1.0*np.arange(Ny)/Ny - y0
        z = 1.0*np.arange(Nz)/Nz - z0

        x = x.reshape(Nx,1,1)
        y = y.reshape(1,Ny,1)
        z = z.reshape(1,1,Nz)
        
        alpha = 30.0
        data += (4.0*alpha*alpha*(x*x+y*y+z*z)-6*alpha)*np.exp(-alpha*(x*x+y*y+z*z))

        exactSolution = np.exp(-alpha*(x*x+y*y+z*z))

    # uniform density inside a ball
    elif testCase==2:
        
        x = 1.0*np.arange(Nx)/Nx - x0
        y = 1.0*np.arange(Ny)/Ny - y0
        z = 1.0*np.arange(Nz)/Nz - z0

        x = x.reshape(Nx,1,1)
        y = y.reshape(1,Ny,1)
        z = z.reshape(1,1,Nz)


        R=0.1
        exactSolution = np.zeros((Nx,Ny,Nz))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    r = np.sqrt( (1.0*i/Nx-x0)**2 + (1.0*j/Ny-y0)**2 + (1.0*k/Nz-z0)**2 )
                    if r < R:
                        data[i,j,k] = 1.0
                        exactSolution[i,j,k] = r**2/6.0
                    else:
                        exactSolution[i,j,k] = -R**3/(3*r)+R**2/2

        # see http://arxiv.org/pdf/1106.0557.pdf
        #exactSolution = -1.0 / (4*np.pi) / np.sqrt( x**2 + y**2 )
        #exactSolution = np.log(x**2+y**2)
            
    # random
    elif testCase==3:

        data += np.random.rand(Nx,Ny,Nz)

        exactSolution = np.zeros((Nx,Ny,Nz))
        
    return exactSolution
        
# solve Laplacian(phi)=rho
def PoissonSolve(rho,method):

    # get data dimensions
    Nx,Ny,Nz=rho.shape    

    dx,dy,dz = 1./Nx, 1./Ny, 1./Nz
    
    # Fourier wave number arrays
    kx=Nx*np.fft.fftfreq(Nx).reshape(Nx,1,1)
    ky=Ny*np.fft.fftfreq(Ny).reshape(1,Ny,1)
    kz=Nz*np.fft.fftfreq(Nz).reshape(1,1,Nz)

    kx = np.repeat( np.repeat(kx,Ny, axis=1), Nz, axis=2)
    ky = np.repeat( np.repeat(ky,Nz, axis=2), Nx, axis=0)
    kz = np.repeat( np.repeat(kz,Nx, axis=0), Ny, axis=1)

    print("kx shape %d %d %d") % ( kx.shape[0] , kx.shape[1] , kx.shape[2] )
    print("ky shape %d %d %d") % ( ky.shape[0] , ky.shape[1] , ky.shape[2] )
    print("kz shape %d %d %d") % ( kz.shape[0] , kz.shape[1] , kz.shape[2] )

    # compute FFT(rho)
    rhoFFT = np.fft.fftn(rho)
    
    #
    # apply inverse Laplacian filter in Fourier space
    #
    # method 0: from Numerical Recipes (section 19.4)
    if method == 0:
        scaleFactor = 2.0*( (np.cos(2*np.pi*kx/Nx)-1)/(dx*dx) +
                            (np.cos(2*np.pi*ky/Ny)-1)/(dy*dy) +
                            (np.cos(2*np.pi*kz/Nz)-1)/(dz*dz) )

    # method 1: use continuous Fourier Transform
    if method == 1:
        scaleFactor = -4.0*np.pi**2*( kx**2 + ky**2 + kz**2 )

    scaleFactor[0,0,0] = 1.0 

    #print("scaleFactor")
    #plt.imshow(scaleFactor[0,:,:])
    #plt.show()
    
    rhoFFT = rhoFFT / scaleFactor
    rhoFFT[0,0,0] = 0.0
    
    return np.fft.ifftn(rhoFFT).real 
                
if __name__ == '__main__':
    
    Nx=32
    Ny=32
    Nz=32

    testCase = 2
    method = 0

    np.random.seed(12)
    
    import sys
    argc=len(sys.argv)
    if argc>1:
        Nx=int(sys.argv[1])
        Ny=Nx
        Nz=Nx
    # if argc>2:
    #     Ny=int(sys.argv[2])
    # if argc>3:
    #     Nz=int(sys.argv[3])
    # N  = Nx*Ny*Nz


    rho=np.zeros((Nx,Ny,Nz))
    phiExact=initData(rho,testCase)

    print("rho")
    plt.imshow(rho[Nx/2,:,:])
    plt.colorbar()
    plt.show()

    # solve Laplacian(phi)=rho
    phi = PoissonSolve(rho, method)

    if testCase==1:
        # compute phi min value
        phi = phi + 1.0 - phi.max()

    if testCase==2:
        phi = phi - phi.min()
        phiExact = phiExact - phiExact.min()
        
    tmp1 =  np.abs(phi-phiExact).sum()
    tmp2 =  np.abs(phi).sum()
    print 'L1 relative error : {}'.format(tmp1/tmp2)
    
    tmp1 =  ((phi-phiExact)**2).sum()
    tmp2 =  (phi**2).sum()
    tmp3 =  (phiExact**2).sum()

    print 'L2 relative error : {}'.format(tmp1/tmp2)
    print 'L2 absolute error : {}'.format(tmp1/Nx/Ny/Nz)
    print 'L2 solution - L2 exact : {}'.format(np.sqrt(tmp2)-np.sqrt(tmp3))

    if testCase==0:
        plt.imshow( phi[:,Ny/4,:]-phiExact[:,Ny/4,:] )
        plt.colorbar()
            
    elif testCase==2:
        #plt.imshow( phi )
        plt.plot(np.sqrt(phi[Nx/2,Ny/2,:]-phi.min()),'r--')
        plt.plot(np.sqrt(phiExact[Nx/2,Ny/2,:]-phiExact.min()),'b--')
        #plt.colorbar()
        
    else:
        #plt.imshow( phi[Nx/2,:,:]-phiExact[Nx/2,:,:] )
        plt.imshow( phi[:,Ny/2,:]-phiExact[:,Ny/2,:] )
        #plt.imshow( phi[Nx/2,:,:] )
        #plt.imshow( phiExact[Nx/2,:,:] )
        plt.colorbar()

    plt.show()
