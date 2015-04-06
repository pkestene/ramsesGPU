"""
   Mini benchmark for FFT-based Poisson solver.

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fftn
#from pyfftw.interfaces.numpy_fft import fftn

def initData(data,testCase):

    x0, y0 = 0.5, 0.5
    
    # get data dimensions
    Nx,Ny=data.shape

    Lx,Ly=1.0,1.0
    
    # sin wave
    if testCase==0:

        # essayer avec une longueur d'onde non entiere
        x = 1.0*np.arange(Nx)/Nx
        y = 1.0*np.arange(Ny)/Ny

        x = np.sin(2.0*np.pi*x) 
        y = np.sin(2.0*np.pi*y) 

        x = x.reshape(Nx,1)
        y = y.reshape(1,Ny)

        data += np.dot(x,y)

        print("data shape %d %d") % ( data.shape[0] , data.shape[1] )

        exactSolution = - np.dot(x,y) / ( (4*np.pi**2) * (1/Lx**2 + 1/Ly**2) )

    # Gaussian
    elif testCase==1:

        x = 1.0*np.arange(Nx)/Nx - x0
        y = 1.0*np.arange(Ny)/Ny - y0

        x = x.reshape(Nx,1)
        y = y.reshape(1,Ny)
        
        alpha = 30.0
        data += 4.0*alpha*(alpha*(x*x+y*y)-1)*np.exp(-alpha*(x*x+y*y))

        exactSolution = np.exp(-alpha*(x*x+y*y))

    # uniform density inside a ball
    elif testCase==2:
        
        x = 1.0*np.arange(Nx)/Nx - x0
        y = 1.0*np.arange(Ny)/Ny - y0

        x = x.reshape(Nx,1)
        y = y.reshape(1,Ny)


        R=0.1
        exactSolution = np.zeros((Nx,Ny))
        for i in range(Nx):
            for j in range(Ny):
                r = np.sqrt( (1.0*i/Nx-x0)**2 + (1.0*j/Ny-y0)**2 )
                if r < R:
                    data[i,j] = 1.0
                    exactSolution[i,j] = r**2/4.0
                else:
                    exactSolution[i,j] = R**2/2.0*np.log(r/R)+R**2/4.0

        # see http://arxiv.org/pdf/1106.0557.pdf
        #exactSolution = -1.0 / (4*np.pi) / np.sqrt( x**2 + y**2)
        #exactSolution = np.log(x**2+y**2)
            
    # random
    elif testCase==3:

        data += np.random.rand(Nx,Ny)

        exactSolution = np.zeros((Nx,Ny))
        
    return exactSolution
        
# solve Laplacian(phi)=rho
def PoissonSolve(rho,method):

    # get data dimensions
    Nx,Ny=rho.shape    

    dx,dy = 1./Nx, 1./Ny
    
    # Fourier wave number arrays
    kx=Nx*np.fft.fftfreq(Nx).reshape(Nx,1)
    ky=Ny*np.fft.fftfreq(Ny).reshape(1,Ny)

    kx = np.dot( kx                        , np.ones(Ny).reshape(1,Ny) )
    ky = np.dot( np.ones(Nx).reshape(Nx,1) , ky)

    print("kx shape %d %d") % ( kx.shape[0] , kx.shape[1] )
    print("ky shape %d %d") % ( ky.shape[0] , ky.shape[1] )

    # compute FFT(rho)
    rhoFFT = np.fft.fftn(rho)
    
    #
    # apply inverse Laplacian filter in Fourier space
    #
    # method 0: from Numerical Recipes (section 19.4)
    if method == 0:
        scaleFactor = 2.0*( (np.cos(2*np.pi*kx/Nx)-1)/(dx*dx) + (np.cos(2*np.pi*ky/Ny)-1)/(dy*dy) )

    # method 1: use continuous Fourier Transform
    if method == 1:
        scaleFactor = -4.0*np.pi**2*( kx**2 + ky**2 )

    scaleFactor[0,0] = 1.0 
        
    rhoFFT = rhoFFT / scaleFactor
    rhoFFT[0,0] = 0.0
    
    return np.fft.ifftn(rhoFFT).real 
                
if __name__ == '__main__':
    
    Nx=100
    Ny=100

    testCase = 2
    method = 0

    np.random.seed(12)
    
    import sys
    argc=len(sys.argv)
    if argc>1:
        Nx=int(sys.argv[1])
        Ny=Nx
    # if argc>2:
    #     Ny=int(sys.argv[2])
    # if argc>3:
    #     Nz=int(sys.argv[3])
    # N  = Nx*Ny*Nz


    rho=np.zeros((Nx,Ny))
    phiExact=initData(rho,testCase)

    plt.imshow(rho)
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
    print 'L2 absolute error : {}'.format(tmp1/Nx/Ny)
    print 'L2 solution - L2 exact : {}'.format(np.sqrt(tmp2)-np.sqrt(tmp3))
    
    if testCase==2:
        #plt.imshow( phi )
        plt.plot(np.sqrt(phi[Nx/2,:]-phi.min()))
        plt.plot(np.sqrt(phiExact[Nx/2,:]-phiExact.min()))
        #plt.colorbar()
        
    else:
        plt.imshow( phi-phiExact )
        plt.colorbar()

    plt.show()
