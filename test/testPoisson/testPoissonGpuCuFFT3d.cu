/**
 * A very simple test for solving Laplacian(\phi) = rho using FFT in 3D.
 *
 * Laplacian operator can be considered as a low-pass filter.
 * Here we implement 2 types of filters :
 *
 * method 0 : see Numerical recipes in C, section 19.4
 * method 1 : just divide right hand side by -(kx^2+ky^2+kz^2) in Fourier
 *
 * Test case 0:  rho(x,y,z) = sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*sin(2*pi*z/Lz)
 * Test case 1:  rho(x,y,z) = (4*alpha*alpha*(x^2+y^2+z^2)-6*alpha)*exp(-alpha*(x^2+y^2+z^2))
 * Test case 2:  rho(x,y,z) = ( r=sqrt(x^2+y^2+z^2) < R ) ? 1 : 0 
 *
 * Example of use:
 * ./testPoissonGpuCuFFT3d --nx 64 --ny 64 --nz 64 --method 1 --test 2
 *
 * \author Pierre Kestener
 * \date April 9, 2015
 */

#include <math.h>
#include <sys/time.h> // for gettimeofday
#include <stdlib.h>

#include <GetPot.h> // for command line arguments
#include "cnpy/cnpy.h"   // for data IO in numpy file format


#define SQR(x) ((x)*(x))

#include <cuda_runtime.h>
#include <cufftw.h>

// cufft wrapper for single / double precision
#if defined(USE_FLOAT)
typedef float FFTW_REAL;
typedef fftwf_complex FFTW_COMPLEX;
typedef fftwf_plan    FFTW_PLAN;
#define FFTW_PLAN_DFT_R2C_2D         fftwf_plan_dft_r2c_2d
#define FFTW_PLAN_DFT_C2R_2D         fftwf_plan_dft_c2r_2d
#define FFTW_PLAN_DFT_R2C_3D         fftwf_plan_dft_r2c_3d
#define FFTW_PLAN_DFT_C2R_3D         fftwf_plan_dft_c2r_3d
#define FFTW_PLAN_DFT_3D             fftwf_plan_dft_3d
#define FFTW_DESTROY_PLAN            fftwf_destroy_plan
#define FFTW_EXECUTE                 fftwf_execute
#define FFTW_CLEANUP                 fftwf_cleanup
#define FFTW_PRINT_PLAN              fftwf_print_plan
#define FFTW_FLOPS                   fftwf_flops
#define FFTW_FREE                    fftwf_free
#else
typedef double FFTW_REAL;
typedef fftw_complex FFTW_COMPLEX;
typedef fftw_plan    FFTW_PLAN;
#define FFTW_PLAN_DFT_R2C_2D         fftw_plan_dft_r2c_2d
#define FFTW_PLAN_DFT_C2R_2D         fftw_plan_dft_c2r_2d
#define FFTW_PLAN_DFT_R2C_3D         fftw_plan_dft_r2c_3d
#define FFTW_PLAN_DFT_C2R_3D         fftw_plan_dft_c2r_3d
#define FFTW_PLAN_DFT_3D             fftw_plan_dft_3d
#define FFTW_DESTROY_PLAN            fftw_destroy_plan
#define FFTW_EXECUTE                 fftw_execute
#define FFTW_CLEANUP                 fftw_cleanup
#define FFTW_PRINT_PLAN              fftw_print_plan
#define FFTW_FLOPS                   fftw_flops
#define FFTW_FREE                    fftw_free
#endif /* USE_FLOAT */


enum {
  TEST_CASE_SIN=0,
  TEST_CASE_GAUSSIAN=1,
  TEST_CASE_UNIFORM_BALL=2
};

/////////////////////////////////////////////////
uint blocksFor(uint elementCount, uint threadCount)
{
  return (elementCount + threadCount - 1) / threadCount;
}


/**
 * apply Poisson kernel to complex Fourier coefficient in input.
 *
 * The results represent the Fourier coefficients of the gravitational potential.
 *
 * Take care that we swapped dimensions, in order to sweep array in row-major format.
 */
#define POISSON_3D_DIMX 32
#define POISSON_3D_DIMY 16

__global__ 
void kernel_poisson_3d(FFTW_COMPLEX *phi_fft, 
		       int nx   , int ny   , int nz,
		       double dx, double dy, double dz, 
		       int methodNb)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int kx = __mul24(bx, POISSON_3D_DIMX) + tx;
  const int ky = __mul24(by, POISSON_3D_DIMY) + ty;


  for (int kz=0; kz<nz; kz++) {
    
    // centered frequency
    int kx_c = kx;
    int ky_c = ky;
    int kz_c = kz;
    
    int nxo2p1 = nx/2+1;
    
    if (kx>nx/2)
      kx_c = kx - nx;
    if (ky>ny/2)
      ky_c = ky - ny;
    if (kz>nz/2)
      kz_c = kz - nz;

    FFTW_REAL scaleFactor=0.0;
    
    if (methodNb == 0) {
      /*
       * method 0 (from Numerical recipes)
       */
      
      scaleFactor=2*( 
		     (cos(1.0*2*M_PI*kx/nx) - 1)/(dx*dx) + 
		     (cos(1.0*2*M_PI*ky/ny) - 1)/(dy*dy) + 
		     (cos(1.0*2*M_PI*kz/nz) - 1)/(dz*dz) )*(nx*ny*nz); 
      
    } else if (methodNb == 1) {
      /*
       * method 1 (just from Continuous Fourier transform of Poisson equation)
       */
      scaleFactor=-4*M_PI*M_PI*(kx_c*kx_c + ky_c*ky_c + kz_c*kz_c)*nx*ny*nz;
    }
    
    // write result
    if (kx < nxo2p1 and ky < ny) {
      
      if (kx!=0 or ky!=0 or kz!=0) {
	
    	phi_fft[kx + nxo2p1*ky + nxo2p1*ny*kz][0] /= scaleFactor;
    	phi_fft[kx + nxo2p1*ky + nxo2p1*ny*kz][1] /= scaleFactor;
	
      } else { // enforce mean value is zero
	
    	phi_fft[kx + nxo2p1*ky + nxo2p1*ny*kz][0] = 0.0;
    	phi_fft[kx + nxo2p1*ky + nxo2p1*ny*kz][1] = 0.0;
	
      }
  
    }

  } // end for kz
  
} // kernel_poisson_3D


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
int main(int argc, char **argv)
{
  
  /* parse command line arguments */
  GetPot cl(argc, argv);

  const ptrdiff_t NX = cl.follow(100,    "--nx");
  const ptrdiff_t NY = cl.follow(100,    "--ny");
  const ptrdiff_t NZ = cl.follow(100,    "--nz");
  
  int NZ2 = 2*(NZ/2+1);
  
  //int NZo2p1 = NZ/2+1;

  // method (variant of FFT-based Poisson solver) : 0 or 1
  const int methodNb   = cl.follow(0, "--method");

  // test case number
  const int testCaseNb = cl.follow(TEST_CASE_SIN, "--test");
  std::cout << "Using test case number : " << testCaseNb << std::endl;

  // time measurement
  //struct timeval tv_start, tv_stop;
  //double deltaT;
  //int N_ITER;

  /*
   * test 3D FFT using in-place transform
   */
  // cpu variables
  FFTW_REAL  *rho      = (FFTW_REAL *) malloc(NX*NY*NZ2*sizeof(FFTW_REAL));
  FFTW_REAL  *solution = (FFTW_REAL *) malloc(NX*NY*NZ2*sizeof(FFTW_REAL));

  // gpu variables
  FFTW_REAL *d_rho;
  cudaMalloc((void**) &d_rho, NX*NY*NZ2*sizeof(FFTW_REAL));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate \"d_rho\"\n");
    return 0;
  }
  FFTW_COMPLEX *d_rhoComplex = (FFTW_COMPLEX *) d_rho;
  
  // CUFFT plan
  // FFTW_PLAN plan_rho_forward;
  // cufftPlan3d(&plan_rho_forward, NX, NY, CUFFT_R2C);

  // FFTW_PLAN plan_rho_backward;
  // cufftPlan3d(&plan_rho_backward, NX, NY, CUFFT_C2R);
  
  FFTW_PLAN plan_rho_forward  = FFTW_PLAN_DFT_R2C_3D(NX, NY, NZ,
						     d_rho, d_rhoComplex,
                                                     FFTW_ESTIMATE);
  FFTW_PLAN plan_rho_backward = FFTW_PLAN_DFT_C2R_3D(NX, NY, NZ,
						     d_rhoComplex, d_rho,
                                                     FFTW_ESTIMATE);

  double Lx=1.0;
  double Ly=1.0;
  double Lz=1.0;

  double dx = Lx/NX;
  double dy = Ly/NY;
  double dz = Lz/NZ;

  double alpha = cl.follow(30.0, "--alpha");
  double x0    = 0.5;
  double y0    = 0.5;
  double z0    = 0.5;

  /*
   * (CPU) initialize rho to some function my_function(x,y,z)
   */
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {


	if (testCaseNb==TEST_CASE_SIN) {
	  
	  double x = 1.0*i/NX;
	  double y = 1.0*j/NY;
	  double z = 1.0*k/NZ;

	  rho[i*NY*NZ2 + j*NZ2 + k] =  sin(2*M_PI*x) * sin(2*M_PI*y) * sin(2*M_PI*z);
	  
	} else if (testCaseNb==TEST_CASE_GAUSSIAN) {

	  double x = 1.0*i/NX - x0;
	  double y = 1.0*j/NY - y0;
	  double z = 1.0*k/NZ - z0;

	  rho[i*NY*NZ2 + j*NZ2 + k] = (4*alpha*alpha*(x*x+y*y+z*z)-6*alpha)*exp(-alpha*(x*x+y*y+z*z));
	  
	} else if (testCaseNb==TEST_CASE_UNIFORM_BALL) {
	  
	  double x = 1.0*i/NX;
	  double y = 1.0*j/NY;
	  double z = 1.0*k/NZ;

	  // uniform ball function center
	  double xC = cl.follow((double) 0.5, "--xC");
	  double yC = cl.follow((double) 0.5, "--yC");
	  double zC = cl.follow((double) 0.5, "--zC");
	  
	  // uniform ball radius
	  double R = cl.follow(0.1, "--radius");
	  
	  double r = sqrt( (x-xC)*(x-xC) + (y-yC)*(y-yC) + (z-zC)*(z-zC) );
	  
	  if ( r < R )
	    rho[i*NY*NZ2 + j*NZ2 + k] = 1.0;
	  else
	    rho[i*NY*NZ2 + j*NZ2 + k] = 0.0;
	}
	
      } // end for k
    } // end for j
  } // end for i

  // save RHS rho
  {
    const unsigned int shape[] = {(unsigned int) NX, 
				  (unsigned int) NY, 
				  (unsigned int) NZ2};
    cnpy::npy_save("rho.npy",rho,shape,3,"w");
  }

  // copy rho onto gpu
  cudaMemcpy(d_rho, rho, sizeof(FFTW_REAL)*NX*NY*NZ2, cudaMemcpyHostToDevice);

  // compute FFT(rho)
  FFTW_EXECUTE(plan_rho_forward);
  
  // compute Fourier coefficient of phi
  FFTW_COMPLEX *d_phiComplex = d_rhoComplex;

  // (GPU) apply poisson kernel 
  {
    
    // swap dimension (row-major to column-major order)
    int nx = NZ;
    int ny = NY;
    int nz = NX;

    double dx_c = dz;
    double dy_c = dy;
    double dz_c = dx;

    dim3 dimBlock(POISSON_3D_DIMX,
		  POISSON_3D_DIMY);
    dim3 dimGrid(blocksFor(nx/2+1, POISSON_3D_DIMX),
		 blocksFor(ny    , POISSON_3D_DIMY));

    kernel_poisson_3d<<<dimGrid, dimBlock>>>(d_phiComplex, 
					     nx  , ny  , nz,
					     dx_c, dy_c, dz_c,
					     methodNb);

  }
  
  // compute FFT^{-1} (phiComplex) to retrieve solution
  FFTW_EXECUTE(plan_rho_backward);

  // retrieve gpu computation
  cudaMemcpy(rho, d_rho, sizeof(FFTW_REAL)*NX*NY*NZ2, cudaMemcpyDeviceToHost);

  
  if (testCaseNb==TEST_CASE_GAUSSIAN) {
    // compute max value, add offset to match analytical solution at the
    // location of max value

    double maxVal = rho[0];
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	for (int k = 0; k < NZ; ++k) {
	  if (rho[i*NY*NZ2 + j*NZ2 + k] > maxVal)
	    maxVal = rho[i*NY*NZ2 + j*NZ2 + k];
	}
      }
    }
    
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	for (int k = 0; k < NZ; ++k) {
	  rho[i*NY*NZ2 + j*NZ2 + k] += 1-maxVal;
	}
      }
    }
    
  } // end TEST_CASE_GAUSSIAN

  if (testCaseNb==TEST_CASE_UNIFORM_BALL) {
    // compute min value

    double minVal = rho[0];
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	for (int k = 0; k < NZ; ++k) {
	  if (rho[i*NY*NZ2 + j*NZ2 + k] < minVal)
	    minVal = rho[i*NY*NZ2 + j*NZ2 + k];
	}
      }
    }
    
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	for (int k = 0; k < NZ; ++k) {
	  rho[i*NY*NZ2 + j*NZ2 + k] -= minVal;
	}
      }
    }
    
  } // end TEST_CASE_UNIFORM_BALL


  // save numerical solution
  {
    const unsigned int shape[] = {(unsigned int) NX, 
				  (unsigned int) NY,
				  (unsigned int) NZ2};
    cnpy::npy_save("phi.npy",rho,shape,3,"w");
  }

  /*
   * compare with "exact" solution
   */
  // compute sum of square ( phi - solution) / sum of square (solution)
  {
    // uniform ball function center
    double xC = cl.follow((double) 0.5, "--xC");
    double yC = cl.follow((double) 0.5, "--yC");
    double zC = cl.follow((double) 0.5, "--zC");

    // uniform ball radius
    double R = cl.follow(0.1, "--radius");

    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	for (int k = 0; k < NZ; ++k) {

	  double sol=0.0;
	  
	  if (testCaseNb==TEST_CASE_SIN) {
	    
	    double x = 1.0*i/NX;
	    double y = 1.0*j/NY;
	    double z = 1.0*k/NZ;

	    sol =  - 
	      sin(2*M_PI*x) * 
	      sin(2*M_PI*y) * 
	      sin(2*M_PI*z) / 
	      ( (4*M_PI*M_PI)*(1.0/Lx/Lx + 1.0/Ly/Ly + 1.0/Lz/Lz) );
	    
	  } else if (testCaseNb==TEST_CASE_GAUSSIAN) {
	    
	    double x = 1.0*i/NX - x0;
	    double y = 1.0*j/NY - y0;
	    double z = 1.0*k/NZ - z0;
	    sol = exp(-alpha*(x*x+y*y+z*z));
	    
	  } else if (testCaseNb==TEST_CASE_UNIFORM_BALL) {
	    
	    double x = 1.0*i/NX;
	    double y = 1.0*j/NY;
	    double z = 1.0*k/NZ;
	    
	    double r = sqrt( (x-xC)*(x-xC) + (y-yC)*(y-yC) + (z-zC)*(z-zC) );
	    
	    if ( r < R ) {
	      sol = r*r/6.0;
	    } else {
	      sol = -R*R*R/(3*r)+R*R/2.0;
	    }
	  } /* end testCase */

	  solution[i*NY*NZ2 + j*NZ2 + k] = sol;
	  
	} // end for k
      } // end for j
    } // end for i
    
    if (testCaseNb==TEST_CASE_UNIFORM_BALL) {
      // compute min value of solution
      double minVal = solution[0];
      for (int i = 0; i < NX; ++i) {
	for (int j = 0; j < NY; ++j) {
	  for (int k = 0; k < NZ; ++k) {
	    if (solution[i*NY*NZ2 + j*NZ2 + k] < minVal)
	      minVal = solution[i*NY*NZ2 + j*NZ2 + k];
	  }
	}
      }

      for (int i = 0; i < NX; ++i) {
	for (int j = 0; j < NY; ++j) {
	  for (int k = 0; k < NZ; ++k) {
	    solution[i*NY*NZ2 + j*NZ2 + k] -= minVal;
	  }
	}
      }
      
    } // end TEST_CASE_UNIFORM_BALL


    // compute L2 difference between FFT-based solution (phi) and 
    // expected analytical solution
    double L2_diff = 0.0;
    double L2_rho  = 0.0;
    
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	for (int k = 0; k < NZ; ++k) {

	  double sol = solution[i*NY*NZ2 + j*NZ2 + k];

	  L2_rho += sol*sol;

	  // rho now contains error
	  rho[i*NY*NZ2 + j*NZ2 + k] -=  sol;
	  L2_diff += rho[i*NY*NZ2 + j*NZ2 + k] * rho[i*NY*NZ2 + j*NZ2 + k];

	} // end for k
      } // end for j
    } // end for i

    std::cout << "L2 relative error between phi and exact solution : " 
	      <<  L2_diff/L2_rho << std::endl;

    // save error array
    {
      const unsigned int shape[] = {(unsigned int) NX, 
				    (unsigned int) NY,
				    (unsigned int) NZ2};
      cnpy::npy_save("error.npy",rho,shape,3,"w");
    }

    // save analytical solution
    {
      const unsigned int shape[] = {(unsigned int) NX, 
				    (unsigned int) NY,
				    (unsigned int) NZ2};
      cnpy::npy_save("solution.npy",solution,shape,3,"w");
    }

  }

  cudaFree(d_rho);

  free(rho);
  free(solution);

  return 0;
  
 } // end main
