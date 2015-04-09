/**
 * A very simple test for solving Laplacian(\phi) = rho using FFT in 2D.
 *
 * Laplacian operator can be considered as a low-pass filter.
 * Here we implement 2 types of filters :
 *
 * method 0 : see Numerical recipes in C, section 19.4
 * method 1 : just divide right hand side by -(kx^2+ky^2) in Fourier
 *
 * Test case 0:  rho(x,y) = sin(2*pi*x/Lx)*sin(2*pi*y/Ly)
 * Test case 1:  rho(x,y) = 4*alpha*(alpha*(x^2+y^2)-1)*exp(-alpha*(x^2+y^2))
 * Test case 2:  rho(x,y) = ( r=sqrt(x^2+y^2) < R ) ? 1 : 0 
 *
 * Example of use:
 * ./testPoissonGpuCuFFT2d --nx 64 --ny 64 --method 1 --test 2
 *
 * \author Pierre Kestener
 * \date April 8, 2015
 */

#include <math.h>
#include <sys/time.h> // for gettimeofday
#include <stdlib.h>

#include <GetPot.h> // for command line arguments
#include <cnpy.h>   // for data IO in numpy file format


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
  TEST_CASE_UNIFORM_DISK=2
};

/////////////////////////////////////////////////
uint blocksFor(uint elementCount, uint threadCount)
{
  return (elementCount + threadCount - 1) / threadCount;
}


/**
 * apply Poisson kernel to complex Fourier coefficient in input.
 *
 * The results represent the Fourier coefficients of the 
 * gravitational potential.
 *
 * Take care that dimensions are swapped, so that we have column-major order
 * inside kernel.
 */
#define POISSON_2D_DIMX 32
#define POISSON_2D_DIMY 16

__global__ 
void kernel_poisson_2d(FFTW_COMPLEX *phi_fft, int nx, int ny, 
		       double dx, double dy,
		       int methodNb)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int kx = __mul24(bx, POISSON_2D_DIMX) + tx;
  const int ky = __mul24(by, POISSON_2D_DIMY) + ty;

  // centered frequency
  int kx_c = kx;
  int ky_c = ky;

  int nxo2p1 = nx/2+1;

  if (kx>nx/2)
    kx_c = kx - nx;
  if (ky>ny/2)
    ky_c = ky - ny;

  // note that factor nx*ny is used here because FFTW fourier coefficients
  // are not scaled
  //FFTW_REAL scaleFactor=2*( cos(2*M_PI*kx/nx) + cos(2*M_PI*ky/ny) - 2)*(nx*ny);
  
  FFTW_REAL scaleFactor=0.0;
  
  if (methodNb == 0) {
    /*
     * method 0 (from Numerical recipes)
     */
    
    scaleFactor=2*( 
		   (cos(1.0*2*M_PI*kx/nx) - 1)/(dx*dx) + 
		   (cos(1.0*2*M_PI*ky/ny) - 1)/(dy*dy) )*(nx*ny); 
    
  } else if (methodNb==1) {
    /*
     * method 1 (just from Continuous Fourier transform of Poisson equation)
     */
    scaleFactor=-4*M_PI*M_PI*(kx_c*kx_c + ky_c*ky_c)*nx*ny;
  }


  // write result
  if (kx < nxo2p1 and ky < ny) {

    if (kx!=0 or ky!=0) {

      phi_fft[kx+nxo2p1*ky][0] /= scaleFactor;
      phi_fft[kx+nxo2p1*ky][1] /= scaleFactor;

    } else { // enforce mean value is zero

      phi_fft[kx+nxo2p1*ky][0] = 0.0;
      phi_fft[kx+nxo2p1*ky][1] = 0.0;

    }
  
  }
  
} // kernel_poisson_2D


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
int main(int argc, char **argv)
{
  
  /* parse command line arguments */
  GetPot cl(argc, argv);

  const ptrdiff_t NX = cl.follow(100,    "--nx");
  const ptrdiff_t NY = cl.follow(100,    "--ny");
  //const ptrdiff_t NZ = cl.follow(100,    "--nz");
  
  int NY2 = 2*(NY/2+1);
  //int NZ2 = 2*(NZ/2+1);
  
  //int NYo2p1 = NY/2+1;

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
   * test 2D FFT using in-place transform
   */
  // cpu variables
  FFTW_REAL  *rho      = (FFTW_REAL *) malloc(NX*NY2*sizeof(FFTW_REAL));
  FFTW_REAL  *solution = (FFTW_REAL *) malloc(NX*NY2*sizeof(FFTW_REAL));

  // gpu variables
  FFTW_REAL *d_rho;
  cudaMalloc((void**) &d_rho, NX*NY2*sizeof(FFTW_REAL));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate \"d_rho\"\n");
    return 0;
  }
  FFTW_COMPLEX *d_rhoComplex = (FFTW_COMPLEX *) d_rho;
  
  // CUFFT plan
  // FFTW_PLAN plan_rho_forward;
  // cufftPlan2d(&plan_rho_forward, NX, NY, CUFFT_R2C);

  // FFTW_PLAN plan_rho_backward;
  // cufftPlan2d(&plan_rho_backward, NX, NY, CUFFT_C2R);
  
  FFTW_PLAN plan_rho_forward  = FFTW_PLAN_DFT_R2C_2D(NX, NY, 
						     d_rho, d_rhoComplex,
                                                     FFTW_ESTIMATE);
  FFTW_PLAN plan_rho_backward = FFTW_PLAN_DFT_C2R_2D(NX, NY, 
						     d_rhoComplex, d_rho,
                                                     FFTW_ESTIMATE);

  double Lx=1.0;
  double Ly=1.0;

  double dx = Lx/NX;
  double dy = Ly/NY;

  double alpha = cl.follow(30.0, "--alpha");
  double x0    = 0.5;
  double y0    = 0.5;

  /*
   * (CPU) initialize rho to some function my_function(x,y)
   */
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      double x = 1.0*i/NX - x0;
      double y = 1.0*j/NY - y0;

      if (testCaseNb==TEST_CASE_SIN) {

	rho[i*NY2 + j] =  sin(2*M_PI*x) * sin(2*M_PI*y);

      } else if (testCaseNb==TEST_CASE_GAUSSIAN) {

	rho[i*NY2 + j] = 4*alpha*(alpha*(x*x+y*y)-1)*exp(-alpha*(x*x+y*y));

      } else if (testCaseNb==TEST_CASE_UNIFORM_DISK) {

	// uniform disk function center
	double xC = cl.follow((double) 0.0, "--xC");
	double yC = cl.follow((double) 0.0, "--yC");

	// uniform disk radius
	double R = cl.follow(0.02, "--radius");

	double r = sqrt( (x-xC)*(x-xC) + (y-yC)*(y-yC) );

	if ( r < R )
	  rho[i*NY2 + j] = 1.0;
	else
	  rho[i*NY2 + j] = 0.0;
      }

    } // end for j
  } // end for i

  // save RHS rho
  {
    const unsigned int shape[] = {(unsigned int) NX, (unsigned int) NY2};
    cnpy::npy_save("rho.npy",rho,shape,2,"w");
  }

  // copy rho onto gpu
  cudaMemcpy(d_rho, rho, sizeof(FFTW_REAL)*NX*NY2, cudaMemcpyHostToDevice);

  // compute FFT(rho)
  FFTW_EXECUTE(plan_rho_forward);
  
  // compute Fourier coefficient of phi
  FFTW_COMPLEX *d_phiComplex = d_rhoComplex;

  // (GPU) apply poisson kernel 
  {

    // swap dimension (row-major to column-major order)
    int nx = NY;
    int ny = NX;
    
    double dx_c = dy;
    double dy_c = dx;

    dim3 dimBlock(POISSON_2D_DIMX,
		  POISSON_2D_DIMY);
    dim3 dimGrid(blocksFor(nx/2+1, POISSON_2D_DIMX),
		 blocksFor(ny    , POISSON_2D_DIMY));
    kernel_poisson_2d<<<dimGrid, dimBlock>>>(d_phiComplex, 
					     nx  , ny  , 
					     dx_c, dy_c,
					     methodNb);

  }
  
  // compute FFT^{-1} (phiComplex) to retrieve solution
  FFTW_EXECUTE(plan_rho_backward);

  // retrieve gpu computation
  cudaMemcpy(rho, d_rho, sizeof(FFTW_REAL)*NX*NY2, cudaMemcpyDeviceToHost);

  
  if (testCaseNb==TEST_CASE_GAUSSIAN) {
    // compute max value, add offset to match analytical solution at the
    // location of max value

    double maxVal = rho[0];
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	if (rho[i*NY2 + j] > maxVal)
	  maxVal = rho[i*NY2 + j];
      }
    }
    
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	rho[i*NY2 + j] += 1-maxVal;
      }
    }
    
  }


  // save numerical solution
  {
    const unsigned int shape[] = {(unsigned int) NX, (unsigned int) NY2};
    cnpy::npy_save("phi.npy",rho,shape,2,"w");
  }

  /*
   * compare with "exact" solution
   */
  // compute sum of square ( phi - solution) / sum of square (solution)
  {
    double L2_diff = 0.0;
    double L2_rho  = 0.0;

    // uniform disk function center
    double xC = cl.follow((double) 0.0, "--xC");
    double yC = cl.follow((double) 0.0, "--yC");

    // uniform disk radius
    double R = cl.follow(0.02, "--radius");

    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {

	double sol=0.0;
	if (testCaseNb==TEST_CASE_SIN) {

	  sol =  - sin(2*M_PI*i/NX) * sin(2*M_PI*j/NY) / ( (4*M_PI*M_PI)*(1.0/Lx/Lx + 1.0/Ly/Ly) );
	
	} else if (testCaseNb==TEST_CASE_GAUSSIAN) {

	  double x = 1.0*i/NX - x0;
	  double y = 1.0*j/NY - y0;
	  sol = exp(-alpha*(x*x+y*y));

	} else if (testCaseNb==TEST_CASE_UNIFORM_DISK) {

	  double x = 1.0*i/NX - x0;
	  double y = 1.0*j/NY - y0;

	  double r = sqrt( (x-xC)*(x-xC) + (y-yC)*(y-yC) );
	  
	  if ( r < R ) {
	    sol = r*r/4.0;
	  } else {
	    sol = R*R/2.0*log(r/R)+R*R/4.0;
	  }
	} /* end testCase */

	// compute L2 difference between FFT-based solution (phi) and 
	// expected analytical solution
	L2_rho += sol*sol;
	rho[i*NY2 + j] -=  sol;
	L2_diff += rho[i*NY2 + j] * rho[i*NY2 + j];

	solution[i*NY2+j] = sol;
      }
    }

    std::cout << "L2 error between phi and exact solution : " 
	      <<  L2_diff/L2_rho << std::endl;

    // save error array
    {
      const unsigned int shape[] = {(unsigned int) NX, (unsigned int) NY2};
      cnpy::npy_save("error.npy",rho,shape,2,"w");
    }

    // save analytical solution
    {
      const unsigned int shape[] = {(unsigned int) NX, (unsigned int) NY2};
      cnpy::npy_save("solution.npy",solution,shape,2,"w");
    }

  }

  cudaFree(d_rho);

  free(rho);
  free(solution);

  return 0;
  
 } // end main
