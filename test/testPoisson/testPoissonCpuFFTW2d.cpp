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
 * ./testPoissonCpuFFTW2d --nx 64 --ny 64 --method 1 --test 2
 *
 * \author Pierre Kestener
 * \date July 3, 2014
 */

#include <math.h>
#include <sys/time.h> // for gettimeofday
#include <stdlib.h>

#include <GetPot.h> // for command line arguments
#include "cnpy/cnpy.h"   // for data IO in numpy file format


#define SQR(x) ((x)*(x))

#include <fftw3.h>
// fftw wrapper for single / double precision
#if defined(USE_FLOAT)
typedef float FFTW_REAL;
typedef fftwf_complex FFTW_COMPLEX;
typedef fftwf_plan    FFTW_PLAN;
#define FFTW_WISDOM_FILENAME         ("fftwf_wisdom.txt")
#define FFTW_PLAN_DFT_R2C_2D         fftwf_plan_dft_r2c_2d
#define FFTW_PLAN_DFT_C2R_2D         fftwf_plan_dft_c2r_2d
#define FFTW_PLAN_DFT_R2C_3D         fftwf_plan_dft_r2c_3d
#define FFTW_PLAN_DFT_C2R_3D         fftwf_plan_dft_c2r_3d
#define FFTW_PLAN_DFT_3D             fftwf_plan_dft_3d
#define FFTW_DESTROY_PLAN            fftwf_destroy_plan
#define FFTW_EXECUTE                 fftwf_execute
#define FFTW_EXPORT_WISDOM_TO_FILE   fftwf_export_wisdom_to_file
#define FFTW_IMPORT_WISDOM_FROM_FILE fftwf_import_wisdom_from_file
#define FFTW_PLAN_WITH_NTHREADS      fftwf_plan_with_nthreads
#define FFTW_INIT_THREADS            fftwf_init_threads
#define FFTW_CLEANUP                 fftwf_cleanup
#define FFTW_CLEANUP_THREADS         fftwf_cleanup_threads
#define FFTW_PRINT_PLAN              fftwf_print_plan
#define FFTW_FLOPS                   fftwf_flops
#define FFTW_MALLOC                  fftwf_malloc
#define FFTW_FREE                    fftwf_free
#else
typedef double FFTW_REAL;
typedef fftw_complex FFTW_COMPLEX;
typedef fftw_plan    FFTW_PLAN;
#define FFTW_WISDOM_FILENAME         ("fftw_wisdom.txt")
#define FFTW_PLAN_DFT_R2C_2D         fftw_plan_dft_r2c_2d
#define FFTW_PLAN_DFT_C2R_2D         fftw_plan_dft_c2r_2d
#define FFTW_PLAN_DFT_R2C_3D         fftw_plan_dft_r2c_3d
#define FFTW_PLAN_DFT_C2R_3D         fftw_plan_dft_c2r_3d
#define FFTW_PLAN_DFT_3D             fftw_plan_dft_3d
#define FFTW_DESTROY_PLAN            fftw_destroy_plan
#define FFTW_EXECUTE                 fftw_execute
#define FFTW_EXPORT_WISDOM_TO_FILE   fftw_export_wisdom_to_file
#define FFTW_IMPORT_WISDOM_FROM_FILE fftw_import_wisdom_from_file
#define FFTW_PLAN_WITH_NTHREADS      fftw_plan_with_nthreads
#define FFTW_INIT_THREADS            fftw_init_threads
#define FFTW_CLEANUP                 fftw_cleanup
#define FFTW_CLEANUP_THREADS         fftw_cleanup_threads
#define FFTW_PRINT_PLAN              fftw_print_plan
#define FFTW_FLOPS                   fftw_flops
#define FFTW_MALLOC                  fftw_malloc
#define FFTW_FREE                    fftw_free
#endif


#ifdef NO_FFTW_EXHAUSTIVE
#define MY_FFTW_FLAGS (FFTW_ESTIMATE)
#else
#define MY_FFTW_FLAGS (FFTW_EXHAUSTIVE)
#endif // NO_FFTW_EXHAUSTIVE

// FFTW_PATIENT, FFTW_MEASURE, FFTW_ESTIMATE
#define O_METHOD  FFTW_ESTIMATE
#define O_METHOD_STR "FFTW_ESTIMATE"

enum {
  TEST_CASE_SIN=0,
  TEST_CASE_GAUSSIAN=1,
  TEST_CASE_UNIFORM_DISK=2
};


int main(int argc, char **argv)
{
  
  /* parse command line arguments */
  GetPot cl(argc, argv);

  const ptrdiff_t NX = cl.follow(100,    "--nx");
  const ptrdiff_t NY = cl.follow(100,    "--ny");
  //const ptrdiff_t NZ = cl.follow(100,    "--nz");
  
  int NY2 = 2*(NY/2+1);
  //int NZ2 = 2*(NZ/2+1);
  
  int NYo2p1 = NY/2+1;

  // method (variant of FFT-based Poisson solver) : 0 or 1
  const int methodNb   = cl.follow(0, "--method");

  // test case number
  const int testCaseNb = cl.follow(TEST_CASE_SIN, "--test");
  std::cout << "Using test case number : " << testCaseNb << std::endl;

  // time measurement
  //struct timeval tv_start, tv_stop;
  //double deltaT;
  //int N_ITER;

  // test in 2D using in-place transform
  FFTW_REAL  *rho      = (FFTW_REAL *) malloc(NX*NY2*sizeof(FFTW_REAL));
  FFTW_REAL  *solution = (FFTW_REAL *) malloc(NX*NY2*sizeof(FFTW_REAL));

  FFTW_COMPLEX *rhoComplex = (FFTW_COMPLEX *) rho;

  FFTW_PLAN plan_rho_forward  = FFTW_PLAN_DFT_R2C_2D(NX, NY, rho, rhoComplex,
						     FFTW_ESTIMATE);
  FFTW_PLAN plan_rho_backward = FFTW_PLAN_DFT_C2R_2D(NX, NY, rhoComplex, rho,
						     FFTW_ESTIMATE);

  double Lx=1.0;
  double Ly=1.0;

  double dx = Lx/NX;
  double dy = Ly/NY;

  double alpha = cl.follow(30.0, "--alpha");
  double x0    = 0.5;
  double y0    = 0.5;

  /*
   * initialize rho to some function my_function(x,y)
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

  // compute FFT(rho)
  FFTW_EXECUTE(plan_rho_forward);
  
  // compute Fourier coefficient of phi
  FFTW_COMPLEX *phiComplex = rhoComplex;
  
  for (int kx=0; kx < NX; kx++)
    for (int ky=0; ky < NYo2p1; ky++) {
      
      double kkx = (double) kx;
      double kky = (double) ky;

      if (kx>NX/2)
	kkx -= NX;
      if (ky>NY/2)
	kky -= NY;
      
      // note that factor NX*NY is used here because FFTW fourier coefficients
      // are not scaled
      //FFTW_REAL scaleFactor=2*( cos(2*M_PI*kx/NX) + cos(2*M_PI*ky/NY) - 2)*(NX*NY);

      FFTW_REAL scaleFactor=0.0;
 
      if (methodNb == 0) {
	/*
	 * method 0 (from Numerical recipes)
	 */
	
	scaleFactor=2*( 
		 (cos(1.0*2*M_PI*kx/NX) - 1)/(dx*dx) + 
		 (cos(1.0*2*M_PI*ky/NY) - 1)/(dy*dy) )*(NX*NY); 
	
      } else if (methodNb==1) {
	/*
	 * method 1 (just from Continuous Fourier transform of Poisson equation)
	 */
	scaleFactor=-4*M_PI*M_PI*(kkx*kkx + kky*kky)*NX*NY;
      }


      if (kx!=0 or ky!=0) {
	phiComplex[kx*NYo2p1+ky][0] /= scaleFactor;
	phiComplex[kx*NYo2p1+ky][1] /= scaleFactor;
      } else { // enforce mean value is zero
	phiComplex[kx*NYo2p1+ky][0] = 0.0;
	phiComplex[kx*NYo2p1+ky][1] = 0.0;
      }
    }
  
  // compute FFT^{-1} (phiComplex) to retrieve solution
  FFTW_EXECUTE(plan_rho_backward);
  
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
    
  } // TEST_CASE_GAUSSIAN

  if (testCaseNb==TEST_CASE_UNIFORM_DISK) {
    // compute min value

    double minVal = rho[0];
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	if (rho[i*NY2 + j] < minVal)
	  minVal = rho[i*NY2 + j];
      }
    }
    
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
	rho[i*NY2 + j] -= minVal;
      }
    }
    
  } // TEST_CASE_UNIFORM_DISK


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

	solution[i*NY2+j] = sol;

      } // end for j
    } // end for i

    if (testCaseNb==TEST_CASE_UNIFORM_DISK) {
      // compute min value of solution
      double minVal = solution[0];
      for (int i = 0; i < NX; ++i) {
	for (int j = 0; j < NY; ++j) {
	  if (solution[i*NY2+j] < minVal)
	    minVal = solution[i*NY2+j];
	} // end for j
      } // end for i

      for (int i = 0; i < NX; ++i) {
	for (int j = 0; j < NY; ++j) {
	  solution[i*NY2+j] -= minVal;
	} // end for j
      } // end for i

    } // end TEST_CASE_UNIFORM_DISK

    // compute L2 difference between FFT-based solution (phi) and 
    // expected analytical solution
    double L2_diff = 0.0;
    double L2_rho  = 0.0;

    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {

	double sol = solution[i*NY2+j];

	L2_rho += sol*sol;

	// rho now contains error
	rho[i*NY2 + j] -=  sol;
	L2_diff += rho[i*NY2 + j] * rho[i*NY2 + j];

      } // end for j
    } // end for i

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

  free(rho);
  free(solution);
  
 } // end main
