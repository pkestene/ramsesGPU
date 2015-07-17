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
 * ./testPoissonCpuFFTW3d --nx 64 --ny 64 --nz 64 --method 1 --test 2

 * \author Pierre Kestener
 * \date April 6, 2015
 */

#include <math.h>
#include <sys/time.h> // for gettimeofday
#include <stdlib.h>

#include <GetPot.h> // for command line arguments
#include <cnpy.h>   // for data IO in numpy file format


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
#endif /* USE_FLOAT */


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
  TEST_CASE_UNIFORM_BALL=2
};


int main(int argc, char **argv)
{
  
  /* parse command line arguments */
  GetPot cl(argc, argv);

  const ptrdiff_t NX = cl.follow(100,    "--nx");
  const ptrdiff_t NY = cl.follow(100,    "--ny");
  const ptrdiff_t NZ = cl.follow(100,    "--nz");
  
  int NZ2 = 2*(NZ/2+1);
  
  int NZo2p1 = NZ/2+1;

  // method (variant of FFT-based Poisson solver) : 0 or 1
  const int methodNb   = cl.follow(0, "--method");

  // test case number
  const int testCaseNb = cl.follow(TEST_CASE_SIN, "--test");
  std::cout << "Using test case number : " << testCaseNb << std::endl;

  // time measurement
  //struct timeval tv_start, tv_stop;
  //double deltaT;
  //int N_ITER;

  // test in 3D using in-place transform
  FFTW_REAL  *rho      = (FFTW_REAL *) malloc(NX*NY*NZ2*sizeof(FFTW_REAL));
  FFTW_REAL  *solution = (FFTW_REAL *) malloc(NX*NY*NZ2*sizeof(FFTW_REAL));

  FFTW_COMPLEX *rhoComplex = (FFTW_COMPLEX *) rho;

  FFTW_PLAN plan_rho_forward  = FFTW_PLAN_DFT_R2C_3D(NX, NY, NZ, 
						     rho, rhoComplex,
						     FFTW_ESTIMATE);
  FFTW_PLAN plan_rho_backward = FFTW_PLAN_DFT_C2R_3D(NX, NY, NZ, 
						     rhoComplex, rho,
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
   * initialize rho to some function my_function(x,y,z)
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

  // compute FFT(rho)
  FFTW_EXECUTE(plan_rho_forward);
  
  // compute Fourier coefficient of phi
  FFTW_COMPLEX *phiComplex = rhoComplex;
  
  for (int kx=0; kx < NX; kx++) {
    for (int ky=0; ky < NY; ky++) {
      for (int kz=0; kz < NZo2p1; kz++) {
      
	double kkx = (double) kx;
	double kky = (double) ky;
	double kkz = (double) kz;

	if (kx>NX/2)
	  kkx -= NX;
	if (ky>NY/2)
	  kky -= NY;
	if (kz>NZ/2)
	  kkz -= NZ;
	
	// note that factor NX*NY*NZ is used here because FFTW 
	// fourier coefficients are not scaled
	//FFTW_REAL scaleFactor=2*( cos(2*M_PI*kx/NX) + cos(2*M_PI*ky/NY) + cos(2*M_PI*kz/NZ) - 3)*(NX*NY*NZ);

	FFTW_REAL scaleFactor=0.0;
 
	if (methodNb == 0) {

	  /*
	   * method 0 (from Numerical recipes)
	   */
	  
	  scaleFactor=2*( 
			 (cos(1.0*2*M_PI*kx/NX) - 1)/(dx*dx) + 
			 (cos(1.0*2*M_PI*ky/NY) - 1)/(dy*dy) + 
			 (cos(1.0*2*M_PI*kz/NZ) - 1)/(dz*dz) )*(NX*NY*NZ);
	  

	} else if (methodNb==1) {

	  /*
	   * method 1 (just from Continuous Fourier transform of 
	   * Poisson equation)
	   */
	  scaleFactor=-4*M_PI*M_PI*(kkx*kkx + kky*kky + kkz*kkz)*NX*NY*NZ;

	}

	
	if (kx!=0 or ky!=0 or kz!=0) {
	  phiComplex[kx*NY*NZo2p1+ky*NZo2p1+kz][0] /= scaleFactor;
	  phiComplex[kx*NY*NZo2p1+ky*NZo2p1+kz][1] /= scaleFactor;
	} else { // enforce mean value is zero
	  phiComplex[kx*NY*NZo2p1+ky*NZo2p1+kz][0] = 0.0;
	  phiComplex[kx*NY*NZo2p1+ky*NZo2p1+kz][1] = 0.0;
	}

      } // end for kz
    } // end for ky
  } // end for kx

  // compute FFT^{-1} (phiComplex) to retrieve solution
  FFTW_EXECUTE(plan_rho_backward);
  
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

  free(rho);
  free(solution);
  
 } // end main
