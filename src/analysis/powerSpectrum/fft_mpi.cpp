/**
 * \file fft_mpi.cpp
 *
 * Parallel FFT computation.
 *
 * \author P. Kestener
 * \date 20/06/2013
 *
 * $Id: fft_mpi.cpp 3394 2014-05-06 10:19:58Z pkestene $
 */

#include <mpi.h>
#include "fft_mpi.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>

/*****************************************************/
/*****************************************************/
/*****************************************************/
void compute_fft_mpi(ConfigMap    &configMap, 
		     double       *dataIn, 
		     double       *dataOut)
{
  
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  
  int nbMpiProc;
  MPI_Comm_size(MPI_COMM_WORLD, &nbMpiProc);
  
#ifdef USE_FFTW3_MPI
  
  // domain global size (fftw uses row-major order)
  ptrdiff_t NX,  NY,  NZ;  // for row-major order   
  ptrdiff_t NXc, NYc, NZc; // for column-major order
  
  // domain local size
  int nx,ny,nz;

  ptrdiff_t  i, j, k; // for row major order
  ptrdiff_t  ic, jc, kc; // for column major order

  //int k0 = 5;

  // fftw resources
  fftw_plan plan;
  fftw_complex *data; //local data of course
  ptrdiff_t alloc_local, local_n0, local_0_start;

  /* read local domain sizes */
  nx=configMap.getInteger("mesh","nx",32);
  ny=configMap.getInteger("mesh","ny",32);
  nz=configMap.getInteger("mesh","nz",32);

  /* global domain sizes (column-major) */
  NXc=nx;
  NYc=ny;
  NZc=nz*nbMpiProc;

  /* global domain sizes (row-major) */
  NX=NZc;
  NY=NYc;
  NZ=NXc;

#ifdef VERBOSE
  if (myRank==0) printf("global geometry : %ld %ld %ld\n",NX,NY,NZ);
#endif // VERBOSE

  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_3d(NX, NY, NZ, MPI_COMM_WORLD,
				       &local_n0, &local_0_start);
  data = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local);
 
#ifdef VERBOSE
  printf("[rank %d] alloc_local %ld local_n0 %ld\n", myRank, alloc_local, local_n0);
#endif // VERBOSE

  /* create plan for forward DFT */
  plan = fftw_mpi_plan_dft_3d(NX, NY, NZ, data, data, MPI_COMM_WORLD,
			      FFTW_FORWARD, FFTW_ESTIMATE);
  
  /* initialize data */
  for (i = 0; i < local_n0; ++i) {
    kc=i;
    for (j = 0; j < NY; ++j) {
      jc=j;
      for (k = 0; k < NZ; ++k) {
	ic=k;
	long int indexC = ic + NXc * (jc + NYc * kc);
	data[(i*NY + j)*NZ + k][0] = dataIn[indexC];
	data[(i*NY + j)*NZ + k][1] = 0;
	//data[(i*NY + j)*NZ + k][0] = drand48()-0.5;//cos(2*M_PI*k0*(i+local_0_start)/NX);
	//data[(i*NY + j)*NZ + k][1] = drand48()-0.5;//sin(2*M_PI*k0*(i+local_0_start)/NX);
      }
    }
#ifdef VERBOSE
    if (i==0) printf("[rank %d] data[i=%ld][j=0] =%g local_0_start = %ld\n",myRank,i,data[0][0],local_0_start);
#endif /* VERBOSE */
  }
  
  /* compute transforms, in-place, as many times as desired */
  fftw_execute(plan);

  // copy data into output array
  fftw_complex *dataOutC = (fftw_complex*) (dataOut);
  for (i = 0; i < local_n0; ++i) {
    for (j = 0; j < NY; ++j) {
      for (k = 0; k < NZ; ++k) {
	dataOutC[(i*NY + j)*NZ + k][0] = data[(i*NY + j)*NZ + k][0];
  	dataOutC[(i*NY + j)*NZ + k][1] = data[(i*NY + j)*NZ + k][1];
      }
    }
  }

  // debug
  // if (myRank==0) {
  //   for (i = 0; i < local_n0; ++i) {
  //     for (j = 0; j < NY; ++j) {
  // 	for (k = 0; k < NZ; ++k) {
  // 	  printf("%g+j%g ",data[(i*NY + j)*NZ + k][0]/(NX*NY*NZ),data[(i*NY + j)*NZ + k][1]/(NX*NY*NZ));
  // 	}
  // 	printf("\n");
  //     }
  //     printf("end of i=%ld\n\n",i+local_0_start);
  //   }
  // }
	  
  fftw_destroy_plan(plan);
  fftw_free(data);

#else

  if (myRank==0) {
    std::cout << "FFTW3 is not enabled !! Can't do anything !\n";
  }

#endif /* USE_FFTW3_MPI */

} // compute_fft_mpi

/*****************************************************/
/*****************************************************/
/*****************************************************/
void compute_ifft_mpi(ConfigMap    &configMap, 
		      double       *dataIn, 
		      double       *dataOut)
{
  
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  
  int nbMpiProc;
  MPI_Comm_size(MPI_COMM_WORLD, &nbMpiProc);
  
#ifdef USE_FFTW3_MPI
  
  // domain global size (fftw uses row-major order)
  ptrdiff_t NX,  NY,  NZ;  // for row-major order   
  ptrdiff_t NXc, NYc, NZc; // for column-major order
  
  // domain local size
  int nx,ny,nz;

  ptrdiff_t  i, j, k; // for row major order
  ptrdiff_t  ic, jc, kc; // for column major order

   // fftw resources
  fftw_plan plan;
  fftw_complex *data; //local data of course
  ptrdiff_t alloc_local, local_n0, local_0_start;

  /* read local domain sizes */
  nx=configMap.getInteger("mesh","nx",32);
  ny=configMap.getInteger("mesh","ny",32);
  nz=configMap.getInteger("mesh","nz",32);

  /* global domain sizes (column-major) */
  NXc=nx;
  NYc=ny;
  NZc=nz*nbMpiProc;

  /* global domain sizes (row-major) */
  NX=NZc;
  NY=NYc;
  NZ=NXc;

#ifdef VERBOSE
  if (myRank==0) printf("global geometry : %ld %ld %ld\n",NX,NY,NZ);
#endif // VERBOSE

  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_3d(NX, NY, NZ, MPI_COMM_WORLD,
				       &local_n0, &local_0_start);
  data = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local);
 
#ifdef VERBOSE
  printf("[rank %d] alloc_local %ld local_n0 %ld\n", myRank, alloc_local, local_n0);
#endif // VERBOSE

  /* create plan for forward DFT */
  plan = fftw_mpi_plan_dft_3d(NX, NY, NZ, data, data, MPI_COMM_WORLD,
			      FFTW_BACKWARD, FFTW_ESTIMATE);
  
  /* initialize data */
  fftw_complex *dataInC = (fftw_complex*) (dataIn);
  for (i = 0; i < local_n0; ++i) {
    kc=i;
    for (j = 0; j < NY; ++j) {
      jc=j;
      for (k = 0; k < NZ; ++k) {
	ic=k;
	data[(i*NY + j)*NZ + k][0] = dataInC[(i*NY + j)*NZ + k][0];
	data[(i*NY + j)*NZ + k][1] = dataInC[(i*NY + j)*NZ + k][1];
	//data[(i*NY + j)*NZ + k][0] = drand48()-0.5;//cos(2*M_PI*k0*(i+local_0_start)/NX);
	//data[(i*NY + j)*NZ + k][1] = drand48()-0.5;//sin(2*M_PI*k0*(i+local_0_start)/NX);
      }
    }
#ifdef VERBOSE
    if (i==0) printf("[rank %d] data[i=%ld][j=0] =%g local_0_start = %ld\n",myRank,i,data[0][0],local_0_start);
#endif /* VERBOSE */
  }
  
  /* compute transforms, in-place, as many times as desired */
  fftw_execute(plan);

  // copy data into output array
  for (i = 0; i < local_n0; ++i) {
    kc=i;
    for (j = 0; j < NY; ++j) {
      jc=j;
      for (k = 0; k < NZ; ++k) {
	ic=k;
	long int indexC = ic + NXc * (jc + NYc * kc);
	dataOut[indexC] = data[(i*NY + j)*NZ + k][0];
  	//data[(i*NY + j)*NZ + k][1] is zero
      }
    }
  }

  // debug
  // if (myRank==0) {
  //   for (i = 0; i < local_n0; ++i) {
  //     for (j = 0; j < NY; ++j) {
  // 	for (k = 0; k < NZ; ++k) {
  // 	  printf("%g+j%g ",data[(i*NY + j)*NZ + k][0]/(NX*NY*NZ),data[(i*NY + j)*NZ + k][1]/(NX*NY*NZ));
  // 	}
  // 	printf("\n");
  //     }
  //     printf("end of i=%ld\n\n",i+local_0_start);
  //   }
  // }
	  
  fftw_destroy_plan(plan);
  fftw_free(data);

#else

  if (myRank==0) {
    std::cout << "FFTW3 is not enabled !! Can't do anything !\n";
  }

#endif /* USE_FFTW3_MPI */

} // compute_ifft_mpi

/*****************************************************/
/*****************************************************/
/*****************************************************/
void compute_power_spectrum_mpi(ConfigMap    &configMap, 
				double       *dataFFT, 
				double       *dataHisto,
				double       *dataPsd)
{

  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  int nbMpiProc;
  MPI_Comm_size(MPI_COMM_WORLD, &nbMpiProc);

  fftw_complex *dataFFTC = (fftw_complex*) (dataFFT);

  int nBins = configMap.getInteger("powerSpectrum","nBins",128);

  // domain global size (fftw uses row-major order)
  ptrdiff_t NX,  NY,  NZ;  // for row-major order   
  ptrdiff_t NXc, NYc, NZc; // for column-major order
  
  // domain local size (column-major)
  int       nxc,nyc,nzc; // column-major
  int       nxr,nyr,nzr;    // row-major

  ptrdiff_t i, j, k; // for row major order
  //ptrdiff_t ic, jc, kc; // for column major order

  /* read local domain sizes (column-major) */
  nxc=configMap.getInteger("mesh","nx",32);
  nyc=configMap.getInteger("mesh","ny",32);
  nzc=configMap.getInteger("mesh","nz",32);

  nxr=nzc;
  nyr=nyc;
  nzr=nxc;

  /* global domain sizes (column-major) */
  NXc=nxc;
  NYc=nyc;
  NZc=nzc*nbMpiProc;

  /* global domain sizes (row-major) */
  NX=NZc;
  NY=NYc;
  NZ=NXc;

  double maxSizeGlobal = NX/2.0;
  double dk = maxSizeGlobal/(nBins-1);

  // use row-major
  // i,j,k are local
  // kx, ky, kz are global
  for (i = 0; i < nxr; ++i) {
    
    int kx = i+myRank*nxr<NX/2 ? i+myRank*nxr : i+myRank*nxr-NX;
    
    for (j = 0; j < nyr; ++j) {

      int ky = j<NY/2 ? j : j-NY;

      for (k = 0; k < nzr; ++k) {

	int kz = k<NZ/2 ? k : k-NZ;

	double fftMod = 
	  dataFFTC[(i*nyr + j)*nzr + k][0]*dataFFTC[(i*nyr + j)*nzr + k][0]+
	  dataFFTC[(i*nyr + j)*nzr + k][1]*dataFFTC[(i*nyr + j)*nzr + k][1];
	
	double kMod = sqrt(kx*kx+ky*ky+kz*kz);
	int index = (int) (kMod/dk+0.5);
	
	if (index < nBins) {
	  dataHisto[index] += 1;
	  dataPsd  [index] += fftMod;
	}
      }
    }
  }

} // compute_power_spectrum_mpi

/*****************************************************/
/*****************************************************/
/*****************************************************/
#ifdef GEN_FBM
void set_power_law_power_spectrum(ConfigMap &configMap,
				  double    *data_fft,
				  std::default_random_engine& generator) {

  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  int nbMpiProc;
  MPI_Comm_size(MPI_COMM_WORLD, &nbMpiProc);

  /* assumes here that  nbMpiProc is 1 */

  fftw_complex *dataFFTC = (fftw_complex*) (data_fft);

  // domain global size (fftw uses row-major order)
  //ptrdiff_t NX,  NY,  NZ;  // for row-major order   
  //ptrdiff_t NXc, NYc, NZc; // for column-major order
  
  // domain local size (column-major)
  int       nxc,nyc,nzc; // column-major
  int       nxr,nyr,nzr;    // row-major

  ptrdiff_t i, j, k; // for row major order
  //ptrdiff_t ic, jc, kc; // for column major order

  /* read local domain sizes (column-major) */
  nxc=configMap.getInteger("mesh","nx",32);
  nyc=configMap.getInteger("mesh","ny",32);
  nzc=configMap.getInteger("mesh","nz",32);

  nxr=nzc;
  nyr=nyc;
  nzr=nxc;

  /* global domain sizes (column-major) */
  //NXc=nxc;
  //NYc=nyc;
  //NZc=nzc;

  /* global domain sizes (row-major) */
  //NX=NZc;
  //NY=NYc;
  //NZ=NXc;

  // Hurst exponent
  double h = configMap.getFloat("fBm","h",0.5);

  // random distribution
  std::normal_distribution<double>       normal_dist(0.0,1.0);
  std::uniform_real_distribution<double> unif_dist(0.0,1.0);

  // use row-major

  // first loop to fill octant 1 and 8
  for (i = 0; i < nxr/2+1; ++i) {
    
    int kx = i;

    int i2;
    if (i==0)
      i2=0;
    else if (i==nxr/2)
      i2=nxr/2;
    else
      i2=nxr-i;

    for (j = 0; j < nyr/2+1; ++j) {

      int ky = j;
      int j2;
      if (j==0)
	j2=0;
      else if (j==nyr/2)
	j2=nyr/2;
      else
	j2=nyr-j;

      for (k = 0; k < nzr/2+1; ++k) {

	int kz = k;
	int k2;
	if (k==0)
	  k2=0;
	else if (k==nzr/2)
	  k2=nzr/2;
	else
	  k2=nzr-k;
	
	double kSquare = kx*kx + ky*ky + kz*kz;
	double radius, phase;

	if (kSquare > 0) {
	  radius = pow(kSquare, -(2*h+3)/4) * normal_dist(generator);
	  phase = 2 * M_PI * unif_dist(generator);
	} else {
	  radius = 1.0;
	  phase  = 0.0;
	}

	// fill Fourier coef so that ifft is real
	dataFFTC[(i *nyr + j )*nzr + k ][0] =   radius*cos(phase);
	dataFFTC[(i *nyr + j )*nzr + k ][1] =   radius*sin(phase);

	dataFFTC[(i2*nyr + j2)*nzr + k2][0] =   radius*cos(phase);
	dataFFTC[(i2*nyr + j2)*nzr + k2][1] = - radius*sin(phase);
		
	
      } // end for k
    } // end for j
  } // end for i

  // make sure that Fourier coef at i=nx/2 ... is real
  dataFFTC[(nxr/2*nyr + 0    )*nzr + 0    ][1] = 0;
  dataFFTC[(    0*nyr + nyr/2)*nzr + 0    ][1] = 0;
  dataFFTC[(    0*nyr + 0    )*nzr + nzr/2][1] = 0;
    
  dataFFTC[(nxr/2*nyr + nyr/2)*nzr + 0    ][1] = 0;
  dataFFTC[(nxr/2*nyr + 0    )*nzr + nzr/2][1] = 0;
  dataFFTC[(    0*nyr + nyr/2)*nzr + nzr/2][1] = 0;

  dataFFTC[(nxr/2*nyr + nyr/2)*nzr + nzr/2][1] = 0;

  // 2nd loop to fill all other octants
  for (i = 1; i < nxr/2; ++i) {

    int kx = i;
    
    for (j = 1; j < nyr/2; ++j) {

      int ky = j;

      for (k = 1; k < nzr/2; ++k) {
	
	int kz = k;

	int i1 = nxr-i;
	int i2 = i;
	  
	int j1 = nyr-j;
	int j2 = j;
	  
	int k1 = nzr-k;
	int k2 = k;

	// fill Fourier coef so that ifft is real
	double kSquare = kx*kx + ky*ky + kz*kz;

	double radius = pow(kSquare, -(2*h+3)/4) * normal_dist(generator);
	double phase = 2 * M_PI * unif_dist(generator);
	
	//dataFFTC[i1,j2,k2] = radius*np.cos(phase) + 1j*radius*np.sin(phase);
	dataFFTC[(i1*nyr + j2)*nzr + k2][0] = radius*cos(phase);
	dataFFTC[(i1*nyr + j2)*nzr + k2][1] = radius*sin(phase);
	//dataFFTC[i2,j1,k1] = radius*np.cos(phase) - 1j*radius*np.sin(phase);
	dataFFTC[(i2*nyr + j1)*nzr + k1][0] =  radius*cos(phase);
	dataFFTC[(i2*nyr + j1)*nzr + k1][1] = -radius*sin(phase);
	  
	radius = pow(kSquare, -(2*h+3)/4) * normal_dist(generator);
	phase = 2 * M_PI * unif_dist(generator);
	
	//dataFFTC[i2,j1,k2] = radius*np.cos(phase) + 1j*radius*np.sin(phase);
	dataFFTC[(i2*nyr + j1)*nzr + k2][0] = radius*cos(phase);
	dataFFTC[(i2*nyr + j1)*nzr + k2][1] = radius*sin(phase);
	//dataFFTC[i1,j2,k1] = radius*np.cos(phase) - 1j*radius*np.sin(phase);
	dataFFTC[(i1*nyr + j2)*nzr + k1][0] =  radius*cos(phase);
	dataFFTC[(i1*nyr + j2)*nzr + k1][1] = -radius*sin(phase);
        
	radius = pow(kSquare, -(2*h+3)/4) * normal_dist(generator);
	phase = 2 * M_PI * unif_dist(generator);
	
	//dataFFTC[i2,j2,k1] = radius*np.cos(phase) + 1j*radius*np.sin(phase);
	dataFFTC[(i2*nyr + j2)*nzr + k1][0] = radius*cos(phase);
	dataFFTC[(i2*nyr + j2)*nzr + k1][1] = radius*sin(phase);
	//dataFFTC[i1,j1,k2] = radius*np.cos(phase) - 1j*radius*np.sin(phase);
	dataFFTC[(i1*nyr + j1)*nzr + k2][0] =  radius*cos(phase);
	dataFFTC[(i1*nyr + j1)*nzr + k2][1] = -radius*sin(phase);

      } // end for k
    } // end for j
  } // end for i


} // set_power_law_power_spectrum
#endif // GEN_FBM
