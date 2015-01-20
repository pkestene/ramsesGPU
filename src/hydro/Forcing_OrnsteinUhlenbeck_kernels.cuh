/*
 * Copyright CEA / Maison de la Simulation
 * Contributors: Pierre Kestener, Sebastien Fromang (May 22, 2012)
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 */

/**
 * \file Forcing_OrnsteinUhlenbeck_kernels.cu
 * \brief Provides the CUDA kernels for performing forcing field update.
 *
 * \author P. Kestener
 *
 * $Id: Forcing_OrnsteinUhlenbeck_kernels.cuh 3236 2014-02-04 00:09:53Z pkestene $
 */
#ifndef FORCING_ORNSTEIN_UHLENBECK_KERNELS_CUH_
#define FORCING_ORNSTEIN_UHLENBECK_KERNELS_CUH_

#include "real_type.h"
#include "constants.h"

#include <curand_kernel.h>

/**
 * CUDA kernel for forcing field mode update.
 *
 */
__global__ 
void init_random_generator_kernel(curandState *state)
{

  int id = threadIdx.x + blockIdx.x * blockDim.x;

  /* Each thread gets same seed, a different sequence 
     number, no offset */
  curand_init(3003, id, 0, &state[id]);

} //  init_random_generator_kernel

/*
 * CUDA kernel for forcing field mode update.
 *
 */
__global__ void update_forcing_field_mode_kernel(double      *d_forcingField,
						 double      *d_projTens,
						 curandState *state,
						 double       timeScaleTurb,
						 double       amplitudeTurb,
						 double       ksi,
						 int          nDim,
						 int          nMode,
						 real_t       dt)
{

  // Block index
  const int bx = blockIdx.x;
  
  // Thread index
  const int tx = threadIdx.x;

  const int iMode = bx * blockDim.x + tx;

  double weight = amplitudeTurb;
  double v      = sqrt(5.0/3.0)* (::gParams.cIso);

  double AAA[3] = {0.0, 0.0, 0.0};
  double BBB[3] = {0.0, 0.0, 0.0};
  double randomNumber;

  /* Copy state to local memory for efficiency */
  curandState localState = state[iMode];
  
  for (int i=0; i<nDim; i++) {
    randomNumber = curand_normal(&localState);
    AAA[i] = randomNumber*sqrt(dt);
  }
  /* Copy state back to global memory */
  state[iMode] = localState;

  for (int j=0; j<nDim ; j++) {
    double summ=0.0;
    for (int i=0; i<nDim; i++) {
      summ += d_projTens[i*nDim*nMode + j*nMode + iMode]*AAA[i];
    }
    BBB[j]=summ;
  }
  
  for (int i=0; i<nDim; i++)
    BBB[i] = BBB[i]*v*sqrt(2.0*weight*weight/timeScaleTurb)/timeScaleTurb;
      
  // now compute df
  for (int i=0; i<nDim; i++)
    BBB[i] = BBB[i] - d_forcingField[i*nMode+iMode]*dt/timeScaleTurb;
  
  // now update forcing field : f(t+dt) = f(t)+df
  double forceRMS = 3.0 / sqrt(1 - 2.0*ksi + 3.0*ksi*ksi);
  for (int i=0; i<nDim; i++) {
    d_forcingField[i*nMode+iMode] += forceRMS * BBB[i];
  }

} // update_forcing_field_mode_kernel 

/*
 * CUDA kernel for forcing field add.
 *
 * \param[in,out] d_U
 * \param[in]     d_forcingField
 * \param[in]     d_mode
 */
__global__ 
void add_forcing_field_kernel(real_t *d_U,
			      double *d_forcingField,
			      double *d_mode,
			      int     nDim,
			      int     nMode,
			      int     ghostWidth,
			      real_t  dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = bx * blockDim.x + tx;
  const int j = by * blockDim.y + ty;

  real_t &xMin = gParams.xMin;
  real_t &yMin = gParams.yMin;
  real_t &zMin = gParams.zMin;
  real_t &dx   = gParams.dx;
  real_t &dy   = gParams.dy;
  real_t &dz   = gParams.dz;

  int &mpiPosX = gParams.mpiPosX;
  int &mpiPosY = gParams.mpiPosY;
  int &mpiPosZ = gParams.mpiPosZ;

  int &nx      = gParams.nx;
  int &ny      = gParams.ny;
  int &nz      = gParams.nz;
  
  int isize = nx+2*ghostWidth;
  int jsize = ny+2*ghostWidth;
  int ksize = nz+2*ghostWidth;
  
  const double twoPi = 2*M_PI;
  
  const int arraySize  = isize * jsize * ksize;

  real_t xPos = xMin + dx/2 + (i-ghostWidth + nx * mpiPosX)*dx;

  real_t yPos = yMin + dy/2 + (j-ghostWidth + ny * mpiPosY)*dy;


  // domain loop
  for (int k=ghostWidth, offset = i + isize * j + isize*jsize*ghostWidth; 
       k<ksize-ghostWidth; 
       k++, offset += isize*jsize ) {

    real_t zPos = zMin + dz/2 + (k-ghostWidth + nz * mpiPosZ)*dz;
    
    if(i >= ghostWidth and i < isize-ghostWidth and 
       j >= ghostWidth and j < jsize-ghostWidth)
      {
	
	// compute velocity forcing field
	double AAA[3];
	for (int iDim=0; iDim<nDim; iDim++) {
	  double summ = 0.0;
	  for (int iMode=0; iMode<nMode; iMode++) {
	    double phase = 
	      xPos*d_mode[0*nMode+iMode] +
	      yPos*d_mode[1*nMode+iMode] +
	      zPos*d_mode[2*nMode+iMode];
	    summ += d_forcingField[iDim*nMode+iMode] * cos(twoPi*phase);
	  } // end for iMode
	  AAA[iDim] = summ;
	} // end for iDim
	
	// read hydro variables
	real_t rho = d_U[offset];
	real_t u   = d_U[offset+IU*arraySize];
	real_t v   = d_U[offset+IV*arraySize];
	real_t w   = d_U[offset+IW*arraySize];
	real_t eTot= d_U[offset+IP*arraySize];

	// compute internal energy
	real_t eInt = 0.5  * ( u*u + v*v + w*w ) / rho;
	eInt = eTot - eInt;
    
	// update velocity field
	u += AAA[0]*dt*rho;
	v += AAA[1]*dt*rho;
	w += AAA[2]*dt*rho;
	
	// write back to memory
	d_U[offset+IU*arraySize] = u;
	d_U[offset+IV*arraySize] = v;
	d_U[offset+IW*arraySize] = w;

	// update total energy
	d_U[offset+IP*arraySize] = eInt + 0.5  * ( u*u + v*v + w*w ) / rho;

      } // end if i,j

  } // end for k

} // add_forcing_field_kernel

#endif // FORCING_ORNSTEIN_UHLENBECK_KERNELS_CUH_
