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
 * \file gravity.cuh
 * \brief CUDA kernel for computing gravity forces (adapted from Dumses).
 *
 * \date 13 Janv 2014
 * \author P. Kestener
 *
 * $Id: gravity.cuh 3261 2014-02-07 20:19:06Z pkestene $
 */
#ifndef GRAVITY_CUH_
#define GRAVITY_CUH_

#include "real_type.h"
#include "constants.h"

#define GRAVITY_PRED_2D_DIMX	16
#define GRAVITY_PRED_2D_DIMY	16
/*
 * CUDA kernel computing gravity predictor (2D data).
 * 
 * \param[in,out]  Q      primitive variables array.
 * \param[in]      dt     time step
 */
__global__ 
void kernel_gravity_predictor_2d(real_t* Q, 
				 int ghostWidth,
				 int pitch, 
				 int imax, 
				 int jmax, 
				 real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = bx * blockDim.x + tx;
  const int j = by * blockDim.y + ty;
  
  const int arraySize  = pitch * jmax;
  const int elemOffset = pitch * j    + i;

  real_t *gravin = gParams.arrayList[A_GRAV];
  
  // update velocity components (IU, IV) with v=v+gravin*dt*half
  if (i>=ghostWidth and i<imax-ghostWidth and
      j>=ghostWidth and j<jmax-ghostWidth)
    {
      
      int offsetQ = elemOffset + IU*arraySize;
      int offsetG = elemOffset;
      Q[offsetQ] += HALF_F * dt * gravin[offsetG]; // component IU
      offsetQ += arraySize; offsetG += arraySize;

      Q[offsetQ] += HALF_F * dt * gravin[offsetG]; // component IV

    }

} // kernel_gravity_predictor_2d

#define GRAVITY_PRED_3D_DIMX	16
#define GRAVITY_PRED_3D_DIMY	16
/*
 * CUDA kernel computing gravity predictor (3D data).
 * 
 * \param[in,out]  Q      primitive variables array.
 * \param[in]      dt     time step
 */
__global__ 
void kernel_gravity_predictor_3d(real_t* Q, 
				 int ghostWidth,
				 int pitch, 
				 int imax, 
				 int jmax, 
				 int kmax, 
				 real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = bx * blockDim.x + tx;
  const int j = by * blockDim.y + ty;
  
  const int arraySize  = pitch * jmax * kmax;

  real_t *gravin = gParams.arrayList[A_GRAV];
  
  // update velocity components (IU, IV, IW) with v=v+gravin*dt*half
  for (int k=ghostWidth, elemOffset = i + pitch * (j + jmax * ghostWidth);
       k < kmax-ghostWidth; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if (i>=ghostWidth and i<imax-ghostWidth and
	j>=ghostWidth and j<jmax-ghostWidth)
      {
	
	int offsetQ = elemOffset + 2*arraySize;
	int offsetG = elemOffset;
	
	Q[offsetQ] += HALF_F * dt * gravin[offsetG]; // component IU
	offsetQ += arraySize; offsetG += arraySize;
	
	Q[offsetQ] += HALF_F * dt * gravin[offsetG]; // component IV
	offsetQ += arraySize; offsetG += arraySize;
	
	Q[offsetQ] += HALF_F * dt * gravin[offsetG]; // component IW

      } // end if i,j

  } // end for k

} // kernel_gravity_predictor_3d

#define GRAVITY_SRC_2D_DIMX	16
#define GRAVITY_SRC_2D_DIMY	16
/*
 * CUDA kernel computing gravity source term (2D data).
 * 
 * \param[in,out]  UNew   conservative variables array.
 * \param[in,out]  UOld   conservative variables array.
 * \param[in]      dt     time step
 */
__global__ 
void kernel_gravity_source_term_2d(real_t* UNew, 
				   real_t* UOld, 
				   int ghostWidth,
				   int pitch, 
				   int imax, 
				   int jmax, 
				   real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = bx * blockDim.x + tx;
  const int j = by * blockDim.y + ty;
  
  const int arraySize  = pitch * jmax;
  const int elemOffset = pitch * j    + i;

  real_t *gravin = gParams.arrayList[A_GRAV];
  
  // update velocity components (IU, IV) with v=v+gravin*dt*half
  if (i>=ghostWidth and i<imax-ghostWidth and
      j>=ghostWidth and j<jmax-ghostWidth)
    {
      
      // read density
      real_t rhoOld = UOld[elemOffset];
      real_t rhoNew = UNew[elemOffset];

      int offsetU = elemOffset + IU*arraySize;
      int offsetG = elemOffset;
      UNew[offsetU] += HALF_F * dt * gravin[offsetG] * (rhoOld + rhoNew); // component IU
      offsetU += arraySize; offsetG += arraySize;

      UNew[offsetU] += HALF_F * dt * gravin[offsetG] * (rhoOld + rhoNew); // component IV

    }

} // kernel_gravity_source_term_2d

#define GRAVITY_SRC_3D_DIMX	16
#define GRAVITY_SRC_3D_DIMY	16
/*
 * CUDA kernel computing gravity source term (3D data).
 * 
 * \param[in,out]  UNew   conservative variables array.
 * \param[in,out]  UOld   conservative variables array.
 * \param[in]      dt     time step
 */
__global__
void kernel_gravity_source_term_3d(real_t* UNew, 
				   real_t* UOld, 
				   int ghostWidth,
				   int pitch, 
				   int imax, 
				   int jmax, 
				   int kmax, 
				   real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = bx * blockDim.x + tx;
  const int j = by * blockDim.y + ty;
  
  const int arraySize  = pitch * jmax * kmax;

  real_t *gravin = gParams.arrayList[A_GRAV];
  
  // update velocity components (IU, IV, IW) with v=v+gravin*dt*half
  for (int k=ghostWidth, elemOffset = i + pitch * (j + jmax * ghostWidth);
       k < kmax-ghostWidth; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if (i>=ghostWidth and i<imax-ghostWidth and
	j>=ghostWidth and j<jmax-ghostWidth)
      {
	
	// read density
	real_t rhoOld = UOld[elemOffset];
	real_t rhoNew = UNew[elemOffset];

	// set indexes offset for reading velocity components U,V,W
	int offsetU = elemOffset + 2*arraySize;

	// set index offset for reading gravity components
	int offsetG = elemOffset;

	// component IU
	UNew[offsetU] += HALF_F * dt * gravin[offsetG] * (rhoOld + rhoNew);
	offsetU += arraySize; offsetG += arraySize;

	// component IV
	UNew[offsetU] += HALF_F * dt * gravin[offsetG] * (rhoOld + rhoNew);
	offsetU += arraySize; offsetG += arraySize;

	// component IW
	UNew[offsetU] += HALF_F * dt * gravin[offsetG] * (rhoOld + rhoNew);

      } // end if i,j

  } // end for k

} // kernel_gravity_source_term_3d

#endif // GRAVITY_CUH_
