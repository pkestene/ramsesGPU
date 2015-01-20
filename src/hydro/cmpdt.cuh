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
 * \file cmpdt.cuh
 * \brief Provides the CUDA kernel for computing time step through a
 * MAX reduction.
 *
 * This routines are directly borrowed from the original CUDA SDK
 * reduction example.
 *
 * \author F. Chateau
 *
 * $Id: cmpdt.cuh 3449 2014-06-16 16:24:38Z pkestene $
 */
#ifndef CMPDT_CUH_
#define CMPDT_CUH_

#include "real_type.h"
#include "constoprim.h"
#include "common_types.h"

#define REDUCE_OP(x, y) x = FMAX(x, y)
#define REDUCE_VAR(array, idx, offset) REDUCE_OP(array[idx], array[idx+offset])

#define CMPDT_BLOCK_SIZE	128

/**
 * Reduction kernel, using FMAX operator
 */
template<unsigned int blockSize>
__global__ void cmpdt_2d(real_t* U, 
			 real_t* g_odata,
			 int pitch,
			 int dimX,
			 int dimY)
{
  extern __shared__ real_t invDt[];

  int n = pitch * dimY;
  
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  // see CUDA documentation of the reduction example, especially the
  // Reduction4 kernel ("Halve the number of blocks, and replace a
  // single load with two loads and first add of the reduction")
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  invDt[tid] = 0;
  //real_t  maxInvDt = 0;

  real_t &dx = ::gParams.dx;
  real_t &dy = ::gParams.dy;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while(i < n)
    {
      real_t q[NVAR_2D];
      real_t c;
      real_t invDt_x, invDt_y;
     
      int iY = i / pitch;
      int iX = i - pitch*iY;

      if (iX>=2 and iX<dimX-2 and 
	  iY>=2 and iY<dimY-2) {
	  
	  computePrimitives_0(U, n, i, c, q);
	  invDt_x = (c + FABS(q[IU])) / dx;
	  invDt_y = (c + FABS(q[IV])) / dy;
	  REDUCE_OP(invDt[tid], invDt_x + invDt_y );
      }
      
      if(i+blockSize < n)
	{
	  iY = (i+blockSize) / pitch;
	  iX = (i+blockSize) - pitch*iY;
	  
	  if (iX>=2 and iX<dimX-2 and 
	      iY>=2 and iY<dimY-2) {
	      computePrimitives_0(U, n, i+blockSize, c, q);
	      invDt_x = (c + FABS(q[IU]))/dx;
	      invDt_y = (c + FABS(q[IV]))/dy;
	      REDUCE_OP(invDt[tid], invDt_x + invDt_y );
	  }
	}

      i += gridSize;
    }
  __syncthreads();

  // do reduction in shared mem
  if(blockSize >= 512) { if(tid < 256) { REDUCE_VAR(invDt, tid, 256); } __syncthreads(); }
  if(blockSize >= 256) { if(tid < 128) { REDUCE_VAR(invDt, tid, 128); } __syncthreads(); }
  if(blockSize >= 128) { if(tid <  64) { REDUCE_VAR(invDt, tid,  64); } __syncthreads(); }

  if(tid < 32)
    {
      volatile real_t* smem = invDt;
      if(blockSize >=  64) { REDUCE_VAR(smem, tid, 32); }
      if(blockSize >=  32) { REDUCE_VAR(smem, tid, 16); }
      if(blockSize >=  16) { REDUCE_VAR(smem, tid,  8); }
      if(blockSize >=   8) { REDUCE_VAR(smem, tid,  4); }
      if(blockSize >=   4) { REDUCE_VAR(smem, tid,  2); }
      if(blockSize >=   2) { REDUCE_VAR(smem, tid,  1); }
    }

  // write result for this block to global mem
  if(tid == 0)
    {
      g_odata[blockIdx.x] = invDt[0];
    }
} // cmpdt_2d


/**
 * Reduction kernel, using FMAX operator, adapted to a real3_t array
 */
template<unsigned int blockSize>
__global__ void cmpdt_3d(real_t* U, 
			 real_t* g_odata, 
			 int pitch,
			 int dimX,
			 int dimY,
			 int dimZ)
{
  
  // here is the work around : use 3 real_t arrays, nothing to be proud of...
  extern __shared__ real_t invDt[];

  int n = pitch*dimY*dimZ;

  real_t &dx = ::gParams.dx;
  real_t &dy = ::gParams.dy;
  real_t &dz = ::gParams.dz;

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  // see CUDA documentation of the reduction example, especially the
  // Reduction4 kernel ("Halve the number of blocks, and replace a
  // single load with two loads and first add of the reduction")
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  invDt[tid] = ZERO_F;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while(i < n)
    {
      real_t q[NVAR_3D];
      real_t c;
      real_t invDt_x, invDt_y, invDt_z;

      int iZ = i / (pitch*dimY);
      int iY = (i - iZ*pitch*dimY) / pitch;
      int iX = i - iY*pitch - iZ*pitch*dimY;
      
      if (iX>=2 and iX<dimX-2 and 
	  iY>=2 and iY<dimY-2 and
	  iZ>=2 and iZ<dimZ-2) {
	computePrimitives_3D_0(U, n, i, c, q);
	invDt_x = (c + FABS(q[IU]))/dx;
	invDt_y = (c + FABS(q[IV]))/dy;
	invDt_z = (c + FABS(q[IW]))/dz;
	REDUCE_OP(invDt[tid], invDt_x + invDt_y + invDt_z);
      }

      if(i+blockSize < n)
	{

	  iZ = (i + blockSize) / (pitch*dimY);
	  iY = (i + blockSize - iZ*pitch*dimY) / pitch;
	  iX = i + blockSize - iY*pitch - iZ*pitch*dimY;

	  if (iX>=2 and iX<dimX-2 and 
	      iY>=2 and iY<dimY-2 and
	      iZ>=2 and iZ<dimZ-2) {
	    computePrimitives_3D_0(U, n, i+blockSize, c, q);
	    invDt_x = (c + FABS(q[IU]))/dx;
	    invDt_y = (c + FABS(q[IV]))/dy;
	    invDt_z = (c + FABS(q[IW]))/dz;
	    REDUCE_OP(invDt[tid], invDt_x + invDt_y + invDt_z);
	  }
	}

      i += gridSize;
    }
  __syncthreads();

  // do reduction in shared mem
  if(blockSize >= 512) { if(tid < 256) { REDUCE_VAR(invDt, tid, 256); } __syncthreads(); }
  if(blockSize >= 256) { if(tid < 128) { REDUCE_VAR(invDt, tid, 128); } __syncthreads(); }
  if(blockSize >= 128) { if(tid <  64) { REDUCE_VAR(invDt, tid,  64); } __syncthreads(); }

  if(tid < 32)
    {
      volatile real_t* smem = invDt;
      if(blockSize >=  64) { REDUCE_VAR(smem, tid, 32); }
      if(blockSize >=  32) { REDUCE_VAR(smem, tid, 16); }
      if(blockSize >=  16) { REDUCE_VAR(smem, tid,  8); }
      if(blockSize >=   8) { REDUCE_VAR(smem, tid,  4); }
      if(blockSize >=   4) { REDUCE_VAR(smem, tid,  2); }
      if(blockSize >=   2) { REDUCE_VAR(smem, tid,  1); }
    }

  // write result for this block to global mem
  if(tid == 0)
    {
      g_odata[blockIdx.x] = invDt[0];
    }
} // cmpdt_3d

#endif /*CMPDT_CUH_*/
