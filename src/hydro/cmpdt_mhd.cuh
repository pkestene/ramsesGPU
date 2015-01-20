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
 * \file cmpdt_mhd.cuh
 * \brief Provides the CUDA kernel for computing MHD time step through a
 * MAX reduction.
 *
 * \sa routines defined in cmpdt.cuh
 *
 * \author P. Kestener
 *
 * $Id: cmpdt_mhd.cuh 2108 2012-05-23 12:07:21Z pkestene $
 */
#ifndef CMPDT_MHD_CUH_
#define CMPDT_MHD_CUH_

#include "real_type.h"
#include "constoprim.h"
#include "common_types.h"
#include "mhd_utils.h"

#define REDUCE_OP(x, y) x = FMAX(x, y)
#define REDUCE_VAR(array, idx, offset) REDUCE_OP(array[idx], array[idx+offset])
//#define REDUCE_2D(array, idx, offset)  REDUCE_VAR(array, idx, offset);
//#define REDUCE_3D(array, idx, offset)  REDUCE_VAR(array, idx, offset);

#define CMPDT_BLOCK_SIZE	128

/**
 * Compute time for 2D MHD.
 * Reduction kernel, using FMAX operator
 */
template<unsigned int blockSize>
__global__ void cmpdt_2d_mhd(real_t* U, 
			     real_t* g_odata, 
			     int pitch, 
			     int dimX,
			     int dimY)
{
  extern __shared__ real_t invDt[];

  int n = pitch*dimY;
  int physicalDim[2];
  physicalDim[0] = pitch;
  physicalDim[1] = dimY;

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  // see CUDA documentation of the reduction example, especially the
  // Reduction4 kernel ("Halve the number of blocks, and replace a
  // single load with two loads and first add of the reduction")
  unsigned int tid = threadIdx.x;
  invDt[tid] = 0;

  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  int iY = i / pitch;
  int iX = i - pitch*iY;

  unsigned int gridSize = blockSize * 2 * gridDim.x;
  invDt[tid] = gParams.smallc / gParams.dx;
  
  int &geometry = ::gParams.geometry;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while(i<n)
    {
      real_t q[NVAR_MHD];
      real_t fastInfoSpeed[3];
      real_t c;
      real_t vx, vy;

      if (iX>2 and iX<dimX-3 and 
	  iY>2 and iY<dimY-3) {
	
	// convert conservative (U) to primitive variables (q)
	computePrimitives_MHD_2D(U, physicalDim, i, c, q, ZERO_F);

	// compute fastest information speeds
	find_speed_info<TWO_D>(q, fastInfoSpeed);
	vx = fastInfoSpeed[IX];
	vy = fastInfoSpeed[IY];
	
	REDUCE_OP(invDt[tid], vx / gParams.dx + vy / gParams.dy);

      } // end if i inside physical domain
      
      // move to i2
      unsigned int i2 = i + blockSize;
      iY = i2 / pitch;
      iX = i2 - pitch*iY;
      if(i2<n and iX>2 and iX<dimX-3 and iY>2 and iY<dimY-3)
	{
	  computePrimitives_MHD_2D(U, physicalDim, i2, c, q, ZERO_F);
	  find_speed_info<TWO_D>(q, fastInfoSpeed);
	  vx = fastInfoSpeed[IX];
	  vy = fastInfoSpeed[IY];
	  if (geometry != GEO_CARTESIAN) {
	    // remember that ghostWidth is 3 in MHD
	    real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (iX-3)*(::gParams.dx);
	    vy /= xPos;
	  }
	  REDUCE_OP(invDt[tid], vx / gParams.dx + vy / gParams.dy);

	} // end if i2 inside physical domain

      i += gridSize;
      iY = i/pitch;
      iX = i - pitch*iY;
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
} // cmpdt_2d_mhd


/**
 * Compute time for 3D MHD.
 * Reduction kernel, using FMAX operator
 */
template<unsigned int blockSize>
__global__ void cmpdt_3d_mhd(real_t* U, 
			     real_t* g_odata, 
			     int pitch, 
			     int dimX,
			     int dimY,
			     int dimZ)
{
  extern __shared__ real_t invDt[];

  int n = pitch*dimY*dimZ;
  int physicalDim[3];
  physicalDim[0] = pitch;
  physicalDim[1] = dimY;
  physicalDim[2] = dimZ;

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  // see CUDA documentation of the reduction example, especially the
  // Reduction4 kernel ("Halve the number of blocks, and replace a
  // single load with two loads and first add of the reduction")
  unsigned int tid = threadIdx.x;
  invDt[tid] = 0;

  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  int iZ  = i / (pitch*dimY);
  int tmp = i - pitch*dimY*iZ;
  int iY  = tmp / pitch;
  int iX  = tmp - pitch * iY;

  unsigned int gridSize = blockSize * 2 * gridDim.x;
  invDt[tid] = gParams.smallc / FMIN(gParams.dx,gParams.dy);

  int    &geometry = ::gParams.geometry;
  real_t &Omega0   = ::gParams.Omega0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while( i < n )
    {
      real_t q[NVAR_MHD];
      real_t fastInfoSpeed[3];
      real_t c;
      real_t vx,vy,vz;
      
      if (iX>2 and iX<dimX-3 and 
	  iY>2 and iY<dimY-3 and
	  iZ>2 and iZ<dimZ-3) {

	// convert conservative (U) to primitive variables (q)
	computePrimitives_MHD_3D(U, physicalDim, i, c, q, ZERO_F);
	
	// compute fastest information speeds
	find_speed_info<THREE_D>(q, fastInfoSpeed);
	vx = fastInfoSpeed[IX];
	vy = fastInfoSpeed[IY];
	if (geometry == GEO_CARTESIAN and Omega0 > 0) {
	  real_t  deltaX   = ::gParams.xMax - ::gParams.xMin;
	  vy += 1.5*Omega0*deltaX/2;
	}
	
	vz = fastInfoSpeed[IZ];
	if (geometry == GEO_SPHERICAL) {
	  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (iX-3)*(::gParams.dx);
	  real_t yPos = ::gParams.yMin + ::gParams.dy/2 + (iY-3)*(::gParams.dy);
	  vz = vz / xPos / sin(yPos);
	}
	
	if (vx / gParams.dx + vy / gParams.dy  + vz / gParams.dz < gParams.smallc / gParams.dx) {
	  PRINTF("cmddt_mhd problem : %d %d %d\n",iX,iY,iZ);
	}

	REDUCE_OP(invDt[tid], vx / gParams.dx + vy / gParams.dy + vz / gParams.dz);

      } // end if i inside physical domain
      
      // move to i2
      unsigned int i2 = i + blockSize;
      iZ  = i2 / (pitch*dimY);
      tmp = i2 - pitch*dimY*iZ;
      iY  = tmp / pitch;
      iX  = tmp - pitch*iY;
      if(i2<n and 
	 iX>2 and iX<dimX-3 and 
	 iY>2 and iY<dimY-3 and
	 iZ>2 and iZ<dimZ-3)
	{
	  computePrimitives_MHD_3D(U, physicalDim, i2, c, q, ZERO_F);
	  find_speed_info<THREE_D>(q, fastInfoSpeed);

	  vx = fastInfoSpeed[IX];
	  vy = fastInfoSpeed[IY];
	  if (geometry == GEO_CARTESIAN and Omega0 > 0) {
	    real_t  deltaX   = ::gParams.xMax - ::gParams.xMin;
	    vy += 1.5*Omega0*deltaX/2;
	  }

	  vz = fastInfoSpeed[IZ];
	  if (geometry == GEO_SPHERICAL) {
	    real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (iX-3)*(::gParams.dx);
	    real_t yPos = ::gParams.yMin + ::gParams.dy/2 + (iY-3)*(::gParams.dy);
	    vz = vz / xPos / sin(yPos);
	  }

	  if (vx / gParams.dx + vy / gParams.dy  + vz / gParams.dz < gParams.smallc / gParams.dx) {
	    PRINTF("cmddt_mhd problem : %d %d %d\n",iX,iY,iZ);
	  }

	  REDUCE_OP(invDt[tid], vx / gParams.dx + vy / gParams.dy + vz / gParams.dz);

	} // end if i2 inside physical domain

      i  += gridSize;
      iZ  = i / pitch*dimY;
      tmp = i - pitch*dimY*iZ;
      iY  = tmp / pitch;
      iX  = tmp - pitch*iY;
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
} // cmpdt_3d_mhd


#endif // CMPDT_MHD_CUH_
