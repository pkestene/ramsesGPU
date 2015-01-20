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
 * \file random_forcing.cuh
 * \brief CUDA kernel for random forcing.
 *
 * \date 4 Sept 2012
 * \author P. Kestener
 *
 * $Id: random_forcing.cuh 3558 2014-10-07 20:30:35Z pkestene $
 */
#ifndef RANDOM_FORCING_CUH_
#define RANDOM_FORCING_CUH_

#include "real_type.h"
#include "constants.h"
#include "common_types.h"
#include <float.h> // for FLT_MAX

#ifdef USE_DOUBLE
#define ADD_RANDOM_FORCING_3D_DIMX	24
#define ADD_RANDOM_FORCING_3D_DIMY	16
#else // simple precision
#define ADD_RANDOM_FORCING_3D_DIMX	24
#define ADD_RANDOM_FORCING_3D_DIMY	16
#endif // USE_DOUBLE

/**
 * CUDA kernel to add random forcing field to velocity (3D data only).
 * 
 * \param[inout] Uin    conservative variables array.
 * \param[in]    randomForcing random forcing field
 * \param[in]    dt time step
 * \param[in]    norm random forcing field normalization
 *
 */
__global__ void kernel_add_random_forcing_3d(real_t* Uin, 
					     real_t* randomForcing,
					     real_t dt,
					     real_t norm,
					     int ghostWidth,
					     int pitch, 
					     int imax, 
					     int jmax,
					     int kmax)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, ADD_RANDOM_FORCING_3D_DIMX) + tx;
  const int j = __mul24(by, ADD_RANDOM_FORCING_3D_DIMY) + ty;
  
  const int arraySize  = pitch * jmax * kmax;

  // conservative variables
  real_t rho, e, u, v, w;

  // random forcing field
  real_t forcing[3];

  for (int k=ghostWidth, elemOffset = i + pitch * (j + jmax * ghostWidth);
       k < kmax-ghostWidth; 
       ++k, elemOffset += (pitch*jmax)) {
    
    
    if (i>=ghostWidth and i<imax-ghostWidth and
	j>=ghostWidth and j<jmax-ghostWidth)
      {
	
	// read input conservative variables
	int offset  = elemOffset;
	int offset2 = elemOffset;

	// read density and total energy
	rho = Uin[offset];  offset += arraySize;
	e   = Uin[offset];  offset += arraySize;

	// read velocity and random forcing field along X
	u           = Uin[offset];            offset  += arraySize;
	forcing[IX] = randomForcing[offset2]; offset2 += arraySize;
	e += u/rho * forcing[IX] * norm +
	  0.5 * SQR( forcing[IX] * norm );

	// read velocity and random forcing field along Y
	v           = Uin[offset];            offset += arraySize;
	forcing[IY] = randomForcing[offset2]; offset2 += arraySize;
	e += v/rho * forcing[IY] * norm +
	  0.5 * SQR( forcing[IY] * norm );

	// read velocity and random forcing field along Z
	w           = Uin[offset];
	forcing[IZ] = randomForcing[offset2];
	e += w/rho * forcing[IZ] * norm +
	  0.5 * SQR( forcing[IZ] * norm );

	// write back energy in external memory
	offset  = elemOffset; offset += arraySize; // go over density
	Uin[offset] = e;      offset += arraySize;

	// write back velocity (in fact momentum, so we multiply by rho)
	Uin[offset] += rho * forcing[IX] * norm;offset += arraySize;
	Uin[offset] += rho * forcing[IY] * norm;offset += arraySize;
	Uin[offset] += rho * forcing[IZ] * norm;

      } // end i-j guard
  } // end for k

} // kernel_add_random_forcing_3d

/*
 * prepare random forcing normalization
 */

#define REDUCE_OP1(x, y) x = (x+y)
#define REDUCE_VAR1(array, idx, offset) REDUCE_OP1(array[idx], array[idx+offset])

#define REDUCE_OP2(x, y) x = FMIN(x, y)
#define REDUCE_VAR2(array, idx, offset) REDUCE_OP2(array[idx], array[idx+offset])

#define REDUCE_OP3(x, y) x = FMAX(x, y)
#define REDUCE_VAR3(array, idx, offset) REDUCE_OP3(array[idx], array[idx+offset])

#define REDUCE_VAR_T(array, idx, size, offset) \
  REDUCE_VAR1(array, idx,         offset); \
  REDUCE_VAR1(array, idx +  size, offset); \
  REDUCE_VAR1(array, idx +2*size, offset); \
  REDUCE_VAR1(array, idx +3*size, offset); \
  REDUCE_VAR1(array, idx +4*size, offset); \
  REDUCE_VAR1(array, idx +5*size, offset); \
  REDUCE_VAR1(array, idx +6*size, offset); \
  REDUCE_VAR2(array, idx +7*size, offset); \
  REDUCE_VAR3(array, idx +8*size, offset);


#define RANDOM_FORCING_BLOCK_SIZE	128

/**
 * Reduction kernel, using FMAX operator, adapted to a real3_t array
 *
 * Here is the list of the 9 reductions:
 * \li 0 sum of rho*v*(delta v)
 * \li 1 sum of rho*(delta v)^2
 * \li 2 sum of rho*v^2/temperature
 * \li 3 sum of v^2/temprature
 * \li 4 sum of rho*v^2
 * \li 5 sum of v^2
 * \li 6 sum of rho^2
 * \li 7 min of rho
 * \li 8 max of rho
 *
 * \param[in] U conservative variables grid array
 * \param[in] randomForcing random forcing field array
 * \param[out] g_odata temporary array holding partial reduction
 * \param[in] ghostWidth 2 for hydro, 3 for MHD
 */
template<unsigned int blockSize>
__global__ void kernel_compute_random_forcing_normalization(real_t* U, 
							    real_t* randomForcing,
							    real_t* g_odata, 
							    int ghostWidth,
							    int pitch,
							    int dimX,
							    int dimY,
							    int dimZ)
{
  
  // temporary array holding partial reduction
  extern __shared__ real_t reduceArray[];
  const int reduceSize = RANDOM_FORCING_BLOCK_SIZE;

  const int n = pitch*dimY*dimZ;
  const int &arraySize = n;

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  // see CUDA documentation of the reduction example, especially the
  // Reduction4 kernel ("Halve the number of blocks, and replace a
  // single load with two loads and first add of the reduction")
  int tid = threadIdx.x;
  int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  int gridSize = blockSize * 2 * gridDim.x;

  // initialize partial reduction array
  reduceArray[tid             ] = ZERO_F;
  reduceArray[tid+  reduceSize] = ZERO_F;
  reduceArray[tid+2*reduceSize] = ZERO_F;
  reduceArray[tid+3*reduceSize] = ZERO_F;
  reduceArray[tid+4*reduceSize] = ZERO_F;
  reduceArray[tid+5*reduceSize] = ZERO_F;
  reduceArray[tid+6*reduceSize] = ZERO_F;
  reduceArray[tid+7*reduceSize] = FLT_MAX; // compute a MIN
  reduceArray[tid+8*reduceSize] = ZERO_F;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while(i < n)
    {
      int iZ = i / (pitch*dimY);
      int iY = (i - iZ*pitch*dimY) / pitch;
      int iX = i - iY*pitch - iZ*pitch*dimY;

     
      if (iX>=ghostWidth and iX<dimX-ghostWidth and 
	  iY>=ghostWidth and iY<dimY-ghostWidth and
	  iZ>=ghostWidth and iZ<dimZ-ghostWidth) {
	
	// read conservative variables
	int offset = i;
	real_t rho = U[offset];  offset += arraySize;
	real_t e   = U[offset];  offset += arraySize;
	real_t u   = U[offset]/rho;  offset += arraySize;
	real_t v   = U[offset]/rho;  offset += arraySize;
	real_t w   = U[offset]/rho;

	// read random forcing
	offset = i;
	real_t uu = randomForcing[offset]; offset += arraySize;
	real_t vv = randomForcing[offset]; offset += arraySize;
	real_t ww = randomForcing[offset]; 

	REDUCE_OP1(reduceArray[tid             ], rho * ( u*uu +  v*vv +  w*ww) );
	REDUCE_OP1(reduceArray[tid+  reduceSize], rho * (uu*uu + vv*vv + ww*ww) );

	// compute temperature
	real_t temperature;
	if (::gParams.cIso >0) {
	  temperature = SQR(::gParams.cIso);
	} else { // use ideal gas eq of state (P over rho)
	  temperature =  (::gParams.gamma0 - ONE_F) * 
	    (e - 0.5 * rho * ( u*u + v*v + w*w ) );
	}

 	REDUCE_OP1(reduceArray[tid+2*reduceSize], rho * (u*u + v*v + w*w) / temperature );
 	REDUCE_OP1(reduceArray[tid+3*reduceSize], (u*u + v*v + w*w) / temperature );
 	REDUCE_OP1(reduceArray[tid+4*reduceSize], rho * (u*u + v*v + w*w) );
 	REDUCE_OP1(reduceArray[tid+5*reduceSize], u*u + v*v + w*w );
  	REDUCE_OP1(reduceArray[tid+6*reduceSize], rho*rho );
  	REDUCE_OP2(reduceArray[tid+7*reduceSize], rho ); // watch out !
  	REDUCE_OP3(reduceArray[tid+8*reduceSize], rho ); // watch out !
      } // end if iX, iY, iZ guard

      if(i+blockSize < n)
	{

	  iZ = (i + blockSize) / (pitch*dimY);
	  iY = (i + blockSize - iZ*pitch*dimY) / pitch;
	  iX = i + blockSize - iY*pitch - iZ*pitch*dimY;

	  if (iX>=ghostWidth and iX<dimX-ghostWidth and 
	      iY>=ghostWidth and iY<dimY-ghostWidth and
	      iZ>=ghostWidth and iZ<dimZ-ghostWidth) {

	    // read conservative variables
	    int offset = i + blockSize;
	    real_t rho = U[offset];  offset += arraySize;
	    real_t e   = U[offset];  offset += arraySize;
	    real_t u   = U[offset]/rho;  offset += arraySize;
	    real_t v   = U[offset]/rho;  offset += arraySize;
	    real_t w   = U[offset]/rho;
	    
	    // read random forcing
	    offset = i + blockSize;
	    real_t uu = randomForcing[offset]; offset += arraySize;
	    real_t vv = randomForcing[offset]; offset += arraySize;
	    real_t ww = randomForcing[offset]; 
	    
	    REDUCE_OP1(reduceArray[tid             ], rho * ( u*uu +  v*vv +  w*ww) );
	    REDUCE_OP1(reduceArray[tid+  reduceSize], rho * (uu*uu + vv*vv + ww*ww) );
	    
	    // compute temperature
	    real_t temperature;
	    if (::gParams.cIso >0) {
	      temperature = SQR(::gParams.cIso);
	    } else { // use ideal gas eq of state (P over rho)
	      temperature =  (::gParams.gamma0 - ONE_F) * 
		(e - 0.5 * rho * ( u*u + v*v + w*w ) );
	    }
	    
	    REDUCE_OP1(reduceArray[tid+2*reduceSize], rho * (u*u + v*v + w*w) / temperature );
	    REDUCE_OP1(reduceArray[tid+3*reduceSize], (u*u + v*v + w*w) / temperature );
	    REDUCE_OP1(reduceArray[tid+4*reduceSize], rho * (u*u + v*v + w*w) );
	    REDUCE_OP1(reduceArray[tid+5*reduceSize], u*u + v*v + w*w );
	    REDUCE_OP1(reduceArray[tid+6*reduceSize], rho*rho );
	    REDUCE_OP2(reduceArray[tid+7*reduceSize], rho ); // watch out !
	    REDUCE_OP3(reduceArray[tid+8*reduceSize], rho ); // watch out !
	    
	  } // end iX, iY, iZ guard
	} // end if (i+blockSize < n)

      i += gridSize;
    }
  __syncthreads();

  // do reduction in shared mem
  if(blockSize >= 512) { if(tid < 256) { REDUCE_VAR_T(reduceArray, tid, reduceSize, 256); } __syncthreads(); }
  
  if(blockSize >= 256) { if(tid < 128) { REDUCE_VAR_T(reduceArray, tid, reduceSize, 128); } __syncthreads(); }
  
  if(blockSize >= 128) { if(tid <  64) { REDUCE_VAR_T(reduceArray, tid, reduceSize,  64); } __syncthreads(); }

  if(tid < 32)
    {
      volatile real_t* smem = reduceArray;
      if(blockSize >=  64) { REDUCE_VAR_T(smem, tid, reduceSize, 32); }
      if(blockSize >=  32) { REDUCE_VAR_T(smem, tid, reduceSize, 16); }
      if(blockSize >=  16) { REDUCE_VAR_T(smem, tid, reduceSize,  8); }
      if(blockSize >=   8) { REDUCE_VAR_T(smem, tid, reduceSize,  4); }
      if(blockSize >=   4) { REDUCE_VAR_T(smem, tid, reduceSize,  2); }
      if(blockSize >=   2) { REDUCE_VAR_T(smem, tid, reduceSize,  1); }
    }

  // write result for this block to global mem
  if(tid == 0)
    {
      g_odata[blockIdx.x             ] = reduceArray[0];
      g_odata[blockIdx.x +  gridDim.x] = reduceArray[0+  reduceSize];
      g_odata[blockIdx.x +2*gridDim.x] = reduceArray[0+2*reduceSize];
      g_odata[blockIdx.x +3*gridDim.x] = reduceArray[0+3*reduceSize];
      g_odata[blockIdx.x +4*gridDim.x] = reduceArray[0+4*reduceSize];
      g_odata[blockIdx.x +5*gridDim.x] = reduceArray[0+5*reduceSize];
      g_odata[blockIdx.x +6*gridDim.x] = reduceArray[0+6*reduceSize];
      g_odata[blockIdx.x +7*gridDim.x] = reduceArray[0+7*reduceSize];
      g_odata[blockIdx.x +8*gridDim.x] = reduceArray[0+8*reduceSize];
    }
} // kernel_compute_random_forcing_normalization

#endif // RANDOM_FORCING_CUH_
