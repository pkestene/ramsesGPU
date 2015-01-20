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
 * \file kurganov-tadmor.cuh
 * \brief Implement GPU kernels for the Kurganov-Tadmor central scheme.
 *
 * \date 10/02/2010
 * \author Pierre Kestener.
 *
 * $Id: kurganov-tadmor.cuh 2108 2012-05-23 12:07:21Z pkestene $
 */
#ifndef KURGANOV_TADMOR_CUH_
#define KURGANOV_TADMOR_CUH_

#include "real_type.h"

// for reconstruction kernel 
#define REC_BLOCK_DIMX		32
#define REC_BLOCK_DIMY		14
#define REC_BLOCK_INNER_DIMX	(REC_BLOCK_DIMX-3)
#define REC_BLOCK_INNER_DIMY	(REC_BLOCK_DIMY-3)

// for reconstruction kernel 
#ifdef USE_DOUBLE
#define PC_BLOCK_DIMX		12
#define PC_BLOCK_DIMY		12
#define PC_BLOCK_INNER_DIMX	(PC_BLOCK_DIMX-3)
#define PC_BLOCK_INNER_DIMY	(PC_BLOCK_DIMY-3)
#else // simple precision
#define PC_BLOCK_DIMX		16
#define PC_BLOCK_DIMY		16
#define PC_BLOCK_INNER_DIMX	(PC_BLOCK_DIMX-3)
#define PC_BLOCK_INNER_DIMY	(PC_BLOCK_DIMY-3)
#endif // USE_DOUBLE

// for computeDt reduction kernel (adapted from cmpdt.cuh, this should
// be better parametrized to mutualize code)
#define REDUCE_OP(x, y) x = FMAX(x, y)
#define REDUCE_VAR(array, idx, var, offset) REDUCE_OP(array[idx].var, array[idx+offset].var)
#define REDUCE(array, idx, offset) REDUCE_VAR(array, idx, x, offset); REDUCE_VAR(array, idx, y, offset)

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

#define CMPDT_BLOCK_SIZE	128


#include "constants.h"
#include "constoprim.h"

#include "kurganov-tadmor.h"


#include <cstdlib> // just in case ...

/** 
 * compute dt step time for Kurganov-Tadmor scheme
 * 
 * @param U       : hydro grid data
 * @param g_odata : partial reduction outpub buffer
 * @param n       : 2d array size (take care of pitch)
 */
template<unsigned int blockSize>
__global__ void computeDt_kt_kernel(real_t* U, real2_t* g_odata, unsigned int n)
{
  extern __shared__ real2_t shared[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  shared[tid].x = 0;
  shared[tid].y = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while(i < n)
    {
      real_t u[NVAR_2D];
      real_t rx,ry;
      int offset;
      for (int k=0; k<NVAR_2D; ++k) {
	offset = i + k*n;
	u[k]=U[offset];
      }
      spectral_radii<NVAR_2D>(u, rx, ry);
      REDUCE_OP(shared[tid].x, rx);
      REDUCE_OP(shared[tid].y, ry);

      if(i+blockSize < n)
	{
	  for (int k=0; k<NVAR_2D; ++k) {
	    offset = i + blockSize + k*n;
	    u[k]=U[offset];
	  }
	  spectral_radii<NVAR_2D>(u, rx, ry);
	  REDUCE_OP(shared[tid].x, rx);
	  REDUCE_OP(shared[tid].y, ry);
	}

      i += gridSize;
    }
  __syncthreads();

  // do reduction in shared mem
  if(blockSize >= 512) { if(tid < 256) { REDUCE(shared, tid, 256); } __syncthreads(); }
  if(blockSize >= 256) { if(tid < 128) { REDUCE(shared, tid, 128); } __syncthreads(); }
  if(blockSize >= 128) { if(tid <  64) { REDUCE(shared, tid,  64); } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
  if(tid < 32)
#endif
    {
      if(blockSize >=  64) { REDUCE(shared, tid, 32); EMUSYNC; }
      if(blockSize >=  32) { REDUCE(shared, tid, 16); EMUSYNC; }
      if(blockSize >=  16) { REDUCE(shared, tid,  8); EMUSYNC; }
      if(blockSize >=   8) { REDUCE(shared, tid,  4); EMUSYNC; }
      if(blockSize >=   4) { REDUCE(shared, tid,  2); EMUSYNC; }
      if(blockSize >=   2) { REDUCE(shared, tid,  1); EMUSYNC; }
    }

  // write result for this block to global mem
  if(tid == 0)
    {
      g_odata[blockIdx.x] = shared[0];
    }
}

/** 
 * Reconstruction step kernel for Kurganov-Tadmor scheme
 * 
 * @param U 
 * @param Uhalf
 * @param pitch 
 * @param isize 
 * @param jsize 
 * @param xLambda 
 * @param yLambda 
 */
template<bool odd>
__global__ void reconstruction_2d_FD2_kt_kernel(real_t* U, real_t* Uhalf, int pitch, int isize, int jsize, real_t xLambda, real_t yLambda)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index
  const int i = __mul24(bx, REC_BLOCK_INNER_DIMX) + tx;
  const int j = __mul24(by, REC_BLOCK_INNER_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jsize);
  const int elemOffset = __umul24(pitch, j) + i;
  
  const int nx = isize-4;
  const int ny = jsize-4;
  
  __shared__ real_t Ushared [REC_BLOCK_DIMY][REC_BLOCK_DIMX];
  __shared__ real_t Uprime  [REC_BLOCK_DIMY][REC_BLOCK_DIMX];  
  __shared__ real_t Uqrime  [REC_BLOCK_DIMY][REC_BLOCK_DIMX];  
  
  Uprime[ty][tx] = 0.0f;
  Uqrime[ty][tx] = 0.0f;
  
  for (unsigned int k=0; k < NVAR_2D; ++k) {
    
    /*
     * copy data from global mem to shared mem
     */
    if( (j<jsize) and (i<isize) ) 
      {
	int offset = elemOffset + k*arraySize;
	Ushared[ty][tx] = U[offset];
      }
    __syncthreads();

    
    /*
     * compute Uprime and Uqrime
     */
    if ( (i>=1) and (i<nx+3) and (j>=1) and (j<ny+3) 
	 and tx>0 and tx<REC_BLOCK_DIMX-1
	 and ty>0 and ty<REC_BLOCK_DIMY-1)
      {

	Uprime[ty][tx]=
	  minmod3(gParams.ALPHA_KT*(Ushared[ty][tx+1] - Ushared[ty][tx  ]), 
		  0.5f            *(Ushared[ty][tx+1] - Ushared[ty][tx-1]), 
		  gParams.ALPHA_KT*(Ushared[ty][tx  ] - Ushared[ty][tx-1]));
	Uqrime[ty][tx]=
	  minmod3(gParams.ALPHA_KT*(Ushared[ty+1][tx] - Ushared[ty  ][tx]), 
		  0.5f            *(Ushared[ty+1][tx] - Ushared[ty-1][tx]), 
		  gParams.ALPHA_KT*(Ushared[ty  ][tx] - Ushared[ty-1][tx]));
	
      }
    __syncthreads();
    
 
    /*
     * compute Uhalf
     */
    if (odd) {

      if ( (i>=1) and (i<nx+2) and (j>=1) and (j<ny+2) 
	   and tx>=1 and tx<REC_BLOCK_DIMX-2
	   and ty>=1 and ty<REC_BLOCK_DIMY-2)
	{
	  real_t tmp;	  
	  tmp = 0.25f*((Ushared[ty  ][tx  ]  +
			Ushared[ty  ][tx+1]  +
			Ushared[ty+1][tx  ]  +
			Ushared[ty+1][tx+1]) +
		       0.25f*((Uprime[ty  ][tx  ] - Uprime[ty  ][tx+1]) +
			      (Uprime[ty+1][tx  ] - Uprime[ty+1][tx+1]) +
			      (Uqrime[ty  ][tx  ] - Uqrime[ty+1][tx  ]) +
			      (Uqrime[ty  ][tx+1] - Uqrime[ty+1][tx+1])));

	  int offset = elemOffset + k*arraySize;
	  Uhalf[offset] = tmp;

	}

    } else {

      if ( (i>=2) and (i<nx+3) and (j>=2) and (j<ny+3) 
	   and tx>1 and tx<REC_BLOCK_DIMX-1
	   and ty>1 and ty<REC_BLOCK_DIMY-1)
	{
	  real_t tmp;
	  tmp = 0.25f*((Ushared[ty-1][tx  ]  +
			Ushared[ty-1][tx-1]  +
			Ushared[ty  ][tx  ]  +
			Ushared[ty  ][tx-1]) +
		       0.25f*((Uprime[ty-1][tx-1] - Uprime[ty-1][tx  ]) +
			      (Uprime[ty  ][tx-1] - Uprime[ty  ][tx  ]) +
			      (Uqrime[ty-1][tx-1] - Uqrime[ty  ][tx-1]) +
			      (Uqrime[ty-1][tx  ] - Uqrime[ty  ][tx  ])));
	  
	  int offset = elemOffset + k*arraySize;
	  Uhalf[offset] = tmp;
	  
	}      
      
    } // end if(odd) ...
    __syncthreads();

  } // end for (int k ...

} // reconstruction_kt_kernel


template<bool odd>
__global__ void predictor_corrector_2d_FD2_kt_kernel(real_t* U, real_t* Uhalf, int pitch, int isize, int jsize, real_t xLambda, real_t yLambda)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index
  const int i = __mul24(bx, PC_BLOCK_INNER_DIMX) + tx;
  const int j = __mul24(by, PC_BLOCK_INNER_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jsize);
  const int elemOffset = __umul24(pitch, j) + i;
  
  const int nx = isize-4;
  const int ny = jsize-4;

  real_t u[NVAR_2D];
  real_t flux_x[NVAR_2D], flux_y[NVAR_2D];

  
  /*
   * copy data from global mem to shared mem
   */
  __shared__ real_t f[NVAR_2D][PC_BLOCK_DIMY][PC_BLOCK_DIMX];
  __shared__ real_t g[NVAR_2D][PC_BLOCK_DIMY][PC_BLOCK_DIMX];
  if( (j<jsize) and (i<isize) ) 
    {
      for (unsigned int k=0; k < NVAR_2D; ++k) {
	int offset = elemOffset + k*arraySize;
	u[k] = U[offset];
      }

      get_flux<NVAR_2D>(u, flux_x, flux_y);

      for (unsigned int k=0; k<NVAR_2D; k++) {
	f[k][ty][tx]=flux_x[k];
	g[k][ty][tx]=flux_y[k];
      }
    }
  __syncthreads();

  //calculate flux derivatives  
  real_t fu, fv, fw;
  real_t gu, gv, gw;
  real_t f_prime, g_qrime;
  if ( (i>=1) and (i<nx+3) and (j>=1) and (j<ny+3) 
	 and tx>0 and tx<PC_BLOCK_DIMX-1
	 and ty>0 and ty<PC_BLOCK_DIMY-1)
    {
      for (int k=0; k<NVAR_2D; k++) {
	fu      = gParams.ALPHA_KT*(f[k][ty][tx+1] - f[k][ty][tx  ]);
	fv      = 0.5f            *(f[k][ty][tx+1] - f[k][ty][tx-1]);
	fw      = gParams.ALPHA_KT*(f[k][ty][tx  ] - f[k][ty][tx-1]);
	f_prime = minmod3(fu, fv, fw);	
	
	gu      = gParams.ALPHA_KT*(g[k][ty+1][tx] - g[k][ty  ][tx]);
	gv      = 0.5f            *(g[k][ty+1][tx] - g[k][ty-1][tx]);
	gw      = gParams.ALPHA_KT*(g[k][ty  ][tx] - g[k][ty-1][tx]);
	g_qrime = minmod3(gu, gv, gw);
	
	// compute predicted value : Ustar
	u[k]   -= 0.5f*(xLambda*f_prime);
	u[k]   -= 0.5f*(yLambda*g_qrime);
      
      }

      // corrector step
      get_flux<NVAR_2D>(u, flux_x, flux_y);
      for (unsigned int k=0; k<NVAR_2D; k++) {
	f[k][ty][tx]=flux_x[k];
	g[k][ty][tx]=flux_y[k];
     
      }

    }
  __syncthreads();

  // finally update U
  if (odd) {
    if ( (i>=1) and (i<nx+2) and (j>=1) and (j<ny+2) 
	 and tx>0 and tx<PC_BLOCK_DIMX-2
	 and ty>0 and ty<PC_BLOCK_DIMY-2)
      {
	for (int k=0; k<NVAR_2D; k++) {
	  int offset = elemOffset + k*arraySize;
	  U[offset] = Uhalf[offset] - 
	    0.5f*(xLambda*((f[k][ty  ][tx+1] - f[k][ty  ][tx  ])  +
			   (f[k][ty+1][tx+1] - f[k][ty+1][tx  ])) + 
		  yLambda*((g[k][ty+1][tx  ] - g[k][ty  ][tx  ])  + 
			   (g[k][ty+1][tx+1] - g[k][ty  ][tx+1])));
	}
      }
  } else {
    if ( (i>=2) and (i<nx+3) and (j>=2) and (j<ny+3) 
	 and tx>1 and tx<PC_BLOCK_DIMX-1
	 and ty>1 and ty<PC_BLOCK_DIMY-1)
      {
	for (int k=0; k<NVAR_2D; k++) {
	  int offset = elemOffset + k*arraySize;
	  U[offset]= Uhalf[offset] -
	    0.5f*(xLambda*((f[k][ty-1][tx  ] - f[k][ty-1][tx-1])  +
			   (f[k][ty  ][tx  ] - f[k][ty  ][tx-1])) +
		  yLambda*((g[k][ty  ][tx-1] - g[k][ty-1][tx-1])  +
			   (g[k][ty  ][tx  ] - g[k][ty-1][tx  ])));
	}
      }
  } // if (odd) ...
  __syncthreads();

} // predictor_corrector_2d_FD2_kt_kernel

#endif // KURGANOV_TADMOR_CUH_
