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
 * \file godunov_trace_v1.cuh
 * \brief Defines the CUDA kernel for the actual Godunov scheme
 * computations using trace computation (necessary to get a second
 * order scheme !).
 *
 * The original kernels without trace are located in godunov_notrace.cuh
 *
 * \date October 2010
 * \author P. Kestener
 *
 * $Id: godunov_trace_v1.cuh 2108 2012-05-23 12:07:21Z pkestene $
 */
#ifndef GODUNOV_TRACE_V1_CUH_
#define GODUNOV_TRACE_V1_CUH_

// Note: T1 (Trace 1) prefix is used not to confuse with macros from
// godunov_notrace.cuh !!!

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define T1_HBLOCK_DIMX		16
#define T1_HBLOCK_DIMY		15
#define T1_HBLOCK_INNER_DIMX	14

#define T1_VBLOCK_DIMX		16
#define T1_VBLOCK_DIMY		15
#define T1_VBLOCK_INNER_DIMY	13
#else // simple precision
#define T1_HBLOCK_DIMX		16
#define T1_HBLOCK_DIMY		30
#define T1_HBLOCK_INNER_DIMX	14

#define T1_VBLOCK_DIMX		16
#define T1_VBLOCK_DIMY		31
#define T1_VBLOCK_INNER_DIMY	29
#endif // USE_DOUBLE

#include "real_type.h"
#include "constants.h"
#include "constoprim.h"
#include "riemann.h"
#include "trace.h"

#include <cstdlib>

/**
 * Directionally split Godunov kernel along X-direction for 2D data
 */
__global__ void godunov_x_2d_v1(real_t* U, real_t* UOut,
				int pitch, int imax, int jmax, 
				const real_t dtdx, const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, T1_HBLOCK_INNER_DIMX) + tx;
  const int j = __mul24(by, T1_HBLOCK_DIMY      ) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;
  
  __shared__ real_t qxm[T1_HBLOCK_DIMX][T1_HBLOCK_DIMY][NVAR_2D];
  real_t qxp[NVAR_2D];
  real_t rhoOld;
  real_t uOld[NVAR_2D];

  if(j >= 2 and j < jmax-2 and 
     i >  0 and i < imax-1)
    {
      // conservative variables
      real_t u[NVAR_2D];
      // primitive variables
      real_t q[NVAR_2D];
      real_t qPlus[NVAR_2D], qMinus[NVAR_2D];
      real_t c, cPlus, cMinus;
      
      // Gather conservative variables and convert to primitive variables
      int offset = elemOffset;
      uOld[ID] = U[offset];  offset += arraySize;
      uOld[IP] = U[offset];  offset += arraySize;
      uOld[IU] = U[offset];  offset += arraySize;
      uOld[IV] = U[offset];
      constoprim_2D(uOld, q, c);
      rhoOld = uOld[ID];

      // next cell along X-direction
      offset = elemOffset+1;
      u[ID] = U[offset];  offset += arraySize;
      u[IP] = U[offset];  offset += arraySize;
      u[IU] = U[offset];  offset += arraySize;
      u[IV] = U[offset];
      constoprim_2D(u, qPlus, cPlus);
      
      // previous cell along X-direction
      offset = elemOffset-1;
      u[ID] = U[offset];  offset += arraySize;
      u[IP] = U[offset];  offset += arraySize;
      u[IU] = U[offset];  offset += arraySize;
      u[IV] = U[offset];
      constoprim_2D(u, qMinus, cMinus);
      
      // Characteristic tracing (compute qxm and qxp)
      trace<NVAR_2D>(q, qPlus, qMinus, c, dtdx, qxm[tx][ty], qxp);
    }  
  __syncthreads();
  
  __shared__ real_t flux[T1_HBLOCK_DIMX][T1_HBLOCK_DIMY][NVAR_2D];
  if(j >= 2 and j < jmax-2 and 
     i >  0 and i < imax   and tx > 0)
    {
      real_t (&qleft)[NVAR_2D] = qxm[tx-1][ty];
      real_t (&qright)[NVAR_2D] = qxp;
      
      // Solve Riemann problem at interfaces and compute fluxes
      real_t qgdnv[NVAR_2D];
      riemann<NVAR_2D>(qleft, qright, qgdnv, flux[tx][ty]);
    }  
  __syncthreads();
  
  if(j >= 2 and j < jmax-2 and 
     i >= 2 and i < imax-2 and tx > 0 and tx < T1_HBLOCK_DIMX-1)
    {
      // Update conservative variables
      int offset = elemOffset;
      UOut[offset] = uOld[ID] + (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx; offset += arraySize;
      UOut[offset] = uOld[IP] + (flux[tx][ty][IP] - flux[tx+1][ty][IP]) * dtdx; offset += arraySize;
      UOut[offset] = uOld[IU] + (flux[tx][ty][IU] - flux[tx+1][ty][IU]) * dtdx; offset += arraySize;
      UOut[offset] = uOld[IV] + (flux[tx][ty][IV] - flux[tx+1][ty][IV]) * dtdx;

      // update momentum using gravity predictor
      real_t rhoDelta = HALF_F * (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx;
      offset = elemOffset + 2*arraySize;
      UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
      UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_y*dt;

    }
} // godunov_x_2d_v1

/**
 * Directionally split Godunov kernel along Y-direction for 2D data.
 */
__global__ void godunov_y_2d_v1(real_t* U, real_t* UOut,
				int pitch, int imax, int jmax, 
				const real_t dtdx, const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, T1_VBLOCK_DIMX      ) + tx;
  const int j = __mul24(by, T1_VBLOCK_INNER_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;
  
  __shared__ real_t qxm[T1_VBLOCK_DIMX][T1_VBLOCK_DIMY][NVAR_2D];
  real_t qxp[NVAR_2D];
  real_t rhoOld;
  real_t uOld[NVAR_2D];

  if(i >= 2 and i < imax-2 and 
     j >  0 and j < jmax-1)
    {
      // conservative variables
      real_t u[NVAR_2D];
      // primitive variables
      real_t q[NVAR_2D];
      real_t qPlus[NVAR_2D], qMinus[NVAR_2D];
      real_t c, cPlus, cMinus;
      
      // Gather conservative variables and convert to primitive variables
      int offset = elemOffset;
      uOld[ID] = U[offset]; offset += arraySize;
      uOld[IP] = U[offset]; offset += arraySize;
      uOld[IV] = U[offset]; offset += arraySize; // watchout! IU and IV are swapped !
      uOld[IU] = U[offset];
      constoprim_2D(uOld, q, c);
      rhoOld = uOld[ID];

      // next cell along Y-direction
      offset = elemOffset+pitch;
      u[ID] = U[offset]; offset += arraySize;
      u[IP] = U[offset]; offset += arraySize;
      u[IV] = U[offset]; offset += arraySize; // IU and IV are swapped !
      u[IU] = U[offset];
      constoprim_2D(u, qPlus, cPlus);

      // previous cell along Y-direction
      offset = elemOffset-pitch;
      u[ID] = U[offset]; offset += arraySize;
      u[IP] = U[offset]; offset += arraySize;
      u[IV] = U[offset]; offset += arraySize; // IU and IV are swapped !
      u[IU] = U[offset];
      constoprim_2D(u, qMinus, cMinus);

      // Characteristic tracing (compute qxm and qxp)
      trace<NVAR_2D>(q, qPlus, qMinus, c, dtdx, qxm[tx][ty], qxp);
    }
  __syncthreads();
  
  __shared__ real_t flux[T1_VBLOCK_DIMX][T1_VBLOCK_DIMY][NVAR_2D];
  if(i >= 2 and i < imax-2 and 
     j >  0 and j < jmax   and ty > 0)
    {
      real_t (&qleft)[NVAR_2D]  = qxm[tx][ty-1];
      real_t (&qright)[NVAR_2D] = qxp;
      
      // Solve Riemann problem at interfaces and compute fluxes
      real_t qgdnv[NVAR_2D];
      riemann<NVAR_2D>(qleft, qright, qgdnv, flux[tx][ty]);
    }  
  __syncthreads();
  
  if(i >= 2 and i < imax-2 and 
     j >= 2 and j < jmax-2 and ty > 0 and ty < T1_VBLOCK_DIMY-1)
    {
      // Update conservative variables : watchout! IU and IV are swapped !
      int offset = elemOffset;
      UOut[offset] = uOld[ID] + (flux[tx][ty][ID]-flux[tx][ty+1][ID]) * dtdx; offset += arraySize;
      UOut[offset] = uOld[IP] + (flux[tx][ty][IP]-flux[tx][ty+1][IP]) * dtdx; offset += arraySize;
      UOut[offset] = uOld[IV] + (flux[tx][ty][IV]-flux[tx][ty+1][IV]) * dtdx; offset += arraySize; 
      UOut[offset] = uOld[IU] + (flux[tx][ty][IU]-flux[tx][ty+1][IU]) * dtdx;

      // update momentum using gravity predictor
      real_t rhoDelta = HALF_F * (flux[tx][ty][ID] - flux[tx][ty+1][ID]) * dtdx;
      offset = elemOffset + 2*arraySize;
      UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
      UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_y*dt;

    }
} // godunov_y_2d_v1


/******************************************
 *** *** *** GODUNOV 3D KERNELS *** *** ***
 ******************************************/

// 3D-kernel block dimensions

// #undef T1_HBLOCK_DIMX
// #undef T1_HBLOCK_DIMY
// #undef T1_HBLOCK_INNER_DIMX
// #undef T1_VBLOCK_DIMX
// #undef T1_VBLOCK_DIMY
// #undef T1_VBLOCK_INNER_DIMY

#ifdef USE_DOUBLE

#define T1_XDIR_BLOCK_DIMX_3D      	16
#define T1_XDIR_BLOCK_DIMY_3D      	15
#define T1_XDIR_BLOCK_INNER_DIMX_3D	14

#define T1_YDIR_BLOCK_DIMX_3D      	16
#define T1_YDIR_BLOCK_DIMY_3D      	15
#define T1_YDIR_BLOCK_INNER_DIMY_3D	13

#define T1_ZDIR_BLOCK_DIMX_3D		16
#define T1_ZDIR_BLOCK_DIMZ_3D		25
#define T1_ZDIR_BLOCK_INNER_DIMZ_3D	23

#else // SINGLE PRECISION

#define T1_XDIR_BLOCK_DIMX_3D		16
#define T1_XDIR_BLOCK_DIMY_3D		24
#define T1_XDIR_BLOCK_INNER_DIMX_3D	14

#define T1_YDIR_BLOCK_DIMX_3D		16
#define T1_YDIR_BLOCK_DIMY_3D		25
#define T1_YDIR_BLOCK_INNER_DIMY_3D	23

#define T1_ZDIR_BLOCK_DIMX_3D		16
#define T1_ZDIR_BLOCK_DIMZ_3D		25
#define T1_ZDIR_BLOCK_INNER_DIMZ_3D	23

#endif // USE_DOUBLE

/**
 * Directionally split Godunov kernel along X-direction for 3D data
 * This kernel does essentially the same computation as godunov_x_2d,
 * except it loops over all posible X-Y planes through the
 * Z-direction.
 * Also note that we must decrease the block dimensions to cope the
 * shared memory limitation.
 */
__global__ void godunov_x_3d_v1(real_t* U, real_t* UOut,
				int pitch, 
				int imax, int jmax, int kmax, 
				const real_t dtdx, const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, T1_XDIR_BLOCK_INNER_DIMX_3D) + tx;
  const int j = __mul24(by, T1_XDIR_BLOCK_DIMY_3D      ) + ty;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t  qxm[T1_XDIR_BLOCK_DIMX_3D][T1_XDIR_BLOCK_DIMY_3D][NVAR_3D];
  real_t qxp[NVAR_3D];
  __shared__ real_t flux[T1_XDIR_BLOCK_DIMX_3D][T1_XDIR_BLOCK_DIMY_3D][NVAR_3D];
  real_t rhoOld;
  real_t uOld[NVAR_3D];

  // loop over all X-Y-planes
  for (int k=2, elemOffset = i + pitch * (j + jmax * 2);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    // 3D array offset (do it in the for loop)
    // int elemOffset = i + pitch * (j + jmax * k);

    // take into account the ghost cells
    if(j >= 2 and j < jmax-2 and
       i >  0 and i < imax-1)
      {
	// conservative variables
	real_t u[NVAR_3D];

	// primitive variables
	real_t q[NVAR_3D], qPlus[NVAR_3D], qMinus[NVAR_3D];
	real_t c, cPlus, cMinus;

	// Gather conservative variables and convert to primitive variables
	int offset = elemOffset;
	uOld[ID] = U[offset];  offset += arraySize;
	uOld[IP] = U[offset];  offset += arraySize;
	uOld[IU] = U[offset];  offset += arraySize;
	uOld[IV] = U[offset];  offset += arraySize;
	uOld[IW] = U[offset];
	constoprim_3D(uOld, q, c);
	rhoOld = uOld[ID];

	// next cell along X-direction
	offset = elemOffset+1;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize;
	u[IU] = U[offset];  offset += arraySize;
	u[IV] = U[offset];  offset += arraySize;
	u[IW] = U[offset];
	constoprim_3D(u, qPlus, cPlus);
      
	// previous cell along X-direction
	offset = elemOffset-1;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize;
	u[IU] = U[offset];  offset += arraySize;
	u[IV] = U[offset];  offset += arraySize;
	u[IW] = U[offset];
	constoprim_3D(u, qMinus, cMinus);

	// Characteristic tracing (compute qxm and qxp)
	trace<NVAR_3D>(q, qPlus, qMinus, c, dtdx, qxm[tx][ty], qxp);
      }    
    __syncthreads();
    
    
    if(j >= 2 and j < jmax-2 and 
       i >  0 and i < imax   and tx > 0)
      {
	real_t (&qleft)[NVAR_3D]  = qxm[tx-1][ty];
	real_t (&qright)[NVAR_3D] = qxp;
	
	// Solve Riemann problem at interfaces and compute fluxes
	real_t qgdnv[NVAR_3D];
	riemann<NVAR_3D>(qleft, qright, qgdnv, flux[tx][ty]);
      }    
    __syncthreads();
    
    if(j >= 2 and j < jmax-2 and 
       i >= 2 and i < imax-2 and tx > 0 and tx < T1_XDIR_BLOCK_DIMX_3D-1)
      {
	// Update conservative variables
	int offset = elemOffset;
	UOut[offset] = uOld[ID] + (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IP] + (flux[tx][ty][IP] - flux[tx+1][ty][IP]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IU] + (flux[tx][ty][IU] - flux[tx+1][ty][IU]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IV] + (flux[tx][ty][IV] - flux[tx+1][ty][IV]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IW] + (flux[tx][ty][IW] - flux[tx+1][ty][IW]) * dtdx;

	// update momentum using gravity predictor
	real_t rhoDelta = HALF_F * (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx;
	offset = elemOffset + 2*arraySize;
	UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
	UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_y*dt; offset += arraySize;
	UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_z*dt;

      }

  } // end loop over k (X-Y-planes location)

} // godunov_x_3d_v1

/**
 * Directionally split Godunov kernel along Y-direction for 3D data
 * This kernel does essentially the same computation as godunov_x_3d,
 * except we have swapped x and y.
 *
 * Also note that we must decrease the block dimensions to cope the
 * shared memory limitation.
 *
 */
__global__ void godunov_y_3d_v1(real_t* U, real_t* UOut,
				int pitch, 
				int imax, int jmax, int kmax, 
				const real_t dtdx, const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, T1_YDIR_BLOCK_DIMX_3D      ) + tx;
  const int j = __mul24(by, T1_YDIR_BLOCK_INNER_DIMY_3D) + ty;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t  qxm[T1_YDIR_BLOCK_DIMX_3D][T1_YDIR_BLOCK_DIMY_3D][NVAR_3D];
  real_t qxp[NVAR_3D];
  __shared__ real_t flux[T1_YDIR_BLOCK_DIMX_3D][T1_YDIR_BLOCK_DIMY_3D][NVAR_3D];
  real_t rhoOld;
  real_t uOld[NVAR_3D];

  // loop over all X-Y-planes
  for (int k=2, elemOffset = i + pitch * (j + jmax * 2);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    // 3D array offset (do it in the for loop)
    // int elemOffset = i + pitch * (j + jmax * k);

    // take into account the ghost cells
    if(i >= 2 and i < imax-2 and
       j >  0 and j < jmax-1)
      {
	// conservative variables
	real_t u[NVAR_3D];
	
	// primitive variables
	real_t q[NVAR_3D], qPlus[NVAR_3D], qMinus[NVAR_3D];
	real_t c, cPlus, cMinus;

	// Gather conservative variables and convert to primitive variables
	int offset = elemOffset;
	uOld[ID] = U[offset];  offset += arraySize;
	uOld[IP] = U[offset];  offset += arraySize;
	uOld[IV] = U[offset];  offset += arraySize; // Watchout IU and IV swapped !
	uOld[IU] = U[offset];  offset += arraySize;
	uOld[IW] = U[offset];
	constoprim_3D(uOld, q, c);
	rhoOld = uOld[ID];

	// next cell along Y-direction
	offset = elemOffset+pitch;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize;
	u[IV] = U[offset];  offset += arraySize; // Watchout IU and IV swapped !
	u[IU] = U[offset];  offset += arraySize;
	u[IW] = U[offset];
	constoprim_3D(u, qPlus, cPlus);

	// previous cell along Y-direction
	offset = elemOffset-pitch;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize;
	u[IV] = U[offset];  offset += arraySize; // Watchout IU and IV swapped !
	u[IU] = U[offset];  offset += arraySize;
	u[IW] = U[offset];
	constoprim_3D(u, qMinus, cMinus);

	// Characteristic tracing (compute qxm and qxp)
	trace<NVAR_3D>(q, qPlus, qMinus, c, dtdx, qxm[tx][ty], qxp);
      }    
    __syncthreads();
    
    
    if(i >= 2 and i < imax-2 and 
       j >  0 and j < jmax   and ty > 0)
      {
	real_t (&qleft)[NVAR_3D]  = qxm[tx][ty-1];
	real_t (&qright)[NVAR_3D] = qxp;
	
	// Solve Riemann problem at interfaces and compute fluxes
	real_t qgdnv[NVAR_3D];
	riemann<NVAR_3D>(qleft, qright, qgdnv, flux[tx][ty]);
      }
    
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and 
       j >= 2 and j < jmax-2 and ty > 0 and ty < T1_YDIR_BLOCK_DIMY_3D-1)
      {
	// Update conservative variables : watchout IU and IV are swapped
	int offset = elemOffset;
	UOut[offset] = uOld[ID] + (flux[tx][ty][ID] - flux[tx][ty+1][ID]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IP] + (flux[tx][ty][IP] - flux[tx][ty+1][IP]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IV] + (flux[tx][ty][IV] - flux[tx][ty+1][IV]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IU] + (flux[tx][ty][IU] - flux[tx][ty+1][IU]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IW] + (flux[tx][ty][IW] - flux[tx][ty+1][IW]) * dtdx;

	// update momentum using gravity predictor
	real_t rhoDelta = HALF_F * (flux[tx][ty][ID] - flux[tx][ty+1][ID]) * dtdx;
	offset = elemOffset + 2*arraySize;
	UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
	UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_y*dt; offset += arraySize;
	UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_z*dt;	
      }

  } // end loop over k (X-Y-planes location)

} // godunov_y_3d_v1

/**
 * Directionally split Godunov kernel along Z-direction for 3D data
 * This kernel does essentially the same computation as godunov_y_3d,
 * except we have swapped y and z.
 *
 */
__global__ void godunov_z_3d_v1(real_t* U, real_t* UOut,
				int pitch, 
				int imax, int jmax, int kmax, 
				const real_t dtdx, const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int bz = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int tz = threadIdx.y;

  const int i = __mul24(bx, T1_ZDIR_BLOCK_DIMX_3D      ) + tx;
  const int k = __mul24(bz, T1_ZDIR_BLOCK_INNER_DIMZ_3D) + tz;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t  qxm[T1_ZDIR_BLOCK_DIMX_3D][T1_ZDIR_BLOCK_DIMZ_3D][NVAR_3D];
  real_t qxp[NVAR_3D];
  __shared__ real_t flux[T1_ZDIR_BLOCK_DIMX_3D][T1_ZDIR_BLOCK_DIMZ_3D][NVAR_3D];
  real_t rhoOld;
  real_t uOld[NVAR_3D];

  // loop over all X-Z-planes
  for (int j=2, elemOffset = i + pitch * (2 + jmax * k);
       j < jmax-2; 
       ++j, elemOffset += pitch) {
    
    // 3D array offset (do it in the for loop)
    // int elemOffset = i + pitch * (j + jmax * k);

    // take into account the ghost cells
    if(i >= 2 and i < imax-2 and
       k >  0 and k < kmax-1)
      {
	// conservative variables
	real_t u[NVAR_3D];

	// primitive variables
	real_t q[NVAR_3D], qPlus[NVAR_3D], qMinus[NVAR_3D];
	real_t c, cPlus, cMinus;

	// Gather conservative variables and convert to primitive variables
	int offset = elemOffset;
	uOld[ID] = U[offset];  offset += arraySize;
	uOld[IP] = U[offset];  offset += arraySize; 
	uOld[IW] = U[offset];  offset += arraySize; // Watchout IU and IW swapped !
	uOld[IV] = U[offset];  offset += arraySize;
	uOld[IU] = U[offset];
	constoprim_3D(uOld, q, c);
	rhoOld = uOld[ID];
	
	// next cell along Z-direction
	offset = elemOffset+pitch*jmax;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize; 
	u[IW] = U[offset];  offset += arraySize; // Watchout IU and IW swapped !
	u[IV] = U[offset];  offset += arraySize;
	u[IU] = U[offset];
	constoprim_3D(u, qPlus, cPlus);

	// previous cell along Z-direction
	offset = elemOffset-pitch*jmax;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize; 
	u[IW] = U[offset];  offset += arraySize; // Watchout IU and IW swapped !
	u[IV] = U[offset];  offset += arraySize;
	u[IU] = U[offset];
	constoprim_3D(u, qMinus, cMinus);

	// Characteristic tracing
	trace<NVAR_3D>(q, qPlus, qMinus, c, dtdx, qxm[tx][tz], qxp);
      }    
    __syncthreads();
    
    
    if(i >= 2 and i < imax-2 and 
       k >  0 and k < kmax   and tz > 0)
      {
	real_t (&qleft)[NVAR_3D]  = qxm[tx][tz-1];
	real_t (&qright)[NVAR_3D] = qxp;
	
	// Solve Riemann problem at interfaces and compute fluxes
	real_t qgdnv[NVAR_3D];
	riemann<NVAR_3D>(qleft, qright, qgdnv, flux[tx][tz]);
      }    
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and 
       k >= 2 and k < kmax-2 and tz > 0 and tz < T1_ZDIR_BLOCK_DIMZ_3D-1)
      {
	// Update conservative variables : watchout IU and IW are swapped
	int offset = elemOffset;
	UOut[offset] = uOld[ID] + (flux[tx][tz][ID] - flux[tx][tz+1][ID]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IP] + (flux[tx][tz][IP] - flux[tx][tz+1][IP]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IW] + (flux[tx][tz][IW] - flux[tx][tz+1][IW]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IV] + (flux[tx][tz][IV] - flux[tx][tz+1][IV]) * dtdx; offset += arraySize;
	UOut[offset] = uOld[IU] + (flux[tx][tz][IU] - flux[tx][tz+1][IU]) * dtdx;
	
	// update momentum using gravity predictor
	real_t rhoDelta = HALF_F * (flux[tx][tz][ID] - flux[tx][tz+1][ID]) * dtdx;
	offset = elemOffset + 2*arraySize;
	UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
	UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_y*dt; offset += arraySize;
	UOut[offset] += (rhoOld+rhoDelta)*gParams.gravity_z*dt;		
      }

  } // end loop over j (X-Z-planes location)
  
} // godunov_z_3d_v1

#endif /*GODUNOV_TRACE_V1_CUH_*/
