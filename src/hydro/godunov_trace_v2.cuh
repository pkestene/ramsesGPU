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
 * \file godunov_trace_v2.cuh
 * \brief Defines the CUDA kernel for the actual Godunov scheme
 * computations using trace computation (necessary to get a second
 * order scheme !).
 *
 * The kernels defined here are functionally the same as the one in
 * godunov_trace_v1.cuh
 *
 * The original kernels without trace are located in godunov_notrace.cuh
 *
 * \date October 2010
 * \author P. Kestener
 *
 * $Id: godunov_trace_v2.cuh 2108 2012-05-23 12:07:21Z pkestene $
 */
#ifndef GODUNOV_TRACE_V2_CUH_
#define GODUNOV_TRACE_V2_CUH_

// Note: T2 (Trace 2) prefix is used not to confuse with macros from
// godunov_notrace.cuh !!!

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define T2_HBLOCK_DIMX		16
#define T2_HBLOCK_DIMY		15
#define T2_HBLOCK_INNER_DIMX	12

#define T2_VBLOCK_DIMX		16
#define T2_VBLOCK_DIMY		15
#define T2_VBLOCK_INNER_DIMY	11
#else // simple precision
#define T2_HBLOCK_DIMX		16
#define T2_HBLOCK_DIMY		30
#define T2_HBLOCK_INNER_DIMX	12

#define T2_VBLOCK_DIMX		16
#define T2_VBLOCK_DIMY		31
#define T2_VBLOCK_INNER_DIMY	27
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
__global__ void godunov_x_2d_v2(real_t* U, real_t* UOut,
				int pitch, int imax, int jmax, 
				const real_t dtdx, const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, T2_HBLOCK_INNER_DIMX) + tx;
  const int j = __mul24(by, T2_HBLOCK_DIMY      ) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;
  
  __shared__ real_t   q[T2_HBLOCK_DIMX][T2_HBLOCK_DIMY][NVAR_2D];
  __shared__ real_t qxm[T2_HBLOCK_DIMX][T2_HBLOCK_DIMY][NVAR_2D];
  real_t qxp[NVAR_2D];
  real_t c;

  qxm[tx][ty][ID] = ZERO_F;
  qxm[tx][ty][IP] = ZERO_F;
  qxm[tx][ty][IU] = ZERO_F;
  qxm[tx][ty][IV] = ZERO_F;
  
  // conservative variables
  real_t u[NVAR_2D];

  // load U and convert to primitive variables
  if(j >= 2 and j < jmax-2 and 
     i >= 0 and i < imax)
    {
      
      // Gather conservative variables
      int offset = elemOffset;
      u[ID] = U[offset];  offset += arraySize;
      u[IP] = U[offset];  offset += arraySize;
      u[IU] = U[offset];  offset += arraySize;
      u[IV] = U[offset];
      
      //Convert to primitive variables
      constoprim_2D(u, q[tx][ty], c);
    }
  __syncthreads();
    
    
  if(j >= 2 and j < jmax-2 and 
     i >  0 and i < imax-1 and tx > 0 and tx < T2_HBLOCK_DIMX-1)
    {
            
      // Characteristic tracing (compute qxm and qxp)
      trace<NVAR_2D>(q[tx][ty], q[tx+1][ty], q[tx-1][ty], c, dtdx, qxm[tx][ty], qxp);
    }
  __syncthreads();
  
  //__shared__ real_t flux[T2_HBLOCK_DIMX][T2_HBLOCK_DIMY][NVAR_2D];
  // re-use q as the flux array;
  real_t (&flux)[T2_HBLOCK_DIMX][T2_HBLOCK_DIMY][NVAR_2D] = q; 
  flux[tx][ty][ID] = ZERO_F;
  flux[tx][ty][IP] = ZERO_F;
  flux[tx][ty][IU] = ZERO_F;
  flux[tx][ty][IV] = ZERO_F;
  __syncthreads();

  if(j >= 2 and j < jmax-2 and 
     i >  0 and i < imax   and tx > 1 and tx < T2_HBLOCK_DIMX-1)
    {
      real_t (&qleft)[NVAR_2D] = qxm[tx-1][ty];
      real_t (&qright)[NVAR_2D] = qxp;
      
      // Solve Riemann problem at interfaces and compute fluxes
      real_t qgdnv[NVAR_2D];
      riemann<NVAR_2D>(qleft, qright, qgdnv, flux[tx][ty]);
    }  
  __syncthreads();
  
  if(j >= 2 and j < jmax-2 and 
     i >= 2 and i < imax-2 and tx > 1 and tx < T2_HBLOCK_DIMX-2)
    {
      // Update conservative variables
      int offset = elemOffset;
      UOut[offset] = u[ID] + (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx; offset += arraySize;
      UOut[offset] = u[IP] + (flux[tx][ty][IP] - flux[tx+1][ty][IP]) * dtdx; offset += arraySize;
      UOut[offset] = u[IU] + (flux[tx][ty][IU] - flux[tx+1][ty][IU]) * dtdx; offset += arraySize;
      UOut[offset] = u[IV] + (flux[tx][ty][IV] - flux[tx+1][ty][IV]) * dtdx;
  
      // update momentum using gravity predictor
      real_t rhoDelta = HALF_F * (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx;
      offset = elemOffset + 2*arraySize;
      UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
      UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_y*dt;

  }
} // godunov_x_2d_v2

/**
 * Directionally split Godunov kernel along Y-direction for 2D data
 */
__global__ void godunov_y_2d_v2(real_t* U, real_t* UOut,
				int pitch, int imax, int jmax, 
				const real_t dtdx, const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, T2_VBLOCK_DIMX      ) + tx;
  const int j = __mul24(by, T2_VBLOCK_INNER_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;
  
  __shared__ real_t   q[T2_VBLOCK_DIMX][T2_VBLOCK_DIMY][NVAR_2D];
  __shared__ real_t qxm[T2_VBLOCK_DIMX][T2_VBLOCK_DIMY][NVAR_2D];
  real_t qxp[NVAR_2D];
  real_t c;
  
  qxm[tx][ty][ID] = ZERO_F;
  qxm[tx][ty][IP] = ZERO_F;
  qxm[tx][ty][IU] = ZERO_F;
  qxm[tx][ty][IV] = ZERO_F;

  // conservative variables
  real_t u[NVAR_2D];

  // load U and convert to primitive variables
  if(i >= 2 and i < imax-2 and 
     j >= 0 and j < jmax)
    {            
      // Gather conservative variables
      int offset = elemOffset;
      u[ID] = U[offset]; offset += arraySize;
      u[IP] = U[offset]; offset += arraySize;
      u[IV] = U[offset]; offset += arraySize; // watchout! IU and IV are swapped !
      u[IU] = U[offset];
      
      //Convert to primitive variables
      constoprim_2D(u, q[tx][ty], c);
    }
  __syncthreads();
  
  if(i >= 2 and i < imax-2 and 
     j >  0 and j < jmax-1 and ty > 0 and ty < T2_VBLOCK_DIMY-1) 
    {
      // Characteristic tracing (compute qxm and qxp)
      trace<NVAR_2D>(q[tx][ty], q[tx][ty+1], q[tx][ty-1], c, dtdx, qxm[tx][ty], qxp);
    }
  __syncthreads();
  
  //__shared__ real_t flux[T2_VBLOCK_DIMX][T2_VBLOCK_DIMY][NVAR_2D];
  // re-use q as the flux array;
  real_t (&flux)[T2_VBLOCK_DIMX][T2_VBLOCK_DIMY][NVAR_2D] = q;
  flux[tx][ty][ID] = ZERO_F;
  flux[tx][ty][IP] = ZERO_F;
  flux[tx][ty][IU] = ZERO_F;
  flux[tx][ty][IV] = ZERO_F;
  __syncthreads();

  if(i >= 2 and i < imax-2 and 
     j >  0 and j < jmax   and ty > 1 and ty < T2_VBLOCK_DIMY-1)
    {
      real_t (&qleft)[NVAR_2D] = qxm[tx][ty-1];
      real_t (&qright)[NVAR_2D] = qxp;
      
      // Solve Riemann problem at interfaces and compute fluxes
      real_t qgdnv[NVAR_2D];
      riemann<NVAR_2D>(qleft, qright, qgdnv, flux[tx][ty]);
    }  
  __syncthreads();
  
  if(i >= 2 and i < imax-2 and 
     j >= 2 and j < jmax-2 and ty > 1 and ty < T2_VBLOCK_DIMY-2)
    {
      // Update conservative variables : watchout! IU and IV are swapped !
      int offset = elemOffset;
      UOut[offset] = u[ID] + (flux[tx][ty][ID]-flux[tx][ty+1][ID]) * dtdx; offset += arraySize;
      UOut[offset] = u[IP] + (flux[tx][ty][IP]-flux[tx][ty+1][IP]) * dtdx; offset += arraySize;
      UOut[offset] = u[IV] + (flux[tx][ty][IV]-flux[tx][ty+1][IV]) * dtdx; offset += arraySize; 
      UOut[offset] = u[IU] + (flux[tx][ty][IU]-flux[tx][ty+1][IU]) * dtdx;

      // update momentum using gravity predictor
      real_t rhoDelta = HALF_F * (flux[tx][ty][ID] - flux[tx][ty+1][ID]) * dtdx;
      offset = elemOffset + 2*arraySize;
      UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
      UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_y*dt;

    }
} // godunov_y_2d_v2


  /******************************************
   *** *** *** GODUNOV 3D KERNELS *** *** ***
   ******************************************/

  // 3D-kernel block dimensions

  // #undef T2_HBLOCK_DIMX
  // #undef T2_HBLOCK_DIMY
  // #undef T2_HBLOCK_INNER_DIMX
  // #undef T2_VBLOCK_DIMX
  // #undef T2_VBLOCK_DIMY
  // #undef T2_VBLOCK_INNER_DIMY

#ifdef USE_DOUBLE
#define T2_XDIR_BLOCK_DIMX_3D      	16
#define T2_XDIR_BLOCK_DIMY_3D      	15
#define T2_XDIR_BLOCK_INNER_DIMX_3D	14

#define T2_YDIR_BLOCK_DIMX_3D      	16
#define T2_YDIR_BLOCK_DIMY_3D      	15
#define T2_YDIR_BLOCK_INNER_DIMY_3D	13

#define T2_ZDIR_BLOCK_DIMX_3D		16
#define T2_ZDIR_BLOCK_DIMZ_3D		25
#define T2_ZDIR_BLOCK_INNER_DIMZ_3D	21

#else // SINGLE PRECISION

#define T2_XDIR_BLOCK_DIMX_3D		16
#define T2_XDIR_BLOCK_DIMY_3D		24
#define T2_XDIR_BLOCK_INNER_DIMX_3D	12

#define T2_YDIR_BLOCK_DIMX_3D		16
#define T2_YDIR_BLOCK_DIMY_3D		25
#define T2_YDIR_BLOCK_INNER_DIMY_3D	21

#define T2_ZDIR_BLOCK_DIMX_3D		16
#define T2_ZDIR_BLOCK_DIMZ_3D		25
#define T2_ZDIR_BLOCK_INNER_DIMZ_3D	21

#endif // USE_DOUBLE

/**
 * Directionally split Godunov kernel along X-direction for 3D data
 * This kernel does essentially the same computation as godunov_x_2d,
 * except it loops over all posible X-Y planes through the
 * Z-direction.
 * Also note that we must decrease the block dimensions to cope the
 * shared memory limitation.
 */
__global__ void godunov_x_3d_v2(real_t* U, real_t* UOut, 
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

  const int i = __mul24(bx, T2_XDIR_BLOCK_INNER_DIMX_3D) + tx;
  const int j = __mul24(by, T2_XDIR_BLOCK_DIMY_3D      ) + ty;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t   q[T2_XDIR_BLOCK_DIMX_3D][T2_XDIR_BLOCK_DIMY_3D][NVAR_3D];
  __shared__ real_t qxm[T2_XDIR_BLOCK_DIMX_3D][T2_XDIR_BLOCK_DIMY_3D][NVAR_3D];
  real_t qxp[NVAR_3D];
  real_t c;

  // loop over all X-Y-planes
  for (int k=2, elemOffset = i + pitch * (j + jmax * 2);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    qxm[tx][ty][ID] = ZERO_F;
    qxm[tx][ty][IP] = ZERO_F;
    qxm[tx][ty][IU] = ZERO_F;
    qxm[tx][ty][IV] = ZERO_F;
    qxm[tx][ty][IW] = ZERO_F;

    // conservative variables
    real_t u[NVAR_3D];
    
    // 3D array offset (do it in the for loop)
    // int elemOffset = i + pitch * (j + jmax * k);

    // take into account the ghost cells
    // load U and convert to primitive variables
    if(j >= 2 and j < jmax-2 and
       i >= 0 and i < imax)
      {
	
	// Gather conservative variables
	int offset = elemOffset;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize;
	u[IU] = U[offset];  offset += arraySize;
	u[IV] = U[offset];  offset += arraySize;
	u[IW] = U[offset];
	
	//Convert to primitive variables
	constoprim_3D(u, q[tx][ty], c);
      }
    __syncthreads();
    
    if(j >= 2 and j < jmax-2 and 
       i >  0 and i < imax-1 and tx > 0 and tx < T2_XDIR_BLOCK_DIMX_3D-1)
      {
	// Characteristic tracing (compute qxm and qxp)
	trace<NVAR_3D>(q[tx][ty], q[tx+1][ty], q[tx-1][ty], c, dtdx, qxm[tx][ty], qxp);
      }
    __syncthreads();
    
    // re-use q as the flux array;
    real_t (&flux)[T2_XDIR_BLOCK_DIMX_3D][T2_XDIR_BLOCK_DIMY_3D][NVAR_3D] = q; 
    flux[tx][ty][ID] = ZERO_F;
    flux[tx][ty][IP] = ZERO_F;
    flux[tx][ty][IU] = ZERO_F;
    flux[tx][ty][IV] = ZERO_F;
    flux[tx][ty][IW] = ZERO_F;
    __syncthreads();

    if(j >= 2 and j < jmax-2 and 
       i >  0 and i < imax   and tx > 1 and tx < T2_XDIR_BLOCK_DIMX_3D-1)
      {
	real_t (&qleft)[NVAR_3D]  = qxm[tx-1][ty];
	real_t (&qright)[NVAR_3D] = qxp;
	
	// Solve Riemann problem at interfaces and compute fluxes
	real_t qgdnv[NVAR_3D];
	riemann<NVAR_3D>(qleft, qright, qgdnv, flux[tx][ty]);
      }    
    __syncthreads();
    
    if(j >= 2 and j < jmax-2 and 
       i >= 2 and i < imax-2 and tx > 1 and tx < T2_XDIR_BLOCK_DIMX_3D-2)
      {
	// Update conservative variables
	int offset = elemOffset;
	UOut[offset] = u[ID] + (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx; offset += arraySize;
	UOut[offset] = u[IP] + (flux[tx][ty][IP] - flux[tx+1][ty][IP]) * dtdx; offset += arraySize;
	UOut[offset] = u[IU] + (flux[tx][ty][IU] - flux[tx+1][ty][IU]) * dtdx; offset += arraySize;
	UOut[offset] = u[IV] + (flux[tx][ty][IV] - flux[tx+1][ty][IV]) * dtdx; offset += arraySize;
	UOut[offset] = u[IW] + (flux[tx][ty][IW] - flux[tx+1][ty][IW]) * dtdx;

	// update momentum using gravity predictor
	real_t rhoDelta = HALF_F * (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx;
	offset = elemOffset + 2*arraySize;
	UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
	UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_y*dt; offset += arraySize;
	UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_z*dt;
      }

  } // end loop over k (X-Y-planes location)

} // godunov_x_3d_v2

/**
 * Directionally split Godunov kernel along Y-direction for 3D data
 * This kernel does essentially the same computation as godunov_x_3d,
 * except we have swapped x and y.
 *
 * Also note that we must decrease the block dimensions to cope the
 * shared memory limitation.
 *
 */
__global__ void godunov_y_3d_v2(real_t* U, real_t* UOut,
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

  const int i = __mul24(bx, T2_YDIR_BLOCK_DIMX_3D      ) + tx;
  const int j = __mul24(by, T2_YDIR_BLOCK_INNER_DIMY_3D) + ty;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t   q[T2_YDIR_BLOCK_DIMX_3D][T2_YDIR_BLOCK_DIMY_3D][NVAR_3D];
  __shared__ real_t qxm[T2_YDIR_BLOCK_DIMX_3D][T2_YDIR_BLOCK_DIMY_3D][NVAR_3D];
  real_t qxp[NVAR_3D];
  real_t c;

  // loop over all X-Y-planes
  for (int k=2, elemOffset = i + pitch * (j + jmax * 2);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    qxm[tx][ty][ID] = ZERO_F;
    qxm[tx][ty][IP] = ZERO_F;
    qxm[tx][ty][IU] = ZERO_F;
    qxm[tx][ty][IV] = ZERO_F;
    qxm[tx][ty][IW] = ZERO_F;

    // conservative variables
    real_t u[NVAR_3D];
    
    // 3D array offset (do it in the for loop)
    // int elemOffset = i + pitch * (j + jmax * k);

    // take into account the ghost cells
    // load U and convert to primitive variables
    if(i >= 2 and i < imax-2 and
       j >= 0 and j < jmax)
      {	
	// Gather conservative variables
	int offset = elemOffset;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize;
	u[IV] = U[offset];  offset += arraySize; // Watchout IU and IV swapped !
	u[IU] = U[offset];  offset += arraySize;
	u[IW] = U[offset];
	
	//Convert to primitive variables
	constoprim_3D(u, q[tx][ty], c);
	
      }
    __syncthreads();

    if(i >= 2 and i < imax-2 and
       j >  0 and j < jmax-1 and ty > 0 and ty < T2_YDIR_BLOCK_DIMY_3D-1)
      {
	// Characteristic tracing (compute qxm and qxp)
	trace<NVAR_3D>(q[tx][ty], q[tx][ty+1], q[tx][ty-1], c, dtdx, qxm[tx][ty], qxp);
      }
    __syncthreads();
    
    // re-use q as the flux array;
    real_t (&flux)[T2_YDIR_BLOCK_DIMX_3D][T2_YDIR_BLOCK_DIMY_3D][NVAR_3D] = q; 
    flux[tx][ty][ID] = ZERO_F;
    flux[tx][ty][IP] = ZERO_F;
    flux[tx][ty][IU] = ZERO_F;
    flux[tx][ty][IV] = ZERO_F;
    flux[tx][ty][IW] = ZERO_F;
    __syncthreads();

    if(i >= 2 and i < imax-2 and 
       j >  0 and j < jmax   and ty > 1 and ty < T2_YDIR_BLOCK_DIMY_3D-1)
      {
	real_t (&qleft)[NVAR_3D]  = qxm[tx][ty-1];
	real_t (&qright)[NVAR_3D] = qxp;
	
	// Solve Riemann problem at interfaces and compute fluxes
	real_t qgdnv[NVAR_3D];
	riemann<NVAR_3D>(qleft, qright, qgdnv, flux[tx][ty]);
      }
    
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and 
       j >= 2 and j < jmax-2 and ty > 1 and ty < T2_YDIR_BLOCK_DIMY_3D-2)
      {
	// Update conservative variables : watchout IU and IV are swapped
	int offset = elemOffset;
	UOut[offset] = u[ID] + (flux[tx][ty][ID] - flux[tx][ty+1][ID]) * dtdx; offset += arraySize;
	UOut[offset] = u[IP] + (flux[tx][ty][IP] - flux[tx][ty+1][IP]) * dtdx; offset += arraySize;
	UOut[offset] = u[IV] + (flux[tx][ty][IV] - flux[tx][ty+1][IV]) * dtdx; offset += arraySize;
	UOut[offset] = u[IU] + (flux[tx][ty][IU] - flux[tx][ty+1][IU]) * dtdx; offset += arraySize;
	UOut[offset] = u[IW] + (flux[tx][ty][IW] - flux[tx][ty+1][IW]) * dtdx;
	
      // update momentum using gravity predictor
      real_t rhoDelta = HALF_F * (flux[tx][ty][ID] - flux[tx][ty+1][ID]) * dtdx;
      offset = elemOffset + 2*arraySize;
      UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
      UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_y*dt; offset += arraySize;
      UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_z*dt;	
      }

  } // end loop over k (X-Y-planes location)

} // godunov_y_3d_v2

/**
 * Directionally split Godunov kernel along Z-direction for 3D data
 * This kernel does essentially the same computation as godunov_y_3d,
 * except we have swapped y and z.
 *
 */
__global__ void godunov_z_3d_v2(real_t* U, real_t* UOut, 
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

  const int i = __mul24(bx, T2_ZDIR_BLOCK_DIMX_3D      ) + tx;
  const int k = __mul24(bz, T2_ZDIR_BLOCK_INNER_DIMZ_3D) + tz;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t   q[T2_ZDIR_BLOCK_DIMX_3D][T2_ZDIR_BLOCK_DIMZ_3D][NVAR_3D];
  __shared__ real_t qxm[T2_ZDIR_BLOCK_DIMX_3D][T2_ZDIR_BLOCK_DIMZ_3D][NVAR_3D];
  real_t qxp[NVAR_3D];
  real_t c;
  
  // loop over all X-Z-planes
  for (int j=2, elemOffset = i + pitch * (2 + jmax * k);
       j < jmax-2; 
       ++j, elemOffset += pitch) {
    
    qxm[tx][tz][ID] = ZERO_F;
    qxm[tx][tz][IP] = ZERO_F;
    qxm[tx][tz][IU] = ZERO_F;
    qxm[tx][tz][IV] = ZERO_F;
    qxm[tx][tz][IW] = ZERO_F;
    
    // conservative variables
    real_t u[NVAR_3D];
    
    // 3D array offset (do it in the for loop)
    // int elemOffset = i + pitch * (j + jmax * k);

    // take into account the ghost cells
    // load U and convert to primitive variables
    if(i >= 2 and i < imax-2 and
       k >= 0 and k < kmax)
      {
	// Gather conservative variables
	int offset = elemOffset;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize; 
	u[IW] = U[offset];  offset += arraySize; // Watchout IU and IW swapped !
	u[IV] = U[offset];  offset += arraySize;
	u[IU] = U[offset];
	
	//Convert to primitive variables
	constoprim_3D(u, q[tx][tz], c);
      }
    __syncthreads();

    if(i >= 2 and i < imax-2 and
       k >  0 and k < kmax-1 and tz > 0 and tz < T2_ZDIR_BLOCK_DIMZ_3D-1)
      {
	// Characteristic tracing
	trace<NVAR_3D>(q[tx][tz], q[tx][tz+1], q[tx][tz-1], c, dtdx, qxm[tx][tz], qxp);
      }
    __syncthreads();
    
    // re-use q as the flux array;
    real_t (&flux)[T2_ZDIR_BLOCK_DIMX_3D][T2_ZDIR_BLOCK_DIMZ_3D][NVAR_3D] = q; 
    flux[tx][tz][ID] = ZERO_F;
    flux[tx][tz][IP] = ZERO_F;
    flux[tx][tz][IU] = ZERO_F;
    flux[tx][tz][IV] = ZERO_F;
    flux[tx][tz][IW] = ZERO_F;
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and 
       k >  0 and k < kmax   and tz > 1 and tz < T2_ZDIR_BLOCK_DIMZ_3D-1)
      {
	real_t (&qleft)[NVAR_3D]  = qxm[tx][tz-1];
	real_t (&qright)[NVAR_3D] = qxp;
	
	// Solve Riemann problem at interfaces and compute fluxes
	real_t qgdnv[NVAR_3D];
	riemann<NVAR_3D>(qleft, qright, qgdnv, flux[tx][tz]);
      }    
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and 
       k >= 2 and k < kmax-2 and tz > 1 and tz < T2_ZDIR_BLOCK_DIMZ_3D-2)
      {
	// Update conservative variables : watchout IU and IW are swapped
	int offset = elemOffset;
	UOut[offset] = u[ID] + (flux[tx][tz][ID] - flux[tx][tz+1][ID]) * dtdx; offset += arraySize;
	UOut[offset] = u[IP] + (flux[tx][tz][IP] - flux[tx][tz+1][IP]) * dtdx; offset += arraySize;
	UOut[offset] = u[IW] + (flux[tx][tz][IW] - flux[tx][tz+1][IW]) * dtdx; offset += arraySize;
	UOut[offset] = u[IV] + (flux[tx][tz][IV] - flux[tx][tz+1][IV]) * dtdx; offset += arraySize;
	UOut[offset] = u[IU] + (flux[tx][tz][IU] - flux[tx][tz+1][IU]) * dtdx;
	
	// update momentum using gravity predictor
	real_t rhoDelta = HALF_F * (flux[tx][tz][ID] - flux[tx][tz+1][ID]) * dtdx;
	offset = elemOffset + 2*arraySize;
	UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_x*dt; offset += arraySize;
	UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_y*dt; offset += arraySize;
	UOut[offset] += (u[ID]+rhoDelta)*gParams.gravity_z*dt;	
      }

  } // end loop over j (X-Z-planes location)
  
} // godunov_z_3d_v2

#endif /*GODUNOV_TRACE_V2_CUH_*/
