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
 * \file godunov.cuh
 * \brief Defines the CUDA kernel for the actual Godunov scheme
 * computations.
 *
 * \author F. Chateau, P. Kestener
 *
 * $Id: godunov.cuh 2108 2012-05-23 12:07:21Z pkestene $
 */
#ifndef GODUNOV_CUH_
#define GODUNOV_CUH_


// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define HBLOCK_DIMX		16
#define HBLOCK_DIMY		15
#define HBLOCK_INNER_DIMX	14

#define VBLOCK_DIMX		16
#define VBLOCK_DIMY		15
#define VBLOCK_INNER_DIMY	13
#else // simple precision
#define HBLOCK_DIMX		16
#define HBLOCK_DIMY		30
#define HBLOCK_INNER_DIMX	14

#define VBLOCK_DIMX		16
#define VBLOCK_DIMY		31
#define VBLOCK_INNER_DIMY	29
#endif // USE_DOUBLE

#include "real_type.h"
#include "constants.h"
#include "constoprim.h"
#include "riemann.h"
#include "cmpflx.h"
#include "trace.h"

#include <cstdlib>

/**
 * Splitting Godunov kernel along X-direction for 2D data
 */
__global__ void godunov_x_2d(real_t* U, int pitch, int imax, int jmax, const real_t dtdx)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, HBLOCK_INNER_DIMX) + tx;
  const int j = __mul24(by, HBLOCK_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j) + i;
  
  __shared__ real_t qxm[HBLOCK_DIMX][HBLOCK_DIMY][NVAR_2D];
  real_t qxp[NVAR_2D];
  if(j >= 2 and j < jmax-2 and i > 0 and i < imax-1)
    {
      // conservative variables
      real_t u[NVAR_2D];
      // primitive variables
      real_t q[NVAR_2D];
      real_t qPlus[NVAR_2D], qMinus[NVAR_2D];
      real_t c, cPlus, cMinus;
      
      // Gather conservative variables
      int offset = elemOffset;
      u[ID] = U[offset];  offset += arraySize;
      u[IP] = U[offset];  offset += arraySize;
      u[IU] = U[offset];  offset += arraySize;
      u[IV] = U[offset];
      
      //Convert to primitive variables
      constoprim_2D(u, q, c);
      
      // next cell
      offset = elemOffset+1;
      u[ID] = U[offset];  offset += arraySize;
      u[IP] = U[offset];  offset += arraySize;
      u[IU] = U[offset];  offset += arraySize;
      u[IV] = U[offset];
      constoprim_2D(u, qPlus, cPlus);
      
      // previous cell
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
  
  __shared__ real_t flux[HBLOCK_DIMX][HBLOCK_DIMY][NVAR_2D];
  if(j >= 2 and j < jmax-2 and i > 0 and i < imax and tx > 0)
    {
      real_t (&qleft)[NVAR_2D] = qxm[tx-1][ty];
      real_t (&qright)[NVAR_2D] = qxp;
      
      // Solve Riemann problem at interfaces
      real_t qgdnv[NVAR_2D];
      riemann(qleft, qright, qgdnv);
      
      // Compute fluxes
      cmpflx(qgdnv, flux[tx][ty]);
    }
  
  __syncthreads();
  
  if(j >= 2 and j < jmax-2 and i >= 2 and i < imax-2 and tx > 0 and tx <= HBLOCK_INNER_DIMX)
    {
      // Update conservative variables
      int offset = elemOffset;
      U[offset] += (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx; offset += arraySize;
      U[offset] += (flux[tx][ty][IP] - flux[tx+1][ty][IP]) * dtdx; offset += arraySize;
      U[offset] += (flux[tx][ty][IU] - flux[tx+1][ty][IU]) * dtdx; offset += arraySize;
      U[offset] += (flux[tx][ty][IV] - flux[tx+1][ty][IV]) * dtdx;
    }
} // godunov_x_2d

/**
 * Splitting Godunov kernel along Y-direction for 2D data
 */
__global__ void godunov_y_2d(real_t* U, int pitch, int imax, int jmax, real_t dtdx)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, VBLOCK_DIMX) + tx;
  const int j = __mul24(by, VBLOCK_INNER_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j) + i;
  
  __shared__ real_t qxm[VBLOCK_DIMX][VBLOCK_DIMY][NVAR_2D];
  real_t qxp[NVAR_2D];
  if(i >= 2 and i < imax-2 and j > 0 and j < jmax-1)
    {
      // conservative variables
      real_t u[NVAR_2D];
      // primitive variables
      real_t q[NVAR_2D];
      real_t qPlus[NVAR_2D], qMinus[NVAR_2D];
      real_t c, cPlus, cMinus;
      
      // Gather conservative variables
      int offset = elemOffset;
      u[ID] = U[offset]; offset += arraySize;
      u[IP] = U[offset]; offset += arraySize;
      u[IV] = U[offset]; offset += arraySize; // watchout! IU and IV are swapped !
      u[IU] = U[offset];
      
      //Convert to primitive variables
      constoprim_2D(u, q, c);
      
      // next cell
      offset = elemOffset+pitch;
      u[ID] = U[offset]; offset += arraySize;
      u[IP] = U[offset]; offset += arraySize;
      u[IV] = U[offset]; offset += arraySize; // IU and IV are swapped !
      u[IU] = U[offset];
      constoprim_2D(u, qPlus, cPlus);

      // previous cell
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
  
  __shared__ real_t flux[VBLOCK_DIMX][VBLOCK_DIMY][NVAR_2D];
  if(i >= 2 and i < imax-2 and j > 0 and j < jmax and ty > 0)
    {
      real_t (&qleft)[NVAR_2D] = qxm[tx][ty-1];
      real_t (&qright)[NVAR_2D] = qxp;
      
      // Solve Riemann problem at interfaces
      real_t qgdnv[NVAR_2D];
      riemann(qleft, qright, qgdnv);
      
      // Compute fluxes
      cmpflx(qgdnv, flux[tx][ty]);
    }
  
  __syncthreads();
  
  if(i >= 2 and i < imax-2 and j >= 2 and j < jmax-2 and ty > 0 and ty <= VBLOCK_INNER_DIMY)
    {
      // Update conservative variables
      int offset = elemOffset;
      U[offset] += (flux[tx][ty][ID]-flux[tx][ty+1][ID]) * dtdx; offset += arraySize;
      U[offset] += (flux[tx][ty][IP]-flux[tx][ty+1][IP]) * dtdx; offset += arraySize;
      U[offset] += (flux[tx][ty][IV]-flux[tx][ty+1][IV]) * dtdx; offset += arraySize; // watchout! IU and IV are swapped !
      U[offset] += (flux[tx][ty][IU]-flux[tx][ty+1][IU]) * dtdx;
    }
} // godunov_y_2d


/******************************************
 *** *** *** GODUNOV 3D KERNELS *** *** ***
 ******************************************/

// 3D-kernel block dimensions

// #undef HBLOCK_DIMX
// #undef HBLOCK_DIMY
// #undef HBLOCK_INNER_DIMX
// #undef VBLOCK_DIMX
// #undef VBLOCK_DIMY
// #undef VBLOCK_INNER_DIMY

#ifdef USE_DOUBLE
#define XDIR_BLOCK_DIMX_3D      	16
#define XDIR_BLOCK_DIMY_3D      	15
#define XDIR_BLOCK_INNER_DIMX_3D	14

#define YDIR_BLOCK_DIMX_3D      	16
#define YDIR_BLOCK_DIMY_3D      	15
#define yDIR_BLOCK_INNER_DIMY_3D	13

#define ZDIR_BLOCK_DIMX_3D		16
#define ZDIR_BLOCK_DIMZ_3D		25
#define ZDIR_BLOCK_INNER_DIMZ_3D	23

#else // SINGLE PRECISION

#define XDIR_BLOCK_DIMX_3D		16
#define XDIR_BLOCK_DIMY_3D		24
#define XDIR_BLOCK_INNER_DIMX_3D	14

#define YDIR_BLOCK_DIMX_3D		16
#define YDIR_BLOCK_DIMY_3D		25
#define YDIR_BLOCK_INNER_DIMY_3D	23

#define ZDIR_BLOCK_DIMX_3D		16
#define ZDIR_BLOCK_DIMZ_3D		25
#define ZDIR_BLOCK_INNER_DIMZ_3D	23

#endif // USE_DOUBLE

/**
 * Splitting Godunov kernel along X-direction for 3D data
 * This kernel does essentially the same computation as godunov_x_2d,
 * except it loops over all posible X-Y planes through the
 * Z-direction.
 * Also note that we must decrease the block dimensions to cope the
 * shared memory limitation.
 */
__global__ void godunov_x_3d(real_t* U, int pitch, 
			     int imax, int jmax, int kmax, 
			     const real_t dtdx)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, XDIR_BLOCK_INNER_DIMX_3D) + tx;
  const int j = __mul24(by, XDIR_BLOCK_DIMY_3D      ) + ty;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t qxm[XDIR_BLOCK_DIMX_3D][XDIR_BLOCK_DIMY_3D][NVAR_3D];
  real_t qxp[NVAR_3D];
  __shared__ real_t flux[XDIR_BLOCK_DIMX_3D][XDIR_BLOCK_DIMY_3D][NVAR_3D];
  

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
	real_t u[NVAR_3D];
	// Gather conservative variables
	int offset = elemOffset;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize;
	u[IU] = U[offset];  offset += arraySize;
	u[IV] = U[offset];  offset += arraySize;
	u[IW] = U[offset];
	
	//Convert to primitive variables
	real_t q[NVAR_3D];
	real_t c;
	constoprim_3D(u, q, c);
	
	// Characteristic tracing
	//trace(q, c, dtdx, qxm, qxp[tx][ty]);
	qxm[tx][ty][ID] = q[ID];
	qxm[tx][ty][IP] = q[IP];
	qxm[tx][ty][IU] = q[IU];
	qxm[tx][ty][IV] = q[IV];
	qxm[tx][ty][IW] = q[IW];
	
	qxp[ID] = q[ID];
	qxp[IP] = q[IP];
	qxp[IU] = q[IU];
	qxp[IV] = q[IV];
	qxp[IW] = q[IW];
      }
    
    __syncthreads();
    
    
    if(j >= 2 and j < jmax-2 and 
       i >  0 and i < imax   and tx > 0)
      {
	real_t (&qleft)[NVAR_3D]  = qxm[tx-1][ty];
	real_t (&qright)[NVAR_3D] = qxp;
	
	// Solve Riemann problem at interfaces
	real_t qgdnv[NVAR_3D];
	riemann<NVAR_3D>(qleft, qright, qgdnv);
	
	// Compute fluxes
	cmpflx<NVAR_3D>(qgdnv, flux[tx][ty]);
      }
    
    __syncthreads();
    
    if(j >= 2 and j < jmax-2 and 
       i >= 2 and i < imax-2 and tx > 0 and tx <= XDIR_BLOCK_INNER_DIMX_3D)
      {
	// Update conservative variables
	int offset = elemOffset;
	U[offset] += (flux[tx][ty][ID] - flux[tx+1][ty][ID]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][ty][IP] - flux[tx+1][ty][IP]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][ty][IU] - flux[tx+1][ty][IU]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][ty][IV] - flux[tx+1][ty][IV]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][ty][IW] - flux[tx+1][ty][IW]) * dtdx;
      }

  } // end loop over k (X-Y-planes location)

} // godunov_x_3d

/**
 * Splitting Godunov kernel along Y-direction for 3D data
 * This kernel does essentially the same computation as godunov_x_3d,
 * except we have swapped x and y.
 *
 * Also note that we must decrease the block dimensions to cope the
 * shared memory limitation.
 *
 */
__global__ void godunov_y_3d(real_t* U, int pitch, 
			     int imax, int jmax, int kmax, 
			     const real_t dtdx)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, YDIR_BLOCK_DIMX_3D      ) + tx;
  const int j = __mul24(by, YDIR_BLOCK_INNER_DIMY_3D) + ty;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t qxm[YDIR_BLOCK_DIMX_3D][YDIR_BLOCK_DIMY_3D][NVAR_3D];
  real_t qxp[NVAR_3D];
  __shared__ real_t flux[YDIR_BLOCK_DIMX_3D][YDIR_BLOCK_DIMY_3D][NVAR_3D];
  

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
	real_t u[NVAR_3D];
	// Gather conservative variables
	int offset = elemOffset;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize;
	u[IV] = U[offset];  offset += arraySize; // Watchout IU and IV swapped !
	u[IU] = U[offset];  offset += arraySize;
	u[IW] = U[offset];
	
	//Convert to primitive variables
	real_t q[NVAR_3D];
	real_t c;
	constoprim_3D(u, q, c);
	
	// Characteristic tracing
	//trace(q, c, dtdx, qxm, qxp[tx][ty]);
	qxm[tx][ty][ID] = q[ID];
	qxm[tx][ty][IP] = q[IP];
	qxm[tx][ty][IU] = q[IU];
	qxm[tx][ty][IV] = q[IV];
	qxm[tx][ty][IW] = q[IW];
	
	qxp[ID] = q[ID];
	qxp[IP] = q[IP];
	qxp[IU] = q[IU];
	qxp[IV] = q[IV];
	qxp[IW] = q[IW];
      }
    
    __syncthreads();
    
    
    if(i >= 2 and i < imax-2 and 
       j >  0 and j < jmax   and ty > 0)
      {
	real_t (&qleft)[NVAR_3D]  = qxm[tx][ty-1];
	real_t (&qright)[NVAR_3D] = qxp;
	
	// Solve Riemann problem at interfaces
	real_t qgdnv[NVAR_3D];
	riemann<NVAR_3D>(qleft, qright, qgdnv);
	
	// Compute fluxes
	cmpflx<NVAR_3D>(qgdnv, flux[tx][ty]);
      }
    
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and 
       j >= 2 and j < jmax-2 and ty > 0 and ty <= YDIR_BLOCK_INNER_DIMY_3D)
      {
	// Update conservative variables
	int offset = elemOffset;
	U[offset] += (flux[tx][ty][ID] - flux[tx][ty+1][ID]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][ty][IP] - flux[tx][ty+1][IP]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][ty][IV] - flux[tx][ty+1][IV]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][ty][IU] - flux[tx][ty+1][IU]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][ty][IW] - flux[tx][ty+1][IW]) * dtdx;
      }

  } // end loop over k (X-Y-planes location)

} // godunov_y_3d

/**
 * Splitting Godunov kernel along Z-direction for 3D data
 * This kernel does essentially the same computation as godunov_y_3d,
 * except we have swapped y and z.
 *
 */
__global__ void godunov_z_3d(real_t* U, int pitch, 
			     int imax, int jmax, int kmax, 
			     const real_t dtdx)
{
  // Block index
  const int bx = blockIdx.x;
  const int bz = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int tz = threadIdx.y;

  const int i = __mul24(bx, ZDIR_BLOCK_DIMX_3D      ) + tx;
  const int k = __mul24(bz, ZDIR_BLOCK_INNER_DIMZ_3D) + tz;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t qxm[ZDIR_BLOCK_DIMX_3D][ZDIR_BLOCK_DIMZ_3D][NVAR_3D];
  real_t qxp[NVAR_3D];
  __shared__ real_t flux[ZDIR_BLOCK_DIMX_3D][ZDIR_BLOCK_DIMZ_3D][NVAR_3D];
  

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
	real_t u[NVAR_3D];
	// Gather conservative variables
	int offset = elemOffset;
	u[ID] = U[offset];  offset += arraySize;
	u[IP] = U[offset];  offset += arraySize; 
	u[IW] = U[offset];  offset += arraySize; // Watchout IU and IW swapped !
	u[IV] = U[offset];  offset += arraySize;
	u[IU] = U[offset];
	
	//Convert to primitive variables
	real_t q[NVAR_3D];
	real_t c;
	constoprim_3D(u, q, c);
	
	// Characteristic tracing
	//trace(q, c, dtdx, qxm, qxp[tx][tz]);
	qxm[tx][tz][ID] = q[ID];
	qxm[tx][tz][IP] = q[IP];
	qxm[tx][tz][IU] = q[IU];
	qxm[tx][tz][IV] = q[IV];
	qxm[tx][tz][IW] = q[IW];
	
	qxp[ID] = q[ID];
	qxp[IP] = q[IP];
	qxp[IU] = q[IU];
	qxp[IV] = q[IV];
	qxp[IW] = q[IW];
      }
    
    __syncthreads();
    
    
    if(i >= 2 and i < imax-2 and 
       k >  0 and k < kmax   and tz > 0)
      {
	real_t (&qleft)[NVAR_3D]  = qxm[tx][tz-1];
	real_t (&qright)[NVAR_3D] = qxp;
	
	// Solve Riemann problem at interfaces
	real_t qgdnv[NVAR_3D];
	riemann<NVAR_3D>(qleft, qright, qgdnv);
	
	// Compute fluxes
	cmpflx<NVAR_3D>(qgdnv, flux[tx][tz]);
      }
    
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and 
       k >= 2 and k < kmax-2 and tz > 0 and tz <= ZDIR_BLOCK_INNER_DIMZ_3D)
      {
	// Update conservative variables
	int offset = elemOffset;
	U[offset] += (flux[tx][tz][ID] - flux[tx][tz+1][ID]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][tz][IP] - flux[tx][tz+1][IP]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][tz][IW] - flux[tx][tz+1][IW]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][tz][IV] - flux[tx][tz+1][IV]) * dtdx; offset += arraySize;
	U[offset] += (flux[tx][tz][IU] - flux[tx][tz+1][IU]) * dtdx;
      }

  } // end loop over j (X-Z-planes location)
  
} // godunov_z_3d

#endif /*GODUNOV_CUH_*/
