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
 * \file relaxingTVD.cuh
 * \brief Defines the CUDA kernel for the relaxing TVD scheme.
 *
 * Please note that the relaxing TVD scheme uses a centered 7 points
 * stencil (3+1+3).
 *
 * \date 1-Feb-2011
 * \author P. Kestener
 *
 * $Id: relaxingTVD.cuh 2108 2012-05-23 12:07:21Z pkestene $
 */
#ifndef RELAXING_TVD_CUH_
#define RELAXING_TVD_CUH_

// 2D kernel block dimensions
#ifdef USE_DOUBLE

#define XDIR_BLOCK_DIMX_2D 28
#define XDIR_BLOCK_DIMX_2D_INNER (XDIR_BLOCK_DIMX_2D-6)
#define XDIR_BLOCK_DIMY_2D 12

#define YDIR_BLOCK_DIMX_2D 16
#define YDIR_BLOCK_DIMY_2D 20
#define YDIR_BLOCK_DIMY_2D_INNER (YDIR_BLOCK_DIMY_2D-6)

#else // single precision

#define XDIR_BLOCK_DIMX_2D 28
#define XDIR_BLOCK_DIMX_2D_INNER (XDIR_BLOCK_DIMX_2D-6)
#define XDIR_BLOCK_DIMY_2D 12

#define YDIR_BLOCK_DIMX_2D 16
#define YDIR_BLOCK_DIMY_2D 20
#define YDIR_BLOCK_DIMY_2D_INNER (YDIR_BLOCK_DIMY_2D-6)

#endif // USE_DOUBLE

#include "real_type.h"
#include "constants.h"
#include "relaxingTVD.h"

/**
 * Directional splitted relaxing TVD kernel for 2D data along X.
 */
__global__ void kernel_relaxing_TVD_2d_xDir(real_t* U, real_t* UOut,
					    int pitch, 
					    int imax, int jmax, 
					    const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, XDIR_BLOCK_DIMX_2D_INNER) + tx;
  const int j = __mul24(by, XDIR_BLOCK_DIMY_2D)       + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  __shared__ real_t      fl[XDIR_BLOCK_DIMY_2D][XDIR_BLOCK_DIMX_2D][NVAR_2D];
  __shared__ real_t      fr[XDIR_BLOCK_DIMY_2D][XDIR_BLOCK_DIMX_2D][NVAR_2D];
  __shared__ real_t      u1[XDIR_BLOCK_DIMY_2D][XDIR_BLOCK_DIMX_2D][NVAR_2D];

  // reuse fr as fu in 1st order half step
  real_t (&fu)[XDIR_BLOCK_DIMY_2D][XDIR_BLOCK_DIMX_2D][NVAR_2D] = fr;
  
  // conservative variables
  real_t u[NVAR_2D];
  real_t w[NVAR_2D];
  real_t c; // speed of sound

  /*
   * do half step using first-order upwind scheme
   */
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax)
    {

      // get current hydro state (conservative variables)
      int offset = elemOffset;
      u[ID] = U[offset]; offset += arraySize;
      u[IP] = U[offset]; offset += arraySize;
      u[IU] = U[offset]; offset += arraySize;
      u[IV] = U[offset];
      
      // get averageFlux (return w and c from a given hydro state u)
      averageFlux<NVAR_2D>(u,w,c);

      // compute left and righ fluxes
      fr[ty][tx][ID] = (u[ID]*c+w[ID])/2;
      fr[ty][tx][IP] = (u[IP]*c+w[IP])/2;
      fr[ty][tx][IU] = (u[IU]*c+w[IU])/2;
      fr[ty][tx][IV] = (u[IV]*c+w[IV])/2;

      if (tx>0) {
	fl[ty][tx-1][ID] = (u[ID]*c-w[ID])/2;
	fl[ty][tx-1][IP] = (u[IP]*c-w[IP])/2;
	fl[ty][tx-1][IU] = (u[IU]*c-w[IU])/2;
	fl[ty][tx-1][IV] = (u[IV]*c-w[IV])/2;
      }
    }
  __syncthreads();

  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and tx<XDIR_BLOCK_DIMX_2D-1)
    {
      // compute fu
      fu[ty][tx][ID] = fr[ty][tx][ID] - fl[ty][tx][ID];
      fu[ty][tx][IP] = fr[ty][tx][IP] - fl[ty][tx][IP];
      fu[ty][tx][IU] = fr[ty][tx][IU] - fl[ty][tx][IU];
      fu[ty][tx][IV] = fr[ty][tx][IV] - fl[ty][tx][IV];
    }
  __syncthreads();

  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     tx > 0 and tx < XDIR_BLOCK_DIMX_2D-1)
    {
      // compute u1
      u1[ty][tx][ID] = u[ID] - (fu[ty][tx][ID]-fu[ty][tx-1][ID])*dt/2;
      u1[ty][tx][IP] = u[IP] - (fu[ty][tx][IP]-fu[ty][tx-1][IP])*dt/2;
      u1[ty][tx][IU] = u[IU] - (fu[ty][tx][IU]-fu[ty][tx-1][IU])*dt/2;
      u1[ty][tx][IV] = u[IV] - (fu[ty][tx][IV]-fu[ty][tx-1][IV])*dt/2;
    }
  __syncthreads();

  /*
   * do full step using second-order TVD scheme
   */
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     tx > 0 and tx < XDIR_BLOCK_DIMX_2D-1)
    {
      // get averageFlux
      averageFlux<NVAR_2D>(u1[ty][tx],w,c);

      // compute left and righ fluxes
      fr[ty][tx][ID]   = (u1[ty][tx][ID]*c+w[ID])/2;
      fr[ty][tx][IP]   = (u1[ty][tx][IP]*c+w[IP])/2;
      fr[ty][tx][IU]   = (u1[ty][tx][IU]*c+w[IU])/2;
      fr[ty][tx][IV]   = (u1[ty][tx][IV]*c+w[IV])/2;

      fl[ty][tx-1][ID] = (u1[ty][tx][ID]*c-w[ID])/2;
      fl[ty][tx-1][IP] = (u1[ty][tx][IP]*c-w[IP])/2;
      fl[ty][tx-1][IU] = (u1[ty][tx][IU]*c-w[IU])/2;
      fl[ty][tx-1][IV] = (u1[ty][tx][IV]*c-w[IV])/2;
    }  
  __syncthreads();

  // we don't need u1 anymore, so reuse it to store dfl
  real_t (&dfl)[XDIR_BLOCK_DIMY_2D][XDIR_BLOCK_DIMX_2D][NVAR_2D] = u1;

  /*
   * right moving waves
   */
  // compute dfl
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     tx>1 and tx<XDIR_BLOCK_DIMX_2D-1)
    {
      dfl[ty][tx][ID] = (fr[ty][tx][ID] - fr[ty][tx-1][ID]) / 2;
      dfl[ty][tx][IP] = (fr[ty][tx][IP] - fr[ty][tx-1][IP]) / 2;
      dfl[ty][tx][IU] = (fr[ty][tx][IU] - fr[ty][tx-1][IU]) / 2;
      dfl[ty][tx][IV] = (fr[ty][tx][IV] - fr[ty][tx-1][IV]) / 2;
    }
  __syncthreads();

  // compute dfr ( dfr[ty][tx] = dfl[ty][tx+1] )

  // compute fr : flux limiter
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     tx>1 and tx<XDIR_BLOCK_DIMX_2D-2)
    {
      vanleer( fr[ty][tx][ID], dfl[ty][tx][ID], dfl[ty][tx+1][ID] );
      vanleer( fr[ty][tx][IP], dfl[ty][tx][IP], dfl[ty][tx+1][IP] );
      vanleer( fr[ty][tx][IU], dfl[ty][tx][IU], dfl[ty][tx+1][IU] );
      vanleer( fr[ty][tx][IV], dfl[ty][tx][IV], dfl[ty][tx+1][IV] );
    }
  __syncthreads();

  /*
   * left moving waves
   */
  // compute dfl
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     tx>0 and tx<XDIR_BLOCK_DIMX_2D-2)
    {
      dfl[ty][tx][ID] = (fl[ty][tx-1][ID] - fl[ty][tx][ID]) / 2;
      dfl[ty][tx][IP] = (fl[ty][tx-1][IP] - fl[ty][tx][IP]) / 2;
      dfl[ty][tx][IU] = (fl[ty][tx-1][IU] - fl[ty][tx][IU]) / 2;
      dfl[ty][tx][IV] = (fl[ty][tx-1][IV] - fl[ty][tx][IV]) / 2;     
    }
  __syncthreads();
  
  // compute dfr ( dfr[ty][tx-1] = dfl[ty][tx] )

  // compute fl : flux limiter
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     tx>0 and tx<XDIR_BLOCK_DIMX_2D-3)
    {
      vanleer( fl[ty][tx][ID], dfl[ty][tx][ID], dfl[ty][tx+1][ID] );
      vanleer( fl[ty][tx][IP], dfl[ty][tx][IP], dfl[ty][tx+1][IP] );
      vanleer( fl[ty][tx][IU], dfl[ty][tx][IU], dfl[ty][tx+1][IU] );
      vanleer( fl[ty][tx][IV], dfl[ty][tx][IV], dfl[ty][tx+1][IV] );
    }
  __syncthreads();
  
  // we don't need dfl (i.e. u1) anymore
  real_t (&fu2)[XDIR_BLOCK_DIMY_2D][XDIR_BLOCK_DIMX_2D][NVAR_2D] = u1;
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     tx>1 and tx<XDIR_BLOCK_DIMX_2D-3)
    {
      fu2[ty][tx][ID] = fr[ty][tx][ID] - fl[ty][tx][ID];
      fu2[ty][tx][IP] = fr[ty][tx][IP] - fl[ty][tx][IP];
      fu2[ty][tx][IU] = fr[ty][tx][IU] - fl[ty][tx][IU];
      fu2[ty][tx][IV] = fr[ty][tx][IV] - fl[ty][tx][IV];
    }
  __syncthreads();
  
  /*
   * hydro update XDIR
   */
  if(j >= 3 and j < jmax-3 and 
     i >= 3 and i < imax-3 and 
     tx>2 and tx<XDIR_BLOCK_DIMX_2D-3)
    {
      int offset = elemOffset;
      UOut[offset] = u[ID]-(fu2[ty][tx][ID]-fu2[ty][tx-1][ID])*dt; offset += arraySize;
      UOut[offset] = u[IP]-(fu2[ty][tx][IP]-fu2[ty][tx-1][IP])*dt; offset += arraySize;
      UOut[offset] = u[IU]-(fu2[ty][tx][IU]-fu2[ty][tx-1][IU])*dt; offset += arraySize;
      UOut[offset] = u[IV]-(fu2[ty][tx][IV]-fu2[ty][tx-1][IV])*dt; offset += arraySize;
    }

} // kernel_relaxing_TVD_2d_xDir

/**
 * Directional splitted relaxing TVD kernel for 2D data along Y.
 */
__global__ void kernel_relaxing_TVD_2d_yDir(real_t* U, real_t* UOut,
					    int pitch, 
					    int imax, int jmax, 
					    const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, YDIR_BLOCK_DIMX_2D)       + tx;
  const int j = __mul24(by, YDIR_BLOCK_DIMY_2D_INNER) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  __shared__ real_t      fl[YDIR_BLOCK_DIMY_2D][YDIR_BLOCK_DIMX_2D][NVAR_2D];
  __shared__ real_t      fr[YDIR_BLOCK_DIMY_2D][YDIR_BLOCK_DIMX_2D][NVAR_2D];
  __shared__ real_t      u1[YDIR_BLOCK_DIMY_2D][YDIR_BLOCK_DIMX_2D][NVAR_2D];

  // reuse fr as fu in 1st order half step
  real_t (&fu)[YDIR_BLOCK_DIMY_2D][YDIR_BLOCK_DIMX_2D][NVAR_2D] = fr;
  
  // conservative variables
  real_t u[NVAR_2D];
  real_t w[NVAR_2D];
  real_t c; // speed of sound

  /*
   * do half step using first-order upwind scheme
   */
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax)
    {

      // get current hydro state (conservative variables) : swap IU
      // and IV !!!
      int offset = elemOffset;
      u[ID] = U[offset]; offset += arraySize;
      u[IP] = U[offset]; offset += arraySize;
      u[IV] = U[offset]; offset += arraySize;
      u[IU] = U[offset];
      
      // get averageFlux (return w and c from a given hydro state u)
      averageFlux<NVAR_2D>(u,w,c);

      // compute left and righ fluxes
      fr[ty][tx][ID] = (u[ID]*c+w[ID])/2;
      fr[ty][tx][IP] = (u[IP]*c+w[IP])/2;
      fr[ty][tx][IU] = (u[IU]*c+w[IU])/2;
      fr[ty][tx][IV] = (u[IV]*c+w[IV])/2;

      if (ty>0) {
	fl[ty-1][tx][ID] = (u[ID]*c-w[ID])/2;
	fl[ty-1][tx][IP] = (u[IP]*c-w[IP])/2;
	fl[ty-1][tx][IU] = (u[IU]*c-w[IU])/2;
	fl[ty-1][tx][IV] = (u[IV]*c-w[IV])/2;
      }
    }
  __syncthreads();

  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and ty<YDIR_BLOCK_DIMY_2D-1)
    {
      // compute fu
      fu[ty][tx][ID] = fr[ty][tx][ID] - fl[ty][tx][ID];
      fu[ty][tx][IP] = fr[ty][tx][IP] - fl[ty][tx][IP];
      fu[ty][tx][IU] = fr[ty][tx][IU] - fl[ty][tx][IU];
      fu[ty][tx][IV] = fr[ty][tx][IV] - fl[ty][tx][IV];
    }
  __syncthreads();

  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     ty > 0 and ty < YDIR_BLOCK_DIMY_2D-1)
    {
      // compute u1
      u1[ty][tx][ID] = u[ID] - (fu[ty][tx][ID]-fu[ty-1][tx][ID])*dt/2;
      u1[ty][tx][IP] = u[IP] - (fu[ty][tx][IP]-fu[ty-1][tx][IP])*dt/2;
      u1[ty][tx][IU] = u[IU] - (fu[ty][tx][IU]-fu[ty-1][tx][IU])*dt/2;
      u1[ty][tx][IV] = u[IV] - (fu[ty][tx][IV]-fu[ty-1][tx][IV])*dt/2;
    }
  __syncthreads();

  /*
   * do full step using second-order TVD scheme
   */
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     ty > 0 and ty < YDIR_BLOCK_DIMY_2D-1)
    {
      // get averageFlux
      averageFlux<NVAR_2D>(u1[ty][tx],w,c);

      // compute left and righ fluxes
      fr[ty][tx][ID]   = (u1[ty][tx][ID]*c+w[ID])/2;
      fr[ty][tx][IP]   = (u1[ty][tx][IP]*c+w[IP])/2;
      fr[ty][tx][IU]   = (u1[ty][tx][IU]*c+w[IU])/2;
      fr[ty][tx][IV]   = (u1[ty][tx][IV]*c+w[IV])/2;

      fl[ty-1][tx][ID] = (u1[ty][tx][ID]*c-w[ID])/2;
      fl[ty-1][tx][IP] = (u1[ty][tx][IP]*c-w[IP])/2;
      fl[ty-1][tx][IU] = (u1[ty][tx][IU]*c-w[IU])/2;
      fl[ty-1][tx][IV] = (u1[ty][tx][IV]*c-w[IV])/2;
    }  
  __syncthreads();

  // we don't need u1 anymore, so reuse it to store dfl
  real_t (&dfl)[YDIR_BLOCK_DIMY_2D][YDIR_BLOCK_DIMX_2D][NVAR_2D] = u1;

  /*
   * right moving waves
   */
  // compute dfl
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     ty>1 and ty<YDIR_BLOCK_DIMY_2D-1)
    {
      dfl[ty][tx][ID] = (fr[ty][tx][ID] - fr[ty-1][tx][ID]) / 2;
      dfl[ty][tx][IP] = (fr[ty][tx][IP] - fr[ty-1][tx][IP]) / 2;
      dfl[ty][tx][IU] = (fr[ty][tx][IU] - fr[ty-1][tx][IU]) / 2;
      dfl[ty][tx][IV] = (fr[ty][tx][IV] - fr[ty-1][tx][IV]) / 2;
    }
  __syncthreads();

  // compute dfr ( dfr[ty][tx] = dfl[ty][tx+1] )

  // compute fr : flux limiter
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     ty>1 and ty<YDIR_BLOCK_DIMY_2D-2)
    {
      vanleer( fr[ty][tx][ID], dfl[ty][tx][ID], dfl[ty+1][tx][ID] );
      vanleer( fr[ty][tx][IP], dfl[ty][tx][IP], dfl[ty+1][tx][IP] );
      vanleer( fr[ty][tx][IU], dfl[ty][tx][IU], dfl[ty+1][tx][IU] );
      vanleer( fr[ty][tx][IV], dfl[ty][tx][IV], dfl[ty+1][tx][IV] );
    }
  __syncthreads();

  /*
   * left moving waves
   */
  // compute dfl
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     ty>0 and ty<YDIR_BLOCK_DIMY_2D-2)
    {
      dfl[ty][tx][ID] = (fl[ty-1][tx][ID] - fl[ty][tx][ID]) / 2;
      dfl[ty][tx][IP] = (fl[ty-1][tx][IP] - fl[ty][tx][IP]) / 2;
      dfl[ty][tx][IU] = (fl[ty-1][tx][IU] - fl[ty][tx][IU]) / 2;
      dfl[ty][tx][IV] = (fl[ty-1][tx][IV] - fl[ty][tx][IV]) / 2;
    }
  __syncthreads();
  
  // compute dfr ( dfr[ty][tx-1] = dfl[ty][tx] )

  // compute fl : flux limiter
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     ty>0 and ty<YDIR_BLOCK_DIMY_2D-3)
    {
      vanleer( fl[ty][tx][ID], dfl[ty][tx][ID], dfl[ty+1][tx][ID] );
      vanleer( fl[ty][tx][IP], dfl[ty][tx][IP], dfl[ty+1][tx][IP] );
      vanleer( fl[ty][tx][IU], dfl[ty][tx][IU], dfl[ty+1][tx][IU] );
      vanleer( fl[ty][tx][IV], dfl[ty][tx][IV], dfl[ty+1][tx][IV] );
    }
  __syncthreads();
  
  // we don't need dfl (i.e. u1) anymore
  real_t (&fu2)[YDIR_BLOCK_DIMY_2D][YDIR_BLOCK_DIMX_2D][NVAR_2D] = u1;
  if(j >= 0 and j < jmax and 
     i >= 0 and i < imax and 
     ty>1 and ty<YDIR_BLOCK_DIMY_2D-3)
    {
      fu2[ty][tx][ID] = fr[ty][tx][ID] - fl[ty][tx][ID];
      fu2[ty][tx][IP] = fr[ty][tx][IP] - fl[ty][tx][IP];
      fu2[ty][tx][IU] = fr[ty][tx][IU] - fl[ty][tx][IU];
      fu2[ty][tx][IV] = fr[ty][tx][IV] - fl[ty][tx][IV];
    }
  __syncthreads();
  
  /*
   * hydro update YDIR
   */
  if(j >= 3 and j < jmax-3 and 
     i >= 3 and i < imax-3 and 
     ty>2 and ty<YDIR_BLOCK_DIMY_2D-3)
    {
      int offset = elemOffset; // watchout swap IU and IV
      UOut[offset] = u[ID]-(fu2[ty][tx][ID]-fu2[ty-1][tx][ID])*dt; offset += arraySize;
      UOut[offset] = u[IP]-(fu2[ty][tx][IP]-fu2[ty-1][tx][IP])*dt; offset += arraySize;
      UOut[offset] = u[IV]-(fu2[ty][tx][IV]-fu2[ty-1][tx][IV])*dt; offset += arraySize;
      UOut[offset] = u[IU]-(fu2[ty][tx][IU]-fu2[ty-1][tx][IU])*dt; offset += arraySize;
    }

} // kernel_relaxing_TVD_2d_yDir

#endif // RELAXING_TVD_CUH_
