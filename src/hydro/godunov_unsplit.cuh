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
 * \file godunov_unsplit.cuh
 * \brief Defines the CUDA kernel for the actual Godunov scheme
 * computations (unsplit version).
 *
 * \date 7 Dec 2010
 * \author P. Kestener
 *
 * $Id: godunov_unsplit.cuh 3450 2014-06-16 22:03:23Z pkestene $
 */
#ifndef GODUNOV_UNSPLIT_CUH_
#define GODUNOV_UNSPLIT_CUH_

#include "real_type.h"
#include "constants.h"
#include "base_type.h" // for qHydroState definition
#include "constoprim.h"
#include "riemann.h"
#include "trace.h"

#include <cstdlib>

/** a dummy device-only swap function */
__device__ inline void swap_value(real_t& a, real_t& b) {
  
  real_t tmp = a;
  a = b;
  b = tmp;
  
} // swap_value

/*
 *
 * Here are CUDA kernels implementing hydro unsplit scheme version 0
 * 
 *
 */


/*****************************************
 *** *** GODUNOV UNSPLIT 2D KERNEL *** ***
 *****************************************/

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define UNSPLIT_BLOCK_DIMX_2D_VOLD	16
#define UNSPLIT_BLOCK_INNER_DIMX_2D_VOLD	(UNSPLIT_BLOCK_DIMX_2D_VOLD-4)
#define UNSPLIT_BLOCK_DIMY_2D_VOLD	7
#define UNSPLIT_BLOCK_INNER_DIMY_2D_VOLD	(UNSPLIT_BLOCK_DIMY_2D_VOLD-4)
#else // simple precision
#define UNSPLIT_BLOCK_DIMX_2D_VOLD	16
#define UNSPLIT_BLOCK_INNER_DIMX_2D_VOLD	(UNSPLIT_BLOCK_DIMX_2D_VOLD-4)
#define UNSPLIT_BLOCK_DIMY_2D_VOLD	14
#define UNSPLIT_BLOCK_INNER_DIMY_2D_VOLD	(UNSPLIT_BLOCK_DIMY_2D_VOLD-4)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 2D data (OLD VERSION).
 *
 * This the first version of the unsplit kernel (it uses quite a lot
 * of shared memory). See kernel_godunov_unsplit_2d for a version that
 * uses half shared memory.
 */
__global__ void kernel_godunov_unsplit_2d_vold(const real_t * __restrict__ Uin, 
					       real_t       *Uout,
					       int pitch, 
					       int imax, 
					       int jmax, 
					       real_t dtdx, 
					       real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_2D_VOLD) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_2D_VOLD) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  __shared__ real_t      q[UNSPLIT_BLOCK_DIMX_2D_VOLD][UNSPLIT_BLOCK_DIMY_2D_VOLD][NVAR_2D];
  __shared__ real_t  qm_x1[UNSPLIT_BLOCK_DIMX_2D_VOLD][UNSPLIT_BLOCK_DIMY_2D_VOLD][NVAR_2D];
  __shared__ real_t  qm_y2[UNSPLIT_BLOCK_DIMX_2D_VOLD][UNSPLIT_BLOCK_DIMY_2D_VOLD][NVAR_2D];
  __shared__ real_t flux_y[UNSPLIT_BLOCK_DIMX_2D_VOLD][UNSPLIT_BLOCK_DIMY_2D_VOLD][NVAR_2D];
  real_t qm[TWO_D][NVAR_2D];
  real_t qp[TWO_D][NVAR_2D];

  // conservative variables
  real_t uIn[NVAR_2D];
  real_t uOut[NVAR_2D];
  real_t c;

  qm_x1[tx][ty][ID] = ZERO_F;
  qm_x1[tx][ty][IP] = ZERO_F;
  qm_x1[tx][ty][IU] = ZERO_F;
  qm_x1[tx][ty][IV] = ZERO_F;

  qm_y2[tx][ty][ID] = ZERO_F;
  qm_y2[tx][ty][IP] = ZERO_F;
  qm_y2[tx][ty][IU] = ZERO_F;
  qm_y2[tx][ty][IV] = ZERO_F;
  
  // load U and convert to primitive variables
  if(i >= 0 and i < imax and 
     j >= 0 and j < jmax)
    {
      
      // Gather conservative variables
      int offset = elemOffset;
      uIn[ID] = Uin[offset];  offset += arraySize;
      uIn[IP] = Uin[offset];  offset += arraySize;
      uIn[IU] = Uin[offset];  offset += arraySize;
      uIn[IV] = Uin[offset];
      
      // copy input state into uOut that will become output state
      uOut[ID] = uIn[ID];
      uOut[IP] = uIn[IP];
      uOut[IU] = uIn[IU];
      uOut[IV] = uIn[IV];

      //Convert to primitive variables
      constoprim_2D(uIn, q[tx][ty], c);
    }
  __syncthreads();

  if(i >= 1 and i < imax-1 and tx > 0 and tx < UNSPLIT_BLOCK_DIMX_2D_VOLD-1 and
     j >= 1 and j < jmax-1 and ty > 0 and ty < UNSPLIT_BLOCK_DIMY_2D_VOLD-1)
    {
      real_t qNeighbors[2*TWO_D][NVAR_2D];
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qNeighbors[0][iVar] = q[tx+1][ty  ][iVar];
	qNeighbors[1][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[2][iVar] = q[tx  ][ty+1][iVar];
	qNeighbors[3][iVar] = q[tx  ][ty-1][iVar];
      }
 	
      // Characteristic tracing (compute qxm and qxp)
      trace_unsplit<TWO_D, NVAR_2D>(q[tx][ty], qNeighbors, c, dtdx, qm, qp);

      // store qm, qp
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qm_x1[tx][ty][iVar] = qm[0][iVar];
	qm_y2[tx][ty][iVar] = qm[1][iVar];
      }

    }
  __syncthreads();

  // re-use q as flux_x
  real_t (&flux_x)[UNSPLIT_BLOCK_DIMX_2D_VOLD][UNSPLIT_BLOCK_DIMY_2D_VOLD][NVAR_2D] = q;
  flux_x[tx][ty][ID] = ZERO_F;
  flux_x[tx][ty][IP] = ZERO_F;
  flux_x[tx][ty][IU] = ZERO_F;
  flux_x[tx][ty][IV] = ZERO_F;
  flux_y[tx][ty][ID] = ZERO_F;
  flux_y[tx][ty][IP] = ZERO_F;
  flux_y[tx][ty][IU] = ZERO_F;
  flux_y[tx][ty][IV] = ZERO_F;
  __syncthreads();

  if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_VOLD-1 and
     j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_VOLD-1)
    {
      real_t qgdnv[NVAR_2D];

      // Solve Riemann problem at X-interfaces and compute fluxes
      real_t (&qleft_x)[NVAR_2D]  = qm_x1[tx-1][ty];
      real_t (&qright_x)[NVAR_2D] = qp[0];      
      riemann<NVAR_2D>(qleft_x, qright_x, qgdnv, flux_x[tx][ty]);
      
      // Solve Riemann problem at Y-interfaces and compute fluxes
      real_t (&qleft_y)[NVAR_2D]  = qm_y2[tx][ty-1];
      real_t (&qright_y)[NVAR_2D] = qp[1];
      // watchout swap IU and IV
      swap_value(qleft_y[IU],qleft_y[IV]);
      swap_value(qright_y[IU],qright_y[IV]);
      riemann<NVAR_2D>(qleft_y, qright_y, qgdnv, flux_y[tx][ty]);
    }  
  __syncthreads();
  
     if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_VOLD-2 and
	j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_VOLD-2)
    {
      // update U with flux_x
      uOut[ID] += (flux_x[tx][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
      uOut[IP] += (flux_x[tx][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
      uOut[IU] += (flux_x[tx][ty][IU]-flux_x[tx+1][ty][IU])*dtdx;
      uOut[IV] += (flux_x[tx][ty][IV]-flux_x[tx+1][ty][IV])*dtdx;
  
      // UPDATE momentum using gravity source term
      real_t rhoDelta = HALF_F * (flux_x[tx][ty][ID] - flux_x[tx+1][ty][ID]) * dtdx;
      uOut[IU] += (uIn[ID]+rhoDelta)*gParams.gravity_x*dt;
      uOut[IV] += (uIn[ID]+rhoDelta)*gParams.gravity_y*dt;
      
      // update U with flux_y : watchout! IU and IV are swapped !
      uOut[ID] += (flux_y[tx][ty][ID]-flux_y[tx][ty+1][ID])*dtdx;
      uOut[IP] += (flux_y[tx][ty][IP]-flux_y[tx][ty+1][IP])*dtdx;
      uOut[IU] += (flux_y[tx][ty][IV]-flux_y[tx][ty+1][IV])*dtdx;
      uOut[IV] += (flux_y[tx][ty][IU]-flux_y[tx][ty+1][IU])*dtdx;

      // update momentum using gravity source term
      rhoDelta = HALF_F * (flux_y[tx][ty][ID] - flux_y[tx][ty+1][ID]) * dtdx;
      uOut[IU] += (uIn[ID]+rhoDelta)*gParams.gravity_x*dt;
      uOut[IV] += (uIn[ID]+rhoDelta)*gParams.gravity_y*dt;

      // actually perform update on external device memory
      int offset = elemOffset;
      Uout[offset] = uOut[ID];  offset += arraySize;
      Uout[offset] = uOut[IP];  offset += arraySize;
      Uout[offset] = uOut[IU];  offset += arraySize;
      Uout[offset] = uOut[IV];

    }
      
} // kernel_godunov_unsplit_2d_vold

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define UNSPLIT_BLOCK_DIMX_2D		16
#define UNSPLIT_BLOCK_INNER_DIMX_2D	(UNSPLIT_BLOCK_DIMX_2D-4)
#define UNSPLIT_BLOCK_DIMY_2D		12
#define UNSPLIT_BLOCK_INNER_DIMY_2D	(UNSPLIT_BLOCK_DIMY_2D-4)
#else // simple precision
#define UNSPLIT_BLOCK_DIMX_2D		16
#define UNSPLIT_BLOCK_INNER_DIMX_2D	(UNSPLIT_BLOCK_DIMX_2D-4)
#define UNSPLIT_BLOCK_DIMY_2D		24
#define UNSPLIT_BLOCK_INNER_DIMY_2D	(UNSPLIT_BLOCK_DIMY_2D-4)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 2D data.
 * 
 * This kernel doesn't eat so much shared memory, but to the price of
 * more computations...
 * We remove array qm_x1 and qm_y2, and recompute what is needed by
 * each thread.
 */
__global__ void kernel_godunov_unsplit_2d(const real_t * __restrict__ Uin, 
					  real_t       *Uout,
					  int pitch, 
					  int imax, 
					  int jmax,
					  real_t dtdx, 
					  real_t dt,
					  bool gravityEnabled)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_2D) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_2D) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  __shared__ real_t      q[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_2D];
  __shared__ real_t flux_y[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_2D];
  real_t qm [TWO_D][NVAR_2D];
  real_t qm1[NVAR_2D];
  real_t qm2[NVAR_2D];

  real_t qp [TWO_D][NVAR_2D];
  real_t qp0[TWO_D][NVAR_2D];

  // conservative variables
  real_t uIn[NVAR_2D];
  real_t uOut[NVAR_2D];
  real_t c;

  real_t *gravin = gParams.arrayList[A_GRAV];

  // load U and convert to primitive variables
  if(i >= 0 and i < imax and 
     j >= 0 and j < jmax)
    {
      
      // Gather conservative variables
      int offset = elemOffset;
      uIn[ID] = Uin[offset];  offset += arraySize;
      uIn[IP] = Uin[offset];  offset += arraySize;
      uIn[IU] = Uin[offset];  offset += arraySize;
      uIn[IV] = Uin[offset];
      
      // copy input state into uOut that will become output state
      uOut[ID] = uIn[ID];
      uOut[IP] = uIn[IP];
      uOut[IU] = uIn[IU];
      uOut[IV] = uIn[IV];

      //Convert to primitive variables
      constoprim_2D(uIn, q[tx][ty], c);
    }
  __syncthreads();

  if(i > 1 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-1 and
     j > 1 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-1)
    {
      real_t qNeighbors[2*TWO_D][NVAR_2D];
 
      // Characteristic tracing : compute qp0
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qNeighbors[0][iVar] = q[tx+1][ty  ][iVar];
	qNeighbors[1][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[2][iVar] = q[tx  ][ty+1][iVar];
	qNeighbors[3][iVar] = q[tx  ][ty-1][iVar];
      }	
      trace_unsplit<TWO_D, NVAR_2D>(q[tx][ty], qNeighbors, c, dtdx, qm, qp0);

      // gravity predictor on velocity component of qp0's
      if (gravityEnabled) {
	qp0[0][IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qp0[0][IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];

	qp0[1][IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qp0[1][IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
      }

      // Characteristic tracing : compute qm_x[1]
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qNeighbors[0][iVar] = q[tx  ][ty  ][iVar];
	qNeighbors[1][iVar] = q[tx-2][ty  ][iVar];
	qNeighbors[2][iVar] = q[tx-1][ty+1][iVar];
	qNeighbors[3][iVar] = q[tx-1][ty-1][iVar];
      }
      trace_unsplit<TWO_D, NVAR_2D>(q[tx-1][ty], qNeighbors, c, dtdx, qm, qp);
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qm1[iVar] = qm[0][iVar];
      }

      // gravity predictor on velocity component of qm1
      if (gravityEnabled) {
	qm1[IU] += HALF_F * dt * gravin[elemOffset-1+IX*arraySize];
	qm1[IV] += HALF_F * dt * gravin[elemOffset-1+IY*arraySize];
      }

      // Characteristic tracing : compute qm_y[2]
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qNeighbors[0][iVar] = q[tx+1][ty-1][iVar];
	qNeighbors[1][iVar] = q[tx-1][ty-1][iVar];
	qNeighbors[2][iVar] = q[tx  ][ty  ][iVar];
	qNeighbors[3][iVar] = q[tx  ][ty-2][iVar];
      }
      trace_unsplit<TWO_D, NVAR_2D>(q[tx][ty-1], qNeighbors, c, dtdx, qm, qp);
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qm2[iVar] = qm[1][iVar];
      }

      // gravity predictor on velocity component of qm2
      if (gravityEnabled) {
	qm2[IU] += HALF_F * dt * gravin[elemOffset-pitch+IX*arraySize];
	qm2[IV] += HALF_F * dt * gravin[elemOffset-pitch+IY*arraySize];
      }

    }
  __syncthreads();

  // re-use q as flux_x
  real_t (&flux_x)[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_2D] = q;
  flux_x[tx][ty][ID] = ZERO_F;
  flux_x[tx][ty][IP] = ZERO_F;
  flux_x[tx][ty][IU] = ZERO_F;
  flux_x[tx][ty][IV] = ZERO_F;
  flux_y[tx][ty][ID] = ZERO_F;
  flux_y[tx][ty][IP] = ZERO_F;
  flux_y[tx][ty][IU] = ZERO_F;
  flux_y[tx][ty][IV] = ZERO_F;
  __syncthreads();

  if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-1 and
     j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-1)
    {
      real_t qgdnv[NVAR_2D];

      // Solve Riemann problem at X-interfaces and compute fluxes
      real_t (&qleft_x)[NVAR_2D]  = qm1;
      real_t (&qright_x)[NVAR_2D] = qp0[0];      
      riemann<NVAR_2D>(qleft_x, qright_x, qgdnv, flux_x[tx][ty]);
      
      // Solve Riemann problem at Y-interfaces and compute fluxes
      real_t (&qleft_y)[NVAR_2D]  = qm2;
      real_t (&qright_y)[NVAR_2D] = qp0[1];
      // watchout swap IU and IV
      swap_value(qleft_y[IU],qleft_y[IV]);
      swap_value(qright_y[IU],qright_y[IV]);
      riemann<NVAR_2D>(qleft_y, qright_y, qgdnv, flux_y[tx][ty]);
    }  
  __syncthreads();
  
     if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-2 and
	j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-2)
    {
      // update U with flux_x
      uOut[ID] += (flux_x[tx][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
      uOut[IP] += (flux_x[tx][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
      uOut[IU] += (flux_x[tx][ty][IU]-flux_x[tx+1][ty][IU])*dtdx;
      uOut[IV] += (flux_x[tx][ty][IV]-flux_x[tx+1][ty][IV])*dtdx;
  
      // update U with flux_y : watchout! IU and IV are swapped !
      uOut[ID] += (flux_y[tx][ty][ID]-flux_y[tx][ty+1][ID])*dtdx;
      uOut[IP] += (flux_y[tx][ty][IP]-flux_y[tx][ty+1][IP])*dtdx;
      uOut[IU] += (flux_y[tx][ty][IV]-flux_y[tx][ty+1][IV])*dtdx;
      uOut[IV] += (flux_y[tx][ty][IU]-flux_y[tx][ty+1][IU])*dtdx;

      // update momentum using gravity source term
      if (gravityEnabled) {
	uOut[IU] += HALF_F*(uIn[ID]+uOut[ID])*dt*gravin[elemOffset+IX*arraySize]*dt;
	uOut[IV] += HALF_F*(uIn[ID]+uOut[ID])*dt*gravin[elemOffset+IY*arraySize]*dt;
      }

      // actually perform update on external device memory
      int offset = elemOffset;
      Uout[offset] = uOut[ID];  offset += arraySize;
      Uout[offset] = uOut[IP];  offset += arraySize;
      Uout[offset] = uOut[IU];  offset += arraySize;
      Uout[offset] = uOut[IV];

    }
      
} // kernel_godunov_unsplit_2d

/*****************************************
 *** *** GODUNOV UNSPLIT 3D KERNEL *** ***
 *****************************************/

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
// newer tested
#define UNSPLIT_BLOCK_DIMX_3D		16
#define UNSPLIT_BLOCK_INNER_DIMX_3D	(UNSPLIT_BLOCK_DIMX_3D-4)
#define UNSPLIT_BLOCK_DIMY_3D		12
#define UNSPLIT_BLOCK_INNER_DIMY_3D	(UNSPLIT_BLOCK_DIMY_3D-4)
#else // simple precision
#define UNSPLIT_BLOCK_DIMX_3D		16
#define UNSPLIT_BLOCK_INNER_DIMX_3D	(UNSPLIT_BLOCK_DIMX_3D-4)
#define UNSPLIT_BLOCK_DIMY_3D		12
#define UNSPLIT_BLOCK_INNER_DIMY_3D	(UNSPLIT_BLOCK_DIMY_3D-4)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 3D data (in place computation).
 * 
 * The main difference with the 2D kernel is that we perform a sweep
 * along the 3rd direction.
 *
 * In place computation is probably not efficient, we need to store to
 * much data (4 planes) into shared memory and then to do not have
 * enough shared memory
 *
 * \note Note that when gravity is enabled, only predictor step is performed here; 
 * the source term computation must be done outside !!!
 *
 */
__global__ void kernel_godunov_unsplit_3d(const real_t * __restrict__ Uin, 
					  real_t       *Uout,
					  int pitch, 
					  int imax, 
					  int jmax,
					  int kmax, 
					  real_t dtdx, 
					  real_t dt,
					  bool gravityEnabled)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_3D) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_3D) + ty;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  // we always store in shared 4 consecutive XY-plans
  __shared__ real_t  q[4][UNSPLIT_BLOCK_DIMX_3D][UNSPLIT_BLOCK_DIMY_3D][NVAR_3D];

  real_t *gravin = gParams.arrayList[A_GRAV];

  // index to address the 4 plans of data
  int low, mid, current, top, tmp;
  low=0;
  mid=1;
  current=2;
  top=3;

  // intermediate variables used to build qleft, qright as input to
  // Riemann problem
  real_t qm [THREE_D][NVAR_3D];
  real_t qm1[NVAR_3D];
  real_t qm2[NVAR_3D];
  real_t qm3[NVAR_3D];

  real_t qp [THREE_D][NVAR_3D];
  real_t qp0[THREE_D][NVAR_3D];

  // conservative variables
  real_t uIn[NVAR_3D];
  real_t uOut[NVAR_3D];
  real_t c[4];

  /*
   * initialize q with the first 4 plans
   */
  for (int k=0, elemOffset = i + pitch * j; 
       k < 4;
       ++k, elemOffset += (pitch*jmax) ) {

    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax)
      {
	
	// Gather conservative variables
	int offset = elemOffset;
	uIn[ID] = Uin[offset];  offset += arraySize;
	uIn[IP] = Uin[offset];  offset += arraySize;
	uIn[IU] = Uin[offset];  offset += arraySize;
	uIn[IV] = Uin[offset];  offset += arraySize;
	uIn[IW] = Uin[offset];
	
	//Convert to primitive variables
	constoprim_3D(uIn, q[k][tx][ty], c[k]);
      }
  } // end loading the first 4 plans
  __syncthreads();

  /*
   * loop over all X-Y-planes starting at z=2 as the current plane.
   * Note that elemOffset is used in the update stage
   */
  for (int k=2, elemOffset = i + pitch * (j + jmax * 2);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i > 1 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D-1 and
       j > 1 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D-1)
      {
	// qNeighbors is used for trace computations
	real_t qNeighbors[2*THREE_D][NVAR_3D];
	
	// Characteristic tracing : compute qp0
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qNeighbors[0][iVar] = q[current][tx+1][ty  ][iVar];
	  qNeighbors[1][iVar] = q[current][tx-1][ty  ][iVar];
	  qNeighbors[2][iVar] = q[current][tx  ][ty+1][iVar];
	  qNeighbors[3][iVar] = q[current][tx  ][ty-1][iVar];
	  qNeighbors[4][iVar] = q[top    ][tx  ][ty  ][iVar];
	  qNeighbors[5][iVar] = q[mid    ][tx  ][ty  ][iVar];
	}	
	trace_unsplit<THREE_D, NVAR_3D>(q[current][tx][ty], 
					qNeighbors, c[current], dtdx, qm, qp0);
	
	// gravity predictor on velocity components of qp0's
	if (gravityEnabled) {
	  qp0[0][IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qp0[0][IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qp0[0][IW] += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	  
	  qp0[1][IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qp0[1][IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qp0[1][IW] += HALF_F * dt * gravin[elemOffset+IZ*arraySize];

	  qp0[2][IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qp0[2][IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qp0[2][IW] += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	}

	// Characteristic tracing : compute qm_x[1] (shift x -1)
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qNeighbors[0][iVar] = q[current][tx  ][ty  ][iVar];
	  qNeighbors[1][iVar] = q[current][tx-2][ty  ][iVar];
	  qNeighbors[2][iVar] = q[current][tx-1][ty+1][iVar];
	  qNeighbors[3][iVar] = q[current][tx-1][ty-1][iVar];
	  qNeighbors[4][iVar] = q[top    ][tx-1][ty  ][iVar];
	  qNeighbors[5][iVar] = q[mid    ][tx-1][ty  ][iVar];
	}
	trace_unsplit<THREE_D, NVAR_3D>(q[current][tx-1][ty], 
					qNeighbors, c[current], dtdx, qm, qp);
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qm1[iVar] = qm[0][iVar];
	}

	// gravity predictor on velocity components of qm1
	if (gravityEnabled) {
	  qm1[IU] += HALF_F * dt * gravin[elemOffset-1+IX*arraySize];
	  qm1[IV] += HALF_F * dt * gravin[elemOffset-1+IY*arraySize];
	  qm1[IW] += HALF_F * dt * gravin[elemOffset-1+IZ*arraySize];
	}
	
	// Characteristic tracing : compute qm_y[2] (shift y -1)
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qNeighbors[0][iVar] = q[current][tx+1][ty-1][iVar];
	  qNeighbors[1][iVar] = q[current][tx-1][ty-1][iVar];
	  qNeighbors[2][iVar] = q[current][tx  ][ty  ][iVar];
	  qNeighbors[3][iVar] = q[current][tx  ][ty-2][iVar];
	  qNeighbors[4][iVar] = q[top    ][tx  ][ty-1][iVar];
	  qNeighbors[5][iVar] = q[mid    ][tx  ][ty-1][iVar];
	}
	trace_unsplit<THREE_D, NVAR_3D>(q[current][tx][ty-1], 
					qNeighbors, c[current], dtdx, qm, qp);
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qm2[iVar] = qm[1][iVar];
	}
	
	// gravity predictor on velocity components of qm2
	if (gravityEnabled) {
	  qm2[IU] += HALF_F * dt * gravin[elemOffset-pitch+IX*arraySize];
	  qm2[IV] += HALF_F * dt * gravin[elemOffset-pitch+IY*arraySize];
	  qm2[IW] += HALF_F * dt * gravin[elemOffset-pitch+IZ*arraySize];
	}

	// Characteristic tracing : compute qm_z[3] (shift z -1)
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qNeighbors[0][iVar] = q[mid    ][tx+1][ty  ][iVar];
	  qNeighbors[1][iVar] = q[mid    ][tx-1][ty  ][iVar];
	  qNeighbors[2][iVar] = q[mid    ][tx  ][ty+1][iVar];
	  qNeighbors[3][iVar] = q[mid    ][tx  ][ty-1][iVar];
	  qNeighbors[4][iVar] = q[current][tx  ][ty  ][iVar];
	  qNeighbors[5][iVar] = q[low    ][tx  ][ty  ][iVar];
	}	
	trace_unsplit<THREE_D, NVAR_3D>(q[mid][tx][ty], 
					qNeighbors, c[mid], dtdx, qm, qp);
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qm3[iVar] = qm[2][iVar];
	}

	// gravity predictor on velocity components of qm3
	if (gravityEnabled) {
	  qm3[IU] += HALF_F * dt * gravin[elemOffset-pitch*jmax+IX*arraySize];
	  qm3[IV] += HALF_F * dt * gravin[elemOffset-pitch*jmax+IY*arraySize];
	  qm3[IW] += HALF_F * dt * gravin[elemOffset-pitch*jmax+IZ*arraySize];
	}

      } 
    // end trace/slope computations, we have all we need to
    // compute qleft/qright now
    __syncthreads();
    
    /*
     * Now, compute fluxes !
     */
    // a nice trick: reuse q[low] to store fluxes, as we don't need then anymore
    // also remember that q[low] will become q[top] after the flux
    // computations and hydro update
    real_t (&flux)[UNSPLIT_BLOCK_DIMX_3D][UNSPLIT_BLOCK_DIMY_3D][NVAR_3D] = q[low];
    
    // solve Riemann problems at X interfaces
    flux[tx][ty][ID] = ZERO_F;
    flux[tx][ty][IP] = ZERO_F;
    flux[tx][ty][IU] = ZERO_F;
    flux[tx][ty][IV] = ZERO_F;
    flux[tx][ty][IW] = ZERO_F;
    __syncthreads();
    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D-1 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D-1)
      {
	real_t qgdnv[NVAR_3D];
	
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t (&qleft_x)[NVAR_3D]  = qm1;
	real_t (&qright_x)[NVAR_3D] = qp0[0];      
	riemann<NVAR_3D>(qleft_x, qright_x, qgdnv, flux[tx][ty]);
      } // end solve Riemann at X interfaces
    __syncthreads();
    
    // update hydro along X
    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D-2 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D-2)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offset = elemOffset;
	uOut[ID] = Uin[offset];  offset += arraySize;
	uOut[IP] = Uin[offset];  offset += arraySize;
	uOut[IU] = Uin[offset];  offset += arraySize;
	uOut[IV] = Uin[offset];  offset += arraySize;
	uOut[IW] = Uin[offset];

	// update U with flux
	uOut[ID] += (flux[tx][ty][ID]-flux[tx+1][ty][ID])*dtdx;
	uOut[IP] += (flux[tx][ty][IP]-flux[tx+1][ty][IP])*dtdx;
	uOut[IU] += (flux[tx][ty][IU]-flux[tx+1][ty][IU])*dtdx;
	uOut[IV] += (flux[tx][ty][IV]-flux[tx+1][ty][IV])*dtdx;
	uOut[IW] += (flux[tx][ty][IW]-flux[tx+1][ty][IW])*dtdx;
	
      } // end update hydro along X
    __syncthreads();
    
    // solve Riemann problems at Y interfaces
    flux[tx][ty][ID] = ZERO_F;
    flux[tx][ty][IP] = ZERO_F;
    flux[tx][ty][IU] = ZERO_F;
    flux[tx][ty][IV] = ZERO_F;
    flux[tx][ty][IW] = ZERO_F;
    __syncthreads();
    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D-1 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D-1)
      {
	real_t qgdnv[NVAR_3D];
	
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t (&qleft_y)[NVAR_3D]  = qm2;
	real_t (&qright_y)[NVAR_3D] = qp0[1];
	// watchout swap IU and IV
	swap_value(qleft_y[IU],qleft_y[IV]);
	swap_value(qright_y[IU],qright_y[IV]);
	riemann<NVAR_3D>(qleft_y, qright_y, qgdnv, flux[tx][ty]);
      } //  end solve Riemann at Y interfaces
    __syncthreads();
    
    // update hydro along Y
    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D-2 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D-2)
      {
	// update U with flux : watchout! IU and IV are swapped !
	uOut[ID] += (flux[tx][ty][ID]-flux[tx][ty+1][ID])*dtdx;
	uOut[IP] += (flux[tx][ty][IP]-flux[tx][ty+1][IP])*dtdx;
	uOut[IU] += (flux[tx][ty][IV]-flux[tx][ty+1][IV])*dtdx;
	uOut[IV] += (flux[tx][ty][IU]-flux[tx][ty+1][IU])*dtdx;
	uOut[IW] += (flux[tx][ty][IW]-flux[tx][ty+1][IW])*dtdx;
	
      } // end update hydro along Y
    __syncthreads();
    
    // solve Riemann problems at Z interfaces
    real_t flux_z[NVAR_3D];
    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D-1 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D-1)
      {
	real_t qgdnv[NVAR_3D];
	
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t (&qleft_z)[NVAR_3D]  = qm3;
	real_t (&qright_z)[NVAR_3D] = qp0[2];
	// watchout swap IU and IW
	swap_value(qleft_z[IU],qleft_z[IW]);
	swap_value(qright_z[IU],qright_z[IW]);
	riemann<NVAR_3D>(qleft_z, qright_z, qgdnv, flux_z);
      } // end solve Riemann at Z interfaces
    __syncthreads();
    
    // update hydro along Z
    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D-2 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D-2)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
    	// update U with flux : watchout! IU and IW are swapped !
    	uOut[ID] += (flux_z[ID])*dtdx;
    	uOut[IP] += (flux_z[IP])*dtdx;
    	uOut[IU] += (flux_z[IW])*dtdx;
    	uOut[IV] += (flux_z[IV])*dtdx;
    	uOut[IW] += (flux_z[IU])*dtdx;
	
    	/*
    	 * update at position z-1.
	 * Note that position z-1 has already been partialy updated in
	 * the previous interation (for loop over k).
    	 */
    	// update U with flux : watchout! IU and IW are swapped !
    	int offset = elemOffset - pitch*jmax;
    	Uout[offset] -= (flux_z[ID])*dtdx; offset += arraySize;
    	Uout[offset] -= (flux_z[IP])*dtdx; offset += arraySize;
    	Uout[offset] -= (flux_z[IW])*dtdx; offset += arraySize;
    	Uout[offset] -= (flux_z[IV])*dtdx; offset += arraySize;
    	Uout[offset] -= (flux_z[IU])*dtdx;
	
	// actually perform the update on external device memory
	offset = elemOffset;
	Uout[offset] = uOut[ID];  offset += arraySize;
	Uout[offset] = uOut[IP];  offset += arraySize;
	Uout[offset] = uOut[IU];  offset += arraySize;
	Uout[offset] = uOut[IV];  offset += arraySize;
	Uout[offset] = uOut[IW];
	
      } // end update along Z
    __syncthreads();
    
    /*
     * swap planes
     */
    tmp = low;
    low = mid;
    mid = current;
    current = top;
    top = tmp;
    __syncthreads();

    /*
     * load new data (located in plane at z=k+2) and place them into top plane
     */
    if (k<kmax-2) {
      if(i >= 0 and i < imax and 
	 j >= 0 and j < jmax)
	{
	  
	  // Gather conservative variables
	  int offset = i + pitch * (j + jmax * (k+2));
	  uIn[ID] = Uin[offset];  offset += arraySize;
	  uIn[IP] = Uin[offset];  offset += arraySize;
	  uIn[IU] = Uin[offset];  offset += arraySize;
	  uIn[IV] = Uin[offset];  offset += arraySize;
	  uIn[IW] = Uin[offset];
	  
	  //Convert to primitive variables
	  constoprim_3D(uIn, q[top][tx][ty], c[top]);
	}
    }
    __syncthreads();
    
  } // end for k
  
} //  kernel_godunov_unsplit_3d

/*
 *
 * Here are CUDA kernel implementing hydro unsplit scheme version 1
 * 
 *
 */

/*******************************************************
 *** COMPUTE PRIMITIVE VARIABLES 2D KERNEL version 1 ***
 *******************************************************/

#ifdef USE_DOUBLE
#define PRIM_VAR_BLOCK_DIMX_2D_V1	16
#define PRIM_VAR_BLOCK_DIMY_2D_V1	16
#else // simple precision
#define PRIM_VAR_BLOCK_DIMX_2D_V1	16
#define PRIM_VAR_BLOCK_DIMY_2D_V1	16
#endif // USE_DOUBLE

/**
 * Compute primitive variables 
 *
 * \param[in]  Uin  input  convervative variable array 
 * \param[out] Qout output primitive variable array
 */
__global__ void kernel_hydro_compute_primitive_variables_2D(const real_t * __restrict__ Uin,
							    real_t       *Qout,
							    int pitch, 
							    int imax, 
							    int jmax)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, PRIM_VAR_BLOCK_DIMX_2D_V1) + tx;
  const int j = __mul24(by, PRIM_VAR_BLOCK_DIMY_2D_V1) + ty;
  
  const int arraySize    = pitch * jmax;

  // conservative variables
  real_t uIn [NVAR_2D];
  real_t c;
  
  // Gather conservative variables (at z=k)
  int elemOffset = i + pitch * j;

  if (i < imax and j < jmax) {

    int offset = elemOffset;
    uIn[ID] = Uin[offset];  offset += arraySize;
    uIn[IP] = Uin[offset];  offset += arraySize;
    uIn[IU] = Uin[offset];  offset += arraySize;
    uIn[IV] = Uin[offset];
    
    //Convert to primitive variables
    real_t qTmp[NVAR_2D];
    constoprim_2D(uIn, qTmp, c);
    
    // copy results into output d_Q at z=k
    offset = elemOffset;
    Qout[offset] = qTmp[ID]; offset += arraySize;
    Qout[offset] = qTmp[IP]; offset += arraySize;
    Qout[offset] = qTmp[IU]; offset += arraySize;
    Qout[offset] = qTmp[IV];

  } // end if

} // kernel_hydro_compute_primitive_variables_2D

/*******************************************************
 *** COMPUTE PRIMITIVE VARIABLES 3D KERNEL version 1 ***
 *******************************************************/

#ifdef USE_DOUBLE
#define PRIM_VAR_BLOCK_DIMX_3D_V1	16
#define PRIM_VAR_BLOCK_DIMY_3D_V1	16
#else // simple precision
#define PRIM_VAR_BLOCK_DIMX_3D_V1	16
#define PRIM_VAR_BLOCK_DIMY_3D_V1	16
#endif // USE_DOUBLE

/**
 * Compute primitive variables 
 *
 * \param[in]  Uin  input  convervative variable array 
 * \param[out] Qout output primitive variable array
 */
__global__ void kernel_hydro_compute_primitive_variables_3D(const real_t * __restrict__ Uin,
							    real_t       *Qout,
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
  
  const int i = __mul24(bx, PRIM_VAR_BLOCK_DIMX_3D_V1) + tx;
  const int j = __mul24(by, PRIM_VAR_BLOCK_DIMY_3D_V1) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // conservative variables
  real_t uIn [NVAR_3D];
  real_t c;

  /*
   * loop over k (i.e. z) to compute primitive variables, and store results
   * in external memory buffer Q.
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {

    if (i < imax and j < jmax) {
      
      // Gather conservative variables (at z=k)
      int offset = elemOffset;
      
      uIn[ID] = Uin[offset];  offset += arraySize;
      uIn[IP] = Uin[offset];  offset += arraySize;
      uIn[IU] = Uin[offset];  offset += arraySize;
      uIn[IV] = Uin[offset];  offset += arraySize;
      uIn[IW] = Uin[offset];
    
      //Convert to primitive variables
      real_t qTmp[NVAR_3D];
      constoprim_3D(uIn, qTmp, c);
    
      // copy results into output d_Q at z=k
      offset = elemOffset;
      Qout[offset] = qTmp[ID]; offset += arraySize;
      Qout[offset] = qTmp[IP]; offset += arraySize;
      Qout[offset] = qTmp[IU]; offset += arraySize;
      Qout[offset] = qTmp[IV]; offset += arraySize;
      Qout[offset] = qTmp[IW];

    } // end if

  } // enf for k

} // kernel_hydro_compute_primitive_variables_3D

/*****************************************
 *** COMPUTE TRACE 2D KERNEL version 1 ***
 *****************************************/

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define TRACE_BLOCK_DIMX_2D_V1	16
#define TRACE_BLOCK_INNER_DIMX_2D_V1	(TRACE_BLOCK_DIMX_2D_V1-2)
#define TRACE_BLOCK_DIMY_2D_V1	16
#define TRACE_BLOCK_INNER_DIMY_2D_V1	(TRACE_BLOCK_DIMY_2D_V1-2)
#else // simple precision
#define TRACE_BLOCK_DIMX_2D_V1	16
#define TRACE_BLOCK_INNER_DIMX_2D_V1	(TRACE_BLOCK_DIMX_2D_V1-2)
#define TRACE_BLOCK_DIMY_2D_V1	16
#define TRACE_BLOCK_INNER_DIMY_2D_V1	(TRACE_BLOCK_DIMY_2D_V1-2)
#endif // USE_DOUBLE

/**
 * Compute trace for hydro 2D (implementation version 1).
 *
 * Output are all that is needed to compute fluxes.
 * \see kernel_hydro_flux_update_unsplit_2d_v1
 *
 * All we do here is call :
 * - slope_unsplit_hydro_2d
 * - trace_unsplit_hydro_2d to get output : qm, qp.
 *
 * \param[in] Uin input conservative variable array
 * \param[in] d_Q input primitive    variable array
 * \param[out] d_qm_x qm state along x
 * \param[out] d_qm_y qm state along y
 * \param[out] d_qp_x qp state along x
 * \param[out] d_qp_y qp state along y
 *
 */
__global__ void kernel_hydro_compute_trace_unsplit_2d_v1(const real_t * __restrict__ Uin, 
							 const real_t * __restrict__ d_Q,
							 real_t* d_qm_x,
							 real_t* d_qm_y,
							 real_t* d_qp_x,
							 real_t* d_qp_y,
							 int pitch, 
							 int imax, 
							 int jmax, 
							 real_t dtdx,
							 real_t dtdy,
							 real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, TRACE_BLOCK_INNER_DIMX_2D_V1) + tx;
  const int j = __mul24(by, TRACE_BLOCK_INNER_DIMY_2D_V1) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  __shared__ real_t q[TRACE_BLOCK_DIMX_2D_V1][TRACE_BLOCK_DIMY_2D_V1][NVAR_2D];

  // qm and qp's are output of the trace step
  real_t qm [TWO_D][NVAR_2D];
  real_t qp [TWO_D][NVAR_2D];

  // read primitive variables from d_Q
  if (i<imax and j<jmax) {
    int offset = elemOffset; // 2D index
    q[tx][ty][ID] = d_Q[offset];  offset += arraySize;
    q[tx][ty][IP] = d_Q[offset];  offset += arraySize;
    q[tx][ty][IU] = d_Q[offset];  offset += arraySize;
    q[tx][ty][IV] = d_Q[offset];
  }
  __syncthreads();

  // slope and trace computation (i.e. dq, and then qm, qp)  
  if(i > 0 and i < imax-1 and tx > 0 and tx < TRACE_BLOCK_DIMX_2D_V1-1 and
     j > 0 and j < jmax-1 and ty > 0 and ty < TRACE_BLOCK_DIMY_2D_V1-1)
    {
      
      int offset = elemOffset; // 2D index
      
      real_t (&qPlusX )[NVAR_2D] = q[tx+1][ty  ];
      real_t (&qMinusX)[NVAR_2D] = q[tx-1][ty  ];
      real_t (&qPlusY )[NVAR_2D] = q[tx  ][ty+1];
      real_t (&qMinusY)[NVAR_2D] = q[tx  ][ty-1];

      // hydro slopes  array
      real_t dq[2][NVAR_2D];
      
      // compute hydro slopes dq
      slope_unsplit_hydro_2d(q[tx][ty], 
			     qPlusX, qMinusX, 
			     qPlusY, qMinusY, 
			     dq);

      // compute qm, qp
      trace_unsplit_hydro_2d(q[tx][ty], dq,
			     dtdx, dtdy,
			     qm, qp);
      
      
      // store qm, qp in external memory
      {
	offset  = elemOffset;
	for (int iVar=0; iVar<NVAR_2D; iVar++) {
	  
	  d_qm_x[offset] = qm[0][iVar];
	  d_qm_y[offset] = qm[1][iVar];
	  
	  d_qp_x[offset] = qp[0][iVar];
	  d_qp_y[offset] = qp[1][iVar];
	  
	  offset  += arraySize;
	} // end for iVar
      } // end store qm, qp

    } // end compute slope and trace
  __syncthreads();

} // kernel_hydro_compute_trace_unsplit_2d_v1

/*****************************************
 *** COMPUTE TRACE 3D KERNEL version 1 ***
 *****************************************/

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
#define TRACE_BLOCK_DIMX_3D_V1	16
#define TRACE_BLOCK_INNER_DIMX_3D_V1	(TRACE_BLOCK_DIMX_3D_V1-2)
#define TRACE_BLOCK_DIMY_3D_V1	16
#define TRACE_BLOCK_INNER_DIMY_3D_V1	(TRACE_BLOCK_DIMY_3D_V1-2)
#else // simple precision
#define TRACE_BLOCK_DIMX_3D_V1	48
#define TRACE_BLOCK_INNER_DIMX_3D_V1	(TRACE_BLOCK_DIMX_3D_V1-2)
#define TRACE_BLOCK_DIMY_3D_V1	10
#define TRACE_BLOCK_INNER_DIMY_3D_V1	(TRACE_BLOCK_DIMY_3D_V1-2)
#endif // USE_DOUBLE

/**
 * Compute trace for hydro 3D (implementation version 1).
 *
 * Output are all that is needed to compute fluxes.
 * \see kernel_hydro_flux_update_unsplit_3d_v1
 *
 * All we do here is call :
 * - slope_unsplit_hydro_3d
 * - trace_unsplit_hydro_3d to get output : qm, qp.
 *
 * \param[in] Uin input conservative variable array
 * \param[in] d_Q input primitive    variable array
 * \param[out] d_qm_x qm state along x
 * \param[out] d_qm_y qm state along y
 * \param[out] d_qm_z qm state along z
 * \param[out] d_qp_x qp state along x
 * \param[out] d_qp_y qp state along y
 * \param[out] d_qp_z qp state along z
 *
 */
__global__ void kernel_hydro_compute_trace_unsplit_3d_v1(const real_t * __restrict__ Uin, 
							 const real_t * __restrict__ d_Q,
							 real_t *d_qm_x,
							 real_t *d_qm_y,
							 real_t *d_qm_z,
							 real_t *d_qp_x,
							 real_t *d_qp_y,
							 real_t *d_qp_z,
							 int pitch, 
							 int imax, 
							 int jmax, 
							 int kmax,
							 real_t dtdx,
							 real_t dtdy,
							 real_t dtdz,
							 real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, TRACE_BLOCK_INNER_DIMX_3D_V1) + tx;
  const int j = __mul24(by, TRACE_BLOCK_INNER_DIMY_3D_V1) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  __shared__ real_t q[TRACE_BLOCK_DIMX_3D_V1][TRACE_BLOCK_DIMY_3D_V1][NVAR_3D];

  // qm and qp's are output of the trace step
  real_t qm [THREE_D][NVAR_3D];
  real_t qp [THREE_D][NVAR_3D];

  // primitive variables at different z
  real_t qZplus1  [NVAR_3D];
  real_t qZminus1 [NVAR_3D];

  /*
   * initialize q (primitive variables ) in the 2 first XY-planes
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < 2; 
       ++k, elemOffset += (pitch*jmax)) {

    if(i < imax and j < jmax) {
      
      int offset = elemOffset;
      
      // read primitive variables from d_Q
      if (k==0) {
	qZminus1[ID] = d_Q[offset];  offset += arraySize;
	qZminus1[IP] = d_Q[offset];  offset += arraySize;
	qZminus1[IU] = d_Q[offset];  offset += arraySize;
	qZminus1[IV] = d_Q[offset];  offset += arraySize;
	qZminus1[IW] = d_Q[offset];
      } else { // k == 1
	q[tx][ty][ID] = d_Q[offset];  offset += arraySize;
	q[tx][ty][IP] = d_Q[offset];  offset += arraySize;
	q[tx][ty][IU] = d_Q[offset];  offset += arraySize;
	q[tx][ty][IV] = d_Q[offset];  offset += arraySize;
	q[tx][ty][IW] = d_Q[offset];
      }

    } // end if
    __syncthreads();

  } // end for k - initialize q

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i < imax and j < jmax) {
      // data fetch :
      // get q at z+1
      int offset = elemOffset + pitch*jmax; // z+1	 
      qZplus1[ID] = d_Q[offset];  offset += arraySize;
      qZplus1[IP] = d_Q[offset];  offset += arraySize;
      qZplus1[IU] = d_Q[offset];  offset += arraySize;
      qZplus1[IV] = d_Q[offset];  offset += arraySize;
      qZplus1[IW] = d_Q[offset];
    } // end if
    __syncthreads();
	     
    // slope and trace computation (i.e. dq, and then qm, qp)
    
    if(i > 0 and i < imax-1 and tx > 0 and tx < TRACE_BLOCK_DIMX_3D_V1-1 and
       j > 0 and j < jmax-1 and ty > 0 and ty < TRACE_BLOCK_DIMY_3D_V1-1)
      {
	
	int offset = elemOffset; // 3D index
	
	real_t (&qPlusX )[NVAR_3D] = q[tx+1][ty  ];
	real_t (&qMinusX)[NVAR_3D] = q[tx-1][ty  ];
	real_t (&qPlusY )[NVAR_3D] = q[tx  ][ty+1];
	real_t (&qMinusY)[NVAR_3D] = q[tx  ][ty-1];
	real_t (&qPlusZ )[NVAR_3D] = qZplus1;  // q[tx][ty] at z+1
	real_t (&qMinusZ)[NVAR_3D] = qZminus1; // q[tx][ty] at z-1
					       // (stored from z
					       // previous
					       // iteration)

	// hydro slopes  array
	real_t dq[3][NVAR_3D];
      
	// compute hydro slopes dq
	/*slope_unsplit_3d(q[tx][ty], 
			 qPlusX, qMinusX, 
			 qPlusY, qMinusY,
			 qPlusZ, qMinusZ,
			 dq);*/
	slope_unsplit_3d_v1(q[tx][ty], qPlusX, qMinusX, dq[0]);
	slope_unsplit_3d_v1(q[tx][ty], qPlusY, qMinusY, dq[1]);
	slope_unsplit_3d_v1(q[tx][ty], qPlusZ, qMinusZ, dq[2]);
	
	// compute qm, qp
	trace_unsplit_hydro_3d(q[tx][ty], dq,
			       dtdx, dtdy, dtdz,
			       qm, qp);
      
	// store qm, qp in external memory
	{
	  offset  = elemOffset;
	  for (int iVar=0; iVar<NVAR_3D; iVar++) {
	    
	    d_qm_x[offset] = qm[0][iVar];
	    d_qm_y[offset] = qm[1][iVar];
	    d_qm_z[offset] = qm[2][iVar];
	    
	    d_qp_x[offset] = qp[0][iVar];
	    d_qp_y[offset] = qp[1][iVar];
	    d_qp_z[offset] = qp[2][iVar];
	    
	    offset  += arraySize;
	  } // end for iVar
	} // end store qm, qp
	
      } // end compute slope and trace
    __syncthreads();

    // rotate buffer
    {
      // q                    become  qZminus1
      // qZplus1 at z+1       become  q
      for (int iVar=0; iVar<NVAR_3D; iVar++) {
	qZminus1 [iVar] = q[tx][ty][iVar];
	q[tx][ty][iVar] = qZplus1  [iVar];
      }
    }
    __syncthreads();
    
  } // end for k

} // kernel_hydro_compute_trace_unsplit_3d_v1

/*****************************************
 *** COMPUTE TRACE 3D KERNEL version 2 ***
 *****************************************/

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
#define TRACE_BLOCK_DIMX_3D_V2	16
#define TRACE_BLOCK_INNER_DIMX_3D_V2	(TRACE_BLOCK_DIMX_3D_V2-2)
#define TRACE_BLOCK_DIMY_3D_V2	16
#define TRACE_BLOCK_INNER_DIMY_3D_V2	(TRACE_BLOCK_DIMY_3D_V2-2)
#else // simple precision
# define TRACE_BLOCK_DIMX_3D_V2	16
# define TRACE_BLOCK_INNER_DIMX_3D_V2	(TRACE_BLOCK_DIMX_3D_V2-2)
# define TRACE_BLOCK_DIMY_3D_V2	8
# define TRACE_BLOCK_INNER_DIMY_3D_V2	(TRACE_BLOCK_DIMY_3D_V2-2)
#endif // USE_DOUBLE

/**
 * Compute trace for hydro 3D (implementation version 1).
 *
 * Output are all that is needed to compute fluxes.
 * \see kernel_hydro_flux_update_unsplit_3d_v1
 *
 * All we do here is call :
 * - slope_unsplit_hydro_3d
 * - trace_unsplit_hydro_3d to get output : qm, qp.
 *
 * \param[in] Uin input conservative variable array
 * \param[in] d_Q input primitive    variable array
 * \param[out] d_qm_x qm state along x
 * \param[out] d_qm_y qm state along y
 * \param[out] d_qm_z qm state along z
 * \param[out] d_qp_x qp state along x
 * \param[out] d_qp_y qp state along y
 * \param[out] d_qp_z qp state along z
 *
 */
__global__ void kernel_hydro_compute_trace_unsplit_3d_v2(const real_t * __restrict__ Uin, 
							 const real_t * __restrict__ d_Q,
							 real_t *d_qm_x,
							 real_t *d_qm_y,
							 real_t *d_qm_z,
							 real_t *d_qp_x,
							 real_t *d_qp_y,
							 real_t *d_qp_z,
							 int pitch, 
							 int imax, 
							 int jmax, 
							 int kmax,
							 real_t dtdx,
							 real_t dtdy,
							 real_t dtdz,
							 real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, TRACE_BLOCK_INNER_DIMX_3D_V2) + tx;
  const int j = __mul24(by, TRACE_BLOCK_INNER_DIMY_3D_V2) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  __shared__ qStateHydro q[TRACE_BLOCK_DIMX_3D_V2][TRACE_BLOCK_DIMY_3D_V2];

  // primitive variables at different z
  __shared__ qStateHydro qZplus1[TRACE_BLOCK_DIMX_3D_V2][TRACE_BLOCK_DIMY_3D_V2];
  __shared__ qStateHydro qZminus1[TRACE_BLOCK_DIMX_3D_V2][TRACE_BLOCK_DIMY_3D_V2];

  // some aliases 
  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  real_t &gamma  = ::gParams.gamma0;
  //real_t &dx     = ::gParams.dx;
  
  /*
   * initialize q (primitive variables ) in the 2 first XY-planes
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < 2; 
       ++k, elemOffset += (pitch*jmax)) {

    if(i < imax and j < jmax) {
      
      int offset = elemOffset;
      
      // read primitive variables from d_Q
      if (k==0) {
	qZminus1[tx][ty].D = d_Q[offset];  offset += arraySize;
	qZminus1[tx][ty].P = d_Q[offset];  offset += arraySize;
	qZminus1[tx][ty].U = d_Q[offset];  offset += arraySize;
	qZminus1[tx][ty].V = d_Q[offset];  offset += arraySize;
	qZminus1[tx][ty].W = d_Q[offset];
      } else { // k == 1
	q[tx][ty].D = d_Q[offset];  offset += arraySize;
	q[tx][ty].P = d_Q[offset];  offset += arraySize;
	q[tx][ty].U = d_Q[offset];  offset += arraySize;
	q[tx][ty].V = d_Q[offset];  offset += arraySize;
	q[tx][ty].W = d_Q[offset];
      }

    } // end if
    __syncthreads();

  } // end for k - initialize q

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i < imax and j < jmax) {
      // data fetch :
      // get q at z+1
      int offset = elemOffset + pitch*jmax; // z+1	 
      qZplus1[tx][ty].D = d_Q[offset];  offset += arraySize;
      qZplus1[tx][ty].P = d_Q[offset];  offset += arraySize;
      qZplus1[tx][ty].U = d_Q[offset];  offset += arraySize;
      qZplus1[tx][ty].V = d_Q[offset];  offset += arraySize;
      qZplus1[tx][ty].W = d_Q[offset];
    } // end if
    __syncthreads();
	     
    // slope and trace computation (i.e. dq, and then qm, qp)
    
    if(i > 0 and i < imax-1 and tx > 0 and tx < TRACE_BLOCK_DIMX_3D_V2-1 and
       j > 0 and j < jmax-1 and ty > 0 and ty < TRACE_BLOCK_DIMY_3D_V2-1)
      {
	
	int offset = elemOffset; // 3D index
	
	// hydro slopes  array
	qStateHydro dqX, dqY, dqZ;
      
	// compute hydro slopes dqX
	{
	  qStateHydro &qPlusX  = q[tx+1][ty  ];
	  qStateHydro &qMinusX = q[tx-1][ty  ];

	  slope_unsplit_3d_v2(q[tx][ty], qPlusX, qMinusX, dqX);
	}

	// compute hydro slopes dqX
	{
	  qStateHydro &qPlusY  = q[tx  ][ty+1];
	  qStateHydro &qMinusY = q[tx  ][ty-1];

	  slope_unsplit_3d_v2(q[tx][ty], qPlusY, qMinusY, dqY);
	}

	// compute hydro slopes dqZ
	{
	  qStateHydro &qPlusZ   = qZplus1[tx][ty];
	  qStateHydro &qMinusZ  = qZminus1[tx][ty];
	  slope_unsplit_3d_v2(q[tx][ty], qPlusZ, qMinusZ, dqZ);
	}
	
	// compute qm, qp
	{
	  // Cell centered values
	  // here we can't use reference, we need to keep q unchanged
	  // to be able to copy in qZminus at the end
	  real_t r = q[tx][ty].D;
	  real_t p = q[tx][ty].P;
	  real_t u = q[tx][ty].U;
	  real_t v = q[tx][ty].V;
	  real_t w = q[tx][ty].W;
	  
	  // Cell centered TVD slopes in X direction
	  real_t& drx = dqX.D; drx *= HALF_F;
	  real_t& dpx = dqX.P; dpx *= HALF_F;
	  real_t& dux = dqX.U; dux *= HALF_F;
	  real_t& dvx = dqX.V; dvx *= HALF_F;
	  real_t& dwx = dqX.W; dwx *= HALF_F;
	  
	  // Cell centered TVD slopes in Y direction
	  real_t& dry = dqY.D; dry *= HALF_F;
	  real_t& dpy = dqY.P; dpy *= HALF_F;
	  real_t& duy = dqY.U; duy *= HALF_F;
	  real_t& dvy = dqY.V; dvy *= HALF_F;
	  real_t& dwy = dqY.W; dwy *= HALF_F;
	  
	  // Cell centered TVD slopes in Z direction
	  real_t& drz = dqZ.D; drz *= HALF_F;
	  real_t& dpz = dqZ.P; dpz *= HALF_F;
	  real_t& duz = dqZ.U; duz *= HALF_F;
	  real_t& dvz = dqZ.V; dvz *= HALF_F;
	  real_t& dwz = dqZ.W; dwz *= HALF_F;
	  
	  
	  // Source terms (including transverse derivatives)
	  real_t sr0, su0, sv0, sw0, sp0;
	  
	  sr0 = (-u*drx-dux*r)*dtdx + (-v*dry-dvy*r)*dtdy + (-w*drz-dwz*r)*dtdz;
	  su0 = (-u*dux-dpx/r)*dtdx + (-v*duy      )*dtdy + (-w*duz      )*dtdz; 
	  sv0 = (-u*dvx      )*dtdx + (-v*dvy-dpy/r)*dtdy + (-w*dvz      )*dtdz;
	  sw0 = (-u*dwx      )*dtdx + (-v*dwy      )*dtdy + (-w*dwz-dpz/r)*dtdz; 
	  sp0 = (-u*dpx-dux*gamma*p)*dtdx + (-v*dpy-dvy*gamma*p)*dtdy + (-w*dpz-dwz*gamma*p)*dtdz;
	  
	  // Update in time the  primitive variables
	  r = r + sr0;
	  u = u + su0;
	  v = v + sv0;
	  w = w + sw0;
	  p = p + sp0;
	  
	  real_t tmp_D, tmp_P;

	  offset  = elemOffset;

	  // Face averaged right state at left interface
	  tmp_D = FMAX(smallR        ,  r - drx);
	  tmp_P = FMAX(smallp * tmp_D,  p - dpx);
	  d_qp_x[offset+ID*arraySize] = tmp_D;
	  d_qp_x[offset+IP*arraySize] = tmp_P;
	  d_qp_x[offset+IU*arraySize] = u - dux;
	  d_qp_x[offset+IV*arraySize] = v - dvx;
	  d_qp_x[offset+IW*arraySize] = w - dwx;
	  
	  // Face averaged left state at right interface
	  tmp_D = FMAX(smallR        ,  r + drx);
	  tmp_P = FMAX(smallp * tmp_D,  p + dpx);
	  d_qm_x[offset+ID*arraySize] = tmp_D;
	  d_qm_x[offset+IP*arraySize] = tmp_P;
	  d_qm_x[offset+IU*arraySize] = u + dux;
	  d_qm_x[offset+IV*arraySize] = v + dvx;
	  d_qm_x[offset+IW*arraySize] = w + dwx;
	  
	  // Face averaged top state at bottom interface
	  tmp_D = FMAX(smallR        ,  r - dry);
	  tmp_P = FMAX(smallp * tmp_D,  p - dpy);
	  d_qp_y[offset+ID*arraySize] = tmp_D;
	  d_qp_y[offset+IP*arraySize] = tmp_P;
	  d_qp_y[offset+IU*arraySize] = u - duy;
	  d_qp_y[offset+IV*arraySize] = v - dvy;
	  d_qp_y[offset+IW*arraySize] = w - dwy;
	  
	  // Face averaged bottom state at top interface
	  tmp_D = FMAX(smallR        ,  r + dry);
	  tmp_P = FMAX(smallp * tmp_D,  p + dpy);
	  d_qm_y[offset+ID*arraySize] = tmp_D;
	  d_qm_y[offset+IP*arraySize] = tmp_P;
	  d_qm_y[offset+IU*arraySize] = u + duy;
	  d_qm_y[offset+IV*arraySize] = v + dvy;
	  d_qm_y[offset+IW*arraySize] = w + dwy;
	  
	  // Face averaged front state at back interface
	  tmp_D = FMAX(smallR        ,  r - drz);
	  tmp_P = FMAX(smallp * tmp_D,  p - dpz);
	  d_qp_z[offset+ID*arraySize] = tmp_D;
	  d_qp_z[offset+IP*arraySize] = tmp_P;
	  d_qp_z[offset+IU*arraySize] = u - duz;
	  d_qp_z[offset+IV*arraySize] = v - dvz;
	  d_qp_z[offset+IW*arraySize] = w - dwz;
	  
	  // Face averaged back state at front interface
	  tmp_D = FMAX(smallR        ,  r + drz);
	  tmp_P = FMAX(smallp * tmp_D,  p + dpz);
	  d_qm_z[offset+ID*arraySize] = tmp_D;
	  d_qm_z[offset+IP*arraySize] = tmp_P;
	  d_qm_z[offset+IU*arraySize] = u + duz;
	  d_qm_z[offset+IV*arraySize] = v + dvz;
	  d_qm_z[offset+IW*arraySize] = w + dwz;

	} // end compute qm, qp
      
      } // end compute slope and trace
    __syncthreads();

    // rotate buffer
    {
      // q                    become  qZminus1
      // qZplus1 at z+1       become  q
      
      qZminus1[tx][ty].D = q[tx][ty].D;
      qZminus1[tx][ty].P = q[tx][ty].P;
      qZminus1[tx][ty].U = q[tx][ty].U;
      qZminus1[tx][ty].V = q[tx][ty].V;
      qZminus1[tx][ty].W = q[tx][ty].W;

      q[tx][ty].D = qZplus1[tx][ty].D;
      q[tx][ty].P = qZplus1[tx][ty].P;
      q[tx][ty].U = qZplus1[tx][ty].U;
      q[tx][ty].V = qZplus1[tx][ty].V;
      q[tx][ty].W = qZplus1[tx][ty].W;

    }
    __syncthreads();
    
  } // end for k

} // kernel_hydro_compute_trace_unsplit_3d_v2

/*********************************************************
 *** UPDATE CONSERVATIVE VAR ARRAY 2D KERNEL version 1 ***
 *********************************************************/
#ifdef USE_DOUBLE
#define UPDATE_BLOCK_DIMX_2D_V1	16
#define UPDATE_BLOCK_INNER_DIMX_2D_V1	(UPDATE_BLOCK_DIMX_2D_V1-1)
#define UPDATE_BLOCK_DIMY_2D_V1	16
#define UPDATE_BLOCK_INNER_DIMY_2D_V1	(UPDATE_BLOCK_DIMY_2D_V1-1)
#else // simple precision
#define UPDATE_BLOCK_DIMX_2D_V1	16
#define UPDATE_BLOCK_INNER_DIMX_2D_V1	(UPDATE_BLOCK_DIMX_2D_V1-1)
#define UPDATE_BLOCK_DIMY_2D_V1	16
#define UPDATE_BLOCK_INNER_DIMY_2D_V1	(UPDATE_BLOCK_DIMY_2D_V1-1)
#endif // USE_DOUBLE

/**
 * Update hydro conservative variables 2D (implementation version 1).
 * 
 * This is the final kernel, that given the qm, qp states compute
 * fluxes and perform update.
 *
 * \see kernel_hydro_compute_trace_unsplit_2d_v1 (computation of qm, qp buffer)
 *
 * \param[in]  Uin  input conservative variable array
 * \param[out] Uout ouput conservative variable array
 * \param[in] d_qm_x qm state along x
 * \param[in] d_qm_y qm state along y
 * \param[in] d_qp_x qp state along x
 * \param[in] d_qp_y qp state along y
 *
 */
__global__ void kernel_hydro_flux_update_unsplit_2d_v1(const real_t * __restrict__ Uin, 
						       real_t       *Uout,
						       const real_t * __restrict__ d_qm_x,
						       const real_t * __restrict__ d_qm_y,
						       const real_t * __restrict__ d_qp_x,
						       const real_t * __restrict__ d_qp_y,
						       int pitch, 
						       int imax, 
						       int jmax,
						       real_t dtdx, 
						       real_t dtdy,
						       real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UPDATE_BLOCK_INNER_DIMX_2D_V1) + tx;
  const int j = __mul24(by, UPDATE_BLOCK_INNER_DIMY_2D_V1) + ty;
  
  const int arraySize    = pitch * jmax;

  // flux computation
  __shared__ real_t flux[UPDATE_BLOCK_DIMX_2D_V1][UPDATE_BLOCK_DIMY_2D_V1][NVAR_2D];

  // qm and qp's are output of the trace step
  //real_t qm [TWO_D][NVAR_2D];
  //real_t qp [TWO_D][NVAR_2D];
 
  // conservative variables
  real_t uOut[NVAR_2D];
  real_t qgdnv[NVAR_2D];
  //real_t c;

  int elemOffset = i + pitch * j;
  
  /*
   * Compute fluxes at X-interfaces.
   */
  // re-use flux as flux_x
  real_t (&flux_x)[UPDATE_BLOCK_DIMX_2D_V1][UPDATE_BLOCK_DIMY_2D_V1][NVAR_2D] = flux;
  flux_x[tx][ty][ID] = ZERO_F;
  flux_x[tx][ty][IP] = ZERO_F;
  flux_x[tx][ty][IU] = ZERO_F;
  flux_x[tx][ty][IV] = ZERO_F;
  __syncthreads();
    
  if(i >= 2 and i < imax-1 and
     j >= 2 and j < jmax-1)
    {
      // Solve Riemann problem at X-interfaces and compute fluxes
      real_t   qleft_x [NVAR_2D];
      real_t   qright_x[NVAR_2D];
      
      // set qleft_x by re-reading qm_x from external memory at location x-1
      int offset = elemOffset-1;
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qleft_x[iVar] = d_qm_x[offset];
	offset += arraySize;
      }
      
      // set qright_x by re-reading qp_x from external memory at location x
      offset = elemOffset;
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qright_x[iVar] = d_qp_x[offset];
	offset += arraySize;
      }
      
      riemann<NVAR_2D>(qleft_x, qright_x, qgdnv, flux_x[tx][ty]);
    }  
  __syncthreads();
    
  // update uOut with flux_x
  if(i >= 2 and i < imax-1 and tx < UPDATE_BLOCK_DIMX_2D_V1-1 and
     j >= 2 and j < jmax-1 and ty < UPDATE_BLOCK_DIMY_2D_V1-1)
    {
      // re-read input state into uOut which will in turn serve to
      // update Uout !
      int offset = elemOffset;
      uOut[ID] = Uin[offset];  offset += arraySize;
      uOut[IP] = Uin[offset];  offset += arraySize;
      uOut[IU] = Uin[offset];  offset += arraySize;
      uOut[IV] = Uin[offset];

      uOut[ID] += flux_x[tx  ][ty][ID]*dtdx;
      uOut[ID] -= flux_x[tx+1][ty][ID]*dtdx;
      
      uOut[IP] += flux_x[tx  ][ty][IP]*dtdx;
      uOut[IP] -= flux_x[tx+1][ty][IP]*dtdx;
      
      uOut[IU] += flux_x[tx  ][ty][IU]*dtdx;
      uOut[IU] -= flux_x[tx+1][ty][IU]*dtdx;
      
      uOut[IV] += flux_x[tx  ][ty][IV]*dtdx;
      uOut[IV] -= flux_x[tx+1][ty][IV]*dtdx;

    }
  __syncthreads();

  /*
   * Compute fluxes at Y-interfaces.
   */
  // re-use flux as flux_y
  real_t (&flux_y)[UPDATE_BLOCK_DIMX_2D_V1][UPDATE_BLOCK_DIMY_2D_V1][NVAR_2D] = flux;
  flux_y[tx][ty][ID] = ZERO_F;
  flux_y[tx][ty][IP] = ZERO_F;
  flux_y[tx][ty][IU] = ZERO_F;
  flux_y[tx][ty][IV] = ZERO_F;
  __syncthreads();
  
  if(i >= 2 and i < imax-1 and 
     j >= 2 and j < jmax-1)
    {
      // Solve Riemann problem at Y-interfaces and compute fluxes
      real_t  qleft_y[NVAR_2D];
      real_t qright_y[NVAR_2D];
      
      // set qleft_y by reading qm_y from external memory at location y-1
      int offset = elemOffset-pitch;
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qleft_y[iVar] = d_qm_y[offset];
	offset += arraySize;
      }
      
      // set qright_y by reading qp_y from external memory at location y
      offset = elemOffset;
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qright_y[iVar] = d_qp_y[offset];
	offset += arraySize;
      }
      
      // watchout swap IU and IV
      swap_value(qleft_y[IU],qleft_y[IV]);
      swap_value(qright_y[IU],qright_y[IV]);
      
      riemann<NVAR_2D>(qleft_y, qright_y, qgdnv, flux_y[tx][ty]);
    }  
  __syncthreads();
  
  // update uOut with flux_y
  if(i >= 2 and i < imax-1 and tx < UPDATE_BLOCK_DIMX_2D_V1-1 and
     j >= 2 and j < jmax-1 and ty < UPDATE_BLOCK_DIMY_2D_V1-1)
    {
      // watchout IU and IV are swapped !
      uOut[ID] += flux_y[tx][ty  ][ID]*dtdy;
      uOut[ID] -= flux_y[tx][ty+1][ID]*dtdy;
      
      uOut[IP] += flux_y[tx][ty  ][IP]*dtdy;
      uOut[IP] -= flux_y[tx][ty+1][IP]*dtdy;
      
      uOut[IU] += flux_y[tx][ty  ][IV]*dtdy;
      uOut[IU] -= flux_y[tx][ty+1][IV]*dtdy;
      
      uOut[IV] += flux_y[tx][ty  ][IU]*dtdy;
      uOut[IV] -= flux_y[tx][ty+1][IU]*dtdy;

      // actually perform the update on external device memory
      int offset = elemOffset;
      
      Uout[offset] = uOut[ID];  offset += arraySize;
      Uout[offset] = uOut[IP];  offset += arraySize;
      Uout[offset] = uOut[IU];  offset += arraySize;
      Uout[offset] = uOut[IV];
    }
  __syncthreads();

} // kernel_hydro_flux_update_unsplit_2d_v1

/*********************************************************
 *** UPDATE CONSERVATIVE VAR ARRAY 3D KERNEL version 1 ***
 *********************************************************/
#ifdef USE_DOUBLE
#define UPDATE_BLOCK_DIMX_3D_V1	16
#define UPDATE_BLOCK_INNER_DIMX_3D_V1	(UPDATE_BLOCK_DIMX_3D_V1-1)
#define UPDATE_BLOCK_DIMY_3D_V1	16
#define UPDATE_BLOCK_INNER_DIMY_3D_V1	(UPDATE_BLOCK_DIMY_3D_V1-1)
#else // simple precision
#define UPDATE_BLOCK_DIMX_3D_V1	48
#define UPDATE_BLOCK_INNER_DIMX_3D_V1	(UPDATE_BLOCK_DIMX_3D_V1-1)
#define UPDATE_BLOCK_DIMY_3D_V1	10
#define UPDATE_BLOCK_INNER_DIMY_3D_V1	(UPDATE_BLOCK_DIMY_3D_V1-1)
#endif // USE_DOUBLE

/**
 * Update hydro conservative variables 3D (implementation version 1).
 * 
 * This is the final kernel, that given the qm, qp states compute
 * fluxes and perform update.
 *
 * \see kernel_hydro_compute_trace_unsplit_3d_v1 (computation of qm, qp buffer)
 *
 * \param[in]  Uin  input conservative variable array
 * \param[out] Uout ouput conservative variable array
 * \param[in] d_qm_x qm state along x
 * \param[in] d_qm_y qm state along y
 * \param[in] d_qm_z qm state along z
 * \param[in] d_qp_x qp state along x
 * \param[in] d_qp_y qp state along y
 * \param[in] d_qp_z qp state along z
 *
 */
__global__ void kernel_hydro_flux_update_unsplit_3d_v1(const real_t * __restrict__ Uin, 
						       real_t       *Uout,
						       const real_t * __restrict__ d_qm_x,
						       const real_t * __restrict__ d_qm_y,
						       const real_t * __restrict__ d_qm_z,
						       const real_t * __restrict__ d_qp_x,
						       const real_t * __restrict__ d_qp_y,
						       const real_t * __restrict__ d_qp_z,
						       int pitch, 
						       int imax, 
						       int jmax,
						       int kmax,
						       real_t dtdx, 
						       real_t dtdy,
						       real_t dtdz,
						       real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UPDATE_BLOCK_INNER_DIMX_3D_V1) + tx;
  const int j = __mul24(by, UPDATE_BLOCK_INNER_DIMY_3D_V1) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // flux computation
  __shared__ real_t flux[UPDATE_BLOCK_DIMX_3D_V1][UPDATE_BLOCK_DIMY_3D_V1][NVAR_3D];

  // qm and qp's are output of the trace step
  //real_t qm [THREE_D][NVAR_3D];
  //real_t qp [THREE_D][NVAR_3D];
 
  // conservative variables
  real_t uOut[NVAR_3D];
  real_t qgdnv[NVAR_3D];
  //real_t c;

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use flux as flux_x
    real_t (&flux_x)[UPDATE_BLOCK_DIMX_3D_V1][UPDATE_BLOCK_DIMY_3D_V1][NVAR_3D] = flux;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    __syncthreads();
    
    if(i >= 2 and i < imax-1 and
       j >= 2 and j < jmax-1 and
       k >= 2 and k < kmax-1)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t   qleft_x [NVAR_3D];
	real_t   qright_x[NVAR_3D];
	
	// set qleft_x by re-reading qm_x from external memory at location x-1
	int offset = elemOffset-1;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qleft_x[iVar] = d_qm_x[offset];
	  offset += arraySize;
	}

	// set qright_x by re-reading qp_x from external memory at location x
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qright_x[iVar] = d_qp_x[offset];
	  offset += arraySize;
	}

	riemann<NVAR_3D>(qleft_x, qright_x, qgdnv, flux_x[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i >= 2 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V1-1 and
       j >= 2 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V1-1 and
       k >= 2 and k < kmax-2)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offset = elemOffset;
	uOut[ID] = Uin[offset];  offset += arraySize;
	uOut[IP] = Uin[offset];  offset += arraySize;
	uOut[IU] = Uin[offset];  offset += arraySize;
	uOut[IV] = Uin[offset];  offset += arraySize;
	uOut[IW] = Uin[offset];

	uOut[ID] += flux_x[tx  ][ty][ID]*dtdx;
	uOut[ID] -= flux_x[tx+1][ty][ID]*dtdx;

	uOut[IP] += flux_x[tx  ][ty][IP]*dtdx;
	uOut[IP] -= flux_x[tx+1][ty][IP]*dtdx;

	uOut[IU] += flux_x[tx  ][ty][IU]*dtdx;
	uOut[IU] -= flux_x[tx+1][ty][IU]*dtdx;

	uOut[IV] += flux_x[tx  ][ty][IV]*dtdx;
	uOut[IV] -= flux_x[tx+1][ty][IV]*dtdx;

	uOut[IW] += flux_x[tx  ][ty][IW]*dtdx;
	uOut[IW] -= flux_x[tx+1][ty][IW]*dtdx;
      }
    __syncthreads();

    /*
     * Compute fluxes at Y-interfaces.
     */
    // re-use flux as flux_y
    real_t (&flux_y)[UPDATE_BLOCK_DIMX_3D_V1][UPDATE_BLOCK_DIMY_3D_V1][NVAR_3D] = flux;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    __syncthreads();
    
    if(i >= 2 and i < imax-1 and 
       j >= 2 and j < jmax-1 and 
       k >= 2 and k < kmax-1)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t  qleft_y[NVAR_3D];
	real_t qright_y[NVAR_3D];
	
	// set qleft_y by reading qm_y from external memory at location y-1
	int offset = elemOffset-pitch;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qleft_y[iVar] = d_qm_y[offset];
	  offset += arraySize;
	}
	
	// set qright_y by reading qp_y from external memory at location y
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qright_y[iVar] = d_qp_y[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IV
	swap_value(qleft_y[IU],qleft_y[IV]);
	swap_value(qright_y[IU],qright_y[IV]);

	riemann<NVAR_3D>(qleft_y, qright_y, qgdnv, flux_y[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_y
    if(i >= 2 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V1-1 and
       j >= 2 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V1-1 and
       k >= 2 and k < kmax-2)
      {
	// watchout IU and IV are swapped !
	uOut[ID] += flux_y[tx][ty  ][ID]*dtdy;
	uOut[ID] -= flux_y[tx][ty+1][ID]*dtdy;

	uOut[IP] += flux_y[tx][ty  ][IP]*dtdy;
	uOut[IP] -= flux_y[tx][ty+1][IP]*dtdy;

	uOut[IU] += flux_y[tx][ty  ][IV]*dtdy;
	uOut[IU] -= flux_y[tx][ty+1][IV]*dtdy;

	uOut[IV] += flux_y[tx][ty  ][IU]*dtdy;
	uOut[IV] -= flux_y[tx][ty+1][IU]*dtdy;

	uOut[IW] += flux_y[tx][ty  ][IW]*dtdy;
	uOut[IW] -= flux_y[tx][ty+1][IW]*dtdy;
      }
    __syncthreads();
    
    /*
     * Compute fluxes at Z-interfaces.
     */
    real_t flux_z[NVAR_3D];
    flux_z[ID] = ZERO_F;
    flux_z[IP] = ZERO_F;
    flux_z[IU] = ZERO_F;
    flux_z[IV] = ZERO_F;
    flux_z[IW] = ZERO_F;
    __syncthreads();
    
    if(i >= 2 and i < imax-1 and tx < UPDATE_BLOCK_DIMX_3D_V1-1 and
       j >= 2 and j < jmax-1 and ty < UPDATE_BLOCK_DIMY_3D_V1-1 and
       k >= 2 and k < kmax-1)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t qleft_z [NVAR_3D];
	real_t qright_z[NVAR_3D];
	
	// set qleft_z by reading qm_z from external memory at location z-1
	int offset = elemOffset - pitch*jmax;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qleft_z[iVar] = d_qm_z[offset];
	  offset += arraySize;
	}
	
        // set qright_z by reading qp_z from external memory at location z
        offset = elemOffset;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qright_z[iVar] = d_qp_z[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IW
	swap_value(qleft_z[IU] ,qleft_z[IW]);
	swap_value(qright_z[IU],qright_z[IW]);
	
	riemann<NVAR_3D>(qleft_z, qright_z, qgdnv, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i >= 2 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V1-1 and
       j >= 2 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V1-1 and
       k >= 2 and k < kmax-1)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	int offset = elemOffset;

	if (k < kmax-2) {
	  // watchout IU and IW are swapped !
	  uOut[ID] += flux_z[ID]*dtdz;
	  uOut[IP] += flux_z[IP]*dtdz;
	  uOut[IU] += flux_z[IW]*dtdz;
	  uOut[IV] += flux_z[IV]*dtdz;
	  uOut[IW] += flux_z[IU]*dtdz;
	  
	  // actually perform the update on external device memory
	  Uout[offset] = uOut[ID];  offset += arraySize;
	  Uout[offset] = uOut[IP];  offset += arraySize;
	  Uout[offset] = uOut[IU];  offset += arraySize;
	  Uout[offset] = uOut[IV];  offset += arraySize;
	  Uout[offset] = uOut[IW];
	}

	if (k>2) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offset = elemOffset - pitch*jmax;
	  Uout[offset] -= flux_z[ID]*dtdz; offset += arraySize;
	  Uout[offset] -= flux_z[IP]*dtdz; offset += arraySize;
	  Uout[offset] -= flux_z[IW]*dtdz; offset += arraySize;
	  Uout[offset] -= flux_z[IV]*dtdz; offset += arraySize;
	  Uout[offset] -= flux_z[IU]*dtdz;
	}

      } // end update along Z
    __syncthreads();

  } // end for k

} // kernel_hydro_flux_update_unsplit_3d_v1

#endif /* GODUNOV_UNSPLIT_CUH_ */
