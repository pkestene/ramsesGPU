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


// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define UNSPLIT_BLOCK_DIMX_2D_V0	16
#define UNSPLIT_BLOCK_INNER_DIMX_2D_V0	(UNSPLIT_BLOCK_DIMX_2D_V0-4)
#define UNSPLIT_BLOCK_DIMY_2D_V0	12
#define UNSPLIT_BLOCK_INNER_DIMY_2D_V0	(UNSPLIT_BLOCK_DIMY_2D_V0-4)
#else // simple precision
#define UNSPLIT_BLOCK_DIMX_2D_V0	16
#define UNSPLIT_BLOCK_INNER_DIMX_2D_V0  (UNSPLIT_BLOCK_DIMX_2D_V0-4)
#define UNSPLIT_BLOCK_DIMY_2D_V0	24
#define UNSPLIT_BLOCK_INNER_DIMY_2D_V0  (UNSPLIT_BLOCK_DIMY_2D_V0-4)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 2D data (version 0).
 * 
 * Primitive variables are assumed to have been already computed. 
 *
 * This kernel doesn't eat so much shared memory, but to the price of
 * more computations...
 * We recompute what is needed by each thread.
 *
 */
__global__ void kernel_godunov_unsplit_2d_v0(real_t *Uout,
					     real_t *qData,
					     int pitch, 
					     int imax, 
					     int jmax,
					     real_t dtdx, 
					     real_t dtdy, 
					     real_t dt,
					     bool gravityEnabled)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_2D_V0) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_2D_V0) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  __shared__ real_t      q[UNSPLIT_BLOCK_DIMX_2D_V0][UNSPLIT_BLOCK_DIMY_2D_V0][NVAR_2D];
  __shared__ real_t deltaU[UNSPLIT_BLOCK_DIMX_2D_V0][UNSPLIT_BLOCK_DIMY_2D_V0][NVAR_2D];

  // primitive variables (local array)
  real_t qLoc[NVAR_2D];
	
  // slopes
  real_t dq[TWO_D][NVAR_2D];
  real_t qNeighbors[2*TWO_D][NVAR_2D];

  // reconstructed state on cell faces
  // aka riemann solver input
  real_t qleft[NVAR_2D];
  real_t qright[NVAR_2D];
  
  // riemann solver output
  real_t qgdnv[NVAR_2D];
  real_t flux_x[NVAR_2D];
  real_t flux_y[NVAR_2D];
	
  real_t *gravin = gParams.arrayList[A_GRAV];

  // load primitive variables
  if(i >= 0 and i < imax and 
     j >= 0 and j < jmax)
    {
      
      // read conservative variables
      int offset = elemOffset;
      q[tx][ty][ID] = qData[offset];  offset += arraySize;
      q[tx][ty][IP] = qData[offset];  offset += arraySize;
      q[tx][ty][IU] = qData[offset];  offset += arraySize;
      q[tx][ty][IV] = qData[offset];
      
      deltaU[tx][ty][ID] = 0;
      deltaU[tx][ty][IP] = 0;
      deltaU[tx][ty][IU] = 0;
      deltaU[tx][ty][IV] = 0;

    }
  __syncthreads();

  // along X
  if(i > 1 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V0-1 and
     j > 1 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V0-1)
    {
 
      // load neighborhood
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qLoc[iVar]          = q[tx  ][ty  ][iVar];
	qNeighbors[0][iVar] = q[tx+1][ty  ][iVar];
	qNeighbors[1][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[2][iVar] = q[tx  ][ty+1][iVar];
	qNeighbors[3][iVar] = q[tx  ][ty-1][iVar];
      }

      // compute slopes in current cell
      slope_unsplit_hydro_2d(qLoc, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     dq);

      // left interface : right state
      trace_unsplit_hydro_2d_by_direction(qLoc, 
					  dq, 
					  dtdx, dtdy, 
					  FACE_XMIN, 
					  qright);

      // get primitive variables state vector in left neighbor along X
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qLoc[iVar]          = q[tx-1][ty  ][iVar];
	qNeighbors[0][iVar] = q[tx  ][ty  ][iVar];
	qNeighbors[1][iVar] = q[tx-2][ty  ][iVar];
	qNeighbors[2][iVar] = q[tx-1][ty+1][iVar];
	qNeighbors[3][iVar] = q[tx-1][ty-1][iVar];
      }

      // compute slopes in left neighbor along X
      slope_unsplit_hydro_2d(qLoc, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     dq);

      // left interface : left state
      trace_unsplit_hydro_2d_by_direction(qLoc,
					  dq,
					  dtdx, dtdy,
					  FACE_XMAX, 
					  qleft);

      if (gravityEnabled) { 
	
	// we need to modify input to flux computation with
	// gravity predictor (half time step)
	qleft[IU]  += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qleft[IV]  += HALF_F * dt * gravin[elemOffset+IY*arraySize];

	qright[IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qright[IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];

      } // end gravityEnabled
      
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      riemann<NVAR_2D>(qleft,qright,qgdnv,flux_x);

      deltaU[tx  ][ty  ][ID] += flux_x[ID]*dtdx;
      deltaU[tx  ][ty  ][IP] += flux_x[IP]*dtdx;
      deltaU[tx  ][ty  ][IU] += flux_x[IU]*dtdx;
      deltaU[tx  ][ty  ][IV] += flux_x[IV]*dtdx;

    }
  __syncthreads();

  if(i > 1 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V0-1 and
     j > 1 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V0-1)
    {

      deltaU[tx-1][ty  ][ID] -= flux_x[ID]*dtdx;
      deltaU[tx-1][ty  ][IP] -= flux_x[IP]*dtdx;
      deltaU[tx-1][ty  ][IU] -= flux_x[IU]*dtdx;
      deltaU[tx-1][ty  ][IV] -= flux_x[IV]*dtdx;

    }
  __syncthreads();


  // along Y
  if(i > 1 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V0-1 and
     j > 1 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V0-1)
    {
 
      // load neighborhood
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qLoc[iVar]          = q[tx  ][ty  ][iVar];
	qNeighbors[0][iVar] = q[tx+1][ty  ][iVar];
	qNeighbors[1][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[2][iVar] = q[tx  ][ty+1][iVar];
	qNeighbors[3][iVar] = q[tx  ][ty-1][iVar];
      }

      // compute slopes in current cell
      slope_unsplit_hydro_2d(qLoc, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     dq);

      // left interface : right state
      trace_unsplit_hydro_2d_by_direction(qLoc, 
					  dq, 
					  dtdx, dtdy, 
					  FACE_YMIN, 
					  qright);

      // get primitive variables state vector in left neighbor along Y
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qLoc[iVar]          = q[tx  ][ty-1][iVar];
	qNeighbors[0][iVar] = q[tx+1][ty-1][iVar];
	qNeighbors[1][iVar] = q[tx-1][ty-1][iVar];
	qNeighbors[2][iVar] = q[tx  ][ty  ][iVar];
	qNeighbors[3][iVar] = q[tx  ][ty-2][iVar];
      }

      // compute slopes in left neighbor along X
      slope_unsplit_hydro_2d(qLoc, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     dq);

      // left interface : left state
      trace_unsplit_hydro_2d_by_direction(qLoc,
					  dq,
					  dtdx, dtdy,
					  FACE_YMAX, 
					  qleft);

      if (gravityEnabled) { 
	
	// we need to modify input to flux computation with
	// gravity predictor (half time step)
	qleft[IU]  += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qleft[IV]  += HALF_F * dt * gravin[elemOffset+IY*arraySize];

	qright[IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qright[IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];

      } // end gravityEnabled
      
      // Solve Riemann problem at X-interfaces and compute Y-fluxes
      swap_value(qleft[IU] ,qleft[IV]);
      swap_value(qright[IU],qright[IV]);
      riemann<NVAR_2D>(qleft,qright,qgdnv,flux_y);

      // watchout IU / IV are swapped
      deltaU[tx  ][ty  ][ID] += flux_y[ID]*dtdy;
      deltaU[tx  ][ty  ][IP] += flux_y[IP]*dtdy;
      deltaU[tx  ][ty  ][IU] += flux_y[IV]*dtdy;
      deltaU[tx  ][ty  ][IV] += flux_y[IU]*dtdy;
    }
  __syncthreads();

  if(i > 1 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V0-1 and
     j > 1 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V0-1)
    {

      // watchout IU / IV are swapped
      deltaU[tx  ][ty-1][ID] -= flux_y[ID]*dtdy;
      deltaU[tx  ][ty-1][IP] -= flux_y[IP]*dtdy;
      deltaU[tx  ][ty-1][IU] -= flux_y[IV]*dtdy;
      deltaU[tx  ][ty-1][IV] -= flux_y[IU]*dtdy;

    }
  __syncthreads();

  //
  // Update on external memory
  //
  if(i > 1 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V0-2 and
     j > 1 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V0-2)

    {

     // actually perform update on external device memory
      int offset = elemOffset;
      Uout[offset] += deltaU[tx][ty][ID];  offset += arraySize;
      Uout[offset] += deltaU[tx][ty][IP];  offset += arraySize;
      Uout[offset] += deltaU[tx][ty][IU];  offset += arraySize;
      Uout[offset] += deltaU[tx][ty][IV];

    }
      
} // kernel_godunov_unsplit_2d_v0

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

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
// newer tested
#define UNSPLIT_BLOCK_DIMX_3D_V0	16
#define UNSPLIT_BLOCK_INNER_DIMX_3D_V0	(UNSPLIT_BLOCK_DIMX_3D_V0-4)
#define UNSPLIT_BLOCK_DIMY_3D_V0	12
#define UNSPLIT_BLOCK_INNER_DIMY_3D_V0	(UNSPLIT_BLOCK_DIMY_3D_V0-4)
#else // simple precision
#define UNSPLIT_BLOCK_DIMX_3D_V0	16
#define UNSPLIT_BLOCK_INNER_DIMX_3D_V0	(UNSPLIT_BLOCK_DIMX_3D_V0-4)
#define UNSPLIT_BLOCK_DIMY_3D_V0	24
#define UNSPLIT_BLOCK_INNER_DIMY_3D_V0	(UNSPLIT_BLOCK_DIMY_3D_V0-4)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 3D data version 0.
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
__global__ void kernel_godunov_unsplit_3d_v0(const real_t * __restrict__ qData, 
					     real_t       *Uout,
					     int pitch, 
					     int imax, 
					     int jmax,
					     int kmax, 
					     real_t dtdx, 
					     real_t dtdy, 
					     real_t dtdz, 
					     real_t dt,
					     bool gravityEnabled)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_3D_V0) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_3D_V0) + ty;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  // we always store in shared 4 consecutive XY-plans
  __shared__ real_t  q[4][UNSPLIT_BLOCK_DIMX_3D_V0][UNSPLIT_BLOCK_DIMY_3D_V0][NVAR_3D];

  real_t *gravin = gParams.arrayList[A_GRAV];

  // index to address the 4 plans of data
  int low, mid, current, top, tmp;
  low=0;
  mid=1;
  current=2;
  top=3;

  // primitive variables
  //real_t qLoc[NVAR_3D];

  // slopes
  real_t dq[THREE_D][NVAR_3D];
  real_t dqN[THREE_D][NVAR_3D];

  // qNeighbors is used for trace computations
  //real_t qNeighbors[2*THREE_D][NVAR_3D];
  
  // reconstructed state on cell faces
  // aka riemann solver input
  real_t qleft[NVAR_3D];
  real_t qright[NVAR_3D];

  real_t  qgdnv[NVAR_3D];
  real_t flux_x[NVAR_3D];
  real_t flux_y[NVAR_3D];
  real_t flux_z[NVAR_3D];


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
	q[k][tx][ty][ID] = qData[offset];  offset += arraySize;
	q[k][tx][ty][IP] = qData[offset];  offset += arraySize;
	q[k][tx][ty][IU] = qData[offset];  offset += arraySize;
	q[k][tx][ty][IV] = qData[offset];  offset += arraySize;
	q[k][tx][ty][IW] = qData[offset];
	
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
    
    if(i > 1 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0-1 and
       j > 1 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0-1)
      {

	//
	// along X
	//

	// compute slopes in current cell
	slope_unsplit_3d(q[current][tx  ][ty  ],
			 q[current][tx+1][ty  ],
			 q[current][tx-1][ty  ],
			 q[current][tx  ][ty+1],
			 q[current][tx  ][ty-1],
			 q[top    ][tx  ][ty  ],
			 q[mid    ][tx  ][ty  ],
			 dq);

	// left interface : right state
	trace_unsplit_hydro_3d_by_direction(q[current][tx][ty],  dq, 
					    dtdx, dtdy, dtdz,
					    FACE_XMIN, qright);

	// compute slopes in neighbor x - 1
	slope_unsplit_3d(q[current][tx-1][ty  ], 
			 q[current][tx  ][ty  ],
			 q[current][tx-2][ty  ],
			 q[current][tx-1][ty+1],
			 q[current][tx-1][ty-1],
			 q[top    ][tx-1][ty  ],
			 q[mid    ][tx-1][ty  ],
			 dqN);

	// left interface : left state
	trace_unsplit_hydro_3d_by_direction(q[current][tx-1][ty], dqN,
					    dtdx, dtdy, dtdz,
					    FACE_XMAX, qleft);

	// gravity predictor on velocity components of qp0's
	if (gravityEnabled) {
	  qleft[IU]  += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qleft[IV]  += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qleft[IW]  += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	  
	  qright[IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qright[IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qright[IW] += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	  
	}

	// Solve Riemann problem at X-interfaces and compute X-fluxes
	riemann<NVAR_3D>(qleft,qright,qgdnv,flux_x);

	//
	// along Y
	//

	// left interface : right state
	trace_unsplit_hydro_3d_by_direction(q[current][tx][ty],  dq, 
					    dtdx, dtdy, dtdz,
					    FACE_YMIN, qright);

	// compute slopes in neighbor y - 1
	slope_unsplit_3d(q[current][tx  ][ty-1], 
			 q[current][tx+1][ty-1],
			 q[current][tx-1][ty-1],
			 q[current][tx  ][ty  ],
			 q[current][tx  ][ty-2],
			 q[top    ][tx  ][ty-1],
			 q[mid    ][tx  ][ty-1],
			 dqN);

	// left interface : left state
	trace_unsplit_hydro_3d_by_direction(q[current][tx][ty-1], dqN,
					    dtdx, dtdy, dtdz,
					    FACE_YMAX, qleft);

	// gravity predictor on velocity components
	if (gravityEnabled) {
	  qleft[IU]  += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qleft[IV]  += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qleft[IW]  += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	  
	  qright[IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qright[IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qright[IW] += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	}
	
	// Solve Riemann problem at Y-interfaces and compute Y-fluxes
	swap_value(qleft[IU] ,qleft[IV]);
	swap_value(qright[IU],qright[IV]);
	riemann<NVAR_3D>(qleft,qright,qgdnv,flux_y);

	//
	// along Z
	//

	// left interface : right state
	trace_unsplit_hydro_3d_by_direction(q[current][tx][ty],  dq, 
					    dtdx, dtdy, dtdz,
					    FACE_ZMIN, qright);

	// compute slopes in neighbor z - 1
	slope_unsplit_3d(q[mid    ][tx  ][ty  ],
			 q[mid    ][tx+1][ty  ],
			 q[mid    ][tx-1][ty  ],
			 q[mid    ][tx  ][ty+1],
			 q[mid    ][tx  ][ty-1],
			 q[current][tx  ][ty  ],
			 q[low    ][tx  ][ty  ],
			 dqN);

	// left interface : left state
	trace_unsplit_hydro_3d_by_direction(q[mid][tx][ty], dqN,
					    dtdx, dtdy, dtdz,
					    FACE_ZMAX, qleft);

	// gravity predictor on velocity components
	if (gravityEnabled) {
	  qleft[IU]  += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qleft[IV]  += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qleft[IW]  += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	  
	  qright[IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qright[IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qright[IW] += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	}
	
	// Solve Riemann problem at Y-interfaces and compute Y-fluxes
	swap_value(qleft[IU] ,qleft[IW]);
	swap_value(qright[IU],qright[IW]);
	riemann<NVAR_3D>(qleft,qright,qgdnv,flux_z);

      } 
    // end trace/slope computations, we have all we need to
    // compute qleft/qright now
    __syncthreads();
    
    /*
     * Now, perform update
     */
    // a nice trick: reuse q[low] to store deltaU, as we don't need then anymore
    // also remember that q[low] will become q[top] after the flux
    // computations and hydro update
    real_t (&deltaU)[UNSPLIT_BLOCK_DIMX_3D_V0][UNSPLIT_BLOCK_DIMY_3D_V0][NVAR_3D] = q[low];
    
    // 
    deltaU[tx][ty][ID] = ZERO_F;
    deltaU[tx][ty][IP] = ZERO_F;
    deltaU[tx][ty][IU] = ZERO_F;
    deltaU[tx][ty][IV] = ZERO_F;
    deltaU[tx][ty][IW] = ZERO_F;
    //__syncthreads();

    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0-1 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0-1)
      {
	// take for flux_y : IU and IV are swapped
	// take for flux_z : IU and IW are swapped
	deltaU[tx  ][ty  ][ID] += (flux_x[ID]*dtdx+flux_y[ID]*dtdy+flux_z[ID]*dtdz);
	deltaU[tx  ][ty  ][IP] += (flux_x[IP]*dtdx+flux_y[IP]*dtdy+flux_z[IP]*dtdz);
	deltaU[tx  ][ty  ][IU] += (flux_x[IU]*dtdx+flux_y[IV]*dtdy+flux_z[IW]*dtdz);
	deltaU[tx  ][ty  ][IV] += (flux_x[IV]*dtdx+flux_y[IU]*dtdy+flux_z[IV]*dtdz);
	deltaU[tx  ][ty  ][IW] += (flux_x[IW]*dtdx+flux_y[IW]*dtdy+flux_z[IU]*dtdz);
      }
    __syncthreads();

    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0-1 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0-1)
      {

	deltaU[tx-1][ty  ][ID] -= flux_x[ID]*dtdx;
	deltaU[tx-1][ty  ][IP] -= flux_x[IP]*dtdx;
	deltaU[tx-1][ty  ][IU] -= flux_x[IU]*dtdx;
	deltaU[tx-1][ty  ][IV] -= flux_x[IV]*dtdx;
	deltaU[tx-1][ty  ][IW] -= flux_x[IW]*dtdx;

      } 
    __syncthreads();

    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0-1 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0-1)
      {

	// take for flux_y : IU and IV are swapped
	deltaU[tx  ][ty-1][ID] -= flux_y[ID]*dtdy;
	deltaU[tx  ][ty-1][IP] -= flux_y[IP]*dtdy;
	deltaU[tx  ][ty-1][IU] -= flux_y[IV]*dtdy;
	deltaU[tx  ][ty-1][IV] -= flux_y[IU]*dtdy;
	deltaU[tx  ][ty-1][IW] -= flux_y[IW]*dtdy;

      } 
    __syncthreads();
    
 
    // update at z and z-1
    if(i >= 2 and i < imax-1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0-2 and
       j >= 2 and j < jmax-1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0-2)
      {

    	/*
    	 * update at position z-1.
	 * Note that position z-1 has already been partialy updated in
	 * the previous interation (for loop over k).
    	 */
    	// update U with flux : watchout! IU and IW are swapped !
    	int offset = elemOffset - pitch*jmax;
    	Uout[offset] -= (flux_z[ID])*dtdz; offset += arraySize;
    	Uout[offset] -= (flux_z[IP])*dtdz; offset += arraySize;
    	Uout[offset] -= (flux_z[IW])*dtdz; offset += arraySize;
    	Uout[offset] -= (flux_z[IV])*dtdz; offset += arraySize;
    	Uout[offset] -= (flux_z[IU])*dtdz;
	
	// actually perform the update on external device memory
	offset = elemOffset;
	Uout[offset] += deltaU[tx][ty][ID];  offset += arraySize;
	Uout[offset] += deltaU[tx][ty][IP];  offset += arraySize;
	Uout[offset] += deltaU[tx][ty][IU];  offset += arraySize;
	Uout[offset] += deltaU[tx][ty][IV];  offset += arraySize;
	Uout[offset] += deltaU[tx][ty][IW];
	
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
	  
	  // Gather primitive variables
	  int offset = i + pitch * (j + jmax * (k+2));
	  q[top][tx][ty][ID] = qData[offset];  offset += arraySize;
	  q[top][tx][ty][IP] = qData[offset];  offset += arraySize;
	  q[top][tx][ty][IU] = qData[offset];  offset += arraySize;
	  q[top][tx][ty][IV] = qData[offset];  offset += arraySize;
	  q[top][tx][ty][IW] = qData[offset];
	}
    }
    __syncthreads();
    
  } // end for k
  
} //  kernel_godunov_unsplit_3d_v0

/*
 *
 * Here are CUDA kernel implementing hydro unsplit scheme version 1
 * 
 *
 */

/*********************************************
 *** COMPUTE PRIMITIVE VARIABLES 2D KERNEL ***
 *********************************************/

#ifdef USE_DOUBLE
#define PRIM_VAR_BLOCK_DIMX_2D	16
#define PRIM_VAR_BLOCK_DIMY_2D	16
#else // simple precision
#define PRIM_VAR_BLOCK_DIMX_2D	16
#define PRIM_VAR_BLOCK_DIMY_2D	16
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
  
  const int i = __mul24(bx, PRIM_VAR_BLOCK_DIMX_2D) + tx;
  const int j = __mul24(by, PRIM_VAR_BLOCK_DIMY_2D) + ty;
  
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

/*********************************************
 *** COMPUTE PRIMITIVE VARIABLES 3D KERNEL ***
 *********************************************/

#ifdef USE_DOUBLE
#define PRIM_VAR_BLOCK_DIMX_3D	16
#define PRIM_VAR_BLOCK_DIMY_3D	16
#else // simple precision
#define PRIM_VAR_BLOCK_DIMX_3D	16
#define PRIM_VAR_BLOCK_DIMY_3D	16
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
  
  const int i = __mul24(bx, PRIM_VAR_BLOCK_DIMX_3D) + tx;
  const int j = __mul24(by, PRIM_VAR_BLOCK_DIMY_3D) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // conservative variables
  real_t uIn [NVAR_3D];
  real_t c;

  /*
   * loop over k (i.e. z) to compute primitive variables, and store results
   * in external memory buffer Q.
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < kmax; 
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
	slope_unsplit_3d_v1(q[tx][ty], qPlusX, qMinusX, dq[IX]);
	slope_unsplit_3d_v1(q[tx][ty], qPlusY, qMinusY, dq[IY]);
	slope_unsplit_3d_v1(q[tx][ty], qPlusZ, qMinusZ, dq[IZ]);
	
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

/******************************************
 *** COMPUTE SLOPES 2D KERNEL version 2 ***
 ******************************************/

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define SLOPES_BLOCK_DIMX_2D_V2	16
#define SLOPES_BLOCK_INNER_DIMX_2D_V2	(SLOPES_BLOCK_DIMX_2D_V2-2)
#define SLOPES_BLOCK_DIMY_2D_V2	16
#define SLOPES_BLOCK_INNER_DIMY_2D_V2	(SLOPES_BLOCK_DIMY_2D_V2-2)
#else // simple precision
#define SLOPES_BLOCK_DIMX_2D_V2	16
#define SLOPES_BLOCK_INNER_DIMX_2D_V2  (SLOPES_BLOCK_DIMX_2D_V2-2)
#define SLOPES_BLOCK_DIMY_2D_V2	24
#define SLOPES_BLOCK_INNER_DIMY_2D_V2  (SLOPES_BLOCK_DIMY_2D_V2-2)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel : computes slopes for 2D data (version 2).
 * 
 * Primitive variables are assumed to have been already computed. 
 *
 */
__global__ void kernel_godunov_slopes_2d_v2(const real_t * __restrict__ qData,
					    real_t * d_slope_x,
					    real_t * d_slope_y,
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
  
  const int i = __mul24(bx, SLOPES_BLOCK_INNER_DIMX_2D_V2) + tx;
  const int j = __mul24(by, SLOPES_BLOCK_INNER_DIMY_2D_V2) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  __shared__ real_t      q[SLOPES_BLOCK_DIMX_2D_V2][SLOPES_BLOCK_DIMY_2D_V2][NVAR_2D];

  // primitive variables (local array)
  real_t qLoc[NVAR_2D];
	
  // slopes
  real_t dq[TWO_D][NVAR_2D];
  real_t qNeighbors[2*TWO_D][NVAR_2D];

  // load primitive variables
  if(i >= 0 and i < imax and 
     j >= 0 and j < jmax)
    {
      
      // read conservative variables
      int offset = elemOffset;
      q[tx][ty][ID] = qData[offset];  offset += arraySize;
      q[tx][ty][IP] = qData[offset];  offset += arraySize;
      q[tx][ty][IU] = qData[offset];  offset += arraySize;
      q[tx][ty][IV] = qData[offset];
      
    }
  __syncthreads();

  // slopes along X
  if(i > 0 and i < imax-1 and tx > 0 and tx < SLOPES_BLOCK_DIMX_2D_V2-1 and
     j > 0 and j < jmax-1 and ty > 0 and ty < SLOPES_BLOCK_DIMY_2D_V2-1)
    {
 
      // load neighborhood
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qLoc[iVar]          = q[tx  ][ty  ][iVar];
	qNeighbors[0][iVar] = q[tx+1][ty  ][iVar];
	qNeighbors[1][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[2][iVar] = q[tx  ][ty+1][iVar];
	qNeighbors[3][iVar] = q[tx  ][ty-1][iVar];
      }

      // compute slopes in current cell
      slope_unsplit_hydro_2d(qLoc, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     dq);
    }
  __syncthreads();

  if(i > 0 and i < imax-1 and tx > 0 and tx < SLOPES_BLOCK_DIMX_2D_V2-1 and
     j > 0 and j < jmax-1 and ty > 0 and ty < SLOPES_BLOCK_DIMY_2D_V2-1)
    {

      // write slopes on global memory
      int offset = elemOffset;
      d_slope_x[offset] = dq[IX][ID];
      d_slope_y[offset] = dq[IY][ID];  offset += arraySize;
      
      d_slope_x[offset] = dq[IX][IP];
      d_slope_y[offset] = dq[IY][IP];  offset += arraySize;
      
      d_slope_x[offset] = dq[IX][IU];
      d_slope_y[offset] = dq[IY][IU];  offset += arraySize;
      
      d_slope_x[offset] = dq[IX][IV];
      d_slope_y[offset] = dq[IY][IV];  offset += arraySize;     
    }

} // kernel_godunov_slopes_2d_v2

/******************************************
 *** COMPUTE SLOPES 3D KERNEL version 2 ***
 ******************************************/

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
#define SLOPES_BLOCK_DIMX_3D_V2	16
#define SLOPES_BLOCK_INNER_DIMX_3D_V2	(SLOPES_BLOCK_DIMX_3D_V2-2)
#define SLOPES_BLOCK_DIMY_3D_V2	16
#define SLOPES_BLOCK_INNER_DIMY_3D_V2	(SLOPES_BLOCK_DIMY_3D_V2-2)
#else // simple precision
#define SLOPES_BLOCK_DIMX_3D_V2	16
#define SLOPES_BLOCK_INNER_DIMX_3D_V2  (SLOPES_BLOCK_DIMX_3D_V2-2)
#define SLOPES_BLOCK_DIMY_3D_V2	24
#define SLOPES_BLOCK_INNER_DIMY_3D_V2  (SLOPES_BLOCK_DIMY_3D_V2-2)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel : computes slopes for 3D data (version 2).
 * 
 * Primitive variables are assumed to have been already computed. 
 *
 */
__global__ void kernel_godunov_slopes_3d_v2(const real_t * __restrict__ qData,
					    real_t * d_slope_x,
					    real_t * d_slope_y,
					    real_t * d_slope_z,
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
  
  const int i = __mul24(bx, SLOPES_BLOCK_INNER_DIMX_3D_V2) + tx;
  const int j = __mul24(by, SLOPES_BLOCK_INNER_DIMY_3D_V2) + ty;

  // 3D array size
  const int arraySize  = pitch * jmax * kmax;

  __shared__ real_t q[3][SLOPES_BLOCK_DIMX_3D_V2][SLOPES_BLOCK_DIMY_3D_V2][NVAR_3D];

  // slopes
  real_t dq[THREE_D][NVAR_3D];

  // index to address the 3 plans of data
  int low, current, top, tmp;
  low=0;
  current=1;
  top=2;

  /*
   * initialize q with the first 3 plans
   */
  for (int k=0, elemOffset = i + pitch * j; 
       k < 3;
       ++k, elemOffset += (pitch*jmax) ) {

    // load primitive variables
    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax)
      {
      
	// read conservative variables
	int offset = elemOffset;
	q[k][tx][ty][ID] = qData[offset];  offset += arraySize;
	q[k][tx][ty][IP] = qData[offset];  offset += arraySize;
	q[k][tx][ty][IU] = qData[offset];  offset += arraySize;
	q[k][tx][ty][IV] = qData[offset];  offset += arraySize;
	q[k][tx][ty][IW] = qData[offset];
	
      }
  } // end  loading the first 3 plans
  __syncthreads();

  /*
   * loop over all X-Y-planes starting at z=1 as the current plane.
   * Note that elemOffset is used in the update stage
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {

    // compute slopes along X, Y and Z at plane z
    if(i > 0 and i < imax-1 and tx > 0 and tx < SLOPES_BLOCK_DIMX_3D_V2-1 and
       j > 0 and j < jmax-1 and ty > 0 and ty < SLOPES_BLOCK_DIMY_3D_V2-1)
      {
 
	// compute slopes in current cell
	slope_unsplit_3d(q[current][tx  ][ty  ],
			 q[current][tx+1][ty  ],
			 q[current][tx-1][ty  ],
			 q[current][tx  ][ty+1],
			 q[current][tx  ][ty-1],
			 q[top    ][tx  ][ty  ],
			 q[low    ][tx  ][ty  ],
			 dq);
      }
    __syncthreads();

    // store slopes in external memory buffer
    if(i > 0 and i < imax-1 and tx > 0 and tx < SLOPES_BLOCK_DIMX_3D_V2-1 and
       j > 0 and j < jmax-1 and ty > 0 and ty < SLOPES_BLOCK_DIMY_3D_V2-1)
      {
	// write slopes on global memory
	int offset = elemOffset;
	d_slope_x[offset] = dq[IX][ID];
	d_slope_y[offset] = dq[IY][ID];
	d_slope_z[offset] = dq[IZ][ID];  offset += arraySize;
	
	d_slope_x[offset] = dq[IX][IP];
	d_slope_y[offset] = dq[IY][IP];
	d_slope_z[offset] = dq[IZ][IP];  offset += arraySize;
	
	d_slope_x[offset] = dq[IX][IU];
	d_slope_y[offset] = dq[IY][IU];
	d_slope_z[offset] = dq[IZ][IU];  offset += arraySize;
	
	d_slope_x[offset] = dq[IX][IV];
	d_slope_y[offset] = dq[IY][IV];
	d_slope_z[offset] = dq[IZ][IV];  offset += arraySize;     
	
	d_slope_x[offset] = dq[IX][IW];
	d_slope_y[offset] = dq[IY][IW];
	d_slope_z[offset] = dq[IZ][IW];  offset += arraySize;     
      
      } // end store slopes
    __syncthreads();

    // load next z-plane primitive variables
    tmp = low;
    low = current;
    current = top;
    top = tmp;
    //__syncthreads();

    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax)
      {
      
	// read conservative variables at z+2
	int offset = elemOffset + pitch*jmax*2;
	q[top][tx][ty][ID] = qData[offset];  offset += arraySize;
	q[top][tx][ty][IP] = qData[offset];  offset += arraySize;
	q[top][tx][ty][IU] = qData[offset];  offset += arraySize;
	q[top][tx][ty][IV] = qData[offset];  offset += arraySize;
	q[top][tx][ty][IW] = qData[offset];
	
      }
    //__syncthreads();

  } // end for

} // kernel_godunov_slopes_3d_v2

/*****************************************
 *** COMPUTE TRACE 2D KERNEL version 2 ***
 *****************************************/

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define TRACE_BLOCK_DIMX_2D_V2	16
#define TRACE_BLOCK_DIMY_2D_V2	16
#else // simple precision
#define TRACE_BLOCK_DIMX_2D_V2	16
#define TRACE_BLOCK_DIMY_2D_V2	16
#endif // USE_DOUBLE

/**
 * Compute trace for hydro 2D by direction (implementation version 2).
 *
 * Output are all that is needed to compute fluxes.
 * \see kernel_hydro_flux_update_unsplit_3d_v2
 *
 * All we do here is call :
 * - trace_unsplit_hydro_2d_by_direction to get output : qleft, right.
 *
 * \param[in] d_slope_x input slopes array
 * \param[in] d_slope_y input slopes array
 * \param[out] d_qm qm state along dir
 * \param[out] d_qp qp state along dir
 */
__global__ void kernel_godunov_trace_by_dir_2d_v2(const real_t * __restrict__ d_Q,
						  const real_t * __restrict__ d_slope_x,
						  const real_t * __restrict__ d_slope_y,
						  real_t * d_qm,
						  real_t * d_qp,
						  int pitch, 
						  int imax, 
						  int jmax,
						  real_t dt,
						  real_t dtdx,
						  real_t dtdy,
						  bool gravityEnabled,
						  int direction)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, TRACE_BLOCK_DIMX_2D_V2) + tx;
  const int j = __mul24(by, TRACE_BLOCK_DIMY_2D_V2) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  real_t qLoc[NVAR_2D];
  real_t dq[TWO_D][NVAR_2D];

  // reconstructed state on cell faces
  // aka riemann solver input
  real_t qleft[NVAR_2D];
  real_t qright[NVAR_2D];

  real_t *gravin = gParams.arrayList[A_GRAV];

  int dir_min, dir_max;
  
  if (direction == IX) {
    dir_min = FACE_XMIN;
    dir_max = FACE_XMAX;
  } else if (direction == IY) {
    dir_min = FACE_YMIN;
    dir_max = FACE_YMAX;
  }

  // load prim var and slopes
  if(i > 0 and i < imax-1 and 
     j > 0 and j < jmax-1 )
    {
      int offset = elemOffset;

      qLoc [ID] = d_Q[offset];
      dq[IX][ID] = d_slope_x[offset];
      dq[IY][ID] = d_slope_y[offset]; offset += arraySize;

      qLoc [IP] = d_Q[offset];
      dq[IX][IP] = d_slope_x[offset];
      dq[IY][IP] = d_slope_y[offset]; offset += arraySize;

      qLoc [IU] = d_Q[offset];
      dq[IX][IU] = d_slope_x[offset];
      dq[IY][IU] = d_slope_y[offset]; offset += arraySize;

      qLoc [IV] = d_Q[offset];
      dq[IX][IV] = d_slope_x[offset];
      dq[IY][IV] = d_slope_y[offset]; offset += arraySize;

      //
      // Compute reconstructed states 
      //
      
      // TAKE CARE here left and right designate the interface location
      // compare to current cell
      // !!! THIS is FUNDAMENTALLY different from v0 and v1 !!!
      
      // left interface 
      trace_unsplit_hydro_2d_by_direction(qLoc, 
					  dq, 
					  dtdx, dtdy, 
					  dir_min,
					  qleft);
      
      // right interface
      trace_unsplit_hydro_2d_by_direction(qLoc,
					  dq,
					  dtdx, dtdy,
					  dir_max, 
					  qright);
      
      if (gravityEnabled) { 
	
	// we need to modify input to flux computation with
	// gravity predictor (half time step)
	
	qleft[IU]  += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qleft[IV]  += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	
	qright[IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qright[IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	
      } // end gravityEnabled

      // store them
      offset = elemOffset;
	
      d_qm[offset] = qleft [ID];
      d_qp[offset] = qright[ID]; offset += arraySize;
	
      d_qm[offset] = qleft [IP];
      d_qp[offset] = qright[IP]; offset += arraySize;

      d_qm[offset] = qleft [IU];
      d_qp[offset] = qright[IU]; offset += arraySize;

      d_qm[offset] = qleft [IV];
      d_qp[offset] = qright[IV]; offset += arraySize;

    }

} // kernel_godunov_trace_by_dir_2d_v2

/*********************************************************
 *** UPDATE CONSERVATIVE VAR ARRAY 2D KERNEL version 2 ***
 *********************************************************/
#ifdef USE_DOUBLE
#define UPDATE_BLOCK_DIMX_2D_V2	16
#define UPDATE_BLOCK_INNER_DIMX_2D_V2	(UPDATE_BLOCK_DIMX_2D_V2-1)
#define UPDATE_BLOCK_DIMY_2D_V2	16
#define UPDATE_BLOCK_INNER_DIMY_2D_V2	(UPDATE_BLOCK_DIMY_2D_V2-1)
#else // simple precision
#define UPDATE_BLOCK_DIMX_2D_V2	16
#define UPDATE_BLOCK_INNER_DIMX_2D_V2	(UPDATE_BLOCK_DIMX_2D_V2-1)
#define UPDATE_BLOCK_DIMY_2D_V2	16
#define UPDATE_BLOCK_INNER_DIMY_2D_V2	(UPDATE_BLOCK_DIMY_2D_V2-1)
#endif // USE_DOUBLE

/**
 * Update hydro conservative variables 2D (implementation version 2).
 * 
 * This is the final kernel, that given the qm, qp states compute
 * fluxes and perform update.
 *
 * \see kernel_hydro_compute_trace_unsplit_2d_v1 (computation of qm, qp buffer)
 *
 * \param[in,out] Uout ouput conservative variable array
 * \param[in] d_qm qm state along dir
 * \param[in] d_qp qp state along dir
 * \param[in] direction : IX, IY or IZ
 *
 */
__global__ void kernel_hydro_flux_update_unsplit_2d_v2(real_t       * Uout,
						       const real_t * __restrict__ d_qm,
						       const real_t * __restrict__ d_qp,
						       int pitch, 
						       int imax, 
						       int jmax,
						       real_t dtdx, 
						       real_t dtdy,
						       real_t dt,
						       int direction)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UPDATE_BLOCK_INNER_DIMX_2D_V2) + tx;
  const int j = __mul24(by, UPDATE_BLOCK_INNER_DIMY_2D_V2) + ty;
  
  const int arraySize    = pitch * jmax;

  // flux computation
  __shared__ real_t flux[UPDATE_BLOCK_DIMX_2D_V2][UPDATE_BLOCK_DIMY_2D_V2][NVAR_2D];

  // conservative variables
  real_t uOut[NVAR_2D];
  real_t qgdnv[NVAR_2D];

  int elemOffset = i + pitch * j;

  int deltaOffset;
  if (direction == IX)
    deltaOffset = -1;
  else if (direction == IY)
    deltaOffset = -pitch;
  
  /*
   * Compute fluxes 
   */
  flux[tx][ty][ID] = ZERO_F;
  flux[tx][ty][IP] = ZERO_F;
  flux[tx][ty][IU] = ZERO_F;
  flux[tx][ty][IV] = ZERO_F;
  __syncthreads();
    
  if(i >= 2 and i < imax-1 and
     j >= 2 and j < jmax-1)
    {
      // Solve Riemann problem at interfaces and compute fluxes
      real_t   qleft [NVAR_2D];
      real_t   qright[NVAR_2D];
      
      // set qleft by re-reading qp from external memory at location x-1 or y-1
      int offset = elemOffset + deltaOffset;
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qleft[iVar] = d_qp[offset];
	offset += arraySize;
      }
      
      // set qright by re-reading qm from external memory at curent location
      offset = elemOffset;
      for (int iVar=0; iVar<NVAR_2D; iVar++) {
	qright[iVar] = d_qm[offset];
	offset += arraySize;
      }

      if (direction == IY) {
	// watchout swap IU and IV
	swap_value(qleft[IU],qleft[IV]);
	swap_value(qright[IU],qright[IV]);
      }
      
      riemann<NVAR_2D>(qleft, qright, qgdnv, flux[tx][ty]);
    }  
  __syncthreads();
    
  // update uOut with flux
  if(i >= 2 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_2D_V2-1 and
     j >= 2 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_2D_V2-1)
    {
      // re-read input state into uOut which will in turn serve to
      // update Uout !
      int offset = elemOffset;
      uOut[ID] = Uout[offset];  offset += arraySize;
      uOut[IP] = Uout[offset];  offset += arraySize;
      uOut[IU] = Uout[offset];  offset += arraySize;
      uOut[IV] = Uout[offset];

      if (direction == IX) {
	uOut[ID] += flux[tx  ][ty][ID]*dtdx;
	uOut[ID] -= flux[tx+1][ty][ID]*dtdx;
	
	uOut[IP] += flux[tx  ][ty][IP]*dtdx;
	uOut[IP] -= flux[tx+1][ty][IP]*dtdx;
	
	uOut[IU] += flux[tx  ][ty][IU]*dtdx;
	uOut[IU] -= flux[tx+1][ty][IU]*dtdx;
	
	uOut[IV] += flux[tx  ][ty][IV]*dtdx;
	uOut[IV] -= flux[tx+1][ty][IV]*dtdx;
      } else if (direction == IY) {
	// watchout IU and IV are swapped !
	uOut[ID] += flux[tx][ty  ][ID]*dtdy;
	uOut[ID] -= flux[tx][ty+1][ID]*dtdy;
	
	uOut[IP] += flux[tx][ty  ][IP]*dtdy;
	uOut[IP] -= flux[tx][ty+1][IP]*dtdy;
	
	uOut[IU] += flux[tx][ty  ][IV]*dtdy;
	uOut[IU] -= flux[tx][ty+1][IV]*dtdy;
	
	uOut[IV] += flux[tx][ty  ][IU]*dtdy;
	uOut[IV] -= flux[tx][ty+1][IU]*dtdy;
      }

      // actually perform the update on external device memory
      offset = elemOffset;
      
      Uout[offset] = uOut[ID];  offset += arraySize;
      Uout[offset] = uOut[IP];  offset += arraySize;
      Uout[offset] = uOut[IU];  offset += arraySize;
      Uout[offset] = uOut[IV];

    }

} // kernel_hydro_flux_update_unsplit_2d_v2

/*********************************************************
 *** UPDATE CONSERVATIVE VAR ARRAY 3D KERNEL version 2 ***
 *********************************************************/
#ifdef USE_DOUBLE
#define UPDATE_BLOCK_DIMX_3D_V2	16
#define UPDATE_BLOCK_INNER_DIMX_3D_V2	(UPDATE_BLOCK_DIMX_3D_V2-1)
#define UPDATE_BLOCK_DIMY_3D_V2	16
#define UPDATE_BLOCK_INNER_DIMY_3D_V2	(UPDATE_BLOCK_DIMY_3D_V2-1)
#else // simple precision
#define UPDATE_BLOCK_DIMX_3D_V2	16
#define UPDATE_BLOCK_INNER_DIMX_3D_V2	(UPDATE_BLOCK_DIMX_3D_V2-1)
#define UPDATE_BLOCK_DIMY_3D_V2	16
#define UPDATE_BLOCK_INNER_DIMY_3D_V2	(UPDATE_BLOCK_DIMY_3D_V2-1)
#endif // USE_DOUBLE

/**
 * Update hydro conservative variables 3D (implementation version 2).
 * 
 * This is the final kernel, that given the qm, qp states compute
 * fluxes and perform update.
 *
 * \param[in,out] Uout ouput conservative variable array
 * \param[in] d_qm qm state along dir
 * \param[in] d_qp qp state along dir
 * \param[in] direction : IX, IY or IZ
 *
 */
__global__ 
void kernel_hydro_flux_update_unsplit_3d_v2(real_t       * Uout,
					    const real_t * __restrict__ d_qm,
					    const real_t * __restrict__ d_qp,
					    int pitch, 
					    int imax, 
					    int jmax,
					    int kmax,
					    real_t dtdx, 
					    real_t dtdy,
					    real_t dtdz,
					    real_t dt,
					    int direction)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UPDATE_BLOCK_INNER_DIMX_3D_V2) + tx;
  const int j = __mul24(by, UPDATE_BLOCK_INNER_DIMY_3D_V2) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // flux computation
  __shared__ real_t flux[UPDATE_BLOCK_DIMX_3D_V2][UPDATE_BLOCK_DIMY_3D_V2][NVAR_3D];

  // conservative variables
  real_t uOut[NVAR_3D];
  real_t qgdnv[NVAR_3D];
  real_t flux_zm1[NVAR_3D];

  int deltaOffset;
  if (direction == IX)
    deltaOffset = -1;
  else if (direction == IY)
    deltaOffset = -pitch;
  else if (direction == IZ)
    deltaOffset = -pitch*jmax;
  
  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=2, elemOffset = i + pitch * (j + jmax * 2);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    /*
     * Compute fluxes at X-interfaces.
     */
    flux[tx][ty][ID] = ZERO_F;
    flux[tx][ty][IP] = ZERO_F;
    flux[tx][ty][IU] = ZERO_F;
    flux[tx][ty][IV] = ZERO_F;
    flux[tx][ty][IW] = ZERO_F;
    __syncthreads();
    
    // Solve Riemann problem at interfaces and compute fluxes
    real_t   qleft [NVAR_3D];
    real_t   qright[NVAR_3D];
    
    if(i >= 2 and i < imax-1 and
       j >= 2 and j < jmax-1 and
       k >= 2 and k < kmax-1)
      {
	
	// set qleft by re-reading qp from external memory at location
	// x-1 or y-1 or z-1
	int offset = elemOffset + deltaOffset;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qleft[iVar] = d_qp[offset];
	  offset += arraySize;
	}

	// set qright by re-reading qm from external memory at current location
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qright[iVar] = d_qm[offset];
	  offset += arraySize;
	}

	if (direction == IY) {
	  // watchout swap IU and IV
	  swap_value(qleft[IU],qleft[IV]);
	  swap_value(qright[IU],qright[IV]);
	} else if (direction == IZ) {
	  // watchout swap IU and IW
	  swap_value(qleft[IU],qleft[IW]);
	  swap_value(qright[IU],qright[IW]);
	}

	riemann<NVAR_3D>(qleft, qright, qgdnv, flux[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux IX or IY or IZ
    if(i >= 2 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V2-1 and
       j >= 2 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V2-1 and
       k >= 2 and k < kmax-1)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !

	if (direction == IX or direction == IY) {
	  int offset = elemOffset;
	  uOut[ID] = Uout[offset];  offset += arraySize;
	  uOut[IP] = Uout[offset];  offset += arraySize;
	  uOut[IU] = Uout[offset];  offset += arraySize;
	  uOut[IV] = Uout[offset];  offset += arraySize;
	  uOut[IW] = Uout[offset];
	  
	  if (direction == IX) {
	    uOut[ID] += flux[tx  ][ty][ID]*dtdx;
	    uOut[ID] -= flux[tx+1][ty][ID]*dtdx;
	    
	    uOut[IP] += flux[tx  ][ty][IP]*dtdx;
	    uOut[IP] -= flux[tx+1][ty][IP]*dtdx;
	    
	    uOut[IU] += flux[tx  ][ty][IU]*dtdx;
	    uOut[IU] -= flux[tx+1][ty][IU]*dtdx;
	    
	    uOut[IV] += flux[tx  ][ty][IV]*dtdx;
	    uOut[IV] -= flux[tx+1][ty][IV]*dtdx;
	    
	    uOut[IW] += flux[tx  ][ty][IW]*dtdx;
	    uOut[IW] -= flux[tx+1][ty][IW]*dtdx;
	  } else if (direction == IY) {
	    // watchout IU and IV are swapped !
	    uOut[ID] += flux[tx][ty  ][ID]*dtdy;
	    uOut[ID] -= flux[tx][ty+1][ID]*dtdy;
	    
	    uOut[IP] += flux[tx][ty  ][IP]*dtdy;
	    uOut[IP] -= flux[tx][ty+1][IP]*dtdy;
	    
	    uOut[IU] += flux[tx][ty  ][IV]*dtdy;
	    uOut[IU] -= flux[tx][ty+1][IV]*dtdy;
	    
	    uOut[IV] += flux[tx][ty  ][IU]*dtdy;
	    uOut[IV] -= flux[tx][ty+1][IU]*dtdy;
	    
	    uOut[IW] += flux[tx][ty  ][IW]*dtdy;
	    uOut[IW] -= flux[tx][ty+1][IW]*dtdy;
	  }

	  if (k < kmax-2) {
	    offset = elemOffset;
	    
	    Uout[offset] = uOut[ID];  offset += arraySize;
	    Uout[offset] = uOut[IP];  offset += arraySize;
	    Uout[offset] = uOut[IU];  offset += arraySize;
	    Uout[offset] = uOut[IV];  offset += arraySize;
	    Uout[offset] = uOut[IW];

	  }

	} else if (direction == IZ) {
	  
	  // at k=2 we do nothing but store flux_zm1 which will be used at k=3
	  
	  if (k > 2 and k < kmax-1) { 
	    /*
	     * update at position z-1.
	     */
	    // watchout! IU and IW are swapped !
	    int offset = elemOffset - pitch*jmax;
	    Uout[offset] += (flux_zm1[ID]-flux[tx][ty][ID])*dtdz; offset += arraySize;
	    Uout[offset] += (flux_zm1[IP]-flux[tx][ty][IP])*dtdz; offset += arraySize;
	    Uout[offset] += (flux_zm1[IW]-flux[tx][ty][IW])*dtdz; offset += arraySize;
	    Uout[offset] += (flux_zm1[IV]-flux[tx][ty][IV])*dtdz; offset += arraySize;
	    Uout[offset] += (flux_zm1[IU]-flux[tx][ty][IU])*dtdz;
	  }

	  // store flux for next z value
	  flux_zm1[ID] = flux[tx][ty][ID];
	  flux_zm1[IP] = flux[tx][ty][IP];
	  flux_zm1[IU] = flux[tx][ty][IU];
	  flux_zm1[IV] = flux[tx][ty][IV];
	  flux_zm1[IW] = flux[tx][ty][IW];

	} // end direction IZ
      }
    __syncthreads();

  } // end for k
 
} // kernel_hydro_flux_update_unsplit_3d_v2

/*****************************************
 *** COMPUTE TRACE 3D KERNEL version 2 ***
 *****************************************/

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
# define TRACE_BLOCK_DIMX_3D_V2	16
# define TRACE_BLOCK_DIMY_3D_V2	16
#else // simple precision
# define TRACE_BLOCK_DIMX_3D_V2	16
# define TRACE_BLOCK_DIMY_3D_V2	16
#endif // USE_DOUBLE

/**
 * Compute trace for hydro 3D by direction (implementation version 2).
 *
 * Output are all that is needed to compute fluxes.
 *
 * All we do here is call :
 * - trace_unsplit_hydro_3d_by_direction to get output : qleft, right.
 *
 * \param[in] d_Q input primitive    variable array
 * \param[in] d_slope_x input slopes array
 * \param[in] d_slope_y input slopes array
 * \param[in] d_slope_z input slopes array
 * \param[out] d_qm qm state along dir
 * \param[out] d_qp qp state along dir
 *
 */
__global__ 
void kernel_godunov_trace_by_dir_3d_v2(const real_t * __restrict__ d_Q,
				       const real_t * __restrict__ d_slope_x,
				       const real_t * __restrict__ d_slope_y,
				       const real_t * __restrict__ d_slope_z,
				       real_t *d_qm,
				       real_t *d_qp,
				       int pitch, 
				       int imax, 
				       int jmax, 
				       int kmax,
				       real_t dt,
				       real_t dtdx,
				       real_t dtdy,
				       real_t dtdz,
				       bool gravityEnabled,
				       int direction)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, TRACE_BLOCK_DIMX_3D_V2) + tx;
  const int j = __mul24(by, TRACE_BLOCK_DIMY_3D_V2) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  real_t qLoc[NVAR_3D];
  real_t dq[THREE_D][NVAR_3D];

  // reconstructed state on cell faces
  // aka riemann solver input
  real_t qleft[NVAR_3D];
  real_t qright[NVAR_3D];

  real_t *gravin = gParams.arrayList[A_GRAV];

  int dir_min, dir_max;
  
  if (direction == IX) {
    dir_min = FACE_XMIN;
    dir_max = FACE_XMAX;
  } else if (direction == IY) {
    dir_min = FACE_YMIN;
    dir_max = FACE_YMAX;
  } else if (direction == IZ) {
    dir_min = FACE_ZMIN;
    dir_max = FACE_ZMAX;
  }

  /*
   * loop over all X-Y-planes starting at z=1 as the current plane.
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i > 0 and i < imax-1 and 
       j > 0 and j < jmax-1)
      {
	// load primitive variables as well as slopes
	int offset = elemOffset;

	qLoc  [ID] = d_Q[offset];
	dq[IX][ID] = d_slope_x[offset];
	dq[IY][ID] = d_slope_y[offset];
	dq[IZ][ID] = d_slope_z[offset]; offset += arraySize;

	qLoc  [IP] = d_Q[offset];
	dq[IX][IP] = d_slope_x[offset];
	dq[IY][IP] = d_slope_y[offset];
	dq[IZ][IP] = d_slope_z[offset]; offset += arraySize;
	
	qLoc  [IU] = d_Q[offset];
	dq[IX][IU] = d_slope_x[offset];
	dq[IY][IU] = d_slope_y[offset];
	dq[IZ][IU] = d_slope_z[offset]; offset += arraySize;
	
	qLoc  [IV] = d_Q[offset];
	dq[IX][IV] = d_slope_x[offset];
	dq[IY][IV] = d_slope_y[offset];
	dq[IZ][IV] = d_slope_z[offset]; offset += arraySize;
	
	qLoc  [IW] = d_Q[offset];
	dq[IX][IW] = d_slope_x[offset];
	dq[IY][IW] = d_slope_y[offset];
	dq[IZ][IW] = d_slope_z[offset]; offset += arraySize;

	//
	// Compute reconstructed states 
	//
	
	// TAKE CARE here left and right designate the interface location
	// compare to current cell
	// !!! THIS is FUNDAMENTALLY different from v0 and v1 !!!
	
	// left interface 
	trace_unsplit_hydro_3d_by_direction(qLoc, 
					    dq, 
					    dtdx, dtdy, dtdz,
					    dir_min,
					    qleft);
	
	// right interface
	trace_unsplit_hydro_3d_by_direction(qLoc,
					    dq,
					    dtdx, dtdy, dtdz,
					    dir_max, 
					    qright);
	
	if (gravityEnabled) { 
	  
	  // we need to modify input to flux computation with
	  // gravity predictor (half time step)
	  
	  qleft[IU]  += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qleft[IV]  += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qleft[IW]  += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	  
	  qright[IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	  qright[IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
	  qright[IW] += HALF_F * dt * gravin[elemOffset+IZ*arraySize];
	  
	} // end gravityEnabled

	// store them in global memory
	offset = elemOffset;
	
	d_qm[offset] = qleft [ID];
	d_qp[offset] = qright[ID]; offset += arraySize;
	
	d_qm[offset] = qleft [IP];
	d_qp[offset] = qright[IP]; offset += arraySize;
	
	d_qm[offset] = qleft [IU];
	d_qp[offset] = qright[IU]; offset += arraySize;
	
	d_qm[offset] = qleft [IV];
	d_qp[offset] = qright[IV]; offset += arraySize;

	d_qm[offset] = qleft [IW];
	d_qp[offset] = qright[IW]; offset += arraySize;
      		
      } // end if (i,j)

  } // end for k

} // kernel_godunov_trace_by_dir_3d_v2

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
						       real_t       *              Uout,
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
						       real_t       *              Uout,
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
