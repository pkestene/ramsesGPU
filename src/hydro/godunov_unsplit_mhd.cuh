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
 * \file godunov_unsplit_mhd.cuh
 * \brief Defines the CUDA kernel for the actual MHD Godunov scheme.
 *
 * \date 13 Apr 2011
 * \author P. Kestener
 *
 * $Id: godunov_unsplit_mhd.cuh 3450 2014-06-16 22:03:23Z pkestene $
 */
#ifndef GODUNOV_UNSPLIT_MHD_CUH_
#define GODUNOV_UNSPLIT_MHD_CUH_

#include "real_type.h"
#include "constants.h"
#include "constoprim.h"
#include "riemann_mhd.h"
#include "trace_mhd.h"

#include <cstdlib>
#include <float.h>

/** a dummy device-only swap function */
__device__ inline void swap_value_(real_t& a, real_t& b) {
  
  real_t tmp = a;
  a = b;
  b = tmp;
  
} // swap_value_

/* mixed precision version of swap routine
 * if USE_MIXED_PRECISION is define, real_riemann_t is typedef'ed to
 * double 
 */
#ifdef USE_MIXED_PRECISION
__device__ inline void swap_value_(real_riemann_t& a, real_riemann_t& b) {
  
  real_riemann_t tmp = a;
  a = b;
  b = tmp;

} // swap_value
#endif // USE_MIXED_PRECISION

/******************************************************************
 * Define some CUDA kernel common to all MHD implementation on GPU
 ******************************************************************/


#ifdef USE_DOUBLE
#define PRIM_VAR_MHD_BLOCK_DIMX_2D	16
#define PRIM_VAR_MHD_BLOCK_DIMY_2D	16
#else // simple precision
#define PRIM_VAR_MHD_BLOCK_DIMX_2D	16
#define PRIM_VAR_MHD_BLOCK_DIMY_2D	16
#endif // USE_DOUBLE

/**
 * Compute MHD primitive variables in 2D.
 *
 * \param[in]  Uin  input  convervative variable array 
 * \param[out] Qout output primitive variable array
 */
__global__ void kernel_mhd_compute_primitive_variables_2D(const real_t * __restrict__ Uin,
							  real_t       *Qout,
							  int pitch, 
							  int imax, 
							  int jmax,
							  real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, PRIM_VAR_MHD_BLOCK_DIMX_2D) + tx;
  const int j = __mul24(by, PRIM_VAR_MHD_BLOCK_DIMY_2D) + ty;
  
  const int arraySize  = pitch * jmax;
  const int elemOffset = pitch * j + i;

  // conservative variables
  real_t uIn [NVAR_MHD];
  real_t c;

  if( i < imax-1 and j < jmax-1 ) {
    
    // Gather conservative variables
    int offset = elemOffset;
    
    uIn[ID] = Uin[offset];  offset += arraySize;
    uIn[IP] = Uin[offset];  offset += arraySize;
    uIn[IU] = Uin[offset];  offset += arraySize;
    uIn[IV] = Uin[offset];  offset += arraySize;
    uIn[IW] = Uin[offset];  offset += arraySize;
    uIn[IA] = Uin[offset];  offset += arraySize;
    uIn[IB] = Uin[offset];  offset += arraySize;
    uIn[IC] = Uin[offset];
    
    // go to magnetic field components and get values from 
    // neighbors on the right
    real_t magFieldNeighbors[3];
    offset = elemOffset + 5 * arraySize;
    magFieldNeighbors[IX] = Uin[offset+1    ];  offset += arraySize;
    magFieldNeighbors[IY] = Uin[offset+pitch];
    magFieldNeighbors[IZ] = ZERO_F;
    
    //Convert to primitive variables
    real_t qTmp[NVAR_MHD];
    constoprim_mhd(uIn, magFieldNeighbors, qTmp, c, dt);
    
    // copy results into output d_Q
    offset = elemOffset;
    Qout[offset] = qTmp[ID]; offset += arraySize;
    Qout[offset] = qTmp[IP]; offset += arraySize;
    Qout[offset] = qTmp[IU]; offset += arraySize;
    Qout[offset] = qTmp[IV]; offset += arraySize;
    Qout[offset] = qTmp[IW]; offset += arraySize;
    Qout[offset] = qTmp[IA]; offset += arraySize;
    Qout[offset] = qTmp[IB]; offset += arraySize;
    Qout[offset] = qTmp[IC];
    
  } // end if

} // kernel_mhd_compute_primitive_variables_2D


#ifdef USE_DOUBLE
#define PRIM_VAR_MHD_BLOCK_DIMX_3D	16
#define PRIM_VAR_MHD_BLOCK_DIMY_3D	16
#else // simple precision
#define PRIM_VAR_MHD_BLOCK_DIMX_3D	16
#define PRIM_VAR_MHD_BLOCK_DIMY_3D	16
#endif // USE_DOUBLE


/**
 * Compute MHD primitive variables in 3D.
 *
 * \param[in]  Uin  input  convervative variable array 
 * \param[out] Qout output primitive variable array
 */
__global__ void kernel_mhd_compute_primitive_variables_3D(const real_t * __restrict__ Uin,
							  real_t       *Qout,
							  int pitch, 
							  int imax, 
							  int jmax,
							  int kmax,
							  real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, PRIM_VAR_MHD_BLOCK_DIMX_3D) + tx;
  const int j = __mul24(by, PRIM_VAR_MHD_BLOCK_DIMY_3D) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // conservative variables
  real_riemann_t uIn [NVAR_MHD];
  real_t c;

  /*
   * loop over k (i.e. z) to compute primitive variables, and store results
   * in external memory buffer Q.
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if( i < imax-1 and j < jmax-1 ) {

      	// Gather conservative variables (at z=k)
	int offset = elemOffset;

	uIn[ID] = static_cast<real_riemann_t>(Uin[offset]);  offset += arraySize;
	uIn[IP] = static_cast<real_riemann_t>(Uin[offset]);  offset += arraySize;
	uIn[IU] = static_cast<real_riemann_t>(Uin[offset]);  offset += arraySize;
	uIn[IV] = static_cast<real_riemann_t>(Uin[offset]);  offset += arraySize;
	uIn[IW] = static_cast<real_riemann_t>(Uin[offset]);  offset += arraySize;
	uIn[IA] = static_cast<real_riemann_t>(Uin[offset]);  offset += arraySize;
	uIn[IB] = static_cast<real_riemann_t>(Uin[offset]);  offset += arraySize;
	uIn[IC] = static_cast<real_riemann_t>(Uin[offset]);
	
	// go to magnetic field components and get values from 
	// neighbors on the right
	real_riemann_t magFieldNeighbors[3];
	offset = elemOffset + 5 * arraySize;
	magFieldNeighbors[IX] = static_cast<real_riemann_t>(Uin[offset+1         ]);  offset += arraySize;
	magFieldNeighbors[IY] = static_cast<real_riemann_t>(Uin[offset+pitch     ]);  offset += arraySize;
	magFieldNeighbors[IZ] = static_cast<real_riemann_t>(Uin[offset+pitch*jmax]);
	
	//Convert to primitive variables
	real_riemann_t qTmp[NVAR_MHD];
	constoprim_mhd(uIn, magFieldNeighbors, qTmp, c, dt);

	// copy results into output d_Q at z=k
	offset = elemOffset;
	Qout[offset] = static_cast<real_t>(qTmp[ID]); offset += arraySize;
	Qout[offset] = static_cast<real_t>(qTmp[IP]); offset += arraySize;
	Qout[offset] = static_cast<real_t>(qTmp[IU]); offset += arraySize;
	Qout[offset] = static_cast<real_t>(qTmp[IV]); offset += arraySize;
	Qout[offset] = static_cast<real_t>(qTmp[IW]); offset += arraySize;
	Qout[offset] = static_cast<real_t>(qTmp[IA]); offset += arraySize;
	Qout[offset] = static_cast<real_t>(qTmp[IB]); offset += arraySize;
	Qout[offset] = static_cast<real_t>(qTmp[IC]);

    } // end if

  } // enf for k

} // kernel_mhd_compute_primitive_variables_3D

/*****************************************
 *** *** GODUNOV UNSPLIT 2D KERNEL *** ***
 *****************************************/

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define UNSPLIT_BLOCK_DIMX_2D		16
#define UNSPLIT_BLOCK_INNER_DIMX_2D	(UNSPLIT_BLOCK_DIMX_2D-5)
#define UNSPLIT_BLOCK_DIMY_2D		12
#define UNSPLIT_BLOCK_INNER_DIMY_2D	(UNSPLIT_BLOCK_DIMY_2D-5)
#else // simple precision
#define UNSPLIT_BLOCK_DIMX_2D		16
#define UNSPLIT_BLOCK_INNER_DIMX_2D	(UNSPLIT_BLOCK_DIMX_2D-5)
#define UNSPLIT_BLOCK_DIMY_2D		20
#define UNSPLIT_BLOCK_INNER_DIMY_2D	(UNSPLIT_BLOCK_DIMY_2D-5)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 2D data.
 * 
 * This kernel doesn't eat so much shared memory, but to the price of
 * more computations...
 * We remove array qm_x1 and qm_y2, and recompute what is needed by
 * each thread.
 *
 * \note Note that when gravity is enabled, only predictor step is performed here; 
 * the source term computation must be done outside !!!

 */
__global__ void kernel_godunov_unsplit_mhd_2d(const real_t * __restrict__ Uin, 
					      real_t       *Uout,
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
  
  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_2D) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_2D) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  const real_t *gravin = gParams.arrayList[A_GRAV];

  // q   : primitive variables
  // bf  : face-centered magnetic field components
  // emf : z-direction electromotive force 
  __shared__ real_t      q[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_MHD];
  __shared__ real_t     bf[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][3];
  __shared__ real_t    emf[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D];

  // the following could be avoided by interleaving flux computation / update (so that flux_x can 
  // be re-used to hold flux_y
  //__shared__ real_t flux_y[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_MHD];

  // qm and qp's are output of the trace step
  real_t qm [TWO_D][NVAR_MHD];
  real_t qm_x10[NVAR_MHD];
  real_t qm_y01[NVAR_MHD];

  real_t qp [TWO_D][NVAR_MHD];
  real_t qp0[TWO_D][NVAR_MHD];

  // in 2D, we only need to compute emfZ
  real_t qEdge[4][NVAR_MHD];       // used in trace_unsplit call
  real_t qEdge_emfZ[4][NVAR_MHD];  // used in compute emf 
  real_t (&qEdge_RT)[NVAR_MHD] = qEdge_emfZ[IRT];
  real_t (&qEdge_RB)[NVAR_MHD] = qEdge_emfZ[IRB];
  real_t (&qEdge_LT)[NVAR_MHD] = qEdge_emfZ[ILT];
  real_t (&qEdge_LB)[NVAR_MHD] = qEdge_emfZ[ILB];

  // conservative variables
  real_t uIn [NVAR_MHD];
  real_t uOut[NVAR_MHD];
  real_t c;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation


  // load U and convert to primitive variables
  if(i >= 0 and i < imax-1 and 
     j >= 0 and j < jmax-1)
    {
      
      // Gather conservative variables
      int offset = elemOffset;
      uIn[ID] = Uin[offset];  offset += arraySize;
      uIn[IP] = Uin[offset];  offset += arraySize;
      uIn[IU] = Uin[offset];  offset += arraySize;
      uIn[IV] = Uin[offset];  offset += arraySize;
      uIn[IW] = Uin[offset];  offset += arraySize;
      uIn[IA] = Uin[offset];  offset += arraySize;
      uIn[IB] = Uin[offset];  offset += arraySize;
      uIn[IC] = Uin[offset];

      // set bf (face-centered magnetic field components)
      bf[tx][ty][0] = uIn[IA];
      bf[tx][ty][1] = uIn[IB];
      bf[tx][ty][2] = uIn[IC];

      // go to magnetic field components and get values from neighbors on the right
      real_t magFieldNeighbors[3];
      offset = elemOffset + 5 * arraySize;
      magFieldNeighbors[IX] = Uin[offset+1    ];  offset += arraySize;
      magFieldNeighbors[IY] = Uin[offset+pitch];
      magFieldNeighbors[IZ] = ZERO_F;

      // copy input state into uOut that will become output state for update
      uOut[ID] = uIn[ID];
      uOut[IP] = uIn[IP];
      uOut[IU] = uIn[IU];
      uOut[IV] = uIn[IV];
      uOut[IW] = uIn[IW];
      uOut[IA] = uIn[IA];
      uOut[IB] = uIn[IB];
      uOut[IC] = uIn[IC];
 
      // Convert to primitive variables
      constoprim_mhd(uIn, magFieldNeighbors, q[tx][ty], c, dt);
    }
  __syncthreads();

  if(i > 1 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-2 and
     j > 1 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-2)
    {
      real_t qNeighbors[3][3][NVAR_MHD]; 
      real_t bfNb[4][4][3];
      
      // prepare neighbors data
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qNeighbors[0][0][iVar] = q[tx-1][ty-1][iVar];
	qNeighbors[0][1][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[0][2][iVar] = q[tx-1][ty+1][iVar];
	qNeighbors[1][0][iVar] = q[tx  ][ty-1][iVar];
	qNeighbors[1][1][iVar] = q[tx  ][ty  ][iVar];
	qNeighbors[1][2][iVar] = q[tx  ][ty+1][iVar];
	qNeighbors[2][0][iVar] = q[tx+1][ty-1][iVar];
	qNeighbors[2][1][iVar] = q[tx+1][ty  ][iVar];
	qNeighbors[2][2][iVar] = q[tx+1][ty+1][iVar];
      }	
      for (int iVar=0; iVar<3; iVar++) {
	for (int ibfy=0; ibfy<4; ibfy++)
	  for (int ibfx=0; ibfx<4; ibfx++) {
	    bfNb[ibfx][ibfy][iVar] = bf[tx-1+ibfx][ty-1+ibfy][iVar];
	  }
      }

      // Characteristic tracing : compute qp0 (analog to qp_x[0][0] and qp_y[0][0])
      trace_unsplit_mhd_2d(qNeighbors, bfNb, c, dtdx, dtdy, xPos, qm, qp0, qEdge);

      // get qEdge_LB
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qEdge_LB[iVar] = qEdge[ILB][iVar];
      }

      // gravity predictor on velocity component of qp0's and qEdge_LB
      if (gravityEnabled) {
	qp0[0][IU]   += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qp0[0][IV]   += HALF_F * dt * gravin[elemOffset+IY*arraySize];

	qp0[1][IU]   += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qp0[1][IV]   += HALF_F * dt * gravin[elemOffset+IY*arraySize];

	qEdge_LB[IU] += HALF_F * dt * gravin[elemOffset+IX*arraySize];
	qEdge_LB[IV] += HALF_F * dt * gravin[elemOffset+IY*arraySize];
      }

      // Shift neighborhood -1 along X direction
      // Characteristic tracing : compute qm_x[1][0]
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qNeighbors[0][0][iVar] = q[tx-2][ty-1][iVar];
	qNeighbors[0][1][iVar] = q[tx-2][ty  ][iVar];
	qNeighbors[0][2][iVar] = q[tx-2][ty+1][iVar];
	qNeighbors[1][0][iVar] = q[tx-1][ty-1][iVar];
	qNeighbors[1][1][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[1][2][iVar] = q[tx-1][ty+1][iVar];
	qNeighbors[2][0][iVar] = q[tx  ][ty-1][iVar];
	qNeighbors[2][1][iVar] = q[tx  ][ty  ][iVar];
	qNeighbors[2][2][iVar] = q[tx  ][ty+1][iVar];
      }
      for (int iVar=0; iVar<3; iVar++) {
	for (int ibfy=0; ibfy<4; ibfy++)
	  for (int ibfx=0; ibfx<4; ibfx++) {
	    bfNb[ibfx][ibfy][iVar] = bf[tx-2+ibfx][ty-1+ibfy][iVar];
	  }
      }
      trace_unsplit_mhd_2d(qNeighbors, bfNb, c, dtdx, dtdy, xPos, qm, qp, qEdge);
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qm_x10  [iVar] = qm[0][iVar];
	qEdge_RB[iVar] = qEdge[IRB][iVar];
      }

      // gravity predictor on velocity component of qm_x10 and qEdge_RB
      if (gravityEnabled) {
	qm_x10[IU]     += HALF_F * dt * gravin[elemOffset-1+IX*arraySize];
	qm_x10[IV]     += HALF_F * dt * gravin[elemOffset-1+IY*arraySize];

	qEdge_RB[IU]   += HALF_F * dt * gravin[elemOffset-1+IX*arraySize];
	qEdge_RB[IV]   += HALF_F * dt * gravin[elemOffset-1+IY*arraySize];
      }


      // Shift neighborhood -1 along Y direction
      // Characteristic tracing : compute qm_y[0][1]
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qNeighbors[0][0][iVar] = q[tx-1][ty-2][iVar];
	qNeighbors[0][1][iVar] = q[tx-1][ty-1][iVar];
	qNeighbors[0][2][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[1][0][iVar] = q[tx  ][ty-2][iVar];
	qNeighbors[1][1][iVar] = q[tx  ][ty-1][iVar];
	qNeighbors[1][2][iVar] = q[tx  ][ty  ][iVar];
	qNeighbors[2][0][iVar] = q[tx+1][ty-2][iVar];
	qNeighbors[2][1][iVar] = q[tx+1][ty-1][iVar];
	qNeighbors[2][2][iVar] = q[tx+1][ty  ][iVar];
      }
      for (int iVar=0; iVar<3; iVar++) {
	for (int ibfy=0; ibfy<4; ibfy++)
	  for (int ibfx=0; ibfx<4; ibfx++) {
	    bfNb[ibfx][ibfy][iVar] = bf[tx-1+ibfx][ty-2+ibfy][iVar];
	  }
      }
      trace_unsplit_mhd_2d(qNeighbors, bfNb, c, dtdx, dtdy, xPos, qm, qp, qEdge);
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qm_y01  [iVar] = qm[1][iVar];
	qEdge_LT[iVar] = qEdge[ILT][iVar];
      }

      // gravity predictor on velocity component of qm_y01 and qEdge_LT
      if (gravityEnabled) {
	qm_y01[IU]       += HALF_F * dt * gravin[elemOffset-pitch+IX*arraySize];
	qm_y01[IV]       += HALF_F * dt * gravin[elemOffset-pitch+IY*arraySize];

	qEdge_LT[IU]     += HALF_F * dt * gravin[elemOffset-pitch+IX*arraySize];
	qEdge_LT[IV]     += HALF_F * dt * gravin[elemOffset-pitch+IY*arraySize];
      }

      // Shift neighborhood -1 along X and Y direction
      // goal : compute qEdge_RT
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qNeighbors[0][0][iVar] = q[tx-2][ty-2][iVar];
	qNeighbors[0][1][iVar] = q[tx-2][ty-1][iVar];
	qNeighbors[0][2][iVar] = q[tx-2][ty  ][iVar];
	qNeighbors[1][0][iVar] = q[tx-1][ty-2][iVar];
	qNeighbors[1][1][iVar] = q[tx-1][ty-1][iVar];
	qNeighbors[1][2][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[2][0][iVar] = q[tx  ][ty-2][iVar];
	qNeighbors[2][1][iVar] = q[tx  ][ty-1][iVar];
	qNeighbors[2][2][iVar] = q[tx  ][ty  ][iVar];
      }
      for (int iVar=0; iVar<3; iVar++) {
	for (int ibfy=0; ibfy<4; ibfy++)
	  for (int ibfx=0; ibfx<4; ibfx++) {
	    bfNb[ibfx][ibfy][iVar] = bf[tx-2+ibfx][ty-2+ibfy][iVar];
	  }
      }
      trace_unsplit_mhd_2d(qNeighbors, bfNb, c, dtdx, dtdy, xPos, qm, qp, qEdge);
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qEdge_RT[iVar] = qEdge[IRT][iVar];
      }

      // gravity predictor on velocity component of qEdge_RT
      if (gravityEnabled) {
	qEdge_RT[IU]       += HALF_F * dt * gravin[elemOffset-1-pitch+IX*arraySize];
	qEdge_RT[IV]       += HALF_F * dt * gravin[elemOffset-1-pitch+IY*arraySize];
      }

    }
  __syncthreads();

  // re-use q as flux_x
  real_t (&flux_x)[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_MHD] = q;
  flux_x[tx][ty][ID] = ZERO_F;
  flux_x[tx][ty][IP] = ZERO_F;
  flux_x[tx][ty][IU] = ZERO_F;
  flux_x[tx][ty][IV] = ZERO_F;
  flux_x[tx][ty][IW] = ZERO_F;
  flux_x[tx][ty][IA] = ZERO_F;
  flux_x[tx][ty][IB] = ZERO_F;
  flux_x[tx][ty][IC] = ZERO_F;
  __syncthreads();

  if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-2 and
     j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-2)
    {
      // Solve Riemann problem at X-interfaces and compute fluxes
      real_riemann_t (&qleft_x)[NVAR_MHD]  = qm_x10;
      real_riemann_t (&qright_x)[NVAR_MHD] = qp0[0];      
      riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);

      /* shear correction */
      if (/* cartesian geometry */ ::gParams.Omega0 > 0) {
	real_t shear_x = -1.5 * ::gParams.Omega0 * (xPos + xPos - ::gParams.dx); // xPos-dx is xPos at position i-1
	flux_x[tx][ty][IC] += shear_x * ( qleft_x[IA] + qright_x[IA] ) / 2; // bn_mean is along direction IA
      } // end shear correction
    }  
  __syncthreads();
  
  if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-3 and
     j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-3)
    {
      // update uOut with flux_x
      uOut[ID] += (flux_x[tx][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
      uOut[IP] += (flux_x[tx][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
      uOut[IU] += (flux_x[tx][ty][IU]-flux_x[tx+1][ty][IU])*dtdx;
      uOut[IV] += (flux_x[tx][ty][IV]-flux_x[tx+1][ty][IV])*dtdx;
      uOut[IW] += (flux_x[tx][ty][IW]-flux_x[tx+1][ty][IW])*dtdx;
      uOut[IC] += (flux_x[tx][ty][IC]-flux_x[tx+1][ty][IC])*dtdx;
      
    }
  
  // re-use q as flux_y
  real_t (&flux_y)[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_MHD] = q;
  flux_y[tx][ty][ID] = ZERO_F;
  flux_y[tx][ty][IP] = ZERO_F;
  flux_y[tx][ty][IU] = ZERO_F;
  flux_y[tx][ty][IV] = ZERO_F;
  flux_y[tx][ty][IW] = ZERO_F;
  flux_y[tx][ty][IA] = ZERO_F;
  flux_y[tx][ty][IB] = ZERO_F;
  flux_y[tx][ty][IC] = ZERO_F;
  
  if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-2 and
     j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-2)
    {
      // Solve Riemann problem at Y-interfaces and compute fluxes
      real_riemann_t (&qleft_y)[NVAR_MHD]  = qm_y01;
      real_riemann_t (&qright_y)[NVAR_MHD] = qp0[1];
      // watchout swap IU and IV
      swap_value_(qleft_y[IU],qleft_y[IV]);
      swap_value_(qleft_y[IA],qleft_y[IB]);
      swap_value_(qright_y[IU],qright_y[IV]);
      swap_value_(qright_y[IA],qright_y[IB]);
      riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);
      
      /* shear correction */
      if (/* cartesian geometry */ ::gParams.Omega0 > 0) {
	real_t shear_y = -1.5 * ::gParams.Omega0 * xPos;
	flux_y[tx][ty][IC] += shear_y * ( qleft_y[IA] + qright_y[IA] ) / 2; // bn_mean is always along direction IA (due to permutation above)
      } // end shear correction

    }  
  __syncthreads();
  
  if(i >= 2 and i < imax-3 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-3 and
     j >= 2 and j < jmax-3 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-3)
    {
      
      // update U with flux_y : watchout! IU and IV are swapped !
      uOut[ID] += (flux_y[tx][ty][ID]-flux_y[tx][ty+1][ID])*dtdx;
      uOut[IP] += (flux_y[tx][ty][IP]-flux_y[tx][ty+1][IP])*dtdx;
      uOut[IU] += (flux_y[tx][ty][IV]-flux_y[tx][ty+1][IV])*dtdx;
      uOut[IV] += (flux_y[tx][ty][IU]-flux_y[tx][ty+1][IU])*dtdx;
      uOut[IW] += (flux_y[tx][ty][IW]-flux_y[tx][ty+1][IW])*dtdx;
      uOut[IC] += (flux_y[tx][ty][IC]-flux_y[tx][ty+1][IC])*dtdx;
            
    }
   __syncthreads();
  
  /* compute EMF to update face-centered magnetic field components :
   * uOut[IA] and uOut[IB] */
  if(i > 1 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-2 and
     j > 1 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-2)
    {
      
      emf[tx][ty] = compute_emf<EMFZ>(qEdge_emfZ,xPos);
    }
   __syncthreads();
  
  if(i >= 3 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D-3 and
     j >= 3 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D-3)
    {
      uOut[IA] -= (emf[tx][ty]-emf[tx  ][ty+1])*dtdy;
      uOut[IB] += (emf[tx][ty]-emf[tx+1][ty  ])*dtdx;

      // actually perform update on external device memory
      int offset = elemOffset;
      Uout[offset] = uOut[ID];  offset += arraySize;
      Uout[offset] = uOut[IP];  offset += arraySize;
      Uout[offset] = uOut[IU];  offset += arraySize;
      Uout[offset] = uOut[IV];  offset += arraySize;
      Uout[offset] = uOut[IW];  offset += arraySize;
      Uout[offset] = uOut[IA];  offset += arraySize;
      Uout[offset] = uOut[IB];  offset += arraySize;
      Uout[offset] = uOut[IC];
    
    }

} // kernel_godunov_unsplit_2d_mhd

/*
 * a second implementation that uses global memory to store qm, qp ad qEdge
 */

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define UNSPLIT_BLOCK_DIMX_2D_V1	16
#define UNSPLIT_BLOCK_INNER_DIMX_2D_V1	(UNSPLIT_BLOCK_DIMX_2D_V1-5)
#define UNSPLIT_BLOCK_DIMY_2D_V1	12
#define UNSPLIT_BLOCK_INNER_DIMY_2D_V1	(UNSPLIT_BLOCK_DIMY_2D_V1-5)
#else // simple precision
#define UNSPLIT_BLOCK_DIMX_2D_V1	16
#define UNSPLIT_BLOCK_INNER_DIMX_2D_V1	(UNSPLIT_BLOCK_DIMX_2D_V1-5)
#define UNSPLIT_BLOCK_DIMY_2D_V1	20
#define UNSPLIT_BLOCK_INNER_DIMY_2D_V1	(UNSPLIT_BLOCK_DIMY_2D_V1-5)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 2D data.
 * 
 * This kernel uses more global memory (store qm, qp and qEdge data).
 *
 * \param[in]     Uin
 * \param[out]    Uout
 * \param[in,out] d_qm_x
 * \param[in,out] d_qm_y
 * \param[in,out] d_qEdge_RT
 * \param[in,out] d_qEdge_RB
 * \param[in,out] d_qEdge_LT
 * \param[out]    d_emf
 *
 */
__global__ void kernel_godunov_unsplit_mhd_2d_v1(const real_t * __restrict__ Uin, 
						 real_t *Uout,
						 real_t *d_qm_x,
						 real_t *d_qm_y,
						 real_t *d_qEdge_RT,
						 real_t *d_qEdge_RB,
						 real_t *d_qEdge_LT,
						 real_t *d_emf,
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
  
  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_2D_V1) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_2D_V1) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  // q   : primitive variables
  // bf  : face-centered magnetic field components
  // emf : z-direction electromotive force 
  __shared__ real_t      q[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1][NVAR_MHD];
  __shared__ real_t     bf[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1][3];
  //__shared__ real_t    emf[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1];
  real_t emf;

  // qm and qp's are output of the trace step
  real_t qm [TWO_D][NVAR_MHD];
  real_t qp [TWO_D][NVAR_MHD];

  // in 2D, we only need to compute emfZ
  real_t qEdge[4][NVAR_MHD];       // used in trace_unsplit call
  real_t qEdge_emfZ[4][NVAR_MHD];  // used in compute emf 
  real_t (&qEdge_RT)[NVAR_MHD] = qEdge_emfZ[IRT];
  real_t (&qEdge_RB)[NVAR_MHD] = qEdge_emfZ[IRB];
  real_t (&qEdge_LT)[NVAR_MHD] = qEdge_emfZ[ILT];
  real_t (&qEdge_LB)[NVAR_MHD] = qEdge_emfZ[ILB];

  // conservative variables
  real_t uIn [NVAR_MHD];
  real_t uOut[NVAR_MHD];
  real_t c;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation
  
  // load U and convert to primitive variables
  if(i >= 0 and i < imax-1 and 
     j >= 0 and j < jmax-1)
    {
      
      // Gather conservative variables
      int offset = elemOffset;
      uIn[ID] = Uin[offset];  offset += arraySize;
      uIn[IP] = Uin[offset];  offset += arraySize;
      uIn[IU] = Uin[offset];  offset += arraySize;
      uIn[IV] = Uin[offset];  offset += arraySize;
      uIn[IW] = Uin[offset];  offset += arraySize;
      uIn[IA] = Uin[offset];  offset += arraySize;
      uIn[IB] = Uin[offset];  offset += arraySize;
      uIn[IC] = Uin[offset];

      // set bf (face-centered magnetic field components)
      bf[tx][ty][0] = uIn[IA];
      bf[tx][ty][1] = uIn[IB];
      bf[tx][ty][2] = uIn[IC];

      // go to magnetic field components and get values from neighbors on the right
      real_t magFieldNeighbors[3];
      offset = elemOffset + 5 * arraySize;
      magFieldNeighbors[IX] = Uin[offset+1    ];  offset += arraySize;
      magFieldNeighbors[IY] = Uin[offset+pitch];
      magFieldNeighbors[IZ] = ZERO_F;

      // copy input state into uOut that will become output state for update
      uOut[ID] = uIn[ID];
      uOut[IP] = uIn[IP];
      uOut[IU] = uIn[IU];
      uOut[IV] = uIn[IV];
      uOut[IW] = uIn[IW];
      uOut[IA] = uIn[IA];
      uOut[IB] = uIn[IB];
      uOut[IC] = uIn[IC];
 
      // Convert to primitive variables
      constoprim_mhd(uIn, magFieldNeighbors, q[tx][ty], c, dt);
    }
  __syncthreads();

  if(i > 1 and i < imax-2 and tx > 0 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-2 and
     j > 1 and j < jmax-2 and ty > 0 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-2)
    {
      real_t qNeighbors[3][3][NVAR_MHD]; 
      real_t bfNb[4][4][3];
      
      // prepare neighbors data
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qNeighbors[0][0][iVar] = q[tx-1][ty-1][iVar];
	qNeighbors[0][1][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[0][2][iVar] = q[tx-1][ty+1][iVar];
	qNeighbors[1][0][iVar] = q[tx  ][ty-1][iVar];
	qNeighbors[1][1][iVar] = q[tx  ][ty  ][iVar];
	qNeighbors[1][2][iVar] = q[tx  ][ty+1][iVar];
	qNeighbors[2][0][iVar] = q[tx+1][ty-1][iVar];
	qNeighbors[2][1][iVar] = q[tx+1][ty  ][iVar];
	qNeighbors[2][2][iVar] = q[tx+1][ty+1][iVar];
      }	
      for (int iVar=0; iVar<3; iVar++) {
	for (int ibfy=0; ibfy<4; ibfy++)
	  for (int ibfx=0; ibfx<4; ibfx++) {
	    bfNb[ibfx][ibfy][iVar] = bf[tx-1+ibfx][ty-1+ibfy][iVar];
	  }
      }

      // Characteristic tracing : compute qm, qp
      trace_unsplit_mhd_2d(qNeighbors, bfNb, 
			   c, dtdx, dtdy, xPos, qm, qp, qEdge);

      // store qm, qEdge in external memory
      int offset = elemOffset;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	d_qm_x[offset]     = qm[0][iVar];
	d_qm_y[offset]     = qm[1][iVar];
	d_qEdge_RT[offset] = qEdge[IRT][iVar];
	d_qEdge_RB[offset] = qEdge[IRB][iVar];
	d_qEdge_LT[offset] = qEdge[ILT][iVar];
	qEdge_LB[iVar]     = qEdge[ILB][iVar]; // qEdge_LB is just a local variable
	offset += arraySize;
      }

    }
  __syncthreads();

  // re-use q as flux_x
  real_t (&flux_x)[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1][NVAR_MHD] = q;
  flux_x[tx][ty][ID] = ZERO_F;
  flux_x[tx][ty][IP] = ZERO_F;
  flux_x[tx][ty][IU] = ZERO_F;
  flux_x[tx][ty][IV] = ZERO_F;
  flux_x[tx][ty][IW] = ZERO_F;
  flux_x[tx][ty][IA] = ZERO_F;
  flux_x[tx][ty][IB] = ZERO_F;
  flux_x[tx][ty][IC] = ZERO_F;
  __syncthreads();

  if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-2 and
     j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-2)
    {
      // Solve Riemann problem at X-interfaces and compute fluxes
      real_riemann_t   qleft_x  [NVAR_MHD];
      real_riemann_t (&qright_x)[NVAR_MHD] = qp[0];
      
      // read qm_x from external memory at location x-1
      int offset = elemOffset-1;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qleft_x[iVar] = d_qm_x[offset];
	offset += arraySize;
      }

      riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);

      /* shear correction */
      if (/* cartesian geometry */ ::gParams.Omega0 > 0) {
	real_t shear_x = -1.5 * ::gParams.Omega0 * (xPos + xPos - ::gParams.dx); // xPos-dx is xPos at position i-1
	flux_x[tx][ty][IC] += shear_x * ( qleft_x[IA] + qright_x[IA] ) / 2; // bn_mean is along direction IA
      } // end shear correction

    }  
  __syncthreads();
  
  if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-3 and
     j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-3)
    {
      // update uOut with flux_x
      uOut[ID] += (flux_x[tx][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
      uOut[IP] += (flux_x[tx][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
      uOut[IU] += (flux_x[tx][ty][IU]-flux_x[tx+1][ty][IU])*dtdx;
      uOut[IV] += (flux_x[tx][ty][IV]-flux_x[tx+1][ty][IV])*dtdx;
      uOut[IW] += (flux_x[tx][ty][IW]-flux_x[tx+1][ty][IW])*dtdx;
      uOut[IC] += (flux_x[tx][ty][IC]-flux_x[tx+1][ty][IC])*dtdx;
      
    }
  
  // re-use q as flux_y
  real_t (&flux_y)[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1][NVAR_MHD] = q;
  flux_y[tx][ty][ID] = ZERO_F;
  flux_y[tx][ty][IP] = ZERO_F;
  flux_y[tx][ty][IU] = ZERO_F;
  flux_y[tx][ty][IV] = ZERO_F;
  flux_y[tx][ty][IW] = ZERO_F;
  flux_y[tx][ty][IA] = ZERO_F;
  flux_y[tx][ty][IB] = ZERO_F;
  flux_y[tx][ty][IC] = ZERO_F;
  
  if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-2 and
     j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-2)
    {
      // Solve Riemann problem at Y-interfaces and compute fluxes
      real_riemann_t   qleft_y  [NVAR_MHD];
      real_riemann_t (&qright_y)[NVAR_MHD] = qp[1];

      // read qm_y from external memory at location y-1
      int offset = elemOffset-pitch;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qleft_y[iVar] = d_qm_y[offset];
	offset += arraySize;
      }


      // watchout swap IU and IV
      swap_value_(qleft_y[IU],qleft_y[IV]);
      swap_value_(qleft_y[IA],qleft_y[IB]);
      swap_value_(qright_y[IU],qright_y[IV]);
      swap_value_(qright_y[IA],qright_y[IB]);
      riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);
      
      /* shear correction */
      if (/* cartesian geometry */ ::gParams.Omega0 > 0) {
	real_t shear_y = -1.5 * ::gParams.Omega0 * xPos;
	flux_y[tx][ty][IC] += shear_y * ( qleft_y[IA] + qright_y[IA] ) / 2; // bn_mean is always along direction IA (due to permutation above)
      } // end shear correction
      
    }  
  __syncthreads();
  
  if(i >= 2 and i < imax-3 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-3 and
     j >= 2 and j < jmax-3 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-3)
    {
      
      // update U with flux_y : watchout! IU and IV are swapped !
      uOut[ID] += (flux_y[tx][ty][ID]-flux_y[tx][ty+1][ID])*dtdy;
      uOut[IP] += (flux_y[tx][ty][IP]-flux_y[tx][ty+1][IP])*dtdy;
      uOut[IU] += (flux_y[tx][ty][IV]-flux_y[tx][ty+1][IV])*dtdy;
      uOut[IV] += (flux_y[tx][ty][IU]-flux_y[tx][ty+1][IU])*dtdy;
      uOut[IW] += (flux_y[tx][ty][IW]-flux_y[tx][ty+1][IW])*dtdy;
      uOut[IC] += (flux_y[tx][ty][IC]-flux_y[tx][ty+1][IC])*dtdy;
            
    }
   __syncthreads();
  
  /* compute EMF to update face-centered magnetic field components :
   * uOut[IA] and uOut[IB] */
   if(i > 1 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-3 and
      j > 1 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-3)
    {

      /*
       * pick qEdge from external memory
       */

      // qEdge_RT at location x-1, y-1
      int offset = elemOffset-1-pitch;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qEdge_RT[iVar] = d_qEdge_RT[offset];
	offset += arraySize;
      }

      // qEdge RB at location x-1, y
      offset = elemOffset-1;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qEdge_RB[iVar] = d_qEdge_RB[offset];
	offset += arraySize;
      }

      // qEdge_LT at locate x, y-1
      offset = elemOffset-pitch;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qEdge_LT[iVar] = d_qEdge_LT[offset];
	offset += arraySize;
      }

      // qEdge_LB at current location (already assigned in a previous stage)
      
      // finally compute emf and store on external memory
      emf = compute_emf<EMFZ>(qEdge_emfZ,xPos);
      offset = elemOffset;
      d_emf[offset] = emf;
    }
   __syncthreads();
  
  if(i >= 3 and i < imax-3 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-3 and
     j >= 3 and j < jmax-3 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-3)
    {
      //uOut[IA] -= (emf[tx][ty]-emf[tx  ][ty+1])*dtdy;
      //uOut[IB] += (emf[tx][ty]-emf[tx+1][ty  ])*dtdx;

      // actually perform update on external device memory
      int offset = elemOffset;
      Uout[offset] = uOut[ID];  offset += arraySize;
      Uout[offset] = uOut[IP];  offset += arraySize;
      Uout[offset] = uOut[IU];  offset += arraySize;
      Uout[offset] = uOut[IV];  offset += arraySize;
      Uout[offset] = uOut[IW];  offset += arraySize;
      Uout[offset] = uOut[IA];  offset += arraySize;
      Uout[offset] = uOut[IB];  offset += arraySize;
      Uout[offset] = uOut[IC];
    
    }

} // kernel_godunov_unsplit_2d_mhd_v1


// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define UNSPLIT_BLOCK_DIMX_2D_V1_EMF	16
#define UNSPLIT_BLOCK_DIMY_2D_V1_EMF	20
#else // simple precision
#define UNSPLIT_BLOCK_DIMX_2D_V1_EMF	16
#define UNSPLIT_BLOCK_DIMY_2D_V1_EMF	20
#endif // USE_DOUBLE

/**
 * MHD kernel designed to be used when implementation version is 1.
 * 
 * This kernel reads emf data from d_emf and perform constraint transport update
 * of magnetic field components.
 *
 * \param[in]  Uin
 * \param[out] Uout
 * \param[in]  d_emf
 */
__global__ void kernel_mhd_2d_update_emf_v1(const real_t * __restrict__ Uin, 
					    real_t       *Uout,
					    const real_t * __restrict__ d_emf,
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
  
  const int i = __mul24(bx, UNSPLIT_BLOCK_DIMX_2D_V1_EMF) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_DIMY_2D_V1_EMF) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  // conservative variables
  real_t uOut[NVAR_MHD];
  
  if(i >= 3 and i < imax-2 and 
     j >= 3 and j < jmax-2)
    {

      int offset = elemOffset + 5 * arraySize;
      
      // read magnetic field components from external memory
      uOut[IA] = Uin[offset];  offset += arraySize;
      uOut[IB] = Uin[offset];  offset += arraySize;
      uOut[IC] = Uin[offset];

      // indexes used to fetch emf's at the right location
      const int ij   = elemOffset;
      const int ip1j = ij + 1;
      const int ijp1 = ij + pitch;
      
      uOut[IA] += ( d_emf[ijp1] - 
		    d_emf[ij  ] ) * dtdy;
      
      uOut[IB] -= ( d_emf[ip1j] - 
		    d_emf[ij  ] ) * dtdx;

      // write back mag field components in external memory
      offset = elemOffset + 5 * arraySize;
      
      Uout[offset] = uOut[IA];  offset += arraySize;
      Uout[offset] = uOut[IB];

    }

} // kernel_godunov_unsplit_mhd_2d_emf_v1

/**
 * Unsplit Godunov kernel in rotating frame for 2D data.
 * 
 * This kernel uses more global memory (store qm, qp and qEdge data).
 */
__global__ void kernel_godunov_unsplit_mhd_rotating_2d_v1(const real_t * __restrict__ Uin, 
							  real_t *Uout,
							  real_t *d_qm_x,
							  real_t *d_qm_y,
							  real_t *d_qEdge_RT,
							  real_t *d_qEdge_RB,
							  real_t *d_qEdge_LT,
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
  
  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_2D_V1) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_2D_V1) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  // q   : primitive variables
  // bf  : face-centered magnetic field components
  // emf : z-direction electromotive force 
  __shared__ real_t      q[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1][NVAR_MHD];
  __shared__ real_t     bf[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1][3];
  __shared__ real_t    emf[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1];

  // qm and qp's are output of the trace step
  real_t qm [TWO_D][NVAR_MHD];
  real_t qp [TWO_D][NVAR_MHD];

  // in 2D, we only need to compute emfZ
  real_t qEdge[4][NVAR_MHD];       // used in trace_unsplit call
  real_t qEdge_emfZ[4][NVAR_MHD];  // used in compute emf 
  real_t (&qEdge_RT)[NVAR_MHD] = qEdge_emfZ[IRT];
  real_t (&qEdge_RB)[NVAR_MHD] = qEdge_emfZ[IRB];
  real_t (&qEdge_LT)[NVAR_MHD] = qEdge_emfZ[ILT];
  real_t (&qEdge_LB)[NVAR_MHD] = qEdge_emfZ[ILB];

  // conservative variables
  real_t uIn [NVAR_MHD];
  real_t uOut[NVAR_MHD];
  real_t c;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  /* shearing box correction on momentum parameters (CARTESIAN only here) */
  real_t &Omega0 = ::gParams.Omega0;
  const real_t lambda = ONE_FOURTH_F * (Omega0*Omega0) *(dt*dt);
  const real_t ratio  = (ONE_F-lambda)/(ONE_F+lambda); 
  const real_t alpha1 =     ONE_F/(ONE_F+lambda);
  const real_t alpha2 = Omega0*dt/(ONE_F+lambda);
  
  // load U and convert to primitive variables
  if(i >= 0 and i < imax-1 and 
     j >= 0 and j < jmax-1)
    {
      
      // Gather conservative variables
      int offset = elemOffset;
      uIn[ID] = Uin[offset];  offset += arraySize;
      uIn[IP] = Uin[offset];  offset += arraySize;
      uIn[IU] = Uin[offset];  offset += arraySize;
      uIn[IV] = Uin[offset];  offset += arraySize;
      uIn[IW] = Uin[offset];  offset += arraySize;
      uIn[IA] = Uin[offset];  offset += arraySize;
      uIn[IB] = Uin[offset];  offset += arraySize;
      uIn[IC] = Uin[offset];

      // set bf (face-centered magnetic field components)
      bf[tx][ty][0] = uIn[IA];
      bf[tx][ty][1] = uIn[IB];
      bf[tx][ty][2] = uIn[IC];

      // go to magnetic field components and get values from neighbors on the right
      real_t magFieldNeighbors[3];
      offset = elemOffset + 5 * arraySize;
      magFieldNeighbors[IX] = Uin[offset+1    ];  offset += arraySize;
      magFieldNeighbors[IY] = Uin[offset+pitch];
      magFieldNeighbors[IZ] = ZERO_F;

      // copy input state into uOut that will become output state for update
      uOut[ID] = uIn[ID];
      uOut[IP] = uIn[IP];
      uOut[IU] = uIn[IU];
      uOut[IV] = uIn[IV];
      uOut[IW] = uIn[IW];
      uOut[IA] = uIn[IA];
      uOut[IB] = uIn[IB];
      uOut[IC] = uIn[IC];

      // some rotating frame corrections
      if (i>2 and i<imax-2 and j>2 and j<jmax-2) {
	real_t dsx =   TWO_F * Omega0 * dt * uIn[IV]/(ONE_F + lambda);
	real_t dsy = -HALF_F * Omega0 * dt * uIn[IU]/(ONE_F + lambda);
	uOut[IU] = uIn[IU]*ratio + dsx;
	uOut[IV] = uIn[IV]*ratio + dsy;
      }
 
      // Convert to primitive variables
      constoprim_mhd(uIn, magFieldNeighbors, q[tx][ty], c, dt);
    }
  __syncthreads();

  if(i > 1 and i < imax-2 and tx > 0 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-2 and
     j > 1 and j < jmax-2 and ty > 0 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-2)
    {
      real_t qNeighbors[3][3][NVAR_MHD]; 
      real_t bfNb[4][4][3];
      
      // prepare neighbors data
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qNeighbors[0][0][iVar] = q[tx-1][ty-1][iVar];
	qNeighbors[0][1][iVar] = q[tx-1][ty  ][iVar];
	qNeighbors[0][2][iVar] = q[tx-1][ty+1][iVar];
	qNeighbors[1][0][iVar] = q[tx  ][ty-1][iVar];
	qNeighbors[1][1][iVar] = q[tx  ][ty  ][iVar];
	qNeighbors[1][2][iVar] = q[tx  ][ty+1][iVar];
	qNeighbors[2][0][iVar] = q[tx+1][ty-1][iVar];
	qNeighbors[2][1][iVar] = q[tx+1][ty  ][iVar];
	qNeighbors[2][2][iVar] = q[tx+1][ty+1][iVar];
      }	
      for (int iVar=0; iVar<3; iVar++) {
	for (int ibfy=0; ibfy<4; ibfy++)
	  for (int ibfx=0; ibfx<4; ibfx++) {
	    bfNb[ibfx][ibfy][iVar] = bf[tx-1+ibfx][ty-1+ibfy][iVar];
	  }
      }

      // Characteristic tracing : compute qm, qp
      trace_unsplit_mhd_2d(qNeighbors, bfNb, 
			   c, dtdx, dtdy, xPos, qm, qp, qEdge);

      // store qm, qEdge in external memory
      int offset = elemOffset;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	d_qm_x[offset]     = qm[0][iVar];
	d_qm_y[offset]     = qm[1][iVar];
	d_qEdge_RT[offset] = qEdge[IRT][iVar];
	d_qEdge_RB[offset] = qEdge[IRB][iVar];
	d_qEdge_LT[offset] = qEdge[ILT][iVar];
	qEdge_LB[iVar]     = qEdge[ILB][iVar]; // qEdge_LB is just a local variable
	offset += arraySize;
      }

    }
  __syncthreads();

  // re-use q as flux_x
  real_t (&flux_x)[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1][NVAR_MHD] = q;
  flux_x[tx][ty][ID] = ZERO_F;
  flux_x[tx][ty][IP] = ZERO_F;
  flux_x[tx][ty][IU] = ZERO_F;
  flux_x[tx][ty][IV] = ZERO_F;
  flux_x[tx][ty][IW] = ZERO_F;
  flux_x[tx][ty][IA] = ZERO_F;
  flux_x[tx][ty][IB] = ZERO_F;
  flux_x[tx][ty][IC] = ZERO_F;
  __syncthreads();

  if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-2 and
     j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-2)
    {
      // Solve Riemann problem at X-interfaces and compute fluxes
      real_riemann_t   qleft_x  [NVAR_MHD];
      real_riemann_t (&qright_x)[NVAR_MHD] = qp[0];
      
      // read qm_x from external memory at location x-1
      int offset = elemOffset-1;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qleft_x[iVar] = d_qm_x[offset];
	offset += arraySize;
      }

      riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);

      /* shear correction */
      if (/* cartesian geometry */ Omega0 > 0) {
	real_t shear_x = -1.5 * Omega0 * (xPos + xPos - ::gParams.dx);      // xPos-dx is xPos at position i-1
	flux_x[tx][ty][IC] += shear_x * ( qleft_x[IA] + qright_x[IA] ) / 2; // bn_mean is along direction IA
      } // end shear correction

    }  
  __syncthreads();
  
  if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-3 and
     j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-3)
    {
      // update uOut with flux_x
      uOut[ID] += (flux_x[tx][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
      uOut[IP] += (flux_x[tx][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
      uOut[IU] += ( alpha1*(flux_x[tx][ty][IU]-flux_x[tx+1][ty][IU]) +
		    alpha2*(flux_x[tx][ty][IV]-flux_x[tx+1][ty][IV])) * dtdx;
      uOut[IV] += ( alpha1*(flux_x[tx][ty][IV]-flux_x[tx+1][ty][IV]) -
		    ONE_FOURTH_F*alpha2*(flux_x[tx][ty][IU]-flux_x[tx+1][ty][IU])) * dtdx;
      uOut[IW] += (flux_x[tx][ty][IW]-flux_x[tx+1][ty][IW])*dtdx;
      uOut[IC] += (flux_x[tx][ty][IC]-flux_x[tx+1][ty][IC])*dtdx;
      
    }
  
  // re-use q as flux_y
  real_t (&flux_y)[UNSPLIT_BLOCK_DIMX_2D_V1][UNSPLIT_BLOCK_DIMY_2D_V1][NVAR_MHD] = q;
  flux_y[tx][ty][ID] = ZERO_F;
  flux_y[tx][ty][IP] = ZERO_F;
  flux_y[tx][ty][IU] = ZERO_F;
  flux_y[tx][ty][IV] = ZERO_F;
  flux_y[tx][ty][IW] = ZERO_F;
  flux_y[tx][ty][IA] = ZERO_F;
  flux_y[tx][ty][IB] = ZERO_F;
  flux_y[tx][ty][IC] = ZERO_F;
  
  if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-2 and
     j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-2)
    {
      // Solve Riemann problem at Y-interfaces and compute fluxes
      real_riemann_t   qleft_y  [NVAR_MHD];
      real_riemann_t (&qright_y)[NVAR_MHD] = qp[1];

      // read qm_y from external memory at location y-1
      int offset = elemOffset-pitch;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qleft_y[iVar] = d_qm_y[offset];
	offset += arraySize;
      }


      // watchout swap IU and IV
      swap_value_(qleft_y[IU],qleft_y[IV]);
      swap_value_(qleft_y[IA],qleft_y[IB]);
      swap_value_(qright_y[IU],qright_y[IV]);
      swap_value_(qright_y[IA],qright_y[IB]);
      riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);
      
      /* shear correction */
      if (/* cartesian geometry */ Omega0 > 0) {
	real_t shear_y = -1.5 * Omega0 * xPos;
	flux_y[tx][ty][IC] += shear_y * ( qleft_y[IA] + qright_y[IA] ) / 2; // bn_mean is always along direction IA (due to permutation above)
      } // end shear correction
      
    }  
  __syncthreads();
  
  if(i >= 2 and i < imax-3 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-3 and
     j >= 2 and j < jmax-3 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-3)
    {
      
      // update U with flux_y : watchout! IU and IV are swapped !
      uOut[ID] += (flux_y[tx][ty][ID]-flux_y[tx][ty+1][ID])*dtdy;
      uOut[IP] += (flux_y[tx][ty][IP]-flux_y[tx][ty+1][IP])*dtdy;
      uOut[IU] += ( alpha1*(flux_y[tx][ty][IV]-flux_y[tx][ty+1][IV]) +
		    alpha2*(flux_y[tx][ty][IU]-flux_y[tx][ty+1][IU])) * dtdy;
      uOut[IV] += ( alpha1*(flux_y[tx][ty][IU]-flux_y[tx][ty+1][IU]) -
		    ONE_FOURTH_F*alpha2*(flux_y[tx][ty][IV]-flux_y[tx][ty+1][IV])) * dtdy;
      uOut[IW] += (flux_y[tx][ty][IW]-flux_y[tx][ty+1][IW])*dtdy;
      uOut[IC] += (flux_y[tx][ty][IC]-flux_y[tx][ty+1][IC])*dtdy;
            
    }
   __syncthreads();
  
  /* compute EMF to update face-centered magnetic field components :
   * uOut[IA] and uOut[IB] */
  if(i > 1 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-2 and
     j > 1 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-2)
    {

      /*
       * pick qEdge from external memory
       */

      // qEdge_RT at location x-1, y-1
      int offset = elemOffset-1-pitch;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qEdge_RT[iVar] = d_qEdge_RT[offset];
	offset += arraySize;
      }

      // qEdge RB at location x-1, y
      offset = elemOffset-1;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qEdge_RB[iVar] = d_qEdge_RB[offset];
	offset += arraySize;
      }

      // qEdge_LT at locate x, y-1
      offset = elemOffset-pitch;
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qEdge_LT[iVar] = d_qEdge_LT[offset];
	offset += arraySize;
      }

      // qEdge_LB at current location (already assigned in a previous stage)
      
      // finally compute emf
      emf[tx][ty] = compute_emf<EMFZ>(qEdge_emfZ,xPos);
    }
   __syncthreads();
  
  if(i >= 3 and i < imax-3 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V1-3 and
     j >= 3 and j < jmax-3 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V1-3)
    {
      uOut[IA] -= (emf[tx][ty]-emf[tx  ][ty+1])*dtdy;
      uOut[IB] += (emf[tx][ty]-emf[tx+1][ty  ])*dtdx;

      // actually perform update on external device memory
      int offset = elemOffset;
      Uout[offset] = uOut[ID];  offset += arraySize;
      Uout[offset] = uOut[IP];  offset += arraySize;
      Uout[offset] = uOut[IU];  offset += arraySize;
      Uout[offset] = uOut[IV];  offset += arraySize;
      Uout[offset] = uOut[IW];  offset += arraySize;
      Uout[offset] = uOut[IA];  offset += arraySize;
      Uout[offset] = uOut[IB];  offset += arraySize;
      Uout[offset] = uOut[IC];
    
    }

} // kernel_godunov_unsplit_2d_mhd_v1





/*****************************************
 *** *** GODUNOV UNSPLIT 3D KERNEL *** ***
 *****************************************/

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
#define UNSPLIT_BLOCK_DIMX_3D_V1	12
#define UNSPLIT_BLOCK_INNER_DIMX_3D_V1  (UNSPLIT_BLOCK_DIMX_3D_V1-5)
#define UNSPLIT_BLOCK_DIMY_3D_V1	7
#define UNSPLIT_BLOCK_INNER_DIMY_3D_V1  (UNSPLIT_BLOCK_DIMY_3D_V1-5)
#else // simple precision
#define UNSPLIT_BLOCK_DIMX_3D_V1	12
#define UNSPLIT_BLOCK_INNER_DIMX_3D_V1	(UNSPLIT_BLOCK_DIMX_3D_V1-5)
#define UNSPLIT_BLOCK_DIMY_3D_V1	7
#define UNSPLIT_BLOCK_INNER_DIMY_3D_V1	(UNSPLIT_BLOCK_DIMY_3D_V1-5)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 3D data (implementation version 1).
 * 
 * This kernel uses more global memory (store qm, qp and qEdge data).
 *
 * Important notice :
 * - d_qm_x and d_qm_y only need to be 2D (we only store the current plane).
 * - d_qm_z needs to be 3D, but with a size of 2 along z direction.
 *
 * \note This kernel is too big (consumes too much shared memory which
 * implies to reduce block width : 12 by 7 is very small !). 
 * Besides it is (very) difficult to debug.
 * This kernel gives quite poor results : for volume size 64x64x128
 * acceleration factor is only 6.5 !!! and it seems to be quite stable
 * when volume size changes.
 */
__global__ void kernel_godunov_unsplit_mhd_3d_v1(const real_t * __restrict__ Uin, 
						 real_t *Uout,
						 real_t *d_qm_x,
						 real_t *d_qm_y,
						 real_t *d_qm_z,
						 real_t *d_qEdge_RT,
						 real_t *d_qEdge_RB,
						 real_t *d_qEdge_LT,
						 real_t *d_qEdge_RT2,
						 real_t *d_qEdge_RB2,
						 real_t *d_qEdge_LT2,
						 real_t *d_qEdge_RT3,
						 real_t *d_qEdge_RB3,
						 real_t *d_qEdge_LT3,
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
  
  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_3D_V1) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_3D_V1) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // global offset in a Z plane
  const int elemOffsetXY = __umul24(pitch, j   ) + i;
  const int arraySize2D  = pitch * jmax;

  // we always store in shared memory 4 consecutive XY-plans
  // q   : primitive variables
  // bf  : face-centered magnetic field components
  // emf : z-direction electromotive force 
  __shared__ real_t      q[4][UNSPLIT_BLOCK_DIMX_3D_V1][UNSPLIT_BLOCK_DIMY_3D_V1][NVAR_MHD];
  __shared__ real_t     bf[4][UNSPLIT_BLOCK_DIMX_3D_V1][UNSPLIT_BLOCK_DIMY_3D_V1][3];

  // index to address the 4 plans of data in shared memory
  int low, mid, current, top, tmp;
  low=0;
  current=1;
  mid=2;
  top=3;

  // indexes to current Z plane and previous one
  int iZcurrent  = 0;
  int iZprevious = 1;

  // qm and qp's are output of the trace step
  real_t qm [THREE_D][NVAR_MHD];
  real_t qp [THREE_D][NVAR_MHD];
  real_t qEdge[4][3][NVAR_MHD];

  real_t qEdge_emfX[4][NVAR_MHD];  // used in compute emf 
  real_t (&qEdge_RT)[NVAR_MHD] = qEdge_emfX[IRT];
  real_t (&qEdge_RB)[NVAR_MHD] = qEdge_emfX[IRB];
  real_t (&qEdge_LT)[NVAR_MHD] = qEdge_emfX[ILT];
  real_t (&qEdge_LB)[NVAR_MHD] = qEdge_emfX[ILB];

  real_t qEdge_emfY[4][NVAR_MHD];  // used in compute emf 
  real_t (&qEdge_RT2)[NVAR_MHD] = qEdge_emfY[IRT];
  real_t (&qEdge_RB2)[NVAR_MHD] = qEdge_emfY[IRB];
  real_t (&qEdge_LT2)[NVAR_MHD] = qEdge_emfY[ILT];
  real_t (&qEdge_LB2)[NVAR_MHD] = qEdge_emfY[ILB];

  real_t qEdge_emfZ[4][NVAR_MHD];  // used in compute emf 
  real_t (&qEdge_RT3)[NVAR_MHD] = qEdge_emfZ[IRT];
  real_t (&qEdge_RB3)[NVAR_MHD] = qEdge_emfZ[IRB];
  real_t (&qEdge_LT3)[NVAR_MHD] = qEdge_emfZ[ILT];
  real_t (&qEdge_LB3)[NVAR_MHD] = qEdge_emfZ[ILB];

  // conservative variables
  real_t uIn [NVAR_MHD];
  real_t uOut[NVAR_MHD];
  real_t c;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  /*
   * initialize q (primitive variables ) in the first 4 plans
   */
  for (int k=0, elemOffset = i + pitch * j; 
       k < 4;
       ++k, elemOffset += (pitch*jmax) ) {

    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1)
      {
	
	// Gather conservative variables
	int offset = elemOffset;
	uIn[ID] = Uin[offset];  offset += arraySize;
	uIn[IP] = Uin[offset];  offset += arraySize;
	uIn[IU] = Uin[offset];  offset += arraySize;
	uIn[IV] = Uin[offset];  offset += arraySize;
	uIn[IW] = Uin[offset];  offset += arraySize;
	uIn[IA] = Uin[offset];  offset += arraySize;
	uIn[IB] = Uin[offset];  offset += arraySize;
	uIn[IC] = Uin[offset];

	// set bf (face-centered magnetic field components)
	bf[k][tx][ty][0] = uIn[IA];
	bf[k][tx][ty][1] = uIn[IB];
	bf[k][tx][ty][2] = uIn[IC];
	
	// go to magnetic field components and get values from 
	// neighbors on the right
	real_t magFieldNeighbors[3];
	offset = elemOffset + 5 * arraySize;
	magFieldNeighbors[IX] = Uin[offset+1         ];  offset += arraySize;
	magFieldNeighbors[IY] = Uin[offset+pitch     ];  offset += arraySize;
	magFieldNeighbors[IZ] = Uin[offset+pitch*jmax];

	//Convert to primitive variables
	constoprim_mhd(uIn, magFieldNeighbors, q[k][tx][ty], c, dt);
      }
  } // end converting to primitive variables in the first 4 plans
  __syncthreads();

  /*
   * loop over all X-Y-planes starting at z=1 as the current plane.
   * Note that elemOffset is used in the update stage
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    /*
     * trace computations
     */
    if(i > 1 and i < imax-2 and tx > 0 and tx < UNSPLIT_BLOCK_DIMX_3D_V1-2 and
       j > 1 and j < jmax-2 and ty > 0 and ty < UNSPLIT_BLOCK_DIMY_3D_V1-2 and
       k > 1 and k < kmax-2)
    {
      real_t qNeighbors[3][3][3][NVAR_MHD]; 
      real_t bfNb      [4][4][4][3];

      // prepare neighbors data
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qNeighbors[0][0][0][iVar] = q[low    ][tx-1][ty-1][iVar];
	qNeighbors[0][1][0][iVar] = q[low    ][tx-1][ty  ][iVar];
	qNeighbors[0][2][0][iVar] = q[low    ][tx-1][ty+1][iVar];
	qNeighbors[1][0][0][iVar] = q[low    ][tx  ][ty-1][iVar];
	qNeighbors[1][1][0][iVar] = q[low    ][tx  ][ty  ][iVar];
	qNeighbors[1][2][0][iVar] = q[low    ][tx  ][ty+1][iVar];
	qNeighbors[2][0][0][iVar] = q[low    ][tx+1][ty-1][iVar];
	qNeighbors[2][1][0][iVar] = q[low    ][tx+1][ty  ][iVar];
	qNeighbors[2][2][0][iVar] = q[low    ][tx+1][ty+1][iVar];

	qNeighbors[0][0][1][iVar] = q[current][tx-1][ty-1][iVar];
	qNeighbors[0][1][1][iVar] = q[current][tx-1][ty  ][iVar];
	qNeighbors[0][2][1][iVar] = q[current][tx-1][ty+1][iVar];
	qNeighbors[1][0][1][iVar] = q[current][tx  ][ty-1][iVar];
	qNeighbors[1][1][1][iVar] = q[current][tx  ][ty  ][iVar];
	qNeighbors[1][2][1][iVar] = q[current][tx  ][ty+1][iVar];
	qNeighbors[2][0][1][iVar] = q[current][tx+1][ty-1][iVar];
	qNeighbors[2][1][1][iVar] = q[current][tx+1][ty  ][iVar];
	qNeighbors[2][2][1][iVar] = q[current][tx+1][ty+1][iVar];

	qNeighbors[0][0][2][iVar] = q[mid    ][tx-1][ty-1][iVar];
	qNeighbors[0][1][2][iVar] = q[mid    ][tx-1][ty  ][iVar];
	qNeighbors[0][2][2][iVar] = q[mid    ][tx-1][ty+1][iVar];
	qNeighbors[1][0][2][iVar] = q[mid    ][tx  ][ty-1][iVar];
	qNeighbors[1][1][2][iVar] = q[mid    ][tx  ][ty  ][iVar];
	qNeighbors[1][2][2][iVar] = q[mid    ][tx  ][ty+1][iVar];
	qNeighbors[2][0][2][iVar] = q[mid    ][tx+1][ty-1][iVar];
	qNeighbors[2][1][2][iVar] = q[mid    ][tx+1][ty  ][iVar];
	qNeighbors[2][2][2][iVar] = q[mid    ][tx+1][ty+1][iVar];
      }	
      for (int iVar=0; iVar<3; iVar++) {
	for (int ibfy=0; ibfy<4; ibfy++)
	  for (int ibfx=0; ibfx<4; ibfx++) {
	    bfNb[ibfx][ibfy][0][iVar] = bf[low    ][tx-1+ibfx][ty-1+ibfy][iVar];
	    bfNb[ibfx][ibfy][1][iVar] = bf[current][tx-1+ibfx][ty-1+ibfy][iVar];
	    bfNb[ibfx][ibfy][2][iVar] = bf[mid    ][tx-1+ibfx][ty-1+ibfy][iVar];
	    bfNb[ibfx][ibfy][3][iVar] = bf[top    ][tx-1+ibfx][ty-1+ibfy][iVar];
	  }
      }

      // Characteristic tracing : compute qm, qp
      trace_unsplit_mhd_3d(qNeighbors, bfNb, 
			   c, dtdx, dtdy, dtdz, xPos, qm, qp, qEdge);

      // store qm, qEdge in external memory
      // iZcurrent is equal either to 0 or 1
      int offset  = elemOffsetXY;
      int offset2 = elemOffsetXY+iZcurrent*arraySize2D;
      

      for (int iVar=0; iVar<NVAR_MHD; iVar++) {

	// remember that d_qm_x and d_qm_y are purely 2D
	d_qm_x[offset]  = qm[0][iVar];
	d_qm_y[offset]  = qm[1][iVar];
	// note that we stored 2 consecutive plane of qm_z
	d_qm_z[offset2] = qm[2][iVar];

	d_qEdge_RT[offset2]  = qEdge[IRT][0][iVar];
	d_qEdge_RB[offset2]  = qEdge[IRB][0][iVar];
	d_qEdge_LT[offset2]  = qEdge[ILT][0][iVar];

	d_qEdge_RT2[offset2] = qEdge[IRT][1][iVar];
	d_qEdge_RB2[offset2] = qEdge[IRB][1][iVar];
	d_qEdge_LT2[offset2] = qEdge[ILT][1][iVar];

	d_qEdge_RT3[offset2] = qEdge[IRT][2][iVar];
	d_qEdge_RB3[offset2] = qEdge[IRB][2][iVar];
	d_qEdge_LT3[offset2] = qEdge[ILT][2][iVar];

	qEdge_LB[iVar]      = qEdge[ILB][0][iVar]; // qEdge_LB  is just a local variable
	qEdge_LB2[iVar]     = qEdge[ILB][1][iVar]; // qEdge_LB2 is just a local variable
	qEdge_LB3[iVar]     = qEdge[ILB][2][iVar]; // qEdge_LB3 is just a local variable

	offset  += arraySize2D;
	offset2 += arraySize2D*2; // because there are 2 planes
      }
      
    } // end trace computations
    __syncthreads();

    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use q[low] as flux_x
    real_t (&flux_x)[UNSPLIT_BLOCK_DIMX_3D_V1][UNSPLIT_BLOCK_DIMY_3D_V1][NVAR_MHD] = q[low];
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    flux_x[tx][ty][IA] = ZERO_F;
    flux_x[tx][ty][IB] = ZERO_F;
    flux_x[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V1-2 and
       j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V1-2 and
       k >= 2 and k < kmax-2)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_riemann_t   qleft_x  [NVAR_MHD];
	real_riemann_t (&qright_x)[NVAR_MHD] = qp[0];
	
	// read qm_x from external memory at location x-1
	int offset = elemOffsetXY-1; // this a 2D index
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_x[iVar] = d_qm_x[offset];
	  offset += arraySize2D;
	}
	
	riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);
	
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V1-3 and
       j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V1-3 and
       k >= 2 and k < kmax-2)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offset = elemOffset;
	uOut[ID] = Uin[offset];  offset += arraySize;
	uOut[IP] = Uin[offset];  offset += arraySize;
	uOut[IU] = Uin[offset];  offset += arraySize;
	uOut[IV] = Uin[offset];  offset += arraySize;
	uOut[IW] = Uin[offset];  offset += arraySize;
	uOut[IA] = Uin[offset];  offset += arraySize;
	uOut[IB] = Uin[offset];  offset += arraySize;
	uOut[IC] = Uin[offset];

	uOut[ID] += (flux_x[tx][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
	uOut[IP] += (flux_x[tx][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
	uOut[IU] += (flux_x[tx][ty][IU]-flux_x[tx+1][ty][IU])*dtdx;
	uOut[IV] += (flux_x[tx][ty][IV]-flux_x[tx+1][ty][IV])*dtdx;
	uOut[IW] += (flux_x[tx][ty][IW]-flux_x[tx+1][ty][IW])*dtdx;
      }
    
    /*
     * Compute fluxes at Y-interfaces.
     */
    // re-use q[low] as flux_y
    real_t (&flux_y)[UNSPLIT_BLOCK_DIMX_3D_V1][UNSPLIT_BLOCK_DIMY_3D_V1][NVAR_MHD] = q[low];
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    flux_y[tx][ty][IA] = ZERO_F;
    flux_y[tx][ty][IB] = ZERO_F;
    flux_y[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V1-2 and
       j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V1-2 and
       k >= 2 and k < kmax-2)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_riemann_t   qleft_y  [NVAR_MHD];
	real_riemann_t (&qright_y)[NVAR_MHD] = qp[1];
	
	// read qm_y from external memory at location y-1
	int offset = elemOffsetXY-pitch; // this a 2D index
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_y[iVar] = d_qm_y[offset];
	  offset += arraySize2D;
	}
	
	// watchout swap IU and IV
	swap_value_(qleft_y[IU],qleft_y[IV]);
	swap_value_(qleft_y[IA],qleft_y[IB]);
	swap_value_(qright_y[IU],qright_y[IV]);
	swap_value_(qright_y[IA],qright_y[IB]);
	riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);
	
	/* shear correction on flux_y */
	if (/* cartesian */ ::gParams.Omega0 > 0 /* and not fargo */) {
	  real_t shear_y = -1.5 * ::gParams.Omega0 * xPos;
	  real_t eMag, eKin, eTot;
	  real_t bn_mean = HALF_F * (qleft_y[IA] + qright_y[IA]);
	  real_t &gamma  = ::gParams.gamma0;
	  
	  if (shear_y > 0) {
	    eMag = HALF_F * (qleft_y[IA]*qleft_y[IA] + qleft_y[IB]*qleft_y[IB] + qleft_y[IC]*qleft_y[IC]);
	    eKin = HALF_F * (qleft_y[IU]*qleft_y[IU] + qleft_y[IV]*qleft_y[IV] + qleft_y[IW]*qleft_y[IW]);
	    eTot = eKin + eMag + qleft_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qleft_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qleft_y[ID]*qleft_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qleft_y[ID]*qleft_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qleft_y[ID]*qleft_y[IW];
	  } else {
	    eMag = HALF_F * (qright_y[IA]*qright_y[IA] + qright_y[IB]*qright_y[IB] + qright_y[IC]*qright_y[IC]);
	    eKin = HALF_F * (qright_y[IU]*qright_y[IU] + qright_y[IV]*qright_y[IV] + qright_y[IW]*qright_y[IW]);
	    eTot = eKin + eMag + qright_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qright_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qright_y[ID]*qright_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qright_y[ID]*qright_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qright_y[ID]*qright_y[IW];		  
	  }
	} // end shear correction on flux_y

      }
    __syncthreads();
    
    // update uOut with flux_y
    if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V1-3 and
       j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V1-3 and
       k >= 2 and k < kmax-2)
      {
	// watchout IU and IV are swapped !
	uOut[ID] += (flux_y[tx][ty][ID]-flux_y[tx][ty+1][ID])*dtdy;
	uOut[IP] += (flux_y[tx][ty][IP]-flux_y[tx][ty+1][IP])*dtdy;
	uOut[IU] += (flux_y[tx][ty][IV]-flux_y[tx][ty+1][IV])*dtdy;
	uOut[IV] += (flux_y[tx][ty][IU]-flux_y[tx][ty+1][IU])*dtdy;
	uOut[IW] += (flux_y[tx][ty][IW]-flux_y[tx][ty+1][IW])*dtdy;
      }
    
    /*
     * Compute fluxes at Z-interfaces.
     */
    real_t flux_z[NVAR_MHD];
    flux_z[ID] = ZERO_F;
    flux_z[IP] = ZERO_F;
    flux_z[IU] = ZERO_F;
    flux_z[IV] = ZERO_F;
    flux_z[IW] = ZERO_F;
    flux_z[IA] = ZERO_F;
    flux_z[IB] = ZERO_F;
    flux_z[IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V1-2 and
       j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V1-2 and
       k >= 2 and k < kmax-2)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_riemann_t   qleft_z  [NVAR_MHD];
	real_riemann_t (&qright_z)[NVAR_MHD] = qp[2];
	
	// read qm_z from external memory at location z-1
	int offset2 = elemOffsetXY-pitch + iZprevious*arraySize2D;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_z[iVar] = d_qm_z[offset2];
	  offset2 += arraySize2D*2; // because there are 2 planes
	}
	
	// watchout swap IU and IW
	swap_value_(qleft_z[IU] ,qleft_z[IW]);
	swap_value_(qleft_z[IA] ,qleft_z[IC]);
	swap_value_(qright_z[IU],qright_z[IW]);
	swap_value_(qright_z[IA],qright_z[IC]);
	riemann_mhd(qleft_z, qright_z, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i >= 2 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V1-3 and
       j >= 2 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V1-3 and
       k >= 2 and k < kmax-2)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	// watchout IU and IW are swapped !
	uOut[ID] += (flux_z[ID])*dtdz;
	uOut[IP] += (flux_z[IP])*dtdz;
	uOut[IU] += (flux_z[IW])*dtdz;
	uOut[IV] += (flux_z[IV])*dtdz;
	uOut[IW] += (flux_z[IU])*dtdz;

	// actually perform the update on external device memory
	int offset = elemOffset;
	Uout[offset] = uOut[ID];  offset += arraySize;
	Uout[offset] = uOut[IP];  offset += arraySize;
	Uout[offset] = uOut[IU];  offset += arraySize;
	Uout[offset] = uOut[IV];  offset += arraySize;
	Uout[offset] = uOut[IW];

    	/*
    	 * update at position z-1.
	 * Note that position z-1 has already been partialy updated in
	 * the previous iteration (for loop over k).
    	 */
    	// watchout! IU and IW are swapped !
	offset = elemOffset - pitch*jmax;
    	Uout[offset] -= (flux_z[ID])*dtdz; offset += arraySize;
    	Uout[offset] -= (flux_z[IP])*dtdz; offset += arraySize;
    	Uout[offset] -= (flux_z[IW])*dtdz; offset += arraySize;
    	Uout[offset] -= (flux_z[IV])*dtdz; offset += arraySize;
    	Uout[offset] -= (flux_z[IU])*dtdz;
      } // end update along Z
    __syncthreads();
    
    /*
     * EMF computations and update face-centered magnetic field components.
     */
    // re-use q[low] as emf
    real_t (&emf)[UNSPLIT_BLOCK_DIMX_3D_V1][UNSPLIT_BLOCK_DIMY_3D_V1][NVAR_MHD] = q[low];
    emf[tx][ty][IX] = ZERO_F; // emfX
    emf[tx][ty][IY] = ZERO_F; // emfY
    emf[tx][ty][IZ] = ZERO_F; // emfZ

    if(i > 1 and i < imax-2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V1-2 and
       j > 1 and j < jmax-2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V1-2 and
       k > 1 and k < kmax-2)
      {
	
	/*
	 * offset into external memory array to qEdge data
	 */

	int offset2         = elemOffsetXY + iZcurrent  * arraySize2D;
	int offset2_zMinus1 = elemOffsetXY + iZprevious * arraySize2D;
 
	
	/*
	 * compute emfZ
	 */

	// qEdge_RT3 at location x-1, y-1
	int offset = offset2-1-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT3[iVar] = d_qEdge_RT3[offset];
	  offset += arraySize2D*2;
	}
	
	// qEdge RB3 at location x-1, y
	offset = offset2-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB3[iVar] = d_qEdge_RB3[offset];
	  offset += arraySize2D*2;
	}
	
	// qEdge_LT3 at locate x, y-1
	offset = offset2-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT3[iVar] = d_qEdge_LT3[offset];
	  offset += arraySize2D*2;
	}
	
	// finally compute emfZ
	emf[tx][ty][IZ] = compute_emf<EMFZ>(qEdge_emfZ,xPos);

	/*
	 * compute emfY (take care RB and LT are swapped)
	 */

	// qEdge_RT2 at location x-1, z-1
	offset = offset2_zMinus1 - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT2[iVar] = d_qEdge_RT2[offset];
	  offset += arraySize2D*2;
	}
	
	// qEdge RB2 (actually LT2) at location x, z-1
	offset = offset2_zMinus1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB2[iVar] = d_qEdge_LT2[offset];
	  offset += arraySize2D*2;
	}
	
	// qEdge_LT2 (actually RB2) at location x-1, z
	offset = offset2-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT2[iVar] = d_qEdge_RB2[offset];
	  offset += arraySize2D*2;
	}
	
	// finally compute emfY
	emf[tx][ty][IY] = compute_emf<EMFY>(qEdge_emfY,xPos);

	/*
	 * compute emfX
	 */
	// qEdge_RT at location y-1, z-1
	offset = offset2_zMinus1 - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT[iVar] = d_qEdge_RT[offset];
	  offset += arraySize2D*2;
	}
	
	// qEdge RB at location y-1, z
	offset = offset2 - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB[iVar] = d_qEdge_RB[offset];
	  offset += arraySize2D*2;
	}
	
	// qEdge_LT at location y, z-1
	offset = offset2_zMinus1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT[iVar] = d_qEdge_LT[offset];
	  offset += arraySize2D*2;
	}
	
	// finally compute emfX
	emf[tx][ty][IX] = compute_emf<EMFX>(qEdge_emfX,xPos);
	
      }
    __syncthreads();

   if(i >= 3 and i < imax-3 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V1-3 and
      j >= 3 and j < jmax-3 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V1-3 and
      k >= 3 and k < kmax-3)
    {
      // actually perform update on external device memory
      
      // First : update at current location x,y,z
      int offset = elemOffset + 5 * arraySize;
 
      // update bx
      Uout[offset] -= (emf[tx  ][ty  ][IZ]-emf[tx  ][ty+1][IZ])*dtdy;  
      Uout[offset] += (emf[tx  ][ty  ][IY])*dtdz; 
      offset += arraySize;
      
      // update by
      Uout[offset] += (emf[tx  ][ty  ][IZ]-emf[tx+1][ty  ][IZ])*dtdx; 
      Uout[offset] -= (emf[tx  ][ty  ][IX])*dtdz;
      offset += arraySize;

      // update bz
      Uout[offset] += (emf[tx+1][ty  ][IY]-emf[tx  ][ty  ][IY])*dtdx;
      Uout[offset] -= (emf[tx  ][ty+1][IX]-emf[tx  ][ty  ][IX])*dtdy;
    
      // Second : update at z-1 !
      offset = elemOffset - pitch*jmax + 5 * arraySize;
      Uout[offset] -= emf[tx][ty][IY]*dtdz; // update bx
      offset += arraySize;
      Uout[offset] += emf[tx][ty][IX]*dtdz; // update by
    }
   
    
    // take care of swapping d_qm_z plans
    iZcurrent  = 1 - iZcurrent;
    iZprevious = 1 - iZprevious;
    
    // swap planes stored in shared memory
    tmp = low;
    low = current;
    current = mid;
    mid = top;
    top = tmp;
    __syncthreads();
    
    /*
     * load new data (located in plane at z=k+3) and place them into top plane
     */
    if (k<kmax-3) {
      if(i >= 0 and i < imax-1 and 
	 j >= 0 and j < jmax-1)
	{
	  
	  // Gather conservative variables
	  int offset = elemOffset + 3*pitch*jmax;
	  uIn[ID] = Uin[offset];  offset += arraySize;
	  uIn[IP] = Uin[offset];  offset += arraySize;
	  uIn[IU] = Uin[offset];  offset += arraySize;
	  uIn[IV] = Uin[offset];  offset += arraySize;
	  uIn[IW] = Uin[offset];  offset += arraySize;
	  uIn[IA] = Uin[offset];  offset += arraySize;
	  uIn[IB] = Uin[offset];  offset += arraySize;
	  uIn[IC] = Uin[offset];

	  // set bf (face-centered magnetic field components)
	  bf[top][tx][ty][0] = uIn[IA];
	  bf[top][tx][ty][1] = uIn[IB];
	  bf[top][tx][ty][2] = uIn[IC];
	  
	  // go to magnetic field components and get values from 
	  // neighbors on the right
	  real_t magFieldNeighbors[3];
	  offset = elemOffset + 3*pitch*jmax + 5*arraySize;
	  magFieldNeighbors[IX] = Uin[offset+1         ];  offset += arraySize;
	  magFieldNeighbors[IY] = Uin[offset+pitch     ];  offset += arraySize;
	  magFieldNeighbors[IZ] = Uin[offset+pitch*jmax];
	  
	  //Convert to primitive variables
	  constoprim_mhd(uIn, magFieldNeighbors, q[top][tx][ty], c, dt);
	}
    }
    __syncthreads();
  
  } // end for k
    
} // kernel_godunov_unsplit_mhd_3d_v1

/************************************************************
 * Define some CUDA kernel to implement MHD version 3 on GPU
 ************************************************************/

#ifdef USE_DOUBLE
#define ELEC_FIELD_BLOCK_DIMX_3D_V3	16
#define ELEC_FIELD_BLOCK_INNER_DIMX_3D_V3	(ELEC_FIELD_BLOCK_DIMX_3D_V3-1)
#define ELEC_FIELD_BLOCK_DIMY_3D_V3	11
#define ELEC_FIELD_BLOCK_INNER_DIMY_3D_V3	(ELEC_FIELD_BLOCK_DIMY_3D_V3-1)
#else // simple precision
#define ELEC_FIELD_BLOCK_DIMX_3D_V3	16
#define ELEC_FIELD_BLOCK_INNER_DIMX_3D_V3	(ELEC_FIELD_BLOCK_DIMX_3D_V3-1)
#define ELEC_FIELD_BLOCK_DIMY_3D_V3	11
#define ELEC_FIELD_BLOCK_INNER_DIMY_3D_V3	(ELEC_FIELD_BLOCK_DIMY_3D_V3-1)
#endif // USE_DOUBLE

/**
 * Compute electric field components (with rotating frame correction terms).
 *
 * \param[in]  Uin  input  convervative variable array 
 * \param[in]  Qin output primitive variable array
 * \param[out] Elec output electric field
 */
__global__ void kernel_mhd_compute_elec_field(const real_t * __restrict__ Uin,
					      const real_t * __restrict__ Qin,
					      real_t       *Elec,
					      int pitch, 
					      int imax, 
					      int jmax,
					      int kmax,
					      real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, ELEC_FIELD_BLOCK_INNER_DIMX_3D_V3) + tx;
  const int j = __mul24(by, ELEC_FIELD_BLOCK_INNER_DIMY_3D_V3) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // primitive variables in shared memory
  __shared__ real_t      q[2][ELEC_FIELD_BLOCK_DIMX_3D_V3][ELEC_FIELD_BLOCK_DIMY_3D_V3][NVAR_MHD];
 
  // face-centered mag field
  __shared__ real_t     bf[2][ELEC_FIELD_BLOCK_DIMX_3D_V3][ELEC_FIELD_BLOCK_DIMY_3D_V3][3];

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  // indexes to current Z plane and previous one
  int iZcurrent = 1;
  int iZprevious= 0;

  /*
   * load q (primitive variables) in the 2 first planes
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < 2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1)
      {
	
	// read conservative variables from Qin buffer
	int offset = elemOffset;
	q[k][tx][ty][ID] = Qin[offset];  offset += arraySize;
	q[k][tx][ty][IP] = Qin[offset];  offset += arraySize;
	q[k][tx][ty][IU] = Qin[offset];  offset += arraySize;
	q[k][tx][ty][IV] = Qin[offset];  offset += arraySize;
	q[k][tx][ty][IW] = Qin[offset];  offset += arraySize;
	q[k][tx][ty][IA] = Qin[offset];  offset += arraySize;
	q[k][tx][ty][IB] = Qin[offset];  offset += arraySize;
	q[k][tx][ty][IC] = Qin[offset];
	
	// read bf (face-centered magnetic field components) from Uin buffer
	offset = elemOffset + 5*arraySize;
	bf[k][tx][ty][0] = Uin[offset];  offset += arraySize;
	bf[k][tx][ty][1] = Uin[offset];  offset += arraySize;
	bf[k][tx][ty][2] = Uin[offset];
      }
    __syncthreads();

  } // end for k

  /*
   * loop over k (i.e. z) to compute Electric field, and store results
   * in external memory buffer Elec.
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {

    if(i >= 1 and i < imax-1 and tx > 0 and tx < ELEC_FIELD_BLOCK_DIMX_3D_V3 and
       j >= 1 and j < jmax-1 and ty > 0 and ty < ELEC_FIELD_BLOCK_DIMY_3D_V3)
      {
	
	/*
	 * compute Ex, Ey, Ez
	 */
	real_riemann_t u, v, w, A, B, C;
	real_t tmpElec;
	int offset = elemOffset;
	
	// Ex
	v  = 0;
	v += q[iZprevious][tx  ][ty-1][IV];
	v += q[iZcurrent ][tx  ][ty-1][IV];
	v += q[iZprevious][tx  ][ty  ][IV];
	v += q[iZcurrent ][tx  ][ty  ][IV];
	v *= ONE_FOURTH_F;

	w  = 0;
	w += q[iZprevious][tx  ][ty-1][IW];
	w += q[iZcurrent ][tx  ][ty-1][IW];
	w += q[iZprevious][tx  ][ty  ][IW];
	w += q[iZcurrent ][tx  ][ty  ][IW];
	w *= ONE_FOURTH_F;

	B  = 0;
	B += bf[iZprevious][tx  ][ty  ][IY];
	B += bf[iZcurrent ][tx  ][ty  ][IY];
	B *= HALF_F;

	C  = 0;
	C += bf[iZcurrent ][tx  ][ty-1][IZ];
	C += bf[iZcurrent ][tx  ][ty  ][IZ];
	C *= HALF_F;

	tmpElec = static_cast<real_t>(v*C-w*B); 
	if (/* cartesian and not fargo */ ::gParams.Omega0 > 0) {
	  tmpElec -= 1.5 * ::gParams.Omega0 * xPos * C;
	}
	Elec[offset] = tmpElec;
	offset += arraySize;
	
	// Ey
	u  = 0;
	u += q[iZprevious][tx-1][ty  ][IU];
	u += q[iZcurrent ][tx-1][ty  ][IU];
	u += q[iZprevious][tx  ][ty  ][IU];
	u += q[iZcurrent ][tx  ][ty  ][IU];
	u *= ONE_FOURTH_F;
	
	w  = 0;
	w += q[iZprevious][tx-1][ty  ][IW];
	w += q[iZcurrent ][tx-1][ty  ][IW];
	w += q[iZprevious][tx  ][ty  ][IW];
	w += q[iZcurrent ][tx  ][ty  ][IW];
	w *= ONE_FOURTH_F;

	A  = 0;
	A += bf[iZprevious][tx  ][ty  ][IX];
	A += bf[iZcurrent ][tx  ][ty  ][IX];
	A *= HALF_F;
	
	C  = 0;
	C += bf[iZcurrent ][tx-1][ty  ][IZ];
	C += bf[iZcurrent ][tx  ][ty  ][IZ];
	C *= HALF_F;

	Elec[offset] = static_cast<real_t>(w*A-u*C); offset += arraySize;
		
	// Ez
	u = 0;
	u += q[iZcurrent ][tx-1][ty-1][IU];
	u += q[iZcurrent ][tx-1][ty  ][IU];
	u += q[iZcurrent ][tx  ][ty-1][IU];
	u += q[iZcurrent ][tx  ][ty  ][IU];
	u *= ONE_FOURTH_F;
	
	v  = 0;
	v += q[iZcurrent ][tx-1][ty-1][IV];
	v += q[iZcurrent ][tx-1][ty  ][IV];
	v += q[iZcurrent ][tx  ][ty-1][IV];
	v += q[iZcurrent ][tx  ][ty  ][IV];
	v *= ONE_FOURTH_F;
	
	A  = 0;
	A += bf[iZcurrent ][tx  ][ty-1][IX];
	A += bf[iZcurrent ][tx  ][ty  ][IX];
	A *= HALF_F;

	B  = 0;
	B += bf[iZcurrent ][tx-1][ty  ][IY];
	B += bf[iZcurrent ][tx  ][ty  ][IY];
	B *= HALF_F;

	tmpElec = static_cast<real_t>(u*B-v*A);
	if (/* cartesian and not fargo */ ::gParams.Omega0 > 0) {
	  tmpElec += 1.5 * ::gParams.Omega0 * (xPos - ::gParams.dx/2) * A;
	}
	Elec[offset] = tmpElec;
      }
    __syncthreads();
    
    /*
     * erase data in iZprevious (not needed anymore) with primitive variables
     * located at z=k+1
     */
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1 and
       k < kmax-1)
      {
	
	// read conservative variables at z=k+1 from Qin
	int offset = elemOffset + pitch*jmax;
	q[iZprevious][tx][ty][ID] = Qin[offset];  offset += arraySize;
	q[iZprevious][tx][ty][IP] = Qin[offset];  offset += arraySize;
	q[iZprevious][tx][ty][IU] = Qin[offset];  offset += arraySize;
	q[iZprevious][tx][ty][IV] = Qin[offset];  offset += arraySize;
	q[iZprevious][tx][ty][IW] = Qin[offset];  offset += arraySize;
	q[iZprevious][tx][ty][IA] = Qin[offset];  offset += arraySize;
	q[iZprevious][tx][ty][IB] = Qin[offset];  offset += arraySize;
	q[iZprevious][tx][ty][IC] = Qin[offset];
	
	// read bf (face-centered magnetic field components) from Uin
	offset = elemOffset + pitch*jmax + 5*arraySize;
	bf[iZprevious][tx][ty][IX] = Uin[offset]; offset += arraySize;
	bf[iZprevious][tx][ty][IY] = Uin[offset]; offset += arraySize;
	bf[iZprevious][tx][ty][IZ] = Uin[offset];
      }
    __syncthreads();
 
    /*
     * swap iZprevious and iZcurrent
     */
    iZprevious = 1-iZprevious;
    iZcurrent  = 1-iZcurrent;
   __syncthreads();
    
  } // end for k

} // kernel_mhd_compute_elec_field


#ifdef USE_DOUBLE
#define PRIM_ELEC_FIELD_BLOCK_DIMX_3D_V3	16
#define PRIM_ELEC_FIELD_BLOCK_INNER_DIMX_3D_V3	(PRIM_ELEC_FIELD_BLOCK_DIMX_3D_V3-1)
#define PRIM_ELEC_FIELD_BLOCK_DIMY_3D_V3	11
#define PRIM_ELEC_FIELD_BLOCK_INNER_DIMY_3D_V3	(PRIM_ELEC_FIELD_BLOCK_DIMY_3D_V3-1)
#else // simple precision
#define PRIM_ELEC_FIELD_BLOCK_DIMX_3D_V3	16
#define PRIM_ELEC_FIELD_BLOCK_INNER_DIMX_3D_V3	(PRIM_ELEC_FIELD_BLOCK_DIMX_3D_V3-1)
#define PRIM_ELEC_FIELD_BLOCK_DIMY_3D_V3	11
#define PRIM_ELEC_FIELD_BLOCK_INNER_DIMY_3D_V3	(PRIM_ELEC_FIELD_BLOCK_DIMY_3D_V3-1)
#endif // USE_DOUBLE

/**
 * Compute primitive variables and electric field components
 *
 * \param[in]  Uin  input  convervative variable array 
 * \param[out] Qout output primitive variable array
 * \param[out] Elec output electric field
 */
__global__ void kernel_mhd_compute_prim_elec_field(const real_t * __restrict__ Uin,
						   real_t       *Qout,
						   real_t       *Elec,
						   int pitch, 
						   int imax, 
						   int jmax,
						   int kmax,
						   real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, PRIM_ELEC_FIELD_BLOCK_INNER_DIMX_3D_V3) + tx;
  const int j = __mul24(by, PRIM_ELEC_FIELD_BLOCK_INNER_DIMY_3D_V3) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // primitive variables in shared memory
  __shared__ real_t      q[2][PRIM_ELEC_FIELD_BLOCK_DIMX_3D_V3][PRIM_ELEC_FIELD_BLOCK_DIMY_3D_V3][NVAR_MHD];
 
  // face-centered mag field
  __shared__ real_t     bf[2][PRIM_ELEC_FIELD_BLOCK_DIMX_3D_V3][PRIM_ELEC_FIELD_BLOCK_DIMY_3D_V3][3];

  // conservative variables
  real_t uIn [NVAR_MHD];
  real_t c;

  // indexes to current Z plane and previous one
  int iZcurrent = 1;
  int iZprevious= 0;

  /*
   * initialize q (primitive variables ) in the 2 first planes
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < 2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1)
      {
	
	// Gather conservative variables
	int offset = elemOffset;
	uIn[ID] = Uin[offset];  offset += arraySize;
	uIn[IP] = Uin[offset];  offset += arraySize;
	uIn[IU] = Uin[offset];  offset += arraySize;
	uIn[IV] = Uin[offset];  offset += arraySize;
	uIn[IW] = Uin[offset];  offset += arraySize;
	uIn[IA] = Uin[offset];  offset += arraySize;
	uIn[IB] = Uin[offset];  offset += arraySize;
	uIn[IC] = Uin[offset];
	
	// set bf (face-centered magnetic field components)
	bf[k][tx][ty][0] = uIn[IA];
	bf[k][tx][ty][1] = uIn[IB];
	bf[k][tx][ty][2] = uIn[IC];
	
	// go to magnetic field components and get values from 
	// neighbors on the right
	real_t magFieldNeighbors[3];
	offset = elemOffset + 5 * arraySize;
	magFieldNeighbors[IX] = Uin[offset+1         ];  offset += arraySize;
	magFieldNeighbors[IY] = Uin[offset+pitch     ];  offset += arraySize;
	magFieldNeighbors[IZ] = Uin[offset+pitch*jmax];
	
	//Convert to primitive variables
	constoprim_mhd(uIn, magFieldNeighbors, q[k][tx][ty], c, dt);

	// copy results into output
	offset = elemOffset;
	Qout[offset] = q[k][tx][ty][ID]; offset += arraySize;
	Qout[offset] = q[k][tx][ty][IP]; offset += arraySize;
	Qout[offset] = q[k][tx][ty][IU]; offset += arraySize;
	Qout[offset] = q[k][tx][ty][IV]; offset += arraySize;
	Qout[offset] = q[k][tx][ty][IW]; offset += arraySize;
	Qout[offset] = q[k][tx][ty][IA]; offset += arraySize;
	Qout[offset] = q[k][tx][ty][IB]; offset += arraySize;
	Qout[offset] = q[k][tx][ty][IC];

      }
    __syncthreads();

  } // end for k

  /*
   * loop over k (i.e. z) to compute Electric field, and store results
   * in external memory buffer Elec.
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {

    if(i >= 1 and i < imax-1 and tx > 0 and tx < PRIM_ELEC_FIELD_BLOCK_DIMX_3D_V3 and
       j >= 1 and j < jmax-1 and ty > 0 and ty < PRIM_ELEC_FIELD_BLOCK_DIMY_3D_V3)
      {
	
	/*
	 * compute Ex, Ey, Ez
	 */
	real_t u, v, w, A, B, C;
	int offset = elemOffset;
	
	// Ex
	v = ONE_FOURTH_F * (  q[iZprevious][tx  ][ty-1][IV] +
		      q[iZcurrent ][tx  ][ty-1][IV] + 
		      q[iZprevious][tx  ][ty  ][IV] +
		      q[iZcurrent ][tx  ][ty  ][IV] );
	
	w = ONE_FOURTH_F * (  q[iZprevious][tx  ][ty-1][IW] +
		      q[iZcurrent ][tx  ][ty-1][IW] + 
		      q[iZprevious][tx  ][ty  ][IW] +
		      q[iZcurrent ][tx  ][ty  ][IW] );
	
	B = HALF_F  * ( bf[iZprevious][tx  ][ty  ][IY] +
			bf[iZcurrent ][tx  ][ty  ][IY] );
	
	C = HALF_F  * ( bf[iZcurrent ][tx  ][ty-1][IZ] +
			bf[iZcurrent ][tx  ][ty  ][IZ] );
	
	Elec[offset] = v*C-w*B; offset += arraySize;
	
	// Ey
	u = ONE_FOURTH_F * (  q[iZprevious][tx-1][ty  ][IU] +
		       q[iZcurrent ][tx-1][ty  ][IU] + 
		       q[iZprevious][tx  ][ty  ][IU] +
		       q[iZcurrent ][tx  ][ty  ][IU] );
	
	w = ONE_FOURTH_F * (  q[iZprevious][tx-1][ty  ][IW] +
		       q[iZcurrent ][tx-1][ty  ][IW] + 
		       q[iZprevious][tx  ][ty  ][IW] +
		       q[iZcurrent ][tx  ][ty  ][IW] );
	
	A = HALF_F  * ( bf[iZprevious][tx  ][ty  ][IX] +
			bf[iZcurrent ][tx  ][ty  ][IX] );
	
	C = HALF_F  * ( bf[iZcurrent ][tx-1][ty  ][IZ] +
			bf[iZcurrent ][tx  ][ty  ][IZ] );
	
	Elec[offset] = w*A-u*C; offset += arraySize;
	
	// Ez
	u = ONE_FOURTH_F * (  q[iZcurrent ][tx-1][ty-1][IU] +
		       q[iZcurrent ][tx-1][ty  ][IU] + 
		       q[iZcurrent ][tx  ][ty-1][IU] +
		       q[iZcurrent ][tx  ][ty  ][IU] );
	
	v = ONE_FOURTH_F * (  q[iZcurrent ][tx-1][ty-1][IV] +
		       q[iZcurrent ][tx-1][ty  ][IV] + 
		       q[iZcurrent ][tx  ][ty-1][IV] +
		       q[iZcurrent ][tx  ][ty  ][IV] );
	
	A = HALF_F  * ( bf[iZcurrent ][tx  ][ty-1][IX] +
			bf[iZcurrent ][tx  ][ty  ][IX] );
	
	B = HALF_F  * ( bf[iZcurrent ][tx-1][ty  ][IY] +
			bf[iZcurrent ][tx  ][ty  ][IY] );
	
	Elec[offset] = u*B-v*A;
	
      }
    __syncthreads();

    /*
     * erase data in iZprevious (not needed anymore) with primitive variables
     * located at z=k+1
     */
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1 and
       k < kmax-1)
      {
	
	// Gather conservative variables (at z=k+1)
	int offset = elemOffset + pitch*jmax;
	uIn[ID] = Uin[offset];  offset += arraySize;
	uIn[IP] = Uin[offset];  offset += arraySize;
	uIn[IU] = Uin[offset];  offset += arraySize;
	uIn[IV] = Uin[offset];  offset += arraySize;
	uIn[IW] = Uin[offset];  offset += arraySize;
	uIn[IA] = Uin[offset];  offset += arraySize;
	uIn[IB] = Uin[offset];  offset += arraySize;
	uIn[IC] = Uin[offset];
	
	// set bf (face-centered magnetic field components)
	bf[iZprevious][tx][ty][IX] = uIn[IA];
	bf[iZprevious][tx][ty][IY] = uIn[IB];
	bf[iZprevious][tx][ty][IZ] = uIn[IC];
	
	// go to magnetic field components and get values from 
	// neighbors on the right
	real_t magFieldNeighbors[3];
	offset = elemOffset + pitch*jmax + 5 * arraySize;
	magFieldNeighbors[IX] = Uin[offset+1         ];  offset += arraySize;
	magFieldNeighbors[IY] = Uin[offset+pitch     ];  offset += arraySize;
	magFieldNeighbors[IZ] = Uin[offset+pitch*jmax];
	
	//Convert to primitive variables
	constoprim_mhd(uIn, magFieldNeighbors, q[iZprevious][tx][ty], c, dt);

	// copy results into output d_Q at z=k+1
	offset = elemOffset + pitch*jmax;
	Qout[offset] = q[iZprevious][tx][ty][ID]; offset += arraySize;
	Qout[offset] = q[iZprevious][tx][ty][IP]; offset += arraySize;
	Qout[offset] = q[iZprevious][tx][ty][IU]; offset += arraySize;
	Qout[offset] = q[iZprevious][tx][ty][IV]; offset += arraySize;
	Qout[offset] = q[iZprevious][tx][ty][IW]; offset += arraySize;
	Qout[offset] = q[iZprevious][tx][ty][IA]; offset += arraySize;
	Qout[offset] = q[iZprevious][tx][ty][IB]; offset += arraySize;
	Qout[offset] = q[iZprevious][tx][ty][IC];
      }
    __syncthreads();
 
    /*
     * swap iZprevious and iZcurrent
     */
    iZprevious = 1-iZprevious;
    iZcurrent  = 1-iZcurrent;
   __syncthreads();
    
  } // end for k

} // kernel_mhd_compute_prim_elec_field


#ifdef USE_DOUBLE
#define MAG_SLOPES_BLOCK_DIMX_3D_V3	16
#define MAG_SLOPES_BLOCK_INNER_DIMX_3D_V3	(MAG_SLOPES_BLOCK_DIMX_3D_V3-2)
#define MAG_SLOPES_BLOCK_DIMY_3D_V3	24
#define MAG_SLOPES_BLOCK_INNER_DIMY_3D_V3	(MAG_SLOPES_BLOCK_DIMY_3D_V3-2)
#else // simple precision
#define MAG_SLOPES_BLOCK_DIMX_3D_V3	16
#define MAG_SLOPES_BLOCK_INNER_DIMX_3D_V3	(MAG_SLOPES_BLOCK_DIMX_3D_V3-2)
#define MAG_SLOPES_BLOCK_DIMY_3D_V3	24
#define MAG_SLOPES_BLOCK_INNER_DIMY_3D_V3	(MAG_SLOPES_BLOCK_DIMY_3D_V3-2)
#endif // USE_DOUBLE

/**
 * Compute magnetic slopes.
 *
 * This kernel warps call to slope_unsplit_mhd_3d routine.
 *
 * \param[in] Uin pointer to main data (hydro variables + magnetic components)
 * \param[out] dA pointer to array of magnetic slopes delta(Bx)
 * \param[out] dB pointer to array of magnetic slopes delta(By)
 * \param[out] dC pointer to array of magnetic slopes delta(Bz)
 * \param[in] pitch
 * \param[in] imax
 * \param[in] jmax
 * \param[in] kmax
 */
__global__ void kernel_mhd_compute_mag_slopes(const real_t * __restrict__ Uin,
					      real_t       *dA,
					      real_t       *dB,
					      real_t       *dC,
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
  
  const int i = __mul24(bx, MAG_SLOPES_BLOCK_INNER_DIMX_3D_V3) + tx;
  const int j = __mul24(by, MAG_SLOPES_BLOCK_INNER_DIMY_3D_V3) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // face-centered mag field (3 planes - 3 components : Bx,By,Bz)
  __shared__ real_t     bf[3][MAG_SLOPES_BLOCK_DIMX_3D_V3][MAG_SLOPES_BLOCK_DIMY_3D_V3][3];


  /*
   * initialize bf (face-centered mag field) in the 3 first planes
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < 3; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax)
      {
	
	// set bf (face-centered magnetic field components)
	int offset = elemOffset + 5*arraySize;
	bf[k][tx][ty][IX] = Uin[offset];  offset += arraySize;
	bf[k][tx][ty][IY] = Uin[offset];  offset += arraySize;
	bf[k][tx][ty][IZ] = Uin[offset];
	
      }

  } // end for k
  __syncthreads();
  
  
  /*
   * loop over k (i.e. z) to compute magnetic slopes, and store results
   * in external memory buffer dA, dB and dC.
   */
  real_t bfSlopes[15];
  real_t dbfSlopes[3][3];
  
  real_t (&dbfX)[3] = dbfSlopes[IX];
  real_t (&dbfY)[3] = dbfSlopes[IY];
  real_t (&dbfZ)[3] = dbfSlopes[IZ];
  
  int low=0;
  int cur=1;
  int top=2;
  int tmp;

  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {

    if(i >= 1 and i < imax-1 and tx > 0 and tx < MAG_SLOPES_BLOCK_DIMX_3D_V3-1 and
       j >= 1 and j < jmax-1 and ty > 0 and ty < MAG_SLOPES_BLOCK_DIMY_3D_V3-1)
      {

	// get magnetic slopes dbf
	bfSlopes[0]  = bf[cur][tx  ][ty  ][IX];
	bfSlopes[1]  = bf[cur][tx  ][ty+1][IX];
	bfSlopes[2]  = bf[cur][tx  ][ty-1][IX];
	bfSlopes[3]  = bf[top][tx  ][ty  ][IX];
	bfSlopes[4]  = bf[low][tx  ][ty  ][IX];
	
	bfSlopes[5]  = bf[cur][tx  ][ty  ][IY];
	bfSlopes[6]  = bf[cur][tx+1][ty  ][IY];
	bfSlopes[7]  = bf[cur][tx-1][ty  ][IY];
	bfSlopes[8]  = bf[top][tx  ][ty  ][IY];
	bfSlopes[9]  = bf[low][tx  ][ty  ][IY];
	
	bfSlopes[10] = bf[cur][tx  ][ty  ][IZ];
	bfSlopes[11] = bf[cur][tx+1][ty  ][IZ];
	bfSlopes[12] = bf[cur][tx-1][ty  ][IZ];
	bfSlopes[13] = bf[cur][tx  ][ty+1][IZ];
	bfSlopes[14] = bf[cur][tx  ][ty-1][IZ];
	
	// compute magnetic slopes
	slope_unsplit_mhd_3d(bfSlopes, dbfSlopes);
	
	// store magnetic slopes
	int offset = elemOffset;
	dA[offset] = dbfX[IX]; offset += arraySize;
	dA[offset] = dbfY[IX]; offset += arraySize;
	dA[offset] = dbfZ[IX];

	offset = elemOffset;
	dB[offset] = dbfX[IY]; offset += arraySize;
	dB[offset] = dbfY[IY]; offset += arraySize;
	dB[offset] = dbfZ[IY];
	
	offset = elemOffset;
	dC[offset] = dbfX[IZ]; offset += arraySize;
	dC[offset] = dbfY[IZ]; offset += arraySize;
	dC[offset] = dbfZ[IZ];
	
      }

    /*
     * erase data in low plane (not needed anymore) with data at z=k+2
     */
    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax)
      {

	// set bf (face-centered magnetic field components)
	int offset = elemOffset + 2*pitch*jmax + 5*arraySize;
	bf[low][tx][ty][IX] = Uin[offset];  offset += arraySize;
	bf[low][tx][ty][IY] = Uin[offset];  offset += arraySize;
	bf[low][tx][ty][IZ] = Uin[offset];
	
      }
    //__syncthreads();
    
    /*
     * permute low, cur and top
     */
    tmp = low;
    low = cur;
    cur = top;
    top = tmp;
   __syncthreads();

  } // end for k

} // kernel_mhd_compute_mag_slopes


#ifdef USE_DOUBLE
#define TRACE_BLOCK_DIMX_3D_V3	16
#define TRACE_BLOCK_INNER_DIMX_3D_V3	(TRACE_BLOCK_DIMX_3D_V3-4)
#define TRACE_BLOCK_DIMY_3D_V3	15
#define TRACE_BLOCK_INNER_DIMY_3D_V3	(TRACE_BLOCK_DIMY_3D_V3-4)
#else // simple precision
#define TRACE_BLOCK_DIMX_3D_V3	16
#define TRACE_BLOCK_INNER_DIMX_3D_V3	(TRACE_BLOCK_DIMX_3D_V3-4)
#define TRACE_BLOCK_DIMY_3D_V3	15
#define TRACE_BLOCK_INNER_DIMY_3D_V3	(TRACE_BLOCK_DIMY_3D_V3-4)
#endif // USE_DOUBLE

/**
 * Compute trace
 *
 * \param[in]  Uin  input MHD conservative variable array
 * \param[out] Uout ouput MHD conservative variable array
 * \param[in] d_Q input primitive variable array
 * \param[in] dA  input mag slopes (Bx)
 * \param[in] dB  input mag slopes (By)
 * \param[in] dC  input mag slopes (Bz)
 * \param[in] Elec input electric field (Ex,Ey,Ez)
 * \param[in,out] d_qm_x qm state along x
 * \param[in,out] d_qm_y qm state along y
 * \param[in,out] d_qm_z qm state along z
 * \param[in,out] d_qEdge_RT 
 * \param[in,out] d_qEdge_RB 
 * \param[in,out] d_qEdge_LT 
 * \param[in,out] d_qEdge_RT2
 * \param[in,out] d_qEdge_RB2
 * \param[in,out] d_qEdge_LT2
 * \param[in,out] d_qEdge_RT3
 * \param[in,out] d_qEdge_RB3
 * \param[in,out] d_qEdge_LT3
 *
 */
__global__ void kernel_mhd_compute_trace(const real_t * __restrict__ Uin,
					 real_t *Uout,
					 const real_t * __restrict__ d_Q,
					 const real_t * __restrict__ d_dA, 
					 const real_t * __restrict__ d_dB,
					 const real_t * __restrict__ d_dC,
					 const real_t * __restrict__ Elec,
					 real_t *d_qm_x,
					 real_t *d_qm_y,
					 real_t *d_qm_z,
					 real_t *d_qEdge_RT,
					 real_t *d_qEdge_RB,
					 real_t *d_qEdge_LT,
					 real_t *d_qEdge_RT2,
					 real_t *d_qEdge_RB2,
					 real_t *d_qEdge_LT2,
					 real_t *d_qEdge_RT3,
					 real_t *d_qEdge_RB3,
					 real_t *d_qEdge_LT3,
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
  
  const int i = __mul24(bx, TRACE_BLOCK_INNER_DIMX_3D_V3) + tx;
  const int j = __mul24(by, TRACE_BLOCK_INNER_DIMY_3D_V3) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // face-centered mag field (3 planes)
  __shared__ real_t      q[TRACE_BLOCK_DIMX_3D_V3][TRACE_BLOCK_DIMY_3D_V3][NVAR_MHD];
  __shared__ real_t     bf[TRACE_BLOCK_DIMX_3D_V3][TRACE_BLOCK_DIMY_3D_V3][3];

  // we only stored transverse magnetic slopes
  // for dA -> Bx along Y and Z
  // for dB -> By along X and Z
  __shared__ real_t     dA[TRACE_BLOCK_DIMX_3D_V3][TRACE_BLOCK_DIMY_3D_V3][2];
  __shared__ real_t     dB[TRACE_BLOCK_DIMX_3D_V3][TRACE_BLOCK_DIMY_3D_V3][2];

  // we only store z component of electric field
  __shared__ real_t     shEz[TRACE_BLOCK_DIMX_3D_V3][TRACE_BLOCK_DIMY_3D_V3];

  // qm and qp's are output of the trace step
  real_t qm [THREE_D][NVAR_MHD];
  real_t qp [THREE_D][NVAR_MHD];
  real_t qEdge[4][3][NVAR_MHD];

  // conservative variables
  real_t uIn [NVAR_MHD];
  real_t uOut[NVAR_MHD];
  //real_t c;

  real_t qZplus1 [NVAR_MHD];
  real_t bfZplus1[3];

  real_t qZminus1 [NVAR_MHD];
  //real_t bfZminus1[3];

  //real_t dA_Y_Zplus1;
  //real_t dA_Z_Zplus1;

  real_t dC_X;
  real_t dC_Y;
  real_t dC_X_Zplus1;
  real_t dC_Y_Zplus1;

  real_t Ex_j_k;
  real_t Ex_j_k1;
  real_t Ex_j1_k;
  real_t Ex_j1_k1;

  real_t Ey_i_k;
  real_t Ey_i_k1;
  real_t Ey_i1_k;
  real_t Ey_i1_k1;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  /*
   * initialize q (primitive variables ) in the 2 first XY-planes
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < 2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1)
      {
	
	int offset = elemOffset;

	// read primitive variables from d_Q
	if (k==0) {
	  qZminus1[ID] = d_Q[offset];  offset += arraySize;
	  qZminus1[IP] = d_Q[offset];  offset += arraySize;
	  qZminus1[IU] = d_Q[offset];  offset += arraySize;
	  qZminus1[IV] = d_Q[offset];  offset += arraySize;
	  qZminus1[IW] = d_Q[offset];  offset += arraySize;
	  qZminus1[IA] = d_Q[offset];  offset += arraySize;
	  qZminus1[IB] = d_Q[offset];  offset += arraySize;
	  qZminus1[IC] = d_Q[offset];
	} else { // k == 1
	  q[tx][ty][ID] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IP] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IU] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IV] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IW] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IA] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IB] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IC] = d_Q[offset];
	}


	// read face-centered magnetic field from Uin
	offset = elemOffset + 5 * arraySize;
	uIn[IA] = Uin[offset];  offset += arraySize;
	uIn[IB] = Uin[offset];  offset += arraySize;
	uIn[IC] = Uin[offset];
	
	// set bf (face-centered magnetic field components)
	if (k==0) {
	  //bfZminus1[0] = uIn[IA];
	  //bfZminus1[1] = uIn[IB];
	  //bfZminus1[2] = uIn[IC];
	} else { // k == 1
	  bf[tx][ty][0] = uIn[IA];
	  bf[tx][ty][1] = uIn[IB];
	  bf[tx][ty][2] = uIn[IC];
	}
		
	// read magnetic slopes dC
	offset = elemOffset;
	dC_X = d_dC[offset]; offset += arraySize;
	dC_Y = d_dC[offset];
	
	// read electric field Ex (at i,j,k and i,j+1,k)
	offset = elemOffset;
	Ex_j_k  = Elec[offset];
	Ex_j1_k = Elec[offset+pitch]; 
	
	// read electric field Ey (at i,j,k and i+1,j,k)
	offset += arraySize;
	Ey_i_k  = Elec[offset];
	Ey_i1_k = Elec[offset+1];
      }
    __syncthreads();
  
  } // end for k

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    // data fetch :
    // get q, bf at z+1
    // get dA, dB at z+1
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1)
      {
	 
	 // read primitive variables at z+1
	 int offset = elemOffset + pitch*jmax; // z+1	 
	 qZplus1[ID] = d_Q[offset];  offset += arraySize;
	 qZplus1[IP] = d_Q[offset];  offset += arraySize;
	 qZplus1[IU] = d_Q[offset];  offset += arraySize;
	 qZplus1[IV] = d_Q[offset];  offset += arraySize;
	 qZplus1[IW] = d_Q[offset];  offset += arraySize;
	 qZplus1[IA] = d_Q[offset];  offset += arraySize;
	 qZplus1[IB] = d_Q[offset];  offset += arraySize;
	 qZplus1[IC] = d_Q[offset];
	 
	 // set bf (read face-centered magnetic field components)
	 offset = elemOffset + pitch*jmax + 5 * arraySize;
	 bfZplus1[IX] = Uin[offset]; offset += arraySize;
	 bfZplus1[IY] = Uin[offset]; offset += arraySize;
	 bfZplus1[IZ] = Uin[offset];
	 
	 // get magnetic slopes dA and dB at z=k
	 // read magnetic slopes dA (along Y and Z) 
	 offset = elemOffset+arraySize;
	 dA[tx][ty][0] = d_dA[offset]; offset += arraySize;
	 dA[tx][ty][1] = d_dA[offset];
	 
	 // read magnetic slopes dB (along X and Z)
	 offset = elemOffset;
	 dB[tx][ty][0] = d_dB[offset]; offset += (2*arraySize);
	 dB[tx][ty][1] = d_dB[offset];
	 
	 // get magnetic slopes dC (along X and Y) at z=k+1
	 offset = elemOffset + pitch*jmax;
	 dC_X_Zplus1 = d_dC[offset]; offset += arraySize;
	 dC_Y_Zplus1 = d_dC[offset];

	 // read electric field (Ex at  i,j,k+1 and i,j+1,k+1)
	 offset = elemOffset + pitch*jmax;
	 Ex_j_k1  = Elec[offset];
	 Ex_j1_k1 = Elec[offset+pitch]; 

	 // read electric field Ey (at i,j,k+1 and i+1,j,k+1)
	 offset += arraySize;
	 Ey_i_k1  = Elec[offset];
	 Ey_i1_k1 = Elec[offset+1];

	 // read electric field (Ez into shared memory) at z=k
	 offset = elemOffset + 2 * arraySize;
	 shEz[tx][ty] = Elec[offset];

       }
     __syncthreads();
     

     // slope and trace computation (i.e. dq, and then qm, qp, qEdge)

     real_t qEdge_emfX[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT)[NVAR_MHD] = qEdge_emfX[IRT];
     real_t (&qEdge_RB)[NVAR_MHD] = qEdge_emfX[IRB];
     real_t (&qEdge_LT)[NVAR_MHD] = qEdge_emfX[ILT];
     real_t (&qEdge_LB)[NVAR_MHD] = qEdge_emfX[ILB];
     
     real_t qEdge_emfY[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT2)[NVAR_MHD] = qEdge_emfY[IRT];
     real_t (&qEdge_RB2)[NVAR_MHD] = qEdge_emfY[IRB];
     real_t (&qEdge_LT2)[NVAR_MHD] = qEdge_emfY[ILT];
     real_t (&qEdge_LB2)[NVAR_MHD] = qEdge_emfY[ILB];
     
     real_t qEdge_emfZ[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT3)[NVAR_MHD] = qEdge_emfZ[IRT];
     real_t (&qEdge_RB3)[NVAR_MHD] = qEdge_emfZ[IRB];
     real_t (&qEdge_LT3)[NVAR_MHD] = qEdge_emfZ[ILT];
     real_t (&qEdge_LB3)[NVAR_MHD] = qEdge_emfZ[ILB];

     if(i >= 1 and i < imax-2 and tx > 0 and tx < TRACE_BLOCK_DIMX_3D_V3-1 and
	j >= 1 and j < jmax-2 and ty > 0 and ty < TRACE_BLOCK_DIMY_3D_V3-1)
       {

	 int offset = elemOffset; // 3D index

	 real_t (&qPlusX )[NVAR_MHD] = q[tx+1][ty  ];
	 real_t (&qMinusX)[NVAR_MHD] = q[tx-1][ty  ];
	 real_t (&qPlusY )[NVAR_MHD] = q[tx  ][ty+1];
	 real_t (&qMinusY)[NVAR_MHD] = q[tx  ][ty-1];
	 real_t (&qPlusZ )[NVAR_MHD] = qZplus1;  // q[tx][ty] at z+1
	 real_t (&qMinusZ)[NVAR_MHD] = qZminus1; // q[tx][ty] at z-1
						 // (stored from z
						 // previous
						 // iteration)
	 // hydro slopes  array
	 real_t dq[3][NVAR_MHD];
	 
	 if (::gParams.slope_type==3) {
	   // positivity preserving slope computation
	   const int di = 1;
	   const int dj = pitch;
	   const int dk = pitch*jmax;

	   for (int nVar=0, offsetBase=elemOffset; 
		nVar<NVAR_MHD; 
		++nVar,offsetBase += arraySize) {

	     real_t vmin = FLT_MAX;
	     real_t vmax = -FLT_MAX;

	     // compute vmin,vmax
	     for (int ii=-1; ii<2; ++ii)
	       for (int jj=-1; jj<2; ++jj)
		 for (int kk=-1; kk<2; ++kk) {
		   offset = offsetBase + ii*di + jj*dj + kk*dk;
		   real_t tmp = d_Q[offset] - q[tx][ty][nVar];
		   vmin = FMIN(vmin,tmp);
		   vmax = FMAX(vmax,tmp);
		 }

	     // x+1,y  ,z   - x-1,y  ,z
	     real_t dfx =  HALF_F * ( d_Q[offsetBase+di] - d_Q[offsetBase-di]);
	     // x  ,y+1,z   - x  ,y-1,z
	     real_t dfy =  HALF_F * ( d_Q[offsetBase+dj] - d_Q[offsetBase-dj]);
	     // x  ,y  ,z+1 - x  ,y  ,z-1
	     real_t dfz =  HALF_F * ( d_Q[offsetBase+dk] - d_Q[offsetBase-dk]);

	     real_t dff  = HALF_F * (FABS(dfx) + FABS(dfy) + FABS(dfz));

	     real_t slop=ONE_F;
	     real_t &dlim=slop;
	     if (dff>ZERO_F) {
	       slop = FMIN(ONE_F, FMIN(FABS(vmin), FABS(vmax))/dff);
	     }

	     dq[IX][nVar] = dlim*dfx;
	     dq[IY][nVar] = dlim*dfy;
	     dq[IZ][nVar] = dlim*dfz;

	   } // end for nVar

	 } else {
	   // compute hydro slopes dq
	   slope_unsplit_hydro_3d(q[tx][ty], 
				  qPlusX, qMinusX, 
				  qPlusY, qMinusY, 
				  qPlusZ, qMinusZ,
				  dq);
	 }
	 
	 // get face-centered magnetic components
	 real_t bfNb[6];
	 bfNb[0] = bf[tx  ][ty  ][0];
	 bfNb[1] = bf[tx+1][ty  ][0];
	 bfNb[2] = bf[tx  ][ty  ][1];
	 bfNb[3] = bf[tx  ][ty+1][1];
	 bfNb[4] = bf[tx  ][ty  ][2];
	 bfNb[5] = bfZplus1      [2];

	 // get dbf (transverse magnetic slopes)
	 real_t dbf[12];
	 dbf[0]  = dA[tx][ty][0]; // dA along Y
	 dbf[1]  = dA[tx][ty][1]; // dA along Z
	 dbf[2]  = dB[tx][ty][0]; // dB along X
	 dbf[3]  = dB[tx][ty][1]; // dB along Z
	 dbf[4]  = dC_X;          // dC along X
	 dbf[5]  = dC_Y;          // dC along Y
	 
	 dbf[6]  = dA[tx+1][ty  ][0];
	 dbf[7]  = dA[tx+1][ty  ][1];
	 dbf[8]  = dB[tx  ][ty+1][0];
	 dbf[9]  = dB[tx  ][ty+1][1];
	 dbf[10] = dC_X_Zplus1;
	 dbf[11] = dC_Y_Zplus1;
	 
	 // get electric field components
	 real_t elecFields[3][2][2];
	 // alias to electric field components
	 real_t (&Ex)[2][2] = elecFields[IX];
	 real_t (&Ey)[2][2] = elecFields[IY];
	 real_t (&Ez)[2][2] = elecFields[IZ];
	 
	 Ex[0][0] = Ex_j_k;
	 Ex[0][1] = Ex_j_k1;
	 Ex[1][0] = Ex_j1_k;
	 Ex[1][1] = Ex_j1_k1;
	 
	 Ey[0][0] = Ey_i_k;
	 Ey[0][1] = Ey_i_k1;
	 Ey[1][0] = Ey_i1_k;
	 Ey[1][1] = Ey_i1_k1;
	 
	 Ez[0][0] = shEz[tx  ][ty  ];
	 Ez[0][1] = shEz[tx  ][ty+1];
	 Ez[1][0] = shEz[tx+1][ty  ];
	 Ez[1][1] = shEz[tx+1][ty+1];
	 
	 // compute qm, qp and qEdge
	 trace_unsplit_mhd_3d_simpler(q[tx][ty], dq, bfNb, dbf, elecFields, 
				      dtdx, dtdy, dtdz, xPos,
				      qm, qp, qEdge);
	 
	 
	 // store qm, qEdge in external memory
	 {
	   offset  = elemOffset;
	   for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	     
	     d_qm_x[offset] = qm[0][iVar];
	     d_qm_y[offset] = qm[1][iVar];
	     d_qm_z[offset] = qm[2][iVar];
	     
	     d_qEdge_RT[offset]  = qEdge[IRT][0][iVar];
	     d_qEdge_RB[offset]  = qEdge[IRB][0][iVar];
	     d_qEdge_LT[offset]  = qEdge[ILT][0][iVar];
	     
	     d_qEdge_RT2[offset] = qEdge[IRT][1][iVar];
	     d_qEdge_RB2[offset] = qEdge[IRB][1][iVar];
	     d_qEdge_LT2[offset] = qEdge[ILT][1][iVar];
	     
	     d_qEdge_RT3[offset] = qEdge[IRT][2][iVar];
	     d_qEdge_RB3[offset] = qEdge[IRB][2][iVar];
	     d_qEdge_LT3[offset] = qEdge[ILT][2][iVar];
	     
	     qEdge_LB[iVar]      = qEdge[ILB][0][iVar]; // qEdge_LB  is just a local variable
	     qEdge_LB2[iVar]     = qEdge[ILB][1][iVar]; // qEdge_LB2 is just a local variable
	     qEdge_LB3[iVar]     = qEdge[ILB][2][iVar]; // qEdge_LB3 is just a local variable
	     
	     offset  += arraySize;
	   } // end for iVar
	 } // end store qm, qp, qEdge

       } // end compute trace
     __syncthreads();


     // q  is copied into qZminus1 so that one can temporarily re-use
     // shared mem array q for flux computation !!!
     for (int iVar=0; iVar<NVAR_MHD; iVar++) {
       qZminus1 [iVar] = q[tx][ty][iVar]; // now q can be re-used for flux
     }
     __syncthreads();

    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use q as flux_x
    real_t (&flux_x)[TRACE_BLOCK_DIMX_3D_V3][TRACE_BLOCK_DIMY_3D_V3][NVAR_MHD] = q;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    flux_x[tx][ty][IA] = ZERO_F;
    flux_x[tx][ty][IB] = ZERO_F;
    flux_x[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    // tx and ty > 1 to be sure that qm_x at location x-1 has been computed
    if(i >= 3 and i < imax-2 and tx > 1 and tx < TRACE_BLOCK_DIMX_3D_V3-1 and
       j >= 3 and j < jmax-2 and ty > 1 and ty < TRACE_BLOCK_DIMY_3D_V3-1 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_riemann_t   qleft_x  [NVAR_MHD];
	real_riemann_t (&qright_x)[NVAR_MHD] = qp[0];
	
	// re-read qm_x from external memory at location x-1
	int offset = elemOffset-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_x[iVar] = d_qm_x[offset];
	  offset += arraySize;
	}
	
	riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);
	
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i >= 3 and i < imax-3 and tx > 1 and tx < TRACE_BLOCK_DIMX_3D_V3-2 and
       j >= 3 and j < jmax-3 and ty > 1 and ty < TRACE_BLOCK_DIMY_3D_V3-2 and
       k >= 3 and k < kmax-3)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offset = elemOffset;
	uOut[ID] = Uin[offset];  offset += arraySize;
	uOut[IP] = Uin[offset];  offset += arraySize;
	uOut[IU] = Uin[offset];  offset += arraySize;
	uOut[IV] = Uin[offset];  offset += arraySize;
	uOut[IW] = Uin[offset];  offset += arraySize;
	uOut[IA] = Uin[offset];  offset += arraySize;
	uOut[IB] = Uin[offset];  offset += arraySize;
	uOut[IC] = Uin[offset];

	uOut[ID] += (flux_x[tx][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
	uOut[IP] += (flux_x[tx][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
	uOut[IU] += (flux_x[tx][ty][IU]-flux_x[tx+1][ty][IU])*dtdx;
	uOut[IV] += (flux_x[tx][ty][IV]-flux_x[tx+1][ty][IV])*dtdx;
	uOut[IW] += (flux_x[tx][ty][IW]-flux_x[tx+1][ty][IW])*dtdx;
      }
    __syncthreads();

    /*
     * Compute fluxes at Y-interfaces.
     */
    // re-use q as flux_y
    real_t (&flux_y)[TRACE_BLOCK_DIMX_3D_V3][TRACE_BLOCK_DIMY_3D_V3][NVAR_MHD] = q;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    flux_y[tx][ty][IA] = ZERO_F;
    flux_y[tx][ty][IB] = ZERO_F;
    flux_y[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    // tx and ty > 1 to be sure that qm_y at location y-1 has been computed
    if(i >= 3 and i < imax-2 and tx > 1 and tx < TRACE_BLOCK_DIMX_3D_V3-1 and
       j >= 3 and j < jmax-2 and ty > 1 and ty < TRACE_BLOCK_DIMY_3D_V3-1 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_riemann_t   qleft_y  [NVAR_MHD];
	real_riemann_t (&qright_y)[NVAR_MHD] = qp[1];
	
	// read qm_y from external memory at location y-1
	int offset = elemOffset-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_y[iVar] = d_qm_y[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IV
	swap_value_(qleft_y[IU],qleft_y[IV]);
	swap_value_(qleft_y[IA],qleft_y[IB]);
	swap_value_(qright_y[IU],qright_y[IV]);
	swap_value_(qright_y[IA],qright_y[IB]);
	riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);

	/* shear correction on flux_y */
	if (/* cartesian */ ::gParams.Omega0 > 0 /* and not fargo */) {
	  real_t shear_y = -1.5 * ::gParams.Omega0 * xPos;
	  real_t eMag, eKin, eTot;
	  real_t bn_mean = HALF_F * (qleft_y[IA] + qright_y[IA]);
	  real_t &gamma  = ::gParams.gamma0;
	  
	  if (shear_y > 0) {
	    eMag = HALF_F * (qleft_y[IA]*qleft_y[IA] + qleft_y[IB]*qleft_y[IB] + qleft_y[IC]*qleft_y[IC]);
	    eKin = HALF_F * (qleft_y[IU]*qleft_y[IU] + qleft_y[IV]*qleft_y[IV] + qleft_y[IW]*qleft_y[IW]);
	    eTot = eKin + eMag + qleft_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qleft_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qleft_y[ID]*qleft_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qleft_y[ID]*qleft_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qleft_y[ID]*qleft_y[IW];
	  } else {
	    eMag = HALF_F * (qright_y[IA]*qright_y[IA] + qright_y[IB]*qright_y[IB] + qright_y[IC]*qright_y[IC]);
	    eKin = HALF_F * (qright_y[IU]*qright_y[IU] + qright_y[IV]*qright_y[IV] + qright_y[IW]*qright_y[IW]);
	    eTot = eKin + eMag + qright_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qright_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qright_y[ID]*qright_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qright_y[ID]*qright_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qright_y[ID]*qright_y[IW];		  
	  }
	} // end shear correction on flux_y
	
      }  
    __syncthreads();
    
    // update uOut with flux_y
    if(i >= 3 and i < imax-3 and tx > 1 and tx < TRACE_BLOCK_DIMX_3D_V3-2 and
       j >= 3 and j < jmax-3 and ty > 1 and ty < TRACE_BLOCK_DIMY_3D_V3-2 and
       k >= 3 and k < kmax-3)
      {
	// watchout IU and IV are swapped !
	uOut[ID] += (flux_y[tx][ty][ID]-flux_y[tx][ty+1][ID])*dtdy;
	uOut[IP] += (flux_y[tx][ty][IP]-flux_y[tx][ty+1][IP])*dtdy;
	uOut[IU] += (flux_y[tx][ty][IV]-flux_y[tx][ty+1][IV])*dtdy;
	uOut[IV] += (flux_y[tx][ty][IU]-flux_y[tx][ty+1][IU])*dtdy;
	uOut[IW] += (flux_y[tx][ty][IW]-flux_y[tx][ty+1][IW])*dtdy;
      }
    __syncthreads();
    
    /*
     * Compute fluxes at Z-interfaces.
     */
    real_t flux_z[NVAR_MHD];
    flux_z[ID] = ZERO_F;
    flux_z[IP] = ZERO_F;
    flux_z[IU] = ZERO_F;
    flux_z[IV] = ZERO_F;
    flux_z[IW] = ZERO_F;
    flux_z[IA] = ZERO_F;
    flux_z[IB] = ZERO_F;
    flux_z[IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and tx > 1 and tx < TRACE_BLOCK_DIMX_3D_V3-1 and
       j >= 3 and j < jmax-2 and ty > 1 and ty < TRACE_BLOCK_DIMY_3D_V3-1 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_riemann_t   qleft_z  [NVAR_MHD];
	real_riemann_t (&qright_z)[NVAR_MHD] = qp[2];
	
	// read qm_z from external memory at location z-1
	int offset = elemOffset - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_z[iVar] = d_qm_z[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IW
	swap_value_(qleft_z[IU] ,qleft_z[IW]);
	swap_value_(qleft_z[IA] ,qleft_z[IC]);
	swap_value_(qright_z[IU],qright_z[IW]);
	swap_value_(qright_z[IA],qright_z[IC]);
	riemann_mhd(qleft_z, qright_z, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i >= 3 and i < imax-3 and tx > 1 and tx < TRACE_BLOCK_DIMX_3D_V3-2 and
       j >= 3 and j < jmax-3 and ty > 1 and ty < TRACE_BLOCK_DIMY_3D_V3-2 and
       k >= 3 and k < kmax-2)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	int offset = elemOffset;

	if (k < kmax-3) {
	  // watchout IU and IW are swapped !
	  uOut[ID] += (flux_z[ID])*dtdz;
	  uOut[IP] += (flux_z[IP])*dtdz;
	  uOut[IU] += (flux_z[IW])*dtdz;
	  uOut[IV] += (flux_z[IV])*dtdz;
	  uOut[IW] += (flux_z[IU])*dtdz;
	  
	  // actually perform the update on external device memory
	  Uout[offset] = uOut[ID];  offset += arraySize;
	  Uout[offset] = uOut[IP];  offset += arraySize;
	  Uout[offset] = uOut[IU];  offset += arraySize;
	  Uout[offset] = uOut[IV];  offset += arraySize;
	  Uout[offset] = uOut[IW];
	}

	if (k>3) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offset = elemOffset - pitch*jmax;
	  Uout[offset] -= (flux_z[ID])*dtdz; offset += arraySize;
	  Uout[offset] -= (flux_z[IP])*dtdz; offset += arraySize;
	  Uout[offset] -= (flux_z[IW])*dtdz; offset += arraySize;
	  Uout[offset] -= (flux_z[IV])*dtdz; offset += arraySize;
	  Uout[offset] -= (flux_z[IU])*dtdz;
	}
      } // end update along Z
    __syncthreads();


    /*
     * EMF computations and update face-centered magnetic field components.
     */
    // re-use q as emf
    real_t (&emf)[TRACE_BLOCK_DIMX_3D_V3][TRACE_BLOCK_DIMY_3D_V3][NVAR_MHD] = q;
    emf[tx][ty][IX] = ZERO_F; // emfX
    emf[tx][ty][IY] = ZERO_F; // emfY
    emf[tx][ty][IZ] = ZERO_F; // emfZ

    if(i > 1 and i < imax-2 and tx > 1 and tx < TRACE_BLOCK_DIMX_3D_V3-1 and
       j > 1 and j < jmax-2 and ty > 1 and ty < TRACE_BLOCK_DIMY_3D_V3-1 and
       k > 1 and k < kmax-2)
      {
	
	/*
	 * offset into external memory array to qEdge data
	 */

	int offset2         = elemOffset;
 
	
	/*
	 * compute emfZ
	 */

	// qEdge_RT3 at location x-1, y-1
	int offset = offset2-1-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT3[iVar] = d_qEdge_RT3[offset];
	  offset += arraySize;
	}
	
	// qEdge RB3 at location x-1, y
	offset = offset2-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB3[iVar] = d_qEdge_RB3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT3 at location x, y-1
	offset = offset2-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT3[iVar] = d_qEdge_LT3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB3 at location x, y (already filled above)

	// finally compute emfZ
	emf[tx][ty][IZ] = compute_emf<EMFZ>(qEdge_emfZ,xPos);

	/*
	 * compute emfY (take care RB and LT are swapped)
	 */

	// qEdge_RT2 at location x-1, z-1
	offset = offset2 - pitch*jmax - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT2[iVar] = d_qEdge_RT2[offset];
	  offset += arraySize;
	}
	
	// qEdge RB2 (actually LT2) at location x, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB2[iVar] = d_qEdge_LT2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT2 (actually RB2) at location x-1, z
	offset = offset2 - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT2[iVar] = d_qEdge_RB2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB2 at location x, y (already filled above)

	// finally compute emfY
	emf[tx][ty][IY] = compute_emf<EMFY>(qEdge_emfY,xPos);

	/*
	 * compute emfX
	 */
	// qEdge_RT at location y-1, z-1
	offset = offset2 - pitch*jmax - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT[iVar] = d_qEdge_RT[offset];
	  offset += arraySize;
	}
	
	// qEdge RB at location y-1, z
	offset = offset2 - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB[iVar] = d_qEdge_RB[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT at location y, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT[iVar] = d_qEdge_LT[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB at location x, y (already filled above)

	// finally compute emfX
	emf[tx][ty][IX] = compute_emf<EMFX>(qEdge_emfX,xPos);
	
      }
    __syncthreads();
    
    //emf[tx][ty][IX] = 0;
    //emf[tx][ty][IY] = 0;
    //emf[tx][ty][IZ] = 0;

    if(i >= 3 and i < imax-2 and tx > 1 and tx < TRACE_BLOCK_DIMX_3D_V3-2 and
       j >= 3 and j < jmax-2 and ty > 1 and ty < TRACE_BLOCK_DIMY_3D_V3-2 and
       k >= 3 and k < kmax-2)
      {
	// actually perform update on external device memory
	
	// First : update at current location x,y,z
	int offset;

	if (k<kmax-3) {

	  offset = elemOffset + 5 * arraySize;

	  // update bx
	  Uout[offset] = bf[tx][ty][0]-(emf[tx][ty][IZ]-emf[tx][ty+1][IZ])*dtdy
	    +(emf[tx][ty][IY])*dtdz; 
	  offset += arraySize;
	  
	  // update by
	  Uout[offset] = bf[tx][ty][1]+(emf[tx][ty][IZ]-emf[tx+1][ty][IZ])*dtdx
	    -(emf[tx][ty][IX])*dtdz;
	  offset += arraySize;
	  
	  // update bz
	  Uout[offset] = bf[tx][ty][2]
	    + (emf[tx+1][ty  ][IY]-emf[tx][ty][IY])*dtdx
	    - (emf[tx  ][ty+1][IX]-emf[tx][ty][IX])*dtdy;
	}

	// Second : update at z-1 !
	if (k>3) {
	  offset = elemOffset - pitch*jmax + 5 * arraySize;
	  Uout[offset] -= emf[tx][ty][IY]*dtdz; // update bx
	  offset += arraySize;
	  Uout[offset] += emf[tx][ty][IX]*dtdz; // update by
	}
      }
    __syncthreads();
    
    {
      // qZplus1 and bfZplus1 at z+1       become  q        and bf
      // dC_X_Zplus1                       becomes dC_X
      // dC_Y_Zplus1                       becomes dC_Y
      // update q with data ar z+1
      // rotate bfZminus1 - bf -bfZplus1
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	q[tx][ty][iVar] = qZplus1[iVar];
      }
      //bfZminus1[0] = bf[tx][ty][0];
      //bfZminus1[1] = bf[tx][ty][1];
      //bfZminus1[2] = bf[tx][ty][2];
      
      bf[tx][ty][0] = bfZplus1[0];
      bf[tx][ty][1] = bfZplus1[1];
      bf[tx][ty][2] = bfZplus1[2];
      
      dC_X = dC_X_Zplus1;
      dC_Y = dC_Y_Zplus1;
      
      Ex_j_k  = Ex_j_k1;
      Ex_j1_k = Ex_j1_k1;
      
      Ey_i_k  = Ey_i_k1;
      Ey_i1_k = Ey_i1_k1;
    }
    __syncthreads();
    
  } // end for k
  
} // kernel_mhd_compute_trace (version 3)


#ifdef USE_DOUBLE
# define TRACE_BLOCK_DIMX_3D_V4	16
# define TRACE_BLOCK_INNER_DIMX_3D_V4	(TRACE_BLOCK_DIMX_3D_V4-2)
# define TRACE_BLOCK_DIMY_3D_V4	14
# define TRACE_BLOCK_INNER_DIMY_3D_V4	(TRACE_BLOCK_DIMY_3D_V4-2)
#else // simple precision
# define TRACE_BLOCK_DIMX_3D_V4	16
# define TRACE_BLOCK_INNER_DIMX_3D_V4	(TRACE_BLOCK_DIMX_3D_V4-2)
# define TRACE_BLOCK_DIMY_3D_V4	14
# define TRACE_BLOCK_INNER_DIMY_3D_V4	(TRACE_BLOCK_DIMY_3D_V4-2)
#endif // USE_DOUBLE

/**
 * Compute trace for MHD (implementation version 4).
 *
 * Output are all that is needed to compute fluxes and EMF.
 * \see kernel_mhd_flux_update_hydro_v4
 *
 * All we do here is call :
 * - slope_unsplit_hydro_3d
 * - trace_unsplit_mhd_3d_simpler to get output : qm, qp, qEdge's
 *
 * \param[in] Uin input MHD conservative variable array
 * \param[in] d_Q input primitive variable array
 * \param[in] dA  input mag slopes (Bx)
 * \param[in] dB  input mag slopes (By)
 * \param[in] dC  input mag slopes (Bz)
 * \param[in] Elec input electric field (Ex,Ey,Ez)
 * \param[out] d_qm_x qm state along x
 * \param[out] d_qm_y qm state along y
 * \param[out] d_qm_z qm state along z
 * \param[out] d_qp_x qp state along x
 * \param[out] d_qp_y qp state along y
 * \param[out] d_qp_z qp state along z
 * \param[out] d_qEdge_RT 
 * \param[out] d_qEdge_RB 
 * \param[out] d_qEdge_LT 
 * \param[out] d_qEdge_LB
 *
 * \param[out] d_qEdge_RT2
 * \param[out] d_qEdge_RB2
 * \param[out] d_qEdge_LT2
 * \param[out] d_qEdge_LB2
 *
 * \param[out] d_qEdge_RT3
 * \param[out] d_qEdge_RB3
 * \param[out] d_qEdge_LT3
 * \param[out] d_qEdge_LB3
 *
 * \note It turns out that it is better to have gravity predictor outside.
 */
__global__ void kernel_mhd_compute_trace_v4(const real_t * __restrict__ Uin,
					    const real_t * __restrict__ d_Q,
					    const real_t * __restrict__ d_dA, 
					    const real_t * __restrict__ d_dB,
					    const real_t * __restrict__ d_dC,
					    const real_t * __restrict__ Elec,
					    real_t *d_qm_x,
					    real_t *d_qm_y,
					    real_t *d_qm_z,
					    real_t *d_qp_x,
					    real_t *d_qp_y,
					    real_t *d_qp_z,
					    real_t *d_qEdge_RT,
					    real_t *d_qEdge_RB,
					    real_t *d_qEdge_LT,
					    real_t *d_qEdge_LB,
					    real_t *d_qEdge_RT2,
					    real_t *d_qEdge_RB2,
					    real_t *d_qEdge_LT2,
					    real_t *d_qEdge_LB2,
					    real_t *d_qEdge_RT3,
					    real_t *d_qEdge_RB3,
					    real_t *d_qEdge_LT3,
					    real_t *d_qEdge_LB3,
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
  
  const int i = __mul24(bx, TRACE_BLOCK_INNER_DIMX_3D_V4) + tx;
  const int j = __mul24(by, TRACE_BLOCK_INNER_DIMY_3D_V4) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // face-centered mag field (3 planes)
  __shared__ real_t      q[TRACE_BLOCK_DIMX_3D_V4][TRACE_BLOCK_DIMY_3D_V4][NVAR_MHD];
  __shared__ real_t     bf[TRACE_BLOCK_DIMX_3D_V4][TRACE_BLOCK_DIMY_3D_V4][3];

  // we only stored transverse magnetic slopes
  // for dA -> Bx along Y and Z
  // for dB -> By along X and Z
  __shared__ real_t     dA[TRACE_BLOCK_DIMX_3D_V4][TRACE_BLOCK_DIMY_3D_V4][2];
  __shared__ real_t     dB[TRACE_BLOCK_DIMX_3D_V4][TRACE_BLOCK_DIMY_3D_V4][2];

  // we only store z component of electric field
  __shared__ real_t     shEz[TRACE_BLOCK_DIMX_3D_V4][TRACE_BLOCK_DIMY_3D_V4];

  // qm and qp's are output of the trace step
  real_t qm [THREE_D][NVAR_MHD];
  real_t qp [THREE_D][NVAR_MHD];
  real_t qEdge[4][3][NVAR_MHD];

  // for gravity predictor
  //const real_t *gravin = gParams.arrayList[A_GRAV];

  // conservative variables
  //real_t c;

  real_t qZplus1 [NVAR_MHD];
  real_t bfZplus1[3];

  real_t qZminus1 [NVAR_MHD];
  //real_t bfZminus1[3];

  //real_t dA_Y_Zplus1;
  //real_t dA_Z_Zplus1;

  real_t dC_X;
  real_t dC_Y;
  real_t dC_X_Zplus1;
  real_t dC_Y_Zplus1;

  real_t Ex_j_k;
  real_t Ex_j_k1;
  real_t Ex_j1_k;
  real_t Ex_j1_k1;

  real_t Ey_i_k;
  real_t Ey_i_k1;
  real_t Ey_i1_k;
  real_t Ey_i1_k1;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  /*
   * initialize q (primitive variables ) in the 2 first XY-planes
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < 2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1)
      {
	
	int offset = elemOffset;

	// read primitive variables from d_Q
	if (k==0) {
	  qZminus1[ID] = d_Q[offset];  offset += arraySize;
	  qZminus1[IP] = d_Q[offset];  offset += arraySize;
	  qZminus1[IU] = d_Q[offset];  offset += arraySize;
	  qZminus1[IV] = d_Q[offset];  offset += arraySize;
	  qZminus1[IW] = d_Q[offset];  offset += arraySize;
	  qZminus1[IA] = d_Q[offset];  offset += arraySize;
	  qZminus1[IB] = d_Q[offset];  offset += arraySize;
	  qZminus1[IC] = d_Q[offset];
	} else { // k == 1
	  q[tx][ty][ID] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IP] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IU] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IV] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IW] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IA] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IB] = d_Q[offset];  offset += arraySize;
	  q[tx][ty][IC] = d_Q[offset];
	}


	// read face-centered magnetic field from Uin
	offset = elemOffset + 5 * arraySize;
	real_t bfX, bfY, bfZ;
	bfX = Uin[offset]; offset += arraySize;
	bfY = Uin[offset]; offset += arraySize;
	bfZ = Uin[offset];
	
	// set bf (face-centered magnetic field components)
	if (k==0) {
	  //bfZminus1[0] = bfX;
	  //bfZminus1[1] = bfY;
	  //bfZminus1[2] = bfZ;
	} else { // k == 1
	  bf[tx][ty][0] = bfX;
	  bf[tx][ty][1] = bfY;
	  bf[tx][ty][2] = bfZ;
	}
		
	// read magnetic slopes dC
	offset = elemOffset;
	dC_X = d_dC[offset]; offset += arraySize;
	dC_Y = d_dC[offset];
	
	// read electric field Ex (at i,j,k and i,j+1,k)
	offset = elemOffset;
	Ex_j_k  = Elec[offset];
	Ex_j1_k = Elec[offset+pitch]; 
	
	// read electric field Ey (at i,j,k and i+1,j,k)
	offset += arraySize;
	Ey_i_k  = Elec[offset];
	Ey_i1_k = Elec[offset+1];
      }
    __syncthreads();
  
  } // end for k

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    // data fetch :
    // get q, bf at z+1
    // get dA, dB at z+1
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1)
      {
	 
	 // read primitive variables at z+1
	 int offset = elemOffset + pitch*jmax; // z+1	 
	 qZplus1[ID] = d_Q[offset];  offset += arraySize;
	 qZplus1[IP] = d_Q[offset];  offset += arraySize;
	 qZplus1[IU] = d_Q[offset];  offset += arraySize;
	 qZplus1[IV] = d_Q[offset];  offset += arraySize;
	 qZplus1[IW] = d_Q[offset];  offset += arraySize;
	 qZplus1[IA] = d_Q[offset];  offset += arraySize;
	 qZplus1[IB] = d_Q[offset];  offset += arraySize;
	 qZplus1[IC] = d_Q[offset];
	 
	 // set bf (read face-centered magnetic field components)
	 offset = elemOffset + pitch*jmax + 5 * arraySize;
	 bfZplus1[IX] = Uin[offset]; offset += arraySize;
	 bfZplus1[IY] = Uin[offset]; offset += arraySize;
	 bfZplus1[IZ] = Uin[offset];
	 
	 // get magnetic slopes dA and dB at z=k
	 // read magnetic slopes dA (along Y and Z) 
	 offset = elemOffset+arraySize;
	 dA[tx][ty][0] = d_dA[offset]; offset += arraySize;
	 dA[tx][ty][1] = d_dA[offset];
	 
	 // read magnetic slopes dB (along X and Z)
	 offset = elemOffset;
	 dB[tx][ty][0] = d_dB[offset]; offset += (2*arraySize);
	 dB[tx][ty][1] = d_dB[offset];
	 
	 // get magnetic slopes dC (along X and Y) at z=k+1
	 offset = elemOffset + pitch*jmax;
	 dC_X_Zplus1 = d_dC[offset]; offset += arraySize;
	 dC_Y_Zplus1 = d_dC[offset];

	 // read electric field (Ex at  i,j,k+1 and i,j+1,k+1)
	 offset = elemOffset + pitch*jmax;
	 Ex_j_k1  = Elec[offset];
	 Ex_j1_k1 = Elec[offset+pitch]; 

	 // read electric field Ey (at i,j,k+1 and i+1,j,k+1)
	 offset += arraySize;
	 Ey_i_k1  = Elec[offset];
	 Ey_i1_k1 = Elec[offset+1];

	 // read electric field (Ez into shared memory) at z=k
	 offset = elemOffset + 2 * arraySize;
	 shEz[tx][ty] = Elec[offset];

       }
     __syncthreads();
     

     // slope and trace computation (i.e. dq, and then qm, qp, qEdge)

     if(i >= 1 and i < imax-2 and tx > 0 and tx < TRACE_BLOCK_DIMX_3D_V4-1 and
	j >= 1 and j < jmax-2 and ty > 0 and ty < TRACE_BLOCK_DIMY_3D_V4-1)
       {

	 int offset = elemOffset; // 3D index

	 real_t (&qPlusX )[NVAR_MHD] = q[tx+1][ty  ];
	 real_t (&qMinusX)[NVAR_MHD] = q[tx-1][ty  ];
	 real_t (&qPlusY )[NVAR_MHD] = q[tx  ][ty+1];
	 real_t (&qMinusY)[NVAR_MHD] = q[tx  ][ty-1];
	 real_t (&qPlusZ )[NVAR_MHD] = qZplus1;  // q[tx][ty] at z+1
	 real_t (&qMinusZ)[NVAR_MHD] = qZminus1; // q[tx][ty] at z-1
						 // (stored from z
						 // previous
						 // iteration)
	 // hydro slopes  array
	 real_t dq[3][NVAR_MHD];
	 
	 if (::gParams.slope_type==3) {
	   // positivity preserving slope computation
	   const int di = 1;
	   const int dj = pitch;
	   const int dk = pitch*jmax;

	   for (int nVar=0, offsetBase=elemOffset; 
		nVar<NVAR_MHD; 
		++nVar,offsetBase += arraySize) {

	     real_t vmin = FLT_MAX;
	     real_t vmax = -FLT_MAX;

	     // compute vmin,vmax
	     for (int ii=-1; ii<2; ++ii)
	       for (int jj=-1; jj<2; ++jj)
		 for (int kk=-1; kk<2; ++kk) {
		   offset = offsetBase + ii*di + jj*dj + kk*dk;
		   real_t tmp = d_Q[offset] - q[tx][ty][nVar];
		   vmin = FMIN(vmin,tmp);
		   vmax = FMAX(vmax,tmp);
		 }

	     // x+1,y  ,z   - x-1,y  ,z
	     real_t dfx =  HALF_F * ( d_Q[offsetBase+di] - d_Q[offsetBase-di]);
	     // x  ,y+1,z   - x  ,y-1,z
	     real_t dfy =  HALF_F * ( d_Q[offsetBase+dj] - d_Q[offsetBase-dj]);
	     // x  ,y  ,z+1 - x  ,y  ,z-1
	     real_t dfz =  HALF_F * ( d_Q[offsetBase+dk] - d_Q[offsetBase-dk]);

	     real_t dff  = HALF_F * (FABS(dfx) + FABS(dfy) + FABS(dfz));

	     real_t slop=ONE_F;
	     real_t &dlim=slop;
	     if (dff>ZERO_F) {
	       slop = FMIN(ONE_F, FMIN(FABS(vmin), FABS(vmax))/dff);
	     }

	     dq[IX][nVar] = dlim*dfx;
	     dq[IY][nVar] = dlim*dfy;
	     dq[IZ][nVar] = dlim*dfz;

	   } // end for nVar

	 } else {
	   // compute hydro slopes dq
	   slope_unsplit_hydro_3d(q[tx][ty], 
				  qPlusX, qMinusX, 
				  qPlusY, qMinusY, 
				  qPlusZ, qMinusZ,
				  dq);
	 }

	 // get face-centered magnetic components
	 real_t bfNb[6];
	 bfNb[0] = bf[tx  ][ty  ][0];
	 bfNb[1] = bf[tx+1][ty  ][0];
	 bfNb[2] = bf[tx  ][ty  ][1];
	 bfNb[3] = bf[tx  ][ty+1][1];
	 bfNb[4] = bf[tx  ][ty  ][2];
	 bfNb[5] = bfZplus1      [2];

	 // get dbf (transverse magnetic slopes)
	 real_t dbf[12];
	 dbf[0]  = dA[tx][ty][0]; // dA along Y
	 dbf[1]  = dA[tx][ty][1]; // dA along Z
	 dbf[2]  = dB[tx][ty][0]; // dB along X
	 dbf[3]  = dB[tx][ty][1]; // dB along Z
	 dbf[4]  = dC_X;          // dC along X
	 dbf[5]  = dC_Y;          // dC along Y
	 
	 dbf[6]  = dA[tx+1][ty  ][0];
	 dbf[7]  = dA[tx+1][ty  ][1];
	 dbf[8]  = dB[tx  ][ty+1][0];
	 dbf[9]  = dB[tx  ][ty+1][1];
	 dbf[10] = dC_X_Zplus1;
	 dbf[11] = dC_Y_Zplus1;
	 
	 // get electric field components
	 real_t elecFields[3][2][2];
	 // alias to electric field components
	 real_t (&Ex)[2][2] = elecFields[IX];
	 real_t (&Ey)[2][2] = elecFields[IY];
	 real_t (&Ez)[2][2] = elecFields[IZ];
	 
	 Ex[0][0] = Ex_j_k;
	 Ex[0][1] = Ex_j_k1;
	 Ex[1][0] = Ex_j1_k;
	 Ex[1][1] = Ex_j1_k1;
	 
	 Ey[0][0] = Ey_i_k;
	 Ey[0][1] = Ey_i_k1;
	 Ey[1][0] = Ey_i1_k;
	 Ey[1][1] = Ey_i1_k1;
	 
	 Ez[0][0] = shEz[tx  ][ty  ];
	 Ez[0][1] = shEz[tx  ][ty+1];
	 Ez[1][0] = shEz[tx+1][ty  ];
	 Ez[1][1] = shEz[tx+1][ty+1];
	 
	 // compute qm, qp and qEdge
	 trace_unsplit_mhd_3d_simpler(q[tx][ty], dq, bfNb, dbf, elecFields, 
				      dtdx, dtdy, dtdz, xPos,
				      qm, qp, qEdge);
	 
	 /*
	  * gravity predictor is moved into a separate kernel (February 5, 2014)
	  * It gives better performance. Trace computation time drops by 13%
	  * on K20 (maxregcount=128)
	  */
	 // gravity predictor on velocity component of qp0's and qEdge_LB
	 // if (gravityEnabled) {
	 //   real_t grav_x = HALF_F * dt * gravin[elemOffset+IX*arraySize];
	 //   real_t grav_y = HALF_F * dt * gravin[elemOffset+IY*arraySize];
	 //   real_t grav_z = HALF_F * dt * gravin[elemOffset+IZ*arraySize];

	 //   qp[0][IU] += grav_x; qp[0][IV] += grav_y; qp[0][IW] += grav_z;
	 //   qp[1][IU] += grav_x; qp[1][IV] += grav_y; qp[1][IW] += grav_z;
	 //   qp[2][IU] += grav_x; qp[2][IV] += grav_y; qp[2][IW] += grav_z;
	   
	 //   qm[0][IU] += grav_x; qp[0][IV] += grav_y; qp[0][IW] += grav_z;
	 //   qm[1][IU] += grav_x; qp[1][IV] += grav_y; qp[1][IW] += grav_z;
	 //   qm[2][IU] += grav_x; qp[2][IV] += grav_y; qp[2][IW] += grav_z;
	   
	 //   qEdge[IRT][0][IU] += grav_x;
	 //   qEdge[IRT][0][IV] += grav_y;
	 //   qEdge[IRT][0][IW] += grav_z;
	 //   qEdge[IRT][1][IU] += grav_x;
	 //   qEdge[IRT][1][IV] += grav_y;
	 //   qEdge[IRT][1][IW] += grav_z;
	 //   qEdge[IRT][2][IU] += grav_x;
	 //   qEdge[IRT][2][IV] += grav_y;
	 //   qEdge[IRT][2][IW] += grav_z;

	 //   qEdge[IRB][0][IU] += grav_x;
	 //   qEdge[IRB][0][IV] += grav_y;
	 //   qEdge[IRB][0][IW] += grav_z;
	 //   qEdge[IRB][1][IU] += grav_x;
	 //   qEdge[IRB][1][IV] += grav_y;
	 //   qEdge[IRB][1][IW] += grav_z;
	 //   qEdge[IRB][2][IU] += grav_x;
	 //   qEdge[IRB][2][IV] += grav_y;
	 //   qEdge[IRB][2][IW] += grav_z;

	 //   qEdge[ILT][0][IU] += grav_x;
	 //   qEdge[ILT][0][IV] += grav_y;
	 //   qEdge[ILT][0][IW] += grav_z;
	 //   qEdge[ILT][1][IU] += grav_x;
	 //   qEdge[ILT][1][IV] += grav_y;
	 //   qEdge[ILT][1][IW] += grav_z;
	 //   qEdge[ILT][2][IU] += grav_x;
	 //   qEdge[ILT][2][IV] += grav_y;
	 //   qEdge[ILT][2][IW] += grav_z;

	 //   qEdge[ILB][0][IU] += grav_x;
	 //   qEdge[ILB][0][IV] += grav_y;
	 //   qEdge[ILB][0][IW] += grav_z;
	 //   qEdge[ILB][1][IU] += grav_x;
	 //   qEdge[ILB][1][IV] += grav_y;
	 //   qEdge[ILB][1][IW] += grav_z;
	 //   qEdge[ILB][2][IU] += grav_x;
	 //   qEdge[ILB][2][IV] += grav_y;
	 //   qEdge[ILB][2][IW] += grav_z;

	 // } // end gravityEnabled
	 
	 // store qm, qp, qEdge in external memory
	 {
	   offset  = elemOffset;
	   for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	     
	     d_qm_x[offset] = qm[0][iVar];
	     d_qm_y[offset] = qm[1][iVar];
	     d_qm_z[offset] = qm[2][iVar];
	     
	     d_qp_x[offset] = qp[0][iVar];
	     d_qp_y[offset] = qp[1][iVar];
	     d_qp_z[offset] = qp[2][iVar];
	     
	     d_qEdge_RT[offset]  = qEdge[IRT][0][iVar];
	     d_qEdge_RB[offset]  = qEdge[IRB][0][iVar];
	     d_qEdge_LT[offset]  = qEdge[ILT][0][iVar];
	     d_qEdge_LB[offset]  = qEdge[ILB][0][iVar];
	     
	     d_qEdge_RT2[offset] = qEdge[IRT][1][iVar];
	     d_qEdge_RB2[offset] = qEdge[IRB][1][iVar];
	     d_qEdge_LT2[offset] = qEdge[ILT][1][iVar];
	     d_qEdge_LB2[offset] = qEdge[ILB][1][iVar];
	     
	     d_qEdge_RT3[offset] = qEdge[IRT][2][iVar];
	     d_qEdge_RB3[offset] = qEdge[IRB][2][iVar];
	     d_qEdge_LT3[offset] = qEdge[ILT][2][iVar];
	     d_qEdge_LB3[offset] = qEdge[ILB][2][iVar];
	     
	     offset  += arraySize;
	   } // end for iVar
	 } // end store qm, qp, qEdge

       } // end compute trace
     __syncthreads();

    // rotate buffer
    {
      // q                                 become  qZminus1
      // qZplus1 and bfZplus1 at z+1       become  q        and bf
      // dC_X_Zplus1                       becomes dC_X
      // dC_Y_Zplus1                       becomes dC_Y
      // update q with data ar z+1
      // rotate bfZminus1 - bf -bfZplus1
      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	qZminus1 [iVar] = q[tx][ty][iVar];
	q[tx][ty][iVar] = qZplus1[iVar];
      }
      //bfZminus1[0] = bf[tx][ty][0];
      //bfZminus1[1] = bf[tx][ty][1];
      //bfZminus1[2] = bf[tx][ty][2];
      
      bf[tx][ty][0] = bfZplus1[0];
      bf[tx][ty][1] = bfZplus1[1];
      bf[tx][ty][2] = bfZplus1[2];
      
      dC_X = dC_X_Zplus1;
      dC_Y = dC_Y_Zplus1;
      
      Ex_j_k  = Ex_j_k1;
      Ex_j1_k = Ex_j1_k1;
      
      Ey_i_k  = Ey_i_k1;
      Ey_i1_k = Ey_i1_k1;
    }
    __syncthreads();
    
  } // end for k
  
} // kernel_mhd_compute_trace_v4


#ifdef USE_DOUBLE
# define GRAV_PRED_BLOCK_DIMX_3D_V4	32
# define GRAV_PRED_BLOCK_DIMY_3D_V4	10
#else // simple precision
# define GRAV_PRED_BLOCK_DIMX_3D_V4	32
# define GRAV_PRED_BLOCK_DIMY_3D_V4	10
#endif // USE_DOUBLE
/**
 * Compute gravity predictor for MHD (implementation version 4).
 *
 * Must be called after kernel_mhd_compute_trace_v4.
 * \see kernel_mhd_compute_trace_v4
 *
 * \param[in,out] d_qm_x qm state along x
 * \param[in,out] d_qm_y qm state along y
 * \param[in,out] d_qm_z qm state along z
 * \param[in,out] d_qp_x qp state along x
 * \param[in,out] d_qp_y qp state along y
 * \param[in,out] d_qp_z qp state along z
 * \param[in,out] d_qEdge_RT 
 * \param[in,out] d_qEdge_RB 
 * \param[in,out] d_qEdge_LT 
 * \param[in,out] d_qEdge_LB
 *
 * \param[in,out] d_qEdge_RT2
 * \param[in,out] d_qEdge_RB2
 * \param[in,out] d_qEdge_LT2
 * \param[in,out] d_qEdge_LB2
 *
 * \param[in,out] d_qEdge_RT3
 * \param[in,out] d_qEdge_RB3
 * \param[in,out] d_qEdge_LT3
 * \param[in,out] d_qEdge_LB3
 *
 */
__global__ void kernel_mhd_compute_gravity_predictor_v4(real_t *d_qm_x,
							real_t *d_qm_y,
							real_t *d_qm_z,
							real_t *d_qp_x,
							real_t *d_qp_y,
							real_t *d_qp_z,
							real_t *d_qEdge_RT,
							real_t *d_qEdge_RB,
							real_t *d_qEdge_LT,
							real_t *d_qEdge_LB,
							real_t *d_qEdge_RT2,
							real_t *d_qEdge_RB2,
							real_t *d_qEdge_LT2,
							real_t *d_qEdge_LB2,
							real_t *d_qEdge_RT3,
							real_t *d_qEdge_RB3,
							real_t *d_qEdge_LT3,
							real_t *d_qEdge_LB3,
							int pitch, 
							int imax, 
							int jmax,
							int kmax,
							real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, GRAV_PRED_BLOCK_DIMX_3D_V4) + tx;
  const int j = __mul24(by, GRAV_PRED_BLOCK_DIMY_3D_V4) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // for gravity predictor
  const real_t *gravin = gParams.arrayList[A_GRAV];

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {

    if(i >= 1 and i < imax-2 and 
       j >= 1 and j < jmax-2)
      {

	int    offset;
	real_t grav_x = HALF_F * dt * gravin[elemOffset+IX*arraySize];
	real_t grav_y = HALF_F * dt * gravin[elemOffset+IY*arraySize];
	real_t grav_z = HALF_F * dt * gravin[elemOffset+IZ*arraySize];

	{
	  offset = elemOffset + IU*arraySize;
	  d_qm_x[offset] += grav_x; d_qp_x[offset] += grav_x;
	  d_qm_y[offset] += grav_x; d_qp_y[offset] += grav_x;
	  d_qm_z[offset] += grav_x; d_qp_z[offset] += grav_x;
	  
	  d_qEdge_RT[offset]  += grav_x;  
	  d_qEdge_RB[offset]  += grav_x; 
	  d_qEdge_LT[offset]  += grav_x; 
	  d_qEdge_LB[offset]  += grav_x; 
	  
	  d_qEdge_RT2[offset]  += grav_x;  
	  d_qEdge_RB2[offset]  += grav_x; 
	  d_qEdge_LT2[offset]  += grav_x; 
	  d_qEdge_LB2[offset]  += grav_x; 
	  
	  d_qEdge_RT3[offset]  += grav_x;  
	  d_qEdge_RB3[offset]  += grav_x; 
	  d_qEdge_LT3[offset]  += grav_x; 
	  d_qEdge_LB3[offset]  += grav_x; 

	} // end grav_x

	{
	  offset = elemOffset + IV*arraySize;
	  d_qm_x[offset] += grav_y; d_qp_x[offset] += grav_y;
	  d_qm_y[offset] += grav_y; d_qp_y[offset] += grav_y;
	  d_qm_z[offset] += grav_y; d_qp_z[offset] += grav_y;
	  
	  d_qEdge_RT[offset]  += grav_y;  
	  d_qEdge_RB[offset]  += grav_y; 
	  d_qEdge_LT[offset]  += grav_y; 
	  d_qEdge_LB[offset]  += grav_y; 
	  
	  d_qEdge_RT2[offset]  += grav_y;  
	  d_qEdge_RB2[offset]  += grav_y; 
	  d_qEdge_LT2[offset]  += grav_y; 
	  d_qEdge_LB2[offset]  += grav_y; 
	  
	  d_qEdge_RT3[offset]  += grav_y;  
	  d_qEdge_RB3[offset]  += grav_y; 
	  d_qEdge_LT3[offset]  += grav_y; 
	  d_qEdge_LB3[offset]  += grav_y; 

	} // end grav_y

	{
	  offset = elemOffset + IW*arraySize;
	  d_qm_x[offset] += grav_z; d_qp_x[offset] += grav_z;
	  d_qm_y[offset] += grav_z; d_qp_y[offset] += grav_z;
	  d_qm_z[offset] += grav_z; d_qp_z[offset] += grav_z;
	  
	  d_qEdge_RT[offset]  += grav_z;  
	  d_qEdge_RB[offset]  += grav_z; 
	  d_qEdge_LT[offset]  += grav_z; 
	  d_qEdge_LB[offset]  += grav_z; 
	  
	  d_qEdge_RT2[offset]  += grav_z;  
	  d_qEdge_RB2[offset]  += grav_z; 
	  d_qEdge_LT2[offset]  += grav_z; 
	  d_qEdge_LB2[offset]  += grav_z; 
	  
	  d_qEdge_RT3[offset]  += grav_z;  
	  d_qEdge_RB3[offset]  += grav_z; 
	  d_qEdge_LT3[offset]  += grav_z; 
	  d_qEdge_LB3[offset]  += grav_z; 

	} // end grav_z

      } // end if i,j

  } // end for k

} // kernel_mhd_compute_gravity_predictor_v4

#ifdef USE_DOUBLE
#define UPDATE_BLOCK_DIMX_3D_V4_OLD	16
#define UPDATE_BLOCK_INNER_DIMX_3D_V4_OLD	(UPDATE_BLOCK_DIMX_3D_V4_OLD-1)
#define UPDATE_BLOCK_DIMY_3D_V4_OLD	16
#define UPDATE_BLOCK_INNER_DIMY_3D_V4_OLD	(UPDATE_BLOCK_DIMY_3D_V4_OLD-1)
#else // simple precision
#define UPDATE_BLOCK_DIMX_3D_V4_OLD	16
#define UPDATE_BLOCK_INNER_DIMX_3D_V4_OLD	(UPDATE_BLOCK_DIMX_3D_V4_OLD-1)
#define UPDATE_BLOCK_DIMY_3D_V4_OLD	16
#define UPDATE_BLOCK_INNER_DIMY_3D_V4_OLD	(UPDATE_BLOCK_DIMY_3D_V4_OLD-1)
#endif // USE_DOUBLE

/**
 * Update MHD conservative variables (implementation version 4).
 * 
 * This is the final kernel, that given the qm, qp, qEdge states compute
 * hydro fluxes, compute EMF, perform hydro update and finally store emf (update 
 * using emf is done later).
 *
 * \see kernel_mhd_compute_trace_v4 (computation of qm, qp, qEdge buffer)
 *
 * \param[in]  Uin  input MHD conservative variable array
 * \param[out] Uout ouput MHD conservative variable array
 * \param[in] d_qm_x qm state along x
 * \param[in] d_qm_y qm state along y
 * \param[in] d_qm_z qm state along z
 * \param[in] d_qp_x qp state along x
 * \param[in] d_qp_y qp state along y
 * \param[in] d_qp_z qp state along z
 * \param[in] d_qEdge_RT 
 * \param[in] d_qEdge_RB 
 * \param[in] d_qEdge_LT 
 * \param[in] d_qEdge_LB 
 * \param[in] d_qEdge_RT2
 * \param[in] d_qEdge_RB2
 * \param[in] d_qEdge_LT2
 * \param[in] d_qEdge_LB2
 * \param[in] d_qEdge_RT3
 * \param[in] d_qEdge_RB3
 * \param[in] d_qEdge_LT3
 * \param[in] d_qEdge_LB3
 * \param[out] d_emf
 *
 */
__global__ void kernel_mhd_flux_update_hydro_v4_old(const real_t *Uin, 
						    real_t       *Uout,
						    const real_t *d_qm_x,
						    const real_t *d_qm_y,
						    const real_t *d_qm_z,
						    const real_t *d_qp_x,
						    const real_t *d_qp_y,
						    const real_t *d_qp_z,
						    const real_t *d_qEdge_RT,
						    const real_t *d_qEdge_RB,
						    const real_t *d_qEdge_LT,
						    const real_t *d_qEdge_LB,
						    const real_t *d_qEdge_RT2,
						    const real_t *d_qEdge_RB2,
						    const real_t *d_qEdge_LT2,
						    const real_t *d_qEdge_LB2,
						    const real_t *d_qEdge_RT3,
						    const real_t *d_qEdge_RB3,
						    const real_t *d_qEdge_LT3,
						    const real_t *d_qEdge_LB3,
						    real_t       *d_emf,
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
  
  const int i = __mul24(bx, UPDATE_BLOCK_INNER_DIMX_3D_V4_OLD) + tx;
  const int j = __mul24(by, UPDATE_BLOCK_INNER_DIMY_3D_V4_OLD) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // flux computation
  __shared__ real_t   flux[UPDATE_BLOCK_DIMX_3D_V4_OLD][UPDATE_BLOCK_DIMY_3D_V4_OLD][NVAR_MHD];

  // qm and qp's are output of the trace step
  //real_t qm [THREE_D][NVAR_MHD];
  //real_t qp [THREE_D][NVAR_MHD];
  //real_t qEdge[4][3][NVAR_MHD];

  // conservative variables
  real_t uOut[NVAR_MHD];
  //real_t c;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  // rotation rate
  real_t &Omega0 = ::gParams.Omega0;

  /*
   * shearing box correction on momentum parameters
   */
  real_t lambda=0, ratio=1, alpha1=1, alpha2=0;
  if (Omega0>0) {
    lambda = Omega0*dt;
    lambda = ONE_FOURTH_F * lambda * lambda;
    ratio  = (ONE_F-lambda)/(ONE_F+lambda);
    alpha1 =          ONE_F/(ONE_F+lambda);
    alpha2 =      Omega0*dt/(ONE_F+lambda);
  }

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use flux as flux_x
    real_t (&flux_x)[UPDATE_BLOCK_DIMX_3D_V4_OLD][UPDATE_BLOCK_DIMY_3D_V4_OLD][NVAR_MHD] = flux;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    flux_x[tx][ty][IA] = ZERO_F;
    flux_x[tx][ty][IB] = ZERO_F;
    flux_x[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and
       j >= 3 and j < jmax-2 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t   qleft_x [NVAR_MHD];
	real_t   qright_x[NVAR_MHD];
	
	// set qleft_x by re-reading qm_x from external memory at location x-1
	int offset = elemOffset-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_x[iVar] = d_qm_x[offset];
	  offset += arraySize;
	}

	// set qright_x by re-reading qp_x from external memory at location x
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_x[iVar] = d_qp_x[offset];
	  offset += arraySize;
	}

	riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i >= 3 and i < imax-3 and tx < UPDATE_BLOCK_DIMX_3D_V4_OLD-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_BLOCK_DIMY_3D_V4_OLD-1 and
       k >= 3 and k < kmax-3)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offset = elemOffset;
	uOut[ID] = Uin[offset];  offset += arraySize;
	uOut[IP] = Uin[offset];  offset += arraySize;
	uOut[IU] = Uin[offset];  offset += arraySize;
	uOut[IV] = Uin[offset];  offset += arraySize;
	uOut[IW] = Uin[offset];  offset += arraySize;
	uOut[IA] = Uin[offset];  offset += arraySize;
	uOut[IB] = Uin[offset];  offset += arraySize;
	uOut[IC] = Uin[offset];

	// rotating frame corrections
	if (Omega0 > 0) {
	  real_t dsx =   TWO_F * Omega0 * dt * uOut[IV]/(ONE_F + lambda);
	  real_t dsy = -HALF_F * Omega0 * dt * uOut[IU]/(ONE_F + lambda);
	  uOut[IU] = uOut[IU]*ratio + dsx;
	  uOut[IV] = uOut[IV]*ratio + dsy;
	}

	if (Omega0 >0) {
	  uOut[ID] += (flux_x[tx  ][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
	  
	  uOut[IP] += (flux_x[tx  ][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
	  
	  uOut[IU] += ( alpha1*(flux_x[tx  ][ty][IU]-flux_x[tx+1][ty][IU]) + 
			alpha2*(flux_x[tx  ][ty][IV]-flux_x[tx+1][ty][IV]) )*dtdx;
	  
	  uOut[IV] += ( alpha1*(flux_x[tx  ][ty][IV]-flux_x[tx+1][ty][IV]) - 0.25*
			alpha2*(flux_x[tx  ][ty][IU]-flux_x[tx+1][ty][IU]) )*dtdx;
	  
	  uOut[IW] += (flux_x[tx  ][ty][IW]-flux_x[tx+1][ty][IW])*dtdx;
	} else {
	  uOut[ID] += (flux_x[tx  ][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
	  
	  uOut[IP] += (flux_x[tx  ][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
	  
	  uOut[IU] += (flux_x[tx  ][ty][IU]-flux_x[tx+1][ty][IU])*dtdx;
	  
	  uOut[IV] += (flux_x[tx  ][ty][IV]-flux_x[tx+1][ty][IV])*dtdx;
	  
	  uOut[IW] += (flux_x[tx  ][ty][IW]-flux_x[tx+1][ty][IW])*dtdx;
	}

      }
    __syncthreads();

    /*
     * Compute fluxes at Y-interfaces.
     */
    // re-use flux as flux_y
    real_t (&flux_y)[UPDATE_BLOCK_DIMX_3D_V4_OLD][UPDATE_BLOCK_DIMY_3D_V4_OLD][NVAR_MHD] = flux;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    flux_y[tx][ty][IA] = ZERO_F;
    flux_y[tx][ty][IB] = ZERO_F;
    flux_y[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and 
       j >= 3 and j < jmax-2 and 
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t  qleft_y[NVAR_MHD];
	real_t qright_y[NVAR_MHD];
	
	// set qleft_y by reading qm_y from external memory at location y-1
	int offset = elemOffset-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_y[iVar] = d_qm_y[offset];
	  offset += arraySize;
	}
	
	// set qright_y by reading qp_y from external memory at location y
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_y[iVar] = d_qp_y[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IV
	swap_value_(qleft_y[IU],qleft_y[IV]);
	swap_value_(qleft_y[IA],qleft_y[IB]);
	swap_value_(qright_y[IU],qright_y[IV]);
	swap_value_(qright_y[IA],qright_y[IB]);

	riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);

	/* shear correction on flux_y */
	if (/* cartesian */ ::gParams.Omega0 > 0 /* and not fargo */) {
	  real_t shear_y = -1.5 * ::gParams.Omega0 * xPos;
	  real_t eMag, eKin, eTot;
	  real_t bn_mean = HALF_F * (qleft_y[IA] + qright_y[IA]);
	  real_t &gamma  = ::gParams.gamma0;
	  
	  if (shear_y > 0) {
	    eMag = HALF_F * (qleft_y[IA]*qleft_y[IA] + qleft_y[IB]*qleft_y[IB] + qleft_y[IC]*qleft_y[IC]);
	    eKin = HALF_F * (qleft_y[IU]*qleft_y[IU] + qleft_y[IV]*qleft_y[IV] + qleft_y[IW]*qleft_y[IW]);
	    eTot = eKin + eMag + qleft_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qleft_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qleft_y[ID]*qleft_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qleft_y[ID]*qleft_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qleft_y[ID]*qleft_y[IW];
	  } else {
	    eMag = HALF_F * (qright_y[IA]*qright_y[IA] + qright_y[IB]*qright_y[IB] + qright_y[IC]*qright_y[IC]);
	    eKin = HALF_F * (qright_y[IU]*qright_y[IU] + qright_y[IV]*qright_y[IV] + qright_y[IW]*qright_y[IW]);
	    eTot = eKin + eMag + qright_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qright_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qright_y[ID]*qright_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qright_y[ID]*qright_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qright_y[ID]*qright_y[IW];		  
	  }
	} // end shear correction on flux_y

      }  
    __syncthreads();
    
    // update uOut with flux_y
    if(i >= 3 and i < imax-3 and tx < UPDATE_BLOCK_DIMX_3D_V4_OLD-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_BLOCK_DIMY_3D_V4_OLD-1 and
       k >= 3 and k < kmax-3)
      {
	// watchout IU and IV are swapped !

	if (Omega0>0) {
	  uOut[ID] += (flux_y[tx][ty  ][ID]-flux_y[tx][ty+1][ID])*dtdy;
	  
	  uOut[IP] += (flux_y[tx][ty  ][IP]-flux_y[tx][ty+1][IP])*dtdy;
	  
	  uOut[IU] += ( alpha1*(flux_y[tx][ty  ][IV]-flux_y[tx][ty+1][IV]) + 
			alpha2*(flux_y[tx][ty  ][IU]-flux_y[tx][ty+1][IU]) )*dtdy;
	  
	  uOut[IV] += ( alpha1*(flux_y[tx][ty  ][IU]-flux_y[tx][ty+1][IU]) - 0.25*
			alpha2*(flux_y[tx][ty  ][IV]-flux_y[tx][ty+1][IV]) )*dtdy;
	  
	  uOut[IW] += (flux_y[tx][ty  ][IW]-flux_y[tx][ty+1][IW])*dtdy;
	} else {
	  uOut[ID] += (flux_y[tx][ty  ][ID]-flux_y[tx][ty+1][ID])*dtdy;
	  
	  uOut[IP] += (flux_y[tx][ty  ][IP]-flux_y[tx][ty+1][IP])*dtdy;
	  
	  uOut[IU] += (flux_y[tx][ty  ][IV]-flux_y[tx][ty+1][IV])*dtdy;
	  
	  uOut[IV] += (flux_y[tx][ty  ][IU]-flux_y[tx][ty+1][IU])*dtdy;
	  
	  uOut[IW] += (flux_y[tx][ty  ][IW]-flux_y[tx][ty+1][IW])*dtdy;
	}
      }
    __syncthreads();
    
    /*
     * Compute fluxes at Z-interfaces.
     */
    real_t flux_z[NVAR_MHD];
    flux_z[ID] = ZERO_F;
    flux_z[IP] = ZERO_F;
    flux_z[IU] = ZERO_F;
    flux_z[IV] = ZERO_F;
    flux_z[IW] = ZERO_F;
    flux_z[IA] = ZERO_F;
    flux_z[IB] = ZERO_F;
    flux_z[IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V4_OLD-1 and
       j >= 3 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V4_OLD-1 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t qleft_z [NVAR_MHD];
	real_t qright_z[NVAR_MHD];
	
	// set qleft_z by reading qm_z from external memory at location z-1
	int offset = elemOffset - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_z[iVar] = d_qm_z[offset];
	  offset += arraySize;
	}
	
        // set qright_z by reading qp_z from external memory at location z
        offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_z[iVar] = d_qp_z[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IW
	swap_value_(qleft_z[IU] ,qleft_z[IW]);
	swap_value_(qleft_z[IA] ,qleft_z[IC]);
	swap_value_(qright_z[IU],qright_z[IW]);
	swap_value_(qright_z[IA],qright_z[IC]);
	
	riemann_mhd(qleft_z, qright_z, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i >= 3 and i < imax-3 and tx < UPDATE_BLOCK_DIMX_3D_V4_OLD-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_BLOCK_DIMY_3D_V4_OLD-1 and
       k >= 3 and k < kmax-2)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	int offset = elemOffset;

	if (k < kmax-3) {
	  // watchout IU and IW are swapped !

	  if (Omega0>0) {
	    uOut[ID] += flux_z[ID]*dtdz;
	    uOut[IP] += flux_z[IP]*dtdz;
	    uOut[IU] += (alpha1*flux_z[IW] +     alpha2*flux_z[IV])*dtdz;
	    uOut[IV] += (alpha1*flux_z[IV] -0.25*alpha2*flux_z[IW])*dtdz;
	    uOut[IW] += flux_z[IU]*dtdz;
	  } else {
	    uOut[ID] += flux_z[ID]*dtdz;
	    uOut[IP] += flux_z[IP]*dtdz;
	    uOut[IU] += flux_z[IW]*dtdz;
	    uOut[IV] += flux_z[IV]*dtdz;
	    uOut[IW] += flux_z[IU]*dtdz;
	  }
	  
	  // actually perform the update on external device memory
	  Uout[offset] = uOut[ID];  offset += arraySize;
	  Uout[offset] = uOut[IP];  offset += arraySize;
	  Uout[offset] = uOut[IU];  offset += arraySize;
	  Uout[offset] = uOut[IV];  offset += arraySize;
	  Uout[offset] = uOut[IW];
	}

	if (k>3) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offset = elemOffset - pitch*jmax;
	  if (Omega0>0) {
	    Uout[offset] -= flux_z[ID]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IP]*dtdz; offset += arraySize;
	    Uout[offset] -= (alpha1*flux_z[IW]+alpha2*flux_z[IV])*dtdz; offset += arraySize;
	    Uout[offset] -= (alpha1*flux_z[IV]-0.25*alpha2*flux_z[IW])*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IU]*dtdz;
	  } else {
	    Uout[offset] -= flux_z[ID]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IP]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IW]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IV]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IU]*dtdz;	    
	  }
	}
      } // end update along Z
    __syncthreads();


    /*
     * EMF computations and update face-centered magnetic field components.
     */

     // intermediate values for EMF computations
     real_t qEdge_emfX[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT)[NVAR_MHD] = qEdge_emfX[IRT];
     real_t (&qEdge_RB)[NVAR_MHD] = qEdge_emfX[IRB];
     real_t (&qEdge_LT)[NVAR_MHD] = qEdge_emfX[ILT];
     real_t (&qEdge_LB)[NVAR_MHD] = qEdge_emfX[ILB];
     
     real_t qEdge_emfY[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT2)[NVAR_MHD] = qEdge_emfY[IRT];
     real_t (&qEdge_RB2)[NVAR_MHD] = qEdge_emfY[IRB];
     real_t (&qEdge_LT2)[NVAR_MHD] = qEdge_emfY[ILT];
     real_t (&qEdge_LB2)[NVAR_MHD] = qEdge_emfY[ILB];
     
     real_t qEdge_emfZ[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT3)[NVAR_MHD] = qEdge_emfZ[IRT];
     real_t (&qEdge_RB3)[NVAR_MHD] = qEdge_emfZ[IRB];
     real_t (&qEdge_LT3)[NVAR_MHD] = qEdge_emfZ[ILT];
     real_t (&qEdge_LB3)[NVAR_MHD] = qEdge_emfZ[ILB];

     real_t emf;
     
     if(i > 1 and i < imax-2 and 
        j > 1 and j < jmax-2 and 
        k > 1 and k < kmax-2)
      {
	
	/*
	 * offset into external memory array to qEdge data
	 */
	
	int offset2         = elemOffset;
	
	
	/*
	 * compute emfZ
	 */

	// qEdge_RT3 at location x-1, y-1
	int offset = offset2-1-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT3[iVar] = d_qEdge_RT3[offset];
	  offset += arraySize;
	}
	
	// qEdge RB3 at location x-1, y
	offset = offset2-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB3[iVar] = d_qEdge_RB3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT3 at location x, y-1
	offset = offset2-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT3[iVar] = d_qEdge_LT3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB3 at location x, y
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB3[iVar] = d_qEdge_LB3[offset];
	  offset += arraySize;
	}

	// finally compute emfZ
	emf = compute_emf<EMFZ>(qEdge_emfZ,xPos);
	offset = offset2 + I_EMFZ*arraySize;
	d_emf[offset] = emf;

	/*
	 * compute emfY (take care RB and LT are swapped)
	 */

	// qEdge_RT2 at location x-1, z-1
	offset = offset2 - pitch*jmax - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT2[iVar] = d_qEdge_RT2[offset];
	  offset += arraySize;
	}
	
	// qEdge RB2 (actually LT2) at location x, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB2[iVar] = d_qEdge_LT2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT2 (actually RB2) at location x-1, z
	offset = offset2 - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT2[iVar] = d_qEdge_RB2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB2 at location x, y
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB2[iVar] = d_qEdge_LB2[offset];
	  offset += arraySize;
	}

	// finally compute emfY
	emf = compute_emf<EMFY>(qEdge_emfY,xPos);
	offset = offset2 + I_EMFY*arraySize;
	d_emf[offset] = emf;

	/*
	 * compute emfX
	 */
	// qEdge_RT at location y-1, z-1
	offset = offset2 - pitch*jmax - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT[iVar] = d_qEdge_RT[offset];
	  offset += arraySize;
	}
	
	// qEdge RB at location y-1, z
	offset = offset2 - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB[iVar] = d_qEdge_RB[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT at location y, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT[iVar] = d_qEdge_LT[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB at location y, z
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB[iVar] = d_qEdge_LB[offset];
	  offset += arraySize;
	}

	// finally compute emfX
	emf = compute_emf<EMFX>(qEdge_emfX,xPos);
	offset = offset2 + I_EMFX*arraySize;
	d_emf[offset] = emf;
	
      }
    __syncthreads();
    
    // if(i >= 3 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V4_OLD-1 and
    //    j >= 3 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V4_OLD-1 and
    //    k >= 3 and k < kmax-2)
    //   {
    // 	// actually perform update on external device memory
	
    // 	// First : update at current location x,y,z
    // 	int offset;
    // 	if (k<kmax-3) {
	  
    // 	  offset = elemOffset + 5 * arraySize;
	  
    // 	  // probably don't need to re-read
    // 	  uOut[IA] = Uin[offset];  offset += arraySize;
    // 	  uOut[IB] = Uin[offset];  offset += arraySize;
    // 	  uOut[IC] = Uin[offset];
	  
    // 	  // update bx
    // 	  uOut[IA] -= emf[tx  ][ty  ][IZ]*dtdy;
    // 	  uOut[IA] += emf[tx  ][ty+1][IZ]*dtdy;  
    // 	  uOut[IA] += emf[tx  ][ty  ][IY]*dtdz;
	  
    // 	  // update by
    // 	  uOut[IB] += emf[tx  ][ty  ][IZ]*dtdx;
    // 	  uOut[IB] -= emf[tx+1][ty  ][IZ]*dtdx; 
    // 	  uOut[IB] -= emf[tx  ][ty  ][IX]*dtdz;
	  
    // 	  // update bz
    // 	  uOut[IC] += emf[tx+1][ty  ][IY]*dtdx;
    // 	  uOut[IC] -= emf[tx  ][ty  ][IY]*dtdx;
    // 	  uOut[IC] -= emf[tx  ][ty+1][IX]*dtdy;
    // 	  uOut[IC] += emf[tx  ][ty  ][IX]*dtdy;

    // 	  // write buffer
    // 	  offset = elemOffset + 5 * arraySize;
    // 	  Uout[offset] = uOut[IA]; offset += arraySize;
    // 	  Uout[offset] = uOut[IB]; offset += arraySize;
    // 	  Uout[offset] = uOut[IC];
    // 	}

    // 	// Second : update at z-1 ! Take care not using Uin here,
    // 	// since Uout already been updated at z-1 in the previous step
    // 	// !!! 
    // 	if (k>3) {
    // 	  offset = elemOffset - pitch*jmax + 5 * arraySize;
    // 	  Uout[offset] -= emf[tx][ty][IY]*dtdz; // update bx
    // 	  offset += arraySize;
    // 	  Uout[offset] += emf[tx][ty][IX]*dtdz; // update by
    // 	}
	
    //   }
    // __syncthreads();
        
  } // end for k

} // kernel_mhd_flux_update_hydro_v4_old

#ifdef USE_DOUBLE
#define UPDATE_BLOCK_DIMX_3D_V4	16
#define UPDATE_BLOCK_INNER_DIMX_3D_V4	(UPDATE_BLOCK_DIMX_3D_V4-1)
#define UPDATE_BLOCK_DIMY_3D_V4	8
#define UPDATE_BLOCK_INNER_DIMY_3D_V4	(UPDATE_BLOCK_DIMY_3D_V4-1)
#else // simple precision
#define UPDATE_BLOCK_DIMX_3D_V4	16
#define UPDATE_BLOCK_INNER_DIMX_3D_V4	(UPDATE_BLOCK_DIMX_3D_V4-1)
#define UPDATE_BLOCK_DIMY_3D_V4	8
#define UPDATE_BLOCK_INNER_DIMY_3D_V4	(UPDATE_BLOCK_DIMY_3D_V4-1)
#endif // USE_DOUBLE

/**
 * Update MHD conservative variables : density, energy and velocity (implementation version 4).
 * 
 * In this kernel, we load qm, qp states and then compute
 * hydro fluxes (using rieman solver) and finally perform hydro update
 *
 * \see kernel_mhd_compute_trace_v4 (computation of qm, qp, qEdge buffer)
 *
 * \param[in]  Uin  input MHD conservative variable array
 * \param[out] Uout ouput MHD conservative variable array
 * \param[in] d_qm_x qm state along x
 * \param[in] d_qm_y qm state along y
 * \param[in] d_qm_z qm state along z
 * \param[in] d_qp_x qp state along x
 * \param[in] d_qp_y qp state along y
 * \param[in] d_qp_z qp state along z
 *
 */
__global__ void kernel_mhd_flux_update_hydro_v4(const real_t * __restrict__ Uin, 
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
  
  const int i = __mul24(bx, UPDATE_BLOCK_INNER_DIMX_3D_V4) + tx;
  const int j = __mul24(by, UPDATE_BLOCK_INNER_DIMY_3D_V4) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  // flux computation
  __shared__ real_t   flux[UPDATE_BLOCK_DIMX_3D_V4][UPDATE_BLOCK_DIMY_3D_V4][NVAR_MHD];

  // qm and qp's are output of the trace step
  //real_t qm [THREE_D][NVAR_MHD];
  //real_t qp [THREE_D][NVAR_MHD];

  // conservative variables
  real_t uOut[NVAR_MHD];
  //real_t c;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  // rotation rate
  real_t &Omega0 = ::gParams.Omega0;

  /*
   * shearing box correction on momentum parameters
   */
  real_t lambda=0, ratio=1, alpha1=1, alpha2=0;
  if (Omega0>0) {
    lambda = Omega0*dt;
    lambda = ONE_FOURTH_F * lambda * lambda;
    ratio  = (ONE_F-lambda)/(ONE_F+lambda);
    alpha1 =          ONE_F/(ONE_F+lambda);
    alpha2 =      Omega0*dt/(ONE_F+lambda);
  }

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use flux as flux_x
    real_t (&flux_x)[UPDATE_BLOCK_DIMX_3D_V4][UPDATE_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    flux_x[tx][ty][IA] = ZERO_F;
    flux_x[tx][ty][IB] = ZERO_F;
    flux_x[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and
       j >= 3 and j < jmax-2 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t   qleft_x [NVAR_MHD];
	real_t   qright_x[NVAR_MHD];
	
	// set qleft_x by re-reading qm_x from external memory at location x-1
	int offset = elemOffset-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_x[iVar] = d_qm_x[offset];
	  offset += arraySize;
	}

	// set qright_x by re-reading qp_x from external memory at location x
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_x[iVar] = d_qp_x[offset];
	  offset += arraySize;
	}

	riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i >= 3 and i < imax-3 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-3)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offset = elemOffset;
	uOut[ID] = Uin[offset];  offset += arraySize;
	uOut[IP] = Uin[offset];  offset += arraySize;
	uOut[IU] = Uin[offset];  offset += arraySize;
	uOut[IV] = Uin[offset];  offset += arraySize;
	uOut[IW] = Uin[offset];  offset += arraySize;
	uOut[IA] = Uin[offset];  offset += arraySize;
	uOut[IB] = Uin[offset];  offset += arraySize;
	uOut[IC] = Uin[offset];

	// rotating frame corrections
	if (Omega0 > 0) {
	  real_t dsx =   TWO_F * Omega0 * dt * uOut[IV]/(ONE_F + lambda);
	  real_t dsy = -HALF_F * Omega0 * dt * uOut[IU]/(ONE_F + lambda);
	  uOut[IU] = uOut[IU]*ratio + dsx;
	  uOut[IV] = uOut[IV]*ratio + dsy;
	}

	if (Omega0 >0) {
	  uOut[ID] += (flux_x[tx  ][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
	  
	  uOut[IP] += (flux_x[tx  ][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
	  
	  uOut[IU] += ( alpha1*(flux_x[tx  ][ty][IU]-flux_x[tx+1][ty][IU]) + 
			alpha2*(flux_x[tx  ][ty][IV]-flux_x[tx+1][ty][IV]) )*dtdx;
	  
	  uOut[IV] += ( alpha1*(flux_x[tx  ][ty][IV]-flux_x[tx+1][ty][IV]) - 0.25*
			alpha2*(flux_x[tx  ][ty][IU]-flux_x[tx+1][ty][IU]) )*dtdx;
	  
	  uOut[IW] += (flux_x[tx  ][ty][IW]-flux_x[tx+1][ty][IW])*dtdx;
	} else {
	  uOut[ID] += (flux_x[tx  ][ty][ID]-flux_x[tx+1][ty][ID])*dtdx;
	  
	  uOut[IP] += (flux_x[tx  ][ty][IP]-flux_x[tx+1][ty][IP])*dtdx;
	  
	  uOut[IU] += (flux_x[tx  ][ty][IU]-flux_x[tx+1][ty][IU])*dtdx;
	  
	  uOut[IV] += (flux_x[tx  ][ty][IV]-flux_x[tx+1][ty][IV])*dtdx;
	  
	  uOut[IW] += (flux_x[tx  ][ty][IW]-flux_x[tx+1][ty][IW])*dtdx;
	}

      }
    __syncthreads();

    /*
     * Compute fluxes at Y-interfaces.
     */
    // re-use flux as flux_y
    real_t (&flux_y)[UPDATE_BLOCK_DIMX_3D_V4][UPDATE_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    flux_y[tx][ty][IA] = ZERO_F;
    flux_y[tx][ty][IB] = ZERO_F;
    flux_y[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and 
       j >= 3 and j < jmax-2 and 
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t  qleft_y[NVAR_MHD];
	real_t qright_y[NVAR_MHD];
	
	// set qleft_y by reading qm_y from external memory at location y-1
	int offset = elemOffset-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_y[iVar] = d_qm_y[offset];
	  offset += arraySize;
	}
	
	// set qright_y by reading qp_y from external memory at location y
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_y[iVar] = d_qp_y[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IV
	swap_value_(qleft_y[IU],qleft_y[IV]);
	swap_value_(qleft_y[IA],qleft_y[IB]);
	swap_value_(qright_y[IU],qright_y[IV]);
	swap_value_(qright_y[IA],qright_y[IB]);

	riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);

	/* shear correction on flux_y */
	if (/* cartesian */ ::gParams.Omega0 > 0 /* and not fargo */) {
	  real_t shear_y = -1.5 * ::gParams.Omega0 * xPos;
	  real_t eMag, eKin, eTot;
	  real_t bn_mean = HALF_F * (qleft_y[IA] + qright_y[IA]);
	  real_t &gamma  = ::gParams.gamma0;
	  
	  if (shear_y > 0) {
	    eMag = HALF_F * (qleft_y[IA]*qleft_y[IA] + qleft_y[IB]*qleft_y[IB] + qleft_y[IC]*qleft_y[IC]);
	    eKin = HALF_F * (qleft_y[IU]*qleft_y[IU] + qleft_y[IV]*qleft_y[IV] + qleft_y[IW]*qleft_y[IW]);
	    eTot = eKin + eMag + qleft_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qleft_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qleft_y[ID]*qleft_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qleft_y[ID]*qleft_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qleft_y[ID]*qleft_y[IW];
	  } else {
	    eMag = HALF_F * (qright_y[IA]*qright_y[IA] + qright_y[IB]*qright_y[IB] + qright_y[IC]*qright_y[IC]);
	    eKin = HALF_F * (qright_y[IU]*qright_y[IU] + qright_y[IV]*qright_y[IV] + qright_y[IW]*qright_y[IW]);
	    eTot = eKin + eMag + qright_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qright_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qright_y[ID]*qright_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qright_y[ID]*qright_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qright_y[ID]*qright_y[IW];		  
	  }
	} // end shear correction on flux_y

      }  
    __syncthreads();
    
    // update uOut with flux_y
    if(i >= 3 and i < imax-3 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-3)
      {
	// watchout IU and IV are swapped !

	if (Omega0>0) {
	  uOut[ID] += (flux_y[tx][ty  ][ID]-flux_y[tx][ty+1][ID])*dtdy;
	  
	  uOut[IP] += (flux_y[tx][ty  ][IP]-flux_y[tx][ty+1][IP])*dtdy;
	  
	  uOut[IU] += ( alpha1*(flux_y[tx][ty  ][IV]-flux_y[tx][ty+1][IV]) + 
			alpha2*(flux_y[tx][ty  ][IU]-flux_y[tx][ty+1][IU]) )*dtdy;
	  
	  uOut[IV] += ( alpha1*(flux_y[tx][ty  ][IU]-flux_y[tx][ty+1][IU]) - 0.25*
			alpha2*(flux_y[tx][ty  ][IV]-flux_y[tx][ty+1][IV]) )*dtdy;
	  
	  uOut[IW] += (flux_y[tx][ty  ][IW]-flux_y[tx][ty+1][IW])*dtdy;
	} else {
	  uOut[ID] += (flux_y[tx][ty  ][ID]-flux_y[tx][ty+1][ID])*dtdy;
	  
	  uOut[IP] += (flux_y[tx][ty  ][IP]-flux_y[tx][ty+1][IP])*dtdy;
	  
	  uOut[IU] += (flux_y[tx][ty  ][IV]-flux_y[tx][ty+1][IV])*dtdy;
	  
	  uOut[IV] += (flux_y[tx][ty  ][IU]-flux_y[tx][ty+1][IU])*dtdy;
	  
	  uOut[IW] += (flux_y[tx][ty  ][IW]-flux_y[tx][ty+1][IW])*dtdy;
	}
      }
    __syncthreads();
    
    /*
     * Compute fluxes at Z-interfaces.
     */
    real_t flux_z[NVAR_MHD];
    flux_z[ID] = ZERO_F;
    flux_z[IP] = ZERO_F;
    flux_z[IU] = ZERO_F;
    flux_z[IV] = ZERO_F;
    flux_z[IW] = ZERO_F;
    flux_z[IA] = ZERO_F;
    flux_z[IB] = ZERO_F;
    flux_z[IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t qleft_z [NVAR_MHD];
	real_t qright_z[NVAR_MHD];
	
	// set qleft_z by reading qm_z from external memory at location z-1
	int offset = elemOffset - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_z[iVar] = d_qm_z[offset];
	  offset += arraySize;
	}
	
        // set qright_z by reading qp_z from external memory at location z
        offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_z[iVar] = d_qp_z[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IW
	swap_value_(qleft_z[IU] ,qleft_z[IW]);
	swap_value_(qleft_z[IA] ,qleft_z[IC]);
	swap_value_(qright_z[IU],qright_z[IW]);
	swap_value_(qright_z[IA],qright_z[IC]);
	
	riemann_mhd(qleft_z, qright_z, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i >= 3 and i < imax-3 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-2)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	int offset = elemOffset;

	if (k < kmax-3) {
	  // watchout IU and IW are swapped !

	  if (Omega0>0) {
	    uOut[ID] += flux_z[ID]*dtdz;
	    uOut[IP] += flux_z[IP]*dtdz;
	    uOut[IU] += (alpha1*flux_z[IW] +     alpha2*flux_z[IV])*dtdz;
	    uOut[IV] += (alpha1*flux_z[IV] -0.25*alpha2*flux_z[IW])*dtdz;
	    uOut[IW] += flux_z[IU]*dtdz;
	  } else {
	    uOut[ID] += flux_z[ID]*dtdz;
	    uOut[IP] += flux_z[IP]*dtdz;
	    uOut[IU] += flux_z[IW]*dtdz;
	    uOut[IV] += flux_z[IV]*dtdz;
	    uOut[IW] += flux_z[IU]*dtdz;
	  }
	  
	  // actually perform the update on external device memory
	  Uout[offset] = uOut[ID];  offset += arraySize;
	  Uout[offset] = uOut[IP];  offset += arraySize;
	  Uout[offset] = uOut[IU];  offset += arraySize;
	  Uout[offset] = uOut[IV];  offset += arraySize;
	  Uout[offset] = uOut[IW];
	}

	if (k>3) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offset = elemOffset - pitch*jmax;
	  if (Omega0>0) {
	    Uout[offset] -= flux_z[ID]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IP]*dtdz; offset += arraySize;
	    Uout[offset] -= (alpha1*flux_z[IW]+alpha2*flux_z[IV])*dtdz; offset += arraySize;
	    Uout[offset] -= (alpha1*flux_z[IV]-0.25*alpha2*flux_z[IW])*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IU]*dtdz;
	  } else {
	    Uout[offset] -= flux_z[ID]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IP]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IW]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IV]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IU]*dtdz;	    
	  }
	}
      } // end update along Z
    __syncthreads();

  } // end for k

} // kernel_mhd_flux_update_hydro_v4


#ifdef USE_DOUBLE
#define COMPUTE_EMF_BLOCK_DIMX_3D_V4	16
#define COMPUTE_EMF_BLOCK_DIMY_3D_V4	16
#else // simple precision
#define COMPUTE_EMF_BLOCK_DIMX_3D_V4	16
#define COMPUTE_EMF_BLOCK_DIMY_3D_V4	16
#endif // USE_DOUBLE

/**
 * Compute emf's and store them in d_emf.
 *
 * In this kernel, all we do is reload qEdge data and compute emfX,
 * emfY and emfZ that will be used later for constraint transport
 * update in kernel_mhd_flux_update_ct_v4
 *
 * \note In kernel kernel_mhd_flux_update_hydro_v4_old, hydro update
 * and emf computation were done in the same kernel.
 *
 * \param[in] d_qEdge_RT 
 * \param[in] d_qEdge_RB 
 * \param[in] d_qEdge_LT 
 * \param[in] d_qEdge_LB 
 * \param[in] d_qEdge_RT2
 * \param[in] d_qEdge_RB2
 * \param[in] d_qEdge_LT2
 * \param[in] d_qEdge_LB2
 * \param[in] d_qEdge_RT3
 * \param[in] d_qEdge_RB3
 * \param[in] d_qEdge_LT3
 * \param[in] d_qEdge_LB3
 * \param[out] d_emf
 *
 */
__global__ void kernel_mhd_compute_emf_v4(const real_t * __restrict__ d_qEdge_RT,
					  const real_t * __restrict__ d_qEdge_RB,
					  const real_t * __restrict__ d_qEdge_LT,
					  const real_t * __restrict__ d_qEdge_LB,
					  const real_t * __restrict__ d_qEdge_RT2,
					  const real_t * __restrict__ d_qEdge_RB2,
					  const real_t * __restrict__ d_qEdge_LT2,
					  const real_t * __restrict__ d_qEdge_LB2,
					  const real_t * __restrict__ d_qEdge_RT3,
					  const real_t * __restrict__ d_qEdge_RB3,
					  const real_t * __restrict__ d_qEdge_LT3,
					  const real_t * __restrict__ d_qEdge_LB3,
					  real_t       *d_emf,
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
  
  const int i = __mul24(bx, COMPUTE_EMF_BLOCK_DIMX_3D_V4) + tx;
  const int j = __mul24(by, COMPUTE_EMF_BLOCK_DIMY_3D_V4) + ty;
  
  const int arraySize    = pitch * jmax * kmax;

  //real_t qEdge[4][3][NVAR_MHD];

  // // conservative variables
  // real_t uOut[NVAR_MHD];
  // real_t c;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  // // rotation rate
  // real_t &Omega0 = ::gParams.Omega0;

  // /*
  //  * shearing box correction on momentum parameters
  //  */
  // real_t lambda=0, ratio=1, alpha1=1, alpha2=0;
  // if (Omega0>0) {
  //   lambda = Omega0*dt;
  //   lambda = ONE_FOURTH_F * lambda * lambda;
  //   ratio  = (ONE_F-lambda)/(ONE_F+lambda);
  //   alpha1 =          ONE_F/(ONE_F+lambda);
  //   alpha2 =      Omega0*dt/(ONE_F+lambda);
  // }

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    /*
     * EMF computations and update face-centered magnetic field components.
     */

     // intermediate values for EMF computations
     real_t qEdge_emfX[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT)[NVAR_MHD] = qEdge_emfX[IRT];
     real_t (&qEdge_RB)[NVAR_MHD] = qEdge_emfX[IRB];
     real_t (&qEdge_LT)[NVAR_MHD] = qEdge_emfX[ILT];
     real_t (&qEdge_LB)[NVAR_MHD] = qEdge_emfX[ILB];
     
     real_t qEdge_emfY[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT2)[NVAR_MHD] = qEdge_emfY[IRT];
     real_t (&qEdge_RB2)[NVAR_MHD] = qEdge_emfY[IRB];
     real_t (&qEdge_LT2)[NVAR_MHD] = qEdge_emfY[ILT];
     real_t (&qEdge_LB2)[NVAR_MHD] = qEdge_emfY[ILB];
     
     real_t qEdge_emfZ[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT3)[NVAR_MHD] = qEdge_emfZ[IRT];
     real_t (&qEdge_RB3)[NVAR_MHD] = qEdge_emfZ[IRB];
     real_t (&qEdge_LT3)[NVAR_MHD] = qEdge_emfZ[ILT];
     real_t (&qEdge_LB3)[NVAR_MHD] = qEdge_emfZ[ILB];

     real_t emf;
     
     if(i > 1 and i < imax-2 and 
        j > 1 and j < jmax-2 and 
        k > 1 and k < kmax-2)
      {
	
	/*
	 * offset into external memory array to qEdge data
	 */
	
	int offset2         = elemOffset;
	
	
	/*
	 * compute emfZ
	 */

	// qEdge_RT3 at location x-1, y-1
	int offset = offset2-1-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT3[iVar] = d_qEdge_RT3[offset];
	  offset += arraySize;
	}
	
	// qEdge RB3 at location x-1, y
	offset = offset2-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB3[iVar] = d_qEdge_RB3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT3 at location x, y-1
	offset = offset2-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT3[iVar] = d_qEdge_LT3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB3 at location x, y
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB3[iVar] = d_qEdge_LB3[offset];
	  offset += arraySize;
	}

	// finally compute emfZ
	emf = compute_emf<EMFZ>(qEdge_emfZ,xPos);
	offset = offset2 + I_EMFZ*arraySize;
	d_emf[offset] = emf;

	/*
	 * compute emfY (take care RB and LT are swapped)
	 */

	// qEdge_RT2 at location x-1, z-1
	offset = offset2 - pitch*jmax - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT2[iVar] = d_qEdge_RT2[offset];
	  offset += arraySize;
	}
	
	// qEdge RB2 (actually LT2) at location x, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB2[iVar] = d_qEdge_LT2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT2 (actually RB2) at location x-1, z
	offset = offset2 - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT2[iVar] = d_qEdge_RB2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB2 at location x, y
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB2[iVar] = d_qEdge_LB2[offset];
	  offset += arraySize;
	}

	// finally compute emfY
	emf = compute_emf<EMFY>(qEdge_emfY,xPos);
	offset = offset2 + I_EMFY*arraySize;
	d_emf[offset] = emf;

	/*
	 * compute emfX
	 */
	// qEdge_RT at location y-1, z-1
	offset = offset2 - pitch*jmax - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT[iVar] = d_qEdge_RT[offset];
	  offset += arraySize;
	}
	
	// qEdge RB at location y-1, z
	offset = offset2 - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB[iVar] = d_qEdge_RB[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT at location y, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT[iVar] = d_qEdge_LT[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB at location y, z
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB[iVar] = d_qEdge_LB[offset];
	  offset += arraySize;
	}

	// finally compute emfX
	emf = compute_emf<EMFX>(qEdge_emfX,xPos);
	offset = offset2 + I_EMFX*arraySize;
	d_emf[offset] = emf;
	
      }
    __syncthreads();
            
  } // end for k

} // kernel_mhd_compute_emf_v4


#ifdef USE_DOUBLE
#define UPDATE_CT_BLOCK_DIMX_3D_V4	16
#define UPDATE_CT_BLOCK_DIMY_3D_V4	16
#else // simple precision
#define UPDATE_CT_BLOCK_DIMX_3D_V4	16
#define UPDATE_CT_BLOCK_DIMY_3D_V4	16
#endif // USE_DOUBLE

/**
 * Update MHD conservative variables (implementation version 4).
 * 
 * This is the final kernel, that given the emf's compute
 * performs magnetic field update, i.e. constraint trasport update
 * (similar to routine ct in Dumses).
 *
 * \param[in]  Uin  input MHD conservative variable array
 * \param[out] Uout ouput MHD conservative variable array
 * \param[in]  d_emf emf input array
 *
 */
__global__ void kernel_mhd_flux_update_ct_v4(const real_t * __restrict__ Uin, 
					     real_t       *Uout,
					     const real_t * __restrict__ d_emf,
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
  
  const int i = __mul24(bx, UPDATE_CT_BLOCK_DIMX_3D_V4) + tx;
  const int j = __mul24(by, UPDATE_CT_BLOCK_DIMY_3D_V4) + ty;
  
  const int arraySize  = pitch * jmax * kmax;
  const int ghostWidth = 3; // MHD

  // conservative variables
  real_t uOut[NVAR_MHD];

   /*
    * loop over k (i.e. z) to perform constraint transport update
    */
  for (int k=ghostWidth, elemOffset = i + pitch * (j + jmax * ghostWidth);
       k < kmax-ghostWidth+1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i >= 3 and i < imax-2 and 
       j >= 3 and j < jmax-2 and 
       k >= 3 and k < kmax-2)
      {
	// First : update at current location x,y,z
	int offset;
  	offset = elemOffset + 5 * arraySize;
	
	// read magnetic field components from external memory
	uOut[IA] = Uin[offset];  offset += arraySize;
	uOut[IB] = Uin[offset];  offset += arraySize;
	uOut[IC] = Uin[offset];
	
	// indexes used to fetch emf's at the right location
	const int ijk   = elemOffset;
	const int ip1jk = ijk + 1;
	const int ijp1k = ijk + pitch;
	const int ijkp1 = ijk + pitch * jmax;

	if (k<kmax-3) { // EMFZ update
	  uOut[IA] += ( d_emf[ijp1k + I_EMFZ*arraySize] - 
			d_emf[ijk   + I_EMFZ*arraySize] ) * dtdy;
	  
	  uOut[IB] -= ( d_emf[ip1jk + I_EMFZ*arraySize] - 
			d_emf[ijk   + I_EMFZ*arraySize] ) * dtdx;
	  
	}
	
	// update BX
	uOut[IA] -= ( d_emf[ijkp1 + I_EMFY*arraySize] -
		      d_emf[ijk   + I_EMFY*arraySize] ) * dtdz;
	
	// update BY
	uOut[IB] += ( d_emf[ijkp1 + I_EMFX*arraySize] -
		      d_emf[ijk   + I_EMFX*arraySize] ) * dtdz;
	
	// update BZ
	uOut[IC] += ( d_emf[ip1jk + I_EMFY*arraySize] -
		      d_emf[ijk   + I_EMFY*arraySize] ) * dtdx;
	uOut[IC] -= ( d_emf[ijp1k + I_EMFX*arraySize] -
		      d_emf[ijk   + I_EMFX*arraySize] ) * dtdy;
	
	// write back mag field components in external memory
	offset = elemOffset + 5 * arraySize;
	
	Uout[offset] = uOut[IA];  offset += arraySize;
	Uout[offset] = uOut[IB];  offset += arraySize;
	Uout[offset] = uOut[IC];
	
      } // end if

  } // end for k

} // kernel_mhd_flux_update_ct_v4


/**
 * Update MHD conservative variables (same as kernel_mhd_flux_update_hydro_v4 but for shearing box)
 * 
 * This kernel performs conservative variables update everywhere except at X-border 
 * (which is done outside).
 *
 * Here we assume Omega is strictly positive.
 *
 * \see kernel_mhd_flux_update_hydro_v4
 *
 * \param[in]  Uin  input MHD conservative variable array
 * \param[out] Uout ouput MHD conservative variable array
 * \param[in] d_qm_x qm state along x
 * \param[in] d_qm_y qm state along y
 * \param[in] d_qm_z qm state along z
 * \param[in] d_qp_x qp state along x
 * \param[in] d_qp_y qp state along y
 * \param[in] d_qp_z qp state along z
 * \param[in] d_qEdge_RT 
 * \param[in] d_qEdge_RB 
 * \param[in] d_qEdge_LT 
 * \param[in] d_qEdge_LB 
 * \param[in] d_qEdge_RT2
 * \param[in] d_qEdge_RB2
 * \param[in] d_qEdge_LT2
 * \param[in] d_qEdge_LB2
 * \param[in] d_qEdge_RT3
 * \param[in] d_qEdge_RB3
 * \param[in] d_qEdge_LT3
 * \param[in] d_qEdge_LB3
 * \param[out] d_emf
 * \param[out] d_shear_flux_xmin
 * \param[out] d_shear_flux_xmax
 *
 */
__global__ void kernel_mhd_flux_update_hydro_v4_shear(const real_t * __restrict__ Uin, 
						      real_t       *Uout,
						      const real_t * __restrict__ d_qm_x,
						      const real_t * __restrict__ d_qm_y,
						      const real_t * __restrict__ d_qm_z,
						      const real_t * __restrict__ d_qp_x,
						      const real_t * __restrict__ d_qp_y,
						      const real_t * __restrict__ d_qp_z,
						      const real_t * __restrict__ d_qEdge_RT,
						      const real_t * __restrict__ d_qEdge_RB,
						      const real_t * __restrict__ d_qEdge_LT,
						      const real_t * __restrict__ d_qEdge_LB,
						      const real_t * __restrict__ d_qEdge_RT2,
						      const real_t * __restrict__ d_qEdge_RB2,
						      const real_t * __restrict__ d_qEdge_LT2,
						      const real_t * __restrict__ d_qEdge_LB2,
						      const real_t * __restrict__ d_qEdge_RT3,
						      const real_t * __restrict__ d_qEdge_RB3,
						      const real_t * __restrict__ d_qEdge_LT3,
						      const real_t * __restrict__ d_qEdge_LB3,
						      real_t       *d_emf,
						      real_t       *d_shear_flux_xmin,
						      real_t       *d_shear_flux_xmax,
						      int pitch, 
						      int imax, 
						      int jmax,
						      int kmax,
						      int pitchB,
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
  
  const int i = __mul24(bx, UPDATE_BLOCK_INNER_DIMX_3D_V4) + tx;
  const int j = __mul24(by, UPDATE_BLOCK_INNER_DIMY_3D_V4) + ty;
  
  const int arraySize    = pitch * jmax * kmax;
  const int arraySize2d  =       pitchB * kmax;

  // flux computation
  __shared__ real_t   flux[UPDATE_BLOCK_DIMX_3D_V4][UPDATE_BLOCK_DIMY_3D_V4][NVAR_MHD];

  // qm and qp's are output of the trace step
  //real_t qm [THREE_D][NVAR_MHD];
  //real_t qp [THREE_D][NVAR_MHD];
  //real_t qEdge[4][3][NVAR_MHD];

  // conservative variables
  real_t uOut[NVAR_MHD];
  //real_t c;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  // rotation rate
  real_t &Omega0 = ::gParams.Omega0;

  /*
   * shearing box correction on momentum parameters
   */
  real_t lambda=0, ratio=1, alpha1=1, alpha2=0;
  
  lambda = Omega0*dt;
  lambda = ONE_FOURTH_F * lambda * lambda;
  ratio  = (ONE_F-lambda)/(ONE_F+lambda);
  alpha1 =          ONE_F/(ONE_F+lambda);
  alpha2 =      Omega0*dt/(ONE_F+lambda);

  /*
   * Copy Uin into Uout (so that Uout has correct border for next kernel calls)
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < kmax; 
       ++k, elemOffset += (pitch*jmax)) {

    if (i<imax and j<jmax and 
	tx < UPDATE_BLOCK_DIMX_3D_V4-1 and 
	ty < UPDATE_BLOCK_DIMY_3D_V4-1) {
      int offset = elemOffset;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];
    }
  } // end copy Uin into Uout

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use flux as flux_x
    real_t (&flux_x)[UPDATE_BLOCK_DIMX_3D_V4][UPDATE_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    flux_x[tx][ty][IA] = ZERO_F;
    flux_x[tx][ty][IB] = ZERO_F;
    flux_x[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and
       j >= 3 and j < jmax-2 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t   qleft_x [NVAR_MHD];
	real_t   qright_x[NVAR_MHD];
	
	// set qleft_x by re-reading qm_x from external memory at location x-1
	int offset = elemOffset-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_x[iVar] = d_qm_x[offset];
	  offset += arraySize;
	}

	// set qright_x by re-reading qp_x from external memory at location x
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_x[iVar] = d_qp_x[offset];
	  offset += arraySize;
	}

	riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i >= 3 and i < imax-3 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-3)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offset = elemOffset;
	uOut[ID] = Uin[offset];  offset += arraySize;
	uOut[IP] = Uin[offset];  offset += arraySize;
	uOut[IU] = Uin[offset];  offset += arraySize;
	uOut[IV] = Uin[offset];  offset += arraySize;
	uOut[IW] = Uin[offset];  offset += arraySize;
	uOut[IA] = Uin[offset];  offset += arraySize;
	uOut[IB] = Uin[offset];  offset += arraySize;
	uOut[IC] = Uin[offset];

	// we need a special treatment at XMIN and XMAX for density update with flux_x
	int mask_XMIN = 1;
	int mask_XMAX = 1;
	if (i==3      and (::gParams.mpiPosX == 0                 ) ) mask_XMIN = 0; // prevent update at XMIN
	if (i==imax-4 and (::gParams.mpiPosX == (::gParams.mx -1) ) ) mask_XMAX = 0; // prevent update at XMAX

	// rotating frame corrections
	{
	  real_t dsx =   TWO_F * Omega0 * dt * uOut[IV]/(ONE_F + lambda);
	  real_t dsy = -HALF_F * Omega0 * dt * uOut[IU]/(ONE_F + lambda);
	  uOut[IU] = uOut[IU]*ratio + dsx;
	  uOut[IV] = uOut[IV]*ratio + dsy;
	}

	{
	  uOut[ID] += (mask_XMIN*flux_x[tx  ][ty][ID]-
		       mask_XMAX*flux_x[tx+1][ty][ID])*dtdx;
	  
	  uOut[IP] += (flux_x[tx  ][ty][IP]-
		       flux_x[tx+1][ty][IP])*dtdx;
	  
	  uOut[IU] += ( alpha1*(flux_x[tx  ][ty][IU]-
				flux_x[tx+1][ty][IU]) + 
			alpha2*(flux_x[tx  ][ty][IV]-
				flux_x[tx+1][ty][IV]) )*dtdx;
	  
	  uOut[IV] += ( alpha1*(flux_x[tx  ][ty][IV]-
				flux_x[tx+1][ty][IV]) - 0.25*
			alpha2*(flux_x[tx  ][ty][IU]-
				flux_x[tx+1][ty][IU]) )*dtdx;
	  
	  uOut[IW] += (flux_x[tx  ][ty][IW]-
		       flux_x[tx+1][ty][IW])*dtdx;
	} 

	if (i==3 and ::gParams.mpiPosX == 0) {
	  /* store flux_xmin */
	  int offsetShear = j+pitchB*k;
	  d_shear_flux_xmin[offsetShear] = flux_x[tx  ][ty][ID]*dtdx; // I_DENS
	}
	if (i==imax-4 and ::gParams.mpiPosX == (::gParams.mx - 1) ) {
	  /* store flux_xmax */
	  int offsetShear = j+pitchB*k;
	  d_shear_flux_xmax[offsetShear] = flux_x[tx+1][ty][ID]*dtdx; // I_DENS	  
	}
	
      }
    __syncthreads();

    /*
     * Compute fluxes at Y-interfaces.
     */
    // re-use flux as flux_y
    real_t (&flux_y)[UPDATE_BLOCK_DIMX_3D_V4][UPDATE_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    flux_y[tx][ty][IA] = ZERO_F;
    flux_y[tx][ty][IB] = ZERO_F;
    flux_y[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and 
       j >= 3 and j < jmax-2 and 
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t  qleft_y[NVAR_MHD];
	real_t qright_y[NVAR_MHD];
	
	// set qleft_y by reading qm_y from external memory at location y-1
	int offset = elemOffset-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_y[iVar] = d_qm_y[offset];
	  offset += arraySize;
	}
	
	// set qright_y by reading qp_y from external memory at location y
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_y[iVar] = d_qp_y[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IV
	swap_value_(qleft_y[IU],qleft_y[IV]);
	swap_value_(qleft_y[IA],qleft_y[IB]);
	swap_value_(qright_y[IU],qright_y[IV]);
	swap_value_(qright_y[IA],qright_y[IB]);

	riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);

	/* shear correction on flux_y */
	/* cartesian and ::gParams.Omega0 > 0  and not fargo */
	{
	  real_t shear_y = -1.5 * ::gParams.Omega0 * xPos;
	  real_t eMag, eKin, eTot;
	  real_t bn_mean = HALF_F * (qleft_y[IA] + qright_y[IA]);
	  real_t &gamma  = ::gParams.gamma0;
	  
	  if (shear_y > 0) {
	    eMag = HALF_F * (qleft_y[IA]*qleft_y[IA] + 
			     qleft_y[IB]*qleft_y[IB] + 
			     qleft_y[IC]*qleft_y[IC]);

	    eKin = HALF_F * (qleft_y[IU]*qleft_y[IU] + 
			     qleft_y[IV]*qleft_y[IV] + 
			     qleft_y[IW]*qleft_y[IW]);

	    eTot = eKin + eMag + qleft_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qleft_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qleft_y[ID]*qleft_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qleft_y[ID]*qleft_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qleft_y[ID]*qleft_y[IW];
	  } else {
	    eMag = HALF_F * (qright_y[IA]*qright_y[IA] + 
			     qright_y[IB]*qright_y[IB] + 
			     qright_y[IC]*qright_y[IC]);

	    eKin = HALF_F * (qright_y[IU]*qright_y[IU] + 
			     qright_y[IV]*qright_y[IV] + 
			     qright_y[IW]*qright_y[IW]);

	    eTot = eKin + eMag + qright_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qright_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qright_y[ID]*qright_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qright_y[ID]*qright_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qright_y[ID]*qright_y[IW];		  
	  }
	} // end shear correction on flux_y

      }  
    __syncthreads();
    
    // update uOut with flux_y
    if(i >= 3 and i < imax-3 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-3)
      {
	// watchout IU and IV are swapped !

	{
	  uOut[ID] += (flux_y[tx][ty  ][ID]-
		       flux_y[tx][ty+1][ID])*dtdy;
	  
	  uOut[IP] += (flux_y[tx][ty  ][IP]-
		       flux_y[tx][ty+1][IP])*dtdy;
	  
	  uOut[IU] += ( alpha1*(flux_y[tx][ty  ][IV]-
				flux_y[tx][ty+1][IV]) + 
			alpha2*(flux_y[tx][ty  ][IU]-
				flux_y[tx][ty+1][IU]) )*dtdy;
	  
	  uOut[IV] += ( alpha1*(flux_y[tx][ty  ][IU]-
				flux_y[tx][ty+1][IU]) - 0.25*
			alpha2*(flux_y[tx][ty  ][IV]-
				flux_y[tx][ty+1][IV]) )*dtdy;
	  
	  uOut[IW] += (flux_y[tx][ty  ][IW]-
		       flux_y[tx][ty+1][IW])*dtdy;
	}
      }
    __syncthreads();
    
    /*
     * Compute fluxes at Z-interfaces.
     */
    real_t flux_z[NVAR_MHD];
    flux_z[ID] = ZERO_F;
    flux_z[IP] = ZERO_F;
    flux_z[IU] = ZERO_F;
    flux_z[IV] = ZERO_F;
    flux_z[IW] = ZERO_F;
    flux_z[IA] = ZERO_F;
    flux_z[IB] = ZERO_F;
    flux_z[IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t qleft_z [NVAR_MHD];
	real_t qright_z[NVAR_MHD];
	
	// set qleft_z by reading qm_z from external memory at location z-1
	int offset = elemOffset - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_z[iVar] = d_qm_z[offset];
	  offset += arraySize;
	}
	
        // set qright_z by reading qp_z from external memory at location z
        offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_z[iVar] = d_qp_z[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IW
	swap_value_(qleft_z[IU] ,qleft_z[IW]);
	swap_value_(qleft_z[IA] ,qleft_z[IC]);
	swap_value_(qright_z[IU],qright_z[IW]);
	swap_value_(qright_z[IA],qright_z[IC]);
	
	riemann_mhd(qleft_z, qright_z, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i >= 3 and i < imax-3 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-2)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	int offset = elemOffset;

	if (k < kmax-3) {
	  // watchout IU and IW are swapped !

	  {
	    uOut[ID] += flux_z[ID]*dtdz;
	    uOut[IP] += flux_z[IP]*dtdz;
	    uOut[IU] += (alpha1*flux_z[IW] +     
			 alpha2*flux_z[IV])*dtdz;
	    uOut[IV] += (alpha1*flux_z[IV] -0.25*
			 alpha2*flux_z[IW])*dtdz;
	    uOut[IW] += flux_z[IU]*dtdz;
	  }
	  
	  // actually perform the update on external device memory
	  Uout[offset] = uOut[ID];  offset += arraySize;
	  Uout[offset] = uOut[IP];  offset += arraySize;
	  Uout[offset] = uOut[IU];  offset += arraySize;
	  Uout[offset] = uOut[IV];  offset += arraySize;
	  Uout[offset] = uOut[IW];
	}

	if (k>3) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offset = elemOffset - pitch*jmax;
	  {
	    Uout[offset] -= flux_z[ID]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IP]*dtdz; offset += arraySize;
	    Uout[offset] -= (alpha1*flux_z[IW]+     
			     alpha2*flux_z[IV])*dtdz; offset += arraySize;
	    Uout[offset] -= (alpha1*flux_z[IV]-0.25*
			     alpha2*flux_z[IW])*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IU]*dtdz;
	  } 
	}
      } // end update along Z
    __syncthreads();


    /*
     * EMF computations and update face-centered magnetic field components.
     */

     // intermediate values for EMF computations
     real_t qEdge_emfX[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT)[NVAR_MHD] = qEdge_emfX[IRT];
     real_t (&qEdge_RB)[NVAR_MHD] = qEdge_emfX[IRB];
     real_t (&qEdge_LT)[NVAR_MHD] = qEdge_emfX[ILT];
     real_t (&qEdge_LB)[NVAR_MHD] = qEdge_emfX[ILB];
     
     real_t qEdge_emfY[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT2)[NVAR_MHD] = qEdge_emfY[IRT];
     real_t (&qEdge_RB2)[NVAR_MHD] = qEdge_emfY[IRB];
     real_t (&qEdge_LT2)[NVAR_MHD] = qEdge_emfY[ILT];
     real_t (&qEdge_LB2)[NVAR_MHD] = qEdge_emfY[ILB];
     
     real_t qEdge_emfZ[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT3)[NVAR_MHD] = qEdge_emfZ[IRT];
     real_t (&qEdge_RB3)[NVAR_MHD] = qEdge_emfZ[IRB];
     real_t (&qEdge_LT3)[NVAR_MHD] = qEdge_emfZ[ILT];
     real_t (&qEdge_LB3)[NVAR_MHD] = qEdge_emfZ[ILB];

    // // re-use flux as emf
    // real_t (&emf)[UPDATE_BLOCK_DIMX_3D_V4][UPDATE_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    // emf[tx][ty][IX] = ZERO_F; // emfX
    // emf[tx][ty][IY] = ZERO_F; // emfY
    // emf[tx][ty][IZ] = ZERO_F; // emfZ

     real_t emf;

    if(i > 2 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
       j > 2 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
       k > 2 and k < kmax-2)
      {
	
	/*
	 * offset into external memory array to qEdge data
	 */

	int offset2         = elemOffset;
 
	
	/*
	 * compute emfZ
	 */

	// qEdge_RT3 at location x-1, y-1
	int offset = offset2-1-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT3[iVar] = d_qEdge_RT3[offset];
	  offset += arraySize;
	}
	
	// qEdge RB3 at location x-1, y
	offset = offset2-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB3[iVar] = d_qEdge_RB3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT3 at location x, y-1
	offset = offset2-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT3[iVar] = d_qEdge_LT3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB3 at location x, y
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB3[iVar] = d_qEdge_LB3[offset];
	  offset += arraySize;
	}

	// finally compute emfZ
	emf = compute_emf<EMFZ>(qEdge_emfZ,xPos);
	offset = offset2 + I_EMFZ*arraySize;
	if (k<kmax-3)
	  d_emf[offset] = emf;

	/*
	 * compute emfY (take care RB and LT are swapped)
	 */

	// qEdge_RT2 at location x-1, z-1
	offset = offset2 - pitch*jmax - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT2[iVar] = d_qEdge_RT2[offset];
	  offset += arraySize;
	}
	
	// qEdge RB2 (actually LT2) at location x, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB2[iVar] = d_qEdge_LT2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT2 (actually RB2) at location x-1, z
	offset = offset2 - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT2[iVar] = d_qEdge_RB2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB2 at location x, y
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB2[iVar] = d_qEdge_LB2[offset];
	  offset += arraySize;
	}

	// finally compute emfY
	emf = compute_emf<EMFY>(qEdge_emfY,xPos);
	offset = offset2 + I_EMFY*arraySize;
	if (j<jmax-3) {
	  d_emf[offset] = emf;
	  
	  // at global XMIN border, store emfY
	  if (i == 3     and ::gParams.mpiPosX == 0) {
	    int offsetShear = j+pitchB*k;
	    d_shear_flux_xmin[offsetShear + arraySize2d*I_EMF_Y] = emf;
	  }
	  
	  // at global XMAX border, store emfY
	  if (i == imax-3 and ::gParams.mpiPosX == (::gParams.mx - 1) ) {
	    int offsetShear = j+pitchB*k;
	    d_shear_flux_xmax[offsetShear + arraySize2d*I_EMF_Y] = emf;
	  }
	}

	/*
	 * compute emfX
	 */
	// qEdge_RT at location y-1, z-1
	offset = offset2 - pitch*jmax - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT[iVar] = d_qEdge_RT[offset];
	  offset += arraySize;
	}
	
	// qEdge RB at location y-1, z
	offset = offset2 - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB[iVar] = d_qEdge_RB[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT at location y, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT[iVar] = d_qEdge_LT[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB at location y, z
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB[iVar] = d_qEdge_LB[offset];
	  offset += arraySize;
	}

	// finally compute emfX
	emf = compute_emf<EMFX>(qEdge_emfX,xPos);
	offset = offset2 + I_EMFX*arraySize;
	if (i<imax-3)
	  d_emf[offset] = emf;	
      }
    __syncthreads();
    
    // if(i >= 3 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V4-1 and
    //    j >= 3 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V4-1 and
    //    k >= 3 and k < kmax-2)
    //   {
    // 	// actually perform update on external device memory
	
    // 	// First : update at current location x,y,z
    // 	int offset;
    // 	if (k<kmax-3) {
	  
    // 	  offset = elemOffset + 5 * arraySize;
	  
    // 	  // probably don't need to re-read
    // 	  uOut[IA] = Uin[offset];  offset += arraySize;
    // 	  uOut[IB] = Uin[offset];  offset += arraySize;
    // 	  uOut[IC] = Uin[offset];
	  
    // 	  // store what's need to be stored at XMIN / XMAX borders

    // 	  if (i==3) {
    // 	    int offsetShear = j+pitchB*k;
    // 	    d_shear_flux_xmin[offsetShear + arraySize2d*I_EMF_Y]       = emf[tx][ty][IY];
    // 	    d_shear_flux_xmin[offsetShear + arraySize2d*I_EMF_Y_REMAP] = emf[tx][ty][IY];
    // 	  }	  
	  
    // 	  if (i==imax-4) {
    // 	    int offsetShear = j+pitchB*k;
    // 	    d_shear_flux_xmax[offsetShear + arraySize2d*I_EMF_Y]       = emf[tx+1][ty][IY];
    // 	    d_shear_flux_xmax[offsetShear + arraySize2d*I_EMF_Y_REMAP] = emf[tx+1][ty][IY];
    // 	    d_shear_flux_xmax[offsetShear + arraySize2d*I_EMF_Z      ] = emf[tx+1][ty][IZ];
    // 	  } 


    // 	  // update bx
    // 	  uOut[IA] -= emf[tx  ][ty  ][IZ]*dtdy;
    // 	  uOut[IA] += emf[tx  ][ty+1][IZ]*dtdy;  
    // 	  if (i==3) {
    // 	    // do nothing
    // 	  } else {
    // 	    uOut[IA] += emf[tx  ][ty  ][IY]*dtdz;
    // 	  }

    // 	  // update by
    // 	  uOut[IB] += emf[tx  ][ty  ][IZ]*dtdx;
    // 	  uOut[IB] -= emf[tx+1][ty  ][IZ]*dtdx;
    // 	  uOut[IB] -= emf[tx  ][ty  ][IX]*dtdz;
	  
    // 	  // update bz
    // 	  if (i==imax-4) {
    // 	    // do nothing
    // 	  } else {
    // 	    uOut[IC] += emf[tx+1][ty  ][IY]*dtdx;
    // 	  }
    // 	  if (i==3) {
    // 	    // do nothing
    // 	  } else {
    // 	    uOut[IC] -= emf[tx  ][ty  ][IY]*dtdx;
    // 	  }
    // 	  uOut[IC] -= emf[tx  ][ty+1][IX]*dtdy;
    // 	  uOut[IC] += emf[tx  ][ty  ][IX]*dtdy;

    // 	  // write buffer
    // 	  offset = elemOffset + 5 * arraySize;
    // 	  if (i<imax-3 and j<jmax-3) {
    // 	    Uout[offset] = uOut[IA]; offset += arraySize;
    // 	    Uout[offset] = uOut[IB]; offset += arraySize;
    // 	    Uout[offset] = uOut[IC];
    // 	  }
    // 	} // end if (k<kmax-3)

    // 	// Second : update at z-1 ! Take care not using Uin here,
    // 	// since Uout already been updated at z-1 in the previous step
    // 	// !!! 
    // 	if (k>3) {
    // 	  if (i<imax-3 and j<jmax-3) {

    // 	    offset = elemOffset - pitch*jmax + 5 * arraySize;
    // 	    if (i==3) {
    // 	      // do nothing
    // 	    } else {
    // 	      Uout[offset] -= emf[tx][ty][IY]*dtdz; // update bx
    // 	    }
    // 	    offset += arraySize;
    // 	    Uout[offset] += emf[tx][ty][IX]*dtdz; // update by

    // 	  } // end if (i<imax-3 and j<jmax-3)

    // 	} // end (k>3)
	
    //   }
    // __syncthreads();
        
  } // end for k

} // kernel_mhd_flux_update_hydro_v4_shear


#ifdef USE_DOUBLE
#define UPDATE_P1_BLOCK_DIMX_3D_V4	16
#define UPDATE_P1_BLOCK_INNER_DIMX_3D_V4	(UPDATE_P1_BLOCK_DIMX_3D_V4-1)
#define UPDATE_P1_BLOCK_DIMY_3D_V4	8
#define UPDATE_P1_BLOCK_INNER_DIMY_3D_V4	(UPDATE_P1_BLOCK_DIMY_3D_V4-1)
#else // simple precision
#define UPDATE_P1_BLOCK_DIMX_3D_V4	16
#define UPDATE_P1_BLOCK_INNER_DIMX_3D_V4	(UPDATE_P1_BLOCK_DIMX_3D_V4-1)
#define UPDATE_P1_BLOCK_DIMY_3D_V4	8
#define UPDATE_P1_BLOCK_INNER_DIMY_3D_V4	(UPDATE_P1_BLOCK_DIMY_3D_V4-1)
#endif // USE_DOUBLE


/**
 * Update MHD conservative variables (same as kernel_mhd_flux_update_hydro_v4 but for shearing box)
 * 
 * This kernel performs conservative variables update everywhere except at X-border 
 * (which is done outside).
 *
 * Here we assume Omega is strictly positive.
 *
 * \see kernel_mhd_flux_update_hydro_v4
 *
 * \param[in]  Uin  input MHD conservative variable array
 * \param[out] Uout ouput MHD conservative variable array
 * \param[in] d_qm_x qm state along x
 * \param[in] d_qm_y qm state along y
 * \param[in] d_qm_z qm state along z
 * \param[in] d_qp_x qp state along x
 * \param[in] d_qp_y qp state along y
 * \param[in] d_qp_z qp state along z
 * \param[out] d_shear_flux_xmin
 * \param[out] d_shear_flux_xmax
 *
 */
__global__ void kernel_mhd_flux_update_hydro_v4_shear_part1(const real_t * __restrict__ Uin, 
							    real_t       *Uout,
							    const real_t * __restrict__ d_qm_x,
							    const real_t * __restrict__ d_qm_y,
							    const real_t * __restrict__ d_qm_z,
							    const real_t * __restrict__ d_qp_x,
							    const real_t * __restrict__ d_qp_y,
							    const real_t * __restrict__ d_qp_z,
							    real_t       *d_shear_flux_xmin,
							    real_t       *d_shear_flux_xmax,
							    int pitch, 
							    int imax, 
							    int jmax,
							    int kmax,
							    int pitchB,
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
  
  const int i = __mul24(bx, UPDATE_P1_BLOCK_INNER_DIMX_3D_V4) + tx;
  const int j = __mul24(by, UPDATE_P1_BLOCK_INNER_DIMY_3D_V4) + ty;
  
  const int arraySize    = pitch * jmax * kmax;
  //const int arraySize2d  =       pitchB * kmax;

  // flux computation
  __shared__ real_t   flux[UPDATE_P1_BLOCK_DIMX_3D_V4][UPDATE_P1_BLOCK_DIMY_3D_V4][NVAR_MHD];

  // conservative variables
  real_t uOut[NVAR_MHD];
  //real_t c;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  // rotation rate
  real_t &Omega0 = ::gParams.Omega0;

  /*
   * shearing box correction on momentum parameters
   */
  real_t lambda=0, ratio=1, alpha1=1, alpha2=0;
  
  lambda = Omega0*dt;
  lambda = ONE_FOURTH_F * lambda * lambda;
  ratio  = (ONE_F-lambda)/(ONE_F+lambda);
  alpha1 =          ONE_F/(ONE_F+lambda);
  alpha2 =      Omega0*dt/(ONE_F+lambda);

  /*
   * Copy Uin into Uout (so that Uout has correct border for next kernel calls)
   */
  for (int k=0, elemOffset = i + pitch * j;
       k < kmax; 
       ++k, elemOffset += (pitch*jmax)) {

    if (i<imax and j<jmax and 
	tx < UPDATE_P1_BLOCK_DIMX_3D_V4-1 and 
	ty < UPDATE_P1_BLOCK_DIMY_3D_V4-1) {
      int offset = elemOffset;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];  offset += arraySize;
      Uout[offset] = Uin[offset];
    }
  } // end copy Uin into Uout

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use flux as flux_x
    real_t (&flux_x)[UPDATE_P1_BLOCK_DIMX_3D_V4][UPDATE_P1_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    flux_x[tx][ty][IA] = ZERO_F;
    flux_x[tx][ty][IB] = ZERO_F;
    flux_x[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and
       j >= 3 and j < jmax-2 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t   qleft_x [NVAR_MHD];
	real_t   qright_x[NVAR_MHD];
	
	// set qleft_x by re-reading qm_x from external memory at location x-1
	int offset = elemOffset-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_x[iVar] = d_qm_x[offset];
	  offset += arraySize;
	}

	// set qright_x by re-reading qp_x from external memory at location x
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_x[iVar] = d_qp_x[offset];
	  offset += arraySize;
	}

	riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i >= 3 and i < imax-3 and tx < UPDATE_P1_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_P1_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-3)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offset = elemOffset;
	uOut[ID] = Uin[offset];  offset += arraySize;
	uOut[IP] = Uin[offset];  offset += arraySize;
	uOut[IU] = Uin[offset];  offset += arraySize;
	uOut[IV] = Uin[offset];  offset += arraySize;
	uOut[IW] = Uin[offset];  offset += arraySize;
	uOut[IA] = Uin[offset];  offset += arraySize;
	uOut[IB] = Uin[offset];  offset += arraySize;
	uOut[IC] = Uin[offset];

	// we need a special treatment at XMIN and XMAX for density update with flux_x
	int mask_XMIN = 1;
	int mask_XMAX = 1;
	if (i==3      and (::gParams.mpiPosX == 0                 ) ) mask_XMIN = 0; // prevent update at XMIN
	if (i==imax-4 and (::gParams.mpiPosX == (::gParams.mx -1) ) ) mask_XMAX = 0; // prevent update at XMAX

	// rotating frame corrections
	{
	  real_t dsx =   TWO_F * Omega0 * dt * uOut[IV]/(ONE_F + lambda);
	  real_t dsy = -HALF_F * Omega0 * dt * uOut[IU]/(ONE_F + lambda);
	  uOut[IU] = uOut[IU]*ratio + dsx;
	  uOut[IV] = uOut[IV]*ratio + dsy;
	}

	{
	  uOut[ID] += (mask_XMIN*flux_x[tx  ][ty][ID]-
		       mask_XMAX*flux_x[tx+1][ty][ID])*dtdx;
	  
	  uOut[IP] += (flux_x[tx  ][ty][IP]-
		       flux_x[tx+1][ty][IP])*dtdx;
	  
	  uOut[IU] += ( alpha1*(flux_x[tx  ][ty][IU]-
				flux_x[tx+1][ty][IU]) + 
			alpha2*(flux_x[tx  ][ty][IV]-
				flux_x[tx+1][ty][IV]) )*dtdx;
	  
	  uOut[IV] += ( alpha1*(flux_x[tx  ][ty][IV]-
				flux_x[tx+1][ty][IV]) - 0.25*
			alpha2*(flux_x[tx  ][ty][IU]-
				flux_x[tx+1][ty][IU]) )*dtdx;
	  
	  uOut[IW] += (flux_x[tx  ][ty][IW]-
		       flux_x[tx+1][ty][IW])*dtdx;
	} 

	if (i==3 and ::gParams.mpiPosX == 0) {
	  /* store flux_xmin */
	  int offsetShear = j+pitchB*k;
	  d_shear_flux_xmin[offsetShear] = flux_x[tx  ][ty][ID]*dtdx; // I_DENS
	}
	if (i==imax-4 and ::gParams.mpiPosX == (::gParams.mx - 1) ) {
	  /* store flux_xmax */
	  int offsetShear = j+pitchB*k;
	  d_shear_flux_xmax[offsetShear] = flux_x[tx+1][ty][ID]*dtdx; // I_DENS	  
	}
	
      }
    __syncthreads();

    /*
     * Compute fluxes at Y-interfaces.
     */
    // re-use flux as flux_y
    real_t (&flux_y)[UPDATE_P1_BLOCK_DIMX_3D_V4][UPDATE_P1_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    flux_y[tx][ty][IA] = ZERO_F;
    flux_y[tx][ty][IB] = ZERO_F;
    flux_y[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and 
       j >= 3 and j < jmax-2 and 
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t  qleft_y[NVAR_MHD];
	real_t qright_y[NVAR_MHD];
	
	// set qleft_y by reading qm_y from external memory at location y-1
	int offset = elemOffset-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_y[iVar] = d_qm_y[offset];
	  offset += arraySize;
	}
	
	// set qright_y by reading qp_y from external memory at location y
	offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_y[iVar] = d_qp_y[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IV
	swap_value_(qleft_y[IU],qleft_y[IV]);
	swap_value_(qleft_y[IA],qleft_y[IB]);
	swap_value_(qright_y[IU],qright_y[IV]);
	swap_value_(qright_y[IA],qright_y[IB]);

	riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);

	/* shear correction on flux_y */
	/* cartesian and ::gParams.Omega0 > 0  and not fargo */
	{
	  real_t shear_y = -1.5 * ::gParams.Omega0 * xPos;
	  real_t eMag, eKin, eTot;
	  real_t bn_mean = HALF_F * (qleft_y[IA] + qright_y[IA]);
	  real_t &gamma  = ::gParams.gamma0;
	  
	  if (shear_y > 0) {
	    eMag = HALF_F * (qleft_y[IA]*qleft_y[IA] + 
			     qleft_y[IB]*qleft_y[IB] + 
			     qleft_y[IC]*qleft_y[IC]);

	    eKin = HALF_F * (qleft_y[IU]*qleft_y[IU] + 
			     qleft_y[IV]*qleft_y[IV] + 
			     qleft_y[IW]*qleft_y[IW]);

	    eTot = eKin + eMag + qleft_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qleft_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qleft_y[ID]*qleft_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qleft_y[ID]*qleft_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qleft_y[ID]*qleft_y[IW];
	  } else {
	    eMag = HALF_F * (qright_y[IA]*qright_y[IA] + 
			     qright_y[IB]*qright_y[IB] + 
			     qright_y[IC]*qright_y[IC]);

	    eKin = HALF_F * (qright_y[IU]*qright_y[IU] + 
			     qright_y[IV]*qright_y[IV] + 
			     qright_y[IW]*qright_y[IW]);

	    eTot = eKin + eMag + qright_y[IP]/(gamma - ONE_F);
	    flux_y[tx][ty][ID] = flux_y[tx][ty][ID] + shear_y * qright_y[ID];
	    flux_y[tx][ty][IP] = flux_y[tx][ty][IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
	    flux_y[tx][ty][IU] = flux_y[tx][ty][IU] + shear_y * qright_y[ID]*qright_y[IU];
	    flux_y[tx][ty][IV] = flux_y[tx][ty][IV] + shear_y * qright_y[ID]*qright_y[IV];
	    flux_y[tx][ty][IW] = flux_y[tx][ty][IW] + shear_y * qright_y[ID]*qright_y[IW];		  
	  }
	} // end shear correction on flux_y

      }  
    __syncthreads();
    
    // update uOut with flux_y
    if(i >= 3 and i < imax-3 and tx < UPDATE_P1_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_P1_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-3)
      {
	// watchout IU and IV are swapped !

	{
	  uOut[ID] += (flux_y[tx][ty  ][ID]-
		       flux_y[tx][ty+1][ID])*dtdy;
	  
	  uOut[IP] += (flux_y[tx][ty  ][IP]-
		       flux_y[tx][ty+1][IP])*dtdy;
	  
	  uOut[IU] += ( alpha1*(flux_y[tx][ty  ][IV]-
				flux_y[tx][ty+1][IV]) + 
			alpha2*(flux_y[tx][ty  ][IU]-
				flux_y[tx][ty+1][IU]) )*dtdy;
	  
	  uOut[IV] += ( alpha1*(flux_y[tx][ty  ][IU]-
				flux_y[tx][ty+1][IU]) - 0.25*
			alpha2*(flux_y[tx][ty  ][IV]-
				flux_y[tx][ty+1][IV]) )*dtdy;
	  
	  uOut[IW] += (flux_y[tx][ty  ][IW]-
		       flux_y[tx][ty+1][IW])*dtdy;
	}
      }
    __syncthreads();
    
    /*
     * Compute fluxes at Z-interfaces.
     */
    real_t flux_z[NVAR_MHD];
    flux_z[ID] = ZERO_F;
    flux_z[IP] = ZERO_F;
    flux_z[IU] = ZERO_F;
    flux_z[IV] = ZERO_F;
    flux_z[IW] = ZERO_F;
    flux_z[IA] = ZERO_F;
    flux_z[IB] = ZERO_F;
    flux_z[IC] = ZERO_F;
    __syncthreads();
    
    if(i >= 3 and i < imax-2 and tx < UPDATE_P1_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-2 and ty < UPDATE_P1_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-2)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t qleft_z [NVAR_MHD];
	real_t qright_z[NVAR_MHD];
	
	// set qleft_z by reading qm_z from external memory at location z-1
	int offset = elemOffset - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_z[iVar] = d_qm_z[offset];
	  offset += arraySize;
	}
	
        // set qright_z by reading qp_z from external memory at location z
        offset = elemOffset;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_z[iVar] = d_qp_z[offset];
	  offset += arraySize;
	}
	
	// watchout swap IU and IW
	swap_value_(qleft_z[IU] ,qleft_z[IW]);
	swap_value_(qleft_z[IA] ,qleft_z[IC]);
	swap_value_(qright_z[IU],qright_z[IW]);
	swap_value_(qright_z[IA],qright_z[IC]);
	
	riemann_mhd(qleft_z, qright_z, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i >= 3 and i < imax-3 and tx < UPDATE_P1_BLOCK_DIMX_3D_V4-1 and
       j >= 3 and j < jmax-3 and ty < UPDATE_P1_BLOCK_DIMY_3D_V4-1 and
       k >= 3 and k < kmax-2)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	int offset = elemOffset;

	if (k < kmax-3) {
	  // watchout IU and IW are swapped !

	  {
	    uOut[ID] += flux_z[ID]*dtdz;
	    uOut[IP] += flux_z[IP]*dtdz;
	    uOut[IU] += (alpha1*flux_z[IW] +     
			 alpha2*flux_z[IV])*dtdz;
	    uOut[IV] += (alpha1*flux_z[IV] -0.25*
			 alpha2*flux_z[IW])*dtdz;
	    uOut[IW] += flux_z[IU]*dtdz;
	  }
	  
	  // actually perform the update on external device memory
	  Uout[offset] = uOut[ID];  offset += arraySize;
	  Uout[offset] = uOut[IP];  offset += arraySize;
	  Uout[offset] = uOut[IU];  offset += arraySize;
	  Uout[offset] = uOut[IV];  offset += arraySize;
	  Uout[offset] = uOut[IW];
	}

	if (k>3) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offset = elemOffset - pitch*jmax;
	  {
	    Uout[offset] -= flux_z[ID]*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IP]*dtdz; offset += arraySize;
	    Uout[offset] -= (alpha1*flux_z[IW]+     
			     alpha2*flux_z[IV])*dtdz; offset += arraySize;
	    Uout[offset] -= (alpha1*flux_z[IV]-0.25*
			     alpha2*flux_z[IW])*dtdz; offset += arraySize;
	    Uout[offset] -= flux_z[IU]*dtdz;
	  } 
	}
      } // end update along Z
    __syncthreads();

  } // end for k

} // kernel_mhd_flux_update_hydro_v4_shear_part1

#ifdef USE_DOUBLE
#define COMPUTE_EMF_BLOCK_DIMX_3D_SHEAR	16
#define COMPUTE_EMF_BLOCK_DIMY_3D_SHEAR	16
#else // simple precision
#define COMPUTE_EMF_BLOCK_DIMX_3D_SHEAR	16
#define COMPUTE_EMF_BLOCK_DIMY_3D_SHEAR	16
#endif // USE_DOUBLE

/**
 * MHD compute emf for shearing box simulations, store them in d_emf
 * 
 * \see kernel_mhd_flux_update_hydro_v4
 *
 * \param[in] d_qEdge_RT 
 * \param[in] d_qEdge_RB 
 * \param[in] d_qEdge_LT 
 * \param[in] d_qEdge_LB 
 * \param[in] d_qEdge_RT2
 * \param[in] d_qEdge_RB2
 * \param[in] d_qEdge_LT2
 * \param[in] d_qEdge_LB2
 * \param[in] d_qEdge_RT3
 * \param[in] d_qEdge_RB3
 * \param[in] d_qEdge_LT3
 * \param[in] d_qEdge_LB3
 * \param[out] d_emf
 * \param[out] d_shear_flux_xmin
 * \param[out] d_shear_flux_xmax
 *
 */
__global__ void kernel_mhd_compute_emf_shear(const real_t * __restrict__ d_qEdge_RT,
					     const real_t * __restrict__ d_qEdge_RB,
					     const real_t * __restrict__ d_qEdge_LT,
					     const real_t * __restrict__ d_qEdge_LB,
					     const real_t * __restrict__ d_qEdge_RT2,
					     const real_t * __restrict__ d_qEdge_RB2,
					     const real_t * __restrict__ d_qEdge_LT2,
					     const real_t * __restrict__ d_qEdge_LB2,
					     const real_t * __restrict__ d_qEdge_RT3,
					     const real_t * __restrict__ d_qEdge_RB3,
					     const real_t * __restrict__ d_qEdge_LT3,
					     const real_t * __restrict__ d_qEdge_LB3,
					     real_t       *d_emf,
					     real_t       *d_shear_flux_xmin,
					     real_t       *d_shear_flux_xmax,
					     int pitch, 
					     int imax, 
					     int jmax,
					     int kmax,
					     int pitchB,
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
  
  const int i = __mul24(bx, COMPUTE_EMF_BLOCK_DIMX_3D_SHEAR ) + tx;
  const int j = __mul24(by, COMPUTE_EMF_BLOCK_DIMY_3D_SHEAR ) + ty;
  
  const int arraySize    = pitch * jmax * kmax;
  const int arraySize2d  =       pitchB * kmax;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-2; 
       ++k, elemOffset += (pitch*jmax)) {
    
    /*
     * EMF computations and update face-centered magnetic field components.
     */

     // intermediate values for EMF computations
     real_t qEdge_emfX[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT)[NVAR_MHD] = qEdge_emfX[IRT];
     real_t (&qEdge_RB)[NVAR_MHD] = qEdge_emfX[IRB];
     real_t (&qEdge_LT)[NVAR_MHD] = qEdge_emfX[ILT];
     real_t (&qEdge_LB)[NVAR_MHD] = qEdge_emfX[ILB];
     
     real_t qEdge_emfY[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT2)[NVAR_MHD] = qEdge_emfY[IRT];
     real_t (&qEdge_RB2)[NVAR_MHD] = qEdge_emfY[IRB];
     real_t (&qEdge_LT2)[NVAR_MHD] = qEdge_emfY[ILT];
     real_t (&qEdge_LB2)[NVAR_MHD] = qEdge_emfY[ILB];
     
     real_t qEdge_emfZ[4][NVAR_MHD];  // used in compute emf 
     real_t (&qEdge_RT3)[NVAR_MHD] = qEdge_emfZ[IRT];
     real_t (&qEdge_RB3)[NVAR_MHD] = qEdge_emfZ[IRB];
     real_t (&qEdge_LT3)[NVAR_MHD] = qEdge_emfZ[ILT];
     real_t (&qEdge_LB3)[NVAR_MHD] = qEdge_emfZ[ILB];

     real_t emf;

    if(i > 2 and i < imax-2 and 
       j > 2 and j < jmax-2 and 
       k > 2 and k < kmax-2)
      {
	
	/*
	 * offset into external memory array to qEdge data
	 */

	int offset2         = elemOffset;
 
	
	/*
	 * compute emfZ
	 */

	// qEdge_RT3 at location x-1, y-1
	int offset = offset2-1-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT3[iVar] = d_qEdge_RT3[offset];
	  offset += arraySize;
	}
	
	// qEdge RB3 at location x-1, y
	offset = offset2-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB3[iVar] = d_qEdge_RB3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT3 at location x, y-1
	offset = offset2-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT3[iVar] = d_qEdge_LT3[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB3 at location x, y
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB3[iVar] = d_qEdge_LB3[offset];
	  offset += arraySize;
	}

	// finally compute emfZ
	emf = compute_emf<EMFZ>(qEdge_emfZ,xPos);
	offset = offset2 + I_EMFZ*arraySize;
	if (k<kmax-3)
	  d_emf[offset] = emf;

	/*
	 * compute emfY (take care RB and LT are swapped)
	 */

	// qEdge_RT2 at location x-1, z-1
	offset = offset2 - pitch*jmax - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT2[iVar] = d_qEdge_RT2[offset];
	  offset += arraySize;
	}
	
	// qEdge RB2 (actually LT2) at location x, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB2[iVar] = d_qEdge_LT2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT2 (actually RB2) at location x-1, z
	offset = offset2 - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT2[iVar] = d_qEdge_RB2[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB2 at location x, y
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB2[iVar] = d_qEdge_LB2[offset];
	  offset += arraySize;
	}

	// finally compute emfY
	emf = compute_emf<EMFY>(qEdge_emfY,xPos);
	offset = offset2 + I_EMFY*arraySize;
	if (j<jmax-3) {
	  d_emf[offset] = emf;
	  
	  // at global XMIN border, store emfY
	  if (i == 3     and ::gParams.mpiPosX == 0) {
	    int offsetShear = j+pitchB*k;
	    d_shear_flux_xmin[offsetShear + arraySize2d*I_EMF_Y] = emf;
	  }
	  
	  // at global XMAX border, store emfY
	  if (i == imax-3 and ::gParams.mpiPosX == (::gParams.mx - 1) ) {
	    int offsetShear = j+pitchB*k;
	    d_shear_flux_xmax[offsetShear + arraySize2d*I_EMF_Y] = emf;
	  }
	} // end if j<jmax-3

	/*
	 * compute emfX
	 */
	// qEdge_RT at location y-1, z-1
	offset = offset2 - pitch*jmax - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT[iVar] = d_qEdge_RT[offset];
	  offset += arraySize;
	}
	
	// qEdge RB at location y-1, z
	offset = offset2 - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB[iVar] = d_qEdge_RB[offset];
	  offset += arraySize;
	}
	
	// qEdge_LT at location y, z-1
	offset = offset2 - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT[iVar] = d_qEdge_LT[offset];
	  offset += arraySize;
	}
	
	// qEdge_LB at location y, z
	offset = offset2;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB[iVar] = d_qEdge_LB[offset];
	  offset += arraySize;
	}

	// finally compute emfX
	emf = compute_emf<EMFX>(qEdge_emfX,xPos);
	offset = offset2 + I_EMFX*arraySize;
	if (i<imax-3)
	  d_emf[offset] = emf;
      }
    __syncthreads();
            
  } // end for k

} // kernel_mhd_compute_emf_shear

#endif /* GODUNOV_UNSPLIT_MHD_CUH_ */
