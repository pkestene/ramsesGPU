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
 * \file godunov_unsplit_mhd_v0.cuh
 * \brief Defines the CUDA kernel for the actual MHD Godunov scheme.
 *
 * \date 18 Sept 2015
 * \author P. Kestener
 *
 */
#ifndef GODUNOV_UNSPLIT_MHD_V0_CUH_
#define GODUNOV_UNSPLIT_MHD_V0_CUH_

#include "real_type.h"
#include "constants.h"
#include "constoprim.h"
#include "riemann_mhd.h"
#include "trace_mhd.h"

#include <cstdlib>
#include <float.h>

/* a dummy device-only swap function */
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
__global__ void kernel_godunov_unsplit_mhd_2d_v0_old(const real_t * __restrict__ Uin, 
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

} // kernel_godunov_unsplit_2d_mhd_v0_old


#endif // GODUNOV_UNSPLIT_MHD_V0_CUH_
