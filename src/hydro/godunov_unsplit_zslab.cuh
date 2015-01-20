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
 * \file godunov_unsplit_zslab.cuh
 * \brief Defines the CUDA kernel for the unsplit Godunov scheme computations (z-slab method).
 *
 * \date 13 Sept 2012
 * \author P. Kestener
 *
 * $Id: godunov_unsplit_zslab.cuh 3450 2014-06-16 22:03:23Z pkestene $
 */
#ifndef GODUNOV_UNSPLIT_ZSLAB_CUH_
#define GODUNOV_UNSPLIT_ZSLAB_CUH_

#include "real_type.h"
#include "constants.h"
#include "base_type.h" // for qHydroState definition
#include "constoprim.h"
#include "riemann.h"
#include "trace.h"

#include "zSlabInfo.h"

#include <cstdlib>

/** a dummy device-only swap function */
__inline__ __device__ void swap_val(real_t& a, real_t& b) {
  
  real_t tmp = a;
  a = b;
  b = tmp;
   
} // swap_val

/*******************************************************
 *** COMPUTE PRIMITIVE VARIABLES 3D KERNEL version 1 ***
 *******************************************************/

#ifdef USE_DOUBLE
#define PRIM_VAR_BLOCK_DIMX_3D_V1Z	16
#define PRIM_VAR_BLOCK_DIMY_3D_V1Z	16
#else // simple precision
#define PRIM_VAR_BLOCK_DIMX_3D_V1Z	16
#define PRIM_VAR_BLOCK_DIMY_3D_V1Z	16
#endif // USE_DOUBLE

/**
 * Compute primitive variables 
 *
 * \param[in]  Uin  input  convervative variable array 
 * \param[out] Qout output primitive variable array
 */
__global__ void 
kernel_hydro_compute_primitive_variables_3D_v1_zslab(const real_t * __restrict__ Uin,
						     real_t       * Qout,
						     int pitch,
						     int imax, 
						     int jmax,
						     int kmax,
						     ZslabInfo zSlabInfo)
{
  
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, PRIM_VAR_BLOCK_DIMX_3D_V1Z) + tx;
  const int j = __mul24(by, PRIM_VAR_BLOCK_DIMY_3D_V1Z) + ty;
  
  const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;

  // conservative variables
  real_t uIn [NVAR_3D];
  real_t c;

  const int &kStart = zSlabInfo.kStart; 
  const int &kStop  = zSlabInfo.kStop; 

  /*
   * loop over k (i.e. z) to compute primitive variables, and store results
   * in external memory buffer Q.
   */
  for (int k=kStart, elemOffset = i + pitch * j + pitch*jmax*kStart;
       k < kStop; 
       ++k, elemOffset += (pitch*jmax)) {

    if (i < imax and j < jmax and k < kmax) {
      
      // Gather conservative variables (at z=k)
      int offsetU = elemOffset;
      
      uIn[ID] = Uin[offsetU];  offsetU += arraySizeU;
      uIn[IP] = Uin[offsetU];  offsetU += arraySizeU;
      uIn[IU] = Uin[offsetU];  offsetU += arraySizeU;
      uIn[IV] = Uin[offsetU];  offsetU += arraySizeU;
      uIn[IW] = Uin[offsetU];
    
      //Convert to primitive variables
      real_t qTmp[NVAR_3D];
      constoprim_3D(uIn, qTmp, c);
    
      // copy results into output d_Q at z=k
      int offsetQ = elemOffset - kStart*pitch*jmax;
      Qout[offsetQ] = qTmp[ID]; offsetQ += arraySizeQ;
      Qout[offsetQ] = qTmp[IP]; offsetQ += arraySizeQ;
      Qout[offsetQ] = qTmp[IU]; offsetQ += arraySizeQ;
      Qout[offsetQ] = qTmp[IV]; offsetQ += arraySizeQ;
      Qout[offsetQ] = qTmp[IW];

    } // end if

  } // enf for k

} // kernel_hydro_compute_primitive_variables_3D

/***********************************************
 *** COMPUTE TRACE 3D KERNEL - zslab version ***
 ***********************************************/

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
#define TRACE_BLOCK_DIMX_3D_V1Z	16
#define TRACE_BLOCK_INNER_DIMX_3D_V1Z	(TRACE_BLOCK_DIMX_3D_V1Z-2)
#define TRACE_BLOCK_DIMY_3D_V1Z	16
#define TRACE_BLOCK_INNER_DIMY_3D_V1Z	(TRACE_BLOCK_DIMY_3D_V1Z-2)
#else // simple precision
#define TRACE_BLOCK_DIMX_3D_V1Z	48
#define TRACE_BLOCK_INNER_DIMX_3D_V1Z	(TRACE_BLOCK_DIMX_3D_V1Z-2)
#define TRACE_BLOCK_DIMY_3D_V1Z	10
#define TRACE_BLOCK_INNER_DIMY_3D_V1Z	(TRACE_BLOCK_DIMY_3D_V1Z-2)
#endif // USE_DOUBLE

/**
 * Compute trace for hydro 3D - z-slab version.
 *
 * Output are all that is needed to compute fluxes.
 * \see kernel_hydro_compute_trace_unsplit_3d_v1
 *
 * All we do here is call :
 * - slope_unsplit_hydro_3d
 * - trace_unsplit_hydro_3d to get output : qm, qp.
 *
 * \param[in]  Uin    input conservative variable array
 * \param[in]  d_Q    input primitive    variable array
 * \param[out] d_qm_x qm state along x
 * \param[out] d_qm_y qm state along y
 * \param[out] d_qm_z qm state along z
 * \param[out] d_qp_x qp state along x
 * \param[out] d_qp_y qp state along y
 * \param[out] d_qp_z qp state along z
 *
 */
__global__ void 
kernel_hydro_compute_trace_unsplit_3d_v1_zslab(const real_t * __restrict__ Uin, 
					       const real_t * __restrict__ d_Q,
					       real_t* d_qm_x,
					       real_t* d_qm_y,
					       real_t* d_qm_z,
					       real_t* d_qp_x,
					       real_t* d_qp_y,
					       real_t* d_qp_z,
					       int pitch, 
					       int imax, 
					       int jmax, 
					       int kmax,
					       real_t dtdx,
					       real_t dtdy,
					       real_t dtdz,
					       real_t dt,
					       ZslabInfo zSlabInfo)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, TRACE_BLOCK_INNER_DIMX_3D_V1Z) + tx;
  const int j = __mul24(by, TRACE_BLOCK_INNER_DIMY_3D_V1Z) + ty;
  
  //const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;

  __shared__ real_t q[TRACE_BLOCK_DIMX_3D_V1Z][TRACE_BLOCK_DIMY_3D_V1Z][NVAR_3D];

  // qm and qp's are output of the trace step
  real_t qm [THREE_D][NVAR_3D];
  real_t qp [THREE_D][NVAR_3D];

  // primitive variables at different z
  real_t qZplus1  [NVAR_3D];
  real_t qZminus1 [NVAR_3D];

  const int &ksizeSlab = zSlabInfo.ksizeSlab;

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
	qZminus1[ID] = d_Q[offset];  offset += arraySizeQ;
	qZminus1[IP] = d_Q[offset];  offset += arraySizeQ;
	qZminus1[IU] = d_Q[offset];  offset += arraySizeQ;
	qZminus1[IV] = d_Q[offset];  offset += arraySizeQ;
	qZminus1[IW] = d_Q[offset];
      } else { // k == 1
	q[tx][ty][ID] = d_Q[offset];  offset += arraySizeQ;
	q[tx][ty][IP] = d_Q[offset];  offset += arraySizeQ;
	q[tx][ty][IU] = d_Q[offset];  offset += arraySizeQ;
	q[tx][ty][IV] = d_Q[offset];  offset += arraySizeQ;
	q[tx][ty][IW] = d_Q[offset];
      }

    } // end if
    __syncthreads();

  } // end for k - initialize q

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < ksizeSlab-1; 
       ++k, elemOffset += (pitch*jmax)) {
    
    if(i < imax and j < jmax) {
      // data fetch :
      // get q at z+1
      int offset = elemOffset + pitch*jmax; // z+1	 
      qZplus1[ID] = d_Q[offset];  offset += arraySizeQ;
      qZplus1[IP] = d_Q[offset];  offset += arraySizeQ;
      qZplus1[IU] = d_Q[offset];  offset += arraySizeQ;
      qZplus1[IV] = d_Q[offset];  offset += arraySizeQ;
      qZplus1[IW] = d_Q[offset];
    } // end if
    __syncthreads();
	     
    // slope and trace computation (i.e. dq, and then qm, qp)
    
    if(i > 0 and i < imax-1 and tx > 0 and tx < TRACE_BLOCK_DIMX_3D_V1Z-1 and
       j > 0 and j < jmax-1 and ty > 0 and ty < TRACE_BLOCK_DIMY_3D_V1Z-1)
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
	    
	    offset  += arraySizeQ;
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

} // kernel_hydro_compute_trace_unsplit_3d_v1_zslab

/***************************************************************
 *** UPDATE CONSERVATIVE VAR ARRAY 3D KERNEL - zslab version ***
 ***************************************************************/
#ifdef USE_DOUBLE
#define UPDATE_BLOCK_DIMX_3D_V1Z	16
#define UPDATE_BLOCK_INNER_DIMX_3D_V1Z	(UPDATE_BLOCK_DIMX_3D_V1Z-1)
#define UPDATE_BLOCK_DIMY_3D_V1Z	16
#define UPDATE_BLOCK_INNER_DIMY_3D_V1Z	(UPDATE_BLOCK_DIMY_3D_V1Z-1)
#else // simple precision
#define UPDATE_BLOCK_DIMX_3D_V1Z	48
#define UPDATE_BLOCK_INNER_DIMX_3D_V1Z	(UPDATE_BLOCK_DIMX_3D_V1Z-1)
#define UPDATE_BLOCK_DIMY_3D_V1Z	10
#define UPDATE_BLOCK_INNER_DIMY_3D_V1Z	(UPDATE_BLOCK_DIMY_3D_V1Z-1)
#endif // USE_DOUBLE

/**
 * Update hydro conservative variables 3D ( z-slab version ).
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
__global__ void 
kernel_hydro_flux_update_unsplit_3d_v1_zslab(const real_t * __restrict__ Uin, 
					     real_t* Uout,
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
					     real_t dt,
					     ZslabInfo zSlabInfo)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UPDATE_BLOCK_INNER_DIMX_3D_V1Z) + tx;
  const int j = __mul24(by, UPDATE_BLOCK_INNER_DIMY_3D_V1Z) + ty;
  
  const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;

  const int &ksizeSlab    = zSlabInfo.ksizeSlab;
  //const int &zSlabId      = zSlabInfo.zSlabId;
  //const int &zSlabNb      = zSlabInfo.zSlabNb;
  const int &kStart       = zSlabInfo.kStart; 

  // flux computation
  __shared__ real_t flux[UPDATE_BLOCK_DIMX_3D_V1Z][UPDATE_BLOCK_DIMY_3D_V1Z][NVAR_3D];

  // qm and qp's are output of the trace step
  //real_t qm [THREE_D][NVAR_3D];
  //real_t qp [THREE_D][NVAR_3D];
 
  // conservative variables
  real_t uOut[NVAR_3D];
  real_t qgdnv[NVAR_3D];
  //real_t c;

  //const int ghostWidth = 2;
  /*int ksizeSlabStopUpdate = ksizeSlab-ghostWidth;
    if (zSlabId == zSlabNb-1) ksizeSlabStopUpdate += 1;*/

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int k=2, elemOffsetQ = i + pitch * (j + jmax * 2);
       k < ksizeSlab-1; 
       ++k, elemOffsetQ += (pitch*jmax)) {
    
    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use flux as flux_x
    real_t (&flux_x)[UPDATE_BLOCK_DIMX_3D_V1Z][UPDATE_BLOCK_DIMY_3D_V1Z][NVAR_3D] = flux;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    __syncthreads();
    
    if(i >= 2 and i < imax-1 and
       j >= 2 and j < jmax-1 and
       k >= 2 and k < ksizeSlab-1)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t   qleft_x [NVAR_3D];
	real_t   qright_x[NVAR_3D];
	
	// set qleft_x by re-reading qm_x from external memory at location x-1
	int offsetQ = elemOffsetQ-1;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qleft_x[iVar] = d_qm_x[offsetQ];
	  offsetQ += arraySizeQ;
	}

	// set qright_x by re-reading qp_x from external memory at location x
	offsetQ = elemOffsetQ;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qright_x[iVar] = d_qp_x[offsetQ];
	  offsetQ += arraySizeQ;
	}

	riemann<NVAR_3D>(qleft_x, qright_x, qgdnv, flux_x[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i >= 2 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V1Z-1 and
       j >= 2 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V1Z-1 and
       k >= 2 and k < ksizeSlab-2)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offsetU = elemOffsetQ + pitch*jmax*kStart;
	uOut[ID] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IP] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IU] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IV] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IW] = Uin[offsetU];

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
    real_t (&flux_y)[UPDATE_BLOCK_DIMX_3D_V1Z][UPDATE_BLOCK_DIMY_3D_V1Z][NVAR_3D] = flux;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    __syncthreads();
    
    if(i >= 2 and i < imax-1 and 
       j >= 2 and j < jmax-1 and 
       k >= 2 and k < ksizeSlab-1)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t  qleft_y[NVAR_3D];
	real_t qright_y[NVAR_3D];
	
	// set qleft_y by reading qm_y from external memory at location y-1
	int offsetQ = elemOffsetQ-pitch;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qleft_y[iVar] = d_qm_y[offsetQ];
	  offsetQ += arraySizeQ;
	}
	
	// set qright_y by reading qp_y from external memory at location y
	offsetQ = elemOffsetQ;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qright_y[iVar] = d_qp_y[offsetQ];
	  offsetQ += arraySizeQ;
	}
	
	// watchout swap IU and IV
	swap_val(qleft_y[IU],qleft_y[IV]);
	swap_val(qright_y[IU],qright_y[IV]);

	riemann<NVAR_3D>(qleft_y, qright_y, qgdnv, flux_y[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_y
    if(i >= 2 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V1Z-1 and
       j >= 2 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V1Z-1 and
       k >= 2 and k < ksizeSlab-2)
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
    
    if(i >= 2 and i < imax-1 and tx < UPDATE_BLOCK_DIMX_3D_V1Z-1 and
       j >= 2 and j < jmax-1 and ty < UPDATE_BLOCK_DIMY_3D_V1Z-1 and
       k >= 2 and k < ksizeSlab-1)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t qleft_z [NVAR_3D];
	real_t qright_z[NVAR_3D];
	
	// set qleft_z by reading qm_z from external memory at location z-1
	int offsetQ = elemOffsetQ - pitch*jmax;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qleft_z[iVar] = d_qm_z[offsetQ];
	  offsetQ += arraySizeQ;
	}
	
        // set qright_z by reading qp_z from external memory at location z
        offsetQ = elemOffsetQ;
	for (int iVar=0; iVar<NVAR_3D; iVar++) {
	  qright_z[iVar] = d_qp_z[offsetQ];
	  offsetQ += arraySizeQ;
	}
	
	// watchout swap IU and IW
	swap_val(qleft_z[IU] ,qleft_z[IW]);
	swap_val(qright_z[IU],qright_z[IW]);
	
	riemann<NVAR_3D>(qleft_z, qright_z, qgdnv, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i >= 2 and i < imax-2 and tx < UPDATE_BLOCK_DIMX_3D_V1Z-1 and
       j >= 2 and j < jmax-2 and ty < UPDATE_BLOCK_DIMY_3D_V1Z-1 and
       k >= 2 and k < ksizeSlab-1)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

	/*
    	 * update current position z.
    	 */
	int offsetU = elemOffsetQ + kStart*pitch*jmax;

	if (k < ksizeSlab-2) {
	  // watchout IU and IW are swapped !
	  uOut[ID] += flux_z[ID]*dtdz;
	  uOut[IP] += flux_z[IP]*dtdz;
	  uOut[IU] += flux_z[IW]*dtdz;
	  uOut[IV] += flux_z[IV]*dtdz;
	  uOut[IW] += flux_z[IU]*dtdz;
	  
	  // actually perform the update on external device memory
	  Uout[offsetU] = uOut[ID];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IP];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IU];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IV];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IW];
	}

	if (k>2) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offsetU = elemOffsetQ + kStart*pitch*jmax - pitch*jmax;
	  Uout[offsetU] -= flux_z[ID]*dtdz; offsetU += arraySizeU;
	  Uout[offsetU] -= flux_z[IP]*dtdz; offsetU += arraySizeU;
	  Uout[offsetU] -= flux_z[IW]*dtdz; offsetU += arraySizeU;
	  Uout[offsetU] -= flux_z[IV]*dtdz; offsetU += arraySizeU;
	  Uout[offsetU] -= flux_z[IU]*dtdz;
	}

      } // end update along Z
    __syncthreads();

  } // end for k

} // kernel_hydro_flux_update_unsplit_3d_v1_zslab

#endif // GODUNOV_UNSPLIT_ZSLAB_CUH_
