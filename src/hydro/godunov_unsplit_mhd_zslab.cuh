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
 * \file godunov_unsplit_mhd_zslab.cuh
 * \brief Defines the CUDA kernel for the actual MHD Godunov scheme, z-slab method.
 *
 * \date Sept 19, 2012
 * \author P. Kestener
 *
 * $Id: godunov_unsplit_mhd_zslab.cuh 3450 2014-06-16 22:03:23Z pkestene $
 */
#ifndef GODUNOV_UNSPLIT_MHD_ZSLAB_CUH_
#define GODUNOV_UNSPLIT_MHD_ZSLAB_CUH_

#include "real_type.h"
#include "constants.h"
#include "constoprim.h"
#include "riemann_mhd.h"
#include "trace_mhd.h"

#include "zSlabInfo.h"

#include <cstdlib>
#include <float.h>

/** a dummy device-only swap function */
__device__ inline void swap_val_(real_t& a, real_t& b) {
  
  real_t tmp = a;
  a = b;
  b = tmp;
  
} // swap_val_

/************************************************************
 * Define some CUDA kernel to implement MHD version 3 on GPU
 ************************************************************/

#ifdef USE_DOUBLE
#define PRIM_VAR_Z_BLOCK_DIMX_3D_V3	16
#define PRIM_VAR_Z_BLOCK_DIMY_3D_V3	16
#else // simple precision
#define PRIM_VAR_Z_BLOCK_DIMX_3D_V3	16
#define PRIM_VAR_Z_BLOCK_DIMY_3D_V3	16
#endif // USE_DOUBLE

/**
 * Compute primitive variables 
 *
 * \param[in]  Uin  input  convervative variable array 
 * \param[out] Qout output primitive variable array
 */
__global__ void kernel_mhd_compute_primitive_variables_zslab(const real_t * __restrict__ Uin,
							     real_t       *Qout,
							     int pitch, 
							     int imax, 
							     int jmax,
							     int kmax,
							     real_t dt,
							     ZslabInfo zSlabInfo)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, PRIM_VAR_Z_BLOCK_DIMX_3D_V3) + tx;
  const int j = __mul24(by, PRIM_VAR_Z_BLOCK_DIMY_3D_V3) + ty;
  
  const int arraySizeU   = pitch * jmax * kmax;
  const int arraySizeQ   = pitch * jmax * zSlabInfo.zSlabWidthG;

  // conservative variables
  real_riemann_t uIn [NVAR_MHD];
  real_t c;

  const int &kStart = zSlabInfo.kStart; 
  const int &kStop  = zSlabInfo.kStop; 

  /*
   * loop over k (i.e. z) to compute primitive variables, and store results
   * in external memory buffer Q.
   */
  for (int kU=kStart, elemOffsetU = i + pitch * j + pitch*jmax*kStart;
       kU < kStop-1; 
       ++kU, elemOffsetU += (pitch*jmax)) {
    
    if( i < imax-1 and j < jmax-1 and kU < kmax-1 ) {

      	// Gather conservative variables (at z=k)
	int offsetU = elemOffsetU;

	uIn[ID] = static_cast<real_riemann_t>(Uin[offsetU]);  offsetU += arraySizeU;
	uIn[IP] = static_cast<real_riemann_t>(Uin[offsetU]);  offsetU += arraySizeU;
	uIn[IU] = static_cast<real_riemann_t>(Uin[offsetU]);  offsetU += arraySizeU;
	uIn[IV] = static_cast<real_riemann_t>(Uin[offsetU]);  offsetU += arraySizeU;
	uIn[IW] = static_cast<real_riemann_t>(Uin[offsetU]);  offsetU += arraySizeU;
	uIn[IA] = static_cast<real_riemann_t>(Uin[offsetU]);  offsetU += arraySizeU;
	uIn[IB] = static_cast<real_riemann_t>(Uin[offsetU]);  offsetU += arraySizeU;
	uIn[IC] = static_cast<real_riemann_t>(Uin[offsetU]);
	
	// go to magnetic field components and get values from 
	// neighbors on the right
	real_riemann_t magFieldNeighbors[3];
	offsetU = elemOffsetU + 5 * arraySizeU;
	magFieldNeighbors[IX] = static_cast<real_riemann_t>(Uin[offsetU+1         ]);  offsetU += arraySizeU;
	magFieldNeighbors[IY] = static_cast<real_riemann_t>(Uin[offsetU+pitch     ]);  offsetU += arraySizeU;
	magFieldNeighbors[IZ] = static_cast<real_riemann_t>(Uin[offsetU+pitch*jmax]);
	
	//Convert to primitive variables
	real_riemann_t qTmp[NVAR_MHD];
	constoprim_mhd(uIn, magFieldNeighbors, qTmp, c, dt);

	// copy results into output d_Q at z=k
	int offsetQ = elemOffsetU - kStart*pitch*jmax;
	Qout[offsetQ] = static_cast<real_t>(qTmp[ID]); offsetQ += arraySizeQ;
	Qout[offsetQ] = static_cast<real_t>(qTmp[IP]); offsetQ += arraySizeQ;
	Qout[offsetQ] = static_cast<real_t>(qTmp[IU]); offsetQ += arraySizeQ;
	Qout[offsetQ] = static_cast<real_t>(qTmp[IV]); offsetQ += arraySizeQ;
	Qout[offsetQ] = static_cast<real_t>(qTmp[IW]); offsetQ += arraySizeQ;
	Qout[offsetQ] = static_cast<real_t>(qTmp[IA]); offsetQ += arraySizeQ;
	Qout[offsetQ] = static_cast<real_t>(qTmp[IB]); offsetQ += arraySizeQ;
	Qout[offsetQ] = static_cast<real_t>(qTmp[IC]);

    } // end if

  } // enf for k

} // kernel_mhd_compute_primitive_variables_zslab

#ifdef USE_DOUBLE
#define ELEC_FIELD_Z_BLOCK_DIMX_3D_V3	16
#define ELEC_FIELD_Z_BLOCK_INNER_DIMX_3D_V3	(ELEC_FIELD_Z_BLOCK_DIMX_3D_V3-1)
#define ELEC_FIELD_Z_BLOCK_DIMY_3D_V3	11
#define ELEC_FIELD_Z_BLOCK_INNER_DIMY_3D_V3	(ELEC_FIELD_Z_BLOCK_DIMY_3D_V3-1)
#else // simple precision
#define ELEC_FIELD_Z_BLOCK_DIMX_3D_V3	16
#define ELEC_FIELD_Z_BLOCK_INNER_DIMX_3D_V3	(ELEC_FIELD_Z_BLOCK_DIMX_3D_V3-1)
#define ELEC_FIELD_Z_BLOCK_DIMY_3D_V3	11
#define ELEC_FIELD_Z_BLOCK_INNER_DIMY_3D_V3	(ELEC_FIELD_Z_BLOCK_DIMY_3D_V3-1)
#endif // USE_DOUBLE

/**
 * Compute electric field components (with rotating frame correction terms).
 *
 * \param[in]  Uin  input  convervative variable array 
 * \param[in]  Qin output primitive variable array
 * \param[out] Elec output electric field
 * \param[in]  zSlabInfo
 */
__global__ void kernel_mhd_compute_elec_field_zslab(const real_t * __restrict__ Uin,
						    const real_t * __restrict__ Qin,
						    real_t       *Elec,
						    int pitch, 
						    int imax, 
						    int jmax,
						    int kmax,
						    real_t dt,
						    ZslabInfo zSlabInfo)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, ELEC_FIELD_Z_BLOCK_INNER_DIMX_3D_V3) + tx;
  const int j = __mul24(by, ELEC_FIELD_Z_BLOCK_INNER_DIMY_3D_V3) + ty;
  
  const int arraySizeU   = pitch * jmax * kmax;
  const int arraySizeQ   = pitch * jmax * zSlabInfo.zSlabWidthG;
  const int &arraySizeL  = arraySizeQ;

  const int &kStart      = zSlabInfo.kStart; 
  //const int &kStop       = zSlabInfo.kStop; 
  const int &ksizeSlab   = zSlabInfo.ksizeSlab;

  // primitive variables in shared memory
  __shared__ real_t   q[2][ELEC_FIELD_Z_BLOCK_DIMX_3D_V3][ELEC_FIELD_Z_BLOCK_DIMY_3D_V3][NVAR_MHD];
 
  // face-centered mag field
  __shared__ real_t   bf[2][ELEC_FIELD_Z_BLOCK_DIMX_3D_V3][ELEC_FIELD_Z_BLOCK_DIMY_3D_V3][3];

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  // indexes to current Z plane and previous one
  int iZcurrent = 1;
  int iZprevious= 0;

  /*
   * load q (primitive variables) in the 2 first planes
   */
  for (int kL=0, elemOffsetL = i + pitch * j;
       kL < 2; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1)
      {
	
	// read conservative variables from Qin buffer
	int offsetL = elemOffsetL;
	q[kL][tx][ty][ID] = Qin[offsetL];  offsetL += arraySizeL;
	q[kL][tx][ty][IP] = Qin[offsetL];  offsetL += arraySizeL;
	q[kL][tx][ty][IU] = Qin[offsetL];  offsetL += arraySizeL;
	q[kL][tx][ty][IV] = Qin[offsetL];  offsetL += arraySizeL;
	q[kL][tx][ty][IW] = Qin[offsetL];  offsetL += arraySizeL;
	q[kL][tx][ty][IA] = Qin[offsetL];  offsetL += arraySizeL;
	q[kL][tx][ty][IB] = Qin[offsetL];  offsetL += arraySizeL;
	q[kL][tx][ty][IC] = Qin[offsetL];
	
	// read bf (face-centered magnetic field components) from Uin buffer
	int offsetU = (elemOffsetL + kStart*pitch*jmax) + 5*arraySizeU;
	bf[kL][tx][ty][0] = Uin[offsetU];  offsetU += arraySizeU;
	bf[kL][tx][ty][1] = Uin[offsetU];  offsetU += arraySizeU;
	bf[kL][tx][ty][2] = Uin[offsetU];
      }
    __syncthreads();

  } // end for k

  /*
   * loop over k (i.e. z) to compute Electric field, and store results
   * in external memory buffer Elec.
   */
  for (int kL=1, elemOffsetL = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-1; 
       ++kL, elemOffsetL += (pitch*jmax)) {

    int kU = kL+kStart; 

    if(i >= 1 and i < imax-1 and tx > 0 and tx < ELEC_FIELD_Z_BLOCK_DIMX_3D_V3 and
       j >= 1 and j < jmax-1 and ty > 0 and ty < ELEC_FIELD_Z_BLOCK_DIMY_3D_V3 and
       kU < kmax-1)
      {
	
	/*
	 * compute Ex, Ey, Ez
	 */
	real_riemann_t u, v, w, A, B, C;
	real_t tmpElec;
	int offsetL = elemOffsetL;
	
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
	Elec[offsetL] = tmpElec;
	offsetL += arraySizeL;
	
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

	Elec[offsetL] = static_cast<real_t>(w*A-u*C); offsetL += arraySizeL;
		
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
	Elec[offsetL] = tmpElec;
      }
    __syncthreads();
    
    /*
     * erase data in iZprevious (not needed anymore) with primitive variables
     * located at z=k+1
     */
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1 and
       kL < ksizeSlab-1)
      {
	
	// read conservative variables at z=k+1 from Qin
	int offsetL = elemOffsetL + pitch*jmax;
	q[iZprevious][tx][ty][ID] = Qin[offsetL];  offsetL += arraySizeL;
	q[iZprevious][tx][ty][IP] = Qin[offsetL];  offsetL += arraySizeL;
	q[iZprevious][tx][ty][IU] = Qin[offsetL];  offsetL += arraySizeL;
	q[iZprevious][tx][ty][IV] = Qin[offsetL];  offsetL += arraySizeL;
	q[iZprevious][tx][ty][IW] = Qin[offsetL];  offsetL += arraySizeL;
	q[iZprevious][tx][ty][IA] = Qin[offsetL];  offsetL += arraySizeL;
	q[iZprevious][tx][ty][IB] = Qin[offsetL];  offsetL += arraySizeL;
	q[iZprevious][tx][ty][IC] = Qin[offsetL];
	
	// read bf (face-centered magnetic field components) from Uin at z+1
	int offsetU = (elemOffsetL + kStart*pitch*jmax) + pitch*jmax + 5*arraySizeU;
	bf[iZprevious][tx][ty][IX] = Uin[offsetU]; offsetU += arraySizeU;
	bf[iZprevious][tx][ty][IY] = Uin[offsetU]; offsetU += arraySizeU;
	bf[iZprevious][tx][ty][IZ] = Uin[offsetU];
      }
    __syncthreads();
 
    /*
     * swap iZprevious and iZcurrent
     */
    iZprevious = 1-iZprevious;
    iZcurrent  = 1-iZcurrent;
   __syncthreads();
    
  } // end for k

} // kernel_mhd_compute_elec_field_zslab

#ifdef USE_DOUBLE
#define MAG_SLOPES_Z_BLOCK_DIMX_3D_V3	16
#define MAG_SLOPES_Z_BLOCK_INNER_DIMX_3D_V3	(MAG_SLOPES_Z_BLOCK_DIMX_3D_V3-2)
#define MAG_SLOPES_Z_BLOCK_DIMY_3D_V3	24
#define MAG_SLOPES_Z_BLOCK_INNER_DIMY_3D_V3	(MAG_SLOPES_Z_BLOCK_DIMY_3D_V3-2)
#else // simple precision
#define MAG_SLOPES_Z_BLOCK_DIMX_3D_V3	16
#define MAG_SLOPES_Z_BLOCK_INNER_DIMX_3D_V3	(MAG_SLOPES_Z_BLOCK_DIMX_3D_V3-2)
#define MAG_SLOPES_Z_BLOCK_DIMY_3D_V3	24
#define MAG_SLOPES_Z_BLOCK_INNER_DIMY_3D_V3	(MAG_SLOPES_Z_BLOCK_DIMY_3D_V3-2)
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
 * \param[in] zSlabInfo
 */
__global__ void kernel_mhd_compute_mag_slopes_zslab(const real_t * __restrict__ Uin,
						    real_t       *dA,
						    real_t       *dB,
						    real_t       *dC,
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
  
  const int i = __mul24(bx, MAG_SLOPES_Z_BLOCK_INNER_DIMX_3D_V3) + tx;
  const int j = __mul24(by, MAG_SLOPES_Z_BLOCK_INNER_DIMY_3D_V3) + ty;
  
  const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;
  const int &arraySizeL   = arraySizeQ;

  const int &kStart       = zSlabInfo.kStart; 
  //const int &kStop        = zSlabInfo.kStop; 
  const int &ksizeSlab    = zSlabInfo.ksizeSlab;

  // face-centered mag field (3 planes - 3 components : Bx,By,Bz)
  __shared__ real_t     bf[3][MAG_SLOPES_Z_BLOCK_DIMX_3D_V3][MAG_SLOPES_Z_BLOCK_DIMY_3D_V3][3];


  /*
   * initialize bf (face-centered mag field) in the 3 first planes
   */
  for (int kL=0, elemOffsetL = i + pitch * j;
       kL < 3; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax)
      {
	
	// set bf (face-centered magnetic field components)
	int offsetU = (elemOffsetL + kStart*pitch*jmax) + 5*arraySizeU;
	bf[kL][tx][ty][IX] = Uin[offsetU];  offsetU += arraySizeU;
	bf[kL][tx][ty][IY] = Uin[offsetU];  offsetU += arraySizeU;
	bf[kL][tx][ty][IZ] = Uin[offsetU];
	
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

  for (int kL=1, elemOffsetL = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-1; 
       ++kL, elemOffsetL += (pitch*jmax)) {

    int kU = kL+kStart;

    if(i >= 1 and i < imax-1 and tx > 0 and tx < MAG_SLOPES_Z_BLOCK_DIMX_3D_V3-1 and
       j >= 1 and j < jmax-1 and ty > 0 and ty < MAG_SLOPES_Z_BLOCK_DIMY_3D_V3-1 and
       kU < kmax-1)
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
	int offsetL = elemOffsetL;
	dA[offsetL] = dbfX[IX]; offsetL += arraySizeL;
	dA[offsetL] = dbfY[IX]; offsetL += arraySizeL;
	dA[offsetL] = dbfZ[IX];

	offsetL = elemOffsetL;
	dB[offsetL] = dbfX[IY]; offsetL += arraySizeL;
	dB[offsetL] = dbfY[IY]; offsetL += arraySizeL;
	dB[offsetL] = dbfZ[IY];
	
	offsetL = elemOffsetL;
	dC[offsetL] = dbfX[IZ]; offsetL += arraySizeL;
	dC[offsetL] = dbfY[IZ]; offsetL += arraySizeL;
	dC[offsetL] = dbfZ[IZ];
	
      } // end if

    /*
     * erase data in low plane (not needed anymore) with data at z=k+2
     */
    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax and 
       kL < ksizeSlab)
      {

	// set bf (face-centered magnetic field components)
	int offsetU = (elemOffsetL + kStart*pitch*jmax) + 2*pitch*jmax + 5*arraySizeU;
	bf[low][tx][ty][IX] = Uin[offsetU];  offsetU += arraySizeU;
	bf[low][tx][ty][IY] = Uin[offsetU];  offsetU += arraySizeU;
	bf[low][tx][ty][IZ] = Uin[offsetU];
	
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

} // kernel_mhd_compute_mag_slopes_zslab

#ifdef USE_DOUBLE
# define TRACE_Z_BLOCK_DIMX_3D_V4	32
# define TRACE_Z_BLOCK_INNER_DIMX_3D_V4	(TRACE_Z_BLOCK_DIMX_3D_V4-2)
# define TRACE_Z_BLOCK_DIMY_3D_V4	6
# define TRACE_Z_BLOCK_INNER_DIMY_3D_V4	(TRACE_Z_BLOCK_DIMY_3D_V4-2)
#else // simple precision
# define TRACE_Z_BLOCK_DIMX_3D_V4	16
# define TRACE_Z_BLOCK_INNER_DIMX_3D_V4	(TRACE_Z_BLOCK_DIMX_3D_V4-2)
# define TRACE_Z_BLOCK_DIMY_3D_V4	14
# define TRACE_Z_BLOCK_INNER_DIMY_3D_V4	(TRACE_Z_BLOCK_DIMY_3D_V4-2)
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
 * \param[in] zSlabInfo
 */
__global__ void kernel_mhd_compute_trace_v4_zslab(const real_t * __restrict__ Uin,
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
						  ZslabInfo zSlabInfo)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, TRACE_Z_BLOCK_INNER_DIMX_3D_V4) + tx;
  const int j = __mul24(by, TRACE_Z_BLOCK_INNER_DIMY_3D_V4) + ty;
  
  const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ   = pitch * jmax * zSlabInfo.zSlabWidthG;
  const int &arraySizeL  = arraySizeQ;

  const int &kStart      = zSlabInfo.kStart; 
  //const int &kStop       = zSlabInfo.kStop; 
  const int &ksizeSlab   = zSlabInfo.ksizeSlab;

  // face-centered mag field (3 planes)
  __shared__ real_t      q[TRACE_Z_BLOCK_DIMX_3D_V4][TRACE_Z_BLOCK_DIMY_3D_V4][NVAR_MHD];
  __shared__ real_t     bf[TRACE_Z_BLOCK_DIMX_3D_V4][TRACE_Z_BLOCK_DIMY_3D_V4][3];

  // we only stored transverse magnetic slopes
  // for dA -> Bx along Y and Z
  // for dB -> By along X and Z
  __shared__ real_t     dA[TRACE_Z_BLOCK_DIMX_3D_V4][TRACE_Z_BLOCK_DIMY_3D_V4][2];
  __shared__ real_t     dB[TRACE_Z_BLOCK_DIMX_3D_V4][TRACE_Z_BLOCK_DIMY_3D_V4][2];

  // we only store z component of electric field
  __shared__ real_t     shEz[TRACE_Z_BLOCK_DIMX_3D_V4][TRACE_Z_BLOCK_DIMY_3D_V4];

  // qm and qp's are output of the trace step
  real_t qm [THREE_D][NVAR_MHD];
  real_t qp [THREE_D][NVAR_MHD];
  real_t qEdge[4][3][NVAR_MHD];

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
  for (int kL=0, elemOffsetL = i + pitch * j;
       kL < 2; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1)
      {
	
	int offsetL = elemOffsetL;

	// read primitive variables from d_Q
	if (kL==0) {
	  qZminus1[ID] = d_Q[offsetL];  offsetL += arraySizeL;
	  qZminus1[IP] = d_Q[offsetL];  offsetL += arraySizeL;
	  qZminus1[IU] = d_Q[offsetL];  offsetL += arraySizeL;
	  qZminus1[IV] = d_Q[offsetL];  offsetL += arraySizeL;
	  qZminus1[IW] = d_Q[offsetL];  offsetL += arraySizeL;
	  qZminus1[IA] = d_Q[offsetL];  offsetL += arraySizeL;
	  qZminus1[IB] = d_Q[offsetL];  offsetL += arraySizeL;
	  qZminus1[IC] = d_Q[offsetL];
	} else { // kL == 1
	  q[tx][ty][ID] = d_Q[offsetL];  offsetL += arraySizeL;
	  q[tx][ty][IP] = d_Q[offsetL];  offsetL += arraySizeL;
	  q[tx][ty][IU] = d_Q[offsetL];  offsetL += arraySizeL;
	  q[tx][ty][IV] = d_Q[offsetL];  offsetL += arraySizeL;
	  q[tx][ty][IW] = d_Q[offsetL];  offsetL += arraySizeL;
	  q[tx][ty][IA] = d_Q[offsetL];  offsetL += arraySizeL;
	  q[tx][ty][IB] = d_Q[offsetL];  offsetL += arraySizeL;
	  q[tx][ty][IC] = d_Q[offsetL];
	}

	// read face-centered magnetic field from Uin
	int offsetU = elemOffsetL + kStart*pitch*jmax + 5 * arraySizeU;
	real_t bfX, bfY, bfZ;
	bfX = Uin[offsetU]; offsetU += arraySizeU;
	bfY = Uin[offsetU]; offsetU += arraySizeU;
	bfZ = Uin[offsetU];
	
	// set bf (face-centered magnetic field components)
	if (kL==0) {
	  //bfZminus1[0] = bfX;
	  //bfZminus1[1] = bfY;
	  //bfZminus1[2] = bfZ;
	} else { // kL == 1
	  bf[tx][ty][0] = bfX;
	  bf[tx][ty][1] = bfY;
	  bf[tx][ty][2] = bfZ;
	}
		
	// read magnetic slopes dC
	offsetL = elemOffsetL;
	dC_X = d_dC[offsetL]; offsetL += arraySizeL;
	dC_Y = d_dC[offsetL];
	
	// read electric field Ex (at i,j,k and i,j+1,k)
	offsetL = elemOffsetL;
	Ex_j_k  = Elec[offsetL];
	Ex_j1_k = Elec[offsetL+pitch]; 
	
	// read electric field Ey (at i,j,k and i+1,j,k)
	offsetL += arraySizeL;
	Ey_i_k  = Elec[offsetL];
	Ey_i1_k = Elec[offsetL+1];
      }
    __syncthreads();
  
  } // end for k

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int kL=1, elemOffsetL = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-2; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
    int kU = kL+kStart; 

    // data fetch :
    // get q, bf at z+1
    // get dA, dB at z+1
    if(i >= 0 and i < imax-1 and 
       j >= 0 and j < jmax-1 and
       kU < kmax-1)
      {
	 
	 // read primitive variables at z+1
	 int offsetL = elemOffsetL + pitch*jmax; // z+1	 
	 qZplus1[ID] = d_Q[offsetL];  offsetL += arraySizeL;
	 qZplus1[IP] = d_Q[offsetL];  offsetL += arraySizeL;
	 qZplus1[IU] = d_Q[offsetL];  offsetL += arraySizeL;
	 qZplus1[IV] = d_Q[offsetL];  offsetL += arraySizeL;
	 qZplus1[IW] = d_Q[offsetL];  offsetL += arraySizeL;
	 qZplus1[IA] = d_Q[offsetL];  offsetL += arraySizeL;
	 qZplus1[IB] = d_Q[offsetL];  offsetL += arraySizeL;
	 qZplus1[IC] = d_Q[offsetL];
	 
	 // set bf (read face-centered magnetic field components) at z+1
	 int offsetU = elemOffsetL + kStart*pitch*jmax + pitch*jmax + 5 * arraySizeU;
	 bfZplus1[IX] = Uin[offsetU]; offsetU += arraySizeU;
	 bfZplus1[IY] = Uin[offsetU]; offsetU += arraySizeU;
	 bfZplus1[IZ] = Uin[offsetU];
	 
	 // get magnetic slopes dA and dB at z=k
	 // read magnetic slopes dA (along Y and Z) 
	 offsetL = elemOffsetL + arraySizeL;
	 dA[tx][ty][0] = d_dA[offsetL]; offsetL += arraySizeL;
	 dA[tx][ty][1] = d_dA[offsetL];
	 
	 // read magnetic slopes dB (along X and Z)
	 offsetL = elemOffsetL;
	 dB[tx][ty][0] = d_dB[offsetL]; offsetL += (2*arraySizeL);
	 dB[tx][ty][1] = d_dB[offsetL];
	 
	 // get magnetic slopes dC (along X and Y) at z=k+1
	 offsetL = elemOffsetL + pitch*jmax;
	 dC_X_Zplus1 = d_dC[offsetL]; offsetL += arraySizeL;
	 dC_Y_Zplus1 = d_dC[offsetL];

	 // read electric field (Ex at  i,j,k+1 and i,j+1,k+1)
	 offsetL = elemOffsetL + pitch*jmax;
	 Ex_j_k1  = Elec[offsetL];
	 Ex_j1_k1 = Elec[offsetL+pitch]; 

	 // read electric field Ey (at i,j,k+1 and i+1,j,k+1)
	 offsetL += arraySizeL;
	 Ey_i_k1  = Elec[offsetL];
	 Ey_i1_k1 = Elec[offsetL+1];

	 // read electric field (Ez into shared memory) at z=k
	 offsetL = elemOffsetL + 2 * arraySizeL;
	 shEz[tx][ty] = Elec[offsetL];

       }
     __syncthreads();
     

     // slope and trace computation (i.e. dq, and then qm, qp, qEdge)

     if(i >= 1 and i < imax-2 and tx > 0 and tx < TRACE_Z_BLOCK_DIMX_3D_V4-1 and
	j >= 1 and j < jmax-2 and ty > 0 and ty < TRACE_Z_BLOCK_DIMY_3D_V4-1)
       {

	 int offsetL = elemOffsetL; // 3D index

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

	   for (int nVar=0, offsetBase=elemOffsetL; 
		nVar<NVAR_MHD; 
		++nVar,offsetBase += arraySizeL) {

	     real_t vmin = FLT_MAX;
	     real_t vmax = -FLT_MAX;

	     // compute vmin,vmax
	     for (int ii=-1; ii<2; ++ii)
	       for (int jj=-1; jj<2; ++jj)
		 for (int kk=-1; kk<2; ++kk) {
		   offsetL = offsetBase + ii*di + jj*dj + kk*dk;
		   real_t tmp = d_Q[offsetL] - q[tx][ty][nVar];
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
	 
	 
	 // store qm, qp, qEdge in external memory
	 {
	   offsetL  = elemOffsetL;
	   for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	     
	     d_qm_x[offsetL] = qm[0][iVar];
	     d_qm_y[offsetL] = qm[1][iVar];
	     d_qm_z[offsetL] = qm[2][iVar];
	     
	     d_qp_x[offsetL] = qp[0][iVar];
	     d_qp_y[offsetL] = qp[1][iVar];
	     d_qp_z[offsetL] = qp[2][iVar];
	     
	     d_qEdge_RT[offsetL]  = qEdge[IRT][0][iVar];
	     d_qEdge_RB[offsetL]  = qEdge[IRB][0][iVar];
	     d_qEdge_LT[offsetL]  = qEdge[ILT][0][iVar];
	     d_qEdge_LB[offsetL]  = qEdge[ILB][0][iVar];
	     
	     d_qEdge_RT2[offsetL] = qEdge[IRT][1][iVar];
	     d_qEdge_RB2[offsetL] = qEdge[IRB][1][iVar];
	     d_qEdge_LT2[offsetL] = qEdge[ILT][1][iVar];
	     d_qEdge_LB2[offsetL] = qEdge[ILB][1][iVar];
	     
	     d_qEdge_RT3[offsetL] = qEdge[IRT][2][iVar];
	     d_qEdge_RB3[offsetL] = qEdge[IRB][2][iVar];
	     d_qEdge_LT3[offsetL] = qEdge[ILT][2][iVar];
	     d_qEdge_LB3[offsetL] = qEdge[ILB][2][iVar];
	     
	     offsetL  += arraySizeL;
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
  
} // kernel_mhd_compute_trace_v4_zslab

#ifdef USE_DOUBLE
# define GRAV_PRED_Z_BLOCK_DIMX_3D_V4	32
# define GRAV_PRED_Z_BLOCK_DIMY_3D_V4	10
#else // simple precision
# define GRAV_PRED_Z_BLOCK_DIMX_3D_V4	32
# define GRAV_PRED_Z_BLOCK_DIMY_3D_V4	10
#endif // USE_DOUBLE
/**
 * Compute gravity predictor for MHD (implementation version 4) with zSlab.
 *
 * Must be called after kernel_mhd_compute_trace_v4.
 * \see kernel_mhd_compute_trace_v4
 *
 * \note All d_q arrays are sized upon zSlab sizes and gravity field array
 * upon sub-domain sizes.
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
 * \param[in]     zSlabInfo
 *
 */
__global__ void kernel_mhd_compute_gravity_predictor_v4_zslab(real_t *d_qm_x,
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
							      real_t dt,
							      ZslabInfo zSlabInfo)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = bx * blockDim.x + tx;
  const int j = by * blockDim.y + ty;
  
  const int arraySizeG    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;
  //const int &arraySizeL   = arraySizeQ;

  const int &kStart       = zSlabInfo.kStart; 
  //const int &kStop        = zSlabInfo.kStop; 
  const int &ksizeSlab    = zSlabInfo.ksizeSlab;

  // for gravity predictor
  const real_t *gravin = gParams.arrayList[A_GRAV];

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int kL=1, elemOffsetQ = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-2; 
       ++kL, elemOffsetQ += (pitch*jmax)) {

    if(i >= 1 and i < imax-2 and 
       j >= 1 and j < jmax-2)
      {

	int    offsetG = elemOffsetQ + kStart*jmax*pitch;
	real_t grav_x = HALF_F * dt * gravin[offsetG+IX*arraySizeG];
	real_t grav_y = HALF_F * dt * gravin[offsetG+IY*arraySizeG];
	real_t grav_z = HALF_F * dt * gravin[offsetG+IZ*arraySizeG];

	{
	  int offsetQ = elemOffsetQ + IU*arraySizeQ;
	  d_qm_x[offsetQ] += grav_x; d_qp_x[offsetQ] += grav_x;
	  d_qm_y[offsetQ] += grav_x; d_qp_y[offsetQ] += grav_x;
	  d_qm_z[offsetQ] += grav_x; d_qp_z[offsetQ] += grav_x;
	  
	  d_qEdge_RT[offsetQ]   += grav_x;  
	  d_qEdge_RB[offsetQ]   += grav_x; 
	  d_qEdge_LT[offsetQ]   += grav_x; 
	  d_qEdge_LB[offsetQ]   += grav_x; 
	  
	  d_qEdge_RT2[offsetQ]  += grav_x;  
	  d_qEdge_RB2[offsetQ]  += grav_x; 
	  d_qEdge_LT2[offsetQ]  += grav_x; 
	  d_qEdge_LB2[offsetQ]  += grav_x; 
	  
	  d_qEdge_RT3[offsetQ]  += grav_x;  
	  d_qEdge_RB3[offsetQ]  += grav_x; 
	  d_qEdge_LT3[offsetQ]  += grav_x; 
	  d_qEdge_LB3[offsetQ]  += grav_x; 

	} // end grav_x

	{
	  int offsetQ = elemOffsetQ + IV*arraySizeQ;
	  d_qm_x[offsetQ] += grav_y; d_qp_x[offsetQ] += grav_y;
	  d_qm_y[offsetQ] += grav_y; d_qp_y[offsetQ] += grav_y;
	  d_qm_z[offsetQ] += grav_y; d_qp_z[offsetQ] += grav_y;
	  
	  d_qEdge_RT[offsetQ]   += grav_y;  
	  d_qEdge_RB[offsetQ]   += grav_y; 
	  d_qEdge_LT[offsetQ]   += grav_y; 
	  d_qEdge_LB[offsetQ]   += grav_y; 
	  
	  d_qEdge_RT2[offsetQ]  += grav_y;  
	  d_qEdge_RB2[offsetQ]  += grav_y; 
	  d_qEdge_LT2[offsetQ]  += grav_y; 
	  d_qEdge_LB2[offsetQ]  += grav_y; 
	  
	  d_qEdge_RT3[offsetQ]  += grav_y;  
	  d_qEdge_RB3[offsetQ]  += grav_y; 
	  d_qEdge_LT3[offsetQ]  += grav_y; 
	  d_qEdge_LB3[offsetQ]  += grav_y; 

	} // end grav_y

	{
	  int offsetQ = elemOffsetQ + IW*arraySizeQ;
	  d_qm_x[offsetQ] += grav_z; d_qp_x[offsetQ] += grav_z;
	  d_qm_y[offsetQ] += grav_z; d_qp_y[offsetQ] += grav_z;
	  d_qm_z[offsetQ] += grav_z; d_qp_z[offsetQ] += grav_z;
	  
	  d_qEdge_RT[offsetQ]   += grav_z;  
	  d_qEdge_RB[offsetQ]   += grav_z; 
	  d_qEdge_LT[offsetQ]   += grav_z; 
	  d_qEdge_LB[offsetQ]   += grav_z; 
	  
	  d_qEdge_RT2[offsetQ]  += grav_z;  
	  d_qEdge_RB2[offsetQ]  += grav_z; 
	  d_qEdge_LT2[offsetQ]  += grav_z; 
	  d_qEdge_LB2[offsetQ]  += grav_z; 
	  
	  d_qEdge_RT3[offsetQ]  += grav_z;  
	  d_qEdge_RB3[offsetQ]  += grav_z; 
	  d_qEdge_LT3[offsetQ]  += grav_z; 
	  d_qEdge_LB3[offsetQ]  += grav_z; 

	} // end grav_z

      } // end if i,j

  } // end for k

} // kernel_mhd_compute_gravity_predictor_v4_zslab

#ifdef USE_DOUBLE
#define UPDATE_Z_BLOCK_DIMX_3D_V4	8
#define UPDATE_Z_BLOCK_INNER_DIMX_3D_V4	(UPDATE_Z_BLOCK_DIMX_3D_V4-1)
#define UPDATE_Z_BLOCK_DIMY_3D_V4	8
#define UPDATE_Z_BLOCK_INNER_DIMY_3D_V4	(UPDATE_Z_BLOCK_DIMY_3D_V4-1)
#else // simple precision
#define UPDATE_Z_BLOCK_DIMX_3D_V4	8
#define UPDATE_Z_BLOCK_INNER_DIMX_3D_V4	(UPDATE_Z_BLOCK_DIMX_3D_V4-1)
#define UPDATE_Z_BLOCK_DIMY_3D_V4	8
#define UPDATE_Z_BLOCK_INNER_DIMY_3D_V4	(UPDATE_Z_BLOCK_DIMY_3D_V4-1)
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
 * \param[in] zSlabInfo
 *
 */
__global__ void kernel_mhd_flux_update_hydro_v4_zslab(const real_t * __restrict__ Uin, 
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
						      real_t dt,
						      ZslabInfo zSlabInfo)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UPDATE_Z_BLOCK_INNER_DIMX_3D_V4) + tx;
  const int j = __mul24(by, UPDATE_Z_BLOCK_INNER_DIMY_3D_V4) + ty;
  
  const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;
  const int &arraySizeL   = arraySizeQ;

  const int &kStart       = zSlabInfo.kStart; 
  //const int &kStop        = zSlabInfo.kStop; 
  const int &ksizeSlab    = zSlabInfo.ksizeSlab;

  // flux computation
  __shared__ real_t   flux[UPDATE_Z_BLOCK_DIMX_3D_V4][UPDATE_Z_BLOCK_DIMY_3D_V4][NVAR_MHD];

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
  for (int kL=1, elemOffsetL = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-2; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use flux as flux_x
    real_t (&flux_x)[UPDATE_Z_BLOCK_DIMX_3D_V4][UPDATE_Z_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    flux_x[tx][ty][IA] = ZERO_F;
    flux_x[tx][ty][IB] = ZERO_F;
    flux_x[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i  >= 3 and i  < imax-2 and
       j  >= 3 and j  < jmax-2 and
       kL >= 3 and kL < ksizeSlab-2)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t   qleft_x [NVAR_MHD];
	real_t   qright_x[NVAR_MHD];
	
	// set qleft_x by re-reading qm_x from external memory at location x-1
	int offsetL = elemOffsetL-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_x[iVar] = d_qm_x[offsetL];
	  offsetL += arraySizeL;
	}

	// set qright_x by re-reading qp_x from external memory at location x
	offsetL = elemOffsetL;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_x[iVar] = d_qp_x[offsetL];
	  offsetL += arraySizeL;
	}

	riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i  >= 3 and i  < imax-3 and tx < UPDATE_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-3 and ty < UPDATE_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-3)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offsetU = elemOffsetL + kStart*pitch*jmax;
	uOut[ID] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IP] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IU] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IV] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IW] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IA] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IB] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IC] = Uin[offsetU];

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
    real_t (&flux_y)[UPDATE_Z_BLOCK_DIMX_3D_V4][UPDATE_Z_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    flux_y[tx][ty][IA] = ZERO_F;
    flux_y[tx][ty][IB] = ZERO_F;
    flux_y[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i  >= 3 and i  < imax-2 and 
       j  >= 3 and j  < jmax-2 and 
       kL >= 3 and kL < ksizeSlab-2)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t  qleft_y[NVAR_MHD];
	real_t qright_y[NVAR_MHD];
	
	// set qleft_y by reading qm_y from external memory at location y-1
	int offsetL = elemOffsetL-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_y[iVar] = d_qm_y[offsetL];
	  offsetL += arraySizeL;
	}
	
	// set qright_y by reading qp_y from external memory at location y
	offsetL = elemOffsetL;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_y[iVar] = d_qp_y[offsetL];
	  offsetL += arraySizeL;
	}
	
	// watchout swap IU and IV
	swap_val_(qleft_y[IU],qleft_y[IV]);
	swap_val_(qleft_y[IA],qleft_y[IB]);
	swap_val_(qright_y[IU],qright_y[IV]);
	swap_val_(qright_y[IA],qright_y[IB]);

	riemann_mhd(qleft_y, qright_y, flux_y[tx][ty]);

	/* shear correction on flux_y */
	if (/* cartesian */ ::gParams.Omega0 > 0 /* and not fargo */) {
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
    if(i  >= 3 and i  < imax-3 and tx < UPDATE_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-3 and ty < UPDATE_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-3)
      {
	// watchout IU and IV are swapped !

	if (Omega0>0) {
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
	} else {
	  uOut[ID] += (flux_y[tx][ty  ][ID]-
		       flux_y[tx][ty+1][ID])*dtdy;
	  
	  uOut[IP] += (flux_y[tx][ty  ][IP]-
		       flux_y[tx][ty+1][IP])*dtdy;
	  
	  uOut[IU] += (flux_y[tx][ty  ][IV]-
		       flux_y[tx][ty+1][IV])*dtdy;
	  
	  uOut[IV] += (flux_y[tx][ty  ][IU]-
		       flux_y[tx][ty+1][IU])*dtdy;
	  
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
    
    if(i  >= 3 and i  < imax-2 and tx < UPDATE_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-2 and ty < UPDATE_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-2)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t qleft_z [NVAR_MHD];
	real_t qright_z[NVAR_MHD];
	
	// set qleft_z by reading qm_z from external memory at location z-1
	int offsetL = elemOffsetL - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_z[iVar] = d_qm_z[offsetL];
	  offsetL += arraySizeL;
	}
	
        // set qright_z by reading qp_z from external memory at location z
        offsetL = elemOffsetL;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_z[iVar] = d_qp_z[offsetL];
	  offsetL += arraySizeL;
	}
	
	// watchout swap IU and IW
	swap_val_(qleft_z[IU] ,qleft_z[IW]);
	swap_val_(qleft_z[IA] ,qleft_z[IC]);
	swap_val_(qright_z[IU],qright_z[IW]);
	swap_val_(qright_z[IA],qright_z[IC]);
	
	riemann_mhd(qleft_z, qright_z, flux_z);
	
      }  
    __syncthreads();
    
    // dirty trick to deal with the last slab...
    //int kLStopUpdate = ksizeSlab-3;
    //if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1)
    //  kLStopUpdate = ksizeSlab-2;

    // update uOut with flux_z
    if(i  >= 3 and i  < imax-3 and tx < UPDATE_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-3 and ty < UPDATE_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-2/*kLStopUpdate*/)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	int offsetU = elemOffsetL + kStart*pitch*jmax;

	if (kL < ksizeSlab-3) {
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
	  Uout[offsetU] = uOut[ID];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IP];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IU];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IV];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IW];
	}

	if ( kL > 3 ) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offsetU = elemOffsetL + kStart*pitch*jmax - pitch*jmax;
	  if (Omega0>0) {
	    Uout[offsetU] -= flux_z[ID]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IP]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= (alpha1*flux_z[IW]+alpha2*flux_z[IV])*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= (alpha1*flux_z[IV]-0.25*alpha2*flux_z[IW])*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IU]*dtdz;
	  } else {
	    Uout[offsetU] -= flux_z[ID]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IP]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IW]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IV]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IU]*dtdz;	    
	  }
	}
      } // end update along Z
    __syncthreads();

  } // end for k

} // kernel_mhd_flux_update_hydro_v4_zslab

#ifdef USE_DOUBLE
#define COMPUTE_EMF_Z_BLOCK_DIMX_3D_V4	16
#define COMPUTE_EMF_Z_BLOCK_DIMY_3D_V4	8
#else // simple precision
#define COMPUTE_EMF_Z_BLOCK_DIMX_3D_V4	16
#define COMPUTE_EMF_Z_BLOCK_DIMY_3D_V4	16
#endif // USE_DOUBLE

/**
 * Compute emf's and store them in d_emf, z-slab method.
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
 * \param[in] zSlabInfo
 *
 */
__global__ void kernel_mhd_compute_emf_v4_zslab(const real_t * __restrict__ d_qEdge_RT,
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
						real_t dt,
						ZslabInfo zSlabInfo)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, COMPUTE_EMF_Z_BLOCK_DIMX_3D_V4) + tx;
  const int j = __mul24(by, COMPUTE_EMF_Z_BLOCK_DIMY_3D_V4) + ty;
  
  //const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;
  const int &arraySizeL   = arraySizeQ;

  //const int &kStart       = zSlabInfo.kStart; 
  //const int &kStop        = zSlabInfo.kStop; 
  const int &ksizeSlab    = zSlabInfo.ksizeSlab;

  //real_t qEdge[4][3][NVAR_MHD];

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  /****************************************
   * loop over k (i.e. z) to compute emf's
   ****************************************/
  for (int kL=1, elemOffsetL = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-2; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
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
     
     if(i  > 1 and i  < imax-2 and 
        j  > 1 and j  < jmax-2 and 
        kL > 1 and kL < ksizeSlab-2)
      {
	
	/*
	 * offset into external memory array to qEdge data
	 */
	
	int offset2L         = elemOffsetL;
	int offsetL;
	
	/*
	 * compute emfZ
	 */

	// qEdge_RT3 at location x-1, y-1
	offsetL = offset2L-1-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT3[iVar] = d_qEdge_RT3[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge RB3 at location x-1, y
	offsetL = offset2L-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB3[iVar] = d_qEdge_RB3[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LT3 at location x, y-1
	offsetL = offset2L-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT3[iVar] = d_qEdge_LT3[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LB3 at location x, y
	offsetL = offset2L;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB3[iVar] = d_qEdge_LB3[offsetL];
	  offsetL += arraySizeL;
	}

	// finally compute emfZ
	emf = compute_emf<EMFZ>(qEdge_emfZ,xPos);
	offsetL = offset2L + I_EMFZ*arraySizeL;
	d_emf[offsetL] = emf;

	/*
	 * compute emfY (take care RB and LT are swapped)
	 */

	// qEdge_RT2 at location x-1, z-1
	offsetL = offset2L - pitch*jmax - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT2[iVar] = d_qEdge_RT2[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge RB2 (actually LT2) at location x, z-1
	offsetL = offset2L - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB2[iVar] = d_qEdge_LT2[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LT2 (actually RB2) at location x-1, z
	offsetL = offset2L - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT2[iVar] = d_qEdge_RB2[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LB2 at location x, y
	offsetL = offset2L;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB2[iVar] = d_qEdge_LB2[offsetL];
	  offsetL += arraySizeL;
	}

	// finally compute emfY
	emf = compute_emf<EMFY>(qEdge_emfY,xPos);
	offsetL = offset2L + I_EMFY*arraySizeL;
	d_emf[offsetL] = emf;

	/*
	 * compute emfX
	 */
	// qEdge_RT at location y-1, z-1
	offsetL = offset2L - pitch*jmax - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT[iVar] = d_qEdge_RT[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge RB at location y-1, z
	offsetL = offset2L - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB[iVar] = d_qEdge_RB[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LT at location y, z-1
	offsetL = offset2L - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT[iVar] = d_qEdge_LT[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LB at location y, z
	offsetL = offset2L;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB[iVar] = d_qEdge_LB[offsetL];
	  offsetL += arraySizeL;
	}

	// finally compute emfX
	emf = compute_emf<EMFX>(qEdge_emfX,xPos);
	offsetL = offset2L + I_EMFX*arraySizeL;
	d_emf[offsetL] = emf;
	
      }
    __syncthreads();
            
  } // end for k

} // kernel_mhd_compute_emf_v4_zslab

#ifdef USE_DOUBLE
#define UPDATE_CT_Z_BLOCK_DIMX_3D_V4	16
#define UPDATE_CT_Z_BLOCK_DIMY_3D_V4	16
#else // simple precision
#define UPDATE_CT_Z_BLOCK_DIMX_3D_V4	16
#define UPDATE_CT_Z_BLOCK_DIMY_3D_V4	16
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
 * \param[in]  zSlabInfo
 *
 */
__global__ void kernel_mhd_flux_update_ct_v4_zslab(const real_t * __restrict__ Uin, 
						   real_t       *Uout,
						   const real_t * __restrict__ d_emf,
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
  
  const int i = __mul24(bx, UPDATE_CT_Z_BLOCK_DIMX_3D_V4) + tx;
  const int j = __mul24(by, UPDATE_CT_Z_BLOCK_DIMY_3D_V4) + ty;
  
  const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;
  const int &arraySizeL   = arraySizeQ;

  const int &kStart       = zSlabInfo.kStart; 
  //const int &kStop        = zSlabInfo.kStop; 
  const int &ksizeSlab    = zSlabInfo.ksizeSlab;

  const int ghostWidth = 3; // MHD

  // conservative variables
  real_t uOut[NVAR_MHD];

  int kLStopUpdate = ksizeSlab-ghostWidth;
  if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1)
    kLStopUpdate = ksizeSlab-ghostWidth+1;

   /*
    * loop over k (i.e. z) to perform constraint transport update
    */
  for (int kL=ghostWidth, elemOffsetL = i + pitch * (j + jmax * ghostWidth);
       kL < kLStopUpdate; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
    if(i >= 3 and i < imax-2 and 
       j >= 3 and j < jmax-2)
      {
	// First : update at current location x,y,z
	int offsetU;
  	offsetU = elemOffsetL + kStart*pitch*jmax + 5 * arraySizeU;
	
	// read magnetic field components from external memory
	uOut[IA] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IB] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IC] = Uin[offsetU];
	
	// indexes used to fetch emf's at the right location
	const int ijk   = elemOffsetL;
	const int ip1jk = ijk + 1;
	const int ijp1k = ijk + pitch;
	const int ijkp1 = ijk + pitch * jmax;

	int kU=kL+kStart;
	if (kU<kmax-3) { // EMFZ update
	  uOut[IA] += ( d_emf[ijp1k + I_EMFZ*arraySizeL] - 
			d_emf[ijk   + I_EMFZ*arraySizeL] ) * dtdy;
	  
	  uOut[IB] -= ( d_emf[ip1jk + I_EMFZ*arraySizeL] - 
			d_emf[ijk   + I_EMFZ*arraySizeL] ) * dtdx;
	  
	}
	
	// update BX
	uOut[IA] -= ( d_emf[ijkp1 + I_EMFY*arraySizeL] -
		      d_emf[ijk   + I_EMFY*arraySizeL] ) * dtdz;
	
	// update BY
	uOut[IB] += ( d_emf[ijkp1 + I_EMFX*arraySizeL] -
		      d_emf[ijk   + I_EMFX*arraySizeL] ) * dtdz;
	
	// update BZ
	uOut[IC] += ( d_emf[ip1jk + I_EMFY*arraySizeL] -
		      d_emf[ijk   + I_EMFY*arraySizeL] ) * dtdx;
	uOut[IC] -= ( d_emf[ijp1k + I_EMFX*arraySizeL] -
		      d_emf[ijk   + I_EMFX*arraySizeL] ) * dtdy;
	
	// write back mag field components in external memory
	offsetU = elemOffsetL + kStart*pitch*jmax + 5 * arraySizeU;
	
	Uout[offsetU] = uOut[IA];  offsetU += arraySizeU;
	Uout[offsetU] = uOut[IB];  offsetU += arraySizeU;
	Uout[offsetU] = uOut[IC];
	
      } // end if

  } // end for k

} // kernel_mhd_flux_update_ct_v4_zslab

/************************************************************/
/***************** SHEARING BOX variants ********************/
/************************************************************/

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
 * \param[in] zSlabInfo
 *
 */
__global__ void kernel_mhd_flux_update_hydro_v4_shear_zslab(const real_t * __restrict__ Uin, 
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
							    real_t dt,
							    ZslabInfo zSlabInfo)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UPDATE_Z_BLOCK_INNER_DIMX_3D_V4) + tx;
  const int j = __mul24(by, UPDATE_Z_BLOCK_INNER_DIMY_3D_V4) + ty;
  
  const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;
  const int &arraySizeL   = arraySizeQ;
  const int arraySize2d   =       pitchB * zSlabInfo.zSlabWidthG; // d_shear_flux_xmin/xmax size

  const int &kStart       = zSlabInfo.kStart; 
  //const int &kStop        = zSlabInfo.kStop; 
  const int &ksizeSlab    = zSlabInfo.ksizeSlab;

  // flux computation
  __shared__ real_t   flux[UPDATE_Z_BLOCK_DIMX_3D_V4][UPDATE_Z_BLOCK_DIMY_3D_V4][NVAR_MHD];

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
   * loop over k (i.e. z) to compute trace
   */
  for (int kL=1, elemOffsetL = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-2; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use flux as flux_x
    real_t (&flux_x)[UPDATE_Z_BLOCK_DIMX_3D_V4][UPDATE_Z_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    flux_x[tx][ty][IA] = ZERO_F;
    flux_x[tx][ty][IB] = ZERO_F;
    flux_x[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i  >= 3 and i  < imax-2 and
       j  >= 3 and j  < jmax-2 and
       kL >= 3 and kL < ksizeSlab-2)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t   qleft_x [NVAR_MHD];
	real_t   qright_x[NVAR_MHD];
	
	// set qleft_x by re-reading qm_x from external memory at location x-1
	int offsetL = elemOffsetL-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_x[iVar] = d_qm_x[offsetL];
	  offsetL += arraySizeL;
	}

	// set qright_x by re-reading qp_x from external memory at location x
	offsetL = elemOffsetL;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_x[iVar] = d_qp_x[offsetL];
	  offsetL += arraySizeL;
	}

	riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i  >= 3 and i  < imax-3 and tx < UPDATE_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-3 and ty < UPDATE_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-3)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offsetU = elemOffsetL + kStart*pitch*jmax;
	uOut[ID] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IP] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IU] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IV] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IW] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IA] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IB] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IC] = Uin[offsetU];

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
	  int offsetShear = j+pitchB*kL;
	  d_shear_flux_xmin[offsetShear] = flux_x[tx  ][ty][ID]*dtdx; // I_DENS
	}
	if (i==imax-4 and ::gParams.mpiPosX == (::gParams.mx - 1) ) {
	  /* store flux_xmax */
	  int offsetShear = j+pitchB*kL;
	  d_shear_flux_xmax[offsetShear] = flux_x[tx+1][ty][ID]*dtdx; // I_DENS	  
	}
	
      }
    __syncthreads();

    /*
     * Compute fluxes at Y-interfaces.
     */
    // re-use flux as flux_y
    real_t (&flux_y)[UPDATE_Z_BLOCK_DIMX_3D_V4][UPDATE_Z_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    flux_y[tx][ty][IA] = ZERO_F;
    flux_y[tx][ty][IB] = ZERO_F;
    flux_y[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i  >= 3 and i  < imax-2 and 
       j  >= 3 and j  < jmax-2 and 
       kL >= 3 and kL < ksizeSlab-2)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t  qleft_y[NVAR_MHD];
	real_t qright_y[NVAR_MHD];
	
	// set qleft_y by reading qm_y from external memory at location y-1
	int offsetL = elemOffsetL-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_y[iVar] = d_qm_y[offsetL];
	  offsetL += arraySizeL;
	}
	
	// set qright_y by reading qp_y from external memory at location y
	offsetL = elemOffsetL;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_y[iVar] = d_qp_y[offsetL];
	  offsetL += arraySizeL;
	}
	
	// watchout swap IU and IV
	swap_val_(qleft_y[IU],qleft_y[IV]);
	swap_val_(qleft_y[IA],qleft_y[IB]);
	swap_val_(qright_y[IU],qright_y[IV]);
	swap_val_(qright_y[IA],qright_y[IB]);

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
    if(i  >= 3 and i  < imax-3 and tx < UPDATE_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-3 and ty < UPDATE_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-3)
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
    
    if(i  >= 3 and i  < imax-2 and tx < UPDATE_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-2 and ty < UPDATE_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-2)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t qleft_z [NVAR_MHD];
	real_t qright_z[NVAR_MHD];
	
	// set qleft_z by reading qm_z from external memory at location z-1
	int offsetL = elemOffsetL - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_z[iVar] = d_qm_z[offsetL];
	  offsetL += arraySizeL;
	}
	
        // set qright_z by reading qp_z from external memory at location z
        offsetL = elemOffsetL;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_z[iVar] = d_qp_z[offsetL];
	  offsetL += arraySizeL;
	}
	
	// watchout swap IU and IW
	swap_val_(qleft_z[IU] ,qleft_z[IW]);
	swap_val_(qleft_z[IA] ,qleft_z[IC]);
	swap_val_(qright_z[IU],qright_z[IW]);
	swap_val_(qright_z[IA],qright_z[IC]);
	
	riemann_mhd(qleft_z, qright_z, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i  >= 3 and i  < imax-3 and tx < UPDATE_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-3 and ty < UPDATE_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-2)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	int offsetU = elemOffsetL + kStart*pitch*jmax;

	if (kL < ksizeSlab-3) {
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
	  Uout[offsetU] = uOut[ID];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IP];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IU];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IV];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IW];
	}

	if (kL > 3) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offsetU = elemOffsetL + kStart*pitch*jmax - pitch*jmax;
	  {
	    Uout[offsetU] -= flux_z[ID]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IP]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= (alpha1*flux_z[IW]+     
			      alpha2*flux_z[IV])*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= (alpha1*flux_z[IV]-0.25*
			      alpha2*flux_z[IW])*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IU]*dtdz;
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
    // real_t (&emf)[UPDATE_Z_BLOCK_DIMX_3D_V4][UPDATE_Z_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    // emf[tx][ty][IX] = ZERO_F; // emfX
    // emf[tx][ty][IY] = ZERO_F; // emfY
    // emf[tx][ty][IZ] = ZERO_F; // emfZ
    
    real_t emf;
    
    if(i  > 2 and i  < imax-2 and tx < UPDATE_Z_BLOCK_DIMX_3D_V4-1 and
       j  > 2 and j  < jmax-2 and ty < UPDATE_Z_BLOCK_DIMY_3D_V4-1 and
       kL > 2 and kL < ksizeSlab-2)
      {
	
	/*
	 * offset into external memory array to qEdge data
	 */

	int offset2L         = elemOffsetL;
 
	
	/*
	 * compute emfZ
	 */

	// qEdge_RT3 at location x-1, y-1
	int offsetL = offset2L-1-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT3[iVar] = d_qEdge_RT3[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge RB3 at location x-1, y
	offsetL = offset2L-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB3[iVar] = d_qEdge_RB3[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LT3 at location x, y-1
	offsetL = offset2L-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT3[iVar] = d_qEdge_LT3[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LB3 at location x, y
	offsetL = offset2L;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB3[iVar] = d_qEdge_LB3[offsetL];
	  offsetL += arraySizeL;
	}

	// finally compute emfZ
	emf = compute_emf<EMFZ>(qEdge_emfZ,xPos);
	offsetL = offset2L + I_EMFZ*arraySizeL;
	if (kL<ksizeSlab-3)
	  d_emf[offsetL] = emf;

	/*
	 * compute emfY (take care RB and LT are swapped)
	 */

	// qEdge_RT2 at location x-1, z-1
	offsetL = offset2L - pitch*jmax - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT2[iVar] = d_qEdge_RT2[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge RB2 (actually LT2) at location x, z-1
	offsetL = offset2L - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB2[iVar] = d_qEdge_LT2[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LT2 (actually RB2) at location x-1, z
	offsetL = offset2L - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT2[iVar] = d_qEdge_RB2[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LB2 at location x, y
	offsetL = offset2L;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB2[iVar] = d_qEdge_LB2[offsetL];
	  offsetL += arraySizeL;
	}

	// finally compute emfY
	emf = compute_emf<EMFY>(qEdge_emfY,xPos);
	offsetL = offset2L + I_EMFY*arraySizeL;
	if (j<jmax-3) {
	  d_emf[offsetL] = emf;
	  
	  // at global XMIN border, store emfY
	  if (i == 3     and ::gParams.mpiPosX == 0) {
	    int offsetShear = j+pitchB*kL;
	    d_shear_flux_xmin[offsetShear + arraySize2d*I_EMF_Y] = emf;
	  }
	  
	  // at global XMAX border, store emfY
	  if (i == imax-3 and ::gParams.mpiPosX == (::gParams.mx - 1) ) {
	    int offsetShear = j+pitchB*kL;
	    d_shear_flux_xmax[offsetShear + arraySize2d*I_EMF_Y] = emf;
	  }
	}

	/*
	 * compute emfX
	 */
	// qEdge_RT at location y-1, z-1
	offsetL = offset2L - pitch*jmax - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT[iVar] = d_qEdge_RT[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge RB at location y-1, z
	offsetL = offset2L - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB[iVar] = d_qEdge_RB[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LT at location y, z-1
	offsetL = offset2L - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT[iVar] = d_qEdge_LT[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LB at location y, z
	offsetL = offset2L;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB[iVar] = d_qEdge_LB[offsetL];
	  offsetL += arraySizeL;
	}

	// finally compute emfX
	emf = compute_emf<EMFX>(qEdge_emfX,xPos);
	offsetL = offset2L + I_EMFX*arraySizeL;
	if (i<imax-3)
	  d_emf[offsetL] = emf;	
      }
    __syncthreads();
            
  } // end for k

} // kernel_mhd_flux_update_hydro_v4_shear_zslab

#ifdef USE_DOUBLE
#define UPDATE_P1_Z_BLOCK_DIMX_3D_V4	16
#define UPDATE_P1_Z_BLOCK_INNER_DIMX_3D_V4	(UPDATE_P1_Z_BLOCK_DIMX_3D_V4-1)
#define UPDATE_P1_Z_BLOCK_DIMY_3D_V4	8
#define UPDATE_P1_Z_BLOCK_INNER_DIMY_3D_V4	(UPDATE_P1_Z_BLOCK_DIMY_3D_V4-1)
#else // simple precision
#define UPDATE_P1_Z_BLOCK_DIMX_3D_V4	16
#define UPDATE_P1_Z_BLOCK_INNER_DIMX_3D_V4	(UPDATE_P1_Z_BLOCK_DIMX_3D_V4-1)
#define UPDATE_P1_Z_BLOCK_DIMY_3D_V4	8
#define UPDATE_P1_Z_BLOCK_INNER_DIMY_3D_V4	(UPDATE_P1_Z_BLOCK_DIMY_3D_V4-1)
#endif // USE_DOUBLE

/**
 * Update MHD conservative variables (same as kernel_mhd_flux_update_hydro_v4 but for shearing box)
 * 
 * This kernel performs conservative variables update everywhere except at X-border 
 * (which is done outside).
 *
 * Here we assume Omega is strictly positive.
 *
 * \see kernel_mhd_flux_update_hydro_v4_shear_zslab
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
 * \param[in] zSlabInfo
 *
 */
__global__ void kernel_mhd_flux_update_hydro_v4_shear_part1_zslab(const real_t * __restrict__ Uin, 
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
								  real_t dt,
								  ZslabInfo zSlabInfo)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, UPDATE_P1_Z_BLOCK_INNER_DIMX_3D_V4) + tx;
  const int j = __mul24(by, UPDATE_P1_Z_BLOCK_INNER_DIMY_3D_V4) + ty;
  
  const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;
  const int &arraySizeL   = arraySizeQ;
  //const int arraySize2d   =       pitchB * zSlabInfo.zSlabWidthG; // d_shear_flux_xmin/xmax size

  const int &kStart       = zSlabInfo.kStart; 
  //const int &kStop        = zSlabInfo.kStop; 
  const int &ksizeSlab    = zSlabInfo.ksizeSlab;

  // flux computation
  __shared__ real_t   flux[UPDATE_P1_Z_BLOCK_DIMX_3D_V4][UPDATE_P1_Z_BLOCK_DIMY_3D_V4][NVAR_MHD];

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
   * loop over k (i.e. z) to compute trace
   */
  for (int kL=1, elemOffsetL = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-2; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
    // update hydro
    /*
     * Compute fluxes at X-interfaces.
     */
    // re-use flux as flux_x
    real_t (&flux_x)[UPDATE_P1_Z_BLOCK_DIMX_3D_V4][UPDATE_P1_Z_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_x[tx][ty][ID] = ZERO_F;
    flux_x[tx][ty][IP] = ZERO_F;
    flux_x[tx][ty][IU] = ZERO_F;
    flux_x[tx][ty][IV] = ZERO_F;
    flux_x[tx][ty][IW] = ZERO_F;
    flux_x[tx][ty][IA] = ZERO_F;
    flux_x[tx][ty][IB] = ZERO_F;
    flux_x[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i  >= 3 and i  < imax-2 and
       j  >= 3 and j  < jmax-2 and
       kL >= 3 and kL < ksizeSlab-2)
      {
	// Solve Riemann problem at X-interfaces and compute fluxes
	real_t   qleft_x [NVAR_MHD];
	real_t   qright_x[NVAR_MHD];
	
	// set qleft_x by re-reading qm_x from external memory at location x-1
	int offsetL = elemOffsetL-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_x[iVar] = d_qm_x[offsetL];
	  offsetL += arraySizeL;
	}

	// set qright_x by re-reading qp_x from external memory at location x
	offsetL = elemOffsetL;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_x[iVar] = d_qp_x[offsetL];
	  offsetL += arraySizeL;
	}

	riemann_mhd(qleft_x, qright_x, flux_x[tx][ty]);
      }  
    __syncthreads();
    
    // update uOut with flux_x
    if(i  >= 3 and i  < imax-3 and tx < UPDATE_P1_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-3 and ty < UPDATE_P1_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-3)
      {
	// re-read input state into uOut which will in turn serve to
	// update Uout !
	int offsetU = elemOffsetL + kStart*pitch*jmax;
	uOut[ID] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IP] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IU] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IV] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IW] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IA] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IB] = Uin[offsetU];  offsetU += arraySizeU;
	uOut[IC] = Uin[offsetU];

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
	  int offsetShear = j+pitchB*kL;
	  d_shear_flux_xmin[offsetShear] = flux_x[tx  ][ty][ID]*dtdx; // I_DENS
	}
	if (i==imax-4 and ::gParams.mpiPosX == (::gParams.mx - 1) ) {
	  /* store flux_xmax */
	  int offsetShear = j+pitchB*kL;
	  d_shear_flux_xmax[offsetShear] = flux_x[tx+1][ty][ID]*dtdx; // I_DENS	  
	}
	
      }
    __syncthreads();

    /*
     * Compute fluxes at Y-interfaces.
     */
    // re-use flux as flux_y
    real_t (&flux_y)[UPDATE_P1_Z_BLOCK_DIMX_3D_V4][UPDATE_P1_Z_BLOCK_DIMY_3D_V4][NVAR_MHD] = flux;
    flux_y[tx][ty][ID] = ZERO_F;
    flux_y[tx][ty][IP] = ZERO_F;
    flux_y[tx][ty][IU] = ZERO_F;
    flux_y[tx][ty][IV] = ZERO_F;
    flux_y[tx][ty][IW] = ZERO_F;
    flux_y[tx][ty][IA] = ZERO_F;
    flux_y[tx][ty][IB] = ZERO_F;
    flux_y[tx][ty][IC] = ZERO_F;
    __syncthreads();
    
    if(i  >= 3 and i  < imax-2 and 
       j  >= 3 and j  < jmax-2 and 
       kL >= 3 and kL < ksizeSlab-2)
      {
	// Solve Riemann problem at Y-interfaces and compute fluxes
	real_t  qleft_y[NVAR_MHD];
	real_t qright_y[NVAR_MHD];
	
	// set qleft_y by reading qm_y from external memory at location y-1
	int offsetL = elemOffsetL-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_y[iVar] = d_qm_y[offsetL];
	  offsetL += arraySizeL;
	}
	
	// set qright_y by reading qp_y from external memory at location y
	offsetL = elemOffsetL;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_y[iVar] = d_qp_y[offsetL];
	  offsetL += arraySizeL;
	}
	
	// watchout swap IU and IV
	swap_val_(qleft_y[IU],qleft_y[IV]);
	swap_val_(qleft_y[IA],qleft_y[IB]);
	swap_val_(qright_y[IU],qright_y[IV]);
	swap_val_(qright_y[IA],qright_y[IB]);

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
    if(i  >= 3 and i  < imax-3 and tx < UPDATE_P1_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-3 and ty < UPDATE_P1_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-3)
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
    
    if(i  >= 3 and i  < imax-2 and tx < UPDATE_P1_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-2 and ty < UPDATE_P1_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-2)
      {
	// Solve Riemann problem at Z-interfaces and compute fluxes
	real_t qleft_z [NVAR_MHD];
	real_t qright_z[NVAR_MHD];
	
	// set qleft_z by reading qm_z from external memory at location z-1
	int offsetL = elemOffsetL - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qleft_z[iVar] = d_qm_z[offsetL];
	  offsetL += arraySizeL;
	}
	
        // set qright_z by reading qp_z from external memory at location z
        offsetL = elemOffsetL;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qright_z[iVar] = d_qp_z[offsetL];
	  offsetL += arraySizeL;
	}
	
	// watchout swap IU and IW
	swap_val_(qleft_z[IU] ,qleft_z[IW]);
	swap_val_(qleft_z[IA] ,qleft_z[IC]);
	swap_val_(qright_z[IU],qright_z[IW]);
	swap_val_(qright_z[IA],qright_z[IC]);
	
	riemann_mhd(qleft_z, qright_z, flux_z);
	
      }  
    __syncthreads();
    
    // update uOut with flux_z
    if(i  >= 3 and i  < imax-3 and tx < UPDATE_P1_Z_BLOCK_DIMX_3D_V4-1 and
       j  >= 3 and j  < jmax-3 and ty < UPDATE_P1_Z_BLOCK_DIMY_3D_V4-1 and
       kL >= 3 and kL < ksizeSlab-2)
      {
	/*
	 * take care that update with flux_z is separated in two stages !!!
	 */

    	/*
    	 * update current position z.
    	 */
	int offsetU = elemOffsetL + kStart*pitch*jmax;

	if (kL < ksizeSlab-3) {
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
	  Uout[offsetU] = uOut[ID];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IP];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IU];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IV];  offsetU += arraySizeU;
	  Uout[offsetU] = uOut[IW];
	}

	if (kL > 3) { 
	  /*
	   * update at position z-1.
	   * Note that position z-1 has already been partialy updated in
	   * the previous iteration (for loop over k).
	   */
	  // watchout! IU and IW are swapped !
	  offsetU = elemOffsetL + kStart*pitch*jmax - pitch*jmax;
	  {
	    Uout[offsetU] -= flux_z[ID]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IP]*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= (alpha1*flux_z[IW]+     
			      alpha2*flux_z[IV])*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= (alpha1*flux_z[IV]-0.25*
			      alpha2*flux_z[IW])*dtdz; offsetU += arraySizeU;
	    Uout[offsetU] -= flux_z[IU]*dtdz;
	  } 
	}
      } // end update along Z
    __syncthreads();

  } // end for k

} // kernel_mhd_flux_update_hydro_v4_shear_part1_zslab

#ifdef USE_DOUBLE
#define COMPUTE_EMF_Z_BLOCK_DIMX_3D_SHEAR	16
#define COMPUTE_EMF_Z_BLOCK_DIMY_3D_SHEAR	16
#else // simple precision
#define COMPUTE_EMF_Z_BLOCK_DIMX_3D_SHEAR	16
#define COMPUTE_EMF_Z_BLOCK_DIMY_3D_SHEAR	16
#endif // USE_DOUBLE

/**
 * MHD compute emf for shearing box simulations, store them in d_emf, zSlab version.
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
__global__ void  kernel_mhd_compute_emf_shear_zslab(const real_t * __restrict__ d_qEdge_RT,
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
						    real_t *d_emf,
						    real_t *d_shear_flux_xmin,
						    real_t *d_shear_flux_xmax,
						    int pitch, 
						    int imax, 
						    int jmax,
						    int kmax,
						    int pitchB,
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
  
  const int i = __mul24(bx, COMPUTE_EMF_Z_BLOCK_DIMX_3D_SHEAR) + tx;
  const int j = __mul24(by, COMPUTE_EMF_Z_BLOCK_DIMY_3D_SHEAR) + ty;
  
  //const int arraySizeU    = pitch * jmax * kmax;
  const int arraySizeQ    = pitch * jmax * zSlabInfo.zSlabWidthG;
  const int &arraySizeL   = arraySizeQ;
  const int arraySize2d   =       pitchB * zSlabInfo.zSlabWidthG; // d_shear_flux_xmin/xmax size

  //const int &kStart       = zSlabInfo.kStart; 
  //const int &kStop        = zSlabInfo.kStop; 
  const int &ksizeSlab    = zSlabInfo.ksizeSlab;

  // x position in real space
  real_t xPos = ::gParams.xMin + ::gParams.dx/2 + (i-3 + ::gParams.nx * ::gParams.mpiPosX)*(::gParams.dx); // needed for shear computation

  /*
   * loop over k (i.e. z) to compute trace
   */
  for (int kL=1, elemOffsetL = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-2; 
       ++kL, elemOffsetL += (pitch*jmax)) {
    
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
    
    if(i  > 2 and i  < imax-2 and
       j  > 2 and j  < jmax-2 and
       kL > 2 and kL < ksizeSlab-2)
      {
	
	/*
	 * offset into external memory array to qEdge data
	 */

	int offset2L         = elemOffsetL;
 
	
	/*
	 * compute emfZ
	 */

	// qEdge_RT3 at location x-1, y-1
	int offsetL = offset2L-1-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT3[iVar] = d_qEdge_RT3[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge RB3 at location x-1, y
	offsetL = offset2L-1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB3[iVar] = d_qEdge_RB3[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LT3 at location x, y-1
	offsetL = offset2L-pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT3[iVar] = d_qEdge_LT3[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LB3 at location x, y
	offsetL = offset2L;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB3[iVar] = d_qEdge_LB3[offsetL];
	  offsetL += arraySizeL;
	}

	// finally compute emfZ
	emf = compute_emf<EMFZ>(qEdge_emfZ,xPos);
	offsetL = offset2L + I_EMFZ*arraySizeL;
	if (kL<ksizeSlab-3)
	  d_emf[offsetL] = emf;

	/*
	 * compute emfY (take care RB and LT are swapped)
	 */

	// qEdge_RT2 at location x-1, z-1
	offsetL = offset2L - pitch*jmax - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT2[iVar] = d_qEdge_RT2[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge RB2 (actually LT2) at location x, z-1
	offsetL = offset2L - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB2[iVar] = d_qEdge_LT2[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LT2 (actually RB2) at location x-1, z
	offsetL = offset2L - 1;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT2[iVar] = d_qEdge_RB2[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LB2 at location x, y
	offsetL = offset2L;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB2[iVar] = d_qEdge_LB2[offsetL];
	  offsetL += arraySizeL;
	}

	// finally compute emfY
	emf = compute_emf<EMFY>(qEdge_emfY,xPos);
	offsetL = offset2L + I_EMFY*arraySizeL;
	if (j<jmax-3) {
	  d_emf[offsetL] = emf;
	  
	  // at global XMIN border, store emfY
	  if (i == 3     and ::gParams.mpiPosX == 0) {
	    int offsetShear = j+pitchB*kL;
	    d_shear_flux_xmin[offsetShear + arraySize2d*I_EMF_Y] = emf;
	  }
	  
	  // at global XMAX border, store emfY
	  if (i == imax-3 and ::gParams.mpiPosX == (::gParams.mx - 1) ) {
	    int offsetShear = j+pitchB*kL;
	    d_shear_flux_xmax[offsetShear + arraySize2d*I_EMF_Y] = emf;
	  }
	} // end if j<(jmax-3)

	/*
	 * compute emfX
	 */
	// qEdge_RT at location y-1, z-1
	offsetL = offset2L - pitch*jmax - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RT[iVar] = d_qEdge_RT[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge RB at location y-1, z
	offsetL = offset2L - pitch;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_RB[iVar] = d_qEdge_RB[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LT at location y, z-1
	offsetL = offset2L - pitch*jmax;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LT[iVar] = d_qEdge_LT[offsetL];
	  offsetL += arraySizeL;
	}
	
	// qEdge_LB at location y, z
	offsetL = offset2L;
	for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	  qEdge_LB[iVar] = d_qEdge_LB[offsetL];
	  offsetL += arraySizeL;
	}

	// finally compute emfX
	emf = compute_emf<EMFX>(qEdge_emfX,xPos);
	offsetL = offset2L + I_EMFX*arraySizeL;
	if (i<imax-3)
	  d_emf[offsetL] = emf;	
      }
    __syncthreads();
            
  } // end for k

} // kernel_mhd_compute_emf_shear_zslab

#endif // GODUNOV_UNSPLIT_MHD_ZSLAB_CUH_
