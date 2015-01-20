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
 * \file viscosity.cuh
 * \brief CUDA kernel for computing viscosity forces (adapted from Dumses).
 *
 * \date 29 Apr 2012
 * \author P. Kestener
 *
 * $Id: viscosity.cuh 2216 2012-07-18 08:28:15Z pkestene $
 */
#ifndef VISCOSITY_CUH_
#define VISCOSITY_CUH_

#include "real_type.h"
#include "constants.h"


#ifdef USE_DOUBLE
#define VISCOSITY_2D_DIMX	24
#define VISCOSITY_2D_DIMY	16
#define VISCOSITY_2D_DIMX_INNER	(VISCOSITY_2D_DIMX-2)
#define VISCOSITY_2D_DIMY_INNER	(VISCOSITY_2D_DIMY-2)
#else // simple precision
#define VISCOSITY_2D_DIMX	24
#define VISCOSITY_2D_DIMY	16
#define VISCOSITY_2D_DIMX_INNER	(VISCOSITY_2D_DIMX-2)
#define VISCOSITY_2D_DIMY_INNER	(VISCOSITY_2D_DIMY-2)
#endif // USE_DOUBLE

/**
 * CUDA kernel computing viscosity forces (2D data).
 * 
 * \param[in]  Uin    conservative variables array.
 * \param[out] flux_x viscosity forces flux along X.
 * \param[out] flux_y viscosity forces flux along Y.
 * \param[in]  dt     time step
 */
__global__ void kernel_viscosity_forces_2d(real_t* Uin, 
					   real_t* flux_x,
					   real_t* flux_y,
					   int ghostWidth,
					   int pitch, 
					   int imax, 
					   int jmax, 
					   real_t dt,
					   real_t dx, 
					   real_t dy)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, VISCOSITY_2D_DIMX_INNER) + tx;
  const int j = __mul24(by, VISCOSITY_2D_DIMY_INNER) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  __shared__ real_t   uIn[VISCOSITY_2D_DIMX][VISCOSITY_2D_DIMY][NVAR_2D];

  real_t dudx[2], dudy[2];
  
  real_t &cIso = ::gParams.cIso;
  real_t &nu   = ::gParams.nu;
  const real_t two3rd = 2./3.;

  // load Uin into shared memory
  if(i >= 0 and i < imax and 
     j >= 0 and j < jmax)
    {
      
      // Gather conservative variables
      int offset = elemOffset;
      uIn[tx][ty][ID] = Uin[offset];                  offset += arraySize;
      uIn[tx][ty][IP] = Uin[offset];                  offset += arraySize;
      uIn[tx][ty][IU] = Uin[offset]/uIn[tx][ty][ID];  offset += arraySize;
      uIn[tx][ty][IV] = Uin[offset]/uIn[tx][ty][ID];

    }
  __syncthreads();

  if (i>=ghostWidth and i<imax-ghostWidth+1 and tx > 0 and tx<VISCOSITY_2D_DIMX-1 and
      j>=ghostWidth and j<jmax-ghostWidth+1 and ty > 0 and ty<VISCOSITY_2D_DIMY-1) 
    {
      real_t u,v;
      real_t uR, uL;
      real_t uRR, uRL, uLR, uLL;
      real_t txx,tyy,txy;
      real_t rho;
      /*
       * 1st direction viscous flux
       */
      rho = HALF_F * ( uIn[tx  ][ty][ID] + uIn[tx-1][ty][ID] );
      
      if (cIso <= 0) {
	u = HALF_F * ( uIn[tx  ][ty][IU] + uIn[tx-1][ty][IU] );
	v = HALF_F * ( uIn[tx  ][ty][IV] + uIn[tx-1][ty][IV] );
      }
      
      // dudx along X
      uR = uIn[tx  ][ty][IU];
      uL = uIn[tx-1][ty][IU];
      dudx[IX] = (uR-uL)/dx;
      
      // dudx along Y
      uR = uIn[tx  ][ty][IV];
      uL = uIn[tx-1][ty][IV];
      dudx[IY] = (uR-uL)/dx;
      
      // dudy along X
      uRR = uIn[tx  ][ty+1][IU];
      uRL = uIn[tx-1][ty+1][IU];
      uLR = uIn[tx  ][ty-1][IU];
      uLL = uIn[tx-1][ty-1][IU];
      uR  = uRR+uRL; 
      uL  = uLR+uLL;
      dudy[IX] = (uR-uL)/dy/4;
      
      // dudy along Y
      uRR = uIn[tx  ][ty+1][IV];
      uRL = uIn[tx-1][ty+1][IV];
      uLR = uIn[tx  ][ty-1][IV];
      uLL = uIn[tx-1][ty-1][IV];
      uR  = uRR+uRL; 
      uL  = uLR+uLL;
      dudy[IY] = (uR-uL)/dy/4;
      
      txx = -two3rd *nu * rho * ( TWO_F*dudx[IX] - dudy[IY] );
      txy = -        nu * rho * (       dudy[IX] + dudx[IY] );
     
      // save results in flux_x
      int offset = elemOffset;
      flux_x[offset] = ZERO_F;  offset += arraySize;
      if (cIso <= 0) {
	flux_x[offset] = (u*txx+v*txy)*dt/dx; offset += arraySize;
      } else {
	flux_x[offset] = ZERO_F; offset += arraySize;
      }
      flux_x[offset] = txx*dt/dx; offset += arraySize;
      flux_x[offset] = txy*dt/dx;

      
      /*
       * 2nd direction viscous flux
       */
      rho = HALF_F * ( uIn[tx][ty  ][ID] + 
		       uIn[tx][ty-1][ID]);
      if (cIso <=0) {
	u = HALF_F * ( uIn[tx][ty  ][IU] + uIn[tx][ty-1][IU] );
	v = HALF_F * ( uIn[tx][ty  ][IV] + uIn[tx][ty-1][IV] );
      }
      
      // dudy along X
      uR = uIn[tx][ty  ][IU];
      uL = uIn[tx][ty-1][IU];
      dudy[IX] = (uR-uL)/dy;
      
      // dudy along Y
      uR = uIn[tx][ty  ][IV];
      uL = uIn[tx][ty-1][IV];
      dudy[IY] = (uR-uL)/dy;
      
      // dudx along X
      uRR = uIn[tx+1][ty  ][IU];
      uRL = uIn[tx+1][ty-1][IU];
      uLR = uIn[tx-1][ty  ][IU];
      uLL = uIn[tx-1][ty-1][IU];
      uR  = uRR+uRL; 
      uL  = uLR+uLL;
      dudx[IX] = (uR-uL)/dx/4;
      
      // dudx along Y
      uRR = uIn[tx+1][ty  ][IV];
      uRL = uIn[tx+1][ty-1][IV];
      uLR = uIn[tx-1][ty  ][IV];
      uLL = uIn[tx-1][ty-1][IV];
      uR  = uRR+uRL; 
      uL  = uLR+uLL;
      dudx[IY] = (uR-uL)/dx/4;
      
      tyy = -two3rd * nu * rho * ( TWO_F * dudy[IY] - dudx[IX] );
      txy = -         nu * rho * (         dudy[IX] + dudx[IY] );
      
      // dave results in flux_y
      offset = elemOffset;
      flux_y[offset] = ZERO_F; offset += arraySize;
      if (cIso <=0) {
	flux_y[offset] = (u*txy+v*tyy)*dt/dy; offset += arraySize;
      } else {
	flux_y[offset] = ZERO_F; offset += arraySize;
      }
      flux_y[offset] = txy*dt/dy; offset += arraySize;
      flux_y[offset] = tyy*dt/dy;
      
    } // end compute viscosity forces flux

} // kernel_viscosity_forces_2d



__global__ void kernel_viscosity_forces_2d_old(real_t* Uin, 
					       real_t* flux_x,
					       real_t* flux_y,
					       int ghostWidth,
					       int pitch, 
					       int imax, 
					       int jmax, 
					       real_t dt,
					       real_t dx, 
					       real_t dy)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, VISCOSITY_2D_DIMX_INNER) + tx;
  const int j = __mul24(by, VISCOSITY_2D_DIMY_INNER) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  __shared__ real_t   uIn[VISCOSITY_2D_DIMX][VISCOSITY_2D_DIMY][NVAR_2D];

  real_t dudx[2], dudy[2];
  
  real_t &cIso = ::gParams.cIso;
  real_t &nu   = ::gParams.nu;
  const real_t two3rd = 2./3.;

  // load Uin into shared memory
  if(i >= 0 and i < imax and 
     j >= 0 and j < jmax)
    {
      
      // Gather conservative variables
      int offset = elemOffset;
      uIn[tx][ty][ID] = Uin[offset];  offset += arraySize;
      uIn[tx][ty][IP] = Uin[offset];  offset += arraySize;
      uIn[tx][ty][IU] = Uin[offset];  offset += arraySize;
      uIn[tx][ty][IV] = Uin[offset];

    }
  __syncthreads();

  if (i>=ghostWidth and i<imax-ghostWidth+1 and tx > 0 and tx<VISCOSITY_2D_DIMX-1 and
      j>=ghostWidth and j<jmax-ghostWidth+1 and ty > 0 and ty<VISCOSITY_2D_DIMY-1) 
    {
      real_t u,v;
      real_t uR, uL;
      real_t uRR, uRL, uLR, uLL;
      real_t txx,tyy,txy;
      real_t rho;
      /*
       * 1st direction viscous flux
       */
      rho = HALF_F * ( uIn[tx  ][ty][ID] + 
		       uIn[tx-1][ty][ID] );
      
      if (cIso <= 0) {
	u = HALF_F * ( uIn[tx  ][ty][IU] / uIn[tx  ][ty][ID] + 
		       uIn[tx-1][ty][IU] / uIn[tx-1][ty][ID] );
	v = HALF_F * ( uIn[tx  ][ty][IV] / uIn[tx  ][ty][ID] + 
		       uIn[tx-1][ty][IV] / uIn[tx-1][ty][ID] );
      }
      
      // dudx along X
      uR = uIn[tx  ][ty][IU] / uIn[tx  ][ty][ID];
      uL = uIn[tx-1][ty][IU] / uIn[tx-1][ty][ID];
      dudx[IX] = (uR-uL)/dx;
      
      // dudx along Y
      uR = uIn[tx  ][ty][IV] / uIn[tx  ][ty][ID];
      uL = uIn[tx-1][ty][IV] / uIn[tx-1][ty][ID];
      dudx[IY] = (uR-uL)/dx;
      
      // dudy along X
      uRR = uIn[tx  ][ty+1][IU] / uIn[tx  ][ty+1][ID];
      uRL = uIn[tx-1][ty+1][IU] / uIn[tx-1][ty+1][ID];
      uLR = uIn[tx  ][ty-1][IU] / uIn[tx  ][ty-1][ID];
      uLL = uIn[tx-1][ty-1][IU] / uIn[tx-1][ty-1][ID];
      uR  = uRR+uRL; 
      uL  = uLR+uLL;
      dudy[IX] = (uR-uL)/dy/4;
      
      // dudy along Y
      uRR = uIn[tx  ][ty+1][IV] / uIn[tx  ][ty+1][ID];
      uRL = uIn[tx-1][ty+1][IV] / uIn[tx-1][ty+1][ID];
      uLR = uIn[tx  ][ty-1][IV] / uIn[tx  ][ty-1][ID];
      uLL = uIn[tx-1][ty-1][IV] / uIn[tx-1][ty-1][ID];
      uR  = uRR+uRL; 
      uL  = uLR+uLL;
      dudy[IY] = (uR-uL)/dy/4;
      
      txx = -two3rd *nu * rho * ( TWO_F*dudx[IX] - dudy[IY] );
      txy = -        nu * rho * (       dudy[IX] + dudx[IY] );
     
      // save results in flux_x
      int offset = elemOffset;
      flux_x[offset] = ZERO_F;  offset += arraySize;
      if (cIso <= 0) {
	flux_x[offset] = (u*txx+v*txy)*dt/dx; offset += arraySize;
      } else {
	flux_x[offset] = ZERO_F; offset += arraySize;
      }
      flux_x[offset] = txx*dt/dx; offset += arraySize;
      flux_x[offset] = txy*dt/dx;

      
      /*
       * 2nd direction viscous flux
       */
      rho = HALF_F * ( uIn[tx][ty  ][ID] + 
		       uIn[tx][ty-1][ID]);
      if (cIso <=0) {
	u = HALF_F * ( uIn[tx][ty  ][IU] / uIn[tx][ty  ][ID] +
		       uIn[tx][ty-1][IU] / uIn[tx][ty-1][ID] );
	v = HALF_F * ( uIn[tx][ty  ][IV] / uIn[tx][ty  ][ID] +
		       uIn[tx][ty-1][IV] / uIn[tx][ty-1][ID] );
      }
      
      // dudy along X
      uR = uIn[tx][ty  ][IU] / uIn[tx][ty  ][ID];
      uL = uIn[tx][ty-1][IU] / uIn[tx][ty-1][ID];
      dudy[IX] = (uR-uL)/dy;
      
      // dudy along Y
      uR = uIn[tx][ty  ][IV] / uIn[tx][ty  ][ID];
      uL = uIn[tx][ty-1][IV] / uIn[tx][ty-1][ID];
      dudy[IY] = (uR-uL)/dy;
      
      // dudx along X
      uRR = uIn[tx+1][ty  ][IU] / uIn[tx+1][ty  ][ID];
      uRL = uIn[tx+1][ty-1][IU] / uIn[tx+1][ty-1][ID];
      uLR = uIn[tx-1][ty  ][IU] / uIn[tx-1][ty  ][ID];
      uLL = uIn[tx-1][ty-1][IU] / uIn[tx-1][ty-1][ID];
      uR  = uRR+uRL; 
      uL  = uLR+uLL;
      dudx[IX] = (uR-uL)/dx/4;
      
      // dudx along Y
      uRR = uIn[tx+1][ty  ][IV] / uIn[tx+1][ty  ][ID];
      uRL = uIn[tx+1][ty-1][IV] / uIn[tx+1][ty-1][ID];
      uLR = uIn[tx-1][ty  ][IV] / uIn[tx-1][ty  ][ID];
      uLL = uIn[tx-1][ty-1][IV] / uIn[tx-1][ty-1][ID];
      uR  = uRR+uRL; 
      uL  = uLR+uLL;
      dudx[IY] = (uR-uL)/dx/4;
      
      tyy = -two3rd * nu * rho * ( TWO_F * dudy[IY] - dudx[IX] );
      txy = -         nu * rho * (         dudy[IX] + dudx[IY] );
      
      // dave results in flux_y
      offset = elemOffset;
      flux_y[offset] = ZERO_F; offset += arraySize;
      if (cIso <=0) {
	flux_y[offset] = (u*txy+v*tyy)*dt/dy; offset += arraySize;
      } else {
	flux_y[offset] = ZERO_F; offset += arraySize;
      }
      flux_y[offset] = txy*dt/dy; offset += arraySize;
      flux_y[offset] = tyy*dt/dy;
      
    } // end compute viscosity forces flux

} // kernel_viscosity_forces_2d_old

//
// TODO : BENCHMARK / OPTIMIZE block sizes for double precision !!!
//
#ifdef USE_DOUBLE
#define VISCOSITY_3D_DIMX	48
#define VISCOSITY_3D_DIMY	8
#define VISCOSITY_3D_DIMX_INNER	(VISCOSITY_3D_DIMX-2)
#define VISCOSITY_3D_DIMY_INNER	(VISCOSITY_3D_DIMY-2)
#else // simple precision
#define VISCOSITY_3D_DIMX	48
#define VISCOSITY_3D_DIMY	10
#define VISCOSITY_3D_DIMX_INNER	(VISCOSITY_3D_DIMX-2)
#define VISCOSITY_3D_DIMY_INNER	(VISCOSITY_3D_DIMY-2)
#endif // USE_DOUBLE

/**
 * CUDA kernel computing viscosity forces (3D data).
 * 
 * \param[in]  Uin    conservative variables array.
 * \param[out] flux_x viscosity forces flux along X.
 * \param[out] flux_y viscosity forces flux along Y.
 * \param[out] flux_z viscosity forces flux along Z.
 * \param[in]  dt     time step
 */
__global__ void kernel_viscosity_forces_3d(real_t* Uin, 
					   real_t* flux_x,
					   real_t* flux_y,
					   real_t* flux_z,
					   int ghostWidth,
					   int pitch, 
					   int imax, 
					   int jmax,
					   int kmax,
					   real_t dt,
					   real_t dx, 
					   real_t dy,
					   real_t dz)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, VISCOSITY_3D_DIMX_INNER) + tx;
  const int j = __mul24(by, VISCOSITY_3D_DIMY_INNER) + ty;
  
  const int arraySize  = pitch * jmax * kmax;

  // we store 3 consecutive plans of data
  __shared__ real_t   uIn[3][VISCOSITY_3D_DIMX][VISCOSITY_3D_DIMY][NVAR_3D];

  // index to address the 3 plans of data
  int low, mid, top, tmp;
  low=0;
  mid=1;
  top=2;

  real_t dudx[3], dudy[3], dudz[3];
  
  real_t &cIso = ::gParams.cIso;
  real_t &nu   = ::gParams.nu;
  const real_t two3rd = 2./3.;

  /*
   * initialize uIn with the first 3 plans 
   * we could start at ghostWidth-1, but for simplicity start at 0 !
   */
  for (int k=0, elemOffset = i + pitch * j; 
       k < 3;
       ++k, elemOffset += (pitch*jmax) ) {

    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax)
      {
	
	// Gather conservative variables
	int offset = elemOffset;
	uIn[k][tx][ty][ID] = Uin[offset];                     offset += arraySize;
	uIn[k][tx][ty][IP] = Uin[offset];                     offset += arraySize;
	uIn[k][tx][ty][IU] = Uin[offset]/uIn[k][tx][ty][ID];  offset += arraySize;
	uIn[k][tx][ty][IV] = Uin[offset]/uIn[k][tx][ty][ID];  offset += arraySize;
	uIn[k][tx][ty][IW] = Uin[offset]/uIn[k][tx][ty][ID];
	
      }
  } // end loading the first 3 plans
  __syncthreads();

  // loop over k starting at k=1 - compute viscosity force fluxes
  // we could for simplicity start at k=ghostWidth (see above)
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {

    if (i>=ghostWidth and i<imax-ghostWidth+1 and tx > 0 and tx<VISCOSITY_3D_DIMX-1 and
	j>=ghostWidth and j<jmax-ghostWidth+1 and ty > 0 and ty<VISCOSITY_3D_DIMY-1) 
      {
	real_t u,v,w;
	real_t uR, uL;
	real_t uRR, uRL, uLR, uLL;
	real_t txx,tyy,tzz,txy,txz,tyz;
	real_t rho;
	
	/*
	 * 1st direction viscous flux
	 */
	rho = HALF_F * ( uIn[mid][tx  ][ty][ID] + 
			 uIn[mid][tx-1][ty][ID] );
	
	if (cIso <=0) {
	  u  = HALF_F * ( uIn[mid][tx  ][ty][IU] + 
			  uIn[mid][tx-1][ty][IU] );
	  v  = HALF_F * ( uIn[mid][tx  ][ty][IV] + 
			  uIn[mid][tx-1][ty][IV] );
	  w  = HALF_F * ( uIn[mid][tx  ][ty][IW] + 
			  uIn[mid][tx-1][ty][IW] );
	}
	
	// dudx along X
	uR = uIn[mid][tx  ][ty][IU];
	uL = uIn[mid][tx-1][ty][IU];
	dudx[IX]=(uR-uL)/dx;
	
	// dudx along Y
	uR = uIn[mid][tx  ][ty][IV];
	uL = uIn[mid][tx-1][ty][IV];
	dudx[IY]=(uR-uL)/dx;
	
	// dudx along Z
	uR = uIn[mid][tx  ][ty][IW];
	uL = uIn[mid][tx-1][ty][IW];
	dudx[IZ]=(uR-uL)/dx;
	
	
	// dudy along X
	uRR = uIn[mid][tx  ][ty+1][IU];
	uRL = uIn[mid][tx-1][ty+1][IU];
	uLR = uIn[mid][tx  ][ty-1][IU];
	uLL = uIn[mid][tx-1][ty-1][IU];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudy[IX] = (uR-uL)/dy/4;
	
	// dudy along Y
	uRR = uIn[mid][tx  ][ty+1][IV];
	uRL = uIn[mid][tx-1][ty+1][IV];
	uLR = uIn[mid][tx  ][ty-1][IV];
	uLL = uIn[mid][tx-1][ty-1][IV];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudy[IY] = (uR-uL)/dy/4;
	
	// dudz along X
	uRR = uIn[top][tx  ][ty][IU];
	uRL = uIn[top][tx-1][ty][IU];
	uLR = uIn[low][tx  ][ty][IU];
	uLL = uIn[low][tx-1][ty][IU];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudz[IX] = (uR-uL)/dz/4;
	
	// dudz along Z
	uRR = uIn[top][tx  ][ty][IW];
	uRL = uIn[top][tx-1][ty][IW];
	uLR = uIn[low][tx  ][ty][IW];
	uLL = uIn[low][tx-1][ty][IW];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudz[IZ] = (uR-uL)/dz/4;
	
	txx = -two3rd * nu * rho * (TWO_F * dudx[IX] - dudy[IY] - dudz[IZ]);
	txy = -         nu * rho * (        dudy[IX] + dudx[IY]           );
	txz = -         nu * rho * (        dudz[IX] + dudx[IZ]           );

	// save results in flux_x
	int offset = elemOffset;
	if (k >= ghostWidth and k<kmax-ghostWidth+1) {
	  offset = elemOffset;
	  flux_x[offset] = ZERO_F;     offset += arraySize; // ID
	  if (cIso <= 0) { // IP
	    flux_x[offset] = (u*txx+v*txy+w*txz)*dt/dx; offset += arraySize;
	  } else {
	    flux_x[offset] = ZERO_F;                    offset += arraySize;
	  }
	  
	  flux_x[offset] = txx*dt/dx;  offset += arraySize; // IU
	  flux_x[offset] = txy*dt/dx;  offset += arraySize; // IV
	  flux_x[offset] = txz*dt/dx;  offset += arraySize; // IW
	}

	/*
	 * 2nd direction viscous flux
	 */
	rho = HALF_F * ( uIn[mid][tx][ty][ID] + uIn[mid][tx][ty-1][ID] );
	
	if (cIso <= 0) {
	  u = HALF_F * ( uIn[mid][tx][ty  ][IU] + 
			 uIn[mid][tx][ty-1][IU] );
	  v = HALF_F * ( uIn[mid][tx][ty  ][IV] + 
			 uIn[mid][tx][ty-1][IV] );
	  w = HALF_F * ( uIn[mid][tx][ty  ][IW] +
			 uIn[mid][tx][ty-1][IW] );
	}
	
	// dudy along X
	uR = uIn[mid][tx][ty  ][IU];
	uL = uIn[mid][tx][ty-1][IU];
	dudy[IX] = (uR-uL)/dy;
	
	// dudy along Y
	uR = uIn[mid][tx][ty  ][IV];
	uL = uIn[mid][tx][ty-1][IV];
	dudy[IY] = (uR-uL)/dy;
	
	// dudy along Z
	uR = uIn[mid][tx][ty  ][IW];
	uL = uIn[mid][tx][ty-1][IW];
	dudy[IZ] = (uR-uL)/dy;
	
	// dudx along X
	uRR = uIn[mid][tx+1][ty  ][IU];
	uRL = uIn[mid][tx+1][ty-1][IU];
	uLR = uIn[mid][tx-1][ty  ][IU];
	uLL = uIn[mid][tx-1][ty-1][IU];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudx[IX]=(uR-uL)/dx/4;
	
	// dudx along Y
	uRR = uIn[mid][tx+1][ty  ][IV];
	uRL = uIn[mid][tx+1][ty-1][IV];
	uLR = uIn[mid][tx-1][ty  ][IV];
	uLL = uIn[mid][tx-1][ty-1][IV];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudx[IY]=(uR-uL)/dx/4;
	
	// dudz along Y
	uRR = uIn[top][tx][ty  ][IV];
	uRL = uIn[top][tx][ty-1][IV];
	uLR = uIn[low][tx][ty  ][IV];
	uLL = uIn[low][tx][ty-1][IV];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudz[IY]=(uR-uL)/dz/4;
	
	// dudz along Z
	uRR = uIn[top][tx][ty  ][IW];
	uRL = uIn[top][tx][ty-1][IW];
	uLR = uIn[low][tx][ty  ][IW];
	uLL = uIn[low][tx][ty-1][IW];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudz[IZ]=(uR-uL)/dz/4;
	
	tyy = -two3rd * nu * rho * (TWO_F * dudy[IY] - dudx[IX] - dudz[IZ] );
	txy = -         nu * rho * (        dudy[IX] + dudx[IY]            );
	tyz = -         nu * rho * (        dudz[IY] + dudy[IZ]            );

	// save results in flux_y
	if (k >= ghostWidth and k<kmax-ghostWidth+1) {
	  offset = elemOffset;
	  flux_y[offset] = ZERO_F;  offset += arraySize; // ID
	  if (cIso <= 0) {
	    flux_y[offset] = (u*txy+v*tyy+w*tyz)*dt/dy;  offset += arraySize; // IP
	  } else {
	    flux_y[offset] = ZERO_F;                     offset += arraySize; // IP
	  }
	  flux_y[offset] = txy*dt/dy;  offset += arraySize; // IU
	  flux_y[offset] = tyy*dt/dy;  offset += arraySize; // IV
	  flux_y[offset] = tyz*dt/dy;  offset += arraySize; // IW
	}
	
	/*
	 * 3rd direction viscous flux
	 */
	rho = HALF_F * ( uIn[mid][tx][ty][ID] + uIn[low][tx][ty][ID] );
	
	if (cIso <= 0) {
	  u = HALF_F * ( uIn[mid][tx][ty][IU] + 
			 uIn[low][tx][ty][IU] );
	  v = HALF_F * ( uIn[mid][tx][ty][IV] + 
			 uIn[low][tx][ty][IV] );
	  w = HALF_F * ( uIn[mid][tx][ty][IW] + 
			 uIn[low][tx][ty][IW] );
	}
	
	// dudz along X
	uR = uIn[mid][tx][ty][IU];
	uL = uIn[low][tx][ty][IU];
	dudz[IX] = (uR-uL)/dz;
	
	// dudz along Y
	uR = uIn[mid][tx][ty][IV];
	uL = uIn[low][tx][ty][IV];
	dudz[IY] = (uR-uL)/dz;
	
	// dudz along Z
	uR = uIn[mid][tx][ty][IW];
	uL = uIn[low][tx][ty][IW];
	dudz[IZ] = (uR-uL)/dz;
	
	// dudx along X
	uRR = uIn[mid][tx+1][ty][IU];
	uRL = uIn[low][tx+1][ty][IU];
	uLR = uIn[mid][tx-1][ty][IU];
	uLL = uIn[low][tx-1][ty][IU];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudx[IX] = (uR-uL)/dx/4;
	
	// dudx along Z
	uRR = uIn[mid][tx+1][ty][IW];
	uRL = uIn[low][tx+1][ty][IW];
	uLR = uIn[mid][tx-1][ty][IW];
	uLL = uIn[low][tx-1][ty][IW];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudx[IZ] = (uR-uL)/dx/4;
	
	// dudy along Y
	uRR = uIn[mid][tx][ty+1][IV];
	uRL = uIn[low][tx][ty+1][IV];
	uLR = uIn[mid][tx][ty-1][IV];
	uLL = uIn[low][tx][ty-1][IV];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudy[IY] = (uR-uL)/dy/4;
	
	// dudy along Z
	uRR = uIn[mid][tx][ty+1][IW];
	uRL = uIn[low][tx][ty+1][IW];
	uLR = uIn[mid][tx][ty-1][IW];
	uLL = uIn[low][tx][ty-1][IW];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudy[IZ] = (uR-uL)/dy/4;
	
	
	tzz = -two3rd * nu * rho * (TWO_F * dudz[IZ] - dudx[IX] - dudy[IY] );
	txz = -         nu * rho * (        dudz[IX] + dudx[IZ]            );
	tyz = -         nu * rho * (        dudz[IY] + dudy[IZ]            );
	
	// save results in flux_z
	if (k >= ghostWidth and k<kmax-ghostWidth+1) {
	  offset = elemOffset;
	  flux_z[offset] = ZERO_F;    offset += arraySize; // ID
	  if (cIso <= 0) {
	    flux_z[offset] = (u*txz+v*tyz+w*tzz)*dt/dz;  offset += arraySize; // IP
	  } else {
	    flux_z[offset] = ZERO_F;                     offset += arraySize; // IP
	  }
	  flux_z[offset] = txz*dt/dz; offset += arraySize; // IU
	  flux_z[offset] = tyz*dt/dz; offset += arraySize; // IV
	  flux_z[offset] = tzz*dt/dz; offset += arraySize; // IW
	}
	
      } // end i-j guard
    __syncthreads();

    // swap planes 
    tmp = low;
    low = mid;
    mid = top;
    top = tmp;
    __syncthreads();
    
    // load k+2 plane into top (k+1)
    if (k+2<kmax) {
      if(i >= 0 and i < imax and 
	 j >= 0 and j < jmax)
	{
	  
	  // Gather conservative variables
	  int offset = i + pitch * (j + jmax * (k+2));
	  uIn[top][tx][ty][ID] = Uin[offset];                       offset += arraySize;
	  uIn[top][tx][ty][IP] = Uin[offset];                       offset += arraySize;
	  uIn[top][tx][ty][IU] = Uin[offset]/uIn[top][tx][ty][ID];  offset += arraySize;
	  uIn[top][tx][ty][IV] = Uin[offset]/uIn[top][tx][ty][ID];  offset += arraySize;
	  uIn[top][tx][ty][IW] = Uin[offset]/uIn[top][tx][ty][ID];
	  
	}
    }
    __syncthreads();

  } // end for k
      
} // kernel_viscosity_forces_3d

__global__ void kernel_viscosity_forces_3d_old(real_t* Uin, 
					       real_t* flux_x,
					       real_t* flux_y,
					       real_t* flux_z,
					       int ghostWidth,
					       int pitch, 
					       int imax, 
					       int jmax,
					       int kmax,
					       real_t dt,
					       real_t dx, 
					       real_t dy,
					       real_t dz)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, VISCOSITY_3D_DIMX_INNER) + tx;
  const int j = __mul24(by, VISCOSITY_3D_DIMY_INNER) + ty;
  
  const int arraySize  = pitch * jmax * kmax;

  // we store 3 consecutive plans of data
  __shared__ real_t   uIn[3][VISCOSITY_3D_DIMX][VISCOSITY_3D_DIMY][NVAR_3D];

  // index to address the 3 plans of data
  int low, mid, top, tmp;
  low=0;
  mid=1;
  top=2;

  real_t dudx[3], dudy[3], dudz[3];
  
  real_t &cIso = ::gParams.cIso;
  real_t &nu   = ::gParams.nu;
  const real_t two3rd = 2./3.;

  /*
   * initialize uIn with the first 3 plans 
   * we could start at ghostWidth-1, but for simplicity start at 0 !
   */
  for (int k=0, elemOffset = i + pitch * j; 
       k < 3;
       ++k, elemOffset += (pitch*jmax) ) {

    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax)
      {
	
	// Gather conservative variables
	int offset = elemOffset;
	uIn[k][tx][ty][ID] = Uin[offset];  offset += arraySize;
	uIn[k][tx][ty][IP] = Uin[offset];  offset += arraySize;
	uIn[k][tx][ty][IU] = Uin[offset];  offset += arraySize;
	uIn[k][tx][ty][IV] = Uin[offset];  offset += arraySize;
	uIn[k][tx][ty][IW] = Uin[offset];
	
      }
  } // end loading the first 3 plans
  __syncthreads();

  // loop over k starting at k=1 - compute viscosity force fluxes
  // we could for simplicity start at k=ghostWidth (see above)
  for (int k=1, elemOffset = i + pitch * (j + jmax * 1);
       k < kmax-1; 
       ++k, elemOffset += (pitch*jmax)) {

    if (i>=ghostWidth and i<imax-ghostWidth+1 and tx > 0 and tx<VISCOSITY_3D_DIMX-1 and
	j>=ghostWidth and j<jmax-ghostWidth+1 and ty > 0 and ty<VISCOSITY_3D_DIMY-1) 
      {
	real_t u,v,w;
	real_t uR, uL;
	real_t uRR, uRL, uLR, uLL;
	real_t txx,tyy,tzz,txy,txz,tyz;
	real_t rho;
	
	/*
	 * 1st direction viscous flux
	 */
	rho = HALF_F * ( uIn[mid][tx  ][ty][ID] + 
			 uIn[mid][tx-1][ty][ID] );
	
	if (cIso <=0) {
	  u  = HALF_F * ( uIn[mid][tx  ][ty][IU] / uIn[mid][tx  ][ty][ID] + 
			  uIn[mid][tx-1][ty][IU] / uIn[mid][tx-1][ty][ID] );
	  v  = HALF_F * ( uIn[mid][tx  ][ty][IV] / uIn[mid][tx  ][ty][ID] + 
			  uIn[mid][tx-1][ty][IV] / uIn[mid][tx-1][ty][ID] );
	  w  = HALF_F * ( uIn[mid][tx  ][ty][IW] / uIn[mid][tx  ][ty][ID] + 
			  uIn[mid][tx-1][ty][IW] / uIn[mid][tx-1][ty][ID] );
	}
	
	// dudx along X
	uR = uIn[mid][tx  ][ty][IU] / uIn[mid][tx  ][ty][ID];
	uL = uIn[mid][tx-1][ty][IU] / uIn[mid][tx-1][ty][ID];
	dudx[IX]=(uR-uL)/dx;
	
	// dudx along Y
	uR = uIn[mid][tx  ][ty][IV] / uIn[mid][tx  ][ty][ID];
	uL = uIn[mid][tx-1][ty][IV] / uIn[mid][tx-1][ty][ID];
	dudx[IY]=(uR-uL)/dx;
	
	// dudx along Z
	uR = uIn[mid][tx  ][ty][IW] / uIn[mid][tx  ][ty][ID];
	uL = uIn[mid][tx-1][ty][IW] / uIn[mid][tx-1][ty][ID];
	dudx[IZ]=(uR-uL)/dx;
	
	
	// dudy along X
	uRR = uIn[mid][tx  ][ty+1][IU] / uIn[mid][tx  ][ty+1][ID];
	uRL = uIn[mid][tx-1][ty+1][IU] / uIn[mid][tx-1][ty+1][ID];
	uLR = uIn[mid][tx  ][ty-1][IU] / uIn[mid][tx  ][ty-1][ID];
	uLL = uIn[mid][tx-1][ty-1][IU] / uIn[mid][tx-1][ty-1][ID];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudy[IX] = (uR-uL)/dy/4;
	
	// dudy along Y
	uRR = uIn[mid][tx  ][ty+1][IV] / uIn[mid][tx  ][ty+1][ID];
	uRL = uIn[mid][tx-1][ty+1][IV] / uIn[mid][tx-1][ty+1][ID];
	uLR = uIn[mid][tx  ][ty-1][IV] / uIn[mid][tx  ][ty-1][ID];
	uLL = uIn[mid][tx-1][ty-1][IV] / uIn[mid][tx-1][ty-1][ID];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudy[IY] = (uR-uL)/dy/4;
	
	// dudz along X
	uRR = uIn[top][tx  ][ty][IU] / uIn[top][tx  ][ty][ID];
	uRL = uIn[top][tx-1][ty][IU] / uIn[top][tx-1][ty][ID];
	uLR = uIn[low][tx  ][ty][IU] / uIn[low][tx  ][ty][ID];
	uLL = uIn[low][tx-1][ty][IU] / uIn[low][tx-1][ty][ID];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudz[IX] = (uR-uL)/dz/4;
	
	// dudz along Z
	uRR = uIn[top][tx  ][ty][IW] / uIn[top][tx  ][ty][ID];
	uRL = uIn[top][tx-1][ty][IW] / uIn[top][tx-1][ty][ID];
	uLR = uIn[low][tx  ][ty][IW] / uIn[low][tx  ][ty][ID];
	uLL = uIn[low][tx-1][ty][IW] / uIn[low][tx-1][ty][ID];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudz[IX] = (uR-uL)/dz/4;
	
	txx = -two3rd * nu * rho * (TWO_F * dudx[IX] - dudy[IY] - dudz[IZ]);
	txy = -         nu * rho * (        dudy[IX] + dudx[IY]           );
	txz = -         nu * rho * (        dudz[IX] + dudx[IZ]           );

	// save results in flux_x
	int offset = elemOffset;
	flux_x[offset] = ZERO_F;     offset += arraySize; // ID
	if (cIso <= 0) { // IP
	  flux_x[offset] = (u*txx+v*txy+w*txz)*dt/dx; offset += arraySize;
	} else {
	  flux_x[offset] = ZERO_F;                    offset += arraySize;
	}

	flux_x[offset] = txx*dt/dx;  offset += arraySize; // IU
	flux_x[offset] = txy*dt/dx;  offset += arraySize; // IV
	flux_x[offset] = txz*dt/dx;  offset += arraySize; // IW

	/*
	 * 2nd direction viscous flux
	 */
	rho = HALF_F * ( uIn[mid][tx][ty][ID] + 
			 uIn[mid][tx][ty-1][ID] );
	
	if (cIso <= 0) {
	  u = HALF_F * ( uIn[mid][tx][ty  ][IU] / uIn[mid][tx][ty  ][ID] + 
			 uIn[mid][tx][ty-1][IU] / uIn[mid][tx][ty-1][ID] );
	  v = HALF_F * ( uIn[mid][tx][ty  ][IV] / uIn[mid][tx][ty  ][ID] + 
			 uIn[mid][tx][ty-1][IV] / uIn[mid][tx][ty-1][ID] );
	  w = HALF_F * ( uIn[mid][tx][ty  ][IW] / uIn[mid][tx][ty  ][ID] +
			 uIn[mid][tx][ty-1][IW] / uIn[mid][tx][ty-1][ID] );
	}
	
	// dudy along X
	uR = uIn[mid][tx][ty  ][IU] / uIn[mid][tx][ty  ][ID];
	uL = uIn[mid][tx][ty-1][IU] / uIn[mid][tx][ty-1][ID];
	dudy[IX] = (uR-uL)/dy;
	
	// dudy along Y
	uR = uIn[mid][tx][ty  ][IV] / uIn[mid][tx][ty  ][ID];
	uL = uIn[mid][tx][ty-1][IV] / uIn[mid][tx][ty-1][ID];
	dudy[IY] = (uR-uL)/dy;
	
	// dudy along Z
	uR = uIn[mid][tx][ty  ][IW] / uIn[mid][tx][ty  ][ID];
	uL = uIn[mid][tx][ty-1][IW] / uIn[mid][tx][ty-1][ID];
	dudy[IZ] = (uR-uL)/dy;
	
	// dudx along X
	uRR = uIn[mid][tx+1][ty  ][IU] / uIn[mid][tx+1][ty  ][ID];
	uRL = uIn[mid][tx+1][ty-1][IU] / uIn[mid][tx+1][ty-1][ID];
	uLR = uIn[mid][tx-1][ty  ][IU] / uIn[mid][tx-1][ty  ][ID];
	uLL = uIn[mid][tx-1][ty-1][IU] / uIn[mid][tx-1][ty-1][ID];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudx[IX]=(uR-uL)/dx/4;
	
	// dudx along Y
	uRR = uIn[mid][tx+1][ty  ][IV] / uIn[mid][tx+1][ty  ][ID];
	uRL = uIn[mid][tx+1][ty-1][IV] / uIn[mid][tx+1][ty-1][ID];
	uLR = uIn[mid][tx-1][ty  ][IV] / uIn[mid][tx-1][ty  ][ID];
	uLL = uIn[mid][tx-1][ty-1][IV] / uIn[mid][tx-1][ty-1][ID];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudx[IY]=(uR-uL)/dx/4;
	
	// dudz along Y
	uRR = uIn[top][tx][ty  ][IV] / uIn[top][tx][ty  ][ID];
	uRL = uIn[top][tx][ty-1][IV] / uIn[top][tx][ty-1][ID];
	uLR = uIn[low][tx][ty  ][IV] / uIn[low][tx][ty  ][ID];
	uLL = uIn[low][tx][ty-1][IV] / uIn[low][tx][ty-1][ID];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudz[IY]=(uR-uL)/dz/4;
	
	// dudz along Z
	uRR = uIn[top][tx][ty  ][IW] / uIn[top][tx][ty  ][ID];
	uRL = uIn[top][tx][ty-1][IW] / uIn[top][tx][ty-1][ID];
	uLR = uIn[low][tx][ty  ][IW] / uIn[low][tx][ty  ][ID];
	uLL = uIn[low][tx][ty-1][IW] / uIn[low][tx][ty-1][ID];
	uR  = uRR+uRL; 
	uL  = uLR+uLL;
	dudz[IZ]=(uR-uL)/dz/4;
	
	tyy = -two3rd * nu * rho * (TWO_F * dudy[IY] - dudx[IX] - dudz[IZ] );
	txy = -         nu * rho * (        dudy[IX] + dudx[IY]            );
	tyz = -         nu * rho * (        dudz[IY] + dudy[IZ]            );

	// save results in flux_y
	offset = elemOffset;
	flux_y[offset] = ZERO_F;  offset += arraySize; // ID
	if (cIso <= 0) {
	  flux_y[offset] = (u*txy+v*tyy+w*tyz)*dt/dy;  offset += arraySize; // IP
	} else {
	  flux_y[offset] = ZERO_F;                     offset += arraySize; // IP
	}
	flux_y[offset] = txy*dt/dy;  offset += arraySize; // IU
	flux_y[offset] = tyy*dt/dy;  offset += arraySize; // IV
	flux_y[offset] = tyz*dt/dy;  offset += arraySize; // IW
	
	/*
	 * 3rd direction viscous flux
	 */
	rho = HALF_F * ( uIn[mid][tx][ty][ID] + 
			 uIn[low][tx][ty][ID] );
	
	if (cIso <= 0) {
	  u = HALF_F * ( uIn[mid][tx][ty][IU]/uIn[mid][tx][ty][ID] + 
			 uIn[low][tx][ty][IU]/uIn[low][tx][ty][ID] );
	  v = HALF_F * ( uIn[mid][tx][ty][IV]/uIn[mid][tx][ty][ID] + 
			 uIn[low][tx][ty][IV]/uIn[low][tx][ty][ID] );
	  w = HALF_F * ( uIn[mid][tx][ty][IW]/uIn[mid][tx][ty][ID] + 
			 uIn[low][tx][ty][IW]/uIn[low][tx][ty][ID] );
	}
	
	// dudz along X
	uR = uIn[mid][tx][ty][IU] / uIn[mid][tx][ty][ID];
	uL = uIn[low][tx][ty][IU] / uIn[low][tx][ty][ID];
	dudz[IX] = (uR-uL)/dz;
	
	// dudz along Y
	uR = uIn[mid][tx][ty][IV] / uIn[mid][tx][ty][ID];
	uL = uIn[low][tx][ty][IV] / uIn[low][tx][ty][ID];
	dudz[IY] = (uR-uL)/dz;
	
	// dudz along Z
	uR = uIn[mid][tx][ty][IW] / uIn[mid][tx][ty][ID];
	uL = uIn[low][tx][ty][IW] / uIn[low][tx][ty][ID];
	dudz[IZ] = (uR-uL)/dz;
	
	// dudx along X
	uRR = uIn[mid][tx+1][ty][IU] / uIn[mid][tx+1][ty][ID];
	uRL = uIn[low][tx+1][ty][IU] / uIn[low][tx+1][ty][ID];
	uLR = uIn[mid][tx-1][ty][IU] / uIn[mid][tx-1][ty][ID];
	uLL = uIn[low][tx-1][ty][IU] / uIn[low][tx-1][ty][ID];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudx[IX] = (uR-uL)/dx/4;
	
	// dudx along Z
	uRR = uIn[mid][tx+1][ty][IW] / uIn[mid][tx+1][ty][ID];
	uRL = uIn[low][tx+1][ty][IW] / uIn[low][tx+1][ty][ID];
	uLR = uIn[mid][tx-1][ty][IW] / uIn[mid][tx-1][ty][ID];
	uLL = uIn[low][tx-1][ty][IW] / uIn[low][tx-1][ty][ID];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudx[IZ] = (uR-uL)/dx/4;
	
	// dudy along Y
	uRR = uIn[mid][tx][ty+1][IV] / uIn[mid][tx][ty+1][ID];
	uRL = uIn[low][tx][ty+1][IV] / uIn[low][tx][ty+1][ID];
	uLR = uIn[mid][tx][ty-1][IV] / uIn[mid][tx][ty-1][ID];
	uLL = uIn[low][tx][ty-1][IV] / uIn[low][tx][ty-1][ID];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudy[IY] = (uR-uL)/dy/4;
	
	// dudy along Z
	uRR = uIn[mid][tx][ty+1][IW] / uIn[mid][tx][ty+1][ID];
	uRL = uIn[low][tx][ty+1][IW] / uIn[low][tx][ty+1][ID];
	uLR = uIn[mid][tx][ty-1][IW] / uIn[mid][tx][ty-1][ID];
	uLL = uIn[low][tx][ty-1][IW] / uIn[low][tx][ty-1][ID];
	uR  = uRR+uRL;
	uL  = uLR+uLL;
	dudy[IZ] = (uR-uL)/dy/4;
	
	
	tzz = -two3rd * nu * rho * (TWO_F * dudz[IZ] - dudx[IX] - dudy[IY] );
	txz = -         nu * rho * (        dudz[IX] + dudx[IZ]            );
	tyz = -         nu * rho * (        dudz[IY] + dudy[IZ]            );
	
	// save results in flux_z
	offset = elemOffset;
	flux_z[offset] = ZERO_F;    offset += arraySize; // ID
	if (cIso <= 0) {
	  flux_z[offset] = (u*txz+v*tyz+w*tzz)*dt/dz;  offset += arraySize; // IP
	} else {
	  flux_z[offset] = ZERO_F;                     offset += arraySize; // IP
	}
	flux_z[offset] = txz*dt/dz; offset += arraySize; // IU
	flux_z[offset] = tyz*dt/dz; offset += arraySize; // IV
	flux_z[offset] = tzz*dt/dz; offset += arraySize; // IW
	
      } // end i-j guard
    __syncthreads();

    // swap planes 
    tmp = low;
    low = mid;
    mid = top;
    top = tmp;
    __syncthreads();
    
    // load k+2 plane into top (k+1)
    if (k+2<kmax) {
      if(i >= 0 and i < imax and 
	 j >= 0 and j < jmax)
	{
	  
	  // Gather conservative variables
	  int offset = i + pitch * (j + jmax * (k+2));
	  uIn[top][tx][ty][ID] = Uin[offset];  offset += arraySize;
	  uIn[top][tx][ty][IP] = Uin[offset];  offset += arraySize;
	  uIn[top][tx][ty][IU] = Uin[offset];  offset += arraySize;
	  uIn[top][tx][ty][IV] = Uin[offset];  offset += arraySize;
	  uIn[top][tx][ty][IW] = Uin[offset];
	  
	}
    }
    __syncthreads();

  } // end for k
      
} // kernel_viscosity_forces_3d_old

#endif // VISCOSITY_CUH_
