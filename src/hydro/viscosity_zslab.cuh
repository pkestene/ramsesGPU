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
 * \file viscosity_zslab.cuh
 * \brief CUDA kernel for computing viscosity forces inside z-slab.
 *
 * \date 14 Sept 2012
 * \author P. Kestener
 *
 * $Id: viscosity_zslab.cuh 3449 2014-06-16 16:24:38Z pkestene $
 */
#ifndef VISCOSITY_ZSLAB_CUH_
#define VISCOSITY_ZSLAB_CUH_

#include "real_type.h"
#include "constants.h"

//
// TODO : BENCHMARK / OPTIMIZE block sizes for double precision !!!
//
#ifdef USE_DOUBLE
#define VISCOSITY_Z_3D_DIMX	48
#define VISCOSITY_Z_3D_DIMY	8
#define VISCOSITY_Z_3D_DIMX_INNER	(VISCOSITY_Z_3D_DIMX-2)
#define VISCOSITY_Z_3D_DIMY_INNER	(VISCOSITY_Z_3D_DIMY-2)
#else // simple precision
#define VISCOSITY_Z_3D_DIMX	48
#define VISCOSITY_Z_3D_DIMY	10
#define VISCOSITY_Z_3D_DIMX_INNER	(VISCOSITY_Z_3D_DIMX-2)
#define VISCOSITY_Z_3D_DIMY_INNER	(VISCOSITY_Z_3D_DIMY-2)
#endif // USE_DOUBLE

/**
 * CUDA kernel computing viscosity forces (3D data) inside z-slab.
 * 
 * \param[in]  Uin    conservative variables array.
 * \param[out] flux_x viscosity forces flux along X.
 * \param[out] flux_y viscosity forces flux along Y.
 * \param[out] flux_z viscosity forces flux along Z.
 * \param[in]  dt     time step
 */
__global__ void kernel_viscosity_forces_3d_zslab(real_t* Uin, 
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
						 real_t dz,
						 ZslabInfo zSlabInfo)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, VISCOSITY_Z_3D_DIMX_INNER) + tx;
  const int j = __mul24(by, VISCOSITY_Z_3D_DIMY_INNER) + ty;
  
  const int arraySizeU = pitch * jmax * kmax;
  const int arraySizeQ = pitch * jmax * zSlabInfo.zSlabWidthG;

  const int &kStart    = zSlabInfo.kStart; 
  //const int &kStop     = zSlabInfo.kStop; 
  const int &ksizeSlab = zSlabInfo.ksizeSlab;

  // we store 3 consecutive plans of data
  __shared__ real_t   uIn[3][VISCOSITY_Z_3D_DIMX][VISCOSITY_Z_3D_DIMY][NVAR_3D];

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
  for (int k=kStart, elemOffsetU = i + pitch * j + pitch*jmax*kStart; 
       k < kStart+3;
       ++k, elemOffsetU += (pitch*jmax) ) {
    
    if(i >= 0 and i < imax and 
       j >= 0 and j < jmax)
      {
	
	int kL = k-kStart;

	// Gather conservative variables
	int offsetU = elemOffsetU;
	uIn[kL][tx][ty][ID] = Uin[offsetU];                      offsetU += arraySizeU;
	uIn[kL][tx][ty][IP] = Uin[offsetU];                      offsetU += arraySizeU;
	uIn[kL][tx][ty][IU] = Uin[offsetU]/uIn[kL][tx][ty][ID];  offsetU += arraySizeU;
	uIn[kL][tx][ty][IV] = Uin[offsetU]/uIn[kL][tx][ty][ID];  offsetU += arraySizeU;
	uIn[kL][tx][ty][IW] = Uin[offsetU]/uIn[kL][tx][ty][ID];
	
      }
  } // end loading the first 3 plans
  __syncthreads();

  // loop over k starting at k=1 - compute viscosity force fluxes
  // we could for simplicity start at k=ghostWidth (see above)
  // kL : k local inside slab
  for (int kL=1, elemOffsetQ = i + pitch * (j + jmax * 1);
       kL < ksizeSlab-1; 
       ++kL, elemOffsetQ += (pitch*jmax)) {

    if (i>=ghostWidth and i<imax-ghostWidth+1 and tx > 0 and tx<VISCOSITY_Z_3D_DIMX-1 and
	j>=ghostWidth and j<jmax-ghostWidth+1 and ty > 0 and ty<VISCOSITY_Z_3D_DIMY-1) 
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
	int offsetQ = elemOffsetQ;
	if (kL >= ghostWidth and kL<ksizeSlab-ghostWidth+1) {
	  offsetQ = elemOffsetQ;
	  flux_x[offsetQ] = ZERO_F;     offsetQ += arraySizeQ; // ID
	  if (cIso <= 0) { // IP
	    flux_x[offsetQ] = (u*txx+v*txy+w*txz)*dt/dx; offsetQ += arraySizeQ;
	  } else {
	    flux_x[offsetQ] = ZERO_F;                    offsetQ += arraySizeQ;
	  }
	  
	  flux_x[offsetQ] = txx*dt/dx;  offsetQ += arraySizeQ; // IU
	  flux_x[offsetQ] = txy*dt/dx;  offsetQ += arraySizeQ; // IV
	  flux_x[offsetQ] = txz*dt/dx;  offsetQ += arraySizeQ; // IW
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
	if (kL >= ghostWidth and kL<ksizeSlab-ghostWidth+1) {
	  offsetQ = elemOffsetQ;
	  flux_y[offsetQ] = ZERO_F;  offsetQ += arraySizeQ; // ID
	  if (cIso <= 0) {
	    flux_y[offsetQ] = (u*txy+v*tyy+w*tyz)*dt/dy;  offsetQ += arraySizeQ; // IP
	  } else {
	    flux_y[offsetQ] = ZERO_F;                     offsetQ += arraySizeQ; // IP
	  }
	  flux_y[offsetQ] = txy*dt/dy;  offsetQ += arraySizeQ; // IU
	  flux_y[offsetQ] = tyy*dt/dy;  offsetQ += arraySizeQ; // IV
	  flux_y[offsetQ] = tyz*dt/dy;  offsetQ += arraySizeQ; // IW
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
	if (kL >= ghostWidth and kL<ksizeSlab-ghostWidth+1) {
	  offsetQ = elemOffsetQ;
	  flux_z[offsetQ] = ZERO_F;    offsetQ += arraySizeQ; // ID
	  if (cIso <= 0) {
	    flux_z[offsetQ] = (u*txz+v*tyz+w*tzz)*dt/dz;  offsetQ += arraySizeQ; // IP
	  } else {
	    flux_z[offsetQ] = ZERO_F;                     offsetQ += arraySizeQ; // IP
	  }
	  flux_z[offsetQ] = txz*dt/dz; offsetQ += arraySizeQ; // IU
	  flux_z[offsetQ] = tyz*dt/dz; offsetQ += arraySizeQ; // IV
	  flux_z[offsetQ] = tzz*dt/dz; offsetQ += arraySizeQ; // IW

	}
	
      } // end i-j guard
    __syncthreads();

    // swap planes 
    tmp = low;
    low = mid;
    mid = top;
    top = tmp;
    __syncthreads();
    
    // load kL+2 plane into top (kL+1)
    if (kL+2<ksizeSlab) {

      if(i >= 0 and i < imax and 
	 j >= 0 and j < jmax)
	{

	  int k = kL + kStart;
	  
	  // Gather conservative variables
	  int offsetU = i + pitch * (j + jmax * (k+2));
	  uIn[top][tx][ty][ID] = Uin[offsetU];                       offsetU += arraySizeU;
	  uIn[top][tx][ty][IP] = Uin[offsetU];                       offsetU += arraySizeU;
	  uIn[top][tx][ty][IU] = Uin[offsetU]/uIn[top][tx][ty][ID];  offsetU += arraySizeU;
	  uIn[top][tx][ty][IV] = Uin[offsetU]/uIn[top][tx][ty][ID];  offsetU += arraySizeU;
	  uIn[top][tx][ty][IW] = Uin[offsetU]/uIn[top][tx][ty][ID];
	  
	}
    }
    __syncthreads();

  } // end for kL
      
} // kernel_viscosity_forces_3d_zslab

#endif // VISCOSITY_ZSLAB_CUH_
