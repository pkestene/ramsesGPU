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
 * \file resistivity_zslab.cuh
 * \brief CUDA kernel for computing resistivity forces (MHD only, adapted from Dumses) using z-slab method.
 *
 * \date Sept 18, 2012
 * \author P. Kestener
 *
 * $Id: resistivity_zslab.cuh 3449 2014-06-16 16:24:38Z pkestene $
 */
#ifndef RESISTIVITY_ZSLAB_CUH_
#define RESISTIVITY_ZSLAB_CUH_

#include "real_type.h"
#include "constants.h"

//
// TODO : BENCHMARK / OPTIMIZE block sizes for double precision !!!
//
#ifdef USE_DOUBLE
#define RESISTIVITY_Z_3D_DIMX	48
#define RESISTIVITY_Z_3D_DIMY	8
#else // simple precision
#define RESISTIVITY_Z_3D_DIMX	48
#define RESISTIVITY_Z_3D_DIMY	10
#endif // USE_DOUBLE

/**
 * CUDA kernel computing resistivity forces (3D data).
 * 
 * Note: don't use shared memory here, L1 cache does its job.
 *
 * \param[in]  Uin    conservative variables array.
 * \param[out] emf    Electromotive force due to resistivity forces.
 */
__global__ void kernel_resistivity_forces_3d_zslab(real_t* Uin, 
						   real_t* emf,
						   int ghostWidth,
						   int pitch, 
						   int imax, 
						   int jmax,
						   int kmax,
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
  
  const int i = __mul24(bx, RESISTIVITY_Z_3D_DIMX) + tx;
  const int j = __mul24(by, RESISTIVITY_Z_3D_DIMY) + ty;
  
  const int arraySizeU = pitch * jmax * kmax;
  const int arraySizeQ = pitch * jmax * zSlabInfo.zSlabWidthG;

  const int &kStart    = zSlabInfo.kStart; 
  //const int &kStop     = zSlabInfo.kStop; 
  const int &ksizeSlab = zSlabInfo.ksizeSlab;

  real_t dbxdy = ZERO_F;
  real_t dbxdz = ZERO_F;

  real_t dbydx = ZERO_F;
  real_t dbydz = ZERO_F;

  real_t dbzdx = ZERO_F;
  real_t dbzdy = ZERO_F;

  real_t jx    = ZERO_F;
  real_t jy    = ZERO_F;
  real_t jz    = ZERO_F;
 
  real_t &eta  = ::gParams.eta;

  /*
   * Compute J=curl(B)
   */
  for (int kL = ghostWidth, elemOffsetQ = i + pitch*j + pitch*jmax*ghostWidth;
       kL < ksizeSlab-ghostWidth+1;
       ++kL, elemOffsetQ += (pitch*jmax) ) {

    if(i >= ghostWidth and i < imax-ghostWidth+1 and 
       j >= ghostWidth and j < jmax-ghostWidth+1)
      {

	int offset_ijkU = elemOffsetQ + kStart*jmax*pitch;
	int offset_ijkQ = elemOffsetQ;

	dbydx = ( Uin[offset_ijkU  +IBY*arraySizeU] -
		  Uin[offset_ijkU-1+IBY*arraySizeU] )/dx;
	dbzdx = ( Uin[offset_ijkU  +IBZ*arraySizeU] -
		  Uin[offset_ijkU-1+IBZ*arraySizeU] )/dx;

	dbxdy = ( Uin[offset_ijkU      +IBX*arraySizeU] -
		  Uin[offset_ijkU-pitch+IBX*arraySizeU] )/dy;
	dbzdy = ( Uin[offset_ijkU      +IBZ*arraySizeU] -
		  Uin[offset_ijkU-pitch+IBZ*arraySizeU] )/dy;
	
	dbxdz = ( Uin[offset_ijkU           +IBX*arraySizeU] -
		  Uin[offset_ijkU-pitch*jmax+IBX*arraySizeU] )/dz;
	dbydz = ( Uin[offset_ijkU           +IBY*arraySizeU] -
		  Uin[offset_ijkU-pitch*jmax+IBY*arraySizeU] )/dz;

	jx = dbzdy-dbydz;
	jy = dbxdz-dbzdx;
	jz = dbydx-dbxdy;

	emf[offset_ijkQ + I_EMFZ*arraySizeQ] = -eta*jz;
	emf[offset_ijkQ + I_EMFY*arraySizeQ] = -eta*jy;
	emf[offset_ijkQ + I_EMFX*arraySizeQ] = -eta*jx;

      } // end i-j guard
    
  } // end for k

} // kernel_resistivity_forces_3d_zslab

#ifdef USE_DOUBLE
#define RESISTIVITY_ENERGY_Z_3D_DIMX	16
#define RESISTIVITY_ENERGY_Z_3D_DIMY	16
#else // simple precision
#define RESISTIVITY_ENERGY_Z_3D_DIMX	48
#define RESISTIVITY_ENERGY_Z_3D_DIMY	10
#endif // USE_DOUBLE

/**
 * CUDA kernel computing resistivity forces (3D data, z-slab).
 * 
 * Note: don't use shared memory here, L1 cache does its job.
 *
 * \param[in]  Uin    conservative variables array.
 * \param[out] emf    Electromotive force due to resistivity forces.
 */
__global__ void kernel_resistivity_energy_flux_3d_zslab(real_t* Uin, 
							real_t* flux_x,
							real_t* flux_y,
							real_t* flux_z,
							int ghostWidth,
							int pitch, 
							int imax, 
							int jmax,
							int kmax,
							real_t dx, 
							real_t dy,
							real_t dz,
							real_t dt,
							ZslabInfo zSlabInfo)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, RESISTIVITY_ENERGY_Z_3D_DIMX) + tx;
  const int j = __mul24(by, RESISTIVITY_ENERGY_Z_3D_DIMY) + ty;
  
  const int arraySizeU  = pitch * jmax * kmax;
  const int arraySizeQ = pitch * jmax * zSlabInfo.zSlabWidthG;

  const int &kStart    = zSlabInfo.kStart; 
  //const int &kStop     = zSlabInfo.kStop; 
  const int &ksizeSlab = zSlabInfo.ksizeSlab;

  real_t Bx,   By,   Bz;
  real_t jx,   jy,   jz;
  real_t jxp1, jyp1, jzp1;
    
  int Dx = 1;
  int Dy = pitch;
  int Dz = pitch*jmax;

  real_t &eta  = ::gParams.eta;

  for (int kL=ghostWidth, elemOffsetQ = i + pitch * (j + jmax * ghostWidth); 
       kL < ksizeSlab-ghostWidth+1;
       ++kL, elemOffsetQ += (pitch*jmax) ) {

    if(i >= ghostWidth and i < imax-ghostWidth+1 and 
       j >= ghostWidth and j < jmax-ghostWidth+1)
      {

	int offset_ijkU = elemOffsetQ + kStart*jmax*pitch;
	int offset_ijkQ = elemOffsetQ;

	// 1st direction energy flux
	
	By = ( Uin[offset_ijkU   +IBY*arraySizeU] + Uin[offset_ijkU-Dx   +IBY*arraySizeU] + 
	       Uin[offset_ijkU+Dy+IBY*arraySizeU] + Uin[offset_ijkU-Dx+Dy+IBY*arraySizeU] )/4;
	Bz = ( Uin[offset_ijkU   +IBZ*arraySizeU] + Uin[offset_ijkU-Dx   +IBZ*arraySizeU] + 
	       Uin[offset_ijkU+Dz+IBZ*arraySizeU] + Uin[offset_ijkU-Dx+Dz+IBZ*arraySizeU] )/4;
	
	jy   = 
	  ( Uin[offset_ijkU   +IBX*arraySizeU] - Uin[offset_ijkU-Dz+IBX*arraySizeU] )/dz -
	  ( Uin[offset_ijkU   +IBZ*arraySizeU] - Uin[offset_ijkU-Dx+IBZ*arraySizeU] )/dx;
	jyp1 = 
	  ( Uin[offset_ijkU+Dz+IBX*arraySizeU] - Uin[offset_ijkU      +IBX*arraySizeU] )/dz -
	  ( Uin[offset_ijkU+Dz+IBZ*arraySizeU] - Uin[offset_ijkU-Dx+Dz+IBZ*arraySizeU] )/dx;
	jy   = (jy+jyp1)/2;
	
	jz   = 
	  ( Uin[offset_ijkU   +IBY*arraySizeU] - Uin[offset_ijkU-Dx+IBY*arraySizeU] )/dx -
	  ( Uin[offset_ijkU   +IBX*arraySizeU] - Uin[offset_ijkU-Dy+IBX*arraySizeU] )/dy;
	jzp1 = 
	  ( Uin[offset_ijkU+Dy+IBY*arraySizeU] - Uin[offset_ijkU-Dx+Dy+IBY*arraySizeU] )/dx -
	  ( Uin[offset_ijkU+Dy+IBX*arraySizeU] - Uin[offset_ijkU      +IBX*arraySizeU] )/dy;
	jz   = (jz+jzp1)/2;

	flux_x[offset_ijkQ + ID*arraySizeQ] = ZERO_F;
	flux_x[offset_ijkQ + IP*arraySizeQ] = - eta*(jy*Bz-jz*By)*dt/dx;
	flux_x[offset_ijkQ + IU*arraySizeQ] = ZERO_F;
	flux_x[offset_ijkQ + IV*arraySizeQ] = ZERO_F;
	flux_x[offset_ijkQ + IW*arraySizeQ] = ZERO_F;
	
	// 2nd direction energy flux
	
	Bx = ( Uin[offset_ijkU   +IBX*arraySizeU] + Uin[offset_ijkU   -Dy+IBX*arraySizeU] +
	       Uin[offset_ijkU+Dx+IBX*arraySizeU] + Uin[offset_ijkU+Dx-Dy+IBX*arraySizeU] )/4;
	
	Bz = ( Uin[offset_ijkU   +IBZ*arraySizeU] + Uin[offset_ijkU-Dy   +IBZ*arraySizeU] + 
	       Uin[offset_ijkU+Dz+IBZ*arraySizeU] + Uin[offset_ijkU-Dy+Dz+IBZ*arraySizeU] )/4;
	
	jx   = 
	  ( Uin[offset_ijkU+IBZ*arraySizeU] - Uin[offset_ijkU-Dy   +IBZ*arraySizeU] )/dy -
	  ( Uin[offset_ijkU+IBY*arraySizeU] - Uin[offset_ijkU   -Dz+IBY*arraySizeU] )/dz;
	jxp1 = 
	  ( Uin[offset_ijkU+Dz+IBZ*arraySizeU] - Uin[offset_ijkU-Dy+Dz+IBZ*arraySizeU] )/dy -
	  ( Uin[offset_ijkU+Dz+IBY*arraySizeU] - Uin[offset_ijkU      +IBY*arraySizeU] )/dz;
	jx = (jx+jxp1)/2;
	
	
	jz   = 
	  ( Uin[offset_ijkU+IBY*arraySizeU] - Uin[offset_ijkU-Dx   +IBY*arraySizeU] )/dx -
	  ( Uin[offset_ijkU+IBX*arraySizeU] - Uin[offset_ijkU   -Dy+IBX*arraySizeU] )/dy;
	jzp1 = 
	  ( Uin[offset_ijkU+Dx+IBY*arraySizeU] - Uin[offset_ijkU      +IBY*arraySizeU] )/dx -
	  ( Uin[offset_ijkU+Dx+IBX*arraySizeU] - Uin[offset_ijkU+Dx-Dy+IBX*arraySizeU] )/dy;
	jz = (jz+jzp1)/2;
	
	flux_y[offset_ijkQ + ID*arraySizeQ] = ZERO_F;
	flux_y[offset_ijkQ + IP*arraySizeQ] = - eta*(jz*Bx-jx*Bz)*dt/dy;
	flux_y[offset_ijkQ + IU*arraySizeQ] = ZERO_F;
	flux_y[offset_ijkQ + IV*arraySizeQ] = ZERO_F;
	flux_y[offset_ijkQ + IW*arraySizeQ] = ZERO_F;
	
	// 3rd direction energy flux
	Bx = ( Uin[offset_ijkU   +IBX*arraySizeU] + Uin[offset_ijkU   -Dz+IBX*arraySizeU] + 
	       Uin[offset_ijkU+Dx+IBX*arraySizeU] + Uin[offset_ijkU+Dx-Dz+IBX*arraySizeU] )/4;
	By = ( Uin[offset_ijkU   +IBY*arraySizeU] + Uin[offset_ijkU   -Dz+IBY*arraySizeU] + 
	       Uin[offset_ijkU+Dy+IBY*arraySizeU] + Uin[offset_ijkU+Dy-Dz+IBY*arraySizeU] )/4;
	
	jx   = 
	  ( Uin[offset_ijkU+IBZ*arraySizeU] - Uin[offset_ijkU-Dy   +IBZ*arraySizeU] )/dy -
	  ( Uin[offset_ijkU+IBY*arraySizeU] - Uin[offset_ijkU   -Dz+IBY*arraySizeU] )/dz;
	jxp1 = 
	  ( Uin[offset_ijkU+Dy+IBZ*arraySizeU] - Uin[offset_ijkU      +IBZ*arraySizeU] )/dy -
	  ( Uin[offset_ijkU+Dy+IBY*arraySizeU] - Uin[offset_ijkU+Dy-Dz+IBY*arraySizeU] )/dz;
	jx   = (jx+jxp1)/2;
	
	jy   = 
	  ( Uin[offset_ijkU+IBX*arraySizeU] - Uin[offset_ijkU   -Dz+IBX*arraySizeU] )/dz -
	  ( Uin[offset_ijkU+IBZ*arraySizeU] - Uin[offset_ijkU-Dx   +IBZ*arraySizeU] )/dx;
	jyp1 = 
	  ( Uin[offset_ijkU+Dx+IBX*arraySizeU] - Uin[offset_ijkU+Dx-Dz+IBX*arraySizeU] )/dz -
	  ( Uin[offset_ijkU+Dx+IBZ*arraySizeU] - Uin[offset_ijkU      +IBZ*arraySizeU] )/dx;
	jy   = (jy+jyp1)/2;
        
	flux_z[offset_ijkQ + ID*arraySizeQ] = ZERO_F;
	flux_z[offset_ijkQ + IP*arraySizeQ] = - eta*(jx*By-jy*Bx)*dt/dz;
	flux_z[offset_ijkQ + IU*arraySizeQ] = ZERO_F;
	flux_z[offset_ijkQ + IV*arraySizeQ] = ZERO_F;
	flux_z[offset_ijkQ + IW*arraySizeQ] = ZERO_F;
	
      } // end i,j guard

  } // end for k

} // kernel_resistivity_energy_flux_3d_zslab

#endif // RESISTIVITY_ZSLAB_CUH_
