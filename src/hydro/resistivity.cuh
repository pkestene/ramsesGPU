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
 * \file resistivity.cuh
 * \brief CUDA kernel for computing resistivity forces (MHD only, adapted from Dumses).
 *
 * \date 30 Apr 2012
 * \author P. Kestener
 *
 * $Id: resistivity.cuh 3449 2014-06-16 16:24:38Z pkestene $
 */
#ifndef RESISTIVITY_CUH_
#define RESISTIVITY_CUH_

#include "real_type.h"
#include "constants.h"

#ifdef USE_DOUBLE
#define RESISTIVITY_2D_DIMX	24
#define RESISTIVITY_2D_DIMY	16
#else // simple precision
#define RESISTIVITY_2D_DIMX	24
#define RESISTIVITY_2D_DIMY	16
#endif // USE_DOUBLE

/**
 * CUDA kernel computing resistivity forces (2D data).
 * 
 * \param[in]  Uin    conservative variables array.
 * \param[out] emf    Electromotive force due to resistivity forces.
 */
__global__ void kernel_resistivity_forces_2d(real_t* Uin, 
					     real_t* emf,
					     int ghostWidth,
					     int pitch, 
					     int imax, 
					     int jmax,
					     real_t dx, 
					     real_t dy)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, RESISTIVITY_2D_DIMX) + tx;
  const int j = __mul24(by, RESISTIVITY_2D_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  real_t dbxdy = ZERO_F;
  //real_t dbxdz = ZERO_F;

  real_t dbydx = ZERO_F;
  //real_t dbydz = ZERO_F;

  //real_t dbzdx = ZERO_F;
  //real_t dbzdy = ZERO_F;

  //real_t jx    = ZERO_F;
  //real_t jy    = ZERO_F;
  real_t jz    = ZERO_F;

  real_t &eta  = ::gParams.eta;

  if(i >= ghostWidth and i < imax-ghostWidth+1 and 
     j >= ghostWidth and j < jmax-ghostWidth+1)
    {
      
      int offset_ij = elemOffset;
      
      dbydx = ( Uin[offset_ij+IBY*arraySize] - Uin[offset_ij-1    +IBY*arraySize] ) / dx;
      //dbzdx = ( Uin[offset_ij+IBZ*arraySize] - Uin[offset_ij-1    +IBZ*arraySize] ) / dx;
      
      dbxdy = ( Uin[offset_ij+IBX*arraySize] - Uin[offset_ij-pitch+IBX*arraySize] ) / dy;
      //dbzdy = ( Uin[offset_ij+IBZ*arraySize] - Uin[offset_ij-pitch+IBZ*arraySize] ) / dy;
      
      //dbxdz = ZERO_F;
      //dbydz = ZERO_F;
      
      //jx = dbzdy - dbydz;
      //jy = dbxdz - dbzdx;
      jz = dbydx - dbxdy;
      
      // note that multiplication by dt is done in ct
      emf[offset_ij+I_EMFZ*arraySize] = -eta*jz;
      /*emf[offset_ij+I_EMFY*arraySize] = -eta*jy;
	emf[offset_ij+I_EMFX*arraySize] = -eta*jx;*/
      
    } // end i,j guard
  
} // kernel_resistivity_forces_2d

#ifdef USE_DOUBLE
#define RESISTIVITY_ENERGY_2D_DIMX	24
#define RESISTIVITY_ENERGY_2D_DIMY	16
#else // simple precision
#define RESISTIVITY_ENERGY_2D_DIMX	24
#define RESISTIVITY_ENERGY_2D_DIMY	16
#endif // USE_DOUBLE

/**
 * CUDA kernel computing resistivity forces (2D data).
 * 
 * \param[in]  Uin    conservative variables array.
 * \param[out] emf    Electromotive force due to resistivity forces.
 */
__global__ void kernel_resistivity_energy_flux_2d(real_t* Uin, 
						  real_t* flux_x,
						  real_t* flux_y,
						  int ghostWidth,
						  int pitch, 
						  int imax, 
						  int jmax,
						  real_t dx, 
						  real_t dy,
						  real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, RESISTIVITY_ENERGY_2D_DIMX) + tx;
  const int j = __mul24(by, RESISTIVITY_ENERGY_2D_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  real_t Bx,   By,   Bz;
  real_t jx,   jy,   jz;
  real_t /*jxp1, jyp1,*/ jzp1;
    
  real_t &eta  = ::gParams.eta;

  if(i >= ghostWidth and i < imax-ghostWidth+1 and 
     j >= ghostWidth and j < jmax-ghostWidth+1)
    {
      
      int offset_ij = elemOffset;
      
      // 1st direction energy flux
      
      By = ( Uin[offset_ij      +IBY*arraySize] + Uin[offset_ij-1      +IBY*arraySize] + 
	     Uin[offset_ij+pitch+IBY*arraySize] + Uin[offset_ij-1+pitch+IBY*arraySize] )/4;
      
      Bz = ( Uin[offset_ij  +IBZ*arraySize] + 
	     Uin[offset_ij-1+IBZ*arraySize] )/2;
      
      jy   = - ( Uin[offset_ij  +IBZ*arraySize] - 
		 Uin[offset_ij-1+IBZ*arraySize] )/dx;
      
      jz   = 
	( Uin[offset_ij+IBY*arraySize] - Uin[offset_ij-1    +IBY*arraySize] )/dx -
	( Uin[offset_ij+IBX*arraySize] - Uin[offset_ij-pitch+IBX*arraySize] )/dy;
      jzp1 = 
	( Uin[offset_ij+pitch+IBY*arraySize] - Uin[offset_ij-1+pitch+IBY*arraySize] )/dx -
	( Uin[offset_ij+pitch+IBX*arraySize] - Uin[offset_ij        +IBX*arraySize] )/dy;
      jz   = (jz+jzp1)/2;
      
      flux_x[offset_ij+ID*arraySize] = ZERO_F;
      flux_x[offset_ij+IP*arraySize] = - eta*(jy*Bz-jz*By)*dt/dx;
      flux_x[offset_ij+IU*arraySize] = ZERO_F;
      flux_x[offset_ij+IV*arraySize] = ZERO_F;
      
      // 2nd direction energy flux
      
      Bx = ( Uin[offset_ij  +IBX*arraySize] + Uin[offset_ij  -pitch+IBX*arraySize] +
	     Uin[offset_ij+1+IBX*arraySize] + Uin[offset_ij+1-pitch+IBX*arraySize] )/4;
      
      Bz = ( Uin[offset_ij  +IBZ*arraySize] + Uin[offset_ij  -pitch+IBZ*arraySize] )/2;
      
      jx = ( Uin[offset_ij      +IBZ*arraySize] -
	     Uin[offset_ij-pitch+IBZ*arraySize] )/dy;
      
      jz   = 
	( Uin[offset_ij+IBY*arraySize] - Uin[offset_ij-1    +IBY*arraySize] )/dx -
	( Uin[offset_ij+IBX*arraySize] - Uin[offset_ij-pitch+IBX*arraySize] )/dy;
      
      jzp1 = 
	( Uin[offset_ij+1+IBY*arraySize] - Uin[offset_ij        +IBY*arraySize] )/dx -
	( Uin[offset_ij+1+IBX*arraySize] - Uin[offset_ij+1-pitch+IBX*arraySize] )/dy;
      
      jz = (jz+jzp1)/2;
      
      flux_y[offset_ij+ID*arraySize] = ZERO_F;
      flux_y[offset_ij+IP*arraySize] = - eta*(jz*Bx-jx*Bz)*dt/dy;
      flux_y[offset_ij+IU*arraySize] = ZERO_F;
      flux_y[offset_ij+IV*arraySize] = ZERO_F;	
      
    } // end guard i,j

} // kernel_resistivity_energy_flux_2d

//
// TODO : BENCHMARK / OPTIMIZE block sizes for double precision !!!
//
#ifdef USE_DOUBLE
#define RESISTIVITY_3D_DIMX	48
#define RESISTIVITY_3D_DIMY	8
#else // simple precision
#define RESISTIVITY_3D_DIMX	48
#define RESISTIVITY_3D_DIMY	10
#endif // USE_DOUBLE

/**
 * CUDA kernel computing resistivity forces (3D data).
 * 
 * Note: don't use shared memory here, L1 cache does its job.
 *
 * \param[in]  Uin    conservative variables array.
 * \param[out] emf    Electromotive force due to resistivity forces.
 */
__global__ void kernel_resistivity_forces_3d(real_t* Uin, 
					     real_t* emf,
					     int ghostWidth,
					     int pitch, 
					     int imax, 
					     int jmax,
					     int kmax,
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
  
  const int i = __mul24(bx, RESISTIVITY_3D_DIMX) + tx;
  const int j = __mul24(by, RESISTIVITY_3D_DIMY) + ty;
  
  const int arraySize  = pitch * jmax * kmax;

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
  for (int k=ghostWidth, elemOffset = i + pitch * (j + jmax * ghostWidth); 
       k < kmax-ghostWidth+1;
       ++k, elemOffset += (pitch*jmax) ) {

    if(i >= ghostWidth and i < imax-ghostWidth+1 and 
       j >= ghostWidth and j < jmax-ghostWidth+1)
      {

	int offset_ijk = elemOffset;
	
	dbydx = ( Uin[offset_ijk  +IBY*arraySize] -
		  Uin[offset_ijk-1+IBY*arraySize] )/dx;
	dbzdx = ( Uin[offset_ijk  +IBZ*arraySize] -
		  Uin[offset_ijk-1+IBZ*arraySize] )/dx;

	dbxdy = ( Uin[offset_ijk      +IBX*arraySize] -
		  Uin[offset_ijk-pitch+IBX*arraySize] )/dy;
	dbzdy = ( Uin[offset_ijk      +IBZ*arraySize] -
		  Uin[offset_ijk-pitch+IBZ*arraySize] )/dy;
	
	dbxdz = ( Uin[offset_ijk           +IBX*arraySize] -
		  Uin[offset_ijk-pitch*jmax+IBX*arraySize] )/dz;
	dbydz = ( Uin[offset_ijk           +IBY*arraySize] -
		  Uin[offset_ijk-pitch*jmax+IBY*arraySize] )/dz;

	jx = dbzdy-dbydz;
	jy = dbxdz-dbzdx;
	jz = dbydx-dbxdy;

	emf[offset_ijk+I_EMFZ*arraySize] = -eta*jz;
	emf[offset_ijk+I_EMFY*arraySize] = -eta*jy;
	emf[offset_ijk+I_EMFX*arraySize] = -eta*jx;

      } // end i-j guard

  } // end for k

} // kernel_resistivity_forces_3d

#ifdef USE_DOUBLE
#define RESISTIVITY_ENERGY_3D_DIMX	16
#define RESISTIVITY_ENERGY_3D_DIMY	16
#else // simple precision
#define RESISTIVITY_ENERGY_3D_DIMX	48
#define RESISTIVITY_ENERGY_3D_DIMY	10
#endif // USE_DOUBLE

/**
 * CUDA kernel computing resistivity forces (3D data).
 * 
 * Note: don't use shared memory here, L1 cache does its job.
 *
 * \param[in]  Uin    conservative variables array.
 * \param[out] emf    Electromotive force due to resistivity forces.
 */
__global__ void kernel_resistivity_energy_flux_3d(real_t* Uin, 
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
						  real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, RESISTIVITY_ENERGY_3D_DIMX) + tx;
  const int j = __mul24(by, RESISTIVITY_ENERGY_3D_DIMY) + ty;
  
  const int arraySize  = pitch * jmax * kmax;

  real_t Bx,   By,   Bz;
  real_t jx,   jy,   jz;
  real_t jxp1, jyp1, jzp1;
    
  int Dx = 1;
  int Dy = pitch;
  int Dz = pitch*jmax;

  real_t &eta  = ::gParams.eta;

  for (int k=ghostWidth, elemOffset = i + pitch * (j + jmax * ghostWidth); 
       k < kmax-ghostWidth+1;
       ++k, elemOffset += (pitch*jmax) ) {

    if(i >= ghostWidth and i < imax-ghostWidth+1 and 
       j >= ghostWidth and j < jmax-ghostWidth+1)
      {

	int offset_ijk = elemOffset;

	// 1st direction energy flux
	
	By = ( Uin[offset_ijk   +IBY*arraySize] + Uin[offset_ijk-Dx   +IBY*arraySize] + 
	       Uin[offset_ijk+Dy+IBY*arraySize] + Uin[offset_ijk-Dx+Dy+IBY*arraySize] )/4;
	Bz = ( Uin[offset_ijk   +IBZ*arraySize] + Uin[offset_ijk-Dx   +IBZ*arraySize] + 
	       Uin[offset_ijk+Dz+IBZ*arraySize] + Uin[offset_ijk-Dx+Dz+IBZ*arraySize] )/4;
	
	jy   = 
	  ( Uin[offset_ijk   +IBX*arraySize] - Uin[offset_ijk-Dz+IBX*arraySize] )/dz -
	  ( Uin[offset_ijk   +IBZ*arraySize] - Uin[offset_ijk-Dx+IBZ*arraySize] )/dx;
	jyp1 = 
	  ( Uin[offset_ijk+Dz+IBX*arraySize] - Uin[offset_ijk      +IBX*arraySize] )/dz -
	  ( Uin[offset_ijk+Dz+IBZ*arraySize] - Uin[offset_ijk-Dx+Dz+IBZ*arraySize] )/dx;
	jy   = (jy+jyp1)/2;
	
	jz   = 
	  ( Uin[offset_ijk   +IBY*arraySize] - Uin[offset_ijk-Dx+IBY*arraySize] )/dx -
	  ( Uin[offset_ijk   +IBX*arraySize] - Uin[offset_ijk-Dy+IBX*arraySize] )/dy;
	jzp1 = 
	  ( Uin[offset_ijk+Dy+IBY*arraySize] - Uin[offset_ijk-Dx+Dy+IBY*arraySize] )/dx -
	  ( Uin[offset_ijk+Dy+IBX*arraySize] - Uin[offset_ijk      +IBX*arraySize] )/dy;
	jz   = (jz+jzp1)/2;

	flux_x[offset_ijk+ID*arraySize] = ZERO_F;
	flux_x[offset_ijk+IP*arraySize] = - eta*(jy*Bz-jz*By)*dt/dx;
	flux_x[offset_ijk+IU*arraySize] = ZERO_F;
	flux_x[offset_ijk+IV*arraySize] = ZERO_F;
	flux_x[offset_ijk+IW*arraySize] = ZERO_F;
	
	// 2nd direction energy flux
	
	Bx = ( Uin[offset_ijk   +IBX*arraySize] + Uin[offset_ijk   -Dy+IBX*arraySize] +
	       Uin[offset_ijk+Dx+IBX*arraySize] + Uin[offset_ijk+Dx-Dy+IBX*arraySize] )/4;
	
	Bz = ( Uin[offset_ijk   +IBZ*arraySize] + Uin[offset_ijk-Dy   +IBZ*arraySize] + 
	       Uin[offset_ijk+Dz+IBZ*arraySize] + Uin[offset_ijk-Dy+Dz+IBZ*arraySize] )/4;
	
	jx   = 
	  ( Uin[offset_ijk+IBZ*arraySize] - Uin[offset_ijk-Dy   +IBZ*arraySize] )/dy -
	  ( Uin[offset_ijk+IBY*arraySize] - Uin[offset_ijk   -Dz+IBY*arraySize] )/dz;
	jxp1 = 
	  ( Uin[offset_ijk+Dz+IBZ*arraySize] - Uin[offset_ijk-Dy+Dz+IBZ*arraySize] )/dy -
	  ( Uin[offset_ijk+Dz+IBY*arraySize] - Uin[offset_ijk      +IBY*arraySize] )/dz;
	jx = (jx+jxp1)/2;
	
	
	jz   = 
	  ( Uin[offset_ijk+IBY*arraySize] - Uin[offset_ijk-Dx   +IBY*arraySize] )/dx -
	  ( Uin[offset_ijk+IBX*arraySize] - Uin[offset_ijk   -Dy+IBX*arraySize] )/dy;
	jzp1 = 
	  ( Uin[offset_ijk+Dx+IBY*arraySize] - Uin[offset_ijk      +IBY*arraySize] )/dx -
	  ( Uin[offset_ijk+Dx+IBX*arraySize] - Uin[offset_ijk+Dx-Dy+IBX*arraySize] )/dy;
	jz = (jz+jzp1)/2;
	
	flux_y[offset_ijk+ID*arraySize] = ZERO_F;
	flux_y[offset_ijk+IP*arraySize] = - eta*(jz*Bx-jx*Bz)*dt/dy;
	flux_y[offset_ijk+IU*arraySize] = ZERO_F;
	flux_y[offset_ijk+IV*arraySize] = ZERO_F;
	flux_y[offset_ijk+IW*arraySize] = ZERO_F;
	
	// 3rd direction energy flux
	Bx = ( Uin[offset_ijk   +IBX*arraySize] + Uin[offset_ijk   -Dz+IBX*arraySize] + 
	       Uin[offset_ijk+Dx+IBX*arraySize] + Uin[offset_ijk+Dx-Dz+IBX*arraySize] )/4;
	By = ( Uin[offset_ijk   +IBY*arraySize] + Uin[offset_ijk   -Dz+IBY*arraySize] + 
	       Uin[offset_ijk+Dy+IBY*arraySize] + Uin[offset_ijk+Dy-Dz+IBY*arraySize] )/4;
	
	jx   = 
	  ( Uin[offset_ijk+IBZ*arraySize] - Uin[offset_ijk-Dy   +IBZ*arraySize] )/dy -
	  ( Uin[offset_ijk+IBY*arraySize] - Uin[offset_ijk   -Dz+IBY*arraySize] )/dz;
	jxp1 = 
	  ( Uin[offset_ijk+Dy+IBZ*arraySize] - Uin[offset_ijk      +IBZ*arraySize] )/dy -
	  ( Uin[offset_ijk+Dy+IBY*arraySize] - Uin[offset_ijk+Dy-Dz+IBY*arraySize] )/dz;
	jx   = (jx+jxp1)/2;
	
	jy   = 
	  ( Uin[offset_ijk+IBX*arraySize] - Uin[offset_ijk   -Dz+IBX*arraySize] )/dz -
	  ( Uin[offset_ijk+IBZ*arraySize] - Uin[offset_ijk-Dx   +IBZ*arraySize] )/dx;
	jyp1 = 
	  ( Uin[offset_ijk+Dx+IBX*arraySize] - Uin[offset_ijk+Dx-Dz+IBX*arraySize] )/dz -
	  ( Uin[offset_ijk+Dx+IBZ*arraySize] - Uin[offset_ijk      +IBZ*arraySize] )/dx;
	jy   = (jy+jyp1)/2;
        
	flux_z[offset_ijk+ID*arraySize] = ZERO_F;
	flux_z[offset_ijk+IP*arraySize] = - eta*(jx*By-jy*Bx)*dt/dz;
	flux_z[offset_ijk+IU*arraySize] = ZERO_F;
	flux_z[offset_ijk+IV*arraySize] = ZERO_F;
	flux_z[offset_ijk+IW*arraySize] = ZERO_F;
	
      } // end i,j guard

  } // end for k

} // kernel_resistivity_energy_flux_3d 

#endif // RESISTIVITY_CUH_
