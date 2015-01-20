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
 * \file hydro_update.cuh
 * \brief CUDA kernel for update conservative variables with flux array.
 *
 * \date 29 Apr 2012
 * \author P. Kestener
 *
 * $Id: hydro_update.cuh 2110 2012-05-23 12:32:02Z pkestene $
 */
#ifndef HYDRO_UPDATE_CUH_
#define HYDRO_UPDATE_CUH_

#include "real_type.h"
#include "constants.h"


#ifdef USE_DOUBLE
#define HYDRO_UPDATE_2D_DIMX	24
#define HYDRO_UPDATE_2D_DIMY	16
#else // simple precision
#define HYDRO_UPDATE_2D_DIMX	24
#define HYDRO_UPDATE_2D_DIMY	16
#endif // USE_DOUBLE

/**
 * CUDA kernel perform hydro update from flux arrays (2D data).
 * 
 * \param[inout] Uin    conservative variables array.
 * \param[in]    flux_x flux along X.
 * \param[in]    flux_y flux along Y.
 *
 */
__global__ void kernel_hydro_update_2d(real_t* Uin, 
				       real_t* flux_x,
				       real_t* flux_y,
				       int ghostWidth,
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
  
  const int i = __mul24(bx, HYDRO_UPDATE_2D_DIMX) + tx;
  const int j = __mul24(by, HYDRO_UPDATE_2D_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  // conservative variables
  real_t uIn[NVAR_2D];

  // flux
  real_t flux[NVAR_2D], fluxNext[NVAR_2D];

  if (i>=ghostWidth and i<imax-ghostWidth and
      j>=ghostWidth and j<jmax-ghostWidth)
    {

      int offset = elemOffset;
      uIn[ID] = Uin[offset];  offset += arraySize;
      uIn[IP] = Uin[offset];  offset += arraySize;
      uIn[IU] = Uin[offset];  offset += arraySize;
      uIn[IV] = Uin[offset];

      // read flux_x
      offset = elemOffset;
      flux[ID] = flux_x[offset];  fluxNext[ID] = flux_x[offset+1]; offset += arraySize;
      flux[IP] = flux_x[offset];  fluxNext[IP] = flux_x[offset+1]; offset += arraySize;
      flux[IU] = flux_x[offset];  fluxNext[IU] = flux_x[offset+1]; offset += arraySize;
      flux[IV] = flux_x[offset];  fluxNext[IV] = flux_x[offset+1];

      // update with flux_x
      uIn[ID] += (flux[ID] - fluxNext[ID]);
      uIn[IP] += (flux[IP] - fluxNext[IP]);
      uIn[IU] += (flux[IU] - fluxNext[IU]);
      uIn[IV] += (flux[IV] - fluxNext[IV]);

      // read flux_y
      offset = elemOffset;
      flux[ID] = flux_y[offset];  fluxNext[ID] = flux_y[offset+pitch]; offset += arraySize;
      flux[IP] = flux_y[offset];  fluxNext[IP] = flux_y[offset+pitch]; offset += arraySize;
      flux[IU] = flux_y[offset];  fluxNext[IU] = flux_y[offset+pitch]; offset += arraySize;
      flux[IV] = flux_y[offset];  fluxNext[IV] = flux_y[offset+pitch];

      // update with flux_y
      uIn[ID] += (flux[ID] - fluxNext[ID]);
      uIn[IP] += (flux[IP] - fluxNext[IP]);
      uIn[IU] += (flux[IU] - fluxNext[IU]);
      uIn[IV] += (flux[IV] - fluxNext[IV]);

      // write back
      offset = elemOffset;
      Uin[offset] = uIn[ID];  offset += arraySize;
      Uin[offset] = uIn[IP];  offset += arraySize;
      Uin[offset] = uIn[IU];  offset += arraySize;
      Uin[offset] = uIn[IV];

    }

} // kernel_hydro_update_2d

/**
 * CUDA kernel perform hydro update (energy only) from flux arrays (2D data).
 * 
 * \param[inout] Uin    conservative variables array.
 * \param[in]    flux_x flux along X.
 * \param[in]    flux_y flux along Y.
 *
 */
__global__ void kernel_hydro_update_energy_2d(real_t* Uin, 
					      real_t* flux_x,
					      real_t* flux_y,
					      int ghostWidth,
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
  
  const int i = __mul24(bx, HYDRO_UPDATE_2D_DIMX) + tx;
  const int j = __mul24(by, HYDRO_UPDATE_2D_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j   ) + i;

  // conservative variables
  real_t uIn_IP;

  // flux
  real_t flux_IP, fluxNext_IP;

  if (i>=ghostWidth and i<imax-ghostWidth and
      j>=ghostWidth and j<jmax-ghostWidth)
    {

      int offset = elemOffset+arraySize;
      uIn_IP = Uin[offset];

      // read flux_x - energy
      offset = elemOffset+arraySize;
      flux_IP     = flux_x[offset];  
      fluxNext_IP = flux_x[offset+1];

      // update with flux_x
      uIn_IP += (flux_IP - fluxNext_IP);

      // read flux_y
      offset = elemOffset+arraySize;
      flux_IP     = flux_y[offset];
      fluxNext_IP = flux_y[offset+pitch];

      // update with flux_y
      uIn_IP += (flux_IP - fluxNext_IP);

      // write back - energy only
      offset = elemOffset+arraySize;
      Uin[offset] = uIn_IP;

    }

} // kernel_hydro_update_energy_2d


/* ***************************************** */
/* ***************************************** */
/* ***************************************** */

#ifdef USE_DOUBLE
#define HYDRO_UPDATE_3D_DIMX	24
#define HYDRO_UPDATE_3D_DIMY	16
#else // simple precision
#define HYDRO_UPDATE_3D_DIMX	24
#define HYDRO_UPDATE_3D_DIMY	16
#endif // USE_DOUBLE

/**
 * CUDA kernel perform hydro update from flux arrays (3D data).
 * 
 * \param[inout] Uin    conservative variables array.
 * \param[in]    flux_x flux along X.
 * \param[in]    flux_y flux along Y.
 * \param[in]    flux_z flux along Z.
 *
 */
__global__ void kernel_hydro_update_3d(real_t* Uin, 
				       real_t* flux_x,
				       real_t* flux_y,
				       real_t* flux_z,
				       int ghostWidth,
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
  
  const int i = __mul24(bx, HYDRO_UPDATE_3D_DIMX) + tx;
  const int j = __mul24(by, HYDRO_UPDATE_3D_DIMY) + ty;
  
  const int arraySize  = pitch * jmax * kmax;

  // conservative variables
  real_t uIn[NVAR_3D];

  // flux
  real_t flux[NVAR_3D], fluxNext[NVAR_3D];

  for (int k=ghostWidth, elemOffset = i + pitch * (j + jmax * ghostWidth);
       k < kmax-ghostWidth; 
       ++k, elemOffset += (pitch*jmax)) {
    
    
    if (i>=ghostWidth and i<imax-ghostWidth and
	j>=ghostWidth and j<jmax-ghostWidth)
      {
	
	// read input conservative variables
	int offset = elemOffset;
	uIn[ID] = Uin[offset];  offset += arraySize;
	uIn[IP] = Uin[offset];  offset += arraySize;
	uIn[IU] = Uin[offset];  offset += arraySize;
	uIn[IV] = Uin[offset];  offset += arraySize;
	uIn[IW] = Uin[offset];
	
	// read flux_x
	offset = elemOffset;
	flux[ID] = flux_x[offset];  fluxNext[ID] = flux_x[offset+1]; offset += arraySize;
	flux[IP] = flux_x[offset];  fluxNext[IP] = flux_x[offset+1]; offset += arraySize;
	flux[IU] = flux_x[offset];  fluxNext[IU] = flux_x[offset+1]; offset += arraySize;
	flux[IV] = flux_x[offset];  fluxNext[IV] = flux_x[offset+1]; offset += arraySize;
	flux[IW] = flux_x[offset];  fluxNext[IW] = flux_x[offset+1];
	
	// update with flux_x
	uIn[ID] += (flux[ID] - fluxNext[ID]);
	uIn[IP] += (flux[IP] - fluxNext[IP]);
	uIn[IU] += (flux[IU] - fluxNext[IU]);
	uIn[IV] += (flux[IV] - fluxNext[IV]);
	uIn[IW] += (flux[IW] - fluxNext[IW]);
	
	// read flux_y
	offset = elemOffset;
	flux[ID] = flux_y[offset];  fluxNext[ID] = flux_y[offset+pitch]; offset += arraySize;
	flux[IP] = flux_y[offset];  fluxNext[IP] = flux_y[offset+pitch]; offset += arraySize;
	flux[IU] = flux_y[offset];  fluxNext[IU] = flux_y[offset+pitch]; offset += arraySize;
	flux[IV] = flux_y[offset];  fluxNext[IV] = flux_y[offset+pitch]; offset += arraySize;
	flux[IW] = flux_y[offset];  fluxNext[IW] = flux_y[offset+pitch];
	
	// update with flux_y
	uIn[ID] += (flux[ID] - fluxNext[ID]);
	uIn[IP] += (flux[IP] - fluxNext[IP]);
	uIn[IU] += (flux[IU] - fluxNext[IU]);
	uIn[IV] += (flux[IV] - fluxNext[IV]);
	uIn[IW] += (flux[IW] - fluxNext[IW]);
	
	// read flux_z
	offset = elemOffset;
	flux[ID] = flux_z[offset];  fluxNext[ID] = flux_z[offset+pitch*jmax]; offset += arraySize;
	flux[IP] = flux_z[offset];  fluxNext[IP] = flux_z[offset+pitch*jmax]; offset += arraySize;
	flux[IU] = flux_z[offset];  fluxNext[IU] = flux_z[offset+pitch*jmax]; offset += arraySize;
	flux[IV] = flux_z[offset];  fluxNext[IV] = flux_z[offset+pitch*jmax]; offset += arraySize;
	flux[IW] = flux_z[offset];  fluxNext[IW] = flux_z[offset+pitch*jmax];
	
	// update with flux_z
	uIn[ID] += (flux[ID] - fluxNext[ID]);
	uIn[IP] += (flux[IP] - fluxNext[IP]);
	uIn[IU] += (flux[IU] - fluxNext[IU]);
	uIn[IV] += (flux[IV] - fluxNext[IV]);
	uIn[IW] += (flux[IW] - fluxNext[IW]);
	
	// write back conservative variables
	offset = elemOffset;
	Uin[offset] = uIn[ID];  offset += arraySize;
	Uin[offset] = uIn[IP];  offset += arraySize;
	Uin[offset] = uIn[IU];  offset += arraySize;
	Uin[offset] = uIn[IV];  offset += arraySize;
	Uin[offset] = uIn[IW];

      } // end i-j guard
  } // end for k

} // kernel_hydro_update_3d

/**
 * CUDA kernel perform hydro update (energy only) from flux arrays (3D data).
 * 
 * \param[inout] Uin    conservative variables array.
 * \param[in]    flux_x flux along X.
 * \param[in]    flux_y flux along Y.
 * \param[in]    flux_z flux along Z.
 *
 */
__global__ void kernel_hydro_update_energy_3d(real_t* Uin, 
					      real_t* flux_x,
					      real_t* flux_y,
					      real_t* flux_z,
					      int ghostWidth,
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
  
  const int i = __mul24(bx, HYDRO_UPDATE_3D_DIMX) + tx;
  const int j = __mul24(by, HYDRO_UPDATE_3D_DIMY) + ty;
  
  const int arraySize  = pitch * jmax * kmax;

  // conservative variables
  real_t uIn_IP;

  // flux
  real_t flux_IP, fluxNext_IP;

  for (int k=ghostWidth, elemOffset = i + pitch * (j + jmax * ghostWidth);
       k < kmax-ghostWidth; 
       ++k, elemOffset += (pitch*jmax)) {
    
    
    if (i>=ghostWidth and i<imax-ghostWidth and
	j>=ghostWidth and j<jmax-ghostWidth)
      {
	
	// read input conservative variables
	int offset = elemOffset+arraySize;
	uIn_IP = Uin[offset];

	// read flux_x
	offset = elemOffset+arraySize;
	flux_IP     = flux_x[offset];  
	fluxNext_IP = flux_x[offset+1];
	
	// update with flux_x
	uIn_IP += (flux_IP - fluxNext_IP);
	
	// read flux_y
	offset = elemOffset+arraySize;
	flux_IP     = flux_y[offset];  
	fluxNext_IP = flux_y[offset+pitch]; 

	// update with flux_y
	uIn_IP += (flux_IP - fluxNext_IP);
	
	// read flux_z
	offset = elemOffset+arraySize;
	flux_IP     = flux_z[offset];  
	fluxNext_IP = flux_z[offset+pitch*jmax];

	// update with flux_z
	uIn_IP += (flux_IP - fluxNext_IP);
	
	// write back conservative variables - energy only
	offset = elemOffset+arraySize;
	Uin[offset] = uIn_IP;

      } // end i-j guard
  } // end for k

} // kernel_hydro_update_energy_3d

#endif // HYDRO_UPDATE_CUH_
