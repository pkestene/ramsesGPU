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
 * \file hydro_update_zslab.cuh
 * \brief CUDA kernel for update conservative variables with flux array inside z-slab.
 *
 * \date 14 Sept 2012
 * \author P. Kestener
 *
 * $Id: hydro_update_zslab.cuh 3449 2014-06-16 16:24:38Z pkestene $
 */
#ifndef HYDRO_UPDATE_ZSLAB_CUH_
#define HYDRO_UPDATE_ZSLAB_CUH_

#include "real_type.h"
#include "constants.h"

/* ***************************************** */
/* ***************************************** */
/* ***************************************** */

#ifdef USE_DOUBLE
#define HYDRO_UPDATE_Z_3D_DIMX	24
#define HYDRO_UPDATE_Z_3D_DIMY	16
#else // simple precision
#define HYDRO_UPDATE_Z_3D_DIMX	24
#define HYDRO_UPDATE_Z_3D_DIMY	16
#endif // USE_DOUBLE

/**
 * CUDA kernel perform hydro update from flux arrays (3D data) inside zslab.
 * 
 * \param[inout] Uin    conservative variables array.
 * \param[in]    flux_x flux along X.
 * \param[in]    flux_y flux along Y.
 * \param[in]    flux_z flux along Z.
 * \param[in]    zSlabInfo
 *
 */
__global__ void kernel_hydro_update_3d_zslab(real_t* Uin, 
					     real_t* flux_x,
					     real_t* flux_y,
					     real_t* flux_z,
					     int ghostWidth,
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
  
  const int i = __mul24(bx, HYDRO_UPDATE_Z_3D_DIMX) + tx;
  const int j = __mul24(by, HYDRO_UPDATE_Z_3D_DIMY) + ty;
  
  const int arraySizeU  = pitch * jmax * kmax;
  const int arraySizeQ  = pitch * jmax * zSlabInfo.zSlabWidthG;

  const int &kStart     = zSlabInfo.kStart; 
  //const int &kStop      = zSlabInfo.kStop; 
  const int &ksizeSlab  = zSlabInfo.ksizeSlab;


  // conservative variables
  real_t uIn[NVAR_3D];

  // flux
  real_t flux[NVAR_3D], fluxNext[NVAR_3D];

  for (int kL=ghostWidth, elemOffsetQ = i + pitch * (j + jmax * ghostWidth);
       kL < ksizeSlab-ghostWidth; 
       ++kL, elemOffsetQ += (pitch*jmax)) {
    
    
    if (i>=ghostWidth and i<imax-ghostWidth and
	j>=ghostWidth and j<jmax-ghostWidth)
      {
	
	// read input conservative variables
	int offsetU = elemOffsetQ + kStart*jmax*pitch;

	uIn[ID] = Uin[offsetU];  offsetU += arraySizeU;
	uIn[IP] = Uin[offsetU];  offsetU += arraySizeU;
	uIn[IU] = Uin[offsetU];  offsetU += arraySizeU;
	uIn[IV] = Uin[offsetU];  offsetU += arraySizeU;
	uIn[IW] = Uin[offsetU];
	
	// read flux_x
	int offsetQ = elemOffsetQ;
	flux[ID] = flux_x[offsetQ];  fluxNext[ID] = flux_x[offsetQ+1]; offsetQ += arraySizeQ;
	flux[IP] = flux_x[offsetQ];  fluxNext[IP] = flux_x[offsetQ+1]; offsetQ += arraySizeQ;
	flux[IU] = flux_x[offsetQ];  fluxNext[IU] = flux_x[offsetQ+1]; offsetQ += arraySizeQ;
	flux[IV] = flux_x[offsetQ];  fluxNext[IV] = flux_x[offsetQ+1]; offsetQ += arraySizeQ;
	flux[IW] = flux_x[offsetQ];  fluxNext[IW] = flux_x[offsetQ+1];
	
	// update with flux_x
	uIn[ID] += (flux[ID] - fluxNext[ID]);
	uIn[IP] += (flux[IP] - fluxNext[IP]);
	uIn[IU] += (flux[IU] - fluxNext[IU]);
	uIn[IV] += (flux[IV] - fluxNext[IV]);
	uIn[IW] += (flux[IW] - fluxNext[IW]);
	
	// read flux_y
	offsetQ = elemOffsetQ;
	flux[ID] = flux_y[offsetQ];  fluxNext[ID] = flux_y[offsetQ+pitch]; offsetQ += arraySizeQ;
	flux[IP] = flux_y[offsetQ];  fluxNext[IP] = flux_y[offsetQ+pitch]; offsetQ += arraySizeQ;
	flux[IU] = flux_y[offsetQ];  fluxNext[IU] = flux_y[offsetQ+pitch]; offsetQ += arraySizeQ;
	flux[IV] = flux_y[offsetQ];  fluxNext[IV] = flux_y[offsetQ+pitch]; offsetQ += arraySizeQ;
	flux[IW] = flux_y[offsetQ];  fluxNext[IW] = flux_y[offsetQ+pitch];
	
	// update with flux_y
	uIn[ID] += (flux[ID] - fluxNext[ID]);
	uIn[IP] += (flux[IP] - fluxNext[IP]);
	uIn[IU] += (flux[IU] - fluxNext[IU]);
	uIn[IV] += (flux[IV] - fluxNext[IV]);
	uIn[IW] += (flux[IW] - fluxNext[IW]);
	
	// read flux_z
	offsetQ = elemOffsetQ;
	flux[ID] = flux_z[offsetQ];  fluxNext[ID] = flux_z[offsetQ+pitch*jmax]; offsetQ += arraySizeQ;
	flux[IP] = flux_z[offsetQ];  fluxNext[IP] = flux_z[offsetQ+pitch*jmax]; offsetQ += arraySizeQ;
	flux[IU] = flux_z[offsetQ];  fluxNext[IU] = flux_z[offsetQ+pitch*jmax]; offsetQ += arraySizeQ;
	flux[IV] = flux_z[offsetQ];  fluxNext[IV] = flux_z[offsetQ+pitch*jmax]; offsetQ += arraySizeQ;
	flux[IW] = flux_z[offsetQ];  fluxNext[IW] = flux_z[offsetQ+pitch*jmax];
	
	// update with flux_z
	uIn[ID] += (flux[ID] - fluxNext[ID]);
	uIn[IP] += (flux[IP] - fluxNext[IP]);
	uIn[IU] += (flux[IU] - fluxNext[IU]);
	uIn[IV] += (flux[IV] - fluxNext[IV]);
	uIn[IW] += (flux[IW] - fluxNext[IW]);
	
	// write back conservative variables
	offsetU = elemOffsetQ + kStart*jmax*pitch;
	Uin[offsetU] = uIn[ID];  offsetU += arraySizeU;
	Uin[offsetU] = uIn[IP];  offsetU += arraySizeU;
	Uin[offsetU] = uIn[IU];  offsetU += arraySizeU;
	Uin[offsetU] = uIn[IV];  offsetU += arraySizeU;
	Uin[offsetU] = uIn[IW];

	/*if (i==2 and j==2)
	  printf("[update] k kL kStart ksizeSlab : %d %d %d %d\n",kL+kStart, kL, kStart, ksizeSlab);*/


      } // end i-j guard
  } // end for k

} // kernel_hydro_update_3d_zslab

/**
 * CUDA kernel perform hydro update (energy only) from flux arrays (3D data, z-slab).
 * 
 * \param[inout] Uin    conservative variables array.
 * \param[in]    flux_x flux along X.
 * \param[in]    flux_y flux along Y.
 * \param[in]    flux_z flux along Z.
 * \param[in]    zSlabInfo
 *
 */
__global__ void kernel_hydro_update_energy_3d_zslab(real_t* Uin, 
						    real_t* flux_x,
						    real_t* flux_y,
						    real_t* flux_z,
						    int ghostWidth,
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
  
  const int i = __mul24(bx, HYDRO_UPDATE_Z_3D_DIMX) + tx;
  const int j = __mul24(by, HYDRO_UPDATE_Z_3D_DIMY) + ty;
  
  const int arraySizeU  = pitch * jmax * kmax;
  const int arraySizeQ  = pitch * jmax * zSlabInfo.zSlabWidthG;

  const int &kStart     = zSlabInfo.kStart; 
  //const int &kStop      = zSlabInfo.kStop; 
  const int &ksizeSlab  = zSlabInfo.ksizeSlab;


  // conservative variables
  real_t uIn_IP;

  // flux
  real_t flux_IP, fluxNext_IP;

  for (int kL=ghostWidth, elemOffsetQ = i + pitch * (j + jmax * ghostWidth);
       kL < ksizeSlab-ghostWidth; 
       ++kL, elemOffsetQ += (pitch*jmax)) {
    
    
    if (i>=ghostWidth and i<imax-ghostWidth and
	j>=ghostWidth and j<jmax-ghostWidth)
      {
	
	// read input conservative variables (skip density)
	int offsetU = (elemOffsetQ +  kStart*jmax*pitch) + arraySizeU;
	uIn_IP = Uin[offsetU];

	// read flux_x
	int offsetQ = elemOffsetQ + arraySizeQ;
	flux_IP     = flux_x[offsetQ];  
	fluxNext_IP = flux_x[offsetQ+1];
	
	// update with flux_x
	uIn_IP += (flux_IP - fluxNext_IP);
	
	// read flux_y
	offsetQ = elemOffsetQ + arraySizeQ;
	flux_IP     = flux_y[offsetQ];  
	fluxNext_IP = flux_y[offsetQ+pitch]; 

	// update with flux_y
	uIn_IP += (flux_IP - fluxNext_IP);
	
	// read flux_z
	offsetQ = elemOffsetQ + arraySizeQ;
	flux_IP     = flux_z[offsetQ];  
	fluxNext_IP = flux_z[offsetQ+pitch*jmax];

	// update with flux_z
	uIn_IP += (flux_IP - fluxNext_IP);
	
	// write back conservative variables - energy only (skip density)
	offsetU = (elemOffsetQ +  kStart*jmax*pitch) + arraySizeU;
	Uin[offsetU] = uIn_IP;

      } // end i-j guard
  } // end for k

} // kernel_hydro_update_energy_3d_zslab

#endif // HYDRO_UPDATE_ZSLAB_CUH_
