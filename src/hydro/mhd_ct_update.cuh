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
 * \file mhd_ct_update.cuh
 * \brief CUDA kernel for update magnetic field with emf (constraint transport).
 *
 * \date 2 May 2012
 * \author P. Kestener
 *
 * $Id: mhd_ct_update.cuh 2110 2012-05-23 12:32:02Z pkestene $
 */
#ifndef CT_UPDATE_CUH_
#define CT_UPDATE_CUH_

#include "real_type.h"
#include "constants.h"

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#define MHD_CT_UPDATE_2D_DIMX	24
#define MHD_CT_UPDATE_2D_DIMY	16
#else // simple precision
#define MHD_CT_UPDATE_2D_DIMX	24
#define MHD_CT_UPDATE_2D_DIMY	16
#endif // USE_DOUBLE

/**
 * CUDA kernel perform magnetic field update (ct) from emf (2D data).
 * 
 * This kernel reads emf data from d_emf and perform constraint transport update
 * of magnetic field components.
 *
 * \param[inout]  Uin
 * \param[in]     d_emf
 */
__global__ void kernel_mhd_ct_update_2d(real_t* Uin, 
					real_t* d_emf,
					int pitch, 
					int imax, 
					int jmax, 
					const real_t dtdx, 
					const real_t dtdy,
					const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, MHD_CT_UPDATE_2D_DIMX) + tx;
  const int j = __mul24(by, MHD_CT_UPDATE_2D_DIMY) + ty;
  
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
      
      Uin[offset] = uOut[IA];  offset += arraySize;
      Uin[offset] = uOut[IB];
      
    } // end i/j guard
  
} // kernel_mhd_ct_update_2d

#ifdef USE_DOUBLE
#define MHD_CT_UPDATE_3D_DIMX	16
#define MHD_CT_UPDATE_3D_DIMY	16
#else // simple precision
#define MHD_CT_UPDATE_3D_DIMX	16
#define MHD_CT_UPDATE_3D_DIMY	16
#endif // USE_DOUBLE

/**
 * CUDA kernel perform magnetic field update (ct) from emf (3D data).
 * 
 * This kernel performs magnetic field update, i.e. constraint trasport update
 * (similar to routine ct in Dumses).
 *
 * \param[inout]  Uin   input MHD conservative variable array
 * \param[in]     d_emf emf input array
 *
 */
__global__ void kernel_mhd_ct_update_3d(real_t* Uin, 
					real_t* d_emf,
					int pitch, 
					int imax, 
					int jmax,
					int kmax,
					const real_t dtdx, 
					const real_t dtdy,
					const real_t dtdz,
					const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, MHD_CT_UPDATE_3D_DIMX) + tx;
  const int j = __mul24(by, MHD_CT_UPDATE_3D_DIMY) + ty;
  
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
	
	Uin[offset] = uOut[IA];  offset += arraySize;
	Uin[offset] = uOut[IB];  offset += arraySize;
	Uin[offset] = uOut[IC];
	
      } // end if i,j guard

  } // end for k

} // kernel_mhd_ct_update_3d

#endif // CT_UPDATE_CUH_
