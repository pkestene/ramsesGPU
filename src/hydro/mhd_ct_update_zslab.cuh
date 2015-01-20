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
 * \file mhd_ct_update_zslab.cuh
 * \brief CUDA kernel for update magnetic field with emf (constraint transport), z-slab implementation.
 *
 * \date Sept 20, 2012
 * \author P. Kestener
 *
 * $Id: mhd_ct_update_zslab.cuh 3449 2014-06-16 16:24:38Z pkestene $
 */
#ifndef CT_UPDATE_ZSLAB_CUH_
#define CT_UPDATE_ZSLAB_CUH_

#include "real_type.h"
#include "constants.h"

#ifdef USE_DOUBLE
#define MHD_CT_UPDATE_Z_3D_DIMX	16
#define MHD_CT_UPDATE_Z_3D_DIMY	16
#else // simple precision
#define MHD_CT_UPDATE_Z_3D_DIMX	16
#define MHD_CT_UPDATE_Z_3D_DIMY	16
#endif // USE_DOUBLE

/**
 * CUDA kernel perform magnetic field update (ct) from emf (3D data).
 * 
 * This kernel performs magnetic field update, i.e. constraint trasport update
 * (similar to routine ct in Dumses).
 *
 * \param[inout]  Uin   input MHD conservative variable array
 * \param[in]     d_emf emf input array
 * \param[in]     zSlabInfo
 *
 */
__global__ void kernel_mhd_ct_update_3d_zslab(real_t* Uin, 
					      real_t* d_emf,
					      int pitch, 
					      int imax, 
					      int jmax,
					      int kmax,
					      const real_t dtdx, 
					      const real_t dtdy,
					      const real_t dtdz,
					      const real_t dt,
					      ZslabInfo zSlabInfo)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int i = __mul24(bx, MHD_CT_UPDATE_Z_3D_DIMX) + tx;
  const int j = __mul24(by, MHD_CT_UPDATE_Z_3D_DIMY) + ty;
  
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
    
    if( i >= 3 and i < imax-2 and 
        j >= 3 and j < jmax-2 )
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
	
	Uin[offsetU] = uOut[IA];  offsetU += arraySizeU;
	Uin[offsetU] = uOut[IB];  offsetU += arraySizeU;
	Uin[offsetU] = uOut[IC];
	
      } // end if i,j guard

  } // end for k

} // kernel_mhd_ct_update_3d_zslab

#endif // CT_UPDATE_ZSLAB_CUH_
