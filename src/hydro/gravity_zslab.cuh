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
 * \file gravity_zslab.cuh
 * \brief CUDA kernel for computing gravity forces (adapted from Dumses).
 *
 * \date 14 Feb 2014
 * \author P. Kestener
 *
 * $Id: gravity_zslab.cuh 3449 2014-06-16 16:24:38Z pkestene $
 */
#ifndef GRAVITY_ZSLAB_CUH_
#define GRAVITY_ZSLAB_CUH_

#include "real_type.h"
#include "constants.h"

#define GRAVITY_PRED_Z_3D_DIMX	16
#define GRAVITY_PRED_Z_3D_DIMY	16
/*
 * CUDA kernel computing gravity predictor (3D data).
 * 
 * We assume here that Q has z dimension sized upon slab,
 * whereas gravity is sized globally.
 *
 * \param[in,out]  Q      primitive variables array.
 * \param[in]      dt     time step
 * \param[in]      zSlabInfo
 */
__global__ 
void kernel_gravity_predictor_3d_zslab(real_t* Q, 
				       int ghostWidth,
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
  
  const int arraySizeG  = pitch * jmax * kmax;
  const int arraySizeQ  = pitch * jmax * zSlabInfo.zSlabWidthG;

  const int &kStart     = zSlabInfo.kStart; 
  //const int &kStop      = zSlabInfo.kStop; 
  const int &ksizeSlab  = zSlabInfo.ksizeSlab;

  real_t *gravin = gParams.arrayList[A_GRAV];
  
  // update velocity components (IU, IV, IW) with v=v+gravin*dt*half
  for (int kL=ghostWidth, elemOffsetQ = i + pitch * (j + jmax * ghostWidth);
       kL < ksizeSlab-ghostWidth; 
       ++kL, elemOffsetQ += (pitch*jmax)) {
    
    if (i>=ghostWidth and i<imax-ghostWidth and
	j>=ghostWidth and j<jmax-ghostWidth)
      {
	
	int offsetQ = elemOffsetQ +                    2*arraySizeQ;
	int offsetG = elemOffsetQ + kStart*jmax*pitch;
	
	Q[offsetQ] += HALF_F * dt * gravin[offsetG]; // component IU
	offsetQ += arraySizeQ; 
	offsetG += arraySizeG;
	
	Q[offsetQ] += HALF_F * dt * gravin[offsetG]; // component IV
	offsetQ += arraySizeQ; 
	offsetG += arraySizeG;
	
	Q[offsetQ] += HALF_F * dt * gravin[offsetG]; // component IW

      } // end if i,j

  } // end for k

} // kernel_gravity_predictor_3d_zslab


#define GRAVITY_SRC_Z_3D_DIMX	16
#define GRAVITY_SRC_Z_3D_DIMY	16
/*
 * CUDA kernel computing gravity source term (3D data).
 * 
 * \param[in,out]  UNew   conservative variables array.
 * \param[in,out]  UOld   conservative variables array.
 * \param[in]      dt     time step
 */
__global__
void kernel_gravity_source_term_3d_zslab(real_t* UNew, 
					 real_t* UOld, 
					 int ghostWidth,
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
  
  const int arraySizeU  = pitch * jmax * kmax;
  //const int arraySizeQ  = pitch * jmax * zSlabInfo.zSlabWidthG;

  const int &kStart     = zSlabInfo.kStart; 
  //const int &kStop      = zSlabInfo.kStop; 
  const int &ksizeSlab  = zSlabInfo.ksizeSlab;

  real_t *gravin = gParams.arrayList[A_GRAV];
  
  // update velocity components (IU, IV, IW) with v=v+gravin*dt*half
  for (int kL=ghostWidth, elemOffsetQ = i + pitch * (j + jmax * ghostWidth);
       kL < ksizeSlab-ghostWidth; 
       ++kL, elemOffsetQ += (pitch*jmax)) {
    
    if (i>=ghostWidth and i<imax-ghostWidth and
	j>=ghostWidth and j<jmax-ghostWidth)
      {
	
	// set indexes offset for reading velocity components U,V,W
	int offsetU = elemOffsetQ + kStart*jmax*pitch + IU*arraySizeU;

	// set index offset for reading gravity components
	int offsetG = elemOffsetQ + kStart*jmax*pitch;

	// read density
	real_t rhoOld = UOld[elemOffsetQ + kStart*jmax*pitch];
	real_t rhoNew = UNew[elemOffsetQ + kStart*jmax*pitch];

	// component IU
	UNew[offsetU] += HALF_F * dt * gravin[offsetG] * (rhoOld + rhoNew);
	offsetU += arraySizeU;
	offsetG += arraySizeU;

	// component IV
	UNew[offsetU] += HALF_F * dt * gravin[offsetG] * (rhoOld + rhoNew);
	offsetU += arraySizeU;
	offsetG += arraySizeU;

	// component IW
	UNew[offsetU] += HALF_F * dt * gravin[offsetG] * (rhoOld + rhoNew);

      } // end if i,j

  } // end for k

} // kernel_gravity_source_term_3d_zslab

#endif // GRAVITY_ZSLAB_CUH_
