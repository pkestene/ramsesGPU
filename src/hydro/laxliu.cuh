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
 * \file laxliu.cuh
 * \brief Implement GPU kernels for the Lax-Liu positive scheme.
 *
 * \date 05/02/2010
 * \author Pierre Kestener.
 *
 * $Id: laxliu.cuh 2108 2012-05-23 12:07:21Z pkestene $
 */
#ifndef LAXLIU_CUH_
#define LAXLIU_CUH_

#define BLOCK_DIMX		32
#define BLOCK_DIMY		14
#define BLOCK_INNER_DIMX	(BLOCK_DIMX-4)
#define BLOCK_INNER_DIMY	(BLOCK_DIMY-4)

#include "constants.h"
#include "constoprim.h"

#include "positiveScheme.h"


#include <cstdlib> // just in case ...

__global__ void laxliu_evolve_kernel(float* a1, float* a2, int pitch, int isize, int jsize, float xLambda, float yLambda)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index
  const int i = __mul24(bx, BLOCK_INNER_DIMX) + tx;
  const int j = __mul24(by, BLOCK_INNER_DIMY) + ty;
  
  const int arraySize  = __umul24(pitch, jsize);
  const int elemOffset = __umul24(pitch, j) + i;
  
  const int nx = isize-4;
  const int ny = jsize-4;

  /*
   * copy data from global mem to shared mem
   */
  __shared__ float data  [NVAR_2D][BLOCK_DIMY][BLOCK_DIMX];
  //__shared__ float data2 [NVAR_2D][BLOCK_DIMY][BLOCK_DIMX];
  if( (j<jsize) and (i<isize) ) 
    {
      int offset = elemOffset;
      data[ID][ty][tx] = a1[offset]; offset += arraySize;
      data[IU][ty][tx] = a1[offset]; offset += arraySize;
      data[IV][ty][tx] = a1[offset]; offset += arraySize;
      data[IP][ty][tx] = a1[offset];
    }
  __syncthreads();


  /*
   * laxliu2d evolve : 1st stage
   */
  __shared__ float fcdf  [NVAR_2D][BLOCK_DIMY][BLOCK_DIMX];
  fcdf[ID][ty][tx] = 0.0f;
  fcdf[IU][ty][tx] = 0.0f;
  fcdf[IV][ty][tx] = 0.0f;
  fcdf[IP][ty][tx] = 0.0f;
  if ( (j>=2) and (j<ny+2) and (i>=1) and (i<nx+2) and tx>0 and tx<BLOCK_DIMX-2 )
    {
      float up[NVAR_2D], um[NVAR_2D];
      float du[NVAR_2D], dup[NVAR_2D], dum[NVAR_2D];
      float fc[NVAR_2D], df[NVAR_2D] = {0, 0};

      for (unsigned int k=0; k < NVAR_2D; ++k)
	{
	  up[k]  = data[k][ty][tx+1];
	  um[k]  = data[k][ty][tx  ];
	  dum[k] = data[k][ty][tx  ] - data[k][ty][tx-1];
	  dup[k] = data[k][ty][tx+2] - data[k][ty][tx+1];
	  du[k]  = data[k][ty][tx+1] - data[k][ty][tx  ];
	}
      
      central_diff_flux<NVAR_2D>(up,um,fc);
      diffusive_flux<NVAR_2D>(up,um,du,dup,dum,df);
      for (unsigned int k=0; k < NVAR_2D; ++k)
	{
	  fcdf[k][ty][tx] = fc[k]+df[k];
	}

    }
    __syncthreads();

    // compute deltaX (inner block)
    if ( (j>=2) and (j<ny+2) and (i>=2) and (i<nx+2) and tx>=2 and tx<BLOCK_DIMX-2 and ty>=2 and ty<BLOCK_DIMY-2 )
      {
	float delta, tmp;
	int offset = elemOffset;
	for (unsigned int k=0; k < NVAR_2D; ++k, offset+=arraySize)
	  {
	    delta      = fcdf[k][ty][tx] - fcdf[k][ty][tx-1];
	    tmp        = data[k][ty][tx] - xLambda*delta;
	    if (k==0 || k==3) {
	      a2[offset] = fmaxf(tmp,::gParams.smallr);
	      //data2[k][ty][tx] = fmaxf(tmp,::gParams.smallr);
	    } else {
	      a2[offset] = tmp;
	      //data2[k][ty][tx] = tmp;
	    }
	  }
      }
    __syncthreads();

  /*
   * laxliu2d evolve : 2nd stage
   */
  fcdf[ID][ty][tx] = 0.0f;
  fcdf[IU][ty][tx] = 0.0f;
  fcdf[IV][ty][tx] = 0.0f;
  fcdf[IP][ty][tx] = 0.0f;
  if ( (j>=1) and (j<ny+2) and (i>=2) and (i<nx+2) and ty>0 and ty<BLOCK_DIMY-2 )
    {
      float up[NVAR_2D], um[NVAR_2D];
      float du[NVAR_2D], dup[NVAR_2D], dum[NVAR_2D];
      float fc[NVAR_2D], df[NVAR_2D];
      
      for (unsigned int k=0; k < NVAR_2D; ++k)
	{
	  // swap directions :
	  int k1=k;
	  if (k==1 || k==2)
	    k1=3-k;
	  
	  up[k]  = data[k1][ty+1][tx];
	  um[k]  = data[k1][ty  ][tx];
	  dum[k] = data[k1][ty  ][tx] - data[k1][ty-1][tx];
	  dup[k] = data[k1][ty+2][tx] - data[k1][ty+1][tx];
	  du[k]  = data[k1][ty+1][tx] - data[k1][ty  ][tx];
	}
      
      central_diff_flux<NVAR_2D>(up,um,fc);
      diffusive_flux<NVAR_2D>(up,um,du,dup,dum,df);
      for (unsigned int k=0; k < NVAR_2D; ++k)
	{
	  fcdf[k][ty][tx] = fc[k]+df[k];
	}
      
    }
  __syncthreads();

    // compute deltaY (inner block)
    if ( (j>=2) and (j<ny+2) and (i>=2) and (i<nx+2) and tx>=2 and tx<BLOCK_DIMX-2 and ty>=2 and ty<BLOCK_DIMY-2 )
      {
	float delta,tmp;
	int offset = elemOffset;
	for (unsigned int k=0; k < NVAR_2D; ++k, offset+=arraySize)
	  {
	    int k1=k;
	    if (k==1 || k==2)
	      k1=3-k;
	    delta       = fcdf[k1][ty][tx] - fcdf[k1][ty-1][tx];
	    //tmp         = data2[k][ty][tx] - yLambda*delta;
	    tmp         = a2[offset]       - yLambda*delta;
	    if (k==0 || k==3) {
	      a2[offset]  = fmaxf(tmp, ::gParams.smallr);
	    } else {
	      a2[offset]  = tmp;
	    }
	  }
      }

} // laxliu_evolve_kernel


__global__ void laxliu_average_kernel(float* a1, float* a2, int pitch, int isize, int jsize)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index
  const int i = __mul24(bx, blockDim.x) + tx;
  const int j = __mul24(by, blockDim.y) + ty;
  
  const int arraySize  = __umul24(pitch, jsize);
  const int elemOffset = __umul24(pitch, j) + i;
  
  //const int nx = isize-4;
  //const int ny = jsize-4;

  int offset = elemOffset;
  for (unsigned int k=0; k < NVAR_2D; ++k, offset+=arraySize)
    {
      a1[offset] = 0.5f*(a1[offset]+a2[offset]);
    }

} // laxliu_average_kernel

#endif // LAXLIU_CUH_
