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
 * \file copyFaces.cuh
 * \brief Some CUDA kernel for copying faces of a 3D simulation domain.
 *
 *
 * \date 12 March 2013
 * \author P. Kestener
 *
 * $Id: copyFaces.cuh 3236 2014-02-04 00:09:53Z pkestene $
 *
 */
#ifndef COPY_FACES_CUH_
#define COPY_FACES_CUH_

#include "real_type.h"
#include "constants.h"

/**
 * Copy X-face data (i.e. corresponding to x=0 plane).
 *
 * \param[in]  d_u
 * \param[out] d_f
 * \param[in]  isize_u
 * \param[in]  jsize_u
 * \param[in]  ksize_u
 * \param[in]  pitch_u
 * \param[in]  isize_f
 * \param[in]  jsize_f
 * \param[in]  pitch_f
 *
 */
__global__ void kernel_copy_face_x(real_t* d_u,
				   real_t* d_f,
				   int isize_u,
				   int jsize_u,
				   int ksize_u,
				   int pitch_u,
				   int isize_f,
				   int jsize_f,
				   int ksize_f,
				   int pitch_f,
				   int nbVar)
{
  
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int j = __mul24(bx, blockDim.x) + tx;
  const int k = __mul24(by, blockDim.y) + ty;

  if (j<jsize_u and k<ksize_u) {
    
    int offset_u,    offset_f;
    int arraySize_u = pitch_u*jsize_u*ksize_u;
    int arraySize_f = pitch_f*jsize_f*ksize_f;

    for (int iVar=0; iVar<nbVar; iVar++) {
      offset_u = pitch_u*(j + jsize_u*k) + iVar*arraySize_u;
      offset_f = pitch_f*(j + jsize_f*k) + iVar*arraySize_f;
      
      d_f[offset_f] = d_u[offset_u];
    }

  }

} // kernel_copy_face_x

/**
 * Copy Y-face data (i.e. corresponding to y=0 plane).
 *
 * \param[in]  d_u
 * \param[out] d_f
 * \param[in]  isize_u
 * \param[in]  jsize_u
 * \param[in]  ksize_u
 * \param[in]  pitch_u
 * \param[in]  isize_f
 * \param[in]  jsize_f
 * \param[in]  pitch_f
 *
 */
__global__ void kernel_copy_face_y(real_t* d_u,
				   real_t* d_f,
				   int isize_u,
				   int jsize_u,
				   int ksize_u,
				   int pitch_u,
				   int isize_f,
				   int jsize_f,
				   int ksize_f,
				   int pitch_f,
				   int nbVar)
{
  
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, blockDim.x) + tx;
  const int k = __mul24(by, blockDim.y) + ty;

  if (i<isize_u and k<ksize_u) {
    
    int offset_u, offset_f;
    int arraySize_u = pitch_u*jsize_u*ksize_u;
    int arraySize_f = pitch_f*jsize_f*ksize_f;

    for (int iVar=0; iVar<nbVar; iVar++) {
      offset_u = i+pitch_u*jsize_u*k + iVar*arraySize_u;
      offset_f = i+pitch_f*jsize_f*k + iVar*arraySize_f;
      
      d_f[offset_f] = d_u[offset_u];
    }

  }

} // kernel_copy_face_y

/**
 * Copy Z-face data (i.e. corresponding to z=0 plane).
 *
 * \param[in]  d_u
 * \param[out] d_f
 * \param[in]  isize_u
 * \param[in]  jsize_u
 * \param[in]  ksize_u
 * \param[in]  pitch_u
 * \param[in]  isize_f
 * \param[in]  jsize_f
 * \param[in]  pitch_f
 *
 */
__global__ void kernel_copy_face_z(real_t* d_u,
				   real_t* d_f,
				   int isize_u,
				   int jsize_u,
				   int ksize_u,
				   int pitch_u,
				   int isize_f,
				   int jsize_f,
				   int ksize_f,
				   int pitch_f,
				   int nbVar)
{
  
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, blockDim.x) + tx;
  const int j = __mul24(by, blockDim.y) + ty;

  if (i<isize_u and j<jsize_u) {
    
    int offset_u, offset_f;
    int arraySize_u = pitch_u*jsize_u*ksize_u;
    int arraySize_f = pitch_f*jsize_f*ksize_f;

    for (int iVar=0; iVar<nbVar; iVar++) {
      offset_u = i+pitch_u*j + iVar*arraySize_u;
      offset_f = i+pitch_f*j + iVar*arraySize_f;
      
      d_f[offset_f] = d_u[offset_u];
    }

  }

} // kernel_copy_face_z

#endif // COPY_FACES_CUH_
