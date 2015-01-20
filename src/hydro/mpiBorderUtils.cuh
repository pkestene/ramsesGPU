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
 * \file mpiBorderUtils.cuh
 * \brief Provides the CUDA kernel for copying border buffer between
 * host and device memory.
 *
 * \date 21 Oct 2010
 * \author Pierre Kestener
 *
 * $Id: mpiBorderUtils.cuh 2108 2012-05-23 12:07:21Z pkestene $
 */
#ifndef MPI_BORDER_UTILS_CUH_
#define MPI_BORDER_UTILS_CUH_

#include "constants.h"

// number of cuda threads per block
#define COPY_BORDER_BLOCK_SIZE    128
#define COPY_BORDER_BLOCK_SIZE_3D  16

#define INDEX_2D_LINEAR(_i,_j,_var,_isize,_jsize) ( (_i) + (_isize) * ( (_j) + (_jsize)*(_var) ) )
#define INDEX_3D_LINEAR(_i,_j,_k,_var,_isize,_jsize,_ksize) ( (_i) + (_isize) * ( (_j) + (_jsize) * ( (_k) + (_ksize) * (_var) ) ) )

// these macros are completely redundant with the ones above, but
// makes the code clearer (to my humble opinion...)
#define INDEX_2D_PITCHED(_i,_j,_var,_pitch,_jsize) ( (_i) + (_pitch) * ( (_j) + (_jsize)*(_var) ) )
#define INDEX_3D_PITCHED(_i,_j,_k,_var,_pitch,_jsize,_ksize) ( (_i) + (_pitch) * ( (_j) + (_jsize) * ( (_k) + (_ksize) * (_var) ) ) )

// =======================================================
// =======================================================
/**
 * cuda kernel for copying 1D border buffer (PITCHED memory type) to 2d array
 */
template<BoundaryLocation boundaryLoc,
	 int              ghostWidth>
__GLOBAL__
void copyDeviceArrayToBorderBufSend_2d_kernel(real_t* border, 
					      int bPitch, 
					      dim3 bDim, 
					      real_t* U, 
					      int pitch, 
					      dim3 domainDim, 
					      int nVar)
{
  
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  
  const int index  = bx * COPY_BORDER_BLOCK_SIZE + tx;

  const uint isize  = domainDim.x;
  const uint jsize  = domainDim.y;
  const uint bJsize = bDim.y;

  // offset used to access data of a MIN or MAX type border
  // default is ghostWidth (for a MIN border) 
  // otherwise it is size-2*ghostWidth
  int offset = ghostWidth;
  if (boundaryLoc == XMAX)
    offset = isize-2*ghostWidth;
  if (boundaryLoc == YMAX)
    offset = jsize-2*ghostWidth;

  // XMIN or XMAX border
  if ( boundaryLoc == XMIN or boundaryLoc == XMAX )
    if (index < jsize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_2D_PITCHED(0,index,iVar,bPitch,bJsize)] = U[INDEX_2D_PITCHED(offset  ,index,iVar,pitch,jsize)];
	border[INDEX_2D_PITCHED(1,index,iVar,bPitch,bJsize)] = U[INDEX_2D_PITCHED(offset+1,index,iVar,pitch,jsize)];
	if (ghostWidth == 3)
	  border[INDEX_2D_PITCHED(2,index,iVar,bPitch,bJsize)] = U[INDEX_2D_PITCHED(offset+2,index,iVar,pitch,jsize)];
      }
    }

  // YMIN or YMAX border
  if ( boundaryLoc == YMIN or boundaryLoc == YMAX )
    if (index < isize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_2D_PITCHED(index,0,iVar,bPitch,bJsize)] = U[INDEX_2D_PITCHED(index,offset  ,iVar,pitch,jsize)];
	border[INDEX_2D_PITCHED(index,1,iVar,bPitch,bJsize)] = U[INDEX_2D_PITCHED(index,offset+1,iVar,pitch,jsize)];
	if (ghostWidth == 3)
	  border[INDEX_2D_PITCHED(index,2,iVar,bPitch,bJsize)] = U[INDEX_2D_PITCHED(index,offset+2,iVar,pitch,jsize)];
      }
    }

} // copyDeviceArrayToBorderBufSend_2d_kernel


// =======================================================
// =======================================================
/**
 * cuda kernel for copying 1D border buffer (LINEAR memory type) to 2d array
 */
template<BoundaryLocation boundaryLoc,
	 int              ghostWidth>
__GLOBAL__
void copyDeviceArrayToBorderBufSend_linear_2d_kernel(real_t* border, 
						     dim3 bDim, 
						     real_t* U, 
						     int pitch, 
						     dim3 domainDim, 
						     int nVar)
{
  
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  
  const int index  = bx * COPY_BORDER_BLOCK_SIZE + tx;

  const uint isize  = domainDim.x;
  const uint jsize  = domainDim.y;
  const uint bIsize = bDim.x;
  const uint bJsize = bDim.y;

  // offset used to access data of a MIN or MAX type border
  // default is ghostWidth (for a MIN border) 
  // otherwise it is size-2*ghostWidth
  int offset = ghostWidth;
  if (boundaryLoc == XMAX)
    offset = isize-2*ghostWidth;
  if (boundaryLoc == YMAX)
    offset = jsize-2*ghostWidth;

  // XMIN or XMAX border
  if ( boundaryLoc == XMIN or boundaryLoc == XMAX )
    if (index < jsize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_2D_LINEAR(0,index,iVar,bIsize,bJsize)] = U[INDEX_2D_PITCHED(offset  ,index,iVar,pitch,jsize)];
	border[INDEX_2D_LINEAR(1,index,iVar,bIsize,bJsize)] = U[INDEX_2D_PITCHED(offset+1,index,iVar,pitch,jsize)];
	if (ghostWidth == 3)
	  border[INDEX_2D_LINEAR(2,index,iVar,bIsize,bJsize)] = U[INDEX_2D_PITCHED(offset+2,index,iVar,pitch,jsize)];
      }
    }

  // YMIN or YMAX border
  if ( boundaryLoc == YMIN or boundaryLoc == YMAX )
    if (index < isize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_2D_LINEAR(index,0,iVar,bIsize,bJsize)] = U[INDEX_2D_PITCHED(index,offset  ,iVar,pitch,jsize)];
	border[INDEX_2D_LINEAR(index,1,iVar,bIsize,bJsize)] = U[INDEX_2D_PITCHED(index,offset+1,iVar,pitch,jsize)];
	if (ghostWidth == 3)
	  border[INDEX_2D_LINEAR(index,2,iVar,bIsize,bJsize)] = U[INDEX_2D_PITCHED(index,offset+2,iVar,pitch,jsize)];
      }
    }

} // copyDeviceArrayToBorderBufSend_linear_2d_kernel


// =======================================================
// =======================================================
/**
 * cuda kernel for copying 2D border (PITCHED memory type) buffer to 3d array
 */
template<BoundaryLocation boundaryLoc,
	 int              ghostWidth>
__GLOBAL__
void copyDeviceArrayToBorderBufSend_3d_kernel(real_t* border, 
					      int bPitch, 
					      dim3 bDim, 
					      real_t* U, 
					      int pitch, 
					      dim3 domainDim, 
					      int nVar)
{
  
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int index1  = bx * COPY_BORDER_BLOCK_SIZE_3D + tx;
  const int index2  = by * COPY_BORDER_BLOCK_SIZE_3D + ty;

  const uint isize = domainDim.x;
  const uint jsize = domainDim.y;
  const uint ksize = domainDim.z;
  
  const uint bJsize = bDim.y;
  const uint bKsize = bDim.z;

  // offset used to access data of a MIN or MAX type border
  // default is ghostWidth (for a MIN border) because there are ghostWidth ghost cells
  // otherwise it is size-2*ghostWidth
  int offset = ghostWidth;
  if (boundaryLoc == XMAX)
    offset = isize-2*ghostWidth;
  if (boundaryLoc == YMAX)
    offset = jsize-2*ghostWidth;
  if (boundaryLoc == ZMAX)
    offset = ksize-2*ghostWidth;

  // XMIN or XMAX border
  if ( boundaryLoc == XMIN or boundaryLoc == XMAX )
    if (index1 < jsize and index2 < ksize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_3D_PITCHED(0       ,index1,index2,iVar,bPitch,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(offset  ,index1,index2,iVar, pitch, jsize, ksize)];
	border[INDEX_3D_PITCHED(1       ,index1,index2,iVar,bPitch,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(offset+1,index1,index2,iVar, pitch, jsize, ksize)];
	if (ghostWidth == 3)
	  border[INDEX_3D_PITCHED(2       ,index1,index2,iVar,bPitch,bJsize,bKsize)] = 
	    U[   INDEX_3D_PITCHED(offset+2,index1,index2,iVar, pitch, jsize, ksize)];
      }
    }

  // YMIN or YMAX border
  if ( boundaryLoc == YMIN or boundaryLoc == YMAX )
    if (index1 < isize and index2 < ksize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_3D_PITCHED(index1,       0,index2,iVar,bPitch,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(index1,offset  ,index2,iVar, pitch, jsize, ksize)];
	border[INDEX_3D_PITCHED(index1,       1,index2,iVar,bPitch,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(index1,offset+1,index2,iVar, pitch, jsize, ksize)];
	if (ghostWidth == 3)
	  border[INDEX_3D_PITCHED(index1,       2,index2,iVar,bPitch,bJsize,bKsize)] = 
	    U[   INDEX_3D_PITCHED(index1,offset+2,index2,iVar, pitch, jsize, ksize)];
      }
    }

  // ZMIN or ZMAX border
  if ( boundaryLoc == ZMIN or boundaryLoc == ZMAX )
    if (index1 < isize and index2 < jsize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_3D_PITCHED(index1,index2,       0,iVar,bPitch,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(index1,index2,offset  ,iVar, pitch, jsize, ksize)];
	border[INDEX_3D_PITCHED(index1,index2,       1,iVar,bPitch,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(index1,index2,offset+1,iVar, pitch, jsize, ksize)];
	if (ghostWidth == 3)
	  border[INDEX_3D_PITCHED(index1,index2,       2,iVar,bPitch,bJsize,bKsize)] = 
	    U[   INDEX_3D_PITCHED(index1,index2,offset+2,iVar, pitch, jsize, ksize)];
      }
    }

} // copyDeviceArrayToBorderBufSend_3d_kernel

// =======================================================
// =======================================================
/**
 * cuda kernel for copying 2D border buffer (LINEAR memory type) to 3d array
 */
template<BoundaryLocation boundaryLoc,
	 int              ghostWidth>
__GLOBAL__
void copyDeviceArrayToBorderBufSend_linear_3d_kernel(real_t* border, 
						     dim3 bDim, 
						     real_t* U, 
						     int pitch, 
						     dim3 domainDim, 
						     int nVar)
{
  
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int index1  = bx * COPY_BORDER_BLOCK_SIZE_3D + tx;
  const int index2  = by * COPY_BORDER_BLOCK_SIZE_3D + ty;

  const uint isize = domainDim.x;
  const uint jsize = domainDim.y;
  const uint ksize = domainDim.z;
  
  const uint bIsize = bDim.x;
  const uint bJsize = bDim.y;
  const uint bKsize = bDim.z;

  // offset used to access data of a MIN or MAX type border
  // default is ghostWidth (for a MIN border) because there are ghostWidth ghost cells
  // otherwise it is size-2*ghostWidth
  int offset = ghostWidth;
  if (boundaryLoc == XMAX)
    offset = isize-2*ghostWidth;
  if (boundaryLoc == YMAX)
    offset = jsize-2*ghostWidth;
  if (boundaryLoc == ZMAX)
    offset = ksize-2*ghostWidth;

  // XMIN or XMAX border
  if ( boundaryLoc == XMIN or boundaryLoc == XMAX )
    if (index1 < jsize and index2 < ksize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_3D_LINEAR (0       ,index1,index2,iVar,bIsize,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(offset  ,index1,index2,iVar, pitch, jsize, ksize)];
	border[INDEX_3D_LINEAR (1       ,index1,index2,iVar,bIsize,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(offset+1,index1,index2,iVar, pitch, jsize, ksize)];
	if (ghostWidth == 3)
	  border[INDEX_3D_LINEAR (2       ,index1,index2,iVar,bIsize,bJsize,bKsize)] = 
	    U[   INDEX_3D_PITCHED(offset+2,index1,index2,iVar, pitch, jsize, ksize)];
      }
    }

  // YMIN or YMAX border
  if ( boundaryLoc == YMIN or boundaryLoc == YMAX )
    if (index1 < isize and index2 < ksize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_3D_LINEAR (index1,       0,index2,iVar,bIsize,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(index1,offset  ,index2,iVar, pitch, jsize, ksize)];
	border[INDEX_3D_LINEAR (index1,       1,index2,iVar,bIsize,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(index1,offset+1,index2,iVar, pitch, jsize, ksize)];
	if (ghostWidth == 3)
	  border[INDEX_3D_LINEAR (index1,       2,index2,iVar,bIsize,bJsize,bKsize)] = 
	    U[   INDEX_3D_PITCHED(index1,offset+2,index2,iVar, pitch, jsize, ksize)];
      }
    }

  // ZMIN or ZMAX border
  if ( boundaryLoc == ZMIN or boundaryLoc == ZMAX )
    if (index1 < isize and index2 < jsize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	border[INDEX_3D_LINEAR (index1,index2,       0,iVar,bIsize,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(index1,index2,offset  ,iVar, pitch, jsize, ksize)];
	border[INDEX_3D_LINEAR (index1,index2,       1,iVar,bIsize,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(index1,index2,offset+1,iVar, pitch, jsize, ksize)];
	if (ghostWidth == 3)
	border[INDEX_3D_LINEAR (index1,index2,       2,iVar,bIsize,bJsize,bKsize)] = 
	  U[   INDEX_3D_PITCHED(index1,index2,offset+2,iVar, pitch, jsize, ksize)];
      }
    }

} // copyDeviceArrayToBorderBufSend_linear_3d_kernel

// =======================================================
// =======================================================
/**
 * cuda kernel for copying 2d array to 1D border buffer (PITCHED
 * memory type)
 * BE VERY CAREFULL U and border MAY NOT HAVE THE SAME PITCH !!!!!!!!!
 */
template<BoundaryLocation boundaryLoc,
	 int              ghostWidth>
__GLOBAL__
void copyBorderBufSendToDeviceArray_2d_kernel(real_t* border, 
					      int bPitch, 
					      dim3 bDim, 
					      real_t* U, 
					      int pitch, 
					      dim3 domainDim, 
					      int nVar)
{
  
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  
  const int index  = bx * COPY_BORDER_BLOCK_SIZE + tx;

  const uint isize  = domainDim.x;
  const uint jsize  = domainDim.y;
  const uint bJsize = bDim.y;

  // offset used to access data of a MIN or MAX type border
  // default is 0 (a MIN border) because we write inside ghost border
  // otherwise it is size-1*ghostWidth
  int offset = 0;
  if (boundaryLoc == XMAX)
    offset = isize-ghostWidth;
  if (boundaryLoc == YMAX)
    offset = jsize-ghostWidth;

  // XMIN or XMAX border
  if ( boundaryLoc == XMIN or boundaryLoc == XMAX )
    if (index < jsize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[INDEX_2D_PITCHED(offset  ,index,iVar,pitch,jsize)] = border[INDEX_2D_PITCHED(0,index,iVar,bPitch,bJsize)];
	U[INDEX_2D_PITCHED(offset+1,index,iVar,pitch,jsize)] = border[INDEX_2D_PITCHED(1,index,iVar,bPitch,bJsize)];
	if (ghostWidth == 3)
	  U[INDEX_2D_PITCHED(offset+2,index,iVar,pitch,jsize)] = border[INDEX_2D_PITCHED(2,index,iVar,bPitch,bJsize)];	
      }
    }

  // YMIN or YMAX border
  if ( boundaryLoc == YMIN or boundaryLoc == YMAX )
    if (index < isize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[INDEX_2D_PITCHED(index,offset  ,iVar,pitch,jsize)] = border[INDEX_2D_PITCHED(index,0,iVar,bPitch,bJsize)];
	U[INDEX_2D_PITCHED(index,offset+1,iVar,pitch,jsize)] = border[INDEX_2D_PITCHED(index,1,iVar,bPitch,bJsize)];
	if (ghostWidth == 3) 
	  U[INDEX_2D_PITCHED(index,offset+2,iVar,pitch,jsize)] = border[INDEX_2D_PITCHED(index,2,iVar,bPitch,bJsize)];
      }
    }

} // copyBorderBufSendToDeviceArray_2d_kernel

// =======================================================
// =======================================================
/**
 * cuda kernel for copying 2d array to 1D border buffer (LINEAR memory type)
 */
template<BoundaryLocation boundaryLoc,
	 int              ghostWidth>
__GLOBAL__
void copyBorderBufSendToDeviceArray_linear_2d_kernel(real_t* border, 
						     dim3 bDim, 
						     real_t* U, 
						     int pitch, 
						     dim3 domainDim, 
						     int nVar)
{
  
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  
  const int index  = bx * COPY_BORDER_BLOCK_SIZE + tx;

  const uint isize  = domainDim.x;
  const uint jsize  = domainDim.y;
  const uint bIsize = bDim.x;
  const uint bJsize = bDim.y;

  // offset used to access data of a MIN or MAX type border
  // default is 0 (for a MIN border) because we write inside ghost border
  // otherwise it is size-1*ghostWidth
  int offset = 0;
  if (boundaryLoc == XMAX)
    offset = isize-ghostWidth;
  if (boundaryLoc == YMAX)
    offset = jsize-ghostWidth;

  // XMIN or XMAX border
  if ( boundaryLoc == XMIN or boundaryLoc == XMAX )
    if (index < jsize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[INDEX_2D_PITCHED(offset  ,index,iVar,pitch,jsize)] = border[INDEX_2D_LINEAR(0,index,iVar,bIsize,bJsize)];
	U[INDEX_2D_PITCHED(offset+1,index,iVar,pitch,jsize)] = border[INDEX_2D_LINEAR(1,index,iVar,bIsize,bJsize)];
	if (ghostWidth == 3)
	  U[INDEX_2D_PITCHED(offset+2,index,iVar,pitch,jsize)] = border[INDEX_2D_LINEAR(2,index,iVar,bIsize,bJsize)];
      }
    }

  // YMIN or YMAX border
  if ( boundaryLoc == YMIN or boundaryLoc == YMAX )
    if (index < isize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[INDEX_2D_PITCHED(index,offset  ,iVar,pitch,jsize)] = border[INDEX_2D_LINEAR(index,0,iVar,bIsize,bJsize)];
	U[INDEX_2D_PITCHED(index,offset+1,iVar,pitch,jsize)] = border[INDEX_2D_LINEAR(index,1,iVar,bIsize,bJsize)];
	if (ghostWidth == 3)
	  U[INDEX_2D_PITCHED(index,offset+2,iVar,pitch,jsize)] = border[INDEX_2D_LINEAR(index,2,iVar,bIsize,bJsize)];
      }
    }

} // copyBorderBufSendToDeviceArray_linear_2d_kernel

/**
 * cuda kernel for copying 3d array to 2D border buffer (PITCHED
 * memory type)
 */
template<BoundaryLocation boundaryLoc,
	 int              ghostWidth>
__GLOBAL__
void copyBorderBufSendToDeviceArray_3d_kernel(real_t* border, 
					      int bPitch, 
					      dim3 bDim, 
					      real_t* U, 
					      int pitch, 
					      dim3 domainDim, 
					      int nVar)
{
  
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int index1  = bx * COPY_BORDER_BLOCK_SIZE_3D + tx;
  const int index2  = by * COPY_BORDER_BLOCK_SIZE_3D + ty;

  const uint isize = domainDim.x;
  const uint jsize = domainDim.y;
  const uint ksize = domainDim.z;

  const uint bJsize = bDim.y;
  const uint bKsize = bDim.z;

  // offset used to access data of a MIN or MAX type border
  // default is 0 (a MIN border) because we write inside ghost border
  // otherwise it is size-1*ghostWidth
  int offset = 0;
  if (boundaryLoc == XMAX)
    offset = isize-ghostWidth;
  if (boundaryLoc == YMAX)
    offset = jsize-ghostWidth;
  if (boundaryLoc == ZMAX)
    offset = ksize-ghostWidth;

  // XMIN or XMAX border
  if ( boundaryLoc == XMIN or boundaryLoc == XMAX )
    if (index1 < jsize and index2 < ksize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[       INDEX_3D_PITCHED(offset  ,index1,index2,iVar, pitch, jsize, ksize)] = 
	  border[INDEX_3D_PITCHED(       0,index1,index2,iVar,bPitch,bJsize,bKsize)];
	U[       INDEX_3D_PITCHED(offset+1,index1,index2,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_PITCHED(       1,index1,index2,iVar,bPitch,bJsize,bKsize)];
	if (ghostWidth == 3) 
	  U[       INDEX_3D_PITCHED(offset+2,index1,index2,iVar, pitch, jsize, ksize)] =
	    border[INDEX_3D_PITCHED(       2,index1,index2,iVar,bPitch,bJsize,bKsize)];
      }
    }

  // YMIN or YMAX border
  if ( boundaryLoc == YMIN or boundaryLoc == YMAX )
    if (index1 < isize and index2 < ksize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[       INDEX_3D_PITCHED(index1,offset  ,index2,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_PITCHED(index1,       0,index2,iVar,bPitch,bJsize,bKsize)];
	U[       INDEX_3D_PITCHED(index1,offset+1,index2,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_PITCHED(index1,       1,index2,iVar,bPitch,bJsize,bKsize)];
	if (ghostWidth == 3)
	  U[       INDEX_3D_PITCHED(index1,offset+2,index2,iVar, pitch, jsize, ksize)] =
	    border[INDEX_3D_PITCHED(index1,       2,index2,iVar,bPitch,bJsize,bKsize)];
      }
    }

  // ZMIN or ZMAX border
  if ( boundaryLoc == ZMIN or boundaryLoc == ZMAX )
    if (index1 < isize and index2 < jsize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[       INDEX_3D_PITCHED(index1,index2,offset  ,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_PITCHED(index1,index2,       0,iVar,bPitch,bJsize,bKsize)];
	U[       INDEX_3D_PITCHED(index1,index2,offset+1,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_PITCHED(index1,index2,       1,iVar,bPitch,bJsize,bKsize)];
	if (ghostWidth == 3) 
	  U[       INDEX_3D_PITCHED(index1,index2,offset+2,iVar, pitch, jsize, ksize)] =
	    border[INDEX_3D_PITCHED(index1,index2,       2,iVar,bPitch,bJsize,bKsize)];
      }
    }

} // copyBorderBufSendToDeviceArray_3d_kernel

/**
 * cuda kernel for copying 3d array to 2D border buffer (LINEAR memory type)
 */
template<BoundaryLocation boundaryLoc,
	 int              ghostWidth>
__GLOBAL__
void copyBorderBufSendToDeviceArray_linear_3d_kernel(real_t* border, 
						     dim3 bDim, 
						     real_t* U, 
						     int pitch, 
						     dim3 domainDim, 
						     int nVar)
{
  
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  const int index1  = bx * COPY_BORDER_BLOCK_SIZE_3D + tx;
  const int index2  = by * COPY_BORDER_BLOCK_SIZE_3D + ty;

  const uint isize = domainDim.x;
  const uint jsize = domainDim.y;
  const uint ksize = domainDim.z;

  const uint bIsize = bDim.x;
  const uint bJsize = bDim.y;
  const uint bKsize = bDim.z;

  // offset used to access data of a MIN or MAX type border
  // default is 0 (for a MIN border) because we write inside ghost border
  // otherwise it is size-1*ghostWidth
  int offset = 0;
  if (boundaryLoc == XMAX)
    offset = isize-ghostWidth;
  if (boundaryLoc == YMAX)
    offset = jsize-ghostWidth;
  if (boundaryLoc == ZMAX)
    offset = ksize-ghostWidth;

  // XMIN or XMAX border
  if ( boundaryLoc == XMIN or boundaryLoc == XMAX )
    if (index1 < jsize and index2 < ksize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[       INDEX_3D_PITCHED(offset  ,index1,index2,iVar, pitch, jsize, ksize)] = 
	  border[INDEX_3D_LINEAR (       0,index1,index2,iVar,bIsize,bJsize,bKsize)];
	U[       INDEX_3D_PITCHED(offset+1,index1,index2,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_LINEAR (       1,index1,index2,iVar,bIsize,bJsize,bKsize)];
	if (ghostWidth == 3)
	  U[       INDEX_3D_PITCHED(offset+2,index1,index2,iVar, pitch, jsize, ksize)] =
	    border[INDEX_3D_LINEAR (       2,index1,index2,iVar,bIsize,bJsize,bKsize)];
      }
    }

  // YMIN or YMAX border
  if ( boundaryLoc == YMIN or boundaryLoc == YMAX )
    if (index1 < isize and index2 < ksize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[       INDEX_3D_PITCHED(index1,offset  ,index2,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_LINEAR (index1,       0,index2,iVar,bIsize,bJsize,bKsize)];
	U[       INDEX_3D_PITCHED(index1,offset+1,index2,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_LINEAR (index1,       1,index2,iVar,bIsize,bJsize,bKsize)];
	if (ghostWidth == 3)
	  U[       INDEX_3D_PITCHED(index1,offset+2,index2,iVar, pitch, jsize, ksize)] =
	    border[INDEX_3D_LINEAR (index1,       2,index2,iVar,bIsize,bJsize,bKsize)];
      }
    }
  
  // ZMIN or ZMAX border
  if ( boundaryLoc == ZMIN or boundaryLoc == ZMAX )
    if (index1 < isize and index2 < jsize) {
      for (int iVar=0; iVar<nVar; iVar++) {
	U[       INDEX_3D_PITCHED(index1,index2,offset  ,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_LINEAR (index1,index2,       0,iVar,bIsize,bJsize,bKsize)];
	U[       INDEX_3D_PITCHED(index1,index2,offset+1,iVar, pitch, jsize, ksize)] =
	  border[INDEX_3D_LINEAR (index1,index2,       1,iVar,bIsize,bJsize,bKsize)];
	if (ghostWidth == 3)
	  U[       INDEX_3D_PITCHED(index1,index2,offset+2,iVar, pitch, jsize, ksize)] =
	    border[INDEX_3D_LINEAR (index1,index2,       2,iVar,bIsize,bJsize,bKsize)];
      }
    }

} // copyBorderBufSendToDeviceArray_linear_3d_kernel

#endif // MPI_BORDER_UTILS_CUH_
