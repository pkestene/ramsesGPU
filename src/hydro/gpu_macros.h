/**
 * \file gpu_macros.h
 * \brief Some useful GPU related macros.
 *
 * \author P. Kestener
 *
 * $Id: gpu_macros.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef GPU_MACROS_H_
#define GPU_MACROS_H_

#include <cmath>
//#include "real_type.h"

#ifdef __CUDACC__
#define __CONSTANT__ __constant__
#define __GLOBAL__ __global__
#define __HOST__ __host__
#define __DEVICE__ __device__ inline
#define SATURATE __saturatef
#else
#define __CONSTANT__
#define __GLOBAL__
#define __HOST__ 
#define __DEVICE__ inline
#define SATURATE saturate_cpu
#endif // __CUDACC__

float saturate_cpu(float a);

#ifdef __CUDACC__
uint blocksFor(uint elementCount, uint threadCount);
#endif // __CUDACC__

/*
 * define some sanity check routines for cuda runtime
 */
#ifdef __CUDACC__
#include "cutil_inline.h"

inline void checkCudaError(const char *msg)
{
  cudaError_t e = cudaThreadSynchronize();
  if( e != cudaSuccess )
    {
      fprintf(stderr, "CUDA Error in %s : %s\n", msg, cudaGetErrorString(e));
    }
  e = cudaGetLastError();
  if( e != cudaSuccess )
    {
      fprintf(stderr, "CUDA Error %s : %s\n", msg, cudaGetErrorString(e));
    }
} // checkCudaError

inline void checkCudaErrorMpi(const char *msg, const int mpiRank)
{
  cudaError_t e = cudaThreadSynchronize();
  if( e != cudaSuccess )
    {
      fprintf(stderr, "[Mpi rank %4d] CUDA Error in %s : %s\n", mpiRank, msg, cudaGetErrorString(e));
    }
  e = cudaGetLastError();
  if( e != cudaSuccess )
    {
      fprintf(stderr, "[Mpi rank %4d] CUDA Error %s : %s\n", mpiRank, msg, cudaGetErrorString(e));
    }
} // checkCudaErrorMpi
#endif // __CUDACC__

#endif // GPU_MACROS_H_
