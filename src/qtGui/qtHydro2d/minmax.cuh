/**
 * \file qtHydro2d/minmax.cuh
 * \brief Implements GPU kernel for computing min and max value of
 * Hydrodynamics arrays.
 * Strongly adapted from cmpdt.cuh
 * 
 *
 * \date 09/03/2010
 * \author Pierre Kestener.
 */
#ifndef MINMAX_CUH_
#define MINMAX_CUH_

#include <real_type.h>
#include <constoprim.h>

#define REDUCE_OP_MAX(x, y) x = FMAX(x, y)
#define REDUCE_OP_MIN(x, y) x = FMIN(x, y)
#define REDUCE_VAR_MAX(array, idx, var, offset) REDUCE_OP_MAX(array[idx].var, array[idx+offset].var)
#define REDUCE_VAR_MIN(array, idx, var, offset) REDUCE_OP_MIN(array[idx].var, array[idx+offset].var)
#define REDUCE(array, idx, offset) REDUCE_VAR_MIN(array, idx, x, offset); REDUCE_VAR_MAX(array, idx, y, offset)

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

#define MINMAX_BLOCK_SIZE	128

/** 
 * compute min and max value of array (density, Ux, Uy or Energy)
 * specified by iVar. 
 * 
 * @param U       : hydrodynamics array
 * @param g_odata : GPU global memory buffer
 * @param n       : section size (pitch taken into account)
 * @param iVar    : control which variable is reduced (density, Ux, Uy or Energy)
 */
template<unsigned int blockSize>
__global__ void minmax_kernel(real_t* U, real2_t* g_odata, unsigned int sectionSize, unsigned int pitch, unsigned int nx, int iVar)
{
  extern __shared__ real2_t minmaxData[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  minmaxData[tid].x =  3.40282347e+38f; // will actually store min
  minmaxData[tid].y = -3.40282347e+38f; // will actually store max

  // one must exclude shadow zone (due to pitched data, which are
  // zero), otherwise one get zero for MIN whatever the true MIN is
  int ix = i - (i/pitch)*pitch;

  int offset = i + sectionSize*iVar;
  int offset2=offset+blockSize;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while(i < sectionSize && ix < nx)
    {
      REDUCE_OP_MIN(minmaxData[tid].x, U[offset]);
      REDUCE_OP_MAX(minmaxData[tid].y, U[offset]);

      int ix2 = (i+blockSize) - ((i+blockSize)/pitch)*pitch;
      if(i+blockSize < sectionSize && ix2 < nx)
	{
	  REDUCE_OP_MIN(minmaxData[tid].x, U[offset2]);
	  REDUCE_OP_MAX(minmaxData[tid].y, U[offset2]);
	}

      i += gridSize;
      ix = i - (i/pitch)*pitch;
    }
  __syncthreads();

  // do reduction in shared mem
  if(blockSize >= 512) { if(tid < 256) { REDUCE(minmaxData, tid, 256); } __syncthreads(); }
  if(blockSize >= 256) { if(tid < 128) { REDUCE(minmaxData, tid, 128); } __syncthreads(); }
  if(blockSize >= 128) { if(tid <  64) { REDUCE(minmaxData, tid,  64); } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
  if(tid < 32)
#endif
    {
      if(blockSize >=  64) { REDUCE(minmaxData, tid, 32); EMUSYNC; }
      if(blockSize >=  32) { REDUCE(minmaxData, tid, 16); EMUSYNC; }
      if(blockSize >=  16) { REDUCE(minmaxData, tid,  8); EMUSYNC; }
      if(blockSize >=   8) { REDUCE(minmaxData, tid,  4); EMUSYNC; }
      if(blockSize >=   4) { REDUCE(minmaxData, tid,  2); EMUSYNC; }
      if(blockSize >=   2) { REDUCE(minmaxData, tid,  1); EMUSYNC; }
    }

  // write result for this block to global mem
  if(tid == 0)
    {
      g_odata[blockIdx.x] = minmaxData[0];
    }
}

#endif /*MINMAX_CUH_*/
