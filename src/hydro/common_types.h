/**
 * \file common_types.h
 * \brief Defines some custom types for compatibility with CUDA.
 *
 * \author P. Kestener
 * \date 29/06/2009
 *
 * $Id: common_types.h 1784 2012-02-21 10:34:58Z pkestene $
 */

#ifndef COMMON_TYPES_H_
#define COMMON_TYPES_H_

#include "real_type.h"

/* if not using CUDA, then defines customs types from vector_types.h */
#ifndef __CUDACC__

typedef unsigned int uint;

/**
 * \struct uint2
 * \brief Two unsigned int structure.
 */
struct uint2
{
  unsigned int x, y;
};
typedef struct uint2 uint2;

inline uint2 make_uint2(unsigned int x, unsigned int y)
{
  uint2 t; t.x = x; t.y = y; return t;
}

/**
 * \struct uint3
 * \brief Three unsigned int structure.
 */
struct uint3
{
  unsigned int x, y, z;
};
typedef struct uint3 uint3;

inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
  uint3 t; t.x = x; t.y = y; t.z = z; return t;
}

/**
 * \struct uint4
 * \brief Four unsigned int structure.
 */
struct uint4
{
  unsigned int x, y, z, w;
};
typedef struct uint4 uint4;

inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
  uint4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

/**
 * \struct float2
 * \brief Two single precision floating point structure.
 */
struct float2
{
  float x, y;
};
typedef struct float2 float2;

/**
 * \struct double2
 * \brief Two double precision floating point structure.
 */
struct double2
{
  double x, y;
};
typedef struct double2 double2;

/**
 * \struct float3
 * \brief Three single precision floating point structure.
 */
struct float3
{
  float x, y, z;
};
typedef struct float3 float3;

/**
 * \struct double3
 * \brief Three double precision floating point structure.
 */
struct double3
{
  double x, y, z;
};
typedef struct double3 double3;

typedef struct dim3 dim3;

/**
 * \struct dim3
 * \brief structure used to set CUDA block sizes.
 */
struct dim3
{
    unsigned int x, y, z;
#if defined(__cplusplus)
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif /* __cplusplus */
};

#endif /* __CUDACC__ */

/**
 * \typedef real2_t
 */
#ifdef USE_DOUBLE
typedef double2 real2_t;
#else
typedef float2 real2_t;
#endif // USE_DOUBLE

/**
 * \typedef real3_t
 */
#ifdef USE_DOUBLE
typedef double3 real3_t;
#else
typedef float3 real3_t;
#endif // USE_DOUBLE

#ifdef __CUDACC__
#ifdef USE_DOUBLE
// the following is only in CUDA 3.0
typedef float4 real4_t;
#else
typedef float4 real4_t;
#endif // USE_DOUBLE
#endif // __CUDACC__

#endif /* COMMON_TYPES_H_ */
