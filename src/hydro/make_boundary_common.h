/**
 * \file make_boundary_common.h
 * \brief Some constant parameter used for CUDA kernels geometry.
 *
 * We put these parameters in a separate header so that it can be safely included
 * in multiple file (no need to have the full CUDA kernel definition included several
 * times).
 *
 * \date 2 Feb 2012
 * \author Pierre Kestener
 *
 * $Id: make_boundary_common.h 1784 2012-02-21 10:34:58Z pkestene $
 *
 */
#ifndef MAKE_BOUNDARY_COMMON_H_
#define MAKE_BOUNDARY_COMMON_H_

/** only usefull for GPU implementation */
#define MK_BOUND_BLOCK_SIZE             128
#define MK_BOUND_BLOCK_SIZE_3D           16
#define MAKE_JET_BLOCK_SIZE             128
#define MAKE_JET_BLOCK_SIZE_3D           16

#endif // MAKE_BOUNDARY_COMMON_H_
