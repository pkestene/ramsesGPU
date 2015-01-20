/**
 * \file make_boundary_base.h
 * \brief Provides several routines of boundary conditions
 * computations on both CPU and GPU (via CUDA kernels) versions of the code.
 *
 * \note The primary author of this file is F. Chateau. 
 * P. Kestener provided modifications to handle the 3d case and also
 * periodic BC. Further developments (relaxing TVD scheme) were required
 * to handle 3 ghost cells at the border (This is implement in
 * February 2011).
 *
 * \note Add a new and simpler version of border conditions routines (make_boundary2). PK.
 *
 * \author F. Chateau and P. Kestener
 *
 * $Id: make_boundary_base.h 3450 2014-06-16 22:03:23Z pkestene $
 */
#ifndef MAKE_BOUNDARY_BASE_H_
#define MAKE_BOUNDARY_BASE_H_

#include "real_type.h"
#include "common_types.h"
#include "constants.h"
#include "gpu_macros.h"
//#include <iostream>

#include "make_boundary_common.h" // for macro  MK_BOUND_BLOCK_SIZE, etc ...

/**
 * This function ensures that the sign inversion is done if necessary when
 * copying a inner cell to a boundary cell.
 * Its specializations handle each particular case where the sign needs to be
 * inverted.
 * Like all device functions it is inlined into the caller function, so there
 * should be no run-time overhead.
 */
template<BoundaryConditionType bct, BoundaryLocation boundaryLoc, int var>
__DEVICE__ void copy_bound_cell(real_t& dest, const real_t& src)
{
  dest = src;
}

template<>
__DEVICE__ void copy_bound_cell<BC_DIRICHLET, XMIN, IU>(real_t& dest, const real_t& src)
{
  dest = -src;
}

template<>
__DEVICE__ void copy_bound_cell<BC_DIRICHLET, XMAX, IU>(real_t& dest, const real_t& src)
{
  dest = -src;
}

template<>
__DEVICE__ void copy_bound_cell<BC_DIRICHLET, YMIN, IV>(real_t& dest, const real_t& src)
{
  dest = -src;
}

template<>
__DEVICE__ void copy_bound_cell<BC_DIRICHLET, YMAX, IV>(real_t& dest, const real_t& src)
{
  dest = -src;
}

template<>
__DEVICE__ void copy_bound_cell<BC_DIRICHLET, ZMIN, IW>(real_t& dest, const real_t& src)
{
  dest = -src;
}

template<>
__DEVICE__ void copy_bound_cell<BC_DIRICHLET, ZMAX, IW>(real_t& dest, const real_t& src)
{
  dest = -src;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/**
 * \brief This function computes the offset of the first cell to access 
 * according to the boundary.
 * It is expanded at compile-time to generate a different function for each
 * boundaries.
 * \param[in] pitch : actual memory size along x direction
 * \param[in] imax  : physical size along x direction (ghost border included)
 * \param[in] jmax  : physical size along y direction (ghost border included)
 * \param[in] kmax  : physical size along z direction (ghost border included)
 * \param[in] arraySize : equal to pitch() * _dim.y * _dim.z (see method section() of Host/DeviceArray)
 * \param[in] m : can be either i, j or k depending on border location
 * \param[in] n : can be either i, j or k depending on border location
 * \param[in] var : can be either ID, IP, IU, IV or IW
 *
 * \return offset inside U array
 */
template<BoundaryLocation boundaryLoc>
__DEVICE__ int computeOffset(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var);

/**
 * This function computes the offset increment between 2 consecutive cells that
 * must be accessed, according to the boundary.
 * It is expanded at compile-time to generate a different function for each
 * boundary location.
 *
 * \param[in,out] offset
 */
template<BoundaryLocation boundaryLoc>
__DEVICE__ void incOffset(int& offset, int pitch, int jmax);

/////////////////////////////////////////
// XMIN bound (i=0)
template<>
__DEVICE__ int computeOffset<XMIN>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var)
{
  (void) imax;
  (void) kmax;

  return var * arraySize +       // array: var
    m * pitch +                  // j: m
    n * pitch * jmax;            // k: n
}

template<>
__DEVICE__ void incOffset<XMIN>(int& offset, int pitch, int jmax)
{
  (void) pitch;
  (void) jmax;
  ++offset;	// jump to the next column (i++)
}

/////////////////////////////////////////
// XMAX bound (i=imax-1)
template<>
__DEVICE__ int computeOffset<XMAX>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var)
{
  (void) kmax;
  return var * arraySize +
    m * pitch +		// j: m
    n * pitch * jmax +  // k: n
    imax-1; 		// i: imax-1
}

template<>
__DEVICE__ void incOffset<XMAX>(int& offset, int pitch, int jmax)
{
  (void) pitch;
  (void) jmax;
  --offset;	// jump to the previous element (i--)
}

/////////////////////////////////////////
// YMIN bound (j=0)
template<>
__DEVICE__ int computeOffset<YMIN>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var)
{
  (void) imax;
  (void) kmax;
  return var * arraySize + m + pitch*jmax*n; // array: var, j: 0, i: m, k: n
}

template<>
__DEVICE__ void incOffset<YMIN>(int& offset, int pitch, int jmax)
{
  (void) jmax;
  offset += pitch; // j++
}

/////////////////////////////////////////
// YMAX bound (j=jmax-1)
template<>
__DEVICE__ int computeOffset<YMAX>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var)
{
  (void) imax;
  (void) kmax;
  return var * arraySize +	// array: var
    pitch*jmax*n +              // k: n
    pitch*jmax - pitch +	// j: dimy-1
    m;				// i: m
}

template<>
__DEVICE__ void incOffset<YMAX>(int& offset, int pitch, int jmax)
{
  (void) jmax;
  offset -= pitch;	// j--
}

/////////////////////////////////////////
// ZMIN bound (k=0)
template<>
__DEVICE__ int computeOffset<ZMIN>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var)
{
  (void) imax;
  (void) jmax;
  (void) kmax;
  return var * arraySize +       // array: var
    m  +                         // i: m
    n * pitch;                   // j: n
}

template<>
__DEVICE__ void incOffset<ZMIN>(int& offset, int pitch, int jmax)
{
  offset += (pitch*jmax);	// jump to the next column (k++)
}

/////////////////////////////////////////
// ZMAX bound (k=kmax-1)
template<>
__DEVICE__ int computeOffset<ZMAX>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var)
{
  (void) imax; 
  return var * arraySize +       // array: var
    pitch * jmax * (kmax-1) +    // k: kmax-1
    m  +                         // i: m
    n * pitch;                   // j: n
}

template<>
__DEVICE__ void incOffset<ZMAX>(int& offset, int pitch, int jmax)
{
  offset -= (pitch*jmax);	// jump to the next column (k--)
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/**
 * This function computes the 4 offsets (2 times 2 ghosts cells) use to set
 * periodic boundary conditions. 
 *
 * It is expanded at compile-time to generate a different function for each
 * boundaries.
 * \param pitch : actual memory size along x direction
 * \param imax  : physical size along x direction (ghost border included)
 * \param jmax  : physical size along y direction (ghost border included)
 * \param kmax  : physical size along z direction (ghost border included)
 * \param arraysize : equal to pitch() * _dim.y * _dim.z (see method section() of Host/DeviceArray)
 * \param m : can be either i, j or k depending on border location
 * \param n : can be either i, j or k depending on border location
 * \param var : can be either ID, IP, IU, IV or IW
 * \param offset0 : output absolute offset of a ghost boundary cell
 * \param offset1 : output absolute offset of the neighboor ghost boundary cell
 * \param offset2 : output absolute offset of the periodic image of cell at offset0
 * \param offset3 : output absolute offset of the periodic image of cell at offset1
 */
template<BoundaryLocation boundaryLoc>
__DEVICE__ void computeOffsetPeriodic(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3);


// XMIN bound (i=0)
template<>
__DEVICE__ void computeOffsetPeriodic<XMIN>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3)
{
  (void) kmax;
  offset0 = var * arraySize + m * pitch + n * pitch * jmax;
  offset1 = offset0 + 1;
  offset2 = offset0 + (imax-4);
  offset3 = offset1 + (imax-4);
}

// XMAX bound (i=imax-1)
template<>
__DEVICE__ void computeOffsetPeriodic<XMAX>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3)
{
  (void) kmax;
  offset0 = var * arraySize + m * pitch + n * pitch * jmax + imax-1;
  offset1 = offset0 - 1;
  offset2 = offset0 - (imax-4);
  offset3 = offset1 - (imax-4);
}

// YMIN bound (j=0)
template<>
__DEVICE__ void computeOffsetPeriodic<YMIN>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3)
{
  (void) imax;
  (void) kmax;
  offset0 = var * arraySize + m + pitch*jmax*n;
  offset1 = offset0 + pitch;
  offset2 = offset0 + (jmax-4)*pitch;
  offset3 = offset1 + (jmax-4)*pitch;
}

// YMAX bound (j=jmax-1)
template<>
__DEVICE__ void computeOffsetPeriodic<YMAX>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3)
{
  (void) imax;
  (void) kmax;
  offset0 = var * arraySize + pitch*jmax*n + pitch*jmax - pitch + m;
  offset1 = offset0 - pitch;
  offset2 = offset0 - (jmax-4)*pitch;
  offset3 = offset1 - (jmax-4)*pitch;
}

// ZMIN bound (k=0)
template<>
__DEVICE__ void computeOffsetPeriodic<ZMIN>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3)
{
  (void) imax;
  offset0 = var * arraySize + m + n * pitch;
  offset1 = offset0 + pitch*jmax;
  offset2 = offset0 + (kmax-4)*pitch*jmax;
  offset3 = offset1 + (kmax-4)*pitch*jmax;
}

// ZMAX bound (k=kmax-1)
template<>
__DEVICE__ void computeOffsetPeriodic<ZMAX>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3)
{
  (void) imax;
  offset0 = var * arraySize + m + n * pitch + (kmax-1) * pitch * jmax;
  offset1 = offset0 - pitch*jmax;
  offset2 = offset0 - (kmax-4)*pitch*jmax;
  offset3 = offset1 - (kmax-4)*pitch*jmax;
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/**
 * This function computes the 6 offsets (2 times 3 ghosts cells) use to set
 * periodic boundary conditions. 
 *
 * It is expanded at compile-time to generate a different function for each
 * boundaries.
 * \param pitch : actual memory size along x direction
 * \param imax  : physical size along x direction (ghost border included)
 * \param jmax  : physical size along y direction (ghost border included)
 * \param kmax  : physical size along z direction (ghost border included)
 * \param arraysize : equal to pitch() * _dim.y * _dim.z (see method section() of Host/DeviceArray)
 * \param m : can be either i, j or k depending on border location
 * \param n : can be either i, j or k depending on border location
 * \param var : can be either ID, IP, IU, IV or IW
 * \param offset0 : output absolute offset of a ghost boundary cell
 * \param offset1 : output absolute offset of the neighboor ghost boundary cell
 * \param offset2 : output absolute offset of the neighboor ghost boundary cell
 * \param offset3 : output absolute offset of the periodic image of cell at offset0
 * \param offset4 : output absolute offset of the periodic image of cell at offset1
 * \param offset5 : output absolute offset of the periodic image of cell at offset2
 */
template<BoundaryLocation boundaryLoc>
__DEVICE__ void computeOffsetPeriodic(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3, int& offset4, int& offset5);

// XMIN bound (i=0)
template<>
__DEVICE__ void computeOffsetPeriodic<XMIN>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3, int& offset4, int& offset5)
{
  (void) kmax;
  offset0 = var * arraySize + m * pitch + n * pitch * jmax;
  offset1 = offset0 + 1;
  offset2 = offset0 + 2;
  offset3 = offset0 + (imax-6);
  offset4 = offset1 + (imax-6);
  offset5 = offset2 + (imax-6);
}

// XMAX bound (i=imax-1)
template<>
__DEVICE__ void computeOffsetPeriodic<XMAX>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3, int& offset4, int& offset5)
{
  (void) kmax;
  offset0 = var * arraySize + m * pitch + n * pitch * jmax + imax-1;
  offset1 = offset0 - 1;
  offset2 = offset0 - 2;
  offset3 = offset0 - (imax-6);
  offset4 = offset1 - (imax-6);
  offset5 = offset2 - (imax-6);
}

// YMIN bound (j=0)
template<>
__DEVICE__ void computeOffsetPeriodic<YMIN>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3, int& offset4, int& offset5)
{
  (void) imax;
  (void) kmax;
  offset0 = var * arraySize + m + pitch*jmax*n;
  offset1 = offset0 + pitch;
  offset2 = offset0 + pitch*2;
  offset3 = offset0 + (jmax-6)*pitch;
  offset4 = offset1 + (jmax-6)*pitch;
  offset5 = offset2 + (jmax-6)*pitch;
}

// YMAX bound (j=jmax-1)
template<>
__DEVICE__ void computeOffsetPeriodic<YMAX>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3, int& offset4, int& offset5)
{
  (void) imax;
  (void) kmax;
  offset0 = var * arraySize + pitch*jmax*n + pitch*jmax - pitch + m;
  offset1 = offset0 - pitch;
  offset2 = offset0 - pitch*2;
  offset3 = offset0 - (jmax-6)*pitch;
  offset4 = offset1 - (jmax-6)*pitch;
  offset5 = offset2 - (jmax-6)*pitch;
}

// ZMIN bound (k=0)
template<>
__DEVICE__ void computeOffsetPeriodic<ZMIN>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3, int& offset4, int& offset5)
{
  (void) imax;
  offset0 = var * arraySize + m + n * pitch;
  offset1 = offset0 + pitch*jmax;
  offset2 = offset0 + pitch*jmax*2;
  offset3 = offset0 + (kmax-6)*pitch*jmax;
  offset4 = offset1 + (kmax-6)*pitch*jmax;
  offset5 = offset2 + (kmax-6)*pitch*jmax;
}

// ZMAX bound (k=kmax-1)
template<>
__DEVICE__ void computeOffsetPeriodic<ZMAX>(int pitch, int imax, int jmax, int kmax, int arraySize, int m, int n, int var, int& offset0, int& offset1, int& offset2, int& offset3, int& offset4, int& offset5)
{
  (void) imax;
  offset0 = var * arraySize + m + n * pitch + (kmax-1) * pitch * jmax;
  offset1 = offset0 - pitch*jmax;
  offset2 = offset0 - pitch*jmax*2;
  offset3 = offset0 - (kmax-6)*pitch*jmax;
  offset4 = offset1 - (kmax-6)*pitch*jmax;
  offset5 = offset2 - (kmax-6)*pitch*jmax;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/**
 * \brief This is the main routine that actually performs the border cells 
 * update.
 * 
 * This function loads the boundary variables at the specified indexes: m, n
 * according to the boundary condition type.
 */
template<BoundaryConditionType bct, BoundaryLocation boundaryLoc, int var>
__DEVICE__ void load_bound_var(real_t* U, int pitch, int dimx, int dimy, int dimz, int arraySize, int m, int n, int nGhosts=2)
{
  int offset = computeOffset<boundaryLoc>(pitch, dimx, dimy, dimz, arraySize, m ,n, var);

  if (nGhosts == 2) { // Hydro Godunov or Hydro Kurganov-Tadmor
    
    if(bct == BC_DIRICHLET)
      {
	real_t& cell0 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t& cell1 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t  cell2 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t  cell3 = U[offset];
	
	copy_bound_cell<bct, boundaryLoc, var>(cell0, cell3);
	copy_bound_cell<bct, boundaryLoc, var>(cell1, cell2);
      }
    else if(bct == BC_NEUMANN)
      {
	real_t& cell0 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t& cell1 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t  cell2 = U[offset];
	copy_bound_cell<bct, boundaryLoc, var>(cell0, cell2);
	copy_bound_cell<bct, boundaryLoc, var>(cell1, cell2);
      }
    else if(bct == BC_PERIODIC)
      {
	int offset1, offset2, offset3;
	computeOffsetPeriodic<boundaryLoc>(pitch, dimx, dimy, dimz, arraySize, m, n, var, offset, offset1, offset2, offset3);
	real_t& cell0 = U[offset]; 
	real_t& cell1 = U[offset1]; 
	real_t  cell2 = U[offset2];
	real_t  cell3 = U[offset3];
	
	copy_bound_cell<bct, boundaryLoc, var>(cell0, cell2);
	copy_bound_cell<bct, boundaryLoc, var>(cell1, cell3);
      }
 
  } else if (nGhosts == 3) { // Hydro relaxing TVD or MHD Godunov
  
    if(bct == BC_DIRICHLET)
      {
	real_t& cell0 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t& cell1 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t& cell2 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t  cell3 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t  cell4 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t  cell5 = U[offset];
	
	copy_bound_cell<bct, boundaryLoc, var>(cell0, cell5);
	copy_bound_cell<bct, boundaryLoc, var>(cell1, cell4);
	copy_bound_cell<bct, boundaryLoc, var>(cell2, cell3);
      }
    else if(bct == BC_NEUMANN)
      {
	real_t& cell0 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t& cell1 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t& cell2 = U[offset]; incOffset<boundaryLoc>(offset, pitch, dimy);
	real_t  cell3 = U[offset];
	copy_bound_cell<bct, boundaryLoc, var>(cell0, cell3);
	copy_bound_cell<bct, boundaryLoc, var>(cell1, cell3);
	copy_bound_cell<bct, boundaryLoc, var>(cell2, cell3);
      }
    else if(bct == BC_PERIODIC)
      {
	int offset1, offset2, offset3, offset4, offset5;
	computeOffsetPeriodic<boundaryLoc>(pitch, dimx, dimy, dimz, arraySize, m, n, var, offset, offset1, offset2, offset3, offset4, offset5);
	real_t& cell0 = U[offset]; 
	real_t& cell1 = U[offset1]; 
	real_t& cell2 = U[offset2];
	real_t  cell3 = U[offset3];
	real_t  cell4 = U[offset4];
	real_t  cell5 = U[offset5];
	
	copy_bound_cell<bct, boundaryLoc, var>(cell0, cell3);
	copy_bound_cell<bct, boundaryLoc, var>(cell1, cell4);
	copy_bound_cell<bct, boundaryLoc, var>(cell2, cell5);
      }

  } // end nGhosts==3

} // load_bound_var

/**
 * \brief Fills a boundary of the grid.
 * 
 * \tparam bct The boundary condition type (Dirichlet, Neumann or periodic)
 * \tparam boundary loc The boundary location (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX) 
 * \param U float pointer to memory array (hydro fields)
 * \param[in] pitch Memory pitch.
 * \param[in] imax physical size along X (ghost cells included)
 * \param[in] jmax physical size along Y (ghost cells included)
 * \param[in] kmax physical size along Z (ghost cells included)
 * \param[in] arraySize physical size allocated to 1 field (ghost cells inc).
 * \param[in] nGhosts number of ghost cells (2 and 3 are supported)
 * \param[in] mhdEnabled boolean which trigger border computation for
 * magnetic field components.
 *
 * bct and boundaryLoc are template parameters because it allows to 
 * automatically generate a highly optimized version of this function for 
 * each combination of the parameters.
 * 
 * Creating a single function handling all these cases would be highly
 * inefficient on GPU hardware.
 * And the last option: creating all these optimized function by hand would
 * require large amounts of copy&paste and redundant code, which would result
 * in bigger and more difficult to maintain code.
 */
template<BoundaryConditionType bct, BoundaryLocation boundaryLoc>
__GLOBAL__ void make_boundary(real_t* U, int pitch, int imax, int jmax, int kmax, int arraySize, int nGhosts=2, bool mhdEnabled=false)
#ifdef __CUDACC__
{

  // for a 2D problem, we only use a 1D grid kernel
  if (kmax==1) /* 2D */
    {
      /* 2D : only bx and tx are used */  
      const int bx = blockIdx.x;
      const int tx = threadIdx.x;
      const int m  = bx * MK_BOUND_BLOCK_SIZE + tx;

      if(m >= 0 and
	 (((boundaryLoc == YMIN or boundaryLoc == YMAX) and m < imax) or
	  ((boundaryLoc == XMIN or boundaryLoc == XMAX) and m < jmax)))
	{
	  load_bound_var<bct, boundaryLoc, ID>(U, pitch, imax, jmax, kmax, arraySize, m, 0, nGhosts);
	  load_bound_var<bct, boundaryLoc, IP>(U, pitch, imax, jmax, kmax, arraySize, m, 0, nGhosts);
	  load_bound_var<bct, boundaryLoc, IU>(U, pitch, imax, jmax, kmax, arraySize, m, 0, nGhosts);
	  load_bound_var<bct, boundaryLoc, IV>(U, pitch, imax, jmax, kmax, arraySize, m, 0, nGhosts);
	  if (mhdEnabled) {
	    load_bound_var<bct, boundaryLoc, IW>(U, pitch, imax, jmax, kmax, arraySize, m, 0, nGhosts);
	    load_bound_var<bct, boundaryLoc, IA>(U, pitch, imax, jmax, kmax, arraySize, m, 0, nGhosts);
	    load_bound_var<bct, boundaryLoc, IB>(U, pitch, imax, jmax, kmax, arraySize, m, 0, nGhosts);
	    load_bound_var<bct, boundaryLoc, IC>(U, pitch, imax, jmax, kmax, arraySize, m, 0, nGhosts);
	  }
	}
    }
  else /* 3D */
    {
      /* for a 3D problem, we use a 2D grid kernel */
      const int bx = blockIdx.x;
      const int tx = threadIdx.x;
      const int m  = bx * MK_BOUND_BLOCK_SIZE_3D + tx;

      const int by = blockIdx.y;
      const int ty = threadIdx.y;
      const int n  = by * MK_BOUND_BLOCK_SIZE_3D + ty;


      /* these 3 boolean test if current thread is addressing a cell
	 located inside the X (resp. Y or Z) face boundary. */
      bool condX, condY, condZ;
      
      condX = 
	(boundaryLoc == XMIN or boundaryLoc == XMAX) and 
	( (m >= 0   ) and (n >= 0   ) ) and
	( (m <  jmax) and (n <  kmax) );
      condY = 
	(boundaryLoc == YMIN or boundaryLoc == YMAX) and 
	( (m >= 0   ) and (n >= 0   ) ) and
	( (m <  imax) and (n <  kmax) );
      condZ = 
	(boundaryLoc == ZMIN or boundaryLoc == ZMAX) and 
	( (m >= 0   ) and (n >= 0   ) ) and
	( (m <  imax) and (n <  jmax) );
      if (condX or condY or condZ) {
	  load_bound_var<bct, boundaryLoc, ID>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	  load_bound_var<bct, boundaryLoc, IP>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	  load_bound_var<bct, boundaryLoc, IU>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	  load_bound_var<bct, boundaryLoc, IV>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);	
	  load_bound_var<bct, boundaryLoc, IW>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	  if (mhdEnabled) {
	    load_bound_var<bct, boundaryLoc, IA>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	    load_bound_var<bct, boundaryLoc, IB>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	    load_bound_var<bct, boundaryLoc, IC>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	  }
      }

	
    }
}
#else // CPU VERSION
{
  if (kmax==1) /* 2D */
    {
      /** IndexMax is set to either imax or jmax */
      int IndexMax = imax; 
      if (boundaryLoc == YMIN or boundaryLoc == YMAX)
	IndexMax = imax;
      if (boundaryLoc == XMIN or boundaryLoc == XMAX)
	IndexMax = jmax;
      
      for (int i=0; i<IndexMax; i++)
	{
	  load_bound_var<bct, boundaryLoc, ID>(U, pitch, imax, jmax, kmax, arraySize, i, 0, nGhosts);
	  load_bound_var<bct, boundaryLoc, IP>(U, pitch, imax, jmax, kmax, arraySize, i, 0, nGhosts);
	  load_bound_var<bct, boundaryLoc, IU>(U, pitch, imax, jmax, kmax, arraySize, i, 0, nGhosts);
	  load_bound_var<bct, boundaryLoc, IV>(U, pitch, imax, jmax, kmax, arraySize, i, 0, nGhosts);
	  if (mhdEnabled) {
	    load_bound_var<bct, boundaryLoc, IW>(U, pitch, imax, jmax, kmax, arraySize, i, 0, nGhosts);
	    load_bound_var<bct, boundaryLoc, IA>(U, pitch, imax, jmax, kmax, arraySize, i, 0, nGhosts);
	    load_bound_var<bct, boundaryLoc, IB>(U, pitch, imax, jmax, kmax, arraySize, i, 0, nGhosts);
	    load_bound_var<bct, boundaryLoc, IC>(U, pitch, imax, jmax, kmax, arraySize, i, 0, nGhosts);
	  }
	}
    } /* end 2D */
  else /* 3D */ 
    {
      int mMax, nMax;
      if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
	mMax = jmax;
	nMax = kmax;
      }
      if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
	mMax = imax;
	nMax = kmax;
      }
      if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {
	mMax = imax;
	nMax = jmax;
      }

      for (int n=0; n<nMax; n++)
	for (int m=0; m<mMax; m++)
	  {
	    load_bound_var<bct, boundaryLoc, ID>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	    load_bound_var<bct, boundaryLoc, IP>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	    load_bound_var<bct, boundaryLoc, IU>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	    load_bound_var<bct, boundaryLoc, IV>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	    load_bound_var<bct, boundaryLoc, IW>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	    if (mhdEnabled) {
	      load_bound_var<bct, boundaryLoc, IA>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	      load_bound_var<bct, boundaryLoc, IB>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	      load_bound_var<bct, boundaryLoc, IC>(U, pitch, imax, jmax, kmax, arraySize, m, n, nGhosts);
	    }
	  }
      
    } /* end 3D */

} // void make_boundary

#endif // __CUDACC__

/**
 * \brief Fills a boundary of the grid (same as make_boundary but stand-alone).
 * 
 * \tparam bct The boundary condition type (Dirichlet, Neumann or periodic)
 * \tparam boundary loc The boundary location (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX) 
 * \param U float pointer to memory array (hydro fields)
 * \param[in] pitch Memory pitch.
 * \param[in] imax physical size along X (ghost cells included)
 * \param[in] jmax physical size along Y (ghost cells included)
 * \param[in] kmax physical size along Z (ghost cells included)
 * \param[in] arraySize physical size allocated to 1 field (ghost cells inc).
 * \param[in] nGhosts number of ghost cells (2 and 3 are supported)
 * \param[in] mhdEnabled boolean which trigger border computation for
 * magnetic field components.
 *
 * bct and boundaryLoc are template parameters because it allows to 
 * automatically generate a highly optimized version of this function for 
 * each combination of the parameters.
 * 
 * Creating a single function handling all these cases would be highly
 * inefficient on GPU hardware.
 * And the last option: creating all these optimized function by hand would
 * require large amounts of copy&paste and redundant code, which would result
 * in bigger and more difficult to maintain code.
 */
template<BoundaryConditionType bct, BoundaryLocation boundaryLoc>
__GLOBAL__ void make_boundary2(real_t* U, int pitch, int imax, int jmax, int kmax, int arraySize, int nGhosts=2, bool mhdEnabled=false)
#ifdef __CUDACC__
{
  int& nbVar = ::gParams.nbVar;
  real_t sign = ONE_F;
  int nx = imax - 2*nGhosts;
  int ny = jmax - 2*nGhosts;
  int nz = kmax - 2*nGhosts;

  (void) mhdEnabled; // avoid compiler warning

  // for a 2D problem, we only use a 1D grid kernel
  if (kmax==1) /* 2D */
    {
      /* 2D : only bx and tx are used */  
      const int bx = blockIdx.x;
      const int tx = threadIdx.x;
      const int index  = bx * MK_BOUND_BLOCK_SIZE + tx;

      /*
       * XMIN
       */
      if (boundaryLoc == XMIN) {
	int i0;
	
	if (index < jmax) {
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int iGhost=0; iGhost<nGhosts; iGhost++) {
	      
	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		i0 = 2*nGhosts-1-iGhost;
		if (iVar==IU) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		i0 = nGhosts;
	      } else { // periodic
		i0 = nx+iGhost;
	      }
	      
	      int offset_in  = iVar*arraySize + i0     + index*pitch;
	      int offset_out = iVar*arraySize + iGhost + index*pitch;
	      U[offset_out]=U[offset_in]*sign;
	      
	    } // end for iGhosts
	  } // end for iVar
	} // end if index
      } // end XMIN
      
      /*
       * XMAX
       */
      if (boundaryLoc == XMAX) {
	int i0;

	if (index < jmax) {
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int iGhost=nx+nGhosts; iGhost<nx+2*nGhosts; iGhost++) {

	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		i0 = 2*nx+2*nGhosts-1-iGhost;
		if (iVar==IU) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		i0 = nx+nGhosts-1;
	      } else { // periodic
		i0 = iGhost-nx;
	      }

	      int offset_in  = iVar*arraySize + i0     + index*pitch;
	      int offset_out = iVar*arraySize + iGhost + index*pitch;
	      U[offset_out] = U[offset_in]*sign;
	      
	    } // end for iGhost
	  } // end for iVar
	} // end if index
      } // end XMAX

      /*
       * YMIN
       */
      if (boundaryLoc == YMIN) {
	int j0;

	if (index < imax) {
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int jGhost=0; jGhost<nGhosts; jGhost++) {
	      
	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		j0 = 2*nGhosts-1-jGhost;
		if (iVar==IV) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		j0 = nGhosts;
	      } else { // periodic
		j0 = ny+jGhost;
	      }

	      int offset_in  = iVar*arraySize + index + j0    *pitch;
	      int offset_out = iVar*arraySize + index + jGhost*pitch;
	      U[offset_out]=U[offset_in]*sign;

	    } // end for jGhost
	  } // end for iVar
	} // end if index
      } // end YMIN

      /*
       * YMAX
       */
      if (boundaryLoc == YMAX) {
	int j0;

	if (index < imax) {
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int jGhost=ny+nGhosts; jGhost<ny+2*nGhosts; jGhost++) {
	      
	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		j0 = 2*ny+2*nGhosts-1-jGhost;
		if (iVar==IV) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		j0 = ny+nGhosts-1;
	      } else { // periodic
		j0 = jGhost-ny;
	      }
	      
	      int offset_in  = iVar*arraySize + index + j0    *pitch;
	      int offset_out = iVar*arraySize + index + jGhost*pitch;
	      U[offset_out] = U[offset_in]*sign;
	      
	    } // end for jGhost
	  } // end for iVar
	} // end if index
      } // end YMAX

    } // end 2D
  else /* 3D */
    {
       /* for a 3D problem, we use a 2D grid kernel */
      const int bx = blockIdx.x;
      const int tx = threadIdx.x;
      const int m  = bx * MK_BOUND_BLOCK_SIZE_3D + tx;

      const int by = blockIdx.y;
      const int ty = threadIdx.y;
      const int n  = by * MK_BOUND_BLOCK_SIZE_3D + ty;
      
      /*
       * XMIN
       */
      if (boundaryLoc == XMIN) { // m is j, n is k
      	int i0;
	const int &j=m;
	const int &k=n;
	
	if (j<jmax and k<kmax) {
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int  iGhost=0; iGhost< nGhosts; iGhost++) {
	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		i0 = 2*nGhosts-1-iGhost;
		if (iVar==IU) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		i0 = nGhosts;
	      } else { // periodic
		i0 = nx+iGhost;
	      }
	    
	      int offset_in  = iVar*arraySize + i0     + j*pitch + k*pitch*jmax;
	      int offset_out = iVar*arraySize + iGhost + j*pitch + k*pitch*jmax;
	      U[offset_out] = U[offset_in]*sign;

	    } // end for iGhost
	  } // end for iVar
	} // end if j and k
      } // end XMIN

      /*
       * XMAX
       */
      if (boundaryLoc == XMAX) { // m is j, n is k
	int i0;
	const int &j=m;
	const int &k=n;

	if (j<jmax and k<kmax) {
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int iGhost=nx+nGhosts; iGhost<nx+2*nGhosts; iGhost++) {
	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		i0 = 2*nx+2*nGhosts-1-iGhost;
		if (iVar==IU) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		i0 = nx+nGhosts-1;
	      } else { // periodic
		i0 = iGhost-nx;
	      }
	      
	      int offset_in  = iVar*arraySize + i0     + j*pitch + k*pitch*jmax;
	      int offset_out = iVar*arraySize + iGhost + j*pitch + k*pitch*jmax;
	      U[offset_out] = U[offset_in]*sign;
	    } // end for iGhost
	  } // end for iVar
	} // end if j and k
      } // end XMAX

      /*
       * YMIN
       */
      if (boundaryLoc == YMIN) { // m is i, n is k
      	int j0;
	const int &i=m;
	const int &k=n;
	
	if (i<imax and k<kmax) {
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int jGhost=0; jGhost<nGhosts; jGhost++) {
	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		j0 = 2*nGhosts-1-jGhost;
		if (iVar==IV) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		j0 = nGhosts;
	      } else { // periodic
		j0 = ny+jGhost;
	      }
	      int offset_in  = iVar*arraySize + i + j0    *pitch + k*pitch*jmax;
	      int offset_out = iVar*arraySize + i + jGhost*pitch + k*pitch*jmax;
	      U[offset_out] = U[offset_in]*sign;
	    } // end for jGhost
	  } // end for iVar
	} // end if i and k
      } // end YMIN

      /*
       * YMAX
       */
      if (boundaryLoc == YMAX) { // m is i, n is k
      	int j0;
	const int &i=m;
	const int &k=n;
	
	if (i<imax and k<kmax) {	
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int jGhost=ny+nGhosts; jGhost<ny+2*nGhosts; jGhost++) {
	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		j0 = 2*ny+2*nGhosts-1-jGhost;
		if (iVar==IV) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		j0 = ny+nGhosts-1;
	      } else { // periodic
		j0 = jGhost-ny;
	      }
	      int offset_in  = iVar*arraySize + i + j0    *pitch + k*pitch*jmax;
	      int offset_out = iVar*arraySize + i + jGhost*pitch + k*pitch*jmax;
	      U[offset_out] = U[offset_in]*sign;
	    } // end for jGhost
	  } // end for iVar
	} // end if i and k
      } // end YMAX

      /*
       * ZMIN
       */
      if (boundaryLoc == ZMIN) { // m is i, n is j
      	int k0;
	const int &i=m;
	const int &j=n;
	
	if (i<imax and j<jmax) {
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int kGhost=0; kGhost<nGhosts; kGhost++) {
	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		k0 = 2*nGhosts-1-kGhost;
		if (iVar==IW) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		k0 = nGhosts;
	      } else { // periodic
		k0 = nz+kGhost;
	      }
	      int offset_in  = iVar*arraySize + i + j*pitch + k0    *pitch*jmax;
	      int offset_out = iVar*arraySize + i + j*pitch + kGhost*pitch*jmax;
	      U[offset_out] = U[offset_in]*sign;
	    } // end for kGhost
	  } // end for iVar
	} // end if i and j
      } // end ZMIN

      /*
       * ZMAX
       */
      if (boundaryLoc == ZMAX) {
      	int k0;
	const int &i=m;
	const int &j=n;
	
	if (i<imax and j<jmax) {	
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int kGhost=nz+nGhosts; kGhost<nz+2*nGhosts; kGhost++) {
	      sign = ONE_F;
	      
	      if (bct == BC_DIRICHLET) {
		k0 = 2*nz+2*nGhosts-1-kGhost;
		if (iVar==IW) sign = -ONE_F;
	      } else if (bct == BC_NEUMANN) {
		k0 = nz+nGhosts-1;
	      } else { // periodic
		k0 = kGhost-nz;
	      }
	      int offset_in  = iVar*arraySize + i + j*pitch + k0    *pitch*jmax;
	      int offset_out = iVar*arraySize + i + j*pitch + kGhost*pitch*jmax;
	      U[offset_out] = U[offset_in]*sign;
	    } // end for kGhost
	  } // end for iVar
	} // end if i and j
      } // end ZMAX
      
    } // end 3D
  
} // end make_boundary2 (GPU version)
#else // CPU version
{
  int& nbVar = ::gParams.nbVar;
  real_t sign = ONE_F;
  int nx = imax - 2*nGhosts;
  int ny = jmax - 2*nGhosts;
  int nz = kmax - 2*nGhosts;

  (void) mhdEnabled; // avoid compiler warning

  if (kmax==1) /* 2D */
    {

      if (boundaryLoc == XMIN) {
	int i0;
      
	for (int iVar=0; iVar<nbVar; iVar++) {
	  for (int  iGhost=0; iGhost< nGhosts; iGhost++) {
	    sign = ONE_F;
	    
	    if (bct == BC_DIRICHLET) {
	      i0 = 2*nGhosts-1-iGhost;
	      if (iVar==IU) sign = -ONE_F;
	    } else if (bct == BC_NEUMANN) {
	      i0 = nGhosts;
	    } else { // periodic
	      i0 = nx+iGhost;
	    }
	    for (int j=0; j<jmax; j++) {
	      int offset_in  = iVar*arraySize + i0     + j*pitch;
	      int offset_out = iVar*arraySize + iGhost + j*pitch;
	      U[offset_out] = U[offset_in]*sign;
	    } // end for j
	  } // end for iGhost
	} // end for iVar
      
      } // end XMIN 

      if (boundaryLoc == XMAX) {
	int i0;

	for (int iVar=0; iVar<nbVar; iVar++) {
	  for (int iGhost=nx+nGhosts; iGhost<nx+2*nGhosts; iGhost++) {
	    sign = ONE_F;

	    if (bct == BC_DIRICHLET) {
	      i0 = 2*nx+2*nGhosts-1-iGhost;
	      if (iVar==IU) sign = -ONE_F;
	    } else if (bct == BC_NEUMANN) {
	      i0 = nx+nGhosts-1;
	    } else { // periodic
	      i0 = iGhost-nx;
	    }
	    for (int j=0; j<jmax; j++) {
	      int offset_in  = iVar*arraySize + i0     + j*pitch;
	      int offset_out = iVar*arraySize + iGhost + j*pitch;
	      U[offset_out] = U[offset_in]*sign;
	    } // end for j
	  } // end for i
	} // end for iVar
      } // end XMAX

      if (boundaryLoc == YMIN) {
	int j0;
	
	for (int iVar=0; iVar<nbVar; iVar++) {
	  for (int jGhost=0; jGhost<nGhosts; jGhost++) {
	    sign = ONE_F;

	    if (bct == BC_DIRICHLET) {
	      j0 = 2*nGhosts-1-jGhost;
	      if (iVar==IV) sign = -ONE_F;
	    } else if (bct == BC_NEUMANN) {
	      j0 = nGhosts;
	    } else { // periodic
	      j0 = ny+jGhost;
	    }
	    for (int i=0; i<imax; i++) {	      
	      int offset_in  = iVar*arraySize + i + j0*pitch;
	      int offset_out = iVar*arraySize + i + jGhost*pitch;
	      U[offset_out] = U[offset_in]*sign;
	    } // end for i
	  } // end for jGhost
	} // end for iVar

      } // end YMIN

      if (boundaryLoc == YMAX) {
	int j0;
	
	for (int iVar=0; iVar<nbVar; iVar++) {
	  for (int jGhost=ny+nGhosts; jGhost<ny+2*nGhosts; jGhost++) {
	    sign = ONE_F;

	    if (bct == BC_DIRICHLET) {
	      j0 = 2*ny+2*nGhosts-1-jGhost;
	      if (iVar==IV) sign = -ONE_F;
	    } else if (bct == BC_NEUMANN) {
	      j0 = ny+nGhosts-1;
	    } else { // periodic
	      j0 = jGhost-ny;
	    }
	    for (int i=0; i<imax; i++) {
	      int offset_in  = iVar*arraySize + i + j0*pitch;
	      int offset_out = iVar*arraySize + i + jGhost*pitch;
	      U[offset_out] = U[offset_in]*sign;
	    } // end for i
	  } // end for jGhost
	} // end for iVar
      } // end YMAX

    } // end 2D 

  else // 3D 

    { 

      /*
       * XMIN
       */
      if (boundaryLoc == XMIN) {
      	int i0;
      
      	for (int iVar=0; iVar<nbVar; iVar++) {
      	  for (int  iGhost=0; iGhost< nGhosts; iGhost++) {
      	    sign = ONE_F;
	    
      	    if (bct == BC_DIRICHLET) {
      	      i0 = 2*nGhosts-1-iGhost;
      	      if (iVar==IU) sign = -ONE_F;
      	    } else if (bct == BC_NEUMANN) {
      	      i0 = nGhosts;
      	    } else { // periodic
      	      i0 = iGhost+nx;
      	    }
      	    for (int k=0; k<kmax; k++) {
      	      for (int j=0; j<jmax; j++) {
      		int offset_in  = iVar*arraySize + i0     + j*pitch + k*pitch*jmax;
      		int offset_out = iVar*arraySize + iGhost + j*pitch + k*pitch*jmax;
      		U[offset_out] = U[offset_in]*sign;
      	      } // end for j
      	    } // end for k
      	  } // end for iGhost
      	} // end for iVar      
      } // end XMIN

      /*
       * XMAX
       */
      if (boundaryLoc == XMAX) {
	int i0;

	for (int iVar=0; iVar<nbVar; iVar++) {
      	  for (int iGhost=nx+nGhosts; iGhost<nx+2*nGhosts; iGhost++) {
      	    sign = ONE_F;
	    
      	    if (bct == BC_DIRICHLET) {
      	      i0 = 2*nx+2*nGhosts-1-iGhost;
      	      if (iVar==IU) sign = -ONE_F;
      	    } else if (bct == BC_NEUMANN) {
      	      i0 = nx+nGhosts-1;
      	    } else { // periodic
      	      i0 = iGhost-nx;
      	    }
	    for (int k=0; k<kmax; k++) {
	      for (int j=0; j<jmax; j++) {
		int offset_in  = iVar*arraySize + i0     + j*pitch + k*pitch*jmax;
		int offset_out = iVar*arraySize + iGhost + j*pitch + k*pitch*jmax;
		U[offset_out] = U[offset_in]*sign;
	      } // end for j
	    } // end for k
	  } // end for i
	} // end for iVar
      } // end XMAX

      /*
       * YMIN
       */
      if (boundaryLoc == YMIN) {
      	int j0;

      	for (int iVar=0; iVar<nbVar; iVar++) {
      	  for (int jGhost=0; jGhost<nGhosts; jGhost++) {
      	    sign = ONE_F;

      	    if (bct == BC_DIRICHLET) {
      	      j0 = 2*nGhosts-1-jGhost;
      	      if (iVar==IV) sign = -ONE_F;
      	    } else if (bct == BC_NEUMANN) {
      	      j0 = nGhosts;
      	    } else { // periodic
      	      j0 = jGhost+ny;
      	    }
 	    for (int k=0; k<kmax; k++) {
	      for (int i=0; i<imax; i++) {	      
		int offset_in  = iVar*arraySize + i + j0    *pitch + k*pitch*jmax;
		int offset_out = iVar*arraySize + i + jGhost*pitch + k*pitch*jmax;
		U[offset_out] = U[offset_in]*sign;
	      } // end for i
	    } // end for k
      	  } // end for jGhost
      	} // end for iVar
      } // end YMIN

      /*
       * YMAX
       */
      if (boundaryLoc == YMAX) {
      	int j0;
	
      	for (int iVar=0; iVar<nbVar; iVar++) {
      	  for (int jGhost=ny+nGhosts; jGhost<ny+2*nGhosts; jGhost++) {
      	    sign = ONE_F;
	    
      	    if (bct == BC_DIRICHLET) {
      	      j0 = 2*ny+2*nGhosts-1-jGhost;
      	      if (iVar==IV) sign = -ONE_F;
      	    } else if (bct == BC_NEUMANN) {
      	      j0 = ny+nGhosts-1;
      	    } else { // periodic
      	      j0 = jGhost-ny;
      	    }
       	    for (int k=0; k<kmax; k++) {
	      for (int i=0; i<imax; i++) {
		int offset_in  = iVar*arraySize + i + j0    *pitch + k*pitch*jmax;
		int offset_out = iVar*arraySize + i + jGhost*pitch + k*pitch*jmax;
		U[offset_out] = U[offset_in]*sign;
	      } // end for i
	    } // end for k
      	  } // end for jGhost
      	} // end for iVar
      } // end YMAX

      /*
       * ZMIN
       */
      if (boundaryLoc == ZMIN) {
      	int k0;
	
      	for (int iVar=0; iVar<nbVar; iVar++) {
      	  for (int kGhost=0; kGhost<nGhosts; kGhost++) {
      	    sign = ONE_F;

      	    if (bct == BC_DIRICHLET) {
      	      k0 = 2*nGhosts-1-kGhost;
      	      if (iVar==IW) sign = -ONE_F;
      	    } else if (bct == BC_NEUMANN) {
      	      k0 = nGhosts;
      	    } else { // periodic
      	      k0 = kGhost+nz;
      	    }
 	    for (int j=0; j<jmax; j++) {
	      for (int i=0; i<imax; i++) {	      
		int offset_in  = iVar*arraySize + i + j*pitch + k0    *pitch*jmax;
		int offset_out = iVar*arraySize + i + j*pitch + kGhost*pitch*jmax;
		U[offset_out] = U[offset_in]*sign;
	      } // end for i
	    } // end for j
      	  } // end for kGhost
      	} // end for iVar
      } // end ZMIN

      /*
       * ZMAX
       */
      if (boundaryLoc == ZMAX) {
      	int k0;
	
      	for (int iVar=0; iVar<nbVar; iVar++) {
      	  for (int kGhost=nz+nGhosts; kGhost<nz+2*nGhosts; kGhost++) {
      	    sign = ONE_F;
	    
      	    if (bct == BC_DIRICHLET) {
      	      k0 = 2*nz+2*nGhosts-1-kGhost;
      	      if (iVar==IW) sign = -ONE_F;
      	    } else if (bct == BC_NEUMANN) {
      	      k0 = nz+nGhosts-1;
      	    } else { // periodic
      	      k0 = kGhost-nz;
      	    }
       	    for (int j=0; j<jmax; j++) {
	      for (int i=0; i<imax; i++) {
		int offset_in  = iVar*arraySize + i + j*pitch + k0    *pitch*jmax;
		int offset_out = iVar*arraySize + i + j*pitch + kGhost*pitch*jmax;
		U[offset_out] = U[offset_in]*sign;
	      } // end for i
	    } // end for j
      	  } // end for kGhost
      	} // end for iVar
      } // end ZMAX
    } // end 3D
  
} // end make_boundary2 (CPU version)
#endif // __CUDACC__



#ifndef __CUDACC__
/**
 * \brief Actual CPU routine to fill a Z direction boundary of the grid in the special case of stratified rotating MHD simulation.
 * 
 * \param[in,out] U float pointer to memory array (hydro fields)
 * \param[in] pitch Memory pitch.
 * \param[in] imax physical size along X (ghost cells included)
 * \param[in] jmax physical size along Y (ghost cells included)
 * \param[in] kmax physical size along Z (ghost cells included)
 * \param[in] arraySize physical size allocated to 1 field (ghost cells inc).
 *
 * \tparam boundaryLoc : only ZMIN or ZMAX are valid here
 *
 * NOTE: nghost is assumed to be 3 and mhdEnabled to true
 *
 * IMPORTANT NOTE: This routine is intended to called by the wrapper 
 * make_boundary2_z_stratified.
 *
 */
template<BoundaryLocation boundaryLoc>
void make_boundary2_z_stratified_cpu(real_t* U, 
				     int     pitch, 
				     int     imax, 
				     int     jmax, 
				     int     kmax, 
				     int     arraySize, 
				     bool    floor)
  
{
  const real_t H      =  gParams.cIso / gParams.Omega0;
  real_t &dx    =  gParams.dx;
  real_t &dy    =  gParams.dy;
  real_t &dz    =  gParams.dz;
  real_t &zMin  =  gParams.zMin;
  real_t &zMax  =  gParams.zMax;
  const real_t factor = -dz / 2.0 / H / H;

  real_t ratio_nyp1,ratio_nyp2,ratio_nyp3;

  //const int nGhosts=3;

  const int zSlice = pitch*jmax;

  if (floor) {
    ratio_nyp1 = ONE_F;
    ratio_nyp2 = ONE_F; 
    ratio_nyp3 = ONE_F;
  } else {

    if (boundaryLoc == ZMIN) {
      ratio_nyp1 = exp( factor*(-2*(zMin+HALF_F*dz)+    dz) );
      ratio_nyp2 = exp( factor*(-2*(zMin+HALF_F*dz)+3.0*dz) );
      ratio_nyp3 = exp( factor*(-2*(zMin+HALF_F*dz)+5.0*dz) );
    }

    if (boundaryLoc == ZMAX) {
      ratio_nyp1 = exp( factor*( 2*(zMax-HALF_F*dz)+    dz) );
      ratio_nyp2 = exp( factor*( 2*(zMax-HALF_F*dz)+3.0*dz) );
      ratio_nyp3 = exp( factor*( 2*(zMax-HALF_F*dz)+5.0*dz) );
    }

  } // end floor


  if (boundaryLoc == ZMIN) {

    for (int j=0; j<jmax; j++) {
      for (int i=0; i<imax; i++) {

	// extrapolating density
	int offset0_d  = ID*arraySize + i + j*pitch +  0 * zSlice;
	int offset1_d  = ID*arraySize + i + j*pitch +  1 * zSlice;
	int offset2_d  = ID*arraySize + i + j*pitch +  2 * zSlice;
	int offset3_d  = ID*arraySize + i + j*pitch +  3 * zSlice;

	real_t rho3 = U[offset3_d];
	real_t rho2 = U[offset3_d]*ratio_nyp1;
	real_t rho1 = U[offset3_d]*ratio_nyp1*ratio_nyp2;
	real_t rho0 = U[offset3_d]*ratio_nyp1*ratio_nyp2*ratio_nyp3;
	
	U[offset2_d] = rho2;
	U[offset1_d] = rho1;
	U[offset0_d] = rho0;

	// zero gradient BC for velocity IU
	{
	  int offset0  = IU*arraySize + i + j*pitch +  0 * zSlice;
	  int offset1  = IU*arraySize + i + j*pitch +  1 * zSlice;
	  int offset2  = IU*arraySize + i + j*pitch +  2 * zSlice;
	  int offset3  = IU*arraySize + i + j*pitch +  3 * zSlice;

	  U[offset0] = U[offset3] / rho3 * rho0;
	  U[offset1] = U[offset3] / rho3 * rho1;
	  U[offset2] = U[offset3] / rho3 * rho2;

	} // end IU

	// zero gradient BC for velocity IV
	{
	  int offset0  = IV*arraySize + i + j*pitch +  0 * zSlice;
	  int offset1  = IV*arraySize + i + j*pitch +  1 * zSlice;
	  int offset2  = IV*arraySize + i + j*pitch +  2 * zSlice;
	  int offset3  = IV*arraySize + i + j*pitch +  3 * zSlice;

	  U[offset0] = U[offset3] / rho3 * rho0;
	  U[offset1] = U[offset3] / rho3 * rho1;
	  U[offset2] = U[offset3] / rho3 * rho2;

	} // end IV

	// now normal velocity IW
	{
	  int offset0  = IW*arraySize + i + j*pitch +  0 * zSlice;
	  int offset1  = IW*arraySize + i + j*pitch +  1 * zSlice;
	  int offset2  = IW*arraySize + i + j*pitch +  2 * zSlice;
	  int offset3  = IW*arraySize + i + j*pitch +  3 * zSlice;

	  real_t w = FMIN( U[offset3] ,ZERO_F);

	  U[offset0] = w;
	  U[offset1] = w;
	  U[offset2] = w;

	} // end IW
	
	// vanishing tangential magnetic field (IA,IB)
	{
	  int offset0  = IA*arraySize + i + j*pitch +  0 * zSlice;
	  int offset1  = IA*arraySize + i + j*pitch +  1 * zSlice;
	  int offset2  = IA*arraySize + i + j*pitch +  2 * zSlice;

	  U[offset0]   = ZERO_F;
	  U[offset1]   = ZERO_F;
	  U[offset2]   = ZERO_F;

	  offset0      = IB*arraySize + i + j*pitch +  0 * zSlice;
	  offset1      = IB*arraySize + i + j*pitch +  1 * zSlice;
	  offset2      = IB*arraySize + i + j*pitch +  2 * zSlice;

	  U[offset0]   = ZERO_F;
	  U[offset1]   = ZERO_F;
	  U[offset2]   = ZERO_F;

	} // end IA,IB

      } // end for i
    } // end for j

    
    // Normal magnetic field (IW)
    for (int j=0; j<jmax-1; j++) {
      for (int i=0; i<imax-1; i++) {
	
	real_t dbxdx, dbydy;
	
	int offset = i + j*pitch;
	
	dbxdx = ( U[offset+1    +IA*arraySize+2*zSlice] -
		  U[offset      +IA*arraySize+2*zSlice] ) / dx;

	dbydy = ( U[offset+pitch+IB*arraySize+2*zSlice] -
		  U[offset      +IB*arraySize+2*zSlice] ) / dy;

	real_t  bz  =  U[offset+IC*arraySize+3*zSlice];
	real_t dbz2 = dz*(dbxdx+dbydy);

	U[offset+IC*arraySize+2*zSlice] = bz + dbz2;

	dbxdx = ( U[offset+1    +IA*arraySize+1*zSlice] -
		  U[offset      +IA*arraySize+1*zSlice] ) / dx;

	dbydy = ( U[offset+pitch+IB*arraySize+1*zSlice] -
		  U[offset      +IB*arraySize+1*zSlice] ) / dy;

	real_t dbz1 = dz*(dbxdx+dbydy);
	U[offset+IC*arraySize+1*zSlice] = bz + dbz2 + dbz1;

	dbxdx = ( U[offset+1    +IA*arraySize+0*zSlice] -
		  U[offset      +IA*arraySize+0*zSlice] ) / dx;

	dbydy = ( U[offset+pitch+IB*arraySize+0*zSlice] -
		  U[offset      +IB*arraySize+0*zSlice] ) / dy;

	real_t dbz0 = dz*(dbxdx+dbydy);
	U[offset+IC*arraySize+0*zSlice] = bz + dbz2 + dbz1 + dbz0;

      } // end for i
    } // end for j

    // end normal magnetic field

  } else if (boundaryLoc == ZMAX) {
  
    for (int j=0; j<jmax; j++) {
      for (int i=0; i<imax; i++) {

	// extrapolating density
	int offset1_d  = ID*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	int offset2_d  = ID*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	int offset3_d  = ID*arraySize + i + j*pitch +  (kmax-3) * zSlice;
	int offset4_d  = ID*arraySize + i + j*pitch +  (kmax-4) * zSlice;

	real_t rho4 = U[offset4_d];
	real_t rho3 = U[offset4_d]*ratio_nyp1;
	real_t rho2 = U[offset4_d]*ratio_nyp1*ratio_nyp2;
	real_t rho1 = U[offset4_d]*ratio_nyp1*ratio_nyp2*ratio_nyp3;
	
	U[offset3_d] = rho3;
	U[offset2_d] = rho2;
	U[offset1_d] = rho1;

	// zero gradient BC for velocity IU
	{
	  int offset1  = IU*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	  int offset2  = IU*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	  int offset3  = IU*arraySize + i + j*pitch +  (kmax-3) * zSlice;
	  int offset4  = IU*arraySize + i + j*pitch +  (kmax-4) * zSlice;
	  
	  U[offset3] = U[offset4] / rho4 * rho3;
	  U[offset2] = U[offset4] / rho4 * rho2;
	  U[offset1] = U[offset4] / rho4 * rho1;
	  
	} // end IU

	// zero gradient BC for velocity IV
	{
	  int offset1  = IV*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	  int offset2  = IV*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	  int offset3  = IV*arraySize + i + j*pitch +  (kmax-3) * zSlice;
	  int offset4  = IV*arraySize + i + j*pitch +  (kmax-4) * zSlice;
	  
	  U[offset3] = U[offset4] / rho4 * rho3;
	  U[offset2] = U[offset4] / rho4 * rho2;
	  U[offset1] = U[offset4] / rho4 * rho1;
	  
	} // end IV

	// now normal velocity IW
	{
	  int offset1  = IW*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	  int offset2  = IW*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	  int offset3  = IW*arraySize + i + j*pitch +  (kmax-3) * zSlice;
	  int offset4  = IW*arraySize + i + j*pitch +  (kmax-4) * zSlice;

	  real_t w = FMAX( U[offset4] ,ZERO_F);

	  U[offset3] = w;
	  U[offset2] = w;
	  U[offset1] = w;

	} // end IW

	// vanishing tangential magnetic field (IA,IB)
	{
	  int offset1  = IA*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	  int offset2  = IA*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	  int offset3  = IA*arraySize + i + j*pitch +  (kmax-3) * zSlice;

	  U[offset3]   = ZERO_F;
	  U[offset2]   = ZERO_F;
	  U[offset1]   = ZERO_F;

	  offset1      = IB*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	  offset2      = IB*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	  offset3      = IB*arraySize + i + j*pitch +  (kmax-3) * zSlice;

	  U[offset3]   = ZERO_F;
	  U[offset2]   = ZERO_F;
	  U[offset1]   = ZERO_F;

	} // end IA,IB

      } // end for i
    } // end for j

    // Normal magnetic field (IW)
    for (int j=0; j<jmax-1; j++) {
      for (int i=0; i<imax-1; i++) {
	
	real_t dbxdx, dbydy;
	
	int offset = i + j*pitch;
	
	dbxdx = ( U[offset+1    +IA*arraySize+(kmax-3)*zSlice] -
		  U[offset      +IA*arraySize+(kmax-3)*zSlice] ) / dx;
	
	dbydy = ( U[offset      +IB*arraySize+(kmax-3)*zSlice] -
		  U[offset      +IB*arraySize+(kmax-3)*zSlice] ) / dy;
	
	real_t  bz  =  U[offset+IC*arraySize+(kmax-3)*zSlice];
	
	real_t dbz1 = dz*(dbxdx+dbydy);
	U[offset+IC*arraySize+(kmax-2)*zSlice] = bz - dbz1;
	
	dbxdx = ( U[offset+1    +IA*arraySize+(kmax-2)*zSlice] -
		  U[offset      +IA*arraySize+(kmax-2)*zSlice] ) / dx;
	
	dbydy = ( U[offset      +IB*arraySize+(kmax-2)*zSlice] -
		  U[offset      +IB*arraySize+(kmax-2)*zSlice] ) / dy;
	
	real_t dbz2 = dz*(dbxdx+dbydy);
	U[offset+IC*arraySize+(kmax-1)*zSlice] = bz - dbz1 - dbz2;

      } // end for i
    } // end for j

    // end normal magnetic field

  } // boundaryLoc == ZMAX
  
} // end make_boundary2_z_stratified_cpu
#endif // ! __CUDACC__


#ifdef __CUDACC__

/**
 * \brief Actual GPU CUDA kernel used to fill a Z direction boundary of the grid in the special case of stratified rotating MHD simulation.
 * 
 * \param[in,out] U float pointer to memory array (hydro fields)
 * \param[in] pitch Memory pitch.
 * \param[in] imax physical size along X (ghost cells included)
 * \param[in] jmax physical size along Y (ghost cells included)
 * \param[in] kmax physical size along Z (ghost cells included)
 * \param[in] arraySize physical size allocated to 1 field (ghost cells inc).
 *
 * \tparam boundaryLoc : only ZMIN or ZMAX are valid here
 *
 * NOTE: nghost is assumed to be 3 and mhdEnabled to true
 *
 * IMPORTANT NOTE: This routine is intended to called by the wrapper 
 * make_boundary2_z_stratified.
 *
 */
template<BoundaryLocation boundaryLoc>
__GLOBAL__
void make_boundary2_z_stratified_gpu_kernel1(real_t* U, 
					     int     pitch, 
					     int     imax, 
					     int     jmax, 
					     int     kmax, 
					     int     arraySize, 
					     bool    floor)
  
{
  const real_t H      =  gParams.cIso / gParams.Omega0;
  //real_t &dx    =  gParams.dx;
  //real_t &dy    =  gParams.dy;
  real_t &dz    =  gParams.dz;
  real_t &zMin  =  gParams.zMin;
  real_t &zMax  =  gParams.zMax;
  const real_t factor = -dz / 2.0 / H / H;

  real_t ratio_nyp1,ratio_nyp2,ratio_nyp3;

  //const int nGhosts=3;

  const int zSlice = pitch*jmax;

  /* for a 3D problem, we use a 2D grid kernel */
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int i  = bx * MK_BOUND_BLOCK_SIZE_3D + tx;
  
  const int by = blockIdx.y;
  const int ty = threadIdx.y;
  const int j  = by * MK_BOUND_BLOCK_SIZE_3D + ty;
  
  if (floor) {
    ratio_nyp1 = ONE_F;
    ratio_nyp2 = ONE_F; 
    ratio_nyp3 = ONE_F;
  } else {

    if (boundaryLoc == ZMIN) {
      ratio_nyp1 = exp( factor*(-2*(zMin+HALF_F*dz)+    dz) );
      ratio_nyp2 = exp( factor*(-2*(zMin+HALF_F*dz)+3.0*dz) );
      ratio_nyp3 = exp( factor*(-2*(zMin+HALF_F*dz)+5.0*dz) );
    }

    if (boundaryLoc == ZMAX) {
      ratio_nyp1 = exp( factor*( 2*(zMax-HALF_F*dz)+    dz) );
      ratio_nyp2 = exp( factor*( 2*(zMax-HALF_F*dz)+3.0*dz) );
      ratio_nyp3 = exp( factor*( 2*(zMax-HALF_F*dz)+5.0*dz) );
    }

  } // end floor

  if (boundaryLoc == ZMIN) {

    if (i<imax and j<jmax) {
      
      // extrapolating density
      int offset0_d  = ID*arraySize + i + j*pitch +  0 * zSlice;
      int offset1_d  = ID*arraySize + i + j*pitch +  1 * zSlice;
      int offset2_d  = ID*arraySize + i + j*pitch +  2 * zSlice;
      int offset3_d  = ID*arraySize + i + j*pitch +  3 * zSlice;
      
      real_t rho3 = U[offset3_d];
      real_t rho2 = U[offset3_d]*ratio_nyp1;
      real_t rho1 = U[offset3_d]*ratio_nyp1*ratio_nyp2;
      real_t rho0 = U[offset3_d]*ratio_nyp1*ratio_nyp2*ratio_nyp3;
      
      U[offset2_d] = rho2;
      U[offset1_d] = rho1;
      U[offset0_d] = rho0;
      
      // zero gradient BC for velocity IU
      {
	int offset0  = IU*arraySize + i + j*pitch +  0 * zSlice;
	int offset1  = IU*arraySize + i + j*pitch +  1 * zSlice;
	int offset2  = IU*arraySize + i + j*pitch +  2 * zSlice;
	int offset3  = IU*arraySize + i + j*pitch +  3 * zSlice;
	
	U[offset0] = U[offset3] / rho3 * rho0;
	U[offset1] = U[offset3] / rho3 * rho1;
	U[offset2] = U[offset3] / rho3 * rho2;
	
      } // end IU
      
      // zero gradient BC for velocity IV
      {
	int offset0  = IV*arraySize + i + j*pitch +  0 * zSlice;
	int offset1  = IV*arraySize + i + j*pitch +  1 * zSlice;
	int offset2  = IV*arraySize + i + j*pitch +  2 * zSlice;
	int offset3  = IV*arraySize + i + j*pitch +  3 * zSlice;
	
	U[offset0] = U[offset3] / rho3 * rho0;
	U[offset1] = U[offset3] / rho3 * rho1;
	U[offset2] = U[offset3] / rho3 * rho2;
	
      } // end IV
      
      // now normal velocity IW
      {
	int offset0  = IW*arraySize + i + j*pitch +  0 * zSlice;
	int offset1  = IW*arraySize + i + j*pitch +  1 * zSlice;
	int offset2  = IW*arraySize + i + j*pitch +  2 * zSlice;
	int offset3  = IW*arraySize + i + j*pitch +  3 * zSlice;
	
	real_t w = FMIN( U[offset3] ,ZERO_F);
	
	U[offset0] = w;
	U[offset1] = w;
	U[offset2] = w;
	
      } // end IW
      
      // vanishing tangential magnetic field (IA,IB)
      {
	int offset0  = IA*arraySize + i + j*pitch +  0 * zSlice;
	int offset1  = IA*arraySize + i + j*pitch +  1 * zSlice;
	int offset2  = IA*arraySize + i + j*pitch +  2 * zSlice;
	
	U[offset0]   = ZERO_F;
	U[offset1]   = ZERO_F;
	U[offset2]   = ZERO_F;
	
	offset0      = IB*arraySize + i + j*pitch +  0 * zSlice;
	offset1      = IB*arraySize + i + j*pitch +  1 * zSlice;
	offset2      = IB*arraySize + i + j*pitch +  2 * zSlice;
	
	U[offset0]   = ZERO_F;
	U[offset1]   = ZERO_F;
	U[offset2]   = ZERO_F;
	
      } // end IA,IB
      
    } // end i<imax and j<jmax

  } else if (boundaryLoc == ZMAX) {
    
    if (i<imax and j<jmax) {
      
      // extrapolating density
      int offset1_d  = ID*arraySize + i + j*pitch +  (kmax-1) * zSlice;
      int offset2_d  = ID*arraySize + i + j*pitch +  (kmax-2) * zSlice;
      int offset3_d  = ID*arraySize + i + j*pitch +  (kmax-3) * zSlice;
      int offset4_d  = ID*arraySize + i + j*pitch +  (kmax-4) * zSlice;
      
      real_t rho4 = U[offset4_d];
      real_t rho3 = U[offset4_d]*ratio_nyp1;
      real_t rho2 = U[offset4_d]*ratio_nyp1*ratio_nyp2;
      real_t rho1 = U[offset4_d]*ratio_nyp1*ratio_nyp2*ratio_nyp3;
      
      U[offset3_d] = rho3;
      U[offset2_d] = rho2;
      U[offset1_d] = rho1;
      
      // zero gradient BC for velocity IU
      {
	int offset1  = IU*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	int offset2  = IU*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	int offset3  = IU*arraySize + i + j*pitch +  (kmax-3) * zSlice;
	int offset4  = IU*arraySize + i + j*pitch +  (kmax-4) * zSlice;
	
	U[offset3] = U[offset4] / rho4 * rho3;
	U[offset2] = U[offset4] / rho4 * rho2;
	U[offset1] = U[offset4] / rho4 * rho1;
	
      } // end IU
      
      // zero gradient BC for velocity IV
      {
	int offset1  = IV*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	int offset2  = IV*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	int offset3  = IV*arraySize + i + j*pitch +  (kmax-3) * zSlice;
	int offset4  = IV*arraySize + i + j*pitch +  (kmax-4) * zSlice;
	
	U[offset3] = U[offset4] / rho4 * rho3;
	U[offset2] = U[offset4] / rho4 * rho2;
	U[offset1] = U[offset4] / rho4 * rho1;
	
      } // end IV
      
      // now normal velocity IW
      {
	int offset1  = IW*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	int offset2  = IW*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	int offset3  = IW*arraySize + i + j*pitch +  (kmax-3) * zSlice;
	int offset4  = IW*arraySize + i + j*pitch +  (kmax-4) * zSlice;
	
	real_t w = FMAX( U[offset4] ,ZERO_F);
	
	U[offset3] = w;
	U[offset2] = w;
	U[offset1] = w;
	
      } // end IW
      
      // vanishing tangential magnetic field (IA,IB)
      {
	int offset1  = IA*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	int offset2  = IA*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	int offset3  = IA*arraySize + i + j*pitch +  (kmax-3) * zSlice;
	
	U[offset3]   = ZERO_F;
	U[offset2]   = ZERO_F;
	U[offset1]   = ZERO_F;
	
	offset1      = IB*arraySize + i + j*pitch +  (kmax-1) * zSlice;
	offset2      = IB*arraySize + i + j*pitch +  (kmax-2) * zSlice;
	offset3      = IB*arraySize + i + j*pitch +  (kmax-3) * zSlice;
	
	U[offset3]   = ZERO_F;
	U[offset2]   = ZERO_F;
	U[offset1]   = ZERO_F;
	
      } // end IA,IB
      
    } // end i<imax and j<jmax

  } // end if boundaryLoc == ZMAX

} // end make_boundary2_z_stratified_gpu_kernel1

/**
 * \brief Actual GPU CUDA kernel used to fill a Z direction boundary of the grid in the special case of stratified rotating MHD simulation.
 * 
 * \param[in,out] U float pointer to memory array (hydro fields)
 * \param[in] pitch Memory pitch.
 * \param[in] imax physical size along X (ghost cells included)
 * \param[in] jmax physical size along Y (ghost cells included)
 * \param[in] kmax physical size along Z (ghost cells included)
 * \param[in] arraySize physical size allocated to 1 field (ghost cells inc).
 *
 * \tparam boundaryLoc : only ZMIN or ZMAX are valid here
 *
 * NOTE: nghost is assumed to be 3 and mhdEnabled to true
 *
 * IMPORTANT NOTE: This routine is intended to called by the wrapper 
 * make_boundary2_z_stratified.
 *
 */
template<BoundaryLocation boundaryLoc>
__GLOBAL__
void make_boundary2_z_stratified_gpu_kernel2(real_t* U, 
					     int     pitch, 
					     int     imax, 
					     int     jmax, 
					     int     kmax, 
					     int     arraySize, 
					     bool    floor)
  
{
  
  //const real_t H      =  gParams.cIso / gParams.Omega0;
  real_t &dx    =  gParams.dx;
  real_t &dy    =  gParams.dy;
  real_t &dz    =  gParams.dz;
  //real_t &zMin  =  gParams.zMin;
  //real_t &zMax  =  gParams.zMax;
  //const real_t factor = -dz / 2.0 / H / H;

  //real_t ratio_nyp1,ratio_nyp2,ratio_nyp3;

  //const int nGhosts=3;

  const int zSlice = pitch*jmax;

  /* for a 3D problem, we use a 2D grid kernel */
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int i  = bx * MK_BOUND_BLOCK_SIZE_3D + tx;
  
  const int by = blockIdx.y;
  const int ty = threadIdx.y;
  const int j  = by * MK_BOUND_BLOCK_SIZE_3D + ty;

  if (boundaryLoc == ZMIN) {

    // Normal magnetic field (IW)
    if ( (i<imax-1) and (j<jmax-1) ) {
      
      real_t dbxdx, dbydy;
      
      int offset = i + j*pitch;
      
      dbxdx = ( U[offset+1    +IA*arraySize+2*zSlice] -
		U[offset      +IA*arraySize+2*zSlice] ) / dx;
      
      dbydy = ( U[offset+pitch+IB*arraySize+2*zSlice] -
		U[offset      +IB*arraySize+2*zSlice] ) / dy;
      
      real_t  bz  =  U[offset+IC*arraySize+3*zSlice];
      real_t dbz2 = dz*(dbxdx+dbydy);
      
      U[offset+IC*arraySize+2*zSlice] = bz + dbz2;
      
      dbxdx = ( U[offset+1    +IA*arraySize+1*zSlice] -
		U[offset      +IA*arraySize+1*zSlice] ) / dx;
      
      dbydy = ( U[offset+pitch+IB*arraySize+1*zSlice] -
		U[offset      +IB*arraySize+1*zSlice] ) / dy;
      
      real_t dbz1 = dz*(dbxdx+dbydy);
      U[offset+IC*arraySize+1*zSlice] = bz + dbz2 + dbz1;
      
      dbxdx = ( U[offset+1    +IA*arraySize+0*zSlice] -
		U[offset      +IA*arraySize+0*zSlice] ) / dx;
      
      dbydy = ( U[offset+pitch+IB*arraySize+0*zSlice] -
		U[offset      +IB*arraySize+0*zSlice] ) / dy;
      
      real_t dbz0 = dz*(dbxdx+dbydy);
      U[offset+IC*arraySize+0*zSlice] = bz + dbz2 + dbz1 + dbz0;
      
    } // end if (i<imax-1) and (j<jmax-1)
    
  } else if (boundaryLoc == ZMAX) {
    
    // Normal magnetic field (IW)
    if ( (i<imax-1) and (j<jmax-1) ) {
      
      real_t dbxdx, dbydy;
      
      int offset = i + j*pitch;
      
      dbxdx = ( U[offset+1    +IA*arraySize+(kmax-3)*zSlice] -
		U[offset      +IA*arraySize+(kmax-3)*zSlice] ) / dx;
      
      dbydy = ( U[offset      +IB*arraySize+(kmax-3)*zSlice] -
		U[offset      +IB*arraySize+(kmax-3)*zSlice] ) / dy;
      
      real_t  bz  =  U[offset+IC*arraySize+(kmax-3)*zSlice];
      
      real_t dbz1 = dz*(dbxdx+dbydy);
      U[offset+IC*arraySize+(kmax-2)*zSlice] = bz - dbz1;
      
      dbxdx = ( U[offset+1    +IA*arraySize+(kmax-2)*zSlice] -
		U[offset      +IA*arraySize+(kmax-2)*zSlice] ) / dx;
      
      dbydy = ( U[offset      +IB*arraySize+(kmax-2)*zSlice] -
		U[offset      +IB*arraySize+(kmax-2)*zSlice] ) / dy;
      
      real_t dbz2 = dz*(dbxdx+dbydy);
      U[offset+IC*arraySize+(kmax-1)*zSlice] = bz - dbz1 - dbz2;
      
    } // end if (i<imax-1) and (j<jmax-1)
    
  } // end boundaryLoc == ZMAX
  
} // make_boundary2_z_stratified_gpu_kernel2

#endif // __CUDACC__

/**
 * \brief Wrapper routine to actual CPU or GPU kernel to fill a Z direction boundary of the grid in the special case of stratified rotating MHD simulation.
 * 
 * \param[in,out] U float pointer to memory array (hydro fields)
 * \param[in] pitch Memory pitch.
 * \param[in] imax physical size along X (ghost cells included)
 * \param[in] jmax physical size along Y (ghost cells included)
 * \param[in] kmax physical size along Z (ghost cells included)
 * \param[in] arraySize physical size allocated to 1 field (ghost cells inc).
 *
 * \tparam boundaryLoc : only ZMIN or ZMAX are valid here
 *
 * NOTE: nghost is assumed to be 3 and mhdEnabled to true
 *
 */
template<BoundaryLocation boundaryLoc>
void make_boundary2_z_stratified(real_t* U, 
				 int pitch, 
				 int imax, 
				 int jmax, 
				 int kmax, 
				 int arraySize, 
				 bool floor)
#ifdef __CUDACC__
{

  /* 
   * need to call :
   * make_boundary2_z_stratified_gpu_kernel1
   * make_boundary2_z_stratified_gpu_kernel2
   */

  dim3 blockCount( blocksFor(imax, MK_BOUND_BLOCK_SIZE_3D),
		   blocksFor(jmax, MK_BOUND_BLOCK_SIZE_3D), 
		   1);
  dim3 threadsPerBlock(MK_BOUND_BLOCK_SIZE_3D, 
		       MK_BOUND_BLOCK_SIZE_3D, 
		       1);

  make_boundary2_z_stratified_gpu_kernel1<boundaryLoc>
    <<<blockCount, threadsPerBlock >>>(U, 
				       pitch, 
				       imax, 
				       jmax, 
				       kmax, 
				       arraySize, 
				       floor);
  
  make_boundary2_z_stratified_gpu_kernel2<boundaryLoc>
    <<<blockCount, threadsPerBlock >>>(U, 
				       pitch, 
				       imax, 
				       jmax, 
				       kmax, 
				       arraySize, 
				       floor);
  
} // end make_boundary2_z_stratified (GPU version)

#else

{ // start make_boundary2_z_stratified (CPU version)


  make_boundary2_z_stratified_cpu<boundaryLoc>(U,
					       pitch,
					       imax,
					       jmax,
					       kmax,
					       arraySize,
					       floor);
  
} // end make_boundary2_z_stratified (CPU version)

#endif // __CUDACC__



#ifdef __CUDACC__

/**
 * \brief This function initializes the upper boundary with values that 
 * simulate a burst of liquid coming into the grid.
 *
 * recall that variable order is : density, pressure, velocity_x, velocity_y
 * \param[in,out] U     : input/output hydro array
 * \param[in] pitch     : pitch size
 * \param[in] arraysize : number of cells offset between 2 variables (i.e. density and pressure)
 * \param[in] ijet      : width (number of cells) used to inject matter
 * \param[in] jet       : real4_t hydro state vector
 * \param[in] offsetJet : offset in pixel to shift injection
 * \param[in] nGhosts   : number of ghost cells
 *
 */
__GLOBAL__ void make_jet_2d(real_t* U, int pitch, int arraySize, int ijet, real4_t jet, int offsetJet, int nGhosts=2)
{
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int k = bx * MAKE_JET_BLOCK_SIZE + tx;
  
  if(k >= nGhosts+offsetJet and k < nGhosts+offsetJet+ijet)
    {

      for (int iGhost = 0; iGhost < nGhosts; ++iGhost) {
	
	int offset = k+iGhost*pitch;
	U[offset] = jet.x; offset += arraySize;
	U[offset] = jet.y; offset += arraySize;
	U[offset] = jet.z; offset += arraySize;
	U[offset] = jet.w;

      }
      
    }

} // make_jet_2d

/** 
 * \brief This is the 3D version of routine make_jet_2d; we just set the 
 * same velocity along x and y axis, and inject matter along z.
 *
 * \sa make_jet_2d
 */
__GLOBAL__ void make_jet_3d(real_t* U, int pitch, int jmax, int arraySize, int ijet, real4_t jet, int offsetJet, int nGhosts=2)
{
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int i = bx * MAKE_JET_BLOCK_SIZE_3D + tx;

  const int by = blockIdx.y;
  const int ty = threadIdx.y;
  const int j = by * MAKE_JET_BLOCK_SIZE_3D + ty;

  if (i >= nGhosts+offsetJet and i < nGhosts+offsetJet+ijet and
      j >= nGhosts+offsetJet and j < nGhosts+offsetJet+ijet 
      /*(i*i+j*j < (nGhosts+offsetJet+ijet)*(nGhosts+offsetJet+ijet) )*/ )
    { 

      for (int iGhost = 0; iGhost < nGhosts; ++iGhost) {

	int offset = i + pitch * j + pitch*jmax*iGhost;
	U[offset] = jet.x; offset += arraySize;
	U[offset] = jet.y; offset += arraySize;
	U[offset] = jet.z; offset += arraySize;
	U[offset] = jet.z; offset += arraySize;
	U[offset] = jet.w;
      
      }

    }

} // make_jet_3d

#endif // __CUDACC__


#endif /*MAKE_BOUNDARY_BASE_H_*/
