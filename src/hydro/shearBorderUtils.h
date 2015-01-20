/**
 * \file shearBorderUtils.h
 * \brief Some utility routines dealing with shearing border buffers.
 *
 * \date 3 February 2012
 * \author Pierre Kestener
 *
 * $Id: shearBorderUtils.h 1792 2012-02-23 11:46:41Z pkestene $
 */
#ifndef SHEAR_BORDER_UTILS_H_
#define SHEAR_BORDER_UTILS_H_

#include "Arrays.h"
#include "constants.h"

#ifdef __CUDACC__
#include "mpiBorderUtils.cuh"
#endif // __CUDACC__

namespace hydroSimu {

  /**
   * function : copyHostArrayToBorderBufShear
   * 
   * Copy array border (XMIN and XMAX) to 2 border buffers
   * Here we assume U is a <b>HostArray</b>. 
   *
   * template parameters:
   * @tparam boundaryLoc : boundary location in source Array
   *                       when used for shearing box, boundaryLoc can only be
   *                       XMIN or XMAX !
   * @tparam dimType     : triggers 2D or 3D specific treatment
   * @tparam ghostWidth  : ghost cell thickness (should be only 2 or
   *                       3); note that ghostWidth is checked to be 2
   *                       or 3 in HydroParameters constructor.
   *
   * argument parameters:
   * @param[out] b reference to a border buffer (destination array)
   * @param[in]  U reference to a hydro simulations array (source array)
   */
  template<
    BoundaryLocation boundaryLoc,
    DimensionType    dimType,
    int              ghostWidth
    >
  void copyHostArrayToBorderBufShear(HostArray<real_t>& b, HostArray<real_t>&U)
  {
    /*
     * array dimension  sanity check
     *
     * This may be bypassed for performance issue since border arrays are
     * allocated in MHDRunGodunov/MHDRunGodunovMpi constructor.
     */
    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
      if (b.dimx() != ghostWidth or 
	  b.dimy() != U.dimy() or 
	  b.dimz() != U.dimz())
	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    }
    if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
      
	throw std::runtime_error(std::string(__FUNCTION__)+": not implemented ");
    }
    if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {
      
	throw std::runtime_error(std::string(__FUNCTION__)+": not implemented ");
    }

    /*
     * Proceed with copy
     */
    int offset = ghostWidth;
    if (boundaryLoc == XMAX)
      offset = U.dimx()-2*ghostWidth;

    /*
     * simple copy when PERIODIC or COPY
     */      
    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {

      if (dimType == TWO_D) {
	  
	for (uint nVar=0; nVar<U.nvar(); ++nVar)
	  for (uint j=0; j<U.dimy(); ++j) {
	    b(0,j,nVar) = U(offset  ,j,nVar);
	    b(1,j,nVar) = U(offset+1,j,nVar);
	    if (ghostWidth == 3)
	      b(2,j,nVar) = U(offset+2,j,nVar);
	  }
	  
      } else { // 3D case
	  
	for (uint nVar=0; nVar<U.nvar(); ++nVar)
	  for (uint k=0; k<U.dimz(); ++k)
	    for (uint j=0; j<U.dimy(); ++j) {
	      b(0,j,k,nVar) = U(offset  ,j,k,nVar);
	      b(1,j,k,nVar) = U(offset+1,j,k,nVar);
	      if (ghostWidth == 3)
		b(2,j,k,nVar) = U(offset+2,j,k,nVar);
	    }
	  
      } // end 3D
	
    } // end if (boundaryLoc == XMIN or boundaryLoc == XMAX)
      
  } // copyHostArrayToBorderBufShear


#ifdef __CUDACC__

  /*******************************************************
   * GPU copy border buf for shearing box routines
   *******************************************************/

  /**
   * function : copyDeviceArrayToBorderBufShear
   * 
   * Copy array border (XMIN and XMAX) to 2 border buffers
   * Here we assume U is a <b>DeviceArray</b>. 
   *
   * template parameters:
   * @tparam boundaryLoc : boundary location in source Array
   *                       when used for shearing box, boundaryLoc can only be
   *                       XMIN or XMAX !
   * @tparam dimType     : triggers 2D or 3D specific treatment
   * @tparam ghostWidth  : ghost cell thickness (should be only 2 or
   *                       3); note that ghostWidth is checked to be 2
   *                       or 3 in HydroParameters constructor.
   *
   * argument parameters:
   * @param[out] b reference to a border buffer (destination array)
   * @param[in]  U reference to a hydro simulations array (source array)
   */
  template<
    BoundaryLocation boundaryLoc,
    DimensionType    dimType,
    int              ghostWidth
    >
  void copyDeviceArrayToBorderBufShear(DeviceArray<real_t>& b, DeviceArray<real_t>&U)
  {
    /*
     * array dimension  sanity check
     *
     * This may be bypassed for performance issue since border arrays are
     * allocated in MHDRunGodunov/MHDRunGodunovMpi constructor.
     */
    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
      if (b.dimx() != ghostWidth or 
	  b.dimy() != U.dimy() or 
	  b.dimz() != U.dimz())
	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    }
    if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
      
	throw std::runtime_error(std::string(__FUNCTION__)+": not implemented ");
    }
    if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {
      
	throw std::runtime_error(std::string(__FUNCTION__)+": not implemented ");
    }

    /*
     * Proceed with copy in device memory
     */
    if (dimType == TWO_D) {

      dim3 dimBlock(COPY_BORDER_BLOCK_SIZE,
		    1,
		    1);
      uint dimMax = U.dimx() < U.dimy() ? U.dimy() : U.dimx();
      dim3 dimGrid(blocksFor(dimMax, COPY_BORDER_BLOCK_SIZE),
		   1,
		   1);
      if ( b.usePitchedMemory() ) {
	::copyDeviceArrayToBorderBufSend_2d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (b.data(), b.pitch(), 
	   dim3(b.dimx(), b.dimy(), b.dimz()),
	   U.data(), U.pitch(), 
	   dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      } else { // b was allocated with cudaMalloc
	::copyDeviceArrayToBorderBufSend_linear_2d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (b.data(), 
	   dim3(b.dimx(), b.dimy(), b.dimz()),
	   U.data(), U.pitch(), 
	   dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      }

    } else { // THREE_D

      dim3 dimBlock(COPY_BORDER_BLOCK_SIZE_3D, 
		    COPY_BORDER_BLOCK_SIZE_3D, 
		    1);
      uint dimMax = U.dimx() < U.dimy() ? U.dimy() : U.dimx();
      dimMax = dimMax < U.dimz() ? U.dimz() : dimMax; 
      dim3 dimGrid(blocksFor(dimMax, COPY_BORDER_BLOCK_SIZE_3D), 
		   blocksFor(dimMax, COPY_BORDER_BLOCK_SIZE_3D),
		   1);
      if ( b.usePitchedMemory() ) {
	::copyDeviceArrayToBorderBufSend_3d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (b.data(), b.pitch(), 
	   dim3(b.dimx(), b.dimy(), b.dimz()), 
	   U.data(), U.pitch(), 
	   dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      } else { // b was allocated with cudaMalloc
	::copyDeviceArrayToBorderBufSend_linear_3d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (b.data(),
	   dim3(b.dimx(), b.dimy(), b.dimz()), 
	   U.data(), U.pitch(), 
	   dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      }
    }


  } // copyDeviceArrayToBorderBufShear

#endif // __CUDACC__


} // namespace hydroSimu

#endif // SHEAR_BORDER_UTILS_H_
