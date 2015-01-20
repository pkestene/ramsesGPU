/**
 * \file mpiBorderUtils.h
 * \brief Some utility routines dealing with MPI border buffers.
 *
 * \date 13 Oct 2010
 * \author Pierre Kestener
 *
 * $Id: mpiBorderUtils.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef MPI_BORDER_UTILS_H_
#define MPI_BORDER_UTILS_H_

#include "Arrays.h"
#include "constants.h"

#ifdef __CUDACC__
#include "mpiBorderUtils.cuh"
#endif // __CUDACC__

namespace hydroSimu {

  /**
   * function : copyBorderBufRecvToHostArray
   *
   * Copy a border buffer (as received by MPI communications) into the
   * right location (given by template parameter boundaryLoc).
   * Here we assume U is a HostArray. 
   * \sa copyBorderBufRecvToDeviceArray
   *
   * template parameters:
   * @tparam boundaryLoc : destination boundary location 
   *                       used to check array dimensions and set offset
   * @tparam dimType     : triggers 2D or 3D specific treatment
   * @tparam ghostWidth  : ghost cell thickness (should be only 2 or
   *                       3); note that ghostWidth is checked to be 2
   *                       or 3 in HydroParameters constructor.
   *
   * argument parameters:
   * @param[out] U reference to a hydro simulations array (destination array)
   * @param[in]  b reference to a border buffer (source array)
   */
  template<
    BoundaryLocation boundaryLoc,
    DimensionType    dimType,
    int              ghostWidth
    >
  void copyBorderBufRecvToHostArray(HostArray<real_t>&U, HostArray<real_t>& b)
  {
    /*
     * array dimension  sanity check
     */
    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
      if (b.dimx() != ghostWidth or 
	  b.dimy() != U.dimy() or 
	  b.dimz() != U.dimz())
	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    }
    if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
      if (b.dimy() != ghostWidth or
	  b.dimx() != U.dimx() or
	  b.dimz() != U.dimz())
	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    }
    if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {
      if (b.dimz() != ghostWidth or
	  b.dimx() != U.dimx() or
	  b.dimy() != U.dimy())
	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    }

    /*
     * we can now proceed with copy.
     */
    int offset = 0;
    if (boundaryLoc == XMAX)
      offset = U.dimx()-ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.dimy()-ghostWidth;
    if (boundaryLoc == ZMAX)
      offset = U.dimz()-ghostWidth;
    

    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
	
      if (dimType == TWO_D) {
	  
	for (uint nVar=0; nVar<U.nvar(); ++nVar)
	  for (uint j=0; j<U.dimy(); ++j) {
	    U(offset  ,j,nVar) = b(0,j,nVar);
	    U(offset+1,j,nVar) = b(1,j,nVar);
	    if (ghostWidth == 3)
	      U(offset+2,j,nVar) = b(2,j,nVar);
	  }
	  
      } else { // 3D case
	  
	for (uint nVar=0; nVar<U.nvar(); ++nVar)
	  for (uint k=0; k<U.dimz(); ++k)
	    for (uint j=0; j<U.dimy(); ++j) {
	      U(offset  ,j,k,nVar) = b(0,j,k,nVar);
	      U(offset+1,j,k,nVar) = b(1,j,k,nVar);
	      if (ghostWidth == 3)
		U(offset+2,j,k,nVar) = b(2,j,k,nVar);
	    }
	  
      }
	
    } else if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
	
      if (dimType == TWO_D) {
	  
	for (uint nVar=0; nVar<U.nvar(); ++nVar)
	  for (uint i=0; i<U.dimx(); ++i) {
	    U(i,offset  ,nVar) = b(i,0,nVar);
	    U(i,offset+1,nVar) = b(i,1,nVar);
	    if (ghostWidth == 3)
	      U(i,offset+2,nVar) = b(i,2,nVar);
	  }
	  
      } else { // 3D case
	  
	for (uint nVar=0; nVar<U.nvar(); ++nVar)
	  for (uint k=0; k<U.dimz(); ++k)
	    for (uint i=0; i<U.dimx(); ++i) {
	      U(i,offset  ,k,nVar) = b(i,0,k,nVar);
	      U(i,offset+1,k,nVar) = b(i,1,k,nVar);
	      if (ghostWidth == 3)
		U(i,offset+2,k,nVar) = b(i,2,k,nVar);
	    }
	  
      }
	
    } else if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {
	
      // always 3D case
      for (uint nVar=0; nVar<U.nvar(); ++nVar)
	for (uint j=0; j<U.dimy(); ++j)
	  for (uint i=0; i<U.dimx(); ++i) {
	    U(i,j,offset  ,nVar) = b(i,j,0,nVar);
	    U(i,j,offset+1,nVar) = b(i,j,1,nVar);
	    if (ghostWidth == 3)
	      U(i,j,offset+2,nVar) = b(i,j,2,nVar);
	  }
	
    }

  } // copyBorderBufSendToHostArray


  /**
   * function : copyHostArrayToBorderBufSend
   * 
   * Copy array border to a border buffer (to be sent by MPI communications) 
   * Here we assume U is a <b>HostArray</b>. 
   * \sa copyBorderBufSendToHostArray
   *
   * template parameters:
   * @tparam boundaryLoc : boundary location in source Array
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
  void copyHostArrayToBorderBufSend(HostArray<real_t>& b, HostArray<real_t>&U)
  {
    /*
     * array dimension  sanity check
     *
     * This may be bypassed for performance issue since border arrays are
     * allocated in HydroRunBaseMpi constructor.
     */
    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
      if (b.dimx() != ghostWidth or 
	  b.dimy() != U.dimy() or 
	  b.dimz() != U.dimz())
	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    }
    if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
      if (b.dimy() != ghostWidth or
	  b.dimx() != U.dimx() or
	  b.dimz() != U.dimz())
	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    }
    if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {
      if (b.dimz() != ghostWidth or
	  b.dimx() != U.dimx() or
	  b.dimy() != U.dimy())
	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    }

    /*
     * Proceed with copy
     */
    int offset = ghostWidth;
    if (boundaryLoc == XMAX)
      offset = U.dimx()-2*ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.dimy()-2*ghostWidth;
    if (boundaryLoc == ZMAX)
      offset = U.dimz()-2*ghostWidth;
    

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
	  
      }
	
    } else if (boundaryLoc == YMIN or boundaryLoc == YMAX) {

      if (dimType == TWO_D) {
	  
	for (uint nVar=0; nVar<U.nvar(); ++nVar)
	  for (uint i=0; i<U.dimx(); ++i) {
	    b(i,0,nVar) = U(i,offset  ,nVar);
	    b(i,1,nVar) = U(i,offset+1,nVar);
	    if (ghostWidth == 3) 
	      b(i,2,nVar) = U(i,offset+2,nVar);
	  }
	  
      } else { // 3D case
	  
	for (uint nVar=0; nVar<U.nvar(); ++nVar)
	  for (uint k=0; k<U.dimz(); ++k)
	    for (uint i=0; i<U.dimx(); ++i) {
	      b(i,0,k,nVar) = U(i,offset  ,k,nVar);
	      b(i,1,k,nVar) = U(i,offset+1,k,nVar);
	      if (ghostWidth == 3)
		b(i,2,k,nVar) = U(i,offset+2,k,nVar);
	    }
	  
      }
    } else if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {

      // always 3D case
      for (uint nVar=0; nVar<U.nvar(); ++nVar)
	for (uint j=0; j<U.dimy(); ++j)
	  for (uint i=0; i<U.dimx(); ++i) {
	    b(i,j,0,nVar) = U(i,j,offset  ,nVar);
	    b(i,j,1,nVar) = U(i,j,offset+1,nVar);
	    if (ghostWidth == 3)
	      b(i,j,2,nVar) = U(i,j,offset+2,nVar);
	  }
	
    } // end (boundaryLoc == ZMIN or boundaryLoc == ZMAX)
      
  } // copyBorderBufRecvToHostArray


#ifdef __CUDACC__

  /*******************************************************
   * GPU copy border buf routines
   *******************************************************/
  /**
   * function : copyBorderBufRecvToDeviceArray
   *
   * Copy a border buffer (as received by MPI communications) into the
   * right location (given by template parameter boundaryLoc) of a
   * Device array.
   *
   * \sa copyDeviceArrayToBorderBufSend
   *
   * template parameters:
   * @tparam boundaryLoc : destination boundary location 
   *                       used to check array dimensions and set offset
   * @tparam dimType     : triggers 2D or 3D specific treatment
   * @tparam ghostWidth  : ghost cell thickness (should be only 2 or
   *                       3); note that ghostWidth is checked to be 2
   *                       or 3 in HydroParameters constructor.
   *
   * argument parameters:
   * @param[out]    U     reference to a hydro simulations array (destination
   *                      device array which is a pitched array)
   * @param[in,out] bTemp reference to a Device array : border buffer same size
   *                      as b, used to make the host-to-device copy; bTemp
   *                      may be linear or pitched 
   * @param[in]     b     reference to a border buffer (source host array)
   */
  template<
    BoundaryLocation boundaryLoc,
    DimensionType    dimType,
    int              ghostWidth
    >
  void copyBorderBufRecvToDeviceArray(DeviceArray<real_t>&U, 
				      DeviceArray<real_t>& bTemp, 
				      HostArray<real_t>& b)
  {
    /*
     * array dimension  sanity check (may be by-passed)
     */
    // if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
    //   if (b.dimx() != ghostWidth or b.dimy() != U.dimy() or b.dimz() != U.dimz())
    // 	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    // }
    // if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
    //   if (b.dimy() != ghostWidth or b.dimx() != U.dimx() or b.dimz() != U.dimz())
    // 	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    // }
    // if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {
    //   if (b.dimz() != ghostWidth or b.dimx() != U.dimx() or b.dimy() != U.dimy())
    // 	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    // }

    /*
     * we can now proceed with copy.
     */

    /*
     * copy border buffer from host memory b to bTemp
     */
    bTemp.copyFromHost(b);

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
      if ( bTemp.usePitchedMemory() ) {
	::copyBorderBufSendToDeviceArray_2d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	    (bTemp.data(), bTemp.pitch(), 
	     dim3(bTemp.dimx(), bTemp.dimy(), bTemp.dimz()), 
	     U.data(), U.pitch(), 
	     dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      } else { // bTemp was allocated with cudaMalloc
	::copyBorderBufSendToDeviceArray_linear_2d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (bTemp.data(), 
	   dim3(bTemp.dimx(), bTemp.dimy(), bTemp.dimz()), 
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
      if ( bTemp.usePitchedMemory() ) {
	::copyBorderBufSendToDeviceArray_3d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (bTemp.data(), bTemp.pitch(), 
	   dim3(bTemp.dimx(), bTemp.dimy(), bTemp.dimz()),
	   U.data(), U.pitch(), 
	   dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      } else { // bTemp was allocated with cudaMalloc
	::copyBorderBufSendToDeviceArray_linear_3d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (bTemp.data(), 
	   dim3(bTemp.dimx(), bTemp.dimy(), bTemp.dimz()),
	   U.data(), U.pitch(), 
	   dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      }
    }
      

  } // copyBorderBufRecvToDeviceArray

  /**
   * function : copyDeviceArrayToBorderBufSend
   * 
   * This is a wrapper to call the CUDA kernel that actually copy
   * array border (device memory) to a border buffer (in host memory
   * to be sent by MPI communications) 
   * Here we assume U is a <b>DeviceArray</b>.
   * \sa CPU version copyHostArrayToBorderBufSend
   *
   * template parameters:
   * @tparam boundaryLoc : boundary location in source Array
   * @tparam dimType     : triggers 2D or 3D specific treatment
   * @tparam ghostWidth  : ghost cell thickness (should be only 2 or
   *                       3); note that ghostWidth is checked to be 2
   *                       or 3 in HydroParameters constructor.
   *
   * argument parameters:
   * @param[out]    b     reference to a Host array   : border buffer (destination array)
   * @param[in,out] bTemp reference to a Device array : border buffer same size
   *                      as b, used to make the device-to-host copy
   * @param[in]     U     reference to a hydro simulations array (source array)
   */
  template<
    BoundaryLocation boundaryLoc,
    DimensionType    dimType,
    int              ghostWidth
    >
  void copyDeviceArrayToBorderBufSend(HostArray<real_t>& b, 
				      DeviceArray<real_t>& bTemp, 
				      DeviceArray<real_t>&U)
  {
    /*
     * array dimension  sanity check 
     *
     * This may be bypassed for performance issue since border arrays are
     * allocated in HydroRunBaseMpi constructor.
     */
    // if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
    //   if (b.dimx() != 2 or b.dimy() != U.dimy() or b.dimz() != U.dimz())
    // 	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    // }
    // if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
    //   if (b.dimy() != 2 or b.dimx() != U.dimx() or b.dimz() != U.dimz())
    // 	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    // }
    // if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {
    //   if (b.dimz() != 2 or b.dimx() != U.dimx() or b.dimy() != U.dimy())
    // 	throw std::runtime_error(std::string(__FUNCTION__)+": non-matching array dimensions ");
    // }

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
      if ( bTemp.usePitchedMemory() ) {
	::copyDeviceArrayToBorderBufSend_2d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (bTemp.data(), bTemp.pitch(), 
	   dim3(bTemp.dimx(), bTemp.dimy(), bTemp.dimz()),
	   U.data(), U.pitch(), 
	   dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      } else { // bTemp was allocated with cudaMalloc
	::copyDeviceArrayToBorderBufSend_linear_2d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (bTemp.data(), 
	   dim3(bTemp.dimx(), bTemp.dimy(), bTemp.dimz()),
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
      if ( bTemp.usePitchedMemory() ) {
	::copyDeviceArrayToBorderBufSend_3d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (bTemp.data(), bTemp.pitch(), 
	   dim3(bTemp.dimx(), bTemp.dimy(), bTemp.dimz()), 
	   U.data(), U.pitch(), 
	   dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      } else { // bTemp was allocated with cudaMalloc
	::copyDeviceArrayToBorderBufSend_linear_3d_kernel<boundaryLoc, ghostWidth><<<dimGrid,dimBlock>>>
	  (bTemp.data(),
	   dim3(bTemp.dimx(), bTemp.dimy(), bTemp.dimz()), 
	   U.data(), U.pitch(), 
	   dim3(U.dimx(), U.dimy(), U.dimz()), U.nvar());
      }
    }
    
    /*
     * copy back results from bTemp to host memory b
     */
    bTemp.copyToHost(b);

  } // copyDeviceArrayToBorderBufSend

#endif // __CUDACC__


} // namespace hydroSimu

#endif // MPI_BORDER_UTILS_H_
