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
 * \file shearingBox_utils.cuh
 * \brief Defines some CUDA kernels for handling shearing box simulations.
 *
 * Shearing box simulations requires some specific kernels:
 * - flux and EMF remapping at XMIN and XMAX borders
 *
 * \date 21 Feb 2012
 * \author P. Kestener
 *
 * $Id: shearingBox_utils.cuh 3450 2014-06-16 22:03:23Z pkestene $
 *
 */
#ifndef SHEARING_BOX_UTILS_CUH_
#define SHEARING_BOX_UTILS_CUH_

#include "real_type.h"
#include "constants.h"

/**
 * Flux/EMF remapping kernel for 3D data at XMIN and XMAX borders (mono GPU only).
 *
 * \param[in]  d_shear_flux_xmin
 * \param[in]  d_shear_flux_xmax
 * \param[out] d_shear_flux_xmin_remap
 * \param[out] d_shear_flux_xmax_remap
 * \param[in]  jsize
 * \param[in]  ksize
 * \param[in]  d_shear_pitched
 * \param[in]  totalTime
 * \param[in]  dt
 */
__global__ void kernel_remapping_mhd_3d(real_t* d_shear_flux_xmin,
					real_t* d_shear_flux_xmax,
					real_t* d_shear_flux_xmin_remap,
					real_t* d_shear_flux_xmax_remap,
					real_t* d_emf,
					const int pitch,
					const int isize,
					const int jsize,
					const int ksize,
					const int d_shear_pitch,
					const real_t totalTime,
					const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int m = __mul24(bx, blockDim.x) + tx;
  const int n = __mul24(by, blockDim.y) + ty;

  // the following code is adapted from Dumses/bval_shear.f90, 
  // subroutines bval_shear_flux and bval_shear_emf
  real_t deltay,epsi,eps;
  int jplus;

  const real_t &dx = ::gParams.dx;
  const real_t &dy = ::gParams.dy;

  const int &nx = ::gParams.nx;
  const int &ny = ::gParams.ny;

  const real_t &Omega0 = ::gParams.Omega0;

  deltay = 1.5 * Omega0 * (dx * nx) * (totalTime+dt/2);
  deltay = FMOD(deltay, (dy * ny) );
  jplus  = (int) (deltay/dy);
  epsi   = FMOD(deltay,  dy);

  // hard-coded for MHD only
  const int ghostWidth = 3;

  const int arraySize   = d_shear_pitch*ksize;
  const int arraySize3d = pitch*jsize*ksize;

  /*
   * perform flux/emf remapping
   */
  const int &j=m;
  const int &k=n;
  
  if (j<jsize and k<ksize) {

   int jremap,jremapp1;
   int offset, offset_remap, offset_remapp1;
   int offset3d_xmin, offset3d_xmax;

    /*
     * inner (i.e. xMin) boundary - flux and emf
     */
    jremap   = j      - jplus - 1;
    jremapp1 = jremap + 1;
    eps      = 1.0-epsi/dy;
    
    if (jremap  < ghostWidth) jremap   += ny;
    if (jremapp1< ghostWidth) jremapp1 += ny;
    
    // physical memory offset
    offset         = j       + d_shear_pitch*k;
    offset_remap   = jremap  + d_shear_pitch*k;
    offset_remapp1 = jremapp1+ d_shear_pitch*k;
    offset3d_xmin  = ghostWidth + pitch * (j + jsize * k);
    offset3d_xmax  = nx + offset3d_xmin;

    // flux
    if (j>=ghostWidth and j<jsize-ghostWidth+1 and 
	k>=ghostWidth and k<ksize-ghostWidth+1) {
      d_shear_flux_xmin_remap[offset + arraySize*I_DENS] =
	d_shear_flux_xmin[offset + arraySize*I_DENS] +
	(1.0-eps) * d_shear_flux_xmax[offset_remap  +arraySize*I_DENS] + 
	eps       * d_shear_flux_xmax[offset_remapp1+arraySize*I_DENS];
      d_shear_flux_xmin_remap[offset + arraySize*I_DENS] *= HALF_F;
    }
    
    // emf
    d_emf[offset3d_xmin + arraySize3d*I_EMFY] += 
      (1.0-eps) * d_shear_flux_xmax[offset_remap  +arraySize*I_EMF_Y] + 
      eps       * d_shear_flux_xmax[offset_remapp1+arraySize*I_EMF_Y];
    d_emf[offset3d_xmin + arraySize3d*I_EMFY] *= HALF_F;
    
    /*
     * outer (i.e. xMax) boundary - flux and emf
     */
    jremap   = j      + jplus;
    jremapp1 = jremap + 1;
    eps      = epsi/dy;
    
    if (jremap   > ny+ghostWidth-1) jremap   -= ny;
    if (jremapp1 > ny+ghostWidth-1) jremapp1 -= ny;
    
    // physical memory offset
    offset         = j       + d_shear_pitch*k;
    offset_remap   = jremap  + d_shear_pitch*k;
    offset_remapp1 = jremapp1+ d_shear_pitch*k;

    // flux
    if (j>=ghostWidth and j<jsize-ghostWidth+1 and
	k>=ghostWidth and k<ksize-ghostWidth+1) {
      d_shear_flux_xmax_remap[offset+arraySize*I_DENS] =
	d_shear_flux_xmax[offset+arraySize*I_DENS] +
	(1.0-eps) * d_shear_flux_xmin[offset_remap  +arraySize*I_DENS] + 
	eps       * d_shear_flux_xmin[offset_remapp1+arraySize*I_DENS];
      d_shear_flux_xmax_remap[offset+arraySize*I_DENS] *= HALF_F;
    }
    
    // emf
    d_emf[offset3d_xmax + arraySize3d*I_EMFY] +=
      (1.0-eps) * d_shear_flux_xmin[offset_remap  +arraySize*I_EMF_Y] + 
      eps       * d_shear_flux_xmin[offset_remapp1+arraySize*I_EMF_Y];
    d_emf[offset3d_xmax + arraySize3d*I_EMFY] *= HALF_F;

  } // end flux/emf remapping  

} // kernel_remapping_mhd_3d

/**
 * Flux/EMF remapping kernel (3D) at XMIN border (MPI only).
 *
 * \param[in]  d_shear_flux_xmin_recv_glob buffer global received from MPI comm
 * \param[in]  d_shear_flux_xmin_toSend    buffer local used in MPI comm (send)
 * \param[out] d_shear_flux_xmin_remap buffer local with density flux and emfY remapped
 * \param[in]  pitch pitch in the 3d buffer
 * \param[in]  jsize
 * \param[in]  ksize
 * \param[in]  d_shear_pitch_global pitch for d_shear_flux_xmin_recv_glob
 * \param[in]  d_shear_pitch_local  pitch for d_shear_flux_xmin_remap
 * \param[in]  totalTime
 * \param[in]  dt
 */
__global__ void kernel_mhd_3d_flux_remapping_xmin(real_t* d_shear_flux_xmin_recv_glob,
						  real_t* d_shear_flux_xmin_toSend,
						  real_t* d_shear_flux_xmin_remap,
						  real_t* d_emf,
						  const int pitch,
						  const int isize,
						  const int jsize,
						  const int ksize,
						  const int d_shear_pitch_global,
						  const int d_shear_pitch_local,
						  const real_t totalTime,
						  const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int m = __mul24(bx, blockDim.x) + tx;
  const int n = __mul24(by, blockDim.y) + ty;

  // the following code is adapted from Dumses/bval_shear.f90, 
  // subroutines bval_shear_flux and bval_shear_emf
  real_t deltay,epsi,eps;
  int jplus;

  const real_t &xMin    = ::gParams.xMin;
  const real_t &yMin    = ::gParams.yMin;
  const real_t &xMax    = ::gParams.xMax;
  const real_t &yMax    = ::gParams.yMax;

  //const real_t &dx      = ::gParams.dx;
  const real_t &dy      = ::gParams.dy;

  //const    int &nx      = ::gParams.nx;
  const    int &ny      = ::gParams.ny;

  const    int &mpiPosY = ::gParams.mpiPosY;
  const    int &my      = ::gParams.my;

  const real_t &Omega0  = ::gParams.Omega0;

  deltay = 1.5 * Omega0 * (xMax - xMin) * (totalTime+dt/2);
  deltay = FMOD(deltay, (yMax - yMin) );
  jplus  = (int) (deltay/dy);
  epsi   = FMOD(deltay,  dy);

  // hard-coded for MHD only
  const int ghostWidth = 3;

  const int arraySize_loc  = d_shear_pitch_local  * ksize;
  const int arraySize_glob = d_shear_pitch_global * ksize;
  const int arraySize3d    = pitch*jsize          * ksize;

  /*
   * perform flux/emf remapping
   */
  const int &j=m;
  const int &k=n;
  
  if (j<jsize and k<ksize) {

   int jremap,jremapp1;
   int offset, offset_remap, offset_remapp1;
   int offset3d_xmin;

    /*
     * inner (i.e. XMIN) boundary - flux and emf
     */
    jremap   = j      + ny*mpiPosY - jplus - 1;
    jremapp1 = jremap + 1;
    eps      = 1.0-epsi/dy;
    
    if (jremap  < ghostWidth) jremap   += ny*my;
    if (jremapp1< ghostWidth) jremapp1 += ny*my;
    
    // physical memory offset
    offset         = j       + d_shear_pitch_local  * k;  // local
    offset_remap   = jremap  + d_shear_pitch_global * k;  // global
    offset_remapp1 = jremapp1+ d_shear_pitch_global * k;  // global
    offset3d_xmin  = ghostWidth + pitch * (j + jsize * k);

    // flux
    if (j>=ghostWidth and j<jsize-ghostWidth+1 and 
	k>=ghostWidth and k<ksize-ghostWidth+1) {
      d_shear_flux_xmin_remap[offset + arraySize_loc*I_DENS] =
	d_shear_flux_xmin_toSend[offset + arraySize_loc*I_DENS] +
	(1.0-eps) * d_shear_flux_xmin_recv_glob[offset_remap  +arraySize_glob*I_DENS] + 
	eps       * d_shear_flux_xmin_recv_glob[offset_remapp1+arraySize_glob*I_DENS];
      d_shear_flux_xmin_remap[offset + arraySize_loc*I_DENS] *= HALF_F;
    }
    
    // emf
    d_emf[offset3d_xmin + arraySize3d*I_EMFY] += 
      (1.0-eps) * d_shear_flux_xmin_recv_glob[offset_remap  +arraySize_glob*I_EMF_Y] + 
      eps       * d_shear_flux_xmin_recv_glob[offset_remapp1+arraySize_glob*I_EMF_Y];
    d_emf[offset3d_xmin + arraySize3d*I_EMFY] *= HALF_F;    

  } // end flux/emf remapping  

} // kernel_mhd_3d_flux_remapping_xmin

/**
 * Flux/EMF remapping kernel (3D) at XMAX border (MPI only).
 *
 * \param[in]  d_shear_flux_xmax_recv_glob buffer global received from MPI comm
 * \param[in]  d_shear_flux_xmax_toSend    buffer local used in MPI comm (send)
 * \param[out] d_shear_flux_xmax_remap buffer local with density flux and emfY remapped
 * \param[in]  pitch pitch in the 3d buffer
 * \param[in]  jsize
 * \param[in]  ksize
 * \param[in]  d_shear_pitch_global pitch for d_shear_flux_xmax_recv_glob
 * \param[in]  d_shear_pitch_local  pitch for d_shear_flux_xmax_remap
 * \param[in]  totalTime
 * \param[in]  dt
 */
__global__ void kernel_mhd_3d_flux_remapping_xmax(real_t* d_shear_flux_xmax_recv_glob,
						  real_t* d_shear_flux_xmax_toSend,
						  real_t* d_shear_flux_xmax_remap,
						  real_t* d_emf,
						  const int pitch,
						  const int isize,
						  const int jsize,
						  const int ksize,
						  const int d_shear_pitch_global,
						  const int d_shear_pitch_local,
						  const real_t totalTime,
						  const real_t dt)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int m = __mul24(bx, blockDim.x) + tx;
  const int n = __mul24(by, blockDim.y) + ty;

  // the following code is adapted from Dumses/bval_shear.f90, 
  // subroutines bval_shear_flux and bval_shear_emf
  real_t deltay,epsi,eps;
  int jplus;

  const real_t &xMin    = ::gParams.xMin;
  const real_t &yMin    = ::gParams.yMin;
  const real_t &xMax    = ::gParams.xMax;
  const real_t &yMax    = ::gParams.yMax;

  //const real_t &dx      = ::gParams.dx;
  const real_t &dy      = ::gParams.dy;

  const    int &nx      = ::gParams.nx;
  const    int &ny      = ::gParams.ny;

  const    int &mpiPosY = ::gParams.mpiPosY;
  const    int &my      = ::gParams.my;

  const real_t &Omega0  = ::gParams.Omega0;

  deltay = 1.5 * Omega0 * (xMax - xMin) * (totalTime+dt/2);
  deltay = FMOD(deltay, (yMax - yMin) );
  jplus  = (int) (deltay/dy);
  epsi   = FMOD(deltay,  dy);

  // hard-coded for MHD only
  const int ghostWidth = 3;

  const int arraySize_loc  = d_shear_pitch_local  * ksize;
  const int arraySize_glob = d_shear_pitch_global * ksize;
  const int arraySize3d    = pitch*jsize          * ksize;

  /*
   * perform flux/emf remapping
   */
  const int &j=m;
  const int &k=n;
  
  if (j<jsize and k<ksize) {

   int jremap,jremapp1;
   int offset, offset_remap, offset_remapp1;
   int offset3d_xmax;

    /*
     * outer (i.e. XMAX) boundary - flux and emf
     */
    jremap   = j      + ny*mpiPosY + jplus;
    jremapp1 = jremap + 1;
    eps      = epsi/dy;
    
    if (jremap   > my*ny+ghostWidth-1) jremap   -= ny*my;
    if (jremapp1 > my*ny+ghostWidth-1) jremapp1 -= ny*my;
    
    // physical memory offset
    offset         = j       + d_shear_pitch_local  * k;  // local
    offset_remap   = jremap  + d_shear_pitch_global * k;  // global
    offset_remapp1 = jremapp1+ d_shear_pitch_global * k;  // global
    offset3d_xmax  = nx + ghostWidth + pitch * (j + jsize * k);

    // flux
    if (j>=ghostWidth and j<jsize-ghostWidth+1 and
    	k>=ghostWidth and k<ksize-ghostWidth+1) {
      d_shear_flux_xmax_remap[offset + arraySize_loc*I_DENS] =
    	d_shear_flux_xmax_toSend[offset + arraySize_loc*I_DENS] +
    	(1.0-eps) * d_shear_flux_xmax_recv_glob[offset_remap  +arraySize_glob*I_DENS] + 
    	eps       * d_shear_flux_xmax_recv_glob[offset_remapp1+arraySize_glob*I_DENS];
      d_shear_flux_xmax_remap[offset + arraySize_loc*I_DENS] *= HALF_F;
    }
    
    // emf
    d_emf[offset3d_xmax + arraySize3d*I_EMFY] +=
      (1.0-eps) * d_shear_flux_xmax_recv_glob[offset_remap  +arraySize_glob*I_EMF_Y] + 
      eps       * d_shear_flux_xmax_recv_glob[offset_remapp1+arraySize_glob*I_EMF_Y];
    d_emf[offset3d_xmax + arraySize3d*I_EMFY] *= HALF_F;

  } // end flux/emf remapping  

} // kernel_mhd_3d_flux_remapping_xmax


/**
 * Update shear borders with remapped data (3D only).
 *
 * \param[in,out] Uout
 * \param[in]     d_shear_flux_xmin
 * \param[in]     d_shear_flux_xmax
 * \param[in]     pitch
 * \param[in]     isize
 * \param[in]     jsize
 * \param[in]     ksize
 * \param[in]     d_shear_pitch
 * \param[in]     totalTime
 * \param[in]     dt
 */
__global__ void kernel_update_shear_borders_3d(real_t *Uout,
					       real_t *d_shear_flux_xmin_remap,
					       real_t *d_shear_flux_xmax_remap,
					       const int pitch,
					       const int isize,
					       const int jsize,
					       const int ksize,
					       const int d_shear_pitch,
					       const real_t totalTime,
					       const real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int m = __mul24(bx, blockDim.x) + tx;
  const int n = __mul24(by, blockDim.y) + ty;

  // other geometry related constants
  //const real_t &dx = ::gParams.dx;
  //const real_t &dy = ::gParams.dy;
  //const real_t &dz = ::gParams.dz;

  //const real_t dtdx = dt/dx;
  //const real_t dtdy = dt/dy;
  //const real_t dtdz = dt/dz;

  const int &nx = ::gParams.nx;

  const int ghostWidth = 3;
  //const int arraySize3d = pitch * jsize * ksize;
  const int arraySize2d = d_shear_pitch * ksize;

  // update xMin/xMax borders with remapped flux/emf
  {
    const int &j=m;
    const int &k=n;

    if (j>=ghostWidth and j<jsize-ghostWidth+1 and
	k>=ghostWidth and k<ksize-ghostWidth+1) {

      int offset3d_jk   = pitch * (j   +         jsize * k);
      //int offset3d_jkm1 = pitch * (j   +         jsize * (k-1) );
      int offset2d_jk   =          j   + d_shear_pitch * k ;
      //int offset2d_jp1k =          j+1 + d_shear_pitch * k ;
      //int offset2d_jkp1 =          j   + d_shear_pitch * (k+1) ;


      // update density
      Uout[   ghostWidth  +offset3d_jk] += d_shear_flux_xmin_remap[offset2d_jk+I_DENS*arraySize2d];
      Uout[nx+ghostWidth-1+offset3d_jk] -= d_shear_flux_xmax_remap[offset2d_jk+I_DENS*arraySize2d];
            
      Uout[   ghostWidth  +offset3d_jk]  = FMAX(Uout[   ghostWidth  +offset3d_jk], gParams.smallr );
      Uout[nx+ghostWidth-1+offset3d_jk]  = FMAX(Uout[nx+ghostWidth-1+offset3d_jk], gParams.smallr );

    } // end if (guard on j and k)
    
  } // end update xMin/xMax borders with remapped flux/emf

} // kernel_update_shear_borders_3d

/**
 * Update xmin shear border with remapped density flux, only usefull in GPU+MPI.
 *
 * \param[in,out] Uout
 * \param[in]     d_shear_flux_remap XMIN or XMAX shear flux remapped data
 * \param[in]     pitch
 * \param[in]     isize
 * \param[in]     jsize
 * \param[in]     ksize
 * \param[in]     d_shear_pitch
 * \param[in]     totalTime
 * \param[in]     dt
 */
template<BoundaryLocation boundaryLoc>
__global__ void kernel_update_shear_border_3d(real_t *Uout,
					      real_t *d_shear_flux_remap,
					      const int pitch,
					      const int isize,
					      const int jsize,
					      const int ksize,
					      const int d_shear_pitch,
					      const real_t totalTime,
					      const real_t dt)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int m = __mul24(bx, blockDim.x) + tx;
  const int n = __mul24(by, blockDim.y) + ty;

  // other geometry related constants
  //const real_t &dx = ::gParams.dx;
  //const real_t &dy = ::gParams.dy;
  //const real_t &dz = ::gParams.dz;

  const int &nx = ::gParams.nx;

  const int ghostWidth = 3;
  //const int arraySize3d = pitch * jsize * ksize;
  const int arraySize2d = d_shear_pitch * ksize;

  // update xMin borders with remapped flux/emf
  {
    const int &j=m;
    const int &k=n;

    if (j>=ghostWidth and j<jsize-ghostWidth+1 and
	k>=ghostWidth and k<ksize-ghostWidth+1) {

      int offset3d_jk   = pitch * (j   +         jsize * k);
      int offset2d_jk   =          j   + d_shear_pitch * k ;


      // update density
      if (boundaryLoc == XMIN) {
	Uout[   ghostWidth  +offset3d_jk] += d_shear_flux_remap[offset2d_jk+I_DENS*arraySize2d];
	Uout[   ghostWidth  +offset3d_jk]  = FMAX(Uout[   ghostWidth  +offset3d_jk], gParams.smallr );
      }
      if (boundaryLoc == XMAX) {
	Uout[nx+ghostWidth-1+offset3d_jk] -= d_shear_flux_remap[offset2d_jk+I_DENS*arraySize2d];
	Uout[nx+ghostWidth-1+offset3d_jk]  = FMAX(Uout[nx+ghostWidth-1+offset3d_jk], gParams.smallr );
      }
            
    } // end if (guard on j and k)
    
  } // end update xMin border with remapped flux/emf

} // kernel_update_shear_border_3d



#endif // SHEARING_BOX_UTILS_CUH_
