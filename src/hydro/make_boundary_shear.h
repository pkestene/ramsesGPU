/**
 * \file make_boundary_shear.h
 * \brief Provides several routines of boundary conditions for shearing box.
 *
 * \date 23 Feb 2012
 * \author P. Kestener
 *
 * $Id: make_boundary_shear.h 3450 2014-06-16 22:03:23Z pkestene $
 */
#ifndef MAKE_BOUNDARY_SHEAR_H_
#define MAKE_BOUNDARY_SHEAR_H_

#include "real_type.h"
#include "common_types.h"
#include "constants.h"
#include "gpu_macros.h"


#ifdef __CUDACC__
/*
 *
 * Shearing box cuda kernels
 *
 */

/**
 * \brief Compute Y-slopes in shear borders (XMIN and XMAX).
 * 
 * \param[out] d_shear_slope_xmin
 * \param[out] d_shear_slope_xmax
 * \param[in]  d_shear_border_xmin
 * \param[in]  d_shear_border_xmax
 * \param[in]  pitch Memory pitch along X direction
 * \param[in]  isize physical size along X (ghost cells included)
 * \param[in]  jsize physical size along Y (ghost cells included)
 * \param[in]  ksize physical size along Z (ghost cells included)
 *
 */
__GLOBAL__ void kernel_compute_shear_border_slopes(real_t* d_shear_slope_xmin, 
						   real_t* d_shear_slope_xmax,
						   real_t* d_shear_border_xmin, 
						   real_t* d_shear_border_xmax, 
						   int pitch, 
						   int isize, 
						   int jsize, 
						   int ksize,
						   int arraySize,
						   int ghostWidth)
{
  // use index j, k as we are dealing XMIN, XMAX borders
  // just to make it clear...

  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int j  = bx * blockDim.x + tx;
  
  const int by = blockIdx.y;
  const int ty = threadIdx.y;
  const int k  = by * blockDim.y + ty;
  
  int &nbVar         = ::gParams.nbVar;
  real_t &slope_type = ::gParams.slope_type;

  real_t dsgn, dlim, dcen, dlft, drgt, slop;

  if (slope_type >= 2 or slope_type <= 3) {
    
    if (j > 0 and 
	j < jsize-1 and
	k < ksize)
      {
	
	for (int i=0; i<ghostWidth; i++) {
	  
	  int ijk   = i + pitch*(j  +jsize*k);
	  int ijp1k = i + pitch*(j+1+jsize*k);
	  int ijm1k = i + pitch*(j-1+jsize*k);
	  
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    
	    if (iVar==IB) { // special treatment for Y-slope of BY
	      
	      // inner (XMIN) BY slope
	      d_shear_slope_xmin   [ijk  +IB*arraySize] = 
		d_shear_border_xmin[ijp1k+IB*arraySize] - 
		d_shear_border_xmin[ijk  +IB*arraySize];
	      
	      // outer (XMAX) BY slope
	      d_shear_slope_xmax   [ijk  +IB*arraySize] = 
		d_shear_border_xmax[ijp1k+IB*arraySize] - 
		d_shear_border_xmax[ijk  +IB*arraySize];
	      
	    } else { // all other components except BY
	      
	      // inner (XMIN) slopes in second coordinate direction 
	      dlft = slope_type * ( d_shear_border_xmin[ijk  +iVar*arraySize] - 
				    d_shear_border_xmin[ijm1k+iVar*arraySize] );
	      drgt = slope_type * ( d_shear_border_xmin[ijp1k+iVar*arraySize] - 
				    d_shear_border_xmin[ijk  +iVar*arraySize] );
	      dcen = HALF_F * (dlft+drgt)/slope_type;
	      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
	      slop = FMIN( FABS(dlft), FABS(drgt) );
	      dlim = slop;
	      if ( (dlft*drgt) <= ZERO_F )
		dlim = ZERO_F;
	      d_shear_slope_xmin[ijk+iVar*arraySize] = dsgn * FMIN( dlim, FABS(dcen) );
	      
	      // outer (XMAX) slopes in second coordinate direction 
	      dlft = slope_type * ( d_shear_border_xmax[ijk  +iVar*arraySize] - 
				    d_shear_border_xmax[ijm1k+iVar*arraySize] );
	      drgt = slope_type * ( d_shear_border_xmax[ijp1k+iVar*arraySize] - 
				    d_shear_border_xmax[ijk  +iVar*arraySize] );
	      dcen = HALF_F * (dlft+drgt)/slope_type;
	      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
	      slop = FMIN( FABS(dlft), FABS(drgt) );
	      dlim = slop;
	      if ( (dlft*drgt) <= ZERO_F )
		dlim = ZERO_F;
	      d_shear_slope_xmax[ijk+iVar*arraySize] = dsgn * FMIN( dlim, FABS(dcen) );
	      
	    } // end if (iVar==IB)
	  } // end for iVar
	  
	} // end for i
	
      } // end slopes computation

  } // end if (slope_type == 2 or slope_type == 3)

} // kernel_compute_shear_border_slopes

/**
 * \brief Compute Y-slopes in shear borders (XMIN and XMAX).
 * 
 * \param[out] U
 * \param[in]  uPitch Memory pitch along X direction
 * \param[in]  uIsize physical size along X (ghost cells included)
 * \param[in]  uJsize physical size along Y (ghost cells included)
 * \param[in]  uKsize physical size along Z (ghost cells included)
 * \param[in]  uArraySize
 * \param[in]  d_shear_slope_xmin
 * \param[in]  d_shear_slope_xmax
 * \param[in]  d_shear_border_xmin
 * \param[in]  d_shear_border_xmax
 * \param[in]  bPitch Memory pitch along X direction
 * \param[in]  bIsize physical size along X (ghost cells included)
 * \param[in]  bJsize physical size along Y (ghost cells included)
 * \param[in]  bKsize physical size along Z (ghost cells included)
 * \param[in]  bArraySize
 * \param[in]  ghostWidth
 *
 * Please note that in the MPI case border buffer are sized along Y-axis up to global Y-size !!
 *
 */
__GLOBAL__ void kernel_perform_final_remapping_shear_borders(real_t* U,
							     int uPitch, 
							     int uIsize, 
							     int uJsize, 
							     int uKsize,
							     int uArraySize,
							     real_t* d_shear_slope_xmin, 
							     real_t* d_shear_slope_xmax,
							     real_t* d_shear_border_xmin, 
							     real_t* d_shear_border_xmax, 
							     int bPitch, 
							     int bIsize, 
							     int bJsize, 
							     int bKsize,
							     int bArraySize,
							     int ghostWidth,
							     real_t totalTime)
{
  // use index j, k as we are dealing XMIN, XMAX borders
  // just to make it clear...

  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int j  = bx * blockDim.x + tx;
  
  const int by = blockIdx.y;
  const int ty = threadIdx.y;
  const int k  = by * blockDim.y + ty;
  
  int &nbVar   = ::gParams.nbVar;

  int &nx      = ::gParams.nx;
  int &ny      = ::gParams.ny;

  real_t &dx      = ::gParams.dx;
  real_t &dy      = ::gParams.dy;

  real_t &Omega0  = ::gParams.Omega0;

  real_t deltay,epsi,eps,lambda;
  int jplus,jremap,jremapp1;
  
  deltay = 1.5 * Omega0 * (dx * nx) * (totalTime);
  deltay = FMOD(deltay, (dy * ny) );
  jplus  = (int) (deltay/dy);
  epsi   = FMOD(deltay,  dy);
  
  if (j>=ghostWidth and j<uJsize-ghostWidth and k<uKsize)
    {

      int ijk_U;
      //int ijk_b;
      int ijk_bremap;
      int ijk_bremapp1;

      /*
       * inner (XMIN) border
       */
      jremap   = j     -jplus-1;
      jremapp1 = jremap+1; 
      eps      = 1.0-epsi/dy;
      
      if (jremap  < ghostWidth) jremap   += ny;
      if (jremapp1< ghostWidth) jremapp1 += ny;
      
      lambda = HALF_F * eps*(eps-1.0);
            
      for (int i=0; i<ghostWidth; i++) {

	ijk_U        = i + uPitch * ( j        + uJsize * k);
	ijk_bremap   = i + bPitch * ( jremap   + bJsize * k);
	ijk_bremapp1 = i + bPitch * ( jremapp1 + bJsize * k);

	for (int iVar=0; iVar<nbVar; iVar++) {
	  
	  if (iVar == IB) { // special treatment for BY magnetic field component
	    
	    U[ijk_U+IB*uArraySize] = 
	      d_shear_border_xmax   [ijk_bremap+IB*bArraySize] + 
	      eps*d_shear_slope_xmax[ijk_bremap+IB*bArraySize];
	    
	  } else { // other components
	    
	    U[ijk_U+iVar*uArraySize] = 
	      (1.0-eps) * d_shear_border_xmax [ijk_bremap  +iVar*bArraySize] +
	      eps       * d_shear_border_xmax [ijk_bremapp1+iVar*bArraySize] + 
	      lambda    * ( d_shear_slope_xmax[ijk_bremap  +iVar*bArraySize] - 
			    d_shear_slope_xmax[ijk_bremapp1+iVar*bArraySize] );
	    
	  } // end if (iVar == IB)
	} // end for iVar
      } // end for i
      
      /*
       * outer (XMAX) border
       */
      jremap   = j     +jplus;
      jremapp1 = jremap+1;
      eps      = epsi/dy;
      
      if (jremap   > ny+ghostWidth-1) jremap   -= ny;
      if (jremapp1 > ny+ghostWidth-1) jremapp1 -= ny;
	  
      lambda = HALF_F * eps*(eps-1.0);
      
      for (int i=0; i<ghostWidth; i++) {

	ijk_U        = nx+ghostWidth+i + uPitch * ( j        + uJsize * k);
	ijk_bremap   =               i + bPitch * ( jremap   + bJsize * k);
	ijk_bremapp1 =               i + bPitch * ( jremapp1 + bJsize * k);

	for (int iVar=0; iVar<nbVar; iVar++) {
	  
	  // update Hydro variables in ghost cells
	  if (iVar < 5) {
	    U[ijk_U+iVar*uArraySize] = 
	      (1.0-eps) * d_shear_border_xmin [ijk_bremap  +iVar*bArraySize] + 
	      eps       * d_shear_border_xmin [ijk_bremapp1+iVar*bArraySize] + 
	      lambda    * ( d_shear_slope_xmin[ijk_bremapp1+iVar*bArraySize] - 
			    d_shear_slope_xmin[ijk_bremap  +iVar*bArraySize] );
	  }
	  if (iVar == IA) { // ! WARNING : do NOT write in first outer ghost cell
	    if (i>0) {
	      U[ijk_U+IA*uArraySize] = 
		(1.0-eps) * d_shear_border_xmin [ijk_bremap  +IA*bArraySize] + 
		eps       * d_shear_border_xmin [ijk_bremapp1+IA*bArraySize] + 
		lambda    * ( d_shear_slope_xmin[ijk_bremapp1+IA*bArraySize] - 
			      d_shear_slope_xmin[ijk_bremap  +IA*bArraySize] );
	    }
	  }
	  if (iVar == IB) {
	    U[ijk_U+IB*uArraySize] = 
	      d_shear_border_xmin     [ijk_bremap+IB*bArraySize] + 
	      eps * d_shear_slope_xmin[ijk_bremap+IB*bArraySize];
	  }
	  if (iVar == IC) {
	    U[ijk_U+IC*uArraySize] = 
	      (1.0-eps) * d_shear_border_xmin [ijk_bremap  +IC*bArraySize] + 
	      eps       * d_shear_border_xmin [ijk_bremapp1+IC*bArraySize] + 
	      lambda    * ( d_shear_slope_xmin[ijk_bremapp1+IC*bArraySize] - 
			    d_shear_slope_xmin[ijk_bremap  +IC*bArraySize] );
	  }
	  
	} // end for iVar      
      } // end for i
      
    } // end remapping shear borders

} // kernel_perform_final_remapping_shear_borders

/**
 * \brief Compute Y-slopes in shear borders (XMIN and XMAX).
 * 
 * \param[out] U
 * \param[in]  uPitch Memory pitch along X direction
 * \param[in]  uIsize physical size along X (ghost cells included)
 * \param[in]  uJsize physical size along Y (ghost cells included)
 * \param[in]  uKsize physical size along Z (ghost cells included)
 * \param[in]  uArraySize
 * \param[in]  d_shear_slope
 * \param[in]  d_shear_border
 * \param[in]  bPitch Memory pitch along X direction
 * \param[in]  bIsize physical size along X (ghost cells included)
 * \param[in]  bJsize physical size along Y (ghost cells included)
 * \param[in]  bKsize physical size along Z (ghost cells included)
 * \param[in]  bArraySize
 * \param[in]  ghostWidth
 *
 * Please note that in the MPI case border buffer are sized along Y-axis up to global Y-size !!
 *
 */
template <BoundaryLocation boundaryLoc>
__GLOBAL__ void kernel_perform_final_remapping_shear_border_mpi(real_t* U,
								int uPitch, 
								int uIsize, 
								int uJsize, 
								int uKsize,
								int uArraySize,
								real_t* d_shear_slope, 
								real_t* d_shear_border, 
								int bPitch, 
								int bIsize, 
								int bJsize, 
								int bKsize,
								int bArraySize,
								int ghostWidth,
								real_t totalTime)
{
  // use index j, k as we are dealing XMIN, XMAX borders
  // just to make it clear...

  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int j  = bx * blockDim.x + tx;
  
  const int by = blockIdx.y;
  const int ty = threadIdx.y;
  const int k  = by * blockDim.y + ty;
  
  const int    &nbVar    = ::gParams.nbVar;

  const int    &nx       = ::gParams.nx;
  const int    &ny       = ::gParams.ny;

  //const int    &mx       = ::gParams.mx;
  const int    &my       = ::gParams.my;

  //const real_t &dx       = ::gParams.dx;
  const real_t &dy       = ::gParams.dy;

  const real_t &Omega0   = ::gParams.Omega0;

  const real_t &xMin     = ::gParams.xMin;
  const real_t &yMin     = ::gParams.yMin;
  const real_t &xMax     = ::gParams.xMax;
  const real_t &yMax     = ::gParams.yMax;

  //const int    &mpiPosX  = ::gParams.mpiPosX;
  const int    &mpiPosY  = ::gParams.mpiPosY;
  //const int    &mpiPosZ  = ::gParams.mpiPosZ;

  real_t deltay,epsi,eps,lambda;
  int jplus,jremap,jremapp1;
  
  deltay = 1.5 * Omega0 * (xMax - xMin) * (totalTime);
  deltay = FMOD(deltay, (yMax - yMin) );
  jplus  = (int) (deltay/dy);
  epsi   = FMOD(deltay,  dy);
  
  if (j>=ghostWidth and j<uJsize-ghostWidth and k<uKsize)
    {

      int ijk_U;
      //int ijk_b;
      int ijk_bremap;
      int ijk_bremapp1;


      if (boundaryLoc == XMIN) {

	/*
	 * inner (XMIN) border
	 */
	jremap   = j      + ny*mpiPosY - jplus - 1;
	jremapp1 = jremap + 1; 
	eps      = 1.0-epsi/dy;
	
	if (jremap  < ghostWidth) jremap   += ny*my;
	if (jremapp1< ghostWidth) jremapp1 += ny*my;
	
	lambda = HALF_F * eps*(eps-1.0);
	
	for (int i=0; i<ghostWidth; i++) {
	  
	  ijk_U        = i + uPitch * ( j        + uJsize * k);
	  ijk_bremap   = i + bPitch * ( jremap   + bJsize * k);
	  ijk_bremapp1 = i + bPitch * ( jremapp1 + bJsize * k);
	  
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    
	    if (iVar == IB) { // special treatment for BY magnetic field component
	      
	      U[ijk_U+IB*uArraySize] = 
		d_shear_border   [ijk_bremap+IB*bArraySize] + 
		eps*d_shear_slope[ijk_bremap+IB*bArraySize];
	      
	    } else { // other components
	      
	      U[ijk_U+iVar*uArraySize] = 
		(1.0-eps) * d_shear_border [ijk_bremap  +iVar*bArraySize] +
		eps       * d_shear_border [ijk_bremapp1+iVar*bArraySize] + 
		lambda    * ( d_shear_slope[ijk_bremap  +iVar*bArraySize] - 
			      d_shear_slope[ijk_bremapp1+iVar*bArraySize] );
	      
	    } // end if (iVar == IB)
	  } // end for iVar
	} // end for i

      } // end if (boundaryLoc == XMIN)


      if (boundaryLoc == XMAX) {

	/*
	 * outer (XMAX) border
	 */
	jremap   = j      + ny*mpiPosY + jplus;
	jremapp1 = jremap + 1;
	eps      = epsi/dy;
	
	if (jremap   > my*ny+ghostWidth-1) jremap   -= ny*my;
	if (jremapp1 > my*ny+ghostWidth-1) jremapp1 -= ny*my;
	
	lambda = HALF_F * eps*(eps-1.0);
	
	for (int i=0; i<ghostWidth; i++) {
	  
	  ijk_U        = nx+ghostWidth+i + uPitch * ( j        + uJsize * k);
	  ijk_bremap   =               i + bPitch * ( jremap   + bJsize * k);
	  ijk_bremapp1 =               i + bPitch * ( jremapp1 + bJsize * k);
	  
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    
	    // update Hydro variables in ghost cells
	    if (iVar < 5) {
	      U[ijk_U+iVar*uArraySize] = 
		(1.0-eps) * d_shear_border [ijk_bremap  +iVar*bArraySize] + 
		eps       * d_shear_border [ijk_bremapp1+iVar*bArraySize] + 
		lambda    * ( d_shear_slope[ijk_bremapp1+iVar*bArraySize] - 
			      d_shear_slope[ijk_bremap  +iVar*bArraySize] );
	    }
	    if (iVar == IA) { // ! WARNING : do NOT write in first outer ghost cell
	      if (i>0) {
		U[ijk_U+IA*uArraySize] = 
		  (1.0-eps) * d_shear_border [ijk_bremap  +IA*bArraySize] + 
		  eps       * d_shear_border [ijk_bremapp1+IA*bArraySize] + 
		  lambda    * ( d_shear_slope[ijk_bremapp1+IA*bArraySize] - 
				d_shear_slope[ijk_bremap  +IA*bArraySize] );
	      }
	    }
	    if (iVar == IB) {
	      U[ijk_U+IB*uArraySize] = 
		d_shear_border     [ijk_bremap+IB*bArraySize] + 
		eps * d_shear_slope[ijk_bremap+IB*bArraySize];
	    }
	    if (iVar == IC) {
	      U[ijk_U+IC*uArraySize] = 
		(1.0-eps) * d_shear_border [ijk_bremap  +IC*bArraySize] + 
		eps       * d_shear_border [ijk_bremapp1+IC*bArraySize] + 
		lambda    * ( d_shear_slope[ijk_bremapp1+IC*bArraySize] - 
			      d_shear_slope[ijk_bremap  +IC*bArraySize] );
	    }
	    
	  } // end for iVar      
	} // end for i

      } // end if (boundaryLoc == XMAX)
      
    } // end remapping shear borders

} // kernel_perform_final_remapping_shear_border_mpi

#endif // __CUDACC__

#endif // MAKE_BOUNDARY_SHEAR_H_
