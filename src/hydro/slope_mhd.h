/**
 * \file slope_mhd.h
 * \brief Compute primitive variables slope dq from a given q state and its
 * adjacent neighbors.
 *
 * \author Pierre Kestener
 * \date March 31, 2011
 *
 * $Id: slope_mhd.h 2427 2012-09-26 15:02:51Z pkestene $
 */
#ifndef SLOPE_MHD_H_
#define SLOPE_MHD_H_

#include "real_type.h"
#include "constants.h"

// =======================================================
// =======================================================
// MHD slope routines
// =======================================================
// =======================================================

/* some dummy utility routines */
__DEVICE__ 
real_t FMAX9_(real_t a0, real_t a1, real_t a2, 
	      real_t a3, real_t a4, real_t a5,
	      real_t a6, real_t a7, real_t a8) 
{
  real_t returnVal = a0;
  returnVal = ( a1 > returnVal) ? a1 : returnVal;
  returnVal = ( a2 > returnVal) ? a2 : returnVal;
  returnVal = ( a3 > returnVal) ? a3 : returnVal;
  returnVal = ( a4 > returnVal) ? a4 : returnVal;
  returnVal = ( a5 > returnVal) ? a5 : returnVal;
  returnVal = ( a6 > returnVal) ? a6 : returnVal;
  returnVal = ( a7 > returnVal) ? a7 : returnVal;
  returnVal = ( a8 > returnVal) ? a8 : returnVal;

  return returnVal;
} // FMAX9_

__DEVICE__ 
real_t FMIN9_(real_t a0, real_t a1, real_t a2, 
	      real_t a3, real_t a4, real_t a5,
	      real_t a6, real_t a7, real_t a8)
{
  real_t returnVal = a0;
  returnVal = ( a1 < returnVal) ? a1 : returnVal;
  returnVal = ( a2 < returnVal) ? a2 : returnVal;
  returnVal = ( a3 < returnVal) ? a3 : returnVal;
  returnVal = ( a4 < returnVal) ? a4 : returnVal;
  returnVal = ( a5 < returnVal) ? a5 : returnVal;
  returnVal = ( a6 < returnVal) ? a6 : returnVal;
  returnVal = ( a7 < returnVal) ? a7 : returnVal;
  returnVal = ( a8 < returnVal) ? a8 : returnVal;

  return returnVal;
} // FMIN9_

/**
 * Compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1,2 and 3.
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 * Loosely adapted from RAMSES/hydro/umuscl.f90: subroutine uslope
 * Interface is changed to become cellwise.
 * Only slope_type 1 and 2 are supported.
 *
 * \param[in]  qNb     : array to primitive variable vector state in the neighborhood
 * \param[out] dq      : reference to an array returning the X and Y slopes
 *
 * 
 */
__DEVICE__ 
void slope_unsplit_hydro_2d(real_t qNb[3][3][NVAR_MHD],
			    real_t (&dq)[2][NVAR_MHD])
{			
  // index of current cell in the neighborhood
  enum {CENTER=1};

  // aliases to input qState neighbors
  real_t (&q      )[NVAR_MHD] = qNb[CENTER  ][CENTER  ];
  real_t (&qPlusX )[NVAR_MHD] = qNb[CENTER+1][CENTER  ];
  real_t (&qMinusX)[NVAR_MHD] = qNb[CENTER-1][CENTER  ];
  real_t (&qPlusY )[NVAR_MHD] = qNb[CENTER  ][CENTER+1]; 
  real_t (&qMinusY)[NVAR_MHD] = qNb[CENTER  ][CENTER-1];

  real_t (&dqX)[NVAR_MHD] = dq[IX];
  real_t (&dqY)[NVAR_MHD] = dq[IY];
  
  if (::gParams.slope_type==0) {
    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {
      dqX[nVar] = ZERO_F;
      dqY[nVar] = ZERO_F;
    }
    return;
  }

  if (::gParams.slope_type==1 or ::gParams.slope_type==2) {  // minmod or average

    real_t dlft, drgt, dcen, dsgn, slop, dlim;
    
    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {

      // slopes in first coordinate direction
      dlft = ::gParams.slope_type*(q     [nVar] - qMinusX[nVar]);
      drgt = ::gParams.slope_type*(qPlusX[nVar] - q      [nVar]);
      dcen = HALF_F * (qPlusX[nVar] - qMinusX[nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqX[nVar] = dsgn * FMIN( dlim, FABS(dcen) );
      
      // slopes in second coordinate direction
      dlft = ::gParams.slope_type*(q     [nVar] - qMinusY[nVar]);
      drgt = ::gParams.slope_type*(qPlusY[nVar] - q      [nVar]);
      dcen = HALF_F * (qPlusY[nVar] - qMinusY[nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqY[nVar] = dsgn * FMIN( dlim, FABS(dcen) );

    } // end for nVar

  } else if (::gParams.slope_type == 3) {
    
    real_t slop, dlim;
    real_t dfll, dflm, dflr, dfml, dfmm, dfmr, dfrl, dfrm, dfrr;
    real_t vmin, vmax;
    real_t dfx, dfy, dff;

    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {
    
      dfll = qNb[CENTER-1][CENTER-1][nVar]-qNb[CENTER][CENTER][nVar];
      dflm = qNb[CENTER-1][CENTER  ][nVar]-qNb[CENTER][CENTER][nVar];
      dflr = qNb[CENTER-1][CENTER+1][nVar]-qNb[CENTER][CENTER][nVar];
      dfml = qNb[CENTER  ][CENTER-1][nVar]-qNb[CENTER][CENTER][nVar];
      dfmm = qNb[CENTER  ][CENTER  ][nVar]-qNb[CENTER][CENTER][nVar];
      dfmr = qNb[CENTER  ][CENTER+1][nVar]-qNb[CENTER][CENTER][nVar];
      dfrl = qNb[CENTER+1][CENTER-1][nVar]-qNb[CENTER][CENTER][nVar];
      dfrm = qNb[CENTER+1][CENTER  ][nVar]-qNb[CENTER][CENTER][nVar];
      dfrr = qNb[CENTER+1][CENTER+1][nVar]-qNb[CENTER][CENTER][nVar];
      
      vmin = FMIN9_(dfll,dflm,dflr,dfml,dfmm,dfmr,dfrl,dfrm,dfrr);
      vmax = FMAX9_(dfll,dflm,dflr,dfml,dfmm,dfmr,dfrl,dfrm,dfrr);
	
      dfx  = HALF_F * (qNb[CENTER+1][CENTER  ][nVar] - qNb[CENTER-1][CENTER  ][nVar]);
      dfy  = HALF_F * (qNb[CENTER  ][CENTER+1][nVar] - qNb[CENTER  ][CENTER-1][nVar]);
      dff  = HALF_F * (FABS(dfx) + FABS(dfy));
	
      if (dff>ZERO_F) {
	slop = FMIN(ONE_F, FMIN(FABS(vmin), FABS(vmax))/dff);
      } else {
	slop = ONE_F;
      }
      
      dlim = slop;
      
      dqX[nVar] = dlim*dfx;
      dqY[nVar] = dlim*dfy;
      
    } // end for nVar
  
  } // end slope_type
  
} // slope_unsplit_hydro_2d

/**
 * Compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1,2 and 3.
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 * Loosely adapted from RAMSES/hydro/umuscl.f90: subroutine uslope
 * Interface is changed to become cellwise.
 * Only slope_type 1 and 2 are supported.
 *
 * \param[in]  q       : current primitive variable state
 * \param[in]  qPlusX  : state in the next neighbor cell along XDIR
 * \param[in]  qMinusX : state in the previous neighbor cell along XDIR
 * \param[in]  qPlusY  : state in the next neighbor cell along YDIR
 * \param[in]  qMinusY : state in the previous neighbor cell along YDIR
 *
 * \param[out] dq      : reference to an array returning the X and Y slopes
 *
 * 
 */
__DEVICE__ 
void slope_unsplit_hydro_2d_simple(real_t q[NVAR_MHD],
				   real_t qPlusX[NVAR_MHD], 
				   real_t qMinusX[NVAR_MHD],
				   real_t qPlusY[NVAR_MHD], 
				   real_t qMinusY[NVAR_MHD],
				   real_t (&dq)[2][NVAR_MHD])
{

  real_t (&dqX)[NVAR_MHD] = dq[IX];
  real_t (&dqY)[NVAR_MHD] = dq[IY];
  
  if (::gParams.slope_type==0) {
    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {
      dqX[nVar] = ZERO_F;
      dqY[nVar] = ZERO_F;
    }
    return;
  }

  if (::gParams.slope_type==1 or ::gParams.slope_type==2) {  // minmod or average

    real_t dlft, drgt, dcen, dsgn, slop, dlim;
    
    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {

      // slopes in first coordinate direction
      dlft = ::gParams.slope_type*(q     [nVar] - qMinusX[nVar]);
      drgt = ::gParams.slope_type*(qPlusX[nVar] - q      [nVar]);
      dcen = HALF_F * (qPlusX[nVar] - qMinusX[nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqX[nVar] = dsgn * FMIN( dlim, FABS(dcen) );
      
      // slopes in second coordinate direction
      dlft = ::gParams.slope_type*(q     [nVar] - qMinusY[nVar]);
      drgt = ::gParams.slope_type*(qPlusY[nVar] - q      [nVar]);
      dcen = HALF_F * (qPlusY[nVar] - qMinusY[nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqY[nVar] = dsgn * FMIN( dlim, FABS(dcen) );

    } // end for nVar

  } // end slope_type
  
} // slope_unsplit_hydro_2d_simple

/**
 * Compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1,2 and 3.
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 * Loosely adapted from RAMSES/hydro/umuscl.f90: subroutine uslope
 * Interface is changed to become cellwise.
 *
 * \param[in]  qNb     : array to primitive variable vector state in the neighborhood
 * \param[out] dq      : reference to an array returning the X, Y and Z slopes
 *
 * \note actually a 3x3x3 neighborhood is only needed for slope_type=3
 * 
 */
__DEVICE__ 
void slope_unsplit_hydro_3d(real_t qNb[3][3][3][NVAR_MHD],
			    real_t (&dq)[3][NVAR_MHD])
{			
  // index of current cell in the neighborhood
  enum {CENTER=1};

  real_t (&dqX)[NVAR_MHD] = dq[IX];
  real_t (&dqY)[NVAR_MHD] = dq[IY];
  real_t (&dqZ)[NVAR_MHD] = dq[IZ];

  if (::gParams.slope_type==0) {
    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {
      dqX[nVar] = ZERO_F;
      dqY[nVar] = ZERO_F;
      dqZ[nVar] = ZERO_F;
    }
    return;
  }


  if (::gParams.slope_type==1 or ::gParams.slope_type==2) {  // minmod or average

    real_t dlft, drgt, dcen, dsgn, slop, dlim;
    
    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {

      // slopes in first coordinate direction
      dlft = ::gParams.slope_type*(qNb[CENTER  ][CENTER][CENTER][nVar] -
				   qNb[CENTER-1][CENTER][CENTER][nVar]);
      drgt = ::gParams.slope_type*(qNb[CENTER+1][CENTER][CENTER][nVar] - 
				   qNb[CENTER  ][CENTER][CENTER][nVar]);
      dcen = HALF_F * (qNb[CENTER+1][CENTER][CENTER][nVar] -
		       qNb[CENTER-1][CENTER][CENTER][nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqX[nVar] = dsgn * FMIN( dlim, FABS(dcen) );

      // slopes in second coordinate direction
      dlft = ::gParams.slope_type*(qNb[CENTER][CENTER  ][CENTER][nVar] -
				   qNb[CENTER][CENTER-1][CENTER][nVar]);
      drgt = ::gParams.slope_type*(qNb[CENTER][CENTER+1][CENTER][nVar] - 
				   qNb[CENTER][CENTER  ][CENTER][nVar]);
      dcen = HALF_F * (qNb[CENTER][CENTER+1][CENTER][nVar] -
		       qNb[CENTER][CENTER-1][CENTER][nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqY[nVar] = dsgn * FMIN( dlim, FABS(dcen) );

      // slopes in third coordinate direction
      dlft = ::gParams.slope_type*(qNb[CENTER][CENTER][CENTER  ][nVar] -
				   qNb[CENTER][CENTER][CENTER-1][nVar]);
      drgt = ::gParams.slope_type*(qNb[CENTER][CENTER][CENTER+1][nVar] - 
				   qNb[CENTER][CENTER][CENTER  ][nVar]);
      dcen = HALF_F * (qNb[CENTER][CENTER][CENTER+1][nVar] -
		       qNb[CENTER][CENTER][CENTER-1][nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqZ[nVar] = dsgn * FMIN( dlim, FABS(dcen) );

    } // end for nVar

  } else if (::gParams.slope_type == 3) {

    real_t slop, dlim;
    real_t dflll, dflml, dflrl, dfmll, dfmml, dfmrl, dfrll, dfrml, dfrrl;
    real_t dfllm, dflmm, dflrm, dfmlm, dfmmm, dfmrm, dfrlm, dfrmm, dfrrm;
    real_t dfllr, dflmr, dflrr, dfmlr, dfmmr, dfmrr, dfrlr, dfrmr, dfrrr;
    real_t vmin, vmax;
    real_t dfx, dfy, dfz, dff;

    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {

      dflll = qNb[CENTER-1][CENTER-1][CENTER-1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dflml = qNb[CENTER-1][CENTER  ][CENTER-1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dflrl = qNb[CENTER-1][CENTER+1][CENTER-1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfmll = qNb[CENTER  ][CENTER-1][CENTER-1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfmml = qNb[CENTER  ][CENTER  ][CENTER-1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfmrl = qNb[CENTER  ][CENTER+1][CENTER-1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfrll = qNb[CENTER+1][CENTER-1][CENTER-1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfrml = qNb[CENTER+1][CENTER  ][CENTER-1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfrrl = qNb[CENTER+1][CENTER+1][CENTER-1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
 
      dfllm = qNb[CENTER-1][CENTER-1][CENTER  ][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dflmm = qNb[CENTER-1][CENTER  ][CENTER  ][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dflrm = qNb[CENTER-1][CENTER+1][CENTER  ][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfmlm = qNb[CENTER  ][CENTER-1][CENTER  ][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfmmm = qNb[CENTER  ][CENTER  ][CENTER  ][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfmrm = qNb[CENTER  ][CENTER+1][CENTER  ][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfrlm = qNb[CENTER+1][CENTER-1][CENTER  ][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfrmm = qNb[CENTER+1][CENTER  ][CENTER  ][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfrrm = qNb[CENTER+1][CENTER+1][CENTER  ][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
 
      dfllr = qNb[CENTER-1][CENTER-1][CENTER+1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dflmr = qNb[CENTER-1][CENTER  ][CENTER+1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dflrr = qNb[CENTER-1][CENTER+1][CENTER+1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfmlr = qNb[CENTER  ][CENTER-1][CENTER+1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfmmr = qNb[CENTER  ][CENTER  ][CENTER+1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfmrr = qNb[CENTER  ][CENTER+1][CENTER+1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfrlr = qNb[CENTER+1][CENTER-1][CENTER+1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfrmr = qNb[CENTER+1][CENTER  ][CENTER+1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
      dfrrr = qNb[CENTER+1][CENTER+1][CENTER+1][nVar]-qNb[CENTER][CENTER][CENTER][nVar];
 
      vmin =             FMIN9_(dflll,dflml,dflrl,dfmll,dfmml,dfmrl,dfrll,dfrml,dfrrl);
      vmin = FMIN( vmin, FMIN9_(dfllm,dflmm,dflrm,dfmlm,dfmmm,dfmrm,dfrlm,dfrmm,dfrrm) );
      vmin = FMIN( vmin, FMIN9_(dfllr,dflmr,dflrr,dfmlr,dfmmr,dfmrr,dfrlr,dfrmr,dfrrr) );
      
      vmax =             FMAX9_(dflll,dflml,dflrl,dfmll,dfmml,dfmrl,dfrll,dfrml,dfrrl);
      vmax = FMAX( vmax, FMAX9_(dfllm,dflmm,dflrm,dfmlm,dfmmm,dfmrm,dfrlm,dfrmm,dfrrm) );
      vmax = FMAX( vmax, FMAX9_(dfllr,dflmr,dflrr,dfmlr,dfmmr,dfmrr,dfrlr,dfrmr,dfrrr) );
      
      dfx  = HALF_F * (qNb[CENTER+1][CENTER  ][CENTER  ][nVar] - 
		       qNb[CENTER-1][CENTER  ][CENTER  ][nVar]);
      dfy  = HALF_F * (qNb[CENTER  ][CENTER+1][CENTER  ][nVar] - 
		       qNb[CENTER  ][CENTER-1][CENTER  ][nVar]);
      dfz  = HALF_F * (qNb[CENTER  ][CENTER  ][CENTER+1][nVar] - 
		       qNb[CENTER  ][CENTER  ][CENTER-1][nVar]);
      dff  = HALF_F * (FABS(dfx) + FABS(dfy) + FABS(dfz));

      if (dff>ZERO_F) {
	slop = FMIN(ONE_F, FMIN(FABS(vmin), FABS(vmax))/dff);
      } else {
	slop = ONE_F;
      }

      dlim = slop;
      
      dqX[nVar] = dlim*dfx;
      dqY[nVar] = dlim*dfy;
      dqZ[nVar] = dlim*dfz;

    } // end for nVar
          
  } // end slope_type == 3
  
} // slope_unsplit_hydro_3d

/**
 * Compute primitive variables slope (vector dq) from q and its
 * neighbors (do not use the full 3x3x3 neighborhood so slope_type=3
 * is NOT available here).
 *
 * This routine is only used in the 3D UNSPLIT integration.
 * Only slope_type 1 and 2 are supported.
 *
 * \param[in]  q       : current primitive variable state
 * \param[in]  qPlusX  : state in the next neighbor cell along XDIR
 * \param[in]  qMinusX : state in the previous neighbor cell along XDIR
 * \param[in]  qPlusY  : state in the next neighbor cell along YDIR
 * \param[in]  qMinusY : state in the previous neighbor cell along YDIR
 * \param[in]  qPlusZ  : state in the next neighbor cell along ZDIR
 * \param[in]  qMinusZ : state in the previous neighbor cell along ZDIR
 * \param[out] dq       : reference to an array returning the X, Y and Z slopes
 *
 * \note this version has the same interface as the pure hydro routine 
 * slope_unsplit_3d located in slope.h
 *
 * \note this version of slope computation should be used inside MHD 
 * implementation version 3
 *
 */
__DEVICE__ 
void slope_unsplit_hydro_3d(real_t q       [NVAR_MHD], 
			    real_t qPlusX  [NVAR_MHD], 
			    real_t qMinusX [NVAR_MHD],
			    real_t qPlusY  [NVAR_MHD], 
			    real_t qMinusY [NVAR_MHD],
			    real_t qPlusZ  [NVAR_MHD], 
			    real_t qMinusZ [NVAR_MHD],
			    real_t (&dq)[3][NVAR_MHD])
{			

  real_t (&dqX)[NVAR_MHD] = dq[IX];
  real_t (&dqY)[NVAR_MHD] = dq[IY];
  real_t (&dqZ)[NVAR_MHD] = dq[IZ];

  if (::gParams.slope_type==0) {
    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {
      dqX[nVar] = ZERO_F;
      dqY[nVar] = ZERO_F;
      dqZ[nVar] = ZERO_F;
    }
    return;
  }

  if (::gParams.slope_type==1 or ::gParams.slope_type==2) {  // minmod or average

    real_t dlft, drgt, dcen, dsgn, slop, dlim;
    
    for (int nVar=0; nVar<NVAR_MHD; ++nVar) {

      // slopes in first coordinate direction
      dlft = ::gParams.slope_type*(q      [nVar] - qMinusX[nVar]);
      drgt = ::gParams.slope_type*(qPlusX [nVar] - q      [nVar]);
      dcen = HALF_F * (qPlusX[nVar] - qMinusX[nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqX[nVar] = dsgn * FMIN( dlim, FABS(dcen) );

      // slopes in second coordinate direction
      dlft = ::gParams.slope_type*(q      [nVar] - qMinusY[nVar]);
      drgt = ::gParams.slope_type*(qPlusY [nVar] - q      [nVar]);
      dcen = HALF_F * (qPlusY[nVar] - qMinusY[nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqY[nVar] = dsgn * FMIN( dlim, FABS(dcen) );

      // slopes in third coordinate direction
      dlft = ::gParams.slope_type*(q      [nVar] - qMinusZ[nVar]);
      drgt = ::gParams.slope_type*(qPlusZ [nVar] - q      [nVar]);
      dcen = HALF_F * (qPlusZ[nVar] - qMinusZ[nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dqZ[nVar] = dsgn * FMIN( dlim, FABS(dcen) );

    } // end for nVar

  } // end slope_type == 1 or 2

} // slope_unsplit_hydro_3d

/**
 * slope_unsplit_mhd_2d computes only magnetic field slopes in 2D; hydro
 * slopes are always computed in slope_unsplit_hydro_2d.
 * 
 * Compute magnetic field slopes (vector dbf) from bf (face-centered)
 * and its neighbors. 
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 * Loosely adapted from RAMSES and DUMSES mhd/umuscl.f90: subroutine uslope
 * Interface is changed to become cellwise.
 *
 * \param[in]  bf  : face centered magnetic field in current
 * and neighboring cells. There are 6 values (3 values for bf_x along
 * y and 3 for bf_y along x).
 * 
 * \param[out] dbf : reference to an array returning magnetic field slopes 
 */
__DEVICE__ 
void slope_unsplit_mhd_2d(real_t bfNeighbors[6],
			  real_t (&dbf)[2][3])
{			
  /* layout for face centered magnetic field */
  real_t &bfx        = bfNeighbors[0];
  real_t &bfx_yplus  = bfNeighbors[1];
  real_t &bfx_yminus = bfNeighbors[2];
  real_t &bfy        = bfNeighbors[3];
  real_t &bfy_xplus  = bfNeighbors[4];
  real_t &bfy_xminus = bfNeighbors[5];
  
  real_t (&dbfX)[3] = dbf[IX];
  real_t (&dbfY)[3] = dbf[IY];

  // default values for magnetic field slopes
  for (int nVar=0; nVar<3; ++nVar) {
    dbfX[nVar] = ZERO_F;
    dbfY[nVar] = ZERO_F;
  }
  
  /*
   * face-centered magnetic field slopes
   */
  // 1D transverse TVD slopes for face-centered magnetic fields
  
  {
    // Bx along direction Y 
    real_t dlft, drgt, dcen, dsgn, slop, dlim;
    dlft = ::gParams.slope_type * (bfx       - bfx_yminus);
    drgt = ::gParams.slope_type * (bfx_yplus - bfx       );
    dcen = HALF_F * (bfx_yplus - bfx_yminus);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = FMIN( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    dbfY[IX] = dsgn * FMIN( dlim, FABS(dcen) );
      
    // By along direction X
    dlft = ::gParams.slope_type * (bfy       - bfy_xminus);
    drgt = ::gParams.slope_type * (bfy_xplus - bfy       );
    dcen = HALF_F * (bfy_xplus - bfy_xminus);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = FMIN( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if( (dlft*drgt) <= ZERO_F )
      dlim=ZERO_F;
    dbfX[IY] = dsgn * FMIN( dlim, FABS(dcen) );
  }

} // slope_unsplit_mhd_2d

/**
 * slope_unsplit_mhd_3d computes only magnetic field slopes in 3D; hydro
 * slopes are always computed in slope_unsplit_hydro_3d.
 * 
 * Compute magnetic field slopes (vector dbf) from bf (face-centered)
 * and its neighbors. 
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 * Loosely adapted from RAMSES and DUMSES mhd/umuscl.f90: subroutine uslope
 * Interface is changed to become cellwise.
 *
 * \param[in]  bf  : face centered magnetic field in current
 * and neighboring cells. There are 15 values (5 values for bf_x along
 * y and z, 5 for bf_y along x and z, 5 for bf_z along x and y).
 * 
 * \param[out] dbf : reference to an array returning magnetic field slopes 
 *
 * \note This routine is called inside trace_unsplit_mhd_3d
 */
__DEVICE__
void slope_unsplit_mhd_3d(real_t bfNeighbors[15],
			  real_t (&dbf)[3][3])
{			
  /* layout for face centered magnetic field */
  real_t &bfx        = bfNeighbors[0];
  real_t &bfx_yplus  = bfNeighbors[1];
  real_t &bfx_yminus = bfNeighbors[2];
  real_t &bfx_zplus  = bfNeighbors[3];
  real_t &bfx_zminus = bfNeighbors[4];

  real_t &bfy        = bfNeighbors[5];
  real_t &bfy_xplus  = bfNeighbors[6];
  real_t &bfy_xminus = bfNeighbors[7];
  real_t &bfy_zplus  = bfNeighbors[8];
  real_t &bfy_zminus = bfNeighbors[9];
  
  real_t &bfz        = bfNeighbors[10];
  real_t &bfz_xplus  = bfNeighbors[11];
  real_t &bfz_xminus = bfNeighbors[12];
  real_t &bfz_yplus  = bfNeighbors[13];
  real_t &bfz_yminus = bfNeighbors[14];
  

  real_t (&dbfX)[3] = dbf[IX];
  real_t (&dbfY)[3] = dbf[IY];
  real_t (&dbfZ)[3] = dbf[IZ];

  // default values for magnetic field slopes
  for (int nVar=0; nVar<3; ++nVar) {
    dbfX[nVar] = ZERO_F;
    dbfY[nVar] = ZERO_F;
    dbfZ[nVar] = ZERO_F;
  }
  
  /*
   * face-centered magnetic field slopes
   */
  // 1D transverse TVD slopes for face-centered magnetic fields
  real_t xslope_type = FMIN(::gParams.slope_type, 2.0);
  real_t dlft, drgt, dcen, dsgn, slop, dlim;
  {
    // Bx along direction Y     
    dlft = xslope_type * (bfx       - bfx_yminus);
    drgt = xslope_type * (bfx_yplus - bfx       );
    dcen = HALF_F      * (bfx_yplus - bfx_yminus);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = FMIN( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    dbfY[IX] = dsgn * FMIN( dlim, FABS(dcen) );
    // Bx along direction Z    
    dlft = xslope_type * (bfx       - bfx_zminus);
    drgt = xslope_type * (bfx_zplus - bfx       );
    dcen = HALF_F      * (bfx_zplus - bfx_zminus);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = FMIN( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    dbfZ[IX] = dsgn * FMIN( dlim, FABS(dcen) );
      
    // By along direction X
    dlft = xslope_type * (bfy       - bfy_xminus);
    drgt = xslope_type * (bfy_xplus - bfy       );
    dcen = HALF_F      * (bfy_xplus - bfy_xminus);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = FMIN( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if( (dlft*drgt) <= ZERO_F )
      dlim=ZERO_F;
    dbfX[IY] = dsgn * FMIN( dlim, FABS(dcen) );
    // By along direction Z
    dlft = xslope_type * (bfy       - bfy_zminus);
    drgt = xslope_type * (bfy_zplus - bfy       );
    dcen = HALF_F      * (bfy_zplus - bfy_zminus);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = FMIN( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if( (dlft*drgt) <= ZERO_F )
      dlim=ZERO_F;
    dbfZ[IY] = dsgn * FMIN( dlim, FABS(dcen) );

    // Bz along direction X
    dlft = xslope_type * (bfz       - bfz_xminus);
    drgt = xslope_type * (bfz_xplus - bfz       );
    dcen = HALF_F      * (bfz_xplus - bfz_xminus);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = FMIN( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if( (dlft*drgt) <= ZERO_F )
      dlim=ZERO_F;
    dbfX[IZ] = dsgn * FMIN( dlim, FABS(dcen) );
    // Bz along direction Y
    dlft = xslope_type * (bfz       - bfz_yminus);
    drgt = xslope_type * (bfz_yplus - bfz       );
    dcen = HALF_F      * (bfz_yplus - bfz_yminus);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = FMIN( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if( (dlft*drgt) <= ZERO_F )
      dlim=ZERO_F;
    dbfY[IZ] = dsgn * FMIN( dlim, FABS(dcen) );

  }

} // slope_unsplit_mhd_3d

#endif // SLOPE_MHD_H_
