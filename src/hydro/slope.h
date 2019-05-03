/**
 * \file slope.h
 * \brief Compute primitive variables slope dq from a given q state and its
 * adjacent neighbors.
 *
 * \author Pierre Kestener
 * \date 16 Nov 2010
 *
 * $Id: slope.h 1832 2012-03-15 14:58:50Z pkestene $
 */
#ifndef SLOPE_H_
#define SLOPE_H_

#include "real_type.h"
#include "constants.h"

#include "base_type.h"

// =======================================================
// =======================================================
// Hydro slope routines
// =======================================================
// =======================================================

/************************************************************************
 * DIRECTIONALLY SPLIT
 ************************************************************************/

/**
 * compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 2D/3D directionally splitted integration.
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 * \param[in]  q      : current primitive variable state
 * \param[in]  qPlus  : state in the next neighbor cell
 * \param[in]  qMinus : state in the previous neighbor cell
 * \param[out] dq     : reference to an array that will return the slope
 */
template <NvarSimulation NVAR>
__DEVICE__ 
void slope(real_t q[NVAR], 
	   real_t qPlus[NVAR], 
	   real_t qMinus[NVAR],
	   real_t (&dq)[NVAR])
{
  
  // local variables
  real_t dsgn, dlim, dcen, dlft, drgt, slop;
  dsgn = ONE_F;

  for (int nVar=0; nVar<NVAR; ++nVar) {
    
    dlft = ::gParams.slope_type*(q    [nVar] - qMinus[nVar]);
    drgt = ::gParams.slope_type*(qPlus[nVar] - q     [nVar]);
    dcen = HALF_F * (qPlus[nVar] - qMinus[nVar]);
    //dsgn = COPYSIGN(ONE_F, dcen);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = FMIN( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt)<= ZERO_F )
      dlim = ZERO_F;
    dq[nVar] = dsgn * FMIN( dlim, FABS(dcen) );
    
  }
  
} // slope

/************************************************************************
 * UNSPLIT 2D
 ************************************************************************/

/**
 * Compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
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
 * \param[out] dq      : reference to an array returning the X and Y slopes
 *
 * \note The very same routine exist in slope_mhd.h with a different
 * interface (q, qPlus, etc... are vector of size NVAR_MHD).
 * 
 */
__DEVICE__ 
void slope_unsplit_hydro_2d(real_t q[NVAR_2D], 
			    real_t qPlusX[NVAR_2D], 
			    real_t qMinusX[NVAR_2D],
			    real_t qPlusY[NVAR_2D], 
			    real_t qMinusY[NVAR_2D],
			    real_t (&dq)[2][NVAR_2D])
{			
  real_t (&dqX)[NVAR_2D] = dq[IX];
  real_t (&dqY)[NVAR_2D] = dq[IY];
  
  if (::gParams.slope_type==0) {
    for (int nVar=0; nVar<NVAR_2D; ++nVar) {
      dqX[nVar] = ZERO_F;
      dqY[nVar] = ZERO_F;
    }
    return;
  }

  if (::gParams.slope_type==1 or ::gParams.slope_type==2) {  // minmod or average

    real_t dlft, drgt, dcen, dsgn, slop, dlim;
    
    for (int nVar=0; nVar<NVAR_2D; ++nVar) {

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

  } // end slope_type == 1 or 2
  
} // slope_unsplit_hydro_2d

/* some dummy utility routines */
__DEVICE__
real_t FMAX9(real_t a0, real_t a1, real_t a2, 
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
} // FMAX9

__DEVICE__
real_t FMIN9(real_t a0, real_t a1, real_t a2, 
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
} // FMIN9

/**
 * compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 2D UNSPLIT integration and
 * slope_type = 3
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 *
 * \param[in]  qNeighbors : vector of state in neighbor cells in the
 * following order :
 * [0][0]:x+0 y+0   [1][0]:x+1 y+0   [2][0]:x-1 y+0
 * [0][1]:x+0 y+1   [1][1]:x+1 y+1   [2][1]:x-1 y+1
 * [0][2]:x+0 y-1   [1][2]:x+1 y-1   [2][2]:x-1 y-1
 * \param[out] dq  : reference to an array that will return the X and Y slopes
 */
__DEVICE__ 
void slope_unsplit_2d_positivity(real_t qNeighbors[3][3][NVAR_2D],
				 real_t (&dq)[2][NVAR_2D])
{
  real_t (&dqX)[NVAR_2D] = dq[0];
  real_t (&dqY)[NVAR_2D] = dq[1];
  
  if (::gParams.slope_type==3) { // positivity preserving 2d unsplit slope

    real_t dfll,dflm,dflr, dfml,dfmm,dfmr, dfrl,dfrm,dfrr;
    real_t vmin, vmax;
    real_t dfx, dfy, dff, slop;

    int L=2, M=0, R=1; // positions (L=left, M=Middle, R=right)
    
    for (int nVar=0; nVar<NVAR_2D; ++nVar) {

      dfll = qNeighbors[L][L][nVar] - qNeighbors[M][M][nVar];
      dflm = qNeighbors[L][M][nVar] - qNeighbors[M][M][nVar];
      dflr = qNeighbors[L][R][nVar] - qNeighbors[M][M][nVar];
      dfml = qNeighbors[M][L][nVar] - qNeighbors[M][M][nVar];
      dfmm = qNeighbors[M][M][nVar] - qNeighbors[M][M][nVar];
      dfmr = qNeighbors[M][R][nVar] - qNeighbors[M][M][nVar];
      dfrl = qNeighbors[R][L][nVar] - qNeighbors[M][M][nVar];
      dfrm = qNeighbors[R][M][nVar] - qNeighbors[M][M][nVar];
      dfrr = qNeighbors[R][R][nVar] - qNeighbors[M][M][nVar];
      
      vmin = FMIN9(dfll,dflm,dflr,dfml,dfmm,dfmr,dfrl,dfrm,dfrr);
      vmax = FMAX9(dfll,dflm,dflr,dfml,dfmm,dfmr,dfrl,dfrm,dfrr);
	
      dfx  = HALF_F * ( qNeighbors[R][M][nVar] - qNeighbors[L][M][nVar] );
      dfy  = HALF_F * ( qNeighbors[M][R][nVar] - qNeighbors[M][L][nVar] );
      dff  = HALF_F * ( FABS(dfx) + FABS(dfy) );
	
      if(dff>ZERO_F)
	slop = FMIN( ONE_F, FMIN( FABS(vmin), FABS(vmax) ) / dff );
      else
	slop = ONE_F;
      
      real_t& dlim = slop;
      
      dqX[nVar] = dlim*dfx;
      dqY[nVar] = dlim*dfy;

    } // end for nVar

  } // end slope_type==3		

} // slope_unsplit_2d_positivity


/************************************************************************
 * UNSPLIT 3D
 ************************************************************************/

/**
 * compute an order 1 slope.
 *
 * \return dq slope computed
 */
__DEVICE__
void slope_order1(const real_t &q,
		  const real_t &qPlus,
		  const real_t &qMinus,
		  real_t &dq)
{
    real_t dlft, drgt;

    dlft = q     - qMinus;
    drgt = qPlus - q      ;
    if( (dlft*drgt) <= ZERO_F )
      dq = ZERO_F;
    else if ( dlft > 0 )
      dq = FMIN( dlft, drgt );
    else
      dq = FMAX( dlft, drgt );
    
} // slope_order1

/**
 * compute an order n slope.
 *
 * \return dq slope computed
 */
__DEVICE__
void slope_order_n(const real_t &q,
		   const real_t &qPlus,
		   const real_t &qMinus,
		   real_t &dq,
		   const real_t slopeParam)
{
  real_t dlft, drgt, dcen, dsgn, slop, dlim;
  
  dlft = slopeParam * (q     - qMinus);
  drgt = slopeParam * (qPlus - q     );
  dcen = HALF_F * (qPlus - qMinus);
  dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
  slop = FMIN( FABS(dlft), FABS(drgt) );
  dlim = slop;
  if ( (dlft*drgt) <= ZERO_F )
    dlim = ZERO_F;
  dq = dsgn * FMIN( dlim, FABS(dcen) );

} // slope_order_n

/**
 * compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1 and 2.
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 * \param[in]  q       : current primitive variable state
 * \param[in]  qPlusX  : state in the next neighbor cell along XDIR
 * \param[in]  qMinusX : state in the previous neighbor cell along XDIR
 * \param[in]  qPlusY  : state in the next neighbor cell along YDIR
 * \param[in]  qMinusY : state in the previous neighbor cell along YDIR
 * \param[in]  qPlusZ  : state in the next neighbor cell along ZDIR
 * \param[in]  qMinusZ : state in the previous neighbor cell along ZDIR
 * \param[out] dq      : reference to an array that will return the X,Y,Z slopes
 */
__DEVICE__ 
void slope_unsplit_3d(real_t q[NVAR_3D], 
		      real_t qPlusX[NVAR_3D], 
		      real_t qMinusX[NVAR_3D],
		      real_t qPlusY[NVAR_3D], 
		      real_t qMinusY[NVAR_3D],
		      real_t qPlusZ[NVAR_3D], 
		      real_t qMinusZ[NVAR_3D],
		      real_t (&dq)[3][NVAR_3D])
{			
  
  real_t (&dqX)[NVAR_3D] = dq[0];
  real_t (&dqY)[NVAR_3D] = dq[1];
  real_t (&dqZ)[NVAR_3D] = dq[2];

  if (::gParams.slope_type==0) {
    for (int nVar=0; nVar<NVAR_3D; ++nVar) {
      dqX[nVar] = ZERO_F;
      dqY[nVar] = ZERO_F;
      dqZ[nVar] = ZERO_F;
    }
    return;
  }

  if (::gParams.slope_type==1) {  // minmod

    real_t dlft, drgt;

    for (int nVar=0; nVar<NVAR_3D; ++nVar) {

      // slopes in first coordinate direction
      dlft = q     [nVar] - qMinusX[nVar];
      drgt = qPlusX[nVar] - q      [nVar];
      if( (dlft*drgt) <= ZERO_F )
	dqX[nVar] = ZERO_F;
      else if ( dlft > 0 )
	dqX[nVar] = FMIN( dlft, drgt );
      else
	dqX[nVar] = FMAX( dlft, drgt );
      
      // slopes in second coordinate direction
      dlft = q     [nVar] - qMinusY[nVar];
      drgt = qPlusY[nVar] - q      [nVar];
      if( (dlft*drgt) <= ZERO_F )
	dqY[nVar] = ZERO_F;
      else if ( dlft > 0 )
	dqY[nVar] = FMIN( dlft, drgt );
      else
	dqY[nVar] = FMAX( dlft, drgt );
      
      // slopes in third coordinate direction 
      dlft = q     [nVar] - qMinusZ[nVar];
      drgt = qPlusZ[nVar] - q      [nVar];
      if( (dlft*drgt) <= ZERO_F )
	dqZ[nVar] = ZERO_F;
      else if ( dlft > 0 )
	dqZ[nVar] = FMIN( dlft, drgt );
      else
	dqZ[nVar] = FMAX( dlft, drgt );
    }
      
  } else if (::gParams.slope_type == 2) { // minmod
 
    real_t dlft, drgt, dcen, dsgn, slop, dlim;
    
    for (int nVar=0; nVar<NVAR_3D; ++nVar) {

      // slopes in first coordinate direction
      dlft = ::gParams.slope_type * (q     [nVar] - qMinusX[nVar]);
      drgt = ::gParams.slope_type * (qPlusX[nVar] - q      [nVar]);
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
      
      // slopes in third coordinate direction
      dlft = ::gParams.slope_type*(q     [nVar] - qMinusZ[nVar]);
      drgt = ::gParams.slope_type*(qPlusZ[nVar] - q      [nVar]);
      dcen = HALF_F * (qPlusZ[nVar] - qMinusZ[nVar]);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F)
	dlim = ZERO_F;
      dqZ[nVar] = dsgn * FMIN( dlim, FABS(dcen) );

    } // end for nVar

  } //  end slope_type == 2

} // slope_unsplit_3d

/**
 * compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1 and 2.
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 * \param[in]  q      : current primitive variable state
 * \param[in]  qPlus  : state in the next neighbor cell along DIR
 * \param[in]  qMinus : state in the previous neighbor cell along DIR
 * \param[out] dq     : reference to an array to return the slopes in one direction
 */
__DEVICE__ 
void slope_unsplit_3d_v1(real_t q[NVAR_3D], 
			 real_t qPlus[NVAR_3D], 
			 real_t qMinus[NVAR_3D],
			 real_t (&dq)[NVAR_3D])
{			
  
  if (::gParams.slope_type==0) {
    for (int nVar=0; nVar<NVAR_3D; ++nVar) {
      dq[nVar] = ZERO_F;
    }
    return;
  }

  if (::gParams.slope_type==1) {  // minmod

    for (int nVar=0; nVar<NVAR_3D; ++nVar) {
      // slopes
      slope_order1(q[nVar], qPlus[nVar], qMinus[nVar], dq[nVar]);
    }
      
  } else if (::gParams.slope_type == 2) { // minmod
 
    for (int nVar=0; nVar<NVAR_3D; ++nVar) {

      // slopes in first coordinate direction
      slope_order_n(q[nVar], qPlus[nVar], qMinus[nVar], dq[nVar], 2);
      
    } // end for nVar

  } //  end slope_type == 2

} // slope_unsplit_3d_v1


/**
 * compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1 and 2.
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 * \param[in]  q      : current primitive variable state
 * \param[in]  qPlus  : state in the next neighbor cell along DIR
 * \param[in]  qMinus : state in the previous neighbor cell along DIR
 * \param[out] dq     : reference to an array that will return the slopes along DIR
 */
__DEVICE__ 
void slope_unsplit_3d_v2(const qStateHydro& q, 
			 const qStateHydro& qPlus, 
			 const qStateHydro& qMinus,
			 qStateHydro& dq)
{			
  
  if (::gParams.slope_type==0) {
    dq.reset();
    return;
  }

  if (::gParams.slope_type==1) {  // minmod

    // slopes 
    slope_order1(q.D, qPlus.D, qMinus.D, dq.D);
    slope_order1(q.P, qPlus.P, qMinus.P, dq.P);
    slope_order1(q.U, qPlus.U, qMinus.U, dq.U);
    slope_order1(q.V, qPlus.V, qMinus.V, dq.V);
    slope_order1(q.W, qPlus.W, qMinus.W, dq.W);
    
  } else if (::gParams.slope_type == 2) { // minmod
    
    // slopes in first coordinate direction
    slope_order_n(q.D, qPlus.D, qMinus.D, dq.D, 2);
    slope_order_n(q.P, qPlus.P, qMinus.P, dq.P, 2);
    slope_order_n(q.U, qPlus.U, qMinus.U, dq.U, 2);
    slope_order_n(q.V, qPlus.V, qMinus.V, dq.V, 2);
    slope_order_n(q.W, qPlus.W, qMinus.W, dq.W, 2);
    
  } //  end slope_type == 2

} // slope_unsplit_3d_v2

/**
 * compute primitive variables slope (vector dq) from q and its neighbors.
 * This routine is only used in the 3D UNSPLIT integration and
 * slope_type = 3
 * 
 * Note that slope_type is a global variable, located in symbol memory when 
 * using the GPU version.
 *
 *
 * \param[in]  qNeighbors : vector of state in neighbor cells in the
 * following order :
 * [0][0][0]:x+0 y+0 z+0   [1][0][0]:x+1 y+0 z+0   [2][0][0]:x-1 y+0 z+0
 * [0][1][0]:x+0 y+1 z+0   [1][1][0]:x+1 y+1 z+0   [2][1][0]:x-1 y+1 z+0
 * [0][2][0]:x+0 y-1 z+0   [1][2][0]:x+1 y-1 z+0   [2][2][0]:x-1 y-1 z+0
 *... etc ...
 *
 * \param[out] dq     : reference to a 3-by-NVAR_3D array returning
 * the X,Y and Z slopes
 */
__DEVICE__ 
void slope_unsplit_3d_positivity(real_t qNeighbors[3][3][3][NVAR_3D],
				 real_t (&dq)[3][NVAR_3D])
{
  
  real_t (&dqX)[NVAR_3D] = dq[0];
  real_t (&dqY)[NVAR_3D] = dq[1];
  real_t (&dqZ)[NVAR_3D] = dq[2];

  if (::gParams.slope_type==3) { // positivity preserving 3d unsplit slope

    real_t dflll,dflml,dflrl, dfmll,dfmml,dfmrl, dfrll,dfrml,dfrrl;
    real_t dfllm,dflmm,dflrm, dfmlm,dfmmm,dfmrm, dfrlm,dfrmm,dfrrm;
    real_t dfllr,dflmr,dflrr, dfmlr,dfmmr,dfmrr, dfrlr,dfrmr,dfrrr;
    real_t vmin, vmax;
    real_t dfx, dfy, dfz, dff, slop;

    int L=2, M=0, R=1; // positions (L=left, M=Middle, R=right)

    for (int nVar=0; nVar<NVAR_3D; ++nVar) {
      
      dflll = qNeighbors[L][L][L][nVar] - qNeighbors[M][M][M][nVar];
      dflml = qNeighbors[L][M][L][nVar] - qNeighbors[M][M][M][nVar];
      dflrl = qNeighbors[L][R][L][nVar] - qNeighbors[M][M][M][nVar];
      dfmll = qNeighbors[M][L][L][nVar] - qNeighbors[M][M][M][nVar];
      dfmml = qNeighbors[M][M][L][nVar] - qNeighbors[M][M][M][nVar];
      dfmrl = qNeighbors[M][R][L][nVar] - qNeighbors[M][M][M][nVar];
      dfrll = qNeighbors[R][L][L][nVar] - qNeighbors[M][M][M][nVar];
      dfrml = qNeighbors[R][M][L][nVar] - qNeighbors[M][M][M][nVar];
      dfrrl = qNeighbors[R][R][L][nVar] - qNeighbors[M][M][M][nVar];
      
      dfllm = qNeighbors[L][L][M][nVar] - qNeighbors[M][M][M][nVar];
      dflmm = qNeighbors[L][M][M][nVar] - qNeighbors[M][M][M][nVar];
      dflrm = qNeighbors[L][R][M][nVar] - qNeighbors[M][M][M][nVar];
      dfmlm = qNeighbors[M][L][M][nVar] - qNeighbors[M][M][M][nVar];
      dfmmm = qNeighbors[M][M][M][nVar] - qNeighbors[M][M][M][nVar];
      dfmrm = qNeighbors[M][R][M][nVar] - qNeighbors[M][M][M][nVar];
      dfrlm = qNeighbors[R][L][M][nVar] - qNeighbors[M][M][M][nVar];
      dfrmm = qNeighbors[R][M][M][nVar] - qNeighbors[M][M][M][nVar];
      dfrrm = qNeighbors[R][R][M][nVar] - qNeighbors[M][M][M][nVar];
      
      dfllr = qNeighbors[L][L][R][nVar] - qNeighbors[M][M][M][nVar];
      dflmr = qNeighbors[L][M][R][nVar] - qNeighbors[M][M][M][nVar];
      dflrr = qNeighbors[L][R][R][nVar] - qNeighbors[M][M][M][nVar];
      dfmlr = qNeighbors[M][L][R][nVar] - qNeighbors[M][M][M][nVar];
      dfmmr = qNeighbors[M][M][R][nVar] - qNeighbors[M][M][M][nVar];
      dfmrr = qNeighbors[M][R][R][nVar] - qNeighbors[M][M][M][nVar];
      dfrlr = qNeighbors[R][L][R][nVar] - qNeighbors[M][M][M][nVar];
      dfrmr = qNeighbors[R][M][R][nVar] - qNeighbors[M][M][M][nVar];
      dfrrr = qNeighbors[R][R][R][nVar] - qNeighbors[M][M][M][nVar];
      
      real_t vmin1 = FMIN9(dflll,dflml,dflrl,dfmll,dfmml,dfmrl,dfrll,dfrml,dfrrl);
      real_t vmin2 = FMIN9(dfllm,dflmm,dflrm,dfmlm,dfmmm,dfmrm,dfrlm,dfrmm,dfrrm);
      real_t vmin3 = FMIN9(dfllr,dflmr,dflrr,dfmlr,dfmmr,dfmrr,dfrlr,dfrmr,dfrrr);
      vmin = FMIN( FMIN(vmin1,vmin2), vmin3 );

      real_t vmax1 = FMAX9(dflll,dflml,dflrl,dfmll,dfmml,dfmrl,dfrll,dfrml,dfrrl);
      real_t vmax2 = FMAX9(dfllm,dflmm,dflrm,dfmlm,dfmmm,dfmrm,dfrlm,dfrmm,dfrrm);
      real_t vmax3 = FMAX9(dfllr,dflmr,dflrr,dfmlr,dfmmr,dfmrr,dfrlr,dfrmr,dfrrr);
      vmax = FMAX( FMAX(vmax1,vmax2), vmax3 );
      
      dfx  = HALF_F * ( qNeighbors[R][M][M][nVar] - qNeighbors[L][M][M][nVar] );
      dfy  = HALF_F * ( qNeighbors[M][R][M][nVar] - qNeighbors[M][L][M][nVar] );
      dfz  = HALF_F * ( qNeighbors[M][M][R][nVar] - qNeighbors[M][M][L][nVar] );
      dff  = HALF_F * ( FABS(dfx) + FABS(dfy) + FABS(dfz) );
      
      if (dff > ZERO_F )
	slop = FMIN( ONE_F, FMIN( FABS(vmin), FABS(vmax) ) / dff );
      else
	slop = ONE_F;
            
      real_t& dlim = slop;
      
      dqX[nVar] = dlim*dfx;
      dqY[nVar] = dlim*dfy;
      dqZ[nVar] = dlim*dfz;

    } // end for nVar
    
  } // end slope_type == 3
  
} // slope_unsplit_3d_positivity


#endif // SLOPE_H_
