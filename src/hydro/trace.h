/**
 * \file trace.h
 * \brief Handle trace computation in the Godunov scheme.
 *
 * \author F. Chateau and P. Kestener
 *
 * $Id: trace.h 3450 2014-06-16 22:03:23Z pkestene $
 */
#ifndef TRACE_H_
#define TRACE_H_

#include "real_type.h"
#include "constants.h"
#include "slope.h"

/**
 * Trace computations for directionally split Godunov scheme.
 *
 * Note that this routine uses global variables iorder, scheme and
 * slope_type.
 *
 * \param[in] q      : Primitive variables state.
 * \param[in] qMinus : state in the previous neighbor cell
 * \param[in] qPlus  : state in the next neighbor cell
 * \param[in] c      : local sound speed.
 * \param[in] dtdx   : dt over dx
 * \param[out] qxm
 * \param[out] qxp
 */
template <NvarSimulation NVAR>
__DEVICE__
void trace(real_t q[NVAR], real_t qPlus[NVAR], real_t qMinus[NVAR], 
	   real_t c, real_t dtdx, 
	   real_t (&qxm)[NVAR], real_t (&qxp)[NVAR])
{
  real_t dq[NVAR];
  for (unsigned int iVar=0; iVar<NVAR; iVar++)
    dq[iVar] = ZERO_F;

  // compute slopes 
  // global var gParams.slope_type is used internally by slope routine
  if(gParams.iorder != 1) {
    slope<NVAR>(q, qPlus, qMinus, dq);
  }

  real_t zerol   = ZERO_F;
  real_t zeror   = ZERO_F;
  real_t project = ZERO_F;
  if(gParams.scheme == MusclScheme) // MUSCL-Hancock method
    {
      zerol   = -ONE_F*100/dtdx;
      zeror   =  ONE_F*100/dtdx;
      project =  ONE_F;
    }
  else if(gParams.scheme == PlmdeScheme) // standard PLMDE
    {
      zerol   = ZERO_F;
      zeror   = ZERO_F;
      project = ONE_F;
    }
  else if(gParams.scheme == CollelaScheme) // Collela's method
    {
      zerol   = ZERO_F;
      zeror   = ZERO_F;
      project = ZERO_F;
    }

  const real_t& cc = c;
  real_t csq = cc*cc;
  const real_t& r = q[ID];
  const real_t& p = q[IP];
  const real_t& u = q[IU];
  const real_t& v = q[IV];

  const real_t& dr = dq[ID];
  const real_t& dp = dq[IP];
  const real_t& du = dq[IU];
  const real_t& dv = dq[IV];

  real_t alpham  = HALF_F*(dp/(r*cc) - du)*r/cc;
  real_t alphap  = HALF_F*(dp/(r*cc) + du)*r/cc;
  real_t alpha0r = dr - dp/csq;
  real_t alpha0v = dv;
  real_t alpha0w = ZERO_F;
  if (NVAR == NVAR_3D)
    alpha0w = dq[IW];

  // Right state
  {
    real_t spminus = (u-cc) < zeror ? (u-cc) * dtdx + ONE_F : project;
    real_t spplus  = (u+cc) < zeror ? (u+cc) * dtdx + ONE_F : project;
    real_t spzero  =  u     < zeror ? u      * dtdx + ONE_F : project;
    real_t apright   = -HALF_F * spplus  * alphap;
    real_t amright   = -HALF_F * spminus * alpham;
    real_t azrright  = -HALF_F * spzero  * alpha0r;
    real_t azv1right = -HALF_F * spzero  * alpha0v;
    real_t acmpright = ZERO_F;
    if (NVAR == NVAR_3D)
      acmpright = -HALF_F * spzero  * alpha0w;

    qxp[ID] = r + (apright + amright + azrright);
    qxp[IP] = p + (apright + amright           ) * csq;
    qxp[IU] = u + (apright - amright           ) * cc / r;
    qxp[IV] = v + azv1right;
    if (NVAR == NVAR_3D)
      qxp[IW] = q[IW] + acmpright;
  }

  // Left state
  {
    real_t spminus = (u-cc) > zerol ? (u-cc) * dtdx - ONE_F : -project;
    real_t spplus  = (u+cc) > zerol ? (u+cc) * dtdx - ONE_F : -project;
    real_t spzero  =  u     > zerol ?  u     * dtdx - ONE_F : -project;
    real_t apleft   = -HALF_F * spplus  * alphap;
    real_t amleft   = -HALF_F * spminus * alpham;
    real_t azrleft  = -HALF_F * spzero  * alpha0r;
    real_t azv1left = -HALF_F * spzero  * alpha0v;
    real_t acmpleft = ZERO_F;
    if (NVAR == NVAR_3D)
      acmpleft = -HALF_F * spzero  * alpha0w;
    
    qxm[ID] = r + (apleft + amleft + azrleft);
    qxm[IP] = p + (apleft + amleft          ) * csq;
    qxm[IU] = u + (apleft - amleft          ) * cc / r;
    qxm[IV] = v + azv1left;
    if (NVAR == NVAR_3D)
      qxm[IW] = q[IW] + acmpleft;
  }
} // trace


/**
 * Compute slopes and then perform trace for unsplit Godunov scheme.
 *
 * \note Note that this routine uses global variables iorder, scheme and
 * slope_type.
 *
 * \note Note that is routine is loosely adapted from trace2d/trace3d found in 
 * Ramses sources (sub-dir hydro, file umuscl.f90) to be now a one cell computation.
 *
 * \param[in] q          : Primitive variables state.
 * \param[in] qNeighbors : state in the neighbor cells (2 neighbors
 * per dimension, in the following order x+, x-, y+, y-, z+, z-)
 * \param[in] c          : local sound speed.
 * \param[in] dtdx       : dt over dx
 * \param[out] qm        : qm state (one per dimension)
 * \param[out] qp        : qp state (one per dimension)
 */
template <DimensionType NDIM, NvarSimulation NVAR>
__DEVICE__
void trace_unsplit(real_t q[NVAR], real_t qNeighbors[2*NDIM][NVAR],
		   real_t c, real_t dtdx, 
		   real_t (&qm)[NDIM][NVAR], real_t (&qp)[NDIM][NVAR])
{
  (void) c;

  real_t& dtdy = dtdx;
  
  // first compute slopes
  if (NDIM == TWO_D) {

    real_t dq[2][NVAR_2D];

    slope_unsplit_hydro_2d(q, 
			   qNeighbors[0], qNeighbors[1], 
			   qNeighbors[2], qNeighbors[3],
			   dq);
    
    // Cell centered values
    real_t& r =  q[ID];
    real_t& p =  q[IP];
    real_t& u =  q[IU];
    real_t& v =  q[IV];

    // TVD slopes in all directions
    real_t& drx = dq[0][ID];
    real_t& dpx = dq[0][IP];
    real_t& dux = dq[0][IU];
    real_t& dvx = dq[0][IV];
      
    real_t& dry = dq[1][ID];
    real_t& dpy = dq[1][IP];
    real_t& duy = dq[1][IU];
    real_t& dvy = dq[1][IV];
      
    // source terms (with transverse derivatives)
    real_t sr0 = -u*drx-v*dry - (dux+dvy)*r;
    real_t sp0 = -u*dpx-v*dpy - (dux+dvy)*gParams.gamma0*p;
    real_t su0 = -u*dux-v*duy - (dpx    )/r;
    real_t sv0 = -u*dvx-v*dvy - (dpy    )/r;
    
    // Right state at left interface
    qp[0][ID] = r - HALF_F*drx + sr0*dtdx*HALF_F;
    qp[0][IP] = p - HALF_F*dpx + sp0*dtdx*HALF_F;
    qp[0][IU] = u - HALF_F*dux + su0*dtdx*HALF_F;
    qp[0][IV] = v - HALF_F*dvx + sv0*dtdx*HALF_F;
    qp[0][ID] = FMAX(gParams.smallr, qp[0][ID]);
      
    // Left state at right interface
    qm[0][ID] = r + HALF_F*drx + sr0*dtdx*HALF_F;
    qm[0][IP] = p + HALF_F*dpx + sp0*dtdx*HALF_F;
    qm[0][IU] = u + HALF_F*dux + su0*dtdx*HALF_F;
    qm[0][IV] = v + HALF_F*dvx + sv0*dtdx*HALF_F;
    qm[0][ID] = FMAX(gParams.smallr, qm[0][ID]);
    
    // Top state at bottom interface
    qp[1][ID] = r - HALF_F*dry + sr0*dtdy*HALF_F;
    qp[1][IP] = p - HALF_F*dpy + sp0*dtdy*HALF_F;
    qp[1][IU] = u - HALF_F*duy + su0*dtdy*HALF_F;
    qp[1][IV] = v - HALF_F*dvy + sv0*dtdy*HALF_F;
    qp[1][ID] = FMAX(gParams.smallr, qp[1][ID]);
      
    // Bottom state at top interface
    qm[1][ID] = r + HALF_F*dry + sr0*dtdy*HALF_F;
    qm[1][IP] = p + HALF_F*dpy + sp0*dtdy*HALF_F;
    qm[1][IU] = u + HALF_F*duy + su0*dtdy*HALF_F;
    qm[1][IV] = v + HALF_F*dvy + sv0*dtdy*HALF_F;
    qm[1][ID] = FMAX(gParams.smallr, qm[1][ID]);

  } else { // THREE_D

    real_t dq[3][NVAR_3D];
    real_t& dtdz = dtdx;

    slope_unsplit_3d(q, 
		     qNeighbors[0], qNeighbors[1], 
		     qNeighbors[2], qNeighbors[3],
		     qNeighbors[4], qNeighbors[5],
		     dq);
  
    // Cell centered values
    real_t& r =  q[ID];
    real_t& p =  q[IP];
    real_t& u =  q[IU];
    real_t& v =  q[IV];
    real_t& w =  q[IW];

    // TVD slopes in all 3 directions
    real_t& drx = dq[0][ID];
    real_t& dpx = dq[0][IP];
    real_t& dux = dq[0][IU];
    real_t& dvx = dq[0][IV];
    real_t& dwx = dq[0][IW];
      
    real_t& dry = dq[1][ID];
    real_t& dpy = dq[1][IP];
    real_t& duy = dq[1][IU];
    real_t& dvy = dq[1][IV];
    real_t& dwy = dq[1][IW];
      
    real_t& drz = dq[2][ID];
    real_t& dpz = dq[2][IP];
    real_t& duz = dq[2][IU];
    real_t& dvz = dq[2][IV];
    real_t& dwz = dq[2][IW];
      
    // Source terms (including transverse derivatives)
    real_t sr0 = -u*drx-v*dry-w*drz - (dux+dvy+dwz)*r;
    real_t sp0 = -u*dpx-v*dpy-w*dpz - (dux+dvy+dwz)*gParams.gamma0*p;
    real_t su0 = -u*dux-v*duy-w*duz - (dpx        )/r;
    real_t sv0 = -u*dvx-v*dvy-w*dvz - (dpy        )/r;
    real_t sw0 = -u*dwx-v*dwy-w*dwz - (dpz        )/r;
      
    // Right state at left interface
    qp[0][ID] = r - HALF_F*drx + sr0*dtdx*HALF_F;
    qp[0][IP] = p - HALF_F*dpx + sp0*dtdx*HALF_F;
    qp[0][IU] = u - HALF_F*dux + su0*dtdx*HALF_F;
    qp[0][IV] = v - HALF_F*dvx + sv0*dtdx*HALF_F;
    qp[0][IW] = w - HALF_F*dwx + sw0*dtdx*HALF_F;
    qp[0][ID] = FMAX(gParams.smallr, qp[0][ID]);
    
    // Left state at left interface
    qm[0][ID] = r + HALF_F*drx + sr0*dtdx*HALF_F;
    qm[0][IP] = p + HALF_F*dpx + sp0*dtdx*HALF_F;
    qm[0][IU] = u + HALF_F*dux + su0*dtdx*HALF_F;
    qm[0][IV] = v + HALF_F*dvx + sv0*dtdx*HALF_F;
    qm[0][IW] = w + HALF_F*dwx + sw0*dtdx*HALF_F;
    qm[0][ID] = FMAX(gParams.smallr, qm[0][ID]);
      
    // Top state at bottom interface
    qp[1][ID] = r - HALF_F*dry + sr0*dtdy*HALF_F;
    qp[1][IP] = p - HALF_F*dpy + sp0*dtdy*HALF_F;
    qp[1][IU] = u - HALF_F*duy + su0*dtdy*HALF_F;
    qp[1][IV] = v - HALF_F*dvy + sv0*dtdy*HALF_F;
    qp[1][IW] = w - HALF_F*dwy + sw0*dtdy*HALF_F;
    qp[1][ID] = FMAX(gParams.smallr, qp[1][ID]);
    
    // Bottom state at top interface
    qm[1][ID] = r + HALF_F*dry + sr0*dtdy*HALF_F;
    qm[1][IP] = p + HALF_F*dpy + sp0*dtdy*HALF_F;
    qm[1][IU] = u + HALF_F*duy + su0*dtdy*HALF_F;
    qm[1][IV] = v + HALF_F*dvy + sv0*dtdy*HALF_F;
    qm[1][IW] = w + HALF_F*dwy + sw0*dtdy*HALF_F;
    qm[1][ID] = FMAX(gParams.smallr, qm[1][ID]);
    
    // Back state at front interface
    qp[2][ID] = r - HALF_F*drz + sr0*dtdz*HALF_F;
    qp[2][IP] = p - HALF_F*dpz + sp0*dtdz*HALF_F;
    qp[2][IU] = u - HALF_F*duz + su0*dtdz*HALF_F;
    qp[2][IV] = v - HALF_F*dvz + sv0*dtdz*HALF_F;
    qp[2][IW] = w - HALF_F*dwz + sw0*dtdz*HALF_F;
    qp[2][ID] = FMAX(gParams.smallr, qp[2][ID]);
    
    // Front state at back interface
    qm[2][ID] = r + HALF_F*drz + sr0*dtdz*HALF_F;
    qm[2][IP] = p + HALF_F*dpz + sp0*dtdz*HALF_F;
    qm[2][IU] = u + HALF_F*duz + su0*dtdz*HALF_F;
    qm[2][IV] = v + HALF_F*dvz + sv0*dtdz*HALF_F;
    qm[2][IW] = w + HALF_F*dwz + sw0*dtdz*HALF_F;
    qm[2][ID] = FMAX(gParams.smallr, qm[2][ID]);
      
  } // end THREE_D

} // trace_unsplit

/**
 * This another implementation of trace computations for 2D data;
 * slopes are computed outside; it is used when unsplitVersion = 1
 *
 * Note that :
 * - hydro slopes computations are done outside this routine
 *
 * \param[in]  q  primitive variable state vector
 * \param[in]  dq primitive variable slopes
 * \param[in]  dtdx dt divided by dx
 * \param[in]  dtdy dt divided by dy
 * \param[out] qm
 * \param[out] qp
 *
 */
__DEVICE__
void trace_unsplit_hydro_2d(real_t q[NVAR_2D],
			    real_t dq[2][NVAR_2D],
			    real_t dtdx,
			    real_t dtdy,
			    real_t (&qm)[2][NVAR_2D], 
			    real_t (&qp)[2][NVAR_2D])
{
  
  // some aliases
  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  real_t &gamma  = ::gParams.gamma0;
  //real_t &dx     = ::gParams.dx;

  // Cell centered values
  real_t r = q[ID];
  real_t p = q[IP];
  real_t u = q[IU];
  real_t v = q[IV];

  // Cell centered TVD slopes in X direction
  real_t& drx = dq[IX][ID];  drx *= HALF_F;
  real_t& dpx = dq[IX][IP];  dpx *= HALF_F;
  real_t& dux = dq[IX][IU];  dux *= HALF_F;
  real_t& dvx = dq[IX][IV];  dvx *= HALF_F;
  
  // Cell centered TVD slopes in Y direction
  real_t& dry = dq[IY][ID];  dry *= HALF_F;
  real_t& dpy = dq[IY][IP];  dpy *= HALF_F;
  real_t& duy = dq[IY][IU];  duy *= HALF_F;
  real_t& dvy = dq[IY][IV];  dvy *= HALF_F;

  // Source terms (including transverse derivatives)
  real_t sr0, su0, sv0, sp0;

  if (true /*cartesian*/) {

    sr0 = (-u*drx-dux*r)      *dtdx + (-v*dry-dvy*r)      *dtdy;
    su0 = (-u*dux-dpx/r)      *dtdx + (-v*duy      )      *dtdy;
    sv0 = (-u*dvx      )      *dtdx + (-v*dvy-dpy/r)      *dtdy;
    sp0 = (-u*dpx-dux*gamma*p)*dtdx + (-v*dpy-dvy*gamma*p)*dtdy;
	
  } // end cartesian

  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  p = p + sp0;

  // Face averaged right state at left interface
  qp[0][ID] = r - drx;
  qp[0][IU] = u - dux;
  qp[0][IV] = v - dvx;
  qp[0][IP] = p - dpx;
  qp[0][ID] = FMAX(smallR,  qp[0][ID]);
  qp[0][IP] = FMAX(smallp * qp[0][ID], qp[0][IP]);
  
  // Face averaged left state at right interface
  qm[0][ID] = r + drx;
  qm[0][IU] = u + dux;
  qm[0][IV] = v + dvx;
  qm[0][IP] = p + dpx;
  qm[0][ID] = FMAX(smallR,  qm[0][ID]);
  qm[0][IP] = FMAX(smallp * qm[0][ID], qm[0][IP]);

  // Face averaged top state at bottom interface
  qp[1][ID] = r - dry;
  qp[1][IU] = u - duy;
  qp[1][IV] = v - dvy;
  qp[1][IP] = p - dpy;
  qp[1][ID] = FMAX(smallR,  qp[1][ID]);
  qp[1][IP] = FMAX(smallp * qp[1][ID], qp[1][IP]);
  
  // Face averaged bottom state at top interface
  qm[1][ID] = r + dry;
  qm[1][IU] = u + duy;
  qm[1][IV] = v + dvy;
  qm[1][IP] = p + dpy;
  qm[1][ID] = FMAX(smallR,  qm[1][ID]);
  qm[1][IP] = FMAX(smallp * qm[1][ID], qm[1][IP]);
  
} // trace_unsplit_hydro_2d

/**
 * This another implementation of trace computations for 2D data;
 * slopes are computed outside; it is used when unsplitVersion = 0
 *
 * Note that :
 * - hydro slopes computations are done outside this routine
 *
 * \param[in]  q  primitive variable state vector
 * \param[in]  dq primitive variable slopes
 * \param[in]  dtdx dt divided by dx
 * \param[in]  dtdy dt divided by dy
 * \param[in]  faceId identify which cell face is to be reconstructed
 * \param[out] qface the reconstructed state
 *
 */
__DEVICE__
void trace_unsplit_hydro_2d_by_direction(real_t q[NVAR_2D],
					 real_t dq[2][NVAR_2D],
					 real_t dtdx,
					 real_t dtdy,
					 int faceId,
					 real_t (&qface)[NVAR_2D])
{
  
  // some aliases
  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  real_t &gamma  = ::gParams.gamma0;

  // Cell centered values
  real_t r = q[ID];
  real_t p = q[IP];
  real_t u = q[IU];
  real_t v = q[IV];

  // Cell centered TVD slopes in X direction
  real_t drx = HALF_F*dq[IX][ID];
  real_t dpx = HALF_F*dq[IX][IP];
  real_t dux = HALF_F*dq[IX][IU];
  real_t dvx = HALF_F*dq[IX][IV];
  
  // Cell centered TVD slopes in Y direction
  real_t dry = HALF_F*dq[IY][ID];
  real_t dpy = HALF_F*dq[IY][IP];
  real_t duy = HALF_F*dq[IY][IU];
  real_t dvy = HALF_F*dq[IY][IV];

  // Source terms (including transverse derivatives)
  real_t sr0, su0, sv0, sp0;

  if (true /*cartesian*/) {

    sr0 = (-u*drx-dux*r)      *dtdx + (-v*dry-dvy*r)      *dtdy;
    su0 = (-u*dux-dpx/r)      *dtdx + (-v*duy      )      *dtdy;
    sv0 = (-u*dvx      )      *dtdx + (-v*dvy-dpy/r)      *dtdy;
    sp0 = (-u*dpx-dux*gamma*p)*dtdx + (-v*dpy-dvy*gamma*p)*dtdy;
	
  } // end cartesian

  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  p = p + sp0;

  if (faceId == FACE_XMIN) {
    // Face averaged right state at left interface
    qface[ID] = r - drx;
    qface[IU] = u - dux;
    qface[IV] = v - dvx;
    qface[IP] = p - dpx;
    qface[ID] = FMAX(smallR,  qface[ID]);
    qface[IP] = FMAX(smallp * qface[ID], qface[IP]);
  }

  if (faceId == FACE_XMAX) {
    // Face averaged left state at right interface
    qface[ID] = r + drx;
    qface[IU] = u + dux;
    qface[IV] = v + dvx;
    qface[IP] = p + dpx;
    qface[ID] = FMAX(smallR,  qface[ID]);
    qface[IP] = FMAX(smallp * qface[ID], qface[IP]);
  }

  if (faceId == FACE_YMIN) {
    // Face averaged top state at bottom interface
    qface[ID] = r - dry;
    qface[IU] = u - duy;
    qface[IV] = v - dvy;
    qface[IP] = p - dpy;
    qface[ID] = FMAX(smallR,  qface[ID]);
    qface[IP] = FMAX(smallp * qface[ID], qface[IP]);
  }
  
  if (faceId == FACE_YMAX) {
    // Face averaged bottom state at top interface
    qface[ID] = r + dry;
    qface[IU] = u + duy;
    qface[IV] = v + dvy;
    qface[IP] = p + dpy;
    qface[ID] = FMAX(smallR,  qface[ID]);
    qface[IP] = FMAX(smallp * qface[ID], qface[IP]);
  }

} // trace_unsplit_hydro_2d_by_direction

/**
 * This another implementation of trace computations for 3D data; it
 * is used when unsplitVersion = 1
 *
 * Note that :
 * - hydro slopes computations are done outside this routine
 *
 *
 * \note There is a MHD version of this routine in file trace_mhd.h
 * named trace_unsplit_mhd_3d_simpler
 *
 * \param[in]  q  primitive variable state vector
 * \param[in]  dq primitive variable slopes
 * \param[in]  dtdx dt divided by dx
 * \param[in]  dtdy dt divided by dy
 * \param[in]  dtdz dt divided by dz
 * \param[out] qm
 * \param[out] qp
 *
 */
__DEVICE__
void trace_unsplit_hydro_3d(real_t q[NVAR_3D],
			    real_t dq[3][NVAR_3D],
			    real_t dtdx,
			    real_t dtdy,
			    real_t dtdz,
			    real_t (&qm)[3][NVAR_3D], 
			    real_t (&qp)[3][NVAR_3D])
{
  
  // some aliases
  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  real_t &gamma  = ::gParams.gamma0;
  //real_t &dx     = ::gParams.dx;

  // Cell centered values
  real_t r = q[ID];
  real_t p = q[IP];
  real_t u = q[IU];
  real_t v = q[IV];
  real_t w = q[IW];            

  // Cell centered TVD slopes in X direction
  real_t& drx = dq[IX][ID];  drx *= HALF_F;
  real_t& dpx = dq[IX][IP];  dpx *= HALF_F;
  real_t& dux = dq[IX][IU];  dux *= HALF_F;
  real_t& dvx = dq[IX][IV];  dvx *= HALF_F;
  real_t& dwx = dq[IX][IW];  dwx *= HALF_F;
  
  // Cell centered TVD slopes in Y direction
  real_t& dry = dq[IY][ID];  dry *= HALF_F;
  real_t& dpy = dq[IY][IP];  dpy *= HALF_F;
  real_t& duy = dq[IY][IU];  duy *= HALF_F;
  real_t& dvy = dq[IY][IV];  dvy *= HALF_F;
  real_t& dwy = dq[IY][IW];  dwy *= HALF_F;

  // Cell centered TVD slopes in Z direction
  real_t& drz = dq[IZ][ID];  drz *= HALF_F;
  real_t& dpz = dq[IZ][IP];  dpz *= HALF_F;
  real_t& duz = dq[IZ][IU];  duz *= HALF_F;
  real_t& dvz = dq[IZ][IV];  dvz *= HALF_F;
  real_t& dwz = dq[IZ][IW];  dwz *= HALF_F;

  // Source terms (including transverse derivatives)
  real_t sr0, su0, sv0, sw0, sp0;

  if (true /*cartesian*/) {

    sr0 = (-u*drx-dux*r)*dtdx + (-v*dry-dvy*r)*dtdy + (-w*drz-dwz*r)*dtdz;
    su0 = (-u*dux-dpx/r)*dtdx + (-v*duy      )*dtdy + (-w*duz      )*dtdz; 
    sv0 = (-u*dvx      )*dtdx + (-v*dvy-dpy/r)*dtdy + (-w*dvz      )*dtdz;
    sw0 = (-u*dwx      )*dtdx + (-v*dwy      )*dtdy + (-w*dwz-dpz/r)*dtdz; 
    sp0 = (-u*dpx-dux*gamma*p)*dtdx + (-v*dpy-dvy*gamma*p)*dtdy + (-w*dpz-dwz*gamma*p)*dtdz;

  } // end cartesian

  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  w = w + sw0;
  p = p + sp0;
  
  // Face averaged right state at left interface
  qp[0][ID] = r - drx;
  qp[0][IU] = u - dux;
  qp[0][IV] = v - dvx;
  qp[0][IW] = w - dwx;
  qp[0][IP] = p - dpx;
  qp[0][ID] = FMAX(smallR,  qp[0][ID]);
  qp[0][IP] = FMAX(smallp * qp[0][ID], qp[0][IP]);
  
  // Face averaged left state at right interface
  qm[0][ID] = r + drx;
  qm[0][IU] = u + dux;
  qm[0][IV] = v + dvx;
  qm[0][IW] = w + dwx;
  qm[0][IP] = p + dpx;
  qm[0][ID] = FMAX(smallR,  qm[0][ID]);
  qm[0][IP] = FMAX(smallp * qm[0][ID], qm[0][IP]);

  // Face averaged top state at bottom interface
  qp[1][ID] = r - dry;
  qp[1][IU] = u - duy;
  qp[1][IV] = v - dvy;
  qp[1][IW] = w - dwy;
  qp[1][IP] = p - dpy;
  qp[1][ID] = FMAX(smallR,  qp[1][ID]);
  qp[1][IP] = FMAX(smallp * qp[1][ID], qp[1][IP]);
  
  // Face averaged bottom state at top interface
  qm[1][ID] = r + dry;
  qm[1][IU] = u + duy;
  qm[1][IV] = v + dvy;
  qm[1][IW] = w + dwy;
  qm[1][IP] = p + dpy;
  qm[1][ID] = FMAX(smallR,  qm[1][ID]);
  qm[1][IP] = FMAX(smallp * qm[1][ID], qm[1][IP]);
  
  // Face averaged front state at back interface
  qp[2][ID] = r - drz;
  qp[2][IU] = u - duz;
  qp[2][IV] = v - dvz;
  qp[2][IW] = w - dwz;
  qp[2][IP] = p - dpz;
  qp[2][ID] = FMAX(smallR,  qp[2][ID]);
  qp[2][IP] = FMAX(smallp * qp[2][ID], qp[2][IP]);
  
  // Face averaged back state at front interface
  qm[2][ID] = r + drz;
  qm[2][IU] = u + duz;
  qm[2][IV] = v + dvz;
  qm[2][IW] = w + dwz;
  qm[2][IP] = p + dpz;
  qm[2][ID] = FMAX(smallR,  qm[2][ID]);
  qm[2][IP] = FMAX(smallp * qm[2][ID], qm[2][IP]);

} // trace_unsplit_hydro_3d

#endif /*TRACE_H_*/
