/**
 * \file trace_mhd.h
 * \brief Handle trace computation in the MHD Godunov scheme.
 *
 * \date March 31, 2011
 * \author Pierre Kestener
 *
 * $Id: trace_mhd.h 3450 2014-06-16 22:03:23Z pkestene $
 */
#ifndef TRACE_MHD_H_
#define TRACE_MHD_H_

#include "real_type.h"
#include "constants.h"
#include "slope_mhd.h"

/**
 * 2D Trace computations for unsplit Godunov scheme.
 *
 * \note Note that this routine uses global variables iorder, scheme and
 * slope_type.
 *
 * \note Note that is routine is loosely adapted from trace2d found in 
 * Dumses and in Ramses sources (sub-dir mhd, file umuscl.f90) to be now a one cell 
 * computation. 
 *
 * \param[in]  qNb        state in neighbor cells (3-by-3 neighborhood indexed as qNb[i][j], for i,j=0,1,2); current center cell is at index (i=j=1).
 * \param[in]  bfNb       face centered magnetic field in neighbor cells (4-by-4 neighborhood indexed as bfNb[i][j] for i,j=0,1,2,3); current cell is located at index (i=j=1)
 * \param[in]  c          local sound speed.
 * \param[in]  dtdx       dt over dx
 * \param[in]  dtdy       dt over dy
 * \param[in]  xPos       x location of current cell (needed for shear computation)
 * \param[out] qm         qm state (one per dimension)
 * \param[out] qp         qp state (one per dimension)
 * \param[out] qEdge      q state on cell edges (qRT, qRB, qLT, qLB)
 */
__DEVICE__ inline
void trace_unsplit_mhd_2d(real_t qNb[3][3][NVAR_MHD],
			  real_t bfNb[4][4][3],
			  real_t c, 
			  real_t dtdx,
			  real_t dtdy,
			  real_t xPos,
			  real_t (&qm)[2][NVAR_MHD], 
			  real_t (&qp)[2][NVAR_MHD],
			  real_t (&qEdge)[4][NVAR_MHD])
{
  (void) c;

  // neighborhood sizes
  enum {Q_SIZE=3, BF_SIZE = 4};

  // index of current cell in the neighborhood
  enum {CENTER=1};

  // alias for q on cell edge (as defined in DUMSES trace2d routine)
  real_t (&qRT)[NVAR_MHD] = qEdge[0];
  real_t (&qRB)[NVAR_MHD] = qEdge[1];
  real_t (&qLT)[NVAR_MHD] = qEdge[2];
  real_t (&qLB)[NVAR_MHD] = qEdge[3];

  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  //real_t &smallP = ::gParams.smallpp;
  real_t &gamma  = ::gParams.gamma0;
  real_t &Omega0 = ::gParams.Omega0;

  real_t (&q)[NVAR_MHD] = qNb[CENTER][CENTER]; // current cell (neighborhood center)

  // compute u,v,A,B,Ez (electric field)
  real_t Ez[2][2];
  for (int di=0; di<2; di++)
    for (int dj=0; dj<2; dj++) {
      
      int centerX = CENTER+di;
      int centerY = CENTER+dj;
      real_t u  = 0.25f *  (qNb[centerX-1][centerY-1][IU] + 
			    qNb[centerX-1][centerY  ][IU] + 
			    qNb[centerX  ][centerY-1][IU] + 
			    qNb[centerX  ][centerY  ][IU]); 
      
      real_t v  = 0.25f *  (qNb[centerX-1][centerY-1][IV] +
			    qNb[centerX-1][centerY  ][IV] +
			    qNb[centerX  ][centerY-1][IV] + 
			    qNb[centerX  ][centerY  ][IV]);
      
      real_t A  = 0.5f  * (bfNb[centerX  ][centerY-1][IX] + 
			   bfNb[centerX  ][centerY  ][IX]);

      real_t B  = 0.5f  * (bfNb[centerX-1][centerY  ][IY] + 
			   bfNb[centerX  ][centerY  ][IY]);
      
      Ez[di][dj] = u*B-v*A;
    }

  // Electric field
  real_t &ELL = Ez[0][0];
  real_t &ELR = Ez[0][1];
  real_t &ERL = Ez[1][0];
  real_t &ERR = Ez[1][1];

  // Cell centered values
  real_t r = q[ID];
  real_t p = q[IP];
  real_t u = q[IU];
  real_t v = q[IV];
  real_t w = q[IW];            
  real_t A = q[IA];
  real_t B = q[IB];
  real_t C = q[IC];            
    
  // Face centered variables
  real_t AL =  bfNb[CENTER  ][CENTER  ][IX];
  real_t AR =  bfNb[CENTER+1][CENTER  ][IX];
  real_t BL =  bfNb[CENTER  ][CENTER  ][IY];
  real_t BR =  bfNb[CENTER  ][CENTER+1][IY];

  // TODO LATER : compute xL, xR and xC using ::gParam
  // this is only needed when doing cylindrical or spherical coordinates

  /*
   * compute dq slopes
   */
  real_t dq[2][NVAR_MHD];

  slope_unsplit_hydro_2d(qNb, dq);

  // slight modification compared to DUMSES (we re-used dq itself,
  // instead of re-declaring new variables, better for the GPU
  // register count

  // Cell centered TVD slopes in X direction
  real_t& drx = dq[IX][ID];  drx *= HALF_F;
  real_t& dpx = dq[IX][IP];  dpx *= HALF_F;
  real_t& dux = dq[IX][IU];  dux *= HALF_F;
  real_t& dvx = dq[IX][IV];  dvx *= HALF_F;
  real_t& dwx = dq[IX][IW];  dwx *= HALF_F;
  real_t& dCx = dq[IX][IC];  dCx *= HALF_F;
  real_t& dBx = dq[IX][IB];  dBx *= HALF_F;
  
  // Cell centered TVD slopes in Y direction
  real_t& dry = dq[IY][ID];  dry *= HALF_F;
  real_t& dpy = dq[IY][IP];  dpy *= HALF_F;
  real_t& duy = dq[IY][IU];  duy *= HALF_F;
  real_t& dvy = dq[IY][IV];  dvy *= HALF_F;
  real_t& dwy = dq[IY][IW];  dwy *= HALF_F;
  real_t& dCy = dq[IY][IC];  dCy *= HALF_F;
  real_t& dAy = dq[IY][IA];  dAy *= HALF_F;
  
  /*
   * compute dbf slopes needed for Face centered TVD slopes in transverse direction
   */
  real_t bfNeighbors[6];
  real_t dbf[2][3];
  real_t (&dbfX)[3] = dbf[IX];
  real_t (&dbfY)[3] = dbf[IY];
  
  bfNeighbors[0] =  bfNb[CENTER  ][CENTER  ][IX];
  bfNeighbors[1] =  bfNb[CENTER  ][CENTER+1][IX];
  bfNeighbors[2] =  bfNb[CENTER  ][CENTER-1][IX];
  bfNeighbors[3] =  bfNb[CENTER  ][CENTER  ][IY];
  bfNeighbors[4] =  bfNb[CENTER+1][CENTER  ][IY];
  bfNeighbors[5] =  bfNb[CENTER-1][CENTER  ][IY];
  
  slope_unsplit_mhd_2d(bfNeighbors, dbf);
  
  // Face centered TVD slopes in transverse direction
  real_t dALy = HALF_F * dbfY[IX];
  real_t dBLx = HALF_F * dbfX[IY];

  // change neighbors to i+1, j and recompute dbf
  bfNeighbors[0] =  bfNb[CENTER+1][CENTER  ][IX];
  bfNeighbors[1] =  bfNb[CENTER+1][CENTER+1][IX];
  bfNeighbors[2] =  bfNb[CENTER+1][CENTER-1][IX];
  bfNeighbors[3] =  bfNb[CENTER+1][CENTER  ][IY];
  bfNeighbors[4] =  bfNb[CENTER+2][CENTER  ][IY];
  bfNeighbors[5] =  bfNb[CENTER  ][CENTER  ][IY];

  slope_unsplit_mhd_2d(bfNeighbors, dbf);  

  real_t dARy = HALF_F * dbfY[IX];

  // change neighbors to i, j+1 and recompute dbf
  bfNeighbors[0] =  bfNb[CENTER  ][CENTER+1][IX];
  bfNeighbors[1] =  bfNb[CENTER  ][CENTER+2][IX];
  bfNeighbors[2] =  bfNb[CENTER  ][CENTER  ][IX];
  bfNeighbors[3] =  bfNb[CENTER  ][CENTER+1][IY];
  bfNeighbors[4] =  bfNb[CENTER+1][CENTER+1][IY];
  bfNeighbors[5] =  bfNb[CENTER-1][CENTER+1][IY];

  slope_unsplit_mhd_2d(bfNeighbors, dbf);

  real_t dBRx = HALF_F * dbfX[IY];
  
  // Cell centered slopes in normal direction
  real_t dAx = HALF_F * (AR - AL);
  real_t dBy = HALF_F * (BR - BL);
  
  // Source terms (including transverse derivatives)
  real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
  real_t sAL0, sAR0, sBL0, sBR0;

  if (true /*cartesian*/) {

    sr0 = (-u*drx-dux*r)                *dtdx + (-v*dry-dvy*r)                *dtdy;
    su0 = (-u*dux-dpx/r-B*dBx/r-C*dCx/r)*dtdx + (-v*duy+B*dAy/r)              *dtdy;
    sv0 = (-u*dvx+A*dBx/r)              *dtdx + (-v*dvy-dpy/r-A*dAy/r-C*dCy/r)*dtdy;
    sw0 = (-u*dwx+A*dCx/r)              *dtdx + (-v*dwy+B*dCy/r)              *dtdy;
    sp0 = (-u*dpx-dux*gamma*p)          *dtdx + (-v*dpy-dvy*gamma*p)          *dtdy;
    sA0 =                                       ( u*dBy+B*duy-v*dAy-A*dvy)    *dtdy;
    sB0 = (-u*dBx-B*dux+v*dAx+A*dvx)    *dtdx ;
    sC0 = ( w*dAx+A*dwx-u*dCx-C*dux)    *dtdx + (-v*dCy-C*dvy+w*dBy+B*dwy)    *dtdy;
    if (Omega0 > ZERO_F) {
      real_t shear = -1.5 * Omega0 * xPos;
      sC0 += (shear * dAx - 1.5 * Omega0 * A) * dtdx;
      sC0 +=  shear * dBy                     * dtdy;
    }

    // Face centered B-field
    sAL0 = +(ELR-ELL)*HALF_F*dtdy;
    sAR0 = +(ERR-ERL)*HALF_F*dtdy;
    sBL0 = -(ERL-ELL)*HALF_F*dtdx;
    sBR0 = -(ERR-ELR)*HALF_F*dtdx;

  } // end cartesian
  
  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  w = w + sw0;
  p = p + sp0;
  A = A + sA0;
  B = B + sB0;
  C = C + sC0;
  
  AL = AL + sAL0;
  AR = AR + sAR0;
  BL = BL + sBL0;
  BR = BR + sBR0;
  
  // Right state at left interface
  qp[0][ID] = r - drx;
  qp[0][IU] = u - dux;
  qp[0][IV] = v - dvx;
  qp[0][IW] = w - dwx;
  qp[0][IP] = p - dpx;
  qp[0][IA] = AL;
  qp[0][IB] = B - dBx;
  qp[0][IC] = C - dCx;
  qp[0][ID] = FMAX(smallR, qp[0][ID]);
  qp[0][IP] = FMAX(smallp*qp[0][ID], qp[0][IP]);
  
  // Left state at right interface
  qm[0][ID] = r + drx;
  qm[0][IU] = u + dux;
  qm[0][IV] = v + dvx;
  qm[0][IW] = w + dwx;
  qm[0][IP] = p + dpx;
  qm[0][IA] = AR;
  qm[0][IB] = B + dBx;
  qm[0][IC] = C + dCx;
  qm[0][ID] = FMAX(smallR, qm[0][ID]);
  qm[0][IP] = FMAX(smallp*qm[0][ID], qm[0][IP]);
  
  // Top state at bottom interface
  qp[1][ID] = r - dry;
  qp[1][IU] = u - duy;
  qp[1][IV] = v - dvy;
  qp[1][IW] = w - dwy;
  qp[1][IP] = p - dpy;
  qp[1][IA] = A - dAy;
  qp[1][IB] = BL;
  qp[1][IC] = C - dCy;
  qp[1][ID] = FMAX(smallR, qp[1][ID]);
  qp[1][IP] = FMAX(smallp*qp[1][ID], qp[1][IP]);
  
  // Bottom state at top interface
  qm[1][ID] = r + dry;
  qm[1][IU] = u + duy;
  qm[1][IV] = v + dvy;
  qm[1][IW] = w + dwy;
  qm[1][IP] = p + dpy;
  qm[1][IA] = A + dAy;
  qm[1][IB] = BR;
  qm[1][IC] = C + dCy;
  qm[1][ID] = FMAX(smallR, qm[1][ID]);
  qm[1][IP] = FMAX(smallp*qm[1][ID], qm[1][IP]);
  
  
  // Right-top state (RT->LL)
  qRT[ID] = r + (+drx+dry);
  qRT[IU] = u + (+dux+duy);
  qRT[IV] = v + (+dvx+dvy);
  qRT[IW] = w + (+dwx+dwy);
  qRT[IP] = p + (+dpx+dpy);
  qRT[IA] = AR+ (   +dARy);
  qRT[IB] = BR+ (+dBRx   );
  qRT[IC] = C + (+dCx+dCy);
  qRT[ID] = FMAX(smallR, qRT[ID]);
  qRT[IP] = FMAX(smallp*qRT[ID], qRT[IP]);
    
  // Right-Bottom state (RB->LR)
  qRB[ID] = r + (+drx-dry);
  qRB[IU] = u + (+dux-duy);
  qRB[IV] = v + (+dvx-dvy);
  qRB[IW] = w + (+dwx-dwy);
  qRB[IP] = p + (+dpx-dpy);
  qRB[IA] = AR+ (   -dARy);
  qRB[IB] = BL+ (+dBLx   );
  qRB[IC] = C + (+dCx-dCy);
  qRB[ID] = FMAX(smallR, qRB[ID]);
  qRB[IP] = FMAX(smallp*qRB[ID], qRB[IP]);
    
  // Left-Bottom state (LB->RR)
  qLB[ID] = r + (-drx-dry);
  qLB[IU] = u + (-dux-duy);
  qLB[IV] = v + (-dvx-dvy);
  qLB[IW] = w + (-dwx-dwy);
  qLB[IP] = p + (-dpx-dpy);
  qLB[IA] = AL+ (   -dALy);
  qLB[IB] = BL+ (-dBLx   );
  qLB[IC] = C + (-dCx-dCy);
  qLB[ID] = FMAX(smallR, qLB[ID]);
  qLB[IP] = FMAX(smallp*qLB[ID], qLB[IP]);
    
  // Left-Top state (LT->RL)
  qLT[ID] = r + (-drx+dry);
  qLT[IU] = u + (-dux+duy);
  qLT[IV] = v + (-dvx+dvy);
  qLT[IW] = w + (-dwx+dwy);
  qLT[IP] = p + (-dpx+dpy);
  qLT[IA] = AL+ (   +dALy);
  qLT[IB] = BR+ (-dBRx   );
  qLT[IC] = C + (-dCx+dCy);
  qLT[ID] = FMAX(smallR, qLT[ID]);
  qLT[IP] = FMAX(smallp*qLT[ID], qLT[IP]);

} // trace_unsplit_mhd_2d

/**
 * 2D Trace computations for unsplit Godunov scheme.
 *
 * \note Note that this routine uses global variables iorder, scheme and
 * slope_type.
 *
 * \note Note that is routine is loosely adapted from trace2d found in 
 * Dumses and in Ramses sources (sub-dir mhd, file umuscl.f90) to be now a one cell 
 * computation. 
 *
 * \param[in]  q          primitive variables state in current cell
 * \param[in]  dq         primitive variable slopes
 * \param[in]  bfNb       face centered magnetic field (only the first 4 are used)
 * \param[in]  dAB        face-centered magnetic slopes in transverse direction dBx/dy and dBy/dx
 * \param[in]  Ez         electric field
 * \param[in]  dtdx       dt over dx
 * \param[in]  dtdy       dt over dy
 * \param[in]  xPos       x location of current cell (needed for shear computation)
 * \param[in]  locationId identify which cell face or edge is to be reconstructed
 * \param[out] qRecons    the reconstructed state
 */
__DEVICE__ inline
void trace_unsplit_mhd_2d_face(real_t q[NVAR_MHD],
			       real_t dq[2][NVAR_MHD],
			       real_t bfNb[TWO_D*2],
			       real_t dAB[TWO_D*2],
			       real_t Ez[2][2],
			       real_t dtdx,
			       real_t dtdy,
			       real_t xPos,
			       int    locationId,
			       real_t (&qRecons)[NVAR_MHD])
{

  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  real_t &gamma  = ::gParams.gamma0;
  real_t &Omega0 = ::gParams.Omega0;

  // Electric field
  real_t &ELL = Ez[0][0];
  real_t &ELR = Ez[0][1];
  real_t &ERL = Ez[1][0];
  real_t &ERR = Ez[1][1];

  // Cell centered values
  real_t r = q[ID];
  real_t p = q[IP];
  real_t u = q[IU];
  real_t v = q[IV];
  real_t w = q[IW];            
  real_t A = q[IA];
  real_t B = q[IB];
  real_t C = q[IC];            
    
  // Face centered variables
  real_t AL =  bfNb[0];
  real_t AR =  bfNb[1];
  real_t BL =  bfNb[2];
  real_t BR =  bfNb[3];

  // TODO LATER : compute xL, xR and xC using ::gParam
  // this is only needed when doing cylindrical or spherical coordinates

  // Cell centered TVD slopes in X direction
  real_t drx = dq[IX][ID] * HALF_F;
  real_t dpx = dq[IX][IP] * HALF_F;
  real_t dux = dq[IX][IU] * HALF_F;
  real_t dvx = dq[IX][IV] * HALF_F;
  real_t dwx = dq[IX][IW] * HALF_F;
  real_t dCx = dq[IX][IC] * HALF_F;
  real_t dBx = dq[IX][IB] * HALF_F;
  
  // Cell centered TVD slopes in Y direction
  real_t dry = dq[IY][ID] * HALF_F;
  real_t dpy = dq[IY][IP] * HALF_F;
  real_t duy = dq[IY][IU] * HALF_F;
  real_t dvy = dq[IY][IV] * HALF_F;
  real_t dwy = dq[IY][IW] * HALF_F;
  real_t dCy = dq[IY][IC] * HALF_F;
  real_t dAy = dq[IY][IA] * HALF_F;
 
  // Face centered TVD slopes in transverse direction
  real_t dALy = HALF_F * dAB[0]; //dbf[IY][0];
  real_t dBLx = HALF_F * dAB[1]; //dbf[IX][1];
  real_t dARy = HALF_F * dAB[2]; //dbf[IY][0];
  real_t dBRx = HALF_F * dAB[3]; //dbf[IX][1];
  
  // Cell centered slopes in normal direction
  real_t dAx = HALF_F * (AR - AL);
  real_t dBy = HALF_F * (BR - BL);
  
  // Source terms (including transverse derivatives)
  real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
  real_t sAL0, sAR0, sBL0, sBR0;

  if (true /*cartesian*/) {

    sr0 = (-u*drx-dux*r)                *dtdx + (-v*dry-dvy*r)                *dtdy;
    su0 = (-u*dux-dpx/r-B*dBx/r-C*dCx/r)*dtdx + (-v*duy+B*dAy/r)              *dtdy;
    sv0 = (-u*dvx+A*dBx/r)              *dtdx + (-v*dvy-dpy/r-A*dAy/r-C*dCy/r)*dtdy;
    sw0 = (-u*dwx+A*dCx/r)              *dtdx + (-v*dwy+B*dCy/r)              *dtdy;
    sp0 = (-u*dpx-dux*gamma*p)          *dtdx + (-v*dpy-dvy*gamma*p)          *dtdy;
    sA0 =                                       ( u*dBy+B*duy-v*dAy-A*dvy)    *dtdy;
    sB0 = (-u*dBx-B*dux+v*dAx+A*dvx)    *dtdx ;
    sC0 = ( w*dAx+A*dwx-u*dCx-C*dux)    *dtdx + (-v*dCy-C*dvy+w*dBy+B*dwy)    *dtdy;
    if (Omega0 > ZERO_F) {
      real_t shear = -1.5 * Omega0 * xPos;
      sC0 += (shear * dAx - 1.5 * Omega0 * A) * dtdx;
      sC0 +=  shear * dBy                     * dtdy;
    }

    // Face centered B-field
    sAL0 = +(ELR-ELL)*HALF_F*dtdy;
    sAR0 = +(ERR-ERL)*HALF_F*dtdy;
    sBL0 = -(ERL-ELL)*HALF_F*dtdx;
    sBR0 = -(ERR-ELR)*HALF_F*dtdx;

  } // end cartesian
  
  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  w = w + sw0;
  p = p + sp0;
  A = A + sA0;
  B = B + sB0;
  C = C + sC0;
  
  AL = AL + sAL0;
  AR = AR + sAR0;
  BL = BL + sBL0;
  BR = BR + sBR0;
  
  if (locationId == FACE_XMIN) {
    // Right state at left interface
    qRecons[ID] = r - drx;
    qRecons[IU] = u - dux;
    qRecons[IV] = v - dvx;
    qRecons[IW] = w - dwx;
    qRecons[IP] = p - dpx;
    qRecons[IA] = AL;
    qRecons[IB] = B - dBx;
    qRecons[IC] = C - dCx;
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  if (locationId == FACE_XMAX) {
    // Left state at right interface
    qRecons[ID] = r + drx;
    qRecons[IU] = u + dux;
    qRecons[IV] = v + dvx;
    qRecons[IW] = w + dwx;
    qRecons[IP] = p + dpx;
    qRecons[IA] = AR;
    qRecons[IB] = B + dBx;
    qRecons[IC] = C + dCx;
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  if (locationId == FACE_YMIN) {
    // Top state at bottom interface
    qRecons[ID] = r - dry;
    qRecons[IU] = u - duy;
    qRecons[IV] = v - dvy;
    qRecons[IW] = w - dwy;
    qRecons[IP] = p - dpy;
    qRecons[IA] = A - dAy;
    qRecons[IB] = BL;
    qRecons[IC] = C - dCy;
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  if (locationId == FACE_YMAX) {
    // Bottom state at top interface
    qRecons[ID] = r + dry;
    qRecons[IU] = u + duy;
    qRecons[IV] = v + dvy;
    qRecons[IW] = w + dwy;
    qRecons[IP] = p + dpy;
    qRecons[IA] = A + dAy;
    qRecons[IB] = BR;
    qRecons[IC] = C + dCy;
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  if (locationId == EDGE_RT) {
    // Right-top state (RT->LL)
    qRecons[ID] = r + (+drx+dry);
    qRecons[IU] = u + (+dux+duy);
    qRecons[IV] = v + (+dvx+dvy);
    qRecons[IW] = w + (+dwx+dwy);
    qRecons[IP] = p + (+dpx+dpy);
    qRecons[IA] = AR+ (   +dARy);
    qRecons[IB] = BR+ (+dBRx   );
    qRecons[IC] = C + (+dCx+dCy);
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  if (locationId == EDGE_RB) {
    // Right-Bottom state (RB->LR)
    qRecons[ID] = r + (+drx-dry);
    qRecons[IU] = u + (+dux-duy);
    qRecons[IV] = v + (+dvx-dvy);
    qRecons[IW] = w + (+dwx-dwy);
    qRecons[IP] = p + (+dpx-dpy);
    qRecons[IA] = AR+ (   -dARy);
    qRecons[IB] = BL+ (+dBLx   );
    qRecons[IC] = C + (+dCx-dCy);
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  if (locationId == EDGE_LB) {
    // Left-Bottom state (LB->RR)
    qRecons[ID] = r + (-drx-dry);
    qRecons[IU] = u + (-dux-duy);
    qRecons[IV] = v + (-dvx-dvy);
    qRecons[IW] = w + (-dwx-dwy);
    qRecons[IP] = p + (-dpx-dpy);
    qRecons[IA] = AL+ (   -dALy);
    qRecons[IB] = BL+ (-dBLx   );
    qRecons[IC] = C + (-dCx-dCy);
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  if (locationId == EDGE_LT) {
    // Left-Top state (LT->RL)
    qRecons[ID] = r + (-drx+dry);
    qRecons[IU] = u + (-dux+duy);
    qRecons[IV] = v + (-dvx+dvy);
    qRecons[IW] = w + (-dwx+dwy);
    qRecons[IP] = p + (-dpx+dpy);
    qRecons[IA] = AL+ (   +dALy);
    qRecons[IB] = BR+ (-dBRx   );
    qRecons[IC] = C + (-dCx+dCy);
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

} // trace_unsplit_mhd_2d_face

/**
 * 2D Trace computations for unsplit Godunov scheme.
 *
 * \note Note that this routine uses global variables iorder, scheme and
 * slope_type.
 *
 * \note Note that is routine is loosely adapted from trace2d found in 
 * Dumses and in Ramses sources (sub-dir mhd, file umuscl.f90) to be now a one cell 
 * computation. 
 *
 * \param[in]  q          primitive variables state in current cell
 * \param[in]  dq         primitive variable slopes
 * \param[in]  bfNb       face centered magnetic field (only the first 4 are used)
 * \param[in]  dAB        face-centered magnetic slopes in transverse direction dBx/dy and dBy/dx
 * \param[in]  Ez         electric field
 * \param[in]  dtdx       dt over dx
 * \param[in]  dtdy       dt over dy
 * \param[in]  xPos       x location of current cell (needed for shear computation)
 * \param[in]  locationId identify which cell face or edge is to be reconstructed
 * \param[out] qRecons    the reconstructed state
 */
__DEVICE__ inline
void trace_unsplit_mhd_2d_face2(real_t q[NVAR_MHD],
				real_t dq[2][NVAR_MHD],
				real_t bfNb[TWO_D*2],
				real_t Ez[2][2],
				real_t dtdx,
				real_t dtdy,
				real_t xPos,
				int    locationId,
				real_t (&qRecons)[NVAR_MHD])
{

  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  real_t &gamma  = ::gParams.gamma0;
  real_t &Omega0 = ::gParams.Omega0;

  // Electric field
  real_t &ELL = Ez[0][0];
  real_t &ELR = Ez[0][1];
  real_t &ERL = Ez[1][0];
  real_t &ERR = Ez[1][1];

  // Cell centered values
  real_t r = q[ID];
  real_t p = q[IP];
  real_t u = q[IU];
  real_t v = q[IV];
  real_t w = q[IW];            
  real_t A = q[IA];
  real_t B = q[IB];
  real_t C = q[IC];            
    
  // Face centered variables
  real_t AL =  bfNb[0];
  real_t AR =  bfNb[1];
  real_t BL =  bfNb[2];
  real_t BR =  bfNb[3];

  // TODO LATER : compute xL, xR and xC using ::gParam
  // this is only needed when doing cylindrical or spherical coordinates

  // Cell centered TVD slopes in X direction
  real_t drx = dq[IX][ID] * HALF_F;
  real_t dpx = dq[IX][IP] * HALF_F;
  real_t dux = dq[IX][IU] * HALF_F;
  real_t dvx = dq[IX][IV] * HALF_F;
  real_t dwx = dq[IX][IW] * HALF_F;
  real_t dCx = dq[IX][IC] * HALF_F;
  real_t dBx = dq[IX][IB] * HALF_F;
  
  // Cell centered TVD slopes in Y direction
  real_t dry = dq[IY][ID] * HALF_F;
  real_t dpy = dq[IY][IP] * HALF_F;
  real_t duy = dq[IY][IU] * HALF_F;
  real_t dvy = dq[IY][IV] * HALF_F;
  real_t dwy = dq[IY][IW] * HALF_F;
  real_t dCy = dq[IY][IC] * HALF_F;
  real_t dAy = dq[IY][IA] * HALF_F;
   
  // Cell centered slopes in normal direction
  real_t dAx = HALF_F * (AR - AL);
  real_t dBy = HALF_F * (BR - BL);
  
  // Source terms (including transverse derivatives)
  real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
  real_t sAL0, sAR0, sBL0, sBR0;

  if (true /*cartesian*/) {

    sr0 = (-u*drx-dux*r)                *dtdx + (-v*dry-dvy*r)                *dtdy;
    su0 = (-u*dux-dpx/r-B*dBx/r-C*dCx/r)*dtdx + (-v*duy+B*dAy/r)              *dtdy;
    sv0 = (-u*dvx+A*dBx/r)              *dtdx + (-v*dvy-dpy/r-A*dAy/r-C*dCy/r)*dtdy;
    sw0 = (-u*dwx+A*dCx/r)              *dtdx + (-v*dwy+B*dCy/r)              *dtdy;
    sp0 = (-u*dpx-dux*gamma*p)          *dtdx + (-v*dpy-dvy*gamma*p)          *dtdy;
    sA0 =                                       ( u*dBy+B*duy-v*dAy-A*dvy)    *dtdy;
    sB0 = (-u*dBx-B*dux+v*dAx+A*dvx)    *dtdx ;
    sC0 = ( w*dAx+A*dwx-u*dCx-C*dux)    *dtdx + (-v*dCy-C*dvy+w*dBy+B*dwy)    *dtdy;
    if (Omega0 > ZERO_F) {
      real_t shear = -1.5 * Omega0 * xPos;
      sC0 += (shear * dAx - 1.5 * Omega0 * A) * dtdx;
      sC0 +=  shear * dBy                     * dtdy;
    }
    
    // Face centered B-field
    sAL0 = +(ELR-ELL)*HALF_F*dtdy;
    sAR0 = +(ERR-ERL)*HALF_F*dtdy;
    sBL0 = -(ERL-ELL)*HALF_F*dtdx;
    sBR0 = -(ERR-ELR)*HALF_F*dtdx;

  } // end cartesian
  
  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  w = w + sw0;
  p = p + sp0;
  A = A + sA0;
  B = B + sB0;
  C = C + sC0;
  
  AL = AL + sAL0;
  AR = AR + sAR0;
  BL = BL + sBL0;
  BR = BR + sBR0;
  
  if (locationId == FACE_XMIN) {
    // Right state at left interface
    qRecons[ID] = r - drx;
    qRecons[IU] = u - dux;
    qRecons[IV] = v - dvx;
    qRecons[IW] = w - dwx;
    qRecons[IP] = p - dpx;
    qRecons[IA] = AL;
    qRecons[IB] = B - dBx;
    qRecons[IC] = C - dCx;
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  else if (locationId == FACE_XMAX) {
    // Left state at right interface
    qRecons[ID] = r + drx;
    qRecons[IU] = u + dux;
    qRecons[IV] = v + dvx;
    qRecons[IW] = w + dwx;
    qRecons[IP] = p + dpx;
    qRecons[IA] = AR;
    qRecons[IB] = B + dBx;
    qRecons[IC] = C + dCx;
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  else if (locationId == FACE_YMIN) {
    // Top state at bottom interface
    qRecons[ID] = r - dry;
    qRecons[IU] = u - duy;
    qRecons[IV] = v - dvy;
    qRecons[IW] = w - dwy;
    qRecons[IP] = p - dpy;
    qRecons[IA] = A - dAy;
    qRecons[IB] = BL;
    qRecons[IC] = C - dCy;
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

  else if (locationId == FACE_YMAX) {
    // Bottom state at top interface
    qRecons[ID] = r + dry;
    qRecons[IU] = u + duy;
    qRecons[IV] = v + dvy;
    qRecons[IW] = w + dwy;
    qRecons[IP] = p + dpy;
    qRecons[IA] = A + dAy;
    qRecons[IB] = BR;
    qRecons[IC] = C + dCy;
    qRecons[ID] = FMAX(smallR, qRecons[ID]);
    qRecons[IP] = FMAX(smallp *qRecons[ID], qRecons[IP]);
  }

} // trace_unsplit_mhd_2d_face

/**
 * 3D Trace computations for unsplit Godunov scheme.
 *
 * \note Note that this routine uses global variables iorder, scheme and
 * slope_type.
 *
 * \note Note that is routine is loosely adapted from trace3d found in 
 * Dumses and in Ramses sources (sub-dir mhd, file umuscl.f90) to be now a one cell 
 * computation. 
 *
 * \param[in]  qNb        state in neighbor cells (3-by-3-by-3 neighborhood indexed as qNb[i][j][k], for i,j,k=0,1,2); current center cell is at index (i=j=k=1).
 * \param[in]  bfNb       face centered magnetic field in neighbor cells (4-by-4-by-4 neighborhood indexed as bfNb[i][j][k] for i,j,k=0,1,2,3); current cell is located at index (i=j=k=1)
 * \param[in]  c          local sound speed.
 * \param[in]  dtdx       dt over dx
 * \param[in]  dtdy       dt over dy
 * \param[in]  dtdz       dt over dy
 * \param[in]  xPos       x location of current cell (needed for shear computation)
 * \param[out] qm         qm state (one per dimension)
 * \param[out] qp         qp state (one per dimension)
 * \param[out] qEdge      q state on cell edges (qRT, qRB, qLT, qLB)
 *
 * This device function is too long, moreover routine
 * slope_unsplit_mhd_3d is called 4 times. 
 *
 * \todo Extract magnetic slopes computations (do them in a separate
 * function/kernel). See trace_unsplit_mhd_3d_v2 routine.
 *
 * \todo Simplify interface, we only need bfNb (face-centered magnetic
 * field neighborhood in a cross shaped stencil)
 */
__DEVICE__ inline
void trace_unsplit_mhd_3d(real_t qNb[3][3][3][NVAR_MHD],
			  real_t bfNb[4][4][4][3],
			  real_t c, 
			  real_t dtdx,
			  real_t dtdy,
			  real_t dtdz,
			  real_t xPos,
			  real_t (&qm)[3][NVAR_MHD], 
			  real_t (&qp)[3][NVAR_MHD],
			  real_t (&qEdge)[4][3][NVAR_MHD])
{
  (void) c;
  
  // neighborhood sizes
  enum {Q_SIZE=3, BF_SIZE = 4};

  // index of current cell in the neighborhood
  enum {CENTER=1};

  // alias for q on cell edge (as defined in DUMSES trace2d routine)
  real_t (&qRT_X)[NVAR_MHD] = qEdge[0][0];
  real_t (&qRB_X)[NVAR_MHD] = qEdge[1][0];
  real_t (&qLT_X)[NVAR_MHD] = qEdge[2][0];
  real_t (&qLB_X)[NVAR_MHD] = qEdge[3][0];

  real_t (&qRT_Y)[NVAR_MHD] = qEdge[0][1];
  real_t (&qRB_Y)[NVAR_MHD] = qEdge[1][1];
  real_t (&qLT_Y)[NVAR_MHD] = qEdge[2][1];
  real_t (&qLB_Y)[NVAR_MHD] = qEdge[3][1];

  real_t (&qRT_Z)[NVAR_MHD] = qEdge[0][2];
  real_t (&qRB_Z)[NVAR_MHD] = qEdge[1][2];
  real_t (&qLT_Z)[NVAR_MHD] = qEdge[2][2];
  real_t (&qLB_Z)[NVAR_MHD] = qEdge[3][2];

  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  real_t &gamma  = ::gParams.gamma0;
  real_t &Omega0 = ::gParams.Omega0;
  real_t &dx     = ::gParams.dx;

  real_t (&q)[NVAR_MHD] = qNb[CENTER][CENTER][CENTER]; // current cell (neighborhood center)

  // compute Ex,Ey,Ez (electric field components)
  real_t Ex[2][2];
  for (int dj=0; dj<2; dj++) {
    for (int dk=0; dk<2; dk++) {
      
      int centerX = CENTER;
      int centerY = CENTER+dj;
      int centerZ = CENTER+dk;

      real_t v = 0.25f * (qNb[centerX][centerY-1][centerZ-1][IV] +
			  qNb[centerX][centerY-1][centerZ  ][IV] +
			  qNb[centerX][centerY  ][centerZ-1][IV] +
			  qNb[centerX][centerY  ][centerZ  ][IV]);
      real_t w = 0.25f * (qNb[centerX][centerY-1][centerZ-1][IW] +
			  qNb[centerX][centerY-1][centerZ  ][IW] +
			  qNb[centerX][centerY  ][centerZ-1][IW] +
			  qNb[centerX][centerY  ][centerZ  ][IW]);
      real_t B = 0.5f * (bfNb[centerX][centerY  ][centerZ-1][IY] +
			 bfNb[centerX][centerY  ][centerZ  ][IY]);
      real_t C = 0.5f * (bfNb[centerX][centerY-1][centerZ  ][IZ] +
			 bfNb[centerX][centerY  ][centerZ  ][IZ]);

      Ex[dj][dk] = v*C-w*B;

      if (/* cartesian */ Omega0>0 /* and not fargo*/) {
	real_t shear = -1.5 * Omega0 * xPos;
	Ex[dj][dk] += shear*C;
      }

    } // end for dk
  } // end for dj
  
  real_t Ey[2][2];
  for (int di=0; di<2; di++) {
    for (int dk=0; dk<2; dk++) {

      int centerX = CENTER+di;
      int centerY = CENTER;
      int centerZ = CENTER+dk;

      real_t u = 0.25f * (qNb[centerX-1][centerY][centerZ-1][IU] + 
			  qNb[centerX-1][centerY][centerZ  ][IU] + 
			  qNb[centerX  ][centerY][centerZ-1][IU] + 
			  qNb[centerX  ][centerY][centerZ  ][IU]);  
      real_t w = 0.25f * (qNb[centerX-1][centerY][centerZ-1][IW] +
			  qNb[centerX-1][centerY][centerZ  ][IW] +
			  qNb[centerX  ][centerY][centerZ-1][IW] +
			  qNb[centerX  ][centerY][centerZ  ][IW]);

      real_t A = 0.5f * (bfNb[centerX  ][centerY][centerZ-1][IX] + 
			 bfNb[centerX  ][centerY][centerZ  ][IX]);
      real_t C = 0.5f * (bfNb[centerX-1][centerY][centerZ  ][IZ] +
			 bfNb[centerX  ][centerY][centerZ  ][IZ]);
      Ey[di][dk] = w*A-u*C;

    } // end for dk
  } // end for di

  real_t Ez[2][2];
  for (int di=0; di<2; di++) {
    for (int dj=0; dj<2; dj++) {
      
      int centerX = CENTER+di;
      int centerY = CENTER+dj;
      int centerZ = CENTER;
      real_t u  = 0.25f *  (qNb[centerX-1][centerY-1][centerZ][IU] + 
			    qNb[centerX-1][centerY  ][centerZ][IU] + 
			    qNb[centerX  ][centerY-1][centerZ][IU] + 
			    qNb[centerX  ][centerY  ][centerZ][IU]); 
      
      real_t v  = 0.25f *  (qNb[centerX-1][centerY-1][centerZ][IV] +
			    qNb[centerX-1][centerY  ][centerZ][IV] +
			    qNb[centerX  ][centerY-1][centerZ][IV] + 
			    qNb[centerX  ][centerY  ][centerZ][IV]);
      
      real_t A  = 0.5f  * (bfNb[centerX  ][centerY-1][centerZ][IX] + 
			   bfNb[centerX  ][centerY  ][centerZ][IX]);

      real_t B  = 0.5f  * (bfNb[centerX-1][centerY  ][centerZ][IY] + 
			   bfNb[centerX  ][centerY  ][centerZ][IY]);
      
      Ez[di][dj] = u*B-v*A;

      if (/* cartesian */ Omega0>0 /* and not fargo*/) {
	real_t shear = -1.5 * Omega0 * (xPos - dx/2);
	Ez[di][dj] -= shear*A;
      }

    } // end for dj
  } // end for di

  // Edge centered electric field in X, Y and Z directions
  real_t &ELL = Ex[0][0];
  real_t &ELR = Ex[0][1];
  real_t &ERL = Ex[1][0];
  real_t &ERR = Ex[1][1];

  real_t &FLL = Ey[0][0];
  real_t &FLR = Ey[0][1];
  real_t &FRL = Ey[1][0];
  real_t &FRR = Ey[1][1];
  
  real_t &GLL = Ez[0][0];
  real_t &GLR = Ez[0][1];
  real_t &GRL = Ez[1][0];
  real_t &GRR = Ez[1][1];
  
  // Cell centered values
  real_t r = q[ID];
  real_t p = q[IP];
  real_t u = q[IU];
  real_t v = q[IV];
  real_t w = q[IW];            
  real_t A = q[IA];
  real_t B = q[IB];
  real_t C = q[IC];            
  
  // Face centered variables
  real_t AL =  bfNb[CENTER  ][CENTER  ][CENTER  ][IX];
  real_t AR =  bfNb[CENTER+1][CENTER  ][CENTER  ][IX];
  real_t BL =  bfNb[CENTER  ][CENTER  ][CENTER  ][IY];
  real_t BR =  bfNb[CENTER  ][CENTER+1][CENTER  ][IY];
  real_t CL =  bfNb[CENTER  ][CENTER  ][CENTER  ][IZ];
  real_t CR =  bfNb[CENTER  ][CENTER  ][CENTER+1][IZ];

  /*
   * compute dq slopes
   */
  real_t dq[3][NVAR_MHD];

  slope_unsplit_hydro_3d(qNb, dq);

  // slight modification compared to DUMSES (we re-used dq itself,
  // instead of re-declaring new variables, better for the GPU
  // register count

  // Cell centered TVD slopes in X direction
  real_t& drx = dq[IX][ID];  drx *= HALF_F;
  real_t& dpx = dq[IX][IP];  dpx *= HALF_F;
  real_t& dux = dq[IX][IU];  dux *= HALF_F;
  real_t& dvx = dq[IX][IV];  dvx *= HALF_F;
  real_t& dwx = dq[IX][IW];  dwx *= HALF_F;
  real_t& dCx = dq[IX][IC];  dCx *= HALF_F;
  real_t& dBx = dq[IX][IB];  dBx *= HALF_F;
  
  // Cell centered TVD slopes in Y direction
  real_t& dry = dq[IY][ID];  dry *= HALF_F;
  real_t& dpy = dq[IY][IP];  dpy *= HALF_F;
  real_t& duy = dq[IY][IU];  duy *= HALF_F;
  real_t& dvy = dq[IY][IV];  dvy *= HALF_F;
  real_t& dwy = dq[IY][IW];  dwy *= HALF_F;
  real_t& dCy = dq[IY][IC];  dCy *= HALF_F;
  real_t& dAy = dq[IY][IA];  dAy *= HALF_F;

  // Cell centered TVD slopes in Z direction
  real_t& drz = dq[IZ][ID];  drz *= HALF_F;
  real_t& dpz = dq[IZ][IP];  dpz *= HALF_F;
  real_t& duz = dq[IZ][IU];  duz *= HALF_F;
  real_t& dvz = dq[IZ][IV];  dvz *= HALF_F;
  real_t& dwz = dq[IZ][IW];  dwz *= HALF_F;
  real_t& dAz = dq[IZ][IA];  dAz *= HALF_F;
  real_t& dBz = dq[IZ][IB];  dBz *= HALF_F;

  /*
   * compute dbf slopes needed for Face centered TVD slopes in transverse direction
   */
  real_t bfNeighbors[15];
  real_t dbf[3][3];
  real_t (&dbfX)[3] = dbf[IX];
  real_t (&dbfY)[3] = dbf[IY];
  real_t (&dbfZ)[3] = dbf[IZ];
  
  bfNeighbors[0]  =  bfNb[CENTER  ][CENTER  ][CENTER  ][IX];
  bfNeighbors[1]  =  bfNb[CENTER  ][CENTER+1][CENTER  ][IX];
  bfNeighbors[2]  =  bfNb[CENTER  ][CENTER-1][CENTER  ][IX];
  bfNeighbors[3]  =  bfNb[CENTER  ][CENTER  ][CENTER+1][IX];
  bfNeighbors[4]  =  bfNb[CENTER  ][CENTER  ][CENTER-1][IX];

  bfNeighbors[5]  =  bfNb[CENTER  ][CENTER  ][CENTER  ][IY];
  bfNeighbors[6]  =  bfNb[CENTER+1][CENTER  ][CENTER  ][IY];
  bfNeighbors[7]  =  bfNb[CENTER-1][CENTER  ][CENTER  ][IY];
  bfNeighbors[8]  =  bfNb[CENTER  ][CENTER  ][CENTER+1][IY];
  bfNeighbors[9]  =  bfNb[CENTER  ][CENTER  ][CENTER-1][IY];
  
  bfNeighbors[10] =  bfNb[CENTER  ][CENTER  ][CENTER  ][IZ];
  bfNeighbors[11] =  bfNb[CENTER+1][CENTER  ][CENTER  ][IZ];
  bfNeighbors[12] =  bfNb[CENTER-1][CENTER  ][CENTER  ][IZ];
  bfNeighbors[13] =  bfNb[CENTER  ][CENTER+1][CENTER  ][IZ];
  bfNeighbors[14] =  bfNb[CENTER  ][CENTER-1][CENTER  ][IZ];
  
  slope_unsplit_mhd_3d(bfNeighbors, dbf);

  // Face centered TVD slopes in transverse direction
  real_t dALy = HALF_F * dbfY[IX];
  real_t dALz = HALF_F * dbfZ[IX];
  real_t dBLx = HALF_F * dbfX[IY];
  real_t dBLz = HALF_F * dbfZ[IY];
  real_t dCLx = HALF_F * dbfX[IZ];
  real_t dCLy = HALF_F * dbfY[IZ];
 
  // change neighbors to i+1, j, k and recompute dbf
  bfNeighbors[0]  =  bfNb[CENTER+1][CENTER  ][CENTER  ][IX];
  bfNeighbors[1]  =  bfNb[CENTER+1][CENTER+1][CENTER  ][IX];
  bfNeighbors[2]  =  bfNb[CENTER+1][CENTER-1][CENTER  ][IX];
  bfNeighbors[3]  =  bfNb[CENTER+1][CENTER  ][CENTER+1][IX];
  bfNeighbors[4]  =  bfNb[CENTER+1][CENTER  ][CENTER-1][IX];
  
  bfNeighbors[5]  =  bfNb[CENTER+1][CENTER  ][CENTER  ][IY];
  bfNeighbors[6]  =  bfNb[CENTER+2][CENTER  ][CENTER  ][IY];
  bfNeighbors[7]  =  bfNb[CENTER  ][CENTER  ][CENTER  ][IY];
  bfNeighbors[8]  =  bfNb[CENTER+1][CENTER  ][CENTER+1][IY];
  bfNeighbors[9]  =  bfNb[CENTER+1][CENTER  ][CENTER-1][IY];
  
  bfNeighbors[10] =  bfNb[CENTER+1][CENTER  ][CENTER  ][IZ];
  bfNeighbors[11] =  bfNb[CENTER+2][CENTER  ][CENTER  ][IZ];
  bfNeighbors[12] =  bfNb[CENTER  ][CENTER  ][CENTER  ][IZ];
  bfNeighbors[13] =  bfNb[CENTER+1][CENTER+1][CENTER  ][IZ];
  bfNeighbors[14] =  bfNb[CENTER+1][CENTER-1][CENTER  ][IZ];
  
  slope_unsplit_mhd_3d(bfNeighbors, dbf);
  real_t dARy = HALF_F * dbfY[IX];
  real_t dARz = HALF_F * dbfZ[IX];

  // change neighbors to i, j+1, k and recompute dbf
  bfNeighbors[0]  =  bfNb[CENTER  ][CENTER+1][CENTER  ][IX];
  bfNeighbors[1]  =  bfNb[CENTER  ][CENTER+2][CENTER  ][IX];
  bfNeighbors[2]  =  bfNb[CENTER  ][CENTER  ][CENTER  ][IX];
  bfNeighbors[3]  =  bfNb[CENTER  ][CENTER+1][CENTER+1][IX];
  bfNeighbors[4]  =  bfNb[CENTER  ][CENTER+1][CENTER-1][IX];

  bfNeighbors[5]  =  bfNb[CENTER  ][CENTER+1][CENTER  ][IY];
  bfNeighbors[6]  =  bfNb[CENTER+1][CENTER+1][CENTER  ][IY];
  bfNeighbors[7]  =  bfNb[CENTER-1][CENTER+1][CENTER  ][IY];
  bfNeighbors[8]  =  bfNb[CENTER  ][CENTER+1][CENTER+1][IY];
  bfNeighbors[9]  =  bfNb[CENTER  ][CENTER+1][CENTER-1][IY];
  
  bfNeighbors[10] =  bfNb[CENTER  ][CENTER+1][CENTER  ][IZ];
  bfNeighbors[11] =  bfNb[CENTER+1][CENTER+1][CENTER  ][IZ];
  bfNeighbors[12] =  bfNb[CENTER-1][CENTER+1][CENTER  ][IZ];
  bfNeighbors[13] =  bfNb[CENTER  ][CENTER+2][CENTER  ][IZ];
  bfNeighbors[14] =  bfNb[CENTER  ][CENTER  ][CENTER  ][IZ];
  
  slope_unsplit_mhd_3d(bfNeighbors, dbf);
  real_t dBRx = HALF_F * dbfX[IY];
  real_t dBRz = HALF_F * dbfZ[IY];

  // change neighbors to i, j, k+1 and recompute dbf
  bfNeighbors[0]  =  bfNb[CENTER  ][CENTER  ][CENTER+1][IX];
  bfNeighbors[1]  =  bfNb[CENTER  ][CENTER+1][CENTER+1][IX];
  bfNeighbors[2]  =  bfNb[CENTER  ][CENTER-1][CENTER+1][IX];
  bfNeighbors[3]  =  bfNb[CENTER  ][CENTER  ][CENTER+2][IX];
  bfNeighbors[4]  =  bfNb[CENTER  ][CENTER  ][CENTER  ][IX];

  bfNeighbors[5]  =  bfNb[CENTER  ][CENTER  ][CENTER+1][IY];
  bfNeighbors[6]  =  bfNb[CENTER+1][CENTER  ][CENTER+1][IY];
  bfNeighbors[7]  =  bfNb[CENTER-1][CENTER  ][CENTER+1][IY];
  bfNeighbors[8]  =  bfNb[CENTER  ][CENTER  ][CENTER+2][IY];
  bfNeighbors[9]  =  bfNb[CENTER  ][CENTER  ][CENTER  ][IY];
  
  bfNeighbors[10] =  bfNb[CENTER  ][CENTER  ][CENTER+1][IZ];
  bfNeighbors[11] =  bfNb[CENTER+1][CENTER  ][CENTER+1][IZ];
  bfNeighbors[12] =  bfNb[CENTER-1][CENTER  ][CENTER+1][IZ];
  bfNeighbors[13] =  bfNb[CENTER  ][CENTER+1][CENTER+1][IZ];
  bfNeighbors[14] =  bfNb[CENTER  ][CENTER-1][CENTER+1][IZ];

  slope_unsplit_mhd_3d(bfNeighbors, dbf);
  real_t dCRx = HALF_F * dbfX[IZ];
  real_t dCRy = HALF_F * dbfY[IZ];

  // Cell centered slopes in normal direction
  real_t dAx = HALF_F * (AR - AL);
  real_t dBy = HALF_F * (BR - BL);
  real_t dCz = HALF_F * (CR - CL);

  // Source terms (including transverse derivatives)
  real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
  real_t sAL0, sAR0, sBL0, sBR0, sCL0, sCR0;

  if (true /*cartesian*/) {

    sr0 = (-u*drx-dux*r)              *dtdx + (-v*dry-dvy*r)              *dtdy + (-w*drz-dwz*r)              *dtdz;
    su0 = (-u*dux-(dpx+B*dBx+C*dCx)/r)*dtdx + (-v*duy+B*dAy/r)            *dtdy + (-w*duz+C*dAz/r)            *dtdz; 
    sv0 = (-u*dvx+A*dBx/r)            *dtdx + (-v*dvy-(dpy+A*dAy+C*dCy)/r)*dtdy + (-w*dvz+C*dBz/r)            *dtdz;
    sw0 = (-u*dwx+A*dCx/r)            *dtdx + (-v*dwy+B*dCy/r)            *dtdy + (-w*dwz-(dpz+A*dAz+B*dBz)/r)*dtdz; 
    sp0 = (-u*dpx-dux*gamma*p)        *dtdx + (-v*dpy-dvy*gamma*p)        *dtdy + (-w*dpz-dwz*gamma*p)        *dtdz;
    sA0 =                                     (u*dBy+B*duy-v*dAy-A*dvy)   *dtdy + (u*dCz+C*duz-w*dAz-A*dwz)   *dtdz;
    sB0 = (v*dAx+A*dvx-u*dBx-B*dux)   *dtdx +                                     (v*dCz+C*dvz-w*dBz-B*dwz)   *dtdz; 
    sC0 = (w*dAx+A*dwx-u*dCx-C*dux)   *dtdx + (w*dBy+B*dwy-v*dCy-C*dvy)   *dtdy;
    if (Omega0>0) {
      real_t shear = -1.5 * Omega0 *xPos;
      sr0 = sr0 -  shear*dry*dtdy;
      su0 = su0 -  shear*duy*dtdy;
      sv0 = sv0 -  shear*dvy*dtdy;
      sw0 = sw0 -  shear*dwy*dtdy;
      sp0 = sp0 -  shear*dpy*dtdy;
      sA0 = sA0 -  shear*dAy*dtdy;
      sB0 = sB0 + (shear*dAx - 1.5 * Omega0 * A * dx)*dtdx + shear*dBz*dtdz;
      sC0 = sC0 -  shear*dCy*dtdy;
    }
	
    // Face-centered B-field
    sAL0 = +(GLR-GLL)*dtdy*HALF_F -(FLR-FLL)*dtdz*HALF_F;
    sAR0 = +(GRR-GRL)*dtdy*HALF_F -(FRR-FRL)*dtdz*HALF_F;
    sBL0 = -(GRL-GLL)*dtdx*HALF_F +(ELR-ELL)*dtdz*HALF_F;
    sBR0 = -(GRR-GLR)*dtdx*HALF_F +(ERR-ERL)*dtdz*HALF_F;
    sCL0 = +(FRL-FLL)*dtdx*HALF_F -(ERL-ELL)*dtdy*HALF_F;
    sCR0 = +(FRR-FLR)*dtdx*HALF_F -(ERR-ELR)*dtdy*HALF_F;

  } // end cartesian

  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  w = w + sw0;
  p = p + sp0;
  A = A + sA0;
  B = B + sB0;
  C = C + sC0;
  
  AL = AL + sAL0;
  AR = AR + sAR0;
  BL = BL + sBL0;
  BR = BR + sBR0;
  CL = CL + sCL0;
  CR = CR + sCR0;

  // Face averaged right state at left interface
  qp[0][ID] = r - drx;
  qp[0][IU] = u - dux;
  qp[0][IV] = v - dvx;
  qp[0][IW] = w - dwx;
  qp[0][IP] = p - dpx;
  qp[0][IA] = AL;
  qp[0][IB] = B - dBx;
  qp[0][IC] = C - dCx;
  qp[0][ID] = FMAX(smallR,  qp[0][ID]);
  qp[0][IP] = FMAX(smallp /** qp[0][ID]*/, qp[0][IP]);
  
  // Face averaged left state at right interface
  qm[0][ID] = r + drx;
  qm[0][IU] = u + dux;
  qm[0][IV] = v + dvx;
  qm[0][IW] = w + dwx;
  qm[0][IP] = p + dpx;
  qm[0][IA] = AR;
  qm[0][IB] = B + dBx;
  qm[0][IC] = C + dCx;
  qm[0][ID] = FMAX(smallR,  qm[0][ID]);
  qm[0][IP] = FMAX(smallp /** qm[0][ID]*/, qm[0][IP]);

  // Face averaged top state at bottom interface
  qp[1][ID] = r - dry;
  qp[1][IU] = u - duy;
  qp[1][IV] = v - dvy;
  qp[1][IW] = w - dwy;
  qp[1][IP] = p - dpy;
  qp[1][IA] = A - dAy;
  qp[1][IB] = BL;
  qp[1][IC] = C - dCy;
  qp[1][ID] = FMAX(smallR,  qp[1][ID]);
  qp[1][IP] = FMAX(smallp /** qp[1][ID]*/, qp[1][IP]);
  
  // Face averaged bottom state at top interface
  qm[1][ID] = r + dry;
  qm[1][IU] = u + duy;
  qm[1][IV] = v + dvy;
  qm[1][IW] = w + dwy;
  qm[1][IP] = p + dpy;
  qm[1][IA] = A + dAy;
  qm[1][IB] = BR;
  qm[1][IC] = C + dCy;
  qm[1][ID] = FMAX(smallR,  qm[1][ID]);
  qm[1][IP] = FMAX(smallp /** qm[1][ID]*/, qm[1][IP]);
  
  // Face averaged front state at back interface
  qp[2][ID] = r - drz;
  qp[2][IU] = u - duz;
  qp[2][IV] = v - dvz;
  qp[2][IW] = w - dwz;
  qp[2][IP] = p - dpz;
  qp[2][IA] = A - dAz;
  qp[2][IB] = B - dBz;
  qp[2][IC] = CL;
  qp[2][ID] = FMAX(smallR,  qp[2][ID]);
  qp[2][IP] = FMAX(smallp /** qp[2][ID]*/, qp[2][IP]);
  
  // Face averaged back state at front interface
  qm[2][ID] = r + drz;
  qm[2][IU] = u + duz;
  qm[2][IV] = v + dvz;
  qm[2][IW] = w + dwz;
  qm[2][IP] = p + dpz;
  qm[2][IA] = A + dAz;
  qm[2][IB] = B + dBz;
  qm[2][IC] = CR;
  qm[2][ID] = FMAX(smallR,  qm[2][ID]);
  qm[2][IP] = FMAX(smallp /** qm[2][ID]*/, qm[2][IP]);

  // X-edge averaged right-top corner state (RT->LL)
  qRT_X[ID] = r + (+dry+drz);
  qRT_X[IU] = u + (+duy+duz);
  qRT_X[IV] = v + (+dvy+dvz);
  qRT_X[IW] = w + (+dwy+dwz);
  qRT_X[IP] = p + (+dpy+dpz);
  qRT_X[IA] = A + (+dAy+dAz);
  qRT_X[IB] = BR+ (   +dBRz);
  qRT_X[IC] = CR+ (+dCRy   );
  qRT_X[ID] = FMAX(smallR,  qRT_X[ID]);
  qRT_X[IP] = FMAX(smallp /** qRT_X[ID]*/, qRT_X[IP]);
  
  // X-edge averaged right-bottom corner state (RB->LR)
  qRB_X[ID] = r + (+dry-drz);
  qRB_X[IU] = u + (+duy-duz);
  qRB_X[IV] = v + (+dvy-dvz);
  qRB_X[IW] = w + (+dwy-dwz);
  qRB_X[IP] = p + (+dpy-dpz);
  qRB_X[IA] = A + (+dAy-dAz);
  qRB_X[IB] = BR+ (   -dBRz);
  qRB_X[IC] = CL+ (+dCLy   );
  qRB_X[ID] = FMAX(smallR,  qRB_X[ID]);
  qRB_X[IP] = FMAX(smallp /** qRB_X[ID]*/, qRB_X[IP]);
  
  // X-edge averaged left-top corner state (LT->RL)
  qLT_X[ID] = r + (-dry+drz);
  qLT_X[IU] = u + (-duy+duz);
  qLT_X[IV] = v + (-dvy+dvz);
  qLT_X[IW] = w + (-dwy+dwz);
  qLT_X[IP] = p + (-dpy+dpz);
  qLT_X[IA] = A + (-dAy+dAz);
  qLT_X[IB] = BL+ (   +dBLz);
  qLT_X[IC] = CR+ (-dCRy   );
  qLT_X[ID] = FMAX(smallR,  qLT_X[ID]);
  qLT_X[IP] = FMAX(smallp /** qLT_X[ID]*/, qLT_X[IP]);
  
  // X-edge averaged left-bottom corner state (LB->RR)
  qLB_X[ID] = r + (-dry-drz);
  qLB_X[IU] = u + (-duy-duz);
  qLB_X[IV] = v + (-dvy-dvz);
  qLB_X[IW] = w + (-dwy-dwz);
  qLB_X[IP] = p + (-dpy-dpz);
  qLB_X[IA] = A + (-dAy-dAz);
  qLB_X[IB] = BL+ (   -dBLz);
  qLB_X[IC] = CL+ (-dCLy   );
  qLB_X[ID] = FMAX(smallR,  qLB_X[ID]);
  qLB_X[IP] = FMAX(smallp /** qLB_X[ID]*/, qLB_X[IP]);
  
  // Y-edge averaged right-top corner state (RT->LL)
  qRT_Y[ID] = r + (+drx+drz);
  qRT_Y[IU] = u + (+dux+duz);
  qRT_Y[IV] = v + (+dvx+dvz);
  qRT_Y[IW] = w + (+dwx+dwz);
  qRT_Y[IP] = p + (+dpx+dpz);
  qRT_Y[IA] = AR+ (   +dARz);
  qRT_Y[IB] = B + (+dBx+dBz);
  qRT_Y[IC] = CR+ (+dCRx   );
  qRT_Y[ID] = FMAX(smallR,  qRT_Y[ID]);
  qRT_Y[IP] = FMAX(smallp /** qRT_Y[ID]*/, qRT_Y[IP]);
  
  // Y-edge averaged right-bottom corner state (RB->LR)
  qRB_Y[ID] = r + (+drx-drz);
  qRB_Y[IU] = u + (+dux-duz);
  qRB_Y[IV] = v + (+dvx-dvz);
  qRB_Y[IW] = w + (+dwx-dwz);
  qRB_Y[IP] = p + (+dpx-dpz);
  qRB_Y[IA] = AR+ (   -dARz);
  qRB_Y[IB] = B + (+dBx-dBz);
  qRB_Y[IC] = CL+ (+dCLx   );
  qRB_Y[ID] = FMAX(smallR,  qRB_Y[ID]);
  qRB_Y[IP] = FMAX(smallp /** qRB_Y[ID]*/, qRB_Y[IP]);
  
  // Y-edge averaged left-top corner state (LT->RL)
  qLT_Y[ID] = r + (-drx+drz);
  qLT_Y[IU] = u + (-dux+duz);
  qLT_Y[IV] = v + (-dvx+dvz);
  qLT_Y[IW] = w + (-dwx+dwz);
  qLT_Y[IP] = p + (-dpx+dpz);
  qLT_Y[IA] = AL+ (   +dALz);
  qLT_Y[IB] = B + (-dBx+dBz);
  qLT_Y[IC] = CR+ (-dCRx   );
  qLT_Y[ID] = FMAX(smallR,  qLT_Y[ID]);
  qLT_Y[IP] = FMAX(smallp /** qLT_Y[ID]*/, qLT_Y[IP]);
  
  // Y-edge averaged left-bottom corner state (LB->RR)
  qLB_Y[ID] = r + (-drx-drz);
  qLB_Y[IU] = u + (-dux-duz);
  qLB_Y[IV] = v + (-dvx-dvz);
  qLB_Y[IW] = w + (-dwx-dwz);
  qLB_Y[IP] = p + (-dpx-dpz);
  qLB_Y[IA] = AL+ (   -dALz);
  qLB_Y[IB] = B + (-dBx-dBz);
  qLB_Y[IC] = CL+ (-dCLx   );
  qLB_Y[ID] = FMAX(smallR,  qLB_Y[ID]);
  qLB_Y[IP] = FMAX(smallp /** qLB_Y[ID]*/, qLB_Y[IP]);
  
  // Z-edge averaged right-top corner state (RT->LL)
  qRT_Z[ID] = r + (+drx+dry);
  qRT_Z[IU] = u + (+dux+duy);
  qRT_Z[IV] = v + (+dvx+dvy);
  qRT_Z[IW] = w + (+dwx+dwy);
  qRT_Z[IP] = p + (+dpx+dpy);
  qRT_Z[IA] = AR+ (   +dARy);
  qRT_Z[IB] = BR+ (+dBRx   );
  qRT_Z[IC] = C + (+dCx+dCy);
  qRT_Z[ID] = FMAX(smallR,  qRT_Z[ID]);
  qRT_Z[IP] = FMAX(smallp /** qRT_Z[ID]*/, qRT_Z[IP]);
  
  // Z-edge averaged right-bottom corner state (RB->LR)
  qRB_Z[ID] = r + (+drx-dry);
  qRB_Z[IU] = u + (+dux-duy);
  qRB_Z[IV] = v + (+dvx-dvy);
  qRB_Z[IW] = w + (+dwx-dwy);
  qRB_Z[IP] = p + (+dpx-dpy);
  qRB_Z[IA] = AR+ (   -dARy);
  qRB_Z[IB] = BL+ (+dBLx   );
  qRB_Z[IC] = C + (+dCx-dCy);
  qRB_Z[ID] = FMAX(smallR,  qRB_Z[ID]);
  qRB_Z[IP] = FMAX(smallp /** qRB_Z[ID]*/, qRB_Z[IP]);
  
  // Z-edge averaged left-top corner state (LT->RL)
  qLT_Z[ID] = r + (-drx+dry);
  qLT_Z[IU] = u + (-dux+duy);
  qLT_Z[IV] = v + (-dvx+dvy);
  qLT_Z[IW] = w + (-dwx+dwy);
  qLT_Z[IP] = p + (-dpx+dpy);
  qLT_Z[IA] = AL+ (   +dALy);
  qLT_Z[IB] = BR+ (-dBRx   );
  qLT_Z[IC] = C + (-dCx+dCy);
  qLT_Z[ID] = FMAX(smallR,  qLT_Z[ID]);
  qLT_Z[IP] = FMAX(smallp /** qLT_Z[ID]*/, qLT_Z[IP]);
  
  // Z-edge averaged left-bottom corner state (LB->RR)
  qLB_Z[ID] = r + (-drx-dry);
  qLB_Z[IU] = u + (-dux-duy);
  qLB_Z[IV] = v + (-dvx-dvy);
  qLB_Z[IW] = w + (-dwx-dwy);
  qLB_Z[IP] = p + (-dpx-dpy);
  qLB_Z[IA] = AL+ (   -dALy);
  qLB_Z[IB] = BL+ (-dBLx   );
  qLB_Z[IC] = C + (-dCx-dCy);
  qLB_Z[ID] = FMAX(smallR,  qLB_Z[ID]);
  qLB_Z[IP] = FMAX(smallp /** qLB_Z[ID]*/, qLB_Z[IP]);
  
} //trace_unsplit_mhd_3d

/**
 * 3D Trace computations for unsplit Godunov scheme.
 *
 * \note Note that this routine uses global variables iorder, scheme and
 * slope_type.
 *
 * \note Note that is routine is loosely adapted from trace3d found in 
 * Dumses and in Ramses sources (sub-dir mhd, file umuscl.f90) to be now a one cell 
 * computation. 
 *
 * \param[in]  q          primitive variables state in current cell.
 * \param[in]  dq         primitive variable slopes
 * \param[in]  bfNb       face centered magnetic field 
 * \param[in]  dABC       face-centered magnetic slopes in transverse direction dBx/dy, dBy/dx, etc ...
 * \param[in]  Exyz       electric field
 * \param[in]  dtdx       dt over dx
 * \param[in]  dtdy       dt over dy
 * \param[in]  dtdz       dt over dy
 * \param[in]  xPos       x location of current cell (needed for shear computation)
 * \param[in]  locationId identify which cell face or edge is to be reconstructed
 * \param[out] qRecons    the reconstructed state
 *
 */
__DEVICE__ inline
void trace_unsplit_mhd_3d_face(real_t q[NVAR_MHD],
			       real_t dq[3][NVAR_MHD],
			       real_t bfNb[THREE_D*2],
			       real_t dABC[THREE_D*4],
			       real_t Exyz[THREE_D][2][2],
			       real_t dtdx,
			       real_t dtdy,
			       real_t dtdz,
			       real_t xPos,
			       int locationId,
			       real_t (&qRecons)[NVAR_MHD])
{
  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  real_t &gamma  = ::gParams.gamma0;
  real_t &Omega0 = ::gParams.Omega0;
  real_t &dx     = ::gParams.dx;

  // neighborhood sizes
  enum {Q_SIZE=3, BF_SIZE = 4};

  // index of current cell in the neighborhood
  enum {CENTER=1};

  // Edge centered electric field in X, Y and Z directions
  real_t &ELL = Exyz[IX][0][0];
  real_t &ELR = Exyz[IX][0][1];
  real_t &ERL = Exyz[IX][1][0];
  real_t &ERR = Exyz[IX][1][1];

  real_t &FLL = Exyz[IY][0][0];
  real_t &FLR = Exyz[IY][0][1];
  real_t &FRL = Exyz[IY][1][0];
  real_t &FRR = Exyz[IY][1][1];
  
  real_t &GLL = Exyz[IZ][0][0];
  real_t &GLR = Exyz[IZ][0][1];
  real_t &GRL = Exyz[IZ][1][0];
  real_t &GRR = Exyz[IZ][1][1];
  
  // Cell centered values
  real_t r = q[ID];
  real_t p = q[IP];
  real_t u = q[IU];
  real_t v = q[IV];
  real_t w = q[IW];            
  real_t A = q[IA];
  real_t B = q[IB];
  real_t C = q[IC];            
  
  // Face centered variables
  real_t AL =  bfNb[0];
  real_t AR =  bfNb[1];
  real_t BL =  bfNb[2];
  real_t BR =  bfNb[3];
  real_t CL =  bfNb[4];
  real_t CR =  bfNb[5];

  // Cell centered TVD slopes in X direction
  real_t drx = dq[IX][ID] * HALF_F;
  real_t dpx = dq[IX][IP] * HALF_F;
  real_t dux = dq[IX][IU] * HALF_F;
  real_t dvx = dq[IX][IV] * HALF_F;
  real_t dwx = dq[IX][IW] * HALF_F;
  real_t dCx = dq[IX][IC] * HALF_F;
  real_t dBx = dq[IX][IB] * HALF_F;
  
  // Cell centered TVD slopes in Y direction
  real_t dry = dq[IY][ID] * HALF_F;
  real_t dpy = dq[IY][IP] * HALF_F;
  real_t duy = dq[IY][IU] * HALF_F;
  real_t dvy = dq[IY][IV] * HALF_F;
  real_t dwy = dq[IY][IW] * HALF_F;
  real_t dCy = dq[IY][IC] * HALF_F;
  real_t dAy = dq[IY][IA] * HALF_F;

  // Cell centered TVD slopes in Z direction
  real_t drz = dq[IZ][ID] * HALF_F;
  real_t dpz = dq[IZ][IP] * HALF_F;
  real_t duz = dq[IZ][IU] * HALF_F;
  real_t dvz = dq[IZ][IV] * HALF_F;
  real_t dwz = dq[IZ][IW] * HALF_F;
  real_t dAz = dq[IZ][IA] * HALF_F;
  real_t dBz = dq[IZ][IB] * HALF_F;

  /*
   * Face centered TVD slopes in transverse direction
   */
  real_t dALy = HALF_F * dABC[0];
  real_t dALz = HALF_F * dABC[1];
  real_t dBLx = HALF_F * dABC[2];
  real_t dBLz = HALF_F * dABC[3];
  real_t dCLx = HALF_F * dABC[4];
  real_t dCLy = HALF_F * dABC[5];
 
  real_t dARy = HALF_F * dABC[6];
  real_t dARz = HALF_F * dABC[7];

  real_t dBRx = HALF_F * dABC[8];
  real_t dBRz = HALF_F * dABC[9];

  real_t dCRx = HALF_F * dABC[10];
  real_t dCRy = HALF_F * dABC[11];

  // Cell centered slopes in normal direction
  real_t dAx = HALF_F * (AR - AL);
  real_t dBy = HALF_F * (BR - BL);
  real_t dCz = HALF_F * (CR - CL);

  // Source terms (including transverse derivatives)
  real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
  real_t sAL0, sAR0, sBL0, sBR0, sCL0, sCR0;

  if (true /*cartesian*/) {

    sr0 = (-u*drx-dux*r)              *dtdx + (-v*dry-dvy*r)              *dtdy + (-w*drz-dwz*r)              *dtdz;
    su0 = (-u*dux-(dpx+B*dBx+C*dCx)/r)*dtdx + (-v*duy+B*dAy/r)            *dtdy + (-w*duz+C*dAz/r)            *dtdz; 
    sv0 = (-u*dvx+A*dBx/r)            *dtdx + (-v*dvy-(dpy+A*dAy+C*dCy)/r)*dtdy + (-w*dvz+C*dBz/r)            *dtdz;
    sw0 = (-u*dwx+A*dCx/r)            *dtdx + (-v*dwy+B*dCy/r)            *dtdy + (-w*dwz-(dpz+A*dAz+B*dBz)/r)*dtdz; 
    sp0 = (-u*dpx-dux*gamma*p)        *dtdx + (-v*dpy-dvy*gamma*p)        *dtdy + (-w*dpz-dwz*gamma*p)        *dtdz;
    sA0 =                                     (u*dBy+B*duy-v*dAy-A*dvy)   *dtdy + (u*dCz+C*duz-w*dAz-A*dwz)   *dtdz;
    sB0 = (v*dAx+A*dvx-u*dBx-B*dux)   *dtdx +                                     (v*dCz+C*dvz-w*dBz-B*dwz)   *dtdz; 
    sC0 = (w*dAx+A*dwx-u*dCx-C*dux)   *dtdx + (w*dBy+B*dwy-v*dCy-C*dvy)   *dtdy;
    
    if (Omega0>0) {
      real_t shear = -1.5 * Omega0 *xPos;
      sr0 = sr0 -  shear*dry*dtdy;
      su0 = su0 -  shear*duy*dtdy;
      sv0 = sv0 -  shear*dvy*dtdy;
      sw0 = sw0 -  shear*dwy*dtdy;
      sp0 = sp0 -  shear*dpy*dtdy;
      sA0 = sA0 -  shear*dAy*dtdy;
      sB0 = sB0 + (shear*dAx - 1.5 * Omega0 * A * dx)*dtdx + shear*dBz*dtdz;
      sC0 = sC0 -  shear*dCy*dtdy;
    }
	
    // Face-centered B-field
    sAL0 = +(GLR-GLL)*dtdy*HALF_F -(FLR-FLL)*dtdz*HALF_F;
    sAR0 = +(GRR-GRL)*dtdy*HALF_F -(FRR-FRL)*dtdz*HALF_F;
    sBL0 = -(GRL-GLL)*dtdx*HALF_F +(ELR-ELL)*dtdz*HALF_F;
    sBR0 = -(GRR-GLR)*dtdx*HALF_F +(ERR-ERL)*dtdz*HALF_F;
    sCL0 = +(FRL-FLL)*dtdx*HALF_F -(ERL-ELL)*dtdy*HALF_F;
    sCR0 = +(FRR-FLR)*dtdx*HALF_F -(ERR-ELR)*dtdy*HALF_F;

  } // end cartesian

  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  w = w + sw0;
  p = p + sp0;
  A = A + sA0;
  B = B + sB0;
  C = C + sC0;
  
  AL = AL + sAL0;
  AR = AR + sAR0;
  BL = BL + sBL0;
  BR = BR + sBR0;
  CL = CL + sCL0;
  CR = CR + sCR0;

  if (locationId == FACE_XMIN) {
    // Face averaged right state at left interface
    qRecons[ID] = r - drx;
    qRecons[IU] = u - dux;
    qRecons[IV] = v - dvx;
    qRecons[IW] = w - dwx;
    qRecons[IP] = p - dpx;
    qRecons[IA] = AL;
    qRecons[IB] = B - dBx;
    qRecons[IC] = C - dCx;
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == FACE_XMAX) {
    // Face averaged left state at right interface
    qRecons[ID] = r + drx;
    qRecons[IU] = u + dux;
    qRecons[IV] = v + dvx;
    qRecons[IW] = w + dwx;
    qRecons[IP] = p + dpx;
    qRecons[IA] = AR;
    qRecons[IB] = B + dBx;
    qRecons[IC] = C + dCx;
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == FACE_YMIN) {
    // Face averaged top state at bottom interface
    qRecons[ID] = r - dry;
    qRecons[IU] = u - duy;
    qRecons[IV] = v - dvy;
    qRecons[IW] = w - dwy;
    qRecons[IP] = p - dpy;
    qRecons[IA] = A - dAy;
    qRecons[IB] = BL;
    qRecons[IC] = C - dCy;
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }
  
  else if (locationId == FACE_YMAX) {
    // Face averaged bottom state at top interface
    qRecons[ID] = r + dry;
    qRecons[IU] = u + duy;
    qRecons[IV] = v + dvy;
    qRecons[IW] = w + dwy;
    qRecons[IP] = p + dpy;
    qRecons[IA] = A + dAy;
    qRecons[IB] = BR;
    qRecons[IC] = C + dCy;
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == FACE_ZMIN) {
    // Face averaged front state at back interface
    qRecons[ID] = r - drz;
    qRecons[IU] = u - duz;
    qRecons[IV] = v - dvz;
    qRecons[IW] = w - dwz;
    qRecons[IP] = p - dpz;
    qRecons[IA] = A - dAz;
    qRecons[IB] = B - dBz;
    qRecons[IC] = CL;
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == FACE_ZMAX) {
    // Face averaged back state at front interface
    qRecons[ID] = r + drz;
    qRecons[IU] = u + duz;
    qRecons[IV] = v + dvz;
    qRecons[IW] = w + dwz;
    qRecons[IP] = p + dpz;
    qRecons[IA] = A + dAz;
    qRecons[IB] = B + dBz;
    qRecons[IC] = CR;
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_RT_X) {
    // X-edge averaged right-top corner state (RT->LL)
    qRecons[ID] = r + (+dry+drz);
    qRecons[IU] = u + (+duy+duz);
    qRecons[IV] = v + (+dvy+dvz);
    qRecons[IW] = w + (+dwy+dwz);
    qRecons[IP] = p + (+dpy+dpz);
    qRecons[IA] = A + (+dAy+dAz);
    qRecons[IB] = BR+ (   +dBRz);
    qRecons[IC] = CR+ (+dCRy   );
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_RB_X) {
    // X-edge averaged right-bottom corner state (RB->LR)
    qRecons[ID] = r + (+dry-drz);
    qRecons[IU] = u + (+duy-duz);
    qRecons[IV] = v + (+dvy-dvz);
    qRecons[IW] = w + (+dwy-dwz);
    qRecons[IP] = p + (+dpy-dpz);
    qRecons[IA] = A + (+dAy-dAz);
    qRecons[IB] = BR+ (   -dBRz);
    qRecons[IC] = CL+ (+dCLy   );
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_LT_X) {
    // X-edge averaged left-top corner state (LT->RL)
    qRecons[ID] = r + (-dry+drz);
    qRecons[IU] = u + (-duy+duz);
    qRecons[IV] = v + (-dvy+dvz);
    qRecons[IW] = w + (-dwy+dwz);
    qRecons[IP] = p + (-dpy+dpz);
    qRecons[IA] = A + (-dAy+dAz);
    qRecons[IB] = BL+ (   +dBLz);
    qRecons[IC] = CR+ (-dCRy   );
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_LB_X) {
    // X-edge averaged left-bottom corner state (LB->RR)
    qRecons[ID] = r + (-dry-drz);
    qRecons[IU] = u + (-duy-duz);
    qRecons[IV] = v + (-dvy-dvz);
    qRecons[IW] = w + (-dwy-dwz);
    qRecons[IP] = p + (-dpy-dpz);
    qRecons[IA] = A + (-dAy-dAz);
    qRecons[IB] = BL+ (   -dBLz);
    qRecons[IC] = CL+ (-dCLy   );
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_RT_Y) {
    // Y-edge averaged right-top corner state (RT->LL)
    qRecons[ID] = r + (+drx+drz);
    qRecons[IU] = u + (+dux+duz);
    qRecons[IV] = v + (+dvx+dvz);
    qRecons[IW] = w + (+dwx+dwz);
    qRecons[IP] = p + (+dpx+dpz);
    qRecons[IA] = AR+ (   +dARz);
    qRecons[IB] = B + (+dBx+dBz);
    qRecons[IC] = CR+ (+dCRx   );
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_RB_Y) {
    // Y-edge averaged right-bottom corner state (RB->LR)
    qRecons[ID] = r + (+drx-drz);
    qRecons[IU] = u + (+dux-duz);
    qRecons[IV] = v + (+dvx-dvz);
    qRecons[IW] = w + (+dwx-dwz);
    qRecons[IP] = p + (+dpx-dpz);
    qRecons[IA] = AR+ (   -dARz);
    qRecons[IB] = B + (+dBx-dBz);
    qRecons[IC] = CL+ (+dCLx   );
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_LT_Y) {
    // Y-edge averaged left-top corner state (LT->RL)
    qRecons[ID] = r + (-drx+drz);
    qRecons[IU] = u + (-dux+duz);
    qRecons[IV] = v + (-dvx+dvz);
    qRecons[IW] = w + (-dwx+dwz);
    qRecons[IP] = p + (-dpx+dpz);
    qRecons[IA] = AL+ (   +dALz);
    qRecons[IB] = B + (-dBx+dBz);
    qRecons[IC] = CR+ (-dCRx   );
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_LB_Y) {
    // Y-edge averaged left-bottom corner state (LB->RR)
    qRecons[ID] = r + (-drx-drz);
    qRecons[IU] = u + (-dux-duz);
    qRecons[IV] = v + (-dvx-dvz);
    qRecons[IW] = w + (-dwx-dwz);
    qRecons[IP] = p + (-dpx-dpz);
    qRecons[IA] = AL+ (   -dALz);
    qRecons[IB] = B + (-dBx-dBz);
    qRecons[IC] = CL+ (-dCLx   );
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_RT_Z) {
    // Z-edge averaged right-top corner state (RT->LL)
    qRecons[ID] = r + (+drx+dry);
    qRecons[IU] = u + (+dux+duy);
    qRecons[IV] = v + (+dvx+dvy);
    qRecons[IW] = w + (+dwx+dwy);
    qRecons[IP] = p + (+dpx+dpy);
    qRecons[IA] = AR+ (   +dARy);
    qRecons[IB] = BR+ (+dBRx   );
    qRecons[IC] = C + (+dCx+dCy);
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_RB_Z) {
    // Z-edge averaged right-bottom corner state (RB->LR)
    qRecons[ID] = r + (+drx-dry);
    qRecons[IU] = u + (+dux-duy);
    qRecons[IV] = v + (+dvx-dvy);
    qRecons[IW] = w + (+dwx-dwy);
    qRecons[IP] = p + (+dpx-dpy);
    qRecons[IA] = AR+ (   -dARy);
    qRecons[IB] = BL+ (+dBLx   );
    qRecons[IC] = C + (+dCx-dCy);
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_LT_Z) {
    // Z-edge averaged left-top corner state (LT->RL)
    qRecons[ID] = r + (-drx+dry);
    qRecons[IU] = u + (-dux+duy);
    qRecons[IV] = v + (-dvx+dvy);
    qRecons[IW] = w + (-dwx+dwy);
    qRecons[IP] = p + (-dpx+dpy);
    qRecons[IA] = AL+ (   +dALy);
    qRecons[IB] = BR+ (-dBRx   );
    qRecons[IC] = C + (-dCx+dCy);
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

  else if (locationId == EDGE_LB_Z) {
    // Z-edge averaged left-bottom corner state (LB->RR)
    qRecons[ID] = r + (-drx-dry);
    qRecons[IU] = u + (-dux-duy);
    qRecons[IV] = v + (-dvx-dvy);
    qRecons[IW] = w + (-dwx-dwy);
    qRecons[IP] = p + (-dpx-dpy);
    qRecons[IA] = AL+ (   -dALy);
    qRecons[IB] = BL+ (-dBLx   );
    qRecons[IC] = C + (-dCx-dCy);
    qRecons[ID] = FMAX(smallR,  qRecons[ID]);
    qRecons[IP] = FMAX(smallp /** qRecons[ID]*/, qRecons[IP]);
  }

} //trace_unsplit_mhd_3d_face

/**
 * This another implementation of trace computations simpler than 
 * trace_unsplit_mhd_3d.
 *
 * By simpler, we mean to design a device function that could lead to better
 * ressource utilization and thus better performances (hopefully).
 *
 * To achieve this goal, several modifications are brought (compared to 
 * trace_unsplit_mhd_3d) :
 * - hydro slopes (call to slope_unsplit_hydro_3d is done outside)
 * - face-centered magnetic field slopes is done outside and before, so it is
 *   an input now
 * - electric field computation is done outside and before (probably in a 
 *   separate CUDA kernel as for the GPU version), so it is now an input 
 *
 *
 */
__DEVICE__ inline
void trace_unsplit_mhd_3d_simpler(real_t q[NVAR_MHD],
				  real_t dq[THREE_D][NVAR_MHD],
				  real_t bfNb[THREE_D*2], /* 2 faces per direction*/
				  real_t dbf[12],
				  real_t elecFields[THREE_D][2][2],
				  real_t dtdx,
				  real_t dtdy,
				  real_t dtdz,
				  real_t xPos,
				  real_t (&qm)[THREE_D][NVAR_MHD], 
				  real_t (&qp)[THREE_D][NVAR_MHD],
				  real_t (&qEdge)[4][3][NVAR_MHD])
{
  
  // inputs
  // alias to electric field components
  real_t (&Ex)[2][2] = elecFields[IX];
  real_t (&Ey)[2][2] = elecFields[IY];
  real_t (&Ez)[2][2] = elecFields[IZ];

  // outputs
  // alias for q on cell edge (as defined in DUMSES trace2d routine)
  real_t (&qRT_X)[NVAR_MHD] = qEdge[0][0];
  real_t (&qRB_X)[NVAR_MHD] = qEdge[1][0];
  real_t (&qLT_X)[NVAR_MHD] = qEdge[2][0];
  real_t (&qLB_X)[NVAR_MHD] = qEdge[3][0];

  real_t (&qRT_Y)[NVAR_MHD] = qEdge[0][1];
  real_t (&qRB_Y)[NVAR_MHD] = qEdge[1][1];
  real_t (&qLT_Y)[NVAR_MHD] = qEdge[2][1];
  real_t (&qLB_Y)[NVAR_MHD] = qEdge[3][1];

  real_t (&qRT_Z)[NVAR_MHD] = qEdge[0][2];
  real_t (&qRB_Z)[NVAR_MHD] = qEdge[1][2];
  real_t (&qLT_Z)[NVAR_MHD] = qEdge[2][2];
  real_t (&qLB_Z)[NVAR_MHD] = qEdge[3][2];

  real_t &gamma  = ::gParams.gamma0;
  real_t &smallR = ::gParams.smallr;
  real_t &smallp = ::gParams.smallp;
  real_t &Omega0 = ::gParams.Omega0;
  real_t &dx     = ::gParams.dx;

  // Edge centered electric field in X, Y and Z directions
  real_t &ELL = Ex[0][0];
  real_t &ELR = Ex[0][1];
  real_t &ERL = Ex[1][0];
  real_t &ERR = Ex[1][1];

  real_t &FLL = Ey[0][0];
  real_t &FLR = Ey[0][1];
  real_t &FRL = Ey[1][0];
  real_t &FRR = Ey[1][1];
  
  real_t &GLL = Ez[0][0];
  real_t &GLR = Ez[0][1];
  real_t &GRL = Ez[1][0];
  real_t &GRR = Ez[1][1];
  
  // Cell centered values
  real_t r = q[ID];
  real_t p = q[IP];
  real_t u = q[IU];
  real_t v = q[IV];
  real_t w = q[IW];            
  real_t A = q[IA];
  real_t B = q[IB];
  real_t C = q[IC];            

  // Face centered variables
  real_t AL =  bfNb[0];
  real_t AR =  bfNb[1];
  real_t BL =  bfNb[2];
  real_t BR =  bfNb[3];
  real_t CL =  bfNb[4];
  real_t CR =  bfNb[5];

  // Cell centered TVD slopes in X direction
  real_t& drx = dq[IX][ID];  drx *= HALF_F;
  real_t& dpx = dq[IX][IP];  dpx *= HALF_F;
  real_t& dux = dq[IX][IU];  dux *= HALF_F;
  real_t& dvx = dq[IX][IV];  dvx *= HALF_F;
  real_t& dwx = dq[IX][IW];  dwx *= HALF_F;
  real_t& dCx = dq[IX][IC];  dCx *= HALF_F;
  real_t& dBx = dq[IX][IB];  dBx *= HALF_F;
  
  // Cell centered TVD slopes in Y direction
  real_t& dry = dq[IY][ID];  dry *= HALF_F;
  real_t& dpy = dq[IY][IP];  dpy *= HALF_F;
  real_t& duy = dq[IY][IU];  duy *= HALF_F;
  real_t& dvy = dq[IY][IV];  dvy *= HALF_F;
  real_t& dwy = dq[IY][IW];  dwy *= HALF_F;
  real_t& dCy = dq[IY][IC];  dCy *= HALF_F;
  real_t& dAy = dq[IY][IA];  dAy *= HALF_F;

  // Cell centered TVD slopes in Z direction
  real_t& drz = dq[IZ][ID];  drz *= HALF_F;
  real_t& dpz = dq[IZ][IP];  dpz *= HALF_F;
  real_t& duz = dq[IZ][IU];  duz *= HALF_F;
  real_t& dvz = dq[IZ][IV];  dvz *= HALF_F;
  real_t& dwz = dq[IZ][IW];  dwz *= HALF_F;
  real_t& dAz = dq[IZ][IA];  dAz *= HALF_F;
  real_t& dBz = dq[IZ][IB];  dBz *= HALF_F;


  // Face centered TVD slopes in transverse direction
  real_t dALy = HALF_F * dbf[0];
  real_t dALz = HALF_F * dbf[1];
  real_t dBLx = HALF_F * dbf[2];
  real_t dBLz = HALF_F * dbf[3];
  real_t dCLx = HALF_F * dbf[4];
  real_t dCLy = HALF_F * dbf[5];

  real_t dARy = HALF_F * dbf[6];
  real_t dARz = HALF_F * dbf[7];
  real_t dBRx = HALF_F * dbf[8];
  real_t dBRz = HALF_F * dbf[9];
  real_t dCRx = HALF_F * dbf[10];
  real_t dCRy = HALF_F * dbf[11];

  // Cell centered slopes in normal direction
  real_t dAx = HALF_F * (AR - AL);
  real_t dBy = HALF_F * (BR - BL);
  real_t dCz = HALF_F * (CR - CL);

  // Source terms (including transverse derivatives)
  real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
  real_t sAL0, sAR0, sBL0, sBR0, sCL0, sCR0;

  if (true /*cartesian*/) {

    sr0 = (-u*drx-dux*r)              *dtdx + (-v*dry-dvy*r)              *dtdy + (-w*drz-dwz*r)              *dtdz;
    su0 = (-u*dux-(dpx+B*dBx+C*dCx)/r)*dtdx + (-v*duy+B*dAy/r)            *dtdy + (-w*duz+C*dAz/r)            *dtdz; 
    sv0 = (-u*dvx+A*dBx/r)            *dtdx + (-v*dvy-(dpy+A*dAy+C*dCy)/r)*dtdy + (-w*dvz+C*dBz/r)            *dtdz;
    sw0 = (-u*dwx+A*dCx/r)            *dtdx + (-v*dwy+B*dCy/r)            *dtdy + (-w*dwz-(dpz+A*dAz+B*dBz)/r)*dtdz; 
    sp0 = (-u*dpx-dux*gamma*p)        *dtdx + (-v*dpy-dvy*gamma*p)        *dtdy + (-w*dpz-dwz*gamma*p)        *dtdz;
    sA0 =                                     (u*dBy+B*duy-v*dAy-A*dvy)   *dtdy + (u*dCz+C*duz-w*dAz-A*dwz)   *dtdz;
    sB0 = (v*dAx+A*dvx-u*dBx-B*dux)   *dtdx +                                     (v*dCz+C*dvz-w*dBz-B*dwz)   *dtdz; 
    sC0 = (w*dAx+A*dwx-u*dCx-C*dux)   *dtdx + (w*dBy+B*dwy-v*dCy-C*dvy)   *dtdy;
    if (Omega0>0) {
      real_t shear = -1.5 * Omega0 *xPos;
      sr0 = sr0 -  shear*dry*dtdy;
      su0 = su0 -  shear*duy*dtdy;
      sv0 = sv0 -  shear*dvy*dtdy;
      sw0 = sw0 -  shear*dwy*dtdy;
      sp0 = sp0 -  shear*dpy*dtdy;
      sA0 = sA0 -  shear*dAy*dtdy;
      sB0 = sB0 + (shear*dAx - 1.5 * Omega0 * A *dx)*dtdx + shear*dBz*dtdz;
      sC0 = sC0 -  shear*dCy*dtdy;
    }
	
    // Face-centered B-field
    sAL0 = +(GLR-GLL)*dtdy*HALF_F -(FLR-FLL)*dtdz*HALF_F;
    sAR0 = +(GRR-GRL)*dtdy*HALF_F -(FRR-FRL)*dtdz*HALF_F;
    sBL0 = -(GRL-GLL)*dtdx*HALF_F +(ELR-ELL)*dtdz*HALF_F;
    sBR0 = -(GRR-GLR)*dtdx*HALF_F +(ERR-ERL)*dtdz*HALF_F;
    sCL0 = +(FRL-FLL)*dtdx*HALF_F -(ERL-ELL)*dtdy*HALF_F;
    sCR0 = +(FRR-FLR)*dtdx*HALF_F -(ERR-ELR)*dtdy*HALF_F;

  } // end cartesian

  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  w = w + sw0;
  p = p + sp0;
  A = A + sA0;
  B = B + sB0;
  C = C + sC0;
  
  AL = AL + sAL0;
  AR = AR + sAR0;
  BL = BL + sBL0;
  BR = BR + sBR0;
  CL = CL + sCL0;
  CR = CR + sCR0;

  // Face averaged right state at left interface
  qp[0][ID] = r - drx;
  qp[0][IU] = u - dux;
  qp[0][IV] = v - dvx;
  qp[0][IW] = w - dwx;
  qp[0][IP] = p - dpx;
  qp[0][IA] = AL;
  qp[0][IB] = B - dBx;
  qp[0][IC] = C - dCx;
  qp[0][ID] = FMAX(smallR,  qp[0][ID]);
  qp[0][IP] = FMAX(smallp /** qp[0][ID]*/, qp[0][IP]);
  
  // Face averaged left state at right interface
  qm[0][ID] = r + drx;
  qm[0][IU] = u + dux;
  qm[0][IV] = v + dvx;
  qm[0][IW] = w + dwx;
  qm[0][IP] = p + dpx;
  qm[0][IA] = AR;
  qm[0][IB] = B + dBx;
  qm[0][IC] = C + dCx;
  qm[0][ID] = FMAX(smallR,  qm[0][ID]);
  qm[0][IP] = FMAX(smallp /** qm[0][ID]*/, qm[0][IP]);

  // Face averaged top state at bottom interface
  qp[1][ID] = r - dry;
  qp[1][IU] = u - duy;
  qp[1][IV] = v - dvy;
  qp[1][IW] = w - dwy;
  qp[1][IP] = p - dpy;
  qp[1][IA] = A - dAy;
  qp[1][IB] = BL;
  qp[1][IC] = C - dCy;
  qp[1][ID] = FMAX(smallR,  qp[1][ID]);
  qp[1][IP] = FMAX(smallp /** qp[1][ID]*/, qp[1][IP]);
  
  // Face averaged bottom state at top interface
  qm[1][ID] = r + dry;
  qm[1][IU] = u + duy;
  qm[1][IV] = v + dvy;
  qm[1][IW] = w + dwy;
  qm[1][IP] = p + dpy;
  qm[1][IA] = A + dAy;
  qm[1][IB] = BR;
  qm[1][IC] = C + dCy;
  qm[1][ID] = FMAX(smallR,  qm[1][ID]);
  qm[1][IP] = FMAX(smallp /** qm[1][ID]*/, qm[1][IP]);
  
  // Face averaged front state at back interface
  qp[2][ID] = r - drz;
  qp[2][IU] = u - duz;
  qp[2][IV] = v - dvz;
  qp[2][IW] = w - dwz;
  qp[2][IP] = p - dpz;
  qp[2][IA] = A - dAz;
  qp[2][IB] = B - dBz;
  qp[2][IC] = CL;
  qp[2][ID] = FMAX(smallR,  qp[2][ID]);
  qp[2][IP] = FMAX(smallp /** qp[2][ID]*/, qp[2][IP]);
  
  // Face averaged back state at front interface
  qm[2][ID] = r + drz;
  qm[2][IU] = u + duz;
  qm[2][IV] = v + dvz;
  qm[2][IW] = w + dwz;
  qm[2][IP] = p + dpz;
  qm[2][IA] = A + dAz;
  qm[2][IB] = B + dBz;
  qm[2][IC] = CR;
  qm[2][ID] = FMAX(smallR,  qm[2][ID]);
  qm[2][IP] = FMAX(smallp /** qm[2][ID]*/, qm[2][IP]);

  // X-edge averaged right-top corner state (RT->LL)
  qRT_X[ID] = r + (+dry+drz);
  qRT_X[IU] = u + (+duy+duz);
  qRT_X[IV] = v + (+dvy+dvz);
  qRT_X[IW] = w + (+dwy+dwz);
  qRT_X[IP] = p + (+dpy+dpz);
  qRT_X[IA] = A + (+dAy+dAz);
  qRT_X[IB] = BR+ (   +dBRz);
  qRT_X[IC] = CR+ (+dCRy   );
  qRT_X[ID] = FMAX(smallR,  qRT_X[ID]);
  qRT_X[IP] = FMAX(smallp /** qRT_X[ID]*/, qRT_X[IP]);
  
  // X-edge averaged right-bottom corner state (RB->LR)
  qRB_X[ID] = r + (+dry-drz);
  qRB_X[IU] = u + (+duy-duz);
  qRB_X[IV] = v + (+dvy-dvz);
  qRB_X[IW] = w + (+dwy-dwz);
  qRB_X[IP] = p + (+dpy-dpz);
  qRB_X[IA] = A + (+dAy-dAz);
  qRB_X[IB] = BR+ (   -dBRz);
  qRB_X[IC] = CL+ (+dCLy   );
  qRB_X[ID] = FMAX(smallR,  qRB_X[ID]);
  qRB_X[IP] = FMAX(smallp /** qRB_X[ID]*/, qRB_X[IP]);
  
  // X-edge averaged left-top corner state (LT->RL)
  qLT_X[ID] = r + (-dry+drz);
  qLT_X[IU] = u + (-duy+duz);
  qLT_X[IV] = v + (-dvy+dvz);
  qLT_X[IW] = w + (-dwy+dwz);
  qLT_X[IP] = p + (-dpy+dpz);
  qLT_X[IA] = A + (-dAy+dAz);
  qLT_X[IB] = BL+ (   +dBLz);
  qLT_X[IC] = CR+ (-dCRy   );
  qLT_X[ID] = FMAX(smallR,  qLT_X[ID]);
  qLT_X[IP] = FMAX(smallp /** qLT_X[ID]*/, qLT_X[IP]);
  
  // X-edge averaged left-bottom corner state (LB->RR)
  qLB_X[ID] = r + (-dry-drz);
  qLB_X[IU] = u + (-duy-duz);
  qLB_X[IV] = v + (-dvy-dvz);
  qLB_X[IW] = w + (-dwy-dwz);
  qLB_X[IP] = p + (-dpy-dpz);
  qLB_X[IA] = A + (-dAy-dAz);
  qLB_X[IB] = BL+ (   -dBLz);
  qLB_X[IC] = CL+ (-dCLy   );
  qLB_X[ID] = FMAX(smallR,  qLB_X[ID]);
  qLB_X[IP] = FMAX(smallp /** qLB_X[ID]*/, qLB_X[IP]);
  
  // Y-edge averaged right-top corner state (RT->LL)
  qRT_Y[ID] = r + (+drx+drz);
  qRT_Y[IU] = u + (+dux+duz);
  qRT_Y[IV] = v + (+dvx+dvz);
  qRT_Y[IW] = w + (+dwx+dwz);
  qRT_Y[IP] = p + (+dpx+dpz);
  qRT_Y[IA] = AR+ (   +dARz);
  qRT_Y[IB] = B + (+dBx+dBz);
  qRT_Y[IC] = CR+ (+dCRx   );
  qRT_Y[ID] = FMAX(smallR,  qRT_Y[ID]);
  qRT_Y[IP] = FMAX(smallp /** qRT_Y[ID]*/, qRT_Y[IP]);
  
  // Y-edge averaged right-bottom corner state (RB->LR)
  qRB_Y[ID] = r + (+drx-drz);
  qRB_Y[IU] = u + (+dux-duz);
  qRB_Y[IV] = v + (+dvx-dvz);
  qRB_Y[IW] = w + (+dwx-dwz);
  qRB_Y[IP] = p + (+dpx-dpz);
  qRB_Y[IA] = AR+ (   -dARz);
  qRB_Y[IB] = B + (+dBx-dBz);
  qRB_Y[IC] = CL+ (+dCLx   );
  qRB_Y[ID] = FMAX(smallR,  qRB_Y[ID]);
  qRB_Y[IP] = FMAX(smallp /** qRB_Y[ID]*/, qRB_Y[IP]);
  
  // Y-edge averaged left-top corner state (LT->RL)
  qLT_Y[ID] = r + (-drx+drz);
  qLT_Y[IU] = u + (-dux+duz);
  qLT_Y[IV] = v + (-dvx+dvz);
  qLT_Y[IW] = w + (-dwx+dwz);
  qLT_Y[IP] = p + (-dpx+dpz);
  qLT_Y[IA] = AL+ (   +dALz);
  qLT_Y[IB] = B + (-dBx+dBz);
  qLT_Y[IC] = CR+ (-dCRx   );
  qLT_Y[ID] = FMAX(smallR,  qLT_Y[ID]);
  qLT_Y[IP] = FMAX(smallp /** qLT_Y[ID]*/, qLT_Y[IP]);
  
  // Y-edge averaged left-bottom corner state (LB->RR)
  qLB_Y[ID] = r + (-drx-drz);
  qLB_Y[IU] = u + (-dux-duz);
  qLB_Y[IV] = v + (-dvx-dvz);
  qLB_Y[IW] = w + (-dwx-dwz);
  qLB_Y[IP] = p + (-dpx-dpz);
  qLB_Y[IA] = AL+ (   -dALz);
  qLB_Y[IB] = B + (-dBx-dBz);
  qLB_Y[IC] = CL+ (-dCLx   );
  qLB_Y[ID] = FMAX(smallR,  qLB_Y[ID]);
  qLB_Y[IP] = FMAX(smallp /** qLB_Y[ID]*/, qLB_Y[IP]);
  
  // Z-edge averaged right-top corner state (RT->LL)
  qRT_Z[ID] = r + (+drx+dry);
  qRT_Z[IU] = u + (+dux+duy);
  qRT_Z[IV] = v + (+dvx+dvy);
  qRT_Z[IW] = w + (+dwx+dwy);
  qRT_Z[IP] = p + (+dpx+dpy);
  qRT_Z[IA] = AR+ (   +dARy);
  qRT_Z[IB] = BR+ (+dBRx   );
  qRT_Z[IC] = C + (+dCx+dCy);
  qRT_Z[ID] = FMAX(smallR,  qRT_Z[ID]);
  qRT_Z[IP] = FMAX(smallp /** qRT_Z[ID]*/, qRT_Z[IP]);
  
  // Z-edge averaged right-bottom corner state (RB->LR)
  qRB_Z[ID] = r + (+drx-dry);
  qRB_Z[IU] = u + (+dux-duy);
  qRB_Z[IV] = v + (+dvx-dvy);
  qRB_Z[IW] = w + (+dwx-dwy);
  qRB_Z[IP] = p + (+dpx-dpy);
  qRB_Z[IA] = AR+ (   -dARy);
  qRB_Z[IB] = BL+ (+dBLx   );
  qRB_Z[IC] = C + (+dCx-dCy);
  qRB_Z[ID] = FMAX(smallR,  qRB_Z[ID]);
  qRB_Z[IP] = FMAX(smallp /** qRB_Z[ID]*/, qRB_Z[IP]);
  
  // Z-edge averaged left-top corner state (LT->RL)
  qLT_Z[ID] = r + (-drx+dry);
  qLT_Z[IU] = u + (-dux+duy);
  qLT_Z[IV] = v + (-dvx+dvy);
  qLT_Z[IW] = w + (-dwx+dwy);
  qLT_Z[IP] = p + (-dpx+dpy);
  qLT_Z[IA] = AL+ (   +dALy);
  qLT_Z[IB] = BR+ (-dBRx   );
  qLT_Z[IC] = C + (-dCx+dCy);
  qLT_Z[ID] = FMAX(smallR,  qLT_Z[ID]);
  qLT_Z[IP] = FMAX(smallp /** qLT_Z[ID]*/, qLT_Z[IP]);
  
  // Z-edge averaged left-bottom corner state (LB->RR)
  qLB_Z[ID] = r + (-drx-dry);
  qLB_Z[IU] = u + (-dux-duy);
  qLB_Z[IV] = v + (-dvx-dvy);
  qLB_Z[IW] = w + (-dwx-dwy);
  qLB_Z[IP] = p + (-dpx-dpy);
  qLB_Z[IA] = AL+ (   -dALy);
  qLB_Z[IB] = BL+ (-dBLx   );
  qLB_Z[IC] = C + (-dCx-dCy);
  qLB_Z[ID] = FMAX(smallR,  qLB_Z[ID]);
  qLB_Z[IP] = FMAX(smallp /** qLB_Z[ID]*/, qLB_Z[IP]);

} // trace_unsplit_mhd_3d_simpler

#endif // TRACE_MHD_H_
