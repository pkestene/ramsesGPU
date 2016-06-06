#include "MHDRunGodunovMpi.h"

#include "constoprim.h"
#include "riemann_mhd.h"
#include "trace_mhd.h"

namespace hydroSimu {

  // =======================================================
  // =======================================================
  void MHDRunGodunovMpi::godunov_unsplit_cpu_v2(HostArray<real_t>& h_UOld, 
						HostArray<real_t>& h_UNew, 
						real_t dt, int nStep)
  {

    // this version also stores electric field and transverse magnetic field
    // it is only 5 percent faster than implementation 1

    (void) nStep;
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;
    (void) dtdz;
    
    // primitive variable domain array
    real_t *Q = h_Q.data();

    // section / domain size
    int arraySize = h_Q.section();

    if (dimType == TWO_D) {

      // fisrt compute electric field Ez
      for (int j=1; j<jsize; j++) {
	for (int i=1; i<isize; i++) {
	  real_t u = ONE_FOURTH_F * ( h_Q(i-1,j-1,IU) +
				      h_Q(i-1,j  ,IU) +
				      h_Q(i  ,j-1,IU) +
				      h_Q(i  ,j  ,IU) );
	  real_t v = ONE_FOURTH_F * ( h_Q(i-1,j-1,IV) +
				      h_Q(i-1,j  ,IV) +
				      h_Q(i  ,j-1,IV) +
				      h_Q(i  ,j  ,IV) );
	  real_t A = HALF_F  * ( h_UOld(i  ,j-1, IA) +
				 h_UOld(i  ,j  , IA) );
	  real_t B = HALF_F  * ( h_UOld(i-1,j  , IB) +
				 h_UOld(i  ,j  , IB) );
	    
	  h_elec(i,j,0) = u*B-v*A;

	} // end for i
      } // end for j

	// second : compute magnetic slopes
      for (int j=1; j<jsize-1; j++) {
	for (int i=1; i<isize-1; i++) {
	    
	  // face-centered magnetic field in the neighborhood
	  real_t bfNeighbors[6]; 
	    
	  // magnetic slopes
	  real_t dbf[2][3];
	  real_t (&dbfX)[3] = dbf[IX];
	  real_t (&dbfY)[3] = dbf[IY];
	    
	  bfNeighbors[0] =  h_UOld(i  ,j  , IA);
	  bfNeighbors[1] =  h_UOld(i  ,j+1, IA);
	  bfNeighbors[2] =  h_UOld(i  ,j-1, IA);
	  bfNeighbors[3] =  h_UOld(i  ,j  , IB);
	  bfNeighbors[4] =  h_UOld(i+1,j  , IB);
	  bfNeighbors[5] =  h_UOld(i-1,j  , IB);
	    
	  // compute magnetic slopes
	  slope_unsplit_mhd_2d(bfNeighbors, dbf);
	    
	  // store magnetic slopes
	  h_dA(i,j,0) = HALF_F * dbfY[IX];
	  h_dB(i,j,0) = HALF_F * dbfX[IY];
	    
	} // end for i
      } // end for j	

	// third : compute qm, qp and qEdge (from trace)
      for (int j=1; j<jsize-2; j++) {
	for (int i=1; i<isize-2; i++) {
	    
	  real_t qNb[3][3][NVAR_MHD];
	  real_t qm[2][NVAR_MHD];
	  real_t qp[2][NVAR_MHD];
	  real_t qEdge[4][NVAR_MHD];

	  // alias for q on cell edge (as defined in DUMSES trace2d routine)
	  real_t (&qRT)[NVAR_MHD] = qEdge[0];
	  real_t (&qRB)[NVAR_MHD] = qEdge[1];
	  real_t (&qLT)[NVAR_MHD] = qEdge[2];
	  real_t (&qLB)[NVAR_MHD] = qEdge[3];


	  // prepare qNb : q state in the 3-by-3 neighborhood
	  // note that current cell (i,j) is in qNb[1][1]
	  // also note that the effective stencil is 4-by-4 since
	  // computation of primitive variable (q) requires mag
	  // field on the right (see computePrimitives_MHD_2D)
	  for (int di=0; di<3; di++)
	    for (int dj=0; dj<3; dj++) {
	      for (int iVar=0; iVar < NVAR_MHD; iVar++) {
		int indexLoc = (i+di-1)+(j+dj-1)*isize; // centered index
		getPrimitiveVector(Q, arraySize, indexLoc, qNb[di][dj]);
	      }
	    }
	    
	  // compute hydro slopes
	  real_t dq[2][NVAR_MHD];
	  slope_unsplit_hydro_2d(qNb, dq);

	  // compute qm, qp, qEdge's
	  //trace_unsplit_mhd_2d(qNb, bfNb, c, dtdx, dtdy, xPos, qm, qp, qEdge);
	  //int iG = i + nx*myMpiPos[0];
	  //real_t xPos = _gParams.xMin + dx/2 + (iG-ghostWidth)*dx;

	  {
	    
	    real_t &smallR = ::gParams.smallr;
	    real_t &smallp = ::gParams.smallp;
	    //real_t &smallP = ::gParams.smallpp;
	    real_t &gamma  = ::gParams.gamma0;

	    // Cell centered values
	    real_t r = qNb[1][1][ID];
	    real_t p = qNb[1][1][IP];
	    real_t u = qNb[1][1][IU];
	    real_t v = qNb[1][1][IV];
	    real_t w = qNb[1][1][IW];            
	    real_t A = qNb[1][1][IA];
	    real_t B = qNb[1][1][IB];
	    real_t C = qNb[1][1][IC];            
	      
	    // Electric field
	    real_t ELL = h_elec(i  ,j  ,0);
	    real_t ELR = h_elec(i  ,j+1,0);
	    real_t ERL = h_elec(i+1,j  ,0);
	    real_t ERR = h_elec(i+1,j+1,0);

	    // Face centered magnetic field components
	    real_t AL =  h_UOld(i  ,j  ,IA);
	    real_t AR =  h_UOld(i+1,j  ,IA);
	    real_t BL =  h_UOld(i  ,j  ,IB);
	    real_t BR =  h_UOld(i  ,j+1,IB);

	    // Cell centered slopes in normal direction
	    real_t dAx = HALF_F * (AR - AL);
	    real_t dBy = HALF_F * (BR - BL);

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

	    // get transverse magnetic slopes previously computed and stored
	    real_t dALy = h_dA(i  ,j  ,0);
	    real_t dBLx = h_dB(i  ,j  ,0);
	    real_t dARy = h_dA(i+1,j  ,0);
	    real_t dBRx = h_dB(i  ,j+1,0);
	      
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
	    qp[0][ID] = FMAX(smallR,  qp[0][ID]);
	    qp[0][IP] = FMAX(smallp * qp[0][ID], qp[0][IP]);
	      
	    // Left state at right interface
	    qm[0][ID] = r + drx;
	    qm[0][IU] = u + dux;
	    qm[0][IV] = v + dvx;
	    qm[0][IW] = w + dwx;
	    qm[0][IP] = p + dpx;
	    qm[0][IA] = AR;
	    qm[0][IB] = B + dBx;
	    qm[0][IC] = C + dCx;
	    qm[0][ID] = FMAX(smallR,  qm[0][ID]);
	    qm[0][IP] = FMAX(smallp * qm[0][ID], qm[0][IP]);
	      
	    // Top state at bottom interface
	    qp[1][ID] = r - dry;
	    qp[1][IU] = u - duy;
	    qp[1][IV] = v - dvy;
	    qp[1][IW] = w - dwy;
	    qp[1][IP] = p - dpy;
	    qp[1][IA] = A - dAy;
	    qp[1][IB] = BL;
	    qp[1][IC] = C - dCy;
	    qp[1][ID] = FMAX(smallR,  qp[1][ID]);
	    qp[1][IP] = FMAX(smallp * qp[1][ID], qp[1][IP]);
	      
	    // Bottom state at top interface
	    qm[1][ID] = r + dry;
	    qm[1][IU] = u + duy;
	    qm[1][IV] = v + dvy;
	    qm[1][IW] = w + dwy;
	    qm[1][IP] = p + dpy;
	    qm[1][IA] = A + dAy;
	    qm[1][IB] = BR;
	    qm[1][IC] = C + dCy;
	    qm[1][ID] = FMAX(smallR,  qm[1][ID]);
	    qm[1][IP] = FMAX(smallp * qm[1][ID], qm[1][IP]);
	      
	      
	    // Right-top state (RT->LL)
	    qRT[ID] = r + (+drx+dry);
	    qRT[IU] = u + (+dux+duy);
	    qRT[IV] = v + (+dvx+dvy);
	    qRT[IW] = w + (+dwx+dwy);
	    qRT[IP] = p + (+dpx+dpy);
	    qRT[IA] = AR+ (   +dARy);
	    qRT[IB] = BR+ (+dBRx   );
	    qRT[IC] = C + (+dCx+dCy);
	    qRT[ID] = FMAX(smallR,  qRT[ID]);
	    qRT[IP] = FMAX(smallp * qRT[ID], qRT[IP]);
	      
	    // Right-Bottom state (RB->LR)
	    qRB[ID] = r + (+drx-dry);
	    qRB[IU] = u + (+dux-duy);
	    qRB[IV] = v + (+dvx-dvy);
	    qRB[IW] = w + (+dwx-dwy);
	    qRB[IP] = p + (+dpx-dpy);
	    qRB[IA] = AR+ (   -dARy);
	    qRB[IB] = BL+ (+dBLx   );
	    qRB[IC] = C + (+dCx-dCy);
	    qRB[ID] = FMAX(smallR,  qRB[ID]);
	    qRB[IP] = FMAX(smallp * qRB[ID], qRB[IP]);
	      
	    // Left-Bottom state (LB->RR)
	    qLB[ID] = r + (-drx-dry);
	    qLB[IU] = u + (-dux-duy);
	    qLB[IV] = v + (-dvx-dvy);
	    qLB[IW] = w + (-dwx-dwy);
	    qLB[IP] = p + (-dpx-dpy);
	    qLB[IA] = AL+ (   -dALy);
	    qLB[IB] = BL+ (-dBLx   );
	    qLB[IC] = C + (-dCx-dCy);
	    qLB[ID] = FMAX(smallR,  qLB[ID]);
	    qLB[IP] = FMAX(smallp * qLB[ID], qLB[IP]);
	      
	    // Left-Top state (LT->RL)
	    qLT[ID] = r + (-drx+dry);
	    qLT[IU] = u + (-dux+duy);
	    qLT[IV] = v + (-dvx+dvy);
	    qLT[IW] = w + (-dwx+dwy);
	    qLT[IP] = p + (-dpx+dpy);
	    qLT[IA] = AL+ (   +dALy);
	    qLT[IB] = BR+ (-dBRx   );
	    qLT[IC] = C + (-dCx+dCy);
	    qLT[ID] = FMAX(smallR,  qLT[ID]);
	    qLT[IP] = FMAX(smallp * qLT[ID], qLT[IP]);
	      
	      
	  }	    
	    
	  // store qm, qp, qEdge : only what is really needed
	  for (int ivar=0; ivar<NVAR_MHD; ivar++) {
	    h_qm_x(i,j,ivar) = qm[0][ivar];
	    h_qp_x(i,j,ivar) = qp[0][ivar];
	    h_qm_y(i,j,ivar) = qm[1][ivar];
	    h_qp_y(i,j,ivar) = qp[1][ivar];
	      
	    h_qEdge_RT(i,j,ivar) = qEdge[IRT][ivar]; 
	    h_qEdge_RB(i,j,ivar) = qEdge[IRB][ivar]; 
	    h_qEdge_LT(i,j,ivar) = qEdge[ILT][ivar]; 
	    h_qEdge_LB(i,j,ivar) = qEdge[ILB][ivar]; 
	  } // end for ivar

	} // end for i
      } // end for j

	// now update hydro variables
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {

	  real_riemann_t qleft[NVAR_MHD];
	  real_riemann_t qright[NVAR_MHD];
	  real_riemann_t flux_x[NVAR_MHD];
	  real_riemann_t flux_y[NVAR_MHD];
	  //int iG = i + nx*myMpiPos[0];
	  //real_t xPos = ::gParams.xMin + dx/2 + (iG-ghostWidth)*dx;

 	  for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	    flux_x[iVar] = 0.0;
	    flux_y[iVar] = 0.0;
	  }

	
	  /*
	   * Solve Riemann problem at X-interfaces and compute
	   * X-fluxes
	   *
	   * Note that continuity of normal component of magnetic
	   * field is ensured inside riemann_mhd routine.
	   */

	  // even in 2D we need to fill IW index (to respect
	  // riemann_mhd interface)
	  qleft[ID]   = h_qm_x(i-1,j,ID);
	  qleft[IP]   = h_qm_x(i-1,j,IP);
	  qleft[IU]   = h_qm_x(i-1,j,IU);
	  qleft[IV]   = h_qm_x(i-1,j,IV);
	  qleft[IW]   = h_qm_x(i-1,j,IW);
	  qleft[IA]   = h_qm_x(i-1,j,IA);
	  qleft[IB]   = h_qm_x(i-1,j,IB);
	  qleft[IC]   = h_qm_x(i-1,j,IC);

	  qright[ID]  = h_qp_x(i  ,j,ID);
	  qright[IP]  = h_qp_x(i  ,j,IP);
	  qright[IU]  = h_qp_x(i  ,j,IU);
	  qright[IV]  = h_qp_x(i  ,j,IV);
	  qright[IW]  = h_qp_x(i  ,j,IW);
	  qright[IA]  = h_qp_x(i  ,j,IA);
	  qright[IB]  = h_qp_x(i  ,j,IB);
	  qright[IC]  = h_qp_x(i  ,j,IC);

	  // compute hydro flux_x
	  riemann_mhd(qleft,qright,flux_x);
	    
	    
	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	  qleft[ID]   = h_qm_y(i,j-1,ID);
	  qleft[IP]   = h_qm_y(i,j-1,IP);
	  qleft[IU]   = h_qm_y(i,j-1,IV); // watchout IU, IV permutation
	  qleft[IV]   = h_qm_y(i,j-1,IU); // watchout IU, IV permutation
	  qleft[IW]   = h_qm_y(i,j-1,IW);
	  qleft[IA]   = h_qm_y(i,j-1,IB); // watchout IA, IB permutation
	  qleft[IB]   = h_qm_y(i,j-1,IA); // watchout IA, IB permutation
	  qleft[IC]   = h_qm_y(i,j-1,IC);

	  qright[ID]  = h_qp_y(i,j  ,ID);
	  qright[IP]  = h_qp_y(i,j  ,IP);
	  qright[IU]  = h_qp_y(i,j  ,IV); // watchout IU, IV permutation
	  qright[IV]  = h_qp_y(i,j  ,IU); // watchout IU, IV permutation
	  qright[IW]  = h_qp_y(i,j  ,IW);
	  qright[IA]  = h_qp_y(i,j  ,IB); // watchout IA, IB permutation
	  qright[IB]  = h_qp_y(i,j  ,IA); // watchout IA, IB permutation
	  qright[IC]  = h_qp_y(i,j  ,IC);
	    
	  // compute hydro flux_y
	  riemann_mhd(qleft,qright,flux_y);

	    
	  /*
	   * update mhd array with hydro fluxes
	   */
	  h_UNew(i-1,j  ,ID) -= flux_x[ID]*dtdx;
	  h_UNew(i-1,j  ,IP) -= flux_x[IP]*dtdx;
	  h_UNew(i-1,j  ,IU) -= flux_x[IU]*dtdx;
	  h_UNew(i-1,j  ,IV) -= flux_x[IV]*dtdx;
	  h_UNew(i-1,j  ,IW) -= flux_x[IW]*dtdx;
	  h_UNew(i-1,j  ,IC) -= flux_x[IC]*dtdx;
	    
	  h_UNew(i  ,j  ,ID) += flux_x[ID]*dtdx;
	  h_UNew(i  ,j  ,IP) += flux_x[IP]*dtdx;
	  h_UNew(i  ,j  ,IU) += flux_x[IU]*dtdx;
	  h_UNew(i  ,j  ,IV) += flux_x[IV]*dtdx;
	  h_UNew(i  ,j  ,IW) += flux_x[IW]*dtdx;
	  h_UNew(i  ,j  ,IC) += flux_x[IC]*dtdx;
	    
	  h_UNew(i  ,j-1,ID) -= flux_y[ID]*dtdy;
	  h_UNew(i  ,j-1,IP) -= flux_y[IP]*dtdy;
	  h_UNew(i  ,j-1,IU) -= flux_y[IV]*dtdy; // watchout IU and IV swapped
	  h_UNew(i  ,j-1,IV) -= flux_y[IU]*dtdy; // watchout IU and IV swapped
	  h_UNew(i  ,j-1,IW) -= flux_y[IW]*dtdy;
	  h_UNew(i  ,j-1,IC) -= flux_y[IC]*dtdy;
	    
	  h_UNew(i  ,j  ,ID) += flux_y[ID]*dtdy;
	  h_UNew(i  ,j  ,IP) += flux_y[IP]*dtdy;
	  h_UNew(i  ,j  ,IU) += flux_y[IV]*dtdy; // watchout IU and IV swapped
	  h_UNew(i  ,j  ,IV) += flux_y[IU]*dtdy; // watchout IU and IV swapped
	  h_UNew(i  ,j  ,IW) += flux_y[IW]*dtdy;
	  h_UNew(i  ,j  ,IC) += flux_y[IC]*dtdy;
	    
	  // now compute EMF's and update magnetic field variables
	  // see DUMSES routine named cmp_mag_flx (TAKE CARE of index
	  // shift appearing in calling arguments)
	    
	  // in 2D, we only need to compute emfZ
	  real_t qEdge_emfZ[4][NVAR_MHD];
	
	  // preparation for calling compute_emf (equivalent to cmp_mag_flx
	  // in DUMSES)
	  // in the following, the 2 first indexes in qEdge_emf array play
	  // the same offset role as in the calling argument of cmp_mag_flx 
	  // in DUMSES (if you see what I mean ?!)
	  for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	    qEdge_emfZ[IRT][iVar] = h_qEdge_RT(i-1,j-1,iVar); 
	    qEdge_emfZ[IRB][iVar] = h_qEdge_RB(i-1,j  ,iVar); 
	    qEdge_emfZ[ILT][iVar] = h_qEdge_LT(i  ,j-1,iVar); 
	    qEdge_emfZ[ILB][iVar] = h_qEdge_LB(i  ,j  ,iVar); 
	  }
	    
	  // actually compute emfZ
	  real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ);
	    
	  // now update h_UNew with emfZ
	  // (Constrained transport for face-centered B-field)
	  h_UNew(i  ,j  ,IA) -= emfZ*dtdy;
	  h_UNew(i  ,j-1,IA) += emfZ*dtdy;
	    
	  h_UNew(i  ,j  ,IB) += emfZ*dtdx;  
	  h_UNew(i-1,j  ,IB) -= emfZ*dtdx;
	    
	} // end for i
      } // end for j
    
    } else { // THREE_D

      std::cout << "CPU - 3D - implementation version 2 - Not implemented !!!" << std::endl;
	
    } // end THREE_D
    
  } // MHDRunGodunovMpi::godunov_unsplit_cpu_v2

} // namespace hydroSimu
