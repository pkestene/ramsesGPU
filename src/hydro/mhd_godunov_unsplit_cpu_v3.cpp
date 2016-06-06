#include "MHDRunGodunov.h"

#include "constoprim.h"
#include "riemann_mhd.h"
#include "trace_mhd.h"

namespace hydroSimu {

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_cpu_v3(HostArray<real_t>& h_UOld, 
  					     HostArray<real_t>& h_UNew, 
  					     real_t dt, int nStep)
  {

    // this version also stores electric field and transverse magnetic field
    // it is only 5 percent faster than implementation 1

    (void) nStep;
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;
        
    if (dimType == TWO_D) {
      
      std::cout << "implementation version 3/4 are not available for 2D simulation. Designed only for 3D problems." << std::endl;
      
    } else { // THREE_D - implementation version 3 / 4
      
      TIMER_START(timerElecField);
      // compute electric field components
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=1; k<ksize-1; k++) {
	for (int j=1; j<jsize-1; j++) {
	  for (int i=1; i<isize-1; i++) {
	      
	    real_t u, v, w, A, B, C;

	    // compute Ex
	    v = ONE_FOURTH_F * ( h_Q   (i  ,j-1,k-1,IV) +
				 h_Q   (i  ,j-1,k  ,IV) +
				 h_Q   (i  ,j  ,k-1,IV) +
				 h_Q   (i  ,j  ,k  ,IV) );
	      
	    w = ONE_FOURTH_F * ( h_Q   (i  ,j-1,k-1,IW) +
				 h_Q   (i  ,j-1,k  ,IW) +
				 h_Q   (i  ,j  ,k-1,IW) +
				 h_Q   (i  ,j  ,k  ,IW) );
	      
	    B = HALF_F  * ( h_UOld(i  ,j  ,k-1,IB) +
			    h_UOld(i  ,j  ,k  ,IB) );
	      
	    C = HALF_F  * ( h_UOld(i  ,j-1,k  ,IC) +
			    h_UOld(i  ,j  ,k  ,IC) );
	      
	    h_elec(i,j,k,IX) = v*C-w*B;

	    // compute Ey
	    u = ONE_FOURTH_F * ( h_Q   (i-1,j  ,k-1,IU) +
				 h_Q   (i-1,j  ,k  ,IU) +
				 h_Q   (i  ,j  ,k-1,IU) +
				 h_Q   (i  ,j  ,k  ,IU) );

	    w = ONE_FOURTH_F * ( h_Q   (i-1,j  ,k-1,IW) +
				 h_Q   (i-1,j  ,k  ,IW) +
				 h_Q   (i  ,j  ,k-1,IW) +
				 h_Q   (i  ,j  ,k  ,IW) );

	    A = HALF_F  * ( h_UOld(i  ,j  ,k-1,IA) +
			    h_UOld(i  ,j  ,k  ,IA) );
	      
	    C = HALF_F  * ( h_UOld(i-1,j  ,k  ,IC) +
			    h_UOld(i  ,j  ,k  ,IC) );
	      
	    h_elec(i,j,k,IY) = w*A-u*C;

	    // compute Ez
	    u = ONE_FOURTH_F * ( h_Q   (i-1,j-1,k  ,IU) +
				 h_Q   (i-1,j  ,k  ,IU) +
				 h_Q   (i  ,j-1,k  ,IU) +
				 h_Q   (i  ,j  ,k  ,IU) );

	    v = ONE_FOURTH_F * ( h_Q   (i-1,j-1,k  ,IV) +
				 h_Q   (i-1,j  ,k  ,IV) +
				 h_Q   (i  ,j-1,k  ,IV) +
				 h_Q   (i  ,j  ,k  ,IV) );
	      
	    A = HALF_F  * ( h_UOld(i  ,j-1,k  ,IA) +
			    h_UOld(i  ,j  ,k  ,IA) );
	      
	    B = HALF_F  * ( h_UOld(i-1,j  ,k  ,IB) +
			    h_UOld(i  ,j  ,k  ,IB) );
	      
	    h_elec(i,j,k,IZ) = u*B-v*A;

	  } // end for i
	} // end for j
      } // end for k
      TIMER_STOP(timerElecField);
	
      if (dumpDataForDebugEnabled) {
	outputHdf5Debug(h_elec, "elec_", nStep);
	outputHdf5Debug(h_Q   , "prim_", nStep);
      }

      TIMER_START(timerMagSlopes);
      // compute magnetic slopes
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=1; k<ksize-1; k++) {
	for (int j=1; j<jsize-1; j++) {
	  for (int i=1; i<isize-1; i++) {

	    real_t bfSlopes[15];
	    real_t dbfSlopes[3][3];

	    real_t (&dbfX)[3] = dbfSlopes[IX];
	    real_t (&dbfY)[3] = dbfSlopes[IY];
	    real_t (&dbfZ)[3] = dbfSlopes[IZ];
	    
	    // get magnetic slopes dbf
	    bfSlopes[0]  = h_UOld(i  ,j  ,k  ,IA);
	    bfSlopes[1]  = h_UOld(i  ,j+1,k  ,IA);
	    bfSlopes[2]  = h_UOld(i  ,j-1,k  ,IA);
	    bfSlopes[3]  = h_UOld(i  ,j  ,k+1,IA);
	    bfSlopes[4]  = h_UOld(i  ,j  ,k-1,IA);
 
	    bfSlopes[5]  = h_UOld(i  ,j  ,k  ,IB);
	    bfSlopes[6]  = h_UOld(i+1,j  ,k  ,IB);
	    bfSlopes[7]  = h_UOld(i-1,j  ,k  ,IB);
	    bfSlopes[8]  = h_UOld(i  ,j  ,k+1,IB);
	    bfSlopes[9]  = h_UOld(i  ,j  ,k-1,IB);
 
	    bfSlopes[10] = h_UOld(i  ,j  ,k  ,IC);
	    bfSlopes[11] = h_UOld(i+1,j  ,k  ,IC);
	    bfSlopes[12] = h_UOld(i-1,j  ,k  ,IC);
	    bfSlopes[13] = h_UOld(i  ,j+1,k  ,IC);
	    bfSlopes[14] = h_UOld(i  ,j-1,k  ,IC);
 
	    // compute magnetic slopes
	    slope_unsplit_mhd_3d(bfSlopes, dbfSlopes);
	      
	    // store magnetic slopes
	    h_dA(i,j,k,0) = dbfX[IX];
	    h_dA(i,j,k,1) = dbfY[IX];
	    h_dA(i,j,k,2) = dbfZ[IX];

	    h_dB(i,j,k,0) = dbfX[IY];
	    h_dB(i,j,k,1) = dbfY[IY];
	    h_dB(i,j,k,2) = dbfZ[IY];

	    h_dC(i,j,k,0) = dbfX[IZ];
	    h_dC(i,j,k,1) = dbfY[IZ];
	    h_dC(i,j,k,2) = dbfZ[IZ];

	  } // end for i
	} // end for j
      } // end for k
      TIMER_STOP(timerMagSlopes);

      TIMER_START(timerTrace);
      // call trace computation routine
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=ghostWidth-2; k<ksize-ghostWidth+1; k++) {
	for (int j=ghostWidth-2; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth-2; i<isize-ghostWidth+1; i++) {
	      
	    real_t q[NVAR_MHD];
	    real_t qPlusX  [NVAR_MHD], qMinusX [NVAR_MHD],
	      qPlusY  [NVAR_MHD], qMinusY [NVAR_MHD],
	      qPlusZ  [NVAR_MHD], qMinusZ [NVAR_MHD];
	    real_t dq[3][NVAR_MHD];
	      
	    real_t bfNb[6];
	    real_t dbf[12];

	    real_t elecFields[3][2][2];
	    // alias to electric field components
	    real_t (&Ex)[2][2] = elecFields[IX];
	    real_t (&Ey)[2][2] = elecFields[IY];
	    real_t (&Ez)[2][2] = elecFields[IZ];

	    real_t qm[THREE_D][NVAR_MHD];
	    real_t qp[THREE_D][NVAR_MHD];
	    real_t qEdge[4][3][NVAR_MHD]; // array for qRT, qRB, qLT, qLB

	    real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

	    if (::gParams.slope_type==3) {
	      real_t qNb[3][3][3][NVAR_MHD];
	      // get primitive variables state vector
	      for (int ii=-1; ii<2; ++ii)
		for (int jj=-1; jj<2; ++jj)
		  for (int kk=-1; kk<2; ++kk)
		    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
		      qNb[ii+1][jj+1][kk+1][iVar] = h_Q(i+ii,j+jj,k+kk,iVar);
		    }
	      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
		q      [iVar] = h_Q(i  ,j  ,k  , iVar);
	      }

	      slope_unsplit_hydro_3d(qNb,dq);

	    } else {
	      // get primitive variables state vector
	      for (int iVar=0; iVar<NVAR_MHD; iVar++) {
		q      [iVar] = h_Q(i  ,j  ,k  , iVar);
		qPlusX [iVar] = h_Q(i+1,j  ,k  , iVar);
		qMinusX[iVar] = h_Q(i-1,j  ,k  , iVar);
		qPlusY [iVar] = h_Q(i  ,j+1,k  , iVar);
		qMinusY[iVar] = h_Q(i  ,j-1,k  , iVar);
		qPlusZ [iVar] = h_Q(i  ,j  ,k+1, iVar);
		qMinusZ[iVar] = h_Q(i  ,j  ,k-1, iVar);
	      }
		
	      // get hydro slopes dq
	      slope_unsplit_hydro_3d(q, 
				     qPlusX, qMinusX, 
				     qPlusY, qMinusY, 
				     qPlusZ, qMinusZ,
				     dq);
	    } // end slope_type = 0,1,2
	      
	    // get face-centered magnetic components
	    bfNb[0] = h_UOld(i  ,j  ,k  ,IA);
	    bfNb[1] = h_UOld(i+1,j  ,k  ,IA);
	    bfNb[2] = h_UOld(i  ,j  ,k  ,IB);
	    bfNb[3] = h_UOld(i  ,j+1,k  ,IB);
	    bfNb[4] = h_UOld(i  ,j  ,k  ,IC);
	    bfNb[5] = h_UOld(i  ,j  ,k+1,IC);
	      
	    // get dbf (transverse magnetic slopes) 
	    dbf[0]  = h_dA(i  ,j  ,k  ,IY);
	    dbf[1]  = h_dA(i  ,j  ,k  ,IZ);
	    dbf[2]  = h_dB(i  ,j  ,k  ,IX);
	    dbf[3]  = h_dB(i  ,j  ,k  ,IZ);
	    dbf[4]  = h_dC(i  ,j  ,k  ,IX);
	    dbf[5]  = h_dC(i  ,j  ,k  ,IY);
	      
	    dbf[6]  = h_dA(i+1,j  ,k  ,IY);
	    dbf[7]  = h_dA(i+1,j  ,k  ,IZ);
	    dbf[8]  = h_dB(i  ,j+1,k  ,IX);
	    dbf[9]  = h_dB(i  ,j+1,k  ,IZ);
	    dbf[10] = h_dC(i  ,j  ,k+1,IX);
	    dbf[11] = h_dC(i  ,j  ,k+1,IY);
	      
	    // get electric field components
	    Ex[0][0] = h_elec(i  ,j  ,k  ,IX);
	    Ex[0][1] = h_elec(i  ,j  ,k+1,IX);
	    Ex[1][0] = h_elec(i  ,j+1,k  ,IX);
	    Ex[1][1] = h_elec(i  ,j+1,k+1,IX);

	    Ey[0][0] = h_elec(i  ,j  ,k  ,IY);
	    Ey[0][1] = h_elec(i  ,j  ,k+1,IY);
	    Ey[1][0] = h_elec(i+1,j  ,k  ,IY);
	    Ey[1][1] = h_elec(i+1,j  ,k+1,IY);

	    Ez[0][0] = h_elec(i  ,j  ,k  ,IZ);
	    Ez[0][1] = h_elec(i  ,j+1,k  ,IZ);
	    Ez[1][0] = h_elec(i+1,j  ,k  ,IZ);
	    Ez[1][1] = h_elec(i+1,j+1,k  ,IZ);

	    // compute qm, qp and qEdge
	    trace_unsplit_mhd_3d_simpler(q, dq, bfNb, dbf, elecFields, 
					 dtdx, dtdy, dtdz, xPos,
					 qm, qp, qEdge);

	    // gravity predictor / modify velocity components
	    if (gravityEnabled) { 

	      real_t grav_x = HALF_F * dt * h_gravity(i,j,k,IX);
	      real_t grav_y = HALF_F * dt * h_gravity(i,j,k,IY);
	      real_t grav_z = HALF_F * dt * h_gravity(i,j,k,IZ);
		
	      qm[0][IU] += grav_x; qm[0][IV] += grav_y; qm[0][IW] += grav_z;
	      qp[0][IU] += grav_x; qp[0][IV] += grav_y; qp[0][IW] += grav_z;

	      qm[1][IU] += grav_x; qm[1][IV] += grav_y; qm[1][IW] += grav_z;
	      qp[1][IU] += grav_x; qp[1][IV] += grav_y; qp[1][IW] += grav_z;

	      qm[2][IU] += grav_x; qm[2][IV] += grav_y; qm[2][IW] += grav_z;
	      qp[2][IU] += grav_x; qp[2][IV] += grav_y; qp[2][IW] += grav_z;

	      qEdge[IRT][0][IU] += grav_x;
	      qEdge[IRT][0][IV] += grav_y;
	      qEdge[IRT][0][IW] += grav_z;
	      qEdge[IRT][1][IU] += grav_x;
	      qEdge[IRT][1][IV] += grav_y;
	      qEdge[IRT][1][IW] += grav_z;
	      qEdge[IRT][2][IU] += grav_x;
	      qEdge[IRT][2][IV] += grav_y;
	      qEdge[IRT][2][IW] += grav_z;

	      qEdge[IRB][0][IU] += grav_x;
	      qEdge[IRB][0][IV] += grav_y;
	      qEdge[IRB][0][IW] += grav_z;
	      qEdge[IRB][1][IU] += grav_x;
	      qEdge[IRB][1][IV] += grav_y;
	      qEdge[IRB][1][IW] += grav_z;
	      qEdge[IRB][2][IU] += grav_x;
	      qEdge[IRB][2][IV] += grav_y;
	      qEdge[IRB][2][IW] += grav_z;

	      qEdge[ILT][0][IU] += grav_x;
	      qEdge[ILT][0][IV] += grav_y;
	      qEdge[ILT][0][IW] += grav_z;
	      qEdge[ILT][1][IU] += grav_x;
	      qEdge[ILT][1][IV] += grav_y;
	      qEdge[ILT][1][IW] += grav_z;
	      qEdge[ILT][2][IU] += grav_x;
	      qEdge[ILT][2][IV] += grav_y;
	      qEdge[ILT][2][IW] += grav_z;

	      qEdge[ILB][0][IU] += grav_x;
	      qEdge[ILB][0][IV] += grav_y;
	      qEdge[ILB][0][IW] += grav_z;
	      qEdge[ILB][1][IU] += grav_x;
	      qEdge[ILB][1][IV] += grav_y;
	      qEdge[ILB][1][IW] += grav_z;
	      qEdge[ILB][2][IU] += grav_x;
	      qEdge[ILB][2][IV] += grav_y;
	      qEdge[ILB][2][IW] += grav_z;

	    } // end gravity predictor

	      // store qm, qp, qEdge : only what is really needed
	    for (int ivar=0; ivar<NVAR_MHD; ivar++) {
	      h_qm_x(i,j,k,ivar) = qm[0][ivar];
	      h_qp_x(i,j,k,ivar) = qp[0][ivar];
	      h_qm_y(i,j,k,ivar) = qm[1][ivar];
	      h_qp_y(i,j,k,ivar) = qp[1][ivar];
	      h_qm_z(i,j,k,ivar) = qm[2][ivar];
	      h_qp_z(i,j,k,ivar) = qp[2][ivar];
		
	      h_qEdge_RT (i,j,k,ivar) = qEdge[IRT][0][ivar]; 
	      h_qEdge_RB (i,j,k,ivar) = qEdge[IRB][0][ivar]; 
	      h_qEdge_LT (i,j,k,ivar) = qEdge[ILT][0][ivar]; 
	      h_qEdge_LB (i,j,k,ivar) = qEdge[ILB][0][ivar]; 

	      h_qEdge_RT2(i,j,k,ivar) = qEdge[IRT][1][ivar]; 
	      h_qEdge_RB2(i,j,k,ivar) = qEdge[IRB][1][ivar]; 
	      h_qEdge_LT2(i,j,k,ivar) = qEdge[ILT][1][ivar]; 
	      h_qEdge_LB2(i,j,k,ivar) = qEdge[ILB][1][ivar]; 

	      h_qEdge_RT3(i,j,k,ivar) = qEdge[IRT][2][ivar]; 
	      h_qEdge_RB3(i,j,k,ivar) = qEdge[IRB][2][ivar]; 
	      h_qEdge_LT3(i,j,k,ivar) = qEdge[ILT][2][ivar]; 
	      h_qEdge_LB3(i,j,k,ivar) = qEdge[ILB][2][ivar]; 
	    } // end for ivar

	  } // end for i
	} // end for j
      } // end for k
      TIMER_STOP(timerTrace);

      TIMER_START(timerUpdate);
      // Finally compute hydro fluxes from rieman solvers, 
      // compute emf's
      // and hydro update
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	      
	    real_riemann_t qleft[NVAR_MHD];
	    real_riemann_t qright[NVAR_MHD];
	    real_riemann_t flux_x[NVAR_MHD];
	    real_riemann_t flux_y[NVAR_MHD];
	    real_riemann_t flux_z[NVAR_MHD];
	    //real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      flux_x[iVar] = 0.0;
	      flux_y[iVar] = 0.0;
	      flux_z[iVar] = 0.0;
	    }

	    /*
	     * Solve Riemann problem at X-interfaces and compute
	     * X-fluxes
	     *
	     * Note that continuity of normal component of magnetic
	     * field is ensured inside riemann_mhd routine.
	     */
	      
	    qleft[ID]   = h_qm_x(i-1,j,k,ID);
	    qleft[IP]   = h_qm_x(i-1,j,k,IP);
	    qleft[IU]   = h_qm_x(i-1,j,k,IU);
	    qleft[IV]   = h_qm_x(i-1,j,k,IV);
	    qleft[IW]   = h_qm_x(i-1,j,k,IW);
	    qleft[IA]   = h_qm_x(i-1,j,k,IA);
	    qleft[IB]   = h_qm_x(i-1,j,k,IB);
	    qleft[IC]   = h_qm_x(i-1,j,k,IC);
	      
	    qright[ID]  = h_qp_x(i  ,j,k,ID);
	    qright[IP]  = h_qp_x(i  ,j,k,IP);
	    qright[IU]  = h_qp_x(i  ,j,k,IU);
	    qright[IV]  = h_qp_x(i  ,j,k,IV);
	    qright[IW]  = h_qp_x(i  ,j,k,IW);
	    qright[IA]  = h_qp_x(i  ,j,k,IA);
	    qright[IB]  = h_qp_x(i  ,j,k,IB);
	    qright[IC]  = h_qp_x(i  ,j,k,IC);
	      
	    // compute hydro flux_x
	    riemann_mhd(qleft,qright,flux_x);

	    /*
	     * Solve Riemann problem at Y-interfaces and compute Y-fluxes
	     */
	    qleft[ID]   = h_qm_y(i,j-1,k,ID);
	    qleft[IP]   = h_qm_y(i,j-1,k,IP);
	    qleft[IU]   = h_qm_y(i,j-1,k,IV); // watchout IU, IV permutation
	    qleft[IV]   = h_qm_y(i,j-1,k,IU); // watchout IU, IV permutation
	    qleft[IW]   = h_qm_y(i,j-1,k,IW);
	    qleft[IA]   = h_qm_y(i,j-1,k,IB); // watchout IA, IB permutation
	    qleft[IB]   = h_qm_y(i,j-1,k,IA); // watchout IA, IB permutation
	    qleft[IC]   = h_qm_y(i,j-1,k,IC);
	      
	    qright[ID]  = h_qp_y(i,j  ,k,ID);
	    qright[IP]  = h_qp_y(i,j  ,k,IP);
	    qright[IU]  = h_qp_y(i,j  ,k,IV); // watchout IU, IV permutation
	    qright[IV]  = h_qp_y(i,j  ,k,IU); // watchout IU, IV permutation
	    qright[IW]  = h_qp_y(i,j  ,k,IW);
	    qright[IA]  = h_qp_y(i,j  ,k,IB); // watchout IA, IB permutation
	    qright[IB]  = h_qp_y(i,j  ,k,IA); // watchout IA, IB permutation
	    qright[IC]  = h_qp_y(i,j  ,k,IC);
	      
	    // compute hydro flux_y
	    riemann_mhd(qleft,qright,flux_y);

	    /*
	     * Solve Riemann problem at Z-interfaces and compute
	     * Z-fluxes
	     */
	    qleft[ID]   = h_qm_z(i,j,k-1,ID);
	    qleft[IP]   = h_qm_z(i,j,k-1,IP);
	    qleft[IU]   = h_qm_z(i,j,k-1,IW); // watchout IU, IW permutation
	    qleft[IV]   = h_qm_z(i,j,k-1,IV);
	    qleft[IW]   = h_qm_z(i,j,k-1,IU); // watchout IU, IW permutation
	    qleft[IA]   = h_qm_z(i,j,k-1,IC); // watchout IA, IC permutation
	    qleft[IB]   = h_qm_z(i,j,k-1,IB);
	    qleft[IC]   = h_qm_z(i,j,k-1,IA); // watchout IA, IC permutation
	      
	    qright[ID]  = h_qp_z(i,j,k  ,ID);
	    qright[IP]  = h_qp_z(i,j,k  ,IP);
	    qright[IU]  = h_qp_z(i,j,k  ,IW); // watchout IU, IW permutation
	    qright[IV]  = h_qp_z(i,j,k  ,IV);
	    qright[IW]  = h_qp_z(i,j,k  ,IU); // watchout IU, IW permutation
	    qright[IA]  = h_qp_z(i,j,k  ,IC); // watchout IA, IC permutation
	    qright[IB]  = h_qp_z(i,j,k  ,IB);
	    qright[IC]  = h_qp_z(i,j,k  ,IA); // watchout IA, IC permutation

	    // compute hydro flux_z
	    riemann_mhd(qleft,qright,flux_z);

	    /*
	     * update mhd array with hydro fluxes.
	     *
	     * \note the "if" guards
	     * prevents from writing in ghost zones, only usefull
	     * when degugging, should be removed later as ghostZones
	     * are anyway erased in make_boudaries routine.
	     */
	    if ( i > ghostWidth       and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {
	      h_UNew(i-1,j  ,k  ,ID) -= flux_x[ID]*dtdx;
	      h_UNew(i-1,j  ,k  ,IP) -= flux_x[IP]*dtdx;
	      h_UNew(i-1,j  ,k  ,IU) -= flux_x[IU]*dtdx;
	      h_UNew(i-1,j  ,k  ,IV) -= flux_x[IV]*dtdx;
	      h_UNew(i-1,j  ,k  ,IW) -= flux_x[IW]*dtdx;
	    }
	      
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_x[ID]*dtdx;
	      h_UNew(i  ,j  ,k  ,IP) += flux_x[IP]*dtdx;
	      h_UNew(i  ,j  ,k  ,IU) += flux_x[IU]*dtdx;
	      h_UNew(i  ,j  ,k  ,IV) += flux_x[IV]*dtdx;
	      h_UNew(i  ,j  ,k  ,IW) += flux_x[IW]*dtdx;
	    }

	    if ( i < isize-ghostWidth and
		 j > ghostWidth       and
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j-1,k  ,ID) -= flux_y[ID]*dtdy;
	      h_UNew(i  ,j-1,k  ,IP) -= flux_y[IP]*dtdy;
	      h_UNew(i  ,j-1,k  ,IU) -= flux_y[IV]*dtdy; // IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IV) -= flux_y[IU]*dtdy; // IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IW) -= flux_y[IW]*dtdy;
	    }
	      
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_y[ID]*dtdy;
	      h_UNew(i  ,j  ,k  ,IP) += flux_y[IP]*dtdy;
	      h_UNew(i  ,j  ,k  ,IU) += flux_y[IV]*dtdy; // IU and IV swapped
	      h_UNew(i  ,j  ,k  ,IV) += flux_y[IU]*dtdy; // IU and IV swapped
	      h_UNew(i  ,j  ,k  ,IW) += flux_y[IW]*dtdy;
	    }

	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and
		 k > ghostWidth ) {
	      h_UNew(i  ,j  ,k-1,ID) -= flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,k-1,IP) -= flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,k-1,IU) -= flux_z[IW]*dtdz; // IU and IW swapped
	      h_UNew(i  ,j  ,k-1,IV) -= flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,k-1,IW) -= flux_z[IU]*dtdz; // IU and IW swapped
	    }

	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,k  ,IP) += flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,k  ,IU) += flux_z[IW]*dtdz; // IU and IW swapped
	      h_UNew(i  ,j  ,k  ,IV) += flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,k  ,IW) += flux_z[IU]*dtdz; // IU and IW swapped
	    }

	    // now compute EMF's and update magnetic field variables
	    // see DUMSES routine named cmp_mag_flx (TAKE CARE of index
	    // shift appearing in calling arguments)
	      
	    real_t qEdge_emfX[4][NVAR_MHD];
	    real_t qEdge_emfY[4][NVAR_MHD];
	    real_t qEdge_emfZ[4][NVAR_MHD];
	      
	    // preparation for calling compute_emf (equivalent to cmp_mag_flx
	    // in DUMSES)
	    // in the following, the 3 first indexes in qEdge_emf array play
	    // the same offset role as in the calling argument of cmp_mag_flx 
	    // in DUMSES (if you see what I mean ?!)

	    // actually compute emfZ 
	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      qEdge_emfZ[IRT][iVar] = h_qEdge_RT3(i-1,j-1,k,iVar); 
	      qEdge_emfZ[IRB][iVar] = h_qEdge_RB3(i-1,j  ,k,iVar); 
	      qEdge_emfZ[ILT][iVar] = h_qEdge_LT3(i  ,j-1,k,iVar); 
	      qEdge_emfZ[ILB][iVar] = h_qEdge_LB3(i  ,j  ,k,iVar); 
	    }
	    real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ);
	    h_emf(i,j,k,I_EMFZ) = emfZ;

	    // actually compute emfY (take care that RB and LT are
	    // swapped !!!)
	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      qEdge_emfY[IRT][iVar] = h_qEdge_RT2(i-1,j  ,k-1,iVar);
	      qEdge_emfY[IRB][iVar] = h_qEdge_LT2(i  ,j  ,k-1,iVar); 
	      qEdge_emfY[ILT][iVar] = h_qEdge_RB2(i-1,j  ,k  ,iVar);
	      qEdge_emfY[ILB][iVar] = h_qEdge_LB2(i  ,j  ,k  ,iVar); 
	    }

	    real_t emfY = compute_emf<EMFY>(qEdge_emfY);
	    h_emf(i,j,k,I_EMFY) = emfY;

	    // actually compute emfX
	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      qEdge_emfX[IRT][iVar] = h_qEdge_RT(i  ,j-1,k-1,iVar);
	      qEdge_emfX[IRB][iVar] = h_qEdge_RB(i  ,j-1,k  ,iVar);
	      qEdge_emfX[ILT][iVar] = h_qEdge_LT(i  ,j  ,k-1,iVar);
	      qEdge_emfX[ILB][iVar] = h_qEdge_LB(i  ,j  ,k  ,iVar);
	    }
	    real_t emfX = compute_emf<EMFX>(qEdge_emfX);
	    h_emf(i,j,k,I_EMFX) = emfX;
	      
	  } // end for i
	} // end for j
      } // end for k

	// gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(h_UNew, h_UOld, dt);
      }

      TIMER_STOP(timerUpdate);

      TIMER_START(timerCtUpdate);
      /*
       * magnetic field update
       */
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {

	    // update with EMFZ
	    if (k<ksize-ghostWidth) {
	      h_UNew(i ,j ,k, IA) += ( h_emf(i  ,j+1, k, I_EMFZ) - 
				       h_emf(i,  j  , k, I_EMFZ) ) * dtdy;
		
	      h_UNew(i ,j ,k, IB) -= ( h_emf(i+1,j  , k, I_EMFZ) - 
				       h_emf(i  ,j  , k, I_EMFZ) ) * dtdx;

	    }

	    // update BX
	    h_UNew(i ,j ,k, IA) -= ( h_emf(i,j,k+1, I_EMFY) -
				     h_emf(i,j,k  , I_EMFY) ) * dtdz;

	    // update BY
	    h_UNew(i ,j ,k, IB) += ( h_emf(i,j,k+1, I_EMFX) -
				     h_emf(i,j,k  , I_EMFX) ) * dtdz;
	       
	    // update BZ
	    h_UNew(i ,j ,k, IC) += ( h_emf(i+1,j  ,k, I_EMFY) -
				     h_emf(i  ,j  ,k, I_EMFY) ) * dtdx;
	    h_UNew(i ,j ,k, IC) -= ( h_emf(i  ,j+1,k, I_EMFX) -
				     h_emf(i  ,j  ,k, I_EMFX) ) * dtdy;

	  } // end for i
	} // end for j
      } // end for k
      TIMER_STOP(timerCtUpdate);

      if (dumpDataForDebugEnabled) {
	outputHdf5Debug(h_debug, "flux_x_", nStep);
	//outputHdf5Debug(h_debug, "emf_", nStep);

	// compute divergence B
	compute_divB(h_UNew,h_debug2);
	outputHdf5Debug(h_debug2, "divB_", nStep);
	  
	outputHdf5Debug(h_dA, "dA_", nStep);
	outputHdf5Debug(h_dB, "dB_", nStep);
	outputHdf5Debug(h_dC, "dC_", nStep);
	  
	outputHdf5Debug(h_qm_x, "qm_x_", nStep);
	outputHdf5Debug(h_qm_y, "qm_y_", nStep);
	outputHdf5Debug(h_qm_z, "qm_z_", nStep);
	  
	outputHdf5Debug(h_qEdge_RT, "qEdge_RT_", nStep);
	outputHdf5Debug(h_qEdge_RB, "qEdge_RB_", nStep);
	outputHdf5Debug(h_qEdge_LT, "qEdge_LT_", nStep);

	outputHdf5Debug(h_qEdge_RT2, "qEdge_RT2_", nStep);
	outputHdf5Debug(h_qEdge_RB2, "qEdge_RB2_", nStep);
	outputHdf5Debug(h_qEdge_LT2, "qEdge_LT2_", nStep);

	outputHdf5Debug(h_qEdge_RT3, "qEdge_RT3_", nStep);
	outputHdf5Debug(h_qEdge_RB3, "qEdge_RB3_", nStep);
	outputHdf5Debug(h_qEdge_LT3, "qEdge_LT3_", nStep);
      }
	
      TIMER_START(timerDissipative);
      // update borders
      real_t &nu  = _gParams.nu;
      real_t &eta = _gParams.eta;
      if (nu>0 or eta>0) {
	make_all_boundaries(h_UNew);
      }

      if (eta>0) {
	// update magnetic field with resistivity emf
	compute_resistivity_emf_3d(h_UNew, h_emf);
	compute_ct_update_3d      (h_UNew, h_emf, dt);

	real_t &cIso = _gParams.cIso;
	if (cIso<=0) { // non-isothermal simulations
	  // compute energy flux
	  compute_resistivity_energy_flux_3d(h_UNew, h_qm_x, h_qm_y, h_qm_z, dt);
	  compute_hydro_update_energy       (h_UNew, h_qm_x, h_qm_y, h_qm_z);
	}
      }

      // compute viscosity forces
      if (nu>0) {
	// re-use h_qm_x and h_qm_y
	HostArray<real_t> &flux_x = h_qm_x;
	HostArray<real_t> &flux_y = h_qm_y;
	HostArray<real_t> &flux_z = h_qm_z;
	  
	compute_viscosity_flux(h_UNew, flux_x, flux_y, flux_z, dt);
	compute_hydro_update  (h_UNew, flux_x, flux_y, flux_z);
	  
      } // end compute viscosity force / update	
      TIMER_STOP(timerDissipative);

      /*
       * random forcing
       */
      if (randomForcingEnabled) {
	  
	real_t norm = compute_random_forcing_normalization(h_UNew, dt);
	  
	add_random_forcing(h_UNew, dt, norm);
	  
      }
      if (randomForcingOrnsteinUhlenbeckEnabled) {
	  
	// add forcing field in real space
	pForcingOrnsteinUhlenbeck->add_forcing_field(h_UNew, dt);
	  
      }

    } // end THREE_D - implementation version 3 / 4

  } // MHDRunGodunov::godunov_unsplit_cpu_v3

} // namespace hydroSimu
