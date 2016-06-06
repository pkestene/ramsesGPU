#include "MHDRunGodunovMpi.h"

#include "constoprim.h"
#include "riemann_mhd.h"
#include "trace_mhd.h"

namespace hydroSimu {

  // =======================================================
  // =======================================================
  void MHDRunGodunovMpi::godunov_unsplit_cpu_v1(HostArray<real_t>& h_UOld, 
						HostArray<real_t>& h_UNew, 
						real_t dt, int nStep)
  {
  
    (void) nStep;
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;
    
    // conservative variable domain array
    real_t *U = h_UOld.data();
    
    // primitive variable domain array
    real_t *Q = h_Q.data();

    // section / domain size
    int arraySize = h_Q.section();

    // this is the less memory scrooge version
    // this version is about 1/3rd faster than implementation 0
    
    if (dimType == TWO_D) {
      
      // first compute qm, qp and qEdge (from trace)
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(2) schedule(auto)
#endif // _OPENMP
      for (int j=ghostWidth-2; j<jsize-ghostWidth+2; j++) {
	for (int i=ghostWidth-2; i<isize-ghostWidth+2; i++) {
	  
	  real_t qNb[3][3][NVAR_MHD];
	  real_t bfNb[4][4][3];
	  real_t qm[TWO_D][NVAR_MHD];
	  real_t qp[TWO_D][NVAR_MHD];
	  real_t qEdge[4][NVAR_MHD]; // array for qRT, qRB, qLT, qLB
	  real_t c=0;
	  
	  int iG = i + nx*myMpiPos[0];
	  real_t xPos = _gParams.xMin + dx/2 + (iG-ghostWidth)*dx;

	  // prepare qNb : q state in the 3-by-3 neighborhood
	  // note that current cell (ii,jj) is in qNb[1][1]
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
	    
	  // prepare bfNb : bf (face centered mag field) in the
	  // 4-by-4 neighborhood
	  // note that current cell (ii,jj) is in bfNb[1][1]
	  for (int di=0; di<4; di++)
	    for (int dj=0; dj<4; dj++) {
	      int indexLoc = (i+di-1)+(j+dj-1)*isize;
	      getMagField(U, arraySize, indexLoc, bfNb[di][dj]);
	    }
	    	    
	  // compute trace 2d finally !!! 
	  trace_unsplit_mhd_2d(qNb, bfNb, c, dtdx, dtdy, xPos, qm, qp, qEdge);
	    
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

	// second compute fluxes from rieman solvers, and update
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
	  h_emf(i,j,I_EMFZ) = emfZ;

	} // end for i
      } // end for j

	/*
	 * magnetic field update (constraint transport)
	 */
#ifdef _OPENMP
#pragma omp parallel default(shared) 
#pragma omp for collapse(2) schedule(auto)
#endif // _OPENMP
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	  // left-face B-field
	  h_UNew(i  ,j  ,IA) += ( h_emf(i  ,j+1, I_EMFZ) - h_emf(i,j, I_EMFZ) )*dtdy;
	  h_UNew(i  ,j  ,IB) -= ( h_emf(i+1,j  , I_EMFZ) - h_emf(i,j, I_EMFZ) )*dtdx;
		    
	} // end for i
      } // end for j

    } else { // THREE_D - implementation version 1
		
      // first compute qm, qp and qEdge (from trace)
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=ghostWidth-2; k<ksize-ghostWidth+1; k++) {
	for (int j=ghostWidth-2; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth-2; i<isize-ghostWidth+1; i++) {
	      
	    real_t qNb[3][3][3][NVAR_MHD];
	    real_t bfNb[4][4][4][3];
	    real_t qm[THREE_D][NVAR_MHD];
	    real_t qp[THREE_D][NVAR_MHD];
	    real_t qEdge[4][3][NVAR_MHD]; // array for qRT, qRB, qLT, qLB
	    real_t c=0;
	      
	    int iG = i + nx*myMpiPos[0];
	    real_t xPos = _gParams.xMin + dx/2 + (iG-ghostWidth)*dx;

	    // prepare qNb : q state in the 3-by-3-by-3 neighborhood
	    // note that current cell (i,j,k) is in qNb[1][1][1]
	    // also note that the effective stencil is 4-by-4-by-4 since
	    // computation of primitive variable (q) requires mag
	    // field on the right (see computePrimitives_MHD_3D)
	    for (int di=0; di<3; di++) {
	      for (int dj=0; dj<3; dj++) {
		for (int dk=0; dk<3; dk++) {
		  for (int iVar=0; iVar < NVAR_MHD; iVar++) {
		    int indexLoc = (i+di-1)+(j+dj-1)*isize+(k+dk-1)*isize*jsize; // centered index
		    getPrimitiveVector(Q, arraySize, indexLoc, qNb[di][dj][dk]);
		  } // end for iVar
		} // end for dk
	      } // end for dj
	    } // end for di

	      // prepare bfNb : bf (face centered mag field) in the
	      // 4-by-4-by-4 neighborhood
	      // note that current cell (i,j,k) is in bfNb[1][1][1]
	    for (int di=0; di<4; di++) {
	      for (int dj=0; dj<4; dj++) {
		for (int dk=0; dk<4; dk++) {
		  int indexLoc = (i+di-1)+(j+dj-1)*isize+(k+dk-1)*isize*jsize;
		  getMagField(U, arraySize, indexLoc, bfNb[di][dj][dk]);
		} // end for dk
	      } // end for dj
	    } // end for di
	      
	      // compute trace 3d finally !!! 
	    trace_unsplit_mhd_3d(qNb, bfNb, c, dtdx, dtdy, dtdz, xPos, qm, qp, qEdge);

	    // store qm, qp, qEdge : only what is really needed
	    for (int ivar=0; ivar<NVAR_MHD; ivar++) {
	      h_qm_x(i,j,k,ivar) = qm[0][ivar];
	      h_qp_x(i,j,k,ivar) = qp[0][ivar];
	      h_qm_y(i,j,k,ivar) = qm[1][ivar];
	      h_qp_y(i,j,k,ivar) = qp[1][ivar];
	      h_qm_z(i,j,k,ivar) = qm[2][ivar];
	      h_qp_z(i,j,k,ivar) = qp[2][ivar];
		
	      h_qEdge_RT(i,j,k,ivar) = qEdge[IRT][0][ivar]; 
	      h_qEdge_RB(i,j,k,ivar) = qEdge[IRB][0][ivar]; 
	      h_qEdge_LT(i,j,k,ivar) = qEdge[ILT][0][ivar]; 
	      h_qEdge_LB(i,j,k,ivar) = qEdge[ILB][0][ivar]; 

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
	
	// second compute fluxes from rieman solvers, and update
      for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	      
	    real_riemann_t qleft[NVAR_MHD];
	    real_riemann_t qright[NVAR_MHD];
	    real_riemann_t flux_x[NVAR_MHD];
	    real_riemann_t flux_y[NVAR_MHD];
	    real_riemann_t flux_z[NVAR_MHD];
	    //int iG = i + nx*myMpiPos[0];
	    //real_t xPos = ::gParams.xMin + dx/2 + (iG-ghostWidth)*dx;

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
	     * update mhd array with hydro fluxes
	     */
	    h_UNew(i-1,j  ,k  ,ID) -= flux_x[ID]*dtdx;
	    h_UNew(i-1,j  ,k  ,IP) -= flux_x[IP]*dtdx;
	    h_UNew(i-1,j  ,k  ,IU) -= flux_x[IU]*dtdx;
	    h_UNew(i-1,j  ,k  ,IV) -= flux_x[IV]*dtdx;
	    h_UNew(i-1,j  ,k  ,IW) -= flux_x[IW]*dtdx;
	      
	    h_UNew(i  ,j  ,k  ,ID) += flux_x[ID]*dtdx;
	    h_UNew(i  ,j  ,k  ,IP) += flux_x[IP]*dtdx;
	    h_UNew(i  ,j  ,k  ,IU) += flux_x[IU]*dtdx;
	    h_UNew(i  ,j  ,k  ,IV) += flux_x[IV]*dtdx;
	    h_UNew(i  ,j  ,k  ,IW) += flux_x[IW]*dtdx;
	      
	    h_UNew(i  ,j-1,k  ,ID) -= flux_y[ID]*dtdy;
	    h_UNew(i  ,j-1,k  ,IP) -= flux_y[IP]*dtdy;
	    h_UNew(i  ,j-1,k  ,IU) -= flux_y[IV]*dtdy; // IU and IV swapped
	    h_UNew(i  ,j-1,k  ,IV) -= flux_y[IU]*dtdy; // IU and IV swapped
	    h_UNew(i  ,j-1,k  ,IW) -= flux_y[IW]*dtdy;

	      
	    h_UNew(i  ,j  ,k  ,ID) += flux_y[ID]*dtdy;
	    h_UNew(i  ,j  ,k  ,IP) += flux_y[IP]*dtdy;
	    h_UNew(i  ,j  ,k  ,IU) += flux_y[IV]*dtdy; // IU and IV swapped
	    h_UNew(i  ,j  ,k  ,IV) += flux_y[IU]*dtdy; // IU and IV swapped
	    h_UNew(i  ,j  ,k  ,IW) += flux_y[IW]*dtdy;
	      
	    h_UNew(i  ,j  ,k-1,ID) -= flux_z[ID]*dtdz;
	    h_UNew(i  ,j  ,k-1,IP) -= flux_z[IP]*dtdz;
	    h_UNew(i  ,j  ,k-1,IU) -= flux_z[IW]*dtdz; // IU and IW swapped
	    h_UNew(i  ,j  ,k-1,IV) -= flux_z[IV]*dtdz;
	    h_UNew(i  ,j  ,k-1,IW) -= flux_z[IU]*dtdz; // IU and IW swapped
	      
	    h_UNew(i  ,j  ,k  ,ID) += flux_z[ID]*dtdz;
	    h_UNew(i  ,j  ,k  ,IP) += flux_z[IP]*dtdz;
	    h_UNew(i  ,j  ,k  ,IU) += flux_z[IW]*dtdz; // IU and IW swapped
	    h_UNew(i  ,j  ,k  ,IV) += flux_z[IV]*dtdz;
	    h_UNew(i  ,j  ,k  ,IW) += flux_z[IU]*dtdz; // IU and IW swapped

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
	      
	    // actually compute emfY (take care that RB and LT are
	    // swapped !!!)
	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      qEdge_emfY[IRT][iVar] = h_qEdge_RT2(i-1,j  ,k-1,iVar);
	      qEdge_emfY[IRB][iVar] = h_qEdge_LT2(i  ,j  ,k-1,iVar); 
	      qEdge_emfY[ILT][iVar] = h_qEdge_RB2(i-1,j  ,k  ,iVar);
	      qEdge_emfY[ILB][iVar] = h_qEdge_LB2(i  ,j  ,k  ,iVar); 
	    }
	    real_t emfY = compute_emf<EMFY>(qEdge_emfY);

	    // actually compute emfX
	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      qEdge_emfX[IRT][iVar] = h_qEdge_RT(i  ,j-1,k-1,iVar);
	      qEdge_emfX[IRB][iVar] = h_qEdge_RB(i  ,j-1,k  ,iVar);
	      qEdge_emfX[ILT][iVar] = h_qEdge_LT(i  ,j  ,k-1,iVar);
	      qEdge_emfX[ILB][iVar] = h_qEdge_LB(i  ,j  ,k  ,iVar);
	    }
	    real_t emfX = compute_emf<EMFX>(qEdge_emfX);

	    // now update h_UNew with emfZ
	    // (Constrained transport for face-centered B-field)
	    h_UNew(i  ,j  ,k  ,IA) -= emfZ*dtdy;
	    h_UNew(i  ,j-1,k  ,IA) += emfZ*dtdy;
	      
	    h_UNew(i  ,j  ,k  ,IB) += emfZ*dtdx;  
	    h_UNew(i-1,j  ,k  ,IB) -= emfZ*dtdx;

	    // now update h_UNew with emfY, emfX
	    h_UNew(i  ,j  ,k  ,IA) += emfY*dtdz;
	    h_UNew(i  ,j  ,k-1,IA) -= emfY*dtdz;
	      
	    h_UNew(i  ,j  ,k  ,IB) -= emfX*dtdz;
	    h_UNew(i  ,j  ,k-1,IB) += emfX*dtdz;
	      
	    h_UNew(i  ,j  ,k  ,IC) -= emfY*dtdx;
	    h_UNew(i-1,j  ,k  ,IC) += emfY*dtdx;
	    h_UNew(i  ,j  ,k  ,IC) += emfX*dtdy;
	    h_UNew(i  ,j-1,k  ,IC) -= emfX*dtdy;
	      
	  } // end for i
	} // end for j
      } // end for k

    } // end THREE_D
    
  } // MHDRunGodunovMpi::godunov_unsplit_cpu_v1

} // namespace hydroSimu
