#include "MHDRunGodunov.h"

#include "constoprim.h"
#include "riemann_mhd.h"
#include "trace_mhd.h"

/** a dummy device-only swap function */
static void swap_v(real_t& a, real_t& b) {
   
  real_t tmp = a;
  a = b;
  b = tmp; 
   
} // swap_v

namespace hydroSimu {

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_cpu_v0(HostArray<real_t>& h_UOld, 
					     HostArray<real_t>& h_UNew, 
					     real_t dt, int nStep)
  {
  
    (void) nStep;
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;
    
    ////////////////////////////////////////////////     
    ////////////////////////////////////////////////     
    if (dimType == TWO_D) {
      
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	  
	  // primitive variables (local array)
	  real_t qLoc[NVAR_MHD];
	  real_t qNeighbors[2*TWO_D][NVAR_MHD];
	  
	  // slopes
	  real_t dq[TWO_D][NVAR_MHD];
	  real_t dbf[TWO_D][THREE_D];
	  real_t bfNb[6];
	  real_t bfNb2[TWO_D*2];
	  real_t dAB[TWO_D*2];

	  // reconstructed state on cell faces
	  // aka riemann solver input
	  real_t qleft_x[NVAR_MHD];
	  real_t qleft_y[NVAR_MHD];
	  real_t qright_x[NVAR_MHD];
	  real_t qright_y[NVAR_MHD];
	  
	  // riemann solver output
	  real_t flux_x[NVAR_MHD];
	  real_t flux_y[NVAR_MHD];

	  // emf
	  real_t Ez[2][2];
	  real_t qEdge_RT[NVAR_MHD];
	  real_t qEdge_RB[NVAR_MHD];
	  real_t qEdge_LT[NVAR_MHD];
	  real_t qEdge_LB[NVAR_MHD];

	  // other variables
	  real_t xPos;
	  
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  // deal with left interface along X !
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	  ////////////////
	  // compute RIGHT state for riemann problem (i,j)
	  ////////////////

	  // get primitive variables state vector
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	    qLoc[iVar]          = h_Q(i  ,j  ,iVar);
	    qNeighbors[0][iVar] = h_Q(i+1,j  ,iVar);
	    qNeighbors[1][iVar] = h_Q(i-1,j  ,iVar);
	    qNeighbors[2][iVar] = h_Q(i  ,j+1,iVar);
	    qNeighbors[3][iVar] = h_Q(i  ,j-1,iVar);
	  
	  } // end for iVar

	  // 1. compute hydro slopes
	  // compute slopes in left neighbor along X
	  slope_unsplit_hydro_2d_simple(qLoc, 
					qNeighbors[0],
					qNeighbors[1],
					qNeighbors[2],
					qNeighbors[3],
					dq);
	  
	  // 2. compute mag slopes (i,j)
	  bfNb[0] =  h_UOld(i  ,j  ,IA);
	  bfNb[1] =  h_UOld(i  ,j+1,IA);
	  bfNb[2] =  h_UOld(i  ,j-1,IA);
	  bfNb[3] =  h_UOld(i  ,j  ,IB);
	  bfNb[4] =  h_UOld(i+1,j  ,IB);
	  bfNb[5] =  h_UOld(i-1,j  ,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[0] = dbf[IY][IX];
	  dAB[1] = dbf[IX][IY];

	  // (i+1,j  )
	  bfNb[0] =  h_UOld(i+1,j  ,IA);
	  bfNb[1] =  h_UOld(i+1,j+1,IA);
	  bfNb[2] =  h_UOld(i+1,j-1,IA);
	  bfNb[3] =  h_UOld(i+1,j  ,IB);
	  bfNb[4] =  h_UOld(i+2,j  ,IB);
	  bfNb[5] =  h_UOld(i  ,j  ,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[2] = dbf[IY][IX];

	  // (i  ,j+1)
	  bfNb[0] =  h_UOld(i  ,j+1,IA);
	  bfNb[1] =  h_UOld(i  ,j+2,IA);
	  bfNb[2] =  h_UOld(i  ,j  ,IA);
	  bfNb[3] =  h_UOld(i  ,j+1,IB);
	  bfNb[4] =  h_UOld(i+1,j+1,IB);
	  bfNb[5] =  h_UOld(i-1,j+1,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[3] = dbf[IX][IY];

	  // 3. compute Ez
	  for (int di=0; di<2; di++)
	    for (int dj=0; dj<2; dj++) {
      
	      int centerX = i+di;
	      int centerY = j+dj;
	      real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,IU) + 
				   h_Q(centerX-1,centerY  ,IU) + 
				   h_Q(centerX  ,centerY-1,IU) + 
				   h_Q(centerX  ,centerY  ,IU) ); 
	      
	      real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,IV) +
				   h_Q(centerX-1,centerY  ,IV) +
				   h_Q(centerX  ,centerY-1,IV) + 
				   h_Q(centerX  ,centerY  ,IV) );
	      
	      real_t A  = 0.5  *  (h_UOld(centerX  ,centerY-1,IA) + 
				   h_UOld(centerX  ,centerY  ,IA) );
	      
	      real_t B  = 0.5  *  (h_UOld(centerX-1,centerY  ,IB) + 
				   h_UOld(centerX  ,centerY  ,IB) );
	      
	      Ez[di][dj] = u*B-v*A;
	    }


	  // 4. perform trace reconstruction
	  //
	  // Compute reconstructed states at left interface along X 
	  // in current cell
	  xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

	  // (i,j)
	  bfNb2[0] =  h_UOld(i  ,j  ,IA);
	  bfNb2[1] =  h_UOld(i+1,j  ,IA);
	  bfNb2[2] =  h_UOld(i  ,j  ,IB);
	  bfNb2[3] =  h_UOld(i  ,j+1,IB);


	  // left interface : right state along x
	  trace_unsplit_mhd_2d_face(qLoc, dq, bfNb2, dAB, Ez,
				    dtdx, dtdy, xPos, FACE_XMIN,
				    qright_x);

	  // left interface : right state along y
	  trace_unsplit_mhd_2d_face(qLoc, dq, bfNb2, dAB, Ez,
				    dtdx, dtdy, xPos, FACE_YMIN,
				    qright_y);

	  // swap qright_y
	  swap_v(qright_y[IU], qright_y[IV]);
	  swap_v(qright_y[IA], qright_y[IB]);


	  // qEdge_LB
	  trace_unsplit_mhd_2d_face(qLoc, dq, bfNb2, dAB, Ez,
				    dtdx, dtdy, xPos, EDGE_LB,
				    qEdge_LB);
	  
	  ////////////////
	  // compute LEFT state for riemann problem (i-1,j)
	  ////////////////
	  
	  // get primitive variables state vector
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	    qLoc[iVar]          = h_Q(i-1,j  ,iVar);
	    qNeighbors[0][iVar] = h_Q(i  ,j  ,iVar);
	    qNeighbors[1][iVar] = h_Q(i-2,j  ,iVar);
	    qNeighbors[2][iVar] = h_Q(i-1,j+1,iVar);
	    qNeighbors[3][iVar] = h_Q(i-1,j-1,iVar);
	  
	  } // end for iVar

	  // 1. compute hydro slopes
	  // compute slopes in left neighbor along X
	  slope_unsplit_hydro_2d_simple(qLoc, 
					qNeighbors[0],
					qNeighbors[1],
					qNeighbors[2],
					qNeighbors[3],
					dq);
	  
	  // 2. compute mag slopes (i-1,j)
	  bfNb[0] =  h_UOld(i-1,j  ,IA);
	  bfNb[1] =  h_UOld(i-1,j+1,IA);
	  bfNb[2] =  h_UOld(i-1,j-1,IA);
	  bfNb[3] =  h_UOld(i-1,j  ,IB);
	  bfNb[4] =  h_UOld(i  ,j  ,IB);
	  bfNb[5] =  h_UOld(i-2,j  ,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[0] = dbf[IY][IX];
	  dAB[1] = dbf[IX][IY];

	  // (i  ,j  )
	  bfNb[0] =  h_UOld(i  ,j  ,IA);
	  bfNb[1] =  h_UOld(i  ,j+1,IA);
	  bfNb[2] =  h_UOld(i  ,j-1,IA);
	  bfNb[3] =  h_UOld(i  ,j  ,IB);
	  bfNb[4] =  h_UOld(i+1,j  ,IB);
	  bfNb[5] =  h_UOld(i-1,j  ,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[2] = dbf[IY][IX];
	  
	  // (i-1,j+1)
	  bfNb[0] =  h_UOld(i-1,j+1,IA);
	  bfNb[1] =  h_UOld(i-1,j+2,IA);
	  bfNb[2] =  h_UOld(i-1,j  ,IA);
	  bfNb[3] =  h_UOld(i-1,j+1,IB);
	  bfNb[4] =  h_UOld(i  ,j+1,IB);
	  bfNb[5] =  h_UOld(i-2,j+1,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[3] = dbf[IX][IY];

	  // 3. compute Ez
	  for (int di=0; di<2; di++)
	    for (int dj=0; dj<2; dj++) {
      
	      int centerX = i-1+di;
	      int centerY = j  +dj;
	      real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,IU) + 
				   h_Q(centerX-1,centerY  ,IU) + 
				   h_Q(centerX  ,centerY-1,IU) + 
				   h_Q(centerX  ,centerY  ,IU) ); 
	      
	      real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,IV) +
				   h_Q(centerX-1,centerY  ,IV) +
				   h_Q(centerX  ,centerY-1,IV) + 
				   h_Q(centerX  ,centerY  ,IV) );
	      
	      real_t A  = 0.5  *  (h_UOld(centerX  ,centerY-1,IA) + 
				   h_UOld(centerX  ,centerY  ,IA) );
	      
	      real_t B  = 0.5  *  (h_UOld(centerX-1,centerY  ,IB) + 
				   h_UOld(centerX  ,centerY  ,IB) );
	      
	      Ez[di][dj] = u*B-v*A;
	    }


	  // 4. perform trace reconstruction
	  //
	  // Compute reconstructed states at left interface along X 
	  // in current cell
	  xPos = ::gParams.xMin + dx/2 + (i-1-ghostWidth)*dx;

	  // (i-1,j)
	  bfNb2[0] =  h_UOld(i-1,j  ,IA);
	  bfNb2[1] =  h_UOld(i  ,j  ,IA);
	  bfNb2[2] =  h_UOld(i-1,j  ,IB);
	  bfNb2[3] =  h_UOld(i-1,j+1,IB);

	  // left interface : left state
	  trace_unsplit_mhd_2d_face(qLoc, dq, bfNb2, dAB, Ez,
				    dtdx, dtdy, xPos, FACE_XMAX,
				    qleft_x);	    

	  // qEdge_RB
	  trace_unsplit_mhd_2d_face(qLoc, dq, bfNb2, dAB, Ez,
				    dtdx, dtdy, xPos, EDGE_RB,
				    qEdge_RB);

	  ////////////////
	  // compute LEFT state for riemann problem (i,j-1)
	  ////////////////

	  // get primitive variables state vector
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	    qLoc[iVar]          = h_Q(i  ,j-1,iVar);
	    qNeighbors[0][iVar] = h_Q(i+1,j-1,iVar);
	    qNeighbors[1][iVar] = h_Q(i-1,j-1,iVar);
	    qNeighbors[2][iVar] = h_Q(i  ,j  ,iVar);
	    qNeighbors[3][iVar] = h_Q(i  ,j-2,iVar);
	  
	  } // end for iVar

	  // 1. compute hydro slopes
	  // compute slopes in left neighbor along X
	  slope_unsplit_hydro_2d_simple(qLoc, 
					qNeighbors[0],
					qNeighbors[1],
					qNeighbors[2],
					qNeighbors[3],
					dq);
	  
	  // 2. compute mag slopes (i,j-1)
	  bfNb[0] =  h_UOld(i  ,j-1,IA);
	  bfNb[1] =  h_UOld(i  ,j  ,IA);
	  bfNb[2] =  h_UOld(i  ,j-2,IA);
	  bfNb[3] =  h_UOld(i  ,j-1,IB);
	  bfNb[4] =  h_UOld(i+1,j-1,IB);
	  bfNb[5] =  h_UOld(i-1,j-1,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[0] = dbf[IY][IX];
	  dAB[1] = dbf[IX][IY];

	  // (i+1,j-1)
	  bfNb[0] =  h_UOld(i+1,j-1,IA);
	  bfNb[1] =  h_UOld(i+1,j  ,IA);
	  bfNb[2] =  h_UOld(i+1,j-2,IA);
	  bfNb[3] =  h_UOld(i+1,j-1,IB);
	  bfNb[4] =  h_UOld(i+2,j-1,IB);
	  bfNb[5] =  h_UOld(i  ,j-1,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[2] = dbf[IY][IX];

	  // (i,j)
	  bfNb[0] =  h_UOld(i  ,j  ,IA);
	  bfNb[1] =  h_UOld(i  ,j+1,IA);
	  bfNb[2] =  h_UOld(i  ,j-1,IA);
	  bfNb[3] =  h_UOld(i  ,j  ,IB);
	  bfNb[4] =  h_UOld(i+1,j  ,IB);
	  bfNb[5] =  h_UOld(i-1,j  ,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[3] = dbf[IX][IY];

	  // 3. compute Ez
	  for (int di=0; di<2; di++)
	    for (int dj=0; dj<2; dj++) {
      
	      int centerX = i  +di;
	      int centerY = j-1+dj;
	      real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,IU) + 
				   h_Q(centerX-1,centerY  ,IU) + 
				   h_Q(centerX  ,centerY-1,IU) + 
				   h_Q(centerX  ,centerY  ,IU) ); 
	      
	      real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,IV) +
				   h_Q(centerX-1,centerY  ,IV) +
				   h_Q(centerX  ,centerY-1,IV) + 
				   h_Q(centerX  ,centerY  ,IV) );
	      
	      real_t A  = 0.5  *  (h_UOld(centerX  ,centerY-1,IA) + 
				   h_UOld(centerX  ,centerY  ,IA) );
	      
	      real_t B  = 0.5  *  (h_UOld(centerX-1,centerY  ,IB) + 
				   h_UOld(centerX  ,centerY  ,IB) );
	      
	      Ez[di][dj] = u*B-v*A;
	    }


	  // 4. perform trace reconstruction
	  //
	  // Compute reconstructed states at left interface along X 
	  // in current cell
	  xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

	  // (i,j-1)
	  bfNb2[0] =  h_UOld(i  ,j-1,IA);
	  bfNb2[1] =  h_UOld(i+1,j-1,IA);
	  bfNb2[2] =  h_UOld(i  ,j-1,IB);
	  bfNb2[3] =  h_UOld(i  ,j  ,IB);

	  // left interface : left state
	  trace_unsplit_mhd_2d_face(qLoc, dq, bfNb2, dAB, Ez,
				    dtdx, dtdy, xPos, FACE_YMAX,
				    qleft_y);

	  // swap qleft_y
	  swap_v(qleft_y[IU], qleft_y[IV]);
	  swap_v(qleft_y[IA], qleft_y[IB]);


	  // qEdge_LT
	  trace_unsplit_mhd_2d_face(qLoc, dq, bfNb2, dAB, Ez,
				    dtdx, dtdy, xPos, EDGE_LT,
				    qEdge_LT);

	  ////////////////
	  // compute reconstructed states at (i-1,j-1)
	  ////////////////

	  // get primitive variables state vector
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	    qLoc[iVar]          = h_Q(i-1,j-1,iVar);
	    qNeighbors[0][iVar] = h_Q(i  ,j-1,iVar);
	    qNeighbors[1][iVar] = h_Q(i-2,j-1,iVar);
	    qNeighbors[2][iVar] = h_Q(i-1,j  ,iVar);
	    qNeighbors[3][iVar] = h_Q(i-1,j-2,iVar);
	  
	  } // end for iVar

	  // 1. compute hydro slopes
	  // compute slopes in left neighbor along X
	  slope_unsplit_hydro_2d_simple(qLoc, 
					qNeighbors[0],
					qNeighbors[1],
					qNeighbors[2],
					qNeighbors[3],
					dq);
	  
	  // 2. compute mag slopes (i-1,j-1)
	  bfNb[0] =  h_UOld(i-1,j-1,IA);
	  bfNb[1] =  h_UOld(i-1,j  ,IA);
	  bfNb[2] =  h_UOld(i-1,j-2,IA);
	  bfNb[3] =  h_UOld(i-1,j-1,IB);
	  bfNb[4] =  h_UOld(i  ,j-1,IB);
	  bfNb[5] =  h_UOld(i-2,j-1,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[0] = dbf[IY][IX];
	  dAB[1] = dbf[IX][IY];
	  
	  // (i  ,j-1)
	  bfNb[0] =  h_UOld(i  ,j-1,IA);
	  bfNb[1] =  h_UOld(i  ,j  ,IA);
	  bfNb[2] =  h_UOld(i  ,j-2,IA);
	  bfNb[3] =  h_UOld(i  ,j-1,IB);
	  bfNb[4] =  h_UOld(i+1,j-1,IB);
	  bfNb[5] =  h_UOld(i-1,j-1,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[2] = dbf[IY][IX];

	  // (i-1,j  )
	  bfNb[0] =  h_UOld(i-1,j  ,IA);
	  bfNb[1] =  h_UOld(i-1,j+1,IA);
	  bfNb[2] =  h_UOld(i-1,j-1,IA);
	  bfNb[3] =  h_UOld(i-1,j  ,IB);
	  bfNb[4] =  h_UOld(i  ,j  ,IB);
	  bfNb[5] =  h_UOld(i-2,j  ,IB);
	  slope_unsplit_mhd_2d(bfNb, dbf);
	  dAB[3] = dbf[IX][IY];

	  // 3. compute Ez
	  for (int di=0; di<2; di++)
	    for (int dj=0; dj<2; dj++) {
      
	      int centerX = i-1+di;
	      int centerY = j-1+dj;
	      real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,IU) + 
				   h_Q(centerX-1,centerY  ,IU) + 
				   h_Q(centerX  ,centerY-1,IU) + 
				   h_Q(centerX  ,centerY  ,IU) ); 
	      
	      real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,IV) +
				   h_Q(centerX-1,centerY  ,IV) +
				   h_Q(centerX  ,centerY-1,IV) + 
				   h_Q(centerX  ,centerY  ,IV) );
	      
	      real_t A  = 0.5  *  (h_UOld(centerX  ,centerY-1,IA) + 
				   h_UOld(centerX  ,centerY  ,IA) );
	      
	      real_t B  = 0.5  *  (h_UOld(centerX-1,centerY  ,IB) + 
				   h_UOld(centerX  ,centerY  ,IB) );
	      
	      Ez[di][dj] = u*B-v*A;
	    }

	  // 4. perform trace reconstruction
	  //
	  // Compute reconstructed states at left interface along X 
	  // in current cell
	  xPos = ::gParams.xMin + dx/2 + (i-1-ghostWidth)*dx;

	  // (i-1,j-1)
	  bfNb2[0] =  h_UOld(i-1,j-1,IA);
	  bfNb2[1] =  h_UOld(i  ,j-1,IA);
	  bfNb2[2] =  h_UOld(i-1,j-1,IB);
	  bfNb2[3] =  h_UOld(i-1,j  ,IB);

	  // qEdge_RT
	  trace_unsplit_mhd_2d_face(qLoc, dq, bfNb2, dAB, Ez,
				    dtdx, dtdy, xPos, EDGE_RT,
				    qEdge_RT);


	  // Now we are ready for solving Riemann problem at left interface
	  if (gravityEnabled) { 
	    // we need to modify input to flux computation with
	    // gravity predictor (half time step)
	    qright_x[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qright_x[IV] += HALF_F * dt * h_gravity(i,j,IY);	    

	    qright_y[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qright_y[IV] += HALF_F * dt * h_gravity(i,j,IY);	    

	    // use i-1,j here ?
	    qleft_x[IU]  += HALF_F * dt * h_gravity(i,j,IX);
	    qleft_x[IV]  += HALF_F * dt * h_gravity(i,j,IY);

	    // use i,j-1 here ?
	    qleft_y[IU]  += HALF_F * dt * h_gravity(i,j,IX);
	    qleft_y[IV]  += HALF_F * dt * h_gravity(i,j,IY);

	    qEdge_RT[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qEdge_RT[IV] += HALF_F * dt * h_gravity(i,j,IY);

	    qEdge_RB[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qEdge_RB[IV] += HALF_F * dt * h_gravity(i,j,IY);

	    qEdge_LT[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qEdge_LT[IV] += HALF_F * dt * h_gravity(i,j,IY);

	    qEdge_LB[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qEdge_LB[IV] += HALF_F * dt * h_gravity(i,j,IY);
	    
	  }
	  
	  // compute hydro flux_x
	  riemann_mhd(qleft_x,qright_x,flux_x);

	  // compute hydro flux_x
	  riemann_mhd(qleft_y,qright_y,flux_y);

	  /*
	   * update mhd array with hydro fluxes
	   */
	  if ( i > ghostWidth       and 
	       j < jsize-ghostWidth) {
	    h_UNew(i-1,j  ,ID) -= flux_x[ID]*dtdx;
	    h_UNew(i-1,j  ,IP) -= flux_x[IP]*dtdx;
	    h_UNew(i-1,j  ,IU) -= flux_x[IU]*dtdx;
	    h_UNew(i-1,j  ,IV) -= flux_x[IV]*dtdx;
	    h_UNew(i-1,j  ,IW) -= flux_x[IW]*dtdx;
	    h_UNew(i-1,j  ,IC) -= flux_x[IC]*dtdx;
	  }
	  
	  if ( i < isize-ghostWidth and 
	       j < jsize-ghostWidth) {
	    h_UNew(i  ,j  ,ID) += flux_x[ID]*dtdx;
	    h_UNew(i  ,j  ,IP) += flux_x[IP]*dtdx;
	    h_UNew(i  ,j  ,IU) += flux_x[IU]*dtdx;
	    h_UNew(i  ,j  ,IV) += flux_x[IV]*dtdx;
	    h_UNew(i  ,j  ,IW) += flux_x[IW]*dtdx;
	    h_UNew(i  ,j  ,IC) += flux_x[IC]*dtdx;
	  }
	  
	  if ( i < isize-ghostWidth and 
	       j > ghostWidth) {
	    h_UNew(i  ,j-1,ID) -= flux_y[ID]*dtdy;
	    h_UNew(i  ,j-1,IP) -= flux_y[IP]*dtdy;
	    h_UNew(i  ,j-1,IU) -= flux_y[IV]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j-1,IV) -= flux_y[IU]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j-1,IW) -= flux_y[IW]*dtdy;
	    h_UNew(i  ,j-1,IC) -= flux_y[IC]*dtdy;
	  }
	  
	  if ( i < isize-ghostWidth and 
	       j < jsize-ghostWidth) {
	    h_UNew(i  ,j  ,ID) += flux_y[ID]*dtdy;
	    h_UNew(i  ,j  ,IP) += flux_y[IP]*dtdy;
	    h_UNew(i  ,j  ,IU) += flux_y[IV]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j  ,IV) += flux_y[IU]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j  ,IW) += flux_y[IW]*dtdy;
	    h_UNew(i  ,j  ,IC) += flux_y[IC]*dtdy;
	  }
	  
	  
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
	    qEdge_emfZ[IRT][iVar] = qEdge_RT[iVar]; 
	    qEdge_emfZ[IRB][iVar] = qEdge_RB[iVar]; 
	    qEdge_emfZ[ILT][iVar] = qEdge_LT[iVar]; 
	    qEdge_emfZ[ILB][iVar] = qEdge_LB[iVar]; 
	  }

	  // actually compute emfZ
	  real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ);
	  h_emf(i,j,I_EMFZ) = emfZ;

	} // end for j
      } // end for i
      
      /*
       * magnetic field update (constraint transport)
       */
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	  // left-face B-field
	  h_UNew(i  ,j  ,IA) += ( h_emf(i  ,j+1, I_EMFZ) - h_emf(i,j, I_EMFZ) )*dtdy;
	  h_UNew(i  ,j  ,IB) -= ( h_emf(i+1,j  , I_EMFZ) - h_emf(i,j, I_EMFZ) )*dtdx;	  
	} // end for i
      } // end for j
      
      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(h_UNew, h_UOld, dt);
      }
 

      ////////////////////////////////////////////////     
      ////////////////////////////////////////////////     
    } else { // THREE_D - implementation version 0
      ////////////////////////////////////////////////
      ////////////////////////////////////////////////     
    
      // Omega0
      real_t &Omega0 = ::gParams.Omega0;

      for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	    
	    // primitive variables (local array)
	    real_t qLoc[NVAR_MHD];
	    real_t qNeighbors[2*THREE_D][NVAR_MHD];
	    
	    // slopes
	    real_t dq[THREE_D][NVAR_MHD];
	    real_t dbf[THREE_D][THREE_D];
	    real_t bfNb[THREE_D*5];
	    real_t bfNb2[THREE_D*2];
	    real_t dABC[THREE_D*4];

	    // reconstructed state on cell faces
	    // aka riemann solver input
	    real_t qleft_x[NVAR_MHD];
	    real_t qleft_y[NVAR_MHD];
	    real_t qleft_z[NVAR_MHD];
	    real_t qright_x[NVAR_MHD];
	    real_t qright_y[NVAR_MHD];
	    real_t qright_z[NVAR_MHD];
	    
	    // riemann solver output
	    real_t flux_x[NVAR_MHD];
	    real_t flux_y[NVAR_MHD];
	    real_t flux_z[NVAR_MHD];
	    
	    // emf
	    real_t Exyz[THREE_D][2][2];
	    real_t qEdge_emfX[4][NVAR_MHD];
	    real_t qEdge_emfY[4][NVAR_MHD];
	    real_t qEdge_emfZ[4][NVAR_MHD];
	    
	    // other variables
	    real_t xPos;

	    int ic=i;
	    int jc=j;
	    int kc=k;

	    ///////////////////////////////////////////////
	    // compute reconstructed states at (i,j,k)
	    ///////////////////////////////////////////////
	    ic=i;
	    jc=j;
	    kc=k;
	    xPos = ::gParams.xMin + dx/2 + (ic-ghostWidth)*dx;

	    // get primitive variables state vector
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLoc[iVar]          = h_Q(ic  ,jc  ,kc  ,iVar);
	      qNeighbors[0][iVar] = h_Q(ic+1,jc  ,kc  ,iVar);
	      qNeighbors[1][iVar] = h_Q(ic-1,jc  ,kc  ,iVar);
	      qNeighbors[2][iVar] = h_Q(ic  ,jc+1,kc  ,iVar);
	      qNeighbors[3][iVar] = h_Q(ic  ,jc-1,kc  ,iVar);
	      qNeighbors[4][iVar] = h_Q(ic  ,jc  ,kc+1,iVar);
	      qNeighbors[5][iVar] = h_Q(ic  ,jc  ,kc-1,iVar);
	      
	    } // end for iVar
	    
	    // 1. compute hydro slopes
	    // compute slopes in left neighbor along X
	    slope_unsplit_hydro_3d(qLoc, 
				   qNeighbors[0],
				   qNeighbors[1],
				   qNeighbors[2],
				   qNeighbors[3],
				   qNeighbors[4],
				   qNeighbors[5],
				   dq);
	      
	    // 2. compute mag slopes @(ic,jc,kc)
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[0] = dbf[IY][IX];
	    dABC[1] = dbf[IZ][IX];
	    dABC[2] = dbf[IX][IY];
	    dABC[3] = dbf[IZ][IY];
	    dABC[4] = dbf[IX][IZ];
	    dABC[5] = dbf[IY][IZ];

	    // change neighbors to ic+1, jc, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic+1,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic+1,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic+1,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic+1,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+2,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic+1,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+2,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic+1,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[6] = dbf[IY][IX];
	    dABC[7] = dbf[IZ][IX];
	    
	    // change neighbors to ic, jc+1, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+2,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc+1,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc+1,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc+1,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc+1,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc+1,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc+1,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+2,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc  ,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[8] = dbf[IX][IY];
	    dABC[9] = dbf[IZ][IY];

	    // change neighbors to ic, jc, kc+1 and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc+1,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+2,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc  ,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc+1,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+2,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc  ,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc+1,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc+1,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc+1,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc+1,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc+1,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[10] = dbf[IX][IZ];
	    dABC[11] = dbf[IY][IZ];

	    // 3. compute Ex,Ey,Ez (electric field components)
	    for (int dj=0; dj<2; dj++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic;
		int centerY = jc+dj;
		int centerZ = kc+dk;
	    
		real_t v = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IV) +
				   h_Q(centerX,centerY-1,centerZ  ,IV) +
				   h_Q(centerX,centerY  ,centerZ-1,IV) +
				   h_Q(centerX,centerY  ,centerZ  ,IV) );

		real_t w = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IW) +
				   h_Q(centerX,centerY-1,centerZ  ,IW) +
				   h_Q(centerX,centerY  ,centerZ-1,IW) +
				   h_Q(centerX,centerY  ,centerZ  ,IW) );

		real_t B = 0.5 * (h_UOld(centerX,centerY  ,centerZ-1,IY) +
				  h_UOld(centerX,centerY  ,centerZ  ,IY) );

		real_t C = 0.5 * (h_UOld(centerX,centerY-1,centerZ  ,IZ) +
				  h_UOld(centerX,centerY  ,centerZ  ,IZ) );
	    
		Exyz[IX][dj][dk] = v*C-w*B;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * xPos;
		  Exyz[IX][dj][dk] += shear*C;
		}
		
	      } // end for dk
	    } // end for dj
  
	    for (int di=0; di<2; di++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic+di;
		int centerY = jc;
		int centerZ = kc+dk;
	    
		real_t u = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IU) + 
				   h_Q(centerX-1,centerY,centerZ  ,IU) + 
				   h_Q(centerX  ,centerY,centerZ-1,IU) + 
				   h_Q(centerX  ,centerY,centerZ  ,IU) );
  
		real_t w = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IW) +
				   h_Q(centerX-1,centerY,centerZ  ,IW) +
				   h_Q(centerX  ,centerY,centerZ-1,IW) +
				   h_Q(centerX  ,centerY,centerZ  ,IW) );
		
		real_t A = 0.5 * (h_UOld(centerX  ,centerY,centerZ-1,IX) + 
				  h_UOld(centerX  ,centerY,centerZ  ,IX) );

		real_t C = 0.5 * (h_UOld(centerX-1,centerY,centerZ  ,IZ) +
				  h_UOld(centerX  ,centerY,centerZ  ,IZ) );

		Exyz[IY][di][dk] = w*A-u*C;
		
	      } // end for dk
	    } // end for di
	    
	    for (int di=0; di<2; di++) {
	      for (int dj=0; dj<2; dj++) {

		int centerX = ic+di;
		int centerY = jc+dj;
		int centerZ = kc;

		real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IU) + 
				     h_Q(centerX-1,centerY  ,centerZ,IU) + 
				     h_Q(centerX  ,centerY-1,centerZ,IU) + 
				     h_Q(centerX  ,centerY  ,centerZ,IU) ); 
		
		real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IV) +
				     h_Q(centerX-1,centerY  ,centerZ,IV) +
				     h_Q(centerX  ,centerY-1,centerZ,IV) + 
				     h_Q(centerX  ,centerY  ,centerZ,IV) );
		
		real_t A  = 0.5  * (h_UOld(centerX  ,centerY-1,centerZ,IA) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IA) );
		
		real_t B  = 0.5  * (h_UOld(centerX-1,centerY  ,centerZ,IB) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IB) );
	    
		Exyz[IZ][di][dj] = u*B-v*A;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * (xPos - dx/2);
		  Exyz[IZ][di][dj] -= shear*A;
		}
		
	      } // end for dj
	    } // end for di

	    // 4. perform trace reconstruction
	    //
	    // Compute reconstructed states at left interface along X 
	    // in current cell
	    
	    // (ic,jc,kc)
	    bfNb2[0] =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb2[1] =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb2[2] =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb2[3] =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb2[4] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb2[5] =  h_UOld(ic  ,jc  ,kc+1,IC);

	    // left interface : right state along x
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, FACE_XMIN,
				      qright_x);

	    // left interface : right state along y
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, FACE_YMIN,
				      qright_y);

	    // left interface : right state along z
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, FACE_ZMIN,
				      qright_z);

	    // swap qright_y
	    swap_v(qright_y[IU], qright_y[IV]);
	    swap_v(qright_y[IA], qright_y[IB]);

	    // swap qright_z
	    swap_v(qright_z[IU], qright_z[IW]);
	    swap_v(qright_z[IA], qright_z[IC]);


	    // EDGE_LB_Z
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_LB_Z,
				      qEdge_emfZ[ILB] );

	    // EDGE_LB_Y
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_LB_Y,
				      qEdge_emfY[ILB] );

	    // EDGE_LB_Y
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_LB_X,
				      qEdge_emfX[ILB] );

	    ///////////////////////////////////////////////
	    // compute reconstructed states at (i-1,j,k)
	    ///////////////////////////////////////////////
	    ic=i-1;
	    jc=j;
	    kc=k;
	    xPos = ::gParams.xMin + dx/2 + (ic-ghostWidth)*dx;
	    
	    // get primitive variables state vector
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLoc[iVar]          = h_Q(ic  ,jc  ,kc  ,iVar);
	      qNeighbors[0][iVar] = h_Q(ic+1,jc  ,kc  ,iVar);
	      qNeighbors[1][iVar] = h_Q(ic-1,jc  ,kc  ,iVar);
	      qNeighbors[2][iVar] = h_Q(ic  ,jc+1,kc  ,iVar);
	      qNeighbors[3][iVar] = h_Q(ic  ,jc-1,kc  ,iVar);
	      qNeighbors[4][iVar] = h_Q(ic  ,jc  ,kc+1,iVar);
	      qNeighbors[5][iVar] = h_Q(ic  ,jc  ,kc-1,iVar);
	      
	    } // end for iVar
	    
	    // 1. compute hydro slopes
	    // compute slopes in left neighbor along X
	    slope_unsplit_hydro_3d(qLoc, 
				   qNeighbors[0],
				   qNeighbors[1],
				   qNeighbors[2],
				   qNeighbors[3],
				   qNeighbors[4],
				   qNeighbors[5],
				   dq);
	      
	    // 2. compute mag slopes @(ic,jc,kc)
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[0] = dbf[IY][IX];
	    dABC[1] = dbf[IZ][IX];
	    dABC[2] = dbf[IX][IY];
	    dABC[3] = dbf[IZ][IY];
	    dABC[4] = dbf[IX][IZ];
	    dABC[5] = dbf[IY][IZ];

	    // change neighbors to ic+1, jc, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic+1,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic+1,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic+1,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic+1,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+2,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic+1,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+2,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic+1,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[6] = dbf[IY][IX];
	    dABC[7] = dbf[IZ][IX];
	    
	    // change neighbors to ic, jc+1, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+2,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc+1,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc+1,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc+1,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc+1,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc+1,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc+1,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+2,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc  ,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[8] = dbf[IX][IY];
	    dABC[9] = dbf[IZ][IY];

	    // change neighbors to ic, jc, kc+1 and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc+1,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+2,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc  ,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc+1,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+2,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc  ,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc+1,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc+1,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc+1,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc+1,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc+1,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[10] = dbf[IX][IZ];
	    dABC[11] = dbf[IY][IZ];

	    // 3. compute Ex,Ey,Ez (electric field components)
	    for (int dj=0; dj<2; dj++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic;
		int centerY = jc+dj;
		int centerZ = kc+dk;
	    
		real_t v = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IV) +
				   h_Q(centerX,centerY-1,centerZ  ,IV) +
				   h_Q(centerX,centerY  ,centerZ-1,IV) +
				   h_Q(centerX,centerY  ,centerZ  ,IV) );

		real_t w = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IW) +
				   h_Q(centerX,centerY-1,centerZ  ,IW) +
				   h_Q(centerX,centerY  ,centerZ-1,IW) +
				   h_Q(centerX,centerY  ,centerZ  ,IW) );

		real_t B = 0.5 * (h_UOld(centerX,centerY  ,centerZ-1,IY) +
				  h_UOld(centerX,centerY  ,centerZ  ,IY) );

		real_t C = 0.5 * (h_UOld(centerX,centerY-1,centerZ  ,IZ) +
				  h_UOld(centerX,centerY  ,centerZ  ,IZ) );
	    
		Exyz[IX][dj][dk] = v*C-w*B;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * xPos;
		  Exyz[IX][dj][dk] += shear*C;
		}
		
	      } // end for dk
	    } // end for dj
  
	    for (int di=0; di<2; di++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic+di;
		int centerY = jc;
		int centerZ = kc+dk;
	    
		real_t u = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IU) + 
				   h_Q(centerX-1,centerY,centerZ  ,IU) + 
				   h_Q(centerX  ,centerY,centerZ-1,IU) + 
				   h_Q(centerX  ,centerY,centerZ  ,IU) );
  
		real_t w = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IW) +
				   h_Q(centerX-1,centerY,centerZ  ,IW) +
				   h_Q(centerX  ,centerY,centerZ-1,IW) +
				   h_Q(centerX  ,centerY,centerZ  ,IW) );
		
		real_t A = 0.5 * (h_UOld(centerX  ,centerY,centerZ-1,IX) + 
				  h_UOld(centerX  ,centerY,centerZ  ,IX) );

		real_t C = 0.5 * (h_UOld(centerX-1,centerY,centerZ  ,IZ) +
				  h_UOld(centerX  ,centerY,centerZ  ,IZ) );

		Exyz[IY][di][dk] = w*A-u*C;
		
	      } // end for dk
	    } // end for di
	    
	    for (int di=0; di<2; di++) {
	      for (int dj=0; dj<2; dj++) {

		int centerX = ic+di;
		int centerY = jc+dj;
		int centerZ = kc;

		real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IU) + 
				     h_Q(centerX-1,centerY  ,centerZ,IU) + 
				     h_Q(centerX  ,centerY-1,centerZ,IU) + 
				     h_Q(centerX  ,centerY  ,centerZ,IU) ); 
		
		real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IV) +
				     h_Q(centerX-1,centerY  ,centerZ,IV) +
				     h_Q(centerX  ,centerY-1,centerZ,IV) + 
				     h_Q(centerX  ,centerY  ,centerZ,IV) );
		
		real_t A  = 0.5  * (h_UOld(centerX  ,centerY-1,centerZ,IA) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IA) );
		
		real_t B  = 0.5  * (h_UOld(centerX-1,centerY  ,centerZ,IB) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IB) );
	    
		Exyz[IZ][di][dj] = u*B-v*A;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * (xPos - dx/2);
		  Exyz[IZ][di][dj] -= shear*A;
		}
		
	      } // end for dj
	    } // end for di

	    // 4. perform trace reconstruction
	    //
	    // Compute reconstructed states at left interface along X 
	    // in current cell
	    
	    // (ic,jc,kc)
	    bfNb2[0] =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb2[1] =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb2[2] =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb2[3] =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb2[4] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb2[5] =  h_UOld(ic  ,jc  ,kc+1,IC);

	    // left interface : left state along x
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, FACE_XMAX,
				      qleft_x);

	    // compute hydro flux_x
	    riemann_mhd(qleft_x,qright_x,flux_x);
	    	      
	    // EDGE_RB_Z
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_RB_Z,
				      qEdge_emfZ[IRB] );
	    
	    // EDGE_RB_Y (swapped with LT)
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_RB_Y,
				      qEdge_emfY[ILT] );
	    

	    ///////////////////////////////////////////////
	    // compute reconstructed states at (i,j-1,k)
	    ///////////////////////////////////////////////
	    ic=i;
	    jc=j-1;
	    kc=k;
	    xPos = ::gParams.xMin + dx/2 + (ic-ghostWidth)*dx;
	    
	    // get primitive variables state vector
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLoc[iVar]          = h_Q(ic  ,jc  ,kc  ,iVar);
	      qNeighbors[0][iVar] = h_Q(ic+1,jc  ,kc  ,iVar);
	      qNeighbors[1][iVar] = h_Q(ic-1,jc  ,kc  ,iVar);
	      qNeighbors[2][iVar] = h_Q(ic  ,jc+1,kc  ,iVar);
	      qNeighbors[3][iVar] = h_Q(ic  ,jc-1,kc  ,iVar);
	      qNeighbors[4][iVar] = h_Q(ic  ,jc  ,kc+1,iVar);
	      qNeighbors[5][iVar] = h_Q(ic  ,jc  ,kc-1,iVar);
	      
	    } // end for iVar
	    
	    // 1. compute hydro slopes
	    // compute slopes in left neighbor along X
	    slope_unsplit_hydro_3d(qLoc, 
				   qNeighbors[0],
				   qNeighbors[1],
				   qNeighbors[2],
				   qNeighbors[3],
				   qNeighbors[4],
				   qNeighbors[5],
				   dq);
	      
	    // 2. compute mag slopes @(ic,jc,kc)
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[0] = dbf[IY][IX];
	    dABC[1] = dbf[IZ][IX];
	    dABC[2] = dbf[IX][IY];
	    dABC[3] = dbf[IZ][IY];
	    dABC[4] = dbf[IX][IZ];
	    dABC[5] = dbf[IY][IZ];

	    // change neighbors to ic+1, jc, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic+1,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic+1,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic+1,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic+1,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+2,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic+1,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+2,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic+1,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[6] = dbf[IY][IX];
	    dABC[7] = dbf[IZ][IX];
	    
	    // change neighbors to ic, jc+1, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+2,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc+1,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc+1,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc+1,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc+1,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc+1,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc+1,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+2,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc  ,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[8] = dbf[IX][IY];
	    dABC[9] = dbf[IZ][IY];

	    // change neighbors to ic, jc, kc+1 and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc+1,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+2,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc  ,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc+1,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+2,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc  ,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc+1,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc+1,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc+1,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc+1,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc+1,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[10] = dbf[IX][IZ];
	    dABC[11] = dbf[IY][IZ];

	    // 3. compute Ex,Ey,Ez (electric field components)
	    for (int dj=0; dj<2; dj++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic;
		int centerY = jc+dj;
		int centerZ = kc+dk;
	    
		real_t v = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IV) +
				   h_Q(centerX,centerY-1,centerZ  ,IV) +
				   h_Q(centerX,centerY  ,centerZ-1,IV) +
				   h_Q(centerX,centerY  ,centerZ  ,IV) );

		real_t w = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IW) +
				   h_Q(centerX,centerY-1,centerZ  ,IW) +
				   h_Q(centerX,centerY  ,centerZ-1,IW) +
				   h_Q(centerX,centerY  ,centerZ  ,IW) );

		real_t B = 0.5 * (h_UOld(centerX,centerY  ,centerZ-1,IY) +
				  h_UOld(centerX,centerY  ,centerZ  ,IY) );

		real_t C = 0.5 * (h_UOld(centerX,centerY-1,centerZ  ,IZ) +
				  h_UOld(centerX,centerY  ,centerZ  ,IZ) );
	    
		Exyz[IX][dj][dk] = v*C-w*B;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * xPos;
		  Exyz[IX][dj][dk] += shear*C;
		}
		
	      } // end for dk
	    } // end for dj
  
	    for (int di=0; di<2; di++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic+di;
		int centerY = jc;
		int centerZ = kc+dk;
	    
		real_t u = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IU) + 
				   h_Q(centerX-1,centerY,centerZ  ,IU) + 
				   h_Q(centerX  ,centerY,centerZ-1,IU) + 
				   h_Q(centerX  ,centerY,centerZ  ,IU) );
  
		real_t w = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IW) +
				   h_Q(centerX-1,centerY,centerZ  ,IW) +
				   h_Q(centerX  ,centerY,centerZ-1,IW) +
				   h_Q(centerX  ,centerY,centerZ  ,IW) );
		
		real_t A = 0.5 * (h_UOld(centerX  ,centerY,centerZ-1,IX) + 
				  h_UOld(centerX  ,centerY,centerZ  ,IX) );

		real_t C = 0.5 * (h_UOld(centerX-1,centerY,centerZ  ,IZ) +
				  h_UOld(centerX  ,centerY,centerZ  ,IZ) );

		Exyz[IY][di][dk] = w*A-u*C;
		
	      } // end for dk
	    } // end for di
	    
	    for (int di=0; di<2; di++) {
	      for (int dj=0; dj<2; dj++) {

		int centerX = ic+di;
		int centerY = jc+dj;
		int centerZ = kc;

		real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IU) + 
				     h_Q(centerX-1,centerY  ,centerZ,IU) + 
				     h_Q(centerX  ,centerY-1,centerZ,IU) + 
				     h_Q(centerX  ,centerY  ,centerZ,IU) ); 
		
		real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IV) +
				     h_Q(centerX-1,centerY  ,centerZ,IV) +
				     h_Q(centerX  ,centerY-1,centerZ,IV) + 
				     h_Q(centerX  ,centerY  ,centerZ,IV) );
		
		real_t A  = 0.5  * (h_UOld(centerX  ,centerY-1,centerZ,IA) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IA) );
		
		real_t B  = 0.5  * (h_UOld(centerX-1,centerY  ,centerZ,IB) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IB) );
	    
		Exyz[IZ][di][dj] = u*B-v*A;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * (xPos - dx/2);
		  Exyz[IZ][di][dj] -= shear*A;
		}
		
	      } // end for dj
	    } // end for di

	    // 4. perform trace reconstruction
	    //
	    // Compute reconstructed states at left interface along X 
	    // in current cell
	    
	    // (ic,jc,kc)
	    bfNb2[0] =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb2[1] =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb2[2] =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb2[3] =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb2[4] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb2[5] =  h_UOld(ic  ,jc  ,kc+1,IC);

	    // left interface : left state along y
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, FACE_YMAX,
				      qleft_y);

	    // swap qleft_y
	    swap_v(qleft_y[IU], qleft_y[IV]);
	    swap_v(qleft_y[IA], qleft_y[IB]);

	    // compute hydro flux_y
	    riemann_mhd(qleft_y,qright_y,flux_y);
	      

	    // EDGE_LT_Z
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_LT_Z,
				      qEdge_emfZ[ILT] );

	    // EDGE_RB_X
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_RB_X,
				      qEdge_emfX[IRB] );

	    ///////////////////////////////////////////////
	    // compute reconstructed states at (i,j,k-1)
	    ///////////////////////////////////////////////
	    ic=i;
	    jc=j;
	    kc=k-1;
	    xPos = ::gParams.xMin + dx/2 + (ic-ghostWidth)*dx;
	    
	    // get primitive variables state vector
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLoc[iVar]          = h_Q(ic  ,jc  ,kc  ,iVar);
	      qNeighbors[0][iVar] = h_Q(ic+1,jc  ,kc  ,iVar);
	      qNeighbors[1][iVar] = h_Q(ic-1,jc  ,kc  ,iVar);
	      qNeighbors[2][iVar] = h_Q(ic  ,jc+1,kc  ,iVar);
	      qNeighbors[3][iVar] = h_Q(ic  ,jc-1,kc  ,iVar);
	      qNeighbors[4][iVar] = h_Q(ic  ,jc  ,kc+1,iVar);
	      qNeighbors[5][iVar] = h_Q(ic  ,jc  ,kc-1,iVar);
	      
	    } // end for iVar
	    
	    // 1. compute hydro slopes
	    // compute slopes in left neighbor along X
	    slope_unsplit_hydro_3d(qLoc, 
				   qNeighbors[0],
				   qNeighbors[1],
				   qNeighbors[2],
				   qNeighbors[3],
				   qNeighbors[4],
				   qNeighbors[5],
				   dq);
	      
	    // 2. compute mag slopes @(ic,jc,kc)
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[0] = dbf[IY][IX];
	    dABC[1] = dbf[IZ][IX];
	    dABC[2] = dbf[IX][IY];
	    dABC[3] = dbf[IZ][IY];
	    dABC[4] = dbf[IX][IZ];
	    dABC[5] = dbf[IY][IZ];

	    // change neighbors to ic+1, jc, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic+1,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic+1,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic+1,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic+1,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+2,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic+1,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+2,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic+1,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[6] = dbf[IY][IX];
	    dABC[7] = dbf[IZ][IX];
	    
	    // change neighbors to ic, jc+1, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+2,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc+1,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc+1,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc+1,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc+1,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc+1,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc+1,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+2,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc  ,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[8] = dbf[IX][IY];
	    dABC[9] = dbf[IZ][IY];

	    // change neighbors to ic, jc, kc+1 and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc+1,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+2,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc  ,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc+1,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+2,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc  ,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc+1,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc+1,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc+1,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc+1,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc+1,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[10] = dbf[IX][IZ];
	    dABC[11] = dbf[IY][IZ];

	    // 3. compute Ex,Ey,Ez (electric field components)
	    for (int dj=0; dj<2; dj++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic;
		int centerY = jc+dj;
		int centerZ = kc+dk;
	    
		real_t v = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IV) +
				   h_Q(centerX,centerY-1,centerZ  ,IV) +
				   h_Q(centerX,centerY  ,centerZ-1,IV) +
				   h_Q(centerX,centerY  ,centerZ  ,IV) );

		real_t w = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IW) +
				   h_Q(centerX,centerY-1,centerZ  ,IW) +
				   h_Q(centerX,centerY  ,centerZ-1,IW) +
				   h_Q(centerX,centerY  ,centerZ  ,IW) );

		real_t B = 0.5 * (h_UOld(centerX,centerY  ,centerZ-1,IY) +
				  h_UOld(centerX,centerY  ,centerZ  ,IY) );

		real_t C = 0.5 * (h_UOld(centerX,centerY-1,centerZ  ,IZ) +
				  h_UOld(centerX,centerY  ,centerZ  ,IZ) );
	    
		Exyz[IX][dj][dk] = v*C-w*B;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * xPos;
		  Exyz[IX][dj][dk] += shear*C;
		}
		
	      } // end for dk
	    } // end for dj
  
	    for (int di=0; di<2; di++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic+di;
		int centerY = jc;
		int centerZ = kc+dk;
	    
		real_t u = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IU) + 
				   h_Q(centerX-1,centerY,centerZ  ,IU) + 
				   h_Q(centerX  ,centerY,centerZ-1,IU) + 
				   h_Q(centerX  ,centerY,centerZ  ,IU) );
  
		real_t w = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IW) +
				   h_Q(centerX-1,centerY,centerZ  ,IW) +
				   h_Q(centerX  ,centerY,centerZ-1,IW) +
				   h_Q(centerX  ,centerY,centerZ  ,IW) );
		
		real_t A = 0.5 * (h_UOld(centerX  ,centerY,centerZ-1,IX) + 
				  h_UOld(centerX  ,centerY,centerZ  ,IX) );

		real_t C = 0.5 * (h_UOld(centerX-1,centerY,centerZ  ,IZ) +
				  h_UOld(centerX  ,centerY,centerZ  ,IZ) );

		Exyz[IY][di][dk] = w*A-u*C;
		
	      } // end for dk
	    } // end for di
	    
	    for (int di=0; di<2; di++) {
	      for (int dj=0; dj<2; dj++) {

		int centerX = ic+di;
		int centerY = jc+dj;
		int centerZ = kc;

		real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IU) + 
				     h_Q(centerX-1,centerY  ,centerZ,IU) + 
				     h_Q(centerX  ,centerY-1,centerZ,IU) + 
				     h_Q(centerX  ,centerY  ,centerZ,IU) ); 
		
		real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IV) +
				     h_Q(centerX-1,centerY  ,centerZ,IV) +
				     h_Q(centerX  ,centerY-1,centerZ,IV) + 
				     h_Q(centerX  ,centerY  ,centerZ,IV) );
		
		real_t A  = 0.5  * (h_UOld(centerX  ,centerY-1,centerZ,IA) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IA) );
		
		real_t B  = 0.5  * (h_UOld(centerX-1,centerY  ,centerZ,IB) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IB) );
	    
		Exyz[IZ][di][dj] = u*B-v*A;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * (xPos - dx/2);
		  Exyz[IZ][di][dj] -= shear*A;
		}
		
	      } // end for dj
	    } // end for di

	    // 4. perform trace reconstruction
	    //
	    // Compute reconstructed states at left interface along X 
	    // in current cell
	    
	    // (ic,jc,kc)
	    bfNb2[0] =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb2[1] =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb2[2] =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb2[3] =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb2[4] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb2[5] =  h_UOld(ic  ,jc  ,kc+1,IC);

	    // left interface : left state along z
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, FACE_ZMAX,
				      qleft_z);

	    // swap qleft_z
	    swap_v(qleft_z[IU], qleft_z[IW]);
	    swap_v(qleft_z[IA], qleft_z[IC]);

	    // compute hydro flux_z
	    riemann_mhd(qleft_z,qright_z,flux_z);
	      
	    // EDGE_LT_Y (!swap)
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_LT_Y,
				      qEdge_emfY[IRB] );

	    // EDGE_LT_X
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_LT_X,
				      qEdge_emfX[ILT] );

	    /*
	     * update mhd array
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
	      
	    ///////////////////////////////////////////////
	    // compute reconstructed states at (i-1,j-1,k)
	    ///////////////////////////////////////////////
	    ic=i-1;
	    jc=j-1;
	    kc=k;
	    xPos = ::gParams.xMin + dx/2 + (ic-ghostWidth)*dx;

	    // get primitive variables state vector
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLoc[iVar]          = h_Q(ic  ,jc  ,kc  ,iVar);
	      qNeighbors[0][iVar] = h_Q(ic+1,jc  ,kc  ,iVar);
	      qNeighbors[1][iVar] = h_Q(ic-1,jc  ,kc  ,iVar);
	      qNeighbors[2][iVar] = h_Q(ic  ,jc+1,kc  ,iVar);
	      qNeighbors[3][iVar] = h_Q(ic  ,jc-1,kc  ,iVar);
	      qNeighbors[4][iVar] = h_Q(ic  ,jc  ,kc+1,iVar);
	      qNeighbors[5][iVar] = h_Q(ic  ,jc  ,kc-1,iVar);
	      
	    } // end for iVar
	    
	    // 1. compute hydro slopes
	    // compute slopes in left neighbor along X
	    slope_unsplit_hydro_3d(qLoc, 
				   qNeighbors[0],
				   qNeighbors[1],
				   qNeighbors[2],
				   qNeighbors[3],
				   qNeighbors[4],
				   qNeighbors[5],
				   dq);
	      
	    // 2. compute mag slopes @(ic,jc,kc)
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[0] = dbf[IY][IX];
	    dABC[1] = dbf[IZ][IX];
	    dABC[2] = dbf[IX][IY];
	    dABC[3] = dbf[IZ][IY];
	    dABC[4] = dbf[IX][IZ];
	    dABC[5] = dbf[IY][IZ];

	    // change neighbors to ic+1, jc, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic+1,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic+1,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic+1,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic+1,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+2,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic+1,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+2,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic+1,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[6] = dbf[IY][IX];
	    dABC[7] = dbf[IZ][IX];
	    
	    // change neighbors to ic, jc+1, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+2,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc+1,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc+1,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc+1,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc+1,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc+1,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc+1,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+2,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc  ,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[8] = dbf[IX][IY];
	    dABC[9] = dbf[IZ][IY];

	    // change neighbors to ic, jc, kc+1 and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc+1,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+2,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc  ,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc+1,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+2,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc  ,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc+1,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc+1,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc+1,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc+1,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc+1,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[10] = dbf[IX][IZ];
	    dABC[11] = dbf[IY][IZ];

	    // 3. compute Ex,Ey,Ez (electric field components)
	    for (int dj=0; dj<2; dj++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic;
		int centerY = jc+dj;
		int centerZ = kc+dk;
	    
		real_t v = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IV) +
				   h_Q(centerX,centerY-1,centerZ  ,IV) +
				   h_Q(centerX,centerY  ,centerZ-1,IV) +
				   h_Q(centerX,centerY  ,centerZ  ,IV) );

		real_t w = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IW) +
				   h_Q(centerX,centerY-1,centerZ  ,IW) +
				   h_Q(centerX,centerY  ,centerZ-1,IW) +
				   h_Q(centerX,centerY  ,centerZ  ,IW) );

		real_t B = 0.5 * (h_UOld(centerX,centerY  ,centerZ-1,IY) +
				  h_UOld(centerX,centerY  ,centerZ  ,IY) );

		real_t C = 0.5 * (h_UOld(centerX,centerY-1,centerZ  ,IZ) +
				  h_UOld(centerX,centerY  ,centerZ  ,IZ) );
	    
		Exyz[IX][dj][dk] = v*C-w*B;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * xPos;
		  Exyz[IX][dj][dk] += shear*C;
		}
		
	      } // end for dk
	    } // end for dj
  
	    for (int di=0; di<2; di++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic+di;
		int centerY = jc;
		int centerZ = kc+dk;
	    
		real_t u = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IU) + 
				   h_Q(centerX-1,centerY,centerZ  ,IU) + 
				   h_Q(centerX  ,centerY,centerZ-1,IU) + 
				   h_Q(centerX  ,centerY,centerZ  ,IU) );
  
		real_t w = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IW) +
				   h_Q(centerX-1,centerY,centerZ  ,IW) +
				   h_Q(centerX  ,centerY,centerZ-1,IW) +
				   h_Q(centerX  ,centerY,centerZ  ,IW) );
		
		real_t A = 0.5 * (h_UOld(centerX  ,centerY,centerZ-1,IX) + 
				  h_UOld(centerX  ,centerY,centerZ  ,IX) );

		real_t C = 0.5 * (h_UOld(centerX-1,centerY,centerZ  ,IZ) +
				  h_UOld(centerX  ,centerY,centerZ  ,IZ) );

		Exyz[IY][di][dk] = w*A-u*C;
		
	      } // end for dk
	    } // end for di
	    
	    for (int di=0; di<2; di++) {
	      for (int dj=0; dj<2; dj++) {

		int centerX = ic+di;
		int centerY = jc+dj;
		int centerZ = kc;

		real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IU) + 
				     h_Q(centerX-1,centerY  ,centerZ,IU) + 
				     h_Q(centerX  ,centerY-1,centerZ,IU) + 
				     h_Q(centerX  ,centerY  ,centerZ,IU) ); 
		
		real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IV) +
				     h_Q(centerX-1,centerY  ,centerZ,IV) +
				     h_Q(centerX  ,centerY-1,centerZ,IV) + 
				     h_Q(centerX  ,centerY  ,centerZ,IV) );
		
		real_t A  = 0.5  * (h_UOld(centerX  ,centerY-1,centerZ,IA) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IA) );
		
		real_t B  = 0.5  * (h_UOld(centerX-1,centerY  ,centerZ,IB) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IB) );
	    
		Exyz[IZ][di][dj] = u*B-v*A;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * (xPos - dx/2);
		  Exyz[IZ][di][dj] -= shear*A;
		}
		
	      } // end for dj
	    } // end for di

	    // 4. perform trace reconstruction
	    //
	    // Compute reconstructed states at left interface along X 
	    // in current cell
	    
	    // (ic,jc,kc)
	    bfNb2[0] =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb2[1] =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb2[2] =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb2[3] =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb2[4] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb2[5] =  h_UOld(ic  ,jc  ,kc+1,IC);

	    // EDGE_RT_Z
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_RT_Z,
				      qEdge_emfZ[IRT] );
	    
	    ///////////////////////////////////////////////
	    // compute reconstructed states at (i-1,j,k-1)
	    ///////////////////////////////////////////////
	    ic=i-1;
	    jc=j;
	    kc=k-1;
	    xPos = ::gParams.xMin + dx/2 + (ic-ghostWidth)*dx;

	    // get primitive variables state vector
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLoc[iVar]          = h_Q(ic  ,jc  ,kc  ,iVar);
	      qNeighbors[0][iVar] = h_Q(ic+1,jc  ,kc  ,iVar);
	      qNeighbors[1][iVar] = h_Q(ic-1,jc  ,kc  ,iVar);
	      qNeighbors[2][iVar] = h_Q(ic  ,jc+1,kc  ,iVar);
	      qNeighbors[3][iVar] = h_Q(ic  ,jc-1,kc  ,iVar);
	      qNeighbors[4][iVar] = h_Q(ic  ,jc  ,kc+1,iVar);
	      qNeighbors[5][iVar] = h_Q(ic  ,jc  ,kc-1,iVar);
	      
	    } // end for iVar
	    
	    // 1. compute hydro slopes
	    // compute slopes in left neighbor along X
	    slope_unsplit_hydro_3d(qLoc, 
				   qNeighbors[0],
				   qNeighbors[1],
				   qNeighbors[2],
				   qNeighbors[3],
				   qNeighbors[4],
				   qNeighbors[5],
				   dq);
	      
	    // 2. compute mag slopes @(ic,jc,kc)
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[0] = dbf[IY][IX];
	    dABC[1] = dbf[IZ][IX];
	    dABC[2] = dbf[IX][IY];
	    dABC[3] = dbf[IZ][IY];
	    dABC[4] = dbf[IX][IZ];
	    dABC[5] = dbf[IY][IZ];

	    // change neighbors to ic+1, jc, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic+1,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic+1,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic+1,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic+1,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+2,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic+1,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+2,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic+1,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[6] = dbf[IY][IX];
	    dABC[7] = dbf[IZ][IX];
	    
	    // change neighbors to ic, jc+1, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+2,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc+1,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc+1,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc+1,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc+1,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc+1,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc+1,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+2,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc  ,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[8] = dbf[IX][IY];
	    dABC[9] = dbf[IZ][IY];

	    // change neighbors to ic, jc, kc+1 and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc+1,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+2,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc  ,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc+1,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+2,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc  ,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc+1,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc+1,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc+1,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc+1,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc+1,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[10] = dbf[IX][IZ];
	    dABC[11] = dbf[IY][IZ];

	    // 3. compute Ex,Ey,Ez (electric field components)
	    for (int dj=0; dj<2; dj++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic;
		int centerY = jc+dj;
		int centerZ = kc+dk;
	    
		real_t v = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IV) +
				   h_Q(centerX,centerY-1,centerZ  ,IV) +
				   h_Q(centerX,centerY  ,centerZ-1,IV) +
				   h_Q(centerX,centerY  ,centerZ  ,IV) );

		real_t w = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IW) +
				   h_Q(centerX,centerY-1,centerZ  ,IW) +
				   h_Q(centerX,centerY  ,centerZ-1,IW) +
				   h_Q(centerX,centerY  ,centerZ  ,IW) );

		real_t B = 0.5 * (h_UOld(centerX,centerY  ,centerZ-1,IY) +
				  h_UOld(centerX,centerY  ,centerZ  ,IY) );

		real_t C = 0.5 * (h_UOld(centerX,centerY-1,centerZ  ,IZ) +
				  h_UOld(centerX,centerY  ,centerZ  ,IZ) );
	    
		Exyz[IX][dj][dk] = v*C-w*B;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * xPos;
		  Exyz[IX][dj][dk] += shear*C;
		}
		
	      } // end for dk
	    } // end for dj
  
	    for (int di=0; di<2; di++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic+di;
		int centerY = jc;
		int centerZ = kc+dk;
	    
		real_t u = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IU) + 
				   h_Q(centerX-1,centerY,centerZ  ,IU) + 
				   h_Q(centerX  ,centerY,centerZ-1,IU) + 
				   h_Q(centerX  ,centerY,centerZ  ,IU) );
  
		real_t w = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IW) +
				   h_Q(centerX-1,centerY,centerZ  ,IW) +
				   h_Q(centerX  ,centerY,centerZ-1,IW) +
				   h_Q(centerX  ,centerY,centerZ  ,IW) );
		
		real_t A = 0.5 * (h_UOld(centerX  ,centerY,centerZ-1,IX) + 
				  h_UOld(centerX  ,centerY,centerZ  ,IX) );

		real_t C = 0.5 * (h_UOld(centerX-1,centerY,centerZ  ,IZ) +
				  h_UOld(centerX  ,centerY,centerZ  ,IZ) );

		Exyz[IY][di][dk] = w*A-u*C;
		
	      } // end for dk
	    } // end for di
	    
	    for (int di=0; di<2; di++) {
	      for (int dj=0; dj<2; dj++) {

		int centerX = ic+di;
		int centerY = jc+dj;
		int centerZ = kc;

		real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IU) + 
				     h_Q(centerX-1,centerY  ,centerZ,IU) + 
				     h_Q(centerX  ,centerY-1,centerZ,IU) + 
				     h_Q(centerX  ,centerY  ,centerZ,IU) ); 
		
		real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IV) +
				     h_Q(centerX-1,centerY  ,centerZ,IV) +
				     h_Q(centerX  ,centerY-1,centerZ,IV) + 
				     h_Q(centerX  ,centerY  ,centerZ,IV) );
		
		real_t A  = 0.5  * (h_UOld(centerX  ,centerY-1,centerZ,IA) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IA) );
		
		real_t B  = 0.5  * (h_UOld(centerX-1,centerY  ,centerZ,IB) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IB) );
	    
		Exyz[IZ][di][dj] = u*B-v*A;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * (xPos - dx/2);
		  Exyz[IZ][di][dj] -= shear*A;
		}
		
	      } // end for dj
	    } // end for di

	    // 4. perform trace reconstruction
	    //
	    // Compute reconstructed states at left interface along X 
	    // in current cell
	    
	    // (ic,jc,kc)
	    bfNb2[0] =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb2[1] =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb2[2] =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb2[3] =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb2[4] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb2[5] =  h_UOld(ic  ,jc  ,kc+1,IC);

	    // EDGE_RT_Y
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_RT_Y,
				      qEdge_emfY[IRT] );

	    ///////////////////////////////////////////////
	    // compute reconstructed states at (i,j-1,k-1)
	    ///////////////////////////////////////////////
	    ic=i;
	    jc=j-1;
	    kc=k-1;
	    xPos = ::gParams.xMin + dx/2 + (ic-ghostWidth)*dx;

	    // get primitive variables state vector
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLoc[iVar]          = h_Q(ic  ,jc  ,kc  ,iVar);
	      qNeighbors[0][iVar] = h_Q(ic+1,jc  ,kc  ,iVar);
	      qNeighbors[1][iVar] = h_Q(ic-1,jc  ,kc  ,iVar);
	      qNeighbors[2][iVar] = h_Q(ic  ,jc+1,kc  ,iVar);
	      qNeighbors[3][iVar] = h_Q(ic  ,jc-1,kc  ,iVar);
	      qNeighbors[4][iVar] = h_Q(ic  ,jc  ,kc+1,iVar);
	      qNeighbors[5][iVar] = h_Q(ic  ,jc  ,kc-1,iVar);
	      
	    } // end for iVar
	    
	    // 1. compute hydro slopes
	    // compute slopes in left neighbor along X
	    slope_unsplit_hydro_3d(qLoc, 
				   qNeighbors[0],
				   qNeighbors[1],
				   qNeighbors[2],
				   qNeighbors[3],
				   qNeighbors[4],
				   qNeighbors[5],
				   dq);
	      
	    // 2. compute mag slopes @(ic,jc,kc)
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[0] = dbf[IY][IX];
	    dABC[1] = dbf[IZ][IX];
	    dABC[2] = dbf[IX][IY];
	    dABC[3] = dbf[IZ][IY];
	    dABC[4] = dbf[IX][IZ];
	    dABC[5] = dbf[IY][IZ];

	    // change neighbors to ic+1, jc, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic+1,jc+1,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic+1,jc-1,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic+1,jc  ,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic+1,jc  ,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic+1,jc  ,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+2,jc  ,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic+1,jc  ,kc-1,IB);

	    bfNb[10] =  h_UOld(ic+1,jc  ,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+2,jc  ,kc  ,IC);
	    bfNb[12] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb[13] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[14] =  h_UOld(ic+1,jc-1,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    // get transverse mag slopes
	    dABC[6] = dbf[IY][IX];
	    dABC[7] = dbf[IZ][IX];
	    
	    // change neighbors to ic, jc+1, kc and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc+1,kc  ,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+2,kc  ,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc+1,kc-1,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc+1,kc  ,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc+1,kc  ,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc+1,kc+1,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc+1,kc-1,IB);

	    bfNb[10] =  h_UOld(ic  ,jc+1,kc  ,IC);
	    bfNb[11] =  h_UOld(ic+1,jc+1,kc  ,IC);
	    bfNb[12] =  h_UOld(ic-1,jc+1,kc  ,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+2,kc  ,IC);
	    bfNb[14] =  h_UOld(ic  ,jc  ,kc  ,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[8] = dbf[IX][IY];
	    dABC[9] = dbf[IZ][IY];

	    // change neighbors to ic, jc, kc+1 and recompute dbf
	    bfNb[0]  =  h_UOld(ic  ,jc  ,kc+1,IA);
	    bfNb[1]  =  h_UOld(ic  ,jc+1,kc+1,IA);
	    bfNb[2]  =  h_UOld(ic  ,jc-1,kc+1,IA);
	    bfNb[3]  =  h_UOld(ic  ,jc  ,kc+2,IA);
	    bfNb[4]  =  h_UOld(ic  ,jc  ,kc  ,IA);

	    bfNb[5]  =  h_UOld(ic  ,jc  ,kc+1,IB);
	    bfNb[6]  =  h_UOld(ic+1,jc  ,kc+1,IB);
	    bfNb[7]  =  h_UOld(ic-1,jc  ,kc+1,IB);
	    bfNb[8]  =  h_UOld(ic  ,jc  ,kc+2,IB);
	    bfNb[9]  =  h_UOld(ic  ,jc  ,kc  ,IB);

	    bfNb[10] =  h_UOld(ic  ,jc  ,kc+1,IC);
	    bfNb[11] =  h_UOld(ic+1,jc  ,kc+1,IC);
	    bfNb[12] =  h_UOld(ic-1,jc  ,kc+1,IC);
	    bfNb[13] =  h_UOld(ic  ,jc+1,kc+1,IC);
	    bfNb[14] =  h_UOld(ic  ,jc-1,kc+1,IC);

	    slope_unsplit_mhd_3d(bfNb, dbf);
	    dABC[10] = dbf[IX][IZ];
	    dABC[11] = dbf[IY][IZ];

	    // 3. compute Ex,Ey,Ez (electric field components)
	    for (int dj=0; dj<2; dj++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic;
		int centerY = jc+dj;
		int centerZ = kc+dk;
	    
		real_t v = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IV) +
				   h_Q(centerX,centerY-1,centerZ  ,IV) +
				   h_Q(centerX,centerY  ,centerZ-1,IV) +
				   h_Q(centerX,centerY  ,centerZ  ,IV) );

		real_t w = 0.25 * (h_Q(centerX,centerY-1,centerZ-1,IW) +
				   h_Q(centerX,centerY-1,centerZ  ,IW) +
				   h_Q(centerX,centerY  ,centerZ-1,IW) +
				   h_Q(centerX,centerY  ,centerZ  ,IW) );

		real_t B = 0.5 * (h_UOld(centerX,centerY  ,centerZ-1,IY) +
				  h_UOld(centerX,centerY  ,centerZ  ,IY) );

		real_t C = 0.5 * (h_UOld(centerX,centerY-1,centerZ  ,IZ) +
				  h_UOld(centerX,centerY  ,centerZ  ,IZ) );
	    
		Exyz[IX][dj][dk] = v*C-w*B;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * xPos;
		  Exyz[IX][dj][dk] += shear*C;
		}
		
	      } // end for dk
	    } // end for dj
  
	    for (int di=0; di<2; di++) {
	      for (int dk=0; dk<2; dk++) {
	    
		int centerX = ic+di;
		int centerY = jc;
		int centerZ = kc+dk;
	    
		real_t u = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IU) + 
				   h_Q(centerX-1,centerY,centerZ  ,IU) + 
				   h_Q(centerX  ,centerY,centerZ-1,IU) + 
				   h_Q(centerX  ,centerY,centerZ  ,IU) );
  
		real_t w = 0.25 * (h_Q(centerX-1,centerY,centerZ-1,IW) +
				   h_Q(centerX-1,centerY,centerZ  ,IW) +
				   h_Q(centerX  ,centerY,centerZ-1,IW) +
				   h_Q(centerX  ,centerY,centerZ  ,IW) );
		
		real_t A = 0.5 * (h_UOld(centerX  ,centerY,centerZ-1,IX) + 
				  h_UOld(centerX  ,centerY,centerZ  ,IX) );

		real_t C = 0.5 * (h_UOld(centerX-1,centerY,centerZ  ,IZ) +
				  h_UOld(centerX  ,centerY,centerZ  ,IZ) );

		Exyz[IY][di][dk] = w*A-u*C;
		
	      } // end for dk
	    } // end for di
	    
	    for (int di=0; di<2; di++) {
	      for (int dj=0; dj<2; dj++) {

		int centerX = ic+di;
		int centerY = jc+dj;
		int centerZ = kc;

		real_t u  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IU) + 
				     h_Q(centerX-1,centerY  ,centerZ,IU) + 
				     h_Q(centerX  ,centerY-1,centerZ,IU) + 
				     h_Q(centerX  ,centerY  ,centerZ,IU) ); 
		
		real_t v  = 0.25 *  (h_Q(centerX-1,centerY-1,centerZ,IV) +
				     h_Q(centerX-1,centerY  ,centerZ,IV) +
				     h_Q(centerX  ,centerY-1,centerZ,IV) + 
				     h_Q(centerX  ,centerY  ,centerZ,IV) );
		
		real_t A  = 0.5  * (h_UOld(centerX  ,centerY-1,centerZ,IA) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IA) );
		
		real_t B  = 0.5  * (h_UOld(centerX-1,centerY  ,centerZ,IB) + 
				    h_UOld(centerX  ,centerY  ,centerZ,IB) );
	    
		Exyz[IZ][di][dj] = u*B-v*A;
		
		if (/* cartesian */ Omega0>0 /* and not fargo*/) {
		  real_t shear = -1.5 * Omega0 * (xPos - dx/2);
		  Exyz[IZ][di][dj] -= shear*A;
		}
		
	      } // end for dj
	    } // end for di

	    // 4. perform trace reconstruction
	    //
	    // Compute reconstructed states at left interface along X 
	    // in current cell
	    
	    // (ic,jc,kc)
	    bfNb2[0] =  h_UOld(ic  ,jc  ,kc  ,IA);
	    bfNb2[1] =  h_UOld(ic+1,jc  ,kc  ,IA);
	    bfNb2[2] =  h_UOld(ic  ,jc  ,kc  ,IB);
	    bfNb2[3] =  h_UOld(ic  ,jc+1,kc  ,IB);
	    bfNb2[4] =  h_UOld(ic  ,jc  ,kc  ,IC);
	    bfNb2[5] =  h_UOld(ic  ,jc  ,kc+1,IC);

	    // EDGE_RT_X
	    trace_unsplit_mhd_3d_face(qLoc, dq, bfNb2, dABC, Exyz,
				      dtdx, dtdy, dtdz, xPos, EDGE_RT_X,
				      qEdge_emfX[IRT] );


	    // preparation for calling compute_emf (equivalent to cmp_mag_flx
	    // in DUMSES)
	    // in the following, the 3 first indexes in qEdge_emf array play
	    // the same offset role as in the calling argument of cmp_mag_flx 
	    // in DUMSES (if you see what I mean ?!)

	    real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ);
	    real_t emfY = compute_emf<EMFY>(qEdge_emfY);
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


  } // MHDRunGodunov::godunov_unsplit_cpu_v0
  //} // mhd_godunov_unsplit_cpu_v0_implem

} // namespace hydroSimu
