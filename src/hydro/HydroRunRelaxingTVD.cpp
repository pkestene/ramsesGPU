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
 * \file HydroRunRelaxingTVD.cpp
 * \brief Implements class HydroRunRelaxingTVD.
 * 
 * 2D/3D Euler equation solver on a cartesian grid using the relaxing TVD scheme.
 *
 * Take care that this scheme requires 3 ghost cells (whereas Godunov
 * requires only 2) !
 *
 *
 * \date 24-Jan-2011
 * \author P. Kestener.
 *
 * $Id: HydroRunRelaxingTVD.cpp 2108 2012-05-23 12:07:21Z pkestene $
 */
#include "HydroRunRelaxingTVD.h"

// include CUDA kernel when necessary
#ifdef __CUDACC__
//#include "cmpdt.cuh"
#include "relaxingTVD.cuh"
#endif // __CUDACC__
#include "constoprim.h"
#include "relaxingTVD.h"

#include "../utils/monitoring/Timer.h"
#include "../utils/monitoring/date.h"
#include <iomanip>

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // HydroRunRelaxingTVD class methods body
  ////////////////////////////////////////////////////////////////////////////////

  HydroRunRelaxingTVD::HydroRunRelaxingTVD(ConfigMap &_configMap)
    : HydroRunBase(_configMap)
  {

    // make sure variable declared as __constant__ are copied to device
    // for current compilation unit
    copyToSymbolMemory();

  } // HydroRunRelaxingTVD::HydroRunRelaxingTVD

  // =======================================================
  // =======================================================
  HydroRunRelaxingTVD::~HydroRunRelaxingTVD()
  {
  } // HydroRunRelaxingTVD::~HydroRunRelaxingTVD

  // =======================================================
  // =======================================================
  /*
   * wrapper that performs the actual CPU or GPU scheme integration step.
   */
  void HydroRunRelaxingTVD::relaxing_tvd_sweep(int nStep, real_t dt)
#ifdef __CUDACC__
  {

    if (dimType == TWO_D) {
      
      relaxing_tvd_gpu(d_U , d_U2, XDIR, dt);
      relaxing_tvd_gpu(d_U2, d_U , YDIR, dt);
      relaxing_tvd_gpu(d_U , d_U2, YDIR, dt);
      relaxing_tvd_gpu(d_U2, d_U , XDIR, dt);
      
    } else { // THREE_D
      
      if (nStep % 3 == 0) {

	relaxing_tvd_gpu(d_U , d_U2, XDIR, dt);
	relaxing_tvd_gpu(d_U2, d_U , YDIR, dt);
	relaxing_tvd_gpu(d_U , d_U2, ZDIR, dt);
	relaxing_tvd_gpu(d_U2, d_U , ZDIR, dt);
	relaxing_tvd_gpu(d_U , d_U2, YDIR, dt);
	relaxing_tvd_gpu(d_U2, d_U , XDIR, dt);

      } else if (nStep % 3 == 1) {
	
	relaxing_tvd_gpu(d_U , d_U2, ZDIR, dt);
	relaxing_tvd_gpu(d_U2, d_U , XDIR, dt);
	relaxing_tvd_gpu(d_U , d_U2, YDIR, dt);
	relaxing_tvd_gpu(d_U2, d_U , YDIR, dt);
	relaxing_tvd_gpu(d_U , d_U2, XDIR, dt);
	relaxing_tvd_gpu(d_U2, d_U , ZDIR, dt);
	
      } else {

	relaxing_tvd_gpu(d_U , d_U2, YDIR, dt);
	relaxing_tvd_gpu(d_U2, d_U , ZDIR, dt);
	relaxing_tvd_gpu(d_U , d_U2, XDIR, dt);
	relaxing_tvd_gpu(d_U2, d_U , XDIR, dt);
	relaxing_tvd_gpu(d_U , d_U2, ZDIR, dt);
	relaxing_tvd_gpu(d_U2, d_U , YDIR, dt);

      }

    } // end THREE_D

  } // HydroRunRelaxingTVD::relaxing_tvd_sweep (GPU)

#else // CPU version

  {   
    if (dimType == TWO_D) {
      
      relaxing_tvd_cpu(h_U , h_U2, XDIR, dt);
      relaxing_tvd_cpu(h_U2, h_U , YDIR, dt);
      relaxing_tvd_cpu(h_U , h_U2, YDIR, dt);
      relaxing_tvd_cpu(h_U2, h_U , XDIR, dt);
      
    } else { // THREE_D
      
      if (nStep % 3 == 0) {

	relaxing_tvd_cpu(h_U , h_U2, XDIR, dt);
	relaxing_tvd_cpu(h_U2, h_U , YDIR, dt);
	relaxing_tvd_cpu(h_U , h_U2, ZDIR, dt);
	relaxing_tvd_cpu(h_U2, h_U , ZDIR, dt);
	relaxing_tvd_cpu(h_U , h_U2, YDIR, dt);
	relaxing_tvd_cpu(h_U2, h_U , XDIR, dt);

      } else if (nStep % 3 == 1) {

       relaxing_tvd_cpu(h_U , h_U2, ZDIR, dt);
       relaxing_tvd_cpu(h_U2, h_U , XDIR, dt);
       relaxing_tvd_cpu(h_U , h_U2, YDIR, dt);
       relaxing_tvd_cpu(h_U2, h_U , YDIR, dt);
       relaxing_tvd_cpu(h_U , h_U2, XDIR, dt);
       relaxing_tvd_cpu(h_U2, h_U , ZDIR, dt);

      } else {

	relaxing_tvd_cpu(h_U , h_U2, YDIR, dt);
	relaxing_tvd_cpu(h_U2, h_U , ZDIR, dt);
	relaxing_tvd_cpu(h_U , h_U2, XDIR, dt);
	relaxing_tvd_cpu(h_U2, h_U , XDIR, dt);
	relaxing_tvd_cpu(h_U , h_U2, ZDIR, dt);
	relaxing_tvd_cpu(h_U2, h_U , YDIR, dt);

      }

    } // end THREE_D

  } // HydroRunRelaxingTVD::relaxing_tvd_sweep (CPU)

#endif // __CUDACC__

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunRelaxingTVD::relaxing_tvd_gpu(DeviceArray<real_t>& d_UOld, 
					     DeviceArray<real_t>& d_UNew,
					     int idim, 
					     real_t dt) 
  {
    
    TIMER_START(timerBoundaries);
    make_boundaries(d_UOld,idim);
    TIMER_STOP(timerBoundaries);
  
    TIMER_START(timerRelaxingTVD);
    if (dimType == TWO_D) {
    
      // launch 2D kernel
      if (idim == XDIR) {
	
	  dim3 dimBlock(XDIR_BLOCK_DIMX_2D, 
			XDIR_BLOCK_DIMY_2D);
	  dim3 dimGrid(blocksFor(isize, XDIR_BLOCK_DIMX_2D_INNER), 
		       blocksFor(jsize, XDIR_BLOCK_DIMY_2D));
	  kernel_relaxing_TVD_2d_xDir<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							     d_UNew.data(),
							     d_UOld.pitch(), 
							     d_UOld.dimx(), 
							     d_UOld.dimy(), 
							     dt);
	  cutilCheckMsg( "kernel_relaxing_TVD_2d_xDir failed" );
      } else { // YDIR

	  dim3 dimBlock(YDIR_BLOCK_DIMX_2D, 
	  		YDIR_BLOCK_DIMY_2D);
	  dim3 dimGrid(blocksFor(isize, YDIR_BLOCK_DIMX_2D), 
	  	       blocksFor(jsize, YDIR_BLOCK_DIMY_2D_INNER));
	  kernel_relaxing_TVD_2d_yDir<<<dimGrid, dimBlock>>>(d_UOld.data(), 
	  						     d_UNew.data(), 
	  						     d_UOld.pitch(), 
	  						     d_UOld.dimx(), 
	  						     d_UOld.dimy(), 
	  						     dt);
	  cutilCheckMsg( "kernel_relaxing_TVD_2d_yDir failed" );
      }

    } else { // THREE_D
    }
    TIMER_STOP(timerRelaxingTVD);

  } // HydroRunRelaxingTVD::relaxing_tvd_gpu

#else

  // =======================================================
  // =======================================================
  void HydroRunRelaxingTVD::relaxing_tvd_cpu(HostArray<real_t>& h_UOld, 
					     HostArray<real_t>& h_UNew,
					     int idim, 
					     real_t dt) 
  {
    
    TIMER_START(timerBoundaries);
    make_boundaries(h_UOld,idim);
    TIMER_STOP(timerBoundaries);
    
    TIMER_START(timerRelaxingTVD);
    if (dimType == TWO_D) {
      
      real_t u[NVAR_2D]     = {0.0f, 0.0f, 0.0f, 0.0f};
      real_t w[NVAR_2D]     = {0.0f, 0.0f, 0.0f, 0.0f};
      
      // dimension splitting
      if (idim == XDIR) {
	
	HostArray<real_t> fr;   fr.allocate(make_uint4(isize,1,1,NVAR_2D));
	HostArray<real_t> fl;   fl.allocate(make_uint4(isize,1,1,NVAR_2D));
	HostArray<real_t> fu;   fu.allocate(make_uint4(isize,1,1,NVAR_2D));
	HostArray<real_t> u1;   u1.allocate(make_uint4(isize,1,1,NVAR_2D));
	HostArray<real_t> dfr; dfr.allocate(make_uint4(isize,1,1,NVAR_2D));
	HostArray<real_t> dfl; dfl.allocate(make_uint4(isize,1,1,NVAR_2D));

	for (int j=3; j<jsize-3; ++j) {

	  /*
	   * do half step using first-order upwind scheme
	   */
	  for (int i=0; i<isize; ++i) {
	    real_t c; // freezing speed

	    // get current hydro state
	    u[ID] = h_U(i,j,0,ID);
	    u[IP] = h_U(i,j,0,IP);
	    u[IU] = h_U(i,j,0,IU);
	    u[IV] = h_U(i,j,0,IV);

	    // get averageFlux
	    averageFlux<NVAR_2D>(u,w,c);
	   
	    // compute left and righ fluxes
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      fr(i,0,0,iVar) = (u[iVar]*c+w[iVar])/2;
	      if (i>0) 
		fl(i-1,0,0,iVar) = (u[iVar]*c-w[iVar])/2;
	    }

	  } // end loop for i

	  for (int i=0; i<isize-1; ++i) {
	    for(int iVar=0; iVar<NVAR_2D; ++iVar)
	      fu(i,0,0,iVar) = fr(i,0,0,iVar)-fl(i,0,0,iVar);
	  }
	  for (int i=1; i<isize-1; ++i) {
	    u1(i,0,0,ID) = h_U(i,j,0,ID) - (fu(i,0,0,ID)-fu(i-1,0,0,ID))*dt/2;
	    u1(i,0,0,IP) = h_U(i,j,0,IP) - (fu(i,0,0,IP)-fu(i-1,0,0,IP))*dt/2;
	    u1(i,0,0,IU) = h_U(i,j,0,IU) - (fu(i,0,0,IU)-fu(i-1,0,0,IU))*dt/2;
	    u1(i,0,0,IV) = h_U(i,j,0,IV) - (fu(i,0,0,IV)-fu(i-1,0,0,IV))*dt/2;
	  }

	  /*
	   * do full step using second-order TVD scheme
	   */
	  for (int i=1; i<isize-1; ++i) {
	    real_t c; // freezing speed

	    // get current hydro state
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      u[iVar] = u1(i,0,0,iVar);
	    }
	   
	    // get averageFlux
	    averageFlux<NVAR_2D>(u,w,c);
	   
	    // compute left and righ fluxes
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      fr(i  ,0,0,iVar) = (u[iVar]*c+w[iVar])/2;
	      fl(i-1,0,0,iVar) = (u[iVar]*c-w[iVar])/2;
	    }

	  } // end for loop i
	
	  /*
	   * right moving waves 
	   */
	  // compute dfl
	  for (int i=2; i<isize-1; ++i) {
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      dfl(i,0,0,iVar) = ( fr(i,0,0,iVar) - fr(i-1,0,0,iVar) )/2;
	    }
	  }
	  // compute dfr
	  for (int i=1; i<isize-2; ++i) {
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      dfr(i,0,0,iVar) = dfl(i+1,0,0,iVar);
	    }
	  }
	  // compute fr : flux limiter
	  for (int i=2; i<isize-2; ++i) {
	    for(int iVar=0; iVar<NVAR_2D; ++iVar) {
	      vanleer( fr(i,0,0,iVar), dfl(i,0,0,iVar), dfr(i,0,0,iVar));
	    }
	  }
      
	  /*
	   * left moving waves 
	   */
	  // compute dfl, dfr
	  for (int i=1; i<isize-2; ++i) {
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      dfl(i  ,0,0,iVar) = ( fl(i-1,0,0,iVar) - fl(i,0,0,iVar) )/2;
	      dfr(i-1,0,0,iVar) = dfl(i,0,0,iVar);
	    }
	  }
	  // compute fl : flux limiter
	  for (int i=1; i<isize-3; ++i) {
	    for(int iVar=0; iVar<NVAR_2D; ++iVar) {
	      vanleer( fl(i,0,0,iVar), dfl(i,0,0,iVar), dfr(i,0,0,iVar));
	    }
	  }
	
	  // compute fu
	  for (int i=2; i<isize-3; ++i) {
	    for(int iVar=0; iVar<NVAR_2D; ++iVar) 
	      fu(i,0,0,iVar) = fr(i,0,0,iVar)-fl(i,0,0,iVar);
	  }

	  /*
	   * hydro update XDIR
	   */
	  for (int i=3; i<isize-3; ++i) {
	    h_UNew(i,j,0,ID) = h_UOld(i,j,0,ID) - (fu(i,0,0,ID)-fu(i-1,0,0,ID))*dt;
	    h_UNew(i,j,0,IP) = h_UOld(i,j,0,IP) - (fu(i,0,0,IP)-fu(i-1,0,0,IP))*dt;
	    h_UNew(i,j,0,IU) = h_UOld(i,j,0,IU) - (fu(i,0,0,IU)-fu(i-1,0,0,IU))*dt;
	    h_UNew(i,j,0,IV) = h_UOld(i,j,0,IV) - (fu(i,0,0,IV)-fu(i-1,0,0,IV))*dt;
	  }
	
	} // end for loop j

      } else if (idim == YDIR) {

	HostArray<real_t> fr;   fr.allocate(make_uint4(1,jsize,1,NVAR_2D));
	HostArray<real_t> fl;   fl.allocate(make_uint4(1,jsize,1,NVAR_2D));
	HostArray<real_t> fu;   fu.allocate(make_uint4(1,jsize,1,NVAR_2D));
	HostArray<real_t> u1;   u1.allocate(make_uint4(1,jsize,1,NVAR_2D));
	HostArray<real_t> dfr; dfr.allocate(make_uint4(1,jsize,1,NVAR_2D));
	HostArray<real_t> dfl; dfl.allocate(make_uint4(1,jsize,1,NVAR_2D));

	for (int i=3; i<isize-3; ++i) {

	  /*
	   * do half step using first-order upwind scheme
	   */
	  for (int j=0; j<jsize; ++j) {
	    real_t c; // freezing speed

	    // get current hydro state (swap IU and IV)
	    u[ID] = h_U(i,j,0,ID);
	    u[IP] = h_U(i,j,0,IP);
	    u[IU] = h_U(i,j,0,IV); // watchout swap
	    u[IV] = h_U(i,j,0,IU); // watchout swap
	   
	    // get averageFlux
	    averageFlux<NVAR_2D>(u,w,c);
	   
	    // compute left and righ fluxes
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      fr(0,j,0,iVar) = (u[iVar]*c+w[iVar])/2;
	      if (j>0) 
		fl(0,j-1,0,iVar) = (u[iVar]*c-w[iVar])/2;
	    }

	  } // end loop for j

	  for (int j=0; j<jsize-1; ++j) {
	    for(int iVar=0; iVar<NVAR_2D; ++iVar)
	      fu(0,j,0,iVar) = fr(0,j,0,iVar)-fl(0,j,0,iVar);
	  }
	  for (int j=1; j<jsize-1; ++j) { // watchout swap IU and IV
	    u1(0,j,0,ID) = h_U(i,j,0,ID) - (fu(0,j,0,ID)-fu(0,j-1,0,ID))*dt/2;
	    u1(0,j,0,IP) = h_U(i,j,0,IP) - (fu(0,j,0,IP)-fu(0,j-1,0,IP))*dt/2;
	    u1(0,j,0,IU) = h_U(i,j,0,IV) - (fu(0,j,0,IU)-fu(0,j-1,0,IU))*dt/2;
	    u1(0,j,0,IV) = h_U(i,j,0,IU) - (fu(0,j,0,IV)-fu(0,j-1,0,IV))*dt/2;
	  }

	  /*
	   * do full step using second-order TVD scheme
	   */
	  for (int j=1; j<jsize-1; ++j) {
	    real_t c; // freezing speed

	    // get current hydro state
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      u[iVar] = u1(0,j,0,iVar);
	    }
	   
	    // get averageFlux
	    averageFlux<NVAR_2D>(u,w,c);
	   
	    // compute left and righ fluxes
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      fr(0,j  ,0,iVar) = (u[iVar]*c+w[iVar])/2;
	      fl(0,j-1,0,iVar) = (u[iVar]*c-w[iVar])/2;
	    }

	  } // end for loop j
	
	  /*
	   * right moving waves 
	   */
	  // compute dfl
	  for (int j=2; j<jsize-1; ++j) {
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      dfl(0,j,0,iVar) = ( fr(0,j,0,iVar) - fr(0,j-1,0,iVar) )/2;
	    }
	  }
	  // compute dfr
	  for (int j=1; j<jsize-2; ++j) {
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      dfr(0,j,0,iVar) = dfl(0,j+1,0,iVar);
	    }
	  }
	  // compute fr : flux limiter
	  for (int j=2; j<jsize-2; ++j) {
	    for(int iVar=0; iVar<NVAR_2D; ++iVar) {
	      vanleer( fr(0,j,0,iVar), dfl(0,j,0,iVar), dfr(0,j,0,iVar));
	    }
	  }
      
	  /*
	   * left moving waves 
	   */
	  // compute dfl, dfr
	  for (int j=1; j<jsize-2; ++j) {
	    for (int iVar=0; iVar<NVAR_2D; ++iVar) {
	      dfl(0,j  ,0,iVar) = ( fl(0,j-1,0,iVar) - fl(0,j,0,iVar) )/2;
	      dfr(0,j-1,0,iVar) = dfl(0,j,0,iVar);
	    }
	  }
	  // compute fl : flux limiter
	  for (int j=1; j<jsize-3; ++j) {
	    for(int iVar=0; iVar<NVAR_2D; ++iVar)
	      vanleer( fl(0,j,0,iVar), dfl(0,j,0,iVar), dfr(0,j,0,iVar));
	  }
	
	  // compute fu
	  for (int j=2; j<jsize-3; ++j) {
	    for(int iVar=0; iVar<NVAR_2D; ++iVar) 
	      fu(0,j,0,iVar) = fr(0,j,0,iVar)-fl(0,j,0,iVar);
	  }

	  /*
	   * hydro update YDIR (take care to swap IU and IV)
	   */
	  for (int j=3; j<jsize-3; ++j) {
	    h_UNew(i,j,0,ID) = h_UOld(i,j,0,ID) - (fu(0,j,0,ID)-fu(0,j-1,0,ID))*dt;
	    h_UNew(i,j,0,IP) = h_UOld(i,j,0,IP) - (fu(0,j,0,IP)-fu(0,j-1,0,IP))*dt;
	    h_UNew(i,j,0,IU) = h_UOld(i,j,0,IU) - (fu(0,j,0,IV)-fu(0,j-1,0,IV))*dt;
	    h_UNew(i,j,0,IV) = h_UOld(i,j,0,IV) - (fu(0,j,0,IU)-fu(0,j-1,0,IU))*dt;	
	  }
	
	} // end for loop i
      
      } // end idim == YDIR 
    
    } else { 

      /*
       * THREE_D
       */

      real_t u[NVAR_3D]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      real_t w[NVAR_3D]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
      // dimension splitting
      if (idim == XDIR) {

	HostArray<real_t> fr;   fr.allocate(make_uint4(isize,1,1,NVAR_3D));
	HostArray<real_t> fl;   fl.allocate(make_uint4(isize,1,1,NVAR_3D));
	HostArray<real_t> fu;   fu.allocate(make_uint4(isize,1,1,NVAR_3D));
	HostArray<real_t> u1;   u1.allocate(make_uint4(isize,1,1,NVAR_3D));
	HostArray<real_t> dfr; dfr.allocate(make_uint4(isize,1,1,NVAR_3D));
	HostArray<real_t> dfl; dfl.allocate(make_uint4(isize,1,1,NVAR_3D));

	for (int k=3; k<ksize-3; ++k) {

	  for (int j=3; j<jsize-3; ++j) {
	  
	    /*
	     * do half step using first-order upwind scheme
	     */
	    for (int i=0; i<isize; ++i) {
	      real_t c; // freezing speed
	    
	      // get current hydro state
	      u[ID] = h_U(i,j,k,ID);
	      u[IP] = h_U(i,j,k,IP);
	      u[IU] = h_U(i,j,k,IU);
	      u[IV] = h_U(i,j,k,IV);
	      u[IW] = h_U(i,j,k,IW);
	    
	      // get averageFlux
	      averageFlux<NVAR_3D>(u,w,c);
	   
	      // compute left and righ fluxes
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		fr(i,0,0,iVar) = (u[iVar]*c+w[iVar])/2;
		if (i>0) 
		  fl(i-1,0,0,iVar) = (u[iVar]*c-w[iVar])/2;
	      }

	    } // end loop for i

	    for (int i=0; i<isize-1; ++i) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar)
		fu(i,0,0,iVar) = fr(i,0,0,iVar)-fl(i,0,0,iVar);
	    }
	    for (int i=1; i<isize-1; ++i) {
	      u1(i,0,0,ID) = h_U(i,j,k,ID) - (fu(i,0,0,ID)-fu(i-1,0,0,ID))*dt/2;
	      u1(i,0,0,IP) = h_U(i,j,k,IP) - (fu(i,0,0,IP)-fu(i-1,0,0,IP))*dt/2;
	      u1(i,0,0,IU) = h_U(i,j,k,IU) - (fu(i,0,0,IU)-fu(i-1,0,0,IU))*dt/2;
	      u1(i,0,0,IV) = h_U(i,j,k,IV) - (fu(i,0,0,IV)-fu(i-1,0,0,IV))*dt/2;
	      u1(i,0,0,IW) = h_U(i,j,k,IW) - (fu(i,0,0,IW)-fu(i-1,0,0,IW))*dt/2;
	    }

	    /*
	     * do full step using second-order TVD scheme
	     */
	    for (int i=1; i<isize-1; ++i) {
	      real_t c; // freezing speed

	      // get current hydro state
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		u[iVar] = u1(i,0,0,iVar);
	      }
	   
	      // get averageFlux
	      averageFlux<NVAR_3D>(u,w,c);
	   
	      // compute left and righ fluxes
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		fr(i  ,0,0,iVar) = (u[iVar]*c+w[iVar])/2;
		fl(i-1,0,0,iVar) = (u[iVar]*c-w[iVar])/2;
	      }

	    } // end for loop i
	
	    /*
	     * right moving waves 
	     */
	    // compute dfl
	    for (int i=2; i<isize-1; ++i) {
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		dfl(i,0,0,iVar) = ( fr(i,0,0,iVar) - fr(i-1,0,0,iVar) )/2;
	      }
	    }
	    // compute dfr
	    for (int i=1; i<isize-2; ++i) {
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		dfr(i,0,0,iVar) = dfl(i+1,0,0,iVar);
	      }
	    }
	    // compute fr : flux limiter
	    for (int i=2; i<isize-2; ++i) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar) {
		vanleer( fr(i,0,0,iVar), dfl(i,0,0,iVar), dfr(i,0,0,iVar));
	      }
	    }
      
	    /*
	     * left moving waves 
	     */
	    // compute dfl, dfr
	    for (int i=1; i<isize-2; ++i) {
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		dfl(i  ,0,0,iVar) = ( fl(i-1,0,0,iVar) - fl(i,0,0,iVar) )/2;
		dfr(i-1,0,0,iVar) = dfl(i,0,0,iVar);
	      }
	    }
	    // compute fl : flux limiter
	    for (int i=1; i<isize-3; ++i) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar) {
		vanleer( fl(i,0,0,iVar), dfl(i,0,0,iVar), dfr(i,0,0,iVar));
	      }
	    }
	
	    // compute fu
	    for (int i=2; i<isize-3; ++i) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar) 
		fu(i,0,0,iVar) = fr(i,0,0,iVar)-fl(i,0,0,iVar);
	    }

	    /*
	     * hydro update XDIR
	     */
	    for (int i=3; i<isize-3; ++i) {
	      h_UNew(i,j,k,ID) = h_UOld(i,j,k,ID) - (fu(i,0,0,ID)-fu(i-1,0,0,ID))*dt;
	      h_UNew(i,j,k,IP) = h_UOld(i,j,k,IP) - (fu(i,0,0,IP)-fu(i-1,0,0,IP))*dt;
	      h_UNew(i,j,k,IU) = h_UOld(i,j,k,IU) - (fu(i,0,0,IU)-fu(i-1,0,0,IU))*dt;
	      h_UNew(i,j,k,IV) = h_UOld(i,j,k,IV) - (fu(i,0,0,IV)-fu(i-1,0,0,IV))*dt;
	      h_UNew(i,j,k,IW) = h_UOld(i,j,k,IW) - (fu(i,0,0,IW)-fu(i-1,0,0,IW))*dt;
	    }
	
	  } // end for loop j

	} // end for loop k
      
      } else if (idim == YDIR) {
      
	HostArray<real_t> fr;   fr.allocate(make_uint4(1,jsize,1,NVAR_3D));
	HostArray<real_t> fl;   fl.allocate(make_uint4(1,jsize,1,NVAR_3D));
	HostArray<real_t> fu;   fu.allocate(make_uint4(1,jsize,1,NVAR_3D));
	HostArray<real_t> u1;   u1.allocate(make_uint4(1,jsize,1,NVAR_3D));
	HostArray<real_t> dfr; dfr.allocate(make_uint4(1,jsize,1,NVAR_3D));
	HostArray<real_t> dfl; dfl.allocate(make_uint4(1,jsize,1,NVAR_3D));
      
	for (int k=3; k<ksize-3; ++k) {
	
	  for (int i=3; i<isize-3; ++i) {
	  
	    /*
	     * do half step using first-order upwind scheme
	     */
	    for (int j=0; j<jsize; ++j) {
	      real_t c; // freezing speed
	      
	      // get current hydro state (swap IU and IV)
	      u[ID] = h_U(i,j,k,ID);
	      u[IP] = h_U(i,j,k,IP);
	      u[IU] = h_U(i,j,k,IV); // watchout swap
	      u[IV] = h_U(i,j,k,IU); // watchout swap
	      u[IW] = h_U(i,j,k,IW);
	      
	      // get averageFlux
	      averageFlux<NVAR_3D>(u,w,c);
	      
	      // compute left and righ fluxes
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		fr(0,j,0,iVar) = (u[iVar]*c+w[iVar])/2;
		if (j>0) 
		  fl(0,j-1,0,iVar) = (u[iVar]*c-w[iVar])/2;
	      }
	      
	    } // end loop for j
	    
	    for (int j=0; j<jsize-1; ++j) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar)
		fu(0,j,0,iVar) = fr(0,j,0,iVar)-fl(0,j,0,iVar);
	    }
	    for (int j=1; j<jsize-1; ++j) { // watchout swap IU and IV
	      u1(0,j,0,ID) = h_U(i,j,k,ID) - (fu(0,j,0,ID)-fu(0,j-1,0,ID))*dt/2;
	      u1(0,j,0,IP) = h_U(i,j,k,IP) - (fu(0,j,0,IP)-fu(0,j-1,0,IP))*dt/2;
	      u1(0,j,0,IU) = h_U(i,j,k,IV) - (fu(0,j,0,IU)-fu(0,j-1,0,IU))*dt/2;
	      u1(0,j,0,IV) = h_U(i,j,k,IU) - (fu(0,j,0,IV)-fu(0,j-1,0,IV))*dt/2;
	      u1(0,j,0,IW) = h_U(i,j,k,IW) - (fu(0,j,0,IW)-fu(0,j-1,0,IW))*dt/2;
	    }
	    
	    /*
	     * do full step using second-order TVD scheme
	     */
	    for (int j=1; j<jsize-1; ++j) {
	      real_t c; // freezing speed
	      
	      // get current hydro state
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		u[iVar] = u1(0,j,0,iVar);
	      }
	      
	      // get averageFlux
	      averageFlux<NVAR_3D>(u,w,c);
	      
	      // compute left and righ fluxes
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		fr(0,j  ,0,iVar) = (u[iVar]*c+w[iVar])/2;
		fl(0,j-1,0,iVar) = (u[iVar]*c-w[iVar])/2;
	      }
	      
	    } // end for loop j
	    
	    /*
	     * right moving waves 
	     */
	    // compute dfl
	    for (int j=2; j<jsize-1; ++j) {
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		dfl(0,j,0,iVar) = ( fr(0,j,0,iVar) - fr(0,j-1,0,iVar) )/2;
	      }
	    }
	    // compute dfr
	    for (int j=1; j<jsize-2; ++j) {
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		dfr(0,j,0,iVar) = dfl(0,j+1,0,iVar);
	      }
	    }
	    // compute fr : flux limiter
	    for (int j=2; j<jsize-2; ++j) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar) {
		vanleer( fr(0,j,0,iVar), dfl(0,j,0,iVar), dfr(0,j,0,iVar));
	      }
	    }
	    
	    /*
	     * left moving waves 
	     */
	    // compute dfl, dfr
	    for (int j=1; j<jsize-2; ++j) {
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		dfl(0,j  ,0,iVar) = ( fl(0,j-1,0,iVar) - fl(0,j,0,iVar) )/2;
		dfr(0,j-1,0,iVar) = dfl(0,j,0,iVar);
	      }
	    }
	    // compute fl : flux limiter
	    for (int j=1; j<jsize-3; ++j) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar)
		vanleer( fl(0,j,0,iVar), dfl(0,j,0,iVar), dfr(0,j,0,iVar));
	    }
	    
	    // compute fu
	    for (int j=2; j<jsize-3; ++j) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar) 
		fu(0,j,0,iVar) = fr(0,j,0,iVar)-fl(0,j,0,iVar);
	    }
	    
	    /*
	     * hydro update YDIR (take care to swap IU and IV)
	     */
	    for (int j=3; j<jsize-3; ++j) {
	      h_UNew(i,j,k,ID) = h_UOld(i,j,k,ID) - (fu(0,j,0,ID)-fu(0,j-1,0,ID))*dt;
	      h_UNew(i,j,k,IP) = h_UOld(i,j,k,IP) - (fu(0,j,0,IP)-fu(0,j-1,0,IP))*dt;
	      h_UNew(i,j,k,IU) = h_UOld(i,j,k,IU) - (fu(0,j,0,IV)-fu(0,j-1,0,IV))*dt;
	      h_UNew(i,j,k,IV) = h_UOld(i,j,k,IV) - (fu(0,j,0,IU)-fu(0,j-1,0,IU))*dt;  
	      h_UNew(i,j,k,IW) = h_UOld(i,j,k,IW) - (fu(0,j,0,IW)-fu(0,j-1,0,IW))*dt;	
	    }
	    
	  } // end for loop i
	  
	} // end for loop k
      
      } else if (idim == ZDIR) {
      
	HostArray<real_t> fr;   fr.allocate(make_uint4(1,1,ksize,NVAR_3D));
	HostArray<real_t> fl;   fl.allocate(make_uint4(1,1,ksize,NVAR_3D));
	HostArray<real_t> fu;   fu.allocate(make_uint4(1,1,ksize,NVAR_3D));
	HostArray<real_t> u1;   u1.allocate(make_uint4(1,1,ksize,NVAR_3D));
	HostArray<real_t> dfr; dfr.allocate(make_uint4(1,1,ksize,NVAR_3D));
	HostArray<real_t> dfl; dfl.allocate(make_uint4(1,1,ksize,NVAR_3D));
      
	for (int j=3; j<jsize-3; ++j) {
	
	  for (int i=3; i<isize-3; ++i) {
	  
	    /*
	     * do half step using first-order upwind scheme
	     */
	    for (int k=0; k<ksize; ++k) {
	      real_t c; // freezing speed
	      
	      // get current hydro state (swap IU and IW)
	      u[ID] = h_U(i,j,k,ID);
	      u[IP] = h_U(i,j,k,IP);
	      u[IU] = h_U(i,j,k,IW); // watchout swap
	      u[IV] = h_U(i,j,k,IV);
	      u[IW] = h_U(i,j,k,IU); // watchout swap
	      
	      // get averageFlux
	      averageFlux<NVAR_3D>(u,w,c);
	      
	      // compute left and righ fluxes
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		fr(0,0,k,iVar) = (u[iVar]*c+w[iVar])/2;
		if (k>0) 
		  fl(0,0,k-1,iVar) = (u[iVar]*c-w[iVar])/2;
	      }
	      
	    } // end loop for k
	    
	    for (int k=0; k<ksize-1; ++k) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar)
		fu(0,0,k,iVar) = fr(0,0,k,iVar)-fl(0,0,k,iVar);
	    }
	    for (int k=1; k<ksize-1; ++k) { // watchout swap IU and IW
	      u1(0,0,k,ID) = h_U(i,j,k,ID) - (fu(0,0,k,ID)-fu(0,0,k-1,ID))*dt/2;
	      u1(0,0,k,IP) = h_U(i,j,k,IP) - (fu(0,0,k,IP)-fu(0,0,k-1,IP))*dt/2;
	      u1(0,0,k,IU) = h_U(i,j,k,IW) - (fu(0,0,k,IU)-fu(0,0,k-1,IU))*dt/2;
	      u1(0,0,k,IV) = h_U(i,j,k,IV) - (fu(0,0,k,IV)-fu(0,0,k-1,IV))*dt/2;
	      u1(0,0,k,IW) = h_U(i,j,k,IU) - (fu(0,0,k,IW)-fu(0,0,k-1,IW))*dt/2;
	    }
	    
	    /*
	     * do full step using second-order TVD scheme
	     */
	    for (int k=1; k<ksize-1; ++k) {
	      real_t c; // freezing speed
	      
	      // get current hydro state
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		u[iVar] = u1(0,0,k,iVar);
	      }
	      
	      // get averageFlux
	      averageFlux<NVAR_3D>(u,w,c);
	      
	      // compute left and righ fluxes
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		fr(0,0,k  ,iVar) = (u[iVar]*c+w[iVar])/2;
		fl(0,0,k-1,iVar) = (u[iVar]*c-w[iVar])/2;
	      }
	      
	    } // end for loop j
	    
	    /*
	     * right moving waves 
	     */
	    // compute dfl
	    for (int k=2; k<ksize-1; ++k) {
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		dfl(0,0,k,iVar) = ( fr(0,0,k,iVar) - fr(0,0,k-1,iVar) )/2;
	      }
	    }
	    // compute dfr
	    for (int k=1; k<ksize-2; ++k) {
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		dfr(0,0,k,iVar) = dfl(0,0,k+1,iVar);
	      }
	    }
	    // compute fr : flux limiter
	    for (int k=2; k<ksize-2; ++k) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar) {
		vanleer( fr(0,0,k,iVar), dfl(0,0,k,iVar), dfr(0,0,k,iVar));
	      }
	    }
	    
	    /*
	     * left moving waves 
	     */
	    // compute dfl, dfr
	    for (int k=1; k<ksize-2; ++k) {
	      for (int iVar=0; iVar<NVAR_3D; ++iVar) {
		dfl(0,0,k  ,iVar) = ( fl(0,0,k-1,iVar) - fl(0,0,k,iVar) )/2;
		dfr(0,0,k-1,iVar) = dfl(0,0,k,iVar);
	      }
	    }
	    // compute fl : flux limiter
	    for (int k=1; k<ksize-3; ++k) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar)
		vanleer( fl(0,0,k,iVar), dfl(0,0,k,iVar), dfr(0,0,k,iVar));
	    }
	    
	    // compute fu
	    for (int k=2; k<ksize-3; ++k) {
	      for(int iVar=0; iVar<NVAR_3D; ++iVar) 
		fu(0,0,k,iVar) = fr(0,0,k,iVar)-fl(0,0,k,iVar);
	    }
	    
	    /*
	     * hydro update ZDIR (take care to swap IU and IW)
	     */
	    for (int k=3; k<ksize-3; ++k) {
	      h_UNew(i,j,k,ID) = h_UOld(i,j,k,ID) - (fu(0,0,k,ID)-fu(0,0,k-1,ID))*dt;
	      h_UNew(i,j,k,IP) = h_UOld(i,j,k,IP) - (fu(0,0,k,IP)-fu(0,0,k-1,IP))*dt;
	      h_UNew(i,j,k,IU) = h_UOld(i,j,k,IU) - (fu(0,0,k,IW)-fu(0,0,k-1,IW))*dt;
	      h_UNew(i,j,k,IV) = h_UOld(i,j,k,IV) - (fu(0,0,k,IV)-fu(0,0,k-1,IV))*dt;
	      h_UNew(i,j,k,IW) = h_UOld(i,j,k,IW) - (fu(0,0,k,IU)-fu(0,0,k-1,IU))*dt;	
	    }
	    
	  } // end for loop i
	  
	} // end for loop j
      
      } // end idim == ZDIR 
    
    } // end THREE_D
    TIMER_STOP(timerRelaxingTVD);

} // HydroRunRelaxingTVD::relaxing_tvd_cpu (CPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  /*
   * main routine to start simulation (must be called once in main application)
   */
  void HydroRunRelaxingTVD::start()
  {

    /*
     * initial condition
     */
    std::cout << "Initialization\n";
    // get (if present) Riemann config number (integer between 0 and 18)
    int configNb = configMap.getInteger("hydro" , "riemann_config_number", 0);
    setRiemannConfId(configNb);
    init_simulation(problem);
    

    // make sure border conditions are OK at beginning of simulation
#ifdef __CUDACC__
    make_boundaries(d_U,XDIR);
    make_boundaries(d_U,YDIR);
    if (dimType == THREE_D)
      make_boundaries(d_U,ZDIR);
#else
    make_boundaries(h_U,XDIR);
    make_boundaries(h_U,YDIR);
    if (dimType == THREE_D)
      make_boundaries(h_U,ZDIR);
#endif
    
    // dump information about computations about to start
    std::cout << "Starting time integration (Relaxing TVD)" << std::endl;
    std::cout << "use directional spliting integration" << std::endl;
    std::cout << "Resolution (nx,ny,nz) " << nx << " " << ny << " " << nz << std::endl;
  
    real_t t  = 0;
    real_t dt = 0;
    int nStep = 0;

    // timing
    Timer timerTotal;
    Timer timerWriteOnDisk;

    // start timer
    timerTotal.start();

    while(t < tEnd && nStep < nStepmax)
      {

	/* Output results */
	if((nStep % nOutput)==0) {

	  timerWriteOnDisk.start();

	  copyGpuToCpu(0);

	  if (outputVtkEnabled)  outputVtk (getDataHost(), nStep);
	  if (outputHdf5Enabled) outputHdf5(getDataHost(), nStep);
	  if (outputXsmEnabled)  outputXsm (getDataHost(), nStep);
	  if (outputPngEnabled)  outputPng (getDataHost(), nStep);

	  timerWriteOnDisk.stop();

	  std::cout << "[" << current_date() << "]"
		    << "  step=" << std::setw(9) << nStep 
		    << " t=" << std::setprecision(8) << t 
		    << " dt=" << dt << std::endl;
	}

	/* one time step integration (nStep increment) */
	oneStepIntegration(nStep, t, dt);
      
      }

    if (outputHdf5Enabled)
      writeXdmfForHdf5Wrapper(nStep);

    // final timing report
    timerTotal.stop();

    printf("Euler relaxing TVD total  time: %5.3f sec\n", timerTotal.elapsed());
    printf("Euler relaxing TVD output time: %5.3f sec (%2.2f %% of total time)\n",timerWriteOnDisk.elapsed(), timerWriteOnDisk.elapsed()/timerTotal.elapsed()*100.);

    /*
     * print timing report if required
     */
#ifdef DO_TIMING
    printf("Euler relaxing TVD boundaries : %5.3f sec (%2.2f %% of total time)\n",timerBoundaries.elapsed(), timerBoundaries.elapsed()/timerTotal.elapsed()*100.);
    printf("Euler relaxing TVD computing  : %5.3f sec (%2.2f %% of total time)\n",timerRelaxingTVD.elapsed(), timerRelaxingTVD.elapsed()/timerTotal.elapsed()*100.);
#endif // DO_TIMING

  } // HydroRunRelaxingTVD::start

  // =======================================================
  // =======================================================
  /*
   * do one time step integration
   */
  void HydroRunRelaxingTVD::oneStepIntegration(int& nStep, real_t& t, real_t& dt) 
  {

    // compute time step dt
    dt=compute_dt(0); // always use h_U or d_U
    dt /= dx; // see original Pen/Trac code

    // perform relaxing tvd sweep over all directions
    relaxing_tvd_sweep(nStep, dt);

    // update nStep and total time
    nStep++;
    t+=dt;

  } // HydroRunRelaxingTVD::oneStepIntegration

} // namespace hydroSimu
