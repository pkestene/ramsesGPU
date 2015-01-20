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
 * \file HydroRunKT.cpp
 * \brief Implements class HydroRunKT
 * 
 * 2D Euler equation solver on a cartesian grid using Kurganov-Tadmor scheme
 *
 * \date 05/02/2010
 * \author P. Kestener.
 *
 * $Id: HydroRunKT.cpp 3452 2014-06-17 10:09:48Z pkestene $
 */
#include "HydroRunKT.h"

// include CUDA kernel when necessary
#ifdef __CUDACC__
#include "kurganov-tadmor.cuh"
#endif // __CUDACC__
#include "constoprim.h"
#include "kurganov-tadmor.h"

//#include <sys/time.h> // for gettimeofday
#include "../utils/monitoring/Timer.h"
#include "../utils/monitoring/date.h"
#include <iomanip>

namespace hydroSimu {

////////////////////////////////////////////////////////////////////////////////
// HydroRunKT class methods body
////////////////////////////////////////////////////////////////////////////////

HydroRunKT::HydroRunKT(ConfigMap &_configMap)
  : HydroRunBase(_configMap), 
    cmpdtBlockCount(192),
    dX(1.0f/nx), dY(1.0f/ny),
    xLambda(0.0f), yLambda(0.0f), odd(true),
    h_spectralRadii(), h_Uhalf()
#ifdef __CUDACC__
    ,
    d_spectralRadii(),
    d_Uhalf()
#else
  , h_Uprime(), h_Uqrime(), h_Ustar()
#endif // __CUDACC__

{
  
  // assert dimType == TWO_D : 3D is not implemented in the Kurganov
  // Tadmor scheme
  assert( (dimType!=THREE_D) && "3D Kurganov-Tadmor scheme is not available !!!");

  h_spectralRadii.allocate(make_uint3(cmpdtBlockCount, 1, 1));
#ifdef __CUDACC__
  d_spectralRadii.allocate(make_uint3(cmpdtBlockCount, 1, 1));
#endif // __CUDACC__
  
    h_Uhalf.allocate(make_uint3(isize, jsize, NVAR_2D));
#ifdef __CUDACC__
    d_Uhalf.allocate(make_uint3(isize, jsize, NVAR_2D));
#else
    h_Uprime.allocate(make_uint3(isize, jsize, NVAR_2D));
    h_Uqrime.allocate(make_uint3(isize, jsize, NVAR_2D));
    h_Ustar.allocate(make_uint3(isize, jsize, NVAR_2D));
    f.allocate(make_uint3(isize, jsize, NVAR_2D));
    g.allocate(make_uint3(isize, jsize, NVAR_2D));
    f_prime.allocate(make_uint3(isize, jsize, NVAR_2D));
    g_qrime.allocate(make_uint3(isize, jsize, NVAR_2D));
#endif // __CUDACC__

  // set resolution
  if (nx>ny) {
    dX = 1.0f/ny;
    dY = dX;
  } else {
    dX = 1.0f/nx;
    dY = dX;    
  }

  // make sure variable declared as __constant__ are copied to device
  // for current compilation unit
  copyToSymbolMemory();

} // HydroRunKT::HydroRunKT

// =======================================================
// =======================================================
HydroRunKT::~HydroRunKT()
{
} // HydroRunKT::~HydroRunKT



// =======================================================
// =======================================================
/** 
 * \brief Compute time step dt.
 * \return dt (real_t)
 */
real_t HydroRunKT::computeDt()
#ifdef __CUDACC__
{
  computeDt_kt_kernel<CMPDT_BLOCK_SIZE><<<cmpdtBlockCount, CMPDT_BLOCK_SIZE, CMPDT_BLOCK_SIZE*sizeof(real2_t)>>>(d_U.data(), d_spectralRadii.data(), d_U.section());
  
  d_spectralRadii.copyToHost(h_spectralRadii);
  real2_t* spectralRadii = h_spectralRadii.data();
  real2_t r_max = {0.0, 0.0};
  for(uint i = 0; i < cmpdtBlockCount; ++i)
    {
      r_max.x = max(r_max.x, spectralRadii[i].x);
      r_max.y = max(r_max.y, spectralRadii[i].y);
    }

  real_t dt;
  if (enableJet)
    dt = cfl / FMAX( r_max.x/dX, FMAX(r_max.y , FMAX(_gParams.smallc, ujet + cjet))/dY );
  else
    dt = cfl / FMAX( r_max.x/dX, r_max.y/dY );

  xLambda=dt/dX;
  yLambda=dt/dY;

  return dt;
}
#else // CPU version
{
  
  real_t rx, ry;
  real_t u[NVAR_2D];
	
  real_t r_maxx=0;
  real_t r_maxy=0;
	
  /*
   * compute max of rx,ry
   */
  for (int i=2; i<nx+2; ++i) {
  
    for (int j=2; j<ny+2; ++j) {
      
      for (int k=0; k<NVAR_2D; ++k)
	u[k]=h_U(i,j,k);

      spectral_radii<NVAR_2D>(u, rx, ry);
      
      if (rx>r_maxx)
	r_maxx=rx;
      if (ry>r_maxy)
	r_maxy=ry;
    }
  }
	
  real_t dt;
  if (enableJet)
    dt = cfl / FMAX( r_maxx/dX, FMAX(r_maxy, FMAX(_gParams.smallc, ujet + cjet))/dY );
  else
    dt = cfl / FMAX( r_maxx/dX, r_maxy/dY );

  xLambda=dt/dX;
  yLambda=dt/dY;
	
  return dt;

} // HydroRunKT::time_step
#endif // __CUDACC__

// =======================================================
// =======================================================
/**
 * main routine to start simulation.
 */
void HydroRunKT::start()
{

  std::cout << "Initialization\n";
  int configNb = configMap.getInteger("hydro", "riemann_config_number", 0);
  setRiemannConfId(configNb);
  init_simulation(problem);

  std::cout << "Starting time integration (Kurganov-Tadmor scheme)" << std::endl;
  real_t  t = 0;
  real_t dt = 0;
  int   nStep = 0;

  std::cout << "nx      : " << nx                << std::endl;
  std::cout << "ny      : " << ny                << std::endl;
  std::cout << "tEnd    : " << tEnd              << std::endl;
  std::cout << "ALPHA_KT: " << _gParams.ALPHA_KT << std::endl;

  odd = true;

  // timing
  Timer timerTotal;
  Timer timerWriteOnDisk;
  
  // start timer
  timerTotal.start();

  while(t < tEnd && nStep < nStepmax)
    {

      /* Output results */
      if((nStep % nOutput)==0) {

	if (outputVtkEnabled)  outputVtk (h_U, nStep);
	if (outputHdf5Enabled) outputHdf5(h_U, nStep);
	if (outputXsmEnabled)  outputXsm (h_U, nStep, ID);
	if (outputPngEnabled)  outputPng (h_U, nStep, ID);

	timerWriteOnDisk.stop();

	std::cout << "[" << current_date() << "]"
		  << "  step=" << std::setw(9) << nStep 
		  << " t=" << std::setprecision(8) << t 
		  << " dt=" << dt << std::endl;
	
      }
      
      /* one time step integration */
      oneStepIntegration(nStep, t, dt);
    }
  
  // final timing report
  timerTotal.stop();

  printf("Euler2d KurganovTadmor total  time: %5.3f sec\n", timerTotal.elapsed());
  printf("Euler2d KurganovTadmor output time: %5.3f sec (%2.2f %% of total time)\n",timerWriteOnDisk.elapsed(), timerWriteOnDisk.elapsed()/timerTotal.elapsed()*100.);

} // HydroRunKT::start


// =======================================================
// =======================================================
/**
 * do one time step integration
 */
  void HydroRunKT::oneStepIntegration(int& nStep, real_t& t, real_t& dt) {

  /* compute new time-step */
  dt=computeDt();
  kt_evolve();
  nStep++;
  t+=dt;
  odd = !odd;

} // HydroRunKT::step


// =======================================================
// =======================================================
void HydroRunKT::copyGpuToCpu(int nStep)
{

  (void) nStep;

#ifdef __CUDACC__
  //d_Uhalf.copyToHost(h_Uhalf);
  d_U.copyToHost(h_U);
#endif // __CUDACC__

} // HydroRunKT::copyGpuToCpu


// =======================================================
// =======================================================
/** 
 * \brief  time step evolve the Kurganov-Tadmor scheme
 *
 */
void HydroRunKT::kt_evolve()
#ifdef __CUDACC__
{
 
  /*
   * apply boundary condition to input
   */
  make_boundaries(d_U,1);
  make_boundaries(d_U,2);

  /*
   * reconstruction
   */
  dim3 dimBlock(REC_BLOCK_DIMX, REC_BLOCK_DIMY);
  dim3 dimGrid(blocksFor(isize, REC_BLOCK_INNER_DIMX), blocksFor(jsize, REC_BLOCK_INNER_DIMY));
  if (odd)
    reconstruction_2d_FD2_kt_kernel<true><<<dimGrid, dimBlock>>>(d_U.data(), d_Uhalf.data(), d_U.pitch(), d_U.dimx(), d_U.dimy(), xLambda, yLambda);
  else
    reconstruction_2d_FD2_kt_kernel<false><<<dimGrid, dimBlock>>>(d_U.data(), d_Uhalf.data(), d_U.pitch(), d_U.dimx(), d_U.dimy(), xLambda, yLambda);
  
  
  /*
   * prediction - correction
   */
  dim3 dimBlock2(PC_BLOCK_DIMX, PC_BLOCK_DIMY);
  dim3 dimGrid2(blocksFor(isize, PC_BLOCK_INNER_DIMX), blocksFor(jsize, PC_BLOCK_INNER_DIMY));
  if (odd)
    predictor_corrector_2d_FD2_kt_kernel<true><<<dimGrid2, dimBlock2>>>(d_U.data(), d_Uhalf.data(), d_U.pitch(), d_U.dimx(), d_U.dimy(), xLambda, yLambda);
  else
    predictor_corrector_2d_FD2_kt_kernel<false><<<dimGrid2, dimBlock2>>>(d_U.data(), d_Uhalf.data(), d_U.pitch(), d_U.dimx(), d_U.dimy(), xLambda, yLambda);


} // HydroRunKT::kt_evolve
#else // CPU version
{

  // apply boundary condition to input
  make_boundaries(h_U,1);
  make_boundaries(h_U,2);
  
  // compute h_Uhalf
  reconstruction_2d_FD2();

  // update U from fluxes
  predictor_corrector_2d_FD2();

} // HydroRunKT::kt_evolve
#endif // __CUDACC__


// =======================================================
// =======================================================
#ifdef __CUDACC__
#else
/** 
 * \brief reconstruction
 *
 */
void HydroRunKT::reconstruction_2d_FD2()
{

  for (int k=0; k<NVAR_2D; k++) {
    
    for (int j=1; j<ny+3; j++) {      
      for (int i=1; i<nx+3; i++) {
	h_Uprime(i,j,k)=minmod3(_gParams.ALPHA_KT*(h_U(i+1,j  ,k) - h_U(i  ,j  ,k)), 
				0.5f             *(h_U(i+1,j  ,k) - h_U(i-1,j  ,k)), 
				_gParams.ALPHA_KT*(h_U(i  ,j  ,k) - h_U(i-1,j  ,k)));
	h_Uqrime(i,j,k)=minmod3(_gParams.ALPHA_KT*(h_U(i  ,j+1,k) - h_U(i  ,j  ,k)),
				0.5f             *(h_U(i  ,j+1,k) - h_U(i  ,j-1,k)),
				_gParams.ALPHA_KT*(h_U(i  ,j  ,k) - h_U(i  ,j-1,k)));
      }
    }
		
    if (odd) {
      
      for (int j=1; j<ny+2; j++) {
	for (int i=1; i<nx+2; i++) 
	  h_Uhalf(i,j,k) = 
	    0.25f*((h_U(i  ,j  ,k) + 
		    h_U(i+1,j  ,k) +
		    h_U(i  ,j+1,k) +
		    h_U(i+1,j+1,k)) +
		   0.25f*((h_Uprime(i  ,j  ,k) - h_Uprime(i+1,j  ,k)) +
			  (h_Uprime(i  ,j+1,k) - h_Uprime(i+1,j+1,k)) +
			  (h_Uqrime(i  ,j  ,k) - h_Uqrime(i  ,j+1,k)) +
			  (h_Uqrime(i+1,j  ,k) - h_Uqrime(i+1,j+1,k))));
      }
      
    } else {
	
      for (int j=2; j<ny+3; j++) {
	for (int i=2; i<nx+3; i++) {
	  h_Uhalf(i,j,k) = 
	    0.25f*((h_U(i  ,j-1,k) + 
		    h_U(i-1,j-1,k) +
		    h_U(i  ,j  ,k) + 
		    h_U(i-1,j  ,k)) + 
		   0.25f*((h_Uprime(i-1,j-1,k) - h_Uprime(i  ,j-1,k)) +
			  (h_Uprime(i-1,j  ,k) - h_Uprime(i  ,j  ,k)) +
			  (h_Uqrime(i-1,j-1,k) - h_Uqrime(i-1,j  ,k)) +
			  (h_Uqrime(i  ,j-1,k) - h_Uqrime(i  ,j  ,k))));
	}
      }
    } // end if(odd)
  } // end for(k=0;k<NVAR_2D;k++)
  
} // HydroRunKT::reconstruction2d_FD2
#endif // __CUDACC__

#ifdef __CUDACC__
#else
// =======================================================
// =======================================================
void HydroRunKT::predictor_corrector_2d_FD2()
{
//   HostArray<real_t> f,g;
//   f.allocate(make_uint3(isize, jsize, NVAR_2D));
//   g.allocate(make_uint3(isize, jsize, NVAR_2D));

//   HostArray<real_t> f_prime,g_qrime;
//   f_prime.allocate(make_uint3(isize, jsize, NVAR_2D));
//   g_qrime.allocate(make_uint3(isize, jsize, NVAR_2D));


  real_t u[NVAR_2D];
  real_t flux_x[NVAR_2D], flux_y[NVAR_2D];

  for (int i=0; i<nx+4; i++) {
    for (int j=0; j<ny+4; j++) {
      for (int k=0; k<NVAR_2D; k++)
	u[k]=h_U(i,j,k);

      get_flux<NVAR_2D>(u, flux_x, flux_y);

      for (int k=0; k<NVAR_2D; k++) {
	f(i,j,k)=flux_x[k];
	g(i,j,k)=flux_y[k];
      }
    }
  }
  
  //calculate flux derivatives  
  real_t fu, fv, fw;
  real_t gu, gv, gw;
  for (int k=0; k<NVAR_2D; k++) {
    for (int j=1; j<ny+3; j++) {
      for (int i=1; i<nx+3; i++) {
	fu = _gParams.ALPHA_KT*(f(i+1,j,k)-f(i  ,j,k));
	fv = 0.5f             *(f(i+1,j,k)-f(i-1,j,k));
	fw = _gParams.ALPHA_KT*(f(i  ,j,k)-f(i-1,j,k));
	f_prime(i,j,k)=minmod3(fu, fv, fw);	
	
	gu = _gParams.ALPHA_KT*(g(i,j+1,k)-g(i,j  ,k));
	gv = 0.5f             *(g(i,j+1,k)-g(i,j-1,k));
	gw = _gParams.ALPHA_KT*(g(i,j  ,k)-g(i,j-1,k));
	g_qrime(i,j,k)=minmod3(gu, gv, gw);
      }
    }
  }
  
  /*
   * calculate predicted values : Ustar
   */
  for(int k=0; k<NVAR_2D; k++) {
    for (int i=1; i<nx+3; i++) {	
      for (int j=1; j<ny+3; j++) {
	h_Ustar(i,j,k)=h_U(i,j,k) - 0.5f*(xLambda*f_prime(i,j,k) + yLambda*g_qrime(i,j,k));
      }
    }
  }

  /*
   * corrector step
   */
  //calculate fluxes
  for (int i=1; i<nx+3; i++) {
    for (int j=1; j<ny+3; j++) {
      for (int k=0; k<NVAR_2D; k++) {
	u[k] = h_Ustar(i,j,k);
      }

      get_flux<NVAR_2D>(u, flux_x, flux_y);
      
      for (int k=0; k<NVAR_2D; k++) {
	f(i,j,k)=flux_x[k];
	g(i,j,k)=flux_y[k];
      }
    }
  }
  
  if(odd) {
    for (int i=1; i<nx+2; i++) {
      for (int j=1; j<ny+2; j++) {
	for (int k=0; k<NVAR_2D; k++) {
	  h_U(i,j,k) = h_Uhalf(i,j,k) - 
	    0.5f*(xLambda*((f(i+1,j  ,k)-f(i  ,j  ,k)) +
			   (f(i+1,j+1,k)-f(i  ,j+1,k))) + 
		  yLambda*((g(i,  j+1,k)-g(i  ,j  ,k)) + 
			   (g(i+1,j+1,k)-g(i+1,j  ,k))));
	}
      }
    }
  } else {
    for (int i=2; i<nx+3; i++) {
      for (int j=2; j<ny+3; j++) {
	for (int k=0; k<NVAR_2D; k++) {
	  h_U(i,j,k)= h_Uhalf(i,j,k) -
	    0.5f*(xLambda*((f(i  ,j-1,k)-f(i-1,j-1,k)) +
			   (f(i  ,j  ,k)-f(i-1,j  ,k))) +
		  yLambda*((g(i-1,j  ,k)-g(i-1,j-1,k)) +
			   (g(i  ,j  ,k)-g(i  ,j-1,k))));
	}
      }
    }
  }


} // HydroRunKT::predictor_corrector_2d_FD2

#endif // __CUDACC__

} // namespace hydroSimu

