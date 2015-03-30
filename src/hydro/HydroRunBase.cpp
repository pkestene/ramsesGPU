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
 * \file HydroRunBase.cpp
 * \brief Implements class HydroRunBase.
 *
 * \author P. Kestener
 *
 * $Id: HydroRunBase.cpp 3595 2014-11-04 12:27:24Z pkestene $
 */
#include "make_boundary_base.h"
#include "HydroRunBase.h"

#include "constoprim.h"

#include "utilities.h"

#include "RandomGen.h"
#include "turbulenceInit.h"
#include "structureFunctions.h"

#include "../utils/monitoring/date.h" // for current_date
#include <cnpy.h>


#include <iomanip> // for std::setprecision
#include <limits> // for std::numeric_limits

// include CUDA kernel when necessary
#ifdef __CUDACC__
#include "cmpdt.cuh"
#include "viscosity.cuh"
#include "viscosity_zslab.cuh"
#include "gravity.cuh"
#include "gravity_zslab.cuh"
#include "hydro_update.cuh"
#include "hydro_update_zslab.cuh"
#include "random_forcing.cuh"
#include "copyFaces.cuh"
#endif // __CUDACC__

/* Graphics Magick C++ API to dump PNG image files */
#ifdef USE_GM
#include <Magick++.h>
#endif // USE_GM

// for vtk file format output
#ifdef USE_VTK
#include "vtk_inc.h"
#endif // USE_VTK

// for HDF5 file format output
#ifdef USE_HDF5
#include <hdf5.h>

#define HDF5_MESG(mesg) \
  std::cerr << "HDF5 :" << mesg << std::endl;

#define HDF5_CHECK(val, mesg) do { \
    if (!val) {								\
      std::cerr << "*** HDF5 ERROR ***\n";				\
      std::cerr << "    HDF5_CHECK (" << mesg << ") failed\n";		\
    } \
} while(0)

#endif // USE_HDF5

// for NETCDF4 output
#ifdef USE_NETCDF4
#include <netcdf.h>
//#include <netcdfcpp.h>
#define NETCDF_ERROR(e) {printf("Error: %s\n", nc_strerror(e)); return;}
#endif // USE_NETCDF4

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // HydroRunBase class methods body
  ////////////////////////////////////////////////////////////////////////////////

  HydroRunBase::HydroRunBase(ConfigMap &_configMap)
    : HydroParameters(_configMap),
      dx(_gParams.dx),
      dy(_gParams.dy),
      dz(_gParams.dz),
      cmpdtBlockCount(192),
      randomForcingBlockCount(192),
      totalTime(0),
#ifdef __CUDACC__
      h_invDt(),
      d_invDt(),
      h_randomForcingNormalization(),
      d_randomForcingNormalization(),
#endif // __CUDACC__
      h_U(),
      h_U2()
#ifdef __CUDACC__
    , d_U()
    , d_U2()
#endif // __CUDACC__
    , h_gravity()
#ifdef __CUDACC__
    , d_gravity()
#endif // __CUDACC__
    , gravityEnabled(false)
    , h_randomForcing()
#ifdef __CUDACC__
    , d_randomForcing()
#endif // __CUDACC__
    , randomForcingEnabled(false)
    , randomForcingEdot(-1.0)
    , randomForcingOrnsteinUhlenbeckEnabled(false)
    , riemannConfId(0)
    , history_hydro_method(NULL)
  {
    // initialization of static members (in the same order as in enum ComponentIndex)
    varNames[ID]  = "density";
    varNames[IP]  = "energy";
    varNames[IU]  = "mx";
    varNames[IV]  = "my";
    varNames[IW]  = "mz";
    varNames[IBX] = "bx";
    varNames[IBY] = "by";
    varNames[IBZ] = "bz";
    
    varPrefix[ID]  = "d";
    varPrefix[IP]  = "p";
    varPrefix[IU]  = "u";
    varPrefix[IV]  = "v";
    varPrefix[IW]  = "w";
    varPrefix[IBX] = "a";
    varPrefix[IBY] = "b";
    varPrefix[IBZ] = "c";

    
    /*
     * hydro / MHD arrays initialization (nbVar is set in
     * HydroParameters class constructor).
     */
    if (dimType == TWO_D) {
      h_U.allocate (make_uint3(isize, jsize, nbVar));
      h_U2.allocate(make_uint3(isize, jsize, nbVar));
    } else {
      h_U.allocate (make_uint4(isize, jsize, ksize, nbVar));
      h_U2.allocate(make_uint4(isize, jsize, ksize, nbVar));    
    }
#ifdef __CUDACC__
    if (dimType == TWO_D) {
      d_U.allocate (make_uint3(isize, jsize, nbVar), gpuMemAllocType);
      d_U2.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
    } else {
      d_U.allocate (make_uint4(isize, jsize, ksize , nbVar), gpuMemAllocType);
      d_U2.allocate(make_uint4(isize, jsize, ksize , nbVar), gpuMemAllocType);
    }
#endif // __CUDACC__

  
#ifdef __CUDACC__	
    // we have to initialize the whole array, because padding zones won't be
    // initialized by the next copyFromHost. As the reduction algorithm consider
    // the array as a 1D array, padding zones must be initialized to zero.
    // Otherwise the timestep computation would return a bad value.
    //cudaMemset(d_U.data(), 0, d_U.sizeBytes());
    //cudaMemset(d_U2.data(), 0, d_U2.sizeBytes());
    d_U.reset();
    d_U2.reset();
    //d_U.copyFromHost(h_U); // load data into the VRAM
#endif // __CUDACC__


    /*
     * compute time step initialization.
     */

#ifdef __CUDACC__

    // for time step computation
    cmpdtBlockCount = std::min(cmpdtBlockCount, blocksFor(h_U.section(), CMPDT_BLOCK_SIZE * 2));

    h_invDt.allocate(make_uint3(cmpdtBlockCount, 1, 1));
    d_invDt.allocate(make_uint3(cmpdtBlockCount, 1, 1));

    // for random forcing reduction
    randomForcingBlockCount = std::min(randomForcingBlockCount, 
				       blocksFor(h_U.section(), 
						 RANDOM_FORCING_BLOCK_SIZE * 2));

    h_randomForcingNormalization.allocate(make_uint3(randomForcingBlockCount*
						     nbRandomForcingReduction, 1, 1));
    d_randomForcingNormalization.allocate(make_uint3(randomForcingBlockCount*
						     nbRandomForcingReduction, 1, 1));
    
    std::cout << "[Random forcing] randomForcingBlockCount = " << 
      randomForcingBlockCount << std::endl;

#endif // __CUDACC__

    /*
     * random forcing enabled ? Only for problem "turbulence"
     */
    if (!problem.compare("turbulence")) {

      randomForcingEnabled = true;
    
      // in that case, we also allocate memory for randomForcing arrays
      if (dimType == THREE_D) {
	h_randomForcing.allocate(make_uint4(isize, jsize, ksize, 3));
#ifdef __CUDACC__
	d_randomForcing.allocate(make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
#endif // __CUDACC__	
      } else {
	std::cerr << "ERROR : \"turbulence\" problem is not available in 2D !!!\n"; 
      }
    } // end if problem turbulence

    /*
     * random forcing enabled ? Only for problem "turbulence-Ornstein-Uhlenbeck"
     */
    if (!problem.compare("turbulence-Ornstein-Uhlenbeck")) {

      randomForcingOrnsteinUhlenbeckEnabled = true;
      
      if (dimType == THREE_D) {
	std::cout << "Ornstein-Uhlenbeck forcing enabled ...\n";

	// first  param is nDim=3
	// second param is nCpu=1 (should always be 1)
	pForcingOrnsteinUhlenbeck = new ForcingOrnsteinUhlenbeck(3, 1, configMap, _gParams);

      } else {
	std::cerr << "ERROR : \"turbulence-Ornstein-Uhlenbeck\" problem is not available in 2D !!!\n"; 
      }

    } // end if problem turbulence-Ornstein-Uhlenbeck

    /*
     * Gravity enabled
     */
    gravityEnabled = configMap.getBool("gravity", "enabled", false);

    // enforce gravityEnabled for some problems
    if ( !problem.compare("Rayleigh-Taylor") or !problem.compare("Keplerian-disk"))
      gravityEnabled = true;
    
    if ( gravityEnabled ) {

      // in that case, we also allocate memory for gravity array
      if (dimType == THREE_D) {
	h_gravity.allocate(make_uint4(isize, jsize, ksize, 3));
#ifdef __CUDACC__
	d_gravity.allocate(make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
#endif // __CUDACC__	

      } else { // TWO_D
	h_gravity.allocate(make_uint3(isize, jsize, 2));
#ifdef __CUDACC__
	d_gravity.allocate(make_uint3(isize, jsize, 2), gpuMemAllocType);
#endif // __CUDACC__	

      } // end TWO_D

      // register data pointers
      _gParams.arrayList[A_GRAV]    = h_gravity.data();
#ifdef __CUDACC__
      _gParams.arrayList[A_GRAV]    = d_gravity.data();
#endif // __CUDACC__	
      
    } // gravity

    /*
     * for Riemann problem test case...
     */
    initRiemannConfig2d(riemannConf);
    riemannConfId = configMap.getInteger("hydro", "riemann_config_number", 0);

    /*
     * VERY important:
     * make sure variables declared as __constant__ are copied to device
     * for current compilation unit
     */
    copyToSymbolMemory();

  } // HydroRunBase::HydroRunBase

  // =======================================================
  // =======================================================
  HydroRunBase::~HydroRunBase()
  {

    if (randomForcingOrnsteinUhlenbeckEnabled)
      delete pForcingOrnsteinUhlenbeck;

  } // HydroRunBase::~HydroRunBase

  // =======================================================
  // =======================================================
  real_t HydroRunBase::compute_dt(int useU)
#ifdef __CUDACC__
  {
    
    // choose between d_U and d_U2
    real_t *uData;
    if (useU == 0)
      uData = d_U.data();
    else
      uData = d_U2.data();

    // inverse time step
    real_t maxInvDt = 0;
 
    if (dimType == TWO_D) {
      
      cmpdt_2d<CMPDT_BLOCK_SIZE>
	<<<cmpdtBlockCount, 
	CMPDT_BLOCK_SIZE, 
	CMPDT_BLOCK_SIZE*sizeof(real_t)>>>(uData, 
					   d_invDt.data(),
					   d_U.pitch(),
					   d_U.dimx(),
					   d_U.dimy());
      checkCudaError("HydroRunBase cmpdt_2d error");
      d_invDt.copyToHost(h_invDt);
      checkCudaError("HydroRunBase d_invDt copy to host error");
      
    } else { // THREE_D

      cmpdt_3d<CMPDT_BLOCK_SIZE>
	<<<cmpdtBlockCount, 
	CMPDT_BLOCK_SIZE, 
	CMPDT_BLOCK_SIZE*sizeof(real_t)>>>(uData, 
					   d_invDt.data(), 
					   d_U.pitch(),
					   d_U.dimx(),
					   d_U.dimy(),
					   d_U.dimz());
      checkCudaError("HydroRunBase cmpdt_3d error");
      d_invDt.copyToHost(h_invDt);
      checkCudaError("HydroRunBase d_invDt copy to host error");
    
    } // end call cuda kernel for invDt reduction

    real_t* invDt = h_invDt.data();
    
    for(uint i = 0; i < cmpdtBlockCount; ++i)	{
      maxInvDt = FMAX ( maxInvDt, invDt[i]);
    }

    if (enableJet) {
      maxInvDt = FMAX ( maxInvDt, (this->ujet + this->cjet)/dx );
    }

    return cfl / maxInvDt;

  } // HydroRunBase::compute_dt -- GPU version
#else // CPU version
  {
    // choose between h_U and h_U2
    real_t *uData;
    if (useU == 0)
      uData = h_U.data();
    else
      uData = h_U2.data();
    
    // inverse time step
    real_t invDt = 0;
    
    if (dimType == TWO_D) {
      
      // for loop over inner region
      for (int j = ghostWidth; j < jsize-ghostWidth; j++)
	for (int i = ghostWidth; i < isize-ghostWidth; i++) {
	  real_t q[NVAR_2D];
	  real_t c;
	  int index = j*isize+i;
	  computePrimitives_0(uData, h_U.section(), index, c, q);
	  real_t vx = c + FABS(q[IU]);
	  real_t vy = c + FABS(q[IV]);
	  
	  invDt = FMAX ( invDt, vx/dx + vy/dy );
	  
	} // end for i,j
      
    } else { // THREE_D
      
      // for loop over inner region
      for (int k = ghostWidth; k < ksize-ghostWidth; k++)
	for (int j = ghostWidth; j < jsize-ghostWidth; j++)
	  for (int i = ghostWidth; i < isize-ghostWidth; i++) {
	    real_t q[NVAR_3D];
	    real_t c;
	    int index = k*isize*jsize + j*isize + i;
	    computePrimitives_3D_0(uData, h_U.section(), index, c, q);
	    real_t vx = c + FABS(q[IU]);
	    real_t vy = c + FABS(q[IV]);
	    real_t vz = c + FABS(q[IW]);
	    
	    invDt = FMAX ( invDt, vx/dx + vy/dy + vz/dz );
	    
	  } // end for i,j,k
      
    } // end THREE_D

    if (enableJet) {
      invDt = FMAX ( invDt, (this->ujet + this->cjet)/dx );
    }

    return cfl / invDt;
    
  } // HydroRunBase::compute_dt -- CPU version
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_viscosity_flux(HostArray<real_t>  &U, 
					    HostArray<real_t>  &flux_x, 
					    HostArray<real_t>  &flux_y, 
					    real_t              dt) 
  {
    real_t &cIso = _gParams.cIso;
    real_t &nu   = _gParams.nu;
    const real_t two3rd = 2./3.;

    if (dimType == TWO_D) {

      real_t dudx[2], dudy[2];

      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {

	  real_t u=0,v=0;
	  real_t uR, uL;
	  real_t uRR, uRL, uLR, uLL;
	  real_t txx,tyy,txy;

	  /*
	   * 1st direction viscous flux
	   */
	  real_t rho = HALF_F * ( U(i,j,ID) + U(i-1,j,ID) );

	  if (cIso <= 0) {
	    u = HALF_F * ( U(i,j,IU)/U(i,j,ID) + U(i-1,j,IU)/U(i-1,j,ID) );
	    v = HALF_F * ( U(i,j,IV)/U(i,j,ID) + U(i-1,j,IV)/U(i-1,j,ID) );
	  }
	  
	  // dudx along X
	  uR = U(i  ,j,IU) / U(i  ,j,ID);
	  uL = U(i-1,j,IU) / U(i-1,j,ID);
	  dudx[IX] = (uR-uL)/dx;

	  // dudx along Y
	  uR = U(i  ,j,IV) / U(i  ,j,ID);
	  uL = U(i-1,j,IV) / U(i-1,j,ID);
	  dudx[IY] = (uR-uL)/dx;
          
	  // dudy along X
	  uRR = U(i  ,j+1,IU) / U(i  ,j+1,ID);
	  uRL = U(i-1,j+1,IU) / U(i-1,j+1,ID);
	  uLR = U(i  ,j-1,IU) / U(i  ,j-1,ID);
	  uLL = U(i-1,j-1,IU) / U(i-1,j-1,ID);
	  uR  = uRR+uRL; 
	  uL  = uLR+uLL;
	  dudy[IX] = (uR-uL)/dy/4;

	  // dudy along Y
	  uRR = U(i  ,j+1,IV) / U(i  ,j+1,ID);
	  uRL = U(i-1,j+1,IV) / U(i-1,j+1,ID);
	  uLR = U(i  ,j-1,IV) / U(i  ,j-1,ID);
	  uLL = U(i-1,j-1,IV) / U(i-1,j-1,ID);
	  uR  = uRR+uRL; 
	  uL  = uLR+uLL;
	  dudy[IY] = (uR-uL)/dy/4;

	  txx = -two3rd *nu * rho * ( TWO_F*dudx[IX] - dudy[IY] );
	  txy = -        nu * rho * (       dudy[IX] + dudx[IY] );

	  flux_x(i,j,ID) = ZERO_F;
	  flux_x(i,j,IU) = txx*dt/dx;
	  flux_x(i,j,IV) = txy*dt/dx;
	  if (cIso <= 0) {
	    flux_x(i,j,IP) = (u*txx+v*txy)*dt/dx;
	  } else {
	    flux_x(i,j,IP) = ZERO_F;
	  }

	  /*
	   * 2nd direction viscous flux
	   */
	  rho = HALF_F * ( U(i,j,ID) + U(i,j-1,ID));
	  if (cIso <=0) {
	    u = HALF_F * ( U(i,j,IU)/U(i,j,ID) + U(i,j-1,IU)/U(i,j-1,ID) );
	    v = HALF_F * ( U(i,j,IV)/U(i,j,ID) + U(i,j-1,IV)/U(i,j-1,ID) );
	  }
	  
	  // dudy along X
	  uR = U(i,j  ,IU) / U(i,j  ,ID);
	  uL = U(i,j-1,IU) / U(i,j-1,ID);
	  dudy[IX] = (uR-uL)/dy;

	  // dudy along Y
	  uR = U(i,j  ,IV) / U(i,j  ,ID);
	  uL = U(i,j-1,IV) / U(i,j-1,ID);
	  dudy[IY] = (uR-uL)/dy;
           
	  // dudx along X
	  uRR = U(i+1,j  ,IU) / U(i+1,j  ,ID);
	  uRL = U(i+1,j-1,IU) / U(i+1,j-1,ID);
	  uLR = U(i-1,j  ,IU) / U(i-1,j  ,ID);
	  uLL = U(i-1,j-1,IU) / U(i-1,j-1,ID);
	  uR  = uRR+uRL; 
	  uL  = uLR+uLL;
	  dudx[IX] = (uR-uL)/dx/4;
           
	  // dudx along Y
	  uRR = U(i+1,j  ,IV) / U(i+1,j  ,ID);
	  uRL = U(i+1,j-1,IV) / U(i+1,j-1,ID);
	  uLR = U(i-1,j  ,IV) / U(i-1,j  ,ID);
	  uLL = U(i-1,j-1,IV) / U(i-1,j-1,ID);
	  uR  = uRR+uRL; 
	  uL  = uLR+uLL;
	  dudx[IY] = (uR-uL)/dx/4;
           
	  tyy = -two3rd * nu * rho * ( TWO_F * dudy[IY] - dudx[IX] );
	  txy = -         nu * rho * (         dudy[IX] + dudx[IY] );

	  flux_y(i,j,ID) = ZERO_F;
	  flux_y(i,j,IU) = txy*dt/dy;
	  flux_y(i,j,IV) = tyy*dt/dy;
	  if (cIso <=0) {
	    flux_y(i,j,IP) = (u*txy+v*tyy)*dt/dy;
	  } else {
	    flux_y(i,j,IP) = ZERO_F;
	  }
	  
	} // end for i
      } // end for j

    } // end TWO_D

  } // HydroRunBase::compute_viscosity_flux for 2D data (CPU version)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_viscosity_flux(DeviceArray<real_t>  &U, 
					    DeviceArray<real_t>  &flux_x, 
					    DeviceArray<real_t>  &flux_y, 
					    real_t                dt) 
  {

    dim3 dimBlock(VISCOSITY_2D_DIMX,
		  VISCOSITY_2D_DIMY);
    dim3 dimGrid(blocksFor(isize, VISCOSITY_2D_DIMX_INNER),
		 blocksFor(jsize, VISCOSITY_2D_DIMY_INNER));

    kernel_viscosity_forces_2d<<< dimGrid, dimBlock >>> (U.data(), flux_x.data(), flux_y.data(), 
							 ghostWidth, U.pitch(),
							 U.dimx(), U.dimy(), dt, dx, dy);
    checkCudaError("in HydroRunBase :: kernel_viscosity_forces_2d");

  } // HydroRunBase::compute_viscosity_flux for 2D data (GPU version)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_viscosity_flux(HostArray<real_t>  &U, 
					    HostArray<real_t>  &flux_x, 
					    HostArray<real_t>  &flux_y, 
					    HostArray<real_t>  &flux_z,
					    real_t              dt) 
  {
    real_t &cIso = _gParams.cIso;
    real_t &nu   = _gParams.nu;
    const real_t two3rd = 2./3.;

    // reset fluxes
    flux_x.reset();
    flux_y.reset();
    flux_z.reset();

    if (dimType == THREE_D) {

      real_t dudx[3], dudy[3], dudz[3];
      
      for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {

	    real_t u=0,v=0,w=0;
	    real_t uR, uL;
	    real_t uRR, uRL, uLR, uLL;
	    real_t txx,tyy,tzz,txy,txz,tyz;
	    
	    real_t rho;

	    /*
	     * 1st direction viscous flux
	     */
	    rho = HALF_F * ( U(i,j,k,ID) + U(i-1,j,k,ID) );

	    if (cIso <=0) {
	      u  = HALF_F * ( U(i,j,k,IU)/U(i,j,k,ID) + U(i-1,j,k,IU)/U(i-1,j,k,ID) );
	      v  = HALF_F * ( U(i,j,k,IV)/U(i,j,k,ID) + U(i-1,j,k,IV)/U(i-1,j,k,ID) );
	      w  = HALF_F * ( U(i,j,k,IW)/U(i,j,k,ID) + U(i-1,j,k,IW)/U(i-1,j,k,ID) );
	    }

	    // dudx along X
	    uR = U(i  ,j,k,IU) / U(i  ,j,k,ID);
	    uL = U(i-1,j,k,IU) / U(i-1,j,k,ID);
	    dudx[IX]=(uR-uL)/dx;

	    // dudx along Y
	    uR = U(i  ,j,k,IV) / U(i  ,j,k,ID);
	    uL = U(i-1,j,k,IV) / U(i-1,j,k,ID);
	    dudx[IY]=(uR-uL)/dx;

	    // dudx along Z
	    uR = U(i  ,j,k,IW) / U(i  ,j,k,ID);
	    uL = U(i-1,j,k,IW) / U(i-1,j,k,ID);
	    dudx[IZ]=(uR-uL)/dx;

	    
	    // dudy along X
	    uRR = U(i  ,j+1,k,IU) / U(i  ,j+1,k,ID);
	    uRL = U(i-1,j+1,k,IU) / U(i-1,j+1,k,ID);
	    uLR = U(i  ,j-1,k,IU) / U(i  ,j-1,k,ID);
	    uLL = U(i-1,j-1,k,IU) / U(i-1,j-1,k,ID);
	    uR  = uRR+uRL; 
	    uL  = uLR+uLL;
	    dudy[IX] = (uR-uL)/dy/4;

	    // dudy along Y
	    uRR = U(i  ,j+1,k,IV) / U(i  ,j+1,k,ID);
	    uRL = U(i-1,j+1,k,IV) / U(i-1,j+1,k,ID);
	    uLR = U(i  ,j-1,k,IV) / U(i  ,j-1,k,ID);
	    uLL = U(i-1,j-1,k,IV) / U(i-1,j-1,k,ID);
	    uR  = uRR+uRL; 
	    uL  = uLR+uLL;
	    dudy[IY] = (uR-uL)/dy/4;

	    // dudz along X
	    uRR = U(i  ,j,k+1,IU) / U(i  ,j,k+1,ID);
	    uRL = U(i-1,j,k+1,IU) / U(i-1,j,k+1,ID);
	    uLR = U(i  ,j,k-1,IU) / U(i  ,j,k-1,ID);
	    uLL = U(i-1,j,k-1,IU) / U(i-1,j,k-1,ID);
	    uR  = uRR+uRL; 
	    uL  = uLR+uLL;
	    dudz[IX] = (uR-uL)/dz/4;

	    // dudz along Z
	    uRR = U(i  ,j,k+1,IW) / U(i  ,j,k+1,ID);
	    uRL = U(i-1,j,k+1,IW) / U(i-1,j,k+1,ID);
	    uLR = U(i  ,j,k-1,IW) / U(i  ,j,k-1,ID);
	    uLL = U(i-1,j,k-1,IW) / U(i-1,j,k-1,ID);
	    uR  = uRR+uRL; 
	    uL  = uLR+uLL;
	    dudz[IZ] = (uR-uL)/dz/4;

	    txx = -two3rd * nu * rho * (TWO_F * dudx[IX] - dudy[IY] - dudz[IZ]);
	    txy = -         nu * rho * (        dudy[IX] + dudx[IY]           );
	    txz = -         nu * rho * (        dudz[IX] + dudx[IZ]           );
	    flux_x(i,j,k,ID) = ZERO_F;
	    flux_x(i,j,k,IU) = txx*dt/dx;
	    flux_x(i,j,k,IV) = txy*dt/dx;
	    flux_x(i,j,k,IW) = txz*dt/dx;
	    if (cIso <= 0) {
	      flux_x(i,j,k,IP) = (u*txx+v*txy+w*txz)*dt/dx;
	    } else {
	      flux_x(i,j,k,IP) = ZERO_F;
	    }

	    /*
	     * 2nd direction viscous flux
	     */
	    rho = HALF_F * ( U(i,j,k,ID) + U(i,j-1,k,ID) );

	    if (cIso <= 0) {
	      u = HALF_F * ( U(i,j,k,IU)/U(i,j,k,ID) + U(i,j-1,k,IU)/U(i,j-1,k,ID) );
	      v = HALF_F * ( U(i,j,k,IV)/U(i,j,k,ID) + U(i,j-1,k,IV)/U(i,j-1,k,ID) );
	      w = HALF_F * ( U(i,j,k,IW)/U(i,j,k,ID) + U(i,j-1,k,IW)/U(i,j-1,k,ID) );
	    }

	    // dudy along X
	    uR = U(i,j  ,k,IU) / U(i,j  ,k,ID);
	    uL = U(i,j-1,k,IU) / U(i,j-1,k,ID);
	    dudy[IX] = (uR-uL)/dy;

	    // dudy along Y
	    uR = U(i,j  ,k,IV) / U(i,j  ,k,ID);
	    uL = U(i,j-1,k,IV) / U(i,j-1,k,ID);
	    dudy[IY] = (uR-uL)/dy;

	    // dudy along Z
	    uR = U(i,j  ,k,IW) / U(i,j  ,k,ID);
	    uL = U(i,j-1,k,IW) / U(i,j-1,k,ID);
	    dudy[IZ] = (uR-uL)/dy;

	    // dudx along X
	    uRR = U(i+1,j  ,k,IU) / U(i+1,j  ,k,ID);
	    uRL = U(i+1,j-1,k,IU) / U(i+1,j-1,k,ID);
	    uLR = U(i-1,j  ,k,IU) / U(i-1,j  ,k,ID);
	    uLL = U(i-1,j-1,k,IU) / U(i-1,j-1,k,ID);
	    uR  = uRR+uRL;
	    uL  = uLR+uLL;
	    dudx[IX]=(uR-uL)/dx/4;

	    // dudx along Y
	    uRR = U(i+1,j  ,k,IV) / U(i+1,j  ,k,ID);
	    uRL = U(i+1,j-1,k,IV) / U(i+1,j-1,k,ID);
	    uLR = U(i-1,j  ,k,IV) / U(i-1,j  ,k,ID);
	    uLL = U(i-1,j-1,k,IV) / U(i-1,j-1,k,ID);
	    uR  = uRR+uRL;
	    uL  = uLR+uLL;
	    dudx[IY]=(uR-uL)/dx/4;

	    // dudz along Y
	    uRR = U(i,j  ,k+1,IV) / U(i,j  ,k+1,ID);
	    uRL = U(i,j-1,k+1,IV) / U(i,j-1,k+1,ID);
	    uLR = U(i,j  ,k-1,IV) / U(i,j  ,k-1,ID);
	    uLL = U(i,j-1,k-1,IV) / U(i,j-1,k-1,ID);
	    uR  = uRR+uRL; 
	    uL  = uLR+uLL;
	    dudz[IY]=(uR-uL)/dz/4;

	    // dudz along Z
	    uRR = U(i,j  ,k+1,IW) / U(i,j  ,k+1,ID);
	    uRL = U(i,j-1,k+1,IW) / U(i,j-1,k+1,ID);
	    uLR = U(i,j  ,k-1,IW) / U(i,j  ,k-1,ID);
	    uLL = U(i,j-1,k-1,IW) / U(i,j-1,k-1,ID);
	    uR  = uRR+uRL; 
	    uL  = uLR+uLL;
	    dudz[IZ]=(uR-uL)/dz/4;

	    tyy = -two3rd * nu * rho * (TWO_F * dudy[IY] - dudx[IX] - dudz[IZ] );
	    txy = -         nu * rho * (        dudy[IX] + dudx[IY]            );
	    tyz = -         nu * rho * (        dudz[IY] + dudy[IZ]            );
	    flux_y(i,j,k,ID) = ZERO_F;
	    flux_y(i,j,k,IU) = txy*dt/dy;
	    flux_y(i,j,k,IV) = tyy*dt/dy;
	    flux_y(i,j,k,IW) = tyz*dt/dy;
	    if (cIso <= 0) {
	      flux_y(i,j,k,IP) = (u*txy+v*tyy+w*tyz)*dt/dy;
	    } else {
	      flux_y(i,j,k,IP) = ZERO_F;
	    }

	    /*
	     * 3rd direction viscous flux
	     */
	    rho = HALF_F * ( U(i,j,k,ID) + U(i,j,k-1,ID) );
	    
	    if (cIso <= 0) {
	      u = HALF_F * ( U(i,j,k,IU)/U(i,j,k,ID) + U(i,j,k-1,IU)/U(i,j,k-1,ID) );
	      v = HALF_F * ( U(i,j,k,IV)/U(i,j,k,ID) + U(i,j,k-1,IV)/U(i,j,k-1,ID) );
	      w = HALF_F * ( U(i,j,k,IW)/U(i,j,k,ID) + U(i,j,k-1,IW)/U(i,j,k-1,ID) );
	    }

	    // dudz along X
	    uR = U(i,j,k  ,IU) / U(i,j,k  ,ID);
	    uL = U(i,j,k-1,IU) / U(i,j,k-1,ID);
	    dudz[IX] = (uR-uL)/dz;

	    // dudz along Y
	    uR = U(i,j,k  ,IV) / U(i,j,k  ,ID);
	    uL = U(i,j,k-1,IV) / U(i,j,k-1,ID);
	    dudz[IY] = (uR-uL)/dz;

	    // dudz along Z
	    uR = U(i,j,k  ,IW) / U(i,j,k  ,ID);
	    uL = U(i,j,k-1,IW) / U(i,j,k-1,ID);
	    dudz[IZ] = (uR-uL)/dz;

	    // dudx along X
	    uRR = U(i+1,j,k  ,IU) / U(i+1,j,k  ,ID);
	    uRL = U(i+1,j,k-1,IU) / U(i+1,j,k-1,ID);
	    uLR = U(i-1,j,k  ,IU) / U(i-1,j,k  ,ID);
	    uLL = U(i-1,j,k-1,IU) / U(i-1,j,k-1,ID);
	    uR  = uRR+uRL;
	    uL  = uLR+uLL;
	    dudx[IX] = (uR-uL)/dx/4;

	    // dudx along Z
	    uRR = U(i+1,j,k  ,IW) / U(i+1,j,k  ,ID);
	    uRL = U(i+1,j,k-1,IW) / U(i+1,j,k-1,ID);
	    uLR = U(i-1,j,k  ,IW) / U(i-1,j,k  ,ID);
	    uLL = U(i-1,j,k-1,IW) / U(i-1,j,k-1,ID);
	    uR  = uRR+uRL;
	    uL  = uLR+uLL;
	    dudx[IZ] = (uR-uL)/dx/4;
	
	    // dudy along Y
	    uRR = U(i,j+1,k  ,IV) / U(i,j+1,k  ,ID);
	    uRL = U(i,j+1,k-1,IV) / U(i,j+1,k-1,ID);
	    uLR = U(i,j-1,k  ,IV) / U(i,j-1,k  ,ID);
	    uLL = U(i,j-1,k-1,IV) / U(i,j-1,k-1,ID);
	    uR  = uRR+uRL;
	    uL  = uLR+uLL;
	    dudy[IY] = (uR-uL)/dy/4;

	    // dudy along Z
	    uRR = U(i,j+1,k  ,IW) / U(i,j+1,k  ,ID);
	    uRL = U(i,j+1,k-1,IW) / U(i,j+1,k-1,ID);
	    uLR = U(i,j-1,k  ,IW) / U(i,j-1,k  ,ID);
	    uLL = U(i,j-1,k-1,IW) / U(i,j-1,k-1,ID);
	    uR  = uRR+uRL;
	    uL  = uLR+uLL;
	    dudy[IZ] = (uR-uL)/dy/4;

	
	    tzz = -two3rd * nu * rho * (TWO_F * dudz[IZ] - dudx[IX] - dudy[IY] );
	    txz = -         nu * rho * (        dudz[IX] + dudx[IZ]            );
	    tyz = -         nu * rho * (        dudz[IY] + dudy[IZ]            );
	    flux_z(i,j,k,ID) = ZERO_F;
	    flux_z(i,j,k,IU) = txz*dt/dz;
	    flux_z(i,j,k,IV) = tyz*dt/dz;
	    flux_z(i,j,k,IW) = tzz*dt/dz;
	    if (cIso <= 0) {
	      flux_z(i,j,k,IP)= (u*txz+v*tyz+w*tzz)*dt/dz;
	    } else {
	      flux_z(i,j,k,IP) = ZERO_F;
	    }

	  } // end for i
	} // end for j
      } // end for k

    } // end THREE_D

  } // HydroRunBase::compute_viscosity_flux for 3D data (CPU version)
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_viscosity_flux(DeviceArray<real_t>  &U, 
					    DeviceArray<real_t>  &flux_x, 
					    DeviceArray<real_t>  &flux_y, 
					    DeviceArray<real_t>  &flux_z, 
					    real_t                dt) 
  {

    // reset fluxes
    flux_x.reset();
    flux_y.reset();
    flux_z.reset();

    dim3 dimBlock(VISCOSITY_3D_DIMX,
		  VISCOSITY_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, VISCOSITY_3D_DIMX_INNER),
		 blocksFor(jsize, VISCOSITY_3D_DIMY_INNER));

    kernel_viscosity_forces_3d<<< dimGrid, dimBlock >>> (U.data(), flux_x.data(), flux_y.data(), flux_z.data(),
							 ghostWidth, U.pitch(),
							 U.dimx(), U.dimy(), U.dimz(), dt, dx, dy, dz);
    checkCudaError("in HydroRunBase :: kernel_viscosity_forces_3d");

  } // HydroRunBase::compute_viscosity_flux for 3D data (GPU version)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_viscosity_flux(HostArray<real_t>  &U, 
					    HostArray<real_t>  &flux_x, 
					    HostArray<real_t>  &flux_y, 
					    HostArray<real_t>  &flux_z,
					    real_t              dt,
					    ZslabInfo           zSlabInfo)
  {
    real_t &cIso = _gParams.cIso;
    real_t &nu   = _gParams.nu;
    const real_t two3rd = 2./3.;
    
    // reset fluxes
    flux_x.reset();
    flux_y.reset();
    flux_z.reset();
    
    if (dimType == THREE_D) {
      
      real_t dudx[3], dudy[3], dudz[3];
      
      // start and stop index of current slab (ghosts included)
      int& kStart = zSlabInfo.kStart;
      int& kStop  = zSlabInfo.kStop;
    
      for (int k = kStart+ghostWidth; 
	   k     < kStop-ghostWidth+1; 
	   k++) {
      
	// local index inside slab
	int kL = k - kStart;

	if (k<ksize-ghostWidth+1) {

	  for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	    for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	    
	      real_t u=0,v=0,w=0;
	      real_t uR, uL;
	      real_t uRR, uRL, uLR, uLL;
	      real_t txx,tyy,tzz,txy,txz,tyz;
	    
	      real_t rho;
	    
	      /*
	       * 1st direction viscous flux
	       */
	      rho = HALF_F * ( U(i,j,k,ID) + U(i-1,j,k,ID) );

	      if (cIso <=0) {
		u  = HALF_F * ( U(i,j,k,IU)/U(i,j,k,ID) + U(i-1,j,k,IU)/U(i-1,j,k,ID) );
		v  = HALF_F * ( U(i,j,k,IV)/U(i,j,k,ID) + U(i-1,j,k,IV)/U(i-1,j,k,ID) );
		w  = HALF_F * ( U(i,j,k,IW)/U(i,j,k,ID) + U(i-1,j,k,IW)/U(i-1,j,k,ID) );
	      }

	      // dudx along X
	      uR = U(i  ,j,k,IU) / U(i  ,j,k,ID);
	      uL = U(i-1,j,k,IU) / U(i-1,j,k,ID);
	      dudx[IX]=(uR-uL)/dx;

	      // dudx along Y
	      uR = U(i  ,j,k,IV) / U(i  ,j,k,ID);
	      uL = U(i-1,j,k,IV) / U(i-1,j,k,ID);
	      dudx[IY]=(uR-uL)/dx;

	      // dudx along Z
	      uR = U(i  ,j,k,IW) / U(i  ,j,k,ID);
	      uL = U(i-1,j,k,IW) / U(i-1,j,k,ID);
	      dudx[IZ]=(uR-uL)/dx;

	    
	      // dudy along X
	      uRR = U(i  ,j+1,k,IU) / U(i  ,j+1,k,ID);
	      uRL = U(i-1,j+1,k,IU) / U(i-1,j+1,k,ID);
	      uLR = U(i  ,j-1,k,IU) / U(i  ,j-1,k,ID);
	      uLL = U(i-1,j-1,k,IU) / U(i-1,j-1,k,ID);
	      uR  = uRR+uRL; 
	      uL  = uLR+uLL;
	      dudy[IX] = (uR-uL)/dy/4;

	      // dudy along Y
	      uRR = U(i  ,j+1,k,IV) / U(i  ,j+1,k,ID);
	      uRL = U(i-1,j+1,k,IV) / U(i-1,j+1,k,ID);
	      uLR = U(i  ,j-1,k,IV) / U(i  ,j-1,k,ID);
	      uLL = U(i-1,j-1,k,IV) / U(i-1,j-1,k,ID);
	      uR  = uRR+uRL; 
	      uL  = uLR+uLL;
	      dudy[IY] = (uR-uL)/dy/4;

	      // dudz along X
	      uRR = U(i  ,j,k+1,IU) / U(i  ,j,k+1,ID);
	      uRL = U(i-1,j,k+1,IU) / U(i-1,j,k+1,ID);
	      uLR = U(i  ,j,k-1,IU) / U(i  ,j,k-1,ID);
	      uLL = U(i-1,j,k-1,IU) / U(i-1,j,k-1,ID);
	      uR  = uRR+uRL; 
	      uL  = uLR+uLL;
	      dudz[IX] = (uR-uL)/dz/4;

	      // dudz along Z
	      uRR = U(i  ,j,k+1,IW) / U(i  ,j,k+1,ID);
	      uRL = U(i-1,j,k+1,IW) / U(i-1,j,k+1,ID);
	      uLR = U(i  ,j,k-1,IW) / U(i  ,j,k-1,ID);
	      uLL = U(i-1,j,k-1,IW) / U(i-1,j,k-1,ID);
	      uR  = uRR+uRL; 
	      uL  = uLR+uLL;
	      dudz[IZ] = (uR-uL)/dz/4;

	      txx = -two3rd * nu * rho * (TWO_F * dudx[IX] - dudy[IY] - dudz[IZ]);
	      txy = -         nu * rho * (        dudy[IX] + dudx[IY]           );
	      txz = -         nu * rho * (        dudz[IX] + dudx[IZ]           );
	      flux_x(i,j,kL,ID) = ZERO_F;
	      flux_x(i,j,kL,IU) = txx*dt/dx;
	      flux_x(i,j,kL,IV) = txy*dt/dx;
	      flux_x(i,j,kL,IW) = txz*dt/dx;
	      if (cIso <= 0) {
		flux_x(i,j,kL,IP) = (u*txx+v*txy+w*txz)*dt/dx;
	      } else {
		flux_x(i,j,kL,IP) = ZERO_F;
	      }

	      /*
	       * 2nd direction viscous flux
	       */
	      rho = HALF_F * ( U(i,j,k,ID) + U(i,j-1,k,ID) );

	      if (cIso <= 0) {
		u = HALF_F * ( U(i,j,k,IU)/U(i,j,k,ID) + U(i,j-1,k,IU)/U(i,j-1,k,ID) );
		v = HALF_F * ( U(i,j,k,IV)/U(i,j,k,ID) + U(i,j-1,k,IV)/U(i,j-1,k,ID) );
		w = HALF_F * ( U(i,j,k,IW)/U(i,j,k,ID) + U(i,j-1,k,IW)/U(i,j-1,k,ID) );
	      }

	      // dudy along X
	      uR = U(i,j  ,k,IU) / U(i,j  ,k,ID);
	      uL = U(i,j-1,k,IU) / U(i,j-1,k,ID);
	      dudy[IX] = (uR-uL)/dy;

	      // dudy along Y
	      uR = U(i,j  ,k,IV) / U(i,j  ,k,ID);
	      uL = U(i,j-1,k,IV) / U(i,j-1,k,ID);
	      dudy[IY] = (uR-uL)/dy;

	      // dudy along Z
	      uR = U(i,j  ,k,IW) / U(i,j  ,k,ID);
	      uL = U(i,j-1,k,IW) / U(i,j-1,k,ID);
	      dudy[IZ] = (uR-uL)/dy;

	      // dudx along X
	      uRR = U(i+1,j  ,k,IU) / U(i+1,j  ,k,ID);
	      uRL = U(i+1,j-1,k,IU) / U(i+1,j-1,k,ID);
	      uLR = U(i-1,j  ,k,IU) / U(i-1,j  ,k,ID);
	      uLL = U(i-1,j-1,k,IU) / U(i-1,j-1,k,ID);
	      uR  = uRR+uRL;
	      uL  = uLR+uLL;
	      dudx[IX]=(uR-uL)/dx/4;

	      // dudx along Y
	      uRR = U(i+1,j  ,k,IV) / U(i+1,j  ,k,ID);
	      uRL = U(i+1,j-1,k,IV) / U(i+1,j-1,k,ID);
	      uLR = U(i-1,j  ,k,IV) / U(i-1,j  ,k,ID);
	      uLL = U(i-1,j-1,k,IV) / U(i-1,j-1,k,ID);
	      uR  = uRR+uRL;
	      uL  = uLR+uLL;
	      dudx[IY]=(uR-uL)/dx/4;

	      // dudz along Y
	      uRR = U(i,j  ,k+1,IV) / U(i,j  ,k+1,ID);
	      uRL = U(i,j-1,k+1,IV) / U(i,j-1,k+1,ID);
	      uLR = U(i,j  ,k-1,IV) / U(i,j  ,k-1,ID);
	      uLL = U(i,j-1,k-1,IV) / U(i,j-1,k-1,ID);
	      uR  = uRR+uRL; 
	      uL  = uLR+uLL;
	      dudz[IY]=(uR-uL)/dz/4;

	      // dudz along Z
	      uRR = U(i,j  ,k+1,IW) / U(i,j  ,k+1,ID);
	      uRL = U(i,j-1,k+1,IW) / U(i,j-1,k+1,ID);
	      uLR = U(i,j  ,k-1,IW) / U(i,j  ,k-1,ID);
	      uLL = U(i,j-1,k-1,IW) / U(i,j-1,k-1,ID);
	      uR  = uRR+uRL; 
	      uL  = uLR+uLL;
	      dudz[IZ]=(uR-uL)/dz/4;

	      tyy = -two3rd * nu * rho * (TWO_F * dudy[IY] - dudx[IX] - dudz[IZ] );
	      txy = -         nu * rho * (        dudy[IX] + dudx[IY]            );
	      tyz = -         nu * rho * (        dudz[IY] + dudy[IZ]            );
	      flux_y(i,j,kL,ID) = ZERO_F;
	      flux_y(i,j,kL,IU) = txy*dt/dy;
	      flux_y(i,j,kL,IV) = tyy*dt/dy;
	      flux_y(i,j,kL,IW) = tyz*dt/dy;
	      if (cIso <= 0) {
		flux_y(i,j,kL,IP) = (u*txy+v*tyy+w*tyz)*dt/dy;
	      } else {
		flux_y(i,j,kL,IP) = ZERO_F;
	      }

	      /*
	       * 3rd direction viscous flux
	       */
	      rho = HALF_F * ( U(i,j,k,ID) + U(i,j,k-1,ID) );
	    
	      if (cIso <= 0) {
		u = HALF_F * ( U(i,j,k,IU)/U(i,j,k,ID) + U(i,j,k-1,IU)/U(i,j,k-1,ID) );
		v = HALF_F * ( U(i,j,k,IV)/U(i,j,k,ID) + U(i,j,k-1,IV)/U(i,j,k-1,ID) );
		w = HALF_F * ( U(i,j,k,IW)/U(i,j,k,ID) + U(i,j,k-1,IW)/U(i,j,k-1,ID) );
	      }

	      // dudz along X
	      uR = U(i,j,k  ,IU) / U(i,j,k  ,ID);
	      uL = U(i,j,k-1,IU) / U(i,j,k-1,ID);
	      dudz[IX] = (uR-uL)/dz;

	      // dudz along Y
	      uR = U(i,j,k  ,IV) / U(i,j,k  ,ID);
	      uL = U(i,j,k-1,IV) / U(i,j,k-1,ID);
	      dudz[IY] = (uR-uL)/dz;

	      // dudz along Z
	      uR = U(i,j,k  ,IW) / U(i,j,k  ,ID);
	      uL = U(i,j,k-1,IW) / U(i,j,k-1,ID);
	      dudz[IZ] = (uR-uL)/dz;

	      // dudx along X
	      uRR = U(i+1,j,k  ,IU) / U(i+1,j,k  ,ID);
	      uRL = U(i+1,j,k-1,IU) / U(i+1,j,k-1,ID);
	      uLR = U(i-1,j,k  ,IU) / U(i-1,j,k  ,ID);
	      uLL = U(i-1,j,k-1,IU) / U(i-1,j,k-1,ID);
	      uR  = uRR+uRL;
	      uL  = uLR+uLL;
	      dudx[IX] = (uR-uL)/dx/4;

	      // dudx along Z
	      uRR = U(i+1,j,k  ,IW) / U(i+1,j,k  ,ID);
	      uRL = U(i+1,j,k-1,IW) / U(i+1,j,k-1,ID);
	      uLR = U(i-1,j,k  ,IW) / U(i-1,j,k  ,ID);
	      uLL = U(i-1,j,k-1,IW) / U(i-1,j,k-1,ID);
	      uR  = uRR+uRL;
	      uL  = uLR+uLL;
	      dudx[IZ] = (uR-uL)/dx/4;
	
	      // dudy along Y
	      uRR = U(i,j+1,k  ,IV) / U(i,j+1,k  ,ID);
	      uRL = U(i,j+1,k-1,IV) / U(i,j+1,k-1,ID);
	      uLR = U(i,j-1,k  ,IV) / U(i,j-1,k  ,ID);
	      uLL = U(i,j-1,k-1,IV) / U(i,j-1,k-1,ID);
	      uR  = uRR+uRL;
	      uL  = uLR+uLL;
	      dudy[IY] = (uR-uL)/dy/4;

	      // dudy along Z
	      uRR = U(i,j+1,k  ,IW) / U(i,j+1,k  ,ID);
	      uRL = U(i,j+1,k-1,IW) / U(i,j+1,k-1,ID);
	      uLR = U(i,j-1,k  ,IW) / U(i,j-1,k  ,ID);
	      uLL = U(i,j-1,k-1,IW) / U(i,j-1,k-1,ID);
	      uR  = uRR+uRL;
	      uL  = uLR+uLL;
	      dudy[IZ] = (uR-uL)/dy/4;

	
	      tzz = -two3rd * nu * rho * (TWO_F * dudz[IZ] - dudx[IX] - dudy[IY] );
	      txz = -         nu * rho * (        dudz[IX] + dudx[IZ]            );
	      tyz = -         nu * rho * (        dudz[IY] + dudy[IZ]            );
	      flux_z(i,j,kL,ID) = ZERO_F;
	      flux_z(i,j,kL,IU) = txz*dt/dz;
	      flux_z(i,j,kL,IV) = tyz*dt/dz;
	      flux_z(i,j,kL,IW) = tzz*dt/dz;
	      if (cIso <= 0) {
		flux_z(i,j,kL,IP)= (u*txz+v*tyz+w*tzz)*dt/dz;
	      } else {
		flux_z(i,j,kL,IP) = ZERO_F;
	      }

	    } // end for i
	  } // end for j
      
	} // end if (k<ksize-ghostWidth+1)

      } // end for k

    } // end THREE_D

  } // HydroRunBase::compute_viscosity_flux for 3D data (CPU version)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_viscosity_flux(DeviceArray<real_t>  &U, 
					    DeviceArray<real_t>  &flux_x, 
					    DeviceArray<real_t>  &flux_y, 
					    DeviceArray<real_t>  &flux_z, 
					    real_t                dt,
					    ZslabInfo             zSlabInfo) 
  {
        
    // take care that the last slab might be truncated
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1) {
      zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
    }
    
    // reset fluxes
    flux_x.reset();
    flux_y.reset();
    flux_z.reset();
    
    dim3 dimBlock(VISCOSITY_Z_3D_DIMX,
		  VISCOSITY_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, VISCOSITY_Z_3D_DIMX_INNER),
		 blocksFor(jsize, VISCOSITY_Z_3D_DIMY_INNER));
    
    kernel_viscosity_forces_3d_zslab<<< dimGrid, dimBlock >>> (U.data(), 
							       flux_x.data(), 
							       flux_y.data(), 
							       flux_z.data(),
							       ghostWidth, U.pitch(),
							       U.dimx(), 
							       U.dimy(), 
							       U.dimz(), 
							       dt, 
							       dx, dy, dz,
							       zSlabInfo);
    checkCudaError("HydroRunBase :: kernel_viscosity_forces_3d_zslab");
    
  } // HydroRunBase::compute_viscosity_flux for 3D data (GPU version)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  real_t HydroRunBase::compute_random_forcing_normalization(HostArray<real_t>  &U, 
							    real_t             dt)
  {

    // reduction - normalization prerequisites
    // 9 values :
    // 0 -> rho*v*(delta v) 
    // 1 -> rho*(delta v)^2
    // 2 -> rho*v^2/temperature
    // 3 -> v^2/temperature
    // 4 -> rho*v^2
    // 5 -> v*v
    // 6 -> rho*rho
    // 7 -> min(rho)
    // 8 -> max(rho)

    real_t reduceValue[nbRandomForcingReduction] = { 0.0, 0.0, 0.0,
						     0.0, 0.0, 0.0,
						     0.0, 0.0, 0.0 };
    // reduceValue[7] is a minimum
    reduceValue[7] = std::numeric_limits<float>::max();

    int64_t nbCells = nx*ny*nz;

    for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  
	  real_t rho = U(i,j,k,ID);
	  real_t u   = U(i,j,k,IU)/rho;
	  real_t v   = U(i,j,k,IV)/rho;
	  real_t w   = U(i,j,k,IW)/rho;
	  real_t uu  = h_randomForcing(i,j,k,IX);
	  real_t vv  = h_randomForcing(i,j,k,IY);
	  real_t ww  = h_randomForcing(i,j,k,IZ);

	  // sum of rho*v*(delta v)
	  reduceValue[0] += rho * (u*uu + v*vv + w*ww);

	  // sum of  rho*(delta v)^2
	  reduceValue[1] += rho*uu*uu;
	  reduceValue[1] += rho*vv*vv;
	  reduceValue[1] += rho*ww*ww;

	  // compute temperature (actually c^2 for isothermal)
	  real_t temperature;
	  if (_gParams.cIso >0) {
	    temperature = SQR(_gParams.cIso);
	  } else { // use ideal gas eq of state (P over rho)
	    temperature =  (_gParams.gamma0 - ONE_F) * 
	      (U(i,j,k,IP) - 0.5 * rho * ( u*u + v*v + w*w ) );
	  }

	  // compute rho*v^2/t
	  reduceValue[2] += rho * u * u / temperature;
	  reduceValue[2] += rho * v * v / temperature;
	  reduceValue[2] += rho * w * w / temperature;
	  
	  // compute v^2/t
	  reduceValue[3] += u * u / temperature;
	  reduceValue[3] += v * v / temperature;
  	  reduceValue[3] += w * w / temperature;

	  // compute rho*v^2
	  reduceValue[4] += rho * u * u;
	  reduceValue[4] += rho * v * v;
	  reduceValue[4] += rho * w * w;

	  // compute v^2
	  reduceValue[5] += u * u;
	  reduceValue[5] += v * v;
	  reduceValue[5] += w * w;

	  // compute rho^2
	  reduceValue[6] += rho * rho;

	  // min density
	  reduceValue[7] = FMIN( reduceValue[7], rho );

	  // max density
	  reduceValue[8] = FMAX( reduceValue[8], rho );

	} // end for i
      } // end for j
    } // end for k

    real_t norm;
    if (randomForcingEdot == 0) {
      norm = 0;
    } else {
      norm = ( SQRT( SQR(reduceValue[0]) + 
		     reduceValue[1] * dt * randomForcingEdot * 2 * nbCells) - 
	       reduceValue[0] ) / reduceValue[1];
    }

    /**/
    // printf("---- %f %f %f %f %f %f %f %f %f\n",reduceValue[0],reduceValue[1],
    // 	   reduceValue[2],reduceValue[3],reduceValue[4],reduceValue[5],
    // 	   reduceValue[6], reduceValue[7], reduceValue[8]);
    /**/
    
    /* Debug:*/
    /*printf("Random forcing normalistation : %f\n",norm);
    printf("Random forcing E_k %f M_m %f M_v %f \n",
	   0.5*reduceValue[4]/nbCells,
	   SQRT(reduceValue[2]/nbCells),
	   SQRT(reduceValue[3]/nbCells) );*/
     /* */

    return norm;

  } // HydroRunBase::compute_random_forcing_normalization

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  real_t HydroRunBase::compute_random_forcing_normalization(DeviceArray<real_t>  &U, 
							    real_t               dt)
  {
    
    // there are nbRandomForcingReduction=9 values to reduce
    kernel_compute_random_forcing_normalization<RANDOM_FORCING_BLOCK_SIZE>
      <<<randomForcingBlockCount, 
      RANDOM_FORCING_BLOCK_SIZE, 
      RANDOM_FORCING_BLOCK_SIZE*sizeof(real_t)*
      nbRandomForcingReduction>>>(U.data(), 
				  d_randomForcing.data(),
				  d_randomForcingNormalization.data(), 
				  ghostWidth,
				  U.pitch(),
				  U.dimx(),
				  U.dimy(),
				  U.dimz());
    checkCudaError("HydroRunBase compute_random_forcing_normalization error");

    // copy back partial reduction on host
    d_randomForcingNormalization.copyToHost(h_randomForcingNormalization);
    checkCudaError("HydroRunBase d_randomForcingNormalization copy to host error");

    // perform final reduction on host
    real_t* reduceArray = h_randomForcingNormalization.data();
    //const int reduceSize = randomForcingBlockCount;

    real_t reduceValue[nbRandomForcingReduction] = { 0.0, 0.0, 0.0,
						     0.0, 0.0, 0.0,
						     0.0, 0.0, 0.0 };
    // reduceValue[7] is a minimum
    reduceValue[7] = std::numeric_limits<float>::max();

    for (uint i = 0; i < randomForcingBlockCount; ++i)	{
      reduceValue[0] = reduceValue[0] + reduceArray[i];
      reduceValue[1] = reduceValue[1] + reduceArray[i +   randomForcingBlockCount];
      reduceValue[2] = reduceValue[2] + reduceArray[i + 2*randomForcingBlockCount];
      reduceValue[3] = reduceValue[3] + reduceArray[i + 3*randomForcingBlockCount];
      reduceValue[4] = reduceValue[4] + reduceArray[i + 4*randomForcingBlockCount];
      reduceValue[5] = reduceValue[5] + reduceArray[i + 5*randomForcingBlockCount];
      reduceValue[6] = reduceValue[6] + reduceArray[i + 6*randomForcingBlockCount];
      reduceValue[7] = FMIN(reduceValue[7], 
			    reduceArray[i + 7*randomForcingBlockCount]);
      reduceValue[8] = FMAX(reduceValue[8],
			    reduceArray[i + 8*randomForcingBlockCount]);
    }

    real_t norm;
    int64_t nbCells = nx*ny*nz;

    if (randomForcingEdot == 0) {
      norm = 0;
    } else {
      norm = ( SQRT( SQR(reduceValue[0]) + 
		     reduceValue[1] * dt * randomForcingEdot * 2 * nbCells ) - 
	       reduceValue[0] ) / reduceValue[1];
    }
    
    /**/
    // printf("---- kk %f %f %f %f %f %f %f %f %f\n",reduceValue[0],reduceValue[1],
    // 	   reduceValue[2],reduceValue[3],reduceValue[4],reduceValue[5],
    // 	   reduceValue[6], reduceValue[7], reduceValue[8]);
    /**/

    /* Debug: */
    /*printf("Random forcing normalistation : %f\n",norm);
    printf("Random forcing E_k %f M_m %f M_v %f \n",
	   0.5*reduceValue[4]/nbCells,
	   SQRT(reduceValue[2]/nbCells),
	   SQRT(reduceValue[3]/nbCells) );*/
    /* */
    
    return norm;
    
  } // HydroRunBase::compute_random_forcing_normalization
#endif // __CUDACC__


  // =======================================================
  // =======================================================
  void HydroRunBase::add_random_forcing(HostArray<real_t>  &U, 
					real_t             dt,
					real_t             norm)
  {

    (void) dt;

    // this is only available in 3D !
    // sanity check already done long before we get here.

    for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
    	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
    	  real_t rho = U(i,j,k,ID);

    	  // update total energy
    	  U(i,j,k,IP) += U(i,j,k,IU)/rho * h_randomForcing(i,j,k,IX) * norm +
    	    0.5 * SQR( h_randomForcing(i,j,k,IX) * norm );
    	  U(i,j,k,IP) += U(i,j,k,IV)/rho * h_randomForcing(i,j,k,IY) * norm +
    	    0.5 * SQR( h_randomForcing(i,j,k,IY) * norm );
    	  U(i,j,k,IP) += U(i,j,k,IW)/rho * h_randomForcing(i,j,k,IZ) * norm +
    	    0.5 * SQR( h_randomForcing(i,j,k,IZ) * norm );

    	  // update velocity (in fact momentum, so we multiply by rho)
    	  U(i,j,k,IU) += rho * h_randomForcing(i,j,k,IX) * norm;
    	  U(i,j,k,IV) += rho * h_randomForcing(i,j,k,IY) * norm;
    	  U(i,j,k,IW) += rho * h_randomForcing(i,j,k,IZ) * norm;
    	}
      }
    }
	  
  } // HydroRunBase::add_random_forcing

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::add_random_forcing(DeviceArray<real_t>  &U, 
					real_t               dt,
					real_t             norm)
  {

    // this is only available in 3D !
    // sanity check already done long before we get here.

    dim3 dimBlock(ADD_RANDOM_FORCING_3D_DIMX,
		  ADD_RANDOM_FORCING_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, ADD_RANDOM_FORCING_3D_DIMX),
		 blocksFor(jsize, ADD_RANDOM_FORCING_3D_DIMY));

    kernel_add_random_forcing_3d<<< dimGrid, dimBlock >>> 
      (U.data(), 
       d_randomForcing.data(),
       dt,
       norm,
       ghostWidth, U.pitch(),
       U.dimx(), U.dimy(), U.dimz());
    
    checkCudaError("in HydroRunBase :: kernel_add_random_forcing_3d");

  } // HydroRunBase::add_random_forcing
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update(HostArray<real_t>  &U, 
					  HostArray<real_t>  &flux_x, 
					  HostArray<real_t>  &flux_y)
  {

    // only update hydro variables (not magnetic field)
    for (int iVar=0; iVar < 4; iVar++) {
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  
	  U(i,j,iVar) += ( flux_x(i  ,j  ,iVar) -
			   flux_x(i+1,j  ,iVar)  );
	  U(i,j,iVar) += ( flux_y(i  ,j  ,iVar) -
			   flux_y(i  ,j+1,iVar)  );
	  
	} // end for i
      } // end for j
    } // end for iVar
    
  } // HydroRunBase::compute_hydro_update (2D case, CPU)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update(DeviceArray<real_t>  &U, 
					  DeviceArray<real_t>  &flux_x, 
					  DeviceArray<real_t>  &flux_y)
  {
    dim3 dimBlock(HYDRO_UPDATE_2D_DIMX,
		  HYDRO_UPDATE_2D_DIMY);
    dim3 dimGrid(blocksFor(isize, HYDRO_UPDATE_2D_DIMX),
		 blocksFor(jsize, HYDRO_UPDATE_2D_DIMY));

    kernel_hydro_update_2d<<< dimGrid, dimBlock >>> (U.data(), flux_x.data(), flux_y.data(), 
						     ghostWidth, U.pitch(),
						     U.dimx(), U.dimy());
    checkCudaError("in HydroRunBase :: kernel_hydro_update_2d");

  } // HydroRunBase::compute_hydro_update (2D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update(HostArray<real_t>  &U, 
					  HostArray<real_t>  &flux_x, 
					  HostArray<real_t>  &flux_y,
					  HostArray<real_t>  &flux_z)
  {

    // only update hydro variables (not magnetic field)
    for (int iVar=0; iVar < 5; iVar++) {
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    U(i,j,k,iVar) += ( flux_x(i  ,j  ,k  ,iVar) -
			       flux_x(i+1,j  ,k  ,iVar)  );
	    U(i,j,k,iVar) += ( flux_y(i  ,j  ,k  ,iVar) -
			       flux_y(i  ,j+1,k  ,iVar)  );
	    U(i,j,k,iVar) += ( flux_z(i  ,j  ,k  ,iVar) -
			       flux_z(i  ,j  ,k+1,iVar)  );
	    
	  } // end for i
	} // end for j
      } // end for k
    } // end for iVar

  } // HydroRunBase::compute_hydro_update (3D case)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update(DeviceArray<real_t>  &U, 
					  DeviceArray<real_t>  &flux_x, 
					  DeviceArray<real_t>  &flux_y, 
					  DeviceArray<real_t>  &flux_z)
  {
    dim3 dimBlock(HYDRO_UPDATE_3D_DIMX,
		  HYDRO_UPDATE_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, HYDRO_UPDATE_3D_DIMX),
		 blocksFor(jsize, HYDRO_UPDATE_3D_DIMY));

    kernel_hydro_update_3d<<< dimGrid, dimBlock >>> (U.data(), flux_x.data(), flux_y.data(), flux_z.data(),
						     ghostWidth, U.pitch(),
						     U.dimx(), U.dimy(), U.dimz());
    checkCudaError("in HydroRunBase :: kernel_hydro_update_3d");

  } // HydroRunBase::compute_hydro_update (3D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update(HostArray<real_t>  &U, 
					  HostArray<real_t>  &flux_x, 
					  HostArray<real_t>  &flux_y,
					  HostArray<real_t>  &flux_z,
					  ZslabInfo           zSlabInfo)
  {
    
    // start and stop index of current slab (ghosts included)
    int& kStart = zSlabInfo.kStart;
    int& kStop  = zSlabInfo.kStop;
    
    // only update hydro variables (not magnetic field)
    for (int iVar=0; iVar < 5; iVar++) {

      for (int k = kStart+ghostWidth; k < kStop-ghostWidth; k++) {

	// local k index
	int kL = k - kStart;

	if (k<ksize-ghostWidth) {
	  
	  for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	    for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	      
	      U(i,j,k,iVar) += ( flux_x(i  ,j  ,kL  ,iVar) -
				 flux_x(i+1,j  ,kL  ,iVar)  );
	      U(i,j,k,iVar) += ( flux_y(i  ,j  ,kL  ,iVar) -
				 flux_y(i  ,j+1,kL  ,iVar)  );
	      U(i,j,k,iVar) += ( flux_z(i  ,j  ,kL  ,iVar) -
				 flux_z(i  ,j  ,kL+1,iVar)  );
	      
	    } // end for i
	  } // end for j

	} // end if (k<ksize-ghostWidth)

      } // end for k

    } // end for iVar

  } // HydroRunBase::compute_hydro_update (3D case)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update(DeviceArray<real_t>  &U, 
					  DeviceArray<real_t>  &flux_x, 
					  DeviceArray<real_t>  &flux_y, 
					  DeviceArray<real_t>  &flux_z,
					  ZslabInfo             zSlabInfo)
  {
    
    // take care that the last slab might be truncated
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1) {
      zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
    }

    // CUDA kernel call
    dim3 dimBlock(HYDRO_UPDATE_Z_3D_DIMX,
		  HYDRO_UPDATE_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, HYDRO_UPDATE_Z_3D_DIMX),
		 blocksFor(jsize, HYDRO_UPDATE_Z_3D_DIMY));

    kernel_hydro_update_3d_zslab<<< dimGrid, dimBlock >>> (U.data(), 
							   flux_x.data(), 
							   flux_y.data(), 
							   flux_z.data(),
							   ghostWidth, 
							   U.pitch(),
							   U.dimx(), 
							   U.dimy(), 
							   U.dimz(),
							   zSlabInfo);
    checkCudaError("HydroRunBase :: kernel_hydro_update_3d_zslab");

  } // HydroRunBase::compute_hydro_update (3D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update_energy(HostArray<real_t>  &U, 
						 HostArray<real_t>  &flux_x, 
						 HostArray<real_t>  &flux_y)
  {
    
    // only update energy
    for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
      for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	
	U(i,j,IP) += ( flux_x(i  ,j  ,IP) -
		       flux_x(i+1,j  ,IP)  );
	U(i,j,IP) += ( flux_y(i  ,j  ,IP) -
		       flux_y(i  ,j+1,IP)  );
	
      } // end for i
    } // end for j
    
  } // HydroRunBase::compute_hydro_update_energy (2D case, CPU)
  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update_energy(DeviceArray<real_t>  &U, 
						 DeviceArray<real_t>  &flux_x, 
						 DeviceArray<real_t>  &flux_y)
  {
    dim3 dimBlock(HYDRO_UPDATE_2D_DIMX,
		  HYDRO_UPDATE_2D_DIMY);
    dim3 dimGrid(blocksFor(isize, HYDRO_UPDATE_2D_DIMX),
		 blocksFor(jsize, HYDRO_UPDATE_2D_DIMY));
    
    kernel_hydro_update_energy_2d<<< dimGrid, 
      dimBlock >>> (U.data(), flux_x.data(), flux_y.data(), 
		    ghostWidth, U.pitch(),
		    U.dimx(), U.dimy());
    checkCudaError("in HydroRunBase :: kernel_hydro_update_energy_2d");

  } // HydroRunBase::compute_hydro_update_energy (2D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update_energy(HostArray<real_t>  &U, 
						 HostArray<real_t>  &flux_x, 
						 HostArray<real_t>  &flux_y,
						 HostArray<real_t>  &flux_z)
  {

    // only update hydro variable energy
    for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  
	  U(i,j,k,IP) += ( flux_x(i  ,j  ,k  ,IP) -
			   flux_x(i+1,j  ,k  ,IP)  );
	  U(i,j,k,IP) += ( flux_y(i  ,j  ,k  ,IP) -
			   flux_y(i  ,j+1,k  ,IP)  );
	  U(i,j,k,IP) += ( flux_z(i  ,j  ,k  ,IP) -
			   flux_z(i  ,j  ,k+1,IP)  );
	  
	} // end for i
      } // end for j
    } // end for k
    
  } // HydroRunBase::compute_hydro_update_energy (3D case)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update_energy(DeviceArray<real_t>  &U, 
						 DeviceArray<real_t>  &flux_x, 
						 DeviceArray<real_t>  &flux_y, 
						 DeviceArray<real_t>  &flux_z)
  {
    dim3 dimBlock(HYDRO_UPDATE_3D_DIMX,
		  HYDRO_UPDATE_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, HYDRO_UPDATE_3D_DIMX),
		 blocksFor(jsize, HYDRO_UPDATE_3D_DIMY));

    kernel_hydro_update_energy_3d<<< dimGrid, 
      dimBlock >>> (U.data(), flux_x.data(), flux_y.data(), flux_z.data(),
		    ghostWidth, U.pitch(),
		    U.dimx(), U.dimy(), U.dimz());
    checkCudaError("in HydroRunBase :: kernel_hydro_update_energy_3d");

  } // HydroRunBase::compute_hydro_update_energy (3D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update_energy(HostArray<real_t>  &U, 
						 HostArray<real_t>  &flux_x, 
						 HostArray<real_t>  &flux_y,
						 HostArray<real_t>  &flux_z,
						 ZslabInfo           zSlabInfo)
  {

    // start and stop index of current slab (ghosts included)
    int& kStart = zSlabInfo.kStart;
    int& kStop  = zSlabInfo.kStop;

    // only update hydro variable energy
    for (int k=kStart+ghostWidth; k<kStop-ghostWidth; k++) {

      // local k index
      int kL = k - kStart;
      
      if (k<ksize-ghostWidth) {

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    U(i,j,k,IP) += ( flux_x(i  ,j  ,kL  ,IP) -
			     flux_x(i+1,j  ,kL  ,IP)  );
	    U(i,j,k,IP) += ( flux_y(i  ,j  ,kL  ,IP) -
			     flux_y(i  ,j+1,kL  ,IP)  );
	    U(i,j,k,IP) += ( flux_z(i  ,j  ,kL  ,IP) -
			     flux_z(i  ,j  ,kL+1,IP)  );
	    
	  } // end for i
	} // end for j

      } // end if (k<ksize-ghostWidth)

    } // end for k
    
  } // HydroRunBase::compute_hydro_update_energy (3D case, z-slab method)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_hydro_update_energy(DeviceArray<real_t>  &U, 
						 DeviceArray<real_t>  &flux_x, 
						 DeviceArray<real_t>  &flux_y, 
						 DeviceArray<real_t>  &flux_z,
						 ZslabInfo             zSlabInfo)
  {

    // take care that the last slab might be truncated
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1) {
      zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
    }

    dim3 dimBlock(HYDRO_UPDATE_Z_3D_DIMX,
		  HYDRO_UPDATE_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, HYDRO_UPDATE_Z_3D_DIMX),
		 blocksFor(jsize, HYDRO_UPDATE_Z_3D_DIMY));

    kernel_hydro_update_energy_3d_zslab<<< dimGrid, dimBlock >>> (U.data(), 
								  flux_x.data(), 
								  flux_y.data(), 
								  flux_z.data(),
								  ghostWidth, 
								  U.pitch(),
								  U.dimx(), 
								  U.dimy(), 
								  U.dimz(),
								  zSlabInfo);
    checkCudaError("in HydroRunBase :: kernel_hydro_update_energy_3d_zslab");

  } // HydroRunBase::compute_hydro_update_energy (3D case, GPU, z-slab method)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_gravity_predictor(HostArray<real_t> &qPrim,
					       real_t  dt)
  {
    
    if (dimType == TWO_D) {

      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	  qPrim(i,j,IU) += HALF_F * dt * h_gravity(i,j,IX); 
	  qPrim(i,j,IV) += HALF_F * dt * h_gravity(i,j,IY);

	} // end for i
      } // end for j

    } else {

      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    qPrim(i,j,k,IU) += HALF_F * dt * h_gravity(i,j,k,IX); 
	    qPrim(i,j,k,IV) += HALF_F * dt * h_gravity(i,j,k,IY);
	    qPrim(i,j,k,IW) += HALF_F * dt * h_gravity(i,j,k,IZ);
	    
	  } // end for i
	} // end for j
      } // end for k
      
    } // end TWO_D / THREE_D

  } // HydroRunBase::compute_gravity_predictor / CPU version

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_gravity_predictor(DeviceArray<real_t> &qPrim,
					       real_t  dt)
  {

    if (dimType == TWO_D) {
      
      dim3 dimBlock(GRAVITY_PRED_2D_DIMX,
		    GRAVITY_PRED_2D_DIMY);
      dim3 dimGrid(blocksFor(isize, GRAVITY_PRED_2D_DIMX),
		   blocksFor(jsize, GRAVITY_PRED_2D_DIMY));

      kernel_gravity_predictor_2d<<<dimGrid, dimBlock>>>(qPrim.data(), 
							 ghostWidth, 
							 qPrim.pitch(),
							 qPrim.dimx(),
							 qPrim.dimy(),
							 dt);

    } else {

      dim3 dimBlock(GRAVITY_PRED_3D_DIMX,
		    GRAVITY_PRED_3D_DIMY);
      dim3 dimGrid(blocksFor(isize, GRAVITY_PRED_3D_DIMX),
		   blocksFor(jsize, GRAVITY_PRED_3D_DIMY));

      kernel_gravity_predictor_3d<<<dimGrid, dimBlock>>>(qPrim.data(), 
							 ghostWidth, 
							 qPrim.pitch(),
							 qPrim.dimx(),
							 qPrim.dimy(),
							 qPrim.dimz(),
							 dt);

    } // end TWO_D / THREE_D
    
  } // HydroRunBase::compute_gravity_predictor / GPU version
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_gravity_predictor(HostArray<real_t> &qPrim,
					       real_t             dt,
					       ZslabInfo          zSlabInfo)
  {
    
    // start and stop index of current slab (ghosts included)
    int& kStart = zSlabInfo.kStart;
    int& kStop  = zSlabInfo.kStop;
    
    // only update hydro variable energy
    for (int k=kStart+ghostWidth; k<kStop-ghostWidth; k++) {
      
      // local k index
      int kL = k - kStart;
      
      if (k<ksize-ghostWidth) {
	
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    qPrim(i,j,kL,IU) += HALF_F * dt * h_gravity(i,j,k,IX); 
	    qPrim(i,j,kL,IV) += HALF_F * dt * h_gravity(i,j,k,IY);
	    qPrim(i,j,kL,IW) += HALF_F * dt * h_gravity(i,j,k,IZ);
	    
	  } // end for i
	} // end for j

      } // if (k<ksize-ghostWidth)

    } // end for k
    
  } // HydroRunBase::compute_gravity_predictor / CPU version / with zSlab

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_gravity_predictor(DeviceArray<real_t> &qPrim,
					       real_t               dt,
					       ZslabInfo            zSlabInfo)
  {

    // take care that the last slab might be truncated
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1) {
      zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
    }

    dim3 dimBlock(GRAVITY_PRED_Z_3D_DIMX,
		  GRAVITY_PRED_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, GRAVITY_PRED_Z_3D_DIMX),
		 blocksFor(jsize, GRAVITY_PRED_Z_3D_DIMY));
    
    kernel_gravity_predictor_3d_zslab<<<dimGrid, dimBlock>>>(qPrim.data(), 
							     ghostWidth, 
							     qPrim.pitch(),
							     qPrim.dimx(),
							     qPrim.dimy(),
							     qPrim.dimz(),
							     dt,
							     zSlabInfo);
    checkCudaError("in HydroRunBase :: kernel_gravity_predictor_3d_zslab");

  } // HydroRunBase::compute_gravity_predictor / GPU version / with zSlab
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_gravity_source_term(HostArray<real_t> &UNew,
						 HostArray<real_t> &UOld,
						 real_t  dt)
  {
    
    if (dimType == TWO_D) {

      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	  real_t rhoOld = UOld(i,j,ID);
	  real_t rhoNew = UNew(i,j,ID);

	  // update momentum
	  UNew(i,j,IU) += HALF_F * dt * h_gravity(i,j,IX) * (rhoOld + rhoNew); 
	  UNew(i,j,IV) += HALF_F * dt * h_gravity(i,j,IY) * (rhoOld + rhoNew);

	} // end for i
      } // end for j

    } else {

      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    real_t rhoOld = UOld(i,j,k,ID);
	    real_t rhoNew = UNew(i,j,k,ID);
	    
	    // update momentum
	    UNew(i,j,k,IU) += HALF_F * dt * h_gravity(i,j,k,IX) * (rhoOld + rhoNew); 
	    UNew(i,j,k,IV) += HALF_F * dt * h_gravity(i,j,k,IY) * (rhoOld + rhoNew);
	    UNew(i,j,k,IW) += HALF_F * dt * h_gravity(i,j,k,IZ) * (rhoOld + rhoNew);
	    
	  } // end for i
	} // end for j
      } // end for k
      
    } // end TWO_D / THREE_D

  } // HydroRunBase::compute_gravity_source_term / CPU version

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_gravity_source_term(DeviceArray<real_t> &UNew,
						 DeviceArray<real_t> &UOld,
						 real_t  dt)
  {
    
    if (dimType == TWO_D) {

      dim3 dimBlock(GRAVITY_SRC_2D_DIMX,
		    GRAVITY_SRC_2D_DIMY);
      dim3 dimGrid(blocksFor(isize, GRAVITY_SRC_2D_DIMX),
		   blocksFor(jsize, GRAVITY_SRC_2D_DIMY));

      kernel_gravity_source_term_2d<<<dimGrid, dimBlock>>>(UNew.data(), 
							   UOld.data(), 
							   ghostWidth, 
							   UNew.pitch(),
							   UNew.dimx(),
							   UNew.dimy(),
							   dt);

    } else {

      dim3 dimBlock(GRAVITY_SRC_3D_DIMX,
		    GRAVITY_SRC_3D_DIMY);
      dim3 dimGrid(blocksFor(isize, GRAVITY_SRC_3D_DIMX),
		   blocksFor(jsize, GRAVITY_SRC_3D_DIMY));

      kernel_gravity_source_term_3d<<<dimGrid, dimBlock>>>(UNew.data(), 
							   UOld.data(), 
							   ghostWidth, 
							   UNew.pitch(),
							   UNew.dimx(),
							   UNew.dimy(),
							   UNew.dimz(),
							   dt);

    }

  } // HydroRunBase::compute_gravity_source_term / GPU version
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBase::compute_gravity_source_term(HostArray<real_t> &UNew,
						 HostArray<real_t> &UOld,
						 real_t  dt,
						 ZslabInfo zSlabInfo)
  {

    // start and stop index of current slab (ghosts included)
    int& kStart = zSlabInfo.kStart;
    int& kStop  = zSlabInfo.kStop;
    
    // only update hydro variable energy
    for (int k=kStart+ghostWidth; k<kStop-ghostWidth; k++) {
      
      // local k index
      //int kL = k - kStart;
      
      if (k<ksize-ghostWidth) {
	
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    real_t rhoOld = UOld(i,j,k,ID);
	    real_t rhoNew = UNew(i,j,k,ID);
	    
	    // update momentum
	    UNew(i,j,k,IU) += HALF_F * dt * h_gravity(i,j,k,IX) * (rhoOld + rhoNew); 
	    UNew(i,j,k,IV) += HALF_F * dt * h_gravity(i,j,k,IY) * (rhoOld + rhoNew);
	    UNew(i,j,k,IW) += HALF_F * dt * h_gravity(i,j,k,IZ) * (rhoOld + rhoNew);
	    
	  } // end for i
	} // end for j

      } // end if (k<ksize-ghostWidth)
    
    } // end for k

  } // HydroRunBase::compute_gravity_source_term / CPU version / zslab

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBase::compute_gravity_source_term(DeviceArray<real_t> &UNew,
						 DeviceArray<real_t> &UOld,
						 real_t  dt,
						 ZslabInfo zSlabInfo)
  {

    // take care that the last slab might be truncated
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1) {
      zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
    }
    
    dim3 dimBlock(GRAVITY_SRC_Z_3D_DIMX,
		  GRAVITY_SRC_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, GRAVITY_SRC_Z_3D_DIMX),
		 blocksFor(jsize, GRAVITY_SRC_Z_3D_DIMY));
    
    kernel_gravity_source_term_3d_zslab<<<dimGrid, dimBlock>>>(UNew.data(), 
							       UOld.data(), 
							       ghostWidth, 
							       UNew.pitch(),
							       UNew.dimx(),
							       UNew.dimy(),
							       UNew.dimz(),
							       dt,
							       zSlabInfo);
    checkCudaError("in HydroRunBase :: kernel_gravity_source_term_3d_zslab");
    
  } // HydroRunBase::compute_gravity_source_term / GPU version / zslab

#endif // __CUDACC__

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  template<BoundaryLocation boundaryLoc>
  void HydroRunBase::make_boundary(DeviceArray<real_t> &U, BoundaryConditionType bct, dim3 blockCount)
  {

    dim3 threadsPerBlock(MK_BOUND_BLOCK_SIZE, 1, 1);
    if (dimType == THREE_D) {
      threadsPerBlock.x = MK_BOUND_BLOCK_SIZE_3D;
      threadsPerBlock.y = MK_BOUND_BLOCK_SIZE_3D;
    }

    if(bct == BC_DIRICHLET)
      {
	::make_boundary2<BC_DIRICHLET, boundaryLoc>
	  <<<blockCount, threadsPerBlock>>>(U.data(),
					    U.pitch(), 
					    U.dimx(), 
					    U.dimy(),
					    U.dimz(),
					    U.section(),
					    ghostWidth,
					    mhdEnabled);
      }
    else if(bct == BC_NEUMANN)
      {
	::make_boundary2<BC_NEUMANN, boundaryLoc>
	  <<<blockCount, threadsPerBlock>>>(U.data(),
					    U.pitch(), 
					    U.dimx(), 
					    U.dimy(), 
					    U.dimz(),
					    U.section(),
					    ghostWidth,
					    mhdEnabled);
      }
    else if(bct == BC_PERIODIC)
      {
	::make_boundary2<BC_PERIODIC, boundaryLoc>
	  <<<blockCount, threadsPerBlock>>>(U.data(),
					    U.pitch(), 
					    U.dimx(), 
					    U.dimy(), 
					    U.dimz(),
					    U.section(),
					    ghostWidth,
					    mhdEnabled);
      }
    else if(bct == BC_Z_STRATIFIED) 
      {
	bool floor = configMap.getBool("MRI", "floor", false);

	// will / can only be call for ZMIN / ZMAX
	// the actuall cuda kernels are called inside
	::make_boundary2_z_stratified<boundaryLoc>(U.data(),
						   U.pitch(), 
						   U.dimx(), 
						   U.dimy(),
						   U.dimz(),
						   U.section(),
						   floor);
      }
    
  }
#else // CPU version
  template<BoundaryLocation boundaryLoc>
  void HydroRunBase::make_boundary(HostArray<real_t> &U, BoundaryConditionType bct, dim3 blockCount)
  {
    (void) blockCount;

    if(bct == BC_DIRICHLET)
      {
	::make_boundary2<BC_DIRICHLET, boundaryLoc>(U.data(),
						    U.pitch(), 
						    U.dimx(), 
						    U.dimy(),
						    U.dimz(),
						    U.section(),
						    ghostWidth,
						    mhdEnabled);
      }
    else if(bct == BC_NEUMANN)
      {
	::make_boundary2<BC_NEUMANN, boundaryLoc>(U.data(),
						  U.pitch(), 
						  U.dimx(), 
						  U.dimy(),
						  U.dimz(),
						  U.section(),
						  ghostWidth,
						  mhdEnabled);
      }
    else if(bct == BC_PERIODIC)
      {
	::make_boundary2<BC_PERIODIC, boundaryLoc>(U.data(),
						   U.pitch(), 
						   U.dimx(), 
						   U.dimy(),
						   U.dimz(),
						   U.section(),
						   ghostWidth,
						   mhdEnabled);
      }
    else if(bct == BC_Z_STRATIFIED) 
      {
	bool floor = configMap.getBool("MRI", "floor", false);

	// will / can only be call for ZMIN / ZMAX
	::make_boundary2_z_stratified<boundaryLoc>(U.data(),
						   U.pitch(), 
						   U.dimx(), 
						   U.dimy(),
						   U.dimz(),
						   U.section(),
						   floor);
      }
  }
#endif // __CUDACC__

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void HydroRunBase::make_boundaries(DeviceArray<real_t> &U, int idim)
  {

    if (dimType == TWO_D) {
    
      if(idim == XDIR) // horizontal boundaries
	{
	  dim3 blockCount(blocksFor(jsize, MK_BOUND_BLOCK_SIZE), 1, 1);
	  make_boundary<XMIN>(U, boundary_xmin, blockCount);
	  make_boundary<XMAX>(U, boundary_xmax, blockCount);
	}
      else // vertical boundaries
	{
	  dim3 blockCount(blocksFor(isize, MK_BOUND_BLOCK_SIZE),1, 1);
	  make_boundary<YMIN>(U, boundary_ymin, blockCount);
	  make_boundary<YMAX>(U, boundary_ymax, blockCount);
	  if (enableJet)
	    make_jet(U);
	}

    } else { // THREE_D

      if(idim == XDIR) // X-boundaries (size jsize x ksize)
	{
	  dim3 blockCount( blocksFor(jsize, MK_BOUND_BLOCK_SIZE_3D),
			   blocksFor(ksize, MK_BOUND_BLOCK_SIZE_3D), 
			   1);
	  make_boundary<XMIN>(U, boundary_xmin, blockCount);
	  make_boundary<XMAX>(U, boundary_xmax, blockCount);
	}
      else if (idim == YDIR) // Y-boundaries (size isize x ksize)
	{
	  dim3 blockCount( blocksFor(isize, MK_BOUND_BLOCK_SIZE_3D),
			   blocksFor(ksize, MK_BOUND_BLOCK_SIZE_3D), 
			   1);
	  make_boundary<YMIN>(U, boundary_ymin, blockCount);
	  make_boundary<YMAX>(U, boundary_ymax, blockCount);
	}
      else // (idim == ZDIR) // Z-boundaries (size isize x jsize)
	{
	  dim3 blockCount( blocksFor(isize, MK_BOUND_BLOCK_SIZE_3D),
			   blocksFor(jsize, MK_BOUND_BLOCK_SIZE_3D), 
			   1);
	  make_boundary<ZMIN>(U, boundary_zmin, blockCount);
	  make_boundary<ZMAX>(U, boundary_zmax, blockCount);
	  if (enableJet)
	    make_jet(U);	
	}

    }

  } // HydroRunBase::make_boundaries
#else // CPU version
  void HydroRunBase::make_boundaries(HostArray<real_t> &U, int idim)
  {

    if (dimType == TWO_D) {
    
      if(idim == XDIR) // horizontal boundaries
	{
	  make_boundary<XMIN>(U, boundary_xmin, 0);
	  make_boundary<XMAX>(U, boundary_xmax, 0);
	}
      else // vertical boundaries
	{
	  make_boundary<YMIN>(U, boundary_ymin, 0);
	  make_boundary<YMAX>(U, boundary_ymax, 0);
	  if (enableJet)
	    make_jet(U);
	}

    } else { // THREE_D

      if(idim == XDIR) // X-boundaries
	{
	  make_boundary<XMIN>(U, boundary_xmin, 0);
	  make_boundary<XMAX>(U, boundary_xmax, 0);
	}
      else if (idim == YDIR) // Y-boundaries
	{
	  make_boundary<YMIN>(U, boundary_ymin, 0);
	  make_boundary<YMAX>(U, boundary_ymax, 0);
	}
      else // Z-boundaries
	{
	  make_boundary<ZMIN>(U, boundary_zmin,0);
	  make_boundary<ZMAX>(U, boundary_zmax,0);
	  if (enableJet)
	    make_jet(U);
	}

    }

  } //HydroRunBase::make_boundaries
#endif // __CUDACC__

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void HydroRunBase::make_all_boundaries(DeviceArray<real_t> &U)
  {  

    make_boundaries(U,XDIR);
    make_boundaries(U,YDIR);
    if (dimType == THREE_D) {
      make_boundaries(U,ZDIR);
    }

  } // HydroRunBase::make_all_boundaries
#else // CPU version
  void HydroRunBase::make_all_boundaries(HostArray<real_t> &U)
  {

    make_boundaries(U,XDIR);
    make_boundaries(U,YDIR);
    if (dimType == THREE_D) {
      make_boundaries(U,ZDIR);
    }

  } //HydroRunBase::make_all_boundaries
#endif // __CUDACC__

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void HydroRunBase::make_jet(DeviceArray<real_t> &U)
  {

    if (dimType == TWO_D) {
      int blockCount = blocksFor(ijet+2+offsetJet, MAKE_JET_BLOCK_SIZE);
      float4 jetState = {djet, pjet/ (_gParams.gamma0 - 1.0f) + 0.5f * djet * ujet * ujet, 0.0f, djet * ujet};
      ::make_jet_2d<<<blockCount, MAKE_JET_BLOCK_SIZE>>>(U.data(),
							 U.pitch(), 
							 U.section(), 
							 ijet, jetState, offsetJet,
							 ghostWidth);
    } else { // THREE_D
      int blockCount = blocksFor(ijet+ghostWidth+offsetJet, MAKE_JET_BLOCK_SIZE_3D);
      float4 jetState = {djet, pjet/ (_gParams.gamma0 - 1.0f) + 0.5f * djet * ujet * ujet, 0.0f, djet * ujet};
      dim3 jetBlockCount(blockCount, blockCount);
      dim3 jetBlockSize(MAKE_JET_BLOCK_SIZE_3D, MAKE_JET_BLOCK_SIZE_3D);
      ::make_jet_3d<<<jetBlockCount, jetBlockSize>>>(U.data(),
						     U.pitch(), 
						     U.dimy(),
						     U.section(), 
						     ijet, jetState, offsetJet,
						     ghostWidth);
    }
  
  } // make_jet (GPU version)
#else // CPU version
  void HydroRunBase::make_jet(HostArray<real_t> &U)
  {

    if (dimType == TWO_D) {
    
      // matter injection in the middle of the YMAX boundary
      for (int j=0; j<ghostWidth; j++)
	for (int i=ghostWidth+offsetJet; i<ghostWidth+offsetJet+ijet; i++) {
	  U(i,j,ID) = djet;
	  U(i,j,IP) = pjet/(_gParams.gamma0-1.)+0.5*djet*ujet*ujet;
	  U(i,j,IU) = 0.0f;
	  U(i,j,IV) = djet*ujet;
	}
    
      /*for (int i=0; i<2; i++)
	for (int j=jsize/2; j<jsize/2+10; j++) {
	U(i,j,ID) = djet;
	U(i,j,IP) = pjet/(gamma0-1.)+0.5*djet*ujet*ujet;
	U(i,j,IU) = djet*ujet;
	U(i,j,IV) = 0.0f;
	}*/
    } else { // THREE_D

      for (int k=0; k<ghostWidth; ++k)
	for (int j=ghostWidth+offsetJet; j<ghostWidth+offsetJet+ijet; ++j)
	  for (int i=ghostWidth+offsetJet; i<ghostWidth+offsetJet+ijet; ++i) 
	    {
	      U(i,j,k,ID) = djet;
	      U(i,j,k,IP) = pjet/(_gParams.gamma0-1.)+0.5*djet*ujet*ujet;
	      U(i,j,k,IU) = 0.0f;
	      U(i,j,k,IV) = 0.0f;
	      U(i,j,k,IW) = djet*ujet;
	    }
    } // end THREE_D
  }
#endif // make_jet (CPU version)

  // =======================================================
  // =======================================================
  /*
   * main routine to start simulation.
   */
  void HydroRunBase::start() {

    //std::cout << "Starting time integration" << std::endl;
  
  } // HydroRunBase::start

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  DeviceArray<real_t>& HydroRunBase::getData(int nStep) {
    if (nStep % 2 == 0)
      return d_U;
    else
      return d_U2;
  }
#else
  HostArray<real_t>& HydroRunBase::getData(int nStep) {
    if (nStep % 2 == 0)
      return h_U;
    else
      return h_U2;
  }
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  HostArray<real_t>& HydroRunBase::getDataHost(int nStep) {
    if (nStep % 2 == 0)
      return h_U;
    else
      return h_U2;
  } // HydroRunBase::getDataHost


  // =======================================================
  // =======================================================
  /**
   * dump computation results into a file (binary format) for current time.
   *
   * \param[in] U HostArray<real_t> data array to save
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] t time
   */
  void HydroRunBase::outputBin(HostArray<real_t> &U, int nStep, real_t t)
  {

    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;

    std::string filename = outputPrefix+outNum.str()+".bin";
    std::fstream outFile;

    // begin output to file
    outFile.open (filename.c_str(), std::ios_base::out);

    if (dimType == TWO_D) {
      outFile << "Outputting array of size=" <<nx<<" "<<ny<<" "<<NVAR_2D<<std::endl;
    } else {
      outFile << "Outputting array of size=" <<nx<<" "<<ny<<" "<<nz<<" "<<NVAR_3D<<std::endl;   
    }
    outFile << t << " " << _gParams.gamma0 << " " << nStep << std::endl;
    real_t *U_data = U.data();

    if (dimType == TWO_D) {
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  int index=i+isize*j;
	  // write density, energy, d*U, d*V
	  outFile.write(reinterpret_cast<char const *>(&U_data[index]), sizeof(real_t));
	  outFile.write(reinterpret_cast<char const *>(&U_data[index+h_U.section()]), sizeof(real_t));
	  outFile.write(reinterpret_cast<char const *>(&U_data[index+2*h_U.section()]), sizeof(real_t));
	  outFile.write(reinterpret_cast<char const *>(&U_data[index+3*h_U.section()]), sizeof(real_t));
	}
    } else { // THREE_D
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    int index=i+isize*j+isize*jsize*k;
	    // write density, energy, d*U, d*V and d*W
	    outFile.write(reinterpret_cast<char const *>(&U_data[index]), sizeof(real_t));
	    outFile.write(reinterpret_cast<char const *>(&U_data[index+h_U.section()]), sizeof(real_t));
	    outFile.write(reinterpret_cast<char const *>(&U_data[index+2*h_U.section()]), sizeof(real_t));
	    outFile.write(reinterpret_cast<char const *>(&U_data[index+3*h_U.section()]), sizeof(real_t));
	    outFile.write(reinterpret_cast<char const *>(&U_data[index+4*h_U.section()]), sizeof(real_t));
	  }  
    } // end THREE_D
  
    outFile.close();

  } // HydroRunBase::outputBin

  // =======================================================
  // =======================================================
  /**
   * \brief dump computation results into a file (Xsmurf format 2D or 3D, 
   * one line ascii header + binary data) for current time.
   *
   * \param[in] U HostArray<real_t> data array to save
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] iVar Define which variable to save (ID, IP, IU, IV, IW).
   */
  void HydroRunBase::outputXsm(HostArray<real_t> &U, int nStep, ComponentIndex iVar)
  {

    // the following ensures that for hydro simulations we will not try
    // to dump magnetic field component !
    if (iVar<0 || iVar>=nbVar)
      return;

    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;

    std::string filename = outputPrefix+"_"+varPrefix[iVar]+"_"+outNum.str()+".xsm";
    std::fstream outFile;

    // begin output to file
    outFile.open (filename.c_str(), std::ios_base::out);

    if (dimType == TWO_D) {
      outFile << "Binary 1 "<<nx<<"x"<<ny<<" "<< nx*ny <<"(" << sizeof(real_t)<<" byte reals)\n";
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  //int index=i+isize*j;
	  // write density, d*U, d*V and energy
	  outFile.write(reinterpret_cast<char const *>(&U(i,j,iVar)), sizeof(real_t));
	}
    } else { // THREE_D
      outFile << "Binary 1 "<<nx<<"x"<<ny<<"x"<<nz<<" "<< nx*ny*nz <<"(" << sizeof(real_t)<<" byte reals)\n";
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    //int index=i+isize*j;
	    // write density, d*U, d*V, d*W or energy
	    outFile.write(reinterpret_cast<char const *>(&U(i,j,k,iVar)), sizeof(real_t));
	  }
    } // end THREE_D

    outFile.close();

  } // HydroRunBase::outputXsm

  // =======================================================
  // =======================================================
  /**
   * dump computation results into a file (png format) for current time.
   * \param[in] nStep The current time step, used to label results filename. 
   *
   */
  void HydroRunBase::outputPng(HostArray<real_t> &U, int nStep, ComponentIndex iVar)
  {
#ifdef USE_GM

    if (dimType == TWO_D) {
      if (iVar<0 || iVar>=nbVar)
	return;
      
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

      std::ostringstream outNum;
      if ( configMap.getBool("output", "latexAnimation", false) ) { 
	// Latex animation requires not-formated
	outNum << nStep;
      } else {
	outNum.width(7);
	outNum.fill('0');
	outNum << nStep;
      }
      std::fstream outFile;
      
      real_t *data = new real_t[nx*ny];
      
      //create Image object 
      Magick::Image outputImage( Magick::Geometry(nx,ny), Magick::Color("white") );
      std::string filename;
      
      /**
       * output variable iVar
       */
      filename = outputPrefix+"_"+varPrefix[iVar]+"_"+outNum.str()+".png";
      // copy iVar array to data
      int k=0;
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  data[k] = U(i,j,iVar);
	  k++;
	}
      // rescale data to range 0.0 - 1.0
      real_t min, max;
      rescaleToZeroOne(data,nx*ny,min,max);
      // fill outputImage with rescaled buffer
      if ( configMap.getBool("output", "colorPng", false) ) { // color from Red to Blue
	for (int j=0; j<ny; ++j)
	  for (int i=0; i<nx; ++i)
	    if (data[i+nx*j]<0.5) {
	      outputImage.pixelColor(i,j,Magick::ColorRGB(1-2*data[i+nx*j],2*data[i+nx*j],0.0));
	    } else {
	      outputImage.pixelColor(i,j,Magick::ColorRGB(0.0, 2-2*data[i+nx*j],2*data[i+nx*j]-1));
	    }   
      } else {
	for (int j=0; j<ny; ++j)
	  for (int i=0; i<nx; ++i)
	    outputImage.pixelColor(i,j,Magick::ColorGray(data[i+nx*j]));
	
	// negate image to have white=0 and black=1
	outputImage.negate();
      } 
      outputImage.write(filename.c_str());
  
      delete [] data;
    
    } else { // THREE_D
      std::cerr << "PNG output is not available for 3D data !!!" << std::endl;
    }

#else
    (void) U;
    (void) nStep;
    (void) iVar;
    std::cout << "PNG output by ImageMagick++ is not available. Please install it !\n";
#endif // USE_GM

  } // HydroRunBase::outputPng

  // =======================================================
  // =======================================================
  /**
   * dump computation results (conservative variables) into a file
   * (Vtk file format, using class vtkImageData); file extension is vti.
   *
   * \see example use of vtkImageData :
   * http://www.vtk.org/Wiki/VTK/Examples/ImageData/IterateImageData .
   * 
   * \see see also
   * http://permalink.gmane.org/gmane.comp.lib.vtk.user/9624
   * which helped me understand the use of PointData and
   * vtkFloatArray to have named component in the resulting XML vti
   * file.
   *
   * \see HydroRunBaseMpi::outputVtk routine.
   *
   * \param[in] U A reference to a hydro simulation HostArray.
   * \param[in] nStep The current time step, used to label results filename. 
   *
   * There are multiple way to use this routine :
   * - If VTK library is available (i.e. USE_VTK defined), we can
   * choose to use VTK API routines or our own hand-written routine
   * via the parameter : output/outpoutVtkHandWritten from init file.
   * - If VTK library is not installed/available, we fall back on the
   * hand written routine.
   *
   * The VTK API based version is fully configurable : there are 3
   * output mode (ascii, base64 or raw binary), the last two one can
   * optionnaly be zlib-compressed.
   * 
   * Our own hand-written version (usefull when VTK is not
   * installed/available) : only ascii and raw binary are implemented.
   * (no compression, no base64 encoding).
   */
  void HydroRunBase::outputVtk(HostArray<real_t> &U, int nStep)
  {

    // check scalar data type
    bool useDouble = false;
#ifdef USE_VTK
    int dataType = VTK_FLOAT;
    if (sizeof(real_t) == sizeof(double)) {
      useDouble = true;
      dataType = VTK_DOUBLE;
    }
#else
    if (sizeof(real_t) == sizeof(double)) {
      useDouble = true;
    }
#endif // USE_VTK

    // which method will we use to dump data (standard VTK API or hand
    // written) ?
    bool outputVtkHandWritten = configMap.getBool("output", "outputVtkHandWritten", false);
#ifndef USE_VTK
    // USE_VTK is not defined so we need to use the hand-written
    // version
    outputVtkHandWritten = true;
#else
    (void) outputVtkHandWritten;
#endif

    // get output mode (ascii or binary)
    bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

    /*
     * build full path filename
     */
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName     = outputPrefix+"_"+outNum.str();
    std::string filename     = baseName+".vti";
    std::string filenameFull = outputDir+"/"+filename;
    
    if (!outputVtkHandWritten) {
      /* use the VTK library API ! */

 #ifdef USE_VTK

      /*
       * write data with XML binary image data using VTK API.
       */
      
      // create a vtkImageData object
      vtkSmartPointer<vtkImageData> imageData = 
	vtkSmartPointer<vtkImageData>::New();
#if HAVE_VTK6
      imageData->SetExtent(0,nx-1,0,ny-1,0,nz-1);
#else
      imageData->SetDimensions(nx, ny, nz);
#endif
      imageData->SetOrigin(0.0, 0.0, 0.0);
      imageData->SetSpacing(1.0,1.0,1.0);
      
#if HAVE_VTK6
#else
      imageData->SetNumberOfScalarComponents(nbVar);
      if (useDouble)
	imageData->SetScalarTypeToDouble();
      else
	imageData->SetScalarTypeToFloat();
      //imageData->AllocateScalars();
#endif
      
      vtkPointData *pointData = imageData->GetPointData();
      
      /*
       * NOTICE :
       * we add array to the pointData object so that we can have named
       * components, which appears to be not possible (when doing a
       * simple imageData->AllocateScalars() and fill data with
       * SetScalarComponentFromFloat or GetScalarPointer for example.
       *
       */
      
      // add density array
      vtkSmartPointer<vtkDataArray> densityArray = 
	vtkDataArray::CreateDataArray(dataType);
      densityArray->SetNumberOfComponents( 1 );
      densityArray->SetNumberOfTuples( nx*ny*nz );
      densityArray->SetName( "density" );
      
      // add energy array
      vtkSmartPointer<vtkDataArray> energyArray = 
	vtkDataArray::CreateDataArray(dataType);
      energyArray->SetNumberOfComponents( 1 );
      energyArray->SetNumberOfTuples( nx*ny*nz );
      energyArray->SetName( "energy" );
      
      // add momentum arrays
      vtkSmartPointer<vtkDataArray> mxArray = 
	vtkDataArray::CreateDataArray(dataType);
      mxArray->SetNumberOfComponents( 1 );
      mxArray->SetNumberOfTuples( nx*ny*nz );
      mxArray->SetName( "mx" );
      vtkSmartPointer<vtkDataArray> myArray = 
	vtkDataArray::CreateDataArray(dataType);
      myArray->SetNumberOfComponents( 1 );
      myArray->SetNumberOfTuples( nx*ny*nz );
      myArray->SetName( "my" );
      vtkSmartPointer<vtkDataArray> mzArray = 
	vtkDataArray::CreateDataArray(dataType);
      mzArray->SetNumberOfComponents( 1 );
      mzArray->SetNumberOfTuples( nx*ny*nz );
      mzArray->SetName( "mz" );
      
      // magnetic component (MHD only)
      vtkSmartPointer<vtkDataArray> bxArray = 
	vtkDataArray::CreateDataArray(dataType);
      bxArray->SetNumberOfComponents( 1 );
      bxArray->SetName( "bx" );
      vtkSmartPointer<vtkDataArray> byArray = 
	vtkDataArray::CreateDataArray(dataType);
      byArray->SetNumberOfComponents( 1 );
      byArray->SetName( "by" );
      vtkSmartPointer<vtkDataArray> bzArray = 
	vtkDataArray::CreateDataArray(dataType);
      bzArray->SetNumberOfComponents( 1 );
      bzArray->SetName( "bz" );
      if (mhdEnabled) {
	// do memory allocation
	bxArray->SetNumberOfTuples( nx*ny*nz );
	byArray->SetNumberOfTuples( nx*ny*nz );
	bzArray->SetNumberOfTuples( nx*ny*nz );
      }
      
      // fill the vtkImageData with scalars from U
      if (dimType == TWO_D) {
	for(int j= ghostWidth; j < jsize-ghostWidth; j++)
	  for(int i = ghostWidth; i < isize-ghostWidth; i++) {
	    int index = i-ghostWidth + nx*(j-ghostWidth);
	    densityArray->SetTuple1(index, U(i,j,ID)); 
	    energyArray->SetTuple1(index, U(i,j,IP));
	    mxArray->SetTuple1(index, U(i,j,IU));
	    myArray->SetTuple1(index, U(i,j,IV));
	    if (mhdEnabled) {
	      mzArray->SetTuple1(index, U(i,j,IW));
	      bxArray->SetTuple1(index, U(i,j,IA));
	      byArray->SetTuple1(index, U(i,j,IB));
	      bzArray->SetTuple1(index, U(i,j,IC));
	    }
	  }
      } else { // THREE_D
	for(int k= ghostWidth; k < ksize-ghostWidth; k++)
	  for(int j= ghostWidth; j < jsize-ghostWidth; j++)
	    for(int i = ghostWidth; i < isize-ghostWidth; i++) {
	      int index = i-ghostWidth + nx*(j-ghostWidth) + nx*ny*(k-ghostWidth);
	      densityArray->SetTuple1(index, U(i,j,k,ID)); 
	      energyArray->SetTuple1(index, U(i,j,k,IP));
	      mxArray->SetTuple1(index, U(i,j,k,IU));
	      myArray->SetTuple1(index, U(i,j,k,IV));
	      mzArray->SetTuple1(index, U(i,j,k,IW));
	      if (mhdEnabled) {
		bxArray->SetTuple1(index, U(i,j,k,IA));
		byArray->SetTuple1(index, U(i,j,k,IB));
		bzArray->SetTuple1(index, U(i,j,k,IC));
	      }
	    }
      }
      
      // add filled data arrays to point data object
      pointData->AddArray( densityArray );
      pointData->AddArray( energyArray );
      pointData->AddArray( mxArray );
      pointData->AddArray( myArray );
      if (dimType == THREE_D and !mhdEnabled)
	pointData->AddArray( mzArray );
      if (mhdEnabled) {
	pointData->AddArray( mzArray );
	pointData->AddArray( bxArray );
	pointData->AddArray( byArray );
	pointData->AddArray( bzArray );
      }
      
      // create image writer
      vtkSmartPointer<vtkXMLImageDataWriter> writer = 
	vtkSmartPointer<vtkXMLImageDataWriter>::New();
#if HAVE_VTK6
      writer->SetInputData(imageData);
#else
      writer->SetInput(imageData);
#endif
      writer->SetFileName(filenameFull.c_str());
      if (outputVtkAscii)
	writer->SetDataModeToAscii();
      
      // do we want base 64 encoding ?? probably not
      // since is it better for data reload (simulation restart). By the
      // way reading/writing raw binary is faster since we don't need to
      // encode !
      bool outputVtkBase64Encoding = configMap.getBool("output", "outputVtkBase64", false);
      if (!outputVtkBase64Encoding)
	writer->EncodeAppendedDataOff();
      
      // if using raw binary or base64, data can be zlib-compressed
      bool outputVtkCompression = configMap.getBool("output", "outputVtkCompression", true);
      if (!outputVtkCompression) {
	writer->SetCompressor(NULL);
      }
      writer->Write();
      
#endif // USE_VTK
      
    } else { // use the hand written routine (no need to have VTK
	     // installed)
      
      /*
       * Hand written procedure (no VTK library linking required).
       * Write XML imageData using either :
       * - ascii 
       * - raw binary
       *
       * Each hydrodynamics field component is written in a separate <DataArray>.
       * Magnetic field component are dumped if mhdEnabled is true !
       */
      std::fstream outFile;
      outFile.open(filenameFull.c_str(), std::ios_base::out);
      
      // domain extent
      int xmin=0, xmax=nx-1, ymin=0, ymax=ny-1, zmin=0, zmax=nz-1;
     
      if (dimType == TWO_D) {
	zmax = 0;
      }

      // if writing raw binary data (file does not respect XML standard)
      if (outputVtkAscii)
	outFile << "<?xml version=\"1.0\"?>" << std::endl;
      
      // write xml data header
      if (isBigEndian())
	outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
      else
	outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
      
      outFile << "  <ImageData WholeExtent=\""
	      << xmin << " " << xmax << " " 
	      << ymin << " " << ymax << " " 
	      << zmin << " " << zmax << ""
	      << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">" << std::endl;
      outFile << "  <Piece Extent=\"" 
	      << xmin << " " << xmax << " " 
	      << ymin << " " << ymax << " " 
	      << zmin << " " << zmax << ""
	      << "\">" << std::endl;
      outFile << "    <PointData>" << std::endl;
      
      if (outputVtkAscii) {
	
	// write ascii data
	if (dimType == TWO_D) {
	  int imin = ghostWidth;
	  int jmin = ghostWidth;
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    if (useDouble) {
	      outFile << "      <DataArray type=\"Float64\" Name=\"" << varNames[iVar]
		      << "\" format=\"ascii\" >" << std::endl;
	    } else {
	      outFile << "      <DataArray type=\"Float32\" Name=\"" << varNames[iVar]
		      << "\" format=\"ascii\" >" << std::endl;
	    }
	    for(int j= jmin; j < jsize-ghostWidth; j++) {
	      for(int i = imin; i < isize-ghostWidth; i++) {
		outFile << std::setprecision (12) << U(i,j,iVar) << " ";
	      }
	      outFile << std::endl;
	    }
	    outFile << "      </DataArray>" << std::endl;
	  }
	} else { // THREE_D
	  int imin = ghostWidth;
	  int jmin = ghostWidth;
	  int kmin = ghostWidth;
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    if (useDouble) {
	      outFile << "      <DataArray type=\"Float64\" Name=\"" << varNames[iVar]
		      << "\" format=\"ascii\" >" << std::endl;
	    } else {
	      outFile << "      <DataArray type=\"Float32\" Name=\"" << varNames[iVar]
		      << "\" format=\"ascii\" >" << std::endl;
	    }
	    for(int k= kmin; k < ksize-ghostWidth; k++) {
	      for(int j= jmin; j < jsize-ghostWidth; j++) {
		for(int i = imin; i < isize-ghostWidth; i++) {
		  outFile << std::setprecision(12) << U(i,j,k,iVar) << " ";
		}
		outFile << std::endl;
	      }
	    }
	    outFile << "      </DataArray>" << std::endl;
	  }
	} // end write ascii data
	
	outFile << "    </PointData>" << std::endl;
	outFile << "    <CellData>" << std::endl;
	outFile << "    </CellData>" << std::endl;
	outFile << "  </Piece>" << std::endl;
	outFile << "  </ImageData>" << std::endl;
	outFile << "</VTKFile>" << std::endl;
	
      } else { // do it using appended format raw binary (no base 64 encoding)
	
	if (dimType == TWO_D) {
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    if (useDouble) {
	      outFile << "     <DataArray type=\"Float64\" Name=\"" ;
	    } else {
	      outFile << "     <DataArray type=\"Float32\" Name=\"" ;
	    }

	    outFile << varNames[iVar]
		    << "\" format=\"appended\" offset=\"" << iVar*nx*ny*sizeof(real_t)+iVar*sizeof(unsigned int) <<"\" />" << std::endl;
	  }
	} else { // THREE_D
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    if (useDouble) {
	      outFile << "     <DataArray type=\"Float64\" Name=\"";
	    } else {
	      outFile << "     <DataArray type=\"Float32\" Name=\"";
	    }
	    outFile << varNames[iVar]
		    << "\" format=\"appended\" offset=\"" << iVar*nx*ny*nz*sizeof(real_t)+iVar*sizeof(unsigned int) <<"\" />" << std::endl;
	  }
	}
	outFile << "    </PointData>" << std::endl;
	outFile << "    <CellData>" << std::endl;
	outFile << "    </CellData>" << std::endl;
	outFile << "  </Piece>" << std::endl;
	outFile << "  </ImageData>" << std::endl;
	
	outFile << "  <AppendedData encoding=\"raw\">" << std::endl;
	
	// write the leading undescore
	outFile << "_";
	// then write heavy data (column major format)
	if (dimType == TWO_D) {
	  unsigned int nbOfWords = nx*ny*sizeof(real_t);
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	    for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	      for (int i=ghostWidth; i<isize-ghostWidth; i++) {
		real_t tmp = U(i,j,iVar);
		outFile.write((char *)&tmp,sizeof(real_t));
	      }
	  }
	} else { // THREE_D
	  unsigned int nbOfWords = nx*ny*nz*sizeof(real_t);
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	    for (int k=ghostWidth; k<ksize-ghostWidth; k++) 
	      for (int j=ghostWidth; j<jsize-ghostWidth; j++) 
		for (int i=ghostWidth; i<isize-ghostWidth; i++) {
		  real_t tmp = U(i,j,k,iVar);
		  outFile.write((char *)&tmp,sizeof(real_t));
		}
	  }
	}
	
	outFile << "  </AppendedData>" << std::endl;
	outFile << "</VTKFile>" << std::endl;
	
      } // end raw binary write
      
      outFile.close();
      
    } // end hand written version
    
  } // HydroRunBase::outputVtk
  
  // =======================================================
  // =======================================================
  /**
   * dump debug array into a file (Vtk file format, using class
   * vtkImageData); file extension is vti
   *
   * use hand written VTK format, in ascii (to ease file comparison) 
   *
   * \see HydroRunBase::outputVtk routine.
   *
   * \param[in] data A reference to a HostArray for debug.
   * \param[in] suffix a string appended to filename.
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] ghostIncluded Do we want ghost cell to be saved as well.
   *
   */
  void HydroRunBase::outputVtkDebug(HostArray<real_t> &data, 
				    const std::string suffix, 
				    int nStep, 
				    bool ghostIncluded)
  {

    // check scalar data type
    bool useDouble = false;
#ifdef USE_VTK
    //int dataType = VTK_FLOAT;
    if (sizeof(real_t) == sizeof(double)) {
      useDouble = true;
      //dataType = VTK_DOUBLE;
    }
#endif // USE_VTK

    // which method will we use to dump data (standard VTK API or hand
    // written) ?
    //bool outputVtkHandWritten = true;

    // get output mode (ascii or binary)
    bool outputVtkAscii = true;

    int nbVarDebug = data.nvar();

    /*
     * build full path filename
     */
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName     = outputPrefix+"_debug_"+suffix+outNum.str();
    std::string filename     = baseName+".vti";
    std::string filenameFull = outputDir+"/"+filename;
    
    /*
     * Hand written procedure (no VTK library linking required).
     * Write XML imageData using either :
     * - ascii 
     * - raw binary
     * Each hydrodynamics field component is written in a separate <DataArray>.
     * Magnetic field component are dumped if mhdEnabled is true !
     */
    std::fstream outFile;
    outFile.open(filenameFull.c_str(), std::ios_base::out);
    
    // domain extent
    int xmin=0, xmax=data.dimx()-1;
    int ymin=0, ymax=data.dimy()-1;
    int zmin=0, zmax=data.dimz()-1;
    if (!ghostIncluded) {
      xmax -= 2*ghostWidth;
      ymax -= 2*ghostWidth;
      zmax -= 2*ghostWidth;
    }
    
    if (dimType == TWO_D) {
      zmax = 0;
    }

    // if writing raw binary data (file does not respect XML standard)
    if (outputVtkAscii)
      outFile << "<?xml version=\"1.0\"?>" << std::endl;
      
    // write xml data header
    if (isBigEndian())
      outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
    else
      outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    
    outFile << "  <ImageData WholeExtent=\""
	    << xmin << " " << xmax << " " 
	    << ymin << " " << ymax << " " 
	    << zmin << " " << zmax << ""
	    << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">" << std::endl;
    outFile << "  <Piece Extent=\"" 
	    << xmin << " " << xmax << " " 
	    << ymin << " " << ymax << " " 
	    << zmin << " " << zmax << ""
	    << "\">" << std::endl;
    outFile << "    <PointData>" << std::endl;
    
    if (outputVtkAscii) {
      
      // write ascii data
      if (dimType == TWO_D) {
	int imin = ghostWidth;
	int jmin = ghostWidth;
	int imax = data.dimx()-ghostWidth;
	int jmax = data.dimy()-ghostWidth;
	if (ghostIncluded) {
	  imin = 0;
	  jmin = 0;
	  imax = isize;
	  jmax = jsize;
	}

	for (int iVar=0; iVar<nbVarDebug; iVar++) {
	  if (useDouble) {
	    outFile << "      <DataArray type=\"Float64\" Name=\"";
	  } else {
	    outFile << "      <DataArray type=\"Float32\" Name=\"";
	  }
	  outFile << "debug" 
		  << iVar  << "\" format=\"ascii\" >" << std::endl;
	  for(int j= jmin; j < jmax; j++) {
	    for(int i = imin; i < imax; i++) {
	      outFile << std::setprecision (12) << data(i,j,iVar) << " ";
	    }
	    outFile << std::endl;
	  }
	  outFile << "      </DataArray>" << std::endl;
	}
      } else { // THREE_D
	int imin = ghostWidth;
	int jmin = ghostWidth;
	int kmin = ghostWidth;
	int imax = data.dimx()-ghostWidth;
	int jmax = data.dimy()-ghostWidth;
	int kmax = data.dimz()-ghostWidth;
	if (ghostIncluded) {
	  imin = 0;
	  jmin = 0;
	  kmin = 0;
	  imax = isize;
	  jmax = jsize;
	  kmax = ksize;
	}
	for (int iVar=0; iVar<nbVarDebug; iVar++) {
	  if (useDouble) {
	    outFile << "      <DataArray type=\"Float64\" Name=\"";
	  } else {
	    outFile << "      <DataArray type=\"Float32\" Name=\"";
	  }
	  outFile << "debug"
		  << iVar  << "\" format=\"ascii\" >" << std::endl;	
	  for(int k= kmin; k < kmax; k++) {
	    for(int j= jmin; j < jmax; j++) {
	      for(int i = imin; i < imax; i++) {
		outFile << std::setprecision (12) << data(i,j,k,iVar) << " ";
	      }
	      outFile << std::endl;
	    }
	  }
	  outFile << "      </DataArray>" << std::endl;
	}
      } // end write ascii data
      
      outFile << "    </PointData>" << std::endl;
      outFile << "    <CellData>" << std::endl;
      outFile << "    </CellData>" << std::endl;
      outFile << "  </Piece>" << std::endl;
      outFile << "  </ImageData>" << std::endl;
      outFile << "</VTKFile>" << std::endl;
      
    } 
    
    outFile.close();
    
  } // HydroRunBase::outputVtkDebug

  // =======================================================
  // =======================================================
  /**
   * dump debug array into a file (wrapper to methods 
   * HydroRunBase::outputVtkDebug and outputHdf5Debug).
   *
   * \param[in] data A reference to a HostArray for debug.
   * \param[in] suffix a string appended to filename.
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] ghostIncluded Do we want ghost cell to be saved as well.
   *
   */
  void HydroRunBase::outputDebug(HostArray<real_t> &data, 
				 const std::string suffix, 
				 int nStep)
  {

    bool vtk  = configMap.getBool("debug","vtk",false);
    bool hdf5 = configMap.getBool("debug","hdf5",false);
    bool ghostIncluded = configMap.getBool("debug","ghostIncluded",false);

    if (vtk)
      outputVtkDebug(data, suffix, nStep, ghostIncluded);

    if (hdf5)
      outputHdf5Debug(data, suffix, nStep);

  } // HydroRunBase::outputDebug

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  /**
   * dump debug array into a file (wrapper to methods 
   * HydroRunBase::outputVtkDebug and outputHdf5Debug).
   *
   * \param[in] data A reference to a HostArray for debug.
   * \param[in] suffix a string appended to filename.
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] ghostIncluded Do we want ghost cell to be saved as well.
   *
   */
  void HydroRunBase::outputDebug(DeviceArray<real_t> &data, 
				 const std::string suffix, 
				 int nStep)
  {

    bool vtk  = configMap.getBool("debug","vtk",false);
    bool hdf5 = configMap.getBool("debug","hdf5",false);
    bool ghostIncluded = configMap.getBool("debug","ghostIncluded",false);


    if (vtk || hdf5)
      data.copyToHost(h_debug);
    
    if (vtk)
      outputVtkDebug(h_debug, suffix, nStep, ghostIncluded);
    
    if (hdf5)
      outputHdf5Debug(h_debug, suffix, nStep);
    
  } // HydroRunBase::outputDebug
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  /**
   * dump computation results (conservative variables) into a file
   * (HDF5 file format) file extension is h5. File can be viewed by
   * hdfview; see also h5dump.
   * \sa writeXdmfForHdf5Wrapper this routine write a Xdmf wrapper file for paraview.
   *
   * \note Take care that HostArray use column-format ordering,
   * whereas C-language and so C API of HDF5 uses raw-major ordering
   * !!! We need to invert dimensions.
   *
   * \note This output routine is the only one to save all fields in a
   * single file.
   *
   * \param[in] U A reference to a hydro simulation HostArray
   * \param[in] nStep The current time step, used to label results filename. 
   *
   * If library HDF5 is not available, do nothing.
   */
  void HydroRunBase::outputHdf5(HostArray<real_t> &U, int nStep, bool ghostIncluded)
  {
#ifdef USE_HDF5
    herr_t status;
    
    // make filename string
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName         = outputPrefix+"_"+outNum.str();
    std::string hdf5Filename     = baseName+".h5";
    std::string hdf5FilenameFull = outputDir+"/"+hdf5Filename;
   
    // data size actually written on disk
    int nxg = nx;
    int nyg = ny;
    int nzg = nz;
    if (ghostIncluded) {
      nxg += 2*ghostWidth;
      nyg += 2*ghostWidth;
      nzg += 2*ghostWidth;
    }

    /*
     * write HDF5 file
     */
    // Create a new file using default properties.
    hid_t file_id = H5Fcreate(hdf5FilenameFull.c_str(), H5F_ACC_TRUNC |  H5F_ACC_DEBUG, H5P_DEFAULT, H5P_DEFAULT);

    // Create the data space for the dataset in memory and in file.
    hsize_t  dims_memory[3];
    hsize_t  dims_file[3];
    hid_t dataspace_memory, dataspace_file;
    if (dimType == TWO_D) {
      dims_memory[0] = U.dimy(); 
      dims_memory[1] = U.dimx();
      dims_file[0] = nyg;
      dims_file[1] = nxg;
      dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
      dataspace_file   = H5Screate_simple(2, dims_file  , NULL);
    } else {
      dims_memory[0] = U.dimz(); 
      dims_memory[1] = U.dimy();
      dims_memory[2] = U.dimx();
      dims_file[0] = nzg;
      dims_file[1] = nyg;
      dims_file[2] = nxg;
      dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
      dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
    }

    // Create the datasets.
    hid_t dataType;
    if (sizeof(real_t) == sizeof(float))
      dataType = H5T_NATIVE_FLOAT;
    else
      dataType = H5T_NATIVE_DOUBLE;
    

    // select data with or without ghost zones
    if (ghostIncluded) {
      if (dimType == TWO_D) {
	hsize_t  start[2] = {0, 0}; // ghost zone width
	hsize_t stride[2] = {1, 1};
	hsize_t  count[2] = {(hsize_t) nyg, (hsize_t) nxg};
	hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = {0, 0, 0}; // ghost zone width
	hsize_t stride[3] = {1, 1, 1};
	hsize_t  count[3] = {(hsize_t) nzg, (hsize_t) nyg, (hsize_t) nxg};
	hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      }      
    } else {
      if (dimType == TWO_D) {
	hsize_t  start[2] = {(hsize_t) ghostWidth, (hsize_t) ghostWidth}; // ghost zone width
	hsize_t stride[2] = {1, 1};
	hsize_t  count[2] = {(hsize_t) ny, (hsize_t) nx};
	hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = {(hsize_t) ghostWidth, (hsize_t) ghostWidth, (hsize_t) ghostWidth}; // ghost zone width
	hsize_t stride[3] = {1, 1, 1};
	hsize_t  count[3] = {(hsize_t) nz, (hsize_t) ny, (hsize_t) nx};
	hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
      }
    }

    /*
     * property list for compression
     */
    // get compression level (0=no compression; 9 is highest level of compression)
    int compressionLevel = configMap.getInteger("output", "outputHdf5CompressionLevel", 0);
    if (compressionLevel < 0 or compressionLevel > 9) {
      std::cerr << "Invalid value for compression level; must be an integer between 0 and 9 !!!" << std::endl;
      std::cerr << "compression level is then set to default value 0; i.e. no compression !!" << std::endl;
      compressionLevel = 0;
    }

    hid_t propList_create_id = H5Pcreate(H5P_DATASET_CREATE);

    if (dimType == TWO_D) {
      const hsize_t chunk_size2D[2] = {(hsize_t) ny, (hsize_t) nx};
      H5Pset_chunk (propList_create_id, 2, chunk_size2D);
    } else { // THREE_D
      const hsize_t chunk_size3D[3] = {(hsize_t) nz, (hsize_t) ny, (hsize_t) nx};
      H5Pset_chunk (propList_create_id, 3, chunk_size3D);
    }
    H5Pset_shuffle (propList_create_id);
    H5Pset_deflate (propList_create_id, compressionLevel);
    
    /*
     * write heavy data to HDF5 file
     */
    real_t* data;

    // write density
    hid_t dataset_id = H5Dcreate2(file_id, "/density", dataType, dataspace_file, 
				  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(U(0,0,ID));
    else
      data = &(U(0,0,0,ID));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);

    // write total energy
    dataset_id = H5Dcreate2(file_id, "/energy", dataType, dataspace_file, 
				  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(U(0,0,IP));
    else
      data = &(U(0,0,0,IP));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
    // write momentum X
    dataset_id = H5Dcreate2(file_id, "/momentum_x", dataType, dataspace_file, 
				  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(U(0,0,IU));
    else
      data = &(U(0,0,0,IU));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
    // write momentum Y
    dataset_id = H5Dcreate2(file_id, "/momentum_y", dataType, dataspace_file, 
				  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(U(0,0,IV));
    else
      data = &(U(0,0,0,IV));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
    // write momentum Z (only if 3D hydro)
    if (dimType == THREE_D and !mhdEnabled) {
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      data = &(U(0,0,0,IW));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    }
    
    if (mhdEnabled) {
      // write momentum mz
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      if (dimType == TWO_D)
	data = &(U(0,0,IW));
      else
	data = &(U(0,0,0,IW));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
      // write magnetic field components
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_x", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      if (dimType == TWO_D)
	data = &(U(0,0,IA));
      else
	data = &(U(0,0,0,IA));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_y", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      if (dimType == TWO_D)
	data = &(U(0,0,IB));
       else
	 data = &(U(0,0,0,IB));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      if (dimType == TWO_D)
	data = &(U(0,0,IC));
      else
	data = &(U(0,0,0,IC));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
    }

    // write time step as an attribute to root group
    hid_t ds_id;
    hid_t attr_id;
    {
       ds_id   = H5Screate(H5S_SCALAR);
       attr_id = H5Acreate2(file_id, "time step", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &nStep);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    // write total time 
    {
      double timeValue = (double) totalTime;

      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "total time", H5T_NATIVE_DOUBLE, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &timeValue);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write geometry information (just to be consistent)
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "nx", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &nx);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "ny", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &ny);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "nz", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &nz);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write information about ghost zone
    {
      int tmpVal = ghostIncluded ? 1 : 0;
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "ghost zone included", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &tmpVal);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write date as an attribute to root group
    std::string dataString = current_date();
    const char *dataChar = dataString.c_str();
    hsize_t   dimsAttr[1] = {1};
    hid_t type = H5Tcopy (H5T_C_S1);
    status = H5Tset_size (type, H5T_VARIABLE);
    hid_t root_id = H5Gopen2(file_id, "/", H5P_DEFAULT);
    hid_t dataspace_id = H5Screate_simple(1, dimsAttr, NULL);
    attr_id = H5Acreate2(root_id, "creation date", type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, type, &dataChar);
    status = H5Aclose(attr_id);
    status = H5Gclose(root_id);
    status = H5Tclose(type);
    status = H5Sclose(dataspace_id);

    // close/release resources.
    H5Pclose(propList_create_id);
    H5Sclose(dataspace_memory);
    H5Sclose(dataspace_file);
    H5Dclose(dataset_id);
    H5Fflush(file_id, H5F_SCOPE_LOCAL);
    H5Fclose(file_id);

    (void) status;

#else

    (void) U;
    (void) nStep;
    (void) ghostIncluded;

    std::cerr << "[HydroRunBase::outputHdf5] Hdf5 not enabled !!!\n";

#endif // USE_HDF5
  } // HydroRunBase::outputHdf5

  // =======================================================
  // =======================================================
  /**
   * Dump array for debug with ghost zones into a HDF5 file.
   * (HDF5 file format) file extension is h5. File can be viewed by
   * hdfview; see also h5dump.
   *
   * \sa writeXdmfForHdf5Wrapper this routine write a Xdmf wrapper file for paraview.
   *
   * \note Take care that HostArray use column-format ordering,
   * whereas C-language and so C API of HDF5 uses raw-major ordering
   * !!! We need to invert dimensions.
   *
   * \note This output routine is the only one to save all fields in a
   * single file.
   *
   * \param[in] data A reference to a HostArray
   * \param[in] suffix a string appended to filename.
   * \param[in] nStep The current time step, used to label results filename. 
   *
   * If library HDF5 is not available, do nothing.
   */
  void HydroRunBase::outputHdf5Debug(HostArray<real_t> &data, const std::string suffix, int nStep)
  {
#ifdef USE_HDF5
    herr_t status;
    
    int nbVarDebug = data.nvar();
    
    // make filename string
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName         = outputPrefix+"_debug_"+suffix+outNum.str();
    std::string hdf5Filename     = baseName+".h5";
    std::string hdf5FilenameFull = outputDir+"/"+hdf5Filename;
   
    // data size actually written on disk
    int nxg = data.dimx();
    int nyg = data.dimy();
    int nzg = data.dimz();

    /*
     * write HDF5 file
     */
    // Create a new file using default properties.
    hid_t file_id = H5Fcreate(hdf5FilenameFull.c_str(), H5F_ACC_TRUNC |  H5F_ACC_DEBUG, H5P_DEFAULT, H5P_DEFAULT);

    // Create the data space for the dataset in memory and in file.
    hsize_t  dims_memory[3];
    hsize_t  dims_file[3];
    hid_t dataspace_memory, dataspace_file;
    if (nzg == 1) {
      dims_memory[0] = data.dimy(); 
      dims_memory[1] = data.dimx();
      dims_file[0] = nyg;
      dims_file[1] = nxg;
      dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
      dataspace_file   = H5Screate_simple(2, dims_file  , NULL);
    } else {
      dims_memory[0] = data.dimz(); 
      dims_memory[1] = data.dimy();
      dims_memory[2] = data.dimx();
      dims_file[0] = nzg;
      dims_file[1] = nyg;
      dims_file[2] = nxg;
      dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
      dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
    }

    // Create the datasets.
    hid_t dataType;
    if (sizeof(real_t) == sizeof(float))
      dataType = H5T_NATIVE_FLOAT;
    else
      dataType = H5T_NATIVE_DOUBLE;
    

    // select data with or without ghost zones
    if (nzg == 1) {
      hsize_t  start[2] = {0, 0}; // ghost zone width
      hsize_t stride[2] = {1, 1};
      hsize_t  count[2] = {(hsize_t) nyg, (hsize_t) nxg};
      hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
      status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
    } else {
      hsize_t  start[3] = {0, 0, 0}; // ghost zone width
      hsize_t stride[3] = {1, 1, 1};
      hsize_t  count[3] = {(hsize_t) nzg, (hsize_t) nyg, (hsize_t) nxg};
      hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
      status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      }      

    /*
     * property list for compression
     */
    // get compression level (0=no compression; 9 is highest level of compression)
    int compressionLevel = configMap.getInteger("output", "outputHdf5CompressionLevel", 0);
    if (compressionLevel < 0 or compressionLevel > 9) {
      std::cerr << "Invalid value for compression level; must be an integer between 0 and 9 !!!" << std::endl;
      std::cerr << "compression level is then set to default value 0; i.e. no compression !!" << std::endl;
      compressionLevel = 0;
    }

    hid_t propList_create_id = H5Pcreate(H5P_DATASET_CREATE);

    if (nzg == 1) {
      const hsize_t chunk_size2D[2] = {(hsize_t) nyg, (hsize_t) nxg};
      H5Pset_chunk (propList_create_id, 2, chunk_size2D);
    } else { // THREE_D
      const hsize_t chunk_size3D[3] = {(hsize_t) nzg, (hsize_t) nyg, (hsize_t) nxg};
      H5Pset_chunk (propList_create_id, 3, chunk_size3D);
    }
    H5Pset_shuffle (propList_create_id);
    H5Pset_deflate (propList_create_id, compressionLevel);
    
    /*
     * write heavy data to HDF5 file
     */
    real_t* dataPtr;

    for (int iVar=0; iVar<nbVarDebug; iVar++) {
      
      std::string dataSetName("/debug");

      std::ostringstream outNum;
      outNum.width(2);
      outNum.fill('0');
      outNum << iVar;
      dataSetName +=outNum.str();

      // write heavy data
      hid_t dataset_id = H5Dcreate2(file_id, dataSetName.c_str(), dataType, 
				    dataspace_file, 
				    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      if (nzg == 1)
	dataPtr = &(data(0,0,iVar));
      else
	dataPtr = &(data(0,0,0,iVar));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, dataPtr);
      
      H5Dclose(dataset_id);


    }

    // write time step as an attribute to root group
    hid_t ds_id   = H5Screate(H5S_SCALAR);
    hid_t attr_id = H5Acreate2(file_id, "time step", H5T_NATIVE_INT, 
				      ds_id,
				      H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_INT, &nStep);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
    
    // write date as an attribute to root group
    std::string dataString = current_date();
    const char *dataChar = dataString.c_str();
    hsize_t   dimsAttr[1] = {1};
    hid_t type = H5Tcopy (H5T_C_S1);
    status = H5Tset_size (type, H5T_VARIABLE);
    hid_t root_id = H5Gopen2(file_id, "/", H5P_DEFAULT);
    hid_t dataspace_id = H5Screate_simple(1, dimsAttr, NULL);
    attr_id = H5Acreate2(root_id, "creation date", type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, type, &dataChar);
    status = H5Aclose(attr_id);
    status = H5Gclose(root_id);
    status = H5Tclose(type);
    status = H5Sclose(dataspace_id);

    // close/release resources.
    H5Pclose(propList_create_id);
    H5Sclose(dataspace_memory);
    H5Sclose(dataspace_file);
    //H5Dclose(dataset_id);
    H5Fclose(file_id);

    (void) status;

#else

    (void) data;
    (void) suffix;
    (void) nStep;

    std::cerr << "[HydroRunBase::outputHdf5Debug] Hdf5 not enabled !!!\n";

#endif // USE_HDF5

  } // HydroRunBase::outputHdf5Debug

  // =======================================================
  // =======================================================
  /**
   * Write a wrapper file using the Xmdf file format (XML) to allow
   * Paraview to read these h5 files as a time series.
   *
   * \param[in] totalNumberOfSteps The number of time steps computed.
   * \param[in] singleStep boolean; if true we only write header for the last step.
   * \param[in] ghostIncluded boolean; if true include ghost cells
   *
   * If library HDF5 is not available, do nothing.
   */
    void HydroRunBase::writeXdmfForHdf5Wrapper(int totalNumberOfSteps, bool singleStep, bool ghostIncluded)
  {
#ifdef USE_HDF5

    // data size actually written on disk
    int nxg = nx;
    int nyg = ny;
    int nzg = nz;
    if (ghostIncluded) {
      nxg += 2*ghostWidth;
      nyg += 2*ghostWidth;
      nzg += 2*ghostWidth;
    }

    // get data type as a string for Xdmf
    std::string dataTypeName;
    if (sizeof(real_t) == sizeof(float))
      dataTypeName = "Float";
    else
      dataTypeName = "Double";

    /*
     * 1. open XDMF and write header lines
     */
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    std::string xdmfFilename = outputPrefix+".xmf";
    if (singleStep) { // add nStep to file name
      std::ostringstream outNum;
      outNum.width(7);
      outNum.fill('0');
      outNum << totalNumberOfSteps;
      xdmfFilename = outputPrefix+"_"+outNum.str()+".xmf";
    }
    std::fstream xdmfFile;
    xdmfFile.open(xdmfFilename.c_str(), std::ios_base::out);

    xdmfFile << "<?xml version=\"1.0\" ?>"                       << std::endl;
    xdmfFile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>"         << std::endl;
    xdmfFile << "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">" << std::endl;
    xdmfFile << "  <Domain>"                                     << std::endl;
    xdmfFile << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;

    // for each time step write a <grid> </grid> item
    int startStep=0;
    int stopStep =totalNumberOfSteps;
    int deltaStep=nOutput;
    if (singleStep) {
      startStep = totalNumberOfSteps;
      stopStep  = totalNumberOfSteps+1;
      deltaStep = 1;
    }

    for (int nStep=startStep; nStep<=stopStep; nStep+=deltaStep) {
 
      std::ostringstream outNum;
      outNum.width(7);
      outNum.fill('0');
      outNum << nStep;

      // take care that the following filename must be exactly the same as in routine outputHdf5 !!!
      std::string baseName         = outputPrefix+"_"+outNum.str();
      std::string hdf5Filename     = outputPrefix+"_"+outNum.str()+".h5";
      std::string hdf5FilenameFull = outputDir+"/"+outputPrefix+"_"+outNum.str()+".h5";

      xdmfFile << "    <Grid Name=\"" << baseName << "\" GridType=\"Uniform\">" << std::endl;
      xdmfFile << "    <Time Value=\"" << nStep << "\" />"                      << std::endl;
      
      // topology CoRectMesh
      if (dimType == TWO_D) 
	xdmfFile << "      <Topology TopologyType=\"2DCoRectMesh\" NumberOfElements=\"" << nyg << " " << nxg << "\"/>" << std::endl;
      else
	xdmfFile << "      <Topology TopologyType=\"3DCoRectMesh\" NumberOfElements=\"" << nzg << " " << nyg << " " << nxg << "\"/>" << std::endl;
      
      // geometry
      if (dimType == TWO_D) {
	xdmfFile << "    <Geometry Type=\"ORIGIN_DXDY\">"        << std::endl;
	xdmfFile << "    <DataStructure"                         << std::endl;
	xdmfFile << "       Name=\"Origin\""                     << std::endl;
	xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
	xdmfFile << "       Dimensions=\"2\""                    << std::endl;
	xdmfFile << "       Format=\"XML\">"                     << std::endl;
	xdmfFile << "       0 0"                                 << std::endl;
	xdmfFile << "    </DataStructure>"                       << std::endl;
	xdmfFile << "    <DataStructure"                         << std::endl;
	xdmfFile << "       Name=\"Spacing\""                    << std::endl;
	xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
	xdmfFile << "       Dimensions=\"2\""                    << std::endl;
	xdmfFile << "       Format=\"XML\">"                     << std::endl;
	xdmfFile << "       1 1"                                 << std::endl;
	xdmfFile << "    </DataStructure>"                       << std::endl;
	xdmfFile << "    </Geometry>"                            << std::endl;
      } else {
	xdmfFile << "    <Geometry Type=\"ORIGIN_DXDYDZ\">"      << std::endl;
	xdmfFile << "    <DataStructure"                         << std::endl;
	xdmfFile << "       Name=\"Origin\""                     << std::endl;
	xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
	xdmfFile << "       Dimensions=\"3\""                    << std::endl;
	xdmfFile << "       Format=\"XML\">"                     << std::endl;
	xdmfFile << "       0 0 0"                               << std::endl;
	xdmfFile << "    </DataStructure>"                       << std::endl;
	xdmfFile << "    <DataStructure"                         << std::endl;
	xdmfFile << "       Name=\"Spacing\""                    << std::endl;
	xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
	xdmfFile << "       Dimensions=\"3\""                    << std::endl;
	xdmfFile << "       Format=\"XML\">"                     << std::endl;
	xdmfFile << "       1 1 1"                               << std::endl;
	xdmfFile << "    </DataStructure>"                       << std::endl;
	xdmfFile << "    </Geometry>"                            << std::endl;
      }
      
      // density
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"density\">" << std::endl;
      xdmfFile << "        <DataStructure"                             << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""    << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                         << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/density"             << std::endl;
      xdmfFile << "        </DataStructure>"                           << std::endl;
      xdmfFile << "      </Attribute>"                                 << std::endl;
      
      // energy
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"energy\">" << std::endl;
      xdmfFile << "        <DataStructure"                              << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""     << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                          << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/energy"             << std::endl;
      xdmfFile << "        </DataStructure>"                            << std::endl;
      xdmfFile << "      </Attribute>"                                  << std::endl;
      
      // momentum X
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_x\">" << std::endl;
      xdmfFile << "        <DataStructure"                                << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/momentum_x"             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
      
      // momentum Y
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_y\">" << std::endl;
      xdmfFile << "        <DataStructure" << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/momentum_y"             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
      
      // momentum Z
      if (dimType == THREE_D and !mhdEnabled) {
	xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_z\">" << std::endl;
	xdmfFile << "        <DataStructure"                                << std::endl;
	xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
	xdmfFile << "           Format=\"HDF\">"                            << std::endl;
	xdmfFile << "           "<<hdf5Filename<<":/momentum_z"             << std::endl;
	xdmfFile << "        </DataStructure>"                              << std::endl;
	xdmfFile << "      </Attribute>"                                    << std::endl;
      }
      
      if (mhdEnabled) {
	// momentum Z
	xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_z\">" << std::endl;
	xdmfFile << "        <DataStructure" << std::endl;
	xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
	if (dimType == TWO_D)
	  xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
	else
	  xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
	xdmfFile << "           Format=\"HDF\">"                            << std::endl;
	xdmfFile << "           "<<hdf5Filename<<":/momentum_z"             << std::endl;
	xdmfFile << "        </DataStructure>"                              << std::endl;
	xdmfFile << "      </Attribute>"                                    << std::endl;

	// magnetic field X
	xdmfFile << "      <Attribute Center=\"Node\" Name=\"magnetic_field_x\">" << std::endl;
	xdmfFile << "        <DataStructure" << std::endl;
	xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
	if (dimType == TWO_D)
	  xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
	else
	  xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
	xdmfFile << "           Format=\"HDF\">"                            << std::endl;
	xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_x"             << std::endl;
	xdmfFile << "        </DataStructure>"                              << std::endl;
	xdmfFile << "      </Attribute>"                                    << std::endl;
	
	// magnetic field Y
	xdmfFile << "      <Attribute Center=\"Node\" Name=\"magnetic_field_y\">" << std::endl;
	xdmfFile << "        <DataStructure" << std::endl;
	xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
	if (dimType == TWO_D)
	  xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
	else
	  xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
	xdmfFile << "           Format=\"HDF\">"                            << std::endl;
	xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_y"             << std::endl;
	xdmfFile << "        </DataStructure>"                              << std::endl;
	xdmfFile << "      </Attribute>"                                    << std::endl;
	
	// magnetic field Z
	xdmfFile << "      <Attribute Center=\"Node\" Name=\"magnetic_field_z\">" << std::endl;
	xdmfFile << "        <DataStructure" << std::endl;
	xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
	if (dimType == TWO_D)
	  xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
	else
	  xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
	xdmfFile << "           Format=\"HDF\">"                            << std::endl;
	xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_z"             << std::endl;
	xdmfFile << "        </DataStructure>"                              << std::endl;
	xdmfFile << "      </Attribute>"                                    << std::endl;
	
      } // end mhdEnabled

      // finalize grid file for the current time step
      xdmfFile << "   </Grid>" << std::endl;
      
    } // end for loop over time step
    
    // finalize Xdmf wrapper file
    xdmfFile << "   </Grid>" << std::endl;
    xdmfFile << " </Domain>" << std::endl;
    xdmfFile << "</Xdmf>"    << std::endl;

#else

    (void) totalNumberOfSteps;
    (void) singleStep;
    (void) ghostIncluded;

    std::cerr << "[HydroRunBase::writeXdmfForHdf5Wrapper] Hdf5 not enabled !!!\n";

#endif // USE_HDF5

  } // HydroRunBase::writeXdmfForHdf5Wrapper

  
  // =======================================================
  // =======================================================
  /**
   * dump computation results (conservative variables) into a file
   * (NetCDF4 file format) file extension is nc. File can be viewed with
   * paraview (at least version >= 3.6.1); see also ncdump.
   *
   * \param[in] U A reference to a hydro simulation HostArray
   * \param[in] nStep The current time step, used to label results filename. 
   *
   * \note we are using C API (instead of C++ API) because to our
   * knowledge the C++ API is not complete (does not allows chunking for
   * example), and it only compatible with Netcdf3 data model.
   * See web documention of the C API at
   * http://www.unidata.ucar.edu/software/netcdf/netcdf-4/newdocs/netcdf-c 
   * Probably that the new C++ API
   * (http://www.unidata.ucar.edu/software/netcdf/docs/cxx4/) could fit
   * our need, but it seems prematurate to use it right now
   *
   * If library NetCDF4 is not available, do nothing.
   */
  void HydroRunBase::outputNetcdf4(HostArray<real_t> &U, int nStep)
  {

#ifdef USE_NETCDF4

#ifdef USE_DOUBLE
#define NC_REAL_T NC_DOUBLE
#define nc_put_var_real_t nc_put_var_double
#else
#define NC_REAL_T NC_FLOAT
#define nc_put_var_real_t nc_put_var_float
#endif // USE_DOUBLE

    // make filename string
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName           = outputPrefix+"_"+outNum.str();
    std::string netcdfFilename     = baseName+".nc";
    std::string netcdfFilenameFull = outputDir+"/"+netcdfFilename;

    /*
     * write NETCDF4 file
     */

    // create file handler
    int ncId, returnVal;
    if ( (returnVal = nc_create(netcdfFilenameFull.c_str(), 
				NC_NETCDF4, &ncId)) ) {
      NETCDF_ERROR(returnVal);
    }

    // create netCDF dimensions (column major order !!!)
    int xDimId, yDimId, zDimId;
    size_t chunkSize[3];
    if (dimType == TWO_D) {
      if ((returnVal = nc_def_dim(ncId, "x", jsize, &xDimId)))
	NETCDF_ERROR(returnVal);
      if ((returnVal = nc_def_dim(ncId, "y", isize, &yDimId)))
	NETCDF_ERROR(returnVal);
      chunkSize[0] = ny;
      chunkSize[1] = nx;
    } else {
      if ((returnVal = nc_def_dim(ncId, "x", ksize, &xDimId)))
	NETCDF_ERROR(returnVal);
      if ((returnVal = nc_def_dim(ncId, "y", jsize, &yDimId)))
	NETCDF_ERROR(returnVal);
      if ((returnVal = nc_def_dim(ncId, "z", isize, &zDimId)))
	NETCDF_ERROR(returnVal);
      chunkSize[0] = nz;
      chunkSize[1] = ny;
      chunkSize[2] = nz;
    }

    int dimIds[3];
    dimIds[0] = xDimId;
    dimIds[1] = yDimId;
    if (dimType == THREE_D) dimIds[2] = zDimId;

    // create variables IDs
    int var_density, var_energy;
    int var_mx, var_my, var_mz;
    int var_bx, var_by, var_bz;

    if (dimType == TWO_D) {
      
      nc_def_var(ncId, "rho",    NC_REAL_T, 2, dimIds, &var_density);
      nc_def_var_chunking(ncId, var_density, NC_CHUNKED, &chunkSize[0]);

      nc_def_var(ncId, "E",     NC_REAL_T, 2, dimIds, &var_energy);
      nc_def_var_chunking(ncId, var_energy, NC_CHUNKED, &chunkSize[0]);

      nc_def_var(ncId, "rho_vx", NC_REAL_T, 2, dimIds, &var_mx);
      nc_def_var_chunking(ncId, var_mx, NC_CHUNKED, &chunkSize[0]);

      nc_def_var(ncId, "rho_vy", NC_REAL_T, 2, dimIds, &var_my);
      nc_def_var_chunking(ncId, var_my, NC_CHUNKED, &chunkSize[0]);

      if (mhdEnabled) {
	nc_def_var(ncId, "rho_vz",       NC_REAL_T, 2, dimIds, &var_mz);
	nc_def_var_chunking(ncId, var_mz, NC_CHUNKED, &chunkSize[0]);

	nc_def_var(ncId, "Bx", NC_REAL_T, 2, dimIds, &var_bx);
	nc_def_var_chunking(ncId, var_bx, NC_CHUNKED, &chunkSize[0]);

	nc_def_var(ncId, "By", NC_REAL_T, 2, dimIds, &var_by);
	nc_def_var_chunking(ncId, var_by, NC_CHUNKED, &chunkSize[0]);

	nc_def_var(ncId, "Bz", NC_REAL_T, 2, dimIds, &var_bz);
	nc_def_var_chunking(ncId, var_bz, NC_CHUNKED, &chunkSize[0]);
     }

    } else { // THREE_D

      nc_def_var(ncId, "rho",    NC_REAL_T, 3, dimIds, &var_density);
      nc_def_var_chunking(ncId, var_density, NC_CHUNKED, &chunkSize[0]);

      nc_def_var(ncId, "E",     NC_REAL_T, 3, dimIds, &var_energy);
      nc_def_var_chunking(ncId, var_energy, NC_CHUNKED, &chunkSize[0]);

      nc_def_var(ncId, "rho_vx", NC_REAL_T, 3, dimIds, &var_mx);
      nc_def_var_chunking(ncId, var_mx, NC_CHUNKED, &chunkSize[0]);

      nc_def_var(ncId, "rho_vy", NC_REAL_T, 3, dimIds, &var_my);
      nc_def_var_chunking(ncId, var_my, NC_CHUNKED, &chunkSize[0]);

      nc_def_var(ncId, "rho_vz", NC_REAL_T, 3, dimIds, &var_mz);
      nc_def_var_chunking(ncId, var_mz, NC_CHUNKED, &chunkSize[0]);

      if (mhdEnabled) {
	nc_def_var(ncId, "Bx", NC_REAL_T, 3, dimIds, &var_bx);
	nc_def_var_chunking(ncId, var_bx, NC_CHUNKED, &chunkSize[0]);

	nc_def_var(ncId, "By", NC_REAL_T, 3, dimIds, &var_by);
	nc_def_var_chunking(ncId, var_by, NC_CHUNKED, &chunkSize[0]);

	nc_def_var(ncId, "Bz", NC_REAL_T, 3, dimIds, &var_bz);
	nc_def_var_chunking(ncId, var_bz, NC_CHUNKED, &chunkSize[0]);
      }    
    }

    // write data to file (using putVar method)
    if (dimType == TWO_D) {
      nc_put_var_real_t(ncId, var_density, &(U(0,0,ID)) );
      nc_put_var_real_t(ncId, var_energy,  &(U(0,0,IP)) );
      nc_put_var_real_t(ncId, var_mx,      &(U(0,0,IU)) );
      nc_put_var_real_t(ncId, var_my,      &(U(0,0,IV)) );
      if (mhdEnabled) {
	nc_put_var_real_t(ncId, var_mz,      &(U(0,0,IW)) );
	nc_put_var_real_t(ncId, var_bx,      &(U(0,0,IA)) );
	nc_put_var_real_t(ncId, var_by,      &(U(0,0,IB)) );
	nc_put_var_real_t(ncId, var_bz,      &(U(0,0,IC)) );
      }
     } else {
      nc_put_var_real_t(ncId, var_density, &(U(0,0,0,ID)) );
      nc_put_var_real_t(ncId, var_energy,  &(U(0,0,0,IP)) );
      nc_put_var_real_t(ncId, var_mx,      &(U(0,0,0,IU)) );
      nc_put_var_real_t(ncId, var_my,      &(U(0,0,0,IV)) );
      nc_put_var_real_t(ncId, var_mz,      &(U(0,0,0,IW)) );
      if (mhdEnabled) {
	nc_put_var_real_t(ncId, var_bx,      &(U(0,0,0,IA)) );
	nc_put_var_real_t(ncId, var_by,      &(U(0,0,0,IB)) );
	nc_put_var_real_t(ncId, var_bz,      &(U(0,0,0,IC)) );
      }
    }


#endif // USE_NETCDF4

  } // HydroRunBase::outputNetcdf4

  // =======================================================
  // =======================================================
  /**
   * dump computation results (conservative variables) into a file
   * (NRRD file format) file extension is nrrd. 
   * NRRD is an acronym for Nearly Raw Raster Data.
   *
   * This output routine is only intended for fun, i.e. use with the webGL
   * toolkit xtk : https://github.com/xtk/X#readme
   *
   * \param[in] U A reference to a hydro simulation HostArray
   * \param[in] nStep The current time step, used to label results filename. 
   *
   */
  void HydroRunBase::outputNrrd(HostArray<real_t> &U, int nStep)
  {

    /*
     * make filename strings for each variable to dump
     */
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;

    // loop over variable to dump
    for (int iVar=0; iVar<nbVar; ++iVar) {

      // build complete filename
      std::string filename = outputDir+"/"+outputPrefix+"_"+varPrefix[iVar]+"_"+outNum.str()+".nrrd";
      
      // begin output to file
      // NOTE : always use float 32 bit, because xtk do not support
      // double precision
      std::fstream outFile;
      outFile.open (filename.c_str(), std::ios_base::out);

      if (dimType == TWO_D) {

 	outFile << "NRRD0004\n";
	outFile << "# Complete NRRD file format specification at:\n";
	outFile << "# http://teem.sourceforge.net/nrrd/format.html\n";
	outFile << "type: float\n";
	outFile << "dimension: 2\n";
	outFile << "sizes: " << nx << " " << ny << std::endl;
	outFile << "space directions: (1,0) (0,1)\n";
	outFile << "endian: little\n";
	outFile << "encoding: raw\n\n";
	
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    float tmpValue = (float) U(i,j,iVar);
	    outFile.write(reinterpret_cast<char const *>(&tmpValue), sizeof(float));
	  }

     } else { // THREE_D

	outFile << "NRRD0004\n";
	outFile << "# Complete NRRD file format specification at:\n";
	outFile << "# http://teem.sourceforge.net/nrrd/format.html\n";
	outFile << "type: float\n";
	outFile << "dimension: 3\n";
	outFile << "sizes: " << nx << " " << ny << " " << nz << std::endl;
	outFile << "space directions: (1,0,0) (0,1,0) (0,0,1)\n";
	outFile << "endian: little\n";
	outFile << "encoding: raw\n\n";

	for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	  for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	    for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	      float tmpValue = (float) U(i,j,k,iVar);
	      outFile.write(reinterpret_cast<char const *>(&tmpValue), sizeof(float));
	    }

      } // end TWO_D / THREE_D

      outFile.close();

    } // end for iVar

  } // HydroRunBase::outputNrrd

  // =======================================================
  // =======================================================
  /**
   * Dump computation results (conservative variables) into a file.
   *
   * This method is just a wrapper that calls the actual dump methods
   * according to the file formats enabled in the configuration file.
   *
   * \param[in] U A reference to a hydro simulation HostArray
   * \param[in] nStep The current time step, used to label results filename. 
   */
  void HydroRunBase::output(HostArray<real_t> &U, int nStep, bool ghostIncluded)
  {

    if (outputVtkEnabled)  outputVtk (U, nStep);
    if (outputHdf5Enabled) outputHdf5(U, nStep, ghostIncluded);
    if (outputNetcdf4Enabled) outputNetcdf4(U, nStep);
    if (outputXsmEnabled)  outputXsm (U, nStep);
    if (outputPngEnabled)  outputPng (U, nStep);
    if (outputNrrdEnabled) outputNrrd(U, nStep);

    // extra output
    if ( !problem.compare("turbulence-Ornstein-Uhlenbeck") ) {
      // need to output forcing parameters
      pForcingOrnsteinUhlenbeck -> output_forcing(nStep);
    }

  } // HydroRunBase::output

  // =======================================================
  // =======================================================
  /**
   * Dump only X,Y and Z faces of the simulation domain.
   *
   * This method is just a wrapper that calls the actual dump methods
   * according to the file formats enabled in the configuration file.
   *
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] outputFormat File format.
   */
  void HydroRunBase::outputFaces(int nStep, FileFormat outputFormat)
  {
    
    HostArray<real_t> &U = getDataHost(nStep);
    
    if (outputFormat == FF_VTK) {  
      outputFacesVtk (U, nStep, IX);
      outputFacesVtk (U, nStep, IY);
      outputFacesVtk (U, nStep, IZ);
    } else if (outputFormat == FF_HDF5)  {
      outputFacesHdf5(U, nStep);
    } else {
      std::cerr << "File format " << outputFormat << " not available for dumping faces..." << std::endl;
    }

  } // HydroRunBase::outputFaces

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  /**
   * Dump only X,Y and Z faces of the simulation domain (GPU version).
   *
   * This method is just a wrapper that calls the actual dump methods
   * according to the file formats enabled in the configuration file.
   *
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] outputFormat File format.
   */
  void HydroRunBase::outputFaces(int nStep, FileFormat outputFormat,
				 HostArray<real_t>   &xface,
				 HostArray<real_t>   &yface,
				 HostArray<real_t>   &zface,
				 DeviceArray<real_t> &d_xface,
				 DeviceArray<real_t> &d_yface,
				 DeviceArray<real_t> &d_zface)
  {

    DeviceArray<real_t> &U = getData(nStep);

    // copy X-face from GPU
    {
      dim3 dimBlock(16, 16);
      dim3 dimGrid(blocksFor(jsize, dimBlock.x),
		   blocksFor(ksize, dimBlock.y));
      kernel_copy_face_x<<< dimGrid, dimBlock >>>(U.data(), d_xface.data(),
      						  U.dimx(), U.dimy(), U.dimz(), U.pitch(),
      						  d_xface.dimx(), d_xface.dimy(), d_xface.dimz(), d_xface.pitch(), U.nvar());
      d_xface.copyToHost(xface);
    }
    
    // copy Y-face from GPU
    {
      dim3 dimBlock(16, 16);
      dim3 dimGrid(blocksFor(isize, dimBlock.x),
    		   blocksFor(ksize, dimBlock.y));
      kernel_copy_face_y<<< dimGrid, dimBlock >>>(U.data(), d_yface.data(),
    						  U.dimx(), U.dimy(), U.dimz(), U.pitch(),
    						  d_yface.dimx(), d_yface.dimy(), d_yface.dimz(), d_yface.pitch(), U.nvar());
      d_yface.copyToHost(yface);
    }

    // copy Z-face from GPU
    {
      dim3 dimBlock(16, 16);
      dim3 dimGrid(blocksFor(isize, dimBlock.x),
    		   blocksFor(jsize, dimBlock.y));
      kernel_copy_face_z<<< dimGrid, dimBlock >>>(U.data(), d_zface.data(),
    						  U.dimx(), U.dimy(), U.dimz(), U.pitch(),
    						  d_zface.dimx(), d_zface.dimy(), d_zface.dimz(), d_zface.pitch(), U.nvar());
      d_zface.copyToHost(zface);
    }

    // dump to file
    if (outputFormat == FF_VTK) {  
      outputFacesVtk (xface, nStep, IX);
      outputFacesVtk (yface, nStep, IY);
      outputFacesVtk (zface, nStep, IZ);
    }  

  } // HydroRunBase::outputFaces
#endif

  // =======================================================
  // =======================================================
  /**
   * Dump computation results (faces of domain) into a HDF5 file.
   *
   * \see outputHdf5
   */
  void HydroRunBase::outputFacesHdf5(HostArray<real_t> &U, int nStep)
  {

    (void) U;
    (void) nStep;
    std::cerr << "[outputFacesHdf5] Not implemented, TODO !!!" << std::endl;

  } // HydroRunBase::outputFacesHdf5

  // =======================================================
  // =======================================================
  /**
   * Dump computation results (faces of domain) into a VTK file,
   * one file per side using type RectilinearGrid (celldata).
   *
   * This routine is only available for 3D problems.
   *
   * \param[in] U A reference to a hydro simulation HostArray.
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] sideDir Identify which faces to dump
   *
   * \see outputVtk
   *
   * \note Since here we use celldata, coordinate arrays must have size+1 length
   */
  void HydroRunBase::outputFacesVtk(HostArray<real_t> &U, 
				    int nStep, 
				    ComponentIndex3D sideDir)
  {

    if (dimType == TWO_D)
      return;

    // check scalar data type
    bool useDouble = false;
    if (sizeof(real_t) == sizeof(double)) {
      useDouble = true;
    }

    //bool outputVtkHandWritten = true;

    // get output mode (ascii or binary)
    bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

    /*
     * build full path filename
     */
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName     = outputPrefix+"_"+outNum.str();
    std::string filename     = baseName;
    if (sideDir == IX)
      filename += "_IX";
    if (sideDir == IY)
      filename += "_IY";
    if (sideDir == IZ)
      filename += "_IZ";
    filename += ".vtr";
    std::string filenameFull = outputDir+"/"+filename;

    /*
     * Hand written procedure (no VTK library linking required).
     * Write XML imageData using either :
     * - ascii 
     * - raw binary
     *
     * Each hydrodynamics field component is written in a separate <DataArray>.
     * Magnetic field component are dumped if mhdEnabled is true !
     */
    std::fstream outFile;
    outFile.open(filenameFull.c_str(), std::ios_base::out);
    
    // domain extent (ghost included)
    //int xmin=0, xmax=isize-1, ymin=0, ymax=jsize-1, zmin=0, zmax=ksize-1;
    int xmin=0, xmax=isize, ymin=0, ymax=jsize, zmin=0, zmax=ksize;
    
    if (sideDir == IX)
      xmax=1;
    if (sideDir == IY)
      ymax=1;
    if (sideDir == IZ)
      zmax=1;

    // if writing raw binary data (file does not respect XML standard)
    if (outputVtkAscii)
      outFile << "<?xml version=\"1.0\"?>" << std::endl;
    
    // write xml data header
    if (isBigEndian())
      outFile << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
    else
      outFile << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    
    outFile << "  <RectilinearGrid WholeExtent=\""
	    << xmin << " " << xmax << " " 
	    << ymin << " " << ymax << " " 
	    << zmin << " " << zmax << "\">" << std::endl;
    outFile << "    <Piece Extent=\"" 
	    << xmin << " " << xmax << " " 
	    << ymin << " " << ymax << " " 
	    << zmin << " " << zmax << ""
	    << "\">" << std::endl;

    // coordinates
    outFile << "      <Coordinates>" << std::endl;
    outFile << "        <DataArray type=\"Int32\" Name=\"x\" format=\"ascii\">" << std::endl;
    for (int i=0; i<xmax+1; i++)
      outFile << i << " ";
    outFile << std::endl;
    outFile << "        </DataArray>" << std::endl;
    outFile << "        <DataArray type=\"Int32\" Name=\"y\" format=\"ascii\">" << std::endl;
    for (int i=0; i<ymax+1; i++)
      outFile << i << " ";
    outFile << std::endl;
    outFile << "        </DataArray>" << std::endl;
    outFile << "        <DataArray type=\"Int32\" Name=\"z\" format=\"ascii\">" << std::endl;
    for (int i=0; i<zmax+1; i++)
      outFile << i << " ";
    outFile << std::endl;
    outFile << "        </DataArray>" << std::endl;
    outFile << "      </Coordinates>" << std::endl;

    outFile << "      <CellData Scalars=\"Conservative variables\">" << std::endl;

    if (outputVtkAscii) {
      
      // write ascii data
      int i0 = 0; int iend = isize; if (sideDir==IX) iend=1;
      int j0 = 0; int jend = jsize; if (sideDir==IY) jend=1;
      int k0 = 0; int kend = ksize; if (sideDir==IZ) kend=1;
      for (int iVar=0; iVar<nbVar; iVar++) {
	if (useDouble) {
	  outFile << "       <DataArray type=\"Float64\" Name=\"" << varNames[iVar]
		  << "\" format=\"ascii\" >" << std::endl;
	} else {
	  outFile << "       <DataArray type=\"Float32\" Name=\"" << varNames[iVar]
		  << "\" format=\"ascii\" >" << std::endl;
	}
	for(int k= k0; k < kend; k++) {
	  for(int j= j0; j < jend; j++) {
	    for(int i = i0; i < iend; i++) {
	      outFile << std::setprecision(12) << U(i,j,k,iVar) << " ";
	    }
	    outFile << std::endl;
	  }
	}
	outFile << "      </DataArray>" << std::endl;
      }
      
      outFile << "    </CellData>" << std::endl;
      outFile << "  </Piece>" << std::endl;
      outFile << "  </RectilinearGrid>" << std::endl;
      outFile << "</VTKFile>" << std::endl;
      
    } else { // do it using appended format raw binary (no base 64 encoding)

      std::cerr << "[outputFacesVtk] Raw Binary Rectilinear file format not implemented. TODO !!" << std::endl;
      
    } // end raw binary write
    
    outFile.close();
    
  } // HydroRunBase::outputFacesVtk

  // =======================================================
  // =======================================================
  /**
   * Dump computation results (faces of domain) into a VTK file,
   * one file per side using type RectilinearGrid (celldata).
   *
   * This routine is only available for 3D problems.
   *
   * \param[in] faceData HostArray with face data.
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] sideDir Identify which faces to dump.
   *
   * \see outputVtk
   * \note faceData should have dimensions matching sideDir
   *       example: isize,jsize,1,nbVar for sideDir=IZ
   */
  // void HydroRunBase::outputFacesVtk(HostArray<real_t> &faceData, 
  // 				    int nStep, 
  // 				    ComponentIndex3D sideDir)
  // {

  //   if (dimType == TWO_D)
  //     return;

  //   // check scalar data type
  //   bool useDouble = false;
  //   if (sizeof(real_t) == sizeof(double)) {
  //     useDouble = true;
  //   }

  //   bool outputVtkHandWritten = true;

  //   // get output mode (ascii or binary)
  //   bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

  //   /*
  //    * build full path filename
  //    */
  //   std::string outputDir    = configMap.getString("output", "outputDir", "./");
  //   std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
  //   std::ostringstream outNum;
  //   outNum.width(7);
  //   outNum.fill('0');
  //   outNum << nStep;
  //   std::string baseName     = outputPrefix+"_"+outNum.str();
  //   std::string filename     = baseName;
  //   if (sideDir == IX)
  //     filename += "_IX";
  //   if (sideDir == IY)
  //     filename += "_IY";
  //   if (sideDir == IZ)
  //     filename += "_IZ";
  //   filename += ".vtr";
  //   std::string filenameFull = outputDir+"/"+filename;

  //   /*
  //    * Hand written procedure (no VTK library linking required).
  //    * Write XML imageData using either :
  //    * - ascii 
  //    * - raw binary
  //    *
  //    * Each hydrodynamics field component is written in a separate <DataArray>.
  //    * Magnetic field component are dumped if mhdEnabled is true !
  //    */
  //   std::fstream outFile;
  //   outFile.open(filenameFull.c_str(), std::ios_base::out);
    
  //   // domain extent (ghost included)
  //   int xmin=0, xmax=isize-1, ymin=0, ymax=jsize-1, zmin=0, zmax=ksize-1;
    
  //   if (sideDir == IX)
  //     xmax=0;
  //   if (sideDir == IY)
  //     ymax=0;
  //   if (sideDir == IZ)
  //     zmax=0;

  //   // if writing raw binary data (file does not respect XML standard)
  //   if (outputVtkAscii)
  //     outFile << "<?xml version=\"1.0\"?>" << std::endl;
    
  //   // write xml data header
  //   if (isBigEndian())
  //     outFile << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
  //   else
  //     outFile << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    
  //   outFile << "  <RectilinearGrid WholeExtent=\""
  // 	    << xmin << " " << xmax << " " 
  // 	    << ymin << " " << ymax << " " 
  // 	    << zmin << " " << zmax << "\">" << std::endl;
  //   outFile << "    <Piece Extent=\"" 
  // 	    << xmin << " " << xmax << " " 
  // 	    << ymin << " " << ymax << " " 
  // 	    << zmin << " " << zmax << ""
  // 	    << "\">" << std::endl;

  //   // coordinates
  //   outFile << "      <Coordinates>" << std::endl;
  //   outFile << "        <DataArray type=\"Int32\" Name=\"x\" format=\"ascii\">" << std::endl;
  //   for (int i=0; i<xmax+1; i++)
  //     outFile << i << " ";
  //   outFile << std::endl;
  //   outFile << "        </DataArray>" << std::endl;
  //   outFile << "        <DataArray type=\"Int32\" Name=\"y\" format=\"ascii\">" << std::endl;
  //   for (int i=0; i<ymax+1; i++)
  //     outFile << i << " ";
  //   outFile << std::endl;
  //   outFile << "        </DataArray>" << std::endl;
  //   outFile << "        <DataArray type=\"Int32\" Name=\"z\" format=\"ascii\">" << std::endl;
  //   for (int i=0; i<zmax+1; i++)
  //     outFile << i << " ";
  //   outFile << std::endl;
  //   outFile << "        </DataArray>" << std::endl;
  //   outFile << "      </Coordinates>" << std::endl;

  //   outFile << "      <CellData Scalars=\"Conservative variables\">" << std::endl;

  //   if (outputVtkAscii) {
      
  //     // write ascii data
  //     int i0 = 0; int iend = isize; if (sideDir==IX) iend=1;
  //     int j0 = 0; int jend = jsize; if (sideDir==IY) jend=1;
  //     int k0 = 0; int kend = ksize; if (sideDir==IZ) kend=1;
  //     for (int iVar=0; iVar<nbVar; iVar++) {
  // 	if (useDouble) {
  // 	  outFile << "       <DataArray type=\"Float64\" Name=\"" << varNames[iVar]
  // 		  << "\" format=\"ascii\" >" << std::endl;
  // 	} else {
  // 	  outFile << "       <DataArray type=\"Float32\" Name=\"" << varNames[iVar]
  // 		  << "\" format=\"ascii\" >" << std::endl;
  // 	}
  // 	for(int k= k0; k < kend; k++) {
  // 	  for(int j= j0; j < jend; j++) {
  // 	    for(int i = i0; i < iend; i++) {
  // 	      outFile << std::setprecision(12) << faceData(i,j,k,iVar) << " ";
  // 	    }
  // 	    outFile << std::endl;
  // 	  }
  // 	}
  // 	outFile << "      </DataArray>" << std::endl;
  //     }
      
  //     outFile << "    </CellData>" << std::endl;
  //     outFile << "  </Piece>" << std::endl;
  //     outFile << "  </RectilinearGrid>" << std::endl;
  //     outFile << "</VTKFile>" << std::endl;
      
  //   } else { // do it using appended format raw binary (no base 64 encoding)

  //     std::cerr << "[outputFacesVtk] Raw Binary Rectilinear file format not implemented. TODO !!" << std::endl;
      
  //   } // end raw binary write
    
  //   outFile.close();
    
  // } // HydroRunBase::outputFacesVtk

  // =======================================================
  // =======================================================
  /**
   * load data from a HDF5 file (previously dumped with outputHdf5).
   * Data are computation results (conservative variables)
   * in HDF5 format.
   *
   * \sa outputHdf5 this routine performs output in HDF5 file
   * \sa writeXdmfForHdf5Wrapper this routine write a Xdmf wrapper file for paraview.
   *
   * \note Take care that HostArray use column-format ordering,
   * whereas C-language and so C API of HDF5 uses raw-major ordering
   * !!! We need to invert dimensions.
   *
   * \note This input routine is the only one that can be used for
   * re-starting a simulation run.
   *
   * \param[out] U A reference to a hydro simulation HostArray
   * \param[in]  filename Name of the input HDF5 file
   * \param[in]  halfResolution boolean, triggers reading half resolution data
   *
   * If library HDF5 is not available, do nothing, just print an error message.
   */
  int HydroRunBase::inputHdf5(HostArray<real_t> &U, 
			      const std::string filename,
			      bool halfResolution)
  {

#ifdef USE_HDF5
    bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);
    //bool halfResolution = configMap.getBool("run","restart_upscale",false);

    herr_t status;
    hid_t  dataset_id;

    // sizes to read
    int nx_r,  ny_r,  nz_r;  // logical sizes
    int nx_rg, ny_rg, nz_rg; // sizes with ghost zones included

    if (halfResolution) {
      nx_r  = nx/2;
      ny_r  = ny/2;
      nz_r  = nz/2;
      
      nx_rg = nx/2+2*ghostWidth;
      ny_rg = ny/2+2*ghostWidth;
      nz_rg = nz/2+2*ghostWidth;

    } else { // use current resolution
      nx_r  = nx;
      ny_r  = ny;
      nz_r  = nz;
      
      nx_rg = nx+2*ghostWidth;
      ny_rg = ny+2*ghostWidth;
      nz_rg = nz+2*ghostWidth;
    }
   
    /*
     * Try to read HDF5 file.
     */
    
    /* Open the file */
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    HDF5_CHECK((file_id >= 0), "H5Fopen "+filename);

    /* build hyperslab handles */
    /* for data in file */
    /* for layout in memory */
    hsize_t  dims_memory[3];
    hsize_t  dims_file[3];
    hid_t dataspace_memory, dataspace_file;

    if (ghostIncluded) {
      
      if (dimType == TWO_D) {
	dims_memory[0] = ny_rg;
	dims_memory[1] = nx_rg;

	dims_file[0]   = ny_rg;
	dims_file[1]   = nx_rg;

	dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(2, dims_file  , NULL);
      } else {
	dims_memory[0] = nz_rg;
	dims_memory[1] = ny_rg;
	dims_memory[2] = nx_rg;

	dims_file[0]   = nz_rg;
	dims_file[1]   = ny_rg;
	dims_file[2]   = nx_rg;

	dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
      }

    } else { // no ghost zones
      
      if (dimType == TWO_D) {
	dims_memory[0] = ny_rg; 
	dims_memory[1] = nx_rg;

	dims_file[0]   = ny_r;
	dims_file[1]   = nx_r;

	dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(2, dims_file  , NULL);
      } else {
	dims_memory[0] = nz_rg;
	dims_memory[1] = ny_rg;
	dims_memory[2] = nx_rg;

	dims_file[0]   = nz_r;
	dims_file[1]   = ny_r;
	dims_file[2]   = nx_r;

	dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
      }

    }


    /* hyperslab parameters */
    if (ghostIncluded) {
      
      if (dimType == TWO_D) {
	hsize_t  start[2] = {0, 0}; // ghost zone included
	hsize_t stride[2] = {1, 1};
	hsize_t  count[2] = {(hsize_t) ny_rg, (hsize_t) nx_rg};
	hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = {0, 0, 0}; // ghost zone included
	hsize_t stride[3] = {1, 1, 1};
	hsize_t  count[3] = {(hsize_t) nz_rg, (hsize_t) ny_rg, (hsize_t) nx_rg};
	hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      }
      
    } else {

      if (dimType == TWO_D) {
	hsize_t  start[2] = {(hsize_t) ghostWidth, (hsize_t) ghostWidth}; // ghost zone width
	hsize_t stride[2] = {1, 1};
	hsize_t  count[2] = {(hsize_t) ny_r, (hsize_t) nx_r};
	hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = {(hsize_t) ghostWidth, (hsize_t) ghostWidth, (hsize_t) ghostWidth}; // ghost zone width
	hsize_t stride[3] = {1, 1, 1};
	hsize_t  count[3] = {(hsize_t) nz_r, (hsize_t) ny_r, (hsize_t) nx_r};
	hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
      }
    
    }

    /* defines data type */
    hid_t dataType, expectedDataType;
    if (sizeof(real_t) == sizeof(float))
      expectedDataType = H5T_NATIVE_FLOAT;
    else
      expectedDataType = H5T_NATIVE_DOUBLE;
    H5T_class_t t_class_expected = H5Tget_class(expectedDataType);

    // pointer to data in memory buffer ( should be h_U.data() )
    real_t* data;

    /*
     * open data set and perform read
     */

    // read density
    dataset_id = H5Dopen2(file_id, "/density", H5P_DEFAULT);
    dataType  = H5Dget_type(dataset_id);
    H5T_class_t t_class = H5Tget_class(dataType);
    if (t_class != t_class_expected) {
      std::cerr << "Wrong HDF5 datatype !!\n";
      std::cerr << "expected     : " << t_class_expected << std::endl;
      std::cerr << "but received : " << t_class          << std::endl;
    }

    if (dimType == TWO_D)
      data = &(U(0,0,ID));
    else
      data = &(U(0,0,0,ID));

    status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		     H5P_DEFAULT, data);
    H5Dclose(dataset_id);

    // read energy
    dataset_id = H5Dopen2(file_id, "/energy", H5P_DEFAULT);

    if (dimType == TWO_D)
      data = &(U(0,0,IP));
    else
      data = &(U(0,0,0,IP));

    status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		     H5P_DEFAULT, data);
    H5Dclose(dataset_id);

    // read momentum X
    dataset_id = H5Dopen2(file_id, "/momentum_x", H5P_DEFAULT);

    if (dimType == TWO_D)
      data = &(U(0,0,IU));
    else
      data = &(U(0,0,0,IU));

    status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		     H5P_DEFAULT, data);
    H5Dclose(dataset_id);

    // read momentum Y
    dataset_id = H5Dopen2(file_id, "/momentum_y", H5P_DEFAULT);

    if (dimType == TWO_D)
      data = &(U(0,0,IV));
    else
      data = &(U(0,0,0,IV));

    status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		     H5P_DEFAULT, data);
    H5Dclose(dataset_id);

    // read momentum Z (only if hydro 3D)
    if (dimType == THREE_D and !mhdEnabled) {
      dataset_id = H5Dopen2(file_id, "/momentum_z", H5P_DEFAULT);

      data = &(U(0,0,0,IW));
      
      status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		       H5P_DEFAULT, data);
      H5Dclose(dataset_id);
    }

    if (mhdEnabled) {
      // read momentum Z
      dataset_id = H5Dopen2(file_id, "/momentum_z", H5P_DEFAULT);
      
      if (dimType == TWO_D)
	data = &(U(0,0,IW));
      else
	data = &(U(0,0,0,IW));
      
      status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		       H5P_DEFAULT, data);
      H5Dclose(dataset_id);

      // read magnetic field components X
      dataset_id = H5Dopen2(file_id, "/magnetic_field_x", H5P_DEFAULT);
      
      if (dimType == TWO_D)
	data = &(U(0,0,IA));
      else
	data = &(U(0,0,0,IA));
      
      status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		       H5P_DEFAULT, data);
      H5Dclose(dataset_id);

      // read magnetic field components Y
      dataset_id = H5Dopen2(file_id, "/magnetic_field_y", H5P_DEFAULT);
      
      if (dimType == TWO_D)
	data = &(U(0,0,IB));
      else
	data = &(U(0,0,0,IB));
      
      status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		       H5P_DEFAULT, data);
      H5Dclose(dataset_id);

      // read magnetic field components Z
      dataset_id = H5Dopen2(file_id, "/magnetic_field_z", H5P_DEFAULT);
      
      if (dimType == TWO_D)
	data = &(U(0,0,IC));
      else
	data = &(U(0,0,0,IC));
      
      status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		       H5P_DEFAULT, data);
      H5Dclose(dataset_id);

    } // end mhdEnabled


    // read time step attribute
    int timeStep;
    hid_t group_id;
    hid_t attr_id;

    {
      group_id  = H5Gopen2(file_id, "/", H5P_DEFAULT);
      attr_id   = H5Aopen(group_id, "time step", H5P_DEFAULT);
      status    = H5Aread(attr_id, H5T_NATIVE_INT, &timeStep);
      status    = H5Aclose(attr_id);
      status    = H5Gclose(group_id);
    }

    // read totalTime
    {
      double readVal;
      group_id  = H5Gopen2(file_id, "/", H5P_DEFAULT);
      attr_id   = H5Aopen(group_id, "total time", H5P_DEFAULT);
      status    = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &readVal);
      status    = H5Aclose(attr_id);
      status    = H5Gclose(group_id);

      totalTime = (real_t) readVal;
    }
    
    // check ghost zones consistency
    // {
    //   int readVal;
    //   group_id  = H5Gopen2(file_id, "/", H5P_DEFAULT);
    //   attr_id   = H5Aopen(group_id, "ghost zone included", H5P_DEFAULT);
    //   status    = H5Aread(attr_id, H5T_NATIVE_INT, &readVal);
    //   status    = H5Aclose(attr_id);
    //   status    = H5Gclose(group_id);
      
    //   if ( readVal != 0 and readVal != 1)
    // 	std::cerr << "HydroRunBase::inputHdf5 : error reading \"ghost zone included\", invalid value\n";
      
    //   if ( (readVal == 0 and ghostIncluded) or (readVal == 1 and !ghostIncluded) )
    // 	std::cerr << "HydroRunBase::inputHdf5 : error reading \"ghost zone included\"; inconsistent value\n";

    // }

    //std::cout << "[DEBUG]  " << "time step read : " << timeStep << std::endl;

    // close/release resources.
    //H5Pclose(propList_create_id);
    H5Sclose(dataspace_memory);
    H5Sclose(dataspace_file);
    //H5Dclose(dataset_id);
    H5Fclose(file_id);

    (void) status;

    return timeStep;

#else

    (void) U;
    (void) filename;
    (void) halfResolution;

    std::cerr << "HDF5 library is not available ! You can't load a data file for restarting the simulation run !!!" << std::endl;
    std::cerr << "Please install HDF5 library !!!" << std::endl;

    return -1;

#endif // USE_HDF5

  } // HydroRunBase::inputHdf5

  // =======================================================
  // =======================================================
  /**
   * Upscale, i.e. increase resolution of input data LowRes into HiRes.
   * This routine is usefull to perform a large resolution run, and taking
   * as initial condition a half-resolution input data (coming from a HDF5
   * file, ghostzones included).
   *
   *
   * \param[out] HiRes  A reference to a HostArray (current resolution)
   * \param[in]  LowRes A reference to a HostArray (half resolution)
   *
   */
  void HydroRunBase::upscale(HostArray<real_t> &HiRes, 
			     const HostArray<real_t> &LowRes)
  {
    
    if (dimType == TWO_D) {

      // loop at high resolution
      for (int j=0; j<jsize; j++) {
	int jLow = (j+ghostWidth)/2;
	
	for (int i=0; i<isize; i++) {
	  int iLow = (i+ghostWidth)/2;
	  
	  // hydro variables : just copy low res value
	  for (int iVar=0; iVar<4; ++iVar) {
	    
	    HiRes(i,j,iVar) = LowRes(iLow, jLow, iVar);
	    
	  } // end for iVar
	  
	  if (mhdEnabled) {
	    HiRes(i,j,IW) = LowRes(iLow, jLow, IW);

	    // magnetic field component : interpolate values so that
	    // div B = 0 is still true !
	    
	    // X-component of magnetic field
	    if (i+ghostWidth-2*iLow == 0) {
	      HiRes(i,j,IBX) = LowRes(iLow, jLow, IBX);
	    } else {
	      HiRes(i,j,IBX) = (LowRes(iLow,   jLow, IBX) +
				LowRes(iLow+1, jLow, IBX) )/2;
	    }
		    
	    // Y-component of magnetic field
	    if (j+ghostWidth-2*jLow == 0) {
	      HiRes(i,j,IBY) = LowRes(iLow, jLow, IBY);
	    } else {
	      HiRes(i,j,IBY) = (LowRes(iLow, jLow,   IBY) +
				LowRes(iLow, jLow+1, IBY) )/2;
	    }
	    
	    // Z-component of magnetic field
	    HiRes(i,j,IBZ) = LowRes(iLow, jLow, IBZ);
	    	    
	  } // end mhdEnabled 
 
	} // end for i
	
      } // end for j

    } else { // THREE_D
      
      // loop at high resolution
      for (int k=0; k<ksize; k++) {
	int kLow = (k+ghostWidth)/2;
	
	for (int j=0; j<jsize; j++) {
	  int jLow = (j+ghostWidth)/2;
	  
	  for (int i=0; i<isize; i++) {
	    int iLow = (i+ghostWidth)/2;
	    
	    // hydro variables : just copy low res value
	    for (int iVar=0; iVar<5; ++iVar) {
	      
	      HiRes(i,j,k,iVar) = LowRes(iLow, jLow, kLow, iVar);
	      
	    } // end for iVar

	    if (mhdEnabled) {
	      // magnetic field component : interpolate values so that
	      // div B = 0 is still true !

	      // X-component of magnetic field
	      if (i+ghostWidth-2*iLow == 0) {
		HiRes(i,j,k,IBX) = LowRes(iLow, jLow, kLow, IBX);
	      } else {
		HiRes(i,j,k,IBX) = (LowRes(iLow,   jLow, kLow, IBX) +
				    LowRes(iLow+1, jLow, kLow, IBX) )/2;
	      }
		    
	      // Y-component of magnetic field
	      if (j+ghostWidth-2*jLow == 0) {
		HiRes(i,j,k,IBY) = LowRes(iLow, jLow, kLow, IBY);
	      } else {
		HiRes(i,j,k,IBY) = (LowRes(iLow, jLow,   kLow, IBY) +
				    LowRes(iLow, jLow+1, kLow, IBY) )/2;
	      }

	      // Z-component of magnetic field
	      if (k+ghostWidth-2*kLow == 0) {
		HiRes(i,j,k,IBZ) = LowRes(iLow, jLow, kLow, IBZ);
	      } else {
		HiRes(i,j,k,IBZ) = (LowRes(iLow, jLow, kLow,   IBZ) +
				    LowRes(iLow, jLow, kLow+1, IBZ) )/2;
	      }

	    } // end mhdEnabled
	    
	  } // end for i

	} // end for j

      } // end for k

    } // end TWO_D / THREE_D

  } // HydroRunBase::upscale

  // =======================================================
  // =======================================================
  void HydroRunBase::init_hydro_jet()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    if (dimType == TWO_D) {
    
      /* jet */
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  //int index = i+isize*j;
	  // fill density, U, V and energy sub-arrays
	  h_U(i,j,ID)=1.0f;
	  h_U(i,j,IP)=1.0f/(_gParams.gamma0-1.0f);
	  h_U(i,j,IU)=0.0f;
	  h_U(i,j,IV)=0.0f;
	}
    
      /* corner grid (not really needed except for Kurganov-Tadmor) */
      if (ghostWidth == 2) {
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j) {
	      h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	      h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	      h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	      h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	    } // end for loop over i,j
	  
	} // end loop over nVar
      }

    } else { // THREE_D
      
      /* jet */
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    // fill density, U, V, W and energy sub-arrays
	    h_U(i,j,k,ID)=1.0f;
	    h_U(i,j,k,IP)=1.0f/(_gParams.gamma0-1.0f);
	    h_U(i,j,k,IU)=0.0f;
	    h_U(i,j,k,IV)=0.0f;
	    h_U(i,j,k,IW)=0.0f;
	  }   
      
      if (ghostWidth == 2) {
	/* fill the 8 grid corner (not really needed except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {     
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j)
	      for (int k=0; k<2; ++k) {
		h_U(     i,     j,     k,nVar) = h_U(   2,   2,   2,nVar);
		h_U(nx+2+i,     j,     k,nVar) = h_U(nx+1,   2,   2,nVar);
		h_U(     i,ny+2+j,     k,nVar) = h_U(   2,ny+1,   2,nVar);
		h_U(nx+2+i,ny+2+j,     k,nVar) = h_U(nx+1,ny+1,   2,nVar);
		
		h_U(     i,     j,nz+2+k,nVar) = h_U(   2,   2,nz+1,nVar);
		h_U(nx+2+i,     j,nz+2+k,nVar) = h_U(nx+1,   2,nz+1,nVar);
		h_U(     i,ny+2+j,nz+2+k,nVar) = h_U(   2,ny+1,nz+1,nVar);
		h_U(nx+2+i,ny+2+j,nz+2+k,nVar) = h_U(nx+1,ny+1,nz+1,nVar);
	      } // end for loop over i,j,k
	} // end for loop over nVar
      }	
    }

  } // HydroRunBase::init_hydro_jet

  // =======================================================
  // =======================================================
  /**
   * The Hydrodynamical Sod Test.
   *
   */
  void HydroRunBase::init_hydro_sod()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    if (dimType == TWO_D) {
  
      /* discontinuity line along diagonal */
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  if (i<isize/2) {
	    h_U(i,j,ID)=1.0f;
	    h_U(i,j,IP)=1.0f/(_gParams.gamma0-1.0f);
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	  } else {
	    h_U(i,j,ID)=0.125f;
	    h_U(i,j,IP)=0.1f/(_gParams.gamma0-1.0f);      
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	  }
	}
    
      if (ghostWidth == 2) {
	/* corner grid (not really needed (except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j) {
	      h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	      h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	      h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	      h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	    } // end for loop over i,j
	  
	} // end loop over nVar
      }
    
    } else { // THREE_D
	
      /* discontinuity line along diagonal */
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    if (i<isize/2) {
	      h_U(i,j,k,ID)=1.0f;
	      h_U(i,j,k,IP)=1.0f/(_gParams.gamma0-1.0f);
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	    } else {
	      h_U(i,j,k,ID)=0.125f;
	      h_U(i,j,k,IP)=0.1f/(_gParams.gamma0-1.0f);      
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	    }
	  }

      if (ghostWidth == 2) {
	/* fill the 8 grid corner (not really needed except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j)
	      for (int k=0; k<2; ++k) {
		h_U(     i,     j,     k,nVar) = h_U(   2,   2,   2,nVar);
		h_U(nx+2+i,     j,     k,nVar) = h_U(nx+1,   2,   2,nVar);
		h_U(     i,ny+2+j,     k,nVar) = h_U(   2,ny+1,   2,nVar);
		h_U(nx+2+i,ny+2+j,     k,nVar) = h_U(nx+1,ny+1,   2,nVar);
		
		h_U(     i,     j,nz+2+k,nVar) = h_U(   2,   2,nz+1,nVar);
		h_U(nx+2+i,     j,nz+2+k,nVar) = h_U(nx+1,   2,nz+1,nVar);
		h_U(     i,ny+2+j,nz+2+k,nVar) = h_U(   2,ny+1,nz+1,nVar);
		h_U(nx+2+i,ny+2+j,nz+2+k,nVar) = h_U(nx+1,ny+1,nz+1,nVar);
	      } // end for loop over i,j,k
	} // end for loop over nVar
      }
    }

  } // HydroRunBase::init_hydro_sod

  // =======================================================
  // =======================================================
  /**
   * The Hydrodynamical Implosion Test.
   * see
   * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
   * for a description of such initial conditions.
   * see also article : Liska, R., & Wendroff, B., "Comparison of Several difference schemes on 1D and 2D Test problems for the Euler equations", http://www-troja.fjfi.cvut.cz/~liska/CompareEuler/compare8/
   */
  void HydroRunBase::init_hydro_implode()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    /* initialize random generator */
    int seed = configMap.getInteger("implode", "seed", 1);
    srand(seed);

    /* initialize density perturbation amplitude */
    real_t amplitude = configMap.getFloat("implode", "amplitude", 0.0);

    if (dimType == TWO_D) {
  
      /* discontinuity line along diagonal */
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  if (((float)i/nx+(float)j/ny)>0.5) {
	    h_U(i,j,ID)=1.0f + amplitude*(1.0*rand()/RAND_MAX-0.5);
	    h_U(i,j,IP)=1.0f/(_gParams.gamma0-1.0f);
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	  } else {
	    h_U(i,j,ID)=0.125f + amplitude*(1.0*rand()/RAND_MAX-0.5);
	    h_U(i,j,IP)=0.14f/(_gParams.gamma0-1.0f);      
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	  }
	}
    
      if (ghostWidth == 2) {
	/* corner grid (not really needed (except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j) {
	      h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	      h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	      h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	      h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	    } // end for loop over i,j
	  
	} // end loop over nVar
      }
    
    } else { // THREE_D
	
      /* discontinuity line along diagonal */
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    if (((float)i/nx+(float)j/ny+(float)k/nz)>0.5) {
	      h_U(i,j,k,ID)=1.0f + amplitude*(1.0*rand()/RAND_MAX-0.5);
	      h_U(i,j,k,IP)=1.0f/(_gParams.gamma0-1.0f);
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	    } else {
	      h_U(i,j,k,ID)=0.125f + amplitude*(1.0*rand()/RAND_MAX-0.5);
	      h_U(i,j,k,IP)=0.14f/(_gParams.gamma0-1.0f);      
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	    }
	  }

      if (ghostWidth == 2) {
	/* fill the 8 grid corner (not really needed except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j)
	      for (int k=0; k<2; ++k) {
		h_U(     i,     j,     k,nVar) = h_U(   2,   2,   2,nVar);
		h_U(nx+2+i,     j,     k,nVar) = h_U(nx+1,   2,   2,nVar);
		h_U(     i,ny+2+j,     k,nVar) = h_U(   2,ny+1,   2,nVar);
		h_U(nx+2+i,ny+2+j,     k,nVar) = h_U(nx+1,ny+1,   2,nVar);
		
		h_U(     i,     j,nz+2+k,nVar) = h_U(   2,   2,nz+1,nVar);
		h_U(nx+2+i,     j,nz+2+k,nVar) = h_U(nx+1,   2,nz+1,nVar);
		h_U(     i,ny+2+j,nz+2+k,nVar) = h_U(   2,ny+1,nz+1,nVar);
		h_U(nx+2+i,ny+2+j,nz+2+k,nVar) = h_U(nx+1,ny+1,nz+1,nVar);
	      } // end for loop over i,j,k
	} // end for loop over nVar
      }
    }

  } // HydroRunBase::init_hydro_implode

  // =======================================================
  // =======================================================
  /**
   * Sperical blast wave test.
   * see
   * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
   * for a description of such initial conditions.
   *
   * parameters:
   * - radius: radius of the initial spherical domain, in real units
   * - center_x, center_y, center_z : cartesian coordinate of the center of the domain. 
   *
   */
  void HydroRunBase::init_hydro_blast()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t &zMin = _gParams.zMin;

    real_t &xMax = _gParams.xMax;
    real_t &yMax = _gParams.yMax;
    real_t &zMax = _gParams.zMax;

    real_t &dx   = _gParams.dx;
    real_t &dy   = _gParams.dy;
    real_t &dz   = _gParams.dz;

    /* get spherical domain parameters */
    real_t radius       = configMap.getFloat("blast","radius",0.25*(xMax-xMin));
    real_t center_x     = configMap.getFloat("blast","center_x",(xMax+xMin)/2);
    real_t center_y     = configMap.getFloat("blast","center_y",(yMax+yMin)/2);
    real_t center_z     = configMap.getFloat("blast","center_z",(zMax+zMin)/2);
    real_t density_in   = configMap.getFloat("blast","density_in"  ,1.0);
    real_t density_out  = configMap.getFloat("blast","density_out" ,1.0);
    real_t pressure_in  = configMap.getFloat("blast","pressure_in" ,10.0);
    real_t pressure_out = configMap.getFloat("blast","pressure_out",0.1);

    // compute square radius
    radius *= radius;

    /* spherical blast wave test */
    if (dimType == TWO_D) {
    
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	
	  // distance to center
	  real_t d2 = 
	    (xPos-center_x)*(xPos-center_x) +
	    (yPos-center_y)*(yPos-center_y);
	
	  if ( d2 < radius) {
	    h_U(i,j,ID)=density_in;
	    h_U(i,j,IP)=pressure_in/(_gParams.gamma0-1.0f);
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	  } else {
	    h_U(i,j,ID)=density_out;
	    h_U(i,j,IP)=pressure_out/(_gParams.gamma0-1.0f);      
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	  }
	} // end for i
      } // end for j
    
      if (ghostWidth == 2) {
	/* corner grid (not really needed (except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {    
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j) {
	      h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	      h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	      h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	      h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	    } // end for loop over i,j
	} // end loop over nVar
      }

    } else { // THREE_D
	
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	real_t zPos = zMin + dz/2 + (k-ghostWidth)*dz;

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	
	    real_t d2 = 
	      (xPos-center_x)*(xPos-center_x) +
	      (yPos-center_y)*(yPos-center_y) +
	      (zPos-center_z)*(zPos-center_z);
	  
	    if ( d2 < radius ) {
	      h_U(i,j,k,ID)=density_in;
	      h_U(i,j,k,IP)=pressure_in/(_gParams.gamma0-1.0f);
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	    } else {
	      h_U(i,j,k,ID)=density_out;
	      h_U(i,j,k,IP)=pressure_out/(_gParams.gamma0-1.0f);      
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	    }
	  } // end for i
	} // end for j
      } // end for k

      if (ghostWidth == 2) {
	/* fill the 8 grid corner (not really needed except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j)
	      for (int k=0; k<2; ++k) {
		h_U(     i,     j,     k,nVar) = h_U(   2,   2,   2,nVar);
		h_U(nx+2+i,     j,     k,nVar) = h_U(nx+1,   2,   2,nVar);
		h_U(     i,ny+2+j,     k,nVar) = h_U(   2,ny+1,   2,nVar);
		h_U(nx+2+i,ny+2+j,     k,nVar) = h_U(nx+1,ny+1,   2,nVar);
		
		h_U(     i,     j,nz+2+k,nVar) = h_U(   2,   2,nz+1,nVar);
		h_U(nx+2+i,     j,nz+2+k,nVar) = h_U(nx+1,   2,nz+1,nVar);
		h_U(     i,ny+2+j,nz+2+k,nVar) = h_U(   2,ny+1,nz+1,nVar);
		h_U(nx+2+i,ny+2+j,nz+2+k,nVar) = h_U(nx+1,ny+1,nz+1,nVar);
	      } // end for loop over i,j,k
	} // end for loop over nVar
      }
    }

  } // HydroRunBase::init_hydro_blast

  // =======================================================
  // =======================================================
  /**
   * Gresho vortex test.
   * see
   * http://arxiv.org/pdf/1409.7395v1.pdf - section 4.2.3
   * http://www.cfd-online.com/Wiki/Gresho_vortex
   *
   *
   */
  void HydroRunBase::init_hydro_Gresho_vortex()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t &zMin = _gParams.zMin;

    real_t &xMax = _gParams.xMax;
    real_t &yMax = _gParams.yMax;
    real_t &zMax = _gParams.zMax;

    real_t &dx   = _gParams.dx;
    real_t &dy   = _gParams.dy;
    real_t &dz   = _gParams.dz;

    real_t center_x     = configMap.getFloat("Gresho_vortex","center_x",(xMax+xMin)/2);
    real_t center_y     = configMap.getFloat("Gresho_vortex","center_y",(yMax+yMin)/2);
    real_t v_bulk_x     = configMap.getFloat("Gresho_vortex","v_bulk_x"  ,0.0);
    real_t v_bulk_y     = configMap.getFloat("Gresho_vortex","v_bulk_y"  ,0.0);
    real_t v_bulk_z     = configMap.getFloat("Gresho_vortex","v_bulk_z"  ,0.0);

    /* 2d Gresho vortex test */
    if (dimType == TWO_D) {
    
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	
	  // distance to center
	  real_t r = sqrt(
			  (xPos-center_x)*(xPos-center_x) +
			  (yPos-center_y)*(yPos-center_y) );
	  
	  real_t phi = atan2(yPos-center_y,xPos-center_x);
	  real_t P, v_phi;

	  if ( r < 0.2 ) {

	    P     = 5 + 12.5*r*r;
	    v_phi = 5*r;

	  } else if ( r < 0.4 ) {
	  
	    P     = 9 + 12.5*r*r - 20*r + 4*log(5*r);
	    v_phi = 2-5*r;

	  } else {

	    P     = 3 + 4*log(2);
	    v_phi = ZERO_F;

	  }
	   
	  h_U(i,j,ID) = ONE_F;
	  h_U(i,j,IU) = -sin(phi) * v_phi + v_bulk_x;
	  h_U(i,j,IV) =  cos(phi) * v_phi + v_bulk_y;
	  h_U(i,j,IP) = P/(_gParams.gamma0-1.0f) +
	    0.5 * ( SQR(h_U(i,j,IU)) + 
		    SQR(h_U(i,j,IV)) ) / h_U(i,j,ID);
	  
	} // end for i
      } // end for j
    
      if (ghostWidth == 2) {
	/* corner grid (not really needed (except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {    
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j) {
	      h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	      h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	      h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	      h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	    } // end for loop over i,j
	} // end loop over nVar
      }

    } else { // THREE_D
	
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	real_t zPos = zMin + dz/2 + (k-ghostWidth)*dz;

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	
	    // distance to center (vortex tube)
	    real_t r = sqrt(
	      (xPos-center_x)*(xPos-center_x) +
	      (yPos-center_y)*(yPos-center_y) );
	  
	    real_t phi = atan2(yPos-center_y,xPos-center_x);
	    real_t P, v_phi;
	    
	    if ( r < 0.2 ) {
	      
	      P     = 5 + 12.5*r*r;
	      v_phi = 5*r;
	      
	    } else if ( r < 0.4 ) {
	      
	      P     = 9 + 12.5*r*r - 20*r + 4*log(5*r);
	      v_phi = 2-5*r;
	      
	    } else {
	      
	      P     = 3 + 4*log(2);
	      v_phi = ZERO_F;
	      
	    }

	    h_U(i,j,k,ID) = ONE_F;
	    h_U(i,j,k,IU) = -sin(phi)*v_phi + v_bulk_x;
	    h_U(i,j,k,IV) =  cos(phi)*v_phi + v_bulk_y;
	    h_U(i,j,k,IW) = v_bulk_z;
	    h_U(i,j,k,IP) = P/(_gParams.gamma0-1.0f) +
	      0.5 * ( SQR(h_U(i,j,k,IU)) + 
		      SQR(h_U(i,j,k,IV)) + 
		      SQR(h_U(i,j,k,IW)) ) / h_U(i,j,k,ID);
	    
	  } // end for i
	} // end for j
      } // end for k
      
      if (ghostWidth == 2) {
	/* fill the 8 grid corner (not really needed except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j)
	      for (int k=0; k<2; ++k) {
		h_U(     i,     j,     k,nVar) = h_U(   2,   2,   2,nVar);
		h_U(nx+2+i,     j,     k,nVar) = h_U(nx+1,   2,   2,nVar);
		h_U(     i,ny+2+j,     k,nVar) = h_U(   2,ny+1,   2,nVar);
		h_U(nx+2+i,ny+2+j,     k,nVar) = h_U(nx+1,ny+1,   2,nVar);
		
		h_U(     i,     j,nz+2+k,nVar) = h_U(   2,   2,nz+1,nVar);
		h_U(nx+2+i,     j,nz+2+k,nVar) = h_U(nx+1,   2,nz+1,nVar);
		h_U(     i,ny+2+j,nz+2+k,nVar) = h_U(   2,ny+1,nz+1,nVar);
		h_U(nx+2+i,ny+2+j,nz+2+k,nVar) = h_U(nx+1,ny+1,nz+1,nVar);
	      } // end for loop over i,j,k
	} // end for loop over nVar
      }
    }

  } // HydroRunBase::init_hydro_Gresho_vortex

  // =======================================================
  // =======================================================
  /**
   * Test of the Kelvin-Helmholtz instability.
   * See
   * http://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
   * for a description of such initial conditions
   *
   * 4 types of perturbations:
   * - rand : multi-mode
   * - sine : single-mode (simple)
   * - sine_athena : single mode with init condition from Athena
   * - sine_robertson : single mode with init condition from article by Robertson et al.
   *
   * "Computational Eulerian hydrodynamics and Galilean invariance", 
   * B.E. Robertson et al, Mon. Not. R. Astron. Soc., 401, 2463-2476, (2010).
   */
  void HydroRunBase::init_hydro_Kelvin_Helmholtz()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    /* initialize random generator */
    int seed = configMap.getInteger("kelvin-helmholtz", "seed", 1);
    srand(seed);

    /* initialize perturbation amplitude */
    real_t amplitude = configMap.getFloat("kelvin-helmholtz", "amplitude", 0.01);
    
    /* perturbation type random / sine / sine_athena */
    bool p_rand_bool  = configMap.getBool("kelvin-helmholtz", "perturbation_rand", true);
    bool p_sine_bool  = configMap.getBool("kelvin-helmholtz", "perturbation_sine", false);
    bool p_sine_athena_bool    = configMap.getBool("kelvin-helmholtz", "perturbation_sine_athena", false);
    bool p_sine_robertson_bool = configMap.getBool("kelvin-helmholtz", "perturbation_sine_robertson", false);

    real_t p_rand  = p_rand_bool  ? 1.0 : 0.0;
    real_t p_sine  = p_sine_bool  ? 1.0 : 0.0;
    real_t p_sine_athena    = p_sine_athena_bool    ? 1.0 : 0.0;
    real_t p_sine_robertson = p_sine_robertson_bool ? 1.0 : 0.0;

    /* inner and outer fluid density */
    real_t rho_inner = configMap.getFloat("kelvin-helmholtz", "rho_inner", 2.0);
    real_t rho_outer = configMap.getFloat("kelvin-helmholtz", "rho_outer", 1.0);
    real_t pressure  = configMap.getFloat("kelvin-helmholtz", "pressure", 2.5);

    // please note that inner_size+outer_size must be smaller than 0.5
    // unit is a fraction of 0.5*ySize
    real_t inner_size = configMap.getFloat("kelvin-helmholtz", "inner_size", 0.2);
    real_t outer_size = configMap.getFloat("kelvin-helmholtz", "outer_size", 0.2);

    real_t vflow_in  = configMap.getFloat("kelvin-helmholtz", "vflow_in", -0.5);
    real_t vflow_out = configMap.getFloat("kelvin-helmholtz", "vflow_out", 0.5);
    
    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t &zMin = _gParams.zMin;

    real_t &xMax = _gParams.xMax;
    real_t &yMax = _gParams.yMax;
    real_t &zMax = _gParams.zMax;

    real_t xSize=xMax-xMin;
    real_t ySize=yMax-yMin;
    real_t zSize=zMax-zMin;

    real_t xCenter = (xMin+xMax)*0.5;
    real_t yCenter = (yMin+yMax)*0.5;
    real_t zCenter = (zMin+zMax)*0.5;

    real_t &dx   = _gParams.dx;
    real_t &dy   = _gParams.dy;
    real_t &dz   = _gParams.dz;

    if (dimType == TWO_D) {

      if (p_rand_bool) {
	
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	    
	    if ( fabs(yPos-yCenter) > outer_size*ySize ) {
	      
	      h_U(i,j,ID) = rho_outer;

	      h_U(i,j,IU) = rho_outer *
		(vflow_out + amplitude * (1.0*rand()/RAND_MAX - 0.5) );

	      h_U(i,j,IV) = rho_outer *
		(0.0f      + amplitude * (1.0*rand()/RAND_MAX - 0.5) );

	      h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f) +
		0.5 * ( SQR(h_U(i,j,IU)) + 
			SQR(h_U(i,j,IV)) ) / h_U(i,j,ID);
	      
	    } else {
	      
	      h_U(i,j,ID) = rho_inner;

	      h_U(i,j,IU) = rho_inner *
		(vflow_in + amplitude * (1.0*rand()/RAND_MAX - 0.5) );

	      h_U(i,j,IV) = rho_inner *
		( 0.0f    + amplitude * (1.0*rand()/RAND_MAX - 0.5) );

	      h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f) +
		0.5 * ( SQR(h_U(i,j,IU)) + 
			SQR(h_U(i,j,IV)) ) / h_U(i,j,ID);

	    }
	  } // end for i
	} // end for j
	
      } else if (p_sine_athena_bool) {

	real_t a = 0.05;
	real_t sigma = 0.2;
	real_t vflow = 0.5;

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;	    
	    
	    h_U(i,j,ID) = rho_inner;
	    h_U(i,j,IU) = rho_inner * vflow * tanh(yPos/a);
	    h_U(i,j,IV) = rho_inner * amplitude * sin(2.0*M_PI*xPos) * exp(-(yPos*yPos)/(sigma*sigma));
	    h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f) +
	      0.5 * ( SQR(h_U(i,j,IU)) + 
		      SQR(h_U(i,j,IV)) ) / h_U(i,j,ID);

	  } // end for i
	} // end for j

      } else if (p_sine_robertson_bool) {

	// perturbation mode number
	int    n      = configMap.getInteger("kelvin-helmholtz", "mode", 4);
	real_t w0     = configMap.getFloat("kelvin-helmholtz", "w0", 0.1);
	real_t deltaY = configMap.getFloat("kelvin-helmholtz", "deltaY", 0.03);

	real_t rho1 = rho_inner;
	real_t rho2 = rho_outer;

	real_t v1 = vflow_in;
	real_t v2 = vflow_out;

	real_t y1 = yMin + 0.25*ySize;
	real_t y2 = yMin + 0.75*ySize;

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	  real_t ramp = 
	    1.0 / ( 1.0 + exp( 2*(yPos-y1)/deltaY ) ) +
	    1.0 / ( 1.0 + exp( 2*(y2-yPos)/deltaY ) );
	  
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;	    
	    
	    h_U(i,j,ID) = rho1 + ramp*(rho2-rho1);
	    h_U(i,j,IU) = h_U(i,j,ID) * (v1 + ramp*(v2-v1));
	    h_U(i,j,IV) = h_U(i,j,ID) * w0 * sin(n*M_PI*xPos);
	    h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f) +
	      0.5 * ( SQR(h_U(i,j,IU)) + 
		      SQR(h_U(i,j,IV)) ) / h_U(i,j,ID);

	  } // end for i
	} // end for j

      } else if (p_sine_bool) {

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;	    

	    real_t perturb_vx = 0;
	    real_t perturb_vy = amplitude * sin(2.0*M_PI*xPos/xSize);
	    
	    if ( fabs(yPos-yCenter) > outer_size*ySize ) {

	      h_U(i,j,ID) = rho_outer;
	      h_U(i,j,IU) = rho_outer * vflow_out * (1.0+perturb_vx);
	      h_U(i,j,IV) = rho_outer * perturb_vy;
	      h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f) +
	      0.5 * ( SQR(h_U(i,j,IU)) + 
		      SQR(h_U(i,j,IV)) ) / h_U(i,j,ID);

	    } else if ( fabs(yPos-yCenter) <= inner_size*ySize ) {

	      h_U(i,j,ID) = rho_inner;
	      h_U(i,j,IU) = rho_inner * vflow_in * (1.0+perturb_vx);
	      h_U(i,j,IV) = rho_inner * perturb_vy;
	      h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f) +
		0.5 * ( SQR(h_U(i,j,IU)) + 
			SQR(h_U(i,j,IV)) ) / h_U(i,j,ID);

	    } else { // interpolate
	      
	      real_t interpSize = outer_size-inner_size;
	      real_t rho_slope = (rho_outer - rho_inner) / (interpSize * ySize);
	      real_t u_slope   = (vflow_out - vflow_in)  / (interpSize * ySize);

	      real_t deltaY;
	      real_t deltaRho;
	      real_t deltaU;
	      if (yPos > yCenter) {
		deltaY   = yPos-(yCenter+inner_size*ySize);
		deltaRho = rho_slope*deltaY;
		deltaU   = u_slope*deltaY;
	      } else {
		deltaY = yPos-(yCenter-inner_size*ySize);
		deltaRho = -rho_slope*deltaY;
		deltaU   = -u_slope*deltaY;
	      }

	      h_U(i,j,ID) = rho_inner + deltaRho;
	      h_U(i,j,IU) = h_U(i,j,ID) * (vflow_in + deltaU)*(1.0+perturb_vx);
	      h_U(i,j,IV) = h_U(i,j,ID) * perturb_vy;
	      h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f) +
	      0.5 * ( SQR(h_U(i,j,IU)) + 
		      SQR(h_U(i,j,IV)) ) / h_U(i,j,ID);

	    }
	  } // end for i
	} // end for j

      }

      if (ghostWidth == 2) {
	/* corner grid (not really needed (except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j) {
	      h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	      h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	      h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	      h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	    } // end for loop over i,j
	  
	} // end loop over nVar
      }

    } else { // THREE_D

      if (p_rand_bool) {

	for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	  real_t zPos = zMin + dz/2 + (k-ghostWidth)*dz;
	  
	  for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	    real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	    
	    for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	      real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	      if ( fabs(zPos-zCenter) > outer_size*zSize ) {

		h_U(i,j,k,ID) = rho_outer;

		h_U(i,j,k,IU) = rho_outer *
		  (vflow_out + amplitude  * (1.0*rand()/RAND_MAX-0.5) );

		h_U(i,j,k,IV) = rho_outer *
		  (0.0       + amplitude  * (1.0*rand()/RAND_MAX-0.5) );

		h_U(i,j,k,IW) = rho_outer *
		  (0.0       + amplitude  * (1.0*rand()/RAND_MAX-0.5) );

		h_U(i,j,k,IP) = pressure/(_gParams.gamma0-1.0f) +
		  0.5 * ( SQR(h_U(i,j,k,IU)) + 
			  SQR(h_U(i,j,k,IV)) + 
			  SQR(h_U(i,j,k,IW)) ) / h_U(i,j,k,ID);

	      } else {

		h_U(i,j,k,ID) = rho_inner;

		h_U(i,j,k,IU) = rho_inner *
		  (vflow_in + amplitude   * (1.0*rand()/RAND_MAX-0.5) );

		h_U(i,j,k,IV) = rho_inner *
		  (0.0      + amplitude   * (1.0*rand()/RAND_MAX-0.5) );

		h_U(i,j,k,IW) = rho_inner *
		  (0.0      + amplitude   * (1.0*rand()/RAND_MAX-0.5) );

		h_U(i,j,k,IP) = pressure/(_gParams.gamma0-1.0f) +
		  0.5 * ( SQR(h_U(i,j,k,IU)) + 
			  SQR(h_U(i,j,k,IV)) + 
			  SQR(h_U(i,j,k,IW)) ) / h_U(i,j,k,ID);

	      }
	    } // end for i
	  } // end for j
	} // end for k

      } else if (p_sine_bool) {

	for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	  real_t zPos = zMin + dz/2 + (k-ghostWidth)*dz;
	  
	  for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	    real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	    
	    for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	      real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	      real_t perturb_vx = 0;
	      real_t perturb_vy = 0;
	      real_t perturb_vz = amplitude * sin(2.0*M_PI*xPos/xSize);

	      if ( fabs(zPos-zCenter) > outer_size*zSize ) {

		h_U(i,j,k,ID) = rho_outer;
		h_U(i,j,k,IU) = rho_outer * vflow_out; 
		h_U(i,j,k,IV) = rho_outer * perturb_vy;
		h_U(i,j,k,IW) = rho_outer * perturb_vz;
		h_U(i,j,k,IP) = pressure/(_gParams.gamma0-1.0f) +
		  0.5 * ( SQR(h_U(i,j,k,IU)) + 
			  SQR(h_U(i,j,k,IV)) + 
			  SQR(h_U(i,j,k,IW)) ) / h_U(i,j,k,ID);
		

	      } else { // if ( fabs(zPos-zCenter) <= inner_size*zSize )

		h_U(i,j,k,ID) = rho_inner;
		h_U(i,j,k,IU) = rho_inner * vflow_in;
		h_U(i,j,k,IV) = rho_inner * perturb_vy;
		h_U(i,j,k,IW) = rho_inner * perturb_vz;
		h_U(i,j,k,IP) = pressure/(_gParams.gamma0-1.0f) +
		  0.5 * ( SQR(h_U(i,j,k,IU)) + 
			  SQR(h_U(i,j,k,IV)) + 
			  SQR(h_U(i,j,k,IW)) ) / h_U(i,j,k,ID);

	      }

	    } // end for i
	  } // end for j
	} // end for k

      } else if (p_sine_robertson_bool) {

	// perturbation mode number
	int    n      = configMap.getInteger("kelvin-helmholtz", "mode", 4);
	real_t w0     = configMap.getFloat("kelvin-helmholtz", "w0", 0.1);
	real_t deltaZ = configMap.getFloat("kelvin-helmholtz", "deltaZ", 0.03);

	real_t rho1 = rho_inner;
	real_t rho2 = rho_outer;

	real_t v1 = vflow_in;
	real_t v2 = vflow_out;

	real_t z1 = zMin + 0.25*zSize;
	real_t z2 = zMin + 0.75*zSize;

	for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	  real_t zPos = zMin + dz/2 + (k-ghostWidth)*dz;

	  real_t ramp = 
	    1.0 / ( 1.0 + exp( 2*(zPos-z1)/deltaZ ) ) +
	    1.0 / ( 1.0 + exp( 2*(z2-zPos)/deltaZ ) );
	  
	  for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	    real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	    	    
	    for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	      real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;	    
	      
	      h_U(i,j,k,ID) = rho1 + ramp*(rho2-rho1);
	      h_U(i,j,k,IU) = h_U(i,j,k,ID) * (v1 + ramp*(v2-v1)) ;
	      h_U(i,j,k,IV) = h_U(i,j,k,ID) * w0 * cos(n*M_PI*xPos);
	      h_U(i,j,k,IW) = h_U(i,j,k,ID) * w0 * sin(n*M_PI*xPos);
	      h_U(i,j,k,IP) = pressure/(_gParams.gamma0-1.0f) +
		0.5 * ( SQR(h_U(i,j,k,IU)) + 
			SQR(h_U(i,j,k,IV)) + 
			SQR(h_U(i,j,k,IW)) ) / h_U(i,j,k,ID);
	      
	    } // end for i
	  } // end for j
	} // end for k

      } 

      if (ghostWidth == 2) {
	/* fill the 8 grid corner (not really needed except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j)
	      for (int k=0; k<2; ++k) {
		h_U(     i,     j,     k,nVar) = h_U(   2,   2,   2,nVar);
		h_U(nx+2+i,     j,     k,nVar) = h_U(nx+1,   2,   2,nVar);
		h_U(     i,ny+2+j,     k,nVar) = h_U(   2,ny+1,   2,nVar);
		h_U(nx+2+i,ny+2+j,     k,nVar) = h_U(nx+1,ny+1,   2,nVar);
		
		h_U(     i,     j,nz+2+k,nVar) = h_U(   2,   2,nz+1,nVar);
		h_U(nx+2+i,     j,nz+2+k,nVar) = h_U(nx+1,   2,nz+1,nVar);
		h_U(     i,ny+2+j,nz+2+k,nVar) = h_U(   2,ny+1,nz+1,nVar);
		h_U(nx+2+i,ny+2+j,nz+2+k,nVar) = h_U(nx+1,ny+1,nz+1,nVar);
	      } // end for loop over i,j,k
	} // end for loop over nVar
      }
    }

  } // HydroRunBase::init_hydro_Kelvin_Helmholtz

  // =======================================================
  // =======================================================
  /**
   * Test of the Rayleigh-Taylor instability.
   * See
   * http://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
   * for a description of such initial conditions
   */
  void HydroRunBase::init_hydro_Rayleigh_Taylor()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    /* initialize perturbation amplitude */
    real_t amplitude = configMap.getFloat("rayleigh-taylor", "amplitude", 0.01);
    real_t        d0 = configMap.getFloat("rayleigh-taylor", "d0", 1.0);
    real_t        d1 = configMap.getFloat("rayleigh-taylor", "d1", 2.0);

    
    bool  randomEnabled = configMap.getBool("rayleigh-taylor", "randomEnabled", false);
    int            seed = configMap.getInteger("rayleigh-taylor", "random_seed", 33);
    if (randomEnabled)
      srand(seed);


    /* static gravity field */
    real_t& gravity_x = _gParams.gravity_x;
    real_t& gravity_y = _gParams.gravity_y;
    real_t& gravity_z = _gParams.gravity_z;
    real_t         P0 = 1.0f/(_gParams.gamma0-1.0f);
      
    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t &zMin = _gParams.zMin;

    real_t &xMax = _gParams.xMax;
    real_t &yMax = _gParams.yMax;
    real_t &zMax = _gParams.zMax;

    real_t Lx = xMax-xMin;
    real_t Ly = yMax-yMin;
    real_t Lz = zMax-zMin;

    if (dimType == TWO_D) {
  
      // the initial condition must ensure the condition of
      // hydrostatic equilibrium for pressure P = P0 - 0.1*\rho*y
      for (int j=0; j<jsize; j++) {
	real_t y = yMin + dy/2 + (j-ghostWidth)*dy;

	for (int i=0; i<isize; i++) {
	  real_t x = xMin + dx/2 + (i-ghostWidth)*dx;

	  // Athena initial conditions
	  // if ( y > 0.0 ) {
	  //   h_U(i,j,ID) = 2.0f;
	  // } else {
	  //   h_U(i,j,ID) = 1.0f;
	  // }
	  // h_U(i,j,IP) = P0 + gravity_x*x + gravity_y*y;
	  // h_U(i,j,IU) = 0.0f;
	  // h_U(i,j,IV) = amplitude*(1+cosf(2*M_PI*x))*(1+cosf(0.5*M_PI*y))/4;

	  if ( y > (yMin+yMax)/2 ) {
	    h_U(i,j,ID) = d1;
	  } else {
	    h_U(i,j,ID) = d0;
	  }
	  h_U(i,j,IP) = P0 + h_U(i,j,ID)*(gravity_x*x + gravity_y*y);
	  h_U(i,j,IU) = 0.0f;
	  if (randomEnabled)
	    h_U(i,j,IV) = amplitude * ( rand() * 1.0 / RAND_MAX - 0.5);
	  else
	    h_U(i,j,IV) = amplitude * 
	      (1+cos(2*M_PI*x/Lx))*
	      (1+cos(2*M_PI*y/Ly))/4;
	}
      }
    
      for (int j=0; j<jsize; j++) {
	for (int i=0; i<isize; i++) {
	  h_gravity(i,j,IX) = gravity_x;
	  h_gravity(i,j,IY) = gravity_y;
	}
      }

      if (ghostWidth == 2) {
	/* corner grid (not really needed (except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j) {
	      h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	      h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	      h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	      h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	    } // end for loop over i,j
	  
	} // end loop over nVar
      }

    } else { // THREE_D

      // the initial condition must ensure the condition of
      // hydrostatic equilibrium for pressure P = P0 - 0.1*\rho*y
      for (int k=0; k<ksize; k++) {
	real_t z = zMin + dz/2 + (k-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  real_t y = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    real_t x = xMin + dx/2 + (i-ghostWidth)*dx;
	    
	    // Athena initial conditions
	    // if ( z > 0.0 ) {
	    //   h_U(i,j,k,ID) = 2.0f;
	    // } else {
	    //   h_U(i,j,k,ID) = 1.0f;
	    // }
	    // h_U(i,j,k,IP) = P0 + gravity_x*x + gravity_y*y + gravity_z*z;
	    // h_U(i,j,k,IU) = 0.0f;
	    // h_U(i,j,k,IV) = 0.0f;
	    // h_U(i,j,k,IW) = amplitude*(1+cosf(2*M_PI*x))*(1+cosf(2*M_PI*y))*(1+cosf(0.5*M_PI*z))/6;

	    if ( z > (zMin+zMax)/2 ) {
	      h_U(i,j,k,ID) = d1;
	    } else {
	      h_U(i,j,k,ID) = d0;
	    }
	    h_U(i,j,k,IP) = P0 + h_U(i,j,k,ID)*(gravity_x*x + gravity_y*y + gravity_z*z);
	    h_U(i,j,k,IU) = 0.0f;
	    h_U(i,j,k,IV) = 0.0f;
	    if (randomEnabled)
	      h_U(i,j,k,IW) = amplitude * ( rand() * 1.0 / RAND_MAX - 0.5);
	    else
	      h_U(i,j,k,IW) = amplitude * 
		(1+cos(2*M_PI*x/Lx))*
		(1+cos(2*M_PI*y/Ly))*
		(1+cos(2*M_PI*z/Lz))/8;

	  }
	}
      }

      for (int k=0; k<ksize; k++) {
	for (int j=0; j<jsize; j++) {
	  for (int i=0; i<isize; i++) {
	    h_gravity(i,j,k,IX) = gravity_x;
	    h_gravity(i,j,k,IY) = gravity_y;
	    h_gravity(i,j,k,IZ) = gravity_z;
	  }
	}
      }

      if (ghostWidth == 2) {
	/* fill the 8 grid corner (not really needed except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j)
	      for (int k=0; k<2; ++k) {
		h_U(     i,     j,     k,nVar) = h_U(   2,   2,   2,nVar);
		h_U(nx+2+i,     j,     k,nVar) = h_U(nx+1,   2,   2,nVar);
		h_U(     i,ny+2+j,     k,nVar) = h_U(   2,ny+1,   2,nVar);
		h_U(nx+2+i,ny+2+j,     k,nVar) = h_U(nx+1,ny+1,   2,nVar);
		
		h_U(     i,     j,nz+2+k,nVar) = h_U(   2,   2,nz+1,nVar);
		h_U(nx+2+i,     j,nz+2+k,nVar) = h_U(nx+1,   2,nz+1,nVar);
		h_U(     i,ny+2+j,nz+2+k,nVar) = h_U(   2,ny+1,nz+1,nVar);
		h_U(nx+2+i,ny+2+j,nz+2+k,nVar) = h_U(nx+1,ny+1,nz+1,nVar);
	      } // end for loop over i,j,k
	} // end for loop over nVar
      }	
    }

#ifdef __CUDACC__
    d_gravity.copyFromHost(h_gravity);
#endif

  } // HydroRunBase::init_hydro_Rayleigh_Taylor

  // =======================================================
  // =======================================================
  /**
   * Keplerian disk test.
   * See
   * http://arxiv.org/pdf/1409.7395v1.pdf - section 4.2.4
   * for a description of such initial conditions.
   *
   */
  void HydroRunBase::init_hydro_Keplerian_disk()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());


    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t &zMin = _gParams.zMin;

    real_t &xMax = _gParams.xMax;
    real_t &yMax = _gParams.yMax;
    real_t &zMax = _gParams.zMax;

    real_t Lx = xMax-xMin;
    real_t Ly = yMax-yMin;
    real_t Lz = zMax-zMin;

    /* initialize parameters */
    real_t epsilon = configMap.getFloat("Keplerian-disk", "epsilon" , 0.01);
    real_t P0      = configMap.getFloat("Keplerian-disk", "pressure", 1e-6);
    real_t xCenter = configMap.getFloat("Keplerian-disk", "xCenter", (xMax+xMin)/2.0);
    real_t yCenter = configMap.getFloat("Keplerian-disk", "yCenter", (yMax+yMin)/2.0);
    real_t grav    = configMap.getFloat("gravity"       , "g"       ,1.0);

    if (dimType == TWO_D) {
  
      for (int j=0; j<jsize; j++) {
	real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	
	for (int i=0; i<isize; i++) {
	  real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	  
	  real_t theta = atan2(yPos-yCenter, xPos-xCenter);
	  
	  // distance to center
	  real_t r = sqrt(
			  (xPos-xCenter)*(xPos-xCenter) +
			  (yPos-yCenter)*(yPos-yCenter) );

	  // orbital velocity
	  real_t velocity = r * pow(r*r+epsilon*epsilon, -3.0/4.0);
	  
	  // static Keplerian potential (g = -grad ( Phi ))
	  // Phi = - (r^2+epsilon^2)^(-1/2)
	  {

	    real_t r_x, r_y;

	    // x -> x+dx
	    r_x = sqrt(
		       (xPos+dx-xCenter)*(xPos+dx-xCenter) +
		       (yPos   -yCenter)*(yPos   -yCenter) );
	    real_t phi_px = -1.0/sqrt(r_x*r_x+epsilon*epsilon);

	    // x -> x-dx
	    r_x = sqrt(
		       (xPos-dx-xCenter)*(xPos-dx-xCenter) +
		       (yPos   -yCenter)*(yPos   -yCenter) );
	    real_t phi_mx = -1.0/sqrt(r_x*r_x+epsilon*epsilon);
	    
	    // y -> y+dy
	    r_y = sqrt(
		       (xPos   -xCenter)*(xPos   -xCenter) +
		       (yPos+dy-yCenter)*(yPos+dy-yCenter) );
	    real_t phi_py = -1.0/sqrt(r_y*r_y+epsilon*epsilon);

	    // y -> y-dy
	    r_y = sqrt(
		       (xPos   -xCenter)*(xPos   -xCenter) +
		       (yPos-dy-yCenter)*(yPos-dy-yCenter) );
	    real_t phi_my = -1.0/sqrt(r_y*r_y+epsilon*epsilon);
	    
	    h_gravity(i,j,IX) = - grav * HALF_F * (phi_px - phi_mx)/dx;
	    h_gravity(i,j,IY) = - grav * HALF_F * (phi_py - phi_my)/dy;

	  }

	  // other variables
	  if ( r < 0.5 ) {
	    h_U(i,j,ID) = 0.01 + pow( r/0.5, 3.0);
	  } else if ( r <= 2 ) {
	    h_U(i,j,ID) = 0.01 + 1;
	  } else if ( r >  2 ) {
	    h_U(i,j,ID) = 0.01 + pow(1 + (r-2)/0.1, -3.0);
	  }

	  h_U(i,j,IU) = -sin(theta) * velocity * h_U(i,j,ID);
	  h_U(i,j,IV) =  cos(theta) * velocity * h_U(i,j,ID);

	  h_U(i,j,IP) = P0 / (_gParams.gamma0 - ONE_F) +
	    0.5 * ( h_U(i,j,IU) * h_U(i,j,IU) +
		    h_U(i,j,IV) * h_U(i,j,IV) ) / h_U(i,j,ID);

	  // real_t eken = 0.5f * (h_U(i,j,IU) * h_U(i,j,IU) + h_U(i,j,IV) * h_U(i,j,IV)) / (h_U(i,j,ID) * h_U(i,j,ID));
	  // real_t eint = h_U(i,j,IP) / h_U(i,j,ID) - eken;

	  // if (eint < 0) {
	  //   printf("KKKK hydro eint < 0  : e %f eken %f diff %f d %f u %f v %f\n",h_U(i,j,IP)/h_U(i,j,ID),eken,h_U(i,j,IP)/h_U(i,j,ID)-eken,h_U(i,j,ID),h_U(i,j,IU),h_U(i,j,IV));
	  // }

  
	} // end for i
      } // end for j
    

      if (ghostWidth == 2) {
	/* corner grid (not really needed (except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j) {
	      h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	      h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	      h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	      h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	    } // end for loop over i,j
	  
	} // end loop over nVar
      }

    } else { // THREE_D

      // cylindrical symetry
      for (int k=0; k<ksize; k++) {
	real_t zPos = zMin + dz/2 + (k-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	    
	    real_t theta = atan2(yPos-yCenter, xPos-xCenter);
	    
	    // distance to center
	    real_t r = sqrt(
			    (xPos-xCenter)*(xPos-xCenter) +
			    (yPos-yCenter)*(yPos-yCenter) );
	    
	    // orbital velocity
	    real_t velocity = r * pow(r*r+epsilon*epsilon, -3.0/4.0);
	    
	    if ( r < 0.5 ) {
	      h_U(i,j,k,ID) = 0.01 + pow( r/0.5, 3.0);
	    } else if ( r <= 2 ) {
	      h_U(i,j,k,ID) = 0.01 + 1;
	    } else {
	      h_U(i,j,k,ID) = 0.01 + pow(1 + (r-2)/0.1, -3.0);
	    }
	    	    
	    h_U(i,j,k,IU) = -sin(theta) * velocity * h_U(i,j,ID);
	    h_U(i,j,k,IV) =  cos(theta) * velocity * h_U(i,j,ID);
	    h_U(i,j,k,IW) =  ZERO_F;

	    h_U(i,j,k,IP) = P0 / (_gParams.gamma0 - ONE_F) +
	      0.5 * ( h_U(i,j,k,IU) * h_U(i,j,k,IU) +
		      h_U(i,j,k,IV) * h_U(i,j,k,IV) +
		      h_U(i,j,k,IW) * h_U(i,j,k,IW) ) / h_U(i,j,k,ID);
	    
	    // static Keplerian potential (g = -grad ( Phi ))
	    // Phi = - (r^2+epsilon^2)^(-1/2)
	    {
	      real_t phi = -1.0/sqrt(r*r+epsilon*epsilon);
	      
	      // x -> x+dx
	      real_t r_x = sqrt(
				(xPos+dx-xCenter)*(xPos+dx-xCenter) +
				(yPos-yCenter)*(yPos-yCenter) );
	      real_t phi_x = -1.0/sqrt(r_x*r_x+epsilon*epsilon);
	      
	      // y -> y+dy
	      real_t r_y = sqrt(
				(xPos-xCenter)*(xPos-xCenter) +
				(yPos+dy-yCenter)*(yPos+dy-yCenter) );
	      real_t phi_y = -1.0/sqrt(r_y*r_y+epsilon*epsilon);
	      
	      h_gravity(i,j,k,IX) = - (phi - phi_x)/dx;
	      h_gravity(i,j,k,IY) = - (phi - phi_y)/dy;
	      h_gravity(i,j,k,IZ) = ZERO_F;
	    }
	    
	  } // end for i
	} // end for j
      } // end for k

      if (ghostWidth == 2) {
	/* fill the 8 grid corner (not really needed except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j)
	      for (int k=0; k<2; ++k) {
		h_U(     i,     j,     k,nVar) = h_U(   2,   2,   2,nVar);
		h_U(nx+2+i,     j,     k,nVar) = h_U(nx+1,   2,   2,nVar);
		h_U(     i,ny+2+j,     k,nVar) = h_U(   2,ny+1,   2,nVar);
		h_U(nx+2+i,ny+2+j,     k,nVar) = h_U(nx+1,ny+1,   2,nVar);
		
		h_U(     i,     j,nz+2+k,nVar) = h_U(   2,   2,nz+1,nVar);
		h_U(nx+2+i,     j,nz+2+k,nVar) = h_U(nx+1,   2,nz+1,nVar);
		h_U(     i,ny+2+j,nz+2+k,nVar) = h_U(   2,ny+1,nz+1,nVar);
		h_U(nx+2+i,ny+2+j,nz+2+k,nVar) = h_U(nx+1,ny+1,nz+1,nVar);
	      } // end for loop over i,j,k
	} // end for loop over nVar
      }	
    }

#ifdef __CUDACC__
    d_gravity.copyFromHost(h_gravity);
#endif

  } // HydroRunBase::init_hydro_Keplerian_disk

  // =======================================================
  // =======================================================
  /**
   * Falling bubble test.
   *
   */
  void HydroRunBase::init_hydro_falling_bubble()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());
    
    /* static gravity field */
    real_t& gravity_x = _gParams.gravity_x;
    real_t& gravity_y = _gParams.gravity_y;
    real_t& gravity_z = _gParams.gravity_z;
    real_t         P0 = 1.0f/(_gParams.gamma0-1.0f);
      
    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t &zMin = _gParams.zMin;

    real_t &xMax = _gParams.xMax;
    real_t &yMax = _gParams.yMax;
    real_t &zMax = _gParams.zMax;

    //real_t Lx = xMax-xMin;
    real_t Ly = yMax-yMin;
    real_t Lz = zMax-zMin;

    /* bubble's initial location */
    real_t radius = configMap.getFloat("falling-bubble", "radius", 0.1);
    real_t    x_c = configMap.getFloat("falling-bubble", "center_x", (xMin+xMax)/2);
    real_t    y_c = configMap.getFloat("falling-bubble", "center_y", yMin+0.8*Ly);
    real_t    z_c = configMap.getFloat("falling-bubble", "center_z", 0.0);

    /* initial falling velocity */
    real_t     v0 = configMap.getFloat("falling-bubble", "v0", 0.0);

    /* d0 is bubble's density */
    real_t     d0 = configMap.getFloat("falling-bubble", "d0", 2.0);
    real_t     d1 = configMap.getFloat("falling-bubble", "d1", 1.0);

    if (dimType == TWO_D) {
  
      // the initial condition must ensure the condition of
      // hydrostatic equilibrium for pressure P = P0 - 0.1*\rho*y
      for (int j=0; j<jsize; j++) {
	real_t y = yMin + dy/2 + (j-ghostWidth)*dy;

	for (int i=0; i<isize; i++) {
	  real_t x = xMin + dx/2 + (i-ghostWidth)*dx;

	  if ( y < yMin + 0.3*Ly ) {
	    h_U(i,j,ID) = d0;
	  } else {
	    h_U(i,j,ID) = d1;
	  }

	  // bubble
	  real_t r2 = (x-x_c)*(x-x_c)+(y-y_c)*(y-y_c);
	  if (r2<radius*radius)
	    h_U(i,j,ID) = d0;

	  h_U(i,j,IP) = P0 + h_U(i,j,ID)*(gravity_x*x + gravity_y*y);
	  h_U(i,j,IU) = ZERO_F;

	  if (r2<radius*radius)
	    h_U(i,j,IV) = v0;
	  else
	    h_U(i,j,IV) = ZERO_F;

	}
      }
    
      for (int j=0; j<jsize; j++) {
	for (int i=0; i<isize; i++) {
	  h_gravity(i,j,IX) = gravity_x;
	  h_gravity(i,j,IY) = gravity_y;
	}
      }

      if (ghostWidth == 2) {
	/* corner grid (not really needed (except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j) {
	      h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	      h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	      h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	      h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	    } // end for loop over i,j
	  
	} // end loop over nVar
      }

    } else { // THREE_D

      // the initial condition must ensure the condition of
      // hydrostatic equilibrium for pressure P = P0 - 0.1*\rho*y
      for (int k=0; k<ksize; k++) {
	real_t z = zMin + dz/2 + (k-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  real_t y = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    real_t x = xMin + dx/2 + (i-ghostWidth)*dx;
	    
	    if ( z < zMin + 0.3*Lz ) {
	      h_U(i,j,ID) = d0;
	    } else {
	      h_U(i,j,ID) = d1;
	    }

	    // bubble
	    real_t r2 = (x-x_c)*(x-x_c)+(y-y_c)*(y-y_c)+(z-z_c)*(z-z_c);
	    if (r2<radius*radius)
	      h_U(i,j,ID) = d0;
	    
	    h_U(i,j,k,IP) = P0 + h_U(i,j,k,ID)*(gravity_x*x + gravity_y*y + gravity_z*z);
	    h_U(i,j,k,IU) = ZERO_F;
	    h_U(i,j,k,IV) = ZERO_F;
	    if (r2<radius*radius)
	      h_U(i,j,k,IW) = v0;
	    else
	      h_U(i,j,k,IW) = ZERO_F;
	    
	  }
	}
      }

      for (int k=0; k<ksize; k++) {
	for (int j=0; j<jsize; j++) {
	  for (int i=0; i<isize; i++) {
	    h_gravity(i,j,k,IX) = gravity_x;
	    h_gravity(i,j,k,IY) = gravity_y;
	    h_gravity(i,j,k,IZ) = gravity_z;
	  }
	}
      }

      if (ghostWidth == 2) {
	/* fill the 8 grid corner (not really needed except for Kurganov-Tadmor) */
	for (int nVar=0; nVar<nbVar; ++nVar) {
	  for (int i=0; i<2; ++i)
	    for (int j=0; j<2; ++j)
	      for (int k=0; k<2; ++k) {
		h_U(     i,     j,     k,nVar) = h_U(   2,   2,   2,nVar);
		h_U(nx+2+i,     j,     k,nVar) = h_U(nx+1,   2,   2,nVar);
		h_U(     i,ny+2+j,     k,nVar) = h_U(   2,ny+1,   2,nVar);
		h_U(nx+2+i,ny+2+j,     k,nVar) = h_U(nx+1,ny+1,   2,nVar);
		
		h_U(     i,     j,nz+2+k,nVar) = h_U(   2,   2,nz+1,nVar);
		h_U(nx+2+i,     j,nz+2+k,nVar) = h_U(nx+1,   2,nz+1,nVar);
		h_U(     i,ny+2+j,nz+2+k,nVar) = h_U(   2,ny+1,nz+1,nVar);
		h_U(nx+2+i,ny+2+j,nz+2+k,nVar) = h_U(nx+1,ny+1,nz+1,nVar);
	      } // end for loop over i,j,k
	} // end for loop over nVar
      }	
    }

#ifdef __CUDACC__
    d_gravity.copyFromHost(h_gravity);
#endif

  } // HydroRunBase::init_hydro_falling_bubble

  // =======================================================
  // =======================================================
  void HydroRunBase::init_hydro_Riemann()
  {

    // reset domain
    memset(h_U.data(),0,h_U.sizeBytes());

    int nb=riemannConfId;
    if (nb<0)
      nb=0;
    else if (nb>NB_RIEMANN_CONFIG-1)
      nb=NB_RIEMANN_CONFIG-1;

    real_t q1[NVAR_2D],q2[NVAR_2D],q3[NVAR_2D],q4[NVAR_2D];

    q1[ID] = riemannConf[nb].pvar[0].rho;
    q1[IP] = riemannConf[nb].pvar[0].p; 
    q1[IU] = riemannConf[nb].pvar[0].u;
    q1[IV] = riemannConf[nb].pvar[0].v;

    q2[ID] = riemannConf[nb].pvar[1].rho;
    q2[IP] = riemannConf[nb].pvar[1].p; 
    q2[IU] = riemannConf[nb].pvar[1].u;
    q2[IV] = riemannConf[nb].pvar[1].v;

    q3[ID] = riemannConf[nb].pvar[2].rho;
    q3[IP] = riemannConf[nb].pvar[2].p; 
    q3[IU] = riemannConf[nb].pvar[2].u;
    q3[IV] = riemannConf[nb].pvar[2].v;

    q4[ID] = riemannConf[nb].pvar[3].rho;
    q4[IP] = riemannConf[nb].pvar[3].p; 
    q4[IU] = riemannConf[nb].pvar[3].u;
    q4[IV] = riemannConf[nb].pvar[3].v;

    primToCons_2D(q1,_gParams.gamma0);
    primToCons_2D(q2,_gParams.gamma0);
    primToCons_2D(q3,_gParams.gamma0);
    primToCons_2D(q4,_gParams.gamma0);  

    for( int j = ghostWidth; j < jsize-ghostWidth; ++j)
      for( int i = ghostWidth; i < isize-ghostWidth; ++i)
	{
	
	  if (i<(ghostWidth+nx/2)) {
	    if (j<(ghostWidth+ny/2)) {
	      // quarter 3
	      h_U(i,j,ID) = q3[ID];
	      h_U(i,j,IP) = q3[IP];
	      h_U(i,j,IU) = q3[IU];
	      h_U(i,j,IV) = q3[IV];
	    } else {
	      // quarter 2
	      h_U(i,j,ID) = q2[ID];
	      h_U(i,j,IP) = q2[IP];
	      h_U(i,j,IU) = q2[IU];
	      h_U(i,j,IV) = q2[IV];
	    }
	  } else {
	    if (j<(ghostWidth+ny/2)) {
	      // quarter 4
	      h_U(i,j,ID) = q4[ID];
	      h_U(i,j,IP) = q4[IP];
	      h_U(i,j,IU) = q4[IU];
	      h_U(i,j,IV) = q4[IV];
	    } else {
	      // quarter 1
	      h_U(i,j,ID) = q1[ID];
	      h_U(i,j,IP) = q1[IP];
	      h_U(i,j,IU) = q1[IU];
	      h_U(i,j,IV) = q1[IV];
	    }     
	  }
	}

    if (ghostWidth == 2) {
      /* fill corner values */
      for (int nVar=0; nVar<nbVar; ++nVar) {    
	for (int i=0; i<2; ++i)
	  for (int j=0; j<2; ++j) {
	    h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	    h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	    h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	    h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	  } // end for loop over i,j
      } // end loop over nVar
    }

  } // HydroRunBase::init_hydro_Riemann

  // =======================================================
  // =======================================================
  /**
   *
   * This initialization routine is inspired by Enzo. 
   * See routine named turboinit by A. Kritsuk in Enzo.
   */
  void HydroRunBase::init_hydro_turbulence()
  {

    // reset domain
    memset(h_U.data(),0,h_U.sizeBytes());

    /* get initial conditions */
    real_t d0 = configMap.getFloat("turbulence", "density",  1.0);
    real_t initialDensityPerturbationAmplitude = 
      configMap.getFloat("turbulence", "initialDensityPerturbationAmplitude", 0.0);

    real_t P0 = configMap.getFloat("turbulence", "pressure", 1.0);

    int seed = configMap.getInteger("turbulence", "random_seed", 33);
    srand(seed);

    if (dimType == TWO_D) {
    
      std::cerr << "Turbulence problem is not available in 2D...." << std::endl;

    } else { // THREE_D

      // initialize h_randomForcing
      init_randomForcing();

      // initialize h_U
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	    // fill density
	    h_U(i,j,k,ID) = d0 * (1.0 + initialDensityPerturbationAmplitude *  ( (float)rand()/(float)(RAND_MAX)   - 0.5 ) );
	    
	    // convert h_randomForce into momentum
	    h_U(i,j,k,IU) = h_U(i,j,k,ID) * h_randomForcing(i,j,k,IX);
	    h_U(i,j,k,IV) = h_U(i,j,k,ID) * h_randomForcing(i,j,k,IY);
	    h_U(i,j,k,IW) = h_U(i,j,k,ID) * h_randomForcing(i,j,k,IZ);

	    // compute total energy
	    h_U(i,j,k,IP) = P0/(_gParams.gamma0-ONE_F) + 
	      0.5 * ( h_U(i,j,k,IU) * h_U(i,j,k,IU) +
		      h_U(i,j,k,IV) * h_U(i,j,k,IV) +
		      h_U(i,j,k,IW) * h_U(i,j,k,IW) ) / h_U(i,j,k,ID);

	  } // end for i,j,k

    } // end THREE_D

  } // HydroRunBase::init_hydro_turbulence

  // =======================================================
  // =======================================================
  /**
   *
   * Initialization for turbulence run using Ornstein-Uhlenbeck forcing.
   *
   */
  void HydroRunBase::init_hydro_turbulence_Ornstein_Uhlenbeck()
  {

    // reset domain
    memset(h_U.data(),0,h_U.sizeBytes());

    /* get initial conditions */
    real_t d0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "density",  1.0);
    real_t initialDensityPerturbationAmplitude = 
      configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "initialDensityPerturbationAmplitude", 0.0);

    real_t P0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "pressure", 1.0);

    int seed = configMap.getInteger("turbulence-Ornstein-Uhlenbeck", "random_seed", 33);
    srand(seed);

    // initialize forcing generator
    pForcingOrnsteinUhlenbeck -> init_forcing();

    // initialize h_U

    if (dimType == TWO_D) {
    
      std::cerr << "Turbulence-Ornstein-Uhlenbeck problem is not available in 2D...." << std::endl;
      
    } else { // THREE_D
      
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    // fill density
	    h_U(i,j,k,ID) = d0 * (1.0 + initialDensityPerturbationAmplitude *  ( (1.0*rand())/RAND_MAX - 0.5 ) );

	    // fill momentum
	    h_U(i,j,k,IU) = ZERO_F;
	    h_U(i,j,k,IV) = ZERO_F;
	    h_U(i,j,k,IW) = ZERO_F;

	    // fill total energy
	    h_U(i,j,k,IP) = P0/(_gParams.gamma0-ONE_F);

	  } // end for i,j,k

    } // end THREE_D

  } // HydroRunBase::init_hydro_turbulence_Ornstein_Uhlenbeck

  // =======================================================
  // =======================================================
  int HydroRunBase::init_simulation(const std::string problemName)
  {

    // test if we are performing a re-start run (default : false)
    bool restartEnabled = configMap.getBool("run","restart",false);
    int timeStep = 0;

    /*
     * check if performing restart run
     */
    if (restartEnabled) { // load data from input data file

      /* initial condition in grid interior */
      memset(h_U.data(),0,h_U.sizeBytes());
     
      // get input filename from configMap
      std::string inputFilename = configMap.getString("run", "restart_filename", "");
      
      // get output directory
      std::string outputDir    = configMap.getString("output", "outputDir", "./");

      // upscale init data from a file twice smaller
      bool restartUpscaleEnabled = configMap.getBool("run","restart_upscale",false);
      
      if (restartUpscaleEnabled) { // load low resolution data from file
      
	// allocate h_input (half resolution, ghost included)
	HostArray<real_t> h_input;
	h_input.allocate(make_uint4(nx/2+2*ghostWidth, 
				    ny/2+2*ghostWidth,
				    nz/2+2*ghostWidth,
				    nbVar));

	// read input date into temporary array h_input
	bool halfResolution=true;
	inputHdf5(h_input, outputDir+"/"+inputFilename, halfResolution);

	// upscale h_input into h_U (i.e. double resolution)
	upscale(h_U, h_input);

      } else { // standard restart

	// read input HDF5 file into h_U buffer , and return time Step
	timeStep = inputHdf5(h_U, outputDir+"/"+inputFilename);

      } // if (restartUpscaleEnabled)

      // in case of turbulence problem, we also need to re-initialize the
      // random forcing field
      if (!problemName.compare("turbulence")) {
	this->init_randomForcing();
      } 

      // in case of Ornstein-Uhlenbeck turbulence problem, 
      // we also need to re-initialize the random forcing field
      if (!problemName.compare("turbulence-Ornstein-Uhlenbeck")) {
	
	bool restartEnabled = true;

	std::string forcing_filename = configMap.getString("turbulence-Ornstein-Uhlenbeck", "forcing_input_file",  "");

	if (restartUpscaleEnabled) {

	  // use default parameter when restarting and upscaling
	  pForcingOrnsteinUhlenbeck -> init_forcing(false);

	} else if ( forcing_filename.size() != 0) {

	  // if forcing filename is provided, we use it
	  pForcingOrnsteinUhlenbeck -> init_forcing(false); // call to allocate
	  pForcingOrnsteinUhlenbeck -> input_forcing(forcing_filename);

	} else {

	  // the forcing parameter filename is build upon configMap information
	  pForcingOrnsteinUhlenbeck -> init_forcing(restartEnabled, timeStep);

	}

      } // end restart problem turbulence-Ornstein-Uhlenbeck

      // some extra stuff that need to be done here
      restart_run_extra_work();

    } else { // perform regular initialization
      
      if (!problemName.compare("jet")) {
	this->init_hydro_jet();
      } else if (!problemName.compare("sod")) {
	this->init_hydro_sod();
      } else if (!problemName.compare("implode")) {
	this->init_hydro_implode();
      } else if (!problemName.compare("blast")) {
	this->init_hydro_blast();
      } else if (!problemName.compare("Gresho-vortex")) {
	this->init_hydro_Gresho_vortex();
      } else if (!problemName.compare("Kelvin-Helmholtz")) {
	this->init_hydro_Kelvin_Helmholtz();
      } else if (!problemName.compare("Rayleigh-Taylor")) {
	this->init_hydro_Rayleigh_Taylor();
      } else if (!problemName.compare("Keplerian-disk")) {
	this->init_hydro_Keplerian_disk();
      } else if (!problemName.compare("falling-bubble")) {
	this->init_hydro_falling_bubble();
      } else if (!problemName.compare("riemann2d")) {
	this->init_hydro_Riemann();
      } else if (!problemName.compare("turbulence")) {
	this->init_hydro_turbulence();
      } else if (!problemName.compare("turbulence-Ornstein-Uhlenbeck")) {
	this->init_hydro_turbulence_Ornstein_Uhlenbeck();
      } else {
	std::cerr << "given problem parameter is: " << problem << std::endl;
	std::cerr << "unknown problem name; please set hydro parameter \"problem\" to a valid value !!!" << std::endl;
      }

    } // end regular initialization

#ifdef __CUDACC__
    d_U.copyFromHost(h_U); // load data into the VRAM
    d_U2.copyFromHost(h_U); // load data into the VRAM
#else
    // copy h_U into h_U2
    h_U.copyTo(h_U2);
#endif // __CUDACC__

    // do we force timeStep to be zero ?
    bool resetTimeStep = configMap.getBool("run","restart_reset_timestep",false);
    if (resetTimeStep)
      timeStep=0;

    return timeStep;

  } // HydroRunBase::init_simulation

  // =======================================================
  // =======================================================
  void HydroRunBase::restart_run_extra_work()
  {

  } // HydroRunBase::restart_run_extra_work

  // =======================================================
  // =======================================================
  void HydroRunBase::init_randomForcing()
  {

    if (dimType == TWO_D) {
    
      std::cerr << "Turbulence problem is not available in 2D...." << std::endl;

    } else { // THREE_D

      real_t d0   = configMap.getFloat("turbulence", "density",  1.0);
      real_t eDot = configMap.getFloat("turbulence", "edot", -1.0);
      
      real_t randomForcingMachNumber = configMap.getFloat("turbulence", "machNumber", 0.0);
      std::cout << "Random forcing Mach number is " << randomForcingMachNumber << std::endl;
      
      /* check parameters as in Enzo */
      /* if eDot is not set in parameter file or negative, it is 
	 set from MacLow1999 formula, see comments in Enzo's file 
	 TurbulenceSimulationInitialize.C */
      if (eDot < 0) {
	real_t boxSize = _gParams.xMax - _gParams.xMin;
	real_t boxMass = boxSize*boxSize*boxSize*d0;
	real_t vRms    = randomForcingMachNumber / sqrt(1.0); // sound speed is one
	/*if (_gParams.cIso > 0)
	  vRms = randomForcingMachNumber * _gParams.cIso;*/
	eDot = 0.81/boxSize*boxMass*vRms*vRms*vRms;
	eDot *= 0.8;
      }
      randomForcingEdot = eDot;
      std::cout << "Using random forcing with eDot : " << eDot << std::endl;
            
      /* turbulence */
      // compute random field
      turbulenceInit(isize, jsize, ksize, 
		     -ghostWidth,-ghostWidth,-ghostWidth,
		     nx, randomForcingMachNumber,
		     &(h_randomForcing(0,0,0,IX)), 
		     &(h_randomForcing(0,0,0,IY)),
		     &(h_randomForcing(0,0,0,IZ)) );
      
#ifdef __CUDACC__
      // we also need to copy
      d_randomForcing.copyFromHost(h_randomForcing);
#endif // __CUDACC__

    } // end THREE_D

  } // HydroRunBase::init_randomForcing

  // =======================================================
  // =======================================================
  void HydroRunBase::copyGpuToCpu(int nStep)
  {

#ifdef __CUDACC__
    if (nStep % 2 == 0)
      d_U.copyToHost(h_U);
    else
      d_U2.copyToHost(h_U2);
#else
    (void) nStep;
#endif // __CUDACC__

  } // HydroRunBase::copyGpuToCpu

  // =======================================================
  // =======================================================
  void HydroRunBase::setupHistory_hydro()
  {
    // set history_hydro_type

    // history enabled ?
    bool historyEnabled = configMap.getBool("history","enabled",false);

    if (historyEnabled) {

      if (!problem.compare("turbulence")) {
	history_hydro_method = &HydroRunBase::history_hydro_turbulence;
      } else {
	history_hydro_method = &HydroRunBase::history_hydro_empty;
      }

    } else { // history disabled

      history_hydro_method = &HydroRunBase::history_hydro_empty;
      
    } 

  } // HydroRunBase::setupHistory_hydro

  // =======================================================
  // =======================================================
  void HydroRunBase::history_hydro(int nStep, real_t dt)
  {

    // call the actual history method
    ((*this).*history_hydro_method)(nStep,dt);

  } // HydroRunBase::history_hydro

  // =======================================================
  // =======================================================
  void HydroRunBase::history_hydro_empty(int nStep, real_t dt)
  {
  
    (void) nStep;
    (void) dt;
  
  } // HydroRunBase::history_hydro_empty

  // =======================================================
  // =======================================================
  void HydroRunBase::history_hydro_turbulence(int nStep, real_t dt)
  {
    
    std::cout << "History for turbulence problem at time " << totalTime << "\n";

    if (dimType == TWO_D) {

      // don't do anything, this problem is not available in 2D

    } else {
      
      // history file name
      std::string fileName = configMap.getString("history",
						 "filename", 
						 "history.txt");
      // get output prefix / outputDir
      std::string outputDir    = configMap.getString("output", "outputDir", "./");
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
      
      // build full path filename
      fileName = outputDir + "/" + outputPrefix + "_" + fileName;
      
      // open history file
      std::ofstream histo;
      histo.open (fileName.c_str(), std::ios::out | std::ios::app | std::ios::ate); 
      
      // if this is the first time we call history, print header
      if (totalTime <= 0) {
	histo << "# history " << current_date() << std::endl;
	
	bool restartEnabled = configMap.getBool("run","restart",false);
	if (restartEnabled)
	  histo << "# history : this is a restart run\n";
	
	// write header (which variables are dumped)
	// Ma_s is the sonic Mach number Ma_s = v_rms/c_s
	// v_rms = sqrt(<v^2)  
	histo << "# totalTime dt mass eKin mean_rho mean_rhovx mean_rhovy mean_rhovz Ma_s\n";

      } // end print header

      // make sure Device data are copied back onto Host memory
      // which data to save ?
      copyGpuToCpu(nStep);
      
      HostArray<real_t> &U = getDataHost(nStep);

      //const double pi = 2*asin(1.0);
      
      double mass       = 0.0, eKin       = 0.0;
      double mean_rhovx = 0.0, mean_rhovy = 0.0, mean_rhovz = 0.0;
      double mean_v2    = 0.0, mean_rho   = 0.0;
      
      // do a local reduction
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	    real_t rho = U(i,j,k,ID);
	    mass += rho;

	    eKin += SQR( U(i,j,k,IU) ) / rho;
	    eKin += SQR( U(i,j,k,IV) ) / rho;
	    eKin += SQR( U(i,j,k,IW) ) / rho;
	    
	    mean_v2 += SQR( U(i,j,k,IU)/rho );
	    mean_v2 += SQR( U(i,j,k,IV)/rho );
	    mean_v2 += SQR( U(i,j,k,IW)/rho );

	    mean_rhovx += U(i,j,k,IU);
	    mean_rhovy += U(i,j,k,IV);
	    mean_rhovz += U(i,j,k,IW);

	    mean_rho += rho;
	    
	  } // end for i
	} // end for j
      } // end for k
      
      double dTau = dx*dy*dz/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin)/
	(_gParams.zMax- _gParams.zMin);
      
      mass    = mass*dTau;

      eKin = eKin*dTau;

      mean_rhovx = mean_rhovx*dTau;
      mean_rhovy = mean_rhovy*dTau;
      mean_rhovz = mean_rhovz*dTau;

      mean_v2  = mean_v2*dTau;
      
      mean_rho = mean_rho*dTau;

      real_t &cIso = _gParams.cIso;
      double Ma_s = -1.0;
      if (cIso >0)
	Ma_s = sqrt(mean_v2)/cIso; 

      histo << totalTime  << "\t" << dt         << "\t" 
	    << mass       << "\t" 
	    << eKin       << "\t"
	    << mean_rho   << "\t"
	    << mean_rhovx << "\t" << mean_rhovy << "\t" << mean_rhovz << "\t"
	    << Ma_s       << "\t" << "\n";
		   
      histo.close();

    } // end THREE_D

    bool structureFunctionsEnabled = configMap.getBool("structureFunctions","enabled",false);
    if ( structureFunctionsEnabled ) {
      HostArray<real_t> &U = getDataHost(nStep);
      structure_functions_hydro(nStep,configMap,_gParams,U);
    }

  } // HydroRunBase::history_hydro_turbulence

  
  // =======================================================
  // =======================================================
  void rescaleToZeroOne(real_t *data, int size, real_t &min, real_t &max) {

    min = data[0];
    max = data[0];

    // get min and max value
    for (int i=1; i<size; ++i) {
      min = (data[i]<min) ? data[i] : min;
      max = (data[i]>max) ? data[i] : max;
    }

    // rescale to range 0.0 - 1.0
    for (int i=0; i<size; ++i) {
      data[i] = (data[i]-min)/(max-min);
    }

  } // rescaleToZeroOne

} // namespace hydroSimu
