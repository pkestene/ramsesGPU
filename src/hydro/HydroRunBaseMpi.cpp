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
 * \file HydroRunBaseMpi.cpp
 * \brief Implements class HydroRunBaseMpi.
 *
 * \date 12 Oct 2010
 * \author Pierre Kestener
 *
 * $Id: HydroRunBaseMpi.cpp 3597 2014-11-04 17:34:47Z pkestene $
 */
#include "make_boundary_base.h"
#include <mpiBorderUtils.h>

#include "HydroRunBaseMpi.h"

#include "constoprim.h"

#include "utilities.h"

#include "turbulenceInit.h" 
#include "structureFunctionsMpi.h"

#include "../utils/monitoring/date.h" // for current_date

#include <limits> // for std::numeric_limits

// include CUDA kernel when necessary
#ifdef __CUDACC__
#include "cmpdt.cuh"
#include "cmpdt_mhd.cuh"
#include "viscosity.cuh"
#include "viscosity_zslab.cuh"
#include "gravity.cuh"
#include "gravity_zslab.cuh"
#include "hydro_update.cuh"
#include "hydro_update_zslab.cuh"
#include "resistivity.cuh"
#include "resistivity_zslab.cuh"
#include "mhd_ct_update.cuh"
#include "mhd_ct_update_zslab.cuh"
#include "random_forcing.cuh"
#include "copyFaces.cuh"
#endif // __CUDACC__
#include "mhd_utils.h"

#include <typeinfo> // for typeid

// for vtk file format output
#ifdef USE_VTK
#include "vtk_inc.h"
#endif // USE_VTK

// for HDF5 file format output
#ifdef USE_HDF5
#include <hdf5.h>
#endif // USE_HDF5

// for Parallel-netCDF support
#ifdef USE_PNETCDF
#include <pnetcdf.h>

#define PNETCDF_HANDLE_ERROR {                        \
    if (err != NC_NOERR)                              \
        printf("Error at line %d (%s)\n", __LINE__,   \
               ncmpi_strerror(err));                  \
}

#endif // USE_PNETCDF

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // HydroRunBaseMpi class methods body
  ////////////////////////////////////////////////////////////////////////////////

  HydroRunBaseMpi::HydroRunBaseMpi(ConfigMap &_configMap)
    : HydroMpiParameters(_configMap),
      dx(_gParams.dx),
      dy(_gParams.dy),
      dz(_gParams.dz),
      mpi_data_type(MpiComm::FLOAT),
      totalTime(0),
      cmpdtBlockCount(192),
      randomForcingBlockCount(192),
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
    , history_method(NULL)     
  {
    
    // runtime determination if we are using float ou double
    mpi_data_type = typeid(1.0f).name() == typeid((real_t)1.0f).name() ? MpiComm::FLOAT : MpiComm::DOUBLE;
    
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
     * allocate memory for main domain (nbVar is set in
     * HydroParameters class constructor)
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
#endif // __CUDACC__


    /*
     * compute time step initialization.
     */

#ifdef __CUDACC__

    // for time step computation (!!! GPU only !!!)
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
    if ( !problem.compare("Rayleigh-Taylor") )
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
      
    } // end gravityEnabled

    /*
     * allocate memory for border buffers.
     * NOTES for performances:
     * - The device border buffer are allocated using cudaMalloc
     *   (linear memory) instead of cudaMalloc (pitched memory) : this
     *   features results in roughly 10x faster transfert between CPU
     *   and GPU (around 2 GBytes/s instead of ~150 MBytes/s !!!)
     * - The host border buffers are allocated using new (PAGEABLE)
     *   when using the pure CPU+MPI version
     * - The host border buffers are allocated using cudaMallocHost
     *   (PINNED) when using the CUDA+MPI version
     */
    if (dimType == TWO_D) {
      
      HostArray<real_t>::HostMemoryAllocType memAllocType;
#ifdef __CUDACC__
      memAllocType = HostArray<real_t>::PINNED;
#else
      memAllocType = HostArray<real_t>::PAGEABLE;
#endif // __CUDACC__

      unsigned int gw = ghostWidth;

      borderBufSend_xmin.allocate(make_uint3(   gw,jsize, nbVar), memAllocType);
      borderBufSend_xmax.allocate(make_uint3(   gw,jsize, nbVar), memAllocType);
      borderBufSend_ymin.allocate(make_uint3(isize,   gw, nbVar), memAllocType);
      borderBufSend_ymax.allocate(make_uint3(isize,   gw, nbVar), memAllocType);

      borderBufRecv_xmin.allocate(make_uint3(   gw,jsize, nbVar), memAllocType);
      borderBufRecv_xmax.allocate(make_uint3(   gw,jsize, nbVar), memAllocType);
      borderBufRecv_ymin.allocate(make_uint3(isize,   gw, nbVar), memAllocType);
      borderBufRecv_ymax.allocate(make_uint3(isize,   gw, nbVar), memAllocType);

#ifdef __CUDACC__
      borderBuffer_device_xdir.allocate(make_uint3(   gw,jsize, nbVar), DeviceArray<real_t>::LINEAR);
      borderBuffer_device_ydir.allocate(make_uint3(isize,   gw, nbVar), DeviceArray<real_t>::LINEAR);
#endif // __CUDACC__

    } else {
      
      HostArray<real_t>::HostMemoryAllocType memAllocType;

      /*
       * In the 3D case, there might be not enought PINNED memory.
       * So we fall back, to PAGEABLE memory.
       */
      // #ifdef __CUDACC__
      //       memAllocType = HostArray<real_t>::PINNED;
      // #else
      //       memAllocType = HostArray<real_t>::PAGEABLE;
      // #endif // __CUDACC__
      memAllocType = HostArray<real_t>::PAGEABLE;

      int &gw = ghostWidth;

      borderBufSend_xmin.allocate(make_uint4(   gw,jsize, ksize, nbVar), memAllocType);
      borderBufSend_xmax.allocate(make_uint4(   gw,jsize, ksize, nbVar), memAllocType);
      borderBufSend_ymin.allocate(make_uint4(isize,   gw, ksize, nbVar), memAllocType);
      borderBufSend_ymax.allocate(make_uint4(isize,   gw, ksize, nbVar), memAllocType);
      borderBufSend_zmin.allocate(make_uint4(isize,jsize,    gw, nbVar), memAllocType);
      borderBufSend_zmax.allocate(make_uint4(isize,jsize,    gw, nbVar), memAllocType);

      borderBufRecv_xmin.allocate(make_uint4(   gw,jsize, ksize, nbVar), memAllocType);
      borderBufRecv_xmax.allocate(make_uint4(   gw,jsize, ksize, nbVar), memAllocType);
      borderBufRecv_ymin.allocate(make_uint4(isize,   gw, ksize, nbVar), memAllocType);
      borderBufRecv_ymax.allocate(make_uint4(isize,   gw, ksize, nbVar), memAllocType);
      borderBufRecv_zmin.allocate(make_uint4(isize,jsize,    gw, nbVar), memAllocType);
      borderBufRecv_zmax.allocate(make_uint4(isize,jsize,    gw, nbVar), memAllocType);

#ifdef __CUDACC__
      borderBuffer_device_xdir.allocate(make_uint4(   gw,jsize, ksize, nbVar), DeviceArray<real_t>::LINEAR);
      borderBuffer_device_ydir.allocate(make_uint4(isize,   gw, ksize, nbVar), DeviceArray<real_t>::LINEAR);
      borderBuffer_device_zdir.allocate(make_uint4(isize,jsize,    gw, nbVar), DeviceArray<real_t>::LINEAR);
#endif // __CUDACC__
    }

    initRiemannConfig2d(riemannConf);
    riemannConfId = configMap.getInteger("hydro", "riemann_config_number", 0);

    // runtime determination if we are using float ou double (for MPI communication)
    data_type = typeid(1.0f).name() == typeid((real_t)1.0f).name() ? hydroSimu::MpiComm::FLOAT : hydroSimu::MpiComm::DOUBLE;

    /*
     * VERY important:
     * make sure variables declared as __constant__ are copied to device
     * for current compilation unit
     */
    copyToSymbolMemory();

  } // HydroRunBaseMpi::HydroRunBaseMpi

  // =======================================================
  // =======================================================
  HydroRunBaseMpi::~HydroRunBaseMpi()
  {

    if (randomForcingOrnsteinUhlenbeckEnabled)
      delete pForcingOrnsteinUhlenbeck;

  } // HydroRunBaseMpi::~HydroRunBaseMpi

  // =======================================================
  // =======================================================
  real_t HydroRunBaseMpi::compute_dt_local(int useU)
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
      checkCudaErrorMpi("compute_dt_local 2D",myRank);
      d_invDt.copyToHost(h_invDt);
      checkCudaErrorMpi("compute_dt_local 2D copyToHost",myRank);

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
      checkCudaErrorMpi("compute_dt_local 3D",myRank);
      d_invDt.copyToHost(h_invDt);
      checkCudaErrorMpi("compute_dt_local 3D copyToHost",myRank);

    } // end call cuda kernel for invDt reduction

    real_t* invDt = h_invDt.data();

    for(uint i = 0; i < cmpdtBlockCount; ++i) {
      maxInvDt = FMAX ( maxInvDt, invDt[i]);
    }
    
    if (enableJet) {
      maxInvDt = FMAX ( maxInvDt, (this->ujet + this->cjet)/dx );
    }
    
    return cfl / maxInvDt;
    
  } // HydroRunBaseMpi::compute_dt_local -- GPU version
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

  } // HydroRunBaseMpi::compute_dt_local -- CPU version
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  real_t HydroRunBaseMpi::compute_dt(int useU)
  {
    real_t dt_local;
    real_t dt;

    // compute dt inside MPI block
    dt_local = compute_dt_local(useU);
    // synchronize all MPI processes
    communicator->synchronize();

    // do a MPI reduction
    //MPI_Reduce(&dt_local, &dt,1,MPI_REAL,MPI_MIN,0,communicator->comm_);
    communicator->allReduce(&dt_local, &dt,1,mpi_data_type,MpiComm::MIN);

    // return global time step
    return dt;

  } // HydroRunBaseMpi::compute_dt

  // =======================================================
  // =======================================================
  real_t HydroRunBaseMpi::compute_dt_mhd_local(int useU)
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

      cmpdt_2d_mhd<CMPDT_BLOCK_SIZE>
	<<<cmpdtBlockCount, 
	CMPDT_BLOCK_SIZE, 
	CMPDT_BLOCK_SIZE*sizeof(real_t)>>>(uData, 
					   d_invDt.data(), 
					   d_U.pitch(), 
					   d_U.dimx(),
					   d_U.dimy());
      checkCudaErrorMpi("HydroRunBaseMpi cmpdt_2d_mhd error",myRank);
      d_invDt.copyToHost(h_invDt);
      checkCudaErrorMpi("HydroRunBaseMpi h_invDt error",myRank);

    } else { // THREE_D

      cmpdt_3d_mhd<CMPDT_BLOCK_SIZE>
	<<<cmpdtBlockCount, 
	CMPDT_BLOCK_SIZE, 
	CMPDT_BLOCK_SIZE*sizeof(real_t)>>>(uData, 
					   d_invDt.data(), 
					   d_U.pitch(),
					   d_U.dimx(),
					   d_U.dimy(),
					   d_U.dimz());
      checkCudaErrorMpi("HydroRunBaseMpi cmpdt_3d_mhd error",myRank);
      d_invDt.copyToHost(h_invDt);
      checkCudaErrorMpi("HydroRunBaseMpi h_invDt error",myRank);
      
    } // end THREE_D

    real_t* invDt = h_invDt.data();
    
    for(uint i = 0; i < cmpdtBlockCount; ++i) {
      //std::cout << "invDt[" << i << "] = " << invDt[i] << std::endl;
      maxInvDt = FMAX ( maxInvDt, invDt[i]);
    }
    
    if (enableJet) {
      maxInvDt = FMAX ( maxInvDt, (this->ujet + this->cjet)/dx );
    }

    //std::cout << "computed dt : " << cfl/maxInvDt << std::endl;

    return cfl / maxInvDt;

  } // HydroRunBaseMpi::compute_dt_mhd_local

#else // CPU version of compute_dt_mhd_local

  {
    // time step inverse
    real_t invDt = gParams.smallc / (cfl * FMIN(dx,dy));

    // choose between h_U and h_U2
    real_t *uData;
    if (useU == 0)
      uData = h_U.data();
    else
      uData = h_U2.data();

    /*
     * 
     */

    // section / domain size
    //int arraySize      =  h_U.section();

    if (dimType == TWO_D) {
            
      int &geometry = ::gParams.geometry;

      int physicalDim[2] = {(int) h_U.pitch(), (int) h_U.dimy()};
      
      // for loop over inner region
      for (int j = ghostWidth; j < jsize-ghostWidth; j++)
	for (int i = ghostWidth; i < isize-ghostWidth; i++)
	  {
	    real_t q[NVAR_MHD];
	    real_t c;
	    int index = j*isize+i;
	    // convert conservative to primitive variables (stored in h_Q)
	    computePrimitives_MHD_2D(uData, physicalDim, index, c, q, ZERO_F);

	    // compute fastest information speeds
	    real_t fastInfoSpeed[3];
	    find_speed_info<TWO_D>(q, fastInfoSpeed);

	    real_t vx = fastInfoSpeed[IX];
	    real_t vy = fastInfoSpeed[IY];

	    if (enableJet) {
	      invDt = FMAX ( FMAX ( invDt, vx / dx + vy / dy ), (this->ujet + this->cjet)/dx );
	    } else {
	      invDt =        FMAX ( invDt, vx / dx + vy / dy );	      
	      if (geometry != GEO_CARTESIAN) {
		int iG = i + nx*myMpiPos[0];
		real_t xPos = ::gParams.xMin + dx/2 + (iG-ghostWidth)*dx;
		invDt =      FMAX ( invDt, vx / dx + vy / dy / xPos );
	      }
	    }
 
	  } // end for(i,j)
            
    } else { // THREE_D

      real_t &Omega0   = ::gParams.Omega0;
      real_t  deltaX   = ::gParams.xMax - ::gParams.xMin; // equivalent to xMax-xMin
      int    &geometry = ::gParams.geometry;

      int physicalDim[3] = {(int) h_U.pitch(), (int) h_U.dimy(), (int) h_U.dimz()};
      
      // for loop over inner region
      for (int k = ghostWidth; k < ksize-ghostWidth; k++)
	for (int j = ghostWidth; j < jsize-ghostWidth; j++)
	  for (int i = ghostWidth; i < isize-ghostWidth; i++)
	    {
	      real_t q[NVAR_MHD];
	      real_t c;
	      int index = k*jsize*isize+j*isize+i;
	      // convert conservative to primitive variables (stored in h_Q)
	      computePrimitives_MHD_3D(uData, physicalDim, index, c, q, ZERO_F);
	      
	      // compute fastest information speeds
	      real_t fastInfoSpeed[3];
	      find_speed_info<THREE_D>(q, fastInfoSpeed);

	      real_t vx = fastInfoSpeed[IX];
	      real_t vy = fastInfoSpeed[IY];
	      if (geometry == GEO_CARTESIAN and Omega0 > 0) {
		vy += 1.5*Omega0*deltaX/2;
	      }
	      real_t vz = fastInfoSpeed[IZ];
	      
	      if (enableJet) {
		invDt = FMAX ( FMAX(  invDt, vx / dx + vy / dy + vz / dz ), (this->ujet + this->cjet)/dx );
	      } else {
		invDt =        FMAX ( invDt, vx / dx + vy / dy + vz / dz);
	      }
	      
	    } // end for(i,j,k)
      
    } // end THREE_D
    
    return cfl / invDt;

  } // HydroRunBaseMpi::compute_dt_mhd_local

#endif // __CUDACC__

  // =======================================================
  // =======================================================
  real_t HydroRunBaseMpi::compute_dt_mhd(int useU)
  {
    real_t dt_local;
    real_t dt;

    // compute dt inside MPI block
    dt_local = compute_dt_mhd_local(useU);
    // synchronize all MPI processes
    communicator->synchronize();

    // do a MPI reduction
    //MPI_Reduce(&dt_local, &dt,1,MPI_REAL,MPI_MIN,0,communicator->comm_);
    communicator->allReduce(&dt_local, &dt,1,mpi_data_type,MpiComm::MIN);

    // return global time step
    return dt;

  } // HydroRunBaseMpi::compute_dt_mhd

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_viscosity_flux(HostArray<real_t>  &U, 
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

  } // HydroRunBaseMpi::compute_viscosity_flux for 2D data (CPU version)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_viscosity_flux(DeviceArray<real_t>  &U, 
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
    checkCudaErrorMpi("in HydroRunBase :: kernel_viscosity_forces_2d",myRank);

  } // HydroRunBaseMpi::compute_viscosity_flux for 2D data (GPU version)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_viscosity_flux(HostArray<real_t>  &U, 
					       HostArray<real_t>  &flux_x, 
					       HostArray<real_t>  &flux_y, 
					       HostArray<real_t>  &flux_z,
					       real_t              dt) 
  {
    real_t &cIso = _gParams.cIso;
    real_t &nu   = _gParams.nu;
    const real_t two3rd = 2./3.;

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

  } // HydroRunBaseMpi::compute_viscosity_flux for 3D data (CPU version)
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_viscosity_flux(DeviceArray<real_t>  &U, 
					       DeviceArray<real_t>  &flux_x, 
					       DeviceArray<real_t>  &flux_y, 
					       DeviceArray<real_t>  &flux_z, 
					       real_t                dt) 
  {

    dim3 dimBlock(VISCOSITY_3D_DIMX,
		  VISCOSITY_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, VISCOSITY_3D_DIMX_INNER),
		 blocksFor(jsize, VISCOSITY_3D_DIMY_INNER));

    kernel_viscosity_forces_3d<<< dimGrid, dimBlock >>> (U.data(), flux_x.data(), flux_y.data(), flux_z.data(),
							 ghostWidth, U.pitch(),
							 U.dimx(), U.dimy(), U.dimz(), dt, dx, dy, dz);
    checkCudaErrorMpi("in HydroRunBase :: kernel_viscosity_forces_3d",myRank);

  } // HydroRunBaseMpi::compute_viscosity_flux for 3D data (GPU version)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_viscosity_flux(HostArray<real_t>  &U, 
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

  } // HydroRunBaseMpi::compute_viscosity_flux for 3D data (CPU version)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_viscosity_flux(DeviceArray<real_t>  &U, 
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
    checkCudaErrorMpi("HydroRunBase :: kernel_viscosity_forces_3d_zslab",myRank);
    
  } // HydroRunBaseMpi::compute_viscosity_flux for 3D data (GPU version)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  real_t HydroRunBaseMpi::compute_random_forcing_normalization(HostArray<real_t>  &U, 
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

    /*
     * local reduction inside MPI sub-domain
     */
    real_t reduceValue[nbRandomForcingReduction] = { 0.0, 0.0, 0.0,
						     0.0, 0.0, 0.0,
						     0.0, 0.0, 0.0 };
    // reduceValue[7] is a minimum
    reduceValue[7] = std::numeric_limits<float>::max();

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

    /*
     * Global reduction over all MPI sub-domains
     */
    real_t reduceValueGlob[nbRandomForcingReduction];
    communicator->synchronize();
    communicator->allReduce(  reduceValue,       reduceValueGlob,    7,mpi_data_type,MpiComm::SUM);
    communicator->allReduce(&(reduceValue[7]), &(reduceValueGlob[7]),1,mpi_data_type,MpiComm::MIN);
    communicator->allReduce(&(reduceValue[8]), &(reduceValueGlob[8]),1,mpi_data_type,MpiComm::MAX);

    int64_t nbCells = (nx*mx)*(ny*my)*(nz*mz);

    real_t norm;
    if (randomForcingEdot == 0) {
      norm = 0;
    } else {
      norm = ( SQRT( SQR(reduceValueGlob[0]) + 
		     reduceValueGlob[1] * dt * randomForcingEdot * 2 * nbCells) - 
	       reduceValueGlob[0] ) / reduceValueGlob[1];
    }

    /**/
    // if (myRank == 0) 
    //   printf("---- %f %f %f %f %f %f %f %f %f\n",reduceValueGlob[0],reduceValueGlob[1],
    // 	     reduceValueGlob[2],reduceValueGlob[3],reduceValueGlob[4],reduceValueGlob[5],
    // 	     reduceValueGlob[6], reduceValueGlob[7], reduceValueGlob[8]);
    // printf("[rank %d]---- %f %f %f %f %f %f %f %f %f\n",myRank,reduceValue[0],reduceValue[1],
    // 	   reduceValue[2],reduceValue[3],reduceValue[4],reduceValue[5],
    // 	   reduceValue[6], reduceValue[7], reduceValue[8]);
    /**/
    
    /* Debug:*/
    /*printf("Random forcing normalistation : %f\n",norm);
      printf("Random forcing E_k %f M_m %f M_v %f \n",
      0.5*reduceValueGlob[4]/nbCells,
      SQRT(reduceValueGlob[2]/nbCells),
      SQRT(reduceValueGlob[3]/nbCells) );*/
     /* */

    return norm;

  } // HydroRunBaseMpi::compute_random_forcing_normalization

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  real_t HydroRunBaseMpi::compute_random_forcing_normalization(DeviceArray<real_t>  &U, 
							       real_t               dt)
  {
    
    /*
     * local reduction inside MPI sub-domain
     */

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
    checkCudaErrorMpi("HydroRunBase compute_random_forcing_normalization error",myRank);

    // copy back partial reduction on host
    d_randomForcingNormalization.copyToHost(h_randomForcingNormalization);
    checkCudaErrorMpi("HydroRunBase d_randomForcingNormalization copy to host error",myRank);

    // perform final reduction on host
    real_t* reduceArray = h_randomForcingNormalization.data();

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

    /*
     * Global reduction
     */
    real_t reduceValueGlob[nbRandomForcingReduction];
    communicator->synchronize();
    communicator->allReduce(  reduceValue,       reduceValueGlob,    7,mpi_data_type,MpiComm::SUM);
    communicator->allReduce(&(reduceValue[7]), &(reduceValueGlob[7]),1,mpi_data_type,MpiComm::MIN);
    communicator->allReduce(&(reduceValue[8]), &(reduceValueGlob[8]),1,mpi_data_type,MpiComm::MAX);

    int64_t nbCells = (nx*mx)*(ny*my)*(nz*mz);

    real_t norm;

    if (randomForcingEdot == 0) {
      norm = 0;
    } else {
      norm = ( SQRT( SQR(reduceValueGlob[0]) + 
		     reduceValueGlob[1] * dt * randomForcingEdot * 2 * nbCells ) - 
	       reduceValueGlob[0] ) / reduceValueGlob[1];
    }
    
    /**/
    // if (myRank==0) printf("---- kk %f %f %f %f %f %f %f %f %f\n",reduceValueGlob[0],reduceValueGlob[1],
    // 			  reduceValueGlob[2],reduceValueGlob[3],reduceValueGlob[4],reduceValueGlob[5],
    // 			  reduceValueGlob[6], reduceValueGlob[7], reduceValueGlob[8]);
    // printf("[rank %d]---- %f %f %f %f %f %f %f %f %f\n",myRank,reduceValue[0],reduceValue[1],
    // 	   reduceValue[2],reduceValue[3],reduceValue[4],reduceValue[5],
    // 	   reduceValue[6], reduceValue[7], reduceValue[8]);
    /**/

    /* Debug: */
    /*
      if (myRank==0) printf("Random forcing normalistation : %f\n",norm);
      printf("Random forcing E_k %f M_m %f M_v %f \n",
      0.5*reduceValueGlob[4]/nbCells,
      SQRT(reduceValueGlob[2]/nbCells),
      SQRT(reduceValueGlob[3]/nbCells) );*/
    /* */
    
    return norm;
    
  } // HydroRunBaseMpi::compute_random_forcing_normalization
#endif // __CUDACC__


  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::add_random_forcing(HostArray<real_t>  &U, 
					   real_t             dt,
					   real_t             norm)
  {

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
	  
  } // HydroRunBaseMpi::add_random_forcing

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::add_random_forcing(DeviceArray<real_t>  &U, 
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
    
    checkCudaErrorMpi("in HydroRunBase :: kernel_add_random_forcing_3d",myRank);

  } // HydroRunBaseMpi::add_random_forcing
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update(HostArray<real_t>  &U, 
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
    
  } // HydroRunBaseMpi::compute_hydro_update (2D case, CPU)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update(DeviceArray<real_t>  &U, 
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
    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_2d",myRank);

  } // HydroRunBaseMpi::compute_hydro_update (2D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update(HostArray<real_t>  &U, 
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

  } // HydroRunBaseMpi::compute_hydro_update (3D case)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update(DeviceArray<real_t>  &U, 
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
    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_3d",myRank);

  } // HydroRunBaseMpi::compute_hydro_update (3D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update(HostArray<real_t>  &U, 
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

  } // HydroRunBaseMpi::compute_hydro_update (3D case)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update(DeviceArray<real_t>  &U, 
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
    checkCudaErrorMpi("HydroRunBase :: kernel_hydro_update_3d_zslab",myRank);

  } // HydroRunBaseMpi::compute_hydro_update (3D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update_energy(HostArray<real_t>  &U, 
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
    
  } // HydroRunBaseMpi::compute_hydro_update_energy (2D case, CPU)
  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update_energy(DeviceArray<real_t>  &U, 
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
    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_energy_2d",myRank);

  } // HydroRunBaseMpi::compute_hydro_update_energy (2D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update_energy(HostArray<real_t>  &U, 
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
    
  } // HydroRunBaseMpi::compute_hydro_update_energy (3D case)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update_energy(DeviceArray<real_t>  &U, 
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
    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_energy_3d",myRank);

  } // HydroRunBaseMpi::compute_hydro_update_energy (3D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update_energy(HostArray<real_t>  &U, 
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
    
  } // HydroRunBaseMpi::compute_hydro_update_energy (3D case, z-slab method)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_hydro_update_energy(DeviceArray<real_t>  &U, 
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
    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_energy_3d_zslab",myRank);

  } // HydroRunBaseMpi::compute_hydro_update_energy (3D case, GPU, z-slab method)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_gravity_predictor(HostArray<real_t> &qPrim,
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

  } // HydroRunBaseMpi::compute_gravity_predictor / CPU version

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_gravity_predictor(DeviceArray<real_t> &qPrim,
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
      checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_predictor_2d", myRank);

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
      checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_predictor_3d", myRank);


    } // end TWO_D / THREE_D
    
  } // HydroRunBaseMpi::compute_gravity_predictor / GPU version
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_gravity_predictor(HostArray<real_t> &qPrim,
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
    
  } // HydroRunBaseMpi::compute_gravity_predictor / CPU version / with zSlab

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_gravity_predictor(DeviceArray<real_t> &qPrim,
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
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_predictor_3d_zslab", myRank);

  } // HydroRunBaseMpi::compute_gravity_predictor / GPU version / with zSlab
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_gravity_source_term(HostArray<real_t> &UNew,
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

  } // HydroRunBaseMpi::compute_gravity_source_term / CPU version

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_gravity_source_term(DeviceArray<real_t> &UNew,
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
      checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_source_term_2d", myRank);

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
      checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_source_term_3d", myRank);
    }

  } // HydroRunBaseMpi::compute_gravity_source_term / GPU version
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_gravity_source_term(HostArray<real_t> &UNew,
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

  } // HydroRunBaseMpi::compute_gravity_source_term / CPU version / zslab

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_gravity_source_term(DeviceArray<real_t> &UNew,
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
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_source_term_3d_zslab", myRank);
    
  } // HydroRunBaseMpi::compute_gravity_source_term / GPU version / zslab

#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_ct_update_2d(HostArray<real_t>  &U, 
					     HostArray<real_t>  &emf,
					     real_t dt)
  {
    
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;

    // only update magnetic field
    for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
      for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	
	U(i  ,j  ,IA) += ( emf(i  ,j+1, I_EMFZ) - emf(i,j, I_EMFZ) )*dtdy;
	U(i  ,j  ,IB) -= ( emf(i+1,j  , I_EMFZ) - emf(i,j, I_EMFZ) )*dtdx;
	
      } // end for i
    } // end for j
    
  } // HydroRunBaseMpi::compute_ct_update_2d (2D case, CPU)
  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_ct_update_2d(DeviceArray<real_t>  &U, 
					     DeviceArray<real_t>  &emf,
					     real_t dt)
  {
    
    dim3 dimBlock(MHD_CT_UPDATE_2D_DIMX,
		  MHD_CT_UPDATE_2D_DIMY);
    dim3 dimGrid(blocksFor(isize, MHD_CT_UPDATE_2D_DIMX),
		 blocksFor(jsize, MHD_CT_UPDATE_2D_DIMY));

    kernel_mhd_ct_update_2d<<< dimGrid, dimBlock >>> (U.data(), 
						      emf.data(),
						      U.pitch(),
						      U.dimx(), 
						      U.dimy(),
						      dt/dx, dt/dy, dt);
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_ct_update_2d", myRank);
    
  } // HydroRunBaseMpi::compute_ct_update_2d (2D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_ct_update_3d(HostArray<real_t>  &U, 
					     HostArray<real_t>  &emf,
					     real_t dt)
  {
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;

    // only update magnetic field
    for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	  
	  // update with EMFZ
	  if (k<ksize-ghostWidth) {
	    U(i ,j ,k, IA) += ( emf(i  ,j+1, k, I_EMFZ) - 
				emf(i,  j  , k, I_EMFZ) ) * dtdy;
	    
	    U(i ,j ,k, IB) -= ( emf(i+1,j  , k, I_EMFZ) - 
				emf(i  ,j  , k, I_EMFZ) ) * dtdx;
	    
	  }
	  
	  // update BX
	  U(i ,j ,k, IA) -= ( emf(i,j,k+1, I_EMFY) -
			      emf(i,j,k  , I_EMFY) ) * dtdz;
	  
	  // update BY
	  U(i ,j ,k, IB) += ( emf(i,j,k+1, I_EMFX) -
			      emf(i,j,k  , I_EMFX) ) * dtdz;
	  
	  // update BZ
	  U(i ,j ,k, IC) += ( emf(i+1,j  ,k, I_EMFY) -
			      emf(i  ,j  ,k, I_EMFY) ) * dtdx;
	  U(i ,j ,k, IC) -= ( emf(i  ,j+1,k, I_EMFX) -
			      emf(i  ,j  ,k, I_EMFX) ) * dtdy;
	  
	} // end for i
      } // end for j
    } // end for k
    
  } // HydroRunBaseMpi::compute_ct_update_3d (3D case, CPU)
  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_ct_update_3d(DeviceArray<real_t>  &U, 
					     DeviceArray<real_t>  &emf,
					     real_t dt)
  {

    dim3 dimBlock(MHD_CT_UPDATE_3D_DIMX,
		  MHD_CT_UPDATE_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, MHD_CT_UPDATE_3D_DIMX),
		 blocksFor(jsize, MHD_CT_UPDATE_3D_DIMY));
    
    kernel_mhd_ct_update_3d<<< dimGrid, dimBlock >>> (U.data(), 
						      emf.data(),
						      U.pitch(),
						      U.dimx(), U.dimy(), U.dimz(),
						      dt/dx, dt/dy, dt/dz, dt);
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_ct_update_3d", myRank);
    
  } // HydroRunBaseMpi::compute_ct_update_3d (3D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_ct_update_3d(HostArray<real_t>  &U, 
					     HostArray<real_t>  &emf,
					     real_t              dt,
					     ZslabInfo           zSlabInfo)
  {
    // start and stop index of current slab (ghosts included)
    int& kStart = zSlabInfo.kStart;
    int& kStop  = zSlabInfo.kStop;

    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;

    int kStopUpdate = kStop-ghostWidth;
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1)
      kStopUpdate = ksize-ghostWidth+1;

    // only update magnetic field
    for (int k = kStart+ghostWidth; k < kStopUpdate; k++) {

      // local k index
      int kL = k - kStart;
      	
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	    
	    // update with EMFZ
	    if (k<ksize-ghostWidth) {

	      U(i ,j ,k, IA) += ( emf(i  ,j+1, kL, I_EMFZ) - 
				  emf(i,  j  , kL, I_EMFZ) ) * dtdy;
	      
	      U(i ,j ,k, IB) -= ( emf(i+1,j  , kL, I_EMFZ) - 
				  emf(i  ,j  , kL, I_EMFZ) ) * dtdx;

	    }
	    
	    // update BX
	    U(i ,j ,k, IA) -= ( emf(i,j,kL+1, I_EMFY) -
				emf(i,j,kL  , I_EMFY) ) * dtdz;
	    
	    // update BY
	    U(i ,j ,k, IB) += ( emf(i,j,kL+1, I_EMFX) -
				emf(i,j,kL  , I_EMFX) ) * dtdz;
	    
	    // update BZ
	    U(i ,j ,k, IC) += ( emf(i+1,j  ,kL, I_EMFY) -
				emf(i  ,j  ,kL, I_EMFY) ) * dtdx;
	    U(i ,j ,k, IC) -= ( emf(i  ,j+1,kL, I_EMFX) -
				emf(i  ,j  ,kL, I_EMFX) ) * dtdy;
	    
	  } // end for i
	} // end for j
	
    } // end for k
    
  } // HydroRunBaseMpi::compute_ct_update_3d (3D case, CPU, z-slab)
  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_ct_update_3d(DeviceArray<real_t>  &U, 
					     DeviceArray<real_t>  &emf,
					     real_t dt,
					     ZslabInfo zSlabInfo)
  {

    dim3 dimBlock(MHD_CT_UPDATE_Z_3D_DIMX,
		  MHD_CT_UPDATE_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, MHD_CT_UPDATE_Z_3D_DIMX),
		 blocksFor(jsize, MHD_CT_UPDATE_Z_3D_DIMY));
    
    kernel_mhd_ct_update_3d_zslab
      <<< dimGrid, dimBlock >>> (U.data(), 
				 emf.data(),
				 U.pitch(),
				 U.dimx(), U.dimy(), U.dimz(),
				 dt/dx, dt/dy, dt/dz, dt,
				 zSlabInfo);
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_ct_update_3d_zslab", myRank);
    
  } // HydroRunBaseMpi::compute_ct_update_3d (3D case, GPU, z-slab)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_resistivity_emf_2d(HostArray<real_t> &U,
						   HostArray<real_t> &emf)
  {
    real_t &eta  = _gParams.eta;

    real_t dbxdy = ZERO_F;
    //real_t dbxdz = ZERO_F;
    
    real_t dbydx = ZERO_F;
    //real_t dbydz = ZERO_F;
    
    //real_t dbzdx = ZERO_F;
    //real_t dbzdy = ZERO_F;
    
    //real_t jx    = ZERO_F;
    //real_t jy    = ZERO_F;
    real_t jz    = ZERO_F;
    
    // Compute J=curl(B)
    for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
      for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	
	dbydx = ( U(i,j,IBY) - U(i-1,j  ,IBY) ) / dx;
	//dbzdx = ( U(i,j,IBZ) - U(i-1,j  ,IBZ) ) / dx;
	
	dbxdy = ( U(i,j,IBX) - U(i  ,j-1,IBX) ) / dy;
	//dbzdy = ( U(i,j,IBZ) - U(i  ,j-1,IBZ) ) / dy;
	
	//dbxdz = ZERO_F;
	//dbydz = ZERO_F;
	
	//jx = dbzdy - dbydz;
	//jy = dbxdz - dbzdx;
	jz = dbydx - dbxdy;
	
	// note that multiplication by dt is done in ct
	emf(i,j,I_EMFZ) = -eta*jz;
	/*emf(i,j,I_EMFY) = -eta*jy;
	  emf(i,j,I_EMFX) = -eta*jx;*/
	
      } // end for i
    } // end for j

  } // HydroRunBaseMpi::compute_resistivity_emf (CPU, 2D)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_resistivity_emf_2d(DeviceArray<real_t> &U,
						   DeviceArray<real_t> &emf)
  {
    dim3 dimBlock(RESISTIVITY_2D_DIMX,
		  RESISTIVITY_2D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_2D_DIMX),
		 blocksFor(jsize, RESISTIVITY_2D_DIMY));

    kernel_resistivity_forces_2d<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    emf.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), dx, dy);
    checkCudaErrorMpi("in HydroRunBase :: kernel_resistivity_forces_2d", myRank);

  } // HydroRunBaseMpi::compute_resistivity_emf_2d (GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_resistivity_emf_3d(HostArray<real_t> &U,
						   HostArray<real_t> &emf)
  {
    real_t &eta  = _gParams.eta;

    real_t dbxdy = ZERO_F;
    real_t dbxdz = ZERO_F;
    
    real_t dbydx = ZERO_F;
    real_t dbydz = ZERO_F;
    
    real_t dbzdx = ZERO_F;
    real_t dbzdy = ZERO_F;
    
    real_t jx=ZERO_F;
    real_t jy=ZERO_F;
    real_t jz=ZERO_F;
    
    // Compute J=curl(B)
    for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	  
	  dbydx = ( U(i,j,k,IBY) - U(i-1,j  ,k  ,IBY) ) / dx;
	  dbzdx = ( U(i,j,k,IBZ) - U(i-1,j  ,k  ,IBZ) ) / dx;
	  
	  dbxdy = ( U(i,j,k,IBX) - U(i  ,j-1,k  ,IBX) ) / dy;
	  dbzdy = ( U(i,j,k,IBZ) - U(i  ,j-1,k  ,IBZ) ) / dy;
	  
	  dbxdz = ( U(i,j,k,IBX) - U(i  ,j  ,k-1,IBX) ) / dz;
	  dbydz = ( U(i,j,k,IBY) - U(i  ,j  ,k-1,IBY) ) / dz;
	  
	  jx = dbzdy - dbydz;
	  jy = dbxdz - dbzdx;
	  jz = dbydx - dbxdy;
	  
	  // note that multiplication by dt is done in ct
	  emf(i,j,k,I_EMFX) = -eta*jx;
	  emf(i,j,k,I_EMFY) = -eta*jy;
	  emf(i,j,k,I_EMFZ) = -eta*jz;
	  
	} // end for i
      } // end for j
    } // end for k

  } // HydroRunBaseMpi::compute_resistivity_emf (CPU, 3D)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_resistivity_emf_3d(DeviceArray<real_t> &U,
						   DeviceArray<real_t> &emf)
  {
    dim3 dimBlock(RESISTIVITY_3D_DIMX,
		  RESISTIVITY_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_3D_DIMX),
		 blocksFor(jsize, RESISTIVITY_3D_DIMY));

    kernel_resistivity_forces_3d<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    emf.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), U.dimz(),
		    dx, dy, dz);
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_forces_3d", myRank);

  } // HydroRunBaseMpi::compute_resistivity_emf_3d (GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_resistivity_emf_3d(HostArray<real_t> &U,
						   HostArray<real_t> &emf,
						   ZslabInfo          zSlabInfo)
  {
    real_t &eta  = _gParams.eta;

    real_t dbxdy = ZERO_F;
    real_t dbxdz = ZERO_F;
    
    real_t dbydx = ZERO_F;
    real_t dbydz = ZERO_F;
    
    real_t dbzdx = ZERO_F;
    real_t dbzdy = ZERO_F;
    
    real_t jx=ZERO_F;
    real_t jy=ZERO_F;
    real_t jz=ZERO_F;
    
    // start and stop index of current slab (ghosts included)
    int& kStart = zSlabInfo.kStart;
    int& kStop  = zSlabInfo.kStop;
    
    // Compute J=curl(B)
    for (int k = kStart+ghostWidth; 
	 k     < kStop-ghostWidth+1; 
	 k++) {

      // local index inside slab
      int kL = k - kStart;

      if (k<ksize-ghostWidth+1) {
	
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	    
	    dbydx = ( U(i,j,k,IBY) - U(i-1,j  ,k  ,IBY) ) / dx;
	    dbzdx = ( U(i,j,k,IBZ) - U(i-1,j  ,k  ,IBZ) ) / dx;
	    
	    dbxdy = ( U(i,j,k,IBX) - U(i  ,j-1,k  ,IBX) ) / dy;
	    dbzdy = ( U(i,j,k,IBZ) - U(i  ,j-1,k  ,IBZ) ) / dy;
	    
	    dbxdz = ( U(i,j,k,IBX) - U(i  ,j  ,k-1,IBX) ) / dz;
	    dbydz = ( U(i,j,k,IBY) - U(i  ,j  ,k-1,IBY) ) / dz;
	    
	    jx = dbzdy - dbydz;
	    jy = dbxdz - dbzdx;
	    jz = dbydx - dbxdy;
	    
	    // note that multiplication by dt is done in ct
	    emf(i,j,kL,I_EMFX) = -eta*jx;
	    emf(i,j,kL,I_EMFY) = -eta*jy;
	    emf(i,j,kL,I_EMFZ) = -eta*jz;
	    
	  } // end for i
	} // end for j

      } // end if (k<ksize-ghostWidth+1)

    } // end for k

  } // HydroRunBaseMpi::compute_resistivity_emf (CPU, 3D, z-slab)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_resistivity_emf_3d(DeviceArray<real_t> &U,
						   DeviceArray<real_t> &emf,
						   ZslabInfo            zSlabInfo)
  {
    // take care that the last slab might be truncated
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1) {
      zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
    }

    dim3 dimBlock(RESISTIVITY_Z_3D_DIMX,
		  RESISTIVITY_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_Z_3D_DIMX),
		 blocksFor(jsize, RESISTIVITY_Z_3D_DIMY));

    kernel_resistivity_forces_3d_zslab<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    emf.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), U.dimz(),
		    dx, dy, dz,
		    zSlabInfo);
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_forces_3d_zslab",myRank);

  } // HydroRunBaseMpi::compute_resistivity_emf_3d (GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_resistivity_energy_flux_2d(HostArray<real_t> &U,
							   HostArray<real_t> &fluxX,
							   HostArray<real_t> &fluxY,
							   real_t             dt)
  {

    real_t bx,   by,   bz;
    real_t jx,   jy,   jz;
    real_t /*jxp1, jyp1,*/ jzp1;

    real_t &eta  = _gParams.eta;

    for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
      for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	
	// 1st direction energy flux

	by = ( U(i,j  ,IBY) + U(i-1,j  ,IBY) + 
	       U(i,j+1,IBY) + U(i-1,j+1,IBY) )/4;
	
	bz = ( U(i  ,j,IBZ) + 
	       U(i-1,j,IBZ) )/2;
	  
	jy   = - ( U(i  ,j  ,IBZ) - 
		   U(i-1,j  ,IBZ) )/dx;
	
	jz   = 
	  ( U(i,j  ,IBY) - U(i-1,j  ,IBY) )/dx -
	  ( U(i,j  ,IBX) - U(i  ,j-1,IBX) )/dy;
	jzp1 = 
	  ( U(i,j+1,IBY) - U(i-1,j+1,IBY) )/dx -
	  ( U(i,j+1,IBX) - U(i  ,j  ,IBX) )/dy;
	jz   = (jz+jzp1)/2;
	
	fluxX(i,j,ID) = ZERO_F;
	fluxX(i,j,IP) = - eta*(jy*bz-jz*by)*dt/dx;
	fluxX(i,j,IU) = ZERO_F;
	fluxX(i,j,IV) = ZERO_F;

	// 2nd direction energy flux
	
	bx = ( U(i  ,j,IBX) + U(i  ,j-1,IBX) +
	       U(i+1,j,IBX) + U(i+1,j-1,IBX) )/4;

	bz = ( U(i  ,j,IBZ) + U(i  ,j-1,IBZ) )/2;
	
	jx = ( U(i  ,j  ,IBZ) -
	       U(i  ,j-1,IBZ) )/dy;
	
	jz   = 
	  ( U(i  ,j,IBY) - U(i-1,j  ,IBY) )/dx -
	  ( U(i  ,j,IBX) - U(i  ,j-1,IBX) )/dy;

	jzp1 = 
	  ( U(i+1,j,IBY) - U(i  ,j  ,IBY) )/dx -
	  ( U(i+1,j,IBX) - U(i+1,j-1,IBX) )/dy;

	jz = (jz+jzp1)/2;
          
	fluxY(i,j,ID) = ZERO_F;
	fluxY(i,j,IP) = - eta*(jz*bx-jx*bz)*dt/dy;
	fluxY(i,j,IU) = ZERO_F;
	fluxY(i,j,IV) = ZERO_F;	
	  	  
	} // end for i
      } // end for j

  } // HydroRunBaseMpi::compute_resistivity_energy_flux_2d
  
  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void HydroRunBaseMpi::compute_resistivity_energy_flux_2d(DeviceArray<real_t> &U,
							   DeviceArray<real_t> &fluxX,
							   DeviceArray<real_t> &fluxY,
							   real_t               dt)
  {

    dim3 dimBlock(RESISTIVITY_ENERGY_2D_DIMX,
		  RESISTIVITY_ENERGY_2D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_ENERGY_2D_DIMX),
		 blocksFor(jsize, RESISTIVITY_ENERGY_2D_DIMY));

    kernel_resistivity_energy_flux_2d<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    fluxX.data(),
		    fluxY.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), dx, dy, dt);
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_energy_flux_2d", myRank);

  } // HydroRunBaseMpi::compute_resistivity_energy_flux_2d (GPU)
#endif // __CUDACC

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_resistivity_energy_flux_3d(HostArray<real_t> &U,
							   HostArray<real_t> &fluxX,
							   HostArray<real_t> &fluxY,
							   HostArray<real_t> &fluxZ,
							   real_t             dt)
  {

    real_t bx,   by,   bz;
    real_t jx,   jy,   jz;
    real_t jxp1, jyp1, jzp1;

    real_t &eta  = _gParams.eta;

    for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
           
	  // 1st direction energy flux

	  by = ( U(i,j  ,k,IBY) + U(i-1,j  ,k,IBY) + 
		 U(i,j+1,k,IBY) + U(i-1,j+1,k,IBY) )/4;
	  bz = ( U(i,j,k  ,IBZ) + U(i-1,j,k  ,IBZ) + 
		 U(i,j,k+1,IBZ) + U(i-1,j,k+1,IBZ) )/4;
	  
	  jy   = 
	    ( U(i,j,k  ,IBX) - U(i  ,j  ,k-1,IBX) )/dz -
	    ( U(i,j,k  ,IBZ) - U(i-1,j  ,k  ,IBZ) )/dx;
	  jyp1 = 
	    ( U(i,j,k+1,IBX) - U(i  ,j  ,k  ,IBX) )/dz -
	    ( U(i,j,k+1,IBZ) - U(i-1,j  ,k+1,IBZ) )/dx;
	  jy   = (jy+jyp1)/2;
	  
	  jz   = 
	    ( U(i,j  ,k,IBY) - U(i-1,j  ,k  ,IBY) )/dx -
	    ( U(i,j  ,k,IBX) - U(i  ,j-1,k  ,IBX) )/dy;
	  jzp1 = 
	    ( U(i,j+1,k,IBY) - U(i-1,j+1,k  ,IBY) )/dx -
	    ( U(i,j+1,k,IBX) - U(i  ,j  ,k  ,IBX) )/dy;
	  jz   = (jz+jzp1)/2;
	  
	  fluxX(i,j,k,ID) = ZERO_F;
	  fluxX(i,j,k,IP) = - eta*(jy*bz-jz*by)*dt/dx;
	  fluxX(i,j,k,IU) = ZERO_F;
	  fluxX(i,j,k,IV) = ZERO_F;
	  fluxX(i,j,k,IW) = ZERO_F;

	  // 2nd direction energy flux
	  
	  bx = ( U(i  ,j,k,IBX) + U(i  ,j-1,k,IBX) +
		 U(i+1,j,k,IBX) + U(i+1,j-1,k,IBX) )/4;
	  
	  
	  bz = ( U(i,j,k  ,IBZ) + U(i,j-1,k  ,IBZ) + 
		 U(i,j,k+1,IBZ) + U(i,j-1,k+1,IBZ) )/4;
	  
	  jx   = 
	    ( U(i,j,k  ,IBZ) - U(i  ,j-1,k  ,IBZ) )/dy -
	    ( U(i,j,k  ,IBY) - U(i  ,j  ,k-1,IBY) )/dz;
	  jxp1 = 
	    ( U(i,j,k+1,IBZ) - U(i  ,j-1,k+1,IBZ) )/dy -
	    ( U(i,j,k+1,IBY) - U(i  ,j  ,k  ,IBY) )/dz;
	  jx = (jx+jxp1)/2;
	    
	  
	  jz   = 
	    ( U(i  ,j,k,IBY) - U(i-1,j  ,k  ,IBY) )/dx -
	    ( U(i  ,j,k,IBX) - U(i  ,j-1,k  ,IBX) )/dy;
	  jzp1 = 
	    ( U(i+1,j,k,IBY) - U(i  ,j  ,k  ,IBY) )/dx -
	    ( U(i+1,j,k,IBX) - U(i+1,j-1,k  ,IBX) )/dy;
	  jz = (jz+jzp1)/2;
	  
	  fluxY(i,j,k,ID) = ZERO_F;
	  fluxY(i,j,k,IP) = - eta*(jz*bx-jx*bz)*dt/dy;
	  fluxY(i,j,k,IU) = ZERO_F;
	  fluxY(i,j,k,IV) = ZERO_F;
	  fluxY(i,j,k,IW) = ZERO_F;
	  
	  // 3rd direction energy flux
	  bx = ( U(i  ,j,k,IBX) + U(i  ,j,k-1,IBX) + 
		 U(i+1,j,k,IBX) + U(i+1,j,k-1,IBX) )/4;
	  by = ( U(i,j  ,k,IBY) + U(i,j  ,k-1,IBY) + 
		 U(i,j+1,k,IBY) + U(i,j+1,k-1,IBY) )/4;
	  
	  jx   = 
	    ( U(i,j  ,k,IBZ) - U(i  ,j-1,k  ,IBZ) )/dy -
	    ( U(i,j  ,k,IBY) - U(i  ,j  ,k-1,IBY) )/dz;
	  jxp1 = 
	    ( U(i,j+1,k,IBZ) - U(i  ,j  ,k  ,IBZ) )/dy -
	    ( U(i,j+1,k,IBY) - U(i  ,j+1,k-1,IBY) )/dz;
	  jx   = (jx+jxp1)/2;

	  jy   = 
	    ( U(i  ,j,k,IBX) - U(i  ,j  ,k-1,IBX) )/dz -
	    ( U(i  ,j,k,IBZ) - U(i-1,j  ,k  ,IBZ) )/dx;
	  jyp1 = 
	    ( U(i+1,j,k,IBX) - U(i+1,j  ,k-1,IBX) )/dz -
	    ( U(i+1,j,k,IBZ) - U(i  ,j  ,k  ,IBZ) )/dx;
	  jy   = (jy+jyp1)/2;
          
	  fluxZ(i,j,k,ID) = ZERO_F;
	  fluxZ(i,j,k,IP) = - eta*(jx*by-jy*bx)*dt/dz;
	  fluxZ(i,j,k,IU) = ZERO_F;
	  fluxZ(i,j,k,IV) = ZERO_F;
	  fluxZ(i,j,k,IW) = ZERO_F;
	  
	} // end for i
      } // end for j
    } // end for k

  } // HydroRunBaseMpi::compute_resistivity_energy_flux_3d
  
  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void HydroRunBaseMpi::compute_resistivity_energy_flux_3d(DeviceArray<real_t> &U,
							   DeviceArray<real_t> &fluxX,
							   DeviceArray<real_t> &fluxY,
							   DeviceArray<real_t> &fluxZ,
							   real_t               dt)
  {

    dim3 dimBlock(RESISTIVITY_ENERGY_3D_DIMX,
		  RESISTIVITY_ENERGY_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_ENERGY_3D_DIMX),
		 blocksFor(jsize, RESISTIVITY_ENERGY_3D_DIMY));

    kernel_resistivity_energy_flux_3d<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    fluxX.data(),
		    fluxY.data(),
		    fluxZ.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), U.dimz(),
		    dx, dy, dz, dt);
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_energy_flux_3d", myRank);

  } // HydroRunBaseMpi::compute_resistivity_energy_flux_3d
#endif // __CUDACC

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_resistivity_energy_flux_3d(HostArray<real_t> &U,
							   HostArray<real_t> &fluxX,
							   HostArray<real_t> &fluxY,
							   HostArray<real_t> &fluxZ,
							   real_t             dt,
							   ZslabInfo          zSlabInfo)
  {

    real_t bx,   by,   bz;
    real_t jx,   jy,   jz;
    real_t jxp1, jyp1, jzp1;
    
    real_t &eta  = _gParams.eta;
    
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
	    
	    // 1st direction energy flux
	    
	    by = ( U(i,j  ,k,IBY) + U(i-1,j  ,k,IBY) + 
		   U(i,j+1,k,IBY) + U(i-1,j+1,k,IBY) )/4;
	    bz = ( U(i,j,k  ,IBZ) + U(i-1,j,k  ,IBZ) + 
		   U(i,j,k+1,IBZ) + U(i-1,j,k+1,IBZ) )/4;
	    
	    jy   = 
	      ( U(i,j,k  ,IBX) - U(i  ,j  ,k-1,IBX) )/dz -
	      ( U(i,j,k  ,IBZ) - U(i-1,j  ,k  ,IBZ) )/dx;
	    jyp1 = 
	      ( U(i,j,k+1,IBX) - U(i  ,j  ,k  ,IBX) )/dz -
	      ( U(i,j,k+1,IBZ) - U(i-1,j  ,k+1,IBZ) )/dx;
	    jy   = (jy+jyp1)/2;
	    
	    jz   = 
	      ( U(i,j  ,k,IBY) - U(i-1,j  ,k  ,IBY) )/dx -
	      ( U(i,j  ,k,IBX) - U(i  ,j-1,k  ,IBX) )/dy;
	    jzp1 = 
	      ( U(i,j+1,k,IBY) - U(i-1,j+1,k  ,IBY) )/dx -
	      ( U(i,j+1,k,IBX) - U(i  ,j  ,k  ,IBX) )/dy;
	    jz   = (jz+jzp1)/2;
	    
	    fluxX(i,j,kL,ID) = ZERO_F;
	    fluxX(i,j,kL,IP) = - eta*(jy*bz-jz*by)*dt/dx;
	    fluxX(i,j,kL,IU) = ZERO_F;
	    fluxX(i,j,kL,IV) = ZERO_F;
	    fluxX(i,j,kL,IW) = ZERO_F;
	    
	    // 2nd direction energy flux
	    
	    bx = ( U(i  ,j,k,IBX) + U(i  ,j-1,k,IBX) +
		   U(i+1,j,k,IBX) + U(i+1,j-1,k,IBX) )/4;
	    
	    
	    bz = ( U(i,j,k  ,IBZ) + U(i,j-1,k  ,IBZ) + 
		   U(i,j,k+1,IBZ) + U(i,j-1,k+1,IBZ) )/4;
	    
	    jx   = 
	      ( U(i,j,k  ,IBZ) - U(i  ,j-1,k  ,IBZ) )/dy -
	      ( U(i,j,k  ,IBY) - U(i  ,j  ,k-1,IBY) )/dz;
	    jxp1 = 
	      ( U(i,j,k+1,IBZ) - U(i  ,j-1,k+1,IBZ) )/dy -
	      ( U(i,j,k+1,IBY) - U(i  ,j  ,k  ,IBY) )/dz;
	    jx = (jx+jxp1)/2;
	    
	    
	    jz   = 
	      ( U(i  ,j,k,IBY) - U(i-1,j  ,k  ,IBY) )/dx -
	      ( U(i  ,j,k,IBX) - U(i  ,j-1,k  ,IBX) )/dy;
	    jzp1 = 
	      ( U(i+1,j,k,IBY) - U(i  ,j  ,k  ,IBY) )/dx -
	      ( U(i+1,j,k,IBX) - U(i+1,j-1,k  ,IBX) )/dy;
	    jz = (jz+jzp1)/2;
	    
	    fluxY(i,j,kL,ID) = ZERO_F;
	    fluxY(i,j,kL,IP) = - eta*(jz*bx-jx*bz)*dt/dy;
	    fluxY(i,j,kL,IU) = ZERO_F;
	    fluxY(i,j,kL,IV) = ZERO_F;
	    fluxY(i,j,kL,IW) = ZERO_F;
	    
	    // 3rd direction energy flux
	    bx = ( U(i  ,j,k,IBX) + U(i  ,j,k-1,IBX) + 
		   U(i+1,j,k,IBX) + U(i+1,j,k-1,IBX) )/4;
	    by = ( U(i,j  ,k,IBY) + U(i,j  ,k-1,IBY) + 
		   U(i,j+1,k,IBY) + U(i,j+1,k-1,IBY) )/4;
	    
	    jx   = 
	      ( U(i,j  ,k,IBZ) - U(i  ,j-1,k  ,IBZ) )/dy -
	      ( U(i,j  ,k,IBY) - U(i  ,j  ,k-1,IBY) )/dz;
	    jxp1 = 
	      ( U(i,j+1,k,IBZ) - U(i  ,j  ,k  ,IBZ) )/dy -
	      ( U(i,j+1,k,IBY) - U(i  ,j+1,k-1,IBY) )/dz;
	    jx   = (jx+jxp1)/2;
	    
	    jy   = 
	      ( U(i  ,j,k,IBX) - U(i  ,j  ,k-1,IBX) )/dz -
	      ( U(i  ,j,k,IBZ) - U(i-1,j  ,k  ,IBZ) )/dx;
	    jyp1 = 
	      ( U(i+1,j,k,IBX) - U(i+1,j  ,k-1,IBX) )/dz -
	      ( U(i+1,j,k,IBZ) - U(i  ,j  ,k  ,IBZ) )/dx;
	    jy   = (jy+jyp1)/2;
	    
	    fluxZ(i,j,kL,ID) = ZERO_F;
	    fluxZ(i,j,kL,IP) = - eta*(jx*by-jy*bx)*dt/dz;
	    fluxZ(i,j,kL,IU) = ZERO_F;
	    fluxZ(i,j,kL,IV) = ZERO_F;
	    fluxZ(i,j,kL,IW) = ZERO_F;
	    
	  } // end for i
	} // end for j

      } // end if (k<ksize-ghostWidth+1)

    } // end for k

  } // HydroRunBaseMpi::compute_resistivity_energy_flux_3d (z-slab)
  
  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void HydroRunBaseMpi::compute_resistivity_energy_flux_3d(DeviceArray<real_t> &U,
							   DeviceArray<real_t> &fluxX,
							   DeviceArray<real_t> &fluxY,
							   DeviceArray<real_t> &fluxZ,
							   real_t               dt,
							   ZslabInfo            zSlabInfo)
  {

    // take care that the last slab might be truncated
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1) {
      zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
    }

    dim3 dimBlock(RESISTIVITY_ENERGY_Z_3D_DIMX,
		  RESISTIVITY_ENERGY_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_ENERGY_Z_3D_DIMX),
		 blocksFor(jsize, RESISTIVITY_ENERGY_Z_3D_DIMY));

    kernel_resistivity_energy_flux_3d_zslab<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    fluxX.data(),
		    fluxY.data(),
		    fluxZ.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), U.dimz(),
		    dx, dy, dz, dt,
		    zSlabInfo);
    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_energy_flux_3d_zslab", myRank);

  } // HydroRunBaseMpi::compute_resistivity_energy_flux_3d (z-slab)
#endif // __CUDACC

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::compute_divB(HostArray<real_t>& h_conserv, 
				     HostArray<real_t>& h_divB) 
  {
    /*
     * compute magnetic field divergence
     */

    if (dimType == TWO_D) {
      
      for (int j=0; j<jsize-1; j++) {

	for (int i=0; i<isize-1; i++) {
	
	  h_divB(i,j,0) = 
	    (h_conserv(i+1,j  ,IA)-h_conserv(i,j,IA))/dx +
	    (h_conserv(i  ,j+1,IB)-h_conserv(i,j,IB))/dy;

	} // end for i

      } // end for j
      
    } else { // THREE_D

      for (int k=0; k<ksize-1; k++) {
	
	for (int j=0; j<jsize-1; j++) {
	  
	  for (int i=0; i<isize-1; i++) {
	    
	    h_divB(i,j,k,0) = 
	      (h_conserv(i+1,j  ,k  ,IA)-h_conserv(i,j,k,IA))/dx +
	      (h_conserv(i  ,j+1,k  ,IB)-h_conserv(i,j,k,IB))/dy +
	      (h_conserv(i  ,j  ,k+1,IC)-h_conserv(i,j,k,IC))/dz;

	  } // end for i
	  
	} // end for j
	
      } // end for k

    } // end THREE_D

  } // HydroRunBaseMpi::compute_divB

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void HydroRunBaseMpi::make_boundaries(DeviceArray<real_t> &U, int idim, bool doExternalBoundaries)
  {
    
    bool doXMin, doXMax;
    bool doYMin, doYMax;
    bool doZMin, doZMax;

    // default behavior is do external boundaries
    doXMin = doXMax = doYMin = doYMax = doZMin = doZMax = true;

    // here is the case used for shearing box border condition
    // internal borders are done but not XMIN and XMAX
    if (doExternalBoundaries == false and idim == XDIR) {
      if (myMpiPos[0] == 0     ) doXMin = false;
      if (myMpiPos[0] == (mx-1)) doXMax = false;
    }

    if (doExternalBoundaries == false and idim == YDIR) {
      if (myMpiPos[1] == 0     ) doYMin = false;
      if (myMpiPos[1] == (my-1)) doYMax = false;
    }

    if (doExternalBoundaries == false and idim == ZDIR) {
      if (myMpiPos[2] == 0     ) doZMin = false;
      if (myMpiPos[2] == (mz-1)) doZMax = false;
    }

    
    if (dimType == TWO_D) {
      
      if(idim == XDIR) // horizontal boundaries
	{
	  dim3 blockCount(blocksFor(jsize, MK_BOUND_BLOCK_SIZE), 1, 1);
	  copy_boundaries<XDIR>(U);
	  transfert_boundaries<XDIR>();
	  if (doXMin) 
	    make_boundary<XMIN>(U, borderBufRecv_xmin, neighborsBC[X_MIN], blockCount);
	  if (doXMax)
	    make_boundary<XMAX>(U, borderBufRecv_xmax, neighborsBC[X_MAX], blockCount);
	}
      else // vertical boundaries
	{
	  dim3 blockCount(blocksFor(isize, MK_BOUND_BLOCK_SIZE),1, 1);
	  copy_boundaries<YDIR>(U);
	  transfert_boundaries<YDIR>();
	  if (doYMin)
	    make_boundary<YMIN>(U, borderBufRecv_ymin, neighborsBC[Y_MIN], blockCount);
	  if (doYMax)
	    make_boundary<YMAX>(U, borderBufRecv_ymax, neighborsBC[Y_MAX], blockCount);
	  if (enableJet)
	    make_jet(U);
	}
      
    } else { // THREE_D
      
      if(idim == XDIR) // X-boundaries
	{
	  dim3 blockCount( blocksFor(jsize, MK_BOUND_BLOCK_SIZE_3D),
			   blocksFor(ksize, MK_BOUND_BLOCK_SIZE_3D), 
			   1);
	  copy_boundaries<XDIR>(U);
	  transfert_boundaries<XDIR>();
	  if (doXMin) 
	    make_boundary<XMIN>(U, borderBufRecv_xmin, neighborsBC[X_MIN], blockCount);
	  if (doXMax) 
	    make_boundary<XMAX>(U, borderBufRecv_xmax, neighborsBC[X_MAX], blockCount);
	}
      else if (idim == YDIR) // Y-boundaries
	{
	  dim3 blockCount( blocksFor(isize, MK_BOUND_BLOCK_SIZE_3D),
			   blocksFor(ksize, MK_BOUND_BLOCK_SIZE_3D), 
			   1);
	  copy_boundaries<YDIR>(U);
	  transfert_boundaries<YDIR>();
	  if (doYMin) 
	    make_boundary<YMIN>(U, borderBufRecv_ymin, neighborsBC[Y_MIN], blockCount);
	  if (doYMax) 
	    make_boundary<YMAX>(U, borderBufRecv_ymax, neighborsBC[Y_MAX], blockCount);
	}
      else // Z-boundaries
	{
	  dim3 blockCount( blocksFor(isize, MK_BOUND_BLOCK_SIZE_3D),
			   blocksFor(jsize, MK_BOUND_BLOCK_SIZE_3D), 
			   1);
	  copy_boundaries<ZDIR>(U);
	  transfert_boundaries<ZDIR>();
	  if (doZMin) 
	    make_boundary<ZMIN>(U, borderBufRecv_zmin, neighborsBC[Z_MIN], blockCount);
	  if (doZMax) 
	    make_boundary<ZMAX>(U, borderBufRecv_zmax, neighborsBC[Z_MAX], blockCount);
	  if (enableJet)
	    make_jet(U);
	}
    } // end THREE_D

  } // HydroRunBaseMpi::make_boundaries
#else // CPU version
  void HydroRunBaseMpi::make_boundaries(HostArray<real_t> &U, int idim, bool doExternalBoundaries)
  {

    bool doXMin, doXMax;
    bool doYMin, doYMax;
    bool doZMin, doZMax;

    // default behavior is do external boundaries
    doXMin = doXMax = doYMin = doYMax = doZMin = doZMax = true;

    // here is the case used for shearing box border condition
    // internal borders are done but not XMIN and XMAX
    if (doExternalBoundaries == false and idim == XDIR) {
      if (myMpiPos[0] == 0     ) doXMin = false;
      if (myMpiPos[0] == (mx-1)) doXMax = false;
    }

    if (doExternalBoundaries == false and idim == YDIR) {
      if (myMpiPos[1] == 0     ) doYMin = false;
      if (myMpiPos[1] == (my-1)) doYMax = false;
    }

    if (doExternalBoundaries == false and idim == ZDIR) {
      if (myMpiPos[2] == 0     ) doZMin = false;
      if (myMpiPos[2] == (mz-1)) doZMax = false;
    }
      
    if (dimType == TWO_D) {
	
      if(idim == XDIR) // horizontal boundaries
	{
	  copy_boundaries<XDIR>(U);
	  transfert_boundaries<XDIR>();
	  if (doXMin) 
	    make_boundary<XMIN>(U, borderBufRecv_xmin, neighborsBC[X_MIN], 0);
	  if (doXMax) 
	    make_boundary<XMAX>(U, borderBufRecv_xmax, neighborsBC[X_MAX], 0);
	}
      else // vertical boundaries
	{
	  copy_boundaries<YDIR>(U);
	  transfert_boundaries<YDIR>();
	  if (doYMin) 
	    make_boundary<YMIN>(U, borderBufRecv_ymin, neighborsBC[Y_MIN], 0);
	  if (doYMax) 
	    make_boundary<YMAX>(U, borderBufRecv_ymax, neighborsBC[Y_MAX], 0);
	  if (enableJet)
	    make_jet(U);
	}
	
    } else { // THREE_D
	
      if(idim == XDIR) // X-boundaries
	{
	  copy_boundaries<XDIR>(U);
	  transfert_boundaries<XDIR>();
	  if (doXMin) 
	    make_boundary<XMIN>(U, borderBufRecv_xmin, neighborsBC[X_MIN], 0);
	  if (doXMax) 
	    make_boundary<XMAX>(U, borderBufRecv_xmax, neighborsBC[X_MAX], 0);
	}
      else if (idim == YDIR) // Y-boundaries
	{
	  copy_boundaries<YDIR>(U);
	  transfert_boundaries<YDIR>();
	  if (doYMin) 
	    make_boundary<YMIN>(U, borderBufRecv_ymin, neighborsBC[Y_MIN], 0);
	  if (doYMax) 
	    make_boundary<YMAX>(U, borderBufRecv_ymax, neighborsBC[Y_MAX], 0);
	}
      else // Z-boundaries
	{
	  copy_boundaries<ZDIR>(U);
	  transfert_boundaries<ZDIR>();
	  if (doZMin) 
	    make_boundary<ZMIN>(U, borderBufRecv_zmin, neighborsBC[Z_MIN], 0);
	  if (doZMax) 
	    make_boundary<ZMAX>(U, borderBufRecv_zmax, neighborsBC[Z_MAX], 0);
	  if (enableJet)
	    make_jet(U);
	}
    } // end THREE_D
    
  } // HydroRunBaseMpi::make_boundaries
#endif // __CUDACC__

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void HydroRunBaseMpi::make_all_boundaries(DeviceArray<real_t> &U)
  {
    
    make_boundaries(U,XDIR); communicator->synchronize();
    make_boundaries(U,YDIR); communicator->synchronize();
    if (dimType == THREE_D) {
      make_boundaries(U,ZDIR); communicator->synchronize();
    }

  } // HydroRunBaseMpi::make_all_boundaries
#else // CPU version
  void HydroRunBaseMpi::make_all_boundaries(HostArray<real_t> &U)
  {

    make_boundaries(U,XDIR); communicator->synchronize();
    make_boundaries(U,YDIR); communicator->synchronize();
    if (dimType == THREE_D) {
      make_boundaries(U,ZDIR); communicator->synchronize();
    }

  } // HydroRunBaseMpi::make_all_boundaries
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  /**
   * main routine to start simulation.
   */
  void HydroRunBaseMpi::start() {
    
    //std::cout << "Starting time integration" << std::endl;
    
  } // HydroRunBaseMpi::start

  // =======================================================
  // =======================================================
  /**
   * implemented in derived class.
   */
  void HydroRunBaseMpi::oneStepIntegration(int& nStep, real_t& t, real_t& dt) {

    (void) nStep;
    (void) t;
    (void) dt;

  } // HydroRunBaseMpi::oneStepIntegration

  // =======================================================
  // =======================================================
  template<int direction>
  void HydroRunBaseMpi::transfert_boundaries()
  {
    TIMER_START(timerBoundariesMpi);

    // OLD MPI comm version
    
    // // do MPI communication
    // MPI_Request reqs[4]; // 2 send + 2 receive
    // MPI_Status stats[4]; // 2 send + 2 receive
    
    // // two borders to send, two borders to receive
    // if (direction == XDIR) {
      
    //   reqs[0] = communicator->Isend(borderBufSend_xmin.data(), 
    // 				    borderBufSend_xmin.size(), 
    // 				    data_type, 
    // 				    neighborsRank[X_MIN], X_MIN);
    //   reqs[1] = communicator->Isend(borderBufSend_xmax.data(), 
    // 				    borderBufSend_xmax.size(), 
    // 				    data_type, 
    // 				    neighborsRank[X_MAX], X_MAX);
    //   reqs[2] = communicator->Irecv(borderBufRecv_xmin.data(), 
    // 				    borderBufRecv_xmin.size(), 
    // 				    data_type, 
    // 				    neighborsRank[X_MIN], X_MAX);
    //   reqs[3] = communicator->Irecv(borderBufRecv_xmax.data(), 
    // 				    borderBufRecv_xmax.size(), 
    // 				    data_type, 
    // 				    neighborsRank[X_MAX], X_MIN);
      
    // } else if (direction == YDIR) { // Y-direction
      
    //   reqs[0] = communicator->Isend(borderBufSend_ymin.data(), 
    // 				    borderBufSend_ymin.size(), 
    // 				    data_type, 
    // 				    neighborsRank[Y_MIN], Y_MIN);
    //   reqs[1] = communicator->Isend(borderBufSend_ymax.data(), 
    // 				    borderBufSend_ymax.size(), 
    // 				    data_type, 
    // 				    neighborsRank[Y_MAX], Y_MAX);
    //   reqs[2] = communicator->Irecv(borderBufRecv_ymin.data(), 
    // 				    borderBufRecv_ymin.size(), 
    // 				    data_type, 
    // 				    neighborsRank[Y_MIN], Y_MAX);
    //   reqs[3] = communicator->Irecv(borderBufRecv_ymax.data(), 
    // 				    borderBufRecv_ymax.size(), 
    // 				    data_type, 
    // 				    neighborsRank[Y_MAX], Y_MIN);
      
    // } else { // Z-direction
      
    //   reqs[0] = communicator->Isend(borderBufSend_zmin.data(), 
    // 				    borderBufSend_zmin.size(), 
    // 				    data_type, 
    // 				    neighborsRank[Z_MIN], Z_MIN);
    //   reqs[1] = communicator->Isend(borderBufSend_zmax.data(), 
    // 				    borderBufSend_zmax.size(), 
    // 				    data_type, 
    // 				    neighborsRank[Z_MAX], Z_MAX);
    //   reqs[2] = communicator->Irecv(borderBufRecv_zmin.data(), 
    // 				    borderBufRecv_zmin.size(), 
    // 				    data_type, 
    // 				    neighborsRank[Z_MIN], Z_MAX);
    //   reqs[3] = communicator->Irecv(borderBufRecv_zmax.data(), 
    // 				    borderBufRecv_zmax.size(), 
    // 				    data_type, 
    // 				    neighborsRank[Z_MAX], Z_MIN);
    // }
    
    // // wait for all MPI comm to finish
    // MPI_Waitall(4, reqs, stats);
    
    // END OLD MPI comm version

    /*
     * use MPI_Sendrecv instead
     */

    // two borders to send, two borders to receive
    if (direction == XDIR) {
      
      communicator->sendrecv(borderBufSend_xmin.data(),
			     borderBufSend_xmin.size(),
			     data_type, neighborsRank[X_MIN], 111,
			     borderBufRecv_xmax.data(),
			     borderBufRecv_xmax.size(),
			     data_type, neighborsRank[X_MAX], 111);

      communicator->sendrecv(borderBufSend_xmax.data(),
			     borderBufSend_xmax.size(),
			     data_type, neighborsRank[X_MAX], 111,
			     borderBufRecv_xmin.data(),
			     borderBufRecv_xmin.size(),
			     data_type, neighborsRank[X_MIN], 111);

    } else if (direction == YDIR) {

      communicator->sendrecv(borderBufSend_ymin.data(),
			     borderBufSend_ymin.size(),
			     data_type, neighborsRank[Y_MIN], 211,
			     borderBufRecv_ymax.data(),
			     borderBufRecv_ymax.size(),
			     data_type, neighborsRank[Y_MAX], 211);

      communicator->sendrecv(borderBufSend_ymax.data(),
			     borderBufSend_ymax.size(),
			     data_type, neighborsRank[Y_MAX], 211,
			     borderBufRecv_ymin.data(),
			     borderBufRecv_ymin.size(),
			     data_type, neighborsRank[Y_MIN], 211);

    } else { // Z direction

      communicator->sendrecv(borderBufSend_zmin.data(),
			     borderBufSend_zmin.size(),
			     data_type, neighborsRank[Z_MIN], 311,
			     borderBufRecv_zmax.data(),
			     borderBufRecv_zmax.size(),
			     data_type, neighborsRank[Z_MAX], 311);

      communicator->sendrecv(borderBufSend_zmax.data(),
			     borderBufSend_zmax.size(),
			     data_type, neighborsRank[Z_MAX], 311,
			     borderBufRecv_zmin.data(),
			     borderBufRecv_zmin.size(),
			     data_type, neighborsRank[Z_MIN], 311);

    }

    TIMER_STOP(timerBoundariesMpi);

  } // HydroRunBaseMpi::transfert_boundaries

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  template<int direction>
  void HydroRunBaseMpi::copy_boundaries(DeviceArray<real_t> &U)
  {
    TIMER_START(timerBoundariesCpuGpu);

    /*
     * prepare buffer border to send (grab the right border that is
     * needed from GPU memory array U)
     *
     * note that for hydro simulation, ghostWidth is to 2
     * note that for MHD   simulation, ghostWidth is to 3
     */

    if (mhdEnabled) {

      if (dimType == TWO_D) {
	if (direction == XDIR) {
	  copyDeviceArrayToBorderBufSend<XMIN, TWO_D, 3>(borderBufSend_xmin, borderBuffer_device_xdir, U);
	  copyDeviceArrayToBorderBufSend<XMAX, TWO_D, 3>(borderBufSend_xmax, borderBuffer_device_xdir, U);
	} else {
	  copyDeviceArrayToBorderBufSend<YMIN, TWO_D, 3>(borderBufSend_ymin, borderBuffer_device_ydir, U);
	  copyDeviceArrayToBorderBufSend<YMAX, TWO_D, 3>(borderBufSend_ymax, borderBuffer_device_ydir, U);	
	}
      } else { // THREE_D
	if (direction == XDIR) {
	  copyDeviceArrayToBorderBufSend<XMIN, THREE_D, 3>(borderBufSend_xmin, borderBuffer_device_xdir, U);
	  copyDeviceArrayToBorderBufSend<XMAX, THREE_D, 3>(borderBufSend_xmax, borderBuffer_device_xdir, U);
	} else if (direction == YDIR) {
	  copyDeviceArrayToBorderBufSend<YMIN, THREE_D, 3>(borderBufSend_ymin, borderBuffer_device_ydir, U);
	  copyDeviceArrayToBorderBufSend<YMAX, THREE_D, 3>(borderBufSend_ymax, borderBuffer_device_ydir, U);	
	} else { // Z direction
	  copyDeviceArrayToBorderBufSend<ZMIN, THREE_D, 3>(borderBufSend_zmin, borderBuffer_device_zdir, U);
	  copyDeviceArrayToBorderBufSend<ZMAX, THREE_D, 3>(borderBufSend_zmax, borderBuffer_device_zdir, U);
	}
      } // end THREE_D

    } else { // hydro

      if (dimType == TWO_D) {
	if (direction == XDIR) {
	  copyDeviceArrayToBorderBufSend<XMIN, TWO_D, 2>(borderBufSend_xmin, borderBuffer_device_xdir, U);
	  copyDeviceArrayToBorderBufSend<XMAX, TWO_D, 2>(borderBufSend_xmax, borderBuffer_device_xdir, U);
	} else {
	  copyDeviceArrayToBorderBufSend<YMIN, TWO_D, 2>(borderBufSend_ymin, borderBuffer_device_ydir, U);
	  copyDeviceArrayToBorderBufSend<YMAX, TWO_D, 2>(borderBufSend_ymax, borderBuffer_device_ydir, U);	
	}
      } else { // THREE_D
	if (direction == XDIR) {
	  copyDeviceArrayToBorderBufSend<XMIN, THREE_D, 2>(borderBufSend_xmin, borderBuffer_device_xdir, U);
	  copyDeviceArrayToBorderBufSend<XMAX, THREE_D, 2>(borderBufSend_xmax, borderBuffer_device_xdir, U);
	} else if (direction == YDIR) {
	  copyDeviceArrayToBorderBufSend<YMIN, THREE_D, 2>(borderBufSend_ymin, borderBuffer_device_ydir, U);
	  copyDeviceArrayToBorderBufSend<YMAX, THREE_D, 2>(borderBufSend_ymax, borderBuffer_device_ydir, U);	
	} else { // Z direction
	  copyDeviceArrayToBorderBufSend<ZMIN, THREE_D, 2>(borderBufSend_zmin, borderBuffer_device_zdir, U);
	  copyDeviceArrayToBorderBufSend<ZMAX, THREE_D, 2>(borderBufSend_zmax, borderBuffer_device_zdir, U);
	}
      } // end THREE_D

    } // end mhdEnabled
    TIMER_STOP(timerBoundariesCpuGpu);
    
  }
#else // CPU version
  template<int direction>
  void HydroRunBaseMpi::copy_boundaries(HostArray<real_t> &U)
  {
    TIMER_START(timerBoundariesCpu);
    
    // prepare buffer border to send from Host array U
    // Note that ghostWidth is 2 for hydro and 3 for MHD

    if (mhdEnabled) {

      if (dimType == TWO_D) {
	if (direction == XDIR) {
	  copyHostArrayToBorderBufSend<XMIN, TWO_D, 3>(borderBufSend_xmin,U);
	  copyHostArrayToBorderBufSend<XMAX, TWO_D, 3>(borderBufSend_xmax,U);
	} else {
	  copyHostArrayToBorderBufSend<YMIN, TWO_D, 3>(borderBufSend_ymin,U);
	  copyHostArrayToBorderBufSend<YMAX, TWO_D, 3>(borderBufSend_ymax,U);	
	}
      } else { // THREE_D
	if (direction == XDIR) {
	  copyHostArrayToBorderBufSend<XMIN, THREE_D, 3>(borderBufSend_xmin,U);
	  copyHostArrayToBorderBufSend<XMAX, THREE_D, 3>(borderBufSend_xmax,U);
	} else if (direction == YDIR) {
	  copyHostArrayToBorderBufSend<YMIN, THREE_D, 3>(borderBufSend_ymin,U);
	  copyHostArrayToBorderBufSend<YMAX, THREE_D, 3>(borderBufSend_ymax,U);	
	} else { // Z direction
	  copyHostArrayToBorderBufSend<ZMIN, THREE_D, 3>(borderBufSend_zmin,U);
	  copyHostArrayToBorderBufSend<ZMAX, THREE_D, 3>(borderBufSend_zmax,U);
	}
      } // end THREE_D

    } else { // hydro

      if (dimType == TWO_D) {
	if (direction == XDIR) {
	  copyHostArrayToBorderBufSend<XMIN, TWO_D, 2>(borderBufSend_xmin,U);
	  copyHostArrayToBorderBufSend<XMAX, TWO_D, 2>(borderBufSend_xmax,U);
	} else {
	  copyHostArrayToBorderBufSend<YMIN, TWO_D, 2>(borderBufSend_ymin,U);
	  copyHostArrayToBorderBufSend<YMAX, TWO_D, 2>(borderBufSend_ymax,U);	
	}
      } else { // THREE_D
	if (direction == XDIR) {
	  copyHostArrayToBorderBufSend<XMIN, THREE_D, 2>(borderBufSend_xmin,U);
	  copyHostArrayToBorderBufSend<XMAX, THREE_D, 2>(borderBufSend_xmax,U);
	} else if (direction == YDIR) {
	  copyHostArrayToBorderBufSend<YMIN, THREE_D, 2>(borderBufSend_ymin,U);
	  copyHostArrayToBorderBufSend<YMAX, THREE_D, 2>(borderBufSend_ymax,U);	
	} else { // Z direction
	  copyHostArrayToBorderBufSend<ZMIN, THREE_D, 2>(borderBufSend_zmin,U);
	  copyHostArrayToBorderBufSend<ZMAX, THREE_D, 2>(borderBufSend_zmax,U);
	}
      } // end THREE_D

    } // end mhdEnabled
    TIMER_STOP(timerBoundariesCpu);
  }
#endif // __CUDACC__


  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  template<BoundaryLocation boundaryLoc>
  void HydroRunBaseMpi::make_boundary(DeviceArray<real_t> &U, HostArray<real_t> &bRecv, BoundaryConditionType bct, dim3 blockCount)
  {
    dim3 threadsPerBlock(MK_BOUND_BLOCK_SIZE, 1, 1);
    if (dimType == THREE_D) {
      threadsPerBlock.x = MK_BOUND_BLOCK_SIZE_3D;
      threadsPerBlock.y = MK_BOUND_BLOCK_SIZE_3D;
    }

    // switch according to boundary condition type
    if(bct == BC_DIRICHLET)
      {
	TIMER_START(timerBoundariesGpu);
	::make_boundary2<BC_DIRICHLET, boundaryLoc>
	    <<<blockCount, threadsPerBlock>>>(U.data(),
					      U.pitch(), 
					      U.dimx(), 
					      U.dimy(),
					      U.dimz(),
					      U.section(),
					      ghostWidth);
	TIMER_STOP(timerBoundariesGpu);
      }
    else if(bct == BC_NEUMANN)
      {
	TIMER_START(timerBoundariesGpu);
	::make_boundary2<BC_NEUMANN, boundaryLoc>
	    <<<blockCount, threadsPerBlock>>>(U.data(),
					      U.pitch(), 
					      U.dimx(), 
					      U.dimy(), 
					      U.dimz(),
					      U.section(),
					      ghostWidth);
	TIMER_STOP(timerBoundariesGpu);
      }
    else if(bct == BC_PERIODIC or bct == BC_COPY)
      {
	// ghostWidth is to 2 for hydro, 3 for MHD

	TIMER_START(timerBoundariesCpuGpu);
	if (mhdEnabled) {

	  if (dimType == TWO_D) {
	    if (boundaryLoc == XMIN or boundaryLoc == XMAX)
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,TWO_D,3>(U,borderBuffer_device_xdir,bRecv);
	    else if (boundaryLoc == YMIN or boundaryLoc == YMAX)
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,TWO_D,3>(U,borderBuffer_device_ydir,bRecv);
	  } else { // THREE_D
	    if (boundaryLoc == XMIN or boundaryLoc == XMAX)
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,THREE_D,3>(U,borderBuffer_device_xdir,bRecv);
	    else if (boundaryLoc == YMIN or boundaryLoc == YMAX)
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,THREE_D,3>(U,borderBuffer_device_ydir,bRecv);
	    else if (boundaryLoc == ZMIN or boundaryLoc == ZMAX)  
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,THREE_D,3>(U,borderBuffer_device_zdir,bRecv);
	  } // end THREE_D

	} else { // hydro

	  if (dimType == TWO_D) {
	    if (boundaryLoc == XMIN or boundaryLoc == XMAX)
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,TWO_D,2>(U,borderBuffer_device_xdir,bRecv);
	    else if (boundaryLoc == YMIN or boundaryLoc == YMAX)
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,TWO_D,2>(U,borderBuffer_device_ydir,bRecv);
	  } else { // THREE_D
	    if (boundaryLoc == XMIN or boundaryLoc == XMAX)
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,THREE_D,2>(U,borderBuffer_device_xdir,bRecv);
	    else if (boundaryLoc == YMIN or boundaryLoc == YMAX)
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,THREE_D,2>(U,borderBuffer_device_ydir,bRecv);
	    else if (boundaryLoc == ZMIN or boundaryLoc == ZMAX)  
	      copyBorderBufRecvToDeviceArray
		<boundaryLoc,THREE_D,2>(U,borderBuffer_device_zdir,bRecv);
	  } // end THREE_D

	} // end mhdEnabled
	TIMER_STOP(timerBoundariesCpuGpu);
      }
  }
#else // CPU version
  template<BoundaryLocation boundaryLoc>
  void HydroRunBaseMpi::make_boundary(HostArray<real_t> &U, HostArray<real_t> &bRecv, BoundaryConditionType bct, dim3 blockCount)
  {
    TIMER_START(timerBoundariesCpu);

    // switch according to boundary condition type
    if(bct == BC_DIRICHLET)
      {
   	::make_boundary2<BC_DIRICHLET, boundaryLoc>(U.data(),
						    U.pitch(), 
						    U.dimx(), 
						    U.dimy(),
						    U.dimz(),
						    U.section(),
						    ghostWidth);
      }
    else if(bct == BC_NEUMANN)
      {
   	::make_boundary2<BC_NEUMANN, boundaryLoc>(U.data(),
						  U.pitch(), 
						  U.dimx(), 
						  U.dimy(),
						  U.dimz(),
						  U.section(),
						  ghostWidth);
      }
    else if(bct == BC_PERIODIC or bct == BC_COPY)
      {

	if (mhdEnabled) {

	  if (dimType == TWO_D)
	    copyBorderBufRecvToHostArray<boundaryLoc,TWO_D, 3>(U,bRecv);
	  else
	    copyBorderBufRecvToHostArray<boundaryLoc,THREE_D, 3>(U,bRecv);

	} else { // hydro

	  if (dimType == TWO_D)
	    copyBorderBufRecvToHostArray<boundaryLoc,TWO_D, 2>(U,bRecv);
	  else
	    copyBorderBufRecvToHostArray<boundaryLoc,THREE_D, 2>(U,bRecv);

	} // end mhdEnabled
      }
    TIMER_STOP(timerBoundariesCpu);
  }
#endif // __CUDACC__

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void HydroRunBaseMpi::make_jet(DeviceArray<real_t> &U)
  {

    // sanity check (do nothing if ijet is larger than one MPI block)
    if (ijet >= nx or ijet >= ny)
      return;

    if (dimType == TWO_D and 
	myMpiPos[0] == 0 and 
	myMpiPos[1] == 0) {

      int blockCount = blocksFor(ijet+2+offsetJet, MAKE_JET_BLOCK_SIZE);
      float4 jetState = {djet, pjet/ (_gParams.gamma0 - 1.0f) + 0.5f * djet * ujet * ujet, 0.0f, djet * ujet};
      ::make_jet_2d<<<blockCount, MAKE_JET_BLOCK_SIZE>>>(U.data(),
							 U.pitch(), 
							 U.section(), 
							 ijet, jetState, offsetJet,
							 ghostWidth);
    } else if (dimType == THREE_D and 
	       myMpiPos[0] == 0 and 
	       myMpiPos[1] == 0 and
	       myMpiPos[2] == 0) { // THREE_D

      int blockCount = blocksFor(ijet+2+offsetJet, MAKE_JET_BLOCK_SIZE_3D);
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
    
  }
#else // CPU version
  void HydroRunBaseMpi::make_jet(HostArray<real_t> &U)
  {
    
    // sanity check (do nothing if ijet is larger than one MPI block)
    if (ijet >= nx or ijet >= ny)
      return;

    /*
     * do this only in one MPI process (rank == 0)
     */
    if (dimType == TWO_D and 
	myMpiPos[0] == 0 and 
	myMpiPos[1] == 0) {

      // matter injection in the middle of the YMIN boundary
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
    } else if (dimType == THREE_D and 
	       myMpiPos[0] == 0 and 
	       myMpiPos[1] == 0 and
	       myMpiPos[2] == 0) { // THREE_D

      for (int k=0; k<ghostWidth; ++k)
	for (int j=ghostWidth+offsetJet; j<ghostWidth+offsetJet+ijet; ++j)
	  for (int i=ghostWidth+offsetJet; i<ghostWidth+offsetJet+ijet; ++i) 
	    {
	      if ( i*i+j*j < (ghostWidth+offsetJet+ijet)*(ghostWidth+offsetJet+ijet) ) {
		U(i,j,k,ID) = djet;
		U(i,j,k,IP) = pjet/(_gParams.gamma0-1.)+0.5*djet*ujet*ujet;
		U(i,j,k,IU) = 0.0f;
		U(i,j,k,IV) = 0.0f;
		U(i,j,k,IW) = djet*ujet;
	      }
	    }
    } // end THREE_D
  }
#endif

  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  DeviceArray<real_t>& HydroRunBaseMpi::getData(int nStep) {
    if (nStep % 2 == 0)
      return d_U;
    else
      return d_U2;
  }
#else
  HostArray<real_t>& HydroRunBaseMpi::getData(int nStep) {
    if (nStep % 2 == 0)
      return h_U;
    else
      return h_U2;
  }
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  HostArray<real_t>& HydroRunBaseMpi::getDataHost(int nStep) {
    if (nStep % 2 == 0)
      return h_U;
    else
      return h_U2;
  } // HydroRunBaseMpi::getDataHost


  // =======================================================
  // =======================================================
  /**
   * dump computation results into a file (Xsmurf format 2D or 3D, one line ascii
   * header + binary data) for current time.
   * \param[in] nStep The current time step, used to label results filename. 
   * \param[in] t time
   * \param[in] iVar Define which variable to save (ID, IP, IU, IV, IW)
   * \param[in] withGhosts Include ghost borders (Usefull for debug).
   */
  void HydroRunBaseMpi::outputXsm(HostArray<real_t> &U, int nStep, ComponentIndex iVar, bool withGhosts)
  {

    if (iVar<0 or iVar>=nbVar)
      return;

    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    std::ostringstream timeFormat;
    timeFormat.width(7);
    timeFormat.fill('0');
    timeFormat << nStep;
    std::ostringstream rankFormat;
    rankFormat.width(5);
    rankFormat.fill('0');
    rankFormat << myRank;

    std::string filename = outputPrefix+"_"+varPrefix[iVar]+"_time"+timeFormat.str()+"_mpi"+rankFormat.str()+".xsm";
    std::fstream outFile;

    // begin output to file
    outFile.open (filename.c_str(), std::ios_base::out);

    if (dimType == TWO_D) {
      if (withGhosts) {
	outFile << "Binary 1 "<<isize<<"x"<<jsize<<" "<< isize*jsize <<"(" << sizeof(real_t)<<" byte reals)\n";
	for (int j=0; j<jsize; j++)
	  for (int i=0; i<isize; i++) {
	    //int index=i+isize*j;
	    // write density, U, V and energy
	    outFile.write(reinterpret_cast<char const *>(&U(i,j,iVar)), sizeof(real_t));
	  }
      } else {
	outFile << "Binary 1 "<<nx<<"x"<<ny<<" "<< nx*ny <<"(" << sizeof(real_t)<<" byte reals)\n";
	for (int j=2; j<jsize-2; j++)
	  for (int i=2; i<isize-2; i++) {
	    //int index=i+isize*j;
	    // write density, U, V and energy
	    outFile.write(reinterpret_cast<char const *>(&U(i,j,iVar)), sizeof(real_t));
	  }
      }
    } else { // THREE_D
      if (withGhosts) {
	outFile << "Binary 1 "<<isize<<"x"<<jsize<<"x"<<ksize<<" "<< isize*jsize*ksize <<"(" << sizeof(real_t)<<" byte reals)\n";
	for (int k=0; k<ksize; k++)
	  for (int j=0; j<jsize; j++)
	    for (int i=0; i<isize; i++) {
	      //int index=i+isize*j;
	      // write density, U, V, W or energy
	      outFile.write(reinterpret_cast<char const *>(&U(i,j,k,iVar)), sizeof(real_t));
	    }
      } else {
	outFile << "Binary 1 "<<nx<<"x"<<ny<<"x"<<nz<<" "<< nx*ny*nz <<"(" << sizeof(real_t)<<" byte reals)\n";
	for (int k=2; k<ksize-2; k++)
	  for (int j=2; j<jsize-2; j++)
	    for (int i=2; i<isize-2; i++) {
	      //int index=i+isize*j;
	      // write density, U, V, W or energy
	      outFile.write(reinterpret_cast<char const *>(&U(i,j,k,iVar)), sizeof(real_t));
	    }
      }
    } // end THREE_D

    outFile.close();

  } // HydroRunBaseMpi::outputXsm

  // =======================================================
  // =======================================================
  /**
   * dump computation results (conservative variables) into a Vtk file
   * : parallel image data; file extension is pvti.
   *
   * \see example use of vtkImageData :
   * http://www.vtk.org/Wiki/VTK/Examples/ImageData/IterateImageData
   *
   * \see Example of parallel vti file, in directory test/mpiBasic, file
   * testVtkXMLPImageDataWriter.cpp 
   *
   * \see Example of use of this routine, outputVtk in MPI version, in directory
   * test/mpiHydro, file testMpiOutputVtk.cpp
   *
   * Note that:
   * - if USE_VTK is defined, we can use the VTK library API (data
   *   are written using either Ascii / Raw binary / Base64 encoding
   *   possibly compressed) or the hand written routine.
   * - if USE_VTK is not defined (VTK library not installed), we
   *   automatically fall-back on the hand-written routine.
   * - the hand-written routine can be choosen (even if VTK is
   *   installed), you just need to set parameter
   *   output/outputVtkHandWritten to yes to write the VTK file "by
   *   hand" using either ascii or raw binary non-compressed format
   *  (handling base64 is a bit anoying).
   *
   *
   * \param[in] U A reference to a hydro simulation HostArray
   * \param[in] nStep The current time step, used to label results filename. 
   *
   * Usefull parameters from initialization file :
   * - output/outputVtkAscii : boolean to enable dump data in ascii format
   * - output/outputVtkBase64 : boolean to enable base64 encoding
   *   (only valid when using the vtk library routines)
   * - output/outputVtkCompression : boolean to enable/disable
   *   compression (only valid for the VTK library routine).
   * - output/outpoutVtkHandWritten : boolean to choose using the
   *   hand written dump routine (no compression implemented) instead
   *   of the VTK library's one. 
   */
  void HydroRunBaseMpi::outputVtk(HostArray<real_t> &U, int nStep)
  {

    // check scalar data type
    bool useDouble = false;
#ifdef USE_VTK
    int dataType = VTK_FLOAT;
    if (sizeof(real_t) == sizeof(double)) {
      useDouble = true;
      dataType = VTK_DOUBLE;
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

    // if using raw binary or base64, data can be zlib-compressed
    bool outputVtkCompression = configMap.getBool("output", "outputVtkCompression", true);
    // if VTK library available, we use the ZLib compressor
    std::string compressor("");
    if (!outputVtkHandWritten and outputVtkCompression)
      std::string compressor = std::string(" compressor=\"vtkZLibDataCompressor\"");
    
    // get output mode (ascii or binary)
    bool outputAscii = configMap.getBool("output", "outputVtkAscii", false);

    /*
     * Write parallel VTK header (.pvti file)
     */

    // prepare filenames
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    std::ostringstream timeFormat;
    timeFormat.width(7);
    timeFormat.fill('0');
    timeFormat << nStep;
    std::ostringstream rankFormat;
    rankFormat.width(5);
    rankFormat.fill('0');
    rankFormat << myRank;
    std::string headerFilename   = outputDir+"/"+outputPrefix+"_time"+timeFormat.str()+".pvti";
    std::string dataFilename     = outputPrefix+"_time"+timeFormat.str()+"_mpi"+rankFormat.str()+".vti";
    std::string dataFilenameFull = outputDir+"/"+outputPrefix+"_time"+timeFormat.str()+"_mpi"+rankFormat.str()+".vti";
    

    /*
     * write pvti header in a separate file.
     */
    if (myRank == 0) {
      
      std::fstream outHeader;

      // open pvti header file
      outHeader.open (headerFilename.c_str(), std::ios_base::out);
      
      outHeader << "<?xml version=\"1.0\"?>" << std::endl;
      if (isBigEndian())
	outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"BigEndian\"" << compressor << ">" << std::endl;
      else
	outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\"" << compressor << ">" << std::endl;
      outHeader << "  <PImageData WholeExtent=\"";
      outHeader << 0 << " " << mx*nx-1 << " ";
      outHeader << 0 << " " << my*ny-1 << " ";
      outHeader << 0 << " " << mz*nz-1 << "\" GhostLevel=\"0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">" << std::endl;
      outHeader << "    <PPointData Scalars=\"Scalars_\">" << std::endl;
      for (int iVar=0; iVar<nbVar; iVar++) {
	if (useDouble) 
	  outHeader << "      <PDataArray type=\"Float64\" Name=\""<< varNames[iVar]<<"\"/>" << std::endl;
	else
	  outHeader << "      <PDataArray type=\"Float32\" Name=\""<< varNames[iVar]<<"\"/>" << std::endl;	  
      }
      outHeader << "    </PPointData>" << std::endl;
      
      // one piece per MPI process
      if (dimType == TWO_D) {
	for (int iPiece=0; iPiece<nProcs; ++iPiece) {
	  std::ostringstream pieceFormat;
	  pieceFormat.width(5);
	  pieceFormat.fill('0');
	  pieceFormat << iPiece;
	  std::string pieceFilename   = outputPrefix+"_time"+timeFormat.str()+"_mpi"+pieceFormat.str()+".vti";
	  // get MPI coords corresponding to MPI rank iPiece
	  int coords[2];
	  communicator->getCoords(iPiece,2,coords);
	  outHeader << "    <Piece Extent=\"";

	  // pieces in first line of column are different (due to the special
	  // pvti file format with overlapping by 1 cell)
	  if (coords[0] == 0)
	    outHeader << 0 << " " << nx-1 << " ";
	  else
	    outHeader << coords[0]*nx-1 << " " << coords[0]*nx+nx-1 << " ";
	  if (coords[1] == 0)
	    outHeader << 0 << " " << ny-1 << " ";
	  else
	    outHeader << coords[1]*ny-1 << " " << coords[1]*ny+ny-1 << " ";
	  outHeader << 0 << " " << 0 << "\" Source=\"";
	  outHeader << pieceFilename << "\"/>" << std::endl;
	} 
      } else { // THREE_D
	for (int iPiece=0; iPiece<nProcs; ++iPiece) {
	  std::ostringstream pieceFormat;
	  pieceFormat.width(5);
	  pieceFormat.fill('0');
	  pieceFormat << iPiece;
	  std::string pieceFilename   = outputPrefix+"_time"+timeFormat.str()+"_mpi"+pieceFormat.str()+".vti";
	  // get MPI coords corresponding to MPI rank iPiece
	  int coords[3];
	  communicator->getCoords(iPiece,3,coords);
	  outHeader << " <Piece Extent=\"";

	  if (coords[0] == 0)
	    outHeader << 0 << " " << nx-1 << " ";
	  else
	    outHeader << coords[0]*nx-1 << " " << coords[0]*nx+nx-1 << " ";

	  if (coords[1] == 0)
	    outHeader << 0 << " " << ny-1 << " ";
	  else
	    outHeader << coords[1]*ny-1 << " " << coords[1]*ny+ny-1 << " ";

	  if (coords[2] == 0)
	    outHeader << 0 << " " << nz-1 << " ";
	  else
	    outHeader << coords[2]*nz-1 << " " << coords[2]*nz+nz-1 << " ";

	  outHeader << "\" Source=\"";
	  outHeader << pieceFilename << "\"/>" << std::endl;
	} 
      }
      outHeader << "</PImageData>" << std::endl;
      outHeader << "</VTKFile>" << std::endl;

      // close header file
      outHeader.close();

    } // end writing pvti header


    /*
     * write data piece by piece, one piece per MPI process
     * - if VTK library available, then use it !!!
     * - otherwise, write data "by hand"...
     */
    if (!outputVtkHandWritten) {
      /* use the VTK library API ! */

#ifdef USE_VTK

      std::fstream outFile;
      
      // create a vtkImageData object
      vtkSmartPointer<vtkImageData> imageData = 
	vtkSmartPointer<vtkImageData>::New();
      //imageData->SetDimensions(nx, ny, nz);
      imageData->SetOrigin(0.0, 0.0, 0.0);
      imageData->SetSpacing(1.0,1.0,1.0);
      imageData->SetNumberOfScalarComponents(nbVar);
      if (useDouble)
	imageData->SetScalarTypeToDouble();
      else
	imageData->SetScalarTypeToFloat();
      //imageData->AllocateScalars();

      int xmin,xmax,ymin,ymax,zmin,zmax;
      if (dimType == TWO_D) {
	xmin=myMpiPos[0]*nx   -1;
	xmax=myMpiPos[0]*nx+nx-1;
	ymin=myMpiPos[1]*ny   -1;
	ymax=myMpiPos[1]*ny+ny-1;
	zmin=0;
	zmax=0;
	if (myMpiPos[0] == 0) {
	  xmin = 0;
	  xmax = nx-1;
	}
	if (myMpiPos[1] == 0) {
	  ymin = 0;
	  ymax = ny-1;
	}
      } else { // THREE_D
	xmin=myMpiPos[0]*nx   -1;
	xmax=myMpiPos[0]*nx+nx-1;
	ymin=myMpiPos[1]*ny   -1;
	ymax=myMpiPos[1]*ny+ny-1;
	zmin=myMpiPos[2]*nz   -1;
	zmax=myMpiPos[2]*nz+nz-1;
	if (myMpiPos[0] == 0) {
	  xmin = 0;
	  xmax = nx-1;
	}
	if (myMpiPos[1] == 0) {
	  ymin = 0;
	  ymax = ny-1;
	}
	if (myMpiPos[2] == 0) {
	  zmin = 0;
	  zmax = nz-1;
	}
      } // end TWO_D
      imageData->SetExtent(xmin,xmax,
			   ymin,ymax,
			   zmin,zmax);
      	
      vtkPointData *pointData = imageData->GetPointData();

      // add density array
      vtkSmartPointer<vtkDataArray> densityArray = 
	vtkDataArray::CreateDataArray(dataType);
      densityArray->SetNumberOfComponents( 1 );
      densityArray->SetNumberOfTuples( (xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1) );
      densityArray->SetName( "density" );

      // add energy array
      vtkSmartPointer<vtkDataArray> energyArray = 
	vtkDataArray::CreateDataArray(dataType);
      energyArray->SetNumberOfComponents( 1 );
      energyArray->SetNumberOfTuples( (xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1) );
      energyArray->SetName( "energy" );

      // add momentum arrays
      vtkSmartPointer<vtkDataArray> mxArray = 
	vtkDataArray::CreateDataArray(dataType);
      mxArray->SetNumberOfComponents( 1 );
      mxArray->SetNumberOfTuples( (xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1) );
      mxArray->SetName( "mx" );
      vtkSmartPointer<vtkDataArray> myArray = 
	vtkDataArray::CreateDataArray(dataType);
      myArray->SetNumberOfComponents( 1 );
      myArray->SetNumberOfTuples( (xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1) );
      myArray->SetName( "my" );
      vtkSmartPointer<vtkDataArray> mzArray = 
	vtkDataArray::CreateDataArray(dataType);
      mzArray->SetNumberOfComponents( 1 );
      mzArray->SetNumberOfTuples( (xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1) );
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
	// do memory allocation for magnetic field component
	bxArray->SetNumberOfTuples( (xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1) );
	byArray->SetNumberOfTuples( (xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1) );
	bzArray->SetNumberOfTuples( (xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1) );
      }

      // fill the vtkImageData with scalars from U
      if (dimType == TWO_D) {
	int xmin = ghostWidth-1;
	int ymin = ghostWidth-1;
	//int extraOffsetX = 0;
	//int extraOffsetY = 0;
	if (myMpiPos[0] == 0) {
	  xmin = ghostWidth;
	  //extraOffsetX = 1;
	}
	if (myMpiPos[1] == 0) {
	  ymin = ghostWidth;
	  //extraOffsetY = 1;
	}
	//int offsetX = myMpiPos[0]*nx-1+extraOffsetX;
	//int offsetY = myMpiPos[1]*ny-1+extraOffsetY;
	int isize2 = isize-ghostWidth-xmin;
	for(int j= ymin; j < jsize-ghostWidth; j++)
	  for(int i = xmin; i < isize-ghostWidth; i++) {
	    //float* tmp = static_cast<float*>( imageData->GetScalarPointer(offsetX+i-xmin,offsetY+j-ymin,0) );
	    //tmp[0] = U(i,j,0);
	    int index = i-xmin + isize2*(j-ymin);
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
	int xmin = ghostWidth-1;
	int ymin = ghostWidth-1;
	int zmin = ghostWidth-1;
	//int extraOffsetX = 0;
	//int extraOffsetY = 0;
	//int extraOffsetZ = 0;
	if (myMpiPos[0] == 0) {
	  xmin = ghostWidth;
	  //extraOffsetX = 1;
	}
	if (myMpiPos[1] == 0) {
	  ymin = ghostWidth;
	  //extraOffsetY = 1;
	}
	if (myMpiPos[2] == 0) {
	  zmin = ghostWidth;
	  //extraOffsetZ = 1;
	}
	//int offsetX = myMpiPos[0]*nx-1+extraOffsetX;
	//int offsetY = myMpiPos[1]*ny-1+extraOffsetY;
	//int offsetZ = myMpiPos[2]*nz-1+extraOffsetZ;
	int isize2 = isize-ghostWidth-xmin;
	int jsize2 = jsize-ghostWidth-ymin;
	for(int k= zmin; k < ksize-ghostWidth; k++)
	  for(int j= ymin; j < jsize-ghostWidth; j++)
	    for(int i = xmin; i < isize-ghostWidth; i++) {
	      //float* tmp = static_cast<float*>( imageData->GetScalarPointer(offsetX+i-xmin,offsetY+j-ymin,offsetZ+k-zmin) );
	      //tmp[0] = U(i,j,k,0);
	      int index = i-xmin + isize2*(j-ymin) + isize2*jsize2*(k-zmin);
	      densityArray->SetTuple1(index, U(i,j,k,ID)); 
	      energyArray->SetTuple1(index, U(i,j,k,IP)); 
	      mxArray->SetTuple1(index, U(i,j,k,IU)); 
	      myArray->SetTuple1(index, U(i,j,k,IV)); 
	      mzArray->SetTuple1(index, U(i,j,k,IW)); 
	      if (mhdEnabled) {
		bxArray->SetTuple1(index, U(i,j,k,IA));
		byArray->SetTuple1(index, U(i,j,k,IB));
		bzArray->SetTuple1(index, U(i,j,k,IC));
	      } // end mhdEnabled
	      
	    } // end for i
      } // end THREE_D
      
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
      writer->SetInput(imageData);
      writer->SetFileName(dataFilenameFull.c_str());
      if (outputAscii)
	writer->SetDataModeToAscii();

      // do we want base 64 encoding ?? probably not
      // since is it better for data reload (simulation restart). By the
      // way reading/writing raw binary is faster since we don't need to
      // encode !
      bool enableBase64Encoding = configMap.getBool("output", "outputVtkBase64", false);
      if (!enableBase64Encoding)
	writer->EncodeAppendedDataOff();
 
      if (!outputVtkCompression) {
	//writer->SetCompressorTypeToNone();
	writer->SetCompressor(NULL);
      }
      writer->Write();

#else // VTK library not available

      // we should never arrive here, since outputVtkHandWritten is force to
      // true when VTK library is not available.
      if (myRank == 0) {
	std::cout << "##################################################" << std::endl;
	std::cout << "VTK library is not available, but you required to " << std::endl;
	std::cout << "dump output result using VTK file format !        " << std::endl;
	std::cout << "To do that, you need to turn on parameter named   " << std::endl;
	std::cout << "output/outputVtkHandWritten in initialization file" << std::endl;
      }

#endif // USE_VTK

    } else { // use the hand written routine (no need to have VTK
	     // installed)

      /*
       * Hand written procedure (no VTK library linking required).
       * Write XML imageData using either :
       * - ascii 
       * - raw binary (in appended XML section)
       *
       * Each hydrodynamics field component is written in a separate <DataArray>.
       * Magnetic field component are dumped if mhdEnabled is true !
       */
      std::fstream outFile;
      outFile.open(dataFilenameFull.c_str(), std::ios_base::out);

      int xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0;
      if (dimType == TWO_D) {
	xmin=myMpiPos[0]*nx   -1;
	xmax=myMpiPos[0]*nx+nx-1;
	ymin=myMpiPos[1]*ny   -1;
	ymax=myMpiPos[1]*ny+ny-1;
	if (myMpiPos[0] == 0) {
	  xmin = 0;
	  xmax = nx-1;
	}
	if (myMpiPos[1] == 0) {
	  ymin = 0;
	  ymax = ny-1;
	}
      } else { // THREE_D
	xmin=myMpiPos[0]*nx   -1;
	xmax=myMpiPos[0]*nx+nx-1;
	ymin=myMpiPos[1]*ny   -1;
	ymax=myMpiPos[1]*ny+ny-1;
	zmin=myMpiPos[2]*nz   -1;
	zmax=myMpiPos[2]*nz+nz-1;
	if (myMpiPos[0] == 0) {
	  xmin = 0;
	  xmax = nx-1;
	}
	if (myMpiPos[1] == 0) {
	  ymin = 0;
	  ymax = ny-1;
	}
	if (myMpiPos[2] == 0) {
	  zmin = 0;
	  zmax = nz-1;
	}
      }

      // if writing raw binary data (file does not respect XML standard)
      if (outputAscii)
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

      if (outputAscii) {

	// write ascii data
	if (dimType == TWO_D) {
	  int xmin = ghostWidth-1;
	  int ymin = ghostWidth-1;
	  if (myMpiPos[0] == 0) {
	    xmin = ghostWidth;
	  }
	  if (myMpiPos[1] == 0) {
	    ymin = ghostWidth;
	  }
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    if (useDouble)
	      outFile << "      <DataArray type=\"Float64\" Name=\"" << varNames[iVar]
		      << "\" format=\"ascii\">" << std::endl;
	    else
	      outFile << "      <DataArray type=\"Float32\" Name=\"" << varNames[iVar]
		      << "\" format=\"ascii\">" << std::endl;
	    for(int j= ymin; j < jsize-ghostWidth; j++) {
	      for(int i = xmin; i < isize-ghostWidth; i++) {
		outFile << U(i,j,iVar) << " ";
	      }
	      outFile << std::endl;
	    }
	    outFile << "      </DataArray>" << std::endl;
	  }
	} else { // THREE_D
	  int xmin = ghostWidth-1;
	  int ymin = ghostWidth-1;
	  int zmin = ghostWidth-1;
	  if (myMpiPos[0] == 0) {
	    xmin = ghostWidth;
	  }
	  if (myMpiPos[1] == 0) {
	    ymin = ghostWidth;
	  }
	  if (myMpiPos[2] == 0) {
	    zmin = ghostWidth;
	  }
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    if (useDouble)
	      outFile << "      <DataArray type=\"Float64\" Name=\"" << varNames[iVar]
		      << "\" format=\"ascii\">" << std::endl;
	    else
	      outFile << "      <DataArray type=\"Float32\" Name=\"" << varNames[iVar]
		      << "\" format=\"ascii\">" << std::endl;
	    for(int k= zmin; k < ksize-ghostWidth; k++) {
	      for(int j= ymin; j < jsize-ghostWidth; j++) {
		for(int i = xmin; i < isize-ghostWidth; i++) {
		  outFile << U(i,j,k,iVar) << " ";
		}
		outFile << std::endl;
	      }
	    }
	    outFile << "      </DataArray>" << std::endl;
	  }
	
	} // end THREE_D write ascii data

	outFile << "    </PointData>" << std::endl;
	outFile << "    <CellData>" << std::endl;
	outFile << "    </CellData>" << std::endl;
	outFile << "  </Piece>" << std::endl;
	outFile << "  </ImageData>" << std::endl;
	outFile << "</VTKFile>" << std::endl;
      
      } else { 
	// dump data using appended format raw binary (no base 64
	// encoding, no Zlib compression)

	int xmin = ghostWidth-1;
	int ymin = ghostWidth-1;
	int zmin = ghostWidth-1;
	int xmax = isize-ghostWidth;
	int ymax = jsize-ghostWidth;
	int zmax = ksize-ghostWidth;
	int nbOfTuples;

	if (myMpiPos[0] == 0) {
	  xmin = ghostWidth;
	}
	if (myMpiPos[1] == 0) {
	  ymin = ghostWidth;
	}

	if (dimType == TWO_D) {
	
	  /* compute extent and nbOfTuples */
	  nbOfTuples = (xmax-xmin)*(ymax-ymin);

	  /* write data array declaration */
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    if (useDouble)
	      outFile << "     <DataArray type=\"Float64\" Name=\"" << varNames[iVar]
		      << "\" format=\"appended\" offset=\"" << iVar*nbOfTuples*sizeof(real_t)+iVar*sizeof(unsigned int) <<"\" />" << std::endl;
	    else
	      outFile << "     <DataArray type=\"Float32\" Name=\"" << varNames[iVar]
		      << "\" format=\"appended\" offset=\"" << iVar*nbOfTuples*sizeof(real_t)+iVar*sizeof(unsigned int) <<"\" />" << std::endl;
	  }
	} else { // THREE_D
	
	  /* compute extent and nbOfTuples */
	  if (myMpiPos[2] == 0) {
	    zmin = ghostWidth;
	  }
	  nbOfTuples = (xmax-xmin)*(ymax-ymin)*(zmax-zmin);

	  for (int iVar=0; iVar<nbVar; iVar++) {
	    if (useDouble)
	      outFile << "     <DataArray type=\"Float64\" Name=\"" << varNames[iVar]
		      << "\" format=\"appended\" offset=\"" << iVar*nbOfTuples*sizeof(real_t)+iVar*sizeof(unsigned int) <<"\" />" << std::endl;
	    else
	      outFile << "     <DataArray type=\"Float32\" Name=\"" << varNames[iVar]
		      << "\" format=\"appended\" offset=\"" << iVar*nbOfTuples*sizeof(real_t)+iVar*sizeof(unsigned int) <<"\" />" << std::endl;
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
	  unsigned int nbOfWords = nbOfTuples*sizeof(real_t);
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	    for (int j=ymin; j<ymax; j++)
	      for (int i=xmin; i<xmax; i++) {
		float tmp = U(i,j,iVar);
		outFile.write((char *)&tmp,sizeof(real_t));
	      }
	  }
	} else { // THREE_D
	  unsigned int nbOfWords = nbOfTuples*sizeof(real_t);
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	    for (int k=zmin; k<zmax; k++) 
	      for (int j=ymin; j<ymax; j++) 
		for (int i=xmin; i<xmax; i++) {
		  float tmp = U(i,j,k,iVar);
		  outFile.write((char *)&tmp,sizeof(real_t));
		}
	  }
	}

	outFile << "  </AppendedData>" << std::endl;
	outFile << "</VTKFile>" << std::endl;

      } // end raw binary write
    
      outFile.close();
  
    } // end hand written routine    
    
  } // HydroRunBaseMpi::outputVtk

  // =======================================================
  // =======================================================
  /**
   * Dump computation results (conservative variables) into a file
   * (HDF5 file format) over MPI. 
   * File extension is h5. File can be viewed by hdfview; see also h5dump.
   *
   * \sa writeXdmfForHdf5Wrapper this routine write a Xdmf wrapper file for paraview.
   * \sa outputHdf5 in class HydroRunBase (serial version)
   *
   * One difference with the serial version, is that we can chose if we want 
   * - only external ghost zones to be saved (using option ghostIncluded)
   * - all ghost zones to be saved (using option allGhostIncluded): 
   *   this one is usefull for a restart run.
   *
   * Take care of parameter reassembleInFile:
   * - when true (default): different MPI pieces will be naturally put into a single
   *                         memory space to restore global topology
   * - when false: all the MPI pieces are written one next 
   *               to another in a contiguous way along Z direction (no reassemble)
   *
   * \warning if reassembleInFile is false, the output file topology
   * depends on the number of MPI task that where used to write it;
   * this routine will need to be modified if we want to perform a
   * upscale restart (need to read piece of data at different
   * location) or a restart with a different MPI configuration. 
   *
   * \warning reassembleInFile=false should be use for very large file
   * because reassembling is VERY VERY SLOW (effective
   * bandwidth can 100 smaller than expected !!!!).
   *
   * In this version, all MPI pieces are written in the same file with parallel
   * IO (MPI pieces are directly re-assembled by hdf5 library, when
   * reassembleInFile is true).
   *
   * \param[in] U A reference to a hydro simulation HostArray
   * \param[in] nStep The current time step, used to label results filename. 
   *
   * If library HDF5 is not available, do nothing.
   */
  void HydroRunBaseMpi::outputHdf5(HostArray<real_t> &U, int nStep)
  {
#ifdef USE_HDF5_PARALLEL
    bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);
    bool allghostIncluded = configMap.getBool("output","allghostIncluded",false);

    bool reassembleInFile = configMap.getBool("output", "reassembleInFile", true);

    // time measurement variables
    double write_timing, max_write_timing, write_bw;
    MPI_Offset write_size, sum_write_size;

    // verbose log ?
    bool hdf5_verbose = configMap.getBool("output","hdf5_verbose",false);

    /*
     * creation date
     */
    std::string stringDate;
    int stringDateSize;
    if (myRank==0) {
      stringDate = current_date();
      stringDateSize = stringDate.size();
    }
    // broadcast stringDate size to all other MPI tasks
    communicator->bcast(&stringDateSize, 1, MpiComm::INT, 0);

    // broadcast stringDate to all other MPI task
    if (myRank != 0) stringDate.reserve(stringDateSize);
    char* cstr = const_cast<char*>(stringDate.c_str());
    communicator->bcast(cstr, stringDateSize, MpiComm::CHAR, 0);


    /*
     * get MPI coords corresponding to MPI rank iPiece
     */
    int coords[3];
    if (dimType == TWO_D) {
      communicator->getCoords(myRank,2,coords);
    } else {
      communicator->getCoords(myRank,3,coords);
    }

    herr_t status;
    (void) status;
    
    // make filename string
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName         = outputPrefix+"_"+outNum.str();
    std::string hdf5Filename     = outputPrefix+"_"+outNum.str()+".h5";
    std::string hdf5FilenameFull = outputDir+"/"+outputPrefix+"_"+outNum.str()+".h5";
   
    // measure time ??
    if (hdf5_verbose) {
      MPI_Barrier(communicator->getComm());
      write_timing = MPI_Wtime();
    }


    /*
     * write HDF5 file
     */
    // Create a new file using property list with parallel I/O access.
    MPI_Info mpi_info     = MPI_INFO_NULL;
    hid_t    propList_create_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(propList_create_id, communicator->getComm(), mpi_info);
    hid_t    file_id  = H5Fcreate(hdf5FilenameFull.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, propList_create_id);
    H5Pclose(propList_create_id);

    // Create the data space for the dataset in memory and in file.
    hsize_t  dims_file[3];
    hsize_t  dims_memory[3];
    hsize_t  dims_chunk[3];
    hid_t dataspace_memory;
    //hid_t dataspace_chunk;
    hid_t dataspace_file;


    /*
     * reassembleInFile is false
     */
    if (!reassembleInFile) {

      if (allghostIncluded or ghostIncluded) {
	
	if (dimType == TWO_D) {
	  
	  dims_file[0] = (ny+2*ghostWidth)*(mx*my);
	  dims_file[1] = (nx+2*ghostWidth);
	  dims_memory[0] = U.dimy(); 
	  dims_memory[1] = U.dimx();
	  dims_chunk[0] = ny+2*ghostWidth;
	  dims_chunk[1] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else { // THREE_D

	  dims_file[0] = (nz+2*ghostWidth)*(mx*my*mz);
	  dims_file[1] =  ny+2*ghostWidth;
	  dims_file[2] =  nx+2*ghostWidth;
	  dims_memory[0] = U.dimz(); 
	  dims_memory[1] = U.dimy();
	  dims_memory[2] = U.dimx();
	  dims_chunk[0] = nz+2*ghostWidth;
	  dims_chunk[1] = ny+2*ghostWidth;
	  dims_chunk[2] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
	  
	} // end THREE_D
      
      } else { // no ghost zones are saved
	
	if (dimType == TWO_D) {
	  
	  dims_file[0] = (ny)*(mx*my);
	  dims_file[1] = nx;
	  dims_memory[0] = U.dimy(); 
	  dims_memory[1] = U.dimx();
	  dims_chunk[0] = ny;
	  dims_chunk[1] = nx;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else {

	  dims_file[0] = (nz)*(mx*my*mz);
	  dims_file[1] = ny;
	  dims_file[2] = nx;
	  dims_memory[0] = U.dimz(); 
	  dims_memory[1] = U.dimy();
	  dims_memory[2] = U.dimx();
	  dims_chunk[0] = nz;
	  dims_chunk[1] = ny;
	  dims_chunk[2] = nx;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
	  
	} // end THREE_D
	      
      } // end - no ghost zones are saved

    } else { 
      /*
       * reassembleInFile is true
       */

      if (allghostIncluded) {
	
	if (dimType == TWO_D) {
	  
	  dims_file[0] = my*(ny+2*ghostWidth);
	  dims_file[1] = mx*(nx+2*ghostWidth);
	  dims_memory[0] = U.dimy(); 
	  dims_memory[1] = U.dimx();
	  dims_chunk[0] = ny+2*ghostWidth;
	  dims_chunk[1] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else {

	  dims_file[0] = mz*(nz+2*ghostWidth);
	  dims_file[1] = my*(ny+2*ghostWidth);
	  dims_file[2] = mx*(nx+2*ghostWidth);
	  dims_memory[0] = U.dimz(); 
	  dims_memory[1] = U.dimy();
	  dims_memory[2] = U.dimx();
	  dims_chunk[0] = nz+2*ghostWidth;
	  dims_chunk[1] = ny+2*ghostWidth;
	  dims_chunk[2] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
	  
	}
	
      } else if (ghostIncluded) { // only external ghost zones
	
	if (dimType == TWO_D) {
	  
	  dims_file[0] = ny*my+2*ghostWidth;
	  dims_file[1] = nx*mx+2*ghostWidth;
	  dims_memory[0] = U.dimy(); 
	  dims_memory[1] = U.dimx();
	  dims_chunk[0] = ny+2*ghostWidth;
	  dims_chunk[1] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else {

	  dims_file[0] = nz*mz+2*ghostWidth;
	  dims_file[1] = ny*my+2*ghostWidth;
	  dims_file[2] = nx*mx+2*ghostWidth;
	  dims_memory[0] = U.dimz(); 
	  dims_memory[1] = U.dimy();
	  dims_memory[2] = U.dimx();
	  dims_chunk[0] = nz+2*ghostWidth;
	  dims_chunk[1] = ny+2*ghostWidth;
	  dims_chunk[2] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);

	}
	
      } else { // no ghost zones are saved
      
	if (dimType == TWO_D) {

	  dims_file[0] = ny*my;
	  dims_file[1] = nx*mx;
	  dims_memory[0] = U.dimy(); 
	  dims_memory[1] = U.dimx();
	  dims_chunk[0] = ny;
	  dims_chunk[1] = nx;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else {

	  dims_file[0] = nz*mz;
	  dims_file[1] = ny*my;
	  dims_file[2] = nx*mx;
	  dims_memory[0] = U.dimz(); 
	  dims_memory[1] = U.dimy();
	  dims_memory[2] = U.dimx();
	  dims_chunk[0] = nz;
	  dims_chunk[1] = ny;
	  dims_chunk[2] = nx;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
	  
	}
	
      } // end ghostIncluded / allghostIncluded

    } // end reassembleInFile is true

    // Create the chunked datasets.
    hid_t dataType;
    if (sizeof(real_t) == sizeof(float))
      dataType = H5T_NATIVE_FLOAT;
    else
      dataType = H5T_NATIVE_DOUBLE;
    

    /*
     * Memory space hyperslab :
     * select data with or without ghost zones
     */
    if (ghostIncluded or allghostIncluded) {
      
      if (dimType == TWO_D) {
	hsize_t  start[2] = { 0, 0 }; // no start offset
	hsize_t stride[2] = { 1, 1 };
	hsize_t  count[2] = { 1, 1 };
	hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = { 0, 0, 0 }; // no start offset
	hsize_t stride[3] = { 1, 1, 1 };
	hsize_t  count[3] = { 1, 1, 1 };
	hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
      }
      
    } else { // no ghost zones
      
      if (dimType == TWO_D) {
	hsize_t  start[2] = { (hsize_t) ghostWidth,  (hsize_t) ghostWidth }; // ghost zone width
	hsize_t stride[2] = {                    1,                     1 };
	hsize_t  count[2] = {                    1,                     1 };
	hsize_t  block[2] = {(hsize_t)           ny, (hsize_t)         nx }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = { (hsize_t) ghostWidth,  (hsize_t) ghostWidth, (hsize_t) ghostWidth }; // ghost zone width
	hsize_t stride[3] = { 1,  1,  1 };
	hsize_t  count[3] = { 1,  1,  1 };
	hsize_t  block[3] = {(hsize_t) nz, (hsize_t) ny, (hsize_t) nx }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
      }
      
    } // end ghostIncluded or allghostIncluded
    
    /*
     * File space hyperslab :
     * select where we want to write our own piece of the global data
     * according to MPI rank.
     */

    /*
     * reassembleInFile is false
     */
    if (!reassembleInFile) {

      if (dimType == TWO_D) {
	
	hsize_t  start[2] = { myRank*dims_chunk[0], 0 };
	//hsize_t  start[2] = { 0, myRank*dims_chunk[1]};
	hsize_t stride[2] = { 1,  1 };
	hsize_t  count[2] = { 1,  1 };
	hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	
      } else { // THREE_D
	
	hsize_t  start[3] = { myRank*dims_chunk[0], 0, 0 };
	hsize_t stride[3] = { 1,  1,  1 };
	hsize_t  count[3] = { 1,  1,  1 };
	hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	
      } // end THREE_D -- allghostIncluded
      
    } else {

      /*
       * reassembleInFile is true
       */

      if (allghostIncluded) {
	
	if (dimType == TWO_D) {
	  
	  hsize_t  start[2] = { coords[1]*dims_chunk[0], coords[0]*dims_chunk[1]};
	  hsize_t stride[2] = { 1,  1 };
	  hsize_t  count[2] = { 1,  1 };
	  hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	} else { // THREE_D
	  
	  hsize_t  start[3] = { coords[2]*dims_chunk[0], coords[1]*dims_chunk[1], coords[0]*dims_chunk[2]};
	  hsize_t stride[3] = { 1,  1,  1 };
	  hsize_t  count[3] = { 1,  1,  1 };
	  hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	}
	
      } else if (ghostIncluded) {
	
	// global offsets
	int gOffsetStartX, gOffsetStartY, gOffsetStartZ;
	
	if (dimType == TWO_D) {
	  gOffsetStartY  = coords[1]*ny;
	  gOffsetStartX  = coords[0]*nx;
	  
	  hsize_t  start[2] = { (hsize_t) gOffsetStartY, (hsize_t) gOffsetStartX };
	  hsize_t stride[2] = { 1,  1 };
	  hsize_t  count[2] = { 1,  1 };
	  hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	} else { // THREE_D
	  
	  gOffsetStartZ  = coords[2]*nz;
	  gOffsetStartY  = coords[1]*ny;
	  gOffsetStartX  = coords[0]*nx;
	  
	  hsize_t  start[3] = { (hsize_t) gOffsetStartZ, (hsize_t) gOffsetStartY, (hsize_t) gOffsetStartX };
	  hsize_t stride[3] = { 1,  1,  1 };
	  hsize_t  count[3] = { 1,  1,  1 };
	  hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	}
	
      } else { // no ghost zones
	
	if (dimType == TWO_D) {
	  
	  hsize_t  start[2] = { coords[1]*dims_chunk[0], coords[0]*dims_chunk[1]};
	  hsize_t stride[2] = { 1,  1 };
	  hsize_t  count[2] = { 1,  1 };
	  hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	} else { // THREE_D
	  
	  hsize_t  start[3] = { coords[2]*dims_chunk[0], coords[1]*dims_chunk[1], coords[0]*dims_chunk[2]};
	  hsize_t stride[3] = { 1,  1,  1 };
	  hsize_t  count[3] = { 1,  1,  1 };
	  hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	} // end THREE_D
	
      } // end ghostIncluded / allghostIncluded

    } // end reassembleInFile is true

    /*
     *
     * write heavy data to HDF5 file
     *
     */
    real_t* data;
    propList_create_id = H5Pcreate(H5P_DATASET_CREATE);
    if (dimType == TWO_D)
      H5Pset_chunk(propList_create_id, 2, dims_chunk);
    else
      H5Pset_chunk(propList_create_id, 3, dims_chunk);

    // please note that HDF5 parallel I/O does not support yet filters
    // so we can't use here H5P_deflate to perform compression !!!
    // Weak solution : call h5repack after the file is created
    // (performance of that has not been tested)

    // take care not to use parallel specific features if the HDF5
    // library available does not support them !!
    hid_t propList_xfer_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(propList_xfer_id, H5FD_MPIO_COLLECTIVE);

    hid_t dataset_id;

    /*
     * write density    
     */
    dataset_id = H5Dcreate2(file_id, "/density", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(U(0,0,ID));
    else
      data = &(U(0,0,0,ID));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    /*
     * write energy
     */
    dataset_id = H5Dcreate2(file_id, "/energy", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(U(0,0,IP));
    else
      data = &(U(0,0,0,IP));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    /*
     * write momentum X
     */
    dataset_id = H5Dcreate2(file_id, "/momentum_x", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(U(0,0,IU));
    else
      data = &(U(0,0,0,IU));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    /*
     * write momentum Y
     */
    dataset_id = H5Dcreate2(file_id, "/momentum_y", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(U(0,0,IV));
    else
      data = &(U(0,0,0,IV));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    /*
     * write momentum Z (only if 3D or MHD enabled)
     */
    if (dimType == THREE_D and !mhdEnabled) {
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      data = &(U(0,0,0,IW));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);
    }
    
    if (mhdEnabled) {
      // write momentum z
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      if (dimType == TWO_D)
	data = &(U(0,0,IW));
      else
	data = &(U(0,0,0,IW));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);

      // write magnetic field components
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_x", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      if (dimType == TWO_D)
	data = &(U(0,0,IA));
      else
	data = &(U(0,0,0,IA));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);
      
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_y", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      if (dimType == TWO_D)
	data = &(U(0,0,IB));
       else
	 data = &(U(0,0,0,IB));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);
      
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      if (dimType == TWO_D)
	data = &(U(0,0,IC));
      else
	data = &(U(0,0,0,IC));
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);

    }

    // write time step number
    hid_t ds_id   = H5Screate(H5S_SCALAR);
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

    // write information about ghost zone
    {
      int tmpVal = ghostIncluded ? 1 : 0;
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "external ghost zones only included", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &tmpVal);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    {
      int tmpVal = allghostIncluded ? 1 : 0;
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "all ghost zones included", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &tmpVal);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write information about reassemble MPI pieces in file
    {
      int tmpVal = reassembleInFile ? 1 : 0;
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "reassemble MPI pieces in file", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &tmpVal);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write local geometry information (just to be consistent)
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
    
    // write MPI topology sizes
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "mx", H5T_NATIVE_INT, 
				 ds_id,
				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &mx);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "my", H5T_NATIVE_INT, 
				 ds_id,
				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &my);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "mz", H5T_NATIVE_INT, 
				 ds_id,
				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &mz);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    /*
     * write creation date
     */
    {
      hsize_t   dimsAttr[1] = {1};
      hid_t memtype = H5Tcopy (H5T_C_S1);
      status = H5Tset_size (memtype, stringDateSize+1);
      hid_t root_id = H5Gopen(file_id, "/", H5P_DEFAULT);
      hid_t dataspace_id = H5Screate_simple(1, dimsAttr, NULL);
      attr_id = H5Acreate(root_id, "creation date", memtype, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, memtype, cstr);
      status = H5Aclose(attr_id);
      status = H5Gclose(root_id);
      status = H5Tclose(memtype);
      status = H5Sclose(dataspace_id);
    }

    // close/release resources.
    H5Pclose(propList_create_id);
    H5Pclose(propList_xfer_id);
    H5Sclose(dataspace_memory);
    H5Sclose(dataspace_file);
    H5Fclose(file_id);


    // verbose log about memory bandwidth
    if (hdf5_verbose) {

      write_timing = MPI_Wtime() - write_timing;
      
      //write_size = nbVar * U.section() * sizeof(real_t);
      write_size = U.sizeBytes();
      sum_write_size = write_size *  nProcs;
      
      MPI_Reduce(&write_timing, &max_write_timing, 1, MPI_DOUBLE, MPI_MAX, 0, communicator->getComm());

      if (myRank==0) {
	printf("########################################################\n");
	printf("################### HDF5 bandwidth #####################\n");
	printf("########################################################\n");
	printf("Local  array size %d x %d x %d reals(%zu bytes), write size = %.2f MB\n",
	       nx+2*ghostWidth,
	       ny+2*ghostWidth,
	       nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*write_size/1048576.0);
	sum_write_size /= 1048576.0;
	printf("Global array size %d x %d x %d reals(%zu bytes), write size = %.2f GB\n",
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*sum_write_size/1024);
	
	write_bw = sum_write_size/max_write_timing;
	printf(" procs    Global array size  exec(sec)  write(MB/s)\n");
	printf("-------  ------------------  ---------  -----------\n");
	printf(" %4d    %4d x %4d x %4d %8.2f  %10.2f\n", nProcs,
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       max_write_timing, write_bw);
	printf("########################################################\n");
      } // end (myRank == 0)

    } // hdf5_verbose

#else

    if (myRank == 0) {
      std::cerr << "Parallel HDF5 library is not available ! You can't load a data file for restarting the simulation run !!!" << std::endl;
      std::cerr << "Please install Parallel HDF5 library !!!" << std::endl;
    }

#endif // USE_HDF5_PARALLEL

  } // HydroRunBaseMpi::outputHdf5

  // =======================================================
  // =======================================================
  /**
   * dump debug array into a file
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
  void HydroRunBaseMpi::outputHdf5Debug(HostArray<real_t> &data, const std::string suffix, int nStep)
  {
#ifdef USE_HDF5
    herr_t status;
    (void) status;

    int nbVarDebug = data.nvar();
    
    // make filename string
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName         = outputPrefix+"_debug_"+suffix+outNum.str();
    std::ostringstream rankNum;
    rankNum.width(5);
    rankNum.fill('0');
    rankNum << myRank;
    baseName = baseName+"_rank_"+rankNum.str();
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

#endif // USE_HDF5
  } // HydroRunBaseMpi::outputHdf5Debug


    // =======================================================
    // =======================================================
    /**
     * Write a wrapper file using the Xmdf file format (XML) to allow
     * Paraview to read these h5 files as a time series.
     *
     * \note This routine is not exactly the same as the one located
     * in HydroRunBase class, only dimensions are changed since each MPI
     * process writes a chunk of total data.
     *
     * \param[in] totalNumberOfSteps The number of time steps computed.
     *
     * If library HDF5 is not available, do nothing.
     */
  void HydroRunBaseMpi::writeXdmfForHdf5Wrapper(int totalNumberOfSteps)
  {
#ifdef USE_HDF5_PARALLEL

    // global sizes
    int nxg = mx*nx;
    int nyg = my*ny;
    int nzg = mz*nz;

    bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);
    if (ghostIncluded) {
      nxg += (2*ghostWidth);
      nyg += (2*ghostWidth);
      nzg += (2*ghostWidth);
    }

    bool allghostIncluded = configMap.getBool("output","allghostIncluded",false);
    if (allghostIncluded) {
      nxg = mx*(nx+2*ghostWidth);
      nyg = my*(ny+2*ghostWidth);
      nzg = mz*(nz+2*ghostWidth);
    }

    bool reassembleInFile = configMap.getBool("output", "reassembleInFile", true);
    if (!reassembleInFile) {
      if (dimType==TWO_D) {
	if (allghostIncluded or ghostIncluded) {
	  nxg = (nx+2*ghostWidth);
	  nyg = (ny+2*ghostWidth)*mx*my;
	} else {
	  nxg = nx;
	  nyg = ny*mx*my;
	}
      } else {
	if (allghostIncluded or ghostIncluded) {
	  nxg = nx+2*ghostWidth;
	  nyg = ny+2*ghostWidth;
	  nzg = (nz+2*ghostWidth)*mx*my*mz;
	} else {
	  nxg = nx;
	  nyg = ny;
	  nzg = nz*mx*my*mz;
	}
      }
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
    std::fstream xdmfFile;
    xdmfFile.open(xdmfFilename.c_str(), std::ios_base::out);

    xdmfFile << "<?xml version=\"1.0\" ?>"                       << std::endl;
    xdmfFile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>"         << std::endl;
    xdmfFile << "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">" << std::endl;
    xdmfFile << "  <Domain>"                                     << std::endl;
    xdmfFile << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;

    // for each time step write a <grid> </grid> item
    for (int nStep=0; nStep<=totalNumberOfSteps; nStep+=nOutput) {
 
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
	xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_x"       << std::endl;
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
	xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_y"       << std::endl;
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
	xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_z"       << std::endl;
	xdmfFile << "        </DataStructure>"                              << std::endl;
	xdmfFile << "      </Attribute>"                                    << std::endl;
	
      }

      // finalize grid file for the current time step
      xdmfFile << "   </Grid>" << std::endl;
      
    } // end for loop over time step
    
      // finalize Xdmf wrapper file
    xdmfFile << "   </Grid>" << std::endl;
    xdmfFile << " </Domain>" << std::endl;
    xdmfFile << "</Xdmf>"    << std::endl;


#endif // USE_HDF5_PARALLEL
  } // HydroRunBaseMpi::writeXdmfForHdf5Wrapper

  // =======================================================
  // =======================================================
  /**
   * Dump computation results (conservative variables) into a file
   * (Parallel-netCDF file format) over MPI. 
   * File extension is nc. 
   * 
   * NetCDF file creation supports:
   * - CDF-2 (using creation mode NC_64BIT_OFFSET)
   * - CDF-5 (using creation mode NC_64BIT_DATA)
   *
   * \note NetCDF file can be viewed by ncBrowse; see also ncdump.
   * \warning ncdump does not support CDF-5 format ! 
   *
   * All MPI pieces are written in the same file with parallel
   * IO (MPI pieces are directly re-assembled by Parallel-netCDF library).
   *
   * \param[in] U A reference to a hydro simulation HostArray
   * \param[in] nStep The current time step, used to label results filename. 
   *
   * If library Parallel-netCDF is not available, do nothing.
   */
  void HydroRunBaseMpi::outputPnetcdf(HostArray<real_t> &U, int nStep)
  {
#ifdef USE_PNETCDF
    //bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);
    //bool allghostIncluded = configMap.getBool("output","allghostIncluded",false);

    // netcdf file id
    int ncFileId;
    int err;

    // file creation mode
    int ncCreationMode = NC_CLOBBER;
    bool useCDF5 = configMap.getBool("output","pnetcdf_cdf5",false);
    if (useCDF5)
      ncCreationMode = NC_CLOBBER|NC_64BIT_DATA;
    else // use CDF-2 file format
      ncCreationMode = NC_CLOBBER|NC_64BIT_OFFSET;

    // verbose log ?
    bool pnetcdf_verbose = configMap.getBool("output","pnetcdf_verbose",false);

    int dimIds[3], varIds[nbVar];
    MPI_Offset starts[3], counts[3], write_size, sum_write_size;
    MPI_Info mpi_info_used;
    //char str[512];

    // time measurement variables
    double write_timing, max_write_timing, write_bw;

    /*
     * creation date
     */
    std::string stringDate;
    int stringDateSize;
    if (myRank==0) {
      stringDate = current_date();
      stringDateSize = stringDate.size();
    }
    // broadcast stringDate size to all other MPI tasks
    communicator->bcast(&stringDateSize, 1, MpiComm::INT, 0);

    // broadcast stringDate to all other MPI task
    if (myRank != 0) stringDate.reserve(stringDateSize);
    char* cstr = const_cast<char*>(stringDate.c_str());
    communicator->bcast(cstr, stringDateSize, MpiComm::CHAR, 0);


    /*
     * get MPI coords corresponding to MPI rank iPiece
     */
    int coords[3];
    if (dimType == TWO_D) {
      communicator->getCoords(myRank,2,coords);
    } else {
      communicator->getCoords(myRank,3,coords);
    }
    
    /*
     * make filename string
     */
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName       = outputPrefix+"_"+outNum.str();
    std::string ncFilename     = outputPrefix+"_"+outNum.str()+".nc";
    std::string ncFilenameFull = outputDir+"/"+ncFilename;

    // measure time ??
    if (pnetcdf_verbose) {
      MPI_Barrier(communicator->getComm());
      write_timing = MPI_Wtime();
    }

    /* 
     * Create NetCDF file
     */
    err = ncmpi_create(communicator->getComm(), ncFilenameFull.c_str(), 
		       ncCreationMode,
                       MPI_INFO_NULL, &ncFileId);
    if (err != NC_NOERR) {
      printf("Error: ncmpi_create() file %s (%s)\n",ncFilenameFull.c_str(),ncmpi_strerror(err));
      MPI_Abort(communicator->getComm(), -1);
      exit(1);
    }
    
    /*
     * Define dimensions
     */
    int gsizes[3];
    if (dimType == TWO_D) {
      gsizes[1] = mx*nx+2*ghostWidth;
      gsizes[0] = my*ny+2*ghostWidth;
      
      err = ncmpi_def_dim(ncFileId, "x", gsizes[0], &dimIds[0]);
      PNETCDF_HANDLE_ERROR;
      
      err = ncmpi_def_dim(ncFileId, "y", gsizes[1], &dimIds[1]);
      PNETCDF_HANDLE_ERROR;
    
    } else { 
      gsizes[2] = mx*nx+2*ghostWidth;
      gsizes[1] = my*ny+2*ghostWidth;
      gsizes[0] = mz*nz+2*ghostWidth;
      
      err = ncmpi_def_dim(ncFileId, "x", gsizes[0], &dimIds[0]);
      PNETCDF_HANDLE_ERROR;
      
      err = ncmpi_def_dim(ncFileId, "y", gsizes[1], &dimIds[1]);
      PNETCDF_HANDLE_ERROR;

      err = ncmpi_def_dim(ncFileId, "z", gsizes[2], &dimIds[2]);
      PNETCDF_HANDLE_ERROR;
    }
    
    /* 
     * Define variables
     */
    nc_type ncDataType;
    MPI_Datatype mpiDataType;

    if (sizeof(real_t) == sizeof(float)) {
      ncDataType  = NC_FLOAT;
      mpiDataType = MPI_FLOAT;
    } else {
      ncDataType  = NC_DOUBLE;
      mpiDataType = MPI_DOUBLE;
    }
    
    if (dimType==TWO_D) {
      err = ncmpi_def_var(ncFileId, "rho", ncDataType, 2, dimIds, &varIds[ID]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "E", ncDataType, 2, dimIds, &varIds[IP]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vx", ncDataType, 2, dimIds, &varIds[IU]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vy", ncDataType, 2, dimIds, &varIds[IV]);
      PNETCDF_HANDLE_ERROR;

      if (mhdEnabled) {
	err = ncmpi_def_var(ncFileId, "rho_vz", ncDataType, 2, dimIds, &varIds[IW]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "Bx", ncDataType, 2, dimIds, &varIds[IA]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "By", ncDataType, 2, dimIds, &varIds[IB]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "Bz", ncDataType, 2, dimIds, &varIds[IC]);
	PNETCDF_HANDLE_ERROR;
      }
      
    } else { // THREE_D

      err = ncmpi_def_var(ncFileId, "rho", ncDataType, 3, dimIds, &varIds[ID]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "E", ncDataType, 3, dimIds, &varIds[IP]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vx", ncDataType, 3, dimIds, &varIds[IU]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vy", ncDataType, 3, dimIds, &varIds[IV]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vz", ncDataType, 3, dimIds, &varIds[IW]);
      PNETCDF_HANDLE_ERROR;

      if (mhdEnabled) {
	err = ncmpi_def_var(ncFileId, "Bx", ncDataType, 3, dimIds, &varIds[IA]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "By", ncDataType, 3, dimIds, &varIds[IB]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "Bz", ncDataType, 3, dimIds, &varIds[IC]);
	PNETCDF_HANDLE_ERROR;
      }

    } // end THREE_D

    /*
     * global attributes
     */
    // did we use CDF-2 or CDF-5
    {
      int useCDF5_int = useCDF5 ? 1 : 0;
      err = ncmpi_put_att_int(ncFileId, NC_GLOBAL, "CDF-5 mode", NC_INT, 1, &useCDF5_int);
      PNETCDF_HANDLE_ERROR;
    }
    
    // write time step number
    err = ncmpi_put_att_int(ncFileId, NC_GLOBAL, "time step", NC_INT, 1, &nStep);
    PNETCDF_HANDLE_ERROR;

    // write total time
    {
      double timeValue = (double) totalTime;
      err = ncmpi_put_att_double(ncFileId, NC_GLOBAL, "total time", NC_DOUBLE, 1, &timeValue);
      PNETCDF_HANDLE_ERROR;
    }

    // for information MPI config used
    {
      std::ostringstream mpiConf;
      mpiConf << "MPI configuration used to write file: "
	      << "mx,my,mz="
	      << mx << "," << my << "," << mz << " "
	      << "nx,ny,nz="
	      << nx << "," << ny << "," << nz;

      err = ncmpi_put_att_text(ncFileId, NC_GLOBAL, "MPI conf", mpiConf.str().size(), mpiConf.str().c_str());
      PNETCDF_HANDLE_ERROR;	    
    }

    /*
     * write creation date
     */
    {
      err = ncmpi_put_att_text(ncFileId, NC_GLOBAL, "creation date", stringDateSize, stringDate.c_str());
      PNETCDF_HANDLE_ERROR;	    
    }

    /* 
     * exit the define mode 
     */
    err = ncmpi_enddef(ncFileId);
    PNETCDF_HANDLE_ERROR;

    /* 
     * Get all the MPI_IO hints used
     */
    err = ncmpi_get_file_info(ncFileId, &mpi_info_used);
    PNETCDF_HANDLE_ERROR;

    /*
     * Write heavy data (take care of row-major / column major format !)
     */
    // write mode
    bool useOverlapMode = configMap.getBool("output","pnetcdf_overlap",false);

    if (useOverlapMode) {

      // first solution (use overlapping domains)
      if (dimType == TWO_D) {

        counts[IY] = nx+2*ghostWidth;
        counts[IX] = ny+2*ghostWidth;
      
        starts[IY] = coords[IX]*nx;
        starts[IX] = coords[IY]*ny;

      } else { // THREE_D

        counts[IZ] = nx+2*ghostWidth;
        counts[IY] = ny+2*ghostWidth;
        counts[IX] = nz+2*ghostWidth;
      
        starts[IZ] = coords[IX]*nx;
        starts[IY] = coords[IY]*ny;
        starts[IX] = coords[IZ]*nz;

      } // end THREE_D

    } else {

      // use non-overlapped domain
      
      if (dimType == TWO_D) {

	counts[IY] = nx;
	counts[IX] = ny;
	
	starts[IY] = coords[IX]*nx;
	starts[IX] = coords[IY]*ny;
	
	// take care of borders along X
	if (coords[IX]==mx-1) {
	  counts[IY] += 2*ghostWidth;
	}
	
	// take care of borders along Y
	if (coords[IY]==my-1) {
	  counts[IX] += 2*ghostWidth;
	}
		
      } else { // THREE_D

	counts[IZ] = nx;
	counts[IY] = ny;
	counts[IX] = nz;

	starts[IZ] = coords[IX]*nx;
	starts[IY] = coords[IY]*ny;
	starts[IX] = coords[IZ]*nz;

	// take care of borders along X
	if (coords[IX]==mx-1) {
	  counts[IZ] += 2*ghostWidth;
	}
	// take care of borders along Y
	if (coords[IY]==my-1) {
	  counts[IY] += 2*ghostWidth;
	}
	// take care of borders along Z
	if (coords[IZ]==mz-1) {
	  counts[IX] += 2*ghostWidth;
	}

      } // end THREE_D
    
    } // end non-overlapped mode


    int nItems = counts[IX]*counts[IY];
    if (dimType==THREE_D)
      nItems *= counts[IZ];

    if (useOverlapMode) {
      
      for (int iVar=0; iVar<nbVar; iVar++) {
	real_t* data;
	if (dimType==TWO_D) {
	  data = &(U(0,0,iVar));
	} else {
	  data = &(U(0,0,0,iVar));
	}
	err = ncmpi_put_vara_all(ncFileId, varIds[iVar], starts, counts, data, nItems, mpiDataType);
	PNETCDF_HANDLE_ERROR;
      } // end for loop writing heavy data
    
    } else { // data need to allocated and copied from U
    
      real_t* data;

      data = (real_t *) malloc(nItems*sizeof(real_t));
      
      int iStop=nx, jStop=ny, kStop=nz;

      if (coords[IX]== mx-1) iStop=nx+2*ghostWidth;
      if (coords[IY]== my-1) jStop=ny+2*ghostWidth;
      if (coords[IZ]== mz-1) kStop=nz+2*ghostWidth;

      for (int iVar=0; iVar<nbVar; iVar++) {
	
	// copy needed data into data !
	if (dimType==TWO_D) {
	  
	  int dI=0;
	  for (int j= 0; j < jStop; j++)
	    for(int i = 0; i < iStop; i++) {
	      data[dI] = U(i,j,iVar);
	      dI++;
	    }
	  
	  err = ncmpi_put_vara_all(ncFileId, varIds[iVar], starts, counts, data, nItems, mpiDataType);
	PNETCDF_HANDLE_ERROR;

	} else { // THREE_D
	  
	  int dI=0;
	  for (int k= 0; k < kStop; k++)
	    for (int j= 0; j < jStop; j++)
	      for(int i = 0; i < iStop; i++) {
		data[dI] = U(i,j,k,iVar);
		dI++;
	      }
	  
	  err = ncmpi_put_vara_all(ncFileId, varIds[iVar], starts, counts, data, nItems, mpiDataType);
	  PNETCDF_HANDLE_ERROR;
	} // THREE_D
	
      } // end for iVar

      free(data);

    } // end non-overlap mode
    
    /* 
     * close the file 
     */
    err = ncmpi_close(ncFileId);
    PNETCDF_HANDLE_ERROR;

    // verbose log about memory bandwidth
    if (pnetcdf_verbose) {

      write_timing = MPI_Wtime() - write_timing;
      
      //write_size = nbVar * U.section() * sizeof(real_t);
      write_size = U.sizeBytes();
      sum_write_size = write_size *  nProcs;
      
      MPI_Reduce(&write_timing, &max_write_timing, 1, MPI_DOUBLE, MPI_MAX, 0, communicator->getComm());

      if (myRank==0) {
	printf("########################################################\n");
	printf("############## Parallel-netCDF bandwidth ###############\n");
	printf("########################################################\n");
	printf("Local  array size %d x %d x %d reals(%zu bytes), write size = %.2f MB\n",
	       nx+2*ghostWidth,
	       ny+2*ghostWidth,
	       nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*write_size/1048576.0);
	sum_write_size /= 1048576.0;
	printf("Global array size %d x %d x %d reals(%zu bytes), write size = %.2f GB\n",
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*sum_write_size/1024);
	
	write_bw = sum_write_size/max_write_timing;
	printf(" procs    Global array size  exec(sec)  write(MB/s)\n");
	printf("-------  ------------------  ---------  -----------\n");
	printf(" %4d    %4d x %4d x %4d %8.2f  %10.2f\n", nProcs,
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       max_write_timing, write_bw);
	printf("########################################################\n");
      } // end (myRank == 0)

    } // pnetcdf_verbose
    
    /*
     * Print MPI information
     */
    bool pnetcdf_print_mpi_info = configMap.getBool("output","pnetcdf_print_mpi_info",false);
    if (pnetcdf_print_mpi_info and myRank==0) {
      
      int     flag;
      char    info_cb_nodes[64], info_cb_buffer_size[64];
      char    info_striping_factor[64], info_striping_unit[64];

      char undefined_char[]="undefined";

      strcpy(info_cb_nodes,        undefined_char);
      strcpy(info_cb_buffer_size,  undefined_char);
      strcpy(info_striping_factor, undefined_char);
      strcpy(info_striping_unit,   undefined_char);
      
      char cb_nodes_char[]       ="cb_nodes";
      char cb_buffer_size_char[] ="cb_buffer_size";
      char striping_factor_char[]="striping_factor";
      char striping_unit_char[]  ="striping_unit";

      MPI_Info_get(mpi_info_used, cb_nodes_char       , 64, info_cb_nodes, &flag);
      MPI_Info_get(mpi_info_used, cb_buffer_size_char , 64, info_cb_buffer_size, &flag);
      MPI_Info_get(mpi_info_used, striping_factor_char, 64, info_striping_factor, &flag);
      MPI_Info_get(mpi_info_used, striping_unit_char  , 64, info_striping_unit, &flag);
      
      printf("MPI hint: cb_nodes        = %s\n", info_cb_nodes);
      printf("MPI hint: cb_buffer_size  = %s\n", info_cb_buffer_size);
      printf("MPI hint: striping_factor = %s\n", info_striping_factor);
      printf("MPI hint: striping_unit   = %s\n", info_striping_unit);
      
    } // pnetcdf_print_mpi_info
    
#endif // USE_PNETCDF   
    
  } // HydroRunBaseMpi::outputPnetcdf

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
  void  HydroRunBaseMpi::output(HostArray<real_t> &U, int nStep)
  {

    if (outputVtkEnabled)  outputVtk (getDataHost(nStep), nStep);
    if (outputHdf5Enabled) outputHdf5(getDataHost(nStep), nStep);
    if (outputPnetcdfEnabled) outputPnetcdf(getDataHost(nStep), nStep);

    // extra output (only proc 0 do it)
    if (myRank == 0) {
      if ( !problem.compare("turbulence-Ornstein-Uhlenbeck") ) {
	// need to output forcing parameters
	pForcingOrnsteinUhlenbeck -> output_forcing(nStep);
      }
    }

  } // HydroRunBaseMpi::output

  // =======================================================
  // =======================================================
  /**
   * Dump faces data using pnetcdf format (for CPU computations).
   *
   * \param[in] nStep
   * \param[in] pnetcdfEnabled
   */
  void HydroRunBaseMpi::outputFaces(int nStep, bool pnetcdfEnabled)
  {
    
    HostArray<real_t> &U = getDataHost(nStep);
    if (pnetcdfEnabled) {
      outputFacesPnetcdf(U,nStep, IX);
      outputFacesPnetcdf(U,nStep, IY);
      outputFacesPnetcdf(U,nStep, IZ);
    }
    
  } // HydroRunBaseMpi::outputFaces

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  /**
   * Dump faces data using pnetcdf format (for GPU simulations).
   *
   * \param[in] nStep
   * \param[in] pnetcdfEnabled
   * \param[inout] h_xface
   * \param[inout] h_yface
   * \param[inout] h_zface
   * \param[inout] d_xface
   * \param[inout] d_yface
   * \param[inout] d_zface
   */
  void HydroRunBaseMpi::outputFaces(int nStep, bool pnetcdfEnabled,
				    HostArray<real_t>   &h_xface,
				    HostArray<real_t>   &h_yface,
				    HostArray<real_t>   &h_zface,
				    DeviceArray<real_t> &d_xface,
				    DeviceArray<real_t> &d_yface,
				    DeviceArray<real_t> &d_zface)
  {
    
    DeviceArray<real_t> &U = getData(nStep);
    if (pnetcdfEnabled) {
      // copy X-face from GPU
      if (myMpiPos[IX]==0) {
	dim3 dimBlock(16, 16);
	dim3 dimGrid(blocksFor(jsize, dimBlock.x),
		     blocksFor(ksize, dimBlock.y));
	kernel_copy_face_x<<< dimGrid, dimBlock >>>(U.data(), d_xface.data(),
						    U.dimx(), U.dimy(), U.dimz(), U.pitch(),
						    d_xface.dimx(), 
						    d_xface.dimy(), 
						    d_xface.dimz(), 
						    d_xface.pitch(), 
						    U.nvar());
	d_xface.copyToHost(h_xface);
      }
      
      // copy Y-face from GPU
      if (myMpiPos[IY]==0) {
	dim3 dimBlock(16, 16);
	dim3 dimGrid(blocksFor(isize, dimBlock.x),
		     blocksFor(ksize, dimBlock.y));
	kernel_copy_face_y<<< dimGrid, dimBlock >>>(U.data(), d_yface.data(),
						    U.dimx(), U.dimy(), U.dimz(), U.pitch(),
						    d_yface.dimx(), 
						    d_yface.dimy(), 
						    d_yface.dimz(), 
						    d_yface.pitch(), 
						    U.nvar());
	d_yface.copyToHost(h_yface);
      }
      
      // copy Z-face from GPU
      if (myMpiPos[IZ]==0) {
	dim3 dimBlock(16, 16);
	dim3 dimGrid(blocksFor(isize, dimBlock.x),
		     blocksFor(jsize, dimBlock.y));
	kernel_copy_face_z<<< dimGrid, dimBlock >>>(U.data(), d_zface.data(),
						    U.dimx(), U.dimy(), U.dimz(), U.pitch(),
						    d_zface.dimx(), 
						    d_zface.dimy(), 
						    d_zface.dimz(), 
						    d_zface.pitch(), 
						    U.nvar());
	d_zface.copyToHost(h_zface);
      }
      
      // dump to file
      outputFacesPnetcdf(h_xface,nStep, IX);
      outputFacesPnetcdf(h_yface,nStep, IY);
      outputFacesPnetcdf(h_zface,nStep, IZ);

    } // if pnetcdfEnabled

  } // HydroRunBaseMpi::outputFaces
  
#endif // __CUDACC__
  
  // =======================================================
  // =======================================================
  /**
   * Dump computation results (conservative variables) into a file
   * (Parallel-netCDF file format) over MPI. 
   * File extension is nc. 
   * 
   * NetCDF file creation supports:
   * - CDF-2 (using creation mode NC_64BIT_OFFSET)
   * - CDF-5 (using creation mode NC_64BIT_DATA)
   *
   * \note NetCDF file can be viewed by ncBrowse; see also ncdump.
   * \warning ncdump does not support CDF-5 format ! 
   *
   * All MPI pieces are written in the same file with parallel
   * IO (MPI pieces are directly re-assembled by Parallel-netCDF library).
   *
   * \param[in] U A reference to a hydro simulation HostArray
   * \param[in] nStep The current time step, used to label results filename. 
   *
   * If library Parallel-netCDF is not available, do nothing.
   */
  void HydroRunBaseMpi::outputFacesPnetcdf(HostArray<real_t> &U, 
					   int nStep, 
					   ComponentIndex3D faceDir)
  {

    int faceColor = (myMpiPos[faceDir]==0) ? 1 : 0;

    // create MPI communicator
    MPI_Comm faceComm;
    MPI_Comm_split(communicator->getComm(), faceColor, myRank, &faceComm);

#ifdef USE_PNETCDF

    // netcdf file id
    int ncFileId;
    int err;

    // file creation mode
    int ncCreationMode = NC_CLOBBER;
    bool useCDF5 = configMap.getBool("output","faces_pnetcdf_cdf5",false);
    if (useCDF5)
      ncCreationMode = NC_CLOBBER|NC_64BIT_DATA;
    else // use CDF-2 file format
      ncCreationMode = NC_CLOBBER|NC_64BIT_OFFSET;

    // verbose log ?
    //bool pnetcdf_verbose = configMap.getBool("output","faces_pnetcdf_verbose",false);

    int dimIds[3], varIds[nbVar];
    MPI_Offset starts[3], counts[3]; // write_size, sum_write_size;
    MPI_Info mpi_info_used;
    //char str[512];

    /*
     * make filename string
     */
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName       = outputPrefix+"_"+outNum.str();
    if (faceDir == IX)
      baseName += "_IX";
    if (faceDir == IY)
      baseName += "_IY";
    if (faceDir == IZ)
      baseName += "_IZ";
    std::string ncFilename     = baseName+".nc";
    std::string ncFilenameFull = outputDir+"/"+ncFilename;

    if (faceColor == 0) {

      // do nothing but wait 
      MPI_Barrier(faceComm);

    } else {
      
      /* 
       * Create NetCDF file
       */
      err = ncmpi_create(faceComm, ncFilenameFull.c_str(), 
			 ncCreationMode,
			 MPI_INFO_NULL, &ncFileId);
      /*
       * Define dimensions
       */
      int gsizes[3];
      if (faceDir==IX) {
	
	gsizes[2] = 1;
	gsizes[1] = my*ny+2*ghostWidth;
	gsizes[0] = mz*nz+2*ghostWidth;

      } else if (faceDir==IY) {

	gsizes[2] = mx*nx+2*ghostWidth;
	gsizes[1] = 1;
	gsizes[0] = mz*nz+2*ghostWidth;
	
      } else { // faceDir == IZ
	
	gsizes[2] = mx*nx+2*ghostWidth;
	gsizes[1] = my*ny+2*ghostWidth;
	gsizes[0] = 1;

      }
      
      err = ncmpi_def_dim(ncFileId, "x", gsizes[0], &dimIds[0]);
      PNETCDF_HANDLE_ERROR;
      
      err = ncmpi_def_dim(ncFileId, "y", gsizes[1], &dimIds[1]);
      PNETCDF_HANDLE_ERROR;
      
      err = ncmpi_def_dim(ncFileId, "z", gsizes[2], &dimIds[2]);
      PNETCDF_HANDLE_ERROR;
      
      /* 
       * Define variables
       */
      nc_type ncDataType;
      MPI_Datatype mpiDataType;
      
      if (sizeof(real_t) == sizeof(float)) {
	ncDataType  = NC_FLOAT;
	mpiDataType = MPI_FLOAT;
      } else {
	ncDataType  = NC_DOUBLE;
	mpiDataType = MPI_DOUBLE;
      }
      
      err = ncmpi_def_var(ncFileId, "rho", ncDataType, 3, dimIds, &varIds[ID]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "E", ncDataType, 3, dimIds, &varIds[IP]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vx", ncDataType, 3, dimIds, &varIds[IU]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vy", ncDataType, 3, dimIds, &varIds[IV]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vz", ncDataType, 3, dimIds, &varIds[IW]);
      PNETCDF_HANDLE_ERROR;
      
      if (mhdEnabled) {
	err = ncmpi_def_var(ncFileId, "Bx", ncDataType, 3, dimIds, &varIds[IA]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "By", ncDataType, 3, dimIds, &varIds[IB]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "Bz", ncDataType, 3, dimIds, &varIds[IC]);
	PNETCDF_HANDLE_ERROR;
      }
      
      /*
       * global attributes
       */
      // did we use CDF-2 or CDF-5
      {
	int useCDF5_int = useCDF5 ? 1 : 0;
	err = ncmpi_put_att_int(ncFileId, NC_GLOBAL, "CDF-5 mode", NC_INT, 1, &useCDF5_int);
	PNETCDF_HANDLE_ERROR;
      }
      
      // write time step number
      err = ncmpi_put_att_int(ncFileId, NC_GLOBAL, "time step", NC_INT, 1, &nStep);
      PNETCDF_HANDLE_ERROR;
      
      // write total time
      {
	double timeValue = (double) totalTime;
	err = ncmpi_put_att_double(ncFileId, NC_GLOBAL, "total time", NC_DOUBLE, 1, &timeValue);
	PNETCDF_HANDLE_ERROR;
      }
      
      /* 
       * exit the define mode 
       */
      err = ncmpi_enddef(ncFileId);
      PNETCDF_HANDLE_ERROR;
      
      /* 
       * Get all the MPI_IO hints used
       */
      err = ncmpi_get_file_info(ncFileId, &mpi_info_used);
      PNETCDF_HANDLE_ERROR;
      
      /*
       * Write heavy data (take care of row-major / column major format !)
       */
      if (faceDir == IX) {

	counts[IZ] = 1;
	counts[IY] = ny;
	counts[IX] = nz;
	
	starts[IZ] = 0;
	starts[IY] = myMpiPos[IY]*ny;
	starts[IX] = myMpiPos[IZ]*nz;
	
	// take care of borders along Y
	if (myMpiPos[IY]==my-1) {
	  counts[IY] += 2*ghostWidth;
	}
	// take care of borders along Z
	if (myMpiPos[IZ]==mz-1) {
	  counts[IX] += 2*ghostWidth;
	}

      } else if (faceDir == IY) {

	counts[IZ] = nx;
	counts[IY] = 1;
	counts[IX] = nz;
	
	starts[IZ] = myMpiPos[IX]*nx;
	starts[IY] = 0;
	starts[IX] = myMpiPos[IZ]*nz;
	
	// take care of borders along X
	if (myMpiPos[IX]==mx-1) {
	  counts[IZ] += 2*ghostWidth;
	}
	// take care of borders along Z
	if (myMpiPos[IZ]==mz-1) {
	  counts[IX] += 2*ghostWidth;
	}

      } else if (faceDir == IZ) {

	counts[IZ] = nx;
	counts[IY] = ny;
	counts[IX] = 1;
	
	starts[IZ] = myMpiPos[IX]*nx;
	starts[IY] = myMpiPos[IY]*ny;
	starts[IX] = 0;
	
	// take care of borders along X
	if (myMpiPos[IX]==mx-1) {
	  counts[IZ] += 2*ghostWidth;
	}
	// take care of borders along Y
	if (myMpiPos[IY]==my-1) {
	  counts[IY] += 2*ghostWidth;
	}

      }
      
      int nItems = counts[IX]*counts[IY]*counts[IZ];

      // data need to allocated and copied from U
    
      real_t* data;

      data = (real_t *) malloc(nItems*sizeof(real_t));
      
      int iStop=nx, jStop=ny, kStop=nz;

      if (myMpiPos[IX]== mx-1) iStop=nx+2*ghostWidth;
      if (myMpiPos[IY]== my-1) jStop=ny+2*ghostWidth;
      if (myMpiPos[IZ]== mz-1) kStop=nz+2*ghostWidth;

      if (faceDir == IX) iStop=1;
      if (faceDir == IY) jStop=1;
      if (faceDir == IZ) kStop=1;

      for (int iVar=0; iVar<nbVar; iVar++) {
	
	// copy needed data into data !
	{ // THREE_D
	  
	  int dI=0;
	  for (int k= 0; k < kStop; k++)
	    for (int j= 0; j < jStop; j++)
	      for(int i = 0; i < iStop; i++) {
		data[dI] = U(i,j,k,iVar);
		dI++;
	      }
	  
	  err = ncmpi_put_vara_all(ncFileId, varIds[iVar], starts, counts, data, nItems, mpiDataType);
	  PNETCDF_HANDLE_ERROR;
	
	} // THREE_D
	
      } // end for iVar

      free(data);

      /* 
       * close the file 
       */
      err = ncmpi_close(ncFileId);
      PNETCDF_HANDLE_ERROR;
      
    } // end faceColor == 0
    
    // make them all reach this point
    MPI_Barrier(communicator->getComm());

#else

    if (myRank==0) std::cerr << "You must enabled Pnetcdf to dump face data..." << std::endl;

#endif // USE_PNETCDF

  } // HydroRunBaseMpi::outputFacesPnetcdf

  // =======================================================
  // =======================================================
  /**
   * load data from a HDF5 file (previously dumped with outputHdf5).
   * Data are computation results (conservative variables)
   * in HDF5 format.
   *
   * One difference with the serial version, is that all MPI process
   * read the same file and extract its corresponding sub-domain.
   *
   * \sa outputHdf5 this routine performs output in HDF5 file
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
   *
   */
  int HydroRunBaseMpi::inputHdf5(HostArray<real_t> &U, 
				 const std::string filename,
				 bool halfResolution)
  {
    
#ifdef USE_HDF5_PARALLEL
    bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);
    bool allghostIncluded = configMap.getBool("output","allghostIncluded",false);

    // verbose log ?
    bool hdf5_verbose = configMap.getBool("run","hdf5_verbose",false);

    // time measurement variables
    double read_timing, max_read_timing, read_bw;
    MPI_Offset read_size, sum_read_size;

    // sizes to read
    int nx_r,  ny_r,  nz_r;  // logical sizes / per sub-domain
    int nx_rg, ny_rg, nz_rg; // sizes with ghost zones included / per sub-domain

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

    // get MPI coords corresponding to MPI rank iPiece
    int coords[3];
    if (dimType == TWO_D) {
      communicator->getCoords(myRank,2,coords);
    } else {
      communicator->getCoords(myRank,3,coords);
    }

    herr_t status;
    (void) status;
    hid_t  dataset_id;

    // TODO
    // here put some cross-check code
    // read geometry (nx,ny,nz) just to be sure to read the same values 
    // as in the current simulations
    // END TODO
    
    /*
     * Create the data space for the dataset in memory and in file.
     */
    hsize_t  dims_file[3];
    hsize_t  dims_memory[3];
    hsize_t  dims_chunk[3];
    hid_t dataspace_memory;
    //hid_t dataspace_chunk;
    hid_t dataspace_file;

    if (allghostIncluded) {
      
      if (dimType == TWO_D) {
	
	dims_file[0] = my*ny_rg;
	dims_file[1] = mx*nx_rg;
	dims_memory[0] = ny_rg; 
	dims_memory[1] = nx_rg;
	dims_chunk[0] = ny_rg;
	dims_chunk[1] = nx_rg;
      	dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

      } else { // THREE_D

	dims_file[0] = mz*nz_rg;
	dims_file[1] = my*ny_rg;
	dims_file[2] = mx*nx_rg;
	dims_memory[0] = nz_rg; 
	dims_memory[1] = ny_rg;
	dims_memory[2] = nx_rg;
	dims_chunk[0] = nz_rg;
	dims_chunk[1] = ny_rg;
	dims_chunk[2] = nx_rg;
	dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(3, dims_file  , NULL);

      }

    } else if (ghostIncluded) {

      if (dimType == TWO_D) {

	dims_file[0] = ny_r*my+2*ghostWidth;
	dims_file[1] = nx_r*mx+2*ghostWidth;
	dims_memory[0] = ny_rg;
	dims_memory[1] = nx_rg;
	dims_chunk[0] = ny_rg;
	dims_chunk[1] = nx_rg;
	dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

      } else { // THREE_D

	dims_file[0] = nz_r*mz+2*ghostWidth;
	dims_file[1] = ny_r*my+2*ghostWidth;
	dims_file[2] = nx_r*mx+2*ghostWidth;
	dims_memory[0] = nz_rg;
	dims_memory[1] = ny_rg;
	dims_memory[2] = nx_rg;
	dims_chunk[0] = nz_rg;
	dims_chunk[1] = ny_rg;
	dims_chunk[2] = nx_rg;
	dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(3, dims_file  , NULL);

      }

    } else { // no ghost zones

      if (dimType == TWO_D) {

	dims_file[0] = ny_r*my;
	dims_file[1] = nx_r*mx;

	dims_memory[0] = ny_rg;
	dims_memory[1] = nx_rg;

	dims_chunk[0] = ny_r;
	dims_chunk[1] = nx_r;

	dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

      } else {

	dims_file[0] = nz_r*mz;
	dims_file[1] = ny_r*my;
	dims_file[2] = nx_r*mx;

	dims_memory[0] = nz_rg;
	dims_memory[1] = ny_rg;
	dims_memory[2] = nx_rg;

	dims_chunk[0] = nz_r;
	dims_chunk[1] = ny_r;
	dims_chunk[2] = nx_r;

	dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	dataspace_file   = H5Screate_simple(3, dims_file  , NULL);

      }

    } // end ghostIncluded / allghostIncluded 

    // set datatype
    hid_t dataType;
    if (sizeof(real_t) == sizeof(float))
      dataType = H5T_NATIVE_FLOAT;
    else
      dataType = H5T_NATIVE_DOUBLE;
    
    /*
     * Memory space hyperslab :
     * select data with or without ghost zones
     */
    if (ghostIncluded or allghostIncluded) {

      if (dimType == TWO_D) {
	hsize_t  start[2] = { 0, 0 }; // ghost zone included
	hsize_t stride[2] = { 1, 1 };
	hsize_t  count[2] = { 1, 1 };
	hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = { 0, 0, 0 }; // ghost zone included
	hsize_t stride[3] = { 1, 1, 1 };
	hsize_t  count[3] = { 1, 1, 1 };
	hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
      }

    } else { // no ghost zones

      if (dimType == TWO_D) {
	hsize_t  start[2] = { (hsize_t) ghostWidth,  (hsize_t) ghostWidth }; // ghost zone width
	hsize_t stride[2] = { 1,  1 };
	hsize_t  count[2] = { 1,  1 };
	hsize_t  block[2] = {(hsize_t) ny_r, (hsize_t) nx_r}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = { (hsize_t) ghostWidth,  (hsize_t) ghostWidth, (hsize_t) ghostWidth }; // ghost zone width
	hsize_t stride[3] = { 1,  1,  1 };
	hsize_t  count[3] = { 1,  1,  1 };
	hsize_t  block[3] = {(hsize_t) nz_r, (hsize_t) ny_r, (hsize_t) nx_r }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
      }
    
    } // end ghostIncluded or allghostIncluded
    
    /*
     * File space hyperslab :
     * select where we want to read our own piece of the global data
     * according to MPI rank.
     */
    if (allghostIncluded) {

      if (dimType == TWO_D) {
	
	hsize_t  start[2] = { coords[1]*dims_chunk[0], coords[0]*dims_chunk[1]};
	hsize_t stride[2] = { 1,  1 };
	hsize_t  count[2] = { 1,  1 };
	hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	
      } else { // THREE_D
	
	hsize_t  start[3] = { coords[2]*dims_chunk[0], coords[1]*dims_chunk[1], coords[0]*dims_chunk[2]};
	hsize_t stride[3] = { 1,  1,  1 };
	hsize_t  count[3] = { 1,  1,  1 };
	hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	
      }

    } else if (ghostIncluded) {

      // global offsets
      int gOffsetStartX, gOffsetStartY, gOffsetStartZ;

      if (dimType == TWO_D) {
	gOffsetStartY  = coords[1]*ny_r;
	gOffsetStartX  = coords[0]*nx_r;

	hsize_t  start[2] = { (hsize_t) gOffsetStartY, (hsize_t) gOffsetStartX };
	hsize_t stride[2] = { 1,  1};
	hsize_t  count[2] = { 1,  1};
	hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	
      } else { // THREE_D
	
	gOffsetStartZ  = coords[2]*nz_r;
	gOffsetStartY  = coords[1]*ny_r;
	gOffsetStartX  = coords[0]*nx_r;

	hsize_t  start[3] = { (hsize_t) gOffsetStartZ, (hsize_t) gOffsetStartY, (hsize_t) gOffsetStartX };
	hsize_t stride[3] = { 1,  1,  1};
	hsize_t  count[3] = { 1,  1,  1};
	hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	
      }

    } else { // no ghost zones
      
      if (dimType == TWO_D) {
	
	hsize_t  start[2] = { coords[1]*dims_chunk[0], coords[0]*dims_chunk[1]};
	hsize_t stride[2] = { 1,  1};
	hsize_t  count[2] = { 1,  1};
	hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	
      } else { // THREE_D
	
	hsize_t  start[3] = { coords[2]*dims_chunk[0], coords[1]*dims_chunk[1], coords[0]*dims_chunk[2]};
	hsize_t stride[3] = { 1,  1,  1};
	hsize_t  count[3] = { 1,  1,  1};
	hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
      }
    } // end ghostIncluded / allghostIncluded

    // measure time ??
    if (hdf5_verbose) {
      MPI_Barrier(communicator->getComm());
      read_timing = MPI_Wtime();
    }

    /*
     * Try parallel read HDF5 file.
     */
    
    /* Set up MPIO file access property lists */
    //MPI_Info mpi_info   = MPI_INFO_NULL;
    hid_t access_plist  = H5Pcreate(H5P_FILE_ACCESS);
    status = H5Pset_fapl_mpio(access_plist, communicator->getComm(), MPI_INFO_NULL);

    /* Open the file */
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, access_plist);
    
    /*
     *
     * read heavy data from HDF5 file
     *
     */
    real_t* data;
    hid_t propList_create_id = H5Pcreate(H5P_DATASET_CREATE);
    if (dimType == TWO_D)
      H5Pset_chunk(propList_create_id, 2, dims_chunk);
    else
      H5Pset_chunk(propList_create_id, 3, dims_chunk);

    // please note that HDF5 parallel I/O does not support yet filters
    // so we can't use here H5P_deflate to perform compression !!!
    // Weak solution : call h5repack after the file is created
    // (performance of that has not been tested)

    hid_t propList_xfer_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(propList_xfer_id, H5FD_MPIO_COLLECTIVE);

    /* read density */
    dataset_id = H5Dopen2(file_id, "/density", H5P_DEFAULT);

    if (dimType == TWO_D)
      data = &(U(0,0,ID));
    else
      data = &(U(0,0,0,ID));

    status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		     propList_xfer_id, data);
    H5Dclose(dataset_id);

    // read energy
    dataset_id = H5Dopen2(file_id, "/energy", H5P_DEFAULT);

    if (dimType == TWO_D)
      data = &(U(0,0,IP));
    else
      data = &(U(0,0,0,IP));

    status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		     propList_xfer_id, data);
    H5Dclose(dataset_id);

    // read momentum X
    dataset_id = H5Dopen2(file_id, "/momentum_x", H5P_DEFAULT);

    if (dimType == TWO_D)
      data = &(U(0,0,IU));
    else
      data = &(U(0,0,0,IU));

    status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		     propList_xfer_id, data);
    H5Dclose(dataset_id);

    // read momentum Y
    dataset_id = H5Dopen2(file_id, "/momentum_y", H5P_DEFAULT);

    if (dimType == TWO_D)
      data = &(U(0,0,IV));
    else
      data = &(U(0,0,0,IV));

    status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		     propList_xfer_id, data);
    H5Dclose(dataset_id);

    // read momentum Z (only if hydro 3D)
    if (dimType == THREE_D and !mhdEnabled) {
      dataset_id = H5Dopen2(file_id, "/momentum_z", H5P_DEFAULT);

      data = &(U(0,0,0,IW));
      
      status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		       propList_xfer_id, data);
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
		       propList_xfer_id, data);
      H5Dclose(dataset_id);

      // read magnetic field components X
      dataset_id = H5Dopen2(file_id, "/magnetic_field_x", H5P_DEFAULT);
      
      if (dimType == TWO_D)
	data = &(U(0,0,IA));
      else
	data = &(U(0,0,0,IA));
      
      status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		       propList_xfer_id, data);
      H5Dclose(dataset_id);

      // read magnetic field components Y
      dataset_id = H5Dopen2(file_id, "/magnetic_field_y", H5P_DEFAULT);
      
      if (dimType == TWO_D)
	data = &(U(0,0,IB));
      else
	data = &(U(0,0,0,IB));
      
      status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		       propList_xfer_id, data);
      H5Dclose(dataset_id);

      // read magnetic field components Z
      dataset_id = H5Dopen2(file_id, "/magnetic_field_z", H5P_DEFAULT);
      
      if (dimType == TWO_D)
	data = &(U(0,0,IC));
      else
	data = &(U(0,0,0,IC));
      
      status = H5Dread(dataset_id, dataType, dataspace_memory, dataspace_file,
		       propList_xfer_id, data);
      H5Dclose(dataset_id);

    } // end mhdEnabled

    /* read time step number (all MPI process need to get it) */
    int timeStep;
    hid_t group_id  = H5Gopen2(file_id, "/", H5P_DEFAULT);
    hid_t attr_id;

    attr_id         = H5Aopen(group_id, "time step", H5P_DEFAULT);
    status          = H5Aread(attr_id, H5T_NATIVE_INT, &timeStep);
    status          = H5Aclose(attr_id);

    // read global time
    double timeValue;
    attr_id         = H5Aopen(group_id, "total time", H5P_DEFAULT);
    status          = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &timeValue);
    status          = H5Aclose(attr_id);
    totalTime = (real_t) timeValue;

    status          = H5Gclose(group_id);

    /* release resources */
    H5Pclose (propList_create_id);
    H5Pclose (access_plist);
    H5Fclose (file_id);

    /*
     * verbose log about memory bandwidth
     */
    if (hdf5_verbose) {

      read_timing = MPI_Wtime() - read_timing;
      
      //read_size = nbVar * U.section() * sizeof(real_t);
      read_size = U.sizeBytes();
      sum_read_size = read_size *  nProcs;
      
      MPI_Reduce(&read_timing, &max_read_timing, 1, MPI_DOUBLE, MPI_MAX, 0, communicator->getComm());

      if (myRank==0) {
	printf("########################################################\n");
	printf("################### HDF5 bandwidth #####################\n");
	printf("########################################################\n");
	printf("Local  array size %d x %d x %d reals(%zu bytes), read size = %.2f MB\n",
	       nx+2*ghostWidth,
	       ny+2*ghostWidth,
	       nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*read_size/1048576.0);
	sum_read_size /= 1048576.0;
	printf("Global array size %d x %d x %d reals(%zu bytes), read size = %.2f GB\n",
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*sum_read_size/1024);
	
	read_bw = sum_read_size/max_read_timing;
	printf(" procs    Global array size  exec(sec)  read(MB/s)\n");
	printf("-------  ------------------  ---------  -----------\n");
	printf(" %4d    %4d x %4d x %4d %8.2f  %10.2f\n", nProcs,
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       max_read_timing, read_bw);
	printf("########################################################\n");

      } // end (myRank==0)

    } // hdf5_verbose

    return timeStep;

#else 

    if (myRank == 0) {
      std::cerr << "Parallel HDF5 library is not available !" << std::endl;
      std::cerr << "You can't load a data file for restarting the simulation run !!!" << std::endl;
      std::cerr << "Please install Parallel HDF5 library !!!" << std::endl;
    }
    return -1;
    
#endif // USE_HDF5_PARALLEL

  } // HydroRunBaseMpi::inputHdf5

  // =======================================================
  // =======================================================
  /**
   * load data from a netcdf file (using Parallel NetCDF library)
   * Data are computation results (conservative variables)
   * in CDF-2 or CDF-5 format.
   *
   * One difference with the serial version, is that all MPI process
   * read the same file and extract its corresponding sub-domain.
   *
   * \sa outputPnetcdf this routine performs output in a netcdf file
   *
   * \note Take care that HostArray use column-format ordering,
   * whereas C-language and so C API of Parallel uses raw-major ordering
   * !!! We need to invert dimensions.
   *
   * \param[out] U A reference to a hydro simulation HostArray
   * \param[in]  filename Name of the input netcdf file
   * \param[in]  halfResolution boolean, triggers reading half resolution data
   *
   * If library Pnetcdf is not available, do nothing, just print an error message.
   *
   */
  int HydroRunBaseMpi::inputPnetcdf(HostArray<real_t> &U, 
				    const std::string filename,
				    bool halfResolution)
  {
    
#ifdef USE_PNETCDF
    // netcdf file id
    int ncFileId;
    int err;

    // file creation mode
    int ncOpenMode = NC_NOWRITE;

    // verbose log ?
    bool pnetcdf_verbose = configMap.getBool("run","pnetcdf_verbose",false);

    int varIds[nbVar];
    MPI_Offset starts[3], counts[3], read_size, sum_read_size;
    MPI_Info mpi_info_used;
    //char str[512];

    // time measurement variables
    double read_timing, max_read_timing, read_bw;

    // sizes to read
    int nx_r,  ny_r,  nz_r;  // logical sizes / per sub-domain
    int nx_rg, ny_rg, nz_rg; // sizes with ghost zones included / per sub-domain

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
     * get MPI coords corresponding to MPI rank iPiece
     */
    int coords[3];
    if (dimType == TWO_D) {
      communicator->getCoords(myRank,2,coords);
    } else {
      communicator->getCoords(myRank,3,coords);
    }

    // measure time ??
    if (pnetcdf_verbose) {
      MPI_Barrier(communicator->getComm());
      read_timing = MPI_Wtime();
    }

    /* 
     * Open NetCDF file
     */
    err = ncmpi_open(communicator->getComm(), filename.c_str(), 
		     ncOpenMode,
		     MPI_INFO_NULL, &ncFileId);
    if (err != NC_NOERR) {
      printf("Error: ncmpi_open() file %s (%s)\n",filename.c_str(),ncmpi_strerror(err));
      MPI_Abort(communicator->getComm(), -1);
      exit(1);
    }
    
    /*
     * Query NetCDF mode
     */
    if (pnetcdf_verbose) {
      int NC_mode;
      err = ncmpi_inq_version(ncFileId, &NC_mode);
      if (myRank==0) {
	if (NC_mode == NC_64BIT_DATA)
	  std::cout << "Pnetcdf Input mode : NC_64BIT_DATA (CDF-5)\n";
	else if (NC_mode == NC_64BIT_OFFSET)
	  std::cout << "Pnetcdf Input mode : NC_64BIT_OFFSET (CDF-2)\n";
	else
	  std::cout << "Pnetcdf Input mode : unknown\n";
      }
    }

    /*
     * Query timeStep (global attribute)
     */
    int timeStep;
    {
      nc_type timeStep_type;
      MPI_Offset timeStep_len;
      err = ncmpi_inq_att (ncFileId, NC_GLOBAL, "time step", 
			   &timeStep_type, 
			   &timeStep_len);
      PNETCDF_HANDLE_ERROR;

      /* read timeStep */
      err = ncmpi_get_att_int(ncFileId, NC_GLOBAL, "time step", &timeStep);
      PNETCDF_HANDLE_ERROR;

      if (pnetcdf_verbose and myRank==0)
	std::cout << "input PnetCDF time step: " << timeStep << std::endl;

    }

    /*
     * Query total time (global attribute)
     */
    {
      nc_type totalTime_type;
      MPI_Offset totalTime_len;
      err = ncmpi_inq_att (ncFileId, NC_GLOBAL, "total time", 
			   &totalTime_type, 
			   &totalTime_len);
      PNETCDF_HANDLE_ERROR;

      /* read total time */
      double timeValue;
      err = ncmpi_get_att_double(ncFileId, NC_GLOBAL, "total time", &timeValue);
      PNETCDF_HANDLE_ERROR;

      totalTime = (real_t) timeValue;
    }

    /*
     * Query information about variables
     */
    {
      int ndims, nvars, ngatts, unlimited;
      err = ncmpi_inq(ncFileId, &ndims, &nvars, &ngatts, &unlimited);
      PNETCDF_HANDLE_ERROR;
      
      // check ndims
      if ( (dimType == TWO_D   and ndims != 2) or
	   (dimType == THREE_D and ndims != 3) ) {
	std::cerr << "inputPnetcdf error; wrong number of dimensions in file " 
		  << filename << std::endl;
	MPI_Abort(communicator->getComm(), -1);
	exit(1);
      }

      // get varIds
      if (dimType == TWO_D) {
	err = ncmpi_inq_varid(ncFileId, "rho", &varIds[ID]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_inq_varid(ncFileId, "E", &varIds[IP]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_inq_varid(ncFileId, "rho_vx", &varIds[IU]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_inq_varid(ncFileId, "rho_vy", &varIds[IV]);
	PNETCDF_HANDLE_ERROR;
	
	if (mhdEnabled) {
	  err = ncmpi_inq_varid(ncFileId, "rho_vz", &varIds[IW]);
	  PNETCDF_HANDLE_ERROR;
	  err = ncmpi_inq_varid(ncFileId, "Bx", &varIds[IA]);
	  PNETCDF_HANDLE_ERROR;
	  err = ncmpi_inq_varid(ncFileId, "By", &varIds[IB]);
	  PNETCDF_HANDLE_ERROR;
	  err = ncmpi_inq_varid(ncFileId, "Bz", &varIds[IC]);
	  PNETCDF_HANDLE_ERROR;	  
	}

      } else { // THREE_D

 	err = ncmpi_inq_varid(ncFileId, "rho", &varIds[ID]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_inq_varid(ncFileId, "E", &varIds[IP]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_inq_varid(ncFileId, "rho_vx", &varIds[IU]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_inq_varid(ncFileId, "rho_vy", &varIds[IV]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_inq_varid(ncFileId, "rho_vz", &varIds[IW]);
	PNETCDF_HANDLE_ERROR;
     
	if (mhdEnabled) {
	  err = ncmpi_inq_varid(ncFileId, "Bx", &varIds[IA]);
	  PNETCDF_HANDLE_ERROR;
	  err = ncmpi_inq_varid(ncFileId, "By", &varIds[IB]);
	  PNETCDF_HANDLE_ERROR;
	  err = ncmpi_inq_varid(ncFileId, "Bz", &varIds[IC]);
	  PNETCDF_HANDLE_ERROR;	
	}

      } // end query varIds

    } // end query information

    /* 
     * Define expected data types (no conversion done here)
     */
    //nc_type ncDataType;
    MPI_Datatype mpiDataType;
    
    if (sizeof(real_t) == sizeof(float)) {
      //ncDataType  = NC_FLOAT;
      mpiDataType = MPI_FLOAT;
    } else {
      //ncDataType  = NC_DOUBLE;
      mpiDataType = MPI_DOUBLE;
    }

    /* 
     * Get all the MPI_IO hints used (just in case, we want to print it after 
     * reading data...
     */
    err = ncmpi_get_file_info(ncFileId, &mpi_info_used);
    PNETCDF_HANDLE_ERROR;

    /*
     * Read heavy data (take care of row-major / column major format !)
     */
    // use overlapping domains
    if (dimType == TWO_D) {
      
      counts[IY] = nx_rg;
      counts[IX] = ny_rg;
      
      starts[IY] = coords[IX]*nx_r;
      starts[IX] = coords[IY]*ny_r;
      
    } else { // THREE_D
      
      counts[IZ] = nx_rg;
      counts[IY] = ny_rg;
      counts[IX] = nz_rg;
      
      starts[IZ] = coords[IX]*nx_r;
      starts[IY] = coords[IY]*ny_r;
      starts[IX] = coords[IZ]*nz_r;
      
    } // end THREE_D

    int nItems = counts[IX]*counts[IY];
    if (dimType==THREE_D)
      nItems *= counts[IZ];

    /*
     * Actual reading
     */
    for (int iVar=0; iVar<nbVar; iVar++) {
      real_t* data;
      if (dimType==TWO_D) {
	data = &(U(0,0,iVar));
      } else {
	data = &(U(0,0,0,iVar));
      }
      err = ncmpi_get_vara_all(ncFileId, varIds[iVar], 
			       starts, counts, data, nItems, mpiDataType);
      PNETCDF_HANDLE_ERROR;
    } // end for loop reading heavy data
    
    /* 
     * close the file 
     */
    err = ncmpi_close(ncFileId);
    PNETCDF_HANDLE_ERROR;

    /*
     * verbose log about memory bandwidth
     */
    if (pnetcdf_verbose) {

      read_timing = MPI_Wtime() - read_timing;
      
      //read_size = nbVar * U.section() * sizeof(real_t);
      read_size = U.sizeBytes();
      sum_read_size = read_size *  nProcs;
      
      MPI_Reduce(&read_timing, &max_read_timing, 1, MPI_DOUBLE, MPI_MAX, 0, communicator->getComm());

      if (myRank==0) {
	printf("########################################################\n");
	printf("############## Parallel-netCDF bandwidth ###############\n");
	printf("########################################################\n");
	printf("Local  array size %d x %d x %d reals(%zu bytes), read size = %.2f MB\n",
	       nx+2*ghostWidth,
	       ny+2*ghostWidth,
	       nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*read_size/1048576.0);
	sum_read_size /= 1048576.0;
	printf("Global array size %d x %d x %d reals(%zu bytes), read size = %.2f GB\n",
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*sum_read_size/1024);
	
	read_bw = sum_read_size/max_read_timing;
	printf(" procs    Global array size  exec(sec)  read(MB/s)\n");
	printf("-------  ------------------  ---------  -----------\n");
	printf(" %4d    %4d x %4d x %4d %8.2f  %10.2f\n", nProcs,
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       max_read_timing, read_bw);
	printf("########################################################\n");

      } // end (myRank==0)

    } // pnetcdf_verbose
    
    /*
     * Print MPI information
     */
    bool pnetcdf_print_mpi_info = configMap.getBool("run","pnetcdf_print_mpi_info",false);
    if (pnetcdf_print_mpi_info and myRank==0) {
      
      int     flag;
      char    info_cb_nodes[64], info_cb_buffer_size[64];
      char    info_striping_factor[64], info_striping_unit[64];
      
      char undefined_char[]="undefined";

      strcpy(info_cb_nodes,        undefined_char);
      strcpy(info_cb_buffer_size,  undefined_char);
      strcpy(info_striping_factor, undefined_char);
      strcpy(info_striping_unit,   undefined_char);
      
      char cb_nodes_char[]       ="cb_nodes";
      char cb_buffer_size_char[] ="cb_buffer_size";
      char striping_factor_char[]="striping_factor";
      char striping_unit_char[]  ="striping_unit";

      MPI_Info_get(mpi_info_used, cb_nodes_char       , 64, info_cb_nodes, &flag);
      MPI_Info_get(mpi_info_used, cb_buffer_size_char , 64, info_cb_buffer_size, &flag);
      MPI_Info_get(mpi_info_used, striping_factor_char, 64, info_striping_factor, &flag);
      MPI_Info_get(mpi_info_used, striping_unit_char  , 64, info_striping_unit, &flag);
      
      printf("MPI hint: cb_nodes        = %s\n", info_cb_nodes);
      printf("MPI hint: cb_buffer_size  = %s\n", info_cb_buffer_size);
      printf("MPI hint: striping_factor = %s\n", info_striping_factor);
      printf("MPI hint: striping_unit   = %s\n", info_striping_unit);
      
    } // pnetcdf_print_mpi_info

    return timeStep;

#else
    
    if (myRank == 0) {
      std::cerr << "Pnetcdf library is not available !" << std::endl;
      std::cerr << "You can't load a data file for restarting the simulation run !!!" << std::endl;
      std::cerr << "Please install Parallel NetCDF library !!!" << std::endl;
    }
    return -1;
    
#endif // USE_PNETCDF
  } // HydroRunBaseMpi::inputPnetcdf

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
  void HydroRunBaseMpi::upscale(HostArray<real_t> &HiRes, 
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

  } // HydroRunBaseMpi::upscale

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::init_hydro_jet()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    if (dimType == TWO_D) {
    
      /* jet */
      for (int j=2; j<jsize-2; j++)
	for (int i=2; i<isize-2; i++) {
	  //int index = i+isize*j;
	  // fill density, U, V and energy sub-arrays
	  h_U(i,j,ID)=1.0f;
	  h_U(i,j,IP)=1.0f/(_gParams.gamma0-1.0f);
	  h_U(i,j,IU)=0.0f;
	  h_U(i,j,IV)=0.0f;
	}
    
      /* corner grid (not really needed except for Kurganov-Tadmor) */
      for (int nVar=0; nVar<nbVar; ++nVar) {

	for (int i=0; i<2; ++i)
	  for (int j=0; j<2; ++j) {
	    h_U(     i,     j,nVar) = h_U(   2,   2,nVar);
	    h_U(nx+2+i,     j,nVar) = h_U(nx+1,   2,nVar);
	    h_U(     i,ny+2+j,nVar) = h_U(   2,ny+1,nVar);
	    h_U(nx+2+i,ny+2+j,nVar) = h_U(nx+1,ny+1,nVar);
	  } // end for loop over i,j

      } // end loop over nVar

    } else { // THREE_D

      /* jet */
      for (int k=2; k<ksize-2; k++)
	for (int j=2; j<jsize-2; j++)
	  for (int i=2; i<isize-2; i++) {
	    // fill density, U, V, W and energy sub-arrays
	    h_U(i,j,k,ID)=1.0f;
	    h_U(i,j,k,IP)=1.0f/(_gParams.gamma0-1.0f);
	    h_U(i,j,k,IU)=0.0f;
	    h_U(i,j,k,IV)=0.0f;
	    h_U(i,j,k,IW)=0.0f;
	  }   

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

  } // HydroRunBaseMpi::init_hydro_jet

    // =======================================================
    // =======================================================
    /*
     * see
     * http://www.astro.princeton.edu/~jstone/tests/implode/Implode.html
     * for a description of such initial conditions
     */
  void HydroRunBaseMpi::init_hydro_implode()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    // amplitude of a random noise added to the vector potential
    const double amp       = configMap.getFloat("implode","amp",0.0);  
    int          seed      = configMap.getInteger("implode","seed",1);

    // initialize random number generator
    seed *= myRank;
    srand48(seed);

    if (dimType == TWO_D) {
  
      /* discontinuity line along diagonal */
      for (int j=2; j<jsize-2; j++)
	for (int i=2; i<isize-2; i++) {
	  // compute global indexes
	  int ii = i + nx*myMpiPos[0];
	  int jj = j + ny*myMpiPos[1];
	  if (((float)ii/nx/mx+(float)jj/ny/my)>1) {
	    h_U(i,j,ID)=1.0f + amp*(drand48()-0.5);
	    h_U(i,j,IP)=1.0f/(_gParams.gamma0-1.0f);
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	  } else {
	    h_U(i,j,ID)=0.125f;
	    h_U(i,j,IP)=0.14f/(_gParams.gamma0-1.0f);      
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	  }
	}
    
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
  
    } else { // THREE_D

      /* discontinuity line along diagonal */
      for (int k=2; k<ksize-2; k++)
	for (int j=2; j<jsize-2; j++)
	  for (int i=2; i<isize-2; i++) {
	    // compute global indexes
	    int ii = i + nx*myMpiPos[0];
	    int jj = j + ny*myMpiPos[1];
	    int kk = k + nz*myMpiPos[2];	    
	    if (((float)ii/nx/mx+(float)jj/ny/my+(float)kk/nz/mz)>1) {
	      h_U(i,j,k,ID)=1.0f + amp*(drand48()-0.5);
	      h_U(i,j,k,IP)=1.0f/(_gParams.gamma0-1.0f);
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	    } else {
	      h_U(i,j,k,ID)=0.125f;
	      h_U(i,j,k,IP)=0.14f/(_gParams.gamma0-1.0f);      
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	    }
	  }

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

  } // HydroRunBaseMpi::init_hydro_implode

  // =======================================================
  // =======================================================
  /*
   * see
   * http://www.astro.princeton.edu/~jstone/tests/blast/blast.html
   * for a description of such initial conditions
   */
  void HydroRunBaseMpi::init_hydro_blast()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    /* get radius */
    int radius          = configMap.getInteger("blast", "radius", nx*mx/4); // global index
    int center_x        = configMap.getInteger("blast", "center_x",nx*mx/2); // global index
    int center_y        = configMap.getInteger("blast", "center_y",ny*my/2); // global index
    int center_z        = configMap.getInteger("blast", "center_z",nz*mz/2); // global index
    real_t density_in   = configMap.getFloat("blast", "density_in", 1.0);
    real_t density_out  = configMap.getFloat("blast", "density_out", 1.0);
    real_t pressure_in  = configMap.getFloat("blast", "pressure_in", 10.0);
    real_t pressure_out = configMap.getFloat("blast", "pressure_out", 0.1);

    // compute square radius
    radius *= radius;

    /* spherical blast wave test */
    if (dimType == TWO_D) {
    
      for (int j=2; j<jsize-2; j++)
	for (int i=2; i<isize-2; i++) {
	  
	  // compute global indexes
	  int ii = i + nx*myMpiPos[0];
	  int jj = j + ny*myMpiPos[1];
	  
	  real_t d2 = 
	    (ii-center_x)*(ii-center_x)+
	    (jj-center_y)*(jj-center_y);
	
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
	}
    
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
  
    } else { // THREE_D
    
      for (int k=2; k<ksize-2; k++)
	for (int j=2; j<jsize-2; j++)
	  for (int i=2; i<isize-2; i++) {
	  
	    // compute global indexes
	    int ii = i + nx*myMpiPos[0];
	    int jj = j + ny*myMpiPos[1];
	    int kk = k + ny*myMpiPos[2];
	    
	    real_t d2 = 
	      (ii-center_x)*(ii-center_x) +
	      (jj-center_y)*(jj-center_y) +
	      (kk-center_z)*(kk-center_z);
	  
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
	  }

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

  } // HydroRunBaseMpi::init_hydro_blast

  // =======================================================
  // =======================================================
  /**
   * Test of the Kelvin-Helmholtz instability.
   * See
   * http://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
   * for a description of such initial conditions
   *
   */
  void HydroRunBaseMpi::init_hydro_Kelvin_Helmholtz()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    /* initialize perturbation amplitude */
    real_t amplitude = configMap.getFloat("kelvin-helmholtz", "amplitude", 0.01);
    
    /* type of perturbation sine or random */
    bool p_sine = configMap.getFloat("kelvin-helmholtz", "perturbation_sine", false);
    bool p_rand = configMap.getFloat("kelvin-helmholtz", "perturbation_rand", true);

    /* initialize random generator (if needed) */
    int seed = configMap.getInteger("kelvin-helmholtz", "seed", 1);
    if (p_rand) {
      seed *= (myRank+1);
      srand(seed);
    }

    /* inner and outer fluid density */
    real_t rho_inner = configMap.getFloat("kelvin-helmholtz", "rho_inner", 2.0);
    real_t rho_outer = configMap.getFloat("kelvin-helmholtz", "rho_outer", 1.0);
    real_t pressure  = configMap.getFloat("kelvin_helmholtz", "pressure", 2.5);

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t &zMin = _gParams.zMin;

    real_t &xMax = _gParams.xMax;
    real_t &yMax = _gParams.yMax;
    real_t &zMax = _gParams.zMax;

    if (dimType == TWO_D) {
  
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	int      jj = j + ny*myMpiPos[1];
	real_t yPos = yMin + dy/2 + (jj-ghostWidth)*dy;
	
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  int      ii = i + nx*myMpiPos[0];
	  real_t xPos = xMin + dx/2 + (ii-ghostWidth)*dx;
	  
	  if ( yPos < yMin+0.25*(yMax-yMin) or
	       yPos > yMin+0.75*(yMax-yMin) ) {

	    h_U(i,j,ID) = rho_outer;
	    h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f);
	    h_U(i,j,IU) = rho_outer *
	      (0.5f + 
	       p_rand*amplitude  * rand()/RAND_MAX +
	       p_sine*amplitude  * sin(2*M_PI*xPos) );
	    h_U(i,j,IV) = rho_outer *
	      (0.0f + 
	       p_rand*amplitude * rand()/RAND_MAX +
	       p_sine*amplitude * sin(2*M_PI*xPos) );

	  } else {

	    h_U(i,j,ID) = rho_inner;
	    h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f);
	    h_U(i,j,IU) = rho_inner *
	      (-0.5f + 
	       p_rand*amplitude * rand()/RAND_MAX +
	       p_sine*amplitude * sin(2*M_PI*xPos) );
	    h_U(i,j,IV) = rho_inner *
	      (0.0f + 
	       p_rand*amplitude * rand()/RAND_MAX +
	       p_sine*amplitude * sin(2*M_PI*xPos) );

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
	int      kk = k + nz*myMpiPos[2];
	real_t zPos = zMin + dz/2 + (kk-ghostWidth)*dz;

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  int      jj = j + ny*myMpiPos[1];
	  real_t yPos = yMin + dy/2 + (jj-ghostWidth)*dy;

	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    int      ii = i + nx*myMpiPos[0];
	    real_t xPos = xMin + dx/2 + (ii-ghostWidth)*dx;

	    if ( zPos < zMin+0.25*(zMax-zMin) or
		 zPos > zMin+0.75*(zMax-zMin) ) {

	      h_U(i,j,k,ID) = rho_outer;
	      h_U(i,j,k,IP) = pressure/(_gParams.gamma0-1.0f);
	      h_U(i,j,k,IU) = rho_outer *
		(0.5f + 
		 p_rand*amplitude * rand()/RAND_MAX +
		 p_sine*amplitude * sin(2*M_PI*xPos) );
	      h_U(i,j,k,IV) = rho_outer *
		(0.0f + 
		 p_rand*amplitude * rand()/RAND_MAX +
		 p_sine*amplitude * sin(2*M_PI*xPos) );
	      h_U(i,j,k,IW) = rho_outer *
		(0.0f + 
		 p_rand * amplitude*rand()/RAND_MAX +
		 p_sine*amplitude * sin(2*M_PI*xPos) );

	    } else {

	      h_U(i,j,k,ID) = rho_inner;
	      h_U(i,j,k,IP) = pressure/(_gParams.gamma0-1.0f);
	      h_U(i,j,k,IU) = rho_inner *
		(-0.5f + 
		 p_rand*amplitude * rand()/RAND_MAX +
		 p_sine*amplitude * sin(2*M_PI*xPos) );
	      h_U(i,j,k,IV) = rho_inner *
		(0.0f + 
		 p_rand*amplitude * rand()/RAND_MAX +
		 p_sine*amplitude * sin(2*M_PI*xPos) );
	      h_U(i,j,k,IW) = rho_inner *
		(0.0f + 
		 p_rand*amplitude * rand()/RAND_MAX +
		 p_sine*amplitude * sin(2*M_PI*xPos) );
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

  } // HydroRunBaseMpi::init_hydro_Kelvin_Helmholtz

  // =======================================================
  // =======================================================
  /**
   * Test of the Rayleigh-Taylor instability.
   * See
   * http://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
   * for a description of such initial conditions
   */
  void HydroRunBaseMpi::init_hydro_Rayleigh_Taylor()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    /* initialize perturbation amplitude */
    real_t amplitude = configMap.getFloat("rayleigh-taylor", "amplitude", 0.01);
    real_t        d0 = configMap.getFloat("rayleigh-taylor", "d0", 1.0);
    real_t        d1 = configMap.getFloat("rayleigh-taylor", "d1", 2.0);

    
    bool  randomEnabled = configMap.getBool("rayleigh-taylor", "randomEnabled", false);
    int            seed = configMap.getInteger("rayleigh-taylor", "random_seed", 33);
    if (randomEnabled) {
      seed *= (myRank+1);
      srand(seed);
    }

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
	int   jj = j + ny*myMpiPos[1];
	real_t y = yMin + dy/2 + (jj-ghostWidth)*dy;
	
	for (int i=0; i<isize; i++) {
	  int   ii = i + nx*myMpiPos[0];
	  real_t x = xMin + dx/2 + (ii-ghostWidth)*dx;
	
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
	int   kk = k + nz*myMpiPos[2];
	real_t z = zMin + dz/2 + (kk-ghostWidth)*dz;

	for (int j=0; j<jsize; j++) {
	  int   jj = j + ny*myMpiPos[1];
	  real_t y = yMin + dy/2 + (jj-ghostWidth)*dy;

	    for (int i=0; i<isize; i++) {
	      int   ii = i + nx*myMpiPos[0];
	      real_t x = xMin + dx/2 + (ii-ghostWidth)*dx;
	    
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

  } // HydroRunBaseMpi::init_hydro_Rayleigh-Taylor

  // =======================================================
  // =======================================================
  /**
   * Falling bubble test.
   *
   */
  void HydroRunBaseMpi::init_hydro_falling_bubble()
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

    real_t Lx = xMax-xMin;
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
	int   jj = j + ny*myMpiPos[1];
	real_t y = yMin + dy/2 + (jj-ghostWidth)*dy;

	for (int i=0; i<isize; i++) {
	  int   ii = i + nx*myMpiPos[0];
	  real_t x = xMin + dx/2 + (ii-ghostWidth)*dx;

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
	int   kk = k + nz*myMpiPos[2];
	real_t z = zMin + dz/2 + (k-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  int   jj = j + ny*myMpiPos[1];
	  real_t y = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    int   ii = i + nx*myMpiPos[0];
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

  } // HydroRunBaseMpi::init_hydro_falling_bubble

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::init_hydro_Riemann()
  {

    // reset domain
    memset(h_U.data(),0,h_U.sizeBytes());

    // each MPI sub-block explore a different 2D Riemann problem
    int nb=riemannConfId*myRank % NB_RIEMANN_CONFIG;

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

    for( int j = 2; j < jsize-2; ++j)
      for( int i = 2; i < isize-2; ++i)
	{
	
	  if (i<(2+nx/2)) {
	    if (j<(2+ny/2)) {
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
	    if (j<(2+ny/2)) {
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

  } // HydroRunBaseMpi::init_hydro_Riemann

  // =======================================================
  // =======================================================
  /**
   *
   * This initialization routine is inspired by Enzo. 
   * See routine named turboinit by A. Kritsuk in Enzo.
   */
  void HydroRunBaseMpi::init_hydro_turbulence()
  {

    // reset domain
    memset(h_U.data(),0,h_U.sizeBytes());

    /* get initial conditions */
    real_t d0 = configMap.getFloat("turbulence", "density",  1.0);
    real_t initialDensityPerturbationAmplitude = 
      configMap.getFloat("turbulence", "initialDensityPerturbationAmplitude", 0.0);

    real_t P0 = configMap.getFloat("turbulence", "pressure", 1.0);

    int seed = configMap.getInteger("turbulence", "random_seed", 33);

    // initialize random number generator
    seed *= (myRank+1);
    srand(seed);

    if (dimType == TWO_D) {
    
      if (myRank==0) std::cerr << "Turbulence problem is not available in 2D...." << std::endl;

    } else { // THREE_D
      
      // initialize h_randomForcing
      init_randomForcing();

      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	    // fill density, U, V, W and energy sub-arrays
	    h_U(i,j,k,ID) = d0 * (1.0 + initialDensityPerturbationAmplitude *  ( (float)rand()/(float)(RAND_MAX)   - 0.5 ) );

	    // convert h_randomForce into momentum
	    h_U(i,j,k,IU) = h_U(i,j,k,ID) * h_randomForcing(i,j,k,IX);
	    h_U(i,j,k,IV) = h_U(i,j,k,ID) * h_randomForcing(i,j,k,IY);
	    h_U(i,j,k,IW) = h_U(i,j,k,ID) * h_randomForcing(i,j,k,IZ);

	    // compute total energy
	    h_U(i,j,k,IP) = P0/(_gParams.gamma0-1.0f) + 
	      0.5 * h_U(i,j,k,ID) * ( h_U(i,j,k,IU) * h_U(i,j,k,IU) +
				      h_U(i,j,k,IV) * h_U(i,j,k,IV) +
				      h_U(i,j,k,IW) * h_U(i,j,k,IW) );
	    
	  } // end for i,j,k

#ifdef __CUDACC__
      // we also need to copy
      d_randomForcing.copyFromHost(h_randomForcing);
#endif // __CUDACC__

    } // end THREE_D

  } // HydroRunBaseMpi::init_hydro_turbulence

  // =======================================================
  // =======================================================
  /**
   *
   * Initialization for turbulence run using Ornstein-Uhlenbeck forcing.
   *
   */
  void HydroRunBaseMpi::init_hydro_turbulence_Ornstein_Uhlenbeck()
  {

    // reset domain
    memset(h_U.data(),0,h_U.sizeBytes());

    /* get initial conditions */
    real_t d0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "density",  1.0);
    real_t initialDensityPerturbationAmplitude = 
      configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "initialDensityPerturbationAmplitude", 0.0);

    real_t P0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "pressure", 1.0);

    int seed = configMap.getInteger("turbulence-Ornstein-Uhlenbeck", "random_seed", 33);
    // initialize random number generator
    seed *= myRank;
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
	    h_U(i,j,k,IP) = P0/(_gParams.gamma0-ONE_F) + 
	      0.5 * h_U(i,j,k,ID) * ( h_U(i,j,k,IU) * h_U(i,j,k,IU) +
				      h_U(i,j,k,IV) * h_U(i,j,k,IV) +
				      h_U(i,j,k,IW) * h_U(i,j,k,IW) );
	    
	  } // end for i,j,k

    } // end THREE_D

  } // HydroRunBaseMpi::init_hydro_turbulence_Ornstein_Uhlenbeck

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::init_mhd_jet()
  {
    
    if (!mhdEnabled) {
      if (myRank == 0)
	std::cerr << "MHD must be enabled to use these initial conditions !!!";
      return;
    }

    /* read Static magnetic field component */
    real_t Bx = configMap.getFloat("jet","BStatic_x",0.0);
    real_t By = configMap.getFloat("jet","BStatic_y",0.0);
    real_t Bz = configMap.getFloat("jet","BStatic_z",0.0);

    // reset domain
    memset(h_U.data(),0,h_U.sizeBytes());
    
    if (dimType == TWO_D) {

      /* jet */
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	  h_U(i,j,ID)=1.0f;
	  h_U(i,j,IU)=0.0f;
	  h_U(i,j,IV)=0.0f;
	  h_U(i,j,IW)=0.0f;
	  h_U(i,j,IA)=Bx;
	  h_U(i,j,IB)=By;
	  h_U(i,j,IC)=Bz;
	}

    } else { // THREE_D

      /* jet */
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    h_U(i,j,k,ID)=1.0f;
	    h_U(i,j,k,IP)=1.0f/(_gParams.gamma0-1.0f) + 0.5*(Bx*Bx+By*By+Bz*Bz);
	    h_U(i,j,k,IU)=0.0f;
	    h_U(i,j,k,IV)=0.0f;
	    h_U(i,j,k,IW)=0.0f;
	    h_U(i,j,k,IA)=Bx;
	    h_U(i,j,k,IB)=By;
	    h_U(i,j,k,IC)=Bz;
	  }
      
    } // end THREE_D

  } // HydroRunBaseMpi::init_mhd_jet
  
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::init_mhd_implode()
  {
    
    if (!mhdEnabled) {
      if (myRank == 0)
	std::cerr << "MHD must be enabled to use these initial conditions !!!";
      return;
    }

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    // amplitude of a random noise added to the vector potential
    const double amp       = configMap.getFloat("implode","amp",0.0);  
    int          seed      = configMap.getInteger("implode","seed",1);

    const double Bx = configMap.getFloat("implode","Bx",0.0);
    const double By = configMap.getFloat("implode","By",0.0);
    const double Bz = configMap.getFloat("implode","Bz",0.0);
 
    // initialize random number generator
    seed *= myRank;
    srand(seed);

    if (dimType == TWO_D) {
  
      /* discontinuity line along diagonal */
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  // compute global indexes
	  int ii = i + nx*myMpiPos[0];
	  int jj = j + ny*myMpiPos[1];
	  if (((float)ii/nx/mx+(float)jj/ny/my)>1) {
	    h_U(i,j,ID)=1.0f + amp * ( (1.0*rand())/RAND_MAX - 0.5);
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	    h_U(i,j,IW)=0.0f;
	    h_U(i,j,IA)=Bx * (1 + amp * ( (1.0*rand())/RAND_MAX - 0.5) );
	    h_U(i,j,IB)=By * (1 + amp * ( (1.0*rand())/RAND_MAX - 0.5) );
	    h_U(i,j,IC)=Bz * (1 + amp * ( (1.0*rand())/RAND_MAX - 0.5) );
	    h_U(i,j,IP)=1.0f/(_gParams.gamma0-1.0f) + 
	      0.5*(h_U(i,j,IA) * h_U(i,j,IA) +
		   h_U(i,j,IB) * h_U(i,j,IB) +
		   h_U(i,j,IC) * h_U(i,j,IC) );
	  } else {
	    h_U(i,j,ID)=0.125f;
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	    h_U(i,j,IW)=0.0f;
	    h_U(i,j,IA)=0.0f;
	    h_U(i,j,IB)=0.0f;
	    h_U(i,j,IC)=0.0f;
	    h_U(i,j,IP)=0.14f/(_gParams.gamma0-1.0f);      
	  }
	}
     
    } else { // THREE_D

      /* discontinuity line along diagonal */
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    // compute global indexes
	    int ii = i + nx*myMpiPos[0];
	    int jj = j + ny*myMpiPos[1];
	    int kk = k + nz*myMpiPos[2];	    
	    if (((float)ii/nx/mx+(float)jj/ny/my+(float)kk/nz/mz)>1) {
	      h_U(i,j,k,ID)=1.0f + amp * ( (1.0*rand())/RAND_MAX - 0.5);
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	      h_U(i,j,k,IA)=Bx * (1 + amp * ( (1.0*rand())/RAND_MAX - 0.5) );
	      h_U(i,j,k,IB)=By * (1 + amp * ( (1.0*rand())/RAND_MAX - 0.5) );
	      h_U(i,j,k,IC)=Bz * (1 + amp * ( (1.0*rand())/RAND_MAX - 0.5) );
	      h_U(i,j,k,IP)=1.0f/(_gParams.gamma0-1.0f) +
		0.5*(h_U(i,j,k,IA) * h_U(i,j,IA) +
		     h_U(i,j,k,IB) * h_U(i,j,IB) +
		     h_U(i,j,k,IC) * h_U(i,j,IC) );
	    } else {
	      h_U(i,j,k,ID)=0.125f;
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	      h_U(i,j,k,IA)=0.0f;
	      h_U(i,j,k,IB)=0.0f;
	      h_U(i,j,k,IC)=0.0f;
	      h_U(i,j,k,IP)=0.14f/(_gParams.gamma0-1.0f);      
	    }
	  } // end for i,j,k
    } // end THREE_D

  } // HydroRunBaseMpi::init_mhd_implode

  // =======================================================
  // =======================================================
  /**
   * Orszag-Tang Vortex problem.
   *
   * adapted from Dumses/patch/ot/condinit.f90 original fortran code.
   * The energy initialization asserts periodic boundary conditions.
   *
   * \sa class MHDRunBase (for sequential mono-CPU / mono GPU) version
   *
   * \sa http://www.astro.virginia.edu/VITA/ATHENA/ot.html
   */
  void HydroRunBaseMpi::init_mhd_Orszag_Tang()
  {
    
    if (!mhdEnabled) {
      if (myRank == 0)
	std::cerr << "MHD must be enabled to use these initial conditions !!!";
      return;
    }

    // reset domain
    memset(h_U.data(),0,h_U.sizeBytes());
    
    const double TwoPi = 4.0*asin(1.0);
    const double B0    = 1.0/sqrt(2.0*TwoPi);
    const double p0    = (double) (_gParams.gamma0/(2.0*TwoPi));
    const double d0    = (double) (_gParams.gamma0*p0);
    const double v0    = 1.0;

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    //real_t &zMin = _gParams.zMin;

    if (dimType == TWO_D) {

      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {

	// global coordinate
	int jG = j + ny*myMpiPos[1];
	double yPos = yMin + dy/2 + (jG-ghostWidth)*dy;

	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {

	  // global coordinate
	  int iG = i + nx*myMpiPos[0];
	  double xPos = xMin + dx/2 + (iG-ghostWidth)*dx;

	  // density initialization
	  h_U(i,j,ID)  = static_cast<real_t>(d0);
          
	  // rho*vx
	  h_U(i,j,IU)  = static_cast<real_t>(-d0*v0*sin(yPos*TwoPi));
	  
	  // rho*vy
	  h_U(i,j,IV)  = static_cast<real_t>( d0*v0*sin(xPos*TwoPi));
	  
	  // rho*vz
	  h_U(i,j,IW) =  ZERO_F;
          
	  // bx
	  h_U(i,j,IBX) = static_cast<real_t>(-B0*sin(    yPos*TwoPi));
          
	  // by
	  h_U(i,j,IBY) = static_cast<real_t>( B0*sin(2.0*xPos*TwoPi));
	  
	  // bz
	  h_U(i,j,IBZ) = ZERO_F;
	  
	} // end for i

      } // end for j

      // total energy (periodic boundary conditions taken into account)
      // see original fortran code in Dumses/patch/ot/condinit.f90
      for (int j=0; j<jsize-1; j++) {
	
	for (int i=0; i<isize-1; i++) {
	  
	  h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
	    0.5 * ( SQR(h_U(i,j,IU)) / h_U(i,j,ID) +
		    SQR(h_U(i,j,IV)) / h_U(i,j,ID) +
		    0.25*SQR(h_U(i,j,IBX) + h_U(i+1,j  ,IBX)) + 
		    0.25*SQR(h_U(i,j,IBY) + h_U(i  ,j+1,IBY)) );
	  
	} // end for i
	
      }	// end for j
      
    } else { // THREE_D
      
      /* get direction (only usefull for 3D) : 0->XY, 1->YZ, 2->ZX */
      int direction = configMap.getInteger("OrszagTang","direction",0);
      if (direction < 0 || direction > 3) {
	direction = 0;
	std::cout << "Orszag-Tang direction set to X-Y plane" << std::endl;
      }
      
      if (direction == 0) { // vortex in X-Y plane

	for (int k=0; k<ksize; k++) {
	
	  for (int j=0; j<jsize; j++) {
	  
	    int jG = j + ny*myMpiPos[1];
	    double yPos = yMin + dy/2 + (jG-ghostWidth)*dy;
	  
	    for (int i=0; i<isize; i++) {
	    
	      int iG = i + ny*myMpiPos[0];
	      double xPos = xMin + dx/2 + (iG-ghostWidth)*dx;
	    
	      // density initialization
	      h_U(i,j,k,ID)  = static_cast<real_t>(d0);
	      
	      // rho*vx
	      h_U(i,j,k,IU)  = static_cast<real_t>(- d0*v0*sin(yPos*TwoPi));
	      
	      // rho*vy
	      h_U(i,j,k,IV)  = static_cast<real_t>(  d0*v0*sin(xPos*TwoPi));
	      
	      // rho*vz
	      h_U(i,j,k,IW) =  ZERO_F;
	      
	      // bx
	      h_U(i,j,k,IBX) = static_cast<real_t>(-B0*sin(    yPos*TwoPi));

	      // by
	      h_U(i,j,k,IBY) = static_cast<real_t>( B0*sin(2.0*xPos*TwoPi));
	      
	      // bz
	      h_U(i,j,k,IBZ) = ZERO_F;
	      
	    } // end for i
	    
	  } // end for j
	  
	} // end for k
	
	// total energy (periodic boundary conditions taken into account)
	// see original fortran code in Dumses/patch/ot/condinit.f90
	for (int k=0; k<ksize; k++) {
	  
	  for (int j=0; j<jsize; j++) {
	    
	    for (int i=0; i<isize; i++) {
	      
	      h_U(i,j,k,IP)  = p0 / (_gParams.gamma0-1.0) +
		0.5 * ( SQR(h_U(i,j,k,IU)) / h_U(i,j,k,ID) +
			SQR(h_U(i,j,k,IV)) / h_U(i,j,k,ID) +
			0.25*SQR(h_U(i,j,k,IBX) + h_U(i+1,j  ,k,IBX)) + 
			0.25*SQR(h_U(i,j,k,IBY) + h_U(i  ,j+1,k,IBY)) );
	      
	    } // end for i
	    
	  } // end for j
	  
	} // end for k
	
      } // end direction == 0

    } // end THREE_D

  } // HydroRunBaseMpi::init_mhd_Orszag_Tang
  
  // =======================================================
  // =======================================================
  /**
   * The 2D/3D MHD field loop advection problem.
   * 
   * Parameters that can be set in the ini file :
   * - radius       : radius of field loop
   * - amplitude    : amplitude of vector potential (and therefore B in loop)
   * - vflow        : flow velocity
   * - densityRatio : density ratio in loop.  Enables density advection and
   *                  thermal conduction tests.
   * The flow is automatically set to run along the diagonal. 
   * - direction : integer 
   *   direction 0 -> field loop in x-y plane (cylinder in 3D)
   *   direction 1 -> field loop in y-z plane (cylinder in 3D)
   *   direction 2 -> field loop in z-x plane (cylinder in 3D)
   *   direction 3 -> rotated cylindrical field loop in 3D.
   *
   * Reference :
   * - T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD
   *   via constrained transport", JCP, 205, 509 (2005)
   * - http://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html
   */
  void HydroRunBaseMpi::init_mhd_field_loop()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    // some constants
    real_t radius    = configMap.getFloat("FieldLoop","radius"   ,1.0f);
    real_t density_in= configMap.getFloat("FieldLoop","density_in", 1.0f);
    real_t amplitude = configMap.getFloat("FieldLoop","amplitude",1.0f);
    real_t vflow     = configMap.getFloat("FieldLoop","vflow"    ,1.0f);

    // amplitude of a random noise added to the vector potential
    const double amp       = configMap.getFloat("FieldLoop","amp",0.01);  
    int          seed      = configMap.getInteger("FieldLoop","seed",0);

    // initialize random number generator
    seed *= myRank;
    srand48(seed);
    
    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    //real_t &zMin = _gParams.zMin;

    const real_t cos_theta = 2.0/sqrt(5.0);
    const real_t sin_theta = sqrt(1-cos_theta*cos_theta);
    
    if (dimType == TWO_D) {

      // vector potential
      HostArray<real_t> Az;
      Az.allocate( make_uint3(isize, jsize, 1) );

      // initialize vector potential
      for (int j=0; j<jsize; j++) {
	
	// global coordinate
	int jG = j + ny*myMpiPos[1];
	real_t yPos = yMin + dy/2 + (jG-ghostWidth)*dy;
	
	for (int i=0; i<isize; i++) {
	  
	  // global coordinate
	  int iG = i + nx*myMpiPos[0];
	  real_t xPos = xMin + dx/2 + (iG-ghostWidth)*dx;

	  real_t r = SQRT(xPos*xPos+yPos*yPos);
	  if ( r < radius ) {
	    Az(i,j,0) = amplitude * ( radius - r );
	  } else {
	    Az(i,j,0) = ZERO_F;
	  }
	
	} // end for i

      } // end for j

      // init MHD
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	
	int jG = j + ny*myMpiPos[1];
	real_t yPos = yMin + dy/2 + (jG-ghostWidth)*dy;
	
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  
	  int iG = i + nx*myMpiPos[0];
	  real_t xPos = xMin + dx/2 + (iG-ghostWidth)*dx;

	  real_t diag = SQRT(1.0*(nx*nx + ny*ny + nz*nz));
	  real_t r    = SQRT(xPos*xPos+yPos*yPos);

	  // density
	  if (r < radius)
	    h_U(i,j,ID) = density_in;
	  else
	    h_U(i,j,ID) = 1.0f;

	  // rho*vx
	  //h_U(i,j,IU) = h_U(i,j,ID)*vflow*nx/diag;
	  h_U(i,j,IU) = h_U(i,j,ID)*vflow*cos_theta;

	  // rho*vy
	  //h_U(i,j,IV) = h_U(i,j,ID)*vflow*ny/diag;
	  h_U(i,j,IV) = h_U(i,j,ID)*vflow*sin_theta;

	  // rho*vz
	  h_U(i,j,IW) = h_U(i,j,ID)*vflow*nz/diag; //ZERO_F;

	  // bx
	  h_U(i,j,IA) =   (Az(i  ,j+1,0) - Az(i,j,0))/dy + amp*(drand48()-0.5);

	  // by
	  h_U(i,j,IB) = - (Az(i+1,j  ,0) - Az(i,j,0))/dx + amp*(drand48()-0.5);

	  // bz
	  h_U(i,j,IC) = ZERO_F;

	  // total energy
	  h_U(i,j,IP) = 1.0f/(_gParams.gamma0-1.0f) + 
	    0.5 * (h_U(i,j,IA) * h_U(i,j,IA) + h_U(i,j,IB) * h_U(i,j,IB)) +
	    0.5 * (h_U(i,j,IU) * h_U(i,j,IU) + h_U(i,j,IV) * h_U(i,j,IV))/h_U(i,j,ID);
	  //h_U(i,j,IP) = ZERO_F;
	} // end for i
      } // end for j

    } else if (dimType == THREE_D) {

      // vector potential
      HostArray<real_t> A;
      A.allocate( make_uint4(isize, jsize, ksize, 3) );

      for (int k=0; k<ksize; k++) {
	
	//int kG = k + nz*myMpiPos[2];
	//real_t zPos = zMin + dz/2 + (kG-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  
	  int jG = j + ny*myMpiPos[1];
	  real_t yPos = yMin + dy/2 + (jG-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    
	    int iG = i + nx*myMpiPos[0];
	    real_t xPos = xMin + dx/2 + (iG-ghostWidth)*dx;
	    
	    A(i,j,k,0) = ZERO_F;
	    A(i,j,k,1) = ZERO_F;
	    A(i,j,k,2) = ZERO_F;
	    real_t r    = SQRT(xPos*xPos+yPos*yPos);
	    if (r < radius) {
	      A(i,j,k,2) = amplitude * (radius - r);
	    }

	  } // end for i

	} // end for j
	
      } // end for k
      
      // init MHD
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {

	//int kG = k + nz*myMpiPos[2];
	//real_t zPos = zMin + dz/2 + (kG-ghostWidth)*dz;

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	
	  int jG = j + ny*myMpiPos[1];
	  real_t yPos = yMin + dy/2 + (jG-ghostWidth)*dy;

	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  
	    int iG = i + nx*myMpiPos[0];
	    real_t xPos = xMin + dx/2 + (iG-ghostWidth)*dx;

	    //real_t diag = SQRT(1.0*(nx*nx + ny*ny + nz*nz));
	    real_t r;
	    r = SQRT(xPos*xPos + yPos*yPos);
	    
	    // density
	    if (r < radius)
	      h_U(i,j,k,ID) = density_in;
	    else
	      h_U(i,j,k,ID) = 1.0f;
	    
	    // rho*vx
	    //h_U(i,j,k,IU) = h_U(i,j,k,ID)*vflow*nx/diag;
	    h_U(i,j,k,IU) = h_U(i,j,k,ID)*vflow*cos_theta*(1+amp*(drand48()-0.5));
	    
	    // rho*vy
	    //h_U(i,j,k,IV) = h_U(i,j,k,ID)*vflow*ny/diag;
	    h_U(i,j,k,IV) = h_U(i,j,k,ID)*vflow*sin_theta*(1+amp*(drand48()-0.5));
	    
	    // rho*vz
	    h_U(i,j,k,IW) = h_U(i,j,k,ID)*vflow*(1+amp*(drand48()-0.5)); //ZERO_F; //h_U(i,j,k,ID)*vflow*nz/diag;
	    
	    // bx
	    h_U(i,j,k,IA) =
	      ( A(i,j+1,k  ,2) - A(i,j,k,2) ) / dy -
	      ( A(i,j  ,k+1,1) - A(i,j,k,1) ) / dz + amp*(drand48()-0.5);
	    
	    // by
	    h_U(i,j,k,IB) = 
	      ( A(i  ,j,k+1,0) - A(i,j,k,0) ) / dz -
	      ( A(i+1,j,k  ,2) - A(i,j,k,2) ) / dx + amp*(drand48()-0.5);
	    
	    // bz
	    h_U(i,j,k,IC) = 
	      ( A(i+1,j  ,k,1) - A(i,j,k,1) ) / dx -
	      ( A(i  ,j+1,k,0) - A(i,j,k,0) ) / dy + amp*(drand48()-0.5);
	    
	    // total energy
	    if (_gParams.cIso>0) {
	      h_U(i,j,k,IP) = ZERO_F;
	    } else {
	      h_U(i,j,k,IP) = 1.0f/(_gParams.gamma0-1.0f) + 
		0.5 * (h_U(i,j,k,IA) * h_U(i,j,k,IA)  + 
		       h_U(i,j,k,IB) * h_U(i,j,k,IB)  +
		       h_U(i,j,k,IC) * h_U(i,j,k,IC)) +
		0.5 * (h_U(i,j,k,IU) * h_U(i,j,k,IU) + 
		       h_U(i,j,k,IV) * h_U(i,j,k,IV) +
		       h_U(i,j,k,IW) * h_U(i,j,k,IW))/h_U(i,j,k,ID);
	    }
	  } // end for i
	} // end for j
      } // end for k
      
    } // end THREE_D
    
  } // HydroRunBaseMpi::init_mhd_field_loop

  // =======================================================
  // =======================================================
  /**
   * The 2D/3D MHD shear wave problem (Shearing border conditions must be activated).
   * 
   * This test aims at verifying that the numerical scheme is OK with shearing box 
   * border conditions.
   *
   * Parameters that can be set in the ini file :
   * - density
   * - delta_vx : amplitude of the X component of velocity field
   *
   * \sa Dumses in patch/tests/shwave
   */
  void HydroRunBaseMpi::init_mhd_shear_wave()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    if (boundary_xmin != BC_SHEARINGBOX or boundary_xmax != BC_SHEARINGBOX) {
      std::cerr << "Shearing box border conditions must enabled along X-direction !!!\n";
      return;
    }

    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    // some constants
    const double TwoPi     = 4.0*asin(1.0);
    const double d0        = 1.0;

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t  xMax = configMap.getFloat("mesh","xmax",1.0);
    real_t  yMax = configMap.getFloat("mesh","ymax",1.0);

    const double Lx        = xMax - xMin;
    const double Ly        = yMax - yMin;

    //const double density   = configMap.getFloat("ShearWave","density" ,1.0);
    const double energy    = configMap.getFloat("ShearWave","energy",1.0);
    const double delta_vx  = (-4.0e-4) *_gParams.cIso;
    const double delta_vy  = ( 1.0e-4) *_gParams.cIso;
    const double kx0       = -4*TwoPi/Lx;
    const double ky0       =    TwoPi/Ly;
    const double xi0       = 0.5* _gParams.Omega0/d0;
    const double delta_rho = (kx0*delta_vy-ky0*delta_vx)/xi0;


    if (dimType == TWO_D) {

      for (int j=0; j<jsize; j++) {
	
	// global coordinate
	int jG = j + ny*myMpiPos[1];
	double yPos = yMin + dy/2 + (jG-ghostWidth)*dy;

	for (int i=0; i<isize; i++) {

	  // global coordinate
	  int iG = i + nx*myMpiPos[0];
	  double xPos = xMin + dx/2 + (iG-ghostWidth)*dx;

	  h_U(i,j,ID) = d0*(1.0-delta_rho*sin(kx0*xPos+ky0*yPos));
	  h_U(i,j,IP) = energy;
	  h_U(i,j,IU) = h_U(i,j,ID)*delta_vx*cos(kx0*xPos+ky0*yPos);
	  h_U(i,j,IV) = h_U(i,j,ID)*delta_vy*cos(kx0*xPos+ky0*yPos);
	  h_U(i,j,IW) = ZERO_F;
	  h_U(i,j,IA) = ZERO_F;
	  h_U(i,j,IB) = ZERO_F;
	  h_U(i,j,IC) = ZERO_F;

	} // end for i
      } // end for j

    } else { // THREE_D

      for (int k=0; k<ksize; k++) {
	
	for (int j=0; j<jsize; j++) {

	  // global coordinate
	  int jG = j + ny*myMpiPos[1];
	  double yPos = yMin + dy/2 + (jG-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    
	    // global coordinate
	    int iG = i + nx*myMpiPos[0];
	    double xPos = xMin + dx/2 + (iG-ghostWidth)*dx;
	    
	    h_U(i,j,k,ID) = d0*(1.0-delta_rho*sin(kx0*xPos+ky0*yPos));
	    h_U(i,j,k,IP) = energy;
	    h_U(i,j,k,IU) = h_U(i,j,ID)*delta_vx*cos(kx0*xPos+ky0*yPos);
	    h_U(i,j,k,IV) = h_U(i,j,ID)*delta_vy*cos(kx0*xPos+ky0*yPos);
	    h_U(i,j,k,IW) = ZERO_F;
	    h_U(i,j,k,IA) = ZERO_F;
	    h_U(i,j,k,IB) = ZERO_F;
	    h_U(i,j,k,IC) = ZERO_F;
	    
	  } // end for i
	} // end for j     
      } // end for k

    } // end THREE_D

  } // HydroRunBaseMpi::init_mhd_shear_wave

  // =======================================================
  // =======================================================
  /**
   * The 3D MHD MRI problem (Shearing border conditions must be activated).
   * 
   * Setup for making a MRI (Magneto-Rotational Instability) simulation.
   *
   * Parameters that can be set in the ini file :
   * - density
   * - beta : real - large value means small magnetic field  /
   *                 small magnetic energy compared to thermal energy
   * - type : string ('noflux', 'pyl', 'fluxZ')
   * - amp  : real - velocity amplitude
   * - seed : int - random number seed
   *
   * \sa Dumses in patch/mri
   */
  void HydroRunBaseMpi::init_mhd_mri()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    if (dimType == TWO_D) {
      std::cerr << "MRI simulations is only available in 3D !\n";
      return;
    }
    
    if (boundary_xmin != BC_SHEARINGBOX or boundary_xmax != BC_SHEARINGBOX) {
      std::cerr << "Shearing box border conditions must enabled along X-direction !!!\n";
      return;
    }

    // Check if Isothermal EOS is enabled, if not print a warning
    if (_gParams.cIso <= 0) {
      std::cout << "############################################################\n";
      std::cout << "#### WARNING: MRI problem called with NON Isothermal EOS !!!\n";
      std::cout << "############################################################\n";
    }


    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    // some constants
    const double TwoPi     = 4.0*asin(1.0);
    const double d0        = configMap.getFloat("MRI","density",1.0);
    const double beta      = configMap.getFloat("MRI", "beta", 400.0);

    const double p0        = d0 * _gParams.cIso * _gParams.cIso; 
    
    real_t &zMin = _gParams.zMin;
    real_t  zMax = configMap.getFloat("mesh","zmax",1.0);
    real_t &Omega0 = _gParams.Omega0;

    double B0;
    std::string type       = configMap.getString("MRI","type","noflux"); 
    if (!type.compare("pyl"))
      B0 = 3.0/2.0 * sqrt( d0 * Omega0 * Omega0 * (zMax-zMin)*(zMax-zMin) / beta);
    else
      B0 = 2.0 * sqrt(p0/beta);

    const double amp       = configMap.getFloat("MRI","amp",0.01);  
    const double d_amp     = configMap.getFloat("MRI","density_fluctuations",0.0);
    int          seed      = configMap.getInteger("MRI","seed",0);
    seed *= myRank;

    real_t &xMin = _gParams.xMin;

    // initialize random number generator
    srand48(seed);

    for (int k=0; k<ksize; k++) {
      
      for (int j=0; j<jsize; j++) {
	
	for (int i=0; i<isize; i++) {

	  int    iG   = i + nx*myMpiPos[0];
	  double xPos = xMin + dx/2 + (iG-ghostWidth)*dx;

	  h_U(i,j,k,ID) = d0 * (1 + d_amp*2*(drand48()-0.5));
	  h_U(i,j,k,IP) = 0;
	  h_U(i,j,k,IU) = d0*amp*(drand48()-0.5)*sqrt(p0);
	  h_U(i,j,k,IV) = d0*amp*(drand48()-0.5)*sqrt(p0);
	  h_U(i,j,k,IW) = d0*amp*(drand48()-0.5)*sqrt(p0);
	  h_U(i,j,k,IA) = ZERO_F;
	  h_U(i,j,k,IB) = ZERO_F; // B0 (Toroidal field)
	  if (!type.compare("noflux")) {
	    h_U(i,j,k,IC) = B0*sin(TwoPi*xPos);
	  } else if (!type.compare("pyl") || !type.compare("fluxZ")){
	    h_U(i,j,k,IC) = B0;
	  } else {
	    h_U(i,j,k,IC) = ZERO_F;
	  }
	  
	} // end for i
      } // end for j     
    } // end for k
    
    /*
     * if gravity is enabled, special init routine
     */
    if (gravityEnabled) {

      // initialize gravity field
      init_mhd_mri_grav_field();

      // modify density and magnetif field
      double zFloor      = configMap.getFloat("MRI", "zFloor"       , 5.0);
      real_t &cIso       = _gParams.cIso;
      double H           = cIso/Omega0;

      for (int k=0; k<ksize; k++) {
	int      kG = k + nz*myMpiPos[2];
	real_t zPos = _gParams.zMin + dz/2 + (kG-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  //int      jG = j + ny*myMpiPos[1];
	  //real_t yPos = _gParams.yMin + dy/2 + (jG-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    //int      iG = i + nx*myMpiPos[0];
	    //real_t xPos = _gParams.xMin + dx/2 + (iG-ghostWidth)*dx;

	    // modify density
	    h_U(i,j,k,ID) = d0 * fmax ( exp(-(zPos*zPos)/2.0/(H*H)), exp(-zFloor*zFloor/2.0) );
	    //h_U(i,j,k,ID) = d0 * exp(-(zPos*zPos)/2.0/(H*H));

	    // enforce azimuthal magnetic field
	    h_U(i,j,k,IA) = ZERO_F;
	    h_U(i,j,k,IB) = ZERO_F;
	    h_U(i,j,k,IC) = ZERO_F;

	    if (zPos < H and zPos > -H)
	      h_U(i,j,k,IB) = B0;

	  } // end for i
	} // end for j
      } // end for k
   
    } // end gravity enabled

  } // HydroRunBaseMpi::init_mhd_mri

  // =======================================================
  // =======================================================
  /**
   * Test of the Kelvin-Helmholtz instability.
   * See
   * http://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
   * for a description of such initial conditions
   *
   */
  void HydroRunBaseMpi::init_mhd_Kelvin_Helmholtz()
  {

    // initialize with a uniform magnetic field
    real_t Bx0 = configMap.getFloat("kelvin-helmholtz", "Bx0",        0.5);
    real_t By0 = configMap.getFloat("kelvin-helmholtz", "By0",        0.0);
    real_t Bz0 = configMap.getFloat("kelvin-helmholtz", "Bz0",        0.0);

    real_t Emag = 0.5 * (Bx0*Bx0 + By0*By0 + Bz0*Bz0);

    // call hydro initialization routine
    init_hydro_Kelvin_Helmholtz();

    if (dimType == TWO_D) {
      
      // initialize magnetic field
      for (int j=0; j<jsize; j++)
	for (int i=0; i<isize; i++) {
	  h_U(i,j,IBX) = Bx0;
	  h_U(i,j,IBY) = By0;
	  h_U(i,j,IBZ) = Bz0;
	  
	  // update energy
	  h_U(i,j,IP) += Emag;
	  
	} // end for i,j

    } else { //THREE_D
   
      // initialize magnetic field
      for (int k=0; k<ksize; k++)
	for (int j=0; j<jsize; j++)
	  for (int i=0; i<isize; i++) {
	    h_U(i,j,k,IBX) = Bx0;
	    h_U(i,j,k,IBY) = By0;
	    h_U(i,j,k,IBZ) = Bz0;
	    
	    // update energy
	    h_U(i,j,k,IP) += Emag;
	    
	  } // end for i,j,k
    
    } // end THREE_D
    
  } // HydroRunBaseMpi::init_mhd_Kelvin_Helmholtz

  // =======================================================
  // =======================================================
  /**
   * The 2D/3D MHD Rayleigh-Taylor instability problem.
   *
   * See
   * http://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
   * for a description of such initial conditions
   */
  void HydroRunBaseMpi::init_mhd_Rayleigh_Taylor()
  {

    // magnetic field initial conditions
    real_t Bx0 = configMap.getFloat("rayleigh-taylor", "bx",  1e-8);
    real_t By0 = configMap.getFloat("rayleigh-taylor", "by",  1e-8);
    real_t Bz0 = configMap.getFloat("rayleigh-taylor", "bz",  1e-8);
    
    real_t Emag = 0.5 * (Bx0*Bx0 + By0*By0 + Bz0*Bz0);

    // call hydro initialization routine
    init_hydro_Rayleigh_Taylor();

    if (dimType == TWO_D) {
      
      // initialize magnetic field
      for (int j=0; j<jsize; j++)
	for (int i=0; i<isize; i++) {
	  h_U(i,j,IBX) = Bx0;
	  h_U(i,j,IBY) = By0;
	  h_U(i,j,IBZ) = Bz0;
	  
	  // update energy
	  h_U(i,j,IP) += Emag;
	  
	} // end for i,j

    } else { //THREE_D
   
      // initialize magnetic field
      for (int k=0; k<ksize; k++)
	for (int j=0; j<jsize; j++)
	  for (int i=0; i<isize; i++) {
	    h_U(i,j,k,IBX) = Bx0;
	    h_U(i,j,k,IBY) = By0;
	    h_U(i,j,k,IBZ) = Bz0;
	    
	    // update energy
	    h_U(i,j,k,IP) += Emag;
	    
	  } // end for i,j,k
    
    } // end THREE_D

  } // HydroRunBaseMpi::init_mhd_Rayleigh_Taylor

  // =======================================================
  // =======================================================
  /**
   * The 3D forcing MHD turbulence problem.
   *
   */
  void HydroRunBaseMpi::init_mhd_turbulence()
  {

    if (dimType == TWO_D) {
      
      if (myRank==0) std::cerr << "Turbulence problem is not available in 2D...." << std::endl;
      
    } else { // THREE_D
      
      // magnetic field initial conditions
      real_t Bx0 = configMap.getFloat("turbulence", "bx",  1e-8);
      real_t By0 = configMap.getFloat("turbulence", "by",  1e-8);
      real_t Bz0 = configMap.getFloat("turbulence", "bz",  1e-8);
      
      // beta plasma is ratio between plasma pressure (p=n*k_B*T)
      // and magnetic pressure (P_mag=B^2/2*mu_0)
      real_t beta = configMap.getFloat("turbulence", "beta",  0.0);

      
      if (beta>0) {
	// use that beta to initialize B along x
	double cIso2 = _gParams.cIso * _gParams.cIso;
	double d0    = configMap.getFloat("turbulence", "density",  1.0);
	Bx0 = sqrt(2*cIso2*d0/beta);
	By0 = 0.0;
	Bz0 = 0.0;
      }

      // call hydro initialization routine
      init_hydro_turbulence();
      
      // initialize magnetic field
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	    h_U(i,j,k,IBX) = Bx0;
	    h_U(i,j,k,IBY) = By0;
	    h_U(i,j,k,IBZ) = Bz0;

	    // update energy
	    h_U(i,j,k,IP) += 0.5 * (Bx0*Bx0 + By0*By0 + Bz0*Bz0);

	  } // end for i,j,k

    } // end THREE_D

  } // HydroRunBaseMpi::init_mhd_turbulence

  // =======================================================
  // =======================================================
  /**
   *
   * The 3D forcing MHD turbulence problem using Ornstein-Uhlenbeck forcing.
   *
   */
  void HydroRunBaseMpi::init_mhd_turbulence_Ornstein_Uhlenbeck()
  {

    if (dimType == TWO_D) {
      
      if (myRank==0) std::cerr << "Turbulence problem is not available in 2D...." << std::endl;
      
    } else { // THREE_D
      
      // magnetic field initial conditions
      real_t Bx0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "bx",  1e-8);
      real_t By0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "by",  1e-8);
      real_t Bz0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "bz",  1e-8);
      
      // beta plasma is ratio between plasma pressure (p=n*k_B*T)
      // and magnetic pressure (P_mag=B^2/2*mu_0)
      real_t beta = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "beta",  0.0);

      
      if (beta>0) {
	// use that beta to initialize B along x
	double cIso2 = _gParams.cIso * _gParams.cIso;
	double d0    = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "density",  1.0);
	Bx0 = sqrt(2*cIso2*d0/beta);
	By0 = 0.0;
	Bz0 = 0.0;
	
	if (cIso2 <= 0.0) { // non-isothermal simulation
	  Bx0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "Bx0",  2.0*d0/beta);
	}

      }

      // call hydro initialization routine
      init_hydro_turbulence_Ornstein_Uhlenbeck();
      
      // initialize magnetic field
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	    h_U(i,j,k,IBX) = Bx0;
	    h_U(i,j,k,IBY) = By0;
	    h_U(i,j,k,IBZ) = Bz0;

	    // update energy
	    h_U(i,j,k,IP) += 0.5 * (Bx0*Bx0 + By0*By0 + Bz0*Bz0);

	  } // end for i,j,k

    } // end THREE_D

  } // HydroRunBaseMpi::init_mhd_turbulence_Ornstein_Uhlenbeck

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::init_mhd_mri_grav_field()
  {

    // initialize gravity field
    if (gravityEnabled) {

      double phi[2][THREE_D] = {0.0};
      h_gravity.reset();
      
      bool smoothGravity = configMap.getBool ("MRI", "smoothGravity", false);
      double zFloor      = configMap.getFloat("MRI", "zFloor"       , 5.0);
      
      real_t &Omega0     = _gParams.Omega0;
      //real_t &cIso       = _gParams.cIso;
      //double H           = cIso/Omega0;
      
      for (int k=0; k<ksize; k++) {
	int      kG = k + nz*myMpiPos[2];
	real_t zPos = _gParams.zMin + dz/2 + (kG-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  //int      jG = j + ny*myMpiPos[1];
	  //real_t yPos = _gParams.yMin + dy/2 + (jG-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    //int      iG = i + nx*myMpiPos[0];
	    //real_t xPos = _gParams.xMin + dx/2 + (iG-ghostWidth)*dx;
	    
	    phi[0][IZ] = HALF_F*Omega0*Omega0*(zPos-dz)*(zPos-dz);
	    phi[1][IZ] = HALF_F*Omega0*Omega0*(zPos+dz)*(zPos+dz);
	    
	    if (smoothGravity) {
              if ( (zPos-dz)>zFloor ) phi[0][IZ] = HALF_F*Omega0*Omega0*zFloor*zFloor;
              if ( (zPos+dz)>zFloor ) phi[1][IZ] = HALF_F*Omega0*Omega0*zFloor*zFloor;
	    }
	    
	    h_gravity(i,j,k,IX) = - HALF_F * ( phi[1][IX] - phi[0][IX] ) / dx;
	    h_gravity(i,j,k,IY) = - HALF_F * ( phi[1][IY] - phi[0][IY] ) / dy;
	    h_gravity(i,j,k,IZ) = - HALF_F * ( phi[1][IZ] - phi[0][IZ] ) / dz;
	    
	  } // end for i
	} // end for j
      } // end for k

      // need to upload gravity field on GPU
#ifdef __CUDACC__
    d_gravity.copyFromHost(h_gravity);
#endif

    } // end gravityEnabled

  } // HydroRunBaseMpi::init_mhd_mri_grav_field
  
  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::restart_run_extra_work()
  {

    if ( (!problem.compare("MRI") ||
	  !problem.compare("Mri") ||
	  !problem.compare("mri") ) and gravityEnabled) {

      // we need to re-generate gravity field
      init_mhd_mri_grav_field();

    } // end extra stuff for restarting a stratified mri run

  } // HydroRunBaseMpi::restart_run_extra_work

  // =======================================================
  // =======================================================
  int HydroRunBaseMpi::init_simulation(const std::string problemName)
  {

    // test if we are performing a re-start run (default : false)
    bool restartEnabled = configMap.getBool("run","restart",false);
    int timeStep = 0;

    if (restartEnabled) { // load data from input data file

      // initial condition in grid interior
      memset(h_U.data(),0,h_U.sizeBytes());
      
      // get input filename from configMap
      std::string inputFilename = configMap.getString("run", "restart_filename", "");

      // check filename extension
      std::string h5Suffix(".h5");
      std::string ncSuffix(".nc");

      bool isHdf5=false, isNcdf=false;
      if (inputFilename.length() >= 3) {
	isHdf5 = (0 == inputFilename.compare (inputFilename.length() - 
					      h5Suffix.length(), 
					      h5Suffix.length(), 
					      h5Suffix) );
	isNcdf = (0 == inputFilename.compare (inputFilename.length() - 
					      ncSuffix.length(), 
					      ncSuffix.length(), 
					      ncSuffix) );
      }

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

	if (isHdf5) {
	  inputHdf5(h_input, outputDir+"/"+inputFilename, halfResolution);
	} else if (isNcdf) {
	  inputPnetcdf(h_input, outputDir+"/"+inputFilename, halfResolution);
	} else {
	  if (myRank == 0) {
	    std::cerr << "Unknown input filename extension !\n";
	    std::cerr << "Should be \".h5\" or \".nc\"\n";
	  }
	}

	// upscale h_input into h_U (i.e. double resolution)
	upscale(h_U, h_input);

      } else { // standard restart
	
	// read input file into h_U buffer , and return time Step
	if (isHdf5) {
	  timeStep = inputHdf5(h_U, outputDir+"/"+inputFilename);
	} else if (isNcdf) {
	  timeStep = inputPnetcdf(h_U, outputDir+"/"+inputFilename);
	} else {
	  if (myRank == 0) {
	    std::cerr << "Unknown input filename extension !\n";
	    std::cerr << "Should be \".h5\" or \".nc\"\n";
	  }
	}

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

    } else { // regular initialization

      // do we perform a MHD problem initialization
      if (mhdEnabled) {
	
	if (!problemName.compare("jet")) {
	  this->init_mhd_jet();
	} else if (!problemName.compare("implode")) {
	  this->init_mhd_implode();
	} else if (!problemName.compare("Orszag-Tang")) {
	  this->init_mhd_Orszag_Tang();
	} else if (!problemName.compare("FieldLoop")  ||
		   !problemName.compare("fieldloop")  ||
		   !problemName.compare("Fieldloop")  ||
		   !problemName.compare("field-loop") || 
		   !problemName.compare("Field-Loop")) {
	  this->init_mhd_field_loop();
	} else if (!problemName.compare("ShearWave") ||
		   !problemName.compare("shearwave") ||
		   !problemName.compare("Shear-Wave") ||
		   !problemName.compare("shear-wave") ||
		   !problemName.compare("Shearwave")) {
	  this->init_mhd_shear_wave();
	} else if (!problemName.compare("MRI") ||
		   !problemName.compare("Mri") ||
		   !problemName.compare("mri")) {
	  this->init_mhd_mri();
	} else if (!problemName.compare("Kelvin-Helmholtz")) {
	  this->init_mhd_Kelvin_Helmholtz();
	} else if (!problemName.compare("Rayleigh-Taylor")) {
	  this->init_mhd_Rayleigh_Taylor();
	} else if (!problemName.compare("turbulence")) {
	  this->init_mhd_turbulence();
	} else if (!problemName.compare("turbulence-Ornstein-Uhlenbeck")) {
	  this->init_mhd_turbulence_Ornstein_Uhlenbeck();
	} else {
	  if (myRank == 0) {
	    std::cerr << "given problem parameter is: " << problem << std::endl;
	    std::cerr << "unknown problem name; please set hydro parameter \"problem\" to a valid value, valid for MHD !!!" << std::endl;
	  }
	}
	
      } else { // hydro initialization
	
	if (!problemName.compare("jet")) {
	  this->init_hydro_jet();
	} else if (!problemName.compare("implode")) {
	  this->init_hydro_implode();
	} else if (!problemName.compare("blast")) {
	  this->init_hydro_blast();
	} else if (!problemName.compare("Kelvin-Helmholtz")) {
	  this->init_hydro_Kelvin_Helmholtz();
	} else if (!problemName.compare("Rayleigh-Taylor")) {
	  this->init_hydro_Rayleigh_Taylor();
	} else if (!problemName.compare("falling-bubble")) {
	  this->init_hydro_falling_bubble();
	} else if (!problemName.compare("riemann2d")) {
	  this->init_hydro_Riemann();
	} else if (!problemName.compare("turbulence")) {
	  this->init_hydro_turbulence();
	} else if (!problemName.compare("turbulence-Ornstein-Uhlenbeck")) {
	  this->init_hydro_turbulence_Ornstein_Uhlenbeck();
	} else {
	  if (myRank == 0) {
	    std::cerr << "given problem parameter is: " << problem << std::endl;
	    std::cerr << "unknown problem name; please set hydro parameter \"problem\" to a valid value !!!" << std::endl;
	  }
	}

      } // end hydro initialization
    
    } // end regular initialization

    // copy data to GPU if necessary
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

  } // HydroRunBaseMpi::init_simulation

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::init_randomForcing()
  {

    if (dimType == TWO_D) {
    
      std::cerr << "Turbulence problem is not available in 2D...." << std::endl;

    } else { // THREE_D

      real_t d0   = configMap.getFloat("turbulence", "density",  1.0);
      real_t eDot = configMap.getFloat("turbulence", "edot", -1.0);
      
      real_t randomForcingMachNumber = configMap.getFloat("turbulence", "machNumber", 0.0);
      if (myRank==0) std::cout << "Random forcing Mach number is " << randomForcingMachNumber << std::endl;
      
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
      if (myRank==0) std::cout << "Using random forcing with eDot : " << eDot << std::endl;
            
      /* turbulence */
      // compute random field
      turbulenceInit(isize, jsize, ksize, 
		     nx*myMpiPos[0]-ghostWidth,
		     ny*myMpiPos[1]-ghostWidth,
		     nz*myMpiPos[2]-ghostWidth,
		     nx*mx, randomForcingMachNumber,
		     &(h_randomForcing(0,0,0,IX)), 
		     &(h_randomForcing(0,0,0,IY)),
		     &(h_randomForcing(0,0,0,IZ)) );
      
#ifdef __CUDACC__
      // we also need to copy
      d_randomForcing.copyFromHost(h_randomForcing);
#endif // __CUDACC__

    } // end THREE_D

  } // HydroRunBaseMpi::init_randomForcing

  // =======================================================
  // =======================================================
  void HydroRunBaseMpi::copyGpuToCpu(int nStep)
  {

#ifdef __CUDACC__
    if (nStep % 2 == 0)
      d_U.copyToHost(h_U);
    else
      d_U2.copyToHost(h_U2);
#endif // __CUDACC__

  } // HydroRunBaseMpi::copyGpuToCpu

  // =======================================================
  // =======================================================
  /*
   * setup history, choose which history method will be called
   */
  void HydroRunBaseMpi::setupHistory()
  {

    // history enabled ?
    bool historyEnabled = configMap.getBool("history","enabled",false);

    if (historyEnabled) {
      
      if (mhdEnabled) {

	if (!problem.compare("MRI") ||
	    !problem.compare("Mri") ||
	    !problem.compare("mri")) {
	  
	  history_method = &HydroRunBaseMpi::history_mhd_mri;
	  
	} else if (!problem.compare("Orszag-Tang") || 
		   !problem.compare("OrszagTang") ) {
	  
	  history_method = &HydroRunBaseMpi::history_mhd_default;
	  
	} else if ( !problem.compare("turbulence") ||
		    !problem.compare("turbulence-Ornstein-Uhlenbeck") ) {
	  
	  history_method = &HydroRunBaseMpi::history_mhd_turbulence;
	  
	} else {
	  
	  history_method = &HydroRunBaseMpi::history_mhd_default;
	  
	}
      
      } else { // pure hydro problem

	if ( !problem.compare("turbulence") ||
	     !problem.compare("turbulence-Ornstein-Uhlenbeck") ) {
	  
	  history_method = &HydroRunBaseMpi::history_hydro_turbulence;
	  
	} else {
	  
	  history_method = &HydroRunBaseMpi::history_hydro_default;
	  
	}
	
      } // end MHD enabled
      
    } else { // history disabled
      
      history_method = &HydroRunBaseMpi::history_empty;
      
    }

    // log some information
    if (myRank ==0) {
      if (historyEnabled) {
	std::cout << "History enabled  !!!\n";
      } else {
	std::cout << "History disabled !!!\n";
      }
    } // end myRank==0


  } // HydroRunBaseMpi::setupHistory

  // =======================================================
  // =======================================================
  /*
   * call history.
   */
  void HydroRunBaseMpi::history(int nStep, real_t dt)
  {
    // call the actual history method
    ((*this).*history_method)(nStep,dt);

  } // HydroRunBaseMpi::history

  // =======================================================
  // =======================================================
  /*
   * Default history, do nothing.
   */
  void HydroRunBaseMpi::history_empty(int nStep, real_t dt)
  {
    (void) nStep;
    (void) dt;

  } // HydroRunBaseMpi::history_empty

  // =======================================================
  // =======================================================
  /*
   * Default history for hydro.
   * only compute total mass (should be constant in time)
   */
  void HydroRunBaseMpi::history_hydro_default(int nStep, real_t dt)
  {
    (void) nStep;
    (void) dt;

    if (myRank==0)
      std::cout << "History at time " << totalTime << "\n";

    // open history file
    std::ofstream histo;
    
    if (myRank == 0) {
      
      // history file name
      std::string fileName = configMap.getString("history",
						 "filename", 
						 "history.txt");
      // get output prefix / outputDir
      std::string outputDir    = configMap.getString("output", "outputDir", "./");
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
      
      // build full path filename
      fileName = outputDir + "/" + outputPrefix + "_" + fileName;
      
      histo.open (fileName.c_str(), std::ios::out | std::ios::app | std::ios::ate); 
      
      // if this is the first time we call history, print header
      if (totalTime <= 0) {
	histo << "# history " << current_date() << std::endl;
	
	bool restartEnabled = configMap.getBool("run","restart",false);
	if (restartEnabled)
	  histo << "# history : this is a restart run\n";
	
	// write header (which variables are dumped)
	histo << "# totalTime dt mass\n";
	
      } // end print header
      
    } // end myRank == 0
    
    // make sure Device data are copied back onto Host memory
    // which data to save ?
    copyGpuToCpu(nStep);
    HostArray<real_t> &U = getDataHost(nStep);
    
    
    // compute local total mass, and local divB
    double mass = 0.0;

    if (dimType == TWO_D) {

      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  
	  mass += U(i,j,ID);
	  
	} // end for i
      } // end for j
      
      double dTau = dx*dy/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin);
      
      mass = mass*dTau;

    } else { // THREE_D
      
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    mass += U(i,j,k,ID);
	    
	  } // end for i
	} // end for j
      } // end for k
      
      double dTau = dx*dy*dz/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin)/
	(_gParams.zMax- _gParams.zMin);
      
      mass = mass*dTau;
      
    } // end THREE_D

    // do volume average (MPI reduction)
    double massT = 0.0;
    
    MPI_Reduce(&mass    ,&massT    ,1,MPI_DOUBLE,MPI_SUM,
	       0,communicator->getComm());

    if (myRank == 0) {
      histo << totalTime   << "\t" << dt     << "\t" 
	    << massT       << "\n";
      
      histo.close();
    } // end myRank == 0

  } // HydroRunBaseMpi::history_hydro_default

  // =======================================================
  // =======================================================
  /*
   * Default history for MHD.
   * only compute total mass (should be constant in time)
   * and divB (should be constant as small as possible)
   */
  void HydroRunBaseMpi::history_mhd_default(int nStep, real_t dt)
  {
    (void) nStep;
    (void) dt;

    if (myRank==0)
      std::cout << "History at time " << totalTime << "\n";

    // open history file
    std::ofstream histo;
    
    if (myRank == 0) {
      
      // history file name
      std::string fileName = configMap.getString("history",
						 "filename", 
						 "history.txt");
      // get output prefix / outputDir
      std::string outputDir    = configMap.getString("output", "outputDir", "./");
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
      
      // build full path filename
      fileName = outputDir + "/" + outputPrefix + "_" + fileName;
      
      histo.open (fileName.c_str(), std::ios::out | std::ios::app | std::ios::ate); 
      
      // if this is the first time we call history, print header
      if (totalTime <= 0) {
	histo << "# history " << current_date() << std::endl;
	
	bool restartEnabled = configMap.getBool("run","restart",false);
	if (restartEnabled)
	  histo << "# history : this is a restart run\n";
	
	// write header (which variables are dumped)
	histo << "# totalTime dt mass divB\n";
	
      } // end print header
      
    } // end myRank == 0
    
    // make sure Device data are copied back onto Host memory
    // which data to save ?
    copyGpuToCpu(nStep);
    HostArray<real_t> &U = getDataHost(nStep);
    
    
    // compute local total mass, and local divB
    double mass = 0.0;
    double divB = 0.0;

    if (dimType == TWO_D) {

      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  
	  mass += U(i,j,ID);
	  
	  divB +=  
	    ( U(i+1,j  ,IBX) - U(i,j,IBX) ) / dx + 
	    ( U(i  ,j+1,IBY) - U(i,j,IBY) ) / dy;
	  
	} // end for i
      } // end for j
      
      double dTau = dx*dy/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin);
      
      mass = mass*dTau;

    } else { // THREE_D
      
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    mass += U(i,j,k,ID);
	    
	    divB +=  
	      ( U(i+1,j  ,k  ,IBX) - U(i,j,k,IBX) ) / dx + 
	      ( U(i  ,j+1,k  ,IBY) - U(i,j,k,IBY) ) / dy + 
	      ( U(i  ,j  ,k+1,IBZ) - U(i,j,k,IBZ) ) / dz;
	    
	  } // end for i
	} // end for j
      } // end for k
      
      double dTau = dx*dy*dz/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin)/
	(_gParams.zMax- _gParams.zMin);
      
      mass = mass*dTau;
      
    } // end THREE_D

    // do volume average (MPI reduction)
    double massT = 0.0, divB_T=0.0;
    
    MPI_Reduce(&mass    ,&massT    ,1,MPI_DOUBLE,MPI_SUM,
	       0,communicator->getComm());

    MPI_Reduce(&divB    ,&divB_T   ,1,MPI_DOUBLE,MPI_SUM,
	       0,communicator->getComm());
    
    if (myRank == 0) {
      histo << totalTime   << "\t" << dt     << "\t" 
	    << massT       << "\t" << divB_T << "\n";
      
      histo.close();
    } // end myRank == 0

  } // HydroRunBaseMpi::history_mhd_default

  // =======================================================
  // =======================================================
  /*
   * history for MRI problem.
   */
  void HydroRunBaseMpi::history_mhd_mri(int nStep, real_t dt)
  {

    if (myRank==0)
      std::cout << "History for MRI problem at time " << totalTime << "\n";

    if (dimType == TWO_D) {
      // don't do anything
    } else {

      // open history file
      std::ofstream histo;
      
      if (myRank == 0) {
	
	// history file name
	std::string fileName = configMap.getString("history",
						   "filename", 
						   "history.txt");
	// get output prefix / outputDir
	std::string outputDir    = configMap.getString("output", "outputDir", "./");
	std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
	
	// build full path filename
	fileName = outputDir + "/" + outputPrefix + "_" + fileName;
	
	histo.open (fileName.c_str(), std::ios::out | std::ios::app | std::ios::ate); 
	
	// if this is the first time we call history, print header
	if (totalTime <= 0) {
	  histo << "# history " << current_date() << std::endl;
	
	  bool restartEnabled = configMap.getBool("run","restart",false);
	  if (restartEnabled)
	    histo << "# history : this is a restart run\n";
	  
	  // write header (which variables are dumped)
	  histo << "# totalTime dt mass maxwell reynolds maxwell+reynolds magp ";
	  histo << "mean_Bx mean_By mean_Bz divB\n";

	} // end print header

      } // end myRank == 0

      // make sure Device data are copied back onto Host memory
      // which data to save ?
      copyGpuToCpu(nStep);
      HostArray<real_t> &U = getDataHost(nStep);
      

      double mass = 0.0, magp = 0.0, maxwell = 0.0;
      double mean_Bx = 0.0, mean_By = 0.0, mean_Bz = 0.0;
      
      // do a local reduction
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    mass += U(i,j,k,ID);
	    magp += 0.25*SQR( U(i,j,k,IBX)+U(i+1,j  ,k  ,IBX) );
	    magp += 0.25*SQR( U(i,j,k,IBY)+U(i  ,j+1,k  ,IBY) );
	    magp += 0.25*SQR( U(i,j,k,IBZ)+U(i  ,j  ,k+1,IBZ) );
	    
	    maxwell -= 0.25 *
	      ( U(i,j,k,IBX) + U(i+1,j  ,k,IBX) ) * 
	      ( U(i,j,k,IBY) + U(i  ,j+1,k,IBY) );
	    
	    mean_Bx += U(i,j,k,IBX);
	    mean_By += U(i,j,k,IBY);
	    mean_Bz += U(i,j,k,IBZ);
	    
	  } // end for i
	} // end for j
      } // end for k
      
      double dTau = dx*dy*dz/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin)/
	(_gParams.zMax- _gParams.zMin);
      
      magp    = magp*dTau/2.;
      mass    = mass*dTau;
      maxwell = maxwell*dTau;
      mean_Bx = mean_Bx*dTau;
      mean_By = mean_By*dTau;
      mean_Bz = mean_Bz*dTau;
      
     
      /* 
       * compute Y-Z averages
       */
      HostArray<double> localMean, localMeanYZ; 
      // 3 field: rho, rhovx/rho, rhovy/rho
      localMean.  allocate( isize, 3 );
      localMeanYZ.allocate( isize, 3 );
      // reset 
      memset( localMean.  data(), 0, localMean.  sizeBytes() );
      memset( localMeanYZ.data(), 0, localMeanYZ.sizeBytes() );

      // average inside sub-domain
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=0; i<isize; i++) {

	    localMean(i,0) += U(i,j,k,ID);
	    localMean(i,1) += U(i,j,k,IU)/U(i,j,k,ID);
	    localMean(i,2) += U(i,j,k,IV)/U(i,j,k,ID);

	  } // end for i
	} // end for j
      } // end for k

      // perform average across MPI topology by gathering localMean arrays
      {
	
	double *tmpData = new double[isize*3*mx*my*mz];
	MPI_Allgather(localMean.data(), isize*3, MPI_DOUBLE, 
		      tmpData         , isize*3, MPI_DOUBLE,
		      communicator->getComm());

	// each MPI process has now a copy of every localMean
	for (int mpiProcNum=0; mpiProcNum<mx*my*mz; mpiProcNum++) {
	  
	  // get cartesian coordiante inside topology and
	  // check all pieces that come from the same X-coordinate to
	  // perform the YZ average
	  int mpiProcCoords[3];
	  communicator->getCoords(mpiProcNum, 3, mpiProcCoords); 

	  if (mpiProcCoords[0] == myMpiPos[0]) {
	    for (int i=0; i<isize; i++) {
	      localMeanYZ(i,0) += tmpData[mpiProcNum*3*isize+        i]; 
	      localMeanYZ(i,1) += tmpData[mpiProcNum*3*isize+  isize+i]; 
	      localMeanYZ(i,2) += tmpData[mpiProcNum*3*isize+2*isize+i]; 
	    }
	  }

	}

	delete[] tmpData;

      }

      for (int i=0; i<isize; i++) {
	localMeanYZ(i,0) /= (ny*my*nz*mz);
  	localMeanYZ(i,1) /= (ny*my*nz*mz);
 	localMeanYZ(i,2) /= (ny*my*nz*mz);
      }


      /*
       * compute Reynolds number and divB
       */
      double reynolds = 0.0;
      double divB = 0.0;
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    reynolds += U(i,j,k,ID) * dTau *
	      ( U(i,j,k,IU) / U(i,j,k,ID) - localMeanYZ(i,1) ) *
	      ( U(i,j,k,IV) / U(i,j,k,ID) - localMeanYZ(i,2) );
	    
	    divB +=  
	      ( U(i+1,j  ,k  ,IBX) - U(i,j,k,IBX) ) / dx + 
	      ( U(i  ,j+1,k  ,IBY) - U(i,j,k,IBY) ) / dy + 
	      ( U(i  ,j  ,k+1,IBZ) - U(i,j,k,IBZ) ) / dz;
	    
	  } // end for i
	} // end for j
      } // end for k

      // do volume average (MPI reduction)
      double massT = 0.0, maxwellT = 0.0, magpT = 0.0, divB_T;
      double reynoldsT = 0.0;
      double mean_B[3], mean_B_T[3];
      mean_B[0] = mean_Bx; mean_B_T[0] = 0.0;
      mean_B[1] = mean_By; mean_B_T[1] = 0.0;
      mean_B[2] = mean_Bz; mean_B_T[2] = 0.0;
      
      MPI_Reduce(&mass    ,&massT    ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&reynolds,&reynoldsT,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&maxwell ,&maxwellT ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&magp    ,&magpT    ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&divB    ,&divB_T   ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce( mean_B  , mean_B_T ,3,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      
      if (myRank == 0) {
	histo << totalTime   << "\t" << dt                 << "\t" 
	      << massT       << "\t" << maxwellT           << "\t" 
	      << reynoldsT   << "\t" << maxwellT+reynoldsT << "\t"
	      << magpT       << "\t" << mean_B_T[0]        << "\t"
	      << mean_B_T[1] << "\t" << mean_B_T[2]        << "\t"
	      << divB_T      << "\n";
	
	histo.close();
      } // end myRank == 0

    } // end THREE_D

  } // HydroRunBaseMpi::history_mhd_mri

  // =======================================================
  // =======================================================
  /*
   * history for turbulence problem in hydro.
   */
  void HydroRunBaseMpi::history_hydro_turbulence(int nStep, real_t dt)
  {

    if (myRank==0)
      std::cout << "History for turbulence problem at time " << totalTime << "\n";

    if (dimType == TWO_D) {

      // don't do anything (this problem is not available in 2D, maybe some day if usefull....)

    } else {

      // open history file
      std::ofstream histo;
      
      if (myRank == 0) {
	
	// history file name
	std::string fileName = configMap.getString("history",
						   "filename", 
						   "history.txt");
	// get output prefix / outputDir
	std::string outputDir    = configMap.getString("output", "outputDir", "./");
	std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
	
	// build full path filename
	fileName = outputDir + "/" + outputPrefix + "_" + fileName;
	
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
	  histo << "# totalTime dt mass eKin mean_rhovx mean_rhovy mean_rhovz Ma_s \n";

	} // end print header

      } // end myRank == 0

      // make sure Device data are copied back onto Host memory
      // which data to save ?
      copyGpuToCpu(nStep);
      HostArray<real_t> &U = getDataHost(nStep);
      
      //const double pi = 2*asin(1.0);
      
      double mass       = 0.0, eKin       = 0.0;
      double mean_rhovx = 0.0, mean_rhovy = 0.0, mean_rhovz = 0.0;
      double mean_v2    = 0.0;

      // do a local reduction
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    real_t rho = U(i,j,k,ID);
	    mass += rho;

	    eKin    += SQR( U(i,j,k,IU) ) / rho;
	    eKin    += SQR( U(i,j,k,IV) ) / rho;
	    eKin    += SQR( U(i,j,k,IW) ) / rho;
	    
	    mean_v2 += SQR( U(i,j,k,IU)/rho );
	    mean_v2 += SQR( U(i,j,k,IV)/rho );
	    mean_v2 += SQR( U(i,j,k,IW)/rho );

	    mean_rhovx += U(i,j,k,IU);
	    mean_rhovy += U(i,j,k,IV);
	    mean_rhovz += U(i,j,k,IW);
	    
	  } // end for i
	} // end for j
      } // end for k
      
      double dTau = dx*dy*dz/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin)/
	(_gParams.zMax- _gParams.zMin);
      
      mass    = mass*dTau;

      eKin    = eKin*dTau;

      mean_rhovx = mean_rhovx*dTau;
      mean_rhovy = mean_rhovy*dTau;
      mean_rhovz = mean_rhovz*dTau;

      mean_v2  = mean_v2*dTau;

      // do volume average (MPI reduction)
      double massT = 0.0;
      double eKinT = 0.0;
      double mean_v2T = 0.0;
      double mean_rhov[3], mean_rhov_T[3];
      mean_rhov[0] = mean_rhovx; mean_rhov_T[0] = 0.0;
      mean_rhov[1] = mean_rhovy; mean_rhov_T[1] = 0.0;
      mean_rhov[2] = mean_rhovz; mean_rhov_T[2] = 0.0;
      
      MPI_Reduce(&mass       ,&massT         ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&mean_v2    ,&mean_v2T      ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&eKin       ,&eKinT         ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce( mean_rhov  , mean_rhov_T   ,3,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      
      if (myRank == 0) {

	real_t &cIso = _gParams.cIso;
	double Ma_s = -1.0;
	if (cIso > 0)
	  Ma_s = sqrt(mean_v2T)/cIso; 
	
	histo << totalTime      << "\t" 
	      << dt             << "\t" 
	      << massT          << "\t" 
	      << eKinT          << "\t" 
	      << mean_rhov[0]   << "\t" 
	      << mean_rhov[1]   << "\t" 
	      << mean_rhov[2]   << "\t"
	      << Ma_s           << "\n";
	
	histo.close();

      } // end myRank == 0

    } // end THREE_D

    bool structureFunctionsEnabled = configMap.getBool("structureFunctions","enabled",false);
    if ( structureFunctionsEnabled ) {
      HostArray<real_t> &U = getDataHost(nStep);
      structure_functions_hydro_mpi(myRank,nStep,configMap,_gParams,U);
    }

  } // HydroRunBaseMpi::history_hydro_turbulence

  // =======================================================
  // =======================================================
  /*
   * history for turbulence problem in MHD.
   */
  void HydroRunBaseMpi::history_mhd_turbulence(int nStep, real_t dt)
  {

    if (myRank==0)
      std::cout << "History for turbulence problem at time " << totalTime << "\n";

    if (dimType == TWO_D) {

      // don't do anything (this problem is not available in 2D, maybe some day if usefull....)

    } else {

      // open history file
      std::ofstream histo;
      
      if (myRank == 0) {
	
	// history file name
	std::string fileName = configMap.getString("history",
						   "filename", 
						   "history.txt");
	// get output prefix / outputDir
	std::string outputDir    = configMap.getString("output", "outputDir", "./");
	std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
	
	// build full path filename
	fileName = outputDir + "/" + outputPrefix + "_" + fileName;
	
	histo.open (fileName.c_str(), std::ios::out | std::ios::app | std::ios::ate); 
	
	// if this is the first time we call history, print header
	if (totalTime <= 0) {
	  histo << "# history " << current_date() << std::endl;
	
	  bool restartEnabled = configMap.getBool("run","restart",false);
	  if (restartEnabled)
	    histo << "# history : this is a restart run\n";
	  
	  // write header (which variables are dumped)
	  // Ma_s is the sonic Mach number Ma_s = v_rms/c_s
	  // Ma_alfven is the alfvenic Mach number (v_rms / v_0A) where 
	  // v_rms = sqrt(<v^2) and v_0A = B_0/sqrt(4 pi rho_0) 
	  // helicity is the mean value < (rho_v) . (B/sqrt(rho)) >
	  histo << "# totalTime dt mass divB eKin eMag helicity mean_B mean_Bx mean_By mean_Bz mean_rhovx mean_rhovy mean_rhovz Ma_s Ma_alfven\n";

	} // end print header

      } // end myRank == 0

      // make sure Device data are copied back onto Host memory
      // which data to save ?
      copyGpuToCpu(nStep);
      HostArray<real_t> &U = getDataHost(nStep);
      
      //const double pi = 2*asin(1.0);
      
      double mass       = 0.0, eKin       = 0.0, eMag       = 0.0;
      double helicity   = 0.0;
      double mean_Bx    = 0.0, mean_By    = 0.0, mean_Bz    = 0.0;
      double mean_rhovx = 0.0, mean_rhovy = 0.0, mean_rhovz = 0.0;
      double mean_v2    = 0.0;

      // do a local reduction
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    real_t rho = U(i,j,k,ID);
	    //real_t bx  = U(i,j,k,IBX);
	    mass += rho;

	    eKin    += SQR( U(i,j,k,IU) ) / rho;
	    eKin    += SQR( U(i,j,k,IV) ) / rho;
	    eKin    += SQR( U(i,j,k,IW) ) / rho;
	    
	    mean_v2 += SQR( U(i,j,k,IU)/rho );
	    mean_v2 += SQR( U(i,j,k,IV)/rho );
	    mean_v2 += SQR( U(i,j,k,IW)/rho );

	    eMag    += SQR( U(i,j,k,IBX) );
	    eMag    += SQR( U(i,j,k,IBY) );
	    eMag    += SQR( U(i,j,k,IBZ) );

	    helicity += U(i,j,k,IU)*U(i,j,k,IBX)/sqrt(rho);
	    helicity += U(i,j,k,IV)*U(i,j,k,IBY)/sqrt(rho);
  	    helicity += U(i,j,k,IW)*U(i,j,k,IBZ)/sqrt(rho);

	    mean_Bx += U(i,j,k,IBX);
	    mean_By += U(i,j,k,IBY);
	    mean_Bz += U(i,j,k,IBZ);

	    mean_rhovx += U(i,j,k,IU);
	    mean_rhovy += U(i,j,k,IV);
	    mean_rhovz += U(i,j,k,IW);
	    
	  } // end for i
	} // end for j
      } // end for k
      
      double dTau = dx*dy*dz/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin)/
	(_gParams.zMax- _gParams.zMin);
      
      mass    = mass*dTau;

      eKin    = eKin*dTau;
      eMag    = eMag*dTau;
      helicity *= dTau;

      mean_Bx = mean_Bx*dTau;
      mean_By = mean_By*dTau;
      mean_Bz = mean_Bz*dTau;

      double mean_B_norm = sqrt( SQR(mean_Bx) + SQR(mean_By) + SQR(mean_Bz) );

      mean_rhovx = mean_rhovx*dTau;
      mean_rhovy = mean_rhovy*dTau;
      mean_rhovz = mean_rhovz*dTau;

      mean_v2  = mean_v2*dTau;

      /*
       * compute divB
       */
      double divB = 0.0;
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    divB +=  
	      ( U(i+1,j  ,k  ,IBX) - U(i,j,k,IBX) ) / dx + 
	      ( U(i  ,j+1,k  ,IBY) - U(i,j,k,IBY) ) / dy + 
	      ( U(i  ,j  ,k+1,IBZ) - U(i,j,k,IBZ) ) / dz;
	    
	  } // end for i
	} // end for j
      } // end for k

      // do volume average (MPI reduction)
      double massT = 0.0, divB_T = 0.0;
      double eKinT = 0.0, eMagT  = 0.0, helicityT = 0.0;
      double mean_v2T = 0.0;
      double mean_B[3], mean_B_T[3];
      mean_B[0] = mean_Bx; mean_B_T[0] = 0.0;
      mean_B[1] = mean_By; mean_B_T[1] = 0.0;
      mean_B[2] = mean_Bz; mean_B_T[2] = 0.0;
      double mean_B_norm_T = 0.0;
      double mean_rhov[3], mean_rhov_T[3];
      mean_rhov[0] = mean_rhovx; mean_rhov_T[0] = 0.0;
      mean_rhov[1] = mean_rhovy; mean_rhov_T[1] = 0.0;
      mean_rhov[2] = mean_rhovz; mean_rhov_T[2] = 0.0;
      
      MPI_Reduce(&mass       ,&massT         ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&mean_v2    ,&mean_v2T      ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&eKin       ,&eKinT         ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&eMag       ,&eMagT         ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&helicity   ,&helicityT     ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&divB       ,&divB_T        ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce(&mean_B_norm,&mean_B_norm_T ,1,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce( mean_B     , mean_B_T      ,3,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      MPI_Reduce( mean_rhov  , mean_rhov_T   ,3,MPI_DOUBLE,MPI_SUM,0,communicator->getComm());
      
      if (myRank == 0) {

	double Ma_alfven = sqrt(mean_v2T)/(mean_B_norm_T/sqrt(4*M_PI*massT));
	real_t &cIso = _gParams.cIso;
	double Ma_s = sqrt(mean_v2T)/cIso; 
	
	histo << totalTime      << "\t" << dt           << "\t" 
	      << massT          << "\t" << divB         << "\t" 
	      << eKinT          << "\t" << eMagT        << "\t"
	      << helicity       << "\t"
	      << mean_B_norm_T  << "\t"
	      << mean_B_T[0]    << "\t" << mean_B_T[1]  << "\t" << mean_B_T[2]  << "\t"
	      << mean_rhov[0]   << "\t" << mean_rhov[1] << "\t" << mean_rhov[2] << "\t"
	      << Ma_s       << "\t" << Ma_alfven  << "\n";
	
	histo.close();

      } // end myRank == 0

    } // end THREE_D
    
    bool structureFunctionsEnabled = configMap.getBool("structureFunctions","enabled",false);
    if ( structureFunctionsEnabled ) {
      HostArray<real_t> &U = getDataHost(nStep);
      structure_functions_mhd_mpi(myRank,nStep,configMap,_gParams,U);
    }

  } // HydroRunBaseMpi::history_mhd_turbulence


} // namespace hydroSimu
