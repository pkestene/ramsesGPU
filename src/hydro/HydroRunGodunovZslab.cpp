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
 * \file HydroRunGodunovZslab.cpp
 * \brief Implements class HydroRunGodunovZslab
 * 
 * 3D-only Euler equation solver on a cartesian grid using Godunov method
 * with approximate Riemann solver (z-slab method).
 *
 * \date September 11, 2012
 * \author P. Kestener
 *
 * $Id: HydroRunGodunovZslab.cpp 3450 2014-06-16 22:03:23Z pkestene $
 */
#include "HydroRunGodunovZslab.h"

// include CUDA kernel when necessary
#ifdef __CUDACC__
#include "godunov_unsplit_zslab.cuh"
#endif // __CUDACC__
#include "constoprim.h"
#include "riemann.h"
#include "trace.h"
#include "zSlabInfo.h"

//#include <sys/time.h> // for gettimeofday
#include "../utils/monitoring/Timer.h"
#include "../utils/monitoring/date.h"
#include <iomanip>
#include "ostream_fmt.h"

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // HydroRunGodunovZslab class methods body
  ////////////////////////////////////////////////////////////////////////////////

  HydroRunGodunovZslab::HydroRunGodunovZslab(ConfigMap &_configMap)
    : HydroRunBase(_configMap)
    ,zSlabNb(1)
    ,zSlabWidth(0)
    ,zSlabWidthG(0)
    ,unsplitVersion(1)
    ,dumpDataForDebugEnabled(false)
  {

    // choose unsplit implementation version
    unsplitVersion = configMap.getInteger("hydro","unsplitVersion", 1);
    if (unsplitVersion !=0 and unsplitVersion !=1 and unsplitVersion !=2) {
      std::cerr << "##################################################" << std::endl;
      std::cerr << "WARNING : you should review your parameter file   " << std::endl;
      std::cerr << "and set hydro/unsplitVersion to a valid number :  " << std::endl;
      std::cerr << " - 0, 1 and 2 are currently available for 2D/3D   " << std::endl;
      std::cerr << "Fall back to the default value : 1                " << std::endl;
      std::cerr << "##################################################" << std::endl;
      unsplitVersion = 1;
    }
    std::cout << "Using Hydro Godunov unsplit implementation version : " 
	      << unsplitVersion 
	      << std::endl;

    // how many z-slabs do we use ?
    zSlabNb = configMap.getInteger("implementation","zSlabNb",0);

    if (zSlabNb <= 0) {
      std::cerr << "ERROR ! You must provide the number of z-slabs (> 0) " << std::endl;
      std::cerr << "to use in parameter file !" << std::endl;
      std::cerr << "Section \"implementation\",\"zSlabNb\"" << std::endl;
    }

    // compute z-slab width
    zSlabWidth  = (ksize + zSlabNb - 1) / zSlabNb;
    zSlabWidthG = zSlabWidth + 2*ghostWidth; 
  
    std::cout << "Using " << zSlabNb << " z-slabs of width " << zSlabWidth 
	      << " (with ghost " << zSlabWidthG << ")" << std::endl;

    /*
     * allways allocate primitive variables array : h_Q / d_Q
     */
#ifdef __CUDACC__

    d_Q.allocate   (make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    
    // register data pointers
    _gParams.arrayList[A_Q]    = d_Q.data();
    
#else
    
    h_Q.allocate   (make_uint4(isize, jsize, zSlabWidthG, nbVar));

    // register data pointers
    _gParams.arrayList[A_Q]    = h_Q.data();

#endif
 
    /*
     * memory allocation specific to a given implementation version
     */
    if (unsplitVersion == 1) {

      // do memory allocation for extra array
#ifdef __CUDACC__
      d_qm_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      d_qm_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      d_qm_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      d_qp_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      d_qp_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      d_qp_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      
      // register data pointers
      _gParams.arrayList[A_QM_X] = d_qm_x.data();
      _gParams.arrayList[A_QM_Y] = d_qm_y.data();
      _gParams.arrayList[A_QM_Z] = d_qm_z.data();
      _gParams.arrayList[A_QP_X] = d_qp_x.data();
      _gParams.arrayList[A_QP_Y] = d_qp_y.data();
      _gParams.arrayList[A_QP_Z] = d_qp_z.data();
#else // CPU version
      h_qm_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      h_qm_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      h_qm_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      h_qp_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      h_qp_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      h_qp_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      
      // register data pointers
      _gParams.arrayList[A_QM_X] = h_qm_x.data();
      _gParams.arrayList[A_QM_Y] = h_qm_y.data();
      _gParams.arrayList[A_QM_Z] = h_qm_z.data();
      _gParams.arrayList[A_QP_X] = h_qp_x.data();
      _gParams.arrayList[A_QP_Y] = h_qp_y.data();
      _gParams.arrayList[A_QP_Z] = h_qp_z.data();
#endif // __CUDACC__
   
    } else if (unsplitVersion == 2) {

#ifdef __CUDACC__
      d_slope_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      d_slope_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      d_slope_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      d_qm.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      d_qp.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
      
      // register data pointers
      _gParams.arrayList[A_SLOPE_X] = d_slope_x.data();
      _gParams.arrayList[A_SLOPE_Y] = d_slope_y.data();
      _gParams.arrayList[A_SLOPE_Z] = d_slope_z.data();
      _gParams.arrayList[A_QM]      = d_qm.data();
      _gParams.arrayList[A_QP]      = d_qp.data();

#else // CPU version

      h_slope_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      h_slope_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      h_slope_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      h_qm.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      h_qp.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
      
      // register data pointers
      _gParams.arrayList[A_SLOPE_X] = h_slope_x.data();
      _gParams.arrayList[A_SLOPE_Y] = h_slope_y.data();
      _gParams.arrayList[A_SLOPE_Z] = h_slope_z.data();
      _gParams.arrayList[A_QM]      = h_qm.data();
      _gParams.arrayList[A_QP]      = h_qp.data();

#endif // __CUDACC__

    } // end unsplitVersion == 2

    // make sure variables declared as __constant__ are copied to device
    // for current compilation unit
    copyToSymbolMemory();

    /*
     * Total memory allocated logging.
     * Just for notice
     */
#ifdef __CUDACC__
    {
      size_t freeMemory, totalMemory;
      cutilSafeCall( cudaMemGetInfo(&freeMemory, &totalMemory) );
      std::cout << "Total memory available on GPU " << totalMemory/1000000. << " MBytes\n";
      std::cout << "Currently free  memory on GPU " <<  freeMemory/1000000. << " MBytes\n";
      std::cout << "Total memory allocated on GPU " << DeviceArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
      std::cout << "Total memory allocated on CPU " << HostArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
    }
#else
    {
      std::cout << "Total memory allocated on CPU " << HostArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
    }
#endif // __CUDACC__

  } // HydroRunGodunovZslab::HydroRunGodunovZslab

  // =======================================================
  // =======================================================
  HydroRunGodunovZslab::~HydroRunGodunovZslab()
  {  

  } // HydroRunGodunovZslab::~HydroRunGodunovZslab

  // =======================================================
  // =======================================================
  void HydroRunGodunovZslab::godunov_unsplit(int nStep, real_t dt)
#ifdef __CUDACC__
  {
    
    if ((nStep%2)==0) {
      godunov_unsplit_gpu(d_U , d_U2, dt, nStep);
    } else {
      godunov_unsplit_gpu(d_U2, d_U , dt, nStep);
    }
    
  } // HydroRunGodunovZslab::godunov_unsplit (GPU version)
#else // CPU version
  {
    
    if ((nStep%2)==0) {
      godunov_unsplit_cpu(h_U , h_U2, dt, nStep);
    } else {
      godunov_unsplit_cpu(h_U2, h_U , dt, nStep);
    }
    
  } // HydroRunGodunovZslab::godunov_unsplit (CPU version)
#endif // __CUDACC__
  
#ifdef __CUDACC__ 
  // =======================================================
  // =======================================================
  void HydroRunGodunovZslab::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
						 DeviceArray<real_t>& d_UNew,
						 real_t dt, int nStep)
  {

    // update boundaries
    TIMER_START(timerBoundaries);
    make_all_boundaries(d_UOld);
    TIMER_STOP(timerBoundaries);

    // copy d_UOld into d_UNew
    d_UOld.copyTo(d_UNew);

    // inner domain integration
    TIMER_START(timerGodunov);

    if (unsplitVersion == 0) {

      godunov_unsplit_gpu_v0(d_UOld, d_UNew, dt, nStep);

    } else if (unsplitVersion == 1) {

      godunov_unsplit_gpu_v1(d_UOld, d_UNew, dt, nStep);
    
    } else if (unsplitVersion == 2) {

      godunov_unsplit_gpu_v2(d_UOld, d_UNew, dt, nStep);
    
    } // end unsplitVersion == 2
    
    TIMER_STOP(timerGodunov);

  } // HydroRunGodunovZslab::godunov_unsplit_gpu

  // =======================================================
  // =======================================================
  void HydroRunGodunovZslab::godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
						    DeviceArray<real_t>& d_UNew,
						    real_t dt, int nStep)
  {

    // loop over z-slab index
    for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {
      
      ZslabInfo zSlabInfo;
      zSlabInfo.zSlabId     = zSlabId;
      zSlabInfo.zSlabNb     = zSlabNb;
      zSlabInfo.zSlabWidthG = zSlabWidthG;
      zSlabInfo.kStart      = zSlabWidth * zSlabId;
      zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;
      zSlabInfo.ksizeSlab   = zSlabWidthG;
      
      // take care that the last slab might be truncated
      if (zSlabId == zSlabNb-1) {
	zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
      }
      
      TIMER_START(timerPrimVar);
      {
	// 3D primitive variables computation kernel    
	dim3 dimBlock(PRIM_VAR_BLOCK_DIMX_3D_Z,
		      PRIM_VAR_BLOCK_DIMY_3D_Z);
	dim3 dimGrid(blocksFor(isize, PRIM_VAR_BLOCK_DIMX_3D_Z), 
		     blocksFor(jsize, PRIM_VAR_BLOCK_DIMY_3D_Z));
	kernel_hydro_compute_primitive_variables_3D_zslab<<<dimGrid, 
	  dimBlock>>>(d_UOld.data(), 
		      d_Q.data(),
		      d_UOld.pitch(),
		      d_UOld.dimx(),
		      d_UOld.dimy(),
		      d_UOld.dimz(),
		      zSlabInfo);
	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_primitive_variables_3D_zslab error");
	
      } // end compute primitive variables 3d kernel
      TIMER_STOP(timerPrimVar);

      {
	// 3D Godunov unsplit kernel    
	dim3 dimBlock(UNSPLIT_BLOCK_DIMX_3D_V0_Z,
		      UNSPLIT_BLOCK_DIMY_3D_V0_Z);
	dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_INNER_DIMX_3D_V0_Z),
		     blocksFor(jsize, UNSPLIT_BLOCK_INNER_DIMY_3D_V0_Z));
	kernel_godunov_unsplit_3d_v0_zslab<<<dimGrid, 
	  dimBlock>>>(d_Q.data(), 
		      d_UNew.data(), 
		      d_UOld.pitch(), 
		      d_UOld.dimx(), 
		      d_UOld.dimy(), 
		      d_UOld.dimz(),
		      dt / dx, 
		      dt / dy, 
		      dt / dz, 
		      dt,
		      gravityEnabled,
		      zSlabInfo);
	checkCudaError("HydroRunGodunovZslab :: kernel_godunov_unsplit_3d_v0_zslab error");

      }
      
      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt, zSlabInfo);
      }
      
    } // end for k inside z-slab
    
    /*************************************
     * DISSIPATIVE TERMS (i.e. viscosity)
     *************************************/
    TIMER_START(timerDissipative);
    real_t &nu = _gParams.nu;
    if (nu>0) {
      std::cerr << "Dissipative terms not implemented (TODO)..." << std::endl;
    } // end compute viscosity force / update
    TIMER_STOP(timerDissipative);
    
  } // HydroRunGodunovZslab::godunov_unsplit_gpu_v0

  // =======================================================
  // =======================================================
  void HydroRunGodunovZslab::godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
						    DeviceArray<real_t>& d_UNew,
						    real_t dt, int nStep)
  {

    // loop over z-slab index
    for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {

      ZslabInfo zSlabInfo;
      zSlabInfo.zSlabId     = zSlabId;
      zSlabInfo.zSlabNb     = zSlabNb;
      zSlabInfo.zSlabWidthG = zSlabWidthG;
      zSlabInfo.kStart      = zSlabWidth * zSlabId;
      zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;
      zSlabInfo.ksizeSlab   = zSlabWidthG;

      // take care that the last slab might be truncated
      if (zSlabId == zSlabNb-1) {
	zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
      }
      
      TIMER_START(timerPrimVar);
      {
	// 3D primitive variables computation kernel    
	dim3 dimBlock(PRIM_VAR_BLOCK_DIMX_3D_Z,
		      PRIM_VAR_BLOCK_DIMY_3D_Z);
	dim3 dimGrid(blocksFor(isize, PRIM_VAR_BLOCK_DIMX_3D_Z), 
		     blocksFor(jsize, PRIM_VAR_BLOCK_DIMY_3D_Z));
	kernel_hydro_compute_primitive_variables_3D_zslab<<<dimGrid, 
	  dimBlock>>>(d_UOld.data(), 
		      d_Q.data(),
		      d_UOld.pitch(),
		      d_UOld.dimx(),
		      d_UOld.dimy(),
		      d_UOld.dimz(),
		      zSlabInfo);
	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_primitive_variables_3D_zslab error");
	
      } // end compute primitive variables 3d kernel
      TIMER_STOP(timerPrimVar);
      
      TIMER_START(timerSlopeTrace);
      {
	// 3D slope / trace computation kernel
	dim3 dimBlock(TRACE_BLOCK_DIMX_3D_V1Z,
		      TRACE_BLOCK_DIMY_3D_V1Z);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_INNER_DIMX_3D_V1Z), 
		     blocksFor(jsize, TRACE_BLOCK_INNER_DIMY_3D_V1Z));
	kernel_hydro_compute_trace_unsplit_3d_v1_zslab<<<dimGrid, 
	  dimBlock>>>(d_UOld.data(),
		      d_Q.data(),
		      d_qm_x.data(),
		      d_qm_y.data(),
		      d_qm_z.data(),
		      d_qp_x.data(),
		      d_qp_y.data(),
		      d_qp_z.data(),
		      d_UOld.pitch(), 
		      d_UOld.dimx(), 
		      d_UOld.dimy(), 
		      d_UOld.dimz(),
		      dt / dx, 
		      dt / dy,
		      dt / dz,
		      dt,
		      zSlabInfo);
	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_trace_unsplit_3d_v1_zslab error");

	if (gravityEnabled) {
	  compute_gravity_predictor(d_qm_x, dt, zSlabInfo);
	  compute_gravity_predictor(d_qm_y, dt, zSlabInfo);
	  compute_gravity_predictor(d_qm_z, dt, zSlabInfo);
	  compute_gravity_predictor(d_qp_x, dt, zSlabInfo);
	  compute_gravity_predictor(d_qp_y, dt, zSlabInfo);
	  compute_gravity_predictor(d_qp_z, dt, zSlabInfo);
	}
	  
      } // end 3D slope / trace computation kernel
      TIMER_STOP(timerSlopeTrace);
      
      TIMER_START(timerUpdate);
      {
	// 3D update hydro kernel
	dim3 dimBlock(UPDATE_BLOCK_DIMX_3D_V1Z,
		      UPDATE_BLOCK_DIMY_3D_V1Z);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_3D_V1Z), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_3D_V1Z));
	kernel_hydro_flux_update_unsplit_3d_v1_zslab<<<dimGrid, 
	  dimBlock>>>(d_UOld.data(),
		      d_UNew.data(),
		      d_qm_x.data(),
		      d_qm_y.data(),
		      d_qm_z.data(),
		      d_qp_x.data(),
		      d_qp_y.data(),
		      d_qp_z.data(),
		      d_UOld.pitch(), 
		      d_UOld.dimx(), 
		      d_UOld.dimy(), 
		      d_UOld.dimz(),
		      dt / dx, 
		      dt / dy,
		      dt / dz,
		      dt,
		      zSlabInfo);
	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_flux_update_unsplit_3d_v1_zslab error");
	
      } // end 3D update hydro kernel

      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt, zSlabInfo);
      }

      TIMER_STOP(timerUpdate);
      
    } // end for zSlabId
    
    // debug
    // {
    // 	HostArray<real_t> h_debug; 
    // 	h_debug.allocate(make_uint4(isize,jsize,ksize,nbVar));
    // 	d_UNew.copyToHost(h_debug);
    // 	outputHdf5Debug(h_debug, "UNew_before_dissip_", nStep);
    // }

    /*************************************
     * DISSIPATIVE TERMS (i.e. viscosity)
     *************************************/
    TIMER_START(timerDissipative);
    real_t &nu = _gParams.nu;
    if (nu>0) {
      // update boundaries before dissipative terms computations
      make_all_boundaries(d_UNew);
    }
    
    // compute viscosity
    if (nu>0) {
      DeviceArray<real_t> &d_flux_x = d_qm_x;
      DeviceArray<real_t> &d_flux_y = d_qm_y;
      DeviceArray<real_t> &d_flux_z = d_qm_z;
      
      // copy d_UNew into d_UOld
      d_UNew.copyTo(d_UOld);

      ZslabInfo zSlabInfo;
      zSlabInfo.zSlabId     = -1;
      zSlabInfo.zSlabNb     = zSlabNb;
      zSlabInfo.zSlabWidthG = zSlabWidthG;
      zSlabInfo.kStart      = -1;
      zSlabInfo.kStop       = -1;
      zSlabInfo.ksizeSlab   = zSlabWidthG;

      // loop over z-slab index
      for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {

	zSlabInfo.zSlabId     = zSlabId;
	zSlabInfo.kStart      = zSlabWidth * zSlabId;
	zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;
	  
	compute_viscosity_flux(d_UOld, d_flux_x, d_flux_y, d_flux_z, dt, zSlabInfo);
	compute_hydro_update  (d_UNew, d_flux_x, d_flux_y, d_flux_z,     zSlabInfo);

      } // end for zSlabId

    } // end compute viscosity force / update  
    TIMER_STOP(timerDissipative);
      
    /*
     * random forcing
     */
    if (randomForcingEnabled) {
      
      real_t norm = compute_random_forcing_normalization(d_UNew, dt);
      
      add_random_forcing(d_UNew, dt, norm);
      
    } // end random forcing
    if (randomForcingOrnsteinUhlenbeckEnabled) {
	
      // add forcing field in real space
      pForcingOrnsteinUhlenbeck->add_forcing_field(d_UNew, dt);
	
    }

  } // HydroRunGodunovZslab::godunov_unsplit_gpu_v1

  // =======================================================
  // =======================================================
  void HydroRunGodunovZslab::godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
						    DeviceArray<real_t>& d_UNew,
						    real_t dt, int nStep)
  {

    // loop over z-slab index
    for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {
      
      ZslabInfo zSlabInfo;
      zSlabInfo.zSlabId     = zSlabId;
      zSlabInfo.zSlabNb     = zSlabNb;
      zSlabInfo.zSlabWidthG = zSlabWidthG;
      zSlabInfo.kStart      = zSlabWidth * zSlabId;
      zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;
      zSlabInfo.ksizeSlab   = zSlabWidthG;
      
      // take care that the last slab might be truncated
      if (zSlabId == zSlabNb-1) {
	zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
      }
      
      TIMER_START(timerPrimVar);
      {
	// 3D primitive variables computation kernel    
	dim3 dimBlock(PRIM_VAR_BLOCK_DIMX_3D_Z,
		      PRIM_VAR_BLOCK_DIMY_3D_Z);
	dim3 dimGrid(blocksFor(isize, PRIM_VAR_BLOCK_DIMX_3D_Z), 
		     blocksFor(jsize, PRIM_VAR_BLOCK_DIMY_3D_Z));
	kernel_hydro_compute_primitive_variables_3D_zslab<<<dimGrid, 
	  dimBlock>>>(d_UOld.data(), 
		      d_Q.data(),
		      d_UOld.pitch(),
		      d_UOld.dimx(),
		      d_UOld.dimy(),
		      d_UOld.dimz(),
		      zSlabInfo);
	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_primitive_variables_3D_zslab error");
	
      } // end compute primitive variables 3d kernel
      TIMER_STOP(timerPrimVar);

      /*
       * 1. Compute and store slopes
       */
      // 3D slopes
      {
	dim3 dimBlock(SLOPES_BLOCK_DIMX_3D_V2_Z,
		      SLOPES_BLOCK_DIMY_3D_V2_Z);
	dim3 dimGrid(blocksFor(isize, SLOPES_BLOCK_INNER_DIMX_3D_V2_Z), 
		     blocksFor(jsize, SLOPES_BLOCK_INNER_DIMY_3D_V2_Z));
	kernel_godunov_slopes_3d_v2_zslab<<<dimGrid,
	  dimBlock>>>(d_Q.data(),
		      d_slope_x.data(),
		      d_slope_y.data(),
		      d_slope_z.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(), 
		      d_Q.dimz(),
		      zSlabInfo);
      } // end slopes 3D

      /*
       * 2. compute reconstructed states along X interfaces
       */
      {
	dim3 dimBlock(TRACE_BLOCK_DIMX_3D_V2_Z,
		      TRACE_BLOCK_DIMY_3D_V2_Z);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_DIMX_3D_V2_Z), 
		     blocksFor(jsize, TRACE_BLOCK_DIMY_3D_V2_Z));
	kernel_godunov_trace_by_dir_3d_v2_zslab<<<dimGrid,
	  dimBlock>>>(d_Q.data(),
		      d_slope_x.data(),
		      d_slope_y.data(),
		      d_slope_z.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      d_Q.dimz(),
		      dt, dt / dx, dt / dy, dt / dz,
		      gravityEnabled,
		      IX,
		      zSlabInfo);
      } // end trace X

      /*
       * 3. Riemann solver at X interface and update
       */
      if (0) {
	dim3 dimBlock(UPDATE_BLOCK_DIMX_3D_V2_Z,
		      UPDATE_BLOCK_DIMY_3D_V2_Z);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_3D_V2_Z), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_3D_V2_Z));
	kernel_hydro_flux_update_unsplit_3d_v2_zslab<<<dimGrid,
	  dimBlock>>>(d_UNew.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      d_Q.dimz(),
		      dt / dx, dt / dy, dt / dz, dt,
		      IX,
		      zSlabInfo);
      
      } // end update X

      /*
       * 4. compute reconstructed states along Y interfaces
       */
      {
	dim3 dimBlock(TRACE_BLOCK_DIMX_3D_V2_Z,
		      TRACE_BLOCK_DIMY_3D_V2_Z);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_DIMX_3D_V2_Z), 
		     blocksFor(jsize, TRACE_BLOCK_DIMY_3D_V2_Z));
	kernel_godunov_trace_by_dir_3d_v2_zslab<<<dimGrid,
	  dimBlock>>>(d_Q.data(),
		      d_slope_x.data(),
		      d_slope_y.data(),
		      d_slope_z.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      d_Q.dimz(),
		      dt, dt / dx, dt / dy, dt / dz,
		      gravityEnabled,
		      IY,
		      zSlabInfo);
      } // end trace Y

      /*
       * 5. Riemann solver at Y interface and update
       */
      if (0) {
	dim3 dimBlock(UPDATE_BLOCK_DIMX_3D_V2_Z,
		      UPDATE_BLOCK_DIMY_3D_V2_Z);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_3D_V2_Z), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_3D_V2_Z));
	kernel_hydro_flux_update_unsplit_3d_v2_zslab<<<dimGrid,
	  dimBlock>>>(d_UNew.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      d_Q.dimz(),
		      dt / dx, dt / dy, dt / dz, dt,
		      IY,
		      zSlabInfo);
      
      } // end update Y

      /*
       * 6. compute reconstructed states along Z interfaces
       */
      {
	dim3 dimBlock(TRACE_BLOCK_DIMX_3D_V2_Z,
		      TRACE_BLOCK_DIMY_3D_V2_Z);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_DIMX_3D_V2_Z), 
		     blocksFor(jsize, TRACE_BLOCK_DIMY_3D_V2_Z));
	kernel_godunov_trace_by_dir_3d_v2_zslab<<<dimGrid,
	  dimBlock>>>(d_Q.data(),
		      d_slope_x.data(),
		      d_slope_y.data(),
		      d_slope_z.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      d_Q.dimz(),
		      dt, dt / dx, dt / dy, dt / dz,
		      gravityEnabled,
		      IZ,
		      zSlabInfo);
      } // end trace Z

      /*
       * 7. Riemann solver at Z interface and update
       */
      if (0) {
	dim3 dimBlock(UPDATE_BLOCK_DIMX_3D_V2_Z,
		      UPDATE_BLOCK_DIMY_3D_V2_Z);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_3D_V2_Z), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_3D_V2_Z));
	kernel_hydro_flux_update_unsplit_3d_v2_zslab<<<dimGrid,
	  dimBlock>>>(d_UNew.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      d_Q.dimz(),
		      dt / dx, dt / dy, dt / dz, dt,
		      IZ,
		      zSlabInfo);
      
      } // end update Z

      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt, zSlabInfo);
      }

    } // end for loop zSlabId

    /*************************************
     * DISSIPATIVE TERMS (i.e. viscosity)
     *************************************/
    TIMER_START(timerDissipative);
    real_t &nu = _gParams.nu;
    if (nu>0) {
      // update boundaries before dissipative terms computations
      make_all_boundaries(d_UNew);
    }
    
    // compute viscosity
    if (nu>0) {
      // re-use slopes arrays
      DeviceArray<real_t> &d_flux_x = d_slope_x;
      DeviceArray<real_t> &d_flux_y = d_slope_y;
      DeviceArray<real_t> &d_flux_z = d_slope_z;
      
      // copy d_UNew into d_UOld
      d_UNew.copyTo(d_UOld);

      ZslabInfo zSlabInfo;
      zSlabInfo.zSlabId     = -1;
      zSlabInfo.zSlabNb     = zSlabNb;
      zSlabInfo.zSlabWidthG = zSlabWidthG;
      zSlabInfo.kStart      = -1;
      zSlabInfo.kStop       = -1;
      zSlabInfo.ksizeSlab   = zSlabWidthG;

      // loop over z-slab index
      for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {

	zSlabInfo.zSlabId     = zSlabId;
	zSlabInfo.kStart      = zSlabWidth * zSlabId;
	zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;
	  
	compute_viscosity_flux(d_UOld, d_flux_x, d_flux_y, d_flux_z, dt, zSlabInfo);
	compute_hydro_update  (d_UNew, d_flux_x, d_flux_y, d_flux_z,     zSlabInfo);

      } // end for zSlabId

    } // end compute viscosity force / update  
    TIMER_STOP(timerDissipative);
      
    /*
     * random forcing
     */
    if (randomForcingEnabled) {
      
      real_t norm = compute_random_forcing_normalization(d_UNew, dt);
      
      add_random_forcing(d_UNew, dt, norm);
      
    } // end random forcing
    if (randomForcingOrnsteinUhlenbeckEnabled) {
	
      // add forcing field in real space
      pForcingOrnsteinUhlenbeck->add_forcing_field(d_UNew, dt);
	
    }

  } // HydroRunGodunovZslab::godunov_unsplit_gpu_v2

#else // CPU version

  // =======================================================
  // =======================================================
  void HydroRunGodunovZslab::godunov_unsplit_cpu(HostArray<real_t>& h_UOld, 
						 HostArray<real_t>& h_UNew, 
						 real_t dt, int nStep)
  {

    (void) nStep;

    TIMER_START(timerBoundaries);
    make_all_boundaries(h_UOld);
    TIMER_STOP(timerBoundaries);

    // copy h_UOld into h_UNew
    // for (int indexGlob=0; indexGlob<h_UOld.size(); indexGlob++) {
    //   h_UNew(indexGlob) = h_UOld(indexGlob);
    // }
    h_UOld.copyTo(h_UNew);

    TIMER_START(timerGodunov);

    if (unsplitVersion == 0) {

      godunov_unsplit_cpu_v0(h_UOld, h_UNew, dt, nStep);

    } else if (unsplitVersion == 1) {

      godunov_unsplit_cpu_v1(h_UOld, h_UNew, dt, nStep);
    
    } else if (unsplitVersion == 2) {

      godunov_unsplit_cpu_v2(h_UOld, h_UNew, dt, nStep);
    
    } // end unsplitVersion == 2

    TIMER_STOP(timerGodunov);

  } // HydroRunGodunovZslab::godunov_unsplit_cpu
  
  // =======================================================
  // =======================================================
  void HydroRunGodunovZslab::godunov_unsplit_cpu_v0(HostArray<real_t>& h_UOld, 
						    HostArray<real_t>& h_UNew, 
						    real_t dt, int nStep)
  {
    
    (void) nStep;

    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;

    // loop over z-slab index
    for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {
      
      // start and stop global index of current slab (ghosts included)
      int kStart    = zSlabWidth * zSlabId;
      //int kStop     = zSlabWidth * zSlabId + zSlabWidthG;
      int ksizeSlab = zSlabWidthG;

      // take care that the last slab might be truncated
      if (zSlabId == zSlabNb-1) {
	ksizeSlab = ksize-kStart;
      }

      /*std::cout << "zSlabId " << zSlabId << " kStart " << kStart
	<< " kStop " << kStop << " ksizeSlab " << ksizeSlab << std::endl; */

      TIMER_START(timerPrimVar);
      // convert conservative to primitive variables (and source term predictor)
      // put results in h_Q object
      convertToPrimitives( h_UOld.data(), zSlabId );
      TIMER_STOP(timerPrimVar);

      int ksizeSlabStopUpdate = ksizeSlab-ghostWidth;
      if (zSlabId == zSlabNb-1) ksizeSlabStopUpdate += 1;

      for (int k=2; k<ksizeSlab-1; k++) {

	// z plane in global domain
	int kG = k+kStart;

	for (int j=2; j<jsize-1; j++) {
	  for (int i=2; i<isize-1; i++) {
	    
	    // primitive variables (local array)
	    real_t qLoc[NVAR_3D];
	    real_t qLocNeighbor[NVAR_3D];
	    real_t qNeighbors[2*THREE_D][NVAR_3D];
	    
	    // slopes
	    real_t dq[THREE_D][NVAR_3D];
	    real_t dqNeighbor[THREE_D][NVAR_3D];

	    // reconstructed state on cell faces
	    // aka riemann solver input
	    real_t qleft[NVAR_3D];
	    real_t qright[NVAR_3D];
	    
	    // riemann solver output
	    real_t qgdnv[NVAR_3D];
	    real_t flux[NVAR_3D];
	    real_t (&flux_x)[NVAR_3D] = flux;
	    real_t (&flux_y)[NVAR_3D] = flux;
	    real_t (&flux_z)[NVAR_3D] = flux;

	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    // deal with left interface along X !
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    
	    // get primitive variables state vector
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLoc[iVar]          = h_Q(i  ,j  ,k  ,iVar);

	      qNeighbors[0][iVar] = h_Q(i+1,j  ,k  ,iVar);
	      qNeighbors[1][iVar] = h_Q(i-1,j  ,k  ,iVar);
	      qNeighbors[2][iVar] = h_Q(i  ,j+1,k  ,iVar);
	      qNeighbors[3][iVar] = h_Q(i  ,j-1,k  ,iVar);
	      qNeighbors[4][iVar] = h_Q(i  ,j  ,k+1,iVar);
	      qNeighbors[5][iVar] = h_Q(i  ,j  ,k-1,iVar);
	  
	    } // end for iVar
	
	    // compute slopes in current cell
	    slope_unsplit_3d(qLoc, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     qNeighbors[4],
			     qNeighbors[5],
			     dq);
	
	    // get primitive variables state vector in left neighbor along X
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLocNeighbor[iVar]  = h_Q(i-1,j  ,k  ,iVar);

	      qNeighbors[0][iVar] = h_Q(i  ,j  ,k  ,iVar);
	      qNeighbors[1][iVar] = h_Q(i-2,j  ,k  ,iVar);
	      qNeighbors[2][iVar] = h_Q(i-1,j+1,k  ,iVar);
	      qNeighbors[3][iVar] = h_Q(i-1,j-1,k  ,iVar);
	      qNeighbors[4][iVar] = h_Q(i-1,j  ,k+1,iVar);
	      qNeighbors[5][iVar] = h_Q(i-1,j  ,k-1,iVar);
	  
	    } // end for iVar
	
	    // compute slopes in left neighbor along X
	    slope_unsplit_3d(qLocNeighbor, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     qNeighbors[4],
			     qNeighbors[5],
			     dqNeighbor);
	    
	    //
	    // Compute reconstructed states at left interface along X
	    // in current cell
	    //
	    
	    // left interface : right state
	    trace_unsplit_hydro_3d_by_direction(qLoc, 
						dq, 
						dtdx, dtdy, dtdz,
						FACE_XMIN, 
						qright);
	    
	    // left interface : left state
	    trace_unsplit_hydro_3d_by_direction(qLocNeighbor,
						dqNeighbor,
						dtdx, dtdy, dtdz,
						FACE_XMAX, 
						qleft);
	    
	    if (gravityEnabled) { 
	      
	      // we need to modify input to flux computation with
	      // gravity predictor (half time step)
	      
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,kG,IZ);
	      
	      qright[IU] += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,kG,IZ);
	      
	    } // end gravityEnabled
	    
	    // Solve Riemann problem at X-interfaces and compute X-fluxes    
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_x);
	    
	    /*
	     * update with flux_x
	     */
	    if ( i  > ghostWidth           and 
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kG < ksize-ghostWidth) {
	      h_UNew(i-1,j  ,kG  ,ID) -= flux_x[ID]*dtdx;
	      h_UNew(i-1,j  ,kG  ,IP) -= flux_x[IP]*dtdx;
	      h_UNew(i-1,j  ,kG  ,IU) -= flux_x[IU]*dtdx;
	      h_UNew(i-1,j  ,kG  ,IV) -= flux_x[IV]*dtdx;
	      h_UNew(i-1,j  ,kG  ,IW) -= flux_x[IW]*dtdx;
	    }
	    
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kG < ksize-ghostWidth) {
	      h_UNew(i  ,j  ,kG  ,ID) += flux_x[ID]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IP) += flux_x[IP]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IU) += flux_x[IU]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IV) += flux_x[IV]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IW) += flux_x[IW]*dtdx;
	    }

	    
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    // deal with left interface along Y !
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    
	    // get primitive variables state vector in left neighbor along Y
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLocNeighbor[iVar]  = h_Q(i  ,j-1,k  ,iVar);
	      qNeighbors[0][iVar] = h_Q(i+1,j-1,k  ,iVar);
	      qNeighbors[1][iVar] = h_Q(i-1,j-1,k  ,iVar);
	      qNeighbors[2][iVar] = h_Q(i  ,j  ,k  ,iVar);
	      qNeighbors[3][iVar] = h_Q(i  ,j-2,k  ,iVar);
	      qNeighbors[4][iVar] = h_Q(i  ,j-1,k+1,iVar);
	      qNeighbors[5][iVar] = h_Q(i  ,j-1,k-1,iVar);
	      
	    } // end for iVar
	    
	    // compute slopes in left neighbor along Y
	    slope_unsplit_3d(qLocNeighbor, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     qNeighbors[4],
			     qNeighbors[5],
			     dqNeighbor);
	    
	    //
	    // Compute reconstructed states at left interface along Y 
	    // in current cell
	    //
	    
	    // left interface : right state
	    trace_unsplit_hydro_3d_by_direction(qLoc, 
						dq, 
						dtdx, dtdy, dtdz,
						FACE_YMIN, 
						qright);
	    
	    // left interface : left state
	    trace_unsplit_hydro_3d_by_direction(qLocNeighbor,
						dqNeighbor,
						dtdx, dtdy, dtdz,
						FACE_YMAX, 
						qleft);
	    
	    if (gravityEnabled) { 
	      // we need to modify input to flux computation with
	      // gravity predictor (half time step)
	      
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,kG,IZ);
	      
	      qright[IU] += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,kG,IZ);
	      
	    } // end gravityEnabled
	    
	    // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	    swapValues(&(qleft[IU]) ,&(qleft[IV]) );
	    swapValues(&(qright[IU]),&(qright[IV]));
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_y);
	    
	    /*
	     * update with flux_y
	     */
	    if ( i  < isize-ghostWidth     and 
		 j  > ghostWidth           and
		 k  < ksizeSlab-ghostWidth and
		 kG < ksize-ghostWidth) {
	      h_UNew(i  ,j-1,kG  ,ID) -= flux_y[ID]*dtdx;
	      h_UNew(i  ,j-1,kG  ,IP) -= flux_y[IP]*dtdx;
	      h_UNew(i  ,j-1,kG  ,IU) -= flux_y[IV]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,kG  ,IV) -= flux_y[IU]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,kG  ,IW) -= flux_y[IW]*dtdx;
	    }
	    
	    if ( i  < isize-ghostWidth     and
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kG < ksize-ghostWidth) {
	      h_UNew(i  ,j  ,kG  ,ID) += flux_y[ID]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IP) += flux_y[IP]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IU) += flux_y[IV]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,kG  ,IV) += flux_y[IU]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,kG  ,IW) += flux_y[IW]*dtdx;
	    }
	    
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    // deal with left interface along Z !
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    
	    // get primitive variables state vector in left neighbor along Z
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      
	      qLocNeighbor[iVar]  = h_Q(i  ,j  ,k-1,iVar);
	      qNeighbors[0][iVar] = h_Q(i+1,j  ,k-1,iVar);
	      qNeighbors[1][iVar] = h_Q(i-1,j  ,k-1,iVar);
	      qNeighbors[2][iVar] = h_Q(i  ,j+1,k-1,iVar);
	      qNeighbors[3][iVar] = h_Q(i  ,j-1,k-1,iVar);
	      qNeighbors[4][iVar] = h_Q(i  ,j  ,k  ,iVar);
	      qNeighbors[5][iVar] = h_Q(i  ,j  ,k-2,iVar);
	      
	    } // end for iVar
	    
	    // compute slopes in left neighbor along Z
	    slope_unsplit_3d(qLocNeighbor, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     qNeighbors[4],
			     qNeighbors[5],
			     dqNeighbor);

	    //
	    // Compute reconstructed states at left interface along Z
	    // in current cell
	    //
	    
	    // left interface : right state
	    trace_unsplit_hydro_3d_by_direction(qLoc, 
						dq, 
						dtdx, dtdy, dtdz,
						FACE_ZMIN, 
						qright);
	    
	    // left interface : left state
	    trace_unsplit_hydro_3d_by_direction(qLocNeighbor,
						dqNeighbor,
						dtdx, dtdy, dtdz,
						FACE_ZMAX, 
						qleft);
	    
	    if (gravityEnabled) { 
	      // we need to modify input to flux computation with
	      // gravity predictor (half time step)
	      
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,kG,IZ);
	      
	      qright[IU] += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,kG,IZ);
	      
	    } // end gravityEnabled
	    
	    // Solve Riemann problem at Y-interfaces and compute Z-fluxes
	    swapValues(&(qleft[IU]) ,&(qleft[IW]) );
	    swapValues(&(qright[IU]),&(qright[IW]));
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_z);

	    /*
	     * update with flux_z
	     */
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  > ghostWidth           and
		 kG > ghostWidth) {
	      h_UNew(i  ,j  ,kG-1,ID) -= flux_z[ID]*dtdx;
	      h_UNew(i  ,j  ,kG-1,IP) -= flux_z[IP]*dtdx;
	      h_UNew(i  ,j  ,kG-1,IU) -= flux_z[IW]*dtdx; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,kG-1,IV) -= flux_z[IV]*dtdx;
	      h_UNew(i  ,j  ,kG-1,IW) -= flux_z[IU]*dtdx; // watchout IU and IW swapped
	    }
	    
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth     and
		 kG < ksize-ghostWidth) {
	      h_UNew(i  ,j  ,kG  ,ID) += flux_z[ID]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IP) += flux_z[IP]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IU) += flux_z[IW]*dtdx; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,kG  ,IV) += flux_z[IV]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IW) += flux_z[IU]*dtdx; // watchout IU and IW swapped
	    }
	  } // end for i
	} // end for j
      } // end for k

      // gravity source term
      if (gravityEnabled) {
	ZslabInfo zSlabInfo;
	zSlabInfo.zSlabId     = zSlabId;
	zSlabInfo.zSlabNb     = zSlabNb;
	zSlabInfo.zSlabWidthG = zSlabWidthG;
	zSlabInfo.kStart      = zSlabWidth * zSlabId;
	zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;
	zSlabInfo.ksizeSlab   = zSlabWidthG;
	
	compute_gravity_source_term(h_UNew, h_UOld, dt, zSlabInfo);
      }

    } // end loop over z-slab

    /*************************************
     * DISSIPATIVE TERMS (i.e. viscosity)
     *************************************/
    TIMER_START(timerDissipative);
    real_t &nu = _gParams.nu;
    if (nu>0) {
      std::cerr << "Dissipative terms not implemented (TODO)..." << std::endl;
    } // end compute viscosity force / update
    TIMER_STOP(timerDissipative);

  } // HydroRunGodunovZslab::godunov_unsplit_cpu_v0

  // =======================================================
  // =======================================================
  void HydroRunGodunovZslab::godunov_unsplit_cpu_v1(HostArray<real_t>& h_UOld, 
						    HostArray<real_t>& h_UNew,
						    real_t dt, int nStep)
  {
    
    (void) nStep;
  
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;
    
    // loop over z-slab index
    for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {
      
      // start and stop global index of current slab (ghosts included)
      int kStart    = zSlabWidth * zSlabId;
      //int kStop     = zSlabWidth * zSlabId + zSlabWidthG;
      int ksizeSlab = zSlabWidthG;

      // take care that the last slab might be truncated
      if (zSlabId == zSlabNb-1) {
	ksizeSlab = ksize-kStart;
      }

      /*std::cout << "zSlabId " << zSlabId << " kStart " << kStart
	<< " kStop " << kStop << " ksizeSlab " << ksizeSlab << std::endl; */

      TIMER_START(timerPrimVar);
      // convert conservative to primitive variables (and source term predictor)
      // put results in h_Q object
      convertToPrimitives( h_UOld.data(), zSlabId );
      TIMER_STOP(timerPrimVar);
      
      TIMER_START(timerSlopeTrace);
      // call trace computation routine
      for (int k=1; k<ksizeSlab-1; k++) {

	for (int j=1; j<jsize-1; j++) {
	  for (int i=1; i<isize-1; i++) {
	    
	    real_t q[NVAR_3D];
	    real_t qPlusX  [NVAR_3D], qMinusX [NVAR_3D],
	      qPlusY  [NVAR_3D], qMinusY [NVAR_3D],
	      qPlusZ  [NVAR_3D], qMinusZ [NVAR_3D];
	    real_t dq[3][NVAR_3D];
	    
	    real_t qm[THREE_D][NVAR_3D];
	    real_t qp[THREE_D][NVAR_3D];
	    
	    // get primitive variables state vector
	    for (int iVar=0; iVar<NVAR_3D; iVar++) {
	      q      [iVar] = h_Q(i  ,j  ,k  , iVar);
	      qPlusX [iVar] = h_Q(i+1,j  ,k  , iVar);
	      qMinusX[iVar] = h_Q(i-1,j  ,k  , iVar);
	      qPlusY [iVar] = h_Q(i  ,j+1,k  , iVar);
	      qMinusY[iVar] = h_Q(i  ,j-1,k  , iVar);
	      qPlusZ [iVar] = h_Q(i  ,j  ,k+1, iVar);
	      qMinusZ[iVar] = h_Q(i  ,j  ,k-1, iVar);
	    }
	    
	    // get hydro slopes dq
	    slope_unsplit_3d(q, 
			     qPlusX, qMinusX, 
			     qPlusY, qMinusY, 
			     qPlusZ, qMinusZ,
			     dq);
	    
	    // compute qm, qp
	    trace_unsplit_hydro_3d(q, dq, 
				   dtdx, dtdy, dtdz,
				   qm, qp);
	    
	    // gravity predictor / modify velocity components
	    if (gravityEnabled) { 
	      int kG = k + kStart;

	      real_t grav_x = HALF_F * dt * h_gravity(i,j,kG,IX);
	      real_t grav_y = HALF_F * dt * h_gravity(i,j,kG,IY);
	      real_t grav_z = HALF_F * dt * h_gravity(i,j,kG,IZ);
		
	      qm[0][IU] += grav_x;
	      qm[0][IV] += grav_y;
	      qm[0][IW] += grav_z;
		
	      qp[0][IU] += grav_x;
	      qp[0][IV] += grav_y;
	      qp[0][IW] += grav_z;
		
	      qm[1][IU] += grav_x;
	      qm[1][IV] += grav_y;
	      qm[1][IW] += grav_z;
		
	      qp[1][IU] += grav_x;
	      qp[1][IV] += grav_y;
	      qp[1][IW] += grav_z;
		
	      qm[2][IU] += grav_x;
	      qm[2][IV] += grav_y;
	      qm[2][IW] += grav_z;
		
	      qp[2][IU] += grav_x;
	      qp[2][IV] += grav_y;
	      qp[2][IW] += grav_z;
	    } // end gravityEnabled
	      
	      // store qm, qp : only what is really needed
	    for (int ivar=0; ivar<NVAR_3D; ivar++) {
	      h_qm_x(i,j,k,ivar) = qm[0][ivar];
	      h_qp_x(i,j,k,ivar) = qp[0][ivar];
	      h_qm_y(i,j,k,ivar) = qm[1][ivar];
	      h_qp_y(i,j,k,ivar) = qp[1][ivar];
	      h_qm_z(i,j,k,ivar) = qm[2][ivar];
	      h_qp_z(i,j,k,ivar) = qp[2][ivar];	      
	    } // end for ivar
	    
	  } // end for i
	} // end for j
      } // end for k
      TIMER_STOP(timerSlopeTrace);
      
      TIMER_START(timerUpdate);
      // Finally compute fluxes from rieman solvers, and update

      int ksizeSlabStopUpdate = ksizeSlab-ghostWidth;
      if (zSlabId == zSlabNb-1) ksizeSlabStopUpdate += 1;

      for (int k=ghostWidth; k<ksizeSlab-ghostWidth+1; k++) {

	// z plane in global domain
	int kU = k+kStart;

	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	    
	    real_riemann_t qleft[NVAR_3D];
	    real_riemann_t qright[NVAR_3D];
	    real_riemann_t flux_x[NVAR_3D];
	    real_riemann_t flux_y[NVAR_3D];
	    real_riemann_t flux_z[NVAR_3D];
	    real_riemann_t qgdnv[NVAR_3D];
	    
	    /*
	     * Solve Riemann problem at X-interfaces and compute
	     * X-fluxes
	     */
	    qleft[ID]   = h_qm_x(i-1,j,k,ID);
	    qleft[IP]   = h_qm_x(i-1,j,k,IP);
	    qleft[IU]   = h_qm_x(i-1,j,k,IU);
	    qleft[IV]   = h_qm_x(i-1,j,k,IV);
	    qleft[IW]   = h_qm_x(i-1,j,k,IW);
	    
	    qright[ID]  = h_qp_x(i  ,j,k,ID);
	    qright[IP]  = h_qp_x(i  ,j,k,IP);
	    qright[IU]  = h_qp_x(i  ,j,k,IU);
	    qright[IV]  = h_qp_x(i  ,j,k,IV);
	    qright[IW]  = h_qp_x(i  ,j,k,IW);
	    
	    // compute hydro flux_x
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_x);
	    
	    /*
	     * Solve Riemann problem at Y-interfaces and compute Y-fluxes
	     */
	    qleft[ID]   = h_qm_y(i,j-1,k,ID);
	    qleft[IP]   = h_qm_y(i,j-1,k,IP);
	    qleft[IU]   = h_qm_y(i,j-1,k,IV); // watchout IU, IV permutation
	    qleft[IV]   = h_qm_y(i,j-1,k,IU); // watchout IU, IV permutation
	    qleft[IW]   = h_qm_y(i,j-1,k,IW);
	    
	    qright[ID]  = h_qp_y(i,j  ,k,ID);
	    qright[IP]  = h_qp_y(i,j  ,k,IP);
	    qright[IU]  = h_qp_y(i,j  ,k,IV); // watchout IU, IV permutation
	    qright[IV]  = h_qp_y(i,j  ,k,IU); // watchout IU, IV permutation
	    qright[IW]  = h_qp_y(i,j  ,k,IW);
	    
	    // compute hydro flux_y
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_y);
	    
	    /*
	     * Solve Riemann problem at Z-interfaces and compute
	     * Z-fluxes
	     */
	    qleft[ID]   = h_qm_z(i,j,k-1,ID);
	    qleft[IP]   = h_qm_z(i,j,k-1,IP);
	    qleft[IU]   = h_qm_z(i,j,k-1,IW); // watchout IU, IW permutation
	    qleft[IV]   = h_qm_z(i,j,k-1,IV);
	    qleft[IW]   = h_qm_z(i,j,k-1,IU); // watchout IU, IW permutation
	    
	    qright[ID]  = h_qp_z(i,j,k  ,ID);
	    qright[IP]  = h_qp_z(i,j,k  ,IP);
	    qright[IU]  = h_qp_z(i,j,k  ,IW); // watchout IU, IW permutation
	    qright[IV]  = h_qp_z(i,j,k  ,IV);
	    qright[IW]  = h_qp_z(i,j,k  ,IU); // watchout IU, IW permutation
	    
	    // compute hydro flux_z
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_z);
	    
	    /*
	     * update hydro array
	     */

	    /*
	     * update with flux_x
	     */
	    if ( i  > ghostWidth           and 
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i-1,j  ,kU  ,ID) -= flux_x[ID]*dtdx;
	      h_UNew(i-1,j  ,kU  ,IP) -= flux_x[IP]*dtdx;
	      h_UNew(i-1,j  ,kU  ,IU) -= flux_x[IU]*dtdx;
	      h_UNew(i-1,j  ,kU  ,IV) -= flux_x[IV]*dtdx;
	      h_UNew(i-1,j  ,kU  ,IW) -= flux_x[IW]*dtdx;
	    }
	      
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,kU  ,ID) += flux_x[ID]*dtdx;
	      h_UNew(i  ,j  ,kU  ,IP) += flux_x[IP]*dtdx;
	      h_UNew(i  ,j  ,kU  ,IU) += flux_x[IU]*dtdx;
	      h_UNew(i  ,j  ,kU  ,IV) += flux_x[IV]*dtdx;
	      h_UNew(i  ,j  ,kU  ,IW) += flux_x[IW]*dtdx;
	    }

	    /*
	     * update with flux_y
	     */
	    if ( i  < isize-ghostWidth     and 
		 j  > ghostWidth           and
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i  ,j-1,kU  ,ID) -= flux_y[ID]*dtdy;
	      h_UNew(i  ,j-1,kU  ,IP) -= flux_y[IP]*dtdy;
	      h_UNew(i  ,j-1,kU  ,IU) -= flux_y[IV]*dtdy; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,kU  ,IV) -= flux_y[IU]*dtdy; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,kU  ,IW) -= flux_y[IW]*dtdy;
	    }
	      
	    if ( i  < isize-ghostWidth     and
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth     ) {
	      h_UNew(i  ,j  ,kU  ,ID) += flux_y[ID]*dtdy;
	      h_UNew(i  ,j  ,kU  ,IP) += flux_y[IP]*dtdy;
	      h_UNew(i  ,j  ,kU  ,IU) += flux_y[IV]*dtdy; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,kU  ,IV) += flux_y[IU]*dtdy; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,kU  ,IW) += flux_y[IW]*dtdy;
	    }
		   
	    /*
	     * update with flux_z
	     */
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  > ghostWidth and
		 kU > ghostWidth ) {
	      h_UNew(i  ,j  ,kU-1,ID) -= flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,kU-1,IP) -= flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,kU-1,IU) -= flux_z[IW]*dtdz; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,kU-1,IV) -= flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,kU-1,IW) -= flux_z[IU]*dtdz; // watchout IU and IW swapped
	    }
	    
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,kU  ,ID) += flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,kU  ,IP) += flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,kU  ,IU) += flux_z[IW]*dtdz; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,kU  ,IV) += flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,kU  ,IW) += flux_z[IU]*dtdz; // watchout IU and IW swapped
	    }
	  } // end for i
	} // end for j
      } // end for k

      // gravity source term
      if (gravityEnabled) {
	ZslabInfo zSlabInfo;
	zSlabInfo.zSlabId     = zSlabId;
	zSlabInfo.zSlabNb     = zSlabNb;
	zSlabInfo.zSlabWidthG = zSlabWidthG;
	zSlabInfo.kStart      = zSlabWidth * zSlabId;
	zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;
	zSlabInfo.ksizeSlab   = zSlabWidthG;

	compute_gravity_source_term(h_UNew, h_UOld, dt, zSlabInfo);
      }

      TIMER_STOP(timerUpdate);

    } // end loop over z-slab
      
    /*************************************
     * DISSIPATIVE TERMS (i.e. viscosity)
     *************************************/
    TIMER_START(timerDissipative);
    real_t &nu = _gParams.nu;
    if (nu>0) {
      // update boundaries before dissipative terms computations
      make_all_boundaries(h_UNew);
    }
    
    // compute viscosity forces
    if (nu>0) {
      // re-use h_qm_x and h_qm_y
      HostArray<real_t> &flux_x = h_qm_x;
      HostArray<real_t> &flux_y = h_qm_y;
      HostArray<real_t> &flux_z = h_qm_z;
      
      // copy h_UNew into h_UOld
      h_UNew.copyTo(h_UOld);

      ZslabInfo zSlabInfo;
      zSlabInfo.zSlabId     = -1;
      zSlabInfo.zSlabNb     = zSlabNb;
      zSlabInfo.zSlabWidthG = zSlabWidthG;
      zSlabInfo.kStart      = -1;
      zSlabInfo.kStop       = -1;
      zSlabInfo.ksizeSlab   = zSlabWidthG;

      // loop over z-slab index
      for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {

	zSlabInfo.zSlabId     = zSlabId;
	zSlabInfo.kStart      = zSlabWidth * zSlabId;
	zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;

	compute_viscosity_flux(h_UOld, flux_x, flux_y, flux_z, dt, zSlabInfo);
	compute_hydro_update  (h_UNew, flux_x, flux_y, flux_z,     zSlabInfo);

      } // end for zSlabId
      
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

  } // HydroRunGodunovZslab::godunov_unsplit_cpu_v1

  // =======================================================
  // =======================================================
  void HydroRunGodunovZslab::godunov_unsplit_cpu_v2(HostArray<real_t>& h_UOld, 
						    HostArray<real_t>& h_UNew,
						    real_t dt, int nStep)
  {
    
    (void) nStep;
  
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;
    
    // loop over z-slab index
    for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {
      
      // start and stop global index of current slab (ghosts included)
      int kStart    = zSlabWidth * zSlabId;
      //int kStop     = zSlabWidth * zSlabId + zSlabWidthG;
      int ksizeSlab = zSlabWidthG;

      // take care that the last slab might be truncated
      if (zSlabId == zSlabNb-1) {
	ksizeSlab = ksize-kStart;
      }

      /*std::cout << "zSlabId " << zSlabId << " kStart " << kStart
	<< " kStop " << kStop << " ksizeSlab " << ksizeSlab << std::endl; */

      TIMER_START(timerPrimVar);
      // convert conservative to primitive variables (and source term predictor)
      // put results in h_Q object
      convertToPrimitives( h_UOld.data(), zSlabId );
      TIMER_STOP(timerPrimVar);

      /*
       * 1. Compute and store slopes
       */
      for (int k=1; k<ksizeSlab-1; k++) {

	for (int j=1; j<jsize-1; j++) {
	  for (int i=1; i<isize-1; i++) {

	    // primitive variables (local array)
	    real_t qLoc[NVAR_3D];
	    real_t qNeighbors[2*THREE_D][NVAR_3D];
	  
	    // slopes
	    real_t dq[THREE_D][NVAR_3D];
	  
	    // get primitive variables state vector
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	      qLoc[iVar]          = h_Q(i  ,j  ,k  ,iVar);
	      qNeighbors[0][iVar] = h_Q(i+1,j  ,k  ,iVar);
	      qNeighbors[1][iVar] = h_Q(i-1,j  ,k  ,iVar);
	      qNeighbors[2][iVar] = h_Q(i  ,j+1,k  ,iVar);
	      qNeighbors[3][iVar] = h_Q(i  ,j-1,k  ,iVar);
	      qNeighbors[4][iVar] = h_Q(i  ,j  ,k+1,iVar);
	      qNeighbors[5][iVar] = h_Q(i  ,j  ,k-1,iVar);
	    
	    } // end for iVar
	
	    // compute slopes in current cell
	    slope_unsplit_3d(qLoc, 
			     qNeighbors[0],
			     qNeighbors[1],
			     qNeighbors[2],
			     qNeighbors[3],
			     qNeighbors[4],
			     qNeighbors[5],
			     dq);
	  
	    // store slopes
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	      h_slope_x(i,j,k,iVar) = dq[0][iVar];
	      h_slope_y(i,j,k,iVar) = dq[1][iVar];
	      h_slope_z(i,j,k,iVar) = dq[2][iVar];
	    }

	  } // end for i
	} // end for j
      } // end for k

      /*
       * 2. Compute reconstructed states along X interfaces
       */
      for (int k=1; k<ksizeSlab-1; k++) {

	// z plane in global domain
	int kG = k+kStart;

	for (int j=1; j<jsize-1; j++) {
	  for (int i=1; i<isize-1; i++) {
	  
	    // primitive variables (local array)
	    real_t qLoc[NVAR_3D];
	  
	    // slopes
	    real_t dq[THREE_D][NVAR_3D];
	  
	    // reconstructed state on cell faces
	    // aka riemann solver input
	    real_t qleft[NVAR_3D];
	    real_t qright[NVAR_3D];
	  
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    // deal with left interface along X !
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  
	    // get current cell slopes and left neighbor
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      qLoc[iVar]  = h_Q      (i  ,j  ,k  ,iVar);
	      dq[0][iVar] = h_slope_x(i  ,j  ,k  ,iVar);
	      dq[1][iVar] = h_slope_y(i  ,j  ,k  ,iVar);
	      dq[2][iVar] = h_slope_z(i  ,j  ,k  ,iVar);
	    
	    } // end for iVar
	  
	    //
	    // Compute reconstructed states at left interface along X in current cell
	    //
	
	    // TAKE CARE here left and right designate the interface location
	    // compare to current cell
	    // !!! THIS is FUNDAMENTALLY different from v0 and v1 !!!

	    // left interface 
	    trace_unsplit_hydro_3d_by_direction(qLoc, 
						dq, 
						dtdx, dtdy, dtdz,
						FACE_XMIN, 
						qleft);
	
	    // right interface
	    trace_unsplit_hydro_3d_by_direction(qLoc,
						dq,
						dtdx, dtdy, dtdz,
						FACE_XMAX, 
						qright);
	  
	    if (gravityEnabled) { 
	    
	      // we need to modify input to flux computation with
	      // gravity predictor (half time step)
	    
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,kG,IZ);
	    
	      qright[IU] += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,kG,IZ);
	    
	    } // end gravityEnabled
	
	    // store them
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      h_qm(i  ,j  ,k  ,iVar) = qleft[iVar];
	      h_qp(i  ,j  ,k  ,iVar) = qright[iVar];
	    
	    }
	  
	  } // end for i
	} // end for j
      } // end for k

      /*
       * 3. Riemann solver at X interface and update
       */
      for (int k=2; k<ksizeSlab-1; k++) {

	// z plane in global domain
	int kG = k+kStart;

	for (int j=2; j<jsize-1; j++) {
	  for (int i=2; i<isize-1; i++) {
	  
	    // reconstructed state on cell faces
	    // aka riemann solver input
	    real_t qleft[NVAR_3D];
	    real_t qright[NVAR_3D];
	  
	    // riemann solver output
	    real_t qgdnv[NVAR_3D];
	    real_t flux_x[NVAR_3D];
	  
	    // read reconstructed states
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      qleft[iVar]  = h_qp(i-1,j  ,k  ,iVar);
	      qright[iVar] = h_qm(i  ,j  ,k  ,iVar);
	    
	    }
	  
	    // Solve Riemann problem at X-interfaces and compute X-fluxes
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_x);
	  
	    /*
	     * update with flux_x
	     */
	    if ( i  > ghostWidth           and 
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kG < ksize-ghostWidth) {
	      h_UNew(i-1,j  ,kG  ,ID) -= flux_x[ID]*dtdx;
	      h_UNew(i-1,j  ,kG  ,IP) -= flux_x[IP]*dtdx;
	      h_UNew(i-1,j  ,kG  ,IU) -= flux_x[IU]*dtdx;
	      h_UNew(i-1,j  ,kG  ,IV) -= flux_x[IV]*dtdx;
	      h_UNew(i-1,j  ,kG  ,IW) -= flux_x[IW]*dtdx;
	    }
	  
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kG < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,kG  ,ID) += flux_x[ID]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IP) += flux_x[IP]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IU) += flux_x[IU]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IV) += flux_x[IV]*dtdx;
	      h_UNew(i  ,j  ,kG  ,IW) += flux_x[IW]*dtdx;
	    }
	  
	  } // end for i
	} // end for j
      } // end for k

      /*
       * 4. Compute reconstructed states along Y interfaces
       */
      for (int k=1; k<ksizeSlab-1; k++) {

	// z plane in global domain
	int kG = k+kStart;

	for (int j=1; j<jsize-1; j++) {
	  for (int i=1; i<isize-1; i++) {
	  
	    // primitive variables (local array)
	    real_t qLoc[NVAR_3D];
	  
	    // slopes
	    real_t dq[THREE_D][NVAR_3D];
	  
	    // reconstructed state on cell faces
	    // aka riemann solver input
	    real_t qleft[NVAR_3D];
	    real_t qright[NVAR_3D];
	  
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    // deal with left interface along Y !
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  
	    // get current cell slopes and left neighbor
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      qLoc[iVar]  = h_Q      (i  ,j  ,k  ,iVar);
	      dq[0][iVar] = h_slope_x(i  ,j  ,k  ,iVar);
	      dq[1][iVar] = h_slope_y(i  ,j  ,k  ,iVar);
	      dq[2][iVar] = h_slope_z(i  ,j  ,k  ,iVar);
	    
	    } // end for iVar
	  
	    //
	    // Compute reconstructed states at left interface along Y in current cell
	    //
	  
	    // TAKE CARE here left and right designate the interface location
	    // compare to current cell
	    // !!! THIS is FUNDAMENTALLY different from v0 and v1 !!!
	  
	    // left interface
	    trace_unsplit_hydro_3d_by_direction(qLoc, 
						dq, 
						dtdx, dtdy, dtdz,
						FACE_YMIN, 
						qleft);
	  
	    // right interface
	    trace_unsplit_hydro_3d_by_direction(qLoc,
						dq,
						dtdx, dtdy, dtdz,
						FACE_YMAX, 
						qright);
	  
	    if (gravityEnabled) { 
	      // we need to modify input to flux computation with
	      // gravity predictor (half time step)
	    
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,kG,IZ);
	    
	      qright[IU] += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,kG,IZ);
	    
	    } // end gravityEnabled
	  
	    // store them
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      h_qm(i  ,j  ,k  ,iVar) = qleft[iVar];
	      h_qp(i  ,j  ,k  ,iVar) = qright[iVar];
	    
	    }
	  
	  } // end for i
	} // end for j
      } // end for k

      /*
       * 5. Riemann solver at Y interface and update
       */
      for (int k=2; k<ksizeSlab-1; k++) {

	// z plane in global domain
	int kG = k+kStart;

	for (int j=2; j<jsize-1; j++) {
	  for (int i=2; i<isize-1; i++) {
	  
	    // reconstructed state on cell faces
	    // aka riemann solver input
	    real_t qleft[NVAR_3D];
	    real_t qright[NVAR_3D];
	  
	    // riemann solver output
	    real_t qgdnv[NVAR_3D];
	    real_t flux_y[NVAR_3D];

	    // read reconstructed states
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      qleft[iVar]  = h_qp(i  ,j-1,k  ,iVar);
	      qright[iVar] = h_qm(i  ,j  ,k  ,iVar);
	    
	    }
	  
	    // Solve Riemann problem at Y-interfaces and compute Y-fluxes	  
	    swapValues(&(qleft[IU]) ,&(qleft[IV]) );
	    swapValues(&(qright[IU]),&(qright[IV]));
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_y);
	
	    /*
	     * update with flux_y
	     */
	    if ( i  < isize-ghostWidth     and 
		 j  > ghostWidth           and
		 k  < ksizeSlab-ghostWidth and
		 kG < ksize-ghostWidth ) {
	      h_UNew(i  ,j-1,kG ,ID) -= flux_y[ID]*dtdx;
	      h_UNew(i  ,j-1,kG ,IP) -= flux_y[IP]*dtdx;
	      h_UNew(i  ,j-1,kG ,IU) -= flux_y[IV]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,kG ,IV) -= flux_y[IU]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,kG ,IW) -= flux_y[IW]*dtdx;
	    }
	  
	    if ( i  < isize-ghostWidth     and
		 j  < jsize-ghostWidth     and
		 k  < ksizeSlab-ghostWidth and
		 kG < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,kG ,ID) += flux_y[ID]*dtdx;
	      h_UNew(i  ,j  ,kG ,IP) += flux_y[IP]*dtdx;
	      h_UNew(i  ,j  ,kG ,IU) += flux_y[IV]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,kG ,IV) += flux_y[IU]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,kG ,IW) += flux_y[IW]*dtdx;
	    }
	  
	  } // end for j
	} // end for i
      } // end for k

      /*
       * 6. Compute reconstructed states along Z interfaces
       */
      for (int k=1; k<ksizeSlab-1; k++) {

	// z plane in global domain
	int kG = k+kStart;

	for (int j=1; j<jsize-1; j++) {
	  for (int i=1; i<isize-1; i++) {
	  
	    // primitive variables (local array)
	    real_t qLoc[NVAR_3D];
	  
	    // slopes
	    real_t dq[THREE_D][NVAR_3D];
	  
	    // reconstructed state on cell faces
	    // aka riemann solver input
	    real_t qleft[NVAR_3D];
	    real_t qright[NVAR_3D];
	  
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    // deal with left interface along Z !
	    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  
	    // get current cell slopes and left neighbor
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      qLoc[iVar]  = h_Q      (i  ,j  ,k  ,iVar);
	      dq[0][iVar] = h_slope_x(i  ,j  ,k  ,iVar);
	      dq[1][iVar] = h_slope_y(i  ,j  ,k  ,iVar);
	      dq[2][iVar] = h_slope_z(i  ,j  ,k  ,iVar);
	    
	    } // end for iVar
	  
	    //
	    // Compute reconstructed states at left interface along Y in current cell
	    //
	  
	    // TAKE CARE here left and right designate the interface location
	    // compare to current cell
	    // !!! THIS is FUNDAMENTALLY different from v0 and v1 !!!
	  
	    // left interface
	    trace_unsplit_hydro_3d_by_direction(qLoc, 
						dq, 
						dtdx, dtdy, dtdz,
						FACE_ZMIN, 
						qleft);
	  
	    // right interface
	    trace_unsplit_hydro_3d_by_direction(qLoc,
						dq,
						dtdx, dtdy, dtdz,
						FACE_ZMAX, 
						qright);
	  
	    if (gravityEnabled) { 
	      // we need to modify input to flux computation with
	      // gravity predictor (half time step)
	    
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,kG,IZ);
	    
	      qright[IU] += HALF_F * dt * h_gravity(i,j,kG,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,kG,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,kG,IZ);
	    
	    } // end gravityEnabled
	  
	    // store them
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      h_qm(i  ,j  ,k  ,iVar) = qleft[iVar];
	      h_qp(i  ,j  ,k  ,iVar) = qright[iVar];
	    
	    }
	  
	  } // end for i
	} // end for j
      } // end for k

      /*
       * 7. Riemann solver at Z interface and update
       */
      for (int k=2; k<ksizeSlab-1; k++) {

	// z plane in global domain
	int kG = k+kStart;

	for (int j=2; j<jsize-1; j++) {
	  for (int i=2; i<isize-1; i++) {
	  
	    // reconstructed state on cell faces
	    // aka riemann solver input
	    real_t qleft[NVAR_3D];
	    real_t qright[NVAR_3D];
	  
	    // riemann solver output
	    real_t qgdnv[NVAR_3D];
	    real_t flux_z[NVAR_3D];

	    // read reconstructed states
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      qleft[iVar]  = h_qp(i  ,j  ,k-1,iVar);
	      qright[iVar] = h_qm(i  ,j  ,k  ,iVar);
	    
	    }
	  
	    // Solve Riemann problem at Z-interfaces and compute Z-fluxes	  
	    swapValues(&(qleft[IU]) ,&(qleft[IW]) );
	    swapValues(&(qright[IU]),&(qright[IW]));
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_z);
	
	    /*
	     * update with flux_z
	     */
	    if ( i  < isize-ghostWidth and 
		 j  < jsize-ghostWidth and
		 k  > ghostWidth       and
		 kG > ghostWidth) {
	      h_UNew(i  ,j  ,kG-1,ID) -= flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,kG-1,IP) -= flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,kG-1,IU) -= flux_z[IW]*dtdz; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,kG-1,IV) -= flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,kG-1,IW) -= flux_z[IU]*dtdz; // watchout IU and IW swapped
	    }
	  
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and 
		 k  < ksizeSlab-ghostWidth and
		 kG < ksize-ghostWidth) {
	      h_UNew(i  ,j  ,kG ,ID) += flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,kG ,IP) += flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,kG ,IU) += flux_z[IW]*dtdz; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,kG ,IV) += flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,kG ,IW) += flux_z[IU]*dtdz; // watchout IU and IW swapped
	    }
	  
	  } // end for j
	} // end for i
      } // end for k

      // gravity source term
      if (gravityEnabled) {
	ZslabInfo zSlabInfo;
	zSlabInfo.zSlabId     = zSlabId;
	zSlabInfo.zSlabNb     = zSlabNb;
	zSlabInfo.zSlabWidthG = zSlabWidthG;
	zSlabInfo.kStart      = zSlabWidth * zSlabId;
	zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;
	zSlabInfo.ksizeSlab   = zSlabWidthG;

	compute_gravity_source_term(h_UNew, h_UOld, dt, zSlabInfo);
      }

    } // end loop over zSlabId

    /*************************************
     * DISSIPATIVE TERMS (i.e. viscosity)
     *************************************/
    TIMER_START(timerDissipative);
    real_t &nu = _gParams.nu;
    if (nu>0) {
      // update boundaries before dissipative terms computations
      make_all_boundaries(h_UNew);
    }
    
    // compute viscosity forces
    if (nu>0) {
      // re-use slope array
      HostArray<real_t> &flux_x = h_slope_x;
      HostArray<real_t> &flux_y = h_slope_y;
      HostArray<real_t> &flux_z = h_slope_z;
      
      // copy h_UNew into h_UOld
      h_UNew.copyTo(h_UOld);

      ZslabInfo zSlabInfo;
      zSlabInfo.zSlabId     = -1;
      zSlabInfo.zSlabNb     = zSlabNb;
      zSlabInfo.zSlabWidthG = zSlabWidthG;
      zSlabInfo.kStart      = -1;
      zSlabInfo.kStop       = -1;
      zSlabInfo.ksizeSlab   = zSlabWidthG;

      // loop over z-slab index
      for (int zSlabId=0; zSlabId < zSlabNb; ++zSlabId) {

	zSlabInfo.zSlabId     = zSlabId;
	zSlabInfo.kStart      = zSlabWidth * zSlabId;
	zSlabInfo.kStop       = zSlabWidth * zSlabId + zSlabWidthG;

	compute_viscosity_flux(h_UOld, flux_x, flux_y, flux_z, dt, zSlabInfo);
	compute_hydro_update  (h_UNew, flux_x, flux_y, flux_z,     zSlabInfo);

      } // end for zSlabId
      
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

  } // HydroRunGodunovZslab::godunov_unsplit_cpu_v2

#endif // __CUDACC__
  
  // =======================================================
  // =======================================================
  /*
   * main routine to start simulation.
   */
  void HydroRunGodunovZslab::start() {
  
    // test if we are performing a re-start run (default : false)
    bool restartEnabled = configMap.getBool("run","restart",false);
  
    // should we include ghost cells in output files ?
    bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);
    
    /*
     * initial condition.
     */
    int  nStep = 0;

    std::cout << "Initialization\n";
    int configNb = configMap.getInteger("hydro" , "riemann_config_number", 0);
    setRiemannConfId(configNb);
    nStep = init_simulation(problem);

    // make sure border conditions are OK at beginning of simulation
    if (restartEnabled and ghostIncluded) {
    
      // we do not need to call make_all_boundaries since h_U/d_U is
      // fully filled from reading input data file (ghost zones already
      // set properly) in init_simulation.
    
    } else { // not a restart run
    
#ifdef __CUDACC__
      make_all_boundaries(d_U);
      d_U.copyTo(d_U2);
#else
      make_all_boundaries(h_U);
      h_U.copyTo(h_U2);
#endif // __CUDACC__

    } // end if (restartEnabled and ghostIncluded)

    // dump information about computations about to start
    std::cout << "Starting time integration (Godunov)" << std::endl;

    std::cout << "use unsplit integration" << std::endl;

    std::cout << "Resolution (nx,ny,nz) " << nx << " " << ny << " " << nz << std::endl;

    // if restart is enabled totalTime is read from input data file
    // if not, we need to set totalTime to zero
    if (!restartEnabled) {

      totalTime = 0;
  
    } else {

      // do we force totalTime to be zero ?
      bool resetTotalTime = configMap.getBool("run","restart_reset_totaltime",false);
      if (resetTotalTime)
	totalTime=0;

      std::cout << "### This is a restarted run ! Current time is " << totalTime << " ###\n";
    }

    real_t dt = compute_dt(0);

    // how often should we print some log
    int nLog = configMap.getInteger("run", "nlog", 0);

    // timing
    Timer timerTotal;
    Timer timerWriteOnDisk;
    Timer timerHistory;

    // choose which history method will be called
    setupHistory_hydro();
    real_t dtHist  = configMap.getFloat("history", "dtHist", 10*dt);
    real_t tHist   = totalTime;

    // start timer
    timerTotal.start();

    while(totalTime < tEnd && nStep < nStepmax)
      {

	/* just some log */
	if (nLog>0 and (nStep % nLog) == 0) {

	  std::cout << "["        << current_date()  << "]"
		    << "  step="  << std::setw(9)    << nStep 
		    << " t="      << fmt(totalTime)
		    << " dt="     << fmt(dt)         << std::endl;
	}

	/* Output results */
	if ((nStep % nOutput)==0) {

	  timerWriteOnDisk.start();
	
	  // make sure Device data are copied back onto Host memory
	  // which data to save ?
	  copyGpuToCpu(nStep);

	  output(getDataHost(nStep), nStep, ghostIncluded);
	
	  timerWriteOnDisk.stop();

	  std::cout << "["        << current_date()  << "]"
		    << "  step="  << std::setw(9)    << nStep 
		    << " t="      << fmt(totalTime)
		    << " dt="     << fmt(dt)
		    << " output " << std::endl;
	}

	// call history ?
	timerHistory.start();
	if (tHist == 0 or 
	    ( (totalTime-dt <= tHist+dtHist) and 
	      (totalTime    >  tHist+dtHist) ) ) {
	  copyGpuToCpu(nStep);
	  history_hydro(nStep,dt);
	  tHist += dtHist;
	}
	timerHistory.stop();

	/* one time step integration (nStep increment) */
	oneStepIntegration(nStep, totalTime, dt);
      
      } // end time loop

    // output last time step
    {
      printf("Final output at step %d\n",nStep);
      timerWriteOnDisk.start();

      // make sure Device data are copied back onto Host memory
      copyGpuToCpu(nStep);
    
      output(getDataHost(nStep), nStep, ghostIncluded);
      timerWriteOnDisk.stop();
    }

    // write Xdmf wrapper file
    if (outputHdf5Enabled) writeXdmfForHdf5Wrapper(nStep, false, ghostIncluded);

    // final timing report
    timerTotal.stop();

    printf("Euler godunov total  time: %5.3f sec\n", timerTotal.elapsed());
    printf("Euler godunov output time: %5.3f sec (%5.2f %% of total time)\n",timerWriteOnDisk.elapsed(), timerWriteOnDisk.elapsed()/timerTotal.elapsed()*100.);

    /*
     * print timing report if required
     */
#ifdef DO_TIMING
    printf("Euler godunov boundaries : %5.3f sec (%5.2f %% of total time)\n",timerBoundaries.elapsed(), timerBoundaries.elapsed()/timerTotal.elapsed()*100.);
    printf("Euler godunov computing  : %5.3f sec (%5.2f %% of total time)\n",timerGodunov.elapsed(), timerGodunov.elapsed()/timerTotal.elapsed()*100.);
  
    printf("Euler hydro prim var    : %5.3f sec (%5.2f %% of computing time)\n",timerPrimVar.elapsed(), timerPrimVar.elapsed()/timerGodunov.elapsed()*100.);
    printf("Euler hydro slope/trace : %5.3f sec (%5.2f %% of computing time)\n",timerSlopeTrace.elapsed(), timerSlopeTrace.elapsed()/timerGodunov.elapsed()*100.);
    printf("Euler hydro update      : %5.3f sec (%5.2f %% of computing time)\n",timerUpdate.elapsed(), timerUpdate.elapsed()/timerGodunov.elapsed()*100.);
    printf("Euler dissipative terms : %5.3f sec (%5.2f %% of computing time)\n",timerDissipative.elapsed(), timerDissipative.elapsed()/timerGodunov.elapsed()*100.);
    printf("Euler history           : %5.3f sec (%5.2f %% of total time)\n",timerHistory.elapsed(), timerHistory.elapsed()/timerTotal.elapsed()*100.);
 

#endif // DO_TIMING

    std::cout  << "####################################\n"
	       << "Global perfomance                   \n" 
	       << 1.0*nStep*(nx)*(ny)*(nz)/(timerTotal.elapsed()-timerWriteOnDisk.elapsed())
	       << " cell updates per seconds (based on wall time)\n"
	       << "####################################\n";
  
  } // HydroRunGodunovZslab::start

    // =======================================================
    // =======================================================
    /*
     * do one time step integration
     */
  void HydroRunGodunovZslab::oneStepIntegration(int& nStep, real_t& t, real_t& dt) {

    // if nStep is even update U  into U2
    // if nStep is odd  update U2 into U
    dt=compute_dt(nStep % 2);
    godunov_unsplit(nStep, dt);
  
    // increment time
    nStep++;
    t+=dt;
  
  } // HydroRunGodunovZslab::oneStepIntegration

  // =======================================================
  // =======================================================
  /*
   * convert to primitive variables.
   *
   * Take care that U is sized upon the global domain, whereas Q is sized
   * upon a local zSlab.
   */
  void HydroRunGodunovZslab::convertToPrimitives(real_t *U, int zSlabId)
  {

#ifdef __CUDACC__

    /* TODO */

#else // CPU version
  
    // primitive variable domain array
    real_t *Q = h_Q.data();
  
    // section / domain size
    int arraySizeQ = h_Q.section();
    int arraySizeU = h_U.section();
  
    // update primitive variables array
    // inside z-slab + ghosts

    // start and stop index of current slab (ghosts included)
    int kStart = zSlabWidth * zSlabId;
    int kStop  = zSlabWidth * zSlabId + zSlabWidthG;

    for (int k = kStart; 
	 k     < kStop; 
	 k++) {

      // take car of last slab might be partly outside domain
      if (k<ksize) {
      
	for (int j=0; j<jsize; j++) {
	  for (int i=0; i<isize; i++) {
	  
	    // primitive variable state vector
	    real_t q[NVAR_3D];
	  
	    int indexLocU = i+j*isize+k*isize*jsize;
	    real_t c;
	  
	    computePrimitives_3D_0(U, arraySizeU, indexLocU, c, q);
	  
	    // copy q state in h_Q
	    int indexLocQ = i+j*isize+(k-kStart)*isize*jsize;
	    int offset = indexLocQ;
	    Q[offset] = q[ID]; offset += arraySizeQ;
	    Q[offset] = q[IP]; offset += arraySizeQ;
	    Q[offset] = q[IU]; offset += arraySizeQ;
	    Q[offset] = q[IV]; offset += arraySizeQ;
	    Q[offset] = q[IW];
	  
	  } // end for i
	} // end for j
    
      } // end if k<ksize

    } // end for k inside z-slab
  
#endif // __CUDACC__
  
  } // HydroRunGodunovZslab::convertToPrimitives

} // namespace hydroSimu
