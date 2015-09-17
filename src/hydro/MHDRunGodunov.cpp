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
 * \file MHDRunGodunov.cpp
 * \brief Implements class MHDRunGodunov
 * 
 * 2D/3D MHD Euler equation solver on a cartesian grid using Godunov method
 * with Riemann solver (HLL, HLLD).
 *
 * \date March, 31 2011
 * \author Pierre Kestener.
 *
 * $Id: MHDRunGodunov.cpp 3450 2014-06-16 22:03:23Z pkestene $
 */
#include "MHDRunGodunov.h"

// include CUDA kernel when necessary
#ifdef __CUDACC__
#include "godunov_unsplit_mhd.cuh"
#include "shearingBox_utils.cuh"
#endif // __CUDACC__
#include "constoprim.h"
#include "riemann_mhd.h"
#include "trace_mhd.h"
//#include "geometry_utils.h"

#include "make_boundary_common.h" // for macro  MK_BOUND_BLOCK_SIZE, etc ...
#include "shearBorderUtils.h"     // for shear border condition
#include "make_boundary_shear.h"  // for shear border condition (slopes, final remap)
#include "../utils/monitoring/date.h"
#include "../utils/monitoring/Timer.h"

#include <iomanip> // for std::setprecision
#include "ostream_fmt.h"

// PAPI support
#ifdef USE_PAPI
#include "../utils/monitoring/PapiInfo.h"
#endif // USE_PAPI

// OpenMP support
#if _OPENMP
# include <omp.h>
#endif

// /** a dummy device-only swap function */
// static void swap_v(real_t& a, real_t& b) {
   
//   real_t tmp = a;
//   a = b;
//   b = tmp; 
   
// } // swap_v

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // MHDRunGodunov class methods body
  ////////////////////////////////////////////////////////////////////////////////
  
  MHDRunGodunov::MHDRunGodunov(ConfigMap &_configMap)
    : MHDRunBase(_configMap)
    , shearingBoxEnabled(false)
    , implementationVersion(0)
    , dumpDataForDebugEnabled(false)
    , debugShear(false)
#ifdef __CUDACC__
    , d_Q()
#else
    , h_Q() /* CPU only */
#endif // __CUDACC__
  {

    /*
     * shearing box enabled ??
     *
     * We need to do this early in constructor, to enforce implementationVersion
     * parameter (used to do memory allocation).
     */

    // normal behavior is to let the code chose if shearing box is enabled by 
    // examining border conditions and if OmegaO is strictly positive
    if (boundary_xmin == BC_SHEARINGBOX and 
	boundary_xmax == BC_SHEARINGBOX and
	_gParams.Omega0 > 0) {
      shearingBoxEnabled = true;
      std::cout << "Using shearing box border conditions with Omega0 = " 
		<< _gParams.Omega0 << std::endl;
    }

    // sanity check
    if ( (boundary_xmin == BC_SHEARINGBOX and boundary_xmax != BC_SHEARINGBOX) or
	 (boundary_xmin != BC_SHEARINGBOX and boundary_xmax == BC_SHEARINGBOX) ) {
      std::cout << "ERROR : you need to set both boundary_xmin and boundary_xmax to 4 (BC_SHEARINGBOX) and set Omega0>0 to enabled shearing box !!\n";
    }
    if (boundary_xmin == BC_SHEARINGBOX and 
	boundary_xmax == BC_SHEARINGBOX and
	_gParams.Omega0 <= 0) {
      std::cout << "##### WARNING : you are trying to use shearing box border\n";
      std::cout << "#####           conditions but Omega0<=0\n";
      std::cout << "#####           You must set a strictly positive Omega0 value\n";
    }

    // make sure that implementationVersion is set to the correct value when
    // using a rotating frame (shearing box or not)
    if (_gParams.Omega0>0 /* or shearingBoxEnabled */) {
      if (dimType == TWO_D)
	implementationVersion = 1;
      else // THREE_D
	implementationVersion = 4;
    }

    /*
     *
     * MEMORY ALLOCATION
     *
     */

    // !!! CPU ONLY : primitive variable array !!!
#ifndef __CUDACC__
    // memory allocation primitive variable array Q
    if (dimType == TWO_D) {
      h_Q.allocate (make_uint3(isize, jsize, nbVar));
    } else {
      h_Q.allocate (make_uint4(isize, jsize, ksize, nbVar));
    }
#endif // __CUDACC__

    // memory allocation for EMF's
#ifdef __CUDACC__
    if (dimType == TWO_D) {
      d_emf.allocate(make_uint3(isize, jsize, 1           ), gpuMemAllocType); // only EMFZ
    } else {
      d_emf.allocate(make_uint4(isize, jsize, ksize, 3    ), gpuMemAllocType); // 3 EMF's
    }
#else
    if (dimType == TWO_D) {
      h_emf.allocate(make_uint3(isize, jsize, 1           )); // only EMFZ
    } else {
      h_emf.allocate(make_uint4(isize, jsize, ksize, 3    )); // 3 EMF's
    }
#endif // __CUDACC__


    // which implementation version will we use (0, 1, 2 or 4 ?)
    // the default version for 3D is 4 (which is faster)
    if (dimType == TWO_D)
      implementationVersion = configMap.getInteger("MHD","implementationVersion",1);
    else
      implementationVersion = configMap.getInteger("MHD","implementationVersion",4);

    if (implementationVersion != 0 and 
	implementationVersion != 1 and
	implementationVersion != 2 and
	implementationVersion != 3 and
	implementationVersion != 4) {
      if (dimType == TWO_D)
	implementationVersion = 1;
      else
	implementationVersion = 4;
      std::cout << "################################################################" << std::endl;
      std::cout << "WARNING : you should review your parameter file and set MHD/implementationVersion"
		<< " to a valid number (0, 1, 2, 3 or 4 currently)" << std::endl;
      std::cout << "implementation version is forced to the default value" << std::endl;
      std::cout << "################################################################" << std::endl;
    }
    std::cout << "Using MHD Godunov implementation version : " << implementationVersion << std::endl;

    

    /*
     * extra memory allocation for a specific implementation version
     */
    if (implementationVersion >= 1) { // less memory scrooge version
#ifdef __CUDACC__
      if (dimType == TWO_D) {
	  
	d_qm_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	d_qm_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	  
	d_qEdge_RT.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	d_qEdge_RB.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	d_qEdge_LT.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	  
      } else { // THREE_D
	  
	if (implementationVersion < 3) { 

	  d_qm_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType); // 2D array
	  d_qm_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType); // 2D array
	  d_qm_z.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType); // 3D array with only 2 plans
	    
	  d_qEdge_RT.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
	  d_qEdge_RB.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
	  d_qEdge_LT.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
	    
	  d_qEdge_RT2.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
	  d_qEdge_RB2.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
	  d_qEdge_LT2.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
	    
	  d_qEdge_RT3.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
	  d_qEdge_RB3.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
	  d_qEdge_LT3.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);

	} else { // implementationVersion == 3 or 4

	  d_qm_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qm_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qm_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qp_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qp_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qp_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	    
	  d_qEdge_RT.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qEdge_RB.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qEdge_LT.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qEdge_LB.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	    
	  d_qEdge_RT2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qEdge_RB2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qEdge_LT2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qEdge_LB2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	    
	  d_qEdge_RT3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qEdge_RB3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qEdge_LT3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qEdge_LB3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	    
	}
      } // end THREE_D
 
#else // CPU
      if (dimType == TWO_D) {

	h_qm_x.allocate(make_uint3(isize, jsize, nbVar));
	h_qm_y.allocate(make_uint3(isize, jsize, nbVar));
	h_qp_x.allocate(make_uint3(isize, jsize, nbVar));
	h_qp_y.allocate(make_uint3(isize, jsize, nbVar));

	h_qEdge_RT.allocate(make_uint3(isize, jsize, nbVar));
	h_qEdge_RB.allocate(make_uint3(isize, jsize, nbVar));
	h_qEdge_LT.allocate(make_uint3(isize, jsize, nbVar));
	h_qEdge_LB.allocate(make_uint3(isize, jsize, nbVar));

      } else { // THREE_D

	h_qm_x.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qm_y.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qm_z.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qp_x.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qp_y.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qp_z.allocate(make_uint4(isize, jsize, ksize, nbVar));

	h_qEdge_RT.allocate (make_uint4(isize, jsize, ksize, nbVar));
	h_qEdge_RB.allocate (make_uint4(isize, jsize, ksize, nbVar));
	h_qEdge_LT.allocate (make_uint4(isize, jsize, ksize, nbVar));
	h_qEdge_LB.allocate (make_uint4(isize, jsize, ksize, nbVar));

	h_qEdge_RT2.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qEdge_RB2.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qEdge_LT2.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qEdge_LB2.allocate(make_uint4(isize, jsize, ksize, nbVar));

	h_qEdge_RT3.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qEdge_RB3.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qEdge_LT3.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_qEdge_LB3.allocate(make_uint4(isize, jsize, ksize, nbVar));

      } // end THREE_D
#endif // __CUDACC__
    } // end implementationVersion >= 1

    if (implementationVersion == 2 || implementationVersion == 3 || implementationVersion == 4) {
#ifdef __CUDACC__
      if (dimType == TWO_D) {
	//d_elec.allocate(make_uint3(isize, jsize, 1), gpuMemAllocType);
      } else { // THREE_D
	d_elec.allocate (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
	d_dA.allocate   (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
	d_dB.allocate   (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
	d_dC.allocate   (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);	
      }
#else // CPU
      if (dimType == TWO_D) { // implementationVersion 2 only
	// in 2D, we only need to compute Ez
	h_elec.allocate (make_uint3(isize, jsize, 1));
	h_dA.allocate   (make_uint3(isize, jsize, 1));
	h_dB.allocate   (make_uint3(isize, jsize, 1));
      } else { // THREE_D 
	h_elec.allocate (make_uint4(isize, jsize, ksize, 3));
	h_dA.allocate   (make_uint4(isize, jsize, ksize, 3));
	h_dB.allocate   (make_uint4(isize, jsize, ksize, 3));
	h_dC.allocate   (make_uint4(isize, jsize, ksize, 3));
      }
#endif // __CUDACC__
    } // end implementationVersion == 2 or 3 or 4

#ifdef __CUDACC__
    if ( (implementationVersion == 3 or implementationVersion == 4) and dimType == THREE_D )
      d_Q.allocate (make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
#endif // __CUDACC__


    /*
     * extra memory allocation for shearing box simulations
     */
    // memory allocation required for shearing box simulations
    // there are 4 components :
    // - 2 for density remapping
    // - 2 for emf remapping
    if (shearingBoxEnabled) {
      if (dimType == TWO_D) {
#ifdef __CUDACC__
#else
	h_shear_flux_xmin.allocate(make_uint3(jsize,1,NUM_COMPONENT_REMAP));
	h_shear_flux_xmax.allocate(make_uint3(jsize,1,NUM_COMPONENT_REMAP));
#endif // __UCUDACC__
	/* WARNING : shearing box is not completely implemented in 2D */
      } else { // THREE_D
#ifdef __CUDACC__
      	d_shear_flux_xmin.allocate      (make_uint3(jsize,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
      	d_shear_flux_xmax.allocate      (make_uint3(jsize,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
      	d_shear_flux_xmin_remap.allocate(make_uint3(jsize,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
      	d_shear_flux_xmax_remap.allocate(make_uint3(jsize,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);

	d_shear_border_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
	d_shear_border_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);

	d_shear_slope_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
 	d_shear_slope_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
#else
      	h_shear_flux_xmin.allocate      (make_uint3(jsize,ksize,NUM_COMPONENT_REMAP));
      	h_shear_flux_xmax.allocate      (make_uint3(jsize,ksize,NUM_COMPONENT_REMAP));
      	h_shear_flux_xmin_remap.allocate(make_uint3(jsize,ksize,NUM_COMPONENT_REMAP));
      	h_shear_flux_xmax_remap.allocate(make_uint3(jsize,ksize,NUM_COMPONENT_REMAP));

	h_shear_border_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar));
	h_shear_border_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar));

	h_shear_slope_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar));
	memset(h_shear_slope_xmin.data(),0,h_shear_slope_xmin.sizeBytes());
 	h_shear_slope_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar));
	memset(h_shear_slope_xmax.data(),0,h_shear_slope_xmax.sizeBytes());
#endif // __CUDACC__


     }
    } // end shearingBoxEnabled

    /*
     * memory allocation for GPU routines debugging
     */
    dumpDataForDebugEnabled = configMap.getBool("debug","dumpData",false);
    if (dumpDataForDebugEnabled) { // we need memory allocation to do that
      if (dimType == THREE_D) {
	h_debug.allocate(make_uint4(isize, jsize, ksize, nbVar));
	h_debug2.allocate(make_uint4(isize, jsize, ksize, 3));
      }
    }

    // shearing box debug
    debugShear = configMap.getBool("debug","shear",false);

    // GPU execution settings
#ifdef __CUDACC__
    bool cachePreferL1 = configMap.getBool("debug","cachePreferL1",true);
    // when using >= sm_20 architecture, prefer L1 cache versus shared memory
    if (cachePreferL1) {
      cudaFuncSetCacheConfig(kernel_mhd_compute_trace_v4, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear_part1, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(kernel_mhd_compute_emf_v4, cudaFuncCachePreferL1);
    }
#endif // __CUDACC__


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

  } // MHDRunGodunov::MHDRunGodunov

  // =======================================================
  // =======================================================
  MHDRunGodunov::~MHDRunGodunov()
  {  
    
  } // MHDRunGodunov::~MHDRunGodunov()
  
  // =======================================================
  // =======================================================
  void MHDRunGodunov::convertToPrimitives(real_t *U, real_t deltaT)
  {

    TIMER_START(timerPrimVar);

#ifdef __CUDACC__

    if (dimType == TWO_D) {

      {
	// 2D primitive variables computation kernel    
	dim3 dimBlock(PRIM_VAR_MHD_BLOCK_DIMX_2D,
		      PRIM_VAR_MHD_BLOCK_DIMY_2D);
	dim3 dimGrid(blocksFor(isize, PRIM_VAR_MHD_BLOCK_DIMX_2D), 
		     blocksFor(jsize, PRIM_VAR_MHD_BLOCK_DIMY_2D));
	kernel_mhd_compute_primitive_variables_2D<<<dimGrid, 
	  dimBlock>>>(U,
		      d_Q.data(),
		      d_Q.pitch(),
		      d_Q.dimx(),
		      d_Q.dimy(),
		      deltaT);
	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_primitive_variables_2D error");
	
      }

    } else { // THREE_D

      {
	// 3D primitive variables computation kernel    
	dim3 dimBlock(PRIM_VAR_MHD_BLOCK_DIMX_3D,
		      PRIM_VAR_MHD_BLOCK_DIMY_3D);
	dim3 dimGrid(blocksFor(isize, PRIM_VAR_MHD_BLOCK_DIMX_3D), 
		     blocksFor(jsize, PRIM_VAR_MHD_BLOCK_DIMY_3D));
	kernel_mhd_compute_primitive_variables_3D<<<dimGrid, 
	  dimBlock>>>(U, 
		      d_Q.data(),
		      d_Q.pitch(),
		      d_Q.dimx(),
		      d_Q.dimy(), 
		      d_Q.dimz(),
		      deltaT);
	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_primitive_variables_3D error");
	
      }

    } // end THREE_D

#else // CPU version

    if (dimType == TWO_D) {
      
      int physicalDim[2] = {(int) h_U.pitch(),
			    (int) h_U.dimy()};
      
      // primitive variable state vector
      real_t q[NVAR_MHD];
      
      // primitive variable domain array
      real_t *Q = h_Q.data();
      
      // section / domain size
      int arraySize = h_Q.section();
      
      // update primitive variables array
      // please note the values of upper bounds (this is because the
      // computePrimitives_MHD_2D routine needs to access magnetic
      // fields in the neighbors on the dimension-wise right !).
      for (int j=0; j<jsize-1; j++) {
	for (int i=0; i<isize-1; i++) {
	  
	  int indexLoc = i+j*isize;
	  real_t c=0;
	  
	  computePrimitives_MHD_2D(U, physicalDim, indexLoc, c, q, deltaT);
	  
	  // copy q state in h_Q
	  int offset = indexLoc;
	  Q[offset] = q[ID]; offset += arraySize;
	  Q[offset] = q[IP]; offset += arraySize;
	  Q[offset] = q[IU]; offset += arraySize;
	  Q[offset] = q[IV]; offset += arraySize;
	  Q[offset] = q[IW]; offset += arraySize;
	  Q[offset] = q[IA]; offset += arraySize;
	  Q[offset] = q[IB]; offset += arraySize;
	  Q[offset] = q[IC];
	  
	} // end for i
      } // end for j
    
    } else { // THREE_D

      int physicalDim[3] = {(int) h_U.pitch(),
			    (int) h_U.dimy(),
			    (int) h_U.dimz()};
      
      // primitive variable state vector
      real_t q[NVAR_MHD];
      
      // primitive variable domain array
      real_t *Q = h_Q.data();
      
      // section / domain size
      int arraySize = h_Q.section();
      
      // update primitive variables array
      // please note the values of upper bounds (this is because the
      // computePrimitives_MHD_3D routine needs to access magnetic
      // fields in the neighbors on the dimension-wise right !).
      for (int k=0; k<ksize-1; k++) {
	for (int j=0; j<jsize-1; j++) {
	  for (int i=0; i<isize-1; i++) {
	    
	    int indexLoc = i+j*isize+k*isize*jsize;
	    real_t c=0;
	    
	    computePrimitives_MHD_3D(U, physicalDim, indexLoc, c, q, deltaT);
	    
	    // copy q state in h_Q
	    int offset = indexLoc;
	    Q[offset] = q[ID]; offset += arraySize;
	    Q[offset] = q[IP]; offset += arraySize;
	    Q[offset] = q[IU]; offset += arraySize;
	    Q[offset] = q[IV]; offset += arraySize;
	    Q[offset] = q[IW]; offset += arraySize;
	    Q[offset] = q[IA]; offset += arraySize;
	    Q[offset] = q[IB]; offset += arraySize;
	    Q[offset] = q[IC];
	    
	  } // end for i
	} // end for j
      } // end for k

    } // end THREE_D

#endif // __CUDACC__

    TIMER_STOP(timerPrimVar);

  } // MHDRunGodunov::convertToPrimitives

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit(int nStep, real_t dt)
#ifdef __CUDACC__
  {
    
    if (_gParams.Omega0>0) {
      
      if ((nStep%2)==0) {
	godunov_unsplit_rotating_gpu(d_U , d_U2, dt, nStep);
      } else {
	godunov_unsplit_rotating_gpu(d_U2, d_U , dt, nStep);
      }

    } else {

      if ((nStep%2)==0) {
	godunov_unsplit_gpu(d_U , d_U2, dt, nStep);
      } else {
	godunov_unsplit_gpu(d_U2, d_U , dt, nStep);
      }

    } // end Omega0>0 (rotating frame)
    
  } // MHDRunGodunov::godunov_unsplit (GPU version)
#else // CPU version
{
  
  if (_gParams.Omega0>0) {

    if ((nStep%2)==0) {
      godunov_unsplit_rotating_cpu(h_U , h_U2, dt, nStep);
    } else {
      godunov_unsplit_rotating_cpu(h_U2, h_U , dt, nStep);
    }

  } else {

    if ((nStep%2)==0) {
      godunov_unsplit_cpu(h_U , h_U2, dt, nStep);
    } else {
      godunov_unsplit_cpu(h_U2, h_U , dt, nStep);
    }

  } // end Omega0>0 (rotating frame)

} // MHDRunGodunov::godunov_unsplit (CPU version)
#endif // __CUDACC__

  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
					  DeviceArray<real_t>& d_UNew,
					  real_t dt, int nStep)
  {

    // update boundaries
    TIMER_START(timerBoundaries);
    make_all_boundaries(d_UOld);
    TIMER_STOP(timerBoundaries);

    // easier for time integration later
    d_UOld.copyTo(d_UNew);

    // gravity term not implemented ?
    if (dimType == THREE_D and gravityEnabled and (implementationVersion < 4) ) {
      std::cerr << "Gravity is not implemented in versions 0 to 3 of 3D MHD.\n";
      std::cerr << "Use implementationVersion=4 instead !\n";
    }

    // inner domain integration
    TIMER_START(timerGodunov);

    if (implementationVersion == 0) {
  
      godunov_unsplit_gpu_v0(d_UOld, d_UNew, dt, nStep);
  
    } else if (implementationVersion == 1) {

      godunov_unsplit_gpu_v1(d_UOld, d_UNew, dt, nStep);
    
    } else if (implementationVersion == 2) {
    
      godunov_unsplit_gpu_v2(d_UOld, d_UNew, dt, nStep);

    } else if (implementationVersion == 3) {
    
      godunov_unsplit_gpu_v3(d_UOld, d_UNew, dt, nStep);

    } else if (implementationVersion == 4) {
    
      godunov_unsplit_gpu_v4(d_UOld, d_UNew, dt, nStep);

    } // end implementationVersion == 4

    TIMER_STOP(timerGodunov);

  } // MHDRunGodunov::godunov_unsplit_gpu

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
					     DeviceArray<real_t>& d_UNew,
					     real_t dt, int nStep)
  {

    if (dimType == TWO_D) {

      // 2D Godunov unsplit kernel
      dim3 dimBlock(UNSPLIT_BLOCK_DIMX_2D,
		    UNSPLIT_BLOCK_DIMY_2D);
      
      dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_INNER_DIMX_2D), 
		   blocksFor(jsize, UNSPLIT_BLOCK_INNER_DIMY_2D));
      kernel_godunov_unsplit_mhd_2d<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							   d_UNew.data(),
							   d_UOld.pitch(), 
							   d_UOld.dimx(), 
							   d_UOld.dimy(), 
							   dt / dx, 
							   dt / dy,
							   dt,
							   gravityEnabled);
      checkCudaError("MHDRunGodunov :: kernel_godunov_unsplit_mhd_2d error");
      
      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt);
      }
      
    } else { // THREE_D

      // unimplemented
      std::cerr << "MHDRunGodunov::godunov_unsplit_gpu_v0 does not implement a 3D version" << std::endl;
      std::cerr << "3D GPU MHD implementation version 0 Will never be implemented ! To poor performances expected." << std::endl;

    } // 3D

  } // MHDRunGodunov::godunov_unsplit_gpu_v0

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
					     DeviceArray<real_t>& d_UNew,
					     real_t dt, int nStep)
  {

    if (dimType == TWO_D) {

      // NOTE : gravity source term is not implemented here
      // use implementation version 0 instead
      
      if (gravityEnabled) {
	std::cerr << "Gravity is not implemented in version 1 of 2D MHD.\n";
	std::cerr << "Use version 0 instead !\n";
      }
      
      {
	
	TIMER_START(timerTrace);
	// 2D Godunov unsplit kernel
	dim3 dimBlock(UNSPLIT_BLOCK_DIMX_2D_V1,
		      UNSPLIT_BLOCK_DIMY_2D_V1);
	
	dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_INNER_DIMX_2D_V1), 
		     blocksFor(jsize, UNSPLIT_BLOCK_INNER_DIMY_2D_V1));
	kernel_godunov_unsplit_mhd_2d_v1<<<dimGrid, dimBlock>>>(d_UOld.data(), 
								d_UNew.data(),
								d_qm_x.data(),
								d_qm_y.data(),
								d_qEdge_RT.data(),
								d_qEdge_RB.data(),
								d_qEdge_LT.data(),
								d_emf.data(),
								d_UOld.pitch(), 
								d_UOld.dimx(), 
								d_UOld.dimy(), 
								dt / dx, 
								dt / dy,
								dt);
	checkCudaError("MHDRunGodunov :: kernel_godunov_unsplit_mhd_2d_v1 error");
	TIMER_STOP(timerTrace);
      }
      
      {
	TIMER_START(timerUpdate);
	// 2D MHD kernel for constraint transport (CT) update
	dim3 dimBlock(UNSPLIT_BLOCK_DIMX_2D_V1_EMF,
		      UNSPLIT_BLOCK_DIMY_2D_V1_EMF);
	
	dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_DIMX_2D_V1_EMF), 
		     blocksFor(jsize, UNSPLIT_BLOCK_DIMY_2D_V1_EMF));
	kernel_mhd_2d_update_emf_v1<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							   d_UNew.data(),
							   d_emf.data(),
							   d_UOld.pitch(), 
							   d_UOld.dimx(), 
							   d_UOld.dimy(), 
							   dt / dx, 
							   dt / dy,
							   dt);
	checkCudaError("MHDRunGodunov :: kernel_mhd_2d_update_emf_v1 error");
	TIMER_STOP(timerUpdate);
      }
      
      TIMER_START(timerDissipative);
      {
	// update borders
	real_t &nu  = _gParams.nu;
	real_t &eta = _gParams.eta;
	if (nu>0 or eta>0) {
	  make_all_boundaries(d_UNew);
	}
	
	if (eta>0) {
	  // update magnetic field with resistivity emf
	  compute_resistivity_emf_2d(d_UNew, d_emf);
	  compute_ct_update_2d      (d_UNew, d_emf, dt);
	  
	  real_t &cIso = _gParams.cIso;
	  if (cIso<=0) { // non-isothermal simulations
	      // compute energy flux
	    compute_resistivity_energy_flux_2d(d_UNew, d_qm_x, d_qm_y, dt);
	    compute_hydro_update_energy       (d_UNew, d_qm_x, d_qm_y);
	  }
	}
	
	// compute viscosity
	if (nu>0) {
	  DeviceArray<real_t> &d_flux_x = d_qm_x;
	  DeviceArray<real_t> &d_flux_y = d_qm_y;
	  
	  compute_viscosity_flux(d_UNew, d_flux_x, d_flux_y, dt);
	  compute_hydro_update  (d_UNew, d_flux_x, d_flux_y);
	} // end compute viscosity force / update  
	
      } // end compute viscosity
      TIMER_STOP(timerDissipative);

    } else { // 3D

      // 3D Godunov unsplit kernel    
      dim3 dimBlock(UNSPLIT_BLOCK_DIMX_3D_V1,
		    UNSPLIT_BLOCK_DIMY_3D_V1);
      dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_INNER_DIMX_3D_V1), 
		   blocksFor(jsize, UNSPLIT_BLOCK_INNER_DIMY_3D_V1));
      kernel_godunov_unsplit_mhd_3d_v1<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							      d_UNew.data(), 
							      d_qm_x.data(),
							      d_qm_y.data(),
							      d_qm_z.data(),
							      d_qEdge_RT.data(),
							      d_qEdge_RB.data(),
							      d_qEdge_LT.data(),
							      d_qEdge_RT2.data(),
							      d_qEdge_RB2.data(),
							      d_qEdge_LT2.data(),
							      d_qEdge_RT3.data(),
							      d_qEdge_RB3.data(),
							      d_qEdge_LT3.data(),
							      d_UOld.pitch(), 
							      d_UOld.dimx(), 
							      d_UOld.dimy(), 
							      d_UOld.dimz(),
							      dt / dx, 
							      dt / dy,
							      dt / dz,
							      dt);
      checkCudaError("MHDRunGodunov :: kernel_godunov_unsplit_mhd_3d_v1 error");

    }

  } // MHDRunGodunov::godunov_unsplit_gpu_v1

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
					     DeviceArray<real_t>& d_UNew,
					     real_t dt, int nStep)
  {

    std::cerr << "2D / 3D GPU MHD implementation version 2 are NOT implemented !" << std::endl;


  } // MHDRunGodunov::godunov_unsplit_gpu_v2

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_gpu_v3(DeviceArray<real_t>& d_UOld, 
					     DeviceArray<real_t>& d_UNew,
					     real_t dt, int nStep)
  {

    TIMER_START(timerPrimVar);
    {
      // 3D primitive variables computation kernel    
      dim3 dimBlock(PRIM_VAR_MHD_BLOCK_DIMX_3D,
		    PRIM_VAR_MHD_BLOCK_DIMY_3D);
      dim3 dimGrid(blocksFor(isize, PRIM_VAR_MHD_BLOCK_DIMX_3D), 
		   blocksFor(jsize, PRIM_VAR_MHD_BLOCK_DIMY_3D));
      kernel_mhd_compute_primitive_variables_3D<<<dimGrid, 
	dimBlock>>>(d_UOld.data(), 
		    d_Q.data(),
		    d_UOld.pitch(),
		    d_UOld.dimx(),
		    d_UOld.dimy(), 
		    d_UOld.dimz(),
		    dt);
      checkCudaError("MHDRunGodunov :: kernel_mhd_compute_primitive_variables_3D error");
      
    }
    TIMER_STOP(timerPrimVar);
    
    TIMER_START(timerElecField);
    {
      // 3D Electric field computation kernel    
      dim3 dimBlock(ELEC_FIELD_BLOCK_DIMX_3D_V3,
		    ELEC_FIELD_BLOCK_DIMY_3D_V3);
      dim3 dimGrid(blocksFor(isize, ELEC_FIELD_BLOCK_INNER_DIMX_3D_V3), 
		   blocksFor(jsize, ELEC_FIELD_BLOCK_INNER_DIMY_3D_V3));
      kernel_mhd_compute_elec_field<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							   d_Q.data(),
							   d_elec.data(),
							   d_UOld.pitch(), 
							   d_UOld.dimx(), 
							   d_UOld.dimy(), 
							   d_UOld.dimz(),
							   dt);
      checkCudaError("MHDRunGodunov :: kernel_mhd_compute_elec_field error");
      
      if (dumpDataForDebugEnabled) {
	d_elec.copyToHost(h_debug2);
	outputHdf5Debug(h_debug2, "elec_", nStep);
      }
      
    }
    TIMER_STOP(timerElecField);
    
    TIMER_START(timerMagSlopes);
    {
      // magnetic slopes computations
      dim3 dimBlock(MAG_SLOPES_BLOCK_DIMX_3D_V3,
		    MAG_SLOPES_BLOCK_DIMY_3D_V3);
      dim3 dimGrid(blocksFor(isize, MAG_SLOPES_BLOCK_INNER_DIMX_3D_V3), 
		   blocksFor(jsize, MAG_SLOPES_BLOCK_INNER_DIMY_3D_V3));
      kernel_mhd_compute_mag_slopes<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							   d_dA.data(),
							   d_dB.data(),
							   d_dC.data(),
							   d_UOld.pitch(), 
							   d_UOld.dimx(), 
							   d_UOld.dimy(), 
							   d_UOld.dimz() );
      checkCudaError("MHDRunGodunov :: kernel_mhd_compute_mag_slopes error");
      
    }
    TIMER_STOP(timerMagSlopes);
    
    TIMER_START(timerTraceUpdate);
    {
      // trace and update
      dim3 dimBlock(TRACE_BLOCK_DIMX_3D_V3,
		    TRACE_BLOCK_DIMY_3D_V3);
      dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_INNER_DIMX_3D_V3), 
		   blocksFor(jsize, TRACE_BLOCK_INNER_DIMY_3D_V3));
      kernel_mhd_compute_trace<<<dimGrid, dimBlock>>>(d_UOld.data(), 
						      d_UNew.data(), 
						      d_Q.data(),
						      d_dA.data(),
						      d_dB.data(),
						      d_dC.data(),
						      d_elec.data(),
						      d_qm_x.data(),
						      d_qm_y.data(),
						      d_qm_z.data(),
						      d_qEdge_RT.data(),
						      d_qEdge_RB.data(),
						      d_qEdge_LT.data(),
						      d_qEdge_RT2.data(),
						      d_qEdge_RB2.data(),
						      d_qEdge_LT2.data(),
						      d_qEdge_RT3.data(),
						      d_qEdge_RB3.data(),
						      d_qEdge_LT3.data(),
						      d_UOld.pitch(), 
						      d_UOld.dimx(), 
						      d_UOld.dimy(), 
						      d_UOld.dimz(),
						      dt / dx, 
						      dt / dy,
						      dt / dz,
						      dt );
      checkCudaError("MHDRunGodunov :: kernel_mhd_compute_trace error");

      // dump data for debug
      if (dumpDataForDebugEnabled) {
	d_dA.copyToHost(h_debug2);
	outputHdf5Debug(h_debug2, "dA_", nStep);
	d_dB.copyToHost(h_debug2);
	outputHdf5Debug(h_debug2, "dB_", nStep);
	d_dC.copyToHost(h_debug2);
	outputHdf5Debug(h_debug2, "dC_", nStep);
	    
	d_qm_x.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qm_x_", nStep);
	d_qm_y.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qm_y_", nStep);
	d_qm_z.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qm_z_", nStep);
	    
	/*d_qp_x.copyToHost(h_debug);
	  outputHdf5Debug(h_debug, "qp_x_", nStep);
	  d_qp_y.copyToHost(h_debug);
	  outputHdf5Debug(h_debug, "qp_y_", nStep);
	  d_qp_z.copyToHost(h_debug);
	  outputHdf5Debug(h_debug, "qp_z_", nStep);*/
	    
	d_qEdge_RT.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qEdge_RT_", nStep);
	d_qEdge_RB.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qEdge_RB_", nStep);
	d_qEdge_LT.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qEdge_LT_", nStep);
	/*d_qEdge_LB.copyToHost(h_debug);
	  outputHdf5Debug(h_debug, "qEdge_LB_", nStep);*/

	// do something to get fluxes and emf arrays...

      } // end dumpDataForDebugEnabled
	      
    } // kernel_mhd_compute_trace
    TIMER_STOP(timerTraceUpdate);

    TIMER_START(timerDissipative);
    // update borders
    real_t &nu  = _gParams.nu;
    real_t &eta = _gParams.eta;
    if (nu>0 or eta>0) {
      make_all_boundaries(d_UNew);
    }

    if (eta>0) {
      // update magnetic field with resistivity emf
      compute_resistivity_emf_3d(d_UNew, d_emf);
      compute_ct_update_3d      (d_UNew, d_emf, dt);
	  
      real_t &cIso = _gParams.cIso;
      if (cIso<=0) { // non-isothermal simulations
	// compute energy flux
	compute_resistivity_energy_flux_3d(d_UNew, d_qm_x, d_qm_y, d_qm_z, dt);
	compute_hydro_update_energy       (d_UNew, d_qm_x, d_qm_y, d_qm_z);
      }
    }

    // compute viscosity
    if (nu>0) {
      DeviceArray<real_t> &d_flux_x = d_qm_x;
      DeviceArray<real_t> &d_flux_y = d_qm_y;
      DeviceArray<real_t> &d_flux_z = d_qm_z;
	  
      compute_viscosity_flux(d_UNew, d_flux_x, d_flux_y, d_flux_z, dt);
      compute_hydro_update  (d_UNew, d_flux_x, d_flux_y, d_flux_z );
    } // end compute viscosity force / update  
    TIMER_STOP(timerDissipative);

    /*
     * random forcing
     */
    if (randomForcingEnabled) {
	  
      real_t norm = compute_random_forcing_normalization(d_UNew, dt);
	  
      add_random_forcing(d_UNew, dt, norm);
	  
    }
    if (randomForcingOrnsteinUhlenbeckEnabled) {
	  
      // add forcing field in real space
      pForcingOrnsteinUhlenbeck->add_forcing_field(d_UNew, dt);
	  
    }
  
  } // MHDRunGodunov::godunov_unsplit_gpu_v3
  
  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_gpu_v4(DeviceArray<real_t>& d_UOld, 
					     DeviceArray<real_t>& d_UNew,
					     real_t dt, int nStep)
  {

    TIMER_START(timerPrimVar);
    {
      // 3D primitive variables computation kernel    
      dim3 dimBlock(PRIM_VAR_MHD_BLOCK_DIMX_3D,
		    PRIM_VAR_MHD_BLOCK_DIMY_3D);
      dim3 dimGrid(blocksFor(isize, PRIM_VAR_MHD_BLOCK_DIMX_3D), 
		   blocksFor(jsize, PRIM_VAR_MHD_BLOCK_DIMY_3D));
      kernel_mhd_compute_primitive_variables_3D<<<dimGrid, 
	dimBlock>>>(d_UOld.data(), 
		    d_Q.data(),
		    d_UOld.pitch(),
		    d_UOld.dimx(),
		    d_UOld.dimy(), 
		    d_UOld.dimz(),
		    dt);
      checkCudaError("MHDRunGodunov :: kernel_mhd_compute_primitive_variables_3D error");
	
    }
    TIMER_STOP(timerPrimVar);

    TIMER_START(timerElecField);
    {
      // 3D Electric field computation kernel    
      dim3 dimBlock(ELEC_FIELD_BLOCK_DIMX_3D_V3,
		    ELEC_FIELD_BLOCK_DIMY_3D_V3);
      dim3 dimGrid(blocksFor(isize, ELEC_FIELD_BLOCK_INNER_DIMX_3D_V3), 
		   blocksFor(jsize, ELEC_FIELD_BLOCK_INNER_DIMY_3D_V3));
      kernel_mhd_compute_elec_field<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							   d_Q.data(),
							   d_elec.data(),
							   d_UOld.pitch(), 
							   d_UOld.dimx(), 
							   d_UOld.dimy(), 
							   d_UOld.dimz(),
							   dt);
      checkCudaError("MHDRunGodunov :: kernel_mhd_compute_elec_field error");

      if (dumpDataForDebugEnabled) {
	d_elec.copyToHost(h_debug2);
	outputHdf5Debug(h_debug2, "elec_", nStep);
      }
	
    }
    TIMER_STOP(timerElecField);

    TIMER_START(timerMagSlopes);
    {
      // magnetic slopes computations
      dim3 dimBlock(MAG_SLOPES_BLOCK_DIMX_3D_V3,
		    MAG_SLOPES_BLOCK_DIMY_3D_V3);
      dim3 dimGrid(blocksFor(isize, MAG_SLOPES_BLOCK_INNER_DIMX_3D_V3), 
		   blocksFor(jsize, MAG_SLOPES_BLOCK_INNER_DIMY_3D_V3));
      kernel_mhd_compute_mag_slopes<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							   d_dA.data(),
							   d_dB.data(),
							   d_dC.data(),
							   d_UOld.pitch(), 
							   d_UOld.dimx(), 
							   d_UOld.dimy(), 
							   d_UOld.dimz() );
      checkCudaError("MHDRunGodunov :: kernel_mhd_compute_mag_slopes error");

    }
    TIMER_STOP(timerMagSlopes);

    TIMER_START(timerTrace);
    // trace
    {
      dim3 dimBlock(TRACE_BLOCK_DIMX_3D_V4,
		    TRACE_BLOCK_DIMY_3D_V4);
      dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_INNER_DIMX_3D_V4), 
		   blocksFor(jsize, TRACE_BLOCK_INNER_DIMY_3D_V4));
      kernel_mhd_compute_trace_v4<<<dimGrid, dimBlock>>>(d_UOld.data(),
							 d_Q.data(),
							 d_dA.data(),
							 d_dB.data(),
							 d_dC.data(),
							 d_elec.data(),
							 d_qm_x.data(),
							 d_qm_y.data(),
							 d_qm_z.data(),
							 d_qp_x.data(),
							 d_qp_y.data(),
							 d_qp_z.data(),
							 d_qEdge_RT.data(),
							 d_qEdge_RB.data(),
							 d_qEdge_LT.data(),
							 d_qEdge_LB.data(),
							 d_qEdge_RT2.data(),
							 d_qEdge_RB2.data(),
							 d_qEdge_LT2.data(),
							 d_qEdge_LB2.data(),
							 d_qEdge_RT3.data(),
							 d_qEdge_RB3.data(),
							 d_qEdge_LT3.data(),
							 d_qEdge_LB3.data(),
							 d_UOld.pitch(), 
							 d_UOld.dimx(), 
							 d_UOld.dimy(), 
							 d_UOld.dimz(),
							 dt / dx, 
							 dt / dy,
							 dt / dz,
							 dt);
      checkCudaError("MHDRunGodunov kernel_mhd_compute_trace_v4 error");

      // dump data for debug
      if (dumpDataForDebugEnabled) {
	d_dA.copyToHost(h_debug2);
	outputHdf5Debug(h_debug2, "dA_", nStep);
	d_dB.copyToHost(h_debug2);
	outputHdf5Debug(h_debug2, "dB_", nStep);
	d_dC.copyToHost(h_debug2);
	outputHdf5Debug(h_debug2, "dC_", nStep);
	    
	d_qm_x.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qm_x_", nStep);
	d_qm_y.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qm_y_", nStep);
	d_qm_z.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qm_z_", nStep);
	    
	/*d_qp_x.copyToHost(h_debug);
	  outputHdf5Debug(h_debug, "qp_x_", nStep);
	  d_qp_y.copyToHost(h_debug);
	  outputHdf5Debug(h_debug, "qp_y_", nStep);
	  d_qp_z.copyToHost(h_debug);
	  outputHdf5Debug(h_debug, "qp_z_", nStep);*/
	    
	d_qEdge_RT.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qEdge_RT_", nStep);
	d_qEdge_RB.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qEdge_RB_", nStep);
	d_qEdge_LT.copyToHost(h_debug);
	outputHdf5Debug(h_debug, "qEdge_LT_", nStep);
	/*d_qEdge_LB.copyToHost(h_debug);
	  outputHdf5Debug(h_debug, "qEdge_LB_", nStep);*/

	// do something to get fluxes and emf arrays...

      } // end dumpDataForDebugEnabled

      // gravity predictor
      if (gravityEnabled) {
	dim3 dimBlock(GRAV_PRED_BLOCK_DIMX_3D_V4,
		      GRAV_PRED_BLOCK_DIMY_3D_V4);
	dim3 dimGrid(blocksFor(isize, GRAV_PRED_BLOCK_DIMX_3D_V4), 
		     blocksFor(jsize, GRAV_PRED_BLOCK_DIMY_3D_V4));
	kernel_mhd_compute_gravity_predictor_v4<<<dimGrid, 
	  dimBlock>>>(d_qm_x.data(),
		      d_qm_y.data(),
		      d_qm_z.data(),
		      d_qp_x.data(),
		      d_qp_y.data(),
		      d_qp_z.data(),
		      d_qEdge_RT.data(),
		      d_qEdge_RB.data(),
		      d_qEdge_LT.data(),
		      d_qEdge_LB.data(),
		      d_qEdge_RT2.data(),
		      d_qEdge_RB2.data(),
		      d_qEdge_LT2.data(),
		      d_qEdge_LB2.data(),
		      d_qEdge_RT3.data(),
		      d_qEdge_RB3.data(),
		      d_qEdge_LT3.data(),
		      d_qEdge_LB3.data(),
		      d_UOld.pitch(), 
		      d_UOld.dimx(), 
		      d_UOld.dimy(), 
		      d_UOld.dimz(),
		      dt);
	checkCudaError("MHDRunGodunov kernel_mhd_compute_gravity_predictor_v4 error");
	  
      } // end gravity predictor

    } // end trace
    TIMER_STOP(timerTrace);
	
    TIMER_START(timerUpdate);	
    // update hydro
    // {
    //   dim3 dimBlock(UPDATE_BLOCK_DIMX_3D_V4_OLD,
    // 		UPDATE_BLOCK_DIMY_3D_V4_OLD);
    //   dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_3D_V4_OLD), 
    // 	       blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_3D_V4_OLD));
    //   kernel_mhd_flux_update_hydro_v4_old<<<dimGrid, dimBlock>>>(d_UOld.data(),
    // 							 d_UNew.data(),
    // 							 d_qm_x.data(),
    // 							 d_qm_y.data(),
    // 							 d_qm_z.data(),
    // 							 d_qp_x.data(),
    // 							 d_qp_y.data(),
    // 							 d_qp_z.data(),
    // 							 d_qEdge_RT.data(),
    // 							 d_qEdge_RB.data(),
    // 							 d_qEdge_LT.data(),
    // 							 d_qEdge_LB.data(),
    // 							 d_qEdge_RT2.data(),
    // 							 d_qEdge_RB2.data(),
    // 							 d_qEdge_LT2.data(),
    // 							 d_qEdge_LB2.data(),
    // 							 d_qEdge_RT3.data(),
    // 							 d_qEdge_RB3.data(),
    // 							 d_qEdge_LT3.data(),
    // 							 d_qEdge_LB3.data(),
    // 							 d_emf.data(),
    // 							 d_UOld.pitch(), 
    // 							 d_UOld.dimx(), 
    // 							 d_UOld.dimy(), 
    // 							 d_UOld.dimz(),
    // 							 dt / dx, 
    // 							 dt / dy,
    // 							 dt / dz,
    // 							 dt );
    //   checkCudaError("MHDRunGodunov kernel_mhd_flux_update_hydro_v4 error");

    // } // end update hydro
    {
      dim3 dimBlock(UPDATE_BLOCK_DIMX_3D_V4,
		    UPDATE_BLOCK_DIMY_3D_V4);
      dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_3D_V4), 
		   blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_3D_V4));
      kernel_mhd_flux_update_hydro_v4<<<dimGrid, dimBlock>>>(d_UOld.data(),
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
							     dt );
      checkCudaError("MHDRunGodunov kernel_mhd_flux_update_hydro_v4 error");

    } // end update hydro

    // gravity source term
    if (gravityEnabled) {
      compute_gravity_source_term(d_UNew, d_UOld, dt);
    }
    TIMER_STOP(timerUpdate);
	
    TIMER_START(timerEmf);
    // compute emf
    {
      dim3 dimBlock(COMPUTE_EMF_BLOCK_DIMX_3D_V4,
		    COMPUTE_EMF_BLOCK_DIMY_3D_V4);
      dim3 dimGrid(blocksFor(isize, COMPUTE_EMF_BLOCK_DIMX_3D_V4), 
		   blocksFor(jsize, COMPUTE_EMF_BLOCK_DIMY_3D_V4));
      kernel_mhd_compute_emf_v4<<<dimGrid, dimBlock>>>(d_qEdge_RT.data(),
						       d_qEdge_RB.data(),
						       d_qEdge_LT.data(),
						       d_qEdge_LB.data(),
						       d_qEdge_RT2.data(),
						       d_qEdge_RB2.data(),
						       d_qEdge_LT2.data(),
						       d_qEdge_LB2.data(),
						       d_qEdge_RT3.data(),
						       d_qEdge_RB3.data(),
						       d_qEdge_LT3.data(),
						       d_qEdge_LB3.data(),
						       d_emf.data(),
						       d_UOld.pitch(), 
						       d_UOld.dimx(), 
						       d_UOld.dimy(), 
						       d_UOld.dimz(),
						       dt / dx, 
						       dt / dy,
						       dt / dz,
						       dt );
      checkCudaError("MHDRunGodunov kernel_mhd_compute_emf_v4 error");

    } // end compute emf
    TIMER_STOP(timerEmf);

    TIMER_START(timerCtUpdate);
    // update magnetic field
    {
      dim3 dimBlock(UPDATE_CT_BLOCK_DIMX_3D_V4,
		    UPDATE_CT_BLOCK_DIMY_3D_V4);
      dim3 dimGrid(blocksFor(isize, UPDATE_CT_BLOCK_DIMX_3D_V4), 
		   blocksFor(jsize, UPDATE_CT_BLOCK_DIMY_3D_V4));
      kernel_mhd_flux_update_ct_v4<<<dimGrid, dimBlock>>>(d_UOld.data(),
							  d_UNew.data(),
							  d_emf.data(),
							  d_UOld.pitch(), 
							  d_UOld.dimx(), 
							  d_UOld.dimy(), 
							  d_UOld.dimz(),
							  dt / dx, 
							  dt / dy,
							  dt / dz,
							  dt );
      checkCudaError("MHDRunGodunov kernel_mhd_flux_update_ct_v4 error");
    } // update magnetic field
    TIMER_STOP(timerCtUpdate);
	
    TIMER_START(timerDissipative);
    // update borders
    real_t &nu  = _gParams.nu;
    real_t &eta = _gParams.eta;
    if (nu>0 or eta>0) {
      make_all_boundaries(d_UNew);
    }

    if (eta>0) {
      // update magnetic field with resistivity emf
      compute_resistivity_emf_3d(d_UNew, d_emf);
      compute_ct_update_3d      (d_UNew, d_emf, dt);
	  
      real_t &cIso = _gParams.cIso;
      if (cIso<=0) { // non-isothermal simulations
	// compute energy flux
	compute_resistivity_energy_flux_3d(d_UNew, d_qm_x, d_qm_y, d_qm_z, dt);
	compute_hydro_update_energy       (d_UNew, d_qm_x, d_qm_y, d_qm_z);
      }
    }
	
    // compute viscosity
    if (nu>0) {
      DeviceArray<real_t> &d_flux_x = d_qm_x;
      DeviceArray<real_t> &d_flux_y = d_qm_y;
      DeviceArray<real_t> &d_flux_z = d_qm_z;
	  
      compute_viscosity_flux(d_UNew, d_flux_x, d_flux_y, d_flux_z, dt);
      compute_hydro_update  (d_UNew, d_flux_x, d_flux_y, d_flux_z );
    } // end compute viscosity force / update  
    TIMER_STOP(timerDissipative);

    /*
     * random forcing
     */
    if (randomForcingEnabled) {
	  
      real_t norm = compute_random_forcing_normalization(d_UNew, dt);
	  
      add_random_forcing(d_UNew, dt, norm);
	  
    }
    if (randomForcingOrnsteinUhlenbeckEnabled) {
	  
      // add forcing field in real space
      pForcingOrnsteinUhlenbeck->add_forcing_field(d_UNew, dt);
	  
    }

  } // MHDRunGodunov::godunov_unsplit_gpu_v4

#else // CPU version

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_cpu(HostArray<real_t>& h_UOld, 
					  HostArray<real_t>& h_UNew, 
					  real_t dt, int nStep)
  {

    /*
     * update h_UOld boundary, before updating h_U
     */
    TIMER_START(timerBoundaries);
    make_all_boundaries(h_UOld);
    TIMER_STOP(timerBoundaries);

    // copy h_UOld into h_UNew
    // for (unsigned int indexGlob=0; indexGlob<h_UOld.size(); indexGlob++) {
    //   h_UNew(indexGlob) = h_UOld(indexGlob);
    // }
    h_UOld.copyTo(h_UNew);

#ifdef USE_PAPI
      // PAPI FLOPS counter
      papiFlops_total.start();
#endif // USE_PAPI      

    /*
     * main computation loop to update h_U : 2D loop over simulation domain location
     */
    TIMER_START(timerGodunov);

    // convert conservative to primitive variables (and source term predictor)
    // put results in h_Q object
    convertToPrimitives(h_UOld.data(), dt);
    
    if (implementationVersion == 0) { // this the memory scrooge version

      godunov_unsplit_cpu_v0(h_UOld, h_UNew, dt, nStep);
           
    } else if (implementationVersion == 1) { 

      godunov_unsplit_cpu_v1(h_UOld, h_UNew, dt, nStep);

    } else if (implementationVersion == 2) {
      
      godunov_unsplit_cpu_v2(h_UOld, h_UNew, dt, nStep);

    } else if (implementationVersion == 3 or implementationVersion == 4) {

      godunov_unsplit_cpu_v3(h_UOld, h_UNew, dt, nStep);
      
    } // end implementationVersion == 3 / 4
    TIMER_STOP(timerGodunov);
  
#ifdef USE_PAPI
    // PAPI FLOPS counter
    papiFlops_total.stop();
#endif // USE_PAPI      

  } // MHDRunGodunov::godunov_unsplit_cpu
  
  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_cpu_v0_old(HostArray<real_t>& h_UOld, 
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

    if (dimType == TWO_D) {
      
	// we need to store qm/qp/qEdge for current position (x,y) and
	// (x-1,y), and (x,y-1) and (x-1,y-1) 
	// that is 4 positions in total
	real_t qm_x[2][2][NVAR_MHD];
	real_t qm_y[2][2][NVAR_MHD];

	real_t qp_x[2][2][NVAR_MHD];
	real_t qp_y[2][2][NVAR_MHD];

	// store what's needed to computed EMF's
	real_t qEdge_emf[2][2][4][NVAR_MHD];

#ifdef _OPENMP
#pragma omp parallel default(shared) private(qm_x,qm_y,qp_x,qp_y,qEdge_emf)
#pragma omp for collapse(2) schedule(auto)
#endif // _OPENMP
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {

	    real_t qNb[3][3][NVAR_MHD];
	    real_t bfNb[4][4][3];
	    real_t qm[TWO_D][NVAR_MHD];
	    real_t qp[TWO_D][NVAR_MHD];
	    real_t qEdge[4][NVAR_MHD]; // array for qRT, qRB, qLT, qLB
	    real_t c=0;

	    // compute qm, qp for the 2x2 positions
	    for (int posY=0; posY<2; posY++) {
	      for (int posX=0; posX<2; posX++) {
  
		int ii=i-posX;
		int jj=j-posY;
	      
		/*
		 * compute qm, qp and qEdge (qRT, qRB, qLT, qLB)
		 */
	      
		// prepare qNb : q state in the 3-by-3 neighborhood
		// note that current cell (ii,jj) is in qNb[1][1]
		// also note that the effective stencil is 4-by-4 since
		// computation of primitive variable (q) requires mag
		// field on the right (see computePrimitives_MHD_2D)
		for (int di=0; di<3; di++)
		  for (int dj=0; dj<3; dj++) {
		    for (int iVar=0; iVar < NVAR_MHD; iVar++) {
		      int indexLoc = (ii+di-1)+(jj+dj-1)*isize; // centered index
		      getPrimitiveVector(Q, arraySize, indexLoc, qNb[di][dj]);
		    }
		  }
	      
		// prepare bfNb : bf (face centered mag field) in the
		// 4-by-4 neighborhood
		// note that current cell (ii,jj) is in bfNb[1][1]
		for (int di=0; di<4; di++)
		  for (int dj=0; dj<4; dj++) {
		    int indexLoc = (ii+di-1)+(jj+dj-1)*isize;
		    getMagField(U, arraySize, indexLoc, bfNb[di][dj]);
		  }
	      
		real_t xPos = ::gParams.xMin + dx/2 + (ii-ghostWidth)*dx;
	      
		// compute trace 2d finally !!! 
		trace_unsplit_mhd_2d(qNb, bfNb, c, dtdx, dtdy, xPos, qm, qp, qEdge);
	      
		// store qm, qp, qEdge : only what is really needed
		for (int ivar=0; ivar<NVAR_MHD; ivar++) {
		  qm_x[posX][posY][ivar] = qm[0][ivar];
		  qp_x[posX][posY][ivar] = qp[0][ivar];
		  qm_y[posX][posY][ivar] = qm[1][ivar];
		  qp_y[posX][posY][ivar] = qp[1][ivar];
		
		  qEdge_emf[posX][posY][IRT][ivar] = qEdge[IRT][ivar]; 
		  qEdge_emf[posX][posY][IRB][ivar] = qEdge[IRB][ivar]; 
		  qEdge_emf[posX][posY][ILT][ivar] = qEdge[ILT][ivar]; 
		  qEdge_emf[posX][posY][ILB][ivar] = qEdge[ILB][ivar]; 
		} // end for ivar
	      
	      } // end for posX
	    } // end for posY
	  	  
	    if (gravityEnabled) { 
	      // we need to modify input to flux computation with
	      // gravity predictor (half time step)
	      for (int posY=0; posY<2; posY++) {
		for (int posX=0; posX<2; posX++) {
		  
		  int ii=i-posX;
		  int jj=j-posY;

		  qm_x[posX][posY][IU] += HALF_F * dt * h_gravity(ii,jj,IX);
		  qm_x[posX][posY][IV] += HALF_F * dt * h_gravity(ii,jj,IY);

		  qp_x[posX][posY][IU] += HALF_F * dt * h_gravity(ii,jj,IX);
		  qp_x[posX][posY][IV] += HALF_F * dt * h_gravity(ii,jj,IY);
	      
		  qm_y[posX][posY][IU] += HALF_F * dt * h_gravity(ii,jj,IX);
		  qm_y[posX][posY][IV] += HALF_F * dt * h_gravity(ii,jj,IY);
	      
		  qp_y[posX][posY][IU] += HALF_F * dt * h_gravity(ii,jj,IX);
		  qp_y[posX][posY][IV] += HALF_F * dt * h_gravity(ii,jj,IY);

		  qEdge_emf[posX][posY][IRT][IU] += HALF_F * dt * h_gravity(ii,jj,IX);
		  qEdge_emf[posX][posY][IRT][IV] += HALF_F * dt * h_gravity(ii,jj,IY);
	      
		  qEdge_emf[posX][posY][IRB][IU] += HALF_F * dt * h_gravity(ii,jj,IX);
		  qEdge_emf[posX][posY][IRB][IV] += HALF_F * dt * h_gravity(ii,jj,IY);
	      
		  qEdge_emf[posX][posY][ILT][IU] += HALF_F * dt * h_gravity(ii,jj,IX);
		  qEdge_emf[posX][posY][ILT][IV] += HALF_F * dt * h_gravity(ii,jj,IY);
	      
		  qEdge_emf[posX][posY][ILB][IU] += HALF_F * dt * h_gravity(ii,jj,IX);
		  qEdge_emf[posX][posY][ILB][IV] += HALF_F * dt * h_gravity(ii,jj,IY);
	      
		} // end for posX
	      } // end posY
	    } // end gravityEnabled

	    real_riemann_t qleft[NVAR_MHD];
	    real_riemann_t qright[NVAR_MHD];
	    real_riemann_t flux_x[NVAR_MHD];
	    real_riemann_t flux_y[NVAR_MHD];

	    /*
	     * Solve Riemann problem at X-interfaces and compute
	     * X-fluxes
	     *
	     * Note that continuity of normal component of magnetic
	     * field is ensured inside riemann_mhd routine.
	     */

	    // even in 2D we need to fill IW index (to respect
	    // riemann_mhd interface)
	    qleft[ID]   = qm_x[1][0][ID];
	    qleft[IP]   = qm_x[1][0][IP];
	    qleft[IU]   = qm_x[1][0][IU];
	    qleft[IV]   = qm_x[1][0][IV];
	    qleft[IW]   = qm_x[1][0][IW];
	    qleft[IA]   = qm_x[1][0][IA];
	    qleft[IB]   = qm_x[1][0][IB];
	    qleft[IC]   = qm_x[1][0][IC];

	    qright[ID]  = qp_x[0][0][ID];
	    qright[IP]  = qp_x[0][0][IP];
	    qright[IU]  = qp_x[0][0][IU];
	    qright[IV]  = qp_x[0][0][IV];
	    qright[IW]  = qp_x[0][0][IW];
	    qright[IA]  = qp_x[0][0][IA];
	    qright[IB]  = qp_x[0][0][IB];
	    qright[IC]  = qp_x[0][0][IC];

	    // compute hydro flux_x
	    riemann_mhd(qleft,qright,flux_x);

	
	    // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	    qleft[ID]   = qm_y[0][1][ID];
	    qleft[IP]   = qm_y[0][1][IP];
	    qleft[IU]   = qm_y[0][1][IV]; // watchout IU, IV permutation
	    qleft[IV]   = qm_y[0][1][IU]; // watchout IU, IV permutation
	    qleft[IW]   = qm_y[0][1][IW];
	    qleft[IA]   = qm_y[0][1][IB]; // watchout IA, IB permutation
	    qleft[IB]   = qm_y[0][1][IA]; // watchout IA, IB permutation
	    qleft[IC]   = qm_y[0][1][IC];

	    qright[ID]  = qp_y[0][0][ID];
	    qright[IP]  = qp_y[0][0][IP];
	    qright[IU]  = qp_y[0][0][IV]; // watchout IU, IV permutation
	    qright[IV]  = qp_y[0][0][IU]; // watchout IU, IV permutation
	    qright[IW]  = qp_y[0][0][IW];
	    qright[IA]  = qp_y[0][0][IB]; // watchout IA, IB permutation
	    qright[IB]  = qp_y[0][0][IA]; // watchout IA, IB permutation
	    qright[IC]  = qp_y[0][0][IC];

	    // compute hydro flux_y
	    riemann_mhd(qleft,qright,flux_y);


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
	      qEdge_emfZ[IRT][iVar] = qEdge_emf[1][1][IRT][iVar]; 
	      qEdge_emfZ[IRB][iVar] = qEdge_emf[1][0][IRB][iVar]; 
	      qEdge_emfZ[ILT][iVar] = qEdge_emf[0][1][ILT][iVar]; 
	      qEdge_emfZ[ILB][iVar] = qEdge_emf[0][0][ILB][iVar]; 
	    }

	    // actually compute emfZ
	    real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ);
	    h_emf(i,j,I_EMFZ) = emfZ;

	  } // end for j
	} // end for i

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

	// gravity source term
	if (gravityEnabled) {
	  compute_gravity_source_term(h_UNew, h_UOld, dt);
	}

    } else { // THREE_D - implementation version 0
    
      // we need to store qm/qp/qEdge for current position (x,y,z) and
      // neighbors in a 2-by-2-by-2 cube
      // that is 8 positions in total

      real_t qm_x[2][2][2][NVAR_MHD];
      real_t qm_y[2][2][2][NVAR_MHD];
      real_t qm_z[2][2][2][NVAR_MHD];
	
      real_t qp_x[2][2][2][NVAR_MHD];
      real_t qp_y[2][2][2][NVAR_MHD];
      real_t qp_z[2][2][2][NVAR_MHD];

      // store what's needed to computed EMF's
      real_t qEdge_emf[2][2][2][4][3][NVAR_MHD];

#ifdef _OPENMP
#pragma omp parallel default(shared) private(qm_x,qm_y,qm_z,qp_x,qp_y,qp_z,qEdge_emf)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPEMP
      for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	      
	    real_t qNb[3][3][3][NVAR_MHD];
	    real_t bfNb[4][4][4][3];
	    real_t qm[THREE_D][NVAR_MHD];
	    real_t qp[THREE_D][NVAR_MHD];
	    real_t qEdge[4][3][NVAR_MHD]; // array for qRT, qRB, qLT, qLB
	    real_t c=0;
	      
	    // compute qm, qp for the 2x2x2 positions
	    for (int posZ=0; posZ<2; posZ++) {
	      for (int posY=0; posY<2; posY++) {
		for (int posX=0; posX<2; posX++) {
		    
		  int ii=i-posX;
		  int jj=j-posY;
		  int kk=k-posZ;
		    
		    
		  /*
		   * compute qm, qp and qEdge (qRT, qRB, qLT, qLB)
		   */
		    
		  // prepare qNb : q state in the 3-by-3-by-3 neighborhood
		  // note that current cell (ii,jj,kk) is in qNb[1][1][1]
		  // also note that the effective stencil is 4-by-4-by-4 since
		  // computation of primitive variable (q) requires mag
		  // field on the right (see computePrimitives_MHD_3D)
		  for (int di=0; di<3; di++) {
		    for (int dj=0; dj<3; dj++) {
		      for (int dk=0; dk<3; dk++) {
			for (int iVar=0; iVar < NVAR_MHD; iVar++) {
			  int indexLoc = (ii+di-1)+(jj+dj-1)*isize+(kk+dk-1)*isize*jsize; // centered index
			  getPrimitiveVector(Q, arraySize, indexLoc, qNb[di][dj][dk]);
			} // end for iVar
		      } // end for dk
		    } // end for dj
		  } // end for di
		    
		    
		    // prepare bfNb : bf (face centered mag field) in the
		    // 4-by-4-by-4 neighborhood
		    // note that current cell (ii,jj,kk) is in bfNb[1][1][1]
		  for (int di=0; di<4; di++) {
		    for (int dj=0; dj<4; dj++) {
		      for (int dk=0; dk<4; dk++) {
			int indexLoc = (ii+di-1)+(jj+dj-1)*isize+(kk+dk-1)*isize*jsize;
			getMagField(U, arraySize, indexLoc, bfNb[di][dj][dk]);
		      } // end for dk
		    } // end for dj
		  } // end for di
		    
		  real_t xPos = ::gParams.xMin + dx/2 + (ii-ghostWidth)*dx;
		    
		  // compute trace 3d finally !!! 
		  trace_unsplit_mhd_3d(qNb, bfNb, c, dtdx, dtdy, dtdz, xPos, qm, qp, qEdge);

		  // store qm, qp, qEdge : only what is really needed
		  for (int ivar=0; ivar<NVAR_MHD; ivar++) {
		    qm_x[posX][posY][posZ][ivar] = qm[0][ivar];
		    qp_x[posX][posY][posZ][ivar] = qp[0][ivar];
		    qm_y[posX][posY][posZ][ivar] = qm[1][ivar];
		    qp_y[posX][posY][posZ][ivar] = qp[1][ivar];
		    qm_z[posX][posY][posZ][ivar] = qm[2][ivar];
		    qp_z[posX][posY][posZ][ivar] = qp[2][ivar];
		      
		    qEdge_emf[posX][posY][posZ][IRT][0][ivar] = qEdge[IRT][0][ivar]; 
		    qEdge_emf[posX][posY][posZ][IRB][0][ivar] = qEdge[IRB][0][ivar]; 
		    qEdge_emf[posX][posY][posZ][ILT][0][ivar] = qEdge[ILT][0][ivar]; 
		    qEdge_emf[posX][posY][posZ][ILB][0][ivar] = qEdge[ILB][0][ivar]; 

		    qEdge_emf[posX][posY][posZ][IRT][1][ivar] = qEdge[IRT][1][ivar]; 
		    qEdge_emf[posX][posY][posZ][IRB][1][ivar] = qEdge[IRB][1][ivar]; 
		    qEdge_emf[posX][posY][posZ][ILT][1][ivar] = qEdge[ILT][1][ivar]; 
		    qEdge_emf[posX][posY][posZ][ILB][1][ivar] = qEdge[ILB][1][ivar]; 

		    qEdge_emf[posX][posY][posZ][IRT][2][ivar] = qEdge[IRT][2][ivar]; 
		    qEdge_emf[posX][posY][posZ][IRB][2][ivar] = qEdge[IRB][2][ivar]; 
		    qEdge_emf[posX][posY][posZ][ILT][2][ivar] = qEdge[ILT][2][ivar]; 
		    qEdge_emf[posX][posY][posZ][ILB][2][ivar] = qEdge[ILB][2][ivar]; 


		  } // end for ivar
		    
		} // end for posZ

	      } // end for posY

	    } // end for posX
	      
	      
	    real_riemann_t qleft[NVAR_MHD];
	    real_riemann_t qright[NVAR_MHD];
	    real_riemann_t flux_x[NVAR_MHD];
	    real_riemann_t flux_y[NVAR_MHD];
	    real_riemann_t flux_z[NVAR_MHD];
	    //real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

	    /*
	     * Solve Riemann problem at X-interfaces and compute
	     * X-fluxes
	     *
	     * Note that continuity of normal component of magnetic
	     * field is ensured inside riemann_mhd routine.
	     */
	      

	    qleft[ID]   = qm_x[1][0][0][ID];
	    qleft[IP]   = qm_x[1][0][0][IP];
	    qleft[IU]   = qm_x[1][0][0][IU];
	    qleft[IV]   = qm_x[1][0][0][IV];
	    qleft[IW]   = qm_x[1][0][0][IW];
	    qleft[IA]   = qm_x[1][0][0][IA];
	    qleft[IB]   = qm_x[1][0][0][IB];
	    qleft[IC]   = qm_x[1][0][0][IC];

	    qright[ID]  = qp_x[0][0][0][ID];
	    qright[IP]  = qp_x[0][0][0][IP];
	    qright[IU]  = qp_x[0][0][0][IU];
	    qright[IV]  = qp_x[0][0][0][IV];
	    qright[IW]  = qp_x[0][0][0][IW];
	    qright[IA]  = qp_x[0][0][0][IA];
	    qright[IB]  = qp_x[0][0][0][IB];
	    qright[IC]  = qp_x[0][0][0][IC];
	      
	    // compute hydro flux_x
	    riemann_mhd(qleft,qright,flux_x);
	      
	    /*
	     * Solve Riemann problem at Y-interfaces and compute
	     * Y-fluxes
	     */
	    qleft[ID]   = qm_y[0][1][0][ID];
	    qleft[IP]   = qm_y[0][1][0][IP];
	    qleft[IU]   = qm_y[0][1][0][IV]; // watchout IU, IV permutation
	    qleft[IV]   = qm_y[0][1][0][IU]; // watchout IU, IV permutation
	    qleft[IW]   = qm_y[0][1][0][IW];
	    qleft[IA]   = qm_y[0][1][0][IB]; // watchout IA, IB permutation
	    qleft[IB]   = qm_y[0][1][0][IA]; // watchout IA, IB permutation
	    qleft[IC]   = qm_y[0][1][0][IC];
	      
	    qright[ID]  = qp_y[0][0][0][ID];
	    qright[IP]  = qp_y[0][0][0][IP];
	    qright[IU]  = qp_y[0][0][0][IV]; // watchout IU, IV permutation
	    qright[IV]  = qp_y[0][0][0][IU]; // watchout IU, IV permutation
	    qright[IW]  = qp_y[0][0][0][IW];
	    qright[IA]  = qp_y[0][0][0][IB]; // watchout IA, IB permutation
	    qright[IB]  = qp_y[0][0][0][IA]; // watchout IA, IB permutation
	    qright[IC]  = qp_y[0][0][0][IC];
	      
	    // compute hydro flux_y
	    riemann_mhd(qleft,qright,flux_y);
	      
	    /*
	     * Solve Riemann problem at Z-interfaces and compute
	     * Z-fluxes
	     */
	    qleft[ID]   = qm_z[0][0][1][ID];
	    qleft[IP]   = qm_z[0][0][1][IP];
	    qleft[IU]   = qm_z[0][0][1][IW]; // watchout IU, IW permutation
	    qleft[IV]   = qm_z[0][0][1][IV];
	    qleft[IW]   = qm_z[0][0][1][IU]; // watchout IU, IW permutation
	    qleft[IA]   = qm_z[0][0][1][IC]; // watchout IA, IC permutation
	    qleft[IB]   = qm_z[0][0][1][IB];
	    qleft[IC]   = qm_z[0][0][1][IA]; // watchout IA, IC permutation
	      
	    qright[ID]  = qp_z[0][0][0][ID];
	    qright[IP]  = qp_z[0][0][0][IP];
	    qright[IU]  = qp_z[0][0][0][IW]; // watchout IU, IW permutation
	    qright[IV]  = qp_z[0][0][0][IV];
	    qright[IW]  = qp_z[0][0][0][IU]; // watchout IU, IW permutation
	    qright[IA]  = qp_z[0][0][0][IC]; // watchout IA, IC permutation
	    qright[IB]  = qp_z[0][0][0][IB];
	    qright[IC]  = qp_z[0][0][0][IA]; // watchout IA, IC permutation
	      
	    // compute hydro flux_z
	    riemann_mhd(qleft,qright,flux_z);
	      
	      
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
	      qEdge_emfZ[IRT][iVar] = qEdge_emf[1][1][0][IRT][2][iVar]; 
	      qEdge_emfZ[IRB][iVar] = qEdge_emf[1][0][0][IRB][2][iVar]; 
	      qEdge_emfZ[ILT][iVar] = qEdge_emf[0][1][0][ILT][2][iVar]; 
	      qEdge_emfZ[ILB][iVar] = qEdge_emf[0][0][0][ILB][2][iVar]; 
	    }
	    real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ);

	    // actually compute emfY (take care that RB and LT are
	    // swapped !!!)
	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      qEdge_emfY[IRT][iVar] = qEdge_emf[1][0][1][IRT][1][iVar]; 
	      qEdge_emfY[IRB][iVar] = qEdge_emf[0][0][1][ILT][1][iVar]; // ! swap
	      qEdge_emfY[ILT][iVar] = qEdge_emf[1][0][0][IRB][1][iVar]; // ! swap
	      qEdge_emfY[ILB][iVar] = qEdge_emf[0][0][0][ILB][1][iVar]; 
	    }
	    real_t emfY = compute_emf<EMFY>(qEdge_emfY);

	    // actually compute emfX
	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      qEdge_emfX[IRT][iVar] = qEdge_emf[0][1][1][IRT][0][iVar]; 
	      qEdge_emfX[IRB][iVar] = qEdge_emf[0][1][0][IRB][0][iVar];
	      qEdge_emfX[ILT][iVar] = qEdge_emf[0][0][1][ILT][0][iVar];
	      qEdge_emfX[ILB][iVar] = qEdge_emf[0][0][0][ILB][0][iVar]; 
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

  } // MHDRunGodunov::godunov_unsplit_cpu_v0_old

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_cpu_v1(HostArray<real_t>& h_UOld, 
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
	
      TIMER_START(timerTrace);
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

	  real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

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
      TIMER_STOP(timerTrace);

      TIMER_START(timerUpdate);
      // second compute fluxes from rieman solvers, and update
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	 
	  real_riemann_t qleft[NVAR_MHD];
	  real_riemann_t qright[NVAR_MHD];
	  real_riemann_t flux_x[NVAR_MHD];
	  real_riemann_t flux_y[NVAR_MHD];
	  //real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

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
      TIMER_STOP(timerUpdate);

      TIMER_START(timerCtUpdate);
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
      TIMER_STOP(timerCtUpdate);

      TIMER_START(timerDissipative);
      // update borders
      real_t &nu  = _gParams.nu;
      real_t &eta = _gParams.eta;
      if (nu>0 or eta>0) {
	make_all_boundaries(h_UNew);
      }

      if (eta>0) {
	// update magnetic field with resistivity emf
	compute_resistivity_emf_2d(h_UNew, h_emf);
	compute_ct_update_2d      (h_UNew, h_emf, dt);

	real_t &cIso = _gParams.cIso;
	if (cIso<=0) { // non-isothermal simulations
	  // compute energy flux
	  compute_resistivity_energy_flux_2d(h_UNew, h_qm_x, h_qm_y, dt);
	  compute_hydro_update_energy       (h_UNew, h_qm_x, h_qm_y);
	}
      }

      // compute viscosity forces
      if (nu>0) {
	// re-use h_qm_x and h_qm_y
	HostArray<real_t> &flux_x = h_qm_x;
	HostArray<real_t> &flux_y = h_qm_y;
	  
	compute_viscosity_flux(h_UNew, flux_x, flux_y, dt);
	compute_hydro_update  (h_UNew, flux_x, flux_y);
	  
      } // end compute viscosity force / update
      TIMER_STOP(timerDissipative);

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
	      
	    real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

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
	    //real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
			
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

    } // end THREE_D

  } // MHDRunGodunov::godunov_unsplit_cpu_v1

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_cpu_v2(HostArray<real_t>& h_UOld, 
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
	  //real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

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
	  //real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
 
	
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
	
      TIMER_START(timerDissipative);
      // update borders
      real_t &nu  = _gParams.nu;
      real_t &eta = _gParams.eta;
      if (nu>0 or eta>0) {
	make_all_boundaries(h_UNew);
      }

      if (eta>0) {
	// update magnetic field with resistivity emf
	compute_resistivity_emf_2d(h_UNew, h_emf);
	compute_ct_update_2d      (h_UNew, h_emf, dt);

	real_t &cIso = _gParams.cIso;
	if (cIso<=0) { // non-isothermal simulations
	  // compute energy flux
	  compute_resistivity_energy_flux_2d(h_UNew, h_qm_x, h_qm_y, dt);
	  compute_hydro_update_energy       (h_UNew, h_qm_x, h_qm_y);
	}
      }

      // compute viscosity forces
      if (nu>0) {
	// re-use h_qm_x and h_qm_y
	HostArray<real_t> &flux_x = h_qm_x;
	HostArray<real_t> &flux_y = h_qm_y;
	  
	compute_viscosity_flux(h_UNew, flux_x, flux_y, dt);
	compute_hydro_update  (h_UNew, flux_x, flux_y);
	  
      } // end compute viscosity force / update
      TIMER_STOP(timerDissipative);
    

    } else { // THREE_D

      std::cout << "CPU - 3D - implementation version 2 - Not implemented !!!" << std::endl;
	
    } // end THREE_D

  } // MHDRunGodunov::godunov_unsplit_cpu_v2

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

  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_cpu_v4(HostArray<real_t>& h_UOld, 
  					     HostArray<real_t>& h_UNew, 
  					     real_t dt, int nStep)
  {

    // there is no version 4 (see version 3)
    (void) h_UOld;
    (void) h_UNew;
    (void) dt;
    (void) nStep;

  } // MHDRunGodunov::godunov_unsplit_cpu_v4

#endif // __CUDACC__
  

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunGodunov::godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
						   DeviceArray<real_t>& d_UNew,
						   real_t dt, int nStep)
  {
    
    // inner domain integration
    TIMER_START(timerGodunov);

    // copy h_UOld into h_UNew
    d_UOld.copyTo(d_UNew);

    if (dimType == TWO_D) {

      // 2D Godunov unsplit kernel
      dim3 dimBlock(UNSPLIT_BLOCK_DIMX_2D_V1,
		    UNSPLIT_BLOCK_DIMY_2D_V1);
      
      dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_INNER_DIMX_2D_V1), 
		   blocksFor(jsize, UNSPLIT_BLOCK_INNER_DIMY_2D_V1));
      kernel_godunov_unsplit_mhd_rotating_2d_v1<<<dimGrid, dimBlock>>>(d_UOld.data(), 
								       d_UNew.data(),
								       d_qm_x.data(),
								       d_qm_y.data(),
								       d_qEdge_RT.data(),
								       d_qEdge_RB.data(),
								       d_qEdge_LT.data(),
								       d_UOld.pitch(), 
								       d_UOld.dimx(), 
								       d_UOld.dimy(), 
								       dt / dx, 
								       dt / dy,
								       dt);
      checkCudaError("MHDRunGodunov :: kernel_godunov_unsplit_mhd_rotating_2d_v1 error");
      
    } else { // THREE_D
	
      TIMER_START(timerPrimVar);
      {
	// 3D primitive variables computation kernel    
	dim3 dimBlock(PRIM_VAR_MHD_BLOCK_DIMX_3D,
		      PRIM_VAR_MHD_BLOCK_DIMY_3D);
	dim3 dimGrid(blocksFor(isize, PRIM_VAR_MHD_BLOCK_DIMX_3D), 
		     blocksFor(jsize, PRIM_VAR_MHD_BLOCK_DIMY_3D));
	kernel_mhd_compute_primitive_variables_3D<<<dimGrid, 
	  dimBlock>>>(d_UOld.data(), 
		      d_Q.data(),
		      d_UOld.pitch(),
		      d_UOld.dimx(),
		      d_UOld.dimy(), 
		      d_UOld.dimz(),
		      dt);
	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_primitive_variables_3D error");
	
      }
      TIMER_STOP(timerPrimVar);

      TIMER_START(timerElecField);
      {
	// 3D Electric field computation kernel    
	dim3 dimBlock(ELEC_FIELD_BLOCK_DIMX_3D_V3,
		      ELEC_FIELD_BLOCK_DIMY_3D_V3);
	dim3 dimGrid(blocksFor(isize, ELEC_FIELD_BLOCK_INNER_DIMX_3D_V3), 
		     blocksFor(jsize, ELEC_FIELD_BLOCK_INNER_DIMY_3D_V3));
	kernel_mhd_compute_elec_field<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							     d_Q.data(),
							     d_elec.data(),
							     d_UOld.pitch(), 
							     d_UOld.dimx(), 
							     d_UOld.dimy(), 
							     d_UOld.dimz(),
							     dt);
	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_elec_field error");
	
	if (dumpDataForDebugEnabled) {
	  d_elec.copyToHost(h_debug2);
	  outputHdf5Debug(h_debug2, "elec_", nStep);
	}
	
      }
      TIMER_STOP(timerElecField);

      TIMER_START(timerMagSlopes);
      {
	// magnetic slopes computations
	dim3 dimBlock(MAG_SLOPES_BLOCK_DIMX_3D_V3,
		      MAG_SLOPES_BLOCK_DIMY_3D_V3);
	dim3 dimGrid(blocksFor(isize, MAG_SLOPES_BLOCK_INNER_DIMX_3D_V3), 
		     blocksFor(jsize, MAG_SLOPES_BLOCK_INNER_DIMY_3D_V3));
	kernel_mhd_compute_mag_slopes<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							     d_dA.data(),
							     d_dB.data(),
							     d_dC.data(),
							     d_UOld.pitch(), 
							     d_UOld.dimx(), 
							     d_UOld.dimy(), 
							     d_UOld.dimz() );
	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_mag_slopes error");
      }
      TIMER_STOP(timerMagSlopes);

      TIMER_START(timerTrace);
      // trace
      {
	dim3 dimBlock(TRACE_BLOCK_DIMX_3D_V4,
		      TRACE_BLOCK_DIMY_3D_V4);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_INNER_DIMX_3D_V4), 
		     blocksFor(jsize, TRACE_BLOCK_INNER_DIMY_3D_V4));
	kernel_mhd_compute_trace_v4<<<dimGrid, dimBlock>>>(d_UOld.data(),
							   d_Q.data(),
							   d_dA.data(),
							   d_dB.data(),
							   d_dC.data(),
							   d_elec.data(),
							   d_qm_x.data(),
							   d_qm_y.data(),
							   d_qm_z.data(),
							   d_qp_x.data(),
							   d_qp_y.data(),
							   d_qp_z.data(),
							   d_qEdge_RT.data(),
							   d_qEdge_RB.data(),
							   d_qEdge_LT.data(),
							   d_qEdge_LB.data(),
							   d_qEdge_RT2.data(),
							   d_qEdge_RB2.data(),
							   d_qEdge_LT2.data(),
							   d_qEdge_LB2.data(),
							   d_qEdge_RT3.data(),
							   d_qEdge_RB3.data(),
							   d_qEdge_LT3.data(),
							   d_qEdge_LB3.data(),
							   d_UOld.pitch(), 
							   d_UOld.dimx(), 
							   d_UOld.dimy(), 
							   d_UOld.dimz(),
							   dt / dx, 
							   dt / dy,
							   dt / dz,
							   dt);
	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_trace_v4 error");
	
	if (dumpDataForDebugEnabled) {
	  // DEBUG SHEAR

	  d_dA.copyToHost(h_debug2);
	  outputHdf5Debug(h_debug2, "dA_", nStep);
	  d_dB.copyToHost(h_debug2);
	  outputHdf5Debug(h_debug2, "dB_", nStep);
	  d_dC.copyToHost(h_debug2);
	  outputHdf5Debug(h_debug2, "dC_", nStep);

	  d_qm_x.copyToHost(h_debug);
	  outputHdf5Debug(  h_debug, "qm_x_", nStep);
	  d_qm_y.copyToHost(h_debug);
	  outputHdf5Debug(  h_debug, "qm_y_", nStep);
	  d_qm_z.copyToHost(h_debug);
	  outputHdf5Debug(  h_debug, "qm_z_", nStep);

	  d_qp_x.copyToHost(h_debug);
	  outputHdf5Debug(  h_debug, "qp_x_", nStep);
	  d_qp_y.copyToHost(h_debug);
	  outputHdf5Debug(  h_debug, "qp_y_", nStep);
	  d_qp_z.copyToHost(h_debug);
	  outputHdf5Debug(  h_debug, "qp_z_", nStep);

	  d_qEdge_RT.copyToHost(h_debug);
	  outputHdf5Debug(      h_debug, "qEdge_RT_", nStep);
	  d_qEdge_RB.copyToHost(h_debug);
	  outputHdf5Debug(      h_debug, "qEdge_RB_", nStep);
	  d_qEdge_LT.copyToHost(h_debug);
	  outputHdf5Debug(      h_debug, "qEdge_LT_", nStep);

	} // end debug
	
	// gravity predictor
	if (gravityEnabled) {
	  dim3 dimBlock(GRAV_PRED_BLOCK_DIMX_3D_V4,
			GRAV_PRED_BLOCK_DIMY_3D_V4);
	  dim3 dimGrid(blocksFor(isize, GRAV_PRED_BLOCK_DIMX_3D_V4), 
		       blocksFor(jsize, GRAV_PRED_BLOCK_DIMY_3D_V4));
	  kernel_mhd_compute_gravity_predictor_v4<<<dimGrid, 
	    dimBlock>>>(d_qm_x.data(),
			d_qm_y.data(),
			d_qm_z.data(),
			d_qp_x.data(),
			d_qp_y.data(),
			d_qp_z.data(),
			d_qEdge_RT.data(),
			d_qEdge_RB.data(),
			d_qEdge_LT.data(),
			d_qEdge_LB.data(),
			d_qEdge_RT2.data(),
			d_qEdge_RB2.data(),
			d_qEdge_LT2.data(),
			d_qEdge_LB2.data(),
			d_qEdge_RT3.data(),
			d_qEdge_RB3.data(),
			d_qEdge_LT3.data(),
			d_qEdge_LB3.data(),
			d_UOld.pitch(), 
			d_UOld.dimx(), 
			d_UOld.dimy(), 
			d_UOld.dimz(),
			dt);
	  checkCudaError("MHDRunGodunov kernel_mhd_compute_gravity_predictor_v4 error");
	  
	} // end gravity predictor

      } // end trace
      TIMER_STOP(timerTrace);	    


      TIMER_START(timerHydroShear);
      // update

      int hydroShearVersion = configMap.getInteger("implementation","hydroShearVersion",1);

      //d_UOld.copyTo(d_UNew);
      
      // the following call should be modified (d_shear_??? are only allocated when 
      // shearingBoxEnabled is true ...
      if (hydroShearVersion == 0) {

	dim3 dimBlock(UPDATE_BLOCK_DIMX_3D_V4,
		      UPDATE_BLOCK_DIMY_3D_V4);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_3D_V4), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_3D_V4));
	kernel_mhd_flux_update_hydro_v4_shear<<<dimGrid, dimBlock>>>(d_UOld.data(),
								     d_UNew.data(),
								     d_qm_x.data(),
								     d_qm_y.data(),
								     d_qm_z.data(),
								     d_qp_x.data(),
								     d_qp_y.data(),
								     d_qp_z.data(),
								     d_qEdge_RT.data(),
								     d_qEdge_RB.data(),
								     d_qEdge_LT.data(),
								     d_qEdge_LB.data(),
								     d_qEdge_RT2.data(),
								     d_qEdge_RB2.data(),
								     d_qEdge_LT2.data(),
								     d_qEdge_LB2.data(),
								     d_qEdge_RT3.data(),
								     d_qEdge_RB3.data(),
								     d_qEdge_LT3.data(),
								     d_qEdge_LB3.data(),
								     d_emf.data(),
								     d_shear_flux_xmin.data(),
								     d_shear_flux_xmax.data(),
								     d_UOld.pitch(), 
								     d_UOld.dimx(), 
								     d_UOld.dimy(), 
								     d_UOld.dimz(),
								     d_shear_flux_xmin.pitch(),
								     dt / dx, 
								     dt / dy,
								     dt / dz,
								     dt );
	checkCudaError("MHDRunGodunov :: kernel_mhd_flux_update_hydro_v4_shear error");
	
      } // end hydroShearVersion == 0
      
      if (hydroShearVersion != 0) {

	{
	  dim3 dimBlock(UPDATE_P1_BLOCK_DIMX_3D_V4,
			UPDATE_P1_BLOCK_DIMY_3D_V4);
	  dim3 dimGrid(blocksFor(isize, UPDATE_P1_BLOCK_INNER_DIMX_3D_V4), 
		       blocksFor(jsize, UPDATE_P1_BLOCK_INNER_DIMY_3D_V4));
	  kernel_mhd_flux_update_hydro_v4_shear_part1<<<dimGrid, dimBlock>>>(d_UOld.data(),
									     d_UNew.data(),
									     d_qm_x.data(),
									     d_qm_y.data(),
									     d_qm_z.data(),
									     d_qp_x.data(),
									     d_qp_y.data(),
									     d_qp_z.data(),
									     d_shear_flux_xmin.data(),
									     d_shear_flux_xmax.data(),
									     d_UOld.pitch(), 
									     d_UOld.dimx(), 
									     d_UOld.dimy(), 
									     d_UOld.dimz(),
									     d_shear_flux_xmin.pitch(),
									     dt / dx, 
									     dt / dy,
									     dt / dz,
									     dt );
	  checkCudaError("MHDRunGodunov :: kernel_mhd_flux_update_hydro_v4_shear_part1 error");
	}

	{
	  dim3 dimBlock(COMPUTE_EMF_BLOCK_DIMX_3D_SHEAR,
			COMPUTE_EMF_BLOCK_DIMY_3D_SHEAR);
	  dim3 dimGrid(blocksFor(isize, COMPUTE_EMF_BLOCK_DIMX_3D_SHEAR), 
		       blocksFor(jsize, COMPUTE_EMF_BLOCK_DIMY_3D_SHEAR));
	  kernel_mhd_compute_emf_shear<<<dimGrid, dimBlock>>>(d_qEdge_RT.data(),
							      d_qEdge_RB.data(),
							      d_qEdge_LT.data(),
							      d_qEdge_LB.data(),
							      d_qEdge_RT2.data(),
							      d_qEdge_RB2.data(),
							      d_qEdge_LT2.data(),
							      d_qEdge_LB2.data(),
							      d_qEdge_RT3.data(),
							      d_qEdge_RB3.data(),
							      d_qEdge_LT3.data(),
							      d_qEdge_LB3.data(),
							      d_emf.data(),
							      d_shear_flux_xmin.data(),
							      d_shear_flux_xmax.data(),
							      d_UOld.pitch(), 
							      d_UOld.dimx(), 
							      d_UOld.dimy(), 
							      d_UOld.dimz(),
							      d_shear_flux_xmin.pitch(),
							      dt / dx, 
							      dt / dy,
							      dt / dz,
							      dt );
	  checkCudaError("MHDRunGodunov :: kernel_mhd_compute_emf_shear error");
	}
      } // end hydroShearVersion != 0

      if (debugShear) {
	// DEBUG SHEAR
	HostArray<real_t> h_shear_debug; 
	h_shear_debug.allocate(make_uint4(isize,jsize,ksize,NVAR_MHD));
	d_UOld.copyToHost(h_shear_debug);
	outputHdf5Debug(h_shear_debug, "UOld_hydroShear_", nStep);
	d_UNew.copyToHost(h_shear_debug);
	outputHdf5Debug(h_shear_debug, "UNew_hydroShear_", nStep);
      }
      
      if (debugShear) {
	// DEBUG SHEAR
	HostArray<real_t> h_shear_debug; 
	h_shear_debug.allocate(make_uint3(jsize,ksize,NUM_COMPONENT_REMAP));
	d_shear_flux_xmin.copyToHost(h_shear_debug);
	outputHdf5Debug(h_shear_debug, "shear_flux_xmin_", nStep);
	d_shear_flux_xmax.copyToHost(h_shear_debug);
	outputHdf5Debug(h_shear_debug, "shear_flux_xmax_", nStep);
      }
	
      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt);
      }
      TIMER_STOP(timerHydroShear);
      

      TIMER_START(timerRemapping);
      // flux and emf remapping
      {
	dim3 dimBlock(16, 16);
	dim3 dimGrid(blocksFor(jsize, dimBlock.x), 
		     blocksFor(ksize, dimBlock.y));
	
	
	kernel_remapping_mhd_3d<<<dimGrid, dimBlock>>>(d_shear_flux_xmin.data(),
						       d_shear_flux_xmax.data(),
						       d_shear_flux_xmin_remap.data(),
						       d_shear_flux_xmax_remap.data(),
						       d_emf.data(),
						       d_UOld.pitch(),
						       d_UOld.dimx(),
						       d_UOld.dimy(),
						       d_UOld.dimz(),
						       d_shear_flux_xmax.pitch(),
						       totalTime,
						       dt);
	
	if (debugShear) {
	  // DEBUG SHEAR
	  HostArray<real_t> h_shear_debug; 
	  h_shear_debug.allocate(make_uint3(jsize,ksize,NUM_COMPONENT_REMAP));
	  d_shear_flux_xmin_remap.copyToHost(h_shear_debug);
	  outputHdf5Debug(h_shear_debug, "shear_flux_xmin_remapped_", nStep);
	  d_shear_flux_xmax_remap.copyToHost(h_shear_debug);
	  outputHdf5Debug(h_shear_debug, "shear_flux_xmax_remapped_", nStep);
	}
	
      } // end flux and emf remapping
      TIMER_STOP(timerRemapping);

      
      TIMER_START(timerShearBorder);
      // update shear borders with density and emfY
      {
	dim3 dimBlock(16, 16);
	dim3 dimGrid(blocksFor(jsize, dimBlock.x), 
		     blocksFor(ksize, dimBlock.y));
	
	kernel_update_shear_borders_3d<<<dimGrid, dimBlock>>>(d_UNew.data(),
							      d_shear_flux_xmin_remap.data(),
							      d_shear_flux_xmax_remap.data(),
							      d_UNew.pitch(), 
							      d_UNew.dimx(), 
							      d_UNew.dimy(), 
							      d_UNew.dimz(),
							      d_shear_flux_xmin_remap.pitch(),
							      totalTime,
							      dt);
	
	if (debugShear) {
	  // DEBUG SHEAR
	  HostArray<real_t> h_shear_debug; 
	  h_shear_debug.allocate(make_uint4(isize,jsize,ksize,NVAR_MHD));
	  d_UNew.copyToHost(h_shear_debug);
	  outputHdf5Debug(h_shear_debug, "gpu_shear_UNew_update2_", nStep);
	}
	
      } // end shear borders with density 
      TIMER_STOP(timerShearBorder);

      TIMER_START(timerCtUpdate);
      // update magnetic field
      {
	dim3 dimBlock(UPDATE_CT_BLOCK_DIMX_3D_V4,
		      UPDATE_CT_BLOCK_DIMY_3D_V4);
	dim3 dimGrid(blocksFor(isize, UPDATE_CT_BLOCK_DIMX_3D_V4), 
		     blocksFor(jsize, UPDATE_CT_BLOCK_DIMY_3D_V4));
	kernel_mhd_flux_update_ct_v4<<<dimGrid, dimBlock>>>(d_UOld.data(),
							    d_UNew.data(),
							    d_emf.data(),
							    d_UOld.pitch(), 
							    d_UOld.dimx(), 
							    d_UOld.dimy(), 
							    d_UOld.dimz(),
							    dt / dx, 
							    dt / dy,
							    dt / dz,
							    dt );
	checkCudaError("MHDRunGodunov kernel_mhd_flux_update_ct_v4 error");
      } // update magnetic field

      if (debugShear) {
	// DEBUG SHEAR
	HostArray<real_t> h_shear_debug; 
	h_shear_debug.allocate(make_uint4(isize,jsize,ksize,3));
	d_emf.copyToHost(h_shear_debug);
	outputHdf5Debug(h_shear_debug, "emf_", nStep);
      }

      if (debugShear) {
	// DEBUG SHEAR
	HostArray<real_t> h_shear_debug; 
	h_shear_debug.allocate(make_uint4(isize,jsize,ksize,NVAR_MHD));
	d_UNew.copyToHost(h_shear_debug);
	outputHdf5Debug(h_shear_debug, "UNew_after_ct_", nStep);
      }
      TIMER_STOP(timerCtUpdate);

      TIMER_START(timerDissipative);
      // update borders
      real_t &nu = _gParams.nu;
      real_t &eta = _gParams.eta;
      if (nu>0 or eta>0) {
	if (shearingBoxEnabled) {
	  
	  make_all_boundaries_shear(d_UNew, dt, nStep);

	} else {

	  make_all_boundaries(d_UNew);

	} // end shearingBoxEnabled
      } // end nu>0
      
      if (eta>0) {
	// update magnetic field with resistivity emf
	compute_resistivity_emf_3d(d_UNew, d_emf);
	compute_ct_update_3d      (d_UNew, d_emf, dt);
	
	real_t &cIso = _gParams.cIso;
	if (cIso<=0) { // non-isothermal simulations
	  // compute energy flux
	  compute_resistivity_energy_flux_3d(d_UNew, d_qm_x, d_qm_y, d_qm_z, dt);
	  compute_hydro_update_energy       (d_UNew, d_qm_x, d_qm_y, d_qm_z);
	}
      }
      
      // compute viscosity
      if (nu>0) {
	DeviceArray<real_t> &d_flux_x = d_qm_x;
	DeviceArray<real_t> &d_flux_y = d_qm_y;
	DeviceArray<real_t> &d_flux_z = d_qm_z;

	compute_viscosity_flux(d_UNew, d_flux_x, d_flux_y, d_flux_z, dt);
	compute_hydro_update  (d_UNew, d_flux_x, d_flux_y, d_flux_z );
      } // end compute viscosity force / update  
      TIMER_STOP(timerDissipative);
      
    } // end THREE_D
    TIMER_STOP(timerGodunov);


    /*
     * update boundaries in UNew
     */
    TIMER_START(timerBoundaries);
    if (shearingBoxEnabled and dimType == THREE_D) {

      make_all_boundaries_shear(d_UNew, dt, nStep);

    } else {

      make_all_boundaries(d_UNew);
    
    } // end if (shearingBoxEnabled and dimType == THREE_D)
    TIMER_STOP(timerBoundaries);


  } // MHDRunGodunov::godunov_unsplit_rotating_gpu

#else
  // =======================================================
  // =======================================================
  // Omega0 is assumed to be strictly positive
  void MHDRunGodunov::godunov_unsplit_rotating_cpu(HostArray<real_t>& h_UOld, 
						   HostArray<real_t>& h_UNew, 
						   real_t dt, int nStep)
  {
    
    /*
     * shearing box correction on momentum parameters
     */
    real_t lambda=0, ratio=1, alpha1=1, alpha2=0;
    
    // geometry
    int &geometry = ::gParams.geometry;
    
    // Omega0
    real_t &Omega0 = ::gParams.Omega0;
    
    if (geometry == GEO_CARTESIAN) {
      lambda = Omega0*dt;
      lambda = ONE_FOURTH_F * lambda * lambda;
      ratio  = (ONE_F-lambda)/(ONE_F+lambda);
      alpha1 =          ONE_F/(ONE_F+lambda);
      alpha2 =      Omega0*dt/(ONE_F+lambda);
    }

    // copy h_UOld into h_UNew
    for (unsigned int indexGlob=0; indexGlob<h_UOld.size(); indexGlob++) {
      h_UNew(indexGlob) = h_UOld(indexGlob);
    }

    // scaling factor to apply to flux when updating hydro state h_U
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;

    // conservative variable domain array
    real_t *U = h_UOld.data();

    // primitive variable domain array
    real_t *Q = h_Q.data();
    
    // section / domain size
    int arraySize = h_Q.section();

    memset(h_emf.data(),0,h_emf.sizeBytes());

    /*
     * main computation loop to update h_U : 2D loop over simulation domain location
     */
    TIMER_START(timerGodunov);

    // convert conservative to primitive variables (and source term predictor)
    // put results in h_Q object
    convertToPrimitives(U, dt);
  

    /*
     * TWO_D
     */
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
	
	  real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
	
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
	  real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
	
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
	  
	  /*
	   * shear correction (assume cartesian here, other coordinate
	   * systems -> TODO)
	   */
	  {
	    real_t shear_x = -1.5 * Omega0 * (xPos + xPos - dx);
	    flux_x[IC] += shear_x * ( qleft[IA] + qright[IA] ) / 2;
	  }

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
	   * shear correction (assume cartesian here, other coordinate
	   * systems -> TODO)
	   */
	  {
	    real_t shear_y = -1.5 * Omega0 * xPos;
	    flux_y[IC] += shear_y * ( qleft[IA] + qright[IA] ) / 2;
	  }
	
	  /*
	   * update mhd array with hydro fluxes
	   */
	  if (geometry == GEO_CARTESIAN) {

	    if (i<isize-ghostWidth and 
		j<jsize-ghostWidth) {
	      real_t dsx =   TWO_F * Omega0 * dt * h_UNew(i,j,IV)/(ONE_F + lambda);
	      real_t dsy = -HALF_F * Omega0 * dt * h_UNew(i,j,IU)/(ONE_F + lambda);
	      h_UNew(i,j,IU) = h_UNew(i,j,IU)*ratio + dsx;
	      h_UNew(i,j,IV) = h_UNew(i,j,IV)*ratio + dsy;
	    }

	    if (shearingBoxEnabled and  i==(nx+ghostWidth) ) {
	      // do not perform update at border XMAX (we need to store density flux and perform a remapping)
	      h_shear_flux_xmax(j,1,I_DENS)       = flux_x[ID]*dtdx;
	    } else {
	      // perform a normal update
	      h_UNew(i-1,j  ,ID) -= flux_x[ID]*dtdx;
	    }
	    h_UNew(i-1,j  ,IP) -= flux_x[IP]*dtdx;
	    h_UNew(i-1,j  ,IU) -= ( alpha1*flux_x[IU] +      alpha2*flux_x[IV] )*dtdx;
	    h_UNew(i-1,j  ,IV) -= ( alpha1*flux_x[IV] - 0.25*alpha2*flux_x[IU] )*dtdx;
	    h_UNew(i-1,j  ,IW) -= flux_x[IW]*dtdx;
	    h_UNew(i-1,j  ,IC) -= flux_x[IC]*dtdx;

	    if (shearingBoxEnabled and  i==ghostWidth ) { 
	      // do not perform update at border XMIN (we need to store density flux and perform a remapping)
	      h_shear_flux_xmin(j,1,I_DENS)       = flux_x[ID]*dtdx;
	    } else {
	      // perform a normal update
	      h_UNew(i  ,j  ,ID) += flux_x[ID]*dtdx;
	    }
	    h_UNew(i  ,j  ,IP) += flux_x[IP]*dtdx;
	    h_UNew(i  ,j  ,IU) += ( alpha1*flux_x[IU] +      alpha2*flux_x[IV] )*dtdx;
	    h_UNew(i  ,j  ,IV) += ( alpha1*flux_x[IV] - 0.25*alpha2*flux_x[IU] )*dtdx;
	    h_UNew(i  ,j  ,IW) += flux_x[IW]*dtdx;
	    h_UNew(i  ,j  ,IC) += flux_x[IC]*dtdx;
	  
	    // watchout IU and IV swapped
	    h_UNew(i  ,j-1,ID) -= flux_y[ID]*dtdy;
	    h_UNew(i  ,j-1,IP) -= flux_y[IP]*dtdy;
	    h_UNew(i  ,j-1,IU) -= ( alpha1*flux_y[IV] +      alpha2*flux_y[IU] )*dtdy;
	    h_UNew(i  ,j-1,IV) -= ( alpha1*flux_y[IU] - 0.25*alpha2*flux_y[IV] )*dtdy;
	    h_UNew(i  ,j-1,IW) -= flux_y[IW]*dtdy;
	    h_UNew(i  ,j-1,IC) -= flux_y[IC]*dtdy;
	  
	    h_UNew(i  ,j  ,ID) += flux_y[ID]*dtdy;
	    h_UNew(i  ,j  ,IP) += flux_y[IP]*dtdy;
	    h_UNew(i  ,j  ,IU) += ( alpha1*flux_y[IV] +      alpha2*flux_y[IU] )*dtdy;
	    h_UNew(i  ,j  ,IV) += ( alpha1*flux_y[IU] - 0.25*alpha2*flux_y[IV] )*dtdy;
	    h_UNew(i  ,j  ,IW) += flux_y[IW]*dtdy;
	    h_UNew(i  ,j  ,IC) += flux_y[IC]*dtdy;
	  } else {
	    /* TODO */
	  } // end if GEO_CARTESIAN
	
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
	  real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ,xPos);
	  h_emf(i,j,I_EMFZ) = emfZ;
	
	} // end for i
      } // end for j

      if (shearingBoxEnabled) {
	// we need to perform flux remapping and update borders

	std::cout << "WARNING :\n";
	std::cout << "2D shearing box border conditions are not fully implemented\n";
	std::cout << "and is not likely to be in the near future....\n";
	
	// the following code is adapted from Dumses/bval_shear.f90, 
	// subroutines bval_shear_flux and bval_shear_emf
	real_t deltay,epsi,eps;
	int jplus,jremap,jremapp1;

	deltay = 1.5 * _gParams.Omega0 * (_gParams.dx * _gParams.nx) * (totalTime+dt/2);
	deltay = FMOD(deltay, (_gParams.dy * _gParams.ny) );
	jplus  = (int) (deltay/dy);
	epsi   = FMOD(deltay,  dy);

	/*
	 * perform flux/emf remapping
	 */
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {

	  /*
	   * inner (i.e. xMin) boundary 
	   */
	  jremap   = j      - jplus - 1;
	  jremapp1 = jremap + 1;
	  eps      = 1.0-epsi/dy;

	  if (jremap  < ghostWidth) jremap   += ny;
	  if (jremapp1< ghostWidth) jremapp1 += ny;

	  h_shear_flux_xmin_remap(j,1,I_DENS) =
	    h_shear_flux_xmin(j,1,I_DENS) +
	    (1.0-eps) * h_shear_flux_xmax(jremap  ,1, I_DENS) + 
	    eps       * h_shear_flux_xmax(jremapp1,1, I_DENS);
	  h_shear_flux_xmin_remap(j,1,I_DENS) *= HALF_F;

	  /*
	   * outer (i.e. xMax) boundary
	   */
	  jremap   = j      + jplus;
	  jremapp1 = jremap + 1;
	  eps      = epsi/dy;

	  if (jremap   > ny+ghostWidth-1) jremap   = jremap   - ny;
	  if (jremapp1 > ny+ghostWidth-1) jremapp1 = jremapp1 - ny;

	  h_shear_flux_xmax_remap(j,1,I_DENS) =
	    h_shear_flux_xmax(j,1,I_DENS) +
	    (1.0-eps) * h_shear_flux_xmin(jremap  ,1, I_DENS) + 
	    eps       * h_shear_flux_xmin(jremapp1,1, I_DENS);
	  h_shear_flux_xmax_remap(j,1,I_DENS) *= HALF_F;

	} // end for j / end flux/emf remapping

	/*
	 * finally update xMin/xMax border with mapped flux/emf
	 */
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {

	  h_UNew(nx+ghostWidth-1,j  ,ID) -= h_shear_flux_xmax_remap(j,1,I_DENS);
	  h_UNew(ghostWidth     ,j  ,ID) += h_shear_flux_xmin_remap(j,1,I_DENS);

	} // end for j / end update xMin/xMax with remapped values
	

      } // end shearing box
      
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
      
      TIMER_START(timerDissipative);
      // update borders
      real_t &nu  = _gParams.nu;
      real_t &eta = _gParams.eta;
      if (nu>0 or eta>0) {
	make_all_boundaries(h_UNew);
      }
      
      if (eta>0) {
	// update magnetic field with resistivity emf
	compute_resistivity_emf_2d(h_UNew, h_emf);
	compute_ct_update_2d      (h_UNew, h_emf, dt);
	
	real_t &cIso = _gParams.cIso;
	if (cIso<=0) { // non-isothermal simulations
	  // compute energy flux
	  compute_resistivity_energy_flux_2d(h_UNew, h_qm_x, h_qm_y, dt);
	  compute_hydro_update_energy       (h_UNew, h_qm_x, h_qm_y);
	}
      }
      
      // compute viscosity forces
      if (nu>0) {
	// re-use h_qm_x and h_qm_y
	HostArray<real_t> &flux_x = h_qm_x;
	HostArray<real_t> &flux_y = h_qm_y;
	
	compute_viscosity_flux(h_UNew, flux_x, flux_y, dt);
	compute_hydro_update  (h_UNew, flux_x, flux_y);
	
      } // end compute viscosity force / update
      TIMER_STOP(timerDissipative);
      
    } else { // THREE_D
    
      /*
       * THREE_D
       */

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
	    real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
	  
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
	  
	    /* if cartesian and not fargo*/
	    {
	      real_t shear = -1.5 * Omega0 * xPos;
	      h_elec(i,j,k,IX) += shear*C;
	    }

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
	    /* if cartesian and not fargo */
	    {
	      real_t shear = -1.5 * Omega0 * (xPos - dx/2);
	      h_elec(i,j,k,IZ) -= shear*A;
	    }

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

      if (debugShear) {
	outputHdf5Debug(h_qEdge_RT, "cpu_qEdge_RT_", nStep);
      }

      if (shearingBoxEnabled and debugShear) {
      	outputHdf5Debug(h_UOld           ,"UOld_hydroShear_before_",nStep);
      	outputHdf5Debug(h_UNew           ,"UNew_hydroShear_before_",nStep);
      }
      
      TIMER_START(timerHydroShear);
      // Finally compute fluxes from rieman solvers, and update
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
	    real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;

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
	     * shear correction (assume cartesian here, other coordinate
	     * systems -> TODO)
	     */
	    /* if cartesian and not fargo */
	    { 
	      // shear correction
	      real_t shear_y = -1.5 * Omega0 * xPos;
	      real_t eMag, eKin, eTot;
	      real_t bn_mean = HALF_F * (qleft[IA] + qright[IA]);
	      real_t &gamma  = ::gParams.gamma0;
	    
	      if (shear_y > 0) {
		eMag = HALF_F * (qleft[IA]*qleft[IA] + 
				 qleft[IB]*qleft[IB] + 
				 qleft[IC]*qleft[IC]);

		eKin = HALF_F * (qleft[IU]*qleft[IU] + 
				 qleft[IV]*qleft[IV] + 
				 qleft[IW]*qleft[IW]);

		eTot = eKin + eMag + qleft[IP]/(gamma - ONE_F);
		flux_y[ID] = flux_y[ID] + shear_y * qleft[ID];
		flux_y[IP] = flux_y[IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
		flux_y[IU] = flux_y[IU] + shear_y * qleft[ID]*qleft[IU];
		flux_y[IV] = flux_y[IV] + shear_y * qleft[ID]*qleft[IV];
		flux_y[IW] = flux_y[IW] + shear_y * qleft[ID]*qleft[IW];
	      } else {
		eMag = HALF_F * (qright[IA]*qright[IA] + 
				 qright[IB]*qright[IB] + 
				 qright[IC]*qright[IC]);

		eKin = HALF_F * (qright[IU]*qright[IU] + 
				 qright[IV]*qright[IV] + 
				 qright[IW]*qright[IW]);

		eTot = eKin + eMag + qright[IP]/(gamma - ONE_F);
		flux_y[ID] = flux_y[ID] + shear_y * qright[ID];
		flux_y[IP] = flux_y[IP] + shear_y * (eTot + eMag - bn_mean*bn_mean);
		flux_y[IU] = flux_y[IU] + shear_y * qright[ID]*qright[IU];
		flux_y[IV] = flux_y[IV] + shear_y * qright[ID]*qright[IV];
		flux_y[IW] = flux_y[IW] + shear_y * qright[ID]*qright[IW];
	      }
	    } // end shear correction

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
	     * when debugging, should be removed later as ghostZones
	     * are anyway erased in make_boudaries routine.
	     */

	    /* only valid for CARTESIAN geometry !!! */

	    if (i<isize-ghostWidth and 
		j<jsize-ghostWidth and 
		k<ksize-ghostWidth) {
	      real_t dsx =   TWO_F * Omega0 * dt * h_UNew(i,j,k,IV)/(ONE_F + lambda);
	      real_t dsy = -HALF_F * Omega0 * dt * h_UNew(i,j,k,IU)/(ONE_F + lambda);
	      h_UNew(i,j,k,IU) = h_UNew(i,j,k,IU)*ratio + dsx;
	      h_UNew(i,j,k,IV) = h_UNew(i,j,k,IV)*ratio + dsy;
	    }

	    /* in shearing box simulation, there is a special treatment for flux_x[ID] */

	    /* 
	     * update with flux_x
	     */
	    if ( i > ghostWidth       and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {

	      if ( shearingBoxEnabled and  i==(nx+ghostWidth) ) {
		// do not perform update at border XMAX (we need to store density flux and 
		// perform a remapping)
		h_shear_flux_xmax(j,k,I_DENS)       = flux_x[ID]*dtdx;
	      } else {
		// perform a normal update
		h_UNew(i-1,j  ,k  ,ID) -= flux_x[ID]*dtdx;
	      }

	      h_UNew(i-1,j  ,k  ,IP) -= flux_x[IP]*dtdx;
	      h_UNew(i-1,j  ,k  ,IU) -= (alpha1*flux_x[IU]+     alpha2*flux_x[IV])*dtdx;
	      h_UNew(i-1,j  ,k  ,IV) -= (alpha1*flux_x[IV]-0.25*alpha2*flux_x[IU])*dtdx;
	      h_UNew(i-1,j  ,k  ,IW) -= flux_x[IW]*dtdx;

	    }
	      
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {

	      if (shearingBoxEnabled and  i==ghostWidth ) { 
		// do not perform update at border XMIN (we need to store density flux and
		// perform a remapping)
		h_shear_flux_xmin(j,k,I_DENS)       = flux_x[ID]*dtdx;
	      } else {
		// perform a normal update
		h_UNew(i  ,j  ,k  ,ID) += flux_x[ID]*dtdx;
	      }
	      h_UNew(i  ,j  ,k  ,IP) += flux_x[IP]*dtdx;
	      h_UNew(i  ,j  ,k  ,IU) += (alpha1*flux_x[IU]+     alpha2*flux_x[IV])*dtdx;
	      h_UNew(i  ,j  ,k  ,IV) += (alpha1*flux_x[IV]-0.25*alpha2*flux_x[IU])*dtdx;
	      h_UNew(i  ,j  ,k  ,IW) += flux_x[IW]*dtdx;

	    }

	    /*
	     * update with flux_y
	     */
	    if ( i < isize-ghostWidth and
		 j > ghostWidth       and
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j-1,k  ,ID) -= flux_y[ID]*dtdy;
	      h_UNew(i  ,j-1,k  ,IP) -= flux_y[IP]*dtdy;
	      h_UNew(i  ,j-1,k  ,IU) -= (alpha1*flux_y[IV]+     alpha2*flux_y[IU])*dtdy; // IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IV) -= (alpha1*flux_y[IU]-0.25*alpha2*flux_y[IV])*dtdy; // IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IW) -= flux_y[IW]*dtdy;
	    }
	      
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_y[ID]*dtdy;
	      h_UNew(i  ,j  ,k  ,IP) += flux_y[IP]*dtdy;
	      h_UNew(i  ,j  ,k  ,IU) += (alpha1*flux_y[IV]+     alpha2*flux_y[IU])*dtdy; // IU and IV swapped
	      h_UNew(i  ,j  ,k  ,IV) += (alpha1*flux_y[IU]-0.25*alpha2*flux_y[IV])*dtdy; // IU and IV swapped
	      h_UNew(i  ,j  ,k  ,IW) += flux_y[IW]*dtdy;
	    }
	    
	    /*
	     * update with flux_z
	     */
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and
		 k > ghostWidth ) {
	      h_UNew(i  ,j  ,k-1,ID) -= flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,k-1,IP) -= flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,k-1,IU) -= (alpha1*flux_z[IW]+     alpha2*flux_z[IV])*dtdz; // IU and IW swapped
	      h_UNew(i  ,j  ,k-1,IV) -= (alpha1*flux_z[IV]-0.25*alpha2*flux_z[IW])*dtdz;
	      h_UNew(i  ,j  ,k-1,IW) -= flux_z[IU]*dtdz; // IU and IW swapped
	    }
	    
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,k  ,IP) += flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,k  ,IU) += (alpha1*flux_z[IW]+     
					 alpha2*flux_z[IV])*dtdz; // IU and IW swapped
	      h_UNew(i  ,j  ,k  ,IV) += (alpha1*flux_z[IV]-0.25*
					 alpha2*flux_z[IW])*dtdz;
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
	    real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ,xPos);
	    if (k<ksize-ghostWidth) {
	      h_emf(i,j,k,I_EMFZ) = emfZ;
	    }

	    // actually compute emfY (take care that RB and LT are
	    // swapped !!!)
	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      qEdge_emfY[IRT][iVar] = h_qEdge_RT2(i-1,j  ,k-1,iVar);
	      qEdge_emfY[IRB][iVar] = h_qEdge_LT2(i  ,j  ,k-1,iVar); 
	      qEdge_emfY[ILT][iVar] = h_qEdge_RB2(i-1,j  ,k  ,iVar);
	      qEdge_emfY[ILB][iVar] = h_qEdge_LB2(i  ,j  ,k  ,iVar); 
	    }
	    real_t emfY = compute_emf<EMFY>(qEdge_emfY,xPos);
	    if (j<jsize-ghostWidth) {
	      h_emf(i,j,k,I_EMFY) = emfY;
	      
	      if (shearingBoxEnabled) {
		if (i == ghostWidth) {
		  h_shear_flux_xmin(j,k,I_EMF_Y) = emfY;
		}
		
		if (i == (nx+ghostWidth)) {
		  h_shear_flux_xmax(j,k,I_EMF_Y) = emfY;
		}
	      }
	    }

	    // actually compute emfX
	    for (int iVar=0; iVar<NVAR_MHD; iVar++) {
	      qEdge_emfX[IRT][iVar] = h_qEdge_RT(i  ,j-1,k-1,iVar);
	      qEdge_emfX[IRB][iVar] = h_qEdge_RB(i  ,j-1,k  ,iVar);
	      qEdge_emfX[ILT][iVar] = h_qEdge_LT(i  ,j  ,k-1,iVar);
	      qEdge_emfX[ILB][iVar] = h_qEdge_LB(i  ,j  ,k  ,iVar);
	    }
	    real_t emfX = compute_emf<EMFX>(qEdge_emfX,xPos);
	    if (i<isize-ghostWidth) {
	      h_emf(i,j,k,I_EMFX) = emfX;
	    }
	      
	    // // now update h_UNew with emfZ
	    // // (Constrained transport for face-centered B-field)
	    // if ( i < isize-ghostWidth+1 and 
	    // 	 j < jsize-ghostWidth and 
	    // 	 k < ksize-ghostWidth ) 
	    //   h_UNew(i  ,j  ,k  ,IA) -= emfZ*dtdy;
	      
	    // if ( i < isize-ghostWidth+1 and
	    // 	 j > ghostWidth       and
	    // 	 k < ksize-ghostWidth )
	    //   h_UNew(i  ,j-1,k  ,IA) += emfZ*dtdy;
	      
	    // if ( i < isize-ghostWidth and 
	    // 	 j < jsize-ghostWidth and 
	    // 	 k < ksize-ghostWidth ) 
	    //   h_UNew(i  ,j  ,k  ,IB) += emfZ*dtdx;  
	      
	    // if ( i > ghostWidth       and 
	    // 	 j < jsize-ghostWidth and 
	    // 	 k < ksize-ghostWidth ) 
	    //   h_UNew(i-1,j  ,k  ,IB) -= emfZ*dtdx;

	    // // now update h_UNew with emfY, emfX

	    // // take care that emfY has a special treatment when shearing box is enabled
	    // // Magnetic field update is delayed...

	    // if (k<ksize-ghostWidth) {
	    //   if (shearingBoxEnabled and i==ghostWidth) {
	    // 	h_shear_flux_xmin(j,k,I_EMF_Y)       = emfY;
	    // 	h_shear_flux_xmin(j,k,I_EMF_Y_REMAP) = emfY;
	    //   }
	    //   if (shearingBoxEnabled and i==(nx+ghostWidth)) {
	    // 	// do not perform update at border XMAX (we need to store emfY
	    // 	// and perform a remapping)
	    // 	h_shear_flux_xmax(j,k,I_EMF_Y)       = emfY;
	    // 	h_shear_flux_xmax(j,k,I_EMF_Y_REMAP) = emfY;
	    // 	h_shear_flux_xmax(j,k,I_EMF_Z)       = emfZ;
	    //   } 
	    // }

	    // if ( i < isize-ghostWidth and 
	    // 	 j < jsize-ghostWidth and 
	    // 	 k < ksize-ghostWidth ) {
	    //   if (shearingBoxEnabled and i==ghostWidth) {
	    // 	// do not perform update at border XMIN (we need to store emfY
	    // 	// and perform a remapping
	    //   } else {
	    // 	// perform normal update
	    // 	h_UNew(i  ,j  ,k  ,IA) += emfY*dtdz;	      
	    // 	h_UNew(i  ,j  ,k  ,IC) -= emfY*dtdx;
	    //   }
	    //   h_UNew(i  ,j  ,k  ,IB) -= emfX*dtdz;
	    //   h_UNew(i  ,j  ,k  ,IC) += emfX*dtdy;
	    // }

	    // if ( i < isize-ghostWidth and 
	    // 	 j < jsize-ghostWidth and
	    // 	     k > ghostWidth ) {
	    //   if (shearingBoxEnabled and i==ghostWidth) {
	    // 	// do not perform update at border XMIN (emfY already stored)
	    //   } else {
	    // 	// perform normal update
	    // 	h_UNew(i  ,j  ,k-1,IA) -= emfY*dtdz;
	    //   }
	    //   h_UNew(i  ,j  ,k-1,IB) += emfX*dtdz;
	    // }

	    // if ( i > ghostWidth       and 
	    // 	 j < jsize-ghostWidth and 
	    // 	 k < ksize-ghostWidth ) {
	    //   if (shearingBoxEnabled and i==(nx+ghostWidth)) {
	    // 	// do not perform update at border XMAX (we need to store emfY
	    // 	// and perform a remapping
	    //   } else {
	    // 	// perform normal update
	    // 	h_UNew(i-1,j  ,k  ,IC) += emfY*dtdx;
	    //   }
	    // }
	      
	    // if ( i < isize-ghostWidth and
	    // 	 j > ghostWidth       and
	    // 	 k < ksize-ghostWidth ) 
	    //   h_UNew(i  ,j-1,k  ,IC) -= emfX*dtdy;
	      
	  } // end for i
	} // end for j
      } // end for k

      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(h_UNew, h_UOld, dt);
      }

      TIMER_STOP(timerHydroShear);

      if (shearingBoxEnabled and debugShear) {
      	outputHdf5Debug(h_shear_flux_xmin,"shear_flux_xmin_",nStep);
      	outputHdf5Debug(h_shear_flux_xmax,"shear_flux_xmax_",nStep);
      	outputHdf5Debug(h_UOld           ,"UOld_hydroShear_",nStep);
      	outputHdf5Debug(h_UNew           ,"UNew_hydroShear_",nStep);
      }

      if (shearingBoxEnabled) {

      	// we need to perform flux remapping and update borders
	TIMER_START(timerRemapping);

	// the following code is adapted from Dumses/bval_shear.f90, 
	// subroutines bval_shear_flux and bval_shear_emf
	real_t deltay,epsi,eps;
	int jplus,jremap,jremapp1;

	deltay = 1.5 * _gParams.Omega0 * (_gParams.dx * _gParams.nx) * (totalTime+dt/2);
	deltay = FMOD(deltay, (_gParams.dy * _gParams.ny) );
	jplus  = (int) (deltay/dy);
	epsi   = FMOD(deltay,  dy);

	/*
	 * perform flux/emf remapping
	 */
	for (int k=0; k<ksize; k++) {
	  for (int j=0; j<jsize; j++) {

	    /*
	     * inner (i.e. xMin) boundary - flux and emf
	     */
	    jremap   = j      - jplus - 1;
	    jremapp1 = jremap + 1;
	    eps      = 1.0-epsi/dy;
	    
	    if (jremap  < ghostWidth) jremap   += ny;
	    if (jremapp1< ghostWidth) jremapp1 += ny;
	    
	    // flux
	    if (j>=ghostWidth and j<jsize-ghostWidth+1 and 
		k>=ghostWidth and k<ksize-ghostWidth+1) {
	      h_shear_flux_xmin_remap(j,k,I_DENS) =
		h_shear_flux_xmin(j,k,I_DENS) +
		(1.0-eps) * h_shear_flux_xmax(jremap  ,k, I_DENS) + 
		eps       * h_shear_flux_xmax(jremapp1,k, I_DENS);
	      h_shear_flux_xmin_remap(j,k,I_DENS) *= HALF_F;
	    }
	    
	    // emf
	    h_emf(ghostWidth,j,k,I_EMFY) += 
	      (1.0-eps) * h_shear_flux_xmax(jremap  ,k,I_EMF_Y) + 
	      eps       * h_shear_flux_xmax(jremapp1,k,I_EMF_Y);
	    h_emf(ghostWidth,j,k,I_EMFY) *= HALF_F;
	    
	    /*
	     * outer (i.e. xMax) boundary - flux and emf
	     */
	    jremap   = j      + jplus;
	    jremapp1 = jremap + 1;
	    eps      = epsi/dy;
	    
	    if (jremap   > ny+ghostWidth-1) jremap   -= ny;
	    if (jremapp1 > ny+ghostWidth-1) jremapp1 -= ny;
	    
	    // flux
	    if (j>=ghostWidth and j<jsize-ghostWidth+1 and
		k>=ghostWidth and k<ksize-ghostWidth+1) {
	      h_shear_flux_xmax_remap(j,k,I_DENS) =
		h_shear_flux_xmax(j,k,I_DENS) +
		(1.0-eps) * h_shear_flux_xmin(jremap  ,k, I_DENS) + 
		eps       * h_shear_flux_xmin(jremapp1,k, I_DENS);
	      h_shear_flux_xmax_remap(j,k,I_DENS) *= HALF_F;
	    }

	    // emf
	    h_emf(nx+ghostWidth,j,k,I_EMFY) +=
	      (1.0-eps) * h_shear_flux_xmin(jremap  , k, I_EMF_Y) + 
	      eps       * h_shear_flux_xmin(jremapp1, k, I_EMF_Y);
	    h_emf(nx+ghostWidth,j,k,I_EMFY) *= HALF_F;
      
	  } // end for j
	} // end for k // end flux/emf remapping
	TIMER_STOP(timerRemapping);

	if (debugShear) {
	  outputHdf5Debug(h_shear_flux_xmin_remap,"shear_flux_xmin_remapped_",nStep);
	  outputHdf5Debug(h_shear_flux_xmax_remap,"shear_flux_xmax_remapped_",nStep);
	}

	/*
	 * finally update xMin/xMax border with remapped flux
	 */
	TIMER_START(timerShearBorder);
	for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
	  for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	    
	    // update density
	    h_UNew(ghostWidth     ,j, k, ID) += h_shear_flux_xmin_remap(j,k,I_DENS);
	    h_UNew(nx+ghostWidth-1,j, k, ID) -= h_shear_flux_xmax_remap(j,k,I_DENS);

	    h_UNew(ghostWidth     ,j, k, ID)  = FMAX(h_UNew(ghostWidth     ,j, k, ID), _gParams.smallr);
	    h_UNew(nx+ghostWidth-1,j, k, ID)  = FMAX(h_UNew(nx+ghostWidth-1,j, k, ID), _gParams.smallr);

	  } // end for j / end update xMin/xMax with remapped values
	} // end for k / end update xMin/xMax with remapped values

	if (debugShear) {
	  outputHdf5Debug(h_UNew,"cpu_shear_UNew_update2_",nStep);
	}	
	TIMER_STOP(timerShearBorder);
	
      } // end shearingBoxEnabled
      
      /*
       * magnetic field update
       */
      TIMER_START(timerCtUpdate);
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
	outputHdf5Debug(h_UNew, "UNew_after_update_", nStep);
	outputHdf5Debug(h_emf, "emf_", nStep);
	
	// compute divergence B
	//compute_divB(h_UNew,h_debug2);
	//outputHdf5Debug(h_debug2, "divB_", nStep);
	
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
      real_t &nu = _gParams.nu;
      real_t &eta = _gParams.eta;
      if (nu>0 or eta>0) {
	if (shearingBoxEnabled) {

	  make_all_boundaries_shear(h_UNew, dt, nStep);
	  	  
	} else {
	  
	  make_all_boundaries(h_UNew);
	  
	} // end if (shearingBoxEnabled and dimType == THREE_D)
	
      } // end nu>0
      
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
      
    } // end THREE_D
    TIMER_STOP(timerGodunov);
  
    /*
     * update h_UNew for next time step
     */
    TIMER_START(timerBoundaries);
    if (shearingBoxEnabled and dimType == THREE_D) {

      make_all_boundaries_shear(h_UNew, dt, nStep);

    } else {

      make_all_boundaries(h_UNew);

    } // end if (shearingBoxEnabled and dimType == THREE_D)
    TIMER_STOP(timerBoundaries);

  } // MHDRunGodunov::godunov_unsplit_rotating_cpu
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  // IMPORTANT NOTICE : shearing border condition is only implemented 
  // for 3D problems and for XDIR direction
#ifdef __CUDACC__
  void MHDRunGodunov::make_boundaries_shear(DeviceArray<real_t> &U, real_t dt, int nStep)
  {

    if (dimType == TWO_D) {

      std::cerr << "Shear border condition in 2D is not implemented !!!" << std::endl; 
      std::cerr << " Will it ever be ???                               " << std::endl;

    } else { // THREE_D

      // copy inner (i.e. xmin) border
      copyDeviceArrayToBorderBufShear<XMIN,THREE_D,3>(d_shear_border_xmin, U);

      // copy outer (i.e. xmax) border
      copyDeviceArrayToBorderBufShear<XMAX,THREE_D,3>(d_shear_border_xmax, U);

      if (debugShear) {
	// DEBUG SHEAR
	HostArray<real_t> h_shear_debug; 
	h_shear_debug.allocate(make_uint4(ghostWidth,jsize,ksize,NVAR_MHD));
	d_shear_border_xmin.copyToHost(h_shear_debug);
	outputHdf5Debug(h_shear_debug, "gpu_shear_border_xmin_", nStep);
	d_shear_border_xmax.copyToHost(h_shear_debug);
	outputHdf5Debug(h_shear_debug, "gpu_shear_border_xmax_", nStep);
      }


      // compute shear border Y-direction slopes and final remapping
      {
	dim3 threadsPerBlock(MK_BOUND_BLOCK_SIZE_3D, 
			     MK_BOUND_BLOCK_SIZE_3D, 
			     1);
	dim3 blockCount( blocksFor(jsize, MK_BOUND_BLOCK_SIZE_3D),
			 blocksFor(ksize, MK_BOUND_BLOCK_SIZE_3D), 
			 1);
	
	int bPitch     = d_shear_slope_xmin.pitch();
	int bDimx      = d_shear_slope_xmin.dimx();
	int bDimy      = d_shear_slope_xmin.dimy();
	int bDimz      = d_shear_slope_xmin.dimz();
	int bArraySize = d_shear_slope_xmin.section();
	
	int uPitch     = U.pitch();
	int uDimx      = U.dimx();
	int uDimy      = U.dimy();
	int uDimz      = U.dimz();
	int uArraySize = U.section();
	
	kernel_compute_shear_border_slopes
	  <<<blockCount, threadsPerBlock>>>(d_shear_slope_xmin.data(),
					    d_shear_slope_xmax.data(),
					    d_shear_border_xmin.data(),
					    d_shear_border_xmax.data(),
					    bPitch,bDimx, bDimy, bDimz, bArraySize,
					    ghostWidth);
      
	if (debugShear) {
	  // DEBUG SHEAR
	  HostArray<real_t> h_shear_debug; 
	  h_shear_debug.allocate(make_uint4(ghostWidth,jsize,ksize,NVAR_MHD));
	  d_shear_slope_xmin.copyToHost(h_shear_debug);
	  outputHdf5Debug(h_shear_debug, "gpu_shear_slope_xmin_", nStep);
	  d_shear_slope_xmax.copyToHost(h_shear_debug);
	  outputHdf5Debug(h_shear_debug, "gpu_shear_slope_xmax_", nStep);
	}

	// perform final remapping in shear borders
	kernel_perform_final_remapping_shear_borders
	  <<<blockCount, threadsPerBlock>>>(U.data(),
					    uPitch, uDimx, uDimy, uDimz, uArraySize,
					    d_shear_slope_xmin.data(),
					    d_shear_slope_xmax.data(),
					    d_shear_border_xmin.data(),
					    d_shear_border_xmax.data(),
					    bPitch, bDimx, bDimy, bDimz, bArraySize,
					    ghostWidth, totalTime+dt);
	
	if (debugShear) {
	  // DEBUG SHEAR
	  HostArray<real_t> h_shear_debug; 
	  h_shear_debug.allocate(make_uint4(isize,jsize,ksize,NVAR_MHD));
	  U.copyToHost(h_shear_debug);
	  outputHdf5Debug(h_shear_debug, "gpu_after_final_remapping_", nStep);
	}

      } // end compute shear border Y-direction slopes and final remapping
       
    } // end THREE_D

  } // MHDRunGodunov::make_boundaries_shear
#else // CPU version
  void MHDRunGodunov::make_boundaries_shear(HostArray<real_t> &U, real_t dt, int nStep)
  {

    if (dimType == TWO_D) {
    
      std::cerr << "Shear border condition in 2D is not implemented !!!" << std::endl;
      std::cerr << " Will it ever be ???                               " << std::endl;

    } else { // THREE_D

      // the following code is adapted from Dumses/bval_shear.f90, 
      // subroutines bval_shear
      real_t deltay,epsi,eps,lambda;
      int jplus,jremap,jremapp1;
      
      deltay = 1.5 * _gParams.Omega0 * (_gParams.dx * _gParams.nx) * (totalTime+dt);
      deltay = FMOD(deltay, (_gParams.dy * _gParams.ny) );
      jplus  = (int) (deltay/dy);
      epsi   = FMOD(deltay,  dy);

      // copy inner (i.e. xmin) border
      copyHostArrayToBorderBufShear<XMIN,THREE_D,3>(h_shear_border_xmin, U);

      // copy outer (i.e. xmax) border
      copyHostArrayToBorderBufShear<XMAX,THREE_D,3>(h_shear_border_xmax, U);

      if (debugShear) {
	// DEBUG SHEAR
	outputHdf5Debug(h_shear_border_xmin, "cpu_shear_border_xmin_", nStep);
	outputHdf5Debug(h_shear_border_xmax, "cpu_shear_border_xmax_", nStep);
      }

      real_t &slope_type=_gParams.slope_type;


      /*
       *
       * compute slopes along Y in XMIN/XMAX borders
       *
       */

      // compute inner (xmin) slopes
      if (slope_type == 0) {
	// do nothing (slopes array are already reset
      }
      
      if(slope_type==1 or slope_type==2) { // minmod or average
	real_t dsgn, dlim, dcen, dlft, drgt, slop;
		
	for (int k=0; k<ksize; k++) {
	  for (int j=1; j<jsize-1; j++) {
	    for (int i=0; i<ghostWidth; i++) {
	      
	      for (int iVar=0; iVar<nbVar; iVar++) {
		
		if (iVar==IB) { // special treatment for Y-slope of BY
		  
		  // inner (XMIN) BY slope
		  h_shear_slope_xmin(i,j,k,IB) = 
		    h_shear_border_xmin(i,j+1,k,IB) - 
		    h_shear_border_xmin(i,j  ,k,IB);
		  
		  // outer (XMAX) BY slope
		  h_shear_slope_xmax(i,j,k,IB) = 
		    h_shear_border_xmax(i,j+1,k,IB) - 
		    h_shear_border_xmax(i,j  ,k,IB);
		    
		} else { // all other components except BY
		  
		  // inner (XMIN) slopes in second coordinate direction 
		  dlft = slope_type * ( h_shear_border_xmin(i,j  ,k,iVar) - 
					h_shear_border_xmin(i,j-1,k,iVar) );
		  drgt = slope_type * ( h_shear_border_xmin(i,j+1,k,iVar) - 
					h_shear_border_xmin(i,j  ,k,iVar) );
		  dcen = HALF_F * (dlft+drgt)/slope_type;
		  dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
		  slop = FMIN( FABS(dlft), FABS(drgt) );
		  dlim = slop;
		  if ( (dlft*drgt) <= ZERO_F )
		    dlim = ZERO_F;
		  h_shear_slope_xmin(i,j,k,iVar) = dsgn * FMIN( dlim, FABS(dcen) );
		  
		  // outer (XMAX) slopes in second coordinate direction 
		  dlft = slope_type * ( h_shear_border_xmax(i,j  ,k,iVar) - 
					h_shear_border_xmax(i,j-1,k,iVar) );
		  drgt = slope_type * ( h_shear_border_xmax(i,j+1,k,iVar) - 
					h_shear_border_xmax(i,j  ,k,iVar) );
		  dcen = HALF_F * (dlft+drgt)/slope_type;
		  dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
		  slop = FMIN( FABS(dlft), FABS(drgt) );
		  dlim = slop;
		  if ( (dlft*drgt) <= ZERO_F )
		    dlim = ZERO_F;
		  h_shear_slope_xmax(i,j,k,iVar) = dsgn * FMIN( dlim, FABS(dcen) );
		  
		} // end if (iVar==IB)
	      } // end for iVar

	    } // end for i
	  } // end for j
	} // end for k
	  
      } // end slope_type==1 or slope_type==2
      
      if (debugShear) {
	// DEBUG SHEAR
	outputHdf5Debug(h_shear_slope_xmin, "cpu_shear_slope_xmin_", nStep);
	outputHdf5Debug(h_shear_slope_xmax, "cpu_shear_slope_xmax_", nStep);
      }


      /*
       *
       * perform final remapping in shear borders
       *
       */
      for (int k=0; k<ksize; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  
	  
	  /*
	   * inner (XMIN) border
	   */
	  jremap   = j     -jplus-1;
	  jremapp1 = jremap+1; 
	  eps      = 1.0-epsi/dy;
	  
	  if (jremap  < ghostWidth) jremap   += ny;
	  if (jremapp1< ghostWidth) jremapp1 += ny;
	  
	  lambda = HALF_F * eps*(eps-1.0);
	      
	  
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int i=0; i<ghostWidth; i++) {

	      if (iVar == IB) { // special treatment for BY magnetic field component
		
		U(i,j,k,IB) = 
		  h_shear_border_xmax(i,jremap,k,IB) + 
		  eps*h_shear_slope_xmax(i,jremap,k,IB);

		// DEBUG : there is a potential problem if initial condition of BY on x-borders is
		// non-zero; the formula above is identically zero in X-MIN border at j==3 !!!
		// if (i==0 and j==3 and k==3) {
		//   printf("DEBUG SHEAR : jremap %d\n",jremap);
		//   printf("DEBUG SHEAR make_boundaries_shear : %f %f %f %f\n",
		// 	 U(i,j,k,IB),
		// 	 h_shear_border_xmax(i,jremap,k,IB),
		// 	 h_shear_slope_xmax(i,jremap,k,IB), eps);
		// }

	      } else { // other components
		
		U(i,j,k,iVar) = 
		  (1.0-eps) * h_shear_border_xmax(i,jremap  ,k,iVar) +
		  eps       * h_shear_border_xmax(i,jremapp1,k,iVar) + 
		  lambda    * ( h_shear_slope_xmax(i,jremap  ,k,iVar) - 
				h_shear_slope_xmax(i,jremapp1,k,iVar) );

	      } // end if (iVar == IB)
	    } // end for i
	  } // end for iVar
	   
	  /*
	   * outer (XMAX) border
	   */
	  jremap   = j     +jplus;
	  jremapp1 = jremap+1;
	  eps      = epsi/dy;
	  
	  if (jremap   > ny+ghostWidth-1) jremap   -= ny;
	  if (jremapp1 > ny+ghostWidth-1) jremapp1 -= ny;
	  
	  lambda = HALF_F * eps*(eps-1.0);
	  
	  for (int iVar=0; iVar<nbVar; iVar++) {
	    for (int i=0; i<ghostWidth; i++) {

	      // update Hydro variables in ghost cells
	      if (iVar < 5) {
		//U(nx+ghostWidth+i,j,k,IB);
		U(nx+ghostWidth+i,j,k,iVar) = 
		  (1.0-eps) * h_shear_border_xmin(i,jremap  ,k,iVar) + 
		  eps       * h_shear_border_xmin(i,jremapp1,k,iVar) + 
		  lambda    * ( h_shear_slope_xmin(i,jremapp1,k,iVar) - 
				h_shear_slope_xmin(i,jremap  ,k,iVar) );
	      }
	      if (iVar == IA) { // ! WARNING : do NOT write in first outer ghost cell
		if (i>0) {
		  U(nx+ghostWidth+i,j,k,IA) = 
		    (1.0-eps) * h_shear_border_xmin(i,jremap  ,k,IA) + 
		    eps       * h_shear_border_xmin(i,jremapp1,k,IA) + 
		    lambda    * ( h_shear_slope_xmin(i,jremapp1,k,IA) - 
				  h_shear_slope_xmin(i,jremap  ,k,IA) );
		}
	      }
	      if (iVar == IB) {
		U(nx+ghostWidth+i,j,k,IB) = h_shear_border_xmin(i,jremap,k,IB) + 
		  eps * h_shear_slope_xmin(i,jremap,k,IB);
	      }
	      if (iVar == IC) {
		U(nx+ghostWidth+i,j,k,IC) = 
		  (1.0-eps) * h_shear_border_xmin(i,jremap  ,k,IC) + 
		  eps       * h_shear_border_xmin(i,jremapp1,k,IC) + 
		  lambda    * ( h_shear_slope_xmin(i,jremapp1,k,IC) - 
				h_shear_slope_xmin(i,jremap  ,k,IC) );
	      }
	      
	    } // end for i
	  } // end for iVar      
	  
	} //end for j
      } // end for k

      if (debugShear)
	outputHdf5Debug(U, "cpu_after_final_remapping_", nStep);
      
    } // end THREE_D
    
  } //MHDRunGodunov::make_boundaries_shear
#endif // __CUDACC__

#ifdef __CUDACC__
  void MHDRunGodunov::make_all_boundaries_shear(DeviceArray<real_t> &U, 
						real_t dt, 
						int nStep)
  {

    // YDIR must be done first !
    make_boundaries(U,YDIR);
    
    make_boundaries_shear(U, dt, nStep);
    
    make_boundaries(U,ZDIR);
    
    make_boundaries(U,YDIR);

  } // MHDRunGodunov::make_all_boundaries_shear -- GPU version
#else // CPU version
  void MHDRunGodunov::make_all_boundaries_shear(HostArray<real_t> &U, 
						real_t dt, 
						int nStep)
  {

    // YDIR must be done first !
    make_boundaries(U,YDIR);

    make_boundaries_shear(U, dt, nStep);

    make_boundaries(U,ZDIR);

    make_boundaries(U,YDIR);
    
  } //MHDRunGodunov::make_all_boundaries_shear -- CPU version
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  /*
   * main routine to start simulation.
   */
  void MHDRunGodunov::start() 
  {

    // test if we are performing a re-start run (default : false)
    bool restartEnabled = configMap.getBool("run","restart",false);
    
    // should we include ghost cells in output files ?
    bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);

    /*
     * initial condition.
     */
    int nStep = 0;

    std::cout << "Initialization\n";
    nStep = init_simulation(problem);

    // make sure border conditions are OK at beginning of simulation
    if (restartEnabled and ghostIncluded) {
      
      // we do not need to call make_all_boundaries since h_U/d_U is
      // fully filled from reading input data file (ghost zones already
      // set properly) in init_simulation.
      
    } else { // not a restart run
      
#ifdef __CUDACC__
    if (shearingBoxEnabled and dimType == THREE_D) {

      make_all_boundaries_shear(d_U, 0, 0);

    } else {

      make_all_boundaries(d_U);

    }

    // initialize d_U2
    d_U.copyTo(d_U2);

#else

    if (shearingBoxEnabled and dimType == THREE_D) {

      make_all_boundaries_shear(h_U,0,0);

    } else {

      make_all_boundaries(h_U);

    } // end if (shearingBoxEnabled and dimType == THREE_D)
   
    // initialize h_U2
    h_U.copyTo(h_U2);

#endif // __CUDACC__

    } // end if (restartEnabled and ghostIncluded)
  
    // dump information about computations about to start
    std::cout << "Starting time integration for MHD (Godunov)" << std::endl;
    std::cout << "use unsplit integration" << std::endl;
    std::cout << "Resolution (nx,ny,nz) " << nx << " " << ny << " " << nz << std::endl;
    std::cout << "Resolution (dx,dy,dz) " << fmt(_gParams.dx) << fmt(_gParams.dy) << fmt(_gParams.dz) << std::endl;

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
    real_t dt = compute_dt_mhd(0);
    std::cout << "Initial dt : " << fmt(dt) << std::endl;
  
    // how often should we print some log
    int nLog = configMap.getInteger("run", "nlog", 0);

    // Do we want to dump faces of domain ?
    int nOutputFaces = configMap.getInteger("run", "nOutputFaces", -1);
    bool outputFacesEnabled = false;
    if (nOutputFaces>0)
      outputFacesEnabled = true;
    bool outputFacesVtk = configMap.getBool("output", "outputFacesVtk", false);
    HostArray<real_t> h_xface, h_yface, h_zface;
    if (outputFacesEnabled == true and dimType==THREE_D) {
      h_xface.allocate(make_uint4(1,     jsize, ksize, nbVar));
      h_yface.allocate(make_uint4(isize, 1,     ksize, nbVar));
      h_zface.allocate(make_uint4(isize, jsize, 1,     nbVar));
    }
#ifdef __CUDACC__
    DeviceArray<real_t> d_xface, d_yface, d_zface;
    if (outputFacesEnabled == true and dimType==THREE_D) {
      d_xface.allocate(make_uint4(1,     jsize, ksize, nbVar));
      d_yface.allocate(make_uint4(isize, 1,     ksize, nbVar));
      d_zface.allocate(make_uint4(isize, jsize, 1,     nbVar));
    }
#endif // __CUDACC__

    // timing
    Timer timerTotal;
    Timer timerWriteOnDisk;
    Timer timerHistory;

    // choose which history method will be called
    setupHistory();
    real_t dtHist  = configMap.getFloat("history", "dtHist", 10*dt);
    real_t tHist   = totalTime;

    // start timer
    timerTotal.start();
  
    while(totalTime < tEnd && nStep < nStepmax)
      {
      
	/* just some log */
	if (nLog>0 and (nStep % nLog) == 0) {
	  
	  std::cout << "["       << current_date()   << "]"
		    << "  step=" << std::setw(9)     << nStep 
		    << " t="     << fmt(totalTime)
		    << " dt="    << fmt(dt,16,12)    << std::endl;
	}

	/* Output results */
	if((nStep % nOutput)==0) {
	
	  timerWriteOnDisk.start();
	
	  // make sure Device data are copied back onto Host memory
	  // which data to save ?
	  copyGpuToCpu(nStep);

	  output(getDataHost(nStep), nStep, ghostIncluded);
	  // dump laplacian / divergence B
	  // {
	  //   HostArray<real_t> h_debug_laplacian;
	  //   h_debug_laplacian.allocate(make_uint4(isize,jsize,ksize,3));
	  //   compute_laplacianB(getDataHost(nStep), h_debug_laplacian);
	  //   outputHdf5Debug(h_debug_laplacian, "laplacian_", nStep);
	  //   compute_divB(getDataHost(nStep), h_debug_laplacian);
	  //   outputHdf5Debug(h_debug_laplacian, "divB_", nStep);
	  // }

	  timerWriteOnDisk.stop();
	
	  std::cout << "["       << current_date()       << "]"
		    << "  step=" << std::setw(9)         << nStep 
		    << " t="     << fmt(totalTime)
		    << " dt="    << fmt(dt,16,12)        << std::endl;


	} // end if nStep%nOutput == 0

	/* Do we want to output faces of domain ? */
	if (outputFacesEnabled and (nStep % nOutputFaces)==0) {
	  // copy faces
#ifdef __CUDACC__
	  if (outputFacesVtk) outputFaces(nStep, FF_VTK, 
					  h_xface, h_yface, h_zface, 
					  d_xface, d_yface, d_zface);
#else
	  if (outputFacesVtk) outputFaces(nStep, FF_VTK);
#endif // __CUDACC__
	}

	// call history ?
	timerHistory.start();
	if (tHist == 0 or 
	    ( (totalTime-dt <= tHist+dtHist) and 
	      (totalTime    >  tHist+dtHist) ) ) {
	  copyGpuToCpu(nStep);
	  history(nStep,dt);
	  tHist += dtHist;
	}
	timerHistory.stop();
      
	/* one time step integration (nStep increment) */
	oneStepIntegration(nStep, totalTime, dt);

      } // end while
  
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

    std::cout << "DEBUG : totalTime " << totalTime << std::endl;
    std::cout << "DEBUG : dt        " << dt        << std::endl;

    printf("Euler MHD godunov total  time: %5.3f sec\n", timerTotal.elapsed());
    printf("Euler MHD godunov output time: %5.3f sec (%5.2f %% of total time)\n",timerWriteOnDisk.elapsed(), timerWriteOnDisk.elapsed()/timerTotal.elapsed()*100.);

#ifndef __CUDACC__
  /*
   * print flops report (CPU only)
   */
#ifdef USE_PAPI
  printf("Euler godunov total floating point operations per cell : %g\n", 1.0*papiFlops_total.getFlop()/(nx*ny*nz*nStep));
  printf("Euler godunov total MFlops                             : %g\n",  papiFlops_total.getFlops());
#endif // USE_PAPI

#endif // __CUDACC__

    /*
     * print timing report if required
     */
#ifdef DO_TIMING
    printf("Euler MHD godunov boundaries : %5.3f sec (%5.2f %% of total time)\n",timerBoundaries.elapsed(), timerBoundaries.elapsed()/timerTotal.elapsed()*100.);
    printf("Euler MHD godunov computing  : %5.3f sec (%5.2f %% of total time)\n",timerGodunov.elapsed(), timerGodunov.elapsed()/timerTotal.elapsed()*100.);
    

    if (shearingBoxEnabled) {
      printf("Euler MHD prim var           : %5.3f sec (%5.2f %% of computing time)\n",timerPrimVar.elapsed(), timerPrimVar.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD Elec field         : %5.3f sec (%5.2f %% of computing time)\n",timerElecField.elapsed(), timerElecField.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD mag slopes         : %5.3f sec (%5.2f %% of computing time)\n",timerMagSlopes.elapsed(), timerMagSlopes.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD trace              : %5.3f sec (%5.2f %% of computing time)\n",timerTrace.elapsed(), timerTrace.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD hydro shear        : %5.3f sec (%5.2f %% of computing time)\n",timerHydroShear.elapsed(), timerHydroShear.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD remapping          : %5.3f sec (%5.2f %% of computing time)\n",timerRemapping.elapsed(), timerRemapping.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD shear border       : %5.3f sec (%5.2f %% of computing time)\n",timerShearBorder.elapsed(), timerShearBorder.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD ct update          : %5.3f sec (%5.2f %% of computing time)\n",timerCtUpdate.elapsed(), timerCtUpdate.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD dissipative terms  : %5.3f sec (%5.2f %% of computing time)\n",timerDissipative.elapsed(), timerDissipative.elapsed()/timerGodunov.elapsed()*100.);

    } else {
      printf("Euler MHD prim var           : %5.3f sec (%5.2f %% of computing time)\n",timerPrimVar.elapsed(), timerPrimVar.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD Elec field         : %5.3f sec (%5.2f %% of computing time)\n",timerElecField.elapsed(), timerElecField.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD mag slopes         : %5.3f sec (%5.2f %% of computing time)\n",timerMagSlopes.elapsed(), timerMagSlopes.elapsed()/timerGodunov.elapsed()*100.);
      // implementation 3 only
      printf("Euler MHD trace-update       : %5.3f sec (%5.2f %% of computing time)\n",timerTraceUpdate.elapsed(), timerTraceUpdate.elapsed()/timerGodunov.elapsed()*100.);
      // implementation 4 only (split trace and update)
      printf("Euler MHD trace              : %5.3f sec (%5.2f %% of computing time)\n",timerTrace.elapsed(), timerTrace.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD update             : %5.3f sec (%5.2f %% of computing time)\n",timerUpdate.elapsed(), timerUpdate.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD compute emf        : %5.3f sec (%5.2f %% of computing time)\n",timerEmf.elapsed(), timerEmf.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD ct update          : %5.3f sec (%5.2f %% of computing time)\n",timerCtUpdate.elapsed(), timerCtUpdate.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler MHD dissipative terms  : %5.3f sec (%5.2f %% of computing time)\n",timerDissipative.elapsed(), timerDissipative.elapsed()/timerGodunov.elapsed()*100.);

    }

    printf("Euler MHD history            : %5.3f sec (%5.2f %% of total time)\n",timerHistory.elapsed(), timerHistory.elapsed()/timerTotal.elapsed()*100.);
    
#endif // DO_TIMING

    std::cout  << "####################################\n"
	       << "Global performance                  \n" 
	       << 1.0*nStep*(nx)*(ny)*(nz)/(timerTotal.elapsed()-timerWriteOnDisk.elapsed())
	       << " cell updates per seconds (based on wall time)\n"
	       << "####################################\n";
    
  } // MHDRunGodunov::start

  // =======================================================
  // =======================================================
  /*
   * do one time step integration
   */
  void MHDRunGodunov::oneStepIntegration(int& nStep, real_t& t, real_t& dt) 
  {
  
    // if nStep is even update U  into U2
    // if nStep is odd  update U2 into U
    dt=compute_dt_mhd(nStep % 2);
    godunov_unsplit(nStep, dt);
  
    // increment time
    nStep++;
    t+=dt;
  
  } // MHDRunGodunov::oneStepIntegration

} // namespace hydroSimu
