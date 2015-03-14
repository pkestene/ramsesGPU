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
 * \file MHDRunGodunovZslab.cpp
 * \brief Implements class MHDRunGodunovZslab
 * 
 * 3D only MHD Euler equation solver on a cartesian grid using Godunov method
 * with Riemann solver (HLL, HLLD).
 *
 * \date Sept 17, 2012
 * \author Pierre Kestener.
 *
 * $Id: MHDRunGodunovZslab.cpp 3450 2014-06-16 22:03:23Z pkestene $
 */
#include "MHDRunGodunovZslab.h"

// include CUDA kernel when necessary
#ifdef __CUDACC__
#include "godunov_unsplit_mhd_zslab.cuh"
#include "shearingBox_utils_zslab.cuh"
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

// OpenMP support
// #if _OPENMP
// # include <omp.h>
// #endif

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // MHDRunGodunovZslab class methods body
  ////////////////////////////////////////////////////////////////////////////////
  
  MHDRunGodunovZslab::MHDRunGodunovZslab(ConfigMap &_configMap)
    : MHDRunBase(_configMap)
    , shearingBoxEnabled(false)
    , zSlabNb(1)
    , zSlabWidth(0)
    , zSlabWidthG(0)
    , implementationVersion(4)
    , dumpDataForDebugEnabled(false)
    , debugShear(false)
#ifdef __CUDACC__
    , d_Q()
#else
    , h_Q() /* CPU only */
#endif // __CUDACC__
  {

    // the default version for 3D is 4 (which is faster)
    // enforce implementationVersion to be 4
    implementationVersion = 4;
    std::cout << "Using MHD Godunov implementation version : " << implementationVersion
	      << " with z-slab" << std::endl;

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
  
    std::cout << "###### Z-Slab implementation ######\n";
    std::cout << "Using " << zSlabNb << " z-slabs of width " << zSlabWidth << std::endl;
    std::cout << "###################################\n";

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

    /*
     *
     * MEMORY ALLOCATION
     *
     */

    // memory allocation primitive variable array Q
#ifdef __CUDACC__
    d_Q.allocate (make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
#else
    h_Q.allocate (make_uint4(isize, jsize, zSlabWidthG, nbVar));
#endif // __CUDACC__

    // memory allocation for EMF's
#ifdef __CUDACC__
    d_emf.allocate(make_uint4(isize, jsize, zSlabWidthG, 3    ), gpuMemAllocType); // 3 EMF's
#else
    h_emf.allocate(make_uint4(isize, jsize, zSlabWidthG, 3    )); // 3 EMF's
#endif // __CUDACC__


    // extra memory allocation for a specific implementation version
#ifdef __CUDACC__
    
    d_qm_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qm_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qm_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qp_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qp_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qp_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    
    d_qEdge_RT.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qEdge_RB.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qEdge_LT.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qEdge_LB.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    
    d_qEdge_RT2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qEdge_RB2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qEdge_LT2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qEdge_LB2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    
    d_qEdge_RT3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qEdge_RB3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qEdge_LT3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    d_qEdge_LB3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
    
    d_elec.allocate (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);
    d_dA.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);
    d_dB.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);
    d_dC.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);	
    
#else // CPU
    
    h_qm_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qm_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qm_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qp_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qp_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qp_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    
    h_qEdge_RT.allocate (make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qEdge_RB.allocate (make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qEdge_LT.allocate (make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qEdge_LB.allocate (make_uint4(isize, jsize, zSlabWidthG, nbVar));
    
    h_qEdge_RT2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qEdge_RB2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qEdge_LT2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qEdge_LB2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    
    h_qEdge_RT3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qEdge_RB3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qEdge_LT3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));
    h_qEdge_LB3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar));

    h_elec.allocate (make_uint4(isize, jsize, zSlabWidthG, 3));
    h_dA.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3));
    h_dB.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3));
    h_dC.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3));
    
#endif // __CUDACC__

    /*
     * extra memory allocation for shearing box simulations
     */
    // memory allocation required for shearing box simulations
    // there are 4 components :
    // - 2 for density remapping
    // - 2 for emf remapping
    if (shearingBoxEnabled) {

#ifdef __CUDACC__
      d_shear_flux_xmin.allocate      (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
      d_shear_flux_xmax.allocate      (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
      d_shear_flux_xmin_remap.allocate(make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
      d_shear_flux_xmax_remap.allocate(make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);

      d_shear_border_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
      d_shear_border_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);

      d_shear_slope_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
      d_shear_slope_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
#else
      h_shear_flux_xmin.allocate      (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP));
      h_shear_flux_xmax.allocate      (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP));
      h_shear_flux_xmin_remap.allocate(make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP));
      h_shear_flux_xmax_remap.allocate(make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP));

      h_shear_border_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar));
      h_shear_border_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar));

      h_shear_slope_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar));
      memset(h_shear_slope_xmin.data(),0,h_shear_slope_xmin.sizeBytes());
      h_shear_slope_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar));
      memset(h_shear_slope_xmax.data(),0,h_shear_slope_xmax.sizeBytes());
#endif // __CUDACC__

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
       cudaFuncSetCacheConfig(kernel_mhd_compute_trace_v4_zslab, cudaFuncCachePreferL1);
       cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_zslab, cudaFuncCachePreferL1);
       cudaFuncSetCacheConfig(kernel_mhd_compute_emf_v4_zslab, cudaFuncCachePreferL1);
       cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear_zslab, cudaFuncCachePreferL1);
       cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear_part1_zslab, cudaFuncCachePreferL1);
       cudaFuncSetCacheConfig(kernel_mhd_compute_emf_shear_zslab, cudaFuncCachePreferL1);
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

  } // MHDRunGodunovZslab::MHDRunGodunovZslab

  // =======================================================
  // =======================================================
  MHDRunGodunovZslab::~MHDRunGodunovZslab()
  {  
    
  } // MHDRunGodunovZslab::~MHDRunGodunovZslab()
  
  // =======================================================
  // =======================================================
  void MHDRunGodunovZslab::convertToPrimitives(real_t *U, real_t timeStep, int zSlabId)
  {

    // this is a CPU-only routine    
#ifndef __CUDACC__

    int physicalDimU[3] = {(int) h_U.pitch(),
			   (int) h_U.dimy(),
			   (int) h_U.dimz()};
    
    // primitive variable state vector
    real_t q[NVAR_MHD];
    
    // primitive variable domain array
    real_t *Q = h_Q.data();
      
    // section / domain size
    int arraySizeQ = h_Q.section();
    //int arraySizeU = h_U.section();

    // start and stop index of current slab (ghosts included)
    int kStart = zSlabWidth * zSlabId;
    int kStop  = zSlabWidth * zSlabId + zSlabWidthG;

    // update primitive variables array
    // please note the values of upper bounds (this is because the
    // computePrimitives_MHD_3D routine needs to access magnetic
    // fields in the neighbors on the dimension-wise right !).
    for (int k = kStart; 
	 k     < kStop; 
	 k++) {
      
      // take car of last slab might be partly outside domain
      if (k<ksize-1) {
	
	for (int j=0; j<jsize-1; j++) {
	  for (int i=0; i<isize-1; i++) {
	    
	    int indexLocU = i+j*isize+k*isize*jsize;
	    real_t c;
	    
	    computePrimitives_MHD_3D(U, physicalDimU, indexLocU, c, q, timeStep);
	    
	    // copy q state in h_Q
	    int indexLocQ = i+j*isize+(k-kStart)*isize*jsize;
	    int offset = indexLocQ;
	    Q[offset] = q[ID]; offset += arraySizeQ;
	    Q[offset] = q[IP]; offset += arraySizeQ;
	    Q[offset] = q[IU]; offset += arraySizeQ;
	    Q[offset] = q[IV]; offset += arraySizeQ;
	    Q[offset] = q[IW]; offset += arraySizeQ;
	    Q[offset] = q[IA]; offset += arraySizeQ;
	    Q[offset] = q[IB]; offset += arraySizeQ;
	    Q[offset] = q[IC];
	    
	  } // end for i
	} // end for j
      } // end if (k<ksize-1)
    } // end for k
    
#endif // __CUDACC__

  } // MHDRunGodunovZslab::convertToPrimitives

  // =======================================================
  // =======================================================
  void MHDRunGodunovZslab::godunov_unsplit(int nStep, real_t dt)
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

  } // MHDRunGodunovZslab::godunov_unsplit (GPU version)
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

} // MHDRunGodunovZslab::godunov_unsplit (CPU version)
#endif // __CUDACC__
  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunGodunovZslab::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
					       DeviceArray<real_t>& d_UNew,
					       real_t dt, int nStep)
  {

    // inner domain integration
    TIMER_START(timerGodunov);
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
	  dim3 dimBlock(PRIM_VAR_Z_BLOCK_DIMX_3D_V3,
			PRIM_VAR_Z_BLOCK_DIMY_3D_V3);
	  dim3 dimGrid(blocksFor(isize, PRIM_VAR_Z_BLOCK_DIMX_3D_V3), 
		       blocksFor(jsize, PRIM_VAR_Z_BLOCK_DIMY_3D_V3));
	  kernel_mhd_compute_primitive_variables_zslab
	    <<<dimGrid, dimBlock>>>(d_UOld.data(), 
				    d_Q.data(),
				    d_UOld.pitch(),
				    d_UOld.dimx(),
				    d_UOld.dimy(), 
				    d_UOld.dimz(),
				    dt,
				    zSlabInfo);
	  checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_primitive_variables_zslab error");
	  
	}
	TIMER_STOP(timerPrimVar);
      
	TIMER_START(timerElecField);
	{
	  // 3D Electric field computation kernel    
	  dim3 dimBlock(ELEC_FIELD_Z_BLOCK_DIMX_3D_V3,
			ELEC_FIELD_Z_BLOCK_DIMY_3D_V3);
	  dim3 dimGrid(blocksFor(isize, ELEC_FIELD_Z_BLOCK_INNER_DIMX_3D_V3), 
		       blocksFor(jsize, ELEC_FIELD_Z_BLOCK_INNER_DIMY_3D_V3));
	  kernel_mhd_compute_elec_field_zslab
	    <<<dimGrid, dimBlock>>>(d_UOld.data(), 
				    d_Q.data(),
				    d_elec.data(),
				    d_UOld.pitch(), 
				    d_UOld.dimx(), 
				    d_UOld.dimy(), 
				    d_UOld.dimz(),
				    dt,
				    zSlabInfo);
	  checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_elec_field_zslab error");
	  
	  if (dumpDataForDebugEnabled) {
	    d_elec.copyToHost(h_debug2);
	    outputHdf5Debug(h_debug2, "elec_", nStep);
	  }
	  
	}
	TIMER_STOP(timerElecField);
      
	TIMER_START(timerMagSlopes);
	{
	  // magnetic slopes computations
	  dim3 dimBlock(MAG_SLOPES_Z_BLOCK_DIMX_3D_V3,
			MAG_SLOPES_Z_BLOCK_DIMY_3D_V3);
	  dim3 dimGrid(blocksFor(isize, MAG_SLOPES_Z_BLOCK_INNER_DIMX_3D_V3), 
		       blocksFor(jsize, MAG_SLOPES_Z_BLOCK_INNER_DIMY_3D_V3));
	  kernel_mhd_compute_mag_slopes_zslab
	    <<<dimGrid, dimBlock>>>(d_UOld.data(), 
				    d_dA.data(),
				    d_dB.data(),
				    d_dC.data(),
				    d_UOld.pitch(),
				    d_UOld.dimx(),
				    d_UOld.dimy(),
				    d_UOld.dimz(),
				    zSlabInfo);
	  checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_mag_slopes_zslab error");
	
	}
	TIMER_STOP(timerMagSlopes);

	TIMER_START(timerTrace);
	// trace
	{
	  dim3 dimBlock(TRACE_Z_BLOCK_DIMX_3D_V4,
			TRACE_Z_BLOCK_DIMY_3D_V4);
	  dim3 dimGrid(blocksFor(isize, TRACE_Z_BLOCK_INNER_DIMX_3D_V4), 
		       blocksFor(jsize, TRACE_Z_BLOCK_INNER_DIMY_3D_V4));
	  kernel_mhd_compute_trace_v4_zslab
	    <<<dimGrid, dimBlock>>>(d_UOld.data(),
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
				    zSlabInfo);
	  checkCudaError("MHDRunGodunovZslab kernel_mhd_compute_trace_v4_zslab error");
	  
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
	    
	    d_qEdge_RT.copyToHost(h_debug);
	    outputHdf5Debug(h_debug, "qEdge_RT_", nStep);
	    d_qEdge_RB.copyToHost(h_debug);
	    outputHdf5Debug(h_debug, "qEdge_RB_", nStep);
	    d_qEdge_LT.copyToHost(h_debug);
	    outputHdf5Debug(h_debug, "qEdge_LT_", nStep);
	    
	  } // end dumpDataForDebugEnabled

	  // gravity predictor
	  if (gravityEnabled) {
	    dim3 dimBlock(GRAV_PRED_Z_BLOCK_DIMX_3D_V4,
			  GRAV_PRED_Z_BLOCK_DIMY_3D_V4);
	    dim3 dimGrid(blocksFor(isize, GRAV_PRED_Z_BLOCK_DIMX_3D_V4), 
			 blocksFor(jsize, GRAV_PRED_Z_BLOCK_DIMY_3D_V4));
	    kernel_mhd_compute_gravity_predictor_v4_zslab<<<dimGrid, 
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
			  dt,
			  zSlabInfo);
	    checkCudaError("MHDRunGodunov kernel_mhd_compute_gravity_predictor_v4_zslab error");
	  
	  } // end gravity predictor

	} // end trace
	TIMER_STOP(timerTrace);

	TIMER_START(timerUpdate);
	// update hydro
	{
	  dim3 dimBlock(UPDATE_Z_BLOCK_DIMX_3D_V4,
			UPDATE_Z_BLOCK_DIMY_3D_V4);
	  dim3 dimGrid(blocksFor(isize, UPDATE_Z_BLOCK_INNER_DIMX_3D_V4), 
		       blocksFor(jsize, UPDATE_Z_BLOCK_INNER_DIMY_3D_V4));
	  kernel_mhd_flux_update_hydro_v4_zslab
	    <<<dimGrid, dimBlock>>>(d_UOld.data(),
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
	  checkCudaError("MHDRunGodunovZslab kernel_mhd_flux_update_hydro_v4_zslab error");

	} // end update hydro

	// gravity source term
	if (gravityEnabled) {
	  compute_gravity_source_term(d_UNew, d_UOld, dt, zSlabInfo);
	}
	TIMER_STOP(timerUpdate);

	TIMER_START(timerEmf);
	// compute emf
	{
	  dim3 dimBlock(COMPUTE_EMF_Z_BLOCK_DIMX_3D_V4,
			COMPUTE_EMF_Z_BLOCK_DIMY_3D_V4);
	  dim3 dimGrid(blocksFor(isize, COMPUTE_EMF_Z_BLOCK_DIMX_3D_V4), 
		       blocksFor(jsize, COMPUTE_EMF_Z_BLOCK_DIMY_3D_V4));
	  kernel_mhd_compute_emf_v4_zslab
	    <<<dimGrid, dimBlock>>>(d_qEdge_RT.data(),
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
				    dt,
				    zSlabInfo);
	  checkCudaError("MHDRunGodunovZslab kernel_mhd_compute_emf_v4_zslab error");
	  
	} // end compute emf
	TIMER_STOP(timerEmf);
	
	TIMER_START(timerCtUpdate);
	// update magnetic field
	{
	  dim3 dimBlock(UPDATE_CT_Z_BLOCK_DIMX_3D_V4,
			UPDATE_CT_Z_BLOCK_DIMY_3D_V4);
	  dim3 dimGrid(blocksFor(isize, UPDATE_CT_Z_BLOCK_DIMX_3D_V4), 
		       blocksFor(jsize, UPDATE_CT_Z_BLOCK_DIMY_3D_V4));
	  kernel_mhd_flux_update_ct_v4_zslab
	    <<<dimGrid, dimBlock>>>(d_UOld.data(),
				    d_UNew.data(),
				    d_emf.data(),
				    d_UOld.pitch(), 
				    d_UOld.dimx(), 
				    d_UOld.dimy(), 
				    d_UOld.dimz(),
				    dt / dx, 
				    dt / dy,
				    dt / dz,
				    dt,
				    zSlabInfo);
	  checkCudaError("MHDRunGodunovZslab kernel_mhd_flux_update_ct_v4_zslab error");
	} // update magnetic field
	TIMER_STOP(timerCtUpdate);
	
      } // end for zSlabId

      /*****************************************************************/
      /*****************************************************************/
      TIMER_START(timerDissipative);
      // update borders
      real_t &nu  = _gParams.nu;
      real_t &eta = _gParams.eta;
      if (nu>0 or eta>0) {
      	make_all_boundaries(d_UNew);
      }

      if (eta>0) {
	// update magnetic field with resistivity emf

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

	  // take care that the last slab might be truncated
	  if (zSlabId == zSlabNb-1) {
	    zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
	  }

	  compute_resistivity_emf_3d(d_UOld, d_emf,     zSlabInfo);
	  compute_ct_update_3d      (d_UNew, d_emf, dt, zSlabInfo);
	  
	  real_t &cIso = _gParams.cIso;
	  if (cIso<=0) { // non-isothermal simulations

	    // compute energy flux
	    compute_resistivity_energy_flux_3d(d_UOld, d_qm_x, d_qm_y, d_qm_z, dt, zSlabInfo);
	    compute_hydro_update_energy       (d_UNew, d_qm_x, d_qm_y, d_qm_z,     zSlabInfo);

	  } // end cIso <= 0

	} // end for zSlabId

      } // end eta>0
      
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
	  
	  // take care that the last slab might be truncated
	  if (zSlabId == zSlabNb-1) {
	    zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
	  }

	  compute_viscosity_flux( d_UOld, d_flux_x, d_flux_y, d_flux_z, dt, zSlabInfo );
	  compute_hydro_update  ( d_UNew, d_flux_x, d_flux_y, d_flux_z,     zSlabInfo );
	} // end for zSlabId

      } // end compute viscosity force / update  

      TIMER_STOP(timerDissipative);
      /*****************************************************************/
      /*****************************************************************/

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


    }
    TIMER_STOP(timerGodunov);

    /*
     * update boundaries
     */
    TIMER_START(timerBoundaries);
    make_all_boundaries(d_UNew);
    TIMER_STOP(timerBoundaries);
    
  } // MHDRunGodunovZslab::godunov_unsplit_gpu

#else // CPU version

  // =======================================================
  // =======================================================
  void MHDRunGodunovZslab::godunov_unsplit_cpu(HostArray<real_t>& h_UOld, 
					       HostArray<real_t>& h_UNew, 
					       real_t dt, int nStep)
  {

    (void) nStep;
    
    // copy h_UOld into h_UNew
    // for (unsigned int indexGlob=0; indexGlob<h_UOld.size(); indexGlob++) {
    //   h_UNew(indexGlob) = h_UOld(indexGlob);
    // }
    h_UOld.copyTo(h_UNew);

    // scaling factor to apply to flux when updating hydro state h_U
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;

    // conservative variable domain array
    real_t *U = h_UOld.data();

    // primitive variable domain array
    //real_t *Q = h_Q.data();
    
    // section / domain size
    //int arraySizeQ = h_Q.section();
    //int arraySizeU = h_U.section();

    /*
     * main computation loop to update h_U : 2D loop over simulation domain location
     */
    TIMER_START(timerGodunov);
    
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
      
      // convert conservative to primitive variables (and source term predictor)
      // put results in h_Q object
      convertToPrimitives(U, dt, zSlabId);
    
      TIMER_START(timerElecField);
      // compute electric field components
      for (int k=1; k<ksizeSlab-1; k++) {

	int kU = k + kStart;

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
	      
	    B = HALF_F  * ( h_UOld(i  ,j  ,kU-1,IB) +
			    h_UOld(i  ,j  ,kU  ,IB) );
	      
	    C = HALF_F  * ( h_UOld(i  ,j-1,kU  ,IC) +
			    h_UOld(i  ,j  ,kU  ,IC) );
	      
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

	    A = HALF_F  * ( h_UOld(i  ,j  ,kU-1,IA) +
			    h_UOld(i  ,j  ,kU  ,IA) );
	      
	    C = HALF_F  * ( h_UOld(i-1,j  ,kU  ,IC) +
			    h_UOld(i  ,j  ,kU  ,IC) );
	      
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
	      
	    A = HALF_F  * ( h_UOld(i  ,j-1,kU  ,IA) +
			    h_UOld(i  ,j  ,kU  ,IA) );
	      
	    B = HALF_F  * ( h_UOld(i-1,j  ,kU  ,IB) +
			    h_UOld(i  ,j  ,kU  ,IB) );
	      
	    h_elec(i,j,k,IZ) = u*B-v*A;

	  } // end for i
	} // end for j
      } // end for k inside z-slab
      TIMER_STOP(timerElecField);

      TIMER_START(timerMagSlopes);
      // compute magnetic slopes
      for (int k=1; k<ksizeSlab-1; k++) {

	int kU = k + kStart;

	for (int j=1; j<jsize-1; j++) {
	  for (int i=1; i<isize-1; i++) {

	    real_t bfSlopes[15];
	    real_t dbfSlopes[3][3];

	    real_t (&dbfX)[3] = dbfSlopes[IX];
	    real_t (&dbfY)[3] = dbfSlopes[IY];
	    real_t (&dbfZ)[3] = dbfSlopes[IZ];
	    
	    // get magnetic slopes dbf
	    bfSlopes[0]  = h_UOld(i  ,j  ,kU  ,IA);
	    bfSlopes[1]  = h_UOld(i  ,j+1,kU  ,IA);
	    bfSlopes[2]  = h_UOld(i  ,j-1,kU  ,IA);
	    bfSlopes[3]  = h_UOld(i  ,j  ,kU+1,IA);
	    bfSlopes[4]  = h_UOld(i  ,j  ,kU-1,IA);
 
	    bfSlopes[5]  = h_UOld(i  ,j  ,kU  ,IB);
	    bfSlopes[6]  = h_UOld(i+1,j  ,kU  ,IB);
	    bfSlopes[7]  = h_UOld(i-1,j  ,kU  ,IB);
	    bfSlopes[8]  = h_UOld(i  ,j  ,kU+1,IB);
	    bfSlopes[9]  = h_UOld(i  ,j  ,kU-1,IB);
 
	    bfSlopes[10] = h_UOld(i  ,j  ,kU  ,IC);
	    bfSlopes[11] = h_UOld(i+1,j  ,kU  ,IC);
	    bfSlopes[12] = h_UOld(i-1,j  ,kU  ,IC);
	    bfSlopes[13] = h_UOld(i  ,j+1,kU  ,IC);
	    bfSlopes[14] = h_UOld(i  ,j-1,kU  ,IC);
 
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
      for (int k=ghostWidth-2; k<ksizeSlab-ghostWidth+1; k++) {

	int kU = k + kStart;

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
	    bfNb[0] = h_UOld(i  ,j  ,kU  ,IA);
	    bfNb[1] = h_UOld(i+1,j  ,kU  ,IA);
	    bfNb[2] = h_UOld(i  ,j  ,kU  ,IB);
	    bfNb[3] = h_UOld(i  ,j+1,kU  ,IB);
	    bfNb[4] = h_UOld(i  ,j  ,kU  ,IC);
	    bfNb[5] = h_UOld(i  ,j  ,kU+1,IC);
	      
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
	      
	      int kG = k + kStart;
	      
	      real_t grav_x = HALF_F * dt * h_gravity(i,j,kG,IX);
	      real_t grav_y = HALF_F * dt * h_gravity(i,j,kG,IY);
	      real_t grav_z = HALF_F * dt * h_gravity(i,j,kG,IZ);
	      
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

      int ksizeSlabStopUpdate = ksizeSlab-ghostWidth;
      if (zSlabId == zSlabNb-1) ksizeSlabStopUpdate += 1;

      for (int k=ghostWidth; k<ksizeSlab-ghostWidth+1/*ksizeSlabStopUpdate*/; k++) {

	int kU = k+kStart;

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
	    /* 
	     * update with flux_x
	     */
	    if ( i  > ghostWidth       and 
		 j  < jsize-ghostWidth and 
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i-1,j  ,kU  ,ID) -= flux_x[ID]*dtdx;
	      h_UNew(i-1,j  ,kU  ,IP) -= flux_x[IP]*dtdx;
	      h_UNew(i-1,j  ,kU  ,IU) -= flux_x[IU]*dtdx;
	      h_UNew(i-1,j  ,kU  ,IV) -= flux_x[IV]*dtdx;
	      h_UNew(i-1,j  ,kU  ,IW) -= flux_x[IW]*dtdx;
	    }
	    
	    if ( i  < isize-ghostWidth and 
		 j  < jsize-ghostWidth and 
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
	    if ( i  < isize-ghostWidth and
		 j  > ghostWidth       and
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i  ,j-1,kU  ,ID) -= flux_y[ID]*dtdy;
	      h_UNew(i  ,j-1,kU  ,IP) -= flux_y[IP]*dtdy;
	      h_UNew(i  ,j-1,kU  ,IU) -= flux_y[IV]*dtdy; // IU and IV swapped
	      h_UNew(i  ,j-1,kU  ,IV) -= flux_y[IU]*dtdy; // IU and IV swapped
	      h_UNew(i  ,j-1,kU  ,IW) -= flux_y[IW]*dtdy;
	    }
	    
	    if ( i  < isize-ghostWidth and 
		 j  < jsize-ghostWidth and 
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,kU  ,ID) += flux_y[ID]*dtdy;
	      h_UNew(i  ,j  ,kU  ,IP) += flux_y[IP]*dtdy;
	      h_UNew(i  ,j  ,kU  ,IU) += flux_y[IV]*dtdy; // IU and IV swapped
	      h_UNew(i  ,j  ,kU  ,IV) += flux_y[IU]*dtdy; // IU and IV swapped
	      h_UNew(i  ,j  ,kU  ,IW) += flux_y[IW]*dtdy;
	    }
	    
	    /* 
	     * update with flux_z
	     */
	    if ( i  < isize-ghostWidth and 
		 j  < jsize-ghostWidth and
		 k  > ghostWidth and
                 kU > ghostWidth ) {
	      h_UNew(i  ,j  ,kU-1,ID) -= flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,kU-1,IP) -= flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,kU-1,IU) -= flux_z[IW]*dtdz; // IU and IW swapped
	      h_UNew(i  ,j  ,kU-1,IV) -= flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,kU-1,IW) -= flux_z[IU]*dtdz; // IU and IW swapped
	    }
	    
	    if ( i  < isize-ghostWidth and 
		 j  < jsize-ghostWidth and 
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,kU  ,ID) += flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,kU  ,IP) += flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,kU  ,IU) += flux_z[IW]*dtdz; // IU and IW swapped
	      h_UNew(i  ,j  ,kU  ,IV) += flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,kU  ,IW) += flux_z[IU]*dtdz; // IU and IW swapped
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

      TIMER_START(timerCtUpdate);
      /*
       * magnetic field update
       */
      for (int k=ghostWidth; k<ksizeSlabStopUpdate; k++) {

	int kU = k+kStart;

	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {

	    // update with EMFZ
	    if (kU < ksize-ghostWidth) {
	      h_UNew(i ,j ,kU, IA) += ( h_emf(i  ,j+1, k, I_EMFZ) - 
					h_emf(i,  j  , k, I_EMFZ) ) * dtdy;
	      
	      h_UNew(i ,j ,kU, IB) -= ( h_emf(i+1,j  , k, I_EMFZ) - 
					h_emf(i  ,j  , k, I_EMFZ) ) * dtdx;

	    }

	    // update BX
	    h_UNew(i ,j ,kU, IA) -= ( h_emf(i,j,k+1, I_EMFY) -
				      h_emf(i,j,k  , I_EMFY) ) * dtdz;
	    
	    // update BY
	    h_UNew(i ,j ,kU, IB) += ( h_emf(i,j,k+1, I_EMFX) -
				      h_emf(i,j,k  , I_EMFX) ) * dtdz;
	    
	    // update BZ
	    h_UNew(i ,j ,kU, IC) += ( h_emf(i+1,j  ,k, I_EMFY) -
				      h_emf(i  ,j  ,k, I_EMFY) ) * dtdx;
	    h_UNew(i ,j ,kU, IC) -= ( h_emf(i  ,j+1,k, I_EMFX) -
				      h_emf(i  ,j  ,k, I_EMFX) ) * dtdy;
	    
	  } // end for i
	} // end for j
      } // end for k
      TIMER_STOP(timerCtUpdate);

    } // end zSlabId loop

    /*****************************************************************/
    /*****************************************************************/
    TIMER_START(timerDissipative);
    // update borders
    real_t &nu  = _gParams.nu;
    real_t &eta = _gParams.eta;
    if (nu>0 or eta>0) {
      make_all_boundaries(h_UNew);
    }

    /*
     * resistive term
     */
    if (eta>0) {

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

	// update magnetic field with resistivity emf
	compute_resistivity_emf_3d(h_UOld, h_emf,     zSlabInfo);
	compute_ct_update_3d      (h_UNew, h_emf, dt, zSlabInfo);

	real_t &cIso = _gParams.cIso;
	if (cIso<=0) { // non-isothermal simulations

	  // compute energy flux
	  compute_resistivity_energy_flux_3d(h_UOld, h_qm_x, h_qm_y, h_qm_z, dt, zSlabInfo);
	  compute_hydro_update_energy       (h_UNew, h_qm_x, h_qm_y, h_qm_z,     zSlabInfo);

	} // end (cIso<=0)

      } // end for zSlabId

    } // end eta>0

    /*
     * compute viscosity forces
     */
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
    /*****************************************************************/
    /*****************************************************************/

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
    

    TIMER_STOP(timerGodunov);
    
    /*
     * update h_Uold boundary, before updating h_U
     */
    TIMER_START(timerBoundaries);
    make_all_boundaries(h_UNew);
    TIMER_STOP(timerBoundaries);
    
  } // MHDRunGodunovZslab::godunov_unsplit_cpu
  
#endif // __CUDACC__


#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunGodunovZslab::godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
							DeviceArray<real_t>& d_UNew,
							real_t dt, int nStep)
  {
    
    // inner domain integration
    TIMER_START(timerGodunov);

    // copy h_UOld into h_UNew
    d_UOld.copyTo(d_UNew);

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
	dim3 dimBlock(PRIM_VAR_Z_BLOCK_DIMX_3D_V3,
		      PRIM_VAR_Z_BLOCK_DIMY_3D_V3);
	dim3 dimGrid(blocksFor(isize, PRIM_VAR_Z_BLOCK_DIMX_3D_V3), 
		     blocksFor(jsize, PRIM_VAR_Z_BLOCK_DIMY_3D_V3));
	kernel_mhd_compute_primitive_variables_zslab
	  <<<dimGrid, dimBlock>>>(d_UOld.data(), 
				  d_Q.data(),
				  d_UOld.pitch(),
				  d_UOld.dimx(),
				  d_UOld.dimy(), 
				  d_UOld.dimz(),
				  dt,
				  zSlabInfo);
	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_primitive_variables_zslab error");
	
      }
      TIMER_STOP(timerPrimVar);

      TIMER_START(timerElecField);
      {
	// 3D Electric field computation kernel    
	dim3 dimBlock(ELEC_FIELD_Z_BLOCK_DIMX_3D_V3,
		      ELEC_FIELD_Z_BLOCK_DIMY_3D_V3);
	dim3 dimGrid(blocksFor(isize, ELEC_FIELD_Z_BLOCK_INNER_DIMX_3D_V3), 
		     blocksFor(jsize, ELEC_FIELD_Z_BLOCK_INNER_DIMY_3D_V3));
	kernel_mhd_compute_elec_field_zslab
	  <<<dimGrid, dimBlock>>>(d_UOld.data(), 
				  d_Q.data(),
				  d_elec.data(),
				  d_UOld.pitch(), 
				  d_UOld.dimx(), 
				  d_UOld.dimy(), 
				  d_UOld.dimz(),
				  dt,
				  zSlabInfo);
	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_elec_field_zslab error");
	
      }
      TIMER_STOP(timerElecField);
      
      TIMER_START(timerMagSlopes);
      {
	// magnetic slopes computations
	dim3 dimBlock(MAG_SLOPES_Z_BLOCK_DIMX_3D_V3,
		      MAG_SLOPES_Z_BLOCK_DIMY_3D_V3);
	dim3 dimGrid(blocksFor(isize, MAG_SLOPES_Z_BLOCK_INNER_DIMX_3D_V3), 
		     blocksFor(jsize, MAG_SLOPES_Z_BLOCK_INNER_DIMY_3D_V3));
	kernel_mhd_compute_mag_slopes_zslab
	  <<<dimGrid, dimBlock>>>(d_UOld.data(), 
				  d_dA.data(),
				  d_dB.data(),
				  d_dC.data(),
				  d_UOld.pitch(), 
				  d_UOld.dimx(), 
				  d_UOld.dimy(), 
				  d_UOld.dimz(),
				  zSlabInfo);
	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_mag_slopes_zslab error");
      }
      TIMER_STOP(timerMagSlopes);
      
      TIMER_START(timerTrace);
      // trace
      {
	dim3 dimBlock(TRACE_Z_BLOCK_DIMX_3D_V4,
		      TRACE_Z_BLOCK_DIMY_3D_V4);
	dim3 dimGrid(blocksFor(isize, TRACE_Z_BLOCK_INNER_DIMX_3D_V4), 
		     blocksFor(jsize, TRACE_Z_BLOCK_INNER_DIMY_3D_V4));
	kernel_mhd_compute_trace_v4_zslab
	  <<<dimGrid, dimBlock>>>(d_UOld.data(),
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
				  zSlabInfo);
	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_trace_v4_zslab error");
	
	// gravity predictor
	if (gravityEnabled) {
	  dim3 dimBlock(GRAV_PRED_Z_BLOCK_DIMX_3D_V4,
			GRAV_PRED_Z_BLOCK_DIMY_3D_V4);
	  dim3 dimGrid(blocksFor(isize, GRAV_PRED_Z_BLOCK_DIMX_3D_V4), 
		       blocksFor(jsize, GRAV_PRED_Z_BLOCK_DIMY_3D_V4));
	  kernel_mhd_compute_gravity_predictor_v4_zslab
	    <<<dimGrid, dimBlock>>>(d_qm_x.data(),
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
				    dt,
				    zSlabInfo);
	  checkCudaError("MHDRunGodunovZslab kernel_mhd_compute_gravity_predictor_v4_zslab error");
	  
	} // end gravity predictor
	
      } // end trace
      TIMER_STOP(timerTrace);	    

      TIMER_START(timerHydroShear);
      // update

      int hydroShearVersion = configMap.getInteger("implementation","hydroShearVersion",1);
    
      // the following call should be modified (d_shear_??? are only allocated when 
      // shearingBoxEnabled is true ...
      if (hydroShearVersion == 0) {
	
	dim3 dimBlock(UPDATE_Z_BLOCK_DIMX_3D_V4,
		      UPDATE_Z_BLOCK_DIMY_3D_V4);
	dim3 dimGrid(blocksFor(isize, UPDATE_Z_BLOCK_INNER_DIMX_3D_V4), 
		     blocksFor(jsize, UPDATE_Z_BLOCK_INNER_DIMY_3D_V4));
	kernel_mhd_flux_update_hydro_v4_shear_zslab
	  <<<dimGrid, dimBlock>>>(d_UOld.data(),
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
				  dt,
				  zSlabInfo);
	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_flux_update_hydro_v4_shear_zslab error");
	
      } // end hydroShearVersion == 0
      
      if (hydroShearVersion != 0) {
	
	{

	  dim3 dimBlock(UPDATE_P1_Z_BLOCK_DIMX_3D_V4,
	  		UPDATE_P1_Z_BLOCK_DIMY_3D_V4);
	  dim3 dimGrid(blocksFor(isize, UPDATE_P1_Z_BLOCK_INNER_DIMX_3D_V4), 
	  	       blocksFor(jsize, UPDATE_P1_Z_BLOCK_INNER_DIMY_3D_V4));
	  kernel_mhd_flux_update_hydro_v4_shear_part1_zslab
	    <<<dimGrid, dimBlock>>>(d_UOld.data(),
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
	  			    dt,
	  			    zSlabInfo);
	  checkCudaError("MHDRunGodunovZslab :: kernel_mhd_flux_update_hydro_v4_shear_part1_zslab error");
	}
	
	{
	  
	  dim3 dimBlock(COMPUTE_EMF_Z_BLOCK_DIMX_3D_SHEAR,
			COMPUTE_EMF_Z_BLOCK_DIMY_3D_SHEAR);
	  dim3 dimGrid(blocksFor(isize, COMPUTE_EMF_Z_BLOCK_DIMX_3D_SHEAR), 
		       blocksFor(jsize, COMPUTE_EMF_Z_BLOCK_DIMY_3D_SHEAR));
	  kernel_mhd_compute_emf_shear_zslab<<<dimGrid, dimBlock>>>(d_qEdge_RT.data(),
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
								    dt,
								    zSlabInfo);
	    checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_emf_shear_zslab error");
	}
      } // end hydroShearVersion != 0
      
      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt, zSlabInfo);
      }

      TIMER_STOP(timerHydroShear);
      
      TIMER_START(timerRemapping);
      // flux and emf remapping
      {
	dim3 dimBlock(16, 16);
	dim3 dimGrid(blocksFor(jsize, dimBlock.x), 
		     blocksFor(ksize, dimBlock.y));
	
	
	kernel_remapping_mhd_3d_zslab
	  <<<dimGrid, dimBlock>>>(d_shear_flux_xmin.data(),
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
				  dt,
				  zSlabInfo);
	
      } // end flux and emf remapping
      TIMER_STOP(timerRemapping);
      
      
      TIMER_START(timerShearBorder);
      // update shear borders with density and emfY
      {
	dim3 dimBlock(16, 16);
	dim3 dimGrid(blocksFor(jsize, dimBlock.x), 
		     blocksFor(ksize, dimBlock.y));
	
	kernel_update_shear_borders_3d_zslab
	  <<<dimGrid, dimBlock>>>(d_UNew.data(),
				  d_shear_flux_xmin_remap.data(),
				  d_shear_flux_xmax_remap.data(),
				  d_UNew.pitch(), 
				  d_UNew.dimx(), 
				  d_UNew.dimy(), 
				  d_UNew.dimz(),
				  d_shear_flux_xmin_remap.pitch(),
				  totalTime,
				  dt,
				  zSlabInfo);
	
      } // end shear borders with density 
      TIMER_STOP(timerShearBorder);
      
      TIMER_START(timerCtUpdate);
      // update magnetic field
      {
	dim3 dimBlock(UPDATE_CT_Z_BLOCK_DIMX_3D_V4,
		      UPDATE_CT_Z_BLOCK_DIMY_3D_V4);
	dim3 dimGrid(blocksFor(isize, UPDATE_CT_Z_BLOCK_DIMX_3D_V4), 
		     blocksFor(jsize, UPDATE_CT_Z_BLOCK_DIMY_3D_V4));
	kernel_mhd_flux_update_ct_v4_zslab
	  <<<dimGrid, dimBlock>>>(d_UOld.data(),
				  d_UNew.data(),
				  d_emf.data(),
				  d_UOld.pitch(), 
				  d_UOld.dimx(), 
				  d_UOld.dimy(), 
				  d_UOld.dimz(),
				  dt / dx, 
				  dt / dy,
				  dt / dz,
				  dt,
				  zSlabInfo);
	checkCudaError("MHDRunGodunovZslab kernel_mhd_flux_update_ct_v4_zslab error");
      } // update magnetic field
      
      TIMER_STOP(timerCtUpdate);
      
    } // end for zSlabId
    
    /*****************************************************************/
    /*****************************************************************/
    TIMER_START(timerDissipative);
    {
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

	  // take care that the last slab might be truncated
	  if (zSlabId == zSlabNb-1) {
	    zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
	  }

	  compute_resistivity_emf_3d(d_UOld, d_emf,     zSlabInfo);
	  compute_ct_update_3d      (d_UNew, d_emf, dt, zSlabInfo);
	
	  real_t &cIso = _gParams.cIso;
	  if (cIso<=0) { // non-isothermal simulations

	    // compute energy flux
	    compute_resistivity_energy_flux_3d(d_UOld, d_qm_x, d_qm_y, d_qm_z, dt, zSlabInfo);
	    compute_hydro_update_energy       (d_UNew, d_qm_x, d_qm_y, d_qm_z,     zSlabInfo);

	  } // end for cIso <= 0

	} // end for zSlabId

      } // end eta>0
      
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
	  
	  // take care that the last slab might be truncated
	  if (zSlabId == zSlabNb-1) {
	    zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
	  }
	  
	  compute_viscosity_flux(d_UOld, d_flux_x, d_flux_y, d_flux_z, dt, zSlabInfo );
	  compute_hydro_update  (d_UNew, d_flux_x, d_flux_y, d_flux_z,     zSlabInfo );
	} // end for zSlabId

      } // end compute viscosity force / update
    }
    TIMER_STOP(timerDissipative);
    /*****************************************************************/
    /*****************************************************************/
      
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
    
    
  } // MHDRunGodunovZslab::godunov_unsplit_rotating_gpu
  
#else
  // =======================================================
  // =======================================================
  // Omega0 is assumed to be strictly positive
  void MHDRunGodunovZslab::godunov_unsplit_rotating_cpu(HostArray<real_t>& h_UOld, 
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
    //real_t *Q = h_Q.data();
    
    // section / domain size
    //int arraySizeQ = h_Q.section();
    //int arraySizeU = h_U.section();
   
    memset(h_emf.data(),0,h_emf.sizeBytes());

    /*
     * main computation loop to update h_U : 2D loop over simulation domain location
     */
    TIMER_START(timerGodunov);
    
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

      h_shear_flux_xmin.reset();
      h_shear_flux_xmax.reset();

      // convert conservative to primitive variables (and source term predictor)
      // put results in h_Q object
      convertToPrimitives(U, dt, zSlabId);
      
      TIMER_START(timerElecField);
      // compute electric field components
      for (int k=1; k<ksizeSlab-1; k++) {

	int kU = k + kStart;

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
	  
	    B = HALF_F  * ( h_UOld(i  ,j  ,kU-1,IB) +
			    h_UOld(i  ,j  ,kU  ,IB) );
	  
	    C = HALF_F  * ( h_UOld(i  ,j-1,kU  ,IC) +
			    h_UOld(i  ,j  ,kU  ,IC) );
	  
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

	    A = HALF_F  * ( h_UOld(i  ,j  ,kU-1,IA) +
			    h_UOld(i  ,j  ,kU  ,IA) );
	      
	    C = HALF_F  * ( h_UOld(i-1,j  ,kU  ,IC) +
			    h_UOld(i  ,j  ,kU  ,IC) );
	      
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
	      
	    A = HALF_F  * ( h_UOld(i  ,j-1,kU  ,IA) +
			    h_UOld(i  ,j  ,kU  ,IA) );
	      
	    B = HALF_F  * ( h_UOld(i-1,j  ,kU  ,IB) +
			    h_UOld(i  ,j  ,kU  ,IB) );
	      
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

      TIMER_START(timerMagSlopes);
      // compute magnetic slopes
      for (int k=1; k<ksizeSlab-1; k++) {

	int kU = k + kStart;

	for (int j=1; j<jsize-1; j++) {
	  for (int i=1; i<isize-1; i++) {

	    real_t bfSlopes[15];
	    real_t dbfSlopes[3][3];

	    real_t (&dbfX)[3] = dbfSlopes[IX];
	    real_t (&dbfY)[3] = dbfSlopes[IY];
	    real_t (&dbfZ)[3] = dbfSlopes[IZ];
	    
	    // get magnetic slopes dbf
	    bfSlopes[0]  = h_UOld(i  ,j  ,kU  ,IA);
	    bfSlopes[1]  = h_UOld(i  ,j+1,kU  ,IA);
	    bfSlopes[2]  = h_UOld(i  ,j-1,kU  ,IA);
	    bfSlopes[3]  = h_UOld(i  ,j  ,kU+1,IA);
	    bfSlopes[4]  = h_UOld(i  ,j  ,kU-1,IA);
 
	    bfSlopes[5]  = h_UOld(i  ,j  ,kU  ,IB);
	    bfSlopes[6]  = h_UOld(i+1,j  ,kU  ,IB);
	    bfSlopes[7]  = h_UOld(i-1,j  ,kU  ,IB);
	    bfSlopes[8]  = h_UOld(i  ,j  ,kU+1,IB);
	    bfSlopes[9]  = h_UOld(i  ,j  ,kU-1,IB);
 
	    bfSlopes[10] = h_UOld(i  ,j  ,kU  ,IC);
	    bfSlopes[11] = h_UOld(i+1,j  ,kU  ,IC);
	    bfSlopes[12] = h_UOld(i-1,j  ,kU  ,IC);
	    bfSlopes[13] = h_UOld(i  ,j+1,kU  ,IC);
	    bfSlopes[14] = h_UOld(i  ,j-1,kU  ,IC);
 
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
      for (int k=ghostWidth-2; k<ksizeSlab-ghostWidth+1; k++) {

	int kU = k + kStart;

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
	    bfNb[0] = h_UOld(i  ,j  ,kU  ,IA);
	    bfNb[1] = h_UOld(i+1,j  ,kU  ,IA);
	    bfNb[2] = h_UOld(i  ,j  ,kU  ,IB);
	    bfNb[3] = h_UOld(i  ,j+1,kU  ,IB);
	    bfNb[4] = h_UOld(i  ,j  ,kU  ,IC);
	    bfNb[5] = h_UOld(i  ,j  ,kU+1,IC);
	      
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
	      
	      int kG = k + kStart;
	      
	      real_t grav_x = HALF_F * dt * h_gravity(i,j,kG,IX);
	      real_t grav_y = HALF_F * dt * h_gravity(i,j,kG,IY);
	      real_t grav_z = HALF_F * dt * h_gravity(i,j,kG,IZ);
	      
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

      TIMER_START(timerHydroShear);
      // Finally compute fluxes from rieman solvers, and update

      int ksizeSlabStopUpdate = ksizeSlab-ghostWidth;
      if (zSlabId == zSlabNb-1) ksizeSlabStopUpdate += 1;

      for (int k=ghostWidth; k<ksizeSlab-ghostWidth+1/*ksizeSlabStopUpdate*/; k++) {
	
	int kU = k+kStart;
	
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

	    if (i  < isize-ghostWidth and 
		j  < jsize-ghostWidth and 
		k  < ksizeSlab-ghostWidth and
		kU < ksize-ghostWidth) {
	      real_t dsx =   TWO_F * Omega0 * dt * h_UNew(i,j,kU,IV)/(ONE_F + lambda);
	      real_t dsy = -HALF_F * Omega0 * dt * h_UNew(i,j,kU,IU)/(ONE_F + lambda);
	      h_UNew(i,j,kU,IU) = h_UNew(i,j,kU,IU)*ratio + dsx;
	      h_UNew(i,j,kU,IV) = h_UNew(i,j,kU,IV)*ratio + dsy;
	    }

	    /* in shearing box simulation, there is a special treatment for flux_x[ID] */
	    
	    /* 
	     * update with flux_x
	     */
	    if ( i  > ghostWidth       and 
		 j  < jsize-ghostWidth and 
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      
	      if ( shearingBoxEnabled and  i==(nx+ghostWidth) ) {
		// do not perform update at border XMAX (we need to store density flux and 
		// perform a remapping)
		h_shear_flux_xmax(j,k,I_DENS)       = flux_x[ID]*dtdx;
	      } else {
		// perform a normal update
		h_UNew(i-1,j  ,kU  ,ID) -= flux_x[ID]*dtdx;
	      }
	      
	      h_UNew(i-1,j  ,kU  ,IP) -= flux_x[IP]*dtdx;
	      h_UNew(i-1,j  ,kU  ,IU) -= (alpha1*flux_x[IU]+     alpha2*flux_x[IV])*dtdx;
	      h_UNew(i-1,j  ,kU  ,IV) -= (alpha1*flux_x[IV]-0.25*alpha2*flux_x[IU])*dtdx;
	      h_UNew(i-1,j  ,kU  ,IW) -= flux_x[IW]*dtdx;
	      
	    }
	    
	    if ( i  < isize-ghostWidth and 
		 j  < jsize-ghostWidth and 
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      
	      if (shearingBoxEnabled and  i==ghostWidth ) {
		// do not perform update at border XMIN (we need to store density flux and
		// perform a remapping)
		h_shear_flux_xmin(j,k,I_DENS)       = flux_x[ID]*dtdx;
	      } else {
		// perform a normal update
		h_UNew(i  ,j  ,kU  ,ID) += flux_x[ID]*dtdx;
	      }
	      h_UNew(i  ,j  ,kU  ,IP) += flux_x[IP]*dtdx;
	      h_UNew(i  ,j  ,kU  ,IU) += (alpha1*flux_x[IU]+     alpha2*flux_x[IV])*dtdx;
	      h_UNew(i  ,j  ,kU  ,IV) += (alpha1*flux_x[IV]-0.25*alpha2*flux_x[IU])*dtdx;
	      h_UNew(i  ,j  ,kU  ,IW) += flux_x[IW]*dtdx;
	    }
	    
	    /*
	     * update with flux_y
	     */
	    if ( i  < isize-ghostWidth and
		 j  > ghostWidth       and
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i  ,j-1,kU  ,ID) -= flux_y[ID]*dtdy;
	      h_UNew(i  ,j-1,kU  ,IP) -= flux_y[IP]*dtdy;
	      h_UNew(i  ,j-1,kU  ,IU) -= (alpha1*flux_y[IV]+     alpha2*flux_y[IU])*dtdy; // IU and IV swapped
	      h_UNew(i  ,j-1,kU  ,IV) -= (alpha1*flux_y[IU]-0.25*alpha2*flux_y[IV])*dtdy; // IU and IV swapped
	      h_UNew(i  ,j-1,kU  ,IW) -= flux_y[IW]*dtdy;
	    }
	    
	    if ( i  < isize-ghostWidth and 
		 j  < jsize-ghostWidth and 
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,kU  ,ID) += flux_y[ID]*dtdy;
	      h_UNew(i  ,j  ,kU  ,IP) += flux_y[IP]*dtdy;
	      h_UNew(i  ,j  ,kU  ,IU) += (alpha1*flux_y[IV]+     alpha2*flux_y[IU])*dtdy; // IU and IV swapped
	      h_UNew(i  ,j  ,kU  ,IV) += (alpha1*flux_y[IU]-0.25*alpha2*flux_y[IV])*dtdy; // IU and IV swapped
	      h_UNew(i  ,j  ,kU  ,IW) += flux_y[IW]*dtdy;
	    }
	    
	    /*
	     * update with flux_z
	     */
	    if ( i  < isize-ghostWidth and 
		 j  < jsize-ghostWidth and
		 k  > ghostWidth and
		 kU > ghostWidth ) {
	      h_UNew(i  ,j  ,kU-1,ID) -= flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,kU-1,IP) -= flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,kU-1,IU) -= (alpha1*flux_z[IW]+
					  alpha2*flux_z[IV])*dtdz; // IU and IW swapped
	      h_UNew(i  ,j  ,kU-1,IV) -= (alpha1*flux_z[IV]-0.25*
					  alpha2*flux_z[IW])*dtdz;
	      h_UNew(i  ,j  ,kU-1,IW) -= flux_z[IU]*dtdz; // IU and IW swapped
	    }
	    
	    if ( i  < isize-ghostWidth and 
		 j  < jsize-ghostWidth and 
		 k  < ksizeSlab-ghostWidth and
		 kU < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,kU  ,ID) += flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,kU  ,IP) += flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,kU  ,IU) += (alpha1*flux_z[IW]+     
					  alpha2*flux_z[IV])*dtdz; // IU and IW swapped
	      h_UNew(i  ,j  ,kU  ,IV) += (alpha1*flux_z[IV]-0.25*
					  alpha2*flux_z[IW])*dtdz;
	      h_UNew(i  ,j  ,kU  ,IW) += flux_z[IU]*dtdz; // IU and IW swapped
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
	    if (kU<ksize-ghostWidth) {
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

      TIMER_STOP(timerHydroShear);

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
	for (int k=0; k<ksizeSlab; k++) {
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
	    if (j>=ghostWidth and j<jsize    -ghostWidth+1 and 
		k>=ghostWidth and k<ksizeSlab-ghostWidth+1) {
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
	    if (j>=ghostWidth and j<jsize    -ghostWidth+1 and
		k>=ghostWidth and k<ksizeSlab-ghostWidth+1) {
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

	/*
	 * finally update xMin/xMax border with remapped flux
	 */
	TIMER_START(timerShearBorder);
	for (int k=ghostWidth; k<ksizeSlabStopUpdate; k++) {
	  for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {

	    int kU = k+kStart;

	    // update density
	    h_UNew(ghostWidth     ,j, kU, ID) += h_shear_flux_xmin_remap(j,k,I_DENS);
	    h_UNew(nx+ghostWidth-1,j, kU, ID) -= h_shear_flux_xmax_remap(j,k,I_DENS);
	    
	    h_UNew(ghostWidth     ,j, kU, ID)  = FMAX(h_UNew(ghostWidth     ,j, kU, ID), gParams.smallr);
	    h_UNew(nx+ghostWidth-1,j, kU, ID)  = FMAX(h_UNew(nx+ghostWidth-1,j, kU, ID), gParams.smallr);

	  } // end for j / end update xMin/xMax with remapped values
	} // end for k / end update xMin/xMax with remapped values

	TIMER_STOP(timerShearBorder);
	
      } // end shearingBoxEnabled
      
      /*
       * magnetic field update
       */
      TIMER_START(timerCtUpdate);
      for (int k=ghostWidth; k<ksizeSlabStopUpdate; k++) {
	
	int kU = k+kStart;

	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	    
	    // update with EMFZ
	    if (kU<ksize-ghostWidth) {
	      h_UNew(i ,j ,kU, IA) += ( h_emf(i  ,j+1, k, I_EMFZ) - 
					h_emf(i,  j  , k, I_EMFZ) ) * dtdy;
	      
	      h_UNew(i ,j ,kU, IB) -= ( h_emf(i+1,j  , k, I_EMFZ) - 
					h_emf(i  ,j  , k, I_EMFZ) ) * dtdx;
	      
	    }
	    
	    // update BX
	    h_UNew(i ,j ,kU, IA) -= ( h_emf(i,j,k+1, I_EMFY) -
				      h_emf(i,j,k  , I_EMFY) ) * dtdz;
	    
	    // update BY
	    h_UNew(i ,j ,kU, IB) += ( h_emf(i,j,k+1, I_EMFX) -
				      h_emf(i,j,k  , I_EMFX) ) * dtdz;
	    
	    // update BZ
	    h_UNew(i ,j ,kU, IC) += ( h_emf(i+1,j  ,k, I_EMFY) -
				      h_emf(i  ,j  ,k, I_EMFY) ) * dtdx;
	    h_UNew(i ,j ,kU, IC) -= ( h_emf(i  ,j+1,k, I_EMFX) -
				      h_emf(i  ,j  ,k, I_EMFX) ) * dtdy;
	    
	  } // end for i
	} // end for j
      } // end for k     
      TIMER_STOP(timerCtUpdate);

    } // end zSlabId loop

    /*****************************************************************/
    /*****************************************************************/
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

	// update magnetic field with resistivity emf
	compute_resistivity_emf_3d(h_UOld, h_emf,     zSlabInfo);
	compute_ct_update_3d      (h_UNew, h_emf, dt, zSlabInfo);
	
	real_t &cIso = _gParams.cIso;
	if (cIso<=0) { // non-isothermal simulations

	  // compute energy flux
	  compute_resistivity_energy_flux_3d(h_UOld, h_qm_x, h_qm_y, h_qm_z, dt, zSlabInfo);
	  compute_hydro_update_energy       (h_UNew, h_qm_x, h_qm_y, h_qm_z,     zSlabInfo);

	} // end (cIso<=0)
      
      } // end for zSlabId

    } // end eta>0

    /*
     * compute viscosity forces
     */
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
    /*****************************************************************/
    /*****************************************************************/

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

  } // MHDRunGodunovZslab::godunov_unsplit_rotating_cpu
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  // IMPORTANT NOTICE : shearing border condition is only implemented 
  // for 3D problems and for XDIR direction
#ifdef __CUDACC__
  void MHDRunGodunovZslab::make_boundaries_shear(DeviceArray<real_t> &U, real_t dt, int nStep)
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

  } // MHDRunGodunovZslab::make_boundaries_shear
#else // CPU version
  void MHDRunGodunovZslab::make_boundaries_shear(HostArray<real_t> &U, real_t dt, int nStep)
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
    
  } //MHDRunGodunovZslab::make_boundaries_shear
#endif // __CUDACC__

#ifdef __CUDACC__
  void MHDRunGodunovZslab::make_all_boundaries_shear(DeviceArray<real_t> &U, 
						     real_t dt, 
						     int nStep)
  {

    // YDIR must be done first !
    make_boundaries(U,YDIR);
    
    make_boundaries_shear(U, dt, nStep);
    
    make_boundaries(U,ZDIR);
    
    make_boundaries(U,YDIR);

  } // MHDRunGodunovZslab::make_all_boundaries_shear -- GPU version
#else // CPU version
  void MHDRunGodunovZslab::make_all_boundaries_shear(HostArray<real_t> &U, 
						     real_t dt, 
						     int nStep)
  {

    // YDIR must be done first !
    make_boundaries(U,YDIR);

    make_boundaries_shear(U, dt, nStep);

    make_boundaries(U,ZDIR);

    make_boundaries(U,YDIR);
    
  } //MHDRunGodunovZslab::make_all_boundaries_shear -- CPU version
#endif // __CUDACC__


  // =======================================================
  // =======================================================
  /*
   * main routine to start simulation.
   */
  void MHDRunGodunovZslab::start() 
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

	  timerWriteOnDisk.stop();
	
	  std::cout << "["       << current_date()       << "]"
		    << "  step=" << std::setw(9)         << nStep 
		    << " t="     << fmt(totalTime)
		    << " dt="    << fmt(dt,16,12)        << std::endl;


	} // end if nStep%nOutput == 0

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
    
  } // MHDRunGodunovZslab::start

  // =======================================================
  // =======================================================
  /*
   * do one time step integration
   */
  void MHDRunGodunovZslab::oneStepIntegration(int& nStep, real_t& t, real_t& dt) 
  {
  
    // if nStep is even update U  into U2
    // if nStep is odd  update U2 into U
    dt=compute_dt_mhd(nStep % 2);
    godunov_unsplit(nStep, dt);
  
    // increment time
    nStep++;
    t+=dt;
  
  } // MHDRunGodunovZslab::oneStepIntegration

} // namespace hydroSimu
