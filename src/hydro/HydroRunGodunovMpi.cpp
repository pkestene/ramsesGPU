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
 * \file HydroRunGodunovMpi.cpp
 * \brief Implements class HydroRunGodunovMpi
 * 
 * 2D/3D Euler equation solver on a cartesian grid using Godunov method
 * with approximate Riemann solver.
 *
 * CUDA and MPI implementation.
 *
 * \date 19 Oct 2010
 * \author Pierre Kestener
 *
 * $Id: HydroRunGodunovMpi.cpp 3450 2014-06-16 22:03:23Z pkestene $
 */
#include "HydroRunGodunovMpi.h"

// include CUDA kernel when necessary
#ifdef __CUDACC__
//#include "cmpdt.cuh"
#include "godunov_notrace.cuh"
#include "godunov_trace_v1.cuh"
#include "godunov_trace_v2.cuh"
#include "godunov_unsplit.cuh"
#endif // __CUDACC__
#include "constoprim.h"
#include "riemann.h"
#include "trace.h"

#include <sys/time.h> // for gettimeofday

#include "../utils/monitoring/Timer.h"
#include "../utils/monitoring/date.h"
#include <iomanip>
#include "ostream_fmt.h"

// OpenMP support
#if _OPENMP
# include <omp.h>
#endif

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // HydroRunGodunovMpi class methods body
  ////////////////////////////////////////////////////////////////////////////////

  HydroRunGodunovMpi::HydroRunGodunovMpi(ConfigMap &_configMap)
    : HydroRunBaseMpi(_configMap)
    ,traceEnabled(true)
    ,traceVersion(1)
    ,unsplitEnabled(true)
    ,unsplitVersion(1)
  {

    // chose between split and unsplit Godunov integration
    unsplitEnabled = configMap.getBool("hydro","unsplit", true);

    // choose unsplit implementation version
    if (unsplitEnabled) {
      unsplitVersion = configMap.getInteger("hydro","unsplitVersion", 1);
      if (unsplitVersion !=0 and unsplitVersion !=1 and unsplitVersion !=2)
	{
	  if (myRank == 0) {
	    std::cerr << "##################################################" << std::endl;
	    std::cerr << "WARNING : you should review your parameter file   " << std::endl;
	    std::cerr << "and set hydro/unsplitVersion to a valid number :  " << std::endl;
	    std::cerr << " - 0, 1 and 2 are currently available for 2D/3D   " << std::endl;
	    std::cerr << "Fall back to the default value : 1                " << std::endl;
	    std::cerr << "##################################################" << std::endl;
	  }
	  unsplitVersion = 1;
	}
      std::cout << "Using Hydro Godunov unsplit implementation version : " << 
	unsplitVersion << std::endl;
      
      /*
       * allways allocate primitive variables array : h_Q / d_Q
       */
#ifdef __CUDACC__

      if (dimType == TWO_D) {
	d_Q.allocate   (make_uint3(isize, jsize, nbVar), gpuMemAllocType);
      } else { // THREE_D
	d_Q.allocate   (make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
      }
      // register data pointers
      _gParams.arrayList[A_Q]    = d_Q.data();
    
#else
    
      if (dimType == TWO_D) {
	h_Q.allocate   (make_uint3(isize, jsize, nbVar));
      } else { // THREE_D
	h_Q.allocate   (make_uint4(isize, jsize, ksize, nbVar));
      }
      // register data pointers
      _gParams.arrayList[A_Q]    = h_Q.data();

#endif

      /*
       * memory allocation specific to a given implementation version
       */
      if (unsplitVersion == 1) {

	// do memory allocation for extra array
#ifdef __CUDACC__
	if (dimType == TWO_D) {

	  d_qm_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	  d_qm_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	  d_qp_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	  d_qp_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);

	  // register data pointers
	  _gParams.arrayList[A_QM_X] = d_qm_x.data();
	  _gParams.arrayList[A_QM_Y] = d_qm_y.data();
	  _gParams.arrayList[A_QP_X] = d_qp_x.data();
	  _gParams.arrayList[A_QP_Y] = d_qp_y.data();

	} else { // THREE_D

	  d_qm_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qm_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qm_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qp_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qp_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qp_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);

	  // register data pointers
	  _gParams.arrayList[A_QM_X] = d_qm_x.data();
	  _gParams.arrayList[A_QM_Y] = d_qm_y.data();
	  _gParams.arrayList[A_QM_Z] = d_qm_z.data();
	  _gParams.arrayList[A_QP_X] = d_qp_x.data();
	  _gParams.arrayList[A_QP_Y] = d_qp_y.data();
	  _gParams.arrayList[A_QP_Z] = d_qp_z.data();

	}
#else // CPU version
	if (dimType == TWO_D) {

	  h_qm_x.allocate(make_uint3(isize, jsize, nbVar));
	  h_qm_y.allocate(make_uint3(isize, jsize, nbVar));
	  h_qp_x.allocate(make_uint3(isize, jsize, nbVar));
	  h_qp_y.allocate(make_uint3(isize, jsize, nbVar));

	  // register data pointers
	  _gParams.arrayList[A_QM_X] = h_qm_x.data();
	  _gParams.arrayList[A_QM_Y] = h_qm_y.data();
	  _gParams.arrayList[A_QP_X] = h_qp_x.data();
	  _gParams.arrayList[A_QP_Y] = h_qp_y.data();

	} else { // THREE_D

	  h_qm_x.allocate(make_uint4(isize, jsize, ksize, nbVar));
	  h_qm_y.allocate(make_uint4(isize, jsize, ksize, nbVar));
	  h_qm_z.allocate(make_uint4(isize, jsize, ksize, nbVar));
	  h_qp_x.allocate(make_uint4(isize, jsize, ksize, nbVar));
	  h_qp_y.allocate(make_uint4(isize, jsize, ksize, nbVar));
	  h_qp_z.allocate(make_uint4(isize, jsize, ksize, nbVar));

	  // register data pointers
	  _gParams.arrayList[A_QM_X] = h_qm_x.data();
	  _gParams.arrayList[A_QM_Y] = h_qm_y.data();
	  _gParams.arrayList[A_QM_Z] = h_qm_z.data();
	  _gParams.arrayList[A_QP_X] = h_qp_x.data();
	  _gParams.arrayList[A_QP_Y] = h_qp_y.data();
	  _gParams.arrayList[A_QP_Z] = h_qp_z.data();

	} // end THREE_D
#endif // __CUDACC__

      } else if (unsplitVersion == 2) {

	// do memory allocation for extra array
#ifdef __CUDACC__
	if (dimType == TWO_D) {

	  d_slope_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	  d_slope_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	  d_qm.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
	  d_qp.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);

	  // register data pointers
	  _gParams.arrayList[A_SLOPE_X] = d_slope_x.data();
	  _gParams.arrayList[A_SLOPE_Y] = d_slope_y.data();
	  _gParams.arrayList[A_QM]      = d_qm.data();
	  _gParams.arrayList[A_QP]      = d_qp.data();

	} else { // THREE_D

	  d_slope_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_slope_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_slope_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qm.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
	  d_qp.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);

	  // register data pointers
	  _gParams.arrayList[A_SLOPE_X] = d_slope_x.data();
	  _gParams.arrayList[A_SLOPE_Y] = d_slope_y.data();
	  _gParams.arrayList[A_SLOPE_Z] = d_slope_z.data();
	  _gParams.arrayList[A_QM]      = d_qm.data();
	  _gParams.arrayList[A_QP]      = d_qp.data();

	} // end THREE_D
#else
	if (dimType == TWO_D) {

	  h_slope_x.allocate(make_uint3(isize, jsize, nbVar));
	  h_slope_y.allocate(make_uint3(isize, jsize, nbVar));

	  h_qm.allocate(make_uint3(isize, jsize, nbVar));
	  h_qp.allocate(make_uint3(isize, jsize, nbVar));

	  // register data pointers
	  _gParams.arrayList[A_SLOPE_X] = h_slope_x.data();
	  _gParams.arrayList[A_SLOPE_Y] = h_slope_y.data();
	  _gParams.arrayList[A_QM]      = h_qm.data();
	  _gParams.arrayList[A_QP]      = h_qp.data();

	} else { // THREE_D

	  h_slope_x.allocate(make_uint4(isize, jsize, ksize, nbVar));
	  h_slope_y.allocate(make_uint4(isize, jsize, ksize, nbVar));
	  h_slope_z.allocate(make_uint4(isize, jsize, ksize, nbVar));

	  h_qm.allocate(make_uint4(isize, jsize, ksize, nbVar));
	  h_qp.allocate(make_uint4(isize, jsize, ksize, nbVar));

	  // register data pointers
	  _gParams.arrayList[A_SLOPE_X] = h_slope_x.data();
	  _gParams.arrayList[A_SLOPE_Y] = h_slope_y.data();
	  _gParams.arrayList[A_SLOPE_Z] = h_slope_z.data();
	  _gParams.arrayList[A_QM]      = h_qm.data();
	  _gParams.arrayList[A_QP]      = h_qp.data();

	} // end THREE_D
#endif // __CUDACC__

      } // end unsplitVersion == 2

    } // end unsplitEnabled

    // make sure variable declared as __constant__ are copied to device
    // for current compilation unit
    copyToSymbolMemory();

    if (myRank==0) {
#ifdef __CUDACC__
      printf("_gParams.arrayList[A_GRAV] = %p\n",_gParams.arrayList[A_GRAV]);
#endif // __CUDACC__
    }

    /*
     * Total memory allocated logging.
     * Just for notice
     */
    if (myRank==0)
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

  } // HydroRunGodunovMpi::HydroRunGodunovMpi

  // =======================================================
  // =======================================================
  HydroRunGodunovMpi::~HydroRunGodunovMpi()
  {  

  } // HydroRunGodunovMpi::~HydroRunGodunovMpi


  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_split(int nStep, real_t dt)
#ifdef __CUDACC__
  {
    if (dimType == TWO_D) {
    
      // one step integration results are always in d_U
      if ((nStep%2)==0) {
	godunov_split_gpu(d_U , d_U2, XDIR, dt);
	godunov_split_gpu(d_U2, d_U , YDIR, dt);
      } else {
	godunov_split_gpu(d_U , d_U2, YDIR, dt);
	godunov_split_gpu(d_U2, d_U , XDIR, dt);
      }

    } else { // THREE_D

      // we check nStep % 6 because, we rotate the 1d operator

      if ((nStep%6)==0) {
	godunov_split_gpu(d_U , d_U2, XDIR, dt);
	godunov_split_gpu(d_U2, d_U , YDIR, dt);
	godunov_split_gpu(d_U , d_U2, ZDIR, dt);
      } else if ((nStep%6)==1) {
	godunov_split_gpu(d_U2, d_U , YDIR,dt);
	godunov_split_gpu(d_U , d_U2, ZDIR,dt);
	godunov_split_gpu(d_U2, d_U , XDIR,dt);
      } else if ((nStep%6)==2) {
	godunov_split_gpu(d_U , d_U2, ZDIR,dt);
	godunov_split_gpu(d_U2, d_U , YDIR,dt);
	godunov_split_gpu(d_U , d_U2, XDIR,dt);
      } else if ((nStep%6)==3) {
	godunov_split_gpu(d_U2, d_U , XDIR, dt);
	godunov_split_gpu(d_U , d_U2, YDIR, dt);
	godunov_split_gpu(d_U2, d_U , ZDIR, dt);
      } else if ((nStep%6)==4) {
	godunov_split_gpu(d_U , d_U2, YDIR,dt);
	godunov_split_gpu(d_U2, d_U , ZDIR,dt);
	godunov_split_gpu(d_U , d_U2, XDIR,dt);
      } else if ((nStep%6)==5) {
	godunov_split_gpu(d_U2, d_U , ZDIR,dt);
	godunov_split_gpu(d_U , d_U2, YDIR,dt);
	godunov_split_gpu(d_U2, d_U , XDIR,dt);
      }

    } // end THREE_D

  } // HydroRunGodunovMpi::godunov_split (GPU version)
#else // CPU version
  {

  if (dimType == TWO_D) {
      
      // one step integration results are always in h_U
      if ((nStep%2)==0) {
	godunov_split_cpu(h_U , h_U2, XDIR, dt);
	godunov_split_cpu(h_U2, h_U , YDIR, dt);
      } else {
	godunov_split_cpu(h_U , h_U2, YDIR, dt);
	godunov_split_cpu(h_U2, h_U , XDIR, dt);
      }
      
    } else { // THREE_D
      
      // we check nStep % 6 because, we rotate the 1d operator
      
      if ((nStep%6)==0) {
	godunov_split_cpu(h_U , h_U2, XDIR, dt);
	godunov_split_cpu(h_U2, h_U , YDIR, dt);
	godunov_split_cpu(h_U , h_U2, ZDIR, dt);
      } else if ((nStep%6)==1) {
	godunov_split_cpu(h_U2, h_U , YDIR,dt);
	godunov_split_cpu(h_U , h_U2, ZDIR,dt);
	godunov_split_cpu(h_U2, h_U , XDIR,dt);
      } else if ((nStep%6)==2) {
	godunov_split_cpu(h_U , h_U2, ZDIR,dt);
	godunov_split_cpu(h_U2, h_U , YDIR,dt);
	godunov_split_cpu(h_U , h_U2, XDIR,dt);
      } else if ((nStep%6)==3) {
	godunov_split_cpu(h_U2, h_U , XDIR, dt);
	godunov_split_cpu(h_U , h_U2, YDIR, dt);
	godunov_split_cpu(h_U2, h_U , ZDIR, dt);
      } else if ((nStep%6)==4) {
	godunov_split_cpu(h_U , h_U2, YDIR,dt);
	godunov_split_cpu(h_U2, h_U , ZDIR,dt);
	godunov_split_cpu(h_U , h_U2, XDIR,dt);
      } else if ((nStep%6)==5) {
	godunov_split_cpu(h_U2, h_U , ZDIR,dt);
	godunov_split_cpu(h_U , h_U2, YDIR,dt);
	godunov_split_cpu(h_U2, h_U , XDIR,dt);
      }
      
    } // end THREE_D
    
  } // HydroRunGodunovMpi::godunov_split (CPU version)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_unsplit(int nStep, real_t dt)
#ifdef __CUDACC__
  {

    if ((nStep%2)==0) {
      godunov_unsplit_gpu(d_U , d_U2, dt);
    } else {
      godunov_unsplit_gpu(d_U2, d_U , dt);
    }

  } // HydroRunGodunovMpi::godunov_unsplit (GPU version)
#else // CPU version
 {
   
   if ((nStep%2)==0) {
     godunov_unsplit_cpu(h_U , h_U2, dt);
   } else {
     godunov_unsplit_cpu(h_U2, h_U , dt);
   }
   
 } // HydroRunGodunovMpi::godunov_unsplit (CPU version)
#endif // __CUDACC__

#ifdef __CUDACC__ 
  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_split_gpu(DeviceArray<real_t>& d_UOld, 
					     DeviceArray<real_t>& d_UNew,
					     int idim, 
					     real_t dt)
  {

    make_boundaries(d_UOld,idim);
    communicator->synchronize();

    TIMER_START(timerGodunov);
    if (dimType == TWO_D) {

      // launch 2D kernel
      if (idim == XDIR)
	{
	  if (traceVersion == 2) {
	    dim3 dimBlock(T2_HBLOCK_DIMX, 
			  T2_HBLOCK_DIMY);
	    dim3 dimGrid(blocksFor(isize, T2_HBLOCK_INNER_DIMX), 
			 blocksFor(jsize, T2_HBLOCK_DIMY));
	    godunov_x_2d_v2<<<dimGrid, dimBlock>>>(d_UOld.data(), 
						   d_UNew.data(),
						   d_UOld.pitch(), 
						   d_UOld.dimx(), 
						   d_UOld.dimy(),
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_x_2d_v2",myRank);
	  } else if (traceVersion == 1) {
	    dim3 dimBlock(T1_HBLOCK_DIMX, 
			  T1_HBLOCK_DIMY);
	    dim3 dimGrid(blocksFor(isize, T1_HBLOCK_INNER_DIMX), 
			 blocksFor(jsize, T1_HBLOCK_DIMY));
	    godunov_x_2d_v1<<<dimGrid, dimBlock>>>(d_UOld.data(), 
						   d_UNew.data(),
						   d_UOld.pitch(), 
						   d_UOld.dimx(), 
						   d_UOld.dimy(),
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_x_2d_v1",myRank);
	  } else { // no trace
	    dim3 dimBlock(HBLOCK_DIMX, 
			  HBLOCK_DIMY);
	    dim3 dimGrid(blocksFor(isize, HBLOCK_INNER_DIMX), 
			 blocksFor(jsize, HBLOCK_DIMY));
	    godunov_x_notrace_2d<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							d_UNew.data(),
							d_UOld.pitch(), 
							d_UOld.dimx(), 
							d_UOld.dimy(), 
							dt / dx, dt);
	    checkCudaErrorMpi("godunov_x_notrace_2d",myRank);
	  }
	}
      else
	{
	  if (traceVersion == 2) {
	    dim3 dimBlock(T2_VBLOCK_DIMX, 
			  T2_VBLOCK_DIMY);
	    dim3 dimGrid(blocksFor(isize, T2_VBLOCK_DIMX), 
			 blocksFor(jsize, T2_VBLOCK_INNER_DIMY));
	    godunov_y_2d_v2<<<dimGrid, dimBlock>>>(d_UOld.data(), 
						   d_UNew.data(),
						   d_UOld.pitch(), 
						   d_UOld.dimx(), 
						   d_UOld.dimy(),
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_y_2d_v2",myRank);
	  } else if (traceVersion == 1) {
	    dim3 dimBlock(T1_VBLOCK_DIMX, 
			  T1_VBLOCK_DIMY);
	    dim3 dimGrid(blocksFor(isize, T1_VBLOCK_DIMX), 
			 blocksFor(jsize, T1_VBLOCK_INNER_DIMY));
	    godunov_y_2d_v1<<<dimGrid, dimBlock>>>(d_UOld.data(), 
						   d_UNew.data(),
						   d_UOld.pitch(), 
						   d_UOld.dimx(), 
						   d_UOld.dimy(),
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_y_2d_v1",myRank);
	  } else {
	    dim3 dimBlock(VBLOCK_DIMX, 
			  VBLOCK_DIMY);
	    dim3 dimGrid(blocksFor(isize, VBLOCK_DIMX), 
			 blocksFor(jsize, VBLOCK_INNER_DIMY));
	    godunov_y_notrace_2d<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							d_UNew.data(),
							d_UOld.pitch(), 
							d_UOld.dimx(), 
							d_UOld.dimy(), 
							dt / dx, dt);
	    checkCudaErrorMpi("godunov_y_notrace_2d",myRank);
	  }
	}

    } else { // THREE_D

      // launch 3D kernel
      if (idim == XDIR)
	{
	  if (traceVersion == 2) {
	    dim3 dimBlock(T2_XDIR_BLOCK_DIMX_3D, 
			  T2_XDIR_BLOCK_DIMY_3D);
	    dim3 dimGrid(blocksFor(isize, T2_XDIR_BLOCK_INNER_DIMX_3D), 
			 blocksFor(jsize, T2_XDIR_BLOCK_DIMY_3D));
	    godunov_x_3d_v2<<<dimGrid, dimBlock>>>(d_UOld.data(), 
						   d_UNew.data(), 
						   d_UOld.pitch(), 
						   d_UOld.dimx(), 
						   d_UOld.dimy(), 
						   d_UOld.dimz(), 
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_x_3d_v2",myRank);
	  } else if (traceVersion == 1) {
	    dim3 dimBlock(T1_XDIR_BLOCK_DIMX_3D, 
			  T1_XDIR_BLOCK_DIMY_3D);
	    dim3 dimGrid(blocksFor(isize, T1_XDIR_BLOCK_INNER_DIMX_3D), 
			 blocksFor(jsize, T1_XDIR_BLOCK_DIMY_3D));
	    godunov_x_3d_v1<<<dimGrid, dimBlock>>>(d_UOld.data(), 
						   d_UNew.data(), 
						   d_UOld.pitch(), 
						   d_UOld.dimx(), 
						   d_UOld.dimy(), 
						   d_UOld.dimz(), 
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_x_3d_v1",myRank);
	  } else {
	    dim3 dimBlock(XDIR_BLOCK_DIMX_3D, 
			  XDIR_BLOCK_DIMY_3D);
	    dim3 dimGrid(blocksFor(isize, XDIR_BLOCK_INNER_DIMX_3D), 
			 blocksFor(jsize, XDIR_BLOCK_DIMY_3D));
	    godunov_x_notrace_3d<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							d_UNew.data(), 
							d_UOld.pitch(), 
							d_UOld.dimx(), 
							d_UOld.dimy(), 
							d_UOld.dimz(), 
							dt / dx, dt);
	    checkCudaErrorMpi("godunov_x_notrace_3d",myRank);
	  }
	}
      else if (idim == YDIR)
	{
	  if (traceVersion == 2) {
	    dim3 dimBlock(T2_YDIR_BLOCK_DIMX_3D, 
			  T2_YDIR_BLOCK_DIMY_3D);
	    dim3 dimGrid(blocksFor(isize, T2_YDIR_BLOCK_DIMX_3D), 
			 blocksFor(jsize, T2_YDIR_BLOCK_INNER_DIMY_3D));
	    godunov_y_3d_v2<<<dimGrid, dimBlock>>>(d_UOld.data(), 
						   d_UNew.data(), 
						   d_UOld.pitch(), 
						   d_UOld.dimx(), 
						   d_UOld.dimy(),
						   d_UOld.dimz(),
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_y_3d_v2",myRank);
	  } else if (traceVersion == 1) {
	    dim3 dimBlock(T1_YDIR_BLOCK_DIMX_3D, 
			  T1_YDIR_BLOCK_DIMY_3D);
	    dim3 dimGrid(blocksFor(isize, T1_YDIR_BLOCK_DIMX_3D), 
			 blocksFor(jsize, T1_YDIR_BLOCK_INNER_DIMY_3D));
	    godunov_y_3d_v1<<<dimGrid, dimBlock>>>(d_UOld.data(),
						   d_UNew.data(),
						   d_UOld.pitch(),
						   d_UOld.dimx(),
						   d_UOld.dimy(),
						   d_UOld.dimz(),
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_y_3d_v1",myRank);
	  } else {
	    dim3 dimBlock(YDIR_BLOCK_DIMX_3D, 
			  YDIR_BLOCK_DIMY_3D);
	    dim3 dimGrid(blocksFor(isize, YDIR_BLOCK_DIMX_3D), 
			 blocksFor(jsize, YDIR_BLOCK_INNER_DIMY_3D));
	    godunov_y_notrace_3d<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							d_UNew.data(), 
							d_UOld.pitch(),
							d_UOld.dimx(), 
							d_UOld.dimy(),
							d_UOld.dimz(),
							dt / dx, dt);
	    checkCudaErrorMpi("godunov_y_notrace_3d",myRank);
	  }
	}
      else // idim == ZDIR
	{
	  if (traceVersion == 2) {
	    dim3 dimBlock(T2_ZDIR_BLOCK_DIMX_3D, 
			  T2_ZDIR_BLOCK_DIMZ_3D);
	    dim3 dimGrid(blocksFor(isize, T2_ZDIR_BLOCK_DIMX_3D), 
			 blocksFor(ksize, T2_ZDIR_BLOCK_INNER_DIMZ_3D));
	    godunov_z_3d_v2<<<dimGrid, dimBlock>>>(d_UOld.data(), 
						   d_UNew.data(), 
						   d_UOld.pitch(),
						   d_UOld.dimx(), 
						   d_UOld.dimy(), 
						   d_UOld.dimz(),
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_z_3d_v2",myRank);
	  } else if (traceVersion == 1) {
	    dim3 dimBlock(T1_ZDIR_BLOCK_DIMX_3D, 
			  T1_ZDIR_BLOCK_DIMZ_3D);
	    dim3 dimGrid(blocksFor(isize, T1_ZDIR_BLOCK_DIMX_3D), 
			 blocksFor(ksize, T1_ZDIR_BLOCK_INNER_DIMZ_3D));
	    godunov_z_3d_v1<<<dimGrid, dimBlock>>>(d_UOld.data(),
						   d_UNew.data(),
						   d_UOld.pitch(),
						   d_UOld.dimx(),
						   d_UOld.dimy(),
						   d_UOld.dimz(), 
						   dt / dx, dt);
	    checkCudaErrorMpi("godunov_z_3d_v1",myRank);
	  } else {
	    dim3 dimBlock(ZDIR_BLOCK_DIMX_3D, 
			  ZDIR_BLOCK_DIMZ_3D);
	    dim3 dimGrid(blocksFor(isize, ZDIR_BLOCK_DIMX_3D), 
			 blocksFor(ksize, ZDIR_BLOCK_INNER_DIMZ_3D));
	    godunov_z_notrace_3d<<<dimGrid, dimBlock>>>(d_UOld.data(), 
							d_UNew.data(), 
							d_UOld.pitch(),
							d_UOld.dimx(), 
							d_UOld.dimy(),
							d_UOld.dimz(), 
							dt / dx, dt);
	    checkCudaErrorMpi("godunov_z_notrace_3d",myRank);
	  }
	}

    } // end THREE_D
    TIMER_STOP(timerGodunov);
  
  } // HydroRunGodunovMpi::godunov_split_gpu

  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
					       DeviceArray<real_t>& d_UNew,
					       real_t dt)
  {
    
    // update boundaries
    make_all_boundaries(d_UOld);
    
    // easier for time integration later
    d_UOld.copyTo(d_UNew);

    // inner domain integration
    TIMER_START(timerGodunov);
    
    /*
     * Whatever implementation version, start by computing primitive variables
     *   
     * convert conservative to primitive variables (and source term predictor)
     * put results in h_Q object
     *
     */
    convertToPrimitives( d_UOld.data() );
  
    if (unsplitVersion == 0) {
  
      godunov_unsplit_gpu_v0(d_UOld, d_UNew, dt);
  
    } else if (unsplitVersion == 1) {

      godunov_unsplit_gpu_v1(d_UOld, d_UNew, dt);
    
    } else if (unsplitVersion == 2) {
    
      godunov_unsplit_gpu_v2(d_UOld, d_UNew, dt);

    } // end unsplitVersion == 2
       
    TIMER_STOP(timerGodunov);

  } // HydroRunGodunovMpi::godunov_unsplit_gpu


  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
						  DeviceArray<real_t>& d_UNew,
						  real_t dt)
  {

    if (dimType == TWO_D) {
    
      // In this version primitive variables are computed as needed inside
      // kernel
      //
      // // 2D Godunov unsplit kernel
      // dim3 dimBlock(UNSPLIT_BLOCK_DIMX_2D,
      // 		    UNSPLIT_BLOCK_DIMY_2D);
      // dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_INNER_DIMX_2D), 
      // 		   blocksFor(jsize, UNSPLIT_BLOCK_INNER_DIMY_2D));
      // kernel_godunov_unsplit_2d<<<dimGrid, dimBlock>>>(d_UOld.data(), 
      // 						       d_UNew.data(),
      // 						       d_UOld.pitch(), 
      // 						       d_UOld.dimx(), 
      // 						       d_UOld.dimy(), 
      // 						       dt / dx, dt,
      // 						       gravityEnabled);
      // checkCudaError("HydroRunGodunov :: kernel_godunov_unsplit_2d error");
    
      {
	dim3 dimBlock(UNSPLIT_BLOCK_DIMX_2D_V0,
		      UNSPLIT_BLOCK_DIMY_2D_V0);
	dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_INNER_DIMX_2D_V0), 
		     blocksFor(jsize, UNSPLIT_BLOCK_INNER_DIMY_2D_V0));
	kernel_godunov_unsplit_2d_v0<<<dimGrid,
	  dimBlock>>>(d_UNew.data(),
		      d_Q.data(),
		      d_UNew.pitch(),
		      d_UNew.dimx(),
		      d_UNew.dimy(),
		      dt / dx, 
		      dt / dy,
		      dt,
		      gravityEnabled);
	checkCudaErrorMpi("HydroRunGodunov :: kernel_godunov_unsplit_2d_v0 error",myRank);
      
      } // end compute unsplit v0
    
    } else if (dimType == THREE_D) {
    
      // // 3D Godunov unsplit kernel    
      // dim3 dimBlock(UNSPLIT_BLOCK_DIMX_3D,
      // 		  UNSPLIT_BLOCK_DIMY_3D);
      // dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_INNER_DIMX_3D), 
      // 		 blocksFor(jsize, UNSPLIT_BLOCK_INNER_DIMY_3D));
      // kernel_godunov_unsplit_3d<<<dimGrid, dimBlock>>>(d_UOld.data(), 
      // 						     d_UNew.data(), 
      // 						     d_UOld.pitch(), 
      // 						     d_UOld.dimx(), 
      // 						     d_UOld.dimy(), 
      // 						     d_UOld.dimz(),
      // 						     dt / dx, dt,
      // 						     gravityEnabled);
      // checkCudaError("HydroRunGodunov :: kernel_godunov_unsplit_3d error");

      // 3D Godunov unsplit kernel    
      dim3 dimBlock(UNSPLIT_BLOCK_DIMX_3D_V0,
		    UNSPLIT_BLOCK_DIMY_3D_V0);
      dim3 dimGrid(blocksFor(isize, UNSPLIT_BLOCK_INNER_DIMX_3D_V0),
		   blocksFor(jsize, UNSPLIT_BLOCK_INNER_DIMY_3D_V0));
      kernel_godunov_unsplit_3d_v0<<<dimGrid, dimBlock>>>(d_Q.data(), 
							  d_UNew.data(), 
							  d_UOld.pitch(), 
							  d_UOld.dimx(), 
							  d_UOld.dimy(), 
							  d_UOld.dimz(),
							  dt / dx, 
							  dt / dy, 
							  dt / dz, 
							  dt,
							  gravityEnabled);
      checkCudaErrorMpi("HydroRunGodunov :: kernel_godunov_unsplit_3d_v0 error",myRank);

    } // end THREE_D

    // gravity source term computation
    if (gravityEnabled) {
      compute_gravity_source_term(d_UNew, d_UOld, dt);
    }

  } // HydroRunGodunovMpi::godunov_unsplit_gpu_v0

  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
						  DeviceArray<real_t>& d_UNew,
						  real_t dt)
  {

    if (dimType == TWO_D) {

      TIMER_START(timerSlopeTrace);
      {
	// 2D slope / trace computation kernel
	dim3 dimBlock(TRACE_BLOCK_DIMX_2D_V1,
		      TRACE_BLOCK_DIMY_2D_V1);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_INNER_DIMX_2D_V1), 
		     blocksFor(jsize, TRACE_BLOCK_INNER_DIMY_2D_V1));
	kernel_hydro_compute_trace_unsplit_2d_v1<<<dimGrid, 
	  dimBlock>>>(d_UOld.data(),
		      d_Q.data(),
		      d_qm_x.data(),
		      d_qm_y.data(),
		      d_qp_x.data(),
		      d_qp_y.data(),
		      d_UOld.pitch(), 
		      d_UOld.dimx(), 
		      d_UOld.dimy(), 
		      dt / dx, 
		      dt / dy,
		      dt);
	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_compute_trace_unsplit_2d_v1 error",myRank);
      
	if (gravityEnabled) {
	  compute_gravity_predictor(d_qm_x, dt);
	  compute_gravity_predictor(d_qm_y, dt);
	  compute_gravity_predictor(d_qp_x, dt);
	  compute_gravity_predictor(d_qp_y, dt);
	}
      
      } // end 2D slope / trace computation kernel
      TIMER_STOP(timerSlopeTrace);
    
      TIMER_START(timerUpdate);
      {
	// 2D update hydro kernel
	dim3 dimBlock(UPDATE_BLOCK_DIMX_2D_V1,
		      UPDATE_BLOCK_DIMY_2D_V1);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_2D_V1), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_2D_V1));
	kernel_hydro_flux_update_unsplit_2d_v1<<<dimGrid, 
	  dimBlock>>>(d_UOld.data(),
		      d_UNew.data(),
		      d_qm_x.data(),
		      d_qm_y.data(),
		      d_qp_x.data(),
		      d_qp_y.data(),
		      d_UOld.pitch(), 
		      d_UOld.dimx(), 
		      d_UOld.dimy(), 
		      dt / dx, 
		      dt / dy,
		      dt );
	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_flux_update_unsplit_2d_v1< error",myRank);
      } // end 2D update hydro kernel
    
      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt);
      }
      TIMER_STOP(timerUpdate);
    
      /*
       * DISSIPATIVE TERMS (i.e. viscosity)
       */
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
      
	compute_viscosity_flux(d_UNew, d_flux_x, d_flux_y, dt);
	compute_hydro_update  (d_UNew, d_flux_x, d_flux_y);
      } // end compute viscosity force / update  
      TIMER_STOP(timerDissipative);

    } else if (dimType == THREE_D) {

      TIMER_START(timerSlopeTrace);
      {
	// 3D slope / trace computation kernel
	dim3 dimBlock(TRACE_BLOCK_DIMX_3D_V1,
		      TRACE_BLOCK_DIMY_3D_V1);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_INNER_DIMX_3D_V1), 
		     blocksFor(jsize, TRACE_BLOCK_INNER_DIMY_3D_V1));
	kernel_hydro_compute_trace_unsplit_3d_v1<<<dimGrid, 
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
		      dt);
	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_compute_trace_unsplit_3d_v1 error",myRank);
      
	if (gravityEnabled) {
	  compute_gravity_predictor(d_qm_x, dt);
	  compute_gravity_predictor(d_qm_y, dt);
	  compute_gravity_predictor(d_qm_z, dt);
	  compute_gravity_predictor(d_qp_x, dt);
	  compute_gravity_predictor(d_qp_y, dt);
	  compute_gravity_predictor(d_qp_z, dt);
	}
      
      } // end 3D slope / trace computation kernel
      TIMER_STOP(timerSlopeTrace);
    
      TIMER_START(timerUpdate);
      {
	// 3D update hydro kernel
	dim3 dimBlock(UPDATE_BLOCK_DIMX_3D_V1,
		      UPDATE_BLOCK_DIMY_3D_V1);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_3D_V1), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_3D_V1));
	kernel_hydro_flux_update_unsplit_3d_v1<<<dimGrid, 
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
		      dt );
	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_flux_update_unsplit_3d_v1 error",myRank);
      
      } // end 3D update hydro kernel
    
      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt);
      }
    
      TIMER_STOP(timerUpdate);
    
      /*
       * DISSIPATIVE TERMS (i.e. viscosity)
       */
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
    
    } // end THREE_D

  } // HydroRunGodunovMpi::godunov_unsplit_gpu_v1

  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
						  DeviceArray<real_t>& d_UNew,
						  real_t dt)
  {

    if (dimType == TWO_D) {

      /*
       * 1. Compute and store slopes
       */
      {
	// 2D slopes
	dim3 dimBlock(SLOPES_BLOCK_DIMX_2D_V2,
		      SLOPES_BLOCK_DIMY_2D_V2);
	dim3 dimGrid(blocksFor(isize, SLOPES_BLOCK_INNER_DIMX_2D_V2), 
		     blocksFor(jsize, SLOPES_BLOCK_INNER_DIMY_2D_V2));
	kernel_godunov_slopes_2d_v2<<<dimGrid,
	  dimBlock>>>(d_Q.data(),
		      d_slope_x.data(),
		      d_slope_y.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy()		    
		      );
      } // end slopes 2D

      /*
       * 2. compute reconstructed states along X interfaces
       */
      {
	dim3 dimBlock(TRACE_BLOCK_DIMX_2D_V2,
		      TRACE_BLOCK_DIMY_2D_V2);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_DIMX_2D_V2), 
		     blocksFor(jsize, TRACE_BLOCK_DIMY_2D_V2));
	kernel_godunov_trace_by_dir_2d_v2<<<dimGrid,
	  dimBlock>>>(d_Q.data(),
		      d_slope_x.data(),
		      d_slope_y.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      dt, dt / dx, dt / dy,
		      gravityEnabled,
		      IX
		      );
      } // end trace X

      /*
       * 3. Riemann solver at X interface and update
       */
      {
	dim3 dimBlock(UPDATE_BLOCK_DIMX_2D_V2,
		      UPDATE_BLOCK_DIMY_2D_V2);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_2D_V2), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_2D_V2));
	kernel_hydro_flux_update_unsplit_2d_v2<<<dimGrid,
	  dimBlock>>>(d_UNew.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      dt / dx, dt / dy, dt,
		      IX
		      );
      
      } // end update X

      /*
       * 4. Compute reconstructed states along Y interfaces
       */
      {
	dim3 dimBlock(TRACE_BLOCK_DIMX_2D_V2,
		      TRACE_BLOCK_DIMY_2D_V2);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_DIMX_2D_V2), 
		     blocksFor(jsize, TRACE_BLOCK_DIMY_2D_V2));
	kernel_godunov_trace_by_dir_2d_v2<<<dimGrid,
	  dimBlock>>>(d_Q.data(),
		      d_slope_x.data(),
		      d_slope_y.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      dt, dt / dx, dt / dy,
		      gravityEnabled,
		      IY
		      );
      } // end trace Y

      /*
       * 5. Riemann solver at Y interface and update
       */
      {
	dim3 dimBlock(UPDATE_BLOCK_DIMX_2D_V2,
		      UPDATE_BLOCK_DIMY_2D_V2);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_2D_V2), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_2D_V2));
	kernel_hydro_flux_update_unsplit_2d_v2<<<dimGrid,
	  dimBlock>>>(d_UNew.data(),
		      d_qm.data(),
		      d_qp.data(),
		      d_Q.pitch(), 
		      d_Q.dimx(), 
		      d_Q.dimy(),
		      dt / dx, dt / dy, dt,
		      IY
		      );
      } // end update Y

      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt);
      }

    } else if (dimType == THREE_D) {
    
      TIMER_START(timerSlopeTrace);
      {
	// 3D slope / trace computation kernel
	dim3 dimBlock(TRACE_BLOCK_DIMX_3D_V2,
		      TRACE_BLOCK_DIMY_3D_V2);
	dim3 dimGrid(blocksFor(isize, TRACE_BLOCK_INNER_DIMX_3D_V2), 
		     blocksFor(jsize, TRACE_BLOCK_INNER_DIMY_3D_V2));
	kernel_hydro_compute_trace_unsplit_3d_v2<<<dimGrid, 
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
		      dt);
	checkCudaError("HydroRunGodunov :: kernel_hydro_compute_trace_unsplit_3d_v2 error");
      } // end 3D slope / trace computation kernel
    
      if (gravityEnabled) {
	compute_gravity_predictor(d_qm_x, dt);
	compute_gravity_predictor(d_qm_y, dt);
	compute_gravity_predictor(d_qm_z, dt);
	compute_gravity_predictor(d_qp_x, dt);
	compute_gravity_predictor(d_qp_y, dt);
	compute_gravity_predictor(d_qp_z, dt);
      }
      TIMER_STOP(timerSlopeTrace);
    
      TIMER_START(timerUpdate);
      {
	// 3D update hydro kernel
	dim3 dimBlock(UPDATE_BLOCK_DIMX_3D_V1,
		      UPDATE_BLOCK_DIMY_3D_V1);
	dim3 dimGrid(blocksFor(isize, UPDATE_BLOCK_INNER_DIMX_3D_V1), 
		     blocksFor(jsize, UPDATE_BLOCK_INNER_DIMY_3D_V1));
	kernel_hydro_flux_update_unsplit_3d_v1<<<dimGrid, 
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
		      dt );
	checkCudaError("HydroRunGodunov :: kernel_hydro_flux_update_unsplit_3d_v1 error");
      
      } // end 3D update hydro kernel
    
      if (gravityEnabled) {
	compute_gravity_source_term(d_UNew, d_UOld, dt);
      }
      TIMER_STOP(timerUpdate);
    
      /*
       * DISSIPATIVE TERMS (i.e. viscosity)
       */
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

    } // end THREE_D

  } // HydroRunGodunovMpi::godunov_unsplit_gpu_v2

#else // CPU version

  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_split_cpu(HostArray<real_t>& h_UOld, 
					     HostArray<real_t>& h_UNew,
					     int idim, 
					     real_t dt)
  {
    
    make_boundaries(h_UOld,idim);
    communicator->synchronize();

    real_t dtdx = dt/dx;

    TIMER_START(timerGodunov);
    if (dimType == TWO_D) {

      if(idim == XDIR) {

	// gather conservative variables and convert to primitives variables
	for (int j=2; j<jsize-2; j++) {

	  real_t qxm[NVAR_2D]     = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qxm1[NVAR_2D]    = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qxp[NVAR_2D]     = {0.0f, 0.0f, 0.0f, 0.0f};
	  //real_t qxp1[NVAR_2D]    = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qleft[NVAR_2D]   = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qright[NVAR_2D]  = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qgdnv[NVAR_2D]   = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t flux[NVAR_2D]    = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t flux1[NVAR_2D]   = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qPlus[NVAR_2D]   = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qMinus[NVAR_2D]  = {0.0f, 0.0f, 0.0f, 0.0f};

	  for (int i=0; i<isize; i++) {
	    int index = i+j*isize;
	    real_t q[NVAR_2D];
	    real_t c, cPlus, cMinus;
	    
	    computePrimitives_0(h_UOld.data(), h_UOld.section(), index, c, q);
	    if (i<isize-1)
	      computePrimitives_0(h_UOld.data(), h_UOld.section(), index+1, cPlus,  qPlus);
	    if (i>0)
	      computePrimitives_0(h_UOld.data(), h_UOld.section(), index-1, cMinus, qMinus);
 
	    // Characteristic tracing : memorize qxm, qxp
	    // and then update
	    for (int ivar=0; ivar<NVAR_2D; ivar++) {
	      qxm1[ivar] = qxm[ivar];
	      //qxp1[ivar] = qxp[ivar];
	    }
	    if (i>0 && i<isize-1) {
	      if (traceEnabled)  {
		trace<NVAR_2D>(q, qPlus, qMinus, c, dtdx, qxm, qxp);
	      } else {
		// the following is a replacement when not using trace
		for (int ivar=0; ivar<NVAR_2D; ivar++) {
		  qxm[ivar] = q[ivar];
		  qxp[ivar] = q[ivar];
		}
	      } // end traceEnabled
	    }
	  
	    // Solve Riemann problem at interfaces and compute fluxes
	    for (int ivar=0; ivar<NVAR_2D; ivar++) {
	      qleft[ivar]   = qxm1[ivar];
	      qright[ivar]  = qxp[ivar];
	      flux1[ivar]   = flux[ivar];
	    }
	    if (i>1)
	      riemann<NVAR_2D>(qleft,qright,qgdnv,flux);

	    // update conservative variables
	    if (i>2 && i<isize-1) {
	      h_UNew(i-1,j,ID) = h_UOld(i-1,j,ID) + (flux1[ID]-flux[ID])*dtdx;
	      h_UNew(i-1,j,IP) = h_UOld(i-1,j,IP) + (flux1[IP]-flux[IP])*dtdx;
	      h_UNew(i-1,j,IU) = h_UOld(i-1,j,IU) + (flux1[IU]-flux[IU])*dtdx;
	      h_UNew(i-1,j,IV) = h_UOld(i-1,j,IV) + (flux1[IV]-flux[IV])*dtdx;
	    }

	  } // for (int i...

	} // for(int j...

      } else { // idim == YDIR

	// gather conservative variables and convert to primitives variables
	for (int i=2; i<isize-2; i++) {

	  real_t qxm[NVAR_2D]     = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qxm1[NVAR_2D]    = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qxp[NVAR_2D]     = {0.0f, 0.0f, 0.0f, 0.0f};
	  //real_t qxp1[NVAR_2D]    = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qleft[NVAR_2D]   = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qright[NVAR_2D]  = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qgdnv[NVAR_2D]   = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t flux[NVAR_2D]    = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t flux1[NVAR_2D]   = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qPlus[NVAR_2D]   = {0.0f, 0.0f, 0.0f, 0.0f};
	  real_t qMinus[NVAR_2D]  = {0.0f, 0.0f, 0.0f, 0.0f};
	
	  for (int j=0; j<jsize; j++) {
	    int index = i+j*isize;
	    real_t q[NVAR_2D];
	    real_t c, cPlus, cMinus;
	    
	    computePrimitives_1(h_UOld.data(), h_UOld.section(), index, c, q);
	    if (j<jsize-1)
	      computePrimitives_1(h_UOld.data(), h_UOld.section(), index+isize, cPlus, qPlus);
	    if (j>0)
	      computePrimitives_1(h_UOld.data(), h_UOld.section(), index-isize, cMinus, qMinus);
 
	    // Characteristic tracing : memorize qxm and qxp and then update
	    for (int ivar=0; ivar<NVAR_2D; ivar++) {
	      qxm1[ivar] = qxm[ivar];
	      //qxp1[ivar] = qxp[ivar];
	    }
	    if (j>0 && j<jsize-1) {
	      if (traceEnabled) {
		trace<NVAR_2D>(q, qPlus, qMinus, c, dtdx, qxm, qxp);
	      } else {
		// the following is a replacement when not using trace
		for (int ivar=0; ivar<NVAR_2D; ivar++) {
		  qxm[ivar] = q[ivar];
		  qxp[ivar] = q[ivar];
		}
	      } // end traceEnabled
	    }
	    
	    // Solve Riemann problem at interfaces and compute fluxes
	    for (int ivar=0; ivar<NVAR_2D; ivar++) {
	      qleft[ivar]   = qxm1[ivar];
	      qright[ivar]  = qxp[ivar];
	      flux1[ivar]   = flux[ivar];
	    }
	    if (j>1)
	      riemann<NVAR_2D>(qleft,qright,qgdnv,flux);

	    // update conservative variables (care that IV and IV are swapped)
	    if (j>2 && j<jsize-1) {
	      h_UNew(i,j-1,ID) = h_UOld(i,j-1,ID) + (flux1[ID]-flux[ID])*dtdx;
	      h_UNew(i,j-1,IP) = h_UOld(i,j-1,IP) + (flux1[IP]-flux[IP])*dtdx;
	      h_UNew(i,j-1,IU) = h_UOld(i,j-1,IU) + (flux1[IV]-flux[IV])*dtdx;
	      h_UNew(i,j-1,IV) = h_UOld(i,j-1,IV) + (flux1[IU]-flux[IU])*dtdx;
	    }

	  } // for (int j...

	} // for(int i...

      }

    } else { // THREE_D

      if(idim == XDIR) {

	// gather conservative variables and convert to primitives variables
	for (int k=2; k<ksize-2; k++) {
	  
	  for (int j=2; j<jsize-2; j++) {
	    
	    real_t qxm[NVAR_3D]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qxm1[NVAR_3D]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qxp[NVAR_3D]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    //real_t qxp1[NVAR_3D]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qleft[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qright[NVAR_3D]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qgdnv[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t flux[NVAR_3D]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t flux1[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qPlus[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qMinus[NVAR_3D]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	  
	    for (int i=0; i<isize; i++) {
	      int index = i + j*isize + k*isize*jsize;
	      real_t q[NVAR_3D];
	      real_t c, cPlus, cMinus;
	    
	      computePrimitives_3D_0(h_UOld.data(), h_UOld.section(), index, c, q);
	      if (i<isize-1)
		computePrimitives_3D_0(h_UOld.data(), h_UOld.section(), index+1, cPlus,  qPlus);
	      if (i>0)
		computePrimitives_3D_0(h_UOld.data(), h_UOld.section(), index-1, cMinus, qMinus);

	      // Characteristic tracing : memorize qxm and qxp and then update
	      for (int ivar=0; ivar<NVAR_3D; ivar++) {
		qxm1[ivar] = qxm[ivar];
		//qxp1[ivar] = qxp[ivar];
	      }
	      if (i>0 && i<isize-1) {
		if (traceEnabled) {
		  trace<NVAR_3D>(q, qPlus, qMinus, c, dtdx, qxm, qxp);
		} else {
		  // the following is a dummy replacement for trace computations
		  for (int ivar=0; ivar<NVAR_3D; ivar++) {
		    qxm[ivar] = q[ivar];
		    qxp[ivar] = q[ivar];
		  }
		} // end traceEnabled
	      }
	    
	      // Solve Riemann problem at interfaces and compute fluxes
	      for (int ivar=0; ivar<NVAR_3D; ivar++) {
		qleft[ivar]   = qxm1[ivar];
		qright[ivar]  = qxp[ivar];
		flux1[ivar]   = flux[ivar];
	      }
	      if (i>1)
		riemann<NVAR_3D>(qleft,qright,qgdnv,flux);
	    
	      // update conservative variables
	      if (i>2 && i<isize-1) {
		h_UNew(i-1,j,k,ID) = h_UOld(i-1,j,k,ID) + (flux1[ID]-flux[ID])*dtdx;
		h_UNew(i-1,j,k,IP) = h_UOld(i-1,j,k,IP) + (flux1[IP]-flux[IP])*dtdx;
		h_UNew(i-1,j,k,IU) = h_UOld(i-1,j,k,IU) + (flux1[IU]-flux[IU])*dtdx;
		h_UNew(i-1,j,k,IV) = h_UOld(i-1,j,k,IV) + (flux1[IV]-flux[IV])*dtdx;
		h_UNew(i-1,j,k,IW) = h_UOld(i-1,j,k,IW) + (flux1[IW]-flux[IW])*dtdx;
	      }
	    
	    } // for (int i...
	  
	  } // for(int j...
  
	} // for(int k...
	
      } else if (idim == YDIR) { // swap indexes (i,j,k) into (j,i,k)

	// gather conservative variables and convert to primitives variables
	for (int k=2; k<ksize-2; k++) {
	  
	  for (int i=2; i<isize-2; i++) {
	    
	    real_t qxm[NVAR_3D]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qxm1[NVAR_3D]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qxp[NVAR_3D]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    //real_t qxp1[NVAR_3D]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qleft[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qright[NVAR_3D]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qgdnv[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t flux[NVAR_3D]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t flux1[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qPlus[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qMinus[NVAR_3D]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	  
	    for (int j=0; j<jsize; j++) {
	      int index = i + j*isize + k*isize*jsize;
	      real_t q[NVAR_3D];
	      real_t c, cPlus, cMinus;
	    
	      computePrimitives_3D_1(h_UOld.data(), h_UOld.section(), index, c, q);
	      if (j<jsize-1)
		computePrimitives_3D_1(h_UOld.data(), h_UOld.section(), index+isize, cPlus,  qPlus);
	      if (j>0)
		computePrimitives_3D_1(h_UOld.data(), h_UOld.section(), index-isize, cMinus, qMinus);
	    
	      // Characteristic tracing : memorize qxm and qxp and then update
	      for (int ivar=0; ivar<NVAR_3D; ivar++) {
		qxm1[ivar] = qxm[ivar];
		//qxp1[ivar] = qxp[ivar];
	      }
	      if (j>0 && j<jsize-1) {
		if (traceEnabled) {
		  trace<NVAR_3D>(q, qPlus, qMinus, c, dtdx, qxm, qxp);
		} else {
		  // the following is a dummy replacement for trace computations
		  for (int ivar=0; ivar<NVAR_3D; ivar++) {
		    qxm[ivar] = q[ivar];
		    qxp[ivar] = q[ivar];
		  }
		} // end traceEnabled
	      }
	    
	      // Solve Riemann problem at interfaces and compute fluxes
	      for (int ivar=0; ivar<NVAR_3D; ivar++) {
		qleft[ivar]   = qxm1[ivar];
		qright[ivar]  = qxp[ivar];
		flux1[ivar]   = flux[ivar];
	      }
	      if (j>1)
		riemann<NVAR_3D>(qleft,qright,qgdnv,flux);
	    
	      // update conservative variables : watchout IU and IV are swapped
	      if (j>2 && j<jsize-1) {
		h_UNew(i,j-1,k,ID) = h_UOld(i,j-1,k,ID) + (flux1[ID]-flux[ID])*dtdx;
		h_UNew(i,j-1,k,IP) = h_UOld(i,j-1,k,IP) + (flux1[IP]-flux[IP])*dtdx;
		h_UNew(i,j-1,k,IU) = h_UOld(i,j-1,k,IU) + (flux1[IV]-flux[IV])*dtdx;
		h_UNew(i,j-1,k,IV) = h_UOld(i,j-1,k,IV) + (flux1[IU]-flux[IU])*dtdx;
		h_UNew(i,j-1,k,IW) = h_UOld(i,j-1,k,IW) + (flux1[IW]-flux[IW])*dtdx;
	      }
	    
	    } // for (int i...
	  
	  } // for(int j...
  
	} // for(int k...
	
      } else { // idim == ZDIR       // swap indexes (i,j,k) into (k,j,i)

	// gather conservative variables and convert to primitives variables
	for (int j=2; j<jsize-2; j++) {
	  
	  for (int i=2; i<isize-2; i++) {
	    
	    real_t qxm[NVAR_3D]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qxm1[NVAR_3D]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qxp[NVAR_3D]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    //real_t qxp1[NVAR_3D]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qleft[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qright[NVAR_3D]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qgdnv[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t flux[NVAR_3D]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t flux1[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qPlus[NVAR_3D]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	    real_t qMinus[NVAR_3D]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	  
	    for (int k=0; k<ksize; k++) {
	      int index = i + j*isize + k*isize*jsize;
	      real_t q[NVAR_3D];
	      real_t c, cPlus, cMinus;
	    
	      computePrimitives_3D_2(h_UOld.data(), h_UOld.section(), index, c, q);
	      if (k<ksize-1)
		computePrimitives_3D_2(h_UOld.data(), h_UOld.section(), index+isize*jsize, cPlus,  qPlus);
	      if (k>0)
		computePrimitives_3D_2(h_UOld.data(), h_UOld.section(), index-isize*jsize, cMinus, qMinus);

	      // Characteristic tracing : memorize qxm and qxp and then update
	      for (int ivar=0; ivar<NVAR_3D; ivar++) {
		qxm1[ivar] = qxm[ivar];
		//qxp1[ivar] = qxp[ivar];
	      }
	      if (k>0 && k<ksize-1) {
		if (traceEnabled) {
		  trace<NVAR_3D>(q, qPlus, qMinus, c, dtdx, qxm, qxp);
		} else {
		  // the following is a dummy replacement for trace computations
		  for (int ivar=0; ivar<NVAR_3D; ivar++) {
		    qxm[ivar] = q[ivar];
		    qxp[ivar] = q[ivar];
		  }
		} // end traceEnabled
	      }
	    
	      // Solve Riemann problem at interfaces and compute fluxes
	      for (int ivar=0; ivar<NVAR_3D; ivar++) {
		qleft[ivar]   = qxm1[ivar];
		qright[ivar]  = qxp[ivar];
		flux1[ivar]   = flux[ivar];
	      }
	      if (k>1)
		riemann<NVAR_3D>(qleft,qright,qgdnv,flux);
	    
	      // update conservative variables : watchout IU and IW are swapped
	      if (k>2 && k<ksize-1) {
		h_UNew(i,j,k-1,ID) = h_UOld(i,j,k-1,ID) + (flux1[ID]-flux[ID])*dtdx;
		h_UNew(i,j,k-1,IP) = h_UOld(i,j,k-1,IP) + (flux1[IP]-flux[IP])*dtdx;
		h_UNew(i,j,k-1,IU) = h_UOld(i,j,k-1,IU) + (flux1[IW]-flux[IW])*dtdx;
		h_UNew(i,j,k-1,IV) = h_UOld(i,j,k-1,IV) + (flux1[IV]-flux[IV])*dtdx;
		h_UNew(i,j,k-1,IW) = h_UOld(i,j,k-1,IW) + (flux1[IU]-flux[IU])*dtdx;
	      }
	    
	    } // for (int i...
	  
	  } // for(int j...
  
	} // for(int k...

      } // end if (idim == ZDIR)

    } // THREE_D
    TIMER_STOP(timerGodunov);

  } // HydroRunGodunovMpi::godunov_split_cpu

  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_unsplit_cpu(HostArray<real_t>& h_UOld, 
					       HostArray<real_t>& h_UNew, 
					       real_t dt)
  {
    make_all_boundaries(h_UOld);

    // copy h_UOld into h_UNew
    // for (int indexGlob=0; indexGlob<h_UOld.size(); indexGlob++) {
    //   h_UNew(indexGlob) = h_UOld(indexGlob);
    // }
    h_UOld.copyTo(h_UNew);

    TIMER_START(timerGodunov);

    /*
     * Whatever implementation version, start by computing primitive variables
     *   
     * convert conservative to primitive variables (and source term predictor)
     * put results in h_Q object
     *
     */
    TIMER_START(timerPrimVar);
    convertToPrimitives( h_UOld.data() );
    TIMER_STOP(timerPrimVar);

    if (unsplitVersion == 0) {

      godunov_unsplit_cpu_v0(h_UOld, h_UNew, dt);

    } else if (unsplitVersion == 1) {

      godunov_unsplit_cpu_v1(h_UOld, h_UNew, dt);
    
    } else if (unsplitVersion == 2) {

      godunov_unsplit_cpu_v2(h_UOld, h_UNew, dt);

    } // end unsplitVersion == 2

    TIMER_STOP(timerGodunov);

  } // HydroRunGodunovMpi::godunov_unsplit_cpu

#endif // __CUDACC__

#ifndef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_unsplit_cpu_v0(HostArray<real_t>& h_UOld,
						  HostArray<real_t>& h_UNew,
						  real_t dt)
  {
    
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;
    
    if (dimType == TWO_D) {

      // #ifdef _OPENMP
      // #pragma omp parallel default(shared) private(qm_x,qm_y,qp_x,qp_y)
      // #pragma omp for collapse(2) schedule(auto)
      // #endif // _OPENMP
      for (int j=2; j<jsize-1; j++) {
	for (int i=2; i<isize-1; i++) {
	
	  // primitive variables (local array)
	  real_t qLoc[NVAR_2D];
	  real_t qLocNeighbor[NVAR_2D];
	  real_t qNeighbors[2*TWO_D][NVAR_2D];
	
	  // slopes
	  real_t dq[TWO_D][NVAR_2D];
	  real_t dqNeighbor[TWO_D][NVAR_2D];
	
	  // reconstructed state on cell faces
	  // aka riemann solver input
	  real_t qleft[NVAR_2D];
	  real_t qright[NVAR_2D];
	
	  // riemann solver output
	  real_t qgdnv[NVAR_2D];
	  real_t flux_x[NVAR_2D];
	  real_t flux_y[NVAR_2D];
	
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  // deal with left interface along X !
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	  // get primitive variables state vector
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    qLoc[iVar]          = h_Q(i  ,j  ,iVar);
	    qNeighbors[0][iVar] = h_Q(i+1,j  ,iVar);
	    qNeighbors[1][iVar] = h_Q(i-1,j  ,iVar);
	    qNeighbors[2][iVar] = h_Q(i  ,j+1,iVar);
	    qNeighbors[3][iVar] = h_Q(i  ,j-1,iVar);
	  
	  } // end for iVar
	
	  // compute slopes in current cell
	  slope_unsplit_hydro_2d(qLoc, 
				 qNeighbors[0],
				 qNeighbors[1],
				 qNeighbors[2],
				 qNeighbors[3],
				 dq);
	
	  // get primitive variables state vector in left neighbor along X
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    qLocNeighbor[iVar]  = h_Q(i-1,j  ,iVar);
	    qNeighbors[0][iVar] = h_Q(i  ,j  ,iVar);
	    qNeighbors[1][iVar] = h_Q(i-2,j  ,iVar);
	    qNeighbors[2][iVar] = h_Q(i-1,j+1,iVar);
	    qNeighbors[3][iVar] = h_Q(i-1,j-1,iVar);
	  
	  } // end for iVar
	
	  // compute slopes in left neighbor along X
	  slope_unsplit_hydro_2d(qLocNeighbor, 
				 qNeighbors[0],
				 qNeighbors[1],
				 qNeighbors[2],
				 qNeighbors[3],
				 dqNeighbor);
	
	  //
	  // Compute reconstructed states at left interface along X in current cell
	  //

	  // left interface : right state
	  trace_unsplit_hydro_2d_by_direction(qLoc, 
					      dq, 
					      dtdx, dtdy, 
					      FACE_XMIN, 
					      qright);
	
	  // left interface : left state
	  trace_unsplit_hydro_2d_by_direction(qLocNeighbor,
					      dqNeighbor,
					      dtdx, dtdy,
					      FACE_XMAX, 
					      qleft);
	
	  if (gravityEnabled) { 
	  
	    // we need to modify input to flux computation with
	    // gravity predictor (half time step)
	  
	    qleft[IU]  += HALF_F * dt * h_gravity(i,j,IX);
	    qleft[IV]  += HALF_F * dt * h_gravity(i,j,IY);
	  
	    qright[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qright[IV] += HALF_F * dt * h_gravity(i,j,IY);
	  
	  } // end gravityEnabled
	
	  // Solve Riemann problem at X-interfaces and compute X-fluxes
	  riemann<NVAR_2D>(qleft,qright,qgdnv,flux_x);

	  /*
	   * update with flux_x
	   */
	  if ( i > ghostWidth       and 
	       j < jsize-ghostWidth ) {
	    h_UNew(i-1,j  ,ID) -= flux_x[ID]*dtdx;
	    h_UNew(i-1,j  ,IP) -= flux_x[IP]*dtdx;
	    h_UNew(i-1,j  ,IU) -= flux_x[IU]*dtdx;
	    h_UNew(i-1,j  ,IV) -= flux_x[IV]*dtdx;
	  }
	
	  if ( i < isize-ghostWidth and 
	       j < jsize-ghostWidth ) {
	    h_UNew(i  ,j  ,ID) += flux_x[ID]*dtdx;
	    h_UNew(i  ,j  ,IP) += flux_x[IP]*dtdx;
	    h_UNew(i  ,j  ,IU) += flux_x[IU]*dtdx;
	    h_UNew(i  ,j  ,IV) += flux_x[IV]*dtdx;
	  }
	  
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  // deal with left interface along Y !
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	  // get primitive variables state vector in left neighbor along Y
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    qLocNeighbor[iVar]  = h_Q(i  ,j-1,iVar);
	    qNeighbors[0][iVar] = h_Q(i+1,j-1,iVar);
	    qNeighbors[1][iVar] = h_Q(i-1,j-1,iVar);
	    qNeighbors[2][iVar] = h_Q(i  ,j  ,iVar);
	    qNeighbors[3][iVar] = h_Q(i  ,j-2,iVar);
	  
	  } // end for iVar
	
	  // compute slopes in left neighbor along Y
	  slope_unsplit_hydro_2d(qLocNeighbor, 
				 qNeighbors[0],
				 qNeighbors[1],
				 qNeighbors[2],
				 qNeighbors[3],
				 dqNeighbor);
	
	  //
	  // Compute reconstructed states at left interface along Y in current cell
	  //
	
	  // left interface : right state
	  trace_unsplit_hydro_2d_by_direction(qLoc, 
					      dq, 
					      dtdx, dtdy, 
					      FACE_YMIN, 
					      qright);
	
	  // left interface : left state
	  trace_unsplit_hydro_2d_by_direction(qLocNeighbor,
					      dqNeighbor,
					      dtdx, dtdy,
					      FACE_YMAX, 
					      qleft);
	
	  if (gravityEnabled) { 
	    // we need to modify input to flux computation with
	    // gravity predictor (half time step)
	  
	    qleft[IU]  += HALF_F * dt * h_gravity(i,j,IX);
	    qleft[IV]  += HALF_F * dt * h_gravity(i,j,IY);
	  
	    qright[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qright[IV] += HALF_F * dt * h_gravity(i,j,IY);
	  
	  } // end gravityEnabled
	
	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes	  
	  swapValues(&(qleft[IU]) ,&(qleft[IV]) );
	  swapValues(&(qright[IU]),&(qright[IV]));
	  riemann<NVAR_2D>(qleft,qright,qgdnv,flux_y);
	
	  /*
	   * update with flux_y
	   */
	  if ( i < isize-ghostWidth and
	       j > ghostWidth       ) {
	    h_UNew(i  ,j-1,ID) -= flux_y[ID]*dtdy;
	    h_UNew(i  ,j-1,IP) -= flux_y[IP]*dtdy;
	    h_UNew(i  ,j-1,IU) -= flux_y[IV]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j-1,IV) -= flux_y[IU]*dtdy; // watchout IU and IV swapped
	  }
	
	  if ( i < isize-ghostWidth and 
	       j < jsize-ghostWidth ) {
	    h_UNew(i  ,j  ,ID) += flux_y[ID]*dtdy;
	    h_UNew(i  ,j  ,IP) += flux_y[IP]*dtdy;
	    h_UNew(i  ,j  ,IU) += flux_y[IV]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j  ,IV) += flux_y[IU]*dtdy; // watchout IU and IV swapped
	  }
	
	} // end for j
      } // end for i

      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(h_UNew, h_UOld, dt);
      }
    
    } else { // THREE_D - Implementation version 0
    
      // #ifdef _OPENMP
      // #pragma omp parallel default(shared) private(qm_x,qm_y,qm_z,qp_x,qp_y,qp_z)
      // #pragma omp for collapse(3) schedule(auto)
      // #endif // _OPENMP
      for (int k=2; k<ksize-1; k++) {
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
	      
	      qleft[IU] += HALF_F * dt * h_gravity(i,j,k,IX);
	      qleft[IV] += HALF_F * dt * h_gravity(i,j,k,IY);
	      qleft[IW] += HALF_F * dt * h_gravity(i,j,k,IZ);
	      
	      qright[IU] += HALF_F * dt * h_gravity(i,j,k,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,k,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,k,IZ);
	      
	    } // end gravityEnabled
	    
	    // Solve Riemann problem at X-interfaces and compute X-fluxes    
	    riemann<NVAR_3D>(qleft,qright,qgdnv,flux_x);
	    
	    /*
	     * update with flux_x
	     */
	    if ( i  > ghostWidth           and 
		 j  < jsize-ghostWidth     and
		 k  < ksize-ghostWidth ) {
	      h_UNew(i-1,j  ,k  ,ID) -= flux_x[ID]*dtdx;
	      h_UNew(i-1,j  ,k  ,IP) -= flux_x[IP]*dtdx;
	      h_UNew(i-1,j  ,k  ,IU) -= flux_x[IU]*dtdx;
	      h_UNew(i-1,j  ,k  ,IV) -= flux_x[IV]*dtdx;
	      h_UNew(i-1,j  ,k  ,IW) -= flux_x[IW]*dtdx;
	    }
	    
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_x[ID]*dtdx;
	      h_UNew(i  ,j  ,k  ,IP) += flux_x[IP]*dtdx;
	      h_UNew(i  ,j  ,k  ,IU) += flux_x[IU]*dtdx;
	      h_UNew(i  ,j  ,k  ,IV) += flux_x[IV]*dtdx;
	      h_UNew(i  ,j  ,k  ,IW) += flux_x[IW]*dtdx;
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
	      
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,k,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,k,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,k,IZ);
	      
	      qright[IU] += HALF_F * dt * h_gravity(i,j,k,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,k,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,k,IZ);
	      
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
		 k  < ksize-ghostWidth ) {
	      h_UNew(i  ,j-1,k  ,ID) -= flux_y[ID]*dtdx;
	      h_UNew(i  ,j-1,k  ,IP) -= flux_y[IP]*dtdx;
	      h_UNew(i  ,j-1,k  ,IU) -= flux_y[IV]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IV) -= flux_y[IU]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IW) -= flux_y[IW]*dtdx;
	    }
	    
	    if ( i  < isize-ghostWidth     and
		 j  < jsize-ghostWidth     and
		 k  < ksize-ghostWidth     ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_y[ID]*dtdx;
	      h_UNew(i  ,j  ,k  ,IP) += flux_y[IP]*dtdx;
	      h_UNew(i  ,j  ,k  ,IU) += flux_y[IV]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,k  ,IV) += flux_y[IU]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,k  ,IW) += flux_y[IW]*dtdx;
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
	      
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,k,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,k,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,k,IZ);
	      
	      qright[IU] += HALF_F * dt * h_gravity(i,j,k,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,k,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,k,IZ);
	      
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
		 k  > ghostWidth ) {
	      h_UNew(i  ,j  ,k-1,ID) -= flux_z[ID]*dtdx;
	      h_UNew(i  ,j  ,k-1,IP) -= flux_z[IP]*dtdx;
	      h_UNew(i  ,j  ,k-1,IU) -= flux_z[IW]*dtdx; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,k-1,IV) -= flux_z[IV]*dtdx;
	      h_UNew(i  ,j  ,k-1,IW) -= flux_z[IU]*dtdx; // watchout IU and IW swapped
	    }
	    
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_z[ID]*dtdx;
	      h_UNew(i  ,j  ,k  ,IP) += flux_z[IP]*dtdx;
	      h_UNew(i  ,j  ,k  ,IU) += flux_z[IW]*dtdx; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,k  ,IV) += flux_z[IV]*dtdx;
	      h_UNew(i  ,j  ,k  ,IW) += flux_z[IU]*dtdx; // watchout IU and IW swapped
	    }
	  } // end for i
	} // end for j
      } // end for k

      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(h_UNew, h_UOld, dt);
      }

    } // end THREE_D - Implementation version 0

} // HydroRunGodunovMpi::godunov_unsplit_cpu_v0
#endif // __CUDACC__

#ifndef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_unsplit_cpu_v1(HostArray<real_t>& h_UOld,
						  HostArray<real_t>& h_UNew,
						  real_t dt)
  {

    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;

    if (dimType == TWO_D) {
    
      // call trace computation routine
      TIMER_START(timerSlopeTrace);
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(2) schedule(auto)
#endif // _OPENMP
      for (int j=1; j<jsize-1; j++) {
	for (int i=1; i<isize-1; i++) {
	
	  real_t q[NVAR_2D];
	  real_t qPlusX  [NVAR_2D], qMinusX [NVAR_2D],
	    qPlusY  [NVAR_2D], qMinusY [NVAR_2D];
	  real_t dq[2][NVAR_2D];
	
	  real_t qm[TWO_D][NVAR_2D];
	  real_t qp[TWO_D][NVAR_2D];
	
	  // get primitive variables state vector
	  for (int iVar=0; iVar<NVAR_2D; iVar++) {
	    q      [iVar] = h_Q(i  ,j  ,iVar);
	    qPlusX [iVar] = h_Q(i+1,j  ,iVar);
	    qMinusX[iVar] = h_Q(i-1,j  ,iVar);
	    qPlusY [iVar] = h_Q(i  ,j+1,iVar);
	    qMinusY[iVar] = h_Q(i  ,j-1,iVar);
	  }
	
	  // get hydro slopes dq
	  slope_unsplit_hydro_2d(q, 
				 qPlusX, qMinusX,
				 qPlusY, qMinusY,
				 dq);
	
	  // compute qm, qp
	  trace_unsplit_hydro_2d(q, dq,
				 dtdx, dtdy, 
				 qm, qp);
	
	  // gravity predictor
	  if (gravityEnabled) { 
	    qm[0][IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qm[0][IV] += HALF_F * dt * h_gravity(i,j,IY);
	  
	    qp[0][IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qp[0][IV] += HALF_F * dt * h_gravity(i,j,IY);
	  
	    qm[1][IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qm[1][IV] += HALF_F * dt * h_gravity(i,j,IY);
	  
	    qp[1][IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qp[1][IV] += HALF_F * dt * h_gravity(i,j,IY);
	  }
	
	  // store qm, qp : only what is really needed
	  for (int ivar=0; ivar<NVAR_2D; ivar++) {
	    h_qm_x(i,j,ivar) = qm[0][ivar];
	    h_qp_x(i,j,ivar) = qp[0][ivar];
	    h_qm_y(i,j,ivar) = qm[1][ivar];
	    h_qp_y(i,j,ivar) = qp[1][ivar];
	  } // end for ivar	
	
	} // end for i
      } // end for j
      TIMER_STOP(timerSlopeTrace);
       
      TIMER_START(timerUpdate);
      // Finally compute fluxes from rieman solvers, and update
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(2) schedule(auto)
#endif // _OPENMP
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	
	  real_riemann_t qleft[NVAR_2D];
	  real_riemann_t qright[NVAR_2D];
	  real_riemann_t flux_x[NVAR_2D];
	  real_riemann_t flux_y[NVAR_2D];
	  real_t qgdnv[NVAR_2D];
	
	  /*
	   * Solve Riemann problem at X-interfaces and compute
	   * X-fluxes
	   */
	  qleft[ID]   = h_qm_x(i-1,j,ID);
	  qleft[IP]   = h_qm_x(i-1,j,IP);
	  qleft[IU]   = h_qm_x(i-1,j,IU);
	  qleft[IV]   = h_qm_x(i-1,j,IV);
	
	  qright[ID]  = h_qp_x(i  ,j,ID);
	  qright[IP]  = h_qp_x(i  ,j,IP);
	  qright[IU]  = h_qp_x(i  ,j,IU);
	  qright[IV]  = h_qp_x(i  ,j,IV);
	
	  // compute hydro flux_x
	  riemann<NVAR_2D>(qleft,qright,qgdnv,flux_x);
	
	  /*
	   * Solve Riemann problem at Y-interfaces and compute Y-fluxes
	   */
	  qleft[ID]   = h_qm_y(i,j-1,ID);
	  qleft[IP]   = h_qm_y(i,j-1,IP);
	  qleft[IU]   = h_qm_y(i,j-1,IV); // watchout IU, IV permutation
	  qleft[IV]   = h_qm_y(i,j-1,IU); // watchout IU, IV permutation
	
	  qright[ID]  = h_qp_y(i,j  ,ID);
	  qright[IP]  = h_qp_y(i,j  ,IP);
	  qright[IU]  = h_qp_y(i,j  ,IV); // watchout IU, IV permutation
	  qright[IV]  = h_qp_y(i,j  ,IU); // watchout IU, IV permutation
	
	  // compute hydro flux_y
	  riemann<NVAR_2D>(qleft,qright,qgdnv,flux_y);
	
	  /*
	   * update hydro array
	   */
	
	  /*
	   * update with flux_x
	   */
	  if ( i > ghostWidth       and 
	       j < jsize-ghostWidth ) {
	    h_UNew(i-1,j  ,ID) -= flux_x[ID]*dtdx;
	    h_UNew(i-1,j  ,IP) -= flux_x[IP]*dtdx;
	    h_UNew(i-1,j  ,IU) -= flux_x[IU]*dtdx;
	    h_UNew(i-1,j  ,IV) -= flux_x[IV]*dtdx;
	  }
	
	  if ( i < isize-ghostWidth and 
	       j < jsize-ghostWidth ) {
	    h_UNew(i  ,j  ,ID) += flux_x[ID]*dtdx;
	    h_UNew(i  ,j  ,IP) += flux_x[IP]*dtdx;
	    h_UNew(i  ,j  ,IU) += flux_x[IU]*dtdx;
	    h_UNew(i  ,j  ,IV) += flux_x[IV]*dtdx;
	  }
	
	  /*
	   * update with flux_y
	   */
	  if ( i < isize-ghostWidth and
	       j > ghostWidth       ) {
	    h_UNew(i  ,j-1,ID) -= flux_y[ID]*dtdy;
	    h_UNew(i  ,j-1,IP) -= flux_y[IP]*dtdy;
	    h_UNew(i  ,j-1,IU) -= flux_y[IV]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j-1,IV) -= flux_y[IU]*dtdy; // watchout IU and IV swapped
	  }
	
	  if ( i < isize-ghostWidth and 
	       j < jsize-ghostWidth ) {
	    h_UNew(i  ,j  ,ID) += flux_y[ID]*dtdy;
	    h_UNew(i  ,j  ,IP) += flux_y[IP]*dtdy;
	    h_UNew(i  ,j  ,IU) += flux_y[IV]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j  ,IV) += flux_y[IU]*dtdy; // watchout IU and IV swapped
	  }
	
	} // end for i
      } // end for j
    
      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(h_UNew, h_UOld, dt);
      }
    
      TIMER_STOP(timerUpdate);
    
      /*
       * DISSIPATIVE TERMS (i.e. viscosity)
       */
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
      
	compute_viscosity_flux(h_UNew, flux_x, flux_y, dt);
	compute_hydro_update  (h_UNew, flux_x, flux_y);
      
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
    
    } else { // THREE_D - unsplit version 1
    
      TIMER_START(timerSlopeTrace);
      // call trace computation routine
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=1; k<ksize-1; k++) {
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
	    /*slope_unsplit_3d_v1(q, qPlusX, qMinusX, dq[0]);
	      slope_unsplit_3d_v1(q, qPlusY, qMinusY, dq[1]);
	      slope_unsplit_3d_v1(q, qPlusZ, qMinusZ, dq[2]);*/
	  
	    // compute qm, qp
	    trace_unsplit_hydro_3d(q, dq, 
				   dtdx, dtdy, dtdz,
				   qm, qp);
	  
	    // gravity predictor / modify velocity components
	    if (gravityEnabled) { 
	      real_t grav_x = HALF_F * dt * h_gravity(i,j,k,IX);
	      real_t grav_y = HALF_F * dt * h_gravity(i,j,k,IY);
	      real_t grav_z = HALF_F * dt * h_gravity(i,j,k,IZ);
	    
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
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
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
	  
	    /*
	     * update with flux_y
	     */
	    if ( i < isize-ghostWidth and
		 j > ghostWidth       and
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j-1,k  ,ID) -= flux_y[ID]*dtdy;
	      h_UNew(i  ,j-1,k  ,IP) -= flux_y[IP]*dtdy;
	      h_UNew(i  ,j-1,k  ,IU) -= flux_y[IV]*dtdy; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IV) -= flux_y[IU]*dtdy; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IW) -= flux_y[IW]*dtdy;
	    }
	  
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_y[ID]*dtdy;
	      h_UNew(i  ,j  ,k  ,IP) += flux_y[IP]*dtdy;
	      h_UNew(i  ,j  ,k  ,IU) += flux_y[IV]*dtdy; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,k  ,IV) += flux_y[IU]*dtdy; // watchout IU and IV swapped
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
	      h_UNew(i  ,j  ,k-1,IU) -= flux_z[IW]*dtdz; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,k-1,IV) -= flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,k-1,IW) -= flux_z[IU]*dtdz; // watchout IU and IW swapped
	    }
	  
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,k  ,IP) += flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,k  ,IU) += flux_z[IW]*dtdz; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,k  ,IV) += flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,k  ,IW) += flux_z[IU]*dtdz; // watchout IU and IW swapped
	    }
	  } // end for i
	} // end for j
      } // end for k
    
      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(h_UNew, h_UOld, dt);
      }
    
      TIMER_STOP(timerUpdate);
    
      /*
       * DISSIPATIVE TERMS (i.e. viscosity)
       */
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
    
    } // end THREE_D  unsplit version 1

  } // HydroRunGodunovMpi::godunov_unsplit_cpu_v1
#endif // __CUDACC__

#ifndef __CUDACC__
  // =======================================================
  // =======================================================
  void HydroRunGodunovMpi::godunov_unsplit_cpu_v2(HostArray<real_t>& h_UOld,
						  HostArray<real_t>& h_UNew,
						  real_t dt)
  {

    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;

    if (dimType == TWO_D) {

      /*
       * 1. Compute and store slopes
       */
      for (int j=1; j<jsize-1; j++) {
	for (int i=1; i<isize-1; i++) {
	
	  // primitive variables (local array)
	  real_t qLoc[NVAR_2D];
	  real_t qNeighbors[2*TWO_D][NVAR_2D];
	
	  // slopes
	  real_t dq[TWO_D][NVAR_2D];
	
	  // get primitive variables state vector
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    qLoc[iVar]          = h_Q(i  ,j  ,iVar);
	    qNeighbors[0][iVar] = h_Q(i+1,j  ,iVar);
	    qNeighbors[1][iVar] = h_Q(i-1,j  ,iVar);
	    qNeighbors[2][iVar] = h_Q(i  ,j+1,iVar);
	    qNeighbors[3][iVar] = h_Q(i  ,j-1,iVar);
	  
	  } // end for iVar
	
	  // compute slopes in current cell
	  slope_unsplit_hydro_2d(qLoc, 
				 qNeighbors[0],
				 qNeighbors[1],
				 qNeighbors[2],
				 qNeighbors[3],
				 dq);

	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    h_slope_x(i,j,iVar) = dq[0][iVar];
	    h_slope_y(i,j,iVar) = dq[1][iVar];
	  }

	} // end for i
      } // end for j

      /*
       * 2. Compute reconstructed states along X interfaces
       */
      for (int j=1; j<jsize-1; j++) {
	for (int i=1; i<isize-1; i++) {

	  // primitive variables (local array)
	  real_t qLoc[NVAR_2D];
	
	  // slopes
	  real_t dq[TWO_D][NVAR_2D];
	
	  // reconstructed state on cell faces
	  // aka riemann solver input
	  real_t qleft[NVAR_2D];
	  real_t qright[NVAR_2D];
	
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  // deal with left interface along X !
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	  // get current cell slopes and left neighbor
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    qLoc[iVar]  = h_Q      (i  ,j  ,iVar);
	    dq[0][iVar] = h_slope_x(i  ,j  ,iVar);
	    dq[1][iVar] = h_slope_y(i  ,j  ,iVar);
	  
	    //qLocN[iVar]  = h_Q      (i-1,j  ,iVar);
	    //dqN[0][iVar] = h_slope_x(i-1,j  ,iVar);
	    //dqN[1][iVar] = h_slope_y(i-1,j  ,iVar);
	  
	  } // end for iVar
	
	  //
	  // Compute reconstructed states at left interface along X in current cell
	  //
	
	  // TAKE CARE here left and right designate the interface location
	  // compare to current cell
	  // !!! THIS is FUNDAMENTALLY different from v0 and v1 !!!

	  // left interface 
	  trace_unsplit_hydro_2d_by_direction(qLoc, 
					      dq, 
					      dtdx, dtdy, 
					      FACE_XMIN, 
					      qleft);
	
	  // right interface
	  trace_unsplit_hydro_2d_by_direction(qLoc,
					      dq,
					      dtdx, dtdy,
					      FACE_XMAX, 
					      qright);
	
	  if (gravityEnabled) { 
	  
	    // we need to modify input to flux computation with
	    // gravity predictor (half time step)
	  
	    qleft[IU]  += HALF_F * dt * h_gravity(i,j,IX);
	    qleft[IV]  += HALF_F * dt * h_gravity(i,j,IY);
	  
	    qright[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qright[IV] += HALF_F * dt * h_gravity(i,j,IY);
	  
	  } // end gravityEnabled
	
	  // store them
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    h_qm(i  ,j  ,iVar) = qleft[iVar];
	    h_qp(i  ,j  ,iVar) = qright[iVar];

	  }

	} // end for i
      } // end for j

      /*
       * 3. Riemann solver at X interface and update
       */
      for (int j=2; j<jsize-1; j++) {
	for (int i=2; i<isize-1; i++) {

	  // reconstructed state on cell faces
	  // aka riemann solver input
	  real_t qleft[NVAR_2D];
	  real_t qright[NVAR_2D];
	
	  // riemann solver output
	  real_t qgdnv[NVAR_2D];
	  real_t flux_x[NVAR_2D];

	  // read reconstructed states
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    qleft[iVar]  = h_qp(i-1,j  ,iVar);
	    qright[iVar] = h_qm(i  ,j  ,iVar);

	  }

	  // Solve Riemann problem at X-interfaces and compute X-fluxes
	  riemann<NVAR_2D>(qleft,qright,qgdnv,flux_x);

	  /*
	   * update with flux_x
	   */
	  if ( i > ghostWidth       and 
	       j < jsize-ghostWidth ) {
	    h_UNew(i-1,j  ,ID) -= flux_x[ID]*dtdx;
	    h_UNew(i-1,j  ,IP) -= flux_x[IP]*dtdx;
	    h_UNew(i-1,j  ,IU) -= flux_x[IU]*dtdx;
	    h_UNew(i-1,j  ,IV) -= flux_x[IV]*dtdx;
	  }
	
	  if ( i < isize-ghostWidth and 
	       j < jsize-ghostWidth ) {
	    h_UNew(i  ,j  ,ID) += flux_x[ID]*dtdx;
	    h_UNew(i  ,j  ,IP) += flux_x[IP]*dtdx;
	    h_UNew(i  ,j  ,IU) += flux_x[IU]*dtdx;
	    h_UNew(i  ,j  ,IV) += flux_x[IV]*dtdx;
	  }
	
	} // end for i
      } // end for j

      /*
       * 4. Compute reconstructed states along Y interfaces
       */
      for (int j=1; j<jsize-1; j++) {
	for (int i=1; i<isize-1; i++) {

	  // primitive variables (local array)
	  real_t qLoc[NVAR_2D];
	
	  // slopes
	  real_t dq[TWO_D][NVAR_2D];
	
	  // reconstructed state on cell faces
	  // aka riemann solver input
	  real_t qleft[NVAR_2D];
	  real_t qright[NVAR_2D];
	
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  // deal with left interface along Y !
	  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	  // get current cell slopes and left neighbor
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    qLoc[iVar]  = h_Q      (i  ,j  ,iVar);
	    dq[0][iVar] = h_slope_x(i  ,j  ,iVar);
	    dq[1][iVar] = h_slope_y(i  ,j  ,iVar);
	  
	    //qLocN[iVar]  = h_Q      (i  ,j-1,iVar);
	    //dqN[0][iVar] = h_slope_x(i  ,j-1,iVar);
	    //dqN[1][iVar] = h_slope_y(i  ,j-1,iVar);
	  
	  } // end for iVar
	
	  //
	  // Compute reconstructed states at left interface along Y in current cell
	  //

	  // TAKE CARE here left and right designate the interface location
	  // compare to current cell
	  // !!! THIS is FUNDAMENTALLY different from v0 and v1 !!!

	  // left interface
	  trace_unsplit_hydro_2d_by_direction(qLoc, 
					      dq, 
					      dtdx, dtdy, 
					      FACE_YMIN, 
					      qleft);
	
	  // right interface
	  trace_unsplit_hydro_2d_by_direction(qLoc,
					      dq,
					      dtdx, dtdy,
					      FACE_YMAX, 
					      qright);
	
	  if (gravityEnabled) { 
	    // we need to modify input to flux computation with
	    // gravity predictor (half time step)
	  
	    qleft[IU]  += HALF_F * dt * h_gravity(i,j,IX);
	    qleft[IV]  += HALF_F * dt * h_gravity(i,j,IY);
	  
	    qright[IU] += HALF_F * dt * h_gravity(i,j,IX);
	    qright[IV] += HALF_F * dt * h_gravity(i,j,IY);
	  
	  } // end gravityEnabled
	
	  // store them
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    h_qm(i  ,j  ,iVar) = qleft[iVar];
	    h_qp(i  ,j  ,iVar) = qright[iVar];

	  }

	} // end for i
      } // end for j

      /*
       * 5. Riemann solver at Y interface and update
       */
      for (int j=2; j<jsize-1; j++) {
	for (int i=2; i<isize-1; i++) {

	  // reconstructed state on cell faces
	  // aka riemann solver input
	  real_t qleft[NVAR_2D];
	  real_t qright[NVAR_2D];
	
	  // riemann solver output
	  real_t qgdnv[NVAR_2D];
	  real_t flux_y[NVAR_2D];

	  // read reconstructed states
	  for ( int iVar=0; iVar<nbVar; iVar++ ) {
	  
	    qleft[iVar]  = h_qp(i  ,j-1,iVar);
	    qright[iVar] = h_qm(i  ,j  ,iVar);

	  }

	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes	  
	  swapValues(&(qleft[IU]) ,&(qleft[IV]) );
	  swapValues(&(qright[IU]),&(qright[IV]));
	  riemann<NVAR_2D>(qleft,qright,qgdnv,flux_y);
	
	  /*
	   * update with flux_y
	   */
	  if ( i < isize-ghostWidth and
	       j > ghostWidth       ) {
	    h_UNew(i  ,j-1,ID) -= flux_y[ID]*dtdy;
	    h_UNew(i  ,j-1,IP) -= flux_y[IP]*dtdy;
	    h_UNew(i  ,j-1,IU) -= flux_y[IV]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j-1,IV) -= flux_y[IU]*dtdy; // watchout IU and IV swapped
	  }
	
	  if ( i < isize-ghostWidth and 
	       j < jsize-ghostWidth ) {
	    h_UNew(i  ,j  ,ID) += flux_y[ID]*dtdy;
	    h_UNew(i  ,j  ,IP) += flux_y[IP]*dtdy;
	    h_UNew(i  ,j  ,IU) += flux_y[IV]*dtdy; // watchout IU and IV swapped
	    h_UNew(i  ,j  ,IV) += flux_y[IU]*dtdy; // watchout IU and IV swapped
	  }
	
	} // end for i
      } // end for j

      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(h_UNew, h_UOld, dt);
      }

    } else if (dimType == THREE_D) { // THREE_D - unsplit version 2

      TIMER_START(timerSlopeTrace);
      /*
       * 1. Compute and store slopes
       */
#ifdef _OPENMP
#pragma omp parallel default(shared)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=1; k<ksize-1; k++) {
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
      for (int k=1; k<ksize-1; k++) {
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
	    
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,k,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,k,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,k,IZ);
	    
	      qright[IU] += HALF_F * dt * h_gravity(i,j,k,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,k,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,k,IZ);
	    
	    } // end gravityEnabled
	
	    // store them
	    for ( int iVar=0; iVar<nbVar; iVar++ ) {
	    
	      h_qm(i  ,j  ,k  ,iVar) = qleft[iVar];
	      h_qp(i  ,j  ,k  ,iVar) = qright[iVar];
	    
	    }
	  
	  } // end for i
	} // end for j
      } // end for k
      TIMER_STOP(timerSlopeTrace);

      /*
       * 3. Riemann solver at X interface and update
       */
      for (int k=2; k<ksize-1; k++) {
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
		 k  < ksize-ghostWidth ) {
	      h_UNew(i-1,j  ,k  ,ID) -= flux_x[ID]*dtdx;
	      h_UNew(i-1,j  ,k  ,IP) -= flux_x[IP]*dtdx;
	      h_UNew(i-1,j  ,k  ,IU) -= flux_x[IU]*dtdx;
	      h_UNew(i-1,j  ,k  ,IV) -= flux_x[IV]*dtdx;
	      h_UNew(i-1,j  ,k  ,IW) -= flux_x[IW]*dtdx;
	    }
	  
	    if ( i  < isize-ghostWidth     and 
		 j  < jsize-ghostWidth     and
		 k  < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_x[ID]*dtdx;
	      h_UNew(i  ,j  ,k  ,IP) += flux_x[IP]*dtdx;
	      h_UNew(i  ,j  ,k  ,IU) += flux_x[IU]*dtdx;
	      h_UNew(i  ,j  ,k  ,IV) += flux_x[IV]*dtdx;
	      h_UNew(i  ,j  ,k  ,IW) += flux_x[IW]*dtdx;
	    }
	  
	  } // end for i
	} // end for j
      } // end for k

      /*
       * 4. Compute reconstructed states along Y interfaces
       */
      for (int k=1; k<ksize-1; k++) {
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
	    
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,k,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,k,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,k,IZ);
	    
	      qright[IU] += HALF_F * dt * h_gravity(i,j,k,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,k,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,k,IZ);
	    
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
      for (int k=2; k<ksize-1; k++) {
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
		 k  < ksize-ghostWidth ) {
	      h_UNew(i  ,j-1,k  ,ID) -= flux_y[ID]*dtdx;
	      h_UNew(i  ,j-1,k  ,IP) -= flux_y[IP]*dtdx;
	      h_UNew(i  ,j-1,k  ,IU) -= flux_y[IV]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IV) -= flux_y[IU]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j-1,k  ,IW) -= flux_y[IW]*dtdx;
	    }
	  
	    if ( i  < isize-ghostWidth     and
		 j  < jsize-ghostWidth     and
		 k  < ksize-ghostWidth     ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_y[ID]*dtdx;
	      h_UNew(i  ,j  ,k  ,IP) += flux_y[IP]*dtdx;
	      h_UNew(i  ,j  ,k  ,IU) += flux_y[IV]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,k  ,IV) += flux_y[IU]*dtdx; // watchout IU and IV swapped
	      h_UNew(i  ,j  ,k  ,IW) += flux_y[IW]*dtdx;
	    }
	  
	  } // end for j
	} // end for i
      } // end for k
    
      /*
       * 6. Compute reconstructed states along Z interfaces
       */
      for (int k=1; k<ksize-1; k++) {
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
	    
	      qleft[IU]  += HALF_F * dt * h_gravity(i,j,k,IX);
	      qleft[IV]  += HALF_F * dt * h_gravity(i,j,k,IY);
	      qleft[IW]  += HALF_F * dt * h_gravity(i,j,k,IZ);
	    
	      qright[IU] += HALF_F * dt * h_gravity(i,j,k,IX);
	      qright[IV] += HALF_F * dt * h_gravity(i,j,k,IY);
	      qright[IW] += HALF_F * dt * h_gravity(i,j,k,IZ);
	    
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
      for (int k=2; k<ksize-1; k++) {
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
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and
		 k > ghostWidth ) {
	      h_UNew(i  ,j  ,k-1,ID) -= flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,k-1,IP) -= flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,k-1,IU) -= flux_z[IW]*dtdz; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,k-1,IV) -= flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,k-1,IW) -= flux_z[IU]*dtdz; // watchout IU and IW swapped
	    }
	  
	    if ( i < isize-ghostWidth and 
		 j < jsize-ghostWidth and 
		 k < ksize-ghostWidth ) {
	      h_UNew(i  ,j  ,k  ,ID) += flux_z[ID]*dtdz;
	      h_UNew(i  ,j  ,k  ,IP) += flux_z[IP]*dtdz;
	      h_UNew(i  ,j  ,k  ,IU) += flux_z[IW]*dtdz; // watchout IU and IW swapped
	      h_UNew(i  ,j  ,k  ,IV) += flux_z[IV]*dtdz;
	      h_UNew(i  ,j  ,k  ,IW) += flux_z[IU]*dtdz; // watchout IU and IW swapped
	    }
	  
	  } // end for j
	} // end for i
      } // end for k

      // gravity source term
      if (gravityEnabled) {
	compute_gravity_source_term(h_UNew, h_UOld, dt);
      }

    } // end THREE_D  unsplit version 2

  } // HydroRunGodunovMpi::godunov_unsplit_cpu_v2
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  /*
   * main routine to start simulation.
   */
  void HydroRunGodunovMpi::start() {

    // test if we are performing a re-start run (default : false)
    bool restartEnabled = configMap.getBool("run","restart",false);

    // should we include ghost cells in output files ?
    bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);
    bool allghostIncluded = configMap.getBool("output","allghostIncluded",false);

    /*
     * initial condition.
     */
    int  nStep = 0;

    std::cout << "Initialization on MPI process " << myRank << std::endl;
    int configNb = configMap.getInteger("hydro" , "riemann_config_number", 0);
    setRiemannConfId(configNb);
    nStep = init_simulation(problem);

    // initialize ghost borders of all MPI blocks
    // needed if the initialization routine called above does not take
    // care of the ghost zones. 
    if (restartEnabled and (ghostIncluded or allghostIncluded)) {

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

    } // end if (restartEnabled and allghostIncluded)

    if (myRank == 0) {
      std::cout << "Trace computation enabled : " << traceEnabled << std::endl;
      if (traceEnabled)
	std::cout << "Trace computation version : " << traceVersion << std::endl;
    }

    if (myRank == 0) {
      std::cout << "Starting time integration (Godunov scheme) on MPI process " << myRank << std::endl;
      if (unsplitEnabled)
	std::cout << "use unsplit integration version : " << unsplitVersion << std::endl;
      else
	std::cout << "use directional spliting integration" << std::endl;
      std::cout << "Resolution per MPI process (nx,ny,nz) " << nx << " " << ny << " " << nz << std::endl;
      std::cout << "MPI grid sizes (mx, my, mz) " << mx  << " " << my << " " << mz << std::endl;
      std::cout << "Trace computation enabled : " << traceEnabled << std::endl;
      if (traceEnabled)
	std::cout << "Trace computation version : " << traceVersion << std::endl;
    }

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

    // Do we want to dump faces of domain ?
    int nOutputFaces = configMap.getInteger("run", "nOutputFaces", -1);
    bool outputFacesEnabled = false;
    if (nOutputFaces>0 and dimType==THREE_D)
      outputFacesEnabled = true;
    bool outputFacesPnetcdfEnabled = configMap.getBool("output", "outputFacesPnetcdf", false);

    // memory allocation for face buffers (in theory not all MPI tasks need to
    // do that)
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

	// just some log 
	if (myRank == 0 and nLog>0 and (nStep % nLog) == 0) {
	  
	  std::cout << "["        << current_date()       << "]"
		    << "  step="  << std::setw(9)         << nStep 
		    << " t="      << fmt(totalTime)
		    << " dt="     << fmt(dt)              << std::endl;
	  
	} // end log

	// Output results
	if((nStep % nOutput)==0) {
	  
	  // call timing routines for output results
	  timerWriteOnDisk.start();

	  // make sure Device data are copied back onto Host memory
	  // which data to save ?
	  copyGpuToCpu(nStep);

	  output(getDataHost(nStep), nStep);

	  timerWriteOnDisk.stop();

	  if (myRank == 0) {
	    std::cout << "["        << current_date()       << "]"
		      << "  step="  << std::setw(9)         << nStep 
		      << " t="      << fmt(totalTime)
		      << " dt="     << fmt(dt)              
		      << " output " << std::endl;
	  }
	} // end output results
	
	/* Output faces results */
	if(outputFacesEnabled) {
	  if ( (nStep % nOutputFaces)==0 ) {
	    
	    outputFaces(nStep, outputFacesPnetcdfEnabled);
	    
	  }
	} // end output faces

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

      } // end while (t < tEnd && nStep < nStepmax)

    // output last time step
    {
      if (myRank == 0) printf("Final output at step %d\n",nStep);
      timerWriteOnDisk.start();
      copyGpuToCpu(nStep);
      output(getDataHost(nStep), nStep);
      timerWriteOnDisk.stop();
    }

    // write Xdmf wrapper file
    if (outputHdf5Enabled and myRank==0) writeXdmfForHdf5Wrapper(nStep);

    // final timing report
    timerTotal.stop();

    printf("Euler godunov total  time             [MPI rank %3d] : %5.3f sec\n", myRank, timerTotal.elapsed());
    printf("Euler godunov output time             [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerWriteOnDisk.elapsed(), timerWriteOnDisk.elapsed()/timerTotal.elapsed()*100.);

    /*
     * print timing report if required
     */
#ifdef DO_TIMING
#ifdef __CUDACC__
    printf("Euler godunov boundaries pure GPU     [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesGpu.elapsed(), timerBoundariesGpu.elapsed()/timerTotal.elapsed()*100.);
    printf("Euler godunov boundaries CPU-GPU comm [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesCpuGpu.elapsed(), timerBoundariesCpuGpu.elapsed()/timerTotal.elapsed()*100.);
    printf("Euler godunov boundaries     MPI comm [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesMpi.elapsed(), timerBoundariesMpi.elapsed()/timerTotal.elapsed()*100.);
    printf("Euler godunov computing               [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerGodunov.elapsed(), timerGodunov.elapsed()/timerTotal.elapsed()*100.);
#else
    printf("Euler godunov boundaries pure CPU     [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesCpu.elapsed(), timerBoundariesCpu.elapsed()/timerTotal.elapsed()*100.);
    printf("Euler godunov boundaries     MPI comm [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesMpi.elapsed(), timerBoundariesMpi.elapsed()/timerTotal.elapsed()*100.);
    printf("Euler godunov computing               [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerGodunov.elapsed(), timerGodunov.elapsed()/timerTotal.elapsed()*100.);
#endif // __CUDACC__

    if (unsplitVersion == 1 or unsplitVersion == 2) {
      
      printf("Euler hydro prim var                  [MPI rank %3d] : %5.3f sec (%5.2f %% of computing time)\n", myRank, timerPrimVar.elapsed(), timerPrimVar.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler hydro slope/trace               [MPI rank %3d] : %5.3f sec (%5.2f %% of computing time)\n", myRank, timerSlopeTrace.elapsed(), timerSlopeTrace.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler hydro update                    [MPI rank %3d] : %5.3f sec (%5.2f %% of computing time)\n", myRank, timerUpdate.elapsed(), timerUpdate.elapsed()/timerGodunov.elapsed()*100.);
      printf("Euler dissipative terms               [MPI rank %3d] : %5.3f sec (%5.2f %% of computing time)\n", myRank, timerDissipative.elapsed(), timerDissipative.elapsed()/timerGodunov.elapsed()*100.);
      
    } 

    printf("History time                          [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerHistory.elapsed(), timerHistory.elapsed()/timerTotal.elapsed()*100.);

#endif // DO_TIMING

    if (myRank==0)
      std::cout  << "####################################\n"
		 << "Global perfomance over all MPI proc \n" 
		 << 1.0*nStep*(nx*mx)*(ny*my)*(nz*mz)/(timerTotal.elapsed()-timerWriteOnDisk.elapsed())
		 << " cell updates per seconds (based on wall time)\n"
		 << "####################################\n";
    
  } // HydroRunGodunovMpi::start

  // =======================================================
  // =======================================================
  /*
   * main routine to start simulation.
   */
  void HydroRunGodunovMpi::oneStepIntegration(int& nStep, real_t& t, real_t& dt) {
    
    if (unsplitEnabled) { // unsplit Godunov integration
      
      // if nStep is even update U  into U2
      // if nStep is odd  update U2 into U
      dt=compute_dt(nStep % 2);
      godunov_unsplit(nStep, dt);

    } else { // directional spliting Godunov integration
     
      // when do we need to re-compute dt ?
      if (dimType == TWO_D) {
	
	/* compute new time-step */
	if ((nStep%2)==0) {
	  dt=compute_dt(); // always use h_U (or d_U)
	  if(nStep==0)
	    dt=dt/2.0;
	}

      } else { // THREE_D

	/* compute new time-step */
	if ((nStep%6)==0) { // current data are in h_U (or d_U)
	  dt=compute_dt(0);
	  if(nStep==0)
	    dt=dt/3.0;
	} else if ((nStep%6)==3) { // current data are in h_U2 (or d_U2)
	  dt=compute_dt(1);
	}

      } // end THREE_D

      /* Directional splitting computations */
      godunov_split(nStep, dt);

    } // end unsplitEnabled

    // increment time
    nStep++;
    t+=dt;
  
  } // HydroRunGodunovMpi::oneStepIntegration

  // =======================================================
  // =======================================================
  /*
   * convert to primitive variables.
   */
  void HydroRunGodunovMpi::convertToPrimitives(real_t *U)
  {

#ifdef __CUDACC__

    if (dimType == TWO_D) {

      TIMER_START(timerPrimVar);
      {
	// 2D primitive variables computation kernel    
	dim3 dimBlock(PRIM_VAR_BLOCK_DIMX_2D,
		      PRIM_VAR_BLOCK_DIMY_2D);
	dim3 dimGrid(blocksFor(isize, PRIM_VAR_BLOCK_DIMX_2D), 
		     blocksFor(jsize, PRIM_VAR_BLOCK_DIMY_2D));
	kernel_hydro_compute_primitive_variables_2D<<<dimGrid, 
	  dimBlock>>>(U, 
		      d_Q.data(),
		      d_Q.pitch(),
		      d_Q.dimx(),
		      d_Q.dimy());
	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_compute_primitive_variables_2D error",myRank);
      
      } // end compute primitive variables 2d kernel
      TIMER_STOP(timerPrimVar);

    } else { // THREE_D

      TIMER_START(timerPrimVar);
      {
	// 3D primitive variables computation kernel    
	dim3 dimBlock(PRIM_VAR_BLOCK_DIMX_3D,
		      PRIM_VAR_BLOCK_DIMY_3D);
	dim3 dimGrid(blocksFor(isize, PRIM_VAR_BLOCK_DIMX_3D), 
		     blocksFor(jsize, PRIM_VAR_BLOCK_DIMY_3D));
	kernel_hydro_compute_primitive_variables_3D<<<dimGrid, 
	  dimBlock>>>(U, 
		      d_Q.data(),
		      d_Q.pitch(),
		      d_Q.dimx(),
		      d_Q.dimy(),
		      d_Q.dimz());
	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_compute_primitive_variables_3D error",myRank);
      
      } // end compute primitive variables 3d kernel
      TIMER_STOP(timerPrimVar);

    } // end THREE_D
  
#else // CPU version

    if (dimType == TWO_D) {
  
      // primitive variable domain array
      real_t *Q = h_Q.data();
    
      // section / domain size
      int arraySize = h_Q.section();
    
      // update primitive variables array
#ifdef _OPENMP
#pragma omp parallel default(shared) private(arraySize)
#pragma omp for collapse(2) schedule(auto)
#endif // _OPENMP
      for (int j=0; j<jsize; j++) {
	for (int i=0; i<isize; i++) {
	
	  // primitive variable state vector
	  real_t q[NVAR_2D];

	  int indexLoc = i+j*isize;
	  real_t c;
	
	  computePrimitives_0(U, arraySize, indexLoc, c, q);
	
	  // copy q state in h_Q
	  int offset = indexLoc;
	  Q[offset] = q[ID]; offset += arraySize;
	  Q[offset] = q[IP]; offset += arraySize;
	  Q[offset] = q[IU]; offset += arraySize;
	  Q[offset] = q[IV];
	
	} // end for i
      } // end for j
    
    } else { // THREE_D
    
      /*int physicalDim[3] = {(int) h_Q.pitch(),
	(int) h_Q.dimy(),
	(int) h_Q.dimz()};*/
        
      // primitive variable domain array
      real_t *Q = h_Q.data();
    
      // section / domain size
      int arraySize = h_Q.section();
    
      // update primitive variables array
#ifdef _OPENMP
#pragma omp parallel default(shared) private(arraySize)
#pragma omp for collapse(3) schedule(auto)
#endif // _OPENMP
      for (int k=0; k<ksize; k++) {
	for (int j=0; j<jsize; j++) {
	  for (int i=0; i<isize; i++) {
	  
	    // primitive variable state vector
	    real_t q[NVAR_3D];

	    int indexLoc = i+j*isize+k*isize*jsize;
	    real_t c;
	  
	    computePrimitives_3D_0(U, arraySize, indexLoc, c, q);
	  
	    // copy q state in h_Q
	    int offset = indexLoc;
	    Q[offset] = q[ID]; offset += arraySize;
	    Q[offset] = q[IP]; offset += arraySize;
	    Q[offset] = q[IU]; offset += arraySize;
	    Q[offset] = q[IV]; offset += arraySize;
	    Q[offset] = q[IW];
	  
	  } // end for i
	} // end for j
      } // end for k
    
    } // end THREE_D
  
#endif // __CUDACC__

  } // HydroRunGodunovMpi::convertToPrimitives

} // namespace hydroSimu
