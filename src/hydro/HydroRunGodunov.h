/**
 * \file HydroRunGodunov.h
 * \brief
 * This file was originally designed by F. Chateau and called HydroRun.h
 * In the process of removing the coupling between fortran and Cuda; the 
 * original file HydroRun was splitted into HydroRunBase class, so that we
 * can derive from it and have access to base routines (boundary
 * condition, output, etc...) inside the derived class which implements
 * the actual computational scheme.
 *
 * The new class HydroRunGodunov derives from HydroRunBase, and
 * implement Godunov's scheme by solving local Riemann problem's at
 * cells interfaces.
 *
 * Further (in October 2010), this file is redesigned to handle both MPI+Cuda 
 * computations. See class HydroRunGodunovMpi.
 * 
 * \author F. Chateau, P. Kestener
 *
 * $Id: HydroRunGodunov.h 3322 2014-03-06 14:53:47Z pkestene $
 */
#ifndef HYDRORUN_GODUNOV_H_
#define HYDRORUN_GODUNOV_H_

#include "real_type.h"
#include "common_types.h"
#include "gpu_macros.h"
#include <cmath>

#include "HydroRunBase.h"

#include "../utils/monitoring/measure_time.h"
#ifdef DO_TIMING
#ifdef __CUDACC__
#include "../utils/monitoring/CudaTimer.h"
#else
#include "../utils/monitoring/Timer.h"
#endif // __CUDACC__
#endif // DO_TIMING

// PAPI support
#ifndef __CUDACC__
#ifdef USE_PAPI
#include "../utils/monitoring/PapiInfo.h"
#endif // USE_PAPI
#endif // __CUDACC__

namespace hydroSimu {

  /**
   * \class HydroRunGodunov HydroRunGodunov.h
   * \brief This class implements hydro simulations using the Godunov numerical
   * scheme.
   * 
   * This class implements different Godunov-based numerical schemes
   * for solving Euler's equations: directionally split and unsplit.
   * Note that in the unsplit case, there are 2 implementations that
   * differ from the amount of required memory; see the following
   * note.
   *
   * \par Unsplit scheme version 0
   * only UOld and UNew are stored, other variables are
   * computed as needed. Algorithm is a single for loop over the
   * domain which calls multiple times local routine trace_unsplit
   * in a large stencil, then calls the riemann routine, and
   * finally performs hydro update<BR>
   * Please note that here trace computations include slopes computations.<BR>
   * available on CPU (2D + 3D)<BR>
   * available on GPU (2D + 3D))
   *
   * \par Unsplit scheme version 1
   * backported using ideas from class MHDRunGodunov (implementation
   * version 4).
   *
   */
  class HydroRunGodunov : public HydroRunBase
  {
  public:
    HydroRunGodunov(ConfigMap &_configMap);
    ~HydroRunGodunov();
       
    //! Godunov integration with directionnal spliting
    void godunov_split(int idim, real_t dt);

    //! Godunov integration using unsplit scheme (brut force, compute
    //! everything that is needed to have the flux update for each
    //! cell without taking care of the fact that some computations
    //! from neighboring cells might be reused.
    //! This routine is just a wrapper to the actual computation done.
    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
    void godunov_unsplit(int nStep, real_t dt);

  private:
#ifdef __CUDACC__
    //! Actual computation of the godunov integration on GPU using
    //! directionally split scheme. 
    void godunov_split_gpu(DeviceArray<real_t>& d_UOld, 
			   DeviceArray<real_t>& d_UNew,
			   int idim, 
			   real_t dt);

    //! Actual computation of the godunov integration using unsplit 
    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
    //! d_UNew are swapped after each iteration).
    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
			     DeviceArray<real_t>& d_UNew,
			     real_t dt, int nStep);
#else
    //! Actual computation of the godunov integration on CPU using
    //! directionally split scheme. 
    void godunov_split_cpu(HostArray<real_t>& h_UOld, 
			   HostArray<real_t>& h_UNew,
			   int idim, 
			   real_t dt);

    //! Actual computation of the godunov integration using unsplit 
    //! scheme on CPU, two array are necessary to make ping-pong (h_UOld and
    //! h_UNew are swapped after each iteration).
    void godunov_unsplit_cpu(HostArray<real_t>& h_UOld, 
			     HostArray<real_t>& h_UNew, 
			     real_t dt, int nStep);

    //! unplitVersion = 0
    //! memory footprint is very low
    //! nothing is stored globally except h_Q
    //! some redundancy in trace computation
    void godunov_unsplit_cpu_v0(HostArray<real_t>& h_UOld, 
				HostArray<real_t>& h_UNew, 
				real_t dt, int nStep);

    //! unplitVersion = 1
    //! memory footprint is medium
    //! reconstructed (trace) states are stored
    //! then perform Riemann flux computation and update
    void godunov_unsplit_cpu_v1(HostArray<real_t>& h_UOld, 
				HostArray<real_t>& h_UNew, 
				real_t dt, int nStep);

    //! unplitVersion = 2
    //! memory footprint is larger than unplitVersion 2
    //! slopes are stored and reconstructed (trace) states computed
    //! as needed
    void godunov_unsplit_cpu_v2(HostArray<real_t>& h_UOld, 
				HostArray<real_t>& h_UNew, 
				real_t dt, int nStep);

#endif // __CUDAC__

  public:
    //! start integration and control output
    void start();

    //! perform only one time step integration
    void oneStepIntegration(int& nStep, real_t& t, real_t& dt);

    //! do we enable trace/slope computations
    void setTraceEnabled(bool _traceEnabled) {traceEnabled = _traceEnabled;};
    
    //! chose trace computation version
    void setTraceVersion(int _traceVersion)  {traceVersion = _traceVersion;};
    
    //! get unplitEnabled bool
    bool getUnsplitEnabled() {return unsplitEnabled;};

  private:

    void convertToPrimitives(real_t *U);

    /** do trace computations */
    bool traceEnabled;
    int  traceVersion;

    /** use unsplit scheme */
    bool unsplitEnabled;
    int  unsplitVersion;

#ifdef __CUDACC__
    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
#else
    HostArray<real_t>   h_Q; //!< !!! CPU ONLY !!! Primitive Data array on CPU
#endif // __CUDACC__

    /** \defgroup implementation1 */
    /*@{*/
#ifdef __CUDACC__
    DeviceArray<real_t> d_qm_x; //!< GPU array for qm state along X
    DeviceArray<real_t> d_qm_y; //!< GPU array for qm state along Y
    DeviceArray<real_t> d_qm_z; //!< GPU array for qm state along Z

    DeviceArray<real_t> d_qp_x; //!< GPU array for qp state along X
    DeviceArray<real_t> d_qp_y; //!< GPU array for qp state along Y
    DeviceArray<real_t> d_qp_z; //!< GPU array for qp state along Z
#else
    HostArray<real_t> h_qm_x;
    HostArray<real_t> h_qm_y;
    HostArray<real_t> h_qm_z;

    HostArray<real_t> h_qp_x;
    HostArray<real_t> h_qp_y;
    HostArray<real_t> h_qp_z;
#endif // __CUDACC__
    /*@}*/

    bool dumpDataForDebugEnabled; /*!< if true, dump almost all intermediate data buffer to a file */

#ifdef DO_TIMING
    /** \defgroup timing monitoring computation time */
    /*@{*/
#ifdef __CUDACC__
    CudaTimer timerBoundaries;
    CudaTimer timerGodunov;

    // other timers
    CudaTimer timerPrimVar;
    CudaTimer timerSlopeTrace;
    CudaTimer timerUpdate;
    CudaTimer timerDissipative;
#else
    Timer     timerBoundaries;
    Timer     timerGodunov;

    // other timers
    Timer     timerPrimVar;
    Timer     timerSlopeTrace;
    Timer     timerUpdate;
    Timer     timerDissipative;

#ifdef USE_PAPI
    // papi counter (cpu only)
    PapiInfo  papiFlops_total;
#endif // USE_PAPI

#endif // __CUDACC__
    /*@}*/
#endif // DO_TIMING

  }; // class HydroRunGodunov
  
} // namespace hydroSimu

#endif /*HYDRORUN_GODUNOV_H_*/
