/**
 * \file HydroRunGodunovMpi.h
 * \brief
 * This class is a direct transposition of HydroRunGodunov but can handle
 * MPI+CUDA computations.
 * 
 * \date 19 Oct 2010
 * \author Pierre Kestener
 *
 * $Id: HydroRunGodunovMpi.h 2431 2012-09-27 13:42:20Z pkestene $
 */
#ifndef HYDRORUN_GODUNOV_MPI_H_
#define HYDRORUN_GODUNOV_MPI_H_

#include "real_type.h"
#include "common_types.h"
#include "gpu_macros.h"
#include <cmath>

#include "HydroRunBaseMpi.h"

namespace hydroSimu {

  /**
   * \class HydroRunGodunovMpi HydroRunGodunovMpi.h
   * \brief This class implements hydro simulations using the Godunov numerical
   * scheme (with MPI parallelization).
   * 
   * This class implements different Godunov-based numerical schemes
   * for solving Euler's equations: directionally split and unsplit.
   * Note that in the unsplit case, there are 2 implementations that
   * differ from the amount of required memory and performance
   * (unsplit version 1 is the best one); 
   *
   * \see HydroRunGodunov class which explains differences between the
   * 2 unsplit versions.
   *
   */
  class HydroRunGodunovMpi : public HydroRunBaseMpi
  {
  public:
    HydroRunGodunovMpi(ConfigMap &_configMap);
    ~HydroRunGodunovMpi();
    
    //! Godunov integration with directionnal spliting
    void godunov_split(int idim, real_t dt);

    //! Godunov integration using unsplit scheme (brut force, compute
    //! everything that is needed to have the flux update for each
    //! cell without taking care of the fact that some computations
    //! from neighboring cells might be reused.
    //! This routine is just a wrapper to the actual computation done.
    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
    void godunov_unsplit(int nStep, real_t dt);

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
    //! slopes are stored and only 1 pair of reconstructed (trace) states
    void godunov_unsplit_cpu_v2(HostArray<real_t>& h_UOld, 
				HostArray<real_t>& h_UNew, 
				real_t dt, int nStep);

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
			     real_t dt);
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
			     real_t dt);
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

    /** \defgroup implementation2 */
    /*@{*/
#ifdef __CUDACC__
    DeviceArray<real_t> d_qm;      //!< only for unsplit version 2
    DeviceArray<real_t> d_qp;      //!< only for unsplit version 2
    DeviceArray<real_t> d_slope_x; //!< only for unsplit version 2
    DeviceArray<real_t> d_slope_y; //!< only for unsplit version 2
    DeviceArray<real_t> d_slope_z; //!< only for unsplit version 2
#else
    HostArray<real_t> h_qm;      //!< only for unsplit version 2
    HostArray<real_t> h_qp;      //!< only for unsplit version 2
    HostArray<real_t> h_slope_x; //!< only for unsplit version 2
    HostArray<real_t> h_slope_y; //!< only for unsplit version 2
    HostArray<real_t> h_slope_z; //!< only for unsplit version 2
#endif // __CUDACC__
    /*@}*/

    /*
     * please note that timer for boundaries computation are declared
     * in HydroRunBaseMpi.h
     */
#ifdef DO_TIMING
#ifdef __CUDACC__
    CudaTimer timerGodunov;
    // other timers
    CudaTimer timerPrimVar;
    CudaTimer timerSlopeTrace;
    CudaTimer timerUpdate;
    CudaTimer timerDissipative;
#else
    Timer     timerGodunov;
    // other timers
    Timer     timerPrimVar;
    Timer     timerSlopeTrace;
    Timer     timerUpdate;
    Timer     timerDissipative;
#endif // __CUDACC__
#endif // DO_TIMING

  }; // class HydroRunGodunovMpi
  
} // namespace hydroSimu

#endif /*HYDRORUN_GODUNOV_MPI_H_*/
