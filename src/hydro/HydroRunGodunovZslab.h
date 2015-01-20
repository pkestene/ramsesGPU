/**
 * \file HydroRunGodunovZslab.h
 * \brief Godunov Hydro run simulation with z-slab method.
 *
 * The new class HydroRunGodunovZslab derives from HydroRunBase, and
 * implement Godunov's scheme by solving local Riemann problem's at
 * cells interfaces and using a piece-by-piece method (z-slab).
 *
 * \date September 11, 2012
 * \author P. Kestener
 *
 * $Id: HydroRunGodunovZslab.h 2431 2012-09-27 13:42:20Z pkestene $
 */
#ifndef HYDRORUN_GODUNOV_ZSLAB_H_
#define HYDRORUN_GODUNOV_ZSLAB_H_

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

namespace hydroSimu {

  /**
   * \class HydroRunGodunovZslab HydroRunGodunovZslab.h
   * \brief This class implements hydro simulations using the Godunov numerical
   * scheme with the z-slab method.
   * 
   * This class implements only an unsplit Godunov-based numerical scheme
   * for solving 3D Euler's equations; but uses a z-slab method.
   *
   * The z-slab method consists in using sub-domain slab of size
   * (nx,ny,z-slab_size) and updating hydro variables
   * slab-by-slab, so that intermediate variables array are sized upon
   * a zslab piece instead of the whole domain.
   *
   */
  class HydroRunGodunovZslab : public HydroRunBase
  {
  public:
    HydroRunGodunovZslab(ConfigMap &_configMap);
    ~HydroRunGodunovZslab();
       
    //! Godunov integration using unsplit scheme (brut force, compute
    //! everything that is needed to have the flux update for each
    //! cell without taking care of the fact that some computations
    //! from neighboring cells might be reused.
    //! This routine is just a wrapper to the actual computation done.
    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
    void godunov_unsplit(int nStep, real_t dt);

  private:
#ifdef __CUDACC__
    //! Actual computation of the godunov integration using unsplit 
    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
    //! d_UNew are swapped after each iteration).
    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
			     DeviceArray<real_t>& d_UNew,
			     real_t dt, int nStep);
#else
    //! Actual computation of the godunov integration using unsplit 
    //! scheme on CPU, two array are necessary to make ping-pong (h_UOld and
    //! h_UNew are swapped after each iteration).
    void godunov_unsplit_cpu(HostArray<real_t>& h_UOld, 
			     HostArray<real_t>& h_UNew, 
			     real_t dt, int nStep);
#endif // __CUDAC__

  public:
    //! start integration and control output
    void start();

    //! perform only one time step integration
    void oneStepIntegration(int& nStep, real_t& t, real_t& dt);
    
  private:

    //! number of z-slabs
    int zSlabNb;

    //! z-slab width (or thickness along z)
    int zSlabWidth;

    //! z-slab width with ghost
    int zSlabWidthG;

    void convertToPrimitives(real_t *U, int zSlabId);
       
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
#endif // __CUDACC__
    /*@}*/
#endif // DO_TIMING

  }; // class HydroRunGodunovZslab
  
} // namespace hydroSimu

#endif /*HYDRORUN_GODUNOV_ZSLAB_H_*/
