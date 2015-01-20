/**
 * \file HydroRunRelaxingTVD.h
 * \brief Implement the relaxing TVD fluid solver by Trac and Pen.
 *
 * The relaxing TVD fluid solver scheme is described in:
 * "A primer on Eulerian Computational Fluid Dynamics for
 * Astrophysics", by H. trac and U.-L. Pen, Publications of the
 * astronomical society of the pacific, vol 115, pp. 303-321, 2003.
 *
 * The scheme is claimed to be significantly faster that other MUSCL
 * Riemann solver scheme since the relaxing TVD method do not require
 * Riemann problem solving to determine the direction upwind of the
 * flow, but instead make a change of variables to express right and
 * left moving flow.
 *
 * \date 24-Jan-2011
 * \author P. Kestener
 *
 * $Id: HydroRunRelaxingTVD.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef HYDRORUN_RELAXING_TVD_H_
#define HYDRORUN_RELAXING_TVD_H_

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
   * \class HydroRunRelaxingTVD HydroRunRelaxingTVD.h
   * \brief This class implements hydro simulations using the relaxing TVD numerical
   * scheme.
   * 
   */
  class HydroRunRelaxingTVD : public HydroRunBase
  {
  public:
    HydroRunRelaxingTVD(ConfigMap &_configMap);
    ~HydroRunRelaxingTVD();
    
    //!  wrapper that performs the actual CPU or GPU scheme
    //!  integration step.
    void relaxing_tvd_sweep(int nStep, real_t dt);
    
  private:
#ifdef __CUDACC__
    //! Actual computation of the relaxing TVD integration on GPU using
    //! a directionally split method. 
    void relaxing_tvd_gpu(DeviceArray<real_t>& d_UOld, 
			  DeviceArray<real_t>& d_UNew,
			  int idim, 
			  real_t dt);
#else
    //! Actual computation of the relaxing TVD integration on CPU using
    //! a directionally split method. 
    void relaxing_tvd_cpu(HostArray<real_t>& h_UOld, 
			  HostArray<real_t>& h_UNew,
			  int idim, 
			  real_t dt);    
#endif // __CUDACC__

  public:
    //! start integration and control output.
    void start();

    //! perform only one time step integration.
    void oneStepIntegration(int& nStep, real_t& t, real_t& dt);
 
  private:
    
#ifdef DO_TIMING
    /** \defgroup timing monitoring computation time */
    /*@{*/
#ifdef __CUDACC__
    CudaTimer timerBoundaries;
    CudaTimer timerRelaxingTVD;
#else
    Timer     timerBoundaries;
    Timer     timerRelaxingTVD;
#endif // __CUDACC__
    /*@}*/
#endif // DO_TIMING
    
  }; // class HydroRunRelaxingTVD

} // namespace hydroSimu

#endif /* HYDRORUN_RELAXING_TVD_H_ */
