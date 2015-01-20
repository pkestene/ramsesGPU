/**
 * \file MHDRunGodunovZslab.h
 * \brief Define class MHDRunGodunovZslab (see MHDRunGodunov) containing
 * the actual numerical scheme for solving MHD with z-slab method.
 *
 * \date Sept 17, 2012
 * \author Pierre Kestener
 *
 * $Id: MHDRunGodunovZslab.h 2410 2012-09-19 09:24:52Z pkestene $
 */
#ifndef MHD_RUN_GODUNOV_ZSLAB_H_
#define MHD_RUN_GODUNOV_ZSLAB_H_

// base class
#include "MHDRunBase.h"

// timing macros
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
   * \class MHDRunGodunovZslab MHDRunGodunovZslab.h
   * \brief This class implements MHD simulations using the Godunov numerical
   * scheme.
   * 
   * This class is similar to HydroRunGodunovZslab.
   *
   * Compared to MHDRunGodunov, only version 4 is implemented with the z-slab way.
   */
  class MHDRunGodunovZslab : public MHDRunBase
  {
  public:
    MHDRunGodunovZslab(ConfigMap &_configMap);
    ~MHDRunGodunovZslab();

    //! convert conservative variables (h_U or h_U2) into primitive
    //! var h_Q and take care of source term predictor.
    //!
    //! This routine is only used in the CPU version of the code.
    //! In the GPU version, the conversion is done on line, inside
    //! kernel as needed.
    //! \param[in] U conservative variable domain array.
    //! \param[in] timeStep time step is needed for predictor computations.
    //! \param[in] zSlabId z-slab index.
    void convertToPrimitives(real_t *U, real_t timeStep, int zSlabId);

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

    /*
     * Numerical scheme for rotating frames
     */
#ifdef __CUDACC__
    //! Rotating frame (Omega0 > 0)
    //! Same routine as godunov_unsplit_gpu but with rotating
    //! frame corrections implemented
    void godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
				      DeviceArray<real_t>& d_UNew,
				      real_t dt, int nStep);
#else
    //! Rotating frame (Omega0 > 0)
    //! Same routine as godunov_unsplit_cpu but with rotating
    //! frame corrections implemented
    void godunov_unsplit_rotating_cpu(HostArray<real_t>& h_UOld, 
				      HostArray<real_t>& h_UNew, 
				      real_t dt, int nStep);
#endif // __CUDACC__

  public:

#ifdef __CUDACC__
    //!  implement shearing border condition
    virtual void make_boundaries_shear(DeviceArray<real_t> &U, real_t dt, int nStep);
#else
    //!  implement shearing border condition
    virtual void make_boundaries_shear(HostArray<real_t>   &U, real_t dt, int nStep);
#endif // __CUDACC__

#ifdef __CUDACC__
    //!  implement shearing border condition
    virtual void make_all_boundaries_shear(DeviceArray<real_t> &U, real_t dt, int nStep);
#else
    //!  implement shearing border condition
    virtual void make_all_boundaries_shear(HostArray<real_t>   &U, real_t dt, int nStep);
#endif // __CUDACC__

    //! start integration and control output
    void start();
    
    //! perform only one time step integration
    void oneStepIntegration(int& nStep, real_t& t, real_t& dt);

    //! shearing box enabled ?
    bool shearingBoxEnabled;

  private:

    //! number of z-slabs
    int zSlabNb;

    //! z-slab width (or thickness along z)
    int zSlabWidth;

    //! z-slab width with ghost
    int zSlabWidthG;

    /** which implementation to use (currently possible values are 0, 1 and 2) */
    int implementationVersion;

    /** if true, dump almost all intermediate data buffer to a file */
    bool dumpDataForDebugEnabled;

    /** if true, enable some special debug (data dump) of shearing box data */
    bool debugShear;

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

    DeviceArray<real_t> d_qEdge_RT;
    DeviceArray<real_t> d_qEdge_RB;
    DeviceArray<real_t> d_qEdge_LT;
    DeviceArray<real_t> d_qEdge_LB;

    DeviceArray<real_t> d_qEdge_RT2;
    DeviceArray<real_t> d_qEdge_RB2;
    DeviceArray<real_t> d_qEdge_LT2;
    DeviceArray<real_t> d_qEdge_LB2;

    DeviceArray<real_t> d_qEdge_RT3;
    DeviceArray<real_t> d_qEdge_RB3;
    DeviceArray<real_t> d_qEdge_LT3;
    DeviceArray<real_t> d_qEdge_LB3;

    DeviceArray<real_t> d_emf; //!< GPU array for electromotive forces
#else
    HostArray<real_t> h_qm_x;
    HostArray<real_t> h_qm_y;
    HostArray<real_t> h_qm_z;

    HostArray<real_t> h_qp_x;
    HostArray<real_t> h_qp_y;
    HostArray<real_t> h_qp_z;

    HostArray<real_t> h_qEdge_RT;
    HostArray<real_t> h_qEdge_RB;
    HostArray<real_t> h_qEdge_LT;
    HostArray<real_t> h_qEdge_LB;

    HostArray<real_t> h_qEdge_RT2;
    HostArray<real_t> h_qEdge_RB2;
    HostArray<real_t> h_qEdge_LT2;
    HostArray<real_t> h_qEdge_LB2;

    HostArray<real_t> h_qEdge_RT3;
    HostArray<real_t> h_qEdge_RB3;
    HostArray<real_t> h_qEdge_LT3;
    HostArray<real_t> h_qEdge_LB3;

    HostArray<real_t> h_emf; //!< CPU array for electromotive forces
#endif // __CUDACC__
    /*@}*/

    /** \defgroup implementation2 */
    /*@{*/
#ifdef __CUDACC__
    DeviceArray<real_t> d_elec; /*!< Device array storing electric field (only
                                used in implementation version 2, 3 */
    DeviceArray<real_t> d_dA; /*!< Device array storing magnetic field slopes along X */
    DeviceArray<real_t> d_dB; /*!< Device array storing magnetic field slopes along Y */
    DeviceArray<real_t> d_dC; /*!< Device array storing magnetic field slopes along Z */
#else // CPU version
    HostArray<real_t> h_elec; /*!< Host array storing electric field (only
                                used in implementation version 2, 3 */
    HostArray<real_t> h_dA; /*!< Host array storing magnetic field slopes along X */
    HostArray<real_t> h_dB; /*!< Host array storing magnetic field slopes along Y */
    HostArray<real_t> h_dC; /*!< Host array storing magnetic field slopes along Z */

#endif // __CUDACC__
    /*@}*/


    // shearing box specific data
    /** \defgroup shearing_box */
    /*@{*/

    /* notice that in the MPI version, only x-border MPI process will 
     * need those buffers */
#ifdef __CUDACC__
    DeviceArray<real_t> d_shear_flux_xmin;    /*!< flux correction data at XMIN */
    DeviceArray<real_t> d_shear_flux_xmax;    /*!< flux correction data at XMAX */
    DeviceArray<real_t> d_shear_flux_xmin_remap;    /*!< flux correction data at XMIN */
    DeviceArray<real_t> d_shear_flux_xmax_remap;    /*!< flux correction data at XMAX */

    DeviceArray<real_t> d_shear_border_xmin;  /*! shearing box border buffer */
    DeviceArray<real_t> d_shear_border_xmax;  /*! shearing box border buffer */

    DeviceArray<real_t> d_shear_slope_xmin;   /*! slopes in XMIN border */
    DeviceArray<real_t> d_shear_slope_xmax;   /*! slopes in XMAX border */
#else
    HostArray<real_t>   h_shear_flux_xmin;    /*!< flux correction data at XMIN */
    HostArray<real_t>   h_shear_flux_xmax;    /*!< flux correction data at XMAX */
    HostArray<real_t>   h_shear_flux_xmin_remap;    /*!< flux correction data at XMIN */
    HostArray<real_t>   h_shear_flux_xmax_remap;    /*!< flux correction data at XMAX */

    HostArray<real_t>   h_shear_border_xmin;   /*! shearing box border buffer */
    HostArray<real_t>   h_shear_border_xmax;   /*! shearing box border buffer */

    HostArray<real_t>   h_shear_slope_xmin;    /*! slopes in XMIN border */
    HostArray<real_t>   h_shear_slope_xmax;    /*! slopes in XMAX border */
#endif // __CUDACC__

    /*@}*/
    
    
#ifdef DO_TIMING
    /** \defgroup timing monitoring computation time */
    /*@{*/
#ifdef __CUDACC__
    CudaTimer timerBoundaries;
    CudaTimer timerGodunov;

    CudaTimer timerTraceUpdate;
    CudaTimer timerUpdate;
    CudaTimer timerEmf;
    CudaTimer timerDissipative;

    // shearing box timers
    CudaTimer timerPrimVar;
    CudaTimer timerElecField;
    CudaTimer timerMagSlopes;
    CudaTimer timerTrace;
    CudaTimer timerHydroShear;
    CudaTimer timerRemapping;
    CudaTimer timerShearBorder;
    CudaTimer timerCtUpdate;

#else
    Timer     timerBoundaries;
    Timer     timerGodunov;

    Timer     timerTraceUpdate;
    Timer     timerUpdate;
    Timer     timerEmf;
    Timer     timerDissipative;

    // shearing box timers
    Timer     timerPrimVar;
    Timer     timerElecField;
    Timer     timerMagSlopes;
    Timer     timerTrace;
    Timer     timerHydroShear;
    Timer     timerRemapping;
    Timer     timerShearBorder;
    Timer     timerCtUpdate;
#endif // __CUDACC__
    /*@}*/
#endif // DO_TIMING

  }; // class MHDRunGodunovZslab
  
} // namespace hydroSimu

#endif // MHD_RUN_GODUNOV_ZSLAB_H_
