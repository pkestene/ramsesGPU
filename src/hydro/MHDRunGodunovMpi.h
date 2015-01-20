/**
 * \file MHDRunGodunovMpi.h
 * \brief Define class MHDRunGodunovMpi (see MHDRunGodunov for mono
 * CPU/GPU version) containing the actual numerical scheme for solving MHD.
 *
 * 2D/3D MHD solver (multi CPU or multi GPU) MPI version.
 * The class MHDRunGodunovMpi is adapted and simplified from the
 * mono-CPU / mono-GPU version: it only implements was is called
 * "implementation version 3 and 4".
 * Note that version 4 should give best performances on GPU.
 *
 * \sa class HydroRunBaseMpi defines MPI communication routine as well
 * as initialization (initial conditions). Here we only defines the
 * numerical scheme.
 *
 *
 * \date August 26, 2011
 * \author Pierre Kestener
 *
 * $Id: MHDRunGodunovMpi.h 2434 2012-09-28 09:00:27Z pkestene $
 */
#ifndef MHD_RUN_GODUNOV_MPI_H_
#define MHD_RUN_GODUNOV_MPI_H_

// base class
#include "HydroRunBaseMpi.h"

namespace hydroSimu {

  /**
   * \class MHDRunGodunovMpi MHDRunGodunovMpi.h
   * \brief This class implements MHD simulations using the Godunov numerical
   * scheme with MPI parallelization.
   * 
   * This class is similar to MHDRunGodunov dedicated to mono-CPU /
   * mono-GPU.
   *
   * Only one implementation of the MHD solver exist (one for 2D and
   * one for 3D). Here are several
   * features of implementation 1 and 3 (see class MHDRunGodunov for other
   * implementations) :
   *
   * \par version 1 (use for 2D)
   * One for loop to compute qm, qp, qedge state for the whole domain;
   * theses states are stored on external memory buffer (h_qm_x,
   * h_qm_y, h_qp_x, h_qp_y, h_qEdge_RT, etc ...).
   * Then a second for loop to compute fluxes and perform
   * hydro+magnetic update.<BR>
   * available on CPU (2D + 3D)<BR>
   * GPU version (2D only); 3D is buggy (prefer using version 3 which is
   * working well)
   *
   * \par version 3 (use for 3D)
   * In this version slopes (hydro and magnetic) computations are done
   * outside the trace routine.<BR> 
   * One for loop to compute Ex,Ey,Ez components and store in external
   * memory buffer h_elec (or d_elec in the GPU version).<BR>
   * One for loop to compute transverse magnetic slopes and store them in h_dA,
   * h_dB, h_dC (resp d_dA, d_dB, d_dC).<BR>
   * One for loop to compute hydro slopes then trace and store qm, qp,
   * qEdge states in external memory buffers.<BR>
   * One for loop to comute riemann problems and perform updates.
   * available on CPU (3D only)<BR>
   * available on GPU (3D only)<BR>
   * Note that this 3D GPU version was really hard to debug (see Trac
   * ticket number 233); this is why implementation 4
   * was created and at the end is faster.
   *
   * \par version 4
   * Initially created to debug version 3;
   * - kernels for for computing primitive variables, electrical field
   *   components and magnetic slopes are identical to version 3
   * - trace computations and flux/update are done in separate kernels
   *   (easier to debug)
   * Kernels are all smaller and easier to debug. Moreover this
   * version provides the best performances (have a look at a domain
   * of size 64^3).
   *
   */
  class MHDRunGodunovMpi : public HydroRunBaseMpi
  {
  public:
    MHDRunGodunovMpi(ConfigMap &_configMap);
    ~MHDRunGodunovMpi();

    //! convert conservative variables (h_U or h_U2) into primitive
    //! var h_Q and take care of source term predictor.
    //!
    //! This routine is only used in the CPU version of the code.
    //! In the GPU version, the conversion is done on line, inside
    //! kernel as needed.
    //! \param[in] U conservative variable domain array
    //! \param[in] timeStep time step is needed for predictor computations
    void convertToPrimitives(real_t *U, real_t timeStep);

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
    virtual void make_boundaries_shear(DeviceArray<real_t> &U, real_t dt, int nStep, bool debug=false);
#else
    //!  implement shearing border condition
    virtual void make_boundaries_shear(HostArray<real_t>   &U, real_t dt, int nStep, bool debug=false);
#endif // __CUDACC__

#ifdef __CUDACC__
    //!  implement shearing border condition
    virtual void make_all_boundaries_shear(DeviceArray<real_t> &U, real_t dt, int nStep, bool doExternalBoundaries=true, bool debug=false);
#else
    //!  implement shearing border condition
    virtual void make_all_boundaries_shear(HostArray<real_t>   &U, real_t dt, int nStep, bool doExternalBoundaries=true, bool debug=false);
#endif // __CUDACC__

    //! start integration and control output
    void start();
    
    //! perform only one time step integration
    void oneStepIntegration(int& nStep, real_t& t, real_t& dt);

    //! shearing box enabled ?
    bool shearingBoxEnabled;

  private:

#ifdef __CUDACC__
    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
#else
    HostArray<real_t>   h_Q; //!< !!! CPU ONLY !!! Primitive Data array on CPU
#endif // __CUDACC__

    /** which implementation to use (currently possible values are 0, 1 and 2) */
    int implementationVersion;

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
    DeviceArray<real_t> d_shear_flux_xmin_toSend;    /*!< flux correction data at XMIN (GPU -> CPU + MPI comm) */
    DeviceArray<real_t> d_shear_flux_xmax_toSend;    /*!< flux correction data at XMAX (GPU -> CPU + MPI comm) */
    DeviceArray<real_t> d_shear_flux_xmin_remap;     /*!< flux correction data at XMIN */
    DeviceArray<real_t> d_shear_flux_xmax_remap;     /*!< flux correction data at XMAX */
    DeviceArray<real_t> d_shear_flux_xmin_recv_glob; /*!< flux correction data at XMIN after MPI comm */
    DeviceArray<real_t> d_shear_flux_xmax_recv_glob; /*!< flux correction data at XMAX after MPI comm */

    DeviceArray<real_t> d_shear_border;              /*! shearing box border buffer */
    DeviceArray<real_t> d_shear_border_xmin_recv_glob; /* shear ghost zones buffer, global size */
    DeviceArray<real_t> d_shear_border_xmax_recv_glob; /* shear ghost zones buffer, global size */

    DeviceArray<real_t> d_shear_slope_xmin_glob;     /*! XMIN border slopes in XMAX border (lives in myMpiPos[0] = mx-1) */
    DeviceArray<real_t> d_shear_slope_xmax_glob;     /*! XMAX border slopes in XMIN border (lives in myMpiPos[0] = 0   ) */

#else

    HostArray<real_t>   h_shear_flux_xmin_remap;    /*!< flux correction data at XMIN */
    HostArray<real_t>   h_shear_flux_xmax_remap;    /*!< flux correction data at XMAX */

#endif // __CUDACC__

    // MPI communication buffers
    HostArray<real_t>   h_shear_flux_xmin_toSend;    /*!< flux correction data at XMIN (MPI comm) */
    HostArray<real_t>   h_shear_flux_xmax_toSend;    /*!< flux correction data at XMAX (MPI comm) */
    HostArray<real_t>   h_shear_flux_xmin_recv1;     /*!< flux correction data at XMIN (MPI comm) */
    HostArray<real_t>   h_shear_flux_xmin_recv2;     /*!< flux correction data at XMIN (MPI comm) */
    HostArray<real_t>   h_shear_flux_xmax_recv1;     /*!< flux correction data at XMAX (MPI comm) */
    HostArray<real_t>   h_shear_flux_xmax_recv2;     /*!< flux correction data at XMAX (MPI comm) */
    HostArray<real_t>   h_shear_flux_xmin_recv_glob; /*!< flux correction data at XMIN after MPI comm */
    HostArray<real_t>   h_shear_flux_xmax_recv_glob; /*!< flux correction data at XMAX after MPI comm */

    HostArray<real_t>   h_shear_border_recv1;
    HostArray<real_t>   h_shear_border_recv2;
    HostArray<real_t>   h_shear_border;             /*! shearing box border buffer, for MPI comm */

    HostArray<real_t>   h_shear_border_xmin_recv_glob; /* shear ghost zones buffer, global size */
    HostArray<real_t>   h_shear_border_xmax_recv_glob; /* shear ghost zones buffer, global size */

    HostArray<real_t>   h_shear_slope_xmin_glob;    /*! XMIN border slopes in XMAX border (lives in myMpiPos[0] = mx-1) */
    HostArray<real_t>   h_shear_slope_xmax_glob;    /*! XMAX border slopes in XMIN border (lives in myMpiPos[0] = 0   ) */

    /*@}*/
    
    
    bool dumpDataForDebugEnabled; /*!< if true, dump almost all intermediate data buffer to a file */
    bool debugShear; /*!< if true, enable some special debug (data dump) of shearing box data */

    /*
     * please note that timer for boundaries computation are declared
     * in HydroRunBaseMpi.h
     */
#ifdef DO_TIMING
#ifdef __CUDACC__
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
    CudaTimer timerMakeShearBorder;
    CudaTimer timerMakeShearBorderSend;
    CudaTimer timerMakeShearBorderSlopes;
    CudaTimer timerMakeShearBorderFinalRemapping;
    CudaTimer timerCtUpdate;

#else
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
    Timer     timerMakeShearBorder;
    Timer     timerCtUpdate;
#endif // __CUDACC__
#endif // DO_TIMING

  }; // class MHDRunGodunovMpi
  
} // namespace hydroSimu

#endif // MHD_RUN_GODUNOV_MPI_H_
