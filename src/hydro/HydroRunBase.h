/**
 * \file HydroRunBase.h
 * \brief Defines a base C++ class to implement hydrodynamics simulations.
 *
 * HydroRunBase class is base class that gather all functionality
 * usefull to real hydrodynamics simulation implementation (see
 * HydroRunGodunov for example).
 *
 * \author P. Kestener
 * \date 28/06/2009
 *
 * $Id: HydroRunBase.h 3595 2014-11-04 12:27:24Z pkestene $
 */
#ifndef HYDRORUN_BASE_H_
#define HYDRORUN_BASE_H_

#include <memory>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "real_type.h"
#include "common_types.h"
#include "gpu_macros.h"
#include "constants.h"
#include <cmath>

#include <ConfigMap.h>
#include <map>
#include "Arrays.h"

//#include <tr1/memory>
//class HydroParameters;
//typedef std::tr1::shared_ptr<HydroParameters> HydroParametersPtr;
#include "HydroParameters.h"

#include "Forcing_OrnsteinUhlenbeck.h"

// some constants for 2D Riemann problem initialization
#include "initHydro.h"
#include "zSlabInfo.h"

namespace hydroSimu {

  /* rescale data buffer in range 0.0 - 1.0 */
  void rescaleToZeroOne(real_t *data, int size, real_t &min, real_t &max);
  
  /**
   * \class HydroRunBase HydroRunBase.h
   * \brief This is the base class containing all usefull methods to hydro
   * simulations (handling array initializations, boundary computations, output files).
   *
   * All classes effectively implementing hydro simulations should
   * inherit from this base class.
   *
   * \note Important note : this class does sequential computations (one CPU or GPU).
   * For parallel computations (multi-CPU/GPU, symbol USE_MPI
   * defined), see class HydroRunBaseMpi (which derives from HydroMpiParameters).
   */
  class HydroRunBase : public HydroParameters
  {
  public:
    HydroRunBase(ConfigMap &_configMap);
    virtual ~HydroRunBase();

    //! component names (5 for hydro + 3 for magnetic field)
    std::map<int,std::string> varNames;

    //! component prefixes (5 for hydro + 3 for magnetic field)
    std::map<int,std::string> varPrefix;

    real_t dx, dy, dz;
    
    //! compute time step.
    //! \param[in] useU if useU=0 then use h_U (or d_U) else use h_U2 (or d_U2)
    //! \return time step
    virtual real_t compute_dt(int useU=0);

    /**
     * compute viscosity fluxes in the 2D case, input arrays must have been allocated before.
     *
     * \param[in]  U      (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[out] flux_x flux along X due to viscosity forces
     * \param[out] flux_y flux along Y due to viscosity forces
     * \param[in]  dt     time step
     *
     */
    void compute_viscosity_flux(HostArray<real_t>  &U, 
				HostArray<real_t>  &flux_x, 
				HostArray<real_t>  &flux_y, 
				real_t              dt);
#ifdef __CUDACC__
    void compute_viscosity_flux(DeviceArray<real_t>  &U, 
				DeviceArray<real_t>  &flux_x, 
				DeviceArray<real_t>  &flux_y, 
				real_t                dt);
#endif // __CUDACC__

    /**
     * compute viscosity fluxes in the 3D case, input arrays must have been allocated before.
     *
     * \param[in]  U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[out] flux_x flux along X due to viscosity forces
     * \param[out] flux_y flux along Y due to viscosity forces
     * \param[out] flux_z flux along Z due to viscosity forces
     * \param[in]  dt     time step
     *
     */
    void compute_viscosity_flux(HostArray<real_t>  &U, 
				HostArray<real_t>  &flux_x, 
				HostArray<real_t>  &flux_y, 
				HostArray<real_t>  &flux_z,
				real_t              dt);
#ifdef __CUDACC__
    void compute_viscosity_flux(DeviceArray<real_t>  &U, 
				DeviceArray<real_t>  &flux_x, 
				DeviceArray<real_t>  &flux_y, 
				DeviceArray<real_t>  &flux_z, 
				real_t                dt);
#endif // __CUDACC__


    /**
     * compute viscosity flux, 3D case inside zslab sub-domain
     *
     * Tobe used inside HydroRunGodunovZslab and MHDRunGodunovZslab.
     */
    void compute_viscosity_flux(HostArray<real_t>  &U, 
				HostArray<real_t>  &flux_x, 
				HostArray<real_t>  &flux_y, 
				HostArray<real_t>  &flux_z,
				real_t              dt,
				ZslabInfo           zSlabInfo);
#ifdef __CUDACC__
    void compute_viscosity_flux(DeviceArray<real_t>  &U, 
				DeviceArray<real_t>  &flux_x, 
				DeviceArray<real_t>  &flux_y, 
				DeviceArray<real_t>  &flux_z,
				real_t                dt,
				ZslabInfo             zSlabInfo);    
#endif // __CUDACC__


    /**
     * compute random forcing normalization.
     *
     * \param[in,out] U  (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in]    dt time step
     *
     */
    real_t compute_random_forcing_normalization(HostArray<real_t>  &U,
						real_t              dt);
#ifdef __CUDACC__
    real_t compute_random_forcing_normalization(DeviceArray<real_t>  &U,
						real_t              dt);    
#endif // __CUDACC__

    /**
     * Add random forcing to velocity field and update total energy.
     *
     * \param[in,out] U  (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in]    dt time step
     * \param[in]    norm random force field normalization
     *
     */
    void add_random_forcing(HostArray<real_t>  &U,
			    real_t              dt,
			    real_t              norm);
#ifdef __CUDACC__
    void add_random_forcing(DeviceArray<real_t>  &U,
			    real_t               dt,
			    real_t              norm);
#endif // __CUDACC__

    /**
     * compute hydro update from fluxes in the 2D case.
     *
     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in]     flux_x flux along X.
     * \param[in]     flux_y flux along Y.
     *
     */
    void compute_hydro_update(HostArray<real_t>  &U, 
			      HostArray<real_t>  &flux_x, 
			      HostArray<real_t>  &flux_y);
#ifdef __CUDACC__
    void compute_hydro_update(DeviceArray<real_t>  &U, 
			      DeviceArray<real_t>  &flux_x, 
			      DeviceArray<real_t>  &flux_y);
#endif // __CUDACC__
    
    /**
     * compute hydro update from fluxes in the 3D case.
     *
     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in]     flux_x flux along X.
     * \param[in]     flux_y flux along Y.
     * \param[in]     flux_z flux along Z.
     *
     */
    void compute_hydro_update(HostArray<real_t>  &U, 
			      HostArray<real_t>  &flux_x,
			      HostArray<real_t>  &flux_y,
			      HostArray<real_t>  &flux_z);
#ifdef __CUDACC__
    void compute_hydro_update(DeviceArray<real_t>  &U, 
			      DeviceArray<real_t>  &flux_x, 
			      DeviceArray<real_t>  &flux_y, 
			      DeviceArray<real_t>  &flux_z);
#endif // __CUDACC__
    
    /**
     * compute hydro update from fluxes in the 3D case, z-slab method.
     *
     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in]     flux_x flux along X.
     * \param[in]     flux_y flux along Y.
     * \param[in]     flux_z flux along Z.
     * \param[in]     zSlabInfo z-slab index.
     *
     */
    void compute_hydro_update(HostArray<real_t>  &U, 
			      HostArray<real_t>  &flux_x,
			      HostArray<real_t>  &flux_y,
			      HostArray<real_t>  &flux_z,
			      ZslabInfo           zSlabInfo);
#ifdef __CUDACC__
    void compute_hydro_update(DeviceArray<real_t>  &U, 
			      DeviceArray<real_t>  &flux_x,
			      DeviceArray<real_t>  &flux_y,
			      DeviceArray<real_t>  &flux_z,
			      ZslabInfo             zSlabInfo);
#endif // __CUDACC__   

    /**
     * compute hydro update energy from fluxes in the 2D case.
     *
     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in]     flux_x flux along X.
     * \param[in]     flux_y flux along Y.
     *
     */
    void compute_hydro_update_energy(HostArray<real_t>  &U, 
				     HostArray<real_t>  &flux_x, 
				     HostArray<real_t>  &flux_y);
#ifdef __CUDACC__
    void compute_hydro_update_energy(DeviceArray<real_t>  &U, 
				     DeviceArray<real_t>  &flux_x, 
				     DeviceArray<real_t>  &flux_y);
#endif // __CUDACC__
    
    /**
     * compute hydro update energy from fluxes in the 3D case.
     *
     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in]     flux_x flux along X.
     * \param[in]     flux_y flux along Y.
     * \param[in]     flux_z flux along Z.
     *
     */
    void compute_hydro_update_energy(HostArray<real_t>  &U, 
				     HostArray<real_t>  &flux_x,
				     HostArray<real_t>  &flux_y,
				     HostArray<real_t>  &flux_z);
#ifdef __CUDACC__
    void compute_hydro_update_energy(DeviceArray<real_t>  &U, 
				     DeviceArray<real_t>  &flux_x, 
				     DeviceArray<real_t>  &flux_y, 
				     DeviceArray<real_t>  &flux_z);
#endif // __CUDACC__
    
    /**
     * compute hydro update energy from fluxes in the 3D case, z-slab method.
     *
     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in]     flux_x flux along X.
     * \param[in]     flux_y flux along Y.
     * \param[in]     flux_z flux along Z.
     * \param[in]     zSlabInfo
     *
     */
    void compute_hydro_update_energy(HostArray<real_t>  &U, 
				     HostArray<real_t>  &flux_x,
				     HostArray<real_t>  &flux_y,
				     HostArray<real_t>  &flux_z,
				     ZslabInfo           zSlabInfo);
#ifdef __CUDACC__
    void compute_hydro_update_energy(DeviceArray<real_t>  &U, 
				     DeviceArray<real_t>  &flux_x, 
				     DeviceArray<real_t>  &flux_y, 
				     DeviceArray<real_t>  &flux_z,
				     ZslabInfo             zSlabInfo);
#endif // __CUDACC__
    
    /**
     * compute gravity predictor.
     *
     * \param[in,out] qPrim 
     * \param[in]     dt
     */
    void compute_gravity_predictor(HostArray<real_t> &qPrim,
				   real_t  dt);
#ifdef __CUDACC__
    void compute_gravity_predictor(DeviceArray<real_t> &qPrim,
				   real_t  dt);
#endif // __CUDACC__

    /**
     * compute gravity predictor, z-slab method, 3D only.
     *
     * \param[in,out] qPrim
     * \param[in]     dt
     */
    void compute_gravity_predictor(HostArray<real_t>  &qPrim, 
				   real_t              dt,
				   ZslabInfo           zSlabInfo);
#ifdef __CUDACC__
    void compute_gravity_predictor(DeviceArray<real_t>  &qPrim, 
				   real_t                dt,
				   ZslabInfo             zSlabInfo);
#endif // __CUDACC__   


    /**
     * compute gravity source term.
     *
     * \param[in,out] UNew
     * \param[in]     UOld
     * \param[in]     dt
     */
    void compute_gravity_source_term(HostArray<real_t> &UNew,
				     HostArray<real_t> &UOld,
				     real_t  dt);
#ifdef __CUDACC__
    void compute_gravity_source_term(DeviceArray<real_t> &UNew,
				     DeviceArray<real_t> &UOld,
				     real_t  dt);
#endif // __CUDACC__

    /**
     * compute gravity source term, z-slab method, 3D only.
     *
     * \param[in,out] UNew
     * \param[in]     UOld
     * \param[in]     dt
     */
    void compute_gravity_source_term(HostArray<real_t> &UNew,
				     HostArray<real_t> &UOld,
				     real_t  dt,
				     ZslabInfo zSlabInfo);
#ifdef __CUDACC__
    void compute_gravity_source_term(DeviceArray<real_t> &UNew,
				     DeviceArray<real_t> &UOld,
				     real_t  dt,
				     ZslabInfo zSlabInfo);
#endif // __CUDACC__

  public:
    //! used in the GPU version to control the number of CUDA blocks in compute dt
    uint cmpdtBlockCount;

    //! used in the GPU version to control the number of CUDA blocks in random forcing normalization
    uint randomForcingBlockCount;

    //! number of required random forcing reduction (prerequisite to normalization)
    static const uint nbRandomForcingReduction = 9;

    //! total (physical) time
    real_t totalTime;

#ifdef __CUDACC__
  protected:
    /** \defgroup data_arrays data arrays */
    /*@{*/
    // Data Arrays (these arrays are only used for the GPU version 
    // of compute_dt reduction routine)
    HostArray  <real_t> h_invDt;
    DeviceArray<real_t> d_invDt;

    HostArray  <real_t> h_randomForcingNormalization;
    DeviceArray<real_t> d_randomForcingNormalization;
   /*@}*/
#endif // __CUDACC__

  public:
    /** \defgroup initial_conditions initialization routines (problem dependent) */
    /*@{*/
    virtual void init_hydro_jet();
    virtual void init_hydro_sod();
    virtual void init_hydro_implode();
    virtual void init_hydro_blast();
    virtual void init_hydro_Kelvin_Helmholtz();
    virtual void init_hydro_Rayleigh_Taylor();
    virtual void init_hydro_falling_bubble();
    virtual void init_hydro_Riemann();
    virtual void init_hydro_turbulence();
    virtual void init_hydro_turbulence_Ornstein_Uhlenbeck();
    //! this a wrapper which calls the actual init routine according
    //! to problemName variable.
    virtual int init_simulation(const std::string problemName);

    virtual void restart_run_extra_work();

    virtual void init_randomForcing();
    /*@}*/

#ifdef __CUDACC__
    //! compute border on GPU (call make_boundary for each borders in direction idim)
    virtual void make_boundaries(DeviceArray<real_t> &U, int idim);
#else
    //! compute border on CPU (call make_boundary for each borders in direction idim)
    virtual void make_boundaries(HostArray<real_t>   &U, int idim);
#endif // __CUDACC__

#ifdef __CUDACC__
    //! compute all borders on GPU (call make_boundary for each borders)
    virtual void make_all_boundaries(DeviceArray<real_t> &U);
#else
    //! compute all borders on CPU (call make_boundary for each borders)
    virtual void make_all_boundaries(HostArray<real_t>   &U);
#endif // __CUDACC__

    //! start simulation (compute multiple time step inside a while loop)
    virtual void start();

    //! compute only one time step integration (must be implemented in
    //! derived class.
    virtual void oneStepIntegration(int& nStep, real_t& t, real_t& dt) = 0;
    
#ifdef __CUDACC__
    //! @return d_U or d_U2
    DeviceArray<real_t>& getData(int nStep=0);
#else
    //! @return h_U or h_U2
    HostArray<real_t>& getData(int nStep=0);
#endif // __CUDACC__

    HostArray<real_t> h_debug; //!< debug array on host */
    HostArray<real_t> h_debug2; //!< debug array on host */
#ifdef __CUDACC__
    DeviceArray<real_t> d_debug; //!< debug array on device */
    DeviceArray<real_t> d_debug2; //!< debug array on device */
#endif // __CUDACC__

    /**
     * \brief return h_U or h_U2 according to nStep parity.
     * \param[in] nStep
     *
     * @return h_U or h_U2
     */
    HostArray<real_t>& getDataHost(int nStep=0);
    
    /**
     * \defgroup OutputRoutines Various output routines
     */
    /*@{*/
    void outputBin(HostArray<real_t> &U, int nStep, real_t);
    void outputXsm(HostArray<real_t> &U, int nStep, ComponentIndex iVar=ID);
    void outputPng(HostArray<real_t> &U, int nStep, ComponentIndex iVar=ID);
    virtual void outputVtk(HostArray<real_t> &U, int nStep);
    virtual void outputVtkDebug(HostArray<real_t> &data, const std::string suffix, int nStep, bool ghostIncluded=false);
    virtual void outputHdf5(HostArray<real_t> &U, int nStep, bool ghostIncluded=false);
    virtual void outputHdf5Debug(HostArray<real_t> &data, const std::string suffix, int nStep);
    virtual void writeXdmfForHdf5Wrapper(int totalNumberOfSteps, bool singleStep = false, bool ghostIncluded=false);
    virtual void outputNetcdf4(HostArray<real_t> &U, int nStep);
    virtual void outputNrrd(HostArray<real_t> &U, int nStep);
    virtual void output(HostArray<real_t> &U, int nStep, bool ghostIncluded=false);
    virtual void outputFaces(int nStep, FileFormat outputFormat);
#ifdef __CUDACC__
    virtual void outputFaces(int nStep, FileFormat outputFormat, 
			     HostArray<real_t>   &xface,
			     HostArray<real_t>   &yface,
			     HostArray<real_t>   &zface,
			     DeviceArray<real_t> &d_xface,
			     DeviceArray<real_t> &d_yface,
			     DeviceArray<real_t> &d_zface);
#endif // __CUDACC__
    virtual void outputFacesHdf5(HostArray<real_t> &U, int nStep);
    virtual void outputFacesVtk(HostArray<real_t> &U, int nStep, 
				ComponentIndex3D sideDir);
    /*@}*/

    virtual int inputHdf5(HostArray<real_t> &U, 
			  const std::string filename, 
			  bool halfResolution=false);
    
    virtual void upscale(HostArray<real_t> &HiRes, 
			 const HostArray<real_t> &LowRes);

    /** 
     * \brief routine to bring back data into host memory.
     * 
     * \param[in] nStep time step number
     *
     * CPU version : does nothing.
     * GPU version copy data back from GPU to host  memory (h_U).
     *
     * The default behavior is to copy d_U into h_U when nStep is even, and 
     * d_U2 into h_U when nStep is odd.
     * 
     * 
     */
    virtual void copyGpuToCpu(int nStep=0);
    
  protected:
#ifdef __CUDACC__
    /**
     * This is a wrapper routine, calling the actual border computation
     * routine from file make_boundary_base.h, according to the border
     * type.
     * <b>GPU version (cuda kernels are defined in make_boundary_base.h).</b>
     */
    template<BoundaryLocation boundaryLoc>
    void make_boundary(DeviceArray<real_t> &U, 
		       BoundaryConditionType bct, 
		       dim3 blockCount);
    /**
     * This routine is only used when doing the jet problem. It takes
     * care of the border to emulate matter injection.
     * <b>GPU version.</b>
     */
    void make_jet(DeviceArray<real_t> &U);
#else // CPU version
    /**
     * This is a wrapper routine, calling the actual border computation
     * routine from file make_boundary_base.h, according to the border
     * type.
     * <b>CPU version.</b>
     */
    template<BoundaryLocation boundaryLoc>
    void make_boundary(HostArray<real_t> &U, 
		       BoundaryConditionType bct, 
		       dim3 blockCount);
    /**
     * This routine is only used when doing the jet problem. It takes
     * care of the border to emulate matter injection.
     * <b>CPU version.</b>
     */
    void make_jet(HostArray<real_t> &U);
#endif // __CUDACC__
    
    /*
     * MAIN data arrays (hydro variables)
     */

    //! Data array on CPU
    HostArray<real_t> h_U;

    //! Extra Data array ( for example, used in the CPU version of the
    //! Godunov unsplit scheme to make ping-pong)
    HostArray<real_t> h_U2;

#ifdef __CUDACC__
    //! Data array on GPU
    DeviceArray<real_t> d_U;
    DeviceArray<real_t> d_U2;
#endif // __CUDACC__

    /*
     * gravity field (3 components)
     */
    HostArray<real_t>   h_gravity;
#ifdef __CUDACC__
    DeviceArray<real_t> d_gravity;
#endif // __CUDACC__

    //! gravity enabled 
    bool gravityEnabled;

    /*
     * random forcing data arrays (3 components)
     */

    //! random forcing field array on CPU
    HostArray<real_t> h_randomForcing;

#ifdef __CUDACC__
    //! random forcing field array on CPU
    DeviceArray<real_t> d_randomForcing;
#endif // __CUDACC__

    //! random forcing enabled 
    bool randomForcingEnabled;

    //! random forcing E dot
    real_t randomForcingEdot;

    //! random forcing (Ornstein-Uhlenbeck) enabled 
    bool randomForcingOrnsteinUhlenbeckEnabled;
    ForcingOrnsteinUhlenbeck *pForcingOrnsteinUhlenbeck;

    //! only used when dealing with a 2D Riemann problem
    RiemannConfig2d riemannConf[NB_RIEMANN_CONFIG];

    //! only used when dealing with a 2D Riemann problem
    int riemannConfId;

  public:
    //! only used when dealing with a 2D Riemann problem
    int getRiemannConfId() {return riemannConfId; };
    
    //! only used when dealing with a 2D Riemann problem
    void setRiemannConfId(int value) {riemannConfId = value; };

    //protected:
    // compute a y-z average of a scalar field (3D only), return a 1D array
    //void compute_yz_mean(HostArray<real_t>& localVar, HostArray<real_t>& meanVar);

  protected:
    /** pointer to a history method */
    void (HydroRunBase::*history_hydro_method)(int nStep, real_t dt);
 
    /** choose history method according to problem */
    void setupHistory_hydro();
    
    /** call the actual history method */
    void history_hydro(int nStep, real_t dt);
    
    /** don't do anything */
    void history_hydro_empty(int nStep, real_t dt);
    
    /** history default: compute total mass. */
    void history_hydro_default(int nStep, real_t dt);

    /** turbulence history */
    void history_hydro_turbulence(int nStep, real_t dt);

  }; // class HydroRunBase
  
} // namespace hydroSimu

#endif /*HYDRORUN_BASE_H_*/
