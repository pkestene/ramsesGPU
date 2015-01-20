/**
 * \file HydroRunBaseMpi.h
 * \brief Defines a base C++ class to implement hydrodynamics
 * simulations in MPI environement.
 *
 * HydroRunBaseMpi class is the symetric to HydroRunBase.
 *
 * \author P. Kestener
 * \date 12 Oct 2010
 *
 * $Id: HydroRunBaseMpi.h 3595 2014-11-04 12:27:24Z pkestene $
 */
#ifndef HYDRORUN_BASE_MPI_H_
#define HYDRORUN_BASE_MPI_H_

#include <memory>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <map>

#include "real_type.h"
#include "common_types.h"
#include "gpu_macros.h"
#include "constants.h"
#include <cmath>

#include "Arrays.h"

//#include <tr1/memory>
//class HydroParameters;
//typedef std::tr1::shared_ptr<HydroMpiParameters> HydroMpiParametersPtr;
#include "HydroMpiParameters.h"

#include "Forcing_OrnsteinUhlenbeck.h"

// some constants for 2D Riemann problem initialization
#include "initHydro.h"
#include "zSlabInfo.h"

// timing measurement
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
   * \class HydroRunBaseMpi HydroRunBaseMpi.h
   * \brief This is the MPI specialized version of the base class
   * containing all usefull methods to hydro simulations (handling
   * array initializations, boundary computations, output files).
   *
   * All classes effectively implementing hydro simulations should
   * inherit from this base class.
   *
   * \note Important note : this class does sequential computations (one CPU or GPU).
   * For parallel computations (multi-CPU/GPU, symbol USE_MPI
   * defined), see class HydroRunBaseMpi (which derives from HydroMpiParameters).
   */
  class HydroRunBaseMpi : public HydroMpiParameters
  {
  public:
    HydroRunBaseMpi(ConfigMap &_configMap);
    virtual ~HydroRunBaseMpi();

    //! component names (5 for hydro + 3 for magnetic field)
    std::map<int,std::string> varNames;

    //! component prefixes (5 for hydro + 3 for magnetic field)
    std::map<int,std::string> varPrefix;

    real_t dx, dy, dz;

    //! compute local (ie intra MPI block) time step
    real_t compute_dt_local(int useU=0);
    //! compute global time step (take into account all MPI blocks)
    real_t compute_dt(int useU=0);

    //! compute local (ie intra MPI block) time step
    //! \param[in] useU if useU=0 then use h_U (or d_U) else use h_U2 (or d_U2)
    //! \return local time step
    real_t compute_dt_mhd_local(int useU=0);

    //! compute global time step.
    //! \param[in] useU if useU=0 then use h_U (or d_U) else use h_U2 (or d_U2)
    //! \return global time step
    real_t compute_dt_mhd(int useU=0);

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

    /**
     * compute magnetic field update from emf in the 2D case.
     *
     * \param[inout] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in] emf emf array.
     *
     */
    void compute_ct_update_2d(HostArray<real_t>  &U, 
			      HostArray<real_t>  &emf,
			      real_t dt);
#ifdef __CUDACC__
    void compute_ct_update_2d(DeviceArray<real_t>  &U, 
			      DeviceArray<real_t>  &emf,
			      real_t dt);
#endif // __CUDACC__
    
    /**
     * compute magnetic field update from emf in the 3D case.
     *
     * \param[inout] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in] emf emf array.
     *
     */
    void compute_ct_update_3d(HostArray<real_t>  &U, 
			      HostArray<real_t>  &emf,
			      real_t dt);
#ifdef __CUDACC__
    void compute_ct_update_3d(DeviceArray<real_t>  &U, 
			      DeviceArray<real_t>  &emf,
			      real_t dt);
#endif // __CUDACC__
    
    /**
     * compute magnetic field update from emf in the 3D case, z-slab method.
     *
     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
     * \param[in] emf emf array.
     * \param[in] zSlabInfo
     *
     */
    void compute_ct_update_3d(HostArray<real_t>  &U, 
			      HostArray<real_t>  &emf,
			      real_t dt,
			      ZslabInfo zSlabInfo);
#ifdef __CUDACC__
    void compute_ct_update_3d(DeviceArray<real_t>  &U, 
			      DeviceArray<real_t>  &emf,
			      real_t dt,
			      ZslabInfo zSlabInfo);
#endif // __CUDACC__

    /**
     * compute resistivity emf's in the 2D case.
     *
     * \param[in]  U input conservative variables array.
     * \param[out] resistiveEmf resistive EMF array.
     * \param[in]  dt time step.
     */
    void compute_resistivity_emf_2d(HostArray<real_t> &U,
				    HostArray<real_t> &emf);
#ifdef __CUDACC__
    void compute_resistivity_emf_2d(DeviceArray<real_t> &U,
				    DeviceArray<real_t> &emf);
#endif // __CUDACC__

    /**
     * compute resistivity emf's in the 3D case.
     *
     * \param[in]  U input conservative variables array.
     * \param[out] resistiveEmf resistive EMF array.
     * \param[in]  dt time step.
     */
    void compute_resistivity_emf_3d(HostArray<real_t> &U,
				    HostArray<real_t> &emf);
#ifdef __CUDACC__
    void compute_resistivity_emf_3d(DeviceArray<real_t> &U,
				    DeviceArray<real_t> &emf);
#endif // __CUDACC__
    
    /**
     * compute resistivity emf's in the 3D case, z-slab method.
     *
     * \param[in]  U input conservative variables array.
     * \param[out] resistiveEmf resistive EMF array.
     * \param[in]  dt time step.
     */
    void compute_resistivity_emf_3d(HostArray<real_t> &U,
				    HostArray<real_t> &emf,
				    ZslabInfo          zSlabInfo);
#ifdef __CUDACC__
    void compute_resistivity_emf_3d(DeviceArray<real_t> &U,
				    DeviceArray<real_t> &emf,
				    ZslabInfo            zSlabInfo);
#endif // __CUDACC__

    /**
     * compute resistivity energy flux in the 2D case, only usefull when performing non-isothermal simulations.
     *
     * \param[in]  U      input conservative variables array.
     * \param[out] flux_x X flux array, only energy is used.
     * \param[out] flux_y Y flux array, only energy is used.
     * \param[in]  dt     time step.
     */
    void compute_resistivity_energy_flux_2d(HostArray<real_t> &U,
					    HostArray<real_t> &flux_x,
					    HostArray<real_t> &flux_y,
					    real_t             dt);
#ifdef __CUDACC__
    void compute_resistivity_energy_flux_2d(DeviceArray<real_t> &U,
					    DeviceArray<real_t> &flux_x,
					    DeviceArray<real_t> &flux_y,
					    real_t               dt);
#endif // __CUDACC__
    
    /**
     * compute resistivity energy flux in the 3D case, only usefull when performing non-isothermal simulations.
     *
     * \param[in]  U      input conservative variables array.
     * \param[out] flux_x X flux array, only energy is used.
     * \param[out] flux_y Y flux array, only energy is used.
     * \param[out] flux_z Z flux array, only energy is used.
     * \param[in]  dt     time step.
     */
    void compute_resistivity_energy_flux_3d(HostArray<real_t> &U,
					    HostArray<real_t> &flux_x,
					    HostArray<real_t> &flux_y,
					    HostArray<real_t> &flux_z,
					    real_t             dt);
#ifdef __CUDACC__
    void compute_resistivity_energy_flux_3d(DeviceArray<real_t> &U,
					    DeviceArray<real_t> &flux_x,
					    DeviceArray<real_t> &flux_y,
					    DeviceArray<real_t> &flux_z,
					    real_t               dt);
#endif // __CUDACC__

    /**
     * compute resistivity energy flux in the 3D case, only usefull when performing non-isothermal simulations (z-slab method).
     *
     * \param[in]  U      input conservative variables array.
     * \param[out] flux_x X flux array, only energy is used.
     * \param[out] flux_y Y flux array, only energy is used.
     * \param[out] flux_z Z flux array, only energy is used.
     * \param[in]  dt     time step.
     * \param[in]  zSlabInfo
     */
    void compute_resistivity_energy_flux_3d(HostArray<real_t> &U,
					    HostArray<real_t> &flux_x,
					    HostArray<real_t> &flux_y,
					    HostArray<real_t> &flux_z,
					    real_t             dt,
					    ZslabInfo          zSlabInfo);
#ifdef __CUDACC__
    void compute_resistivity_energy_flux_3d(DeviceArray<real_t> &U,
					    DeviceArray<real_t> &flux_x,
					    DeviceArray<real_t> &flux_y,
					    DeviceArray<real_t> &flux_z,
					    real_t               dt,
					    ZslabInfo            zSlabInfo);
#endif // __CUDACC__
    
    //! compute magnetic field divergence at cell centers
    //! using face-centered values
    void compute_divB(HostArray<real_t>& h_conserv,
		      HostArray<real_t>& h_divB);

    //! initialized in constructor to either MpiComm::FLOAT or MpiComm::DOUBLE
    int mpi_data_type;

    //! total (physical) time
    real_t totalTime;

  protected:
    //! used in the GPU version to control the number of CUDA block of
    //! the compute time step kernel.
    uint cmpdtBlockCount;

    //! used in the GPU version to control the number of CUDA blocks in random forcing normalization
    uint randomForcingBlockCount;

    //! number of required random forcing reduction (prerequisite to normalization)
    static const uint nbRandomForcingReduction = 9;

#ifdef __CUDACC__
  protected:
    // Data Arrays (these arrays are only used for the GPU version 
    // of compute_dt reduction routine)
    HostArray  <real_t> h_invDt;
    DeviceArray<real_t> d_invDt;

    HostArray  <real_t> h_randomForcingNormalization;
    DeviceArray<real_t> d_randomForcingNormalization;
#endif // __CUDACC__

  public:
    /** \defgroup initial_conditions initialization routines (problem dependent) */
    /*@{*/
    virtual void init_hydro_jet();
    virtual void init_hydro_implode();
    virtual void init_hydro_blast();
    virtual void init_hydro_Kelvin_Helmholtz();
    virtual void init_hydro_Rayleigh_Taylor();
    virtual void init_hydro_falling_bubble();
    virtual void init_hydro_Riemann();
    virtual void init_hydro_turbulence();
    virtual void init_hydro_turbulence_Ornstein_Uhlenbeck();
    virtual void init_mhd_jet();
    virtual void init_mhd_implode();
    virtual void init_mhd_Orszag_Tang();
    virtual void init_mhd_field_loop();
    virtual void init_mhd_shear_wave();
    virtual void init_mhd_mri();
    virtual void init_mhd_Kelvin_Helmholtz();
    virtual void init_mhd_Rayleigh_Taylor();
    virtual void init_mhd_turbulence();
    virtual void init_mhd_turbulence_Ornstein_Uhlenbeck();

    void init_mhd_mri_grav_field();
    virtual void restart_run_extra_work();

    //! this a wrapper which calls the actual init routine according
    //! to problemName variable.
    virtual int init_simulation(const std::string problemName);

    virtual void init_randomForcing();
    /*@}*/

#ifdef __CUDACC__
    //! compute border on GPU (call make_boundary for each borders in direction idim)
    virtual void make_boundaries(DeviceArray<real_t> &U, int idim, bool doExternalBorders=true);
#else
    //! compute border on CPU (call make_boundary for each borders in direction idim)
    virtual void make_boundaries(HostArray<real_t>   &U, int idim, bool doExternalBorders=true);
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
    void outputXsm(HostArray<real_t> &U, int nStep, ComponentIndex iVar=ID, bool withGhosts=false);
    virtual void outputVtk(HostArray<real_t> &U, int nStep);
    virtual void outputHdf5(HostArray<real_t> &U, int nStep);
    virtual void outputHdf5Debug(HostArray<real_t> &data, const std::string suffix, int nStep);
    virtual void writeXdmfForHdf5Wrapper(int totalNumberOfSteps);
    virtual void outputPnetcdf(HostArray<real_t> &U, int nStep);
    virtual void output(HostArray<real_t> &U, int nStep);

    // dump face data
    virtual void outputFaces(int nStep, bool pnetcdfEnabled);
#ifdef __CUDACC__
    virtual void outputFaces(int nStep, bool pnetcdfEnabled,
			     HostArray<real_t>   &h_xface,
			     HostArray<real_t>   &h_yface,
			     HostArray<real_t>   &h_zface,
			     DeviceArray<real_t> &d_xface,
			     DeviceArray<real_t> &d_yface,
			     DeviceArray<real_t> &d_zface);
#endif // __CUDACC__
    virtual void outputFacesPnetcdf(HostArray<real_t> &U, int nStep, ComponentIndex3D faceDir);
    /*@}*/
    
    /**
     * \defgroup InputRoutines Various input routines
     */
    /*@{*/
    virtual int inputHdf5(HostArray<real_t> &U, 
			  const std::string filename,
			  bool halfResolution=false);
    virtual int inputPnetcdf(HostArray<real_t> &U, 
			     const std::string filename,
			     bool halfResolution=false);
    /*@}*/

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

    //! \defgroup BorderBuffer data arrays for border exchange handling
    //! @{
    HostArray<real_t> borderBufSend_xmin;
    HostArray<real_t> borderBufSend_xmax;
    HostArray<real_t> borderBufSend_ymin;
    HostArray<real_t> borderBufSend_ymax;
    HostArray<real_t> borderBufSend_zmin;
    HostArray<real_t> borderBufSend_zmax;

    HostArray<real_t> borderBufRecv_xmin;
    HostArray<real_t> borderBufRecv_xmax;
    HostArray<real_t> borderBufRecv_ymin;
    HostArray<real_t> borderBufRecv_ymax;
    HostArray<real_t> borderBufRecv_zmin;
    HostArray<real_t> borderBufRecv_zmax;

#ifdef __CUDACC__
    DeviceArray<real_t> borderBuffer_device_xdir;
    DeviceArray<real_t> borderBuffer_device_ydir;
    DeviceArray<real_t> borderBuffer_device_zdir;
#endif // __CUDACC__

    //! @}

    /**
     * transfert border buffer (MPI communication)
     */
    template<int direction>
    void transfert_boundaries();

#ifdef __CUDACC__
    /**
     * transfert border buffer (MPI communication):
     * 1. copy border to buffer
     * 2. send and receiver these buffers
     *
     * \tparam direction XDIR, YDIR or ZDIR
     */
    template<int direction>
    void copy_boundaries(DeviceArray<real_t> &U);
    /**
     * take care of an inner or outer border. This routine is called
     * once per border and per time step.
     *
     * 1. copy border to buffer
     * 2. send and receiver these buffers
     * 3. copy back in place the buffers
     * 4. (GPU only) copy border to Device data array
     *
     * <b>GPU version.</b>
     */
    template<BoundaryLocation boundaryLoc>
    void make_boundary(DeviceArray<real_t> &U, 
		       HostArray<real_t> &bRecv, 
		       BoundaryConditionType bct,
		       dim3 blockCount);
    /**
     * This routine is only used when doing the jet problem. It takes
     * care of the border to emulate matter injection.
     * <b>CPU version.</b>
     */
    void make_jet(DeviceArray<real_t> &U);
#else // CPU version
    /**
     * transfert border buffer (MPI communication):
     * 1. copy border to buffer
     * 2. send and receiver these buffers
     */
    template<int direction>
    void copy_boundaries(HostArray<real_t> &U);
    /**
     * take care of an inner or outer border. This routine is called
     * once per border and per time step.
     *
     * 1. handle outer boundaries (Dirichlet, Neumann, ...). 
     *    Possibly, no MPI communication involved here.
     * 2. handle inner boundaries : copy back in place the buffers
     *
     * @tparam boundaryLoc Boundary Location where 
     * <b>CPU version.</b>
     */
    template<BoundaryLocation boundaryLoc>
    void make_boundary(HostArray<real_t> &U, 
		       HostArray<real_t> &bRecv, 
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
     * MAIN data arrays (for conservative variables)
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

    //! runtime determination if we are using float ou double (for MPI communication)
    int data_type;

#ifdef DO_TIMING
#ifdef __CUDACC__
    CudaTimer timerBoundariesCpuGpu;
    CudaTimer timerBoundariesGpu;
    CudaTimer timerBoundariesMpi;
#else
    Timer     timerBoundariesCpu;
    Timer     timerBoundariesMpi;
#endif // __CUDACC__
#endif // DO_TIMING

  public:
    //! only used when dealing with a 2D Riemann problem
    int getRiemannConfId() {return riemannConfId; };
    
    //! only used when dealing with a 2D Riemann problem
    void setRiemannConfId(int value) {riemannConfId = value; };

  protected:
    /** pointer to a history method */
    void (HydroRunBaseMpi::*history_method)(int nStep, real_t dt);
    
    /** choose history method according to problem */
    void setupHistory();
    
    /** call the actual history method */
    void history(int nStep, real_t dt);
    
    /** don't do anything */
    void history_empty(int nStep, real_t dt);
    
    /** history default for hydro : compute total mass. */
    void history_hydro_default(int nStep, real_t dt);

    /** history default for mhd   : compute total mass and divB. */
    void history_mhd_default(int nStep, real_t dt);

    /** MRI history */
    void history_mhd_mri(int nStep, real_t dt);

    /** turbulence history for hydro */
    void history_hydro_turbulence(int nStep, real_t dt);
    
    /** turbulence history for mhd */
    void history_mhd_turbulence(int nStep, real_t dt);
    
  }; // class HydroRunBaseMpi
  
} // namespace hydroSimu

#endif /*HYDRORUN_BASE_H_*/
