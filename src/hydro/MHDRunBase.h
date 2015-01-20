/**
 * \file MHDRunBase.h
 * \brief Defines a base C++ class to implement magneto-hydrodynamics (MHD) simulations.
 *
 * MHDRunBase class is base class that gather all functionality
 * usefull to real MHD simulation implementation (see
 * MhdRunGodunov for example).
 *
 * MHDRunBase derives from HydroRunBase (so that the Gui classes like
 * HydroWindow in Glut can be reused as is) but almost every method
 * are redefined.
 *
 * \author P. Kestener
 * \date March, 25 2011
 *
 * $Id: MHDRunBase.h 3464 2014-06-29 21:16:23Z pkestene $
 */
#ifndef MHDRUN_BASE_H_
#define MHDRUN_BASE_H_

#include "HydroRunBase.h"

namespace hydroSimu {
  
  /**
   * \class MHDRunBase MHDRunBase.h
   * \brief This is the base class containing all usefull methods to MHD
   * simulations (handling array initializations, boundary computations, output files).
   *
   * All classes effectively implementing MHD simulations should
   * inherit from this base class.
   *
   * \note Important note : this class does sequential computations (one CPU or GPU).
   * For parallel computations (multi-CPU/GPU, symbol USE_MPI
   * defined), see class MHDRunBaseMpi.
   *
   * \sa class HydroRunBase
   */
  class MHDRunBase : public HydroRunBase
  {
  public:
    MHDRunBase(ConfigMap &_configMap);
    virtual ~MHDRunBase();

    /** compute time step.
     * \param[in] useU if useU=0 then use h_U (or d_U) else use h_U2 (or d_U2)
     * \return time step
     */
    real_t compute_dt_mhd(int useU=0);

    /**
     * compute magnetic field update from emf in the 2D case.
     *
     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
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
     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
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

    //! compute magnetic field laplacian
    void compute_laplacianB(HostArray<real_t>& h_conserv,
			    HostArray<real_t>& h_laplacianB);

    /** \addtogroup initial_conditions initialization routines for MHD problems */
    /** @{*/
    virtual int init_simulation(const std::string problemName);
    void init_Orszag_Tang();
    void init_mhd_jet();
    void init_mhd_sod();
    void init_mhd_BrioWu();
    void init_mhd_rotor();
    void init_mhd_field_loop();
    void init_mhd_current_sheet();
    void init_mhd_inertial_wave();
    void init_mhd_shear_wave();
    void init_mhd_mri();
    void init_mhd_Kelvin_Helmholtz();
    void init_mhd_Rayleigh_Taylor();
    void init_mhd_turbulence();
    void init_mhd_turbulence_Ornstein_Uhlenbeck();

    void init_mhd_mri_grav_field();
    virtual void restart_run_extra_work();
    /** @}*/

  protected:
    /** pointer to a history method */
    void (MHDRunBase::*history_method)(int nStep, real_t dt);
    
    /** choose history method according to problem */
    void setupHistory();
    
    /** call the actual history method */
    void history(int nStep, real_t dt);
    
    /** don't do anything */
    void history_empty(int nStep, real_t dt);
    
    /** history default: compute total mass and divB. */
    void history_default(int nStep, real_t dt);
    
    /** inertial wave history */
    void history_inertial_wave(int nStep, real_t dt);
    
    /** MRI history */
    void history_mri(int nStep, real_t dt);
    
    /** Turbulence history */
    void history_turbulence(int nStep, real_t dt);

  }; // class MHDRunBase
  
} // namespace hydroSimu

#endif // MHDRUN_BASE_H_
