/**
 * \file HydroParameters.h
 * \brief Defines a C++ structures gathering simulation parameters.
 *
 * Hydrodynamics simulation parameters are declared in the same order as in
 * original namelists from fortran file read_params.f90.
 *
 * \author P. Kestener
 * \date 28/06/2009
 *
 * $Id: HydroParameters.h 3236 2014-02-04 00:09:53Z pkestene $
 */


#ifndef HYDROPARAMETERS_H_
#define HYDROPARAMETERS_H_

#include "real_type.h"
#include "constants.h"
#include "gpu_macros.h"
#include "Arrays.h"
#include <iostream>
#include "utils/config/ConfigMap.h"
#include <string>
#include <cctype> // for std::tolower
#include <algorithm> // for std::transform
#include <cmath>
#include <cstring> // for memset

#include <cassert>
/* 
 * usefull macro for assertion avoiding dummy compiler warning
 * see http://stackoverflow.com/questions/1712713/suppress-c-compiler-warning-from-a-specific-line
 * This macro only works if the message has no white space !
 */
//#define myAssert(exp, msg) { const bool msg(true); assert(msg && (exp)); }

/**
 * \struct HydroParameters HydroParameters.h
 * \brief This is the base class containing all usefull parameters to perform
 * computations.
 *
 * HydroRunBase inherits from this base class.
 */
struct HydroParameters
{
  /**
   * constructor
   * @param _configMap : a reference to a ConfigMap object
   * @param _initGpu : [only used in GPU version] a boolean that tells to 
   *         initialize here the device. 
   *         When using a CUDA+MPI version, running on an achitecture
   *         with multiple GPU per node, it may be usefull to choose the device,
   *         not here buf after getting MPI information (MPI rank for example).
   *         See class HydroMpiParameters' constructor, for a example of 
   *         calling cudaSetDevice, to be ensure that each MPI processes
   *         comunicates with a unique GPU device.
   */
  HydroParameters(ConfigMap &_configMap, bool _initGpu);

  /**
   * this routine copy hydro parameters (defined below) to global variables.
   * GPU version copy them to __constant__ memory area on device.
   * Note that since __constant__ variable have static storage, this
   * routine must be called in each compilation unit they are used.
   */
  void copyToSymbolMemory();

  ConfigMap configMap; //!< a key-value map output by our INI file reader.

  /** \defgroup param_run run parameters */
  /**@{*/
  int nStepmax; //!< maximun number of time steps.
  real_t tEnd;  //!< end of simulation time.
  int nOutput;  //!< number of time step between 2 consecutive outputs.
  /**@}*/

  /** \defgroup param_mesh mesh parameters */
  /**@{*/
  int nx;    //!< logical size along X (without ghost cells).
  int ny;    //!< logical size along Y (without ghost cells).
  int nz;    //!< logical size along Z (without ghost cells).
  int nbVar; //!< number of fields in simulation (density, velocities, ...)
  DimensionType dimType; //!< 2D or 3D.
  //float xMin, xMax, yMin, yMax, zMin, zMax; //!< domain geometry
  //float dx;  //!< resolution along X axis.
  //float dy;  //!< resolution along Y axis.
  //float dz;  //!< resolution along Z axis.
  BoundaryConditionType boundary_xmin; //!< Boundary Condition Type
  BoundaryConditionType boundary_xmax; //!< Boundary Condition Type
  BoundaryConditionType boundary_ymin; //!< Boundary Condition Type
  BoundaryConditionType boundary_ymax; //!< Boundary Condition Type
  BoundaryConditionType boundary_zmin; //!< Boundary Condition Type
  BoundaryConditionType boundary_zmax; //!< Boundary Condition Type
  int ghostWidth; //!< number of ghost cells on each border (2 for Hydro, 3 for MHD).
  /**@}*/
  
  /** \defgroup param_hydro hydro parameters */
  /**@{*/
  GlobalConstants _gParams;  //!< a structure gathering simulation
			     //!< parameters (used in the GPU version of
			     //!< the code).
  std::string problem;       //!< a string used to set initial conditions.
  real_t cfl;                //!< Courant-Friedrich-Lewy parameter
  real_t _slope_type;        //!< type of slope computation (only
			     //!< usefull in case of Godunov scheme)
  /**@}*/

  /** \defgroup param_mhd mhd parameters */
  /**@{*/
  bool mhdEnabled;
  /**@}*/

  
  /** \defgroup param_jet jet parameters */
  /**@{*/
  int   enableJet;  //!< enable jet initial and border conditions.
  int   ijet;       //!< width (in cell unit) of the injection jet.
  real_t djet;      //!< density of the injected fluid.
  real_t ujet;      //!< velocity of the injected fluid.
  real_t pjet;      //!< pressure of the injected fluid.
  real_t cjet;      //!< sound speed of the injected fluid.
  int offsetJet;    //!< position or offset (in cell unit) of the injection jet.
  /**@}*/
  
  /* other parameters (not associated to a particular namelist) */
  int imin; //!< index minimum at X border
  int imax; //!< index maximum at X border
  int jmin; //!< index minimum at Y border
  int jmax; //!< index maximum at Y border
  int kmin; //!< index minimum at Z border
  int kmax; //!< index maximum at Z border

  int isize; //!< total size along X direction (including ghost cells).
  int jsize; //!< total size along Y direction (including ghost cells).
  int ksize; //!< total size along Z direction (including ghost cells).

  /** \defgroup output_mode control output mode */
  /**@{*/
  bool outputVtkEnabled;  //!< enable VTK output file format (using VTI).
  bool outputHdf5Enabled; //!< enable HDF5 output file format.
  bool outputNetcdf4Enabled; //!< enable NetCDF4 output file format.
  bool outputPnetcdfEnabled; //!< enable Parallel-NetCDF output file format.
  bool outputXsmEnabled;  //!< enable Xsmurf output file format (binary
			  //!< + one line ascii header)
  bool outputPngEnabled;  //!< enable PNG output file format (only for
			  //!< 2D simulation runs).
  bool outputNrrdEnabled; //!< enable NRRD output file format.
  /**@}*/

#ifdef __CUDACC__
  /** \defgroup implementation algorithm and implementation related parameters */
  /**@{*/
  /** Device memory allocation type */
  hydroSimu::DeviceArray<real_t>::DeviceMemoryAllocType gpuMemAllocType;
  /**@}*/
#endif // __CUDACC__

}; // struct HydroParameters

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// ------ STRUCT HydroParameters
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
inline HydroParameters::HydroParameters(ConfigMap &_configMap, 
					bool _initGpu=true) 
  : configMap(_configMap),
    nStepmax(0), tEnd(0.0), nOutput(0),
    nx(0), ny(0), nz(0), nbVar(0), dimType(TWO_D), 
    /*xMin(0.0), xMax(0.0), yMin(0.0), yMax(0.0), zMin(0.0), zMax(0.0),
      dx(0.0), dy(0.0), dz(0.0),*/
    boundary_xmin(), boundary_xmax(),
    boundary_ymin(), boundary_ymax(),
    boundary_zmin(), boundary_zmax(),
    ghostWidth(2),
    problem(""), cfl(0.0), _slope_type(0.0),
    mhdEnabled(false),
    enableJet(0), ijet(0), djet(0.0), ujet(0.0), pjet(0.0), cjet(0.0), offsetJet(0),
    imin(0), imax(0), jmin(0), jmax(0), kmin(0), kmax(0), 
    isize(0), jsize(0), ksize(0),
    outputVtkEnabled(false), outputHdf5Enabled(false), 
    outputNetcdf4Enabled(false), 
    outputPnetcdfEnabled(false),
    outputXsmEnabled(false), outputPngEnabled(false), outputNrrdEnabled(false)
#ifdef __CUDACC__
  ,gpuMemAllocType(hydroSimu::DeviceArray<real_t>::PITCHED)
#endif // __CUDACC__
{

  /* 
   * NOTE : default values are taken from hydro_module.f90 
   */

  /* initialize RUN parameters */
  nStepmax = configMap.getInteger("run","nstepmax",1000);
  tEnd     = configMap.getFloat  ("run","tend",0.0);
  nOutput  = configMap.getInteger("run","noutput",100);
  
  /* initialize MESH parameters */
  nx = configMap.getInteger("mesh","nx", 2);
  ny = configMap.getInteger("mesh","ny", 2);
  nz = configMap.getInteger("mesh","nz", 1);
  if (nz == 1) {
    dimType = TWO_D;
    nbVar = NVAR_2D;
  } else {
    dimType = THREE_D;
    nbVar = NVAR_3D;
  }
  // the follwing is necessary so that grid resolution can be used on GPU
  // (put in constant memory space)
  _gParams.nx = nx;
  _gParams.ny = ny;
  _gParams.nz = nz;

  // default value for MPI cartesian coordinate (relevant values will be set in
  // HydroMpiParameters constructor).
  _gParams.mpiPosX = 0;
  _gParams.mpiPosY = 0;
  _gParams.mpiPosZ = 0; 

  // default value for MPI mesh size, i.e. number of MPI process in each direction
  // relevant values will be set in HydroMpiParameters constructor.
  _gParams.mx = 1;
  _gParams.my = 1;
  _gParams.mz = 1; 

  /* test if we are doing MHD (default : no) */
  mhdEnabled = configMap.getBool("MHD","enable", false);
  if (mhdEnabled)
    nbVar = NVAR_MHD;

  // copy HydroParameters::nbVar into gParams.nbVar
  _gParams.nbVar = nbVar;

  /* domain geometry */
  _gParams.xMin = configMap.getFloat("mesh","xmin",0.0);
  _gParams.xMax = configMap.getFloat("mesh","xmax",1.0);
  _gParams.yMin = configMap.getFloat("mesh","ymin",0.0);
  _gParams.yMax = configMap.getFloat("mesh","ymax",1.0);
  _gParams.zMin = configMap.getFloat("mesh","zmin",0.0);
  _gParams.zMax = configMap.getFloat("mesh","zmax",1.0);

  _gParams.dx = (_gParams.xMax- _gParams.xMin)/nx;
  _gParams.dy = (_gParams.yMax- _gParams.yMin)/ny;
  _gParams.dz = (_gParams.zMax- _gParams.zMin)/nz;

  /* grid geometry type */
  _gParams.geometry = configMap.getInteger("mesh","geometry", GEO_CARTESIAN);

  /* boundary condition types */
  boundary_xmin = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_xmin", BC_DIRICHLET));
  boundary_xmax = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_xmax", BC_DIRICHLET));
  boundary_ymin = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_ymin", BC_DIRICHLET));
  boundary_ymax = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_ymax", BC_DIRICHLET));
  boundary_zmin = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_zmin", BC_DIRICHLET));
  boundary_zmax = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_zmax", BC_DIRICHLET));
  
  ghostWidth = configMap.getInteger("mesh","ghostWidth", 2);
  if (ghostWidth !=2 and ghostWidth !=3) {
    std::cout << "wrong ghostWidth parameter: " <<  ghostWidth << std::endl;
    std::cout << "You need to change your parameter file" << std::endl;
    std::cout << "ghostWidth is set to 2 !!!" << std::endl;
    ghostWidth = 2;
  }
  // if MHD is enabled make sure we use the correct ghostWidth
  if (mhdEnabled) {
    ghostWidth = 3;
    std::cout << "MHD is enabled : we use ghostWidth=3" << std::endl;
  }

  /* initialize HYDRO parameters */
  cfl           = _configMap.getFloat("hydro", "cfl", 0.5);
  std::cout << "### CFL ### " << cfl << std::endl;
  // cfl can't be zero (take care that if env variable LANG is not correct
  // float variable are not correctly parsed (in French "," is used instead of 
  // "." (this is a mess !!)
  if (!cfl) {
    cfl = 0.5;
    std::cout << "### warning ###, parameter cfl was zero; set to 0.5\n";
  }

  /* get problem name (default problem is unknown) */
  problem = configMap.getString("hydro","problem", "unknown");

  std::cout << "problem : " << problem << std::endl;

  /* initialize global parameters (used in both the CPU and the GPU code */

  /* if cIso != ZERO, then we use isothermal equation of state */
  _gParams.cIso          = configMap.getFloat("hydro","cIso",0);
  _gParams.gamma0        = configMap.getFloat("hydro","gamma0", 1.4);
  if (_gParams.cIso > 0) {
    std::cout << "Using Isothermal  with cIso = " << _gParams.cIso << std::endl;
  } else {
    std::cout << "Using perfect gas equation of state with gamma = " << _gParams.gamma0 << std::endl;
  }

  /** 
   * \todo add a switch variable here to change smalle, smallp and smallpp
   * when doing isothermal simulations.
   */
  _gParams.smallr        = configMap.getFloat("hydro","smallr", 1e-10);
  _gParams.smallc        = configMap.getFloat("hydro","smallc", 1e-10);
  _gParams.niter_riemann = configMap.getInteger("hydro","niter_riemann", 10);
  _gParams.iorder        = configMap.getInteger("hydro","iorder", 2);
  _gParams.smalle        = 1e-7; // _gParams.smallc*_gParams.smallc/_gParams.gamma0/(_gParams.gamma0-1.0); /* never used ? */
  _gParams.smallp        = _gParams.smallc * _gParams.smallc / _gParams.gamma0;
  if (_gParams.cIso>0)   _gParams.smallp = _gParams.smallr * _gParams.cIso * _gParams.cIso;
  _gParams.smallpp       = _gParams.smallr * _gParams.smallp;
  _gParams.gamma6        = (_gParams.gamma0 + 1.0f)/(2.0f * _gParams.gamma0);
  _gParams.Omega0        = configMap.getFloat("MHD","omega0", 0.0f);
  _gParams.ALPHA         = configMap.getFloat("hydro","ALPHA", 0.9f);
  _gParams.BETA          = configMap.getFloat("hydro","BETA" , 0.1f);
  _gParams.XLAMBDA       = configMap.getFloat("hydro","XLAMBDA" , 0.25f);
  _gParams.YLAMBDA       = configMap.getFloat("hydro","YLAMBDA" , 0.25f);
  _gParams.ALPHA_KT      = configMap.getFloat("hydro","ALPHA_KT" , 1.4f);
  _gParams.slope_type    = configMap.getFloat("hydro","slope_type",1.0);
  if (configMap.getInteger("hydro", "traceVersion", 1) == 0)
    _gParams.slope_type = 0.0;
  _gParams.gravity_x     = configMap.getFloat("gravity", "static_field_x", 0.0f);
  _gParams.gravity_y     = configMap.getFloat("gravity", "static_field_y", 0.0f);
  _gParams.gravity_z     = configMap.getFloat("gravity", "static_field_z", 0.0f);

  // read dissipative term paramaters
  _gParams.nu            = configMap.getFloat("hydro","nu", 0.0); // viscosity
  _gParams.eta           = configMap.getFloat("MHD"  ,"eta",0.0); // resistivity

  // scheme
  std::string scheme_names[] = {"muscl", "plmde", "collela"};
  std::string scheme_str = configMap.getString("hydro","scheme", "muscl");
  if(scheme_str.compare(scheme_names[0]) == 0) // MUSCL-Hancock method
    {
      _gParams.scheme = MusclScheme;
    }
  else if(scheme_str.compare(scheme_names[1]) == 0) // standard PLMDE
    {
      _gParams.scheme = PlmdeScheme;
    }
  else if(scheme_str.compare(scheme_names[2]) == 0) // Collela's method
    {
      _gParams.scheme = CollelaScheme;
    }
  else
    _gParams.scheme = UnknownScheme;
  std::cout << "GPU : scheme : " << _gParams.scheme << std::endl;

  /*
   * Riemann solver (used in hydro fluxes)
   */
  // default value
  std::string riemannSolver_str = configMap.getString("hydro","riemannSolver", "approx");
  _gParams.riemannSolver = APPROX;
  // transform to lower case parsed string
  std::transform(riemannSolver_str.begin(),
		 riemannSolver_str.end(),
		 riemannSolver_str.begin(),
		 ::tolower);
  // create a map with allowed values
  std::map<std::string, RiemannSolverType> riemannSolverMap;
  riemannSolverMap["approx"]=APPROX;
  riemannSolverMap["hll"]=HLL;
  riemannSolverMap["hllc"]=HLLC;
  if (mhdEnabled) {
    riemannSolverMap["hlld"]=HLLD;
    riemannSolverMap["llf"]=LLF;
  }
  // traverse map to find
  bool riemannSolverFound = false;
  std::map<std::string, RiemannSolverType>::iterator it;
  for ( it=riemannSolverMap.begin() ; it != riemannSolverMap.end(); it++ ) {
    if ( 0 == riemannSolver_str.compare( (*it).first ) ) {
      _gParams.riemannSolver = (*it).second;
      riemannSolverFound = true;
      break;
    }
  }
  std::cout << "Riemann solver (hydro flux) found   : " << riemannSolverFound << std::endl;
  std::cout << "Riemann solver (hydro flux) string  : " << riemannSolver_str << std::endl;
  std::cout << "Riemann solver (hydro flux) type    : " << _gParams.riemannSolver << std::endl;
  
  /*
   * Riemann solver (used in MHD fluxes)
   * see parameter Riemann2d in routine cmp_mag_flx in DUMSES
   */
  // default value
  std::string magRiemannSolver_str = configMap.getString("MHD","magRiemannSolver", "hlld");
  _gParams.magRiemannSolver = MAG_HLLD;
  if (mhdEnabled) {
    // transform to lower case parsed string
    std::transform(magRiemannSolver_str.begin(),
		   magRiemannSolver_str.end(),
		   magRiemannSolver_str.begin(),
		   ::tolower);
    // create a map with allowed values
    std::map<std::string, MagneticRiemannSolverType> magRiemannSolverMap;
    magRiemannSolverMap["hlld"]   = MAG_HLLD;
    magRiemannSolverMap["hllf"]   = MAG_HLLF;
    magRiemannSolverMap["hlla"]   = MAG_HLLA;
    magRiemannSolverMap["roe"]    = MAG_ROE;
    magRiemannSolverMap["llf"]    = MAG_LLF;
    magRiemannSolverMap["upwind"] = MAG_UPWIND;
    // traverse map to find
    bool magRiemannSolverFound = false;
    std::map<std::string, MagneticRiemannSolverType>::iterator it;
    for ( it=magRiemannSolverMap.begin() ; it != magRiemannSolverMap.end(); it++ ) {
      if ( 0 == magRiemannSolver_str.compare( (*it).first ) ) {
	_gParams.magRiemannSolver = (*it).second;
	magRiemannSolverFound = true;
	break;
      }
    }
    std::cout << "Riemann solver (mag flux) found   : " << magRiemannSolverFound << std::endl;
    std::cout << "Riemann solver (mag flux) string  : " << magRiemannSolver_str << std::endl;
    std::cout << "Riemann solver (mag flux) type    : " << _gParams.magRiemannSolver << std::endl;
  } // end mhdEnabled

  /*
   * Global constants
   */
#ifdef __CUDACC__
  // copy global constants into device constant memory : see routine
  // copyToSymbolMemory() : this routine must be called once in every
  // compilation/translation unit, since __constant__ variables are
  // file-scoped !!!
#else // CPU version
  // we copy this structure into a global variable, just for symetry
  // with the CUDA code, to be able to call the same functions
  // declared with keywords __device__ and __host__
  ::gParams = _gParams;
#endif // __CUDACC__

  /* initialize JET parameters */
  if (!problem.compare("jet"))
    enableJet=1;
  else
    enableJet=0;
  ijet = configMap.getInteger("jet","ijet",0);
  djet = configMap.getFloat  ("jet","djet",1.0);
  ujet = configMap.getFloat  ("jet","ujet",0.0);
  pjet = configMap.getFloat  ("jet","pjet",0.0);
  cjet = SQRT(_gParams.gamma0*pjet/djet);
  offsetJet = configMap.getInteger("jet","offsetJet",0);

  /* initialize other parameters; assume we use a border of 2 ghost cells */  
  imin = 0;
  imax = nx-1+2*ghostWidth;
  jmin = 0;
  jmax = ny-1+2*ghostWidth;
  
  kmin = 0;
  if (nz == 1) {
    kmax = 0;
  } else {
    kmax = nz-1+2*ghostWidth;
  }
  
  isize = imax - imin + 1;
  jsize = jmax - jmin + 1;
  ksize = kmax - kmin + 1;

  // do we want VTK output ?
  outputVtkEnabled  = configMap.getBool("output","outputVtk", true);
  
  // do we want HDF5 output ?
  outputHdf5Enabled = configMap.getBool("output","outputHdf5", true);

  // do we want NETCDF4 output ?
  outputNetcdf4Enabled = configMap.getBool("output","outputNetcdf4", false);

  // do we want Parallel NETCDF output ? Only valid/activated for MPI run
  outputPnetcdfEnabled = configMap.getBool("output","outputPnetcdf", false);

  // do we want Xsmurf (raw binary + one line ascii header) output ?
  // Please note that this format is used in test script test_run.sh
  // to compare CPU/GPU performances
  outputXsmEnabled  = configMap.getBool("output","outputXsm", false);

  // do we want Png (ImageMagick must be available and detected by configure,
  // otherwise this does nothing)
  outputPngEnabled  = configMap.getBool("output","outputPng", false);

  // do we want NRRD output ?
  outputNrrdEnabled = configMap.getBool("output","outputNrrd", false);

  // GPU memory allocation type
#ifdef __CUDACC__
  std::string gpuAllocString = configMap.getString("implementation","DeviceMemoryAllocType", "PITCHED");
  
  if (!gpuAllocString.compare("LINEAR")) {
    gpuMemAllocType = hydroSimu::DeviceArray<real_t>::LINEAR;
    std::cout << "Using GPU memory allocation type : LINEAR" << std::endl;
  } else if (!gpuAllocString.compare("PITCHED")) {
    gpuMemAllocType = hydroSimu::DeviceArray<real_t>::PITCHED;
    std::cout << "Using GPU memory allocation type : PITCHED" << std::endl;
  } else {
    std::cout << "WARNING: unknown GPU memory allocation type !!!" << std::endl;
    std::cout << "Possible values are LINEAR and PITCHED)        " << std::endl;
    std::cout << "We will use the default value.                 " << std::endl;
  }
#endif // __CUDACC__

  if (_initGpu) {
    /*
     * choose a CUDA device (if running the GPU version)
     */
#ifdef __CUDACC__  
    cutilSafeCall( cudaSetDevice(cutGetMaxGflopsDeviceId()) );
    cudaDeviceProp deviceProp;
    int myDevId = -1;
    cutilSafeCall( cudaGetDevice( &myDevId ) );
    cutilSafeCall( cudaGetDeviceProperties( &deviceProp, myDevId ) );
    std::cout << "[GPU] myDevId : " << myDevId << " (" << deviceProp.name << ")" << std::endl;
#endif // __CUDACC__
   
  
    /*
     * copy some usefull parameter to constant memory space located on device 
     * (if using GPU, otherwise this routine does nothing...) 
     */
    copyToSymbolMemory();
  }

} // HydroParameters::HydroParameters

// -------------------------------------------------------------------------------
// -------------------------------------------------------------------------------
inline void HydroParameters::copyToSymbolMemory()
{

#ifdef __CUDACC__
  cutilSafeCall( cudaMemcpyToSymbol(::gParams, &_gParams, sizeof(GlobalConstants), 0, cudaMemcpyHostToDevice ) );
#else
  gParams = _gParams;
#endif // __CUDACC__



} // HydroParameters::copyToSymbolMemory

void inline swapValues(real_t *a, real_t *b)
{
  real_t tmp = *a;
  
  *a = *b;
  *b = tmp;

} // swapValues


#endif /*HYDROPARAMETERS_H_*/
