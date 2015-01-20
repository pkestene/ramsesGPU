/**
 * \file constants.h
 * \brief Defines some usefull enumerations and constants.
 *
 * \author F. Chateau and P. Kestener
 *
 * $Id: constants.h 3472 2014-07-02 21:20:49Z pkestene $
 */
#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include "real_type.h"
#include "gpu_macros.h"

// memory alignment 
#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(__PGI) // PGI
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

/**
 * number of variables for hydro simulations (size of vector state
 * used in Riemann solver routines) or MHD simulations.
 *
 * when doing MHD simulations (2D or 3D), always use 5 hydro + 3
 * magnetic components
 */
enum NvarSimulation {
  NVAR_2D=4, /*!< Hydro-only, 2D */
  NVAR_3D=5, /*!< Hydro-only, 3D */
  NVAR_MHD=8 /*!< MHD, 2D or 3D */
};

//! Grid geometry type
enum GeometryType {
  GEO_CARTESIAN = 0,
  GEO_CYLINDRICAL = 1,
  GEO_SPHERICAL = 2
};

//! hydro/MHD field indexes
enum ComponentIndex {
  ID=0,  /*!< ID Density field index */
  IP=1,  /*!< IP Pressure/Energy field index */
  IU=2,  /*!< X velocity / momentum index */
  IV=3,  /*!< Y velocity / momentum index */ 
  IW=4,  /*!< Z velocity / momentum index */ 
  IBX=5, /*!< X component of magnetic field */ 
  IBY=6, /*!< Y component of magnetic field */ 
  IBZ=7, /*!< Z component of magnetic field */
  IA=5,  /*!< X component of magnetic field */ 
  IB=6,  /*!< Y component of magnetic field */ 
  IC=7  /*!< Z component of magnetic field */
};

//! enum used in shearing box computations
enum ShearRemapIndex {
  I_DENS=0,
  I_EMF_Y=1
};

//! enum number of component for remapping buffer 
//! (number of item in enum ShearRemapIndex)
enum ShearRemapComponent {
  NUM_COMPONENT_REMAP=2
};

//! enum 3D component index
enum ComponentIndex3D {
  IX = 0,
  IY = 1,
  IZ = 2
};

//! dimension of the problem
enum DimensionType {
  TWO_D = 2, 
  THREE_D = 3
};

//! numerical scheme for hydrodynamics
enum NumScheme {
  GODUNOV,  /*!< Godunov type scheme (Riemann solver + slope + trace) */
  KURGANOV, /*!< Kurganov-Tadmor centered scheme (Riemann solver-free) */
  RELAXING  /*!< Relaxing TVD scheme (Riemann solver-free) */
};

//! used in trace computation
enum Scheme {UnknownScheme, MusclScheme, PlmdeScheme, CollelaScheme};

//! Riemann solver type for hydro fluxes
enum RiemannSolverType {
  APPROX, /*!< quasi-exact Riemann solver (hydro-only) */ 
  HLL,    /*!< HLL hydro and MHD Riemann solver */
  HLLC,   /*!< HLLC hydro-only Riemann solver */ 
  HLLD,   /*!< HLLD MHD-only Riemann solver */
  LLF     /*!< Local Lax-Friedrich Riemann solver (MHD only) */
};

//! Riemann solver type for magnetic fluxes
enum MagneticRiemannSolverType {
  MAG_HLLD,
  MAG_HLLF,
  MAG_HLLA,
  MAG_ROE, /* not implemented (will probably never be) */
  MAG_LLF,
  MAG_UPWIND /* not implemented */
};

// gravitational constant (6.674x10−11 N m^2 kg^−2 in SI units)
#define GRAV_UNIV_CST (1.0)

enum SelfGravityMethod {
  SG_FFT_DECOMP1D, /* using FFTW + MPI or cuFFT + MPI */
  SG_FFT_DECOMP2D, /* P3DFFT (CPU) or DiGPFFT (GPU) */
  SG_MULTIGRID     /* TO DO */
};

//! enum edge index (use in MHD - EMF computations)
enum EdgeIndex {
  IRT = 0, /*!< RT (Right - Top   ) */
  IRB = 1, /*!< RB (Right - Bottom) */
  ILT = 2, /*!< LT (Left  - Top   ) */
  ILB = 3  /*!< LB (Left  - Bottom) */
};

//! another enum defining edge index (use in MHD - EMF computations)
enum EdgeIndex2 {
  ILL = 0,
  IRL = 1,
  ILR = 2,
  IRR = 3
};

//! enum used in MHD - EMF computations
enum EmfDir {
  EMFX = 0,
  EMFY = 1,
  EMFZ = 2
};

//! EMF indexes (EMFZ is first because in 2D, we only need EMFZ)
enum EmfIndex {
  I_EMFZ=0,
  I_EMFY=1,
  I_EMFX=2
};

//! location of the outside boundary
enum BoundaryLocation {
  XMIN = 0, 
  XMAX = 1, 
  YMIN = 2, 
  YMAX = 3, 
  ZMIN = 4,
  ZMAX = 5
};

//! type of boundary condition (note that BC_COPY is only used in the
//! MPI version for inside boundary)
enum BoundaryConditionType {
  BC_UNDEFINED, 
  BC_DIRICHLET,    /*!< reflecting border condition */
  BC_NEUMANN,      /*!< absorbing border condition */
  BC_PERIODIC,     /*!< periodic border condition */
  BC_SHEARINGBOX,  /*!< shearing box border condition (MHD only, only for X direction */
  BC_COPY,         /*!< only used in MPI parallelized version */
  BC_Z_STRATIFIED  /*!< only usefull for stratified MRI problem */
};

//! direction used in directional splitting scheme
enum Direction {XDIR=1, YDIR=2, ZDIR=3};

//! File Format 
enum FileFormat {
  FF_HDF5    = 0, /*!< for both mono/multi GPU applications */
  FF_NETCDF  = 1, /*!< for mono GPU applications */
  FF_PNETCDF = 2, /*!< for both mono/multi GPU applications (best performances) */
  FF_VTK     = 3, /*!< binary/ascii supported */
  FF_XSM     = 4, /*!< Xsmurf compatible */
  FF_NRRD    = 5, /*!< usefull for doing webgl animations */
  FF_BIN     = 6  /*!< not really used */
};

/** list of array pointers (mostly usefull only in GPU version
 *  so pointers to data can be read from the GlobalConstants structure
 *  instead of being passed to CUDA kernels as arguments) */
enum ArrayList {
  A_Q,  
  A_QM_X,
  A_QM_Y,
  A_QM_Z,
  A_QP_X,
  A_QP_Y,
  A_QP_Z,
  A_QEDGE_RT,
  A_QEDGE_RB,
  A_QEDGE_LT,
  A_QEDGE_LB,
  A_QEDGE_RT2,
  A_QEDGE_RB2,
  A_QEDGE_LT2,
  A_QEDGE_LB2,
  A_QEDGE_RT3,
  A_QEDGE_RB3,
  A_QEDGE_LT3,
  A_QEDGE_LB3,
  A_EMF,
  A_ELEC,
  A_DA,
  A_DB,
  A_DC,
  A_GRAV
};

// above enum should not have more than ARRAY_LIST_MAX items
#define ARRAY_LIST_MAX (32)

/**
 * \brief A simple structure designed to gather all parameters that
 * should go to constant memory in the CUDA/GPU version (i.e. to be
 * copied to device memory using cudaMemcpyToSymbol).
 */
struct MY_ALIGN(8) GlobalConstants
{
  real_t xMin, yMin, zMin; /*!< coordinate at origin */
  real_t xMax, yMax, zMax; /*!< domain */
  real_t dx, dy, dz; /*!< simulation resolution */
  int    nx, ny, nz; /*!< grid resolution */
  int    mpiPosX, mpiPosY, mpiPosZ; /*!< cartesian coordinate of MPI process */
  int    mx,my,mz;   /*!< mpi topology mesh sizes */
  int geometry;  /*!< grid geometry (cartesian, cylindrical or spherical) */
  int    nbVar;  /*!< number of fields in simulation (density, velocities, ...) */
  real_t gamma0; /*!< Heat capacity ratio (adiabatic index). \f$\gamma = C_p / C_v$*/
  real_t smallr; /*!< small density cut-off */
  real_t smallc; /*!< small speed of sound cut-off */
  int    niter_riemann; /*!< maximum number of iteration in Riemann solver approx */
  int    iorder; /*!< order of the numerical scheme */
  Scheme scheme; /*!< name of the numerical scheme */
  RiemannSolverType riemannSolver; /*!< identify the Riemann solver
                                      for hydro fluxes */
  MagneticRiemannSolverType magRiemannSolver;  /*!< identify the Riemann
						 solver for magnetic fluxes
						 (parameter named
						 riemann2d in dumses) */
  real_t cIso;   /*!< isothermal sound speed (if non-zero, use isothermal EOS) */
  real_t smalle; /*!< small internal energy cut-off */
  real_t smallp; /*!< small pressure cut-off */
  real_t smallpp; /*!< small pressure cut-off */
  real_t gamma6;
  real_t Omega0; /*!< Omega0 angular velocity used to computed Coriolis force.*/
  real_t ALPHA;   /*!< for used in Lax-Liu scheme */
  real_t BETA;    /*!< for used in Lax-Liu scheme */
  real_t XLAMBDA; /*!< for used in Lax-Liu scheme */
  real_t YLAMBDA; /*!< for used in Lax-Liu scheme */
  real_t ALPHA_KT; // for Kurganov-Tadmor scheme,  used in minmod routine should be between 1 and 2
  real_t slope_type; /*!< used in slope (trace computations) */
  real_t gravity_x;
  real_t gravity_y;
  real_t gravity_z;
  real_t nu;       /*!< viscosity */
  real_t eta;      /*!< resistivity (for MHD only) */
  real_t *arrayList[ARRAY_LIST_MAX];
}; // GlobalConstants

// variadic uggly macro to make nvcc happy when compiling for hardware < 2.0
#ifdef __CUDACC__
# if __CUDA_ARCH__ >= 200
#  define PRINTF(...)  printf(__VA_ARGS__)
# else
#  define PRINTF(...)
# endif

#else
# define PRINTF(...)  fprintf (stderr, __VA_ARGS__)
#endif // __CUDACC__


/** make these variables live in device constant memory if using nvcc
 * compiler; they have to be globals, because 
 * "memory qualifier on data member is not allowed"
 */
#ifdef __CUDACC__

__CONSTANT__ GlobalConstants gParams;


#else

extern GlobalConstants gParams;

#endif // __CUDACC__

#endif /*CONSTANTS_H_*/
