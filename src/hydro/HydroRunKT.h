/**
 * \file HydroRunKT.h
 * \brief
 * Class HydroRunKT derives from HydroRunBase, and
 * implement a fully discrete central scheme as described by Kurganov
 * and Tadmor (KT) to solve Euler equation.
 *
 * Some of the code is derived and adapted from CentPack
 * http://www.cscamm.umd.edu/centpack/
 * See also original articles:
 * - Kurganov and Tadmor, "Solution of two-dimensional
 * Riemann problems for gas dynamics without Riemann problem solver",
 * Numer. Methods Partial Differential Equations, vol 18, pp 548-608, 2002.
 * - Kurganov and Tadmor, "New high-resolution central
 * schemes for nonlinear conservation laws and convection-diffusion
 * equations.", J. Comput. Phys., 160(1):241-282, 2000. 
 * 
 * \date 05/02/2010
 * \author P. Kestener
 *
 * $Id: HydroRunKT.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef HYDRORUN_KT_H_
#define HYDRORUN_KT_H_

#include "real_type.h"
#include "common_types.h"
#include "gpu_macros.h"
#include <cmath>

#include "HydroRunBase.h"

namespace hydroSimu {

/**
 * \class HydroRunKT HydroRunKT.h
 * \brief This class implements hydro simulations using the Kurganov-Tadmor numerical
 * scheme.
 */
class HydroRunKT : public HydroRunBase
{
public:
  HydroRunKT(ConfigMap &_configMap);
  ~HydroRunKT();

  void start(); 
  void oneStepIntegration(int& nStep, real_t& t, real_t& dt);

  uint cmpdtBlockCount;
  void copyGpuToCpu(int nStep=0);

protected:

#ifdef __CUDACC__
  void kt_evolve();
#else
  void kt_evolve();
  void reconstruction_2d_FD2();
  void predictor_corrector_2d_FD2();
#endif
  real_t computeDt();

  real_t dX, dY;
  real_t xLambda, yLambda;
  bool odd;

private:

  // Data Arrays
  HostArray<real2_t> h_spectralRadii;
#ifdef __CUDACC__
  DeviceArray<real2_t> d_spectralRadii;
#endif // __CUDACC__

  HostArray<real_t> h_Uhalf;
#ifdef __CUDACC__
  DeviceArray<real_t> d_Uhalf;
#else
  HostArray<real_t> h_Uprime;
  HostArray<real_t> h_Uqrime;
  HostArray<real_t> h_Ustar;
  HostArray<real_t> f,g;
  HostArray<real_t> f_prime,g_qrime;
#endif // __CUDACC__

};

} // namespace hydroSimu

#endif /*HYDRORUN_KT_H_*/
