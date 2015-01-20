/**
 * \file HydroRunLaxLiu.h
 * \brief Defines a class to perfrom hydro simulations following the
 * scheme proposed in article: P.D. Lax, X.-D. Liu, "Solution of
 * Two-Dimensional Riemann Problems of Gas Dynamics by Positive
 * Schemes", SIAM J. Sci. Comput., vol. 19, pp. 319-340, (1998).
 *
 * \date January 2010
 * \author P. Kestener
 *
 * $Id: HydroRunLaxLiu.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef HYDRORUN_LAXLIU_H_
#define HYDRORUN_LAXLIU_H_

#include "common_types.h"
#include "gpu_macros.h"
#include <cmath>

#include "HydroRunBase.h"

namespace hydroSimu {

/**
 * \class HydroRunLaxLiu HydroRunLaxLiu.h
 * \brief This class derives from HydroRunBase, and
 * implement a positive scheme alogorithm to solve Euler equation.
 *
 * Some of the code is derived and adapted from Fortran found in
 * original article by Lax and Liu "Solution of the two-dimensional
 * Riemann problems of gas dynamics by positive schemes", SIMA
 * J. Sci. Comput., vol 19, pp 319-340, 1998.
 */
class HydroRunLaxLiu : public HydroRunBase
{
public:
  HydroRunLaxLiu(ConfigMap &_configMap);
  ~HydroRunLaxLiu();

  void init_hydro_jet();
  void start(); 

  //! perform only one time step integration
  void oneStepIntegration(int& nStep, real_t& t, real_t& dt);

protected:
#ifdef __CUDACC__
  void laxliu_evolve(DeviceArray<float> &a1, DeviceArray<float> &a2);
  void averageArray (DeviceArray<float> &a1, DeviceArray<float> &a2);
#else
  void laxliu_evolve(HostArray<float>   &a1, HostArray<float>   &a2);
  void averageArray (HostArray<float>   &a1, HostArray<float>   &a2);
#endif


private:

  // Data Arrays
  HostArray<float> h_U1;
  HostArray<float> h_tmp1;
#ifdef __CUDACC__
  DeviceArray<float> d_U1;
  DeviceArray<float> d_tmp1;
#endif // __CUDACC__

};

} // namespace hydroSimu

#endif /*HYDRORUN_LAXLIU_H_*/
