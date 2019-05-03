/**
 * \file structureFunctions.h
 *
 * Compute structure functions for mono-CPU or mono-GPU.
 * See src/analysis/structureFunctionsMpi.cpp for algorithm description.
 *
 * \author Pierre Kestener
 * \date 29 June 2014
 *
 * $Id: structureFunctions.h 3465 2014-06-29 21:28:48Z pkestene $
 */
#ifndef STRUCTURE_FUNCTIONS_H_
#define STRUCTURE_FUNCTIONS_H_

#include "real_type.h"
#include "Arrays.h"
#include "utils/config/ConfigMap.h"
#include "constants.h"

namespace hydroSimu {

  // =======================================================
  // =======================================================
  void structure_functions_hydro(int nStep, 
				 ConfigMap &configMap,
				 GlobalConstants &_gParams,
				 HostArray<real_t> &U);
  
  // =======================================================
  // =======================================================
  void structure_functions_mhd(int nStep,
			       ConfigMap &configMap,
			       GlobalConstants &_gParams,
			       HostArray<real_t> &U);

} // namespace hydroSimu

#endif // STRUCTURE_FUNCTIONS_H_

