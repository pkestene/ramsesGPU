/**
 * \file structureFunctionsMpi.h
 *
 * Compute structure functions with MPI enabled.
 * See src/analysis/structureFunctionsMpi.cpp for algorithm description.
 *
 * \author Pierre Kestener
 * \date 29 June 2014
 *
 * $Id: structureFunctionsMpi.h 3467 2014-06-30 10:21:44Z pkestene $
 */
#ifndef STRUCTURE_FUNCTIONS_MPI_H_
#define STRUCTURE_FUNCTIONS_MPI_H_

#include "real_type.h"
#include "Arrays.h"
#include "utils/config/ConfigMap.h"
#include "constants.h"

namespace hydroSimu {

  // =======================================================
  // =======================================================
  void structure_functions_hydro_mpi(int myRank,
				     int nStep, 
				     ConfigMap &configMap,
				     GlobalConstants &_gParams,
				     HostArray<real_t> &U);
  
  // =======================================================
  // =======================================================
  void structure_functions_mhd_mpi(int myRank,
				   int nStep,
				   ConfigMap &configMap,
				   GlobalConstants &_gParams,
				   HostArray<real_t> &U);

} // namespace hydroSimu

#endif // STRUCTURE_FUNCTIONS_MPI_H_
