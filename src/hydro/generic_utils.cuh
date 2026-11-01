/**
 * Copyright CEA / Maison de la Simulation
 * Contributors: Pierre Kestener, Sebastien Fromang (May 22, 2012)
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 */

/**
 * \file generic_utils.cuh
 * \brief Defines some useful utilities.
 */
#ifndef GENERIC_UTILS_CUH_H_
#define GENERIC_UTILS_CUH_H_

#include "real_type.h"

/** a dummy device-only swap function */
__inline__ __device__ void
swap_value(real_t & a, real_t & b)
{

  real_t tmp = a;
  a = b;
  b = tmp;

} // swap_value

#endif // GENERIC_UTILS_CUH_H_
