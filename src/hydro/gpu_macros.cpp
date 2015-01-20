/*
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
 * \file gpu_macros.cpp
 * \brief Some useful GPU related macros.
 *
 * \author P. Kestener
 *
 * $Id: gpu_macros.cpp 2108 2012-05-23 12:07:21Z pkestene $
 */
#include "gpu_macros.h"

float saturate_cpu(float a) {
  
  if (isnanf(a)) return 0.0f;
  return a >= 1.0f ? 1.0f : a <= 0.0f ? 0.0f : a;

};

#ifdef __CUDACC__
uint blocksFor(uint elementCount, uint threadCount)
{
  return (elementCount + threadCount - 1) / threadCount;
}
#endif // __CUDACC__
