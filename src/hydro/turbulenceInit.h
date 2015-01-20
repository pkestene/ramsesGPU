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
 * \file turbulenceInit.h
 * \brief Create random fiels used to initialize velocity field in a turbulence simulation.
 *
 * This turbulence initialization is adapted from Enzo (by A. Kritsuk) in file
 * turboinit.f
 *
 * \author P. Kestener
 * \date 01/09/2012
 *
 * $Id: turbulenceInit.h 2395 2012-09-14 12:45:17Z pkestene $
 *
 */
#ifndef TURBULENCE_INIT_H_
#define TURBULENCE_INIT_H_

#include "real_type.h"

void turbulenceInit(int sizeX,   int sizeY,   int sizeZ,
		    int offsetX, int offsetY, int offsetZ,
		    int nbox, real_t randomForcingMachNumber,
		    real_t *u, real_t *v, real_t *w);
#endif // TURBULENCE_INIT_H_
