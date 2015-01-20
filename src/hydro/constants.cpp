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
 * \file constants.cpp
 * \brief Create an instance of the C++ GlobalConstants structure gathering simulation parameters.
 *
 * The GlobalConstants structure contains hydrodynamics simulation
 * parameters from the original fortran namelist located in
 * read_params.f90, and also parameters for the new numerical schemes
 * (Kurganov-Tadmor, Lax-Liu, ...) and also for MHD.
 *
 * \author P. Kestener
 * \date 28/06/2009
 *
 * $Id: constants.cpp 3234 2014-02-03 16:40:34Z pkestene $
 */
#include "constants.h"
#include <cstdlib>

#ifndef __CUDACC__
GlobalConstants gParams;

// define reference to gParams fields, just to minimize changes with
// the old version
/*real_t& gamma0        = gParams.gamma0;
real_t& smallr        = gParams.smallr;
real_t& smallc        = gParams.smallc;
int   & niter_riemann = gParams.niter_riemann;
int   & iorder        = gParams.iorder;
Scheme& scheme        = gParams.scheme;
real_t& smalle        = gParams.smalle;
real_t& smallp        = gParams.smallp;
real_t& smallpp       = gParams.smallpp;
real_t& gamma6        = gParams.gamma6;
real_t& ALPHA         = gParams.ALPHA;
real_t& BETA          = gParams.BETA;
real_t& XLAMBDA       = gParams.XLAMBDA;
real_t& YLAMBDA       = gParams.YLAMBDA;
real_t& ALPHA_KT      = gParams.ALPHA_KT;
real_t& slope_type    = gParams.slope_type;*/

#endif // __CUDACC__


