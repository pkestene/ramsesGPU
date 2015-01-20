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
 * \file Forcing_OrnsteinUhlenbeck.h
 * \brief Implementation of ForcingOrnsteinUhlenbeck class
 *
 * \author P. Kestener
 * \date 19/12/2013
 *
 * $Id: Forcing_OrnsteinUhlenbeck.h 3465 2014-06-29 21:28:48Z pkestene $
 */
#ifndef FORCING_ORNSTEIN_UHLENBECK_H_
#define FORCING_ORNSTEIN_UHLENBECK_H_

#include "RandomGen.h"

#include "real_type.h"
#include "constants.h" // for gParams
#include "Arrays.h"
#include <ConfigMap.h>

#ifdef __CUDACC__
#include <curand_kernel.h> // for curandState type
#endif

namespace hydroSimu {

  /**
   * \class ForcingOrnsteinUhlenbeck
   *
   * Some reference about Ornstein-Uhlenbeck process:
   *
   * "Exact numerical simulation of the Ornstein-Uhlenbeck process and
   * its integral", Daniel T. Gillespie, PhysRevE, vol 54, num 2, 1995.
   * See equations (1.1) and (1.9a)
   */
  class ForcingOrnsteinUhlenbeck
  {
    
  public:
    static const int nMode = 31;

    /** constructor */
    ForcingOrnsteinUhlenbeck(int _nDim, 
			     int _nCpu, 
			     ConfigMap &_configMap,
			     GlobalConstants &_gParams);

    /** destructor */
    virtual ~ForcingOrnsteinUhlenbeck();

    double timeScaleTurb;
    double amplitudeTurb;
    double ksi;          //!< new name for parturfor (used in litterature)
    int    init_random;  //
    int    nDim;         //!< dimension (only 3 is supported, for now)
    int    nCpu;         //!< number of MPI tasks
    
    RandomGen *pRandomGen;  //!< the random number generator for CPU computation

    ConfigMap configMap; //!< Global config map (used here to determine IO filenames

    // some array used in the forcing term computation
    double *mode;         //!< Fourier mode array (total size is  nDim*nMode)
    double *forcingField; //!< Forcing Field in Fourrier domain (total size is  nDim*nMode)
    double *projTens;     //!< Projection tensor (total size is  nDim*nDim*nMode)
    int    *gaussSeed;    //!< Gaussian random generator seed array (total size is  nCpu*4)
    int     forceSeed[4]; //!< The seed used to generated Gaussian N(0,1) random number

#ifdef __CUDACC__
    double *d_mode, *d_forcingField, *d_projTens; // CUDA pointers
    curandState *deviceStates; //!< Array of random generator states
#endif

    /** memory allocation */
    void allocate();
    
    /** memory free */
    void free();
    
    /** init forcing parameters (read them from file upon restart) */
    virtual void init_forcing(bool restartEnabled=false, int nStep=0);
    
    /** output forcing parameters to file */
    virtual void output_forcing(int nStep);

    /** input forcing parameters (mode, forcingField, gaussSeed) from file */
    virtual void input_forcing(std::string forcing_filename="", int nStep=0);

    /** compute new forcing field modes at t+dt */
    virtual void update_forcing_field_mode(real_t dt);

    /** add forcing field (modify velocity field and total energy) */
#ifdef __CUDACC__
    virtual void add_forcing_field(DeviceArray<real_t> &U,
				   real_t dt);
#else
    virtual void add_forcing_field(HostArray<real_t> &U,
				   real_t dt);
#endif

  }; // class ForcingOrnsteinUhlenbeck

} // namespace hydroSimu

#endif /* FORCING_ORNSTEIN_UHLENBECK_H_ */
