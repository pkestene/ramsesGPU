/**
 * \file PapiInfo.cpp
 * \brief A simple PAPI interface class.
 *
 * Parts of this class is inspired by file sc_flops.c found in library
 * libsc (https://github.com/cburstedde/libsc).
 *
 * \author Pierre Kestener
 * \date March 3rd, 2014
 *
 * $Id$
 */
#include "PapiInfo.h"

#include <time.h>
#include <sys/time.h> // for gettimeofday and struct timeval

#include <stdio.h>
#include <papi.h>

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // PapiInfo class methods body
  ////////////////////////////////////////////////////////////////////////////////
  
  // =======================================================
  // =======================================================
  PapiInfo::PapiInfo()
  {

    crtime  = 0.0f;
    cptime  = 0.0f;
    cflpops = 0;
    irtime  = 0.0f;
    iptime  = 0.0f;
    iflpops = 0;
    mflops  = 0.0;
    float tmp;

    // initialize PAPI counters
    PAPI_flops (&irtime, &iptime, &iflpops, &tmp);

  } // PapiInfo::PapiInfo
  
  // =======================================================
  // =======================================================
  PapiInfo::~PapiInfo()
  {

  } // PapiInfo::~PapiInfo

  // =======================================================
  // =======================================================
  void PapiInfo::start()
  {
    
    float tmp;

    papiTimer.start();
    PAPI_flops (&irtime, &iptime, &iflpops, &tmp);

  } // PapiInfo::start

  // =======================================================
  // =======================================================
  void PapiInfo::stop()
  {
    
    float rtime, ptime;
    long long int flpops;
    float tmp;
    
    PAPI_flops (&rtime, &ptime, &flpops, &tmp);
    papiTimer.stop();
    
    // add increment from previous call to start values to accumulator counters
    crtime  = rtime  - irtime;
    cptime  = ptime  - iptime;
    cflpops += flpops - iflpops;
    
    mflops = 1.0 * cflpops / papiTimer.elapsed() * 1e-6;

  } // PapiInfo::stop

  // =======================================================
  // =======================================================
  double PapiInfo::getFlops()
  {
    
    return mflops;

  } // PapiInfo::getFlops

  // =======================================================
  // =======================================================
  long long int PapiInfo::getFlop()
  {
    
    return cflpops;

  } // PapiInfo::getFlop

  // =======================================================
  // =======================================================
  double PapiInfo::elapsed()
  {
    
    return papiTimer.elapsed();

  } // PapiInfo::elapsed

} // namespace hydroSimu
