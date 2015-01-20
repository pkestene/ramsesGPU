/**
 * \file PapiInfo.h
 * \brief A simple PAPI interface class.
 *
 * Parts of this class is adapted from file sc_flops.c found in library
 * libsc (https://github.com/cburstedde/libsc)
 *
 * \author Pierre Kestener
 * \date March 3rd, 2014
 *
 * $Id$
 */
#ifndef PAPI_INFO_H_
#define PAPI_INFO_H_

#include "Timer.h"

namespace hydroSimu {
  
  class PapiInfo
  {
  public:
    /**
     * constructor
     */
    PapiInfo();
    
    /**
     * destructor
     */
    ~PapiInfo();

    void start();
    void stop();
    double getFlops();
    long long int getFlop();
    double elapsed();

  protected:
    /* Wall clock time */
    Timer               papiTimer;
    
    /* cumulative counters */
    float               crtime;   /* cumulative real time */
    float               cptime;   /* cumulative process time */
    long long int       cflpops;  /* cumulative floating point operations */

    /* values used in start routine */
    float               irtime;   /* interval real time */
    float               iptime;   /* interval process time */
    long long int       iflpops;  /* interval floating point operations */

    double              mflops;   /* MFlop/s rate  */
    
  }; // class PapiInfo

} // namespace hydroSimu

#endif // PAPI_INFO_H_
