/**
 * \file Timer.h
 * \brief A simple timer class.
 *
 * \author Pierre Kestener
 * \date 29 Oct 2010
 *
 * $Id: Timer.h 1783 2012-02-21 10:20:07Z pkestene $
 */
#ifndef MONITORING_TIMER_H_
#define MONITORING_TIMER_H_

#include <time.h>
#include <sys/time.h> // for gettimeofday and struct timeval

typedef struct timeval timeval_t;

namespace hydroSimu {

  /**
   * \brief a simple Timer class.
   * If MPI is enabled, should we use MPI_WTime instead of gettimeofday (?!?)
   */
  class Timer
  {
  public:
    /** default constructor, timing starts rightaway */
    Timer();
    
    Timer(double t);
    Timer(Timer const& aTimer);
    ~Timer();

    /** start time measure */
    void start();
    
    /** stop time measure and add result to total_time */
    void stop();

    /** return elapsed time in seconds (as stored in total_time) */
    double elapsed() const;

  protected:
    timeval_t start_time;

    /** store total accumulated timings */
    double    total_time;

  }; // class Timer

} // namespace hydroSimu

#endif // MONITORING_TIMER_H_
