/**
 * \file Timer.cpp
 * \brief a simpe Timer class implementation.
 * 
 * \author Pierre Kestener
 * \date 29 Oct 2010
 *
 * $Id: Timer.cpp 1783 2012-02-21 10:20:07Z pkestene $
 */

#include "Timer.h"

#include <stdexcept>

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // Timer class methods body
  ////////////////////////////////////////////////////////////////////////////////
  
  // =======================================================
  // =======================================================
  Timer::Timer() {
    start_time.tv_sec = 0;
    start_time.tv_usec = 0;
    total_time = 0.0;
    start();
  } // Timer::Timer

  // =======================================================
  // =======================================================
  Timer::Timer(double t) 
  {
    
    //start_time.tv_sec = time_t(t);
    //start_time.tv_usec = (t - start_time.tv_sec) * 1e6;
    start_time.tv_sec = 0;
    start_time.tv_usec = 0;
    total_time = t;
    
  } // Timer::Timer

  // =======================================================
  // =======================================================
  Timer::Timer(Timer const& aTimer) : start_time(aTimer.start_time), total_time(aTimer.total_time)
  {
  } // Timer::Timer

  // =======================================================
  // =======================================================
  Timer::~Timer()
  {
  } // Timer::~Timer

  // =======================================================
  // =======================================================
  void Timer::start() 
  {

    if (-1 == gettimeofday(&start_time, 0))
      throw std::runtime_error("Timer: Couldn't initialize start_time time");
    
  } // Timer::start
  
  // =======================================================
  // =======================================================
  void Timer::stop()
  {
    double accum;
    timeval now;
    if (-1 == gettimeofday(&now, 0))
      throw std::runtime_error("Couldn't get current time");
    
    if (now.tv_sec == start_time.tv_sec)
      accum = double(now.tv_usec - start_time.tv_usec) * 1e-6;
    else
      accum = double(now.tv_sec - start_time.tv_sec) + 
	(double(now.tv_usec - start_time.tv_usec) * 1e-6);
    
    total_time += accum;

  } // Timer::stop
  
  // =======================================================
  // =======================================================
  double Timer::elapsed() const
  {

    return total_time;

  } // Timer::elapsed
} // namespace hydroSimu
