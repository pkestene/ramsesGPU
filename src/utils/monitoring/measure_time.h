/**
 * \file measure_time.h
 * \brief Simple timing macros.
 *
 * \author Pierre Kestener
 * \date 31 Oct 2010
 *
 * $Id: measure_time.h 1783 2012-02-21 10:20:07Z pkestene $
 */
#ifndef MEASURE_TIME_H_
#define MEASURE_TIME_H_

#ifdef DO_TIMING
#define MEASURE_TIME(timer, functionToCall) \
  timer##.start();			    \
  functionToCall;			    \
  timer##.stop()
#define TIMER_START(timer) timer.start()
#define TIMER_STOP(timer) timer.stop()
#else
#define MEASURE_TIME(timer, functionToCall) \
  functionToCall;
#define TIMER_START(timer) 
#define TIMER_STOP(timer) 
#endif // DO_TIMING

#endif // MEASURE_TIME_H_
