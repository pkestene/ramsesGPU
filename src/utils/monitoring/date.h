/**
 * \file date.h
 * \brief A simple macro to return a formatted string with date.
 *
 * \author Pierre Kestener
 * \date 31 Oct 2010
 *
 * $Id: date.h 1783 2012-02-21 10:20:07Z pkestene $
 */
#ifndef MONITORING_DATE_H_
#define MONITORING_DATE_H_

#include <string> // for std::string
#include <time.h> // for time_t, localtime, strftime

inline
const std::string current_date()
{
  /* get current time */
  time_t     now = time(NULL); 
  
  /* Format and print the time, "ddd yyyy-mm-dd hh:mm:ss zzz" */
  struct tm  *ts; 
  ts = localtime(&now);

  char       buf[80];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %Z", ts);
  
  return std::string(buf);

} // current_date

#endif // MONITORING_DATE_H_
