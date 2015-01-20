/**
 * \file utilities.h
 * \brief Some utility macros.
 *
 * \date 9-March-2011
 * \author P. Kestener
 *
 * $Id: utilities.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef UTILITIES_H_
#define UTILITIES_H_

inline bool isBigEndian()
{
  const int i = 1;
  return ( (*(char*)&i) == 0 );
}

#endif /* UTILITIES_H_ */
