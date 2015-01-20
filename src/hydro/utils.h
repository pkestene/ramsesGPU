/**
 * \file utils.h
 * \brief Some general use utilities.
 *
 * \author F. Chateau
 *
 * $Id: utils.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef UTILS_H_
#define UTILS_H_

enum FPUPrecision
{
	FPUSingle = 0, FPUDouble = 1, FPUExtended = 2
};

void unix_gettimeofday(int* sec, int* usec);
void set_fpu_precision(enum FPUPrecision* prec);

#endif /*UTILS_H_*/
