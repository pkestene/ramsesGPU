/**
 * \file utils.c
 * \brief Some general use utilities.
 *
 * \author F. Chateau
 */
#include "utils.h"
#include <sys/time.h>
#include <fpu_control.h>

/* A Fortran-callable gettimeofday routine to give access
 * to the wall clock timer.
 */
void unix_gettimeofday(int* sec, int* usec)
{
	struct timeval tp;
	gettimeofday(&tp, 0);
	*sec = tp.tv_sec;
	*usec = tp.tv_usec;
}

void set_fpu_precision(enum FPUPrecision* prec)
{
	unsigned precisionFlag = (*prec == FPUSingle ? _FPU_SINGLE : (*prec == FPUDouble ? _FPU_DOUBLE : _FPU_EXTENDED));
	unsigned originalCW;
	_FPU_GETCW(originalCW);
	unsigned cw = (originalCW & ~_FPU_EXTENDED) | precisionFlag;
	_FPU_SETCW(cw);
}
