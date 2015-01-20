/**
 * \file fft_mpi.h
 *
 * Parallel FFT computation.
 *
 * \author P. Kestener
 * \date 20/06/2013
 *
 * $Id: fft_mpi.h 3394 2014-05-06 10:19:58Z pkestene $
 */


#ifndef FFT_MPI_H_
#define FFT_MPI_H_

#ifdef USE_FFTW3_MPI
#include <fftw3-mpi.h>
#endif /* USE_FFTW3_MPI */

#include <ConfigMap.h>

#ifdef GEN_FBM
#include <random> // c++11
#endif // GEN_FBM

/**
 * Compute forward transform FFT (real to complex).
 *
 * dataIn must have been allocated with size nx*ny*nz (local domain sizes)
 * dataOut must also be allocated in the calling routine.
 *
 */
void compute_fft_mpi(ConfigMap    &configMap, 
		     double       *dataIn, 
		     double       *dataOut);

/**
 * Compute backward transform IFFT (complex to real).
 *
 * dataIn must have been allocated with size nx*ny*nz (local domain sizes)
 * dataOut must also be allocated in the calling routine.
 *
 */
void compute_ifft_mpi(ConfigMap    &configMap, 
		      double       *dataIn, 
		      double       *dataOut);

/**
 *
 * dataIn (fft coef) must have been allocated with size nx*ny*nz (local domain sizes)
 * dataOut (local spectrum) must also be allocated in the calling routine.
 *
 * @param[in]  dataFFT
 * @param[out] dataHisto
 * @param[out] dataPsd
 */
void compute_power_spectrum_mpi(ConfigMap    &configMap, 
				double       *dataFFT, 
				double       *dataHisto,
				double       *dataPsd);

#ifdef GEN_FBM
/**
 * generate Fourier coefficients for a 3D Fractional Browniam motion realization.
 */
void set_power_law_power_spectrum(ConfigMap &configMap,
				  double    *data_fft,
				  std::default_random_engine& generator);

#endif /* GEN_FBM */

#endif /* FFT_MPI_H_ */
