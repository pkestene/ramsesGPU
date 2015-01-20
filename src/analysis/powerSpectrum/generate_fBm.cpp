/**
 * \file generateFbm.cpp
 *
 * This file contains a simple programs to generate a realization the
 * 3D fractional Brownian process
 * http://en.wikipedia.org/wiki/Fractional_Brownian_motion
 *
 * Data are then dump into a pnetcdf file comptatible with RamsesGPU.
 * Only rho_vx is populated with the Brownian motion process.
 *
 * Spectral method : simply set a power-law Fourier spectrum and then apply 
 * inverse FFT (using the distributed implementation provided by fftw3-mpi).
 *
 * \sa see also :
 * http://www.maths.uq.edu.au/~kroese/ps/MCSpatial.pdf
 * http://wstein.org/home/wstein/www/home/simuw/simuw08/refs/fractal/dieker-mandjes-2003-on_spectral_simulation_of_fractional_Brownian_motion.pdf
 * http://www2.isye.gatech.edu/~adieker3/fbm/thesisold.pdf
 * http://www.keithlantz.net/2011/11/using-fourier-synthesis-to-generate-a-fractional-brownian-motion-surface/
 * http://www2.isye.gatech.edu/~adieker3/publications/specsim.pdf
 * http://bringerp.free.fr/Files/Captain%20Blood/Saupe87d.pdf
 * http://www.maths.uq.edu.au/~kroese/ps/MCSpatial.pdf
 *
 * \note Can only be used with 1 MPI proc : mpirun -np 1 ./generate_fBm --param fBm.ini
 *
 * \author P. Kestener
 * \date April 14, 2014
 *
 * $Id$
 */

#include <math.h>
#include <iostream>
#include <fstream>

#ifdef USE_FFTW3_MPI
#include <fftw3-mpi.h>
#endif /* USE_FFTW3_MPI */

#include <GetPot.h>
#include <ConfigMap.h>
#include <cnpy.h>

#include <Arrays.h>
using hydroSimu::HostArray;

#include "constants.h"

#include "pnetcdf_io.h"
#include "fft_mpi.h"

int main(int argc, char **argv){

#ifndef USE_FFTW3_MPI
  std::cout << "fftw3-mpi is not available; please enable to build this application\n";
  return 0;
#endif // USE_FFTW3_MPI

#ifndef USE_PNETCDF
  std::cout << "Parallel-NetCDF is not available; please enable to build this application\n";
  return 0;
#endif // USE_PNETCDF


#if defined(USE_FFTW3_MPI) && defined(USE_PNETCDF)
 
  /* parse command line arguments */
  GetPot cl(argc, argv);
  
  /* set default configuration parameter fileName */
  const std::string default_param_file = "fBm.ini";
  const std::string param_file = cl.follow(default_param_file.c_str(),    "--param");

  /* parse parameters from input file */
  ConfigMap configMap(param_file);
  
  /* ******************************************* */
  int myRank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // initialize MPI fftw
  fftw_mpi_init();


  /* 
   * Sanity check
   */
  /* read mpi geometry */
  // mpi geometry
  int mx,my,mz;
  mx=configMap.getInteger("mpi","mx",1);
  my=configMap.getInteger("mpi","my",1);
  mz=configMap.getInteger("mpi","mz",1);

  int nbMpiProc;
  MPI_Comm_size(MPI_COMM_WORLD, &nbMpiProc);
  if (mz!=1 || mx!=1 || my!=1) {
    std::cout << "Invalid configuration : check parameter file\n";
    return -1;
  }

  /*
   * geometry
   */
  // read local domain size
  int nx=configMap.getInteger("mesh","nx",32);
  int ny=configMap.getInteger("mesh","ny",32);
  int nz=configMap.getInteger("mesh","nz",32);

  int NX=nx*mx, NY=ny*my, NZ=nz*mz;

  HostArray<double> data_fBm;
  data_fBm.allocate(make_uint4(nx, ny, nz, 1));

  // if generating a vector field, we need 3 components
  HostArray<double> data_fBm2, data_fBm3;
  bool vectorFieldEnabled = configMap.getBool("output", "vectorFieldEnabled",false);
  if (vectorFieldEnabled) {
      data_fBm2.allocate(make_uint4(nx, ny, nz, 1));
      data_fBm3.allocate(make_uint4(nx, ny, nz, 1));
  }
    

  HostArray<double> data_fft;
  data_fft.allocate(make_uint4(nx, ny, nz, 2)); // complex = 2 double's

  // generate fBm
  {

    unsigned int r_seed = configMap.getInteger("fBm","random_seed",12);
    std::default_random_engine generator(r_seed);

    // put a power-law Fourier power spectrum in data_fft
    set_power_law_power_spectrum(configMap, data_fft.data(), generator);

    // ifft (in place)
    compute_ifft_mpi(configMap, data_fft.data(), data_fBm.data());
    
    if (vectorFieldEnabled) {
      set_power_law_power_spectrum(configMap, data_fft.data(), generator);
      compute_ifft_mpi(configMap, data_fft.data(), data_fBm2.data());

      set_power_law_power_spectrum(configMap, data_fft.data(), generator);
      compute_ifft_mpi(configMap, data_fft.data(), data_fBm3.data());
    }

    // add ghost in netcdf output
    bool ghostIncluded = configMap.getBool("output", "ghostIncluded",false);

    // do we want scaled output (normalized inverse fft ?)
    bool scaledOutput = configMap.getBool("output", "scaled",false);
    double scale = 1.0;
    if (scaledOutput)
      scale = 1/(NX*NY*NZ);
    

    // allocate memory for buffer to write
    HostArray<double> data_toDump;
    int ghostWidth=3;
    if (ghostIncluded)
      data_toDump.allocate(make_uint4(nx+2*ghostWidth, 
				      ny+2*ghostWidth, 
				      nz+2*ghostWidth, 
				      8));
    else
      data_toDump.allocate(make_uint4(nx,
				      ny,
				      nz,
				      8));
    data_toDump.reset();

    // copy fBm into rho_vx (index IU) adn density = 1.0
    if (ghostIncluded) {

      for(int k=0; k<nz; ++k)
	for(int j=0; j<ny; ++j)
	  for(int i=0; i<nx; ++i) {
	    data_toDump(i+ghostWidth,
			j+ghostWidth,
			k+ghostWidth,ID) = 1.0;
	    data_toDump(i+ghostWidth,
			j+ghostWidth,
			k+ghostWidth,IU) = data_fBm(i,j,k,0)*scale;
	  }

      if (vectorFieldEnabled) {
	for(int k=0; k<nz; ++k)
	  for(int j=0; j<ny; ++j)
	    for(int i=0; i<nx; ++i) {
	      data_toDump(i+ghostWidth,
			  j+ghostWidth,
			  k+ghostWidth,IV) = data_fBm2(i,j,k,0)*scale;
	      data_toDump(i+ghostWidth,
			j+ghostWidth,
			  k+ghostWidth,IW) = data_fBm3(i,j,k,0)*scale;
	    }
      } // end vectorFieldEnabled
      
    } else {

      for(int k=0; k<nz; ++k)
	for(int j=0; j<ny; ++j)
	  for(int i=0; i<nx; ++i) {
	    data_toDump(i, j, k, ID) = 1.0;
	    data_toDump(i, j, k, IU) = data_fBm(i,j,k,0)*scale;
	  }

      if (vectorFieldEnabled) {
	for(int k=0; k<nz; ++k)
	  for(int j=0; j<ny; ++j)
	    for(int i=0; i<nx; ++i) {
	      data_toDump(i, j, k, IV) = data_fBm2(i,j,k,0)*scale;
	      data_toDump(i, j, k, IW) = data_fBm3(i,j,k,0)*scale;
	    }
      } // end vectorFieldEnabled

    } // ghostIncluded
    
    // save data in pnetcdf format
    std::string output_file = configMap.getString("output", 
						  "outputFile", 
						  "./fBm.nc");
    write_pnetcdf(output_file,data_toDump,configMap);
    

  } // end power spectrum rho

  // cleanup MPI fftw
  fftw_mpi_cleanup();

  if (myRank==0) printf("Finalize MPI environment...\n");
  MPI_Finalize();

  return EXIT_SUCCESS;

#endif // defined(USE_FFTW3_MPI) && defined(USE_PNETCDF)

} // main
