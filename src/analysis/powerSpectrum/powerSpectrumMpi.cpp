/**
 * \file powerSpectrumMpi.cpp
 *
 * This file contains a simple programs to compute power spectrum of a large data
 * array stored in a netcdf file (use Parallel-NetCDF to load data) and then compute
 * FFT (using the distributed implementation provided by fftw3-mpi).
 *
 * \author P. Kestener
 * \date 20/06/2013
 *
 * $Id: powerSpectrumMpi.cpp 2890 2013-07-01 13:47:27Z pkestene $
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
  const std::string default_param_file = "fft.ini";
  const std::string param_file = cl.follow(default_param_file.c_str(),    "--param");

  /* parse parameters from input file */
  ConfigMap configMap(param_file);
  
  const std::string input_file    = cl.follow("test.nc", "--in");
  const std::string output_prefix = cl.follow("psdG",    "--out");
  const bool        rhoEnabled    = cl.follow(false,     "--rho");
  const bool        EkEnabled     = cl.follow(false,     "--Ek");
  const bool        EmEnabled     = cl.follow(false,     "--Em");

  /* ******************************************* */
  int myRank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // check input file 
  if (input_file.size() == 0) {
    std::cout << "Wrong input.\n";
  } else {
    if (myRank==0) std::cout << "input file used : " << input_file << std::endl;
  }


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
  if (mx*my*mz != nbMpiProc || mx!=1 || my!=1) {
    std::cout << "Invalid configuration : check parameter file\n";
    return -1;
  }

  /*
   * Read data
   */
  // read local domain size
  int nx=configMap.getInteger("mesh","nx",32);
  int ny=configMap.getInteger("mesh","ny",32);
  int nz=configMap.getInteger("mesh","nz",32);

  int NX=nx*mx, NY=ny*my, NZ=nz*mz;
  double maxSizeGlobal = NX/2.0;
  int    nBins = configMap.getInteger("powerSpectrum","nBins",128);
  double dk=maxSizeGlobal/(nBins-1);

  HostArray<double> data_read;
  data_read.allocate(make_uint4(nx, ny, nz, 1));

  HostArray<double> fft_out;
  fft_out.allocate(make_uint4(nx, ny, nz, 2)); // complex = 2 double's
  
  HostArray<double> histo, psd;
  histo.allocate(make_uint4(nBins,1,1,1)); histo.reset();
  psd.allocate(make_uint4(nBins,1,1,1));   psd.reset();

  HostArray<double> histoG, psdG;
  histoG.allocate(make_uint4(nBins,1,1,1)); histoG.reset();
  psdG.allocate(make_uint4(nBins,1,1,1));   psdG.reset();
 
  HostArray<double> freq; 
  freq.allocate(make_uint4(nBins,1,1,1));
  for (int nBin=0; nBin<nBins; nBin++) {
    freq(nBin)=nBin*dk;
  }

  // ////////////////////
  // power spectrum  rho
  // ////////////////////
  if (rhoEnabled) {
    // read data
    read_pnetcdf(input_file,ID,configMap,data_read);
    HostArray<double> &rho = data_read;
    
    // fft
    compute_fft_mpi(configMap, rho.data(), fft_out.data());
    
    // power spectrum
    histo.reset();
    psd.reset();
    compute_power_spectrum_mpi(configMap, fft_out.data(), histo.data(), psd.data());
    
    // gather all result by performing reduce
    histoG.reset();
    psdG.reset();   
    MPI_Reduce(histo.data(), histoG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(psd.data(), psdG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // normalize spectrum psdG
    for (int nBin=0; nBin<nBins; nBin++) {
      if (histoG(nBin) > 0) {
    	psdG(nBin) /= histoG(nBin);
	psdG(nBin) *= 4./3.*M_PI*( (nBin+1)*(nBin+1)*(nBin+1)-nBin*nBin*nBin )*dk;
      }
    }
    
    // save data to file
    if (myRank==0) {
      // save array freq to file
      const unsigned int shape[] = {(unsigned int) nBins};
      if (configMap.getBool("powerSpectrum","dumpFreq",false)) {
    	std::string output_file=output_prefix+"_rho_powerspec_freq.npy";
    	cnpy::npy_save(output_file.c_str(),freq.data(),shape,1,"w");
      }
      
      // save array psdG to file
      std::string output_file=output_prefix+"_rho_powerspec_bins.npy";
      cnpy::npy_save(output_file.c_str(),psdG.data(),shape,1,"w");
      
      std::cout << "rho power spectrum computed...\n";
      
    }
  
  } // end power spectrum rho

  // ////////////////////////////////////
  // power spectrum  Ek / Ek2 / velocity
  // ////////////////////////////////////

  // Ek  spectrum is rho^(1/2)v
  // Ek2 spectrum is rho^(1/3)v

  if (EkEnabled) {
    HostArray<double> Ek, Ek2, velocity;
    Ek.allocate(make_uint4(nx, ny, nz, 1));
    Ek.reset();
    Ek2.allocate(make_uint4(nx, ny, nz, 1));
    Ek2.reset();
    velocity.allocate(make_uint4(nx, ny, nz, 1));
    velocity.reset();

    // read data rho_vx
    read_pnetcdf(input_file,IU,configMap,data_read);
    data_read *= data_read;
    Ek += data_read;
    
    // read data rho_vy
    read_pnetcdf(input_file,IV,configMap,data_read);
    data_read *= data_read;
    Ek += data_read;

    // read data rho_vz
    read_pnetcdf(input_file,IW,configMap,data_read);
    data_read *= data_read;
    Ek += data_read;

    // read data rho
    read_pnetcdf(input_file,ID,configMap,data_read);
    //Ek /= data_read;
    for (int i=0; i<nx*ny*nz; i++) {
      velocity(i)=Ek(i);                            // rho^2     * v^2
      Ek(i) /= data_read(i);                        // rho       * v^2
      Ek(i) = sqrt(Ek(i));                          // rho^(1/2) * v
      velocity(i) /= (data_read(i)*data_read(i));   //             v^2
      velocity(i) = sqrt(velocity(i));              //             v
      Ek2(i) = velocity(i)*pow(data_read(i),1.0/3); // rho^(1/3) * v
    }

    // Ek power spectrum
    {
      // fft
      compute_fft_mpi(configMap, Ek.data(), fft_out.data());
      
      // power spectrum
      histo.reset();
      psd.reset();
      
      compute_power_spectrum_mpi(configMap, fft_out.data(), histo.data(), psd.data());
      
      // gather all result by performing reduce
      histoG.reset();
      psdG.reset();
      
      MPI_Reduce(histo.data(), histoG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(psd.data(), psdG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      
      // normalize spectrum psdG
      for (int nBin=0; nBin<nBins; nBin++) {
	if (histoG(nBin) > 0) {
	  psdG(nBin) /= histoG(nBin);
	  psdG(nBin) *= 4./3.*M_PI*( (nBin+1)*(nBin+1)*(nBin+1)-nBin*nBin*nBin )*dk;
	}
      }
      
      // save data to file
      if (myRank==0) {
	// save array freq to file
	const unsigned int shape[] = {(unsigned int) nBins};
	if (configMap.getBool("powerSpectrum","dumpFreq",false)) {
	  std::string output_file=output_prefix+"_Ek_powerspec_freq.npy";
	  cnpy::npy_save(output_file.c_str(),freq.data(),shape,1,"w");
	}
	
	// save array psdG to file
	std::string output_file=output_prefix+"_Ek_powerspec_bins.npy";
	cnpy::npy_save(output_file.c_str(),psdG.data(),shape,1,"w");
	
	std::cout << "Ek power spectrum computed...\n";
	
      }
    } // end Ek power spectrum

    // Ek2 power spectrum
    {
      // fft
      compute_fft_mpi(configMap, Ek2.data(), fft_out.data());
      
      // power spectrum
      histo.reset();
      psd.reset();
      
      compute_power_spectrum_mpi(configMap, fft_out.data(), histo.data(), psd.data());
      
      // gather all result by performing reduce
      histoG.reset();
      psdG.reset();
      
      MPI_Reduce(histo.data(), histoG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(psd.data(), psdG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      
      // normalize spectrum psdG
      for (int nBin=0; nBin<nBins; nBin++) {
	if (histoG(nBin) > 0) {
	  psdG(nBin) /= histoG(nBin);
	  psdG(nBin) *= 4./3.*M_PI*( (nBin+1)*(nBin+1)*(nBin+1)-nBin*nBin*nBin )*dk;
	}
      }
      
      // save data to file
      if (myRank==0) {
	// save array freq to file
	const unsigned int shape[] = {(unsigned int) nBins};
	if (configMap.getBool("powerSpectrum","dumpFreq",false)) {
	  std::string output_file=output_prefix+"_Ek2_powerspec_freq.npy";
	  cnpy::npy_save(output_file.c_str(),freq.data(),shape,1,"w");
	}
	
	// save array psdG to file
	std::string output_file=output_prefix+"_Ek2_powerspec_bins.npy";
	cnpy::npy_save(output_file.c_str(),psdG.data(),shape,1,"w");
	
	std::cout << "Ek2 power spectrum computed...\n";
	
      }
    } // end Ek2 power spectrum

    // velocity power spectrum
    {
      // fft
      compute_fft_mpi(configMap, velocity.data(), fft_out.data());
      
      // power spectrum
      histo.reset();
      psd.reset();
      
      compute_power_spectrum_mpi(configMap, fft_out.data(), histo.data(), psd.data());
      
      // gather all result by performing reduce
      histoG.reset();
      psdG.reset();
      
      MPI_Reduce(histo.data(), histoG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(psd.data(), psdG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      
      // normalize spectrum psdG
      for (int nBin=0; nBin<nBins; nBin++) {
	if (histoG(nBin) > 0) {
	  psdG(nBin) /= histoG(nBin);
	  psdG(nBin) *= 4./3.*M_PI*( (nBin+1)*(nBin+1)*(nBin+1)-nBin*nBin*nBin )*dk;
	}
      }
      
      // save data to file
      if (myRank==0) {
	// save array freq to file
	const unsigned int shape[] = {(unsigned int) nBins};
	if (configMap.getBool("powerSpectrum","dumpFreq",false)) {
	  std::string output_file=output_prefix+"_v_powerspec_freq.npy";
	  cnpy::npy_save(output_file.c_str(),freq.data(),shape,1,"w");
	}
	
	// save array psdG to file
	std::string output_file=output_prefix+"_v_powerspec_bins.npy";
	cnpy::npy_save(output_file.c_str(),psdG.data(),shape,1,"w");
	
	std::cout << "velocity power spectrum computed...\n";
	
      }
    } // end velocity power spectrum

  } // power spectrum Ek / velocity

  // ///////////////////
  // power spectrum  Em
  // ///////////////////
  if (EmEnabled) {
    HostArray<double> Em;
    Em.allocate(make_uint4(nx, ny, nz, 1));
    Em.reset();

    // read data and compute magnetic energy
    read_pnetcdf(input_file,IA,configMap,data_read);
    data_read *= data_read;
    Em += data_read;

    read_pnetcdf(input_file,IB,configMap,data_read);
    data_read *= data_read;
    Em += data_read;

    read_pnetcdf(input_file,IC,configMap,data_read);
    data_read *= data_read;
    Em += data_read;

    for (int i=0; i<nx*ny*nz; i++) {
      Em(i) = sqrt(Em(i));
    }

    // fft
    compute_fft_mpi(configMap, Em.data(), fft_out.data());
    
    // power spectrum
    histo.reset();
    psd.reset();

    compute_power_spectrum_mpi(configMap, fft_out.data(), histo.data(), psd.data());
    
    // gather all result by performing reduce
    histoG.reset();
    psdG.reset();
    
    MPI_Reduce(histo.data(), histoG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(psd.data(), psdG.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // normalize spectrum psdG
    for (int nBin=0; nBin<nBins; nBin++) {
      if (histoG(nBin) > 0) {
	psdG(nBin) /= histoG(nBin);
	psdG(nBin) *= 4./3.*M_PI*( (nBin+1)*(nBin+1)*(nBin+1)-nBin*nBin*nBin )*dk;
      }
    }
    
    // save data to file
    if (myRank==0) {
      // save array freq to file
      const unsigned int shape[] = {(unsigned int) nBins};
      if (configMap.getBool("powerSpectrum","dumpFreq",false)) {
	std::string output_file=output_prefix+"_Em_powerspec_freq.npy";
	cnpy::npy_save(output_file.c_str(),freq.data(),shape,1,"w");
      }
      
      // save array psdG to file
      std::string output_file=output_prefix+"_Em_powerspec_bins.npy";
      cnpy::npy_save(output_file.c_str(),psdG.data(),shape,1,"w");
      
      std::cout << "Em power spectrum computed...\n";
      
    }
  } // power spectrum Em

  fftw_mpi_cleanup();

  MPI_Finalize();

  if (myRank==0) printf("MPI finalized...\n");

  return 0;

#endif // defined(USE_FFTW3_MPI) && defined(USE_PNETCDF)

}
