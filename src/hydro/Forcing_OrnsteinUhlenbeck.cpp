/*
 * Copyright CEA / Maison de la Simulation
 * Contributors: Pierre Kestener, Sebastien Fromang (May 22, 2012)
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use, 
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info". 
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 */

/**
 * \file Forcing_OrnsteinUhlenbeck.cpp
 * \brief Implementation of ForcingOrnsteinUhlenbeck class
 *
 * \author P. Kestener
 * \date 19/12/2013
 *
 * $Id: Forcing_OrnsteinUhlenbeck.cpp 3465 2014-06-29 21:28:48Z pkestene $
 */
#include "Forcing_OrnsteinUhlenbeck.h"

// for I/O with numpy readable files
#include "utils/cnpy/cnpy.h"
#include <cmath> // for copysign

#ifdef __CUDACC__
#include "cutil_inline_runtime.h"

#include "Forcing_OrnsteinUhlenbeck_kernels.cuh"
#endif // __CUDACC__

namespace hydroSimu {

  // static member
  const int ForcingOrnsteinUhlenbeck::nMode;

  //////////////////////////////////////////////////////////////////////////
  // ForcingOrnsteinUhlenbeck class methods body
  //////////////////////////////////////////////////////////////////////////
  ForcingOrnsteinUhlenbeck::ForcingOrnsteinUhlenbeck(int _nDim,
						     int _nCpu,
						     ConfigMap &_configMap,
						     GlobalConstants &_gParams) : 
    timeScaleTurb(0.1),
    amplitudeTurb(0.0001),
    ksi(0.0),
    init_random(600),
    nDim(_nDim),
    nCpu(_nCpu),
    configMap(_configMap)
  {

    // read timeScaleTurb, amplitudeTurb, ksi, init_random from input file
    timeScaleTurb = configMap.getFloat("turbulence-Ornstein-Uhlenbeck","timeScaleTurb", 0.1);
    amplitudeTurb = configMap.getFloat("turbulence-Ornstein-Uhlenbeck","amplitudeTurb", 0.0001);
    ksi           = configMap.getFloat("turbulence-Ornstein-Uhlenbeck","ksi", 0.0);
    init_random   = configMap.getInteger("turbulence-Ornstein-Uhlenbeck","init_random", 600);

    /*
     * VERY important:
     * make sure variables declared as __constant__ are copied to device
     * for current compilation unit
     */
#ifdef __CUDACC__
    cutilSafeCall( cudaMemcpyToSymbol(::gParams, &_gParams, sizeof(GlobalConstants), 0, cudaMemcpyHostToDevice ) );
#else
    (void) _gParams;
#endif // __CUDACC__
    
  } // ForcingOrnsteinUhlenbeck::ForcingOrnsteinUhlenbeck

  // =======================================================
  // =======================================================
  ForcingOrnsteinUhlenbeck::~ForcingOrnsteinUhlenbeck()
  {
    
    free();

  } // ForcingOrnsteinUhlenbeck::~ForcingOrnsteinUhlenbeck

  // =======================================================
  // =======================================================
  void ForcingOrnsteinUhlenbeck::allocate()
  {
    
    mode         = new double[nDim*nMode];
    forcingField = new double[nDim*nMode];
    projTens     = new double[nDim*nDim*nMode];
    gaussSeed    = new int   [nCpu*4];

    for (int i=0; i<nDim*nMode; i++) {
      mode        [i]=0.0;
      forcingField[i]=0.0;
    }

    for (int i=0; i<nDim*nDim*nMode; i++)
      projTens[i]=0.0;

    for (int i=0; i<nCpu*4; i++)
      gaussSeed[i]=0;

    pRandomGen = new RandomGen();

#ifdef __CUDACC__

    cutilSafeCall( cudaMalloc((void**) &d_mode        , nDim*nMode*sizeof(double)) );
    cutilSafeCall( cudaMalloc((void**) &d_forcingField, nDim*nMode*sizeof(double)) );
    cutilSafeCall( cudaMalloc((void**) &d_projTens    , nDim*nDim*nMode*sizeof(double)) );

    cutilSafeCall( cudaMalloc((void**) &deviceStates  , nMode * sizeof(curandState) ) );

#endif

  } // ForcingOrnsteinUhlenbeck::allocate

  // =======================================================
  // =======================================================
  void ForcingOrnsteinUhlenbeck::free()
  {

    if (mode)
      delete[] mode;

    if (forcingField)
      delete[] forcingField;

    if (projTens)
      delete[] projTens;

    if (gaussSeed) 
      delete[] gaussSeed;

    delete pRandomGen;

#ifdef __CUDACC__

    cutilSafeCall( cudaFree(d_mode        ) );
    cutilSafeCall( cudaFree(d_forcingField) );
    cutilSafeCall( cudaFree(d_projTens    ) );

    cutilSafeCall( cudaFree(deviceStates  ) );
    
#endif // __CUDACC__

  } // ForcingOrnsteinUhlenbeck::free 

  // =======================================================
  // =======================================================
  void ForcingOrnsteinUhlenbeck::init_forcing(bool restartEnabled, int nStep)
  {

    allocate();

    // identity matrix
    double ID[nDim][nDim];
    for (int j=0; j<nDim; j++)
      for (int i=0; i<nDim; i++)
	ID[j][i] = 0.0;
    for (int i=0; i<nDim; i++)
	ID[i][i] = 0.0;
    
    // initialize Gaussian random number generator seeds 
    pRandomGen->rans(nCpu,init_random,gaussSeed);

    forceSeed[0] = gaussSeed[0];
    forceSeed[1] = gaussSeed[1];
    forceSeed[2] = gaussSeed[2];
    forceSeed[3] = gaussSeed[3];
      
    // initialize the forcing field Fourier modes
    mode[ 0]=0.0 ; mode[1*nMode+ 0]=0.0 ; mode[2*nMode+ 0]=2.0;
    mode[ 1]=0.0 ; mode[1*nMode+ 1]=0.0 ; mode[2*nMode+ 1]=3.0;
    mode[ 2]=0.0 ; mode[1*nMode+ 2]=1.0 ; mode[2*nMode+ 2]=2.0;
    mode[ 3]=0.0 ; mode[1*nMode+ 3]=1.0 ; mode[2*nMode+ 3]=3.0;
    mode[ 4]=0.0 ; mode[1*nMode+ 4]=2.0 ; mode[2*nMode+ 4]=0.0;
    mode[ 5]=0.0 ; mode[1*nMode+ 5]=2.0 ; mode[2*nMode+ 5]=1.0;
    mode[ 6]=0.0 ; mode[1*nMode+ 6]=2.0 ; mode[2*nMode+ 6]=2.0;
    mode[ 7]=0.0 ; mode[1*nMode+ 7]=3.0 ; mode[2*nMode+ 7]=0.0;
    mode[ 8]=0.0 ; mode[1*nMode+ 8]=3.0 ; mode[2*nMode+ 8]=1.0;
    mode[ 9]=1.0 ; mode[1*nMode+ 9]=0.0 ; mode[2*nMode+ 9]=2.0;
    mode[10]=1.0 ; mode[1*nMode+10]=0.0 ; mode[2*nMode+10]=3.0;
    mode[11]=1.0 ; mode[1*nMode+11]=1.0 ; mode[2*nMode+11]=2.0;
    mode[12]=1.0 ; mode[1*nMode+12]=1.0 ; mode[2*nMode+12]=3.0;
    mode[13]=1.0 ; mode[1*nMode+13]=2.0 ; mode[2*nMode+13]=0.0;
    mode[14]=1.0 ; mode[1*nMode+14]=2.0 ; mode[2*nMode+14]=1.0;
    mode[15]=1.0 ; mode[1*nMode+15]=2.0 ; mode[2*nMode+15]=2.0;
    mode[16]=1.0 ; mode[1*nMode+16]=3.0 ; mode[2*nMode+16]=0.0;
    mode[17]=1.0 ; mode[1*nMode+17]=3.0 ; mode[2*nMode+17]=1.0;
    mode[18]=2.0 ; mode[1*nMode+18]=0.0 ; mode[2*nMode+18]=0.0;
    mode[19]=2.0 ; mode[1*nMode+19]=0.0 ; mode[2*nMode+19]=1.0;
    mode[20]=2.0 ; mode[1*nMode+20]=0.0 ; mode[2*nMode+20]=2.0;
    mode[21]=2.0 ; mode[1*nMode+21]=1.0 ; mode[2*nMode+21]=0.0;
    mode[22]=2.0 ; mode[1*nMode+22]=1.0 ; mode[2*nMode+22]=1.0;
    mode[23]=2.0 ; mode[1*nMode+23]=1.0 ; mode[2*nMode+23]=2.0;
    mode[24]=2.0 ; mode[1*nMode+24]=2.0 ; mode[2*nMode+24]=0.0;
    mode[25]=2.0 ; mode[1*nMode+25]=2.0 ; mode[2*nMode+25]=1.0;
    mode[26]=2.0 ; mode[1*nMode+26]=2.0 ; mode[2*nMode+26]=2.0;
    mode[27]=3.0 ; mode[1*nMode+27]=0.0 ; mode[2*nMode+27]=0.0;
    mode[28]=3.0 ; mode[1*nMode+28]=0.0 ; mode[2*nMode+28]=1.0;
    mode[29]=3.0 ; mode[1*nMode+29]=1.0 ; mode[2*nMode+29]=0.0;
    mode[30]=3.0 ; mode[1*nMode+30]=1.0 ; mode[2*nMode+30]=1.0;

    // loop over modes
    for (int iMode=0; iMode<nMode; iMode++) {
      double sum=0.0;
      double randomNumber;
      
      // compute mode "energy"
      for (int iDim=0; iDim<nDim; iDim++) {
        pRandomGen->gaussDev( forceSeed, randomNumber );
        mode[iDim*nMode + iMode] = copysign( mode[iDim*nMode + iMode],randomNumber);
	sum = sum + mode[iDim*nMode + iMode]*mode[iDim*nMode + iMode];
      }

      // compute projection tenseur
      // see for example http://arxiv.org/pdf/astro-ph/0407616.pdf
      // equation (8)
      // ksi = 1 : forcing field is purely solenoidal (i.e. div-free)
      // ksi = 0 : forcing field is purely compressive (i.e. curl-free)
      for (int j=0; j<nDim; j++) {
	for (int i=0; i<nDim; i++) {
	  projTens[i*nDim*nMode + j*nMode + iMode] = 
	    ksi * ID[i][j] +
	    (1.0 - 2.0 * ksi) * 
	    mode[j*nMode+iMode] * 
	    mode[i*nMode+iMode]/sum;
	}
      }
    }

    /* ------------------------------------------------------ */
    /* For a restart run, read forcing file                   */
    /* and fill mode, projTens, forcingField and seedGauss array */
    /* ------------------------------------------------------ */
    if (restartEnabled) { // perform a restart

      /*
       * build full path filename
       */
      std::string outputDir    = configMap.getString("output", "outputDir", "./");
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
      
      std::ostringstream outNum;
      outNum.width(7);
      outNum.fill('0');
      outNum << nStep;
      std::string baseName     = outputPrefix+"_forcing_"+outNum.str();
      std::string filename     = baseName+".npz";
      std::string filenameFull = outputDir+"/"+filename;
      
      // read nMode from file into nMode2
      {
	cnpy::NpyArray nMode_npy = cnpy::npz_load(filenameFull,"nMode");
	int *nMode2 = reinterpret_cast<int*>(nMode_npy.data);
	
	if (nMode2[0] != nMode) {
	  std::cerr << "File forc.tmp is not compatible\n";
	  std::cerr << "Found    nMode =" << nMode2[0] << "\n";
	  std::cerr << "Expected nMode =" << nMode << "\n";
	}
	nMode_npy.destruct();
      }

      // read nCpu  actually nCpu should always be 1 (we only need a single random seed)
      int nCpuRead;
      {
	cnpy::NpyArray nCpu_npy = cnpy::npz_load(filenameFull,"nCpu");
	int *nCpu2 = reinterpret_cast<int*>(nCpu_npy.data);
	
	if (nCpu2[0] != nCpu) {
	  std::cerr << "nCpu (read) =" << nCpu2[0] << "is different from current nCpu "
		    << nCpu << "\n";
	  std::cerr << "Using default gaussSeed instead !\n ";
	}
	nCpuRead = nCpu2[0];
	nCpu_npy.destruct();
      }
      
      // read mode
      {
	cnpy::NpyArray mode_npy = cnpy::npz_load(filenameFull,"mode");
	double *pMode = reinterpret_cast<double*>(mode_npy.data);
	
	for(int i = 0; i<nDim*nMode; i++)
	  mode[i] = pMode[i];

	mode_npy.destruct();
      }

      // read forcingField
      {
	cnpy::NpyArray forcingField_npy = cnpy::npz_load(filenameFull,"forcingField");
	double *pForcingField = reinterpret_cast<double*>(forcingField_npy.data);
	
	for(int i = 0; i<nDim*nMode; i++)
	  forcingField[i] = pForcingField[i];

	forcingField_npy.destruct();
      }

      // read projTens
      {
	cnpy::NpyArray projTens_npy = cnpy::npz_load(filenameFull,"projTens");
	double *pProjTens = reinterpret_cast<double*>(projTens_npy.data);
	
	for(int i = 0; i<nDim*nDim*nMode; i++)
	  projTens[i] = pProjTens[i];

	projTens_npy.destruct();
      }

      // read gaussSeed
      {
	cnpy::NpyArray gaussSeed_npy = cnpy::npz_load(filenameFull,"gaussSeed");
	int *pGaussSeed = reinterpret_cast<int*>(gaussSeed_npy.data);
	
	if (nCpuRead == nCpu) {
	  for(int i = 0; i<nCpu*4; i++)
	    gaussSeed[i] = pGaussSeed[i];
	} else { // re-generate initial random state
	  pRandomGen->rans(nCpu,init_random,gaussSeed);
	}

	gaussSeed_npy.destruct();
      }

      // copy gaussSeed[0] into forceSeed
      forceSeed[0] = gaussSeed[0];
      forceSeed[1] = gaussSeed[1];
      forceSeed[2] = gaussSeed[2];
      forceSeed[3] = gaussSeed[3];

    } // end restart

#ifdef __CUDACC__

    // copy parameters into GPU memory
    cutilSafeCall( cudaMemcpy( d_mode,         mode,         
			       nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
    cutilSafeCall( cudaMemcpy( d_forcingField, forcingField, 
			       nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
    cutilSafeCall( cudaMemcpy( d_projTens,     projTens,
			       nDim*nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );

    // initialize random generator on GPU
    init_random_generator_kernel<<<(nMode+31)/32,32>>>(deviceStates);

#endif // __CUDACC__


  } // ForcingOrnsteinUhlenbeck::init_forcing

  // =======================================================
  // =======================================================
  void ForcingOrnsteinUhlenbeck::output_forcing(int nStep)
  {

#ifdef __CUDACC__

    // copy parameters from GPU memory
    cutilSafeCall( cudaMemcpy( mode,         d_mode,         
			       nDim*nMode*sizeof(double), cudaMemcpyDeviceToHost ) );
    cutilSafeCall( cudaMemcpy( forcingField, d_forcingField, 
			       nDim*nMode*sizeof(double), cudaMemcpyDeviceToHost ) );
    cutilSafeCall( cudaMemcpy( projTens,     d_projTens,
			       nDim*nDim*nMode*sizeof(double), cudaMemcpyDeviceToHost ) );

#endif // __CUDACC__

    /*
     * build full path filename
     */
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << nStep;
    std::string baseName     = outputPrefix+"_forcing_"+outNum.str();
    std::string filename     = baseName+".npz";
    std::string filenameFull = outputDir+"/"+filename;
    
    // write nMode
    unsigned int shape_scalar[] = {1};
    cnpy::npz_save(filenameFull,"nMode",&nMode,shape_scalar,1,"w");

    // write nCpu
    cnpy::npz_save(filenameFull,"nCpu",&nCpu,shape_scalar,1,"a");

    // write mode (watch outshape order !)
    unsigned int shape_mode[] = {(unsigned int) nMode, (unsigned int) nDim};
    cnpy::npz_save(filenameFull,"mode",mode,shape_mode,2,"a");

    // write forcingField (watch outshape order !)
    unsigned int shape_forcingField[] = {(unsigned int) nMode, (unsigned int) nDim};
    cnpy::npz_save(filenameFull,"forcingField",forcingField,shape_forcingField,2,"a");

    // write projTens (watch outshape order !)
    unsigned int shape_projTens[] = {(unsigned int)  nMode, (unsigned int) nDim, (unsigned int) nDim};
    cnpy::npz_save(filenameFull,"projTens",projTens,shape_projTens,3,"a");

    // write gaussSeed (watch outshape order !)
    unsigned int shape_gaussSeed[] = {4, (unsigned int) nCpu};
    cnpy::npz_save(filenameFull,"gaussSeed",gaussSeed,shape_gaussSeed,2,"a");

  } // ForcingOrnsteinUhlenbeck::output_forcing

  // =======================================================
  // =======================================================
  void ForcingOrnsteinUhlenbeck::input_forcing(std::string forcing_filename,
					       int nStep)
  {

    std::string filenameFull;

    if ( !forcing_filename.size() ) {
      
      /*
       * build full path filename using the exact same parameter from a previous run
       */
      std::string outputDir    = configMap.getString("output", "outputDir", "./");
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
      
      std::ostringstream outNum;
      outNum.width(7);
      outNum.fill('0');
      outNum << nStep;
      std::string baseName     = outputPrefix+"_forcing_"+outNum.str();
      std::string filename     = baseName+".npz";
      filenameFull = outputDir+"/"+filename;

    } else {

      filenameFull = forcing_filename;

    }

    // load the entire file
    cnpy::npz_t my_npz = cnpy::npz_load(filenameFull);
    
    // read nMode
    cnpy::NpyArray data_nMode = my_npz["nMode"];
    int *nModePtr = reinterpret_cast<int*>(data_nMode.data);
    if ( nModePtr[0] != nMode ) {
      std::cerr << "[Forcing_Ornstein-Uhlenbeck] Error when reading nMode from file "
		<< filenameFull << "\n";
    }
    
    // read nCpu
    cnpy::NpyArray data_nCpu = my_npz["nCpu"];
    int *nCpuPtr = reinterpret_cast<int*>(data_nCpu.data);
    if ( nCpuPtr[0] != nCpu ) {
      std::cerr << "[Forcing_Ornstein-Uhlenbeck] Warning : nCpu read from file is different from the current one; using default gaussSeed instead !\n ";
    }
    
    // read mode (watch outshape order !)
    cnpy::NpyArray data_mode = my_npz["mode"];
    double *modePtr = reinterpret_cast<double*>(data_mode.data);
    for (int i=0; i<nDim*nMode; i++)
      mode[i] = modePtr[i];
    
    // read forcingField (watch outshape order !)
    cnpy::NpyArray data_forcingField = my_npz["forcingField"];
    double *forcingFieldPtr = reinterpret_cast<double*>(data_forcingField.data);
    for (int i=0; i<nDim*nMode; i++)
      forcingField[i] = forcingFieldPtr[i];

    // read projTens (watch outshape order !)
    cnpy::NpyArray data_projTens = my_npz["projTens"];
    double *projTensPtr = reinterpret_cast<double*>(data_projTens.data);
    for (int i=0; i<nDim*nDim*nMode; i++)
      projTens[i] = projTensPtr[i];
       
    // read gaussSeed (watch outshape order !)
    cnpy::NpyArray data_gaussSeed = my_npz["gaussSeed"];
    int *gaussSeedPtr = reinterpret_cast<int*>(data_gaussSeed.data);
    if (nCpuPtr[0] == nCpu) {
      for (int i=0; i<nCpu*4; i++)
	gaussSeed[i] = gaussSeedPtr[i];
    } else { // use default values
      pRandomGen->rans(nCpu,init_random,gaussSeed);
    }

    // destroy my_npz
    my_npz.destruct();

#ifdef __CUDACC__

    // copy parameters into GPU memory
    cutilSafeCall( cudaMemcpy( d_mode,         mode,         
			       nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
    cutilSafeCall( cudaMemcpy( d_forcingField, forcingField, 
			       nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
    cutilSafeCall( cudaMemcpy( d_projTens,     projTens,
			       nDim*nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );

#endif // __CUDACC__

  } // ForcingOrnsteinUhlenbeck::input_forcing

  // =======================================================
  // =======================================================
  void ForcingOrnsteinUhlenbeck::update_forcing_field_mode(real_t dt)
  {

    /*
     * the aim here is to compute the new forcing field modes at t+dt
     * f(t+dt) = f(t) + df
     */

#ifdef __CUDACC__

    // GPU version (1D parallelization of iMode loop)

    update_forcing_field_mode_kernel<<<(nMode+31)/32,32>>>(d_forcingField,
							   d_projTens,
							   deviceStates,
							   timeScaleTurb,
							   amplitudeTurb,
							   ksi,
							   nDim,
							   nMode,
							   dt);
    
#else

    // CPU version

    double weight = amplitudeTurb;
    double v      = sqrt(5.0/3.0)*gParams.cIso;

    for (int iMode=0; iMode<nMode; iMode++) {

      double AAA[3] = {0.0, 0.0, 0.0};
      double BBB[3] = {0.0, 0.0, 0.0};
      double randomNumber;

      for (int i=0; i<nDim; i++) {
	pRandomGen->gaussDev(forceSeed, randomNumber);
	AAA[i] = randomNumber*sqrt(dt);
      }

      for (int j=0; j<nDim ; j++) {
	double summ=0.0;
	for (int i=0; i<nDim; i++) {
	  summ += projTens[i*nDim*nMode + j*nMode + iMode]*AAA[i];
	}
	BBB[j]=summ;
      }

      for (int i=0; i<nDim; i++)
	BBB[i] = BBB[i]*v*sqrt(2.0*weight*weight/timeScaleTurb)/timeScaleTurb;
      
      // now compute df
      for (int i=0; i<nDim; i++)
	BBB[i] = BBB[i] - forcingField[i*nMode+iMode]*dt/timeScaleTurb;
      
      // now update forcing field : f(t+dt) = f(t)+df
      double forceRMS = 3.0 / sqrt(1 - 2.0*ksi + 3.0*ksi*ksi);
      for (int i=0; i<nDim; i++) {
	forcingField[i*nMode+iMode] += forceRMS * BBB[i];
      }

    } // end for iMode

#endif // __CUDACC__    

  } // ForcingOrnsteinUhlenbeck::update_forcing_field_mode

#ifdef __CUDACC__

  // =======================================================
  // =======================================================
  void ForcingOrnsteinUhlenbeck::add_forcing_field(DeviceArray<real_t> &U,
						   real_t dt)
  {

    bool mhdEnabled = configMap.getBool("MHD","enable", false);
    int ghostWidth = 2;
    if (mhdEnabled) ghostWidth = 3;

    int isize = U.dimx();
    int jsize = U.dimy();

    // 1. update Fourier modes of forcing field
    update_forcing_field_mode(dt);

    // 2. update phase with space dependencies
    // 3. update hydro variables
    dim3 dimBlock(16,16);
    dim3 dimGrid(blocksFor(isize, dimBlock.x),
		 blocksFor(jsize, dimBlock.y));

    add_forcing_field_kernel<<<dimGrid,dimBlock>>>(U.data(), d_forcingField, d_mode, 
						   nDim, nMode, ghostWidth, dt);

  } // ForcingOrnsteinUhlenbeck::add_forcing_field // GPU version

#else

  // =======================================================
  // =======================================================
  void ForcingOrnsteinUhlenbeck::add_forcing_field(HostArray<real_t> &U,
						   real_t dt)
  {

    bool mhdEnabled = configMap.getBool("MHD","enable", false);
    int ghostWidth = 2;
    if (mhdEnabled) ghostWidth = 3;

    int isize = U.dimx();
    int jsize = U.dimy();
    int ksize = U.dimz();

    int nx = gParams.nx;
    int ny = gParams.ny;
    int nz = gParams.nz;

    const double twoPi = 2*M_PI;
    
    // 1. update Fourier modes of forcing field
    update_forcing_field_mode(dt);

    // 2. update phase with space dependencies
    // 3. update hydro variables
    for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
      
      real_t zPos = gParams.zMin + gParams.dz/2 + (k-ghostWidth + nz * gParams.mpiPosZ)*gParams.dz;
      
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	
	real_t yPos = gParams.yMin + gParams.dy/2 + (j-ghostWidth + ny * gParams.mpiPosY)*gParams.dy;
	
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	
	  real_t xPos = gParams.xMin + gParams.dx/2 + (i-ghostWidth + nx * gParams.mpiPosX)*gParams.dx;
	  
	  double phase[nMode];
	  memset(phase, 0, sizeof(phase)); // clear phase buffer

	  // compute phase factor at given i,j,k cell location
	  for (int iMode=0; iMode<nMode; iMode++)
	    phase[iMode] = 
	      xPos*mode[0*nMode+iMode] +
	      yPos*mode[1*nMode+iMode] +
	      zPos*mode[2*nMode+iMode];

	  double AAA[3];
	  for (int iDim=0; iDim<nDim; iDim++) {
	    double summ = 0.0;
	    for (int iMode=0; iMode<nMode; iMode++) {
	      summ += forcingField[iDim*nMode+iMode] * cos(twoPi*phase[iMode]);
	    } // end for iMode
	    AAA[iDim] = summ;
	  } // end for iDim

	  // compute internal energy
	  real_t eInt = 0.5  * (
				U(i,j,k,IU) * U(i,j,k,IU) + 
				U(i,j,k,IV) * U(i,j,k,IV) + 
				U(i,j,k,IW) * U(i,j,k,IW) ) / U(i,j,k,ID);
	  eInt = U(i,j,k,IP) - eInt;
				 
	  // update velocity field
	  real_t rho = U(i,j,k,ID);
	  U(i,j,k,IU) += AAA[0]*dt*rho;
	  U(i,j,k,IV) += AAA[1]*dt*rho;
	  U(i,j,k,IW) += AAA[2]*dt*rho;

	  // update total energy
	  U(i,j,k,IP) = eInt + 0.5  * (
				       U(i,j,k,IU) * U(i,j,k,IU) + 
				       U(i,j,k,IV) * U(i,j,k,IV) + 
				       U(i,j,k,IW) * U(i,j,k,IW) ) / U(i,j,k,ID);

	} // end for i
      } // end for j
    } // end for k

  } // ForcingOrnsteinUhlenbeck::add_forcing_field // CPU version

#endif // __CUDACC__

} // namespace hydroSimu
