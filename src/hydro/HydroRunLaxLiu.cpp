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
 * \file HydroRunLaxLiu.cpp
 * \brief Implements class HydroRunLaxLiu
 * 
 * 2D Euler equation solver on a cartesian grid using Lax-Liu method
 * (i.e. positive scheme)
 *
 * \date January 2010
 * \author P. Kestener
 *
 * $Id: HydroRunLaxLiu.cpp 2108 2012-05-23 12:07:21Z pkestene $
 */
#include "HydroRunLaxLiu.h"

// include CUDA kernel when necessary
#ifdef __CUDACC__
#include "laxliu.cuh"
#endif // __CUDACC__
#include "constoprim.h"
#include "positiveScheme.h"

namespace hydroSimu {

////////////////////////////////////////////////////////////////////////////////
// HydroRunLaxLiu class methods body
////////////////////////////////////////////////////////////////////////////////

HydroRunLaxLiu::HydroRunLaxLiu(ConfigMap &_configMap)
  : HydroRunBase(_configMap),
    h_U1(),
    h_tmp1()
#ifdef __CUDACC__
  ,
    d_U1(),
    d_tmp1()
#endif // __CUDACC__

{

    // assert dimType == TWO_D (3D is not implemented in the Kurganov
  // Tadmor scheme
  assert((dimType!=THREE_D)&&"3D Lax-Liu scheme is not available !!!");

  h_U1.allocate(make_uint3(isize, jsize, NVAR_2D));
  h_tmp1.allocate(make_uint3(isize, jsize, NVAR_2D));
#ifdef __CUDACC__
  d_U1.allocate(make_uint3(isize, jsize, NVAR_2D));
  d_tmp1.allocate(make_uint3(isize, jsize, NVAR_2D));
#endif // __CUDACC__

  // make sure variable declared as __constant__ are copied to device
  // for current compilation unit
  copyToSymbolMemory();

} // HydroRunLaxLiu::HydroRunLaxLiu

// =======================================================
// =======================================================
HydroRunLaxLiu::~HydroRunLaxLiu()
{
} // HydroRunLaxLiu::~HydroRunLaxLiu


// =======================================================
// =======================================================
/** 
 * \brief 2D laxliu evolve : evaluate future values of rho, rhou, rhov, E_t
 * 
 * local variables 
 * up        : \f$ U_{j+1} \f$ 
 * um        : \f$ U_{j} \f$
 * dup       : \f$ U_{j+2}-U_{j+1} \f$
 * dum       : \f$ U_{j}  -U_{j-1} \f$
 * du        : \f$ U_{j+1}-U_{j} \f$
 * dw        : \f$ R^{-1}*(U_{j+1}-U_{j}) \f$, \f$ R\f$ is the Roe matrix
 * dwf       : Diffusive flux in char fields
 *
 * fc        : Central differencing flux
 * df=R*dwf  : Diffusive flux
 * f =fc+df  : flux in x direction
 * g         : flux in y direction
 *
 */
#ifdef __CUDACC__
void HydroRunLaxLiu::laxliu_evolve(DeviceArray<float> &a1, DeviceArray<float> &a2)
{
 
  // apply boundary condition to input a1
  make_boundaries(a1,1);
  make_boundaries(a1,2);

  dim3 dimBlock(BLOCK_DIMX, BLOCK_DIMY);
  dim3 dimGrid(blocksFor(isize, BLOCK_INNER_DIMX), blocksFor(jsize, BLOCK_INNER_DIMY));
  laxliu_evolve_kernel<<<dimGrid, dimBlock>>>(a1.data(), a2.data(), a1.pitch(), a1.dimx(), a1.dimy(),_gParams.XLAMBDA,_gParams.YLAMBDA);
  
} // HydroRunLaxLiu::laxliu_evolve
#else // CPU version
void HydroRunLaxLiu::laxliu_evolve(HostArray<float>   &a1, HostArray<float>   &a2)
{

  // apply boundary condition to input a1
  make_boundaries(a1,1);
  make_boundaries(a1,2);
  

  // local variables :
  float up[NVAR_2D], um[NVAR_2D], du[NVAR_2D], dup[NVAR_2D], dum[NVAR_2D];
  float fc[NVAR_2D], df[NVAR_2D] = {0, 0};
  
  float *data     = a1.data();
  float *dataOut  = a2.data();

  int offset2D = a1.section(); 
  int pitch    = a1.pitch();
  
  // laxliu2d evolve : 1st stage
  for( int j = 2; j < ny+2; ++j ) { 
    
    float tmp[NVAR_2D]    = {0.0f, 0.0f, 0.0f, 0.0f};
    float tmpOld[NVAR_2D] = {0.0f, 0.0f, 0.0f, 0.0f};
    float delta[NVAR_2D]  = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for( int i = 1; i < nx+2; ++i) {
      
      unsigned int index = j*pitch+i;
      
      for (unsigned int k=0; k < NVAR_2D; ++k) {
	int offset = k*offset2D+index;
	up[k]  = data[offset+1];
	um[k]  = data[offset];
	dum[k] = data[offset]  -data[offset-1];
	dup[k] = data[offset+2]-data[offset+1];
	du[k]  = data[offset+1]-data[offset];
      }
      
      central_diff_flux<NVAR_2D>(up,um,fc);
      diffusive_flux<NVAR_2D>(up,um,du,dup,dum,df);
      
      for (unsigned int k=0; k < NVAR_2D; ++k) {
	tmp[k]    = fc[k]+df[k];
	delta[k]  = tmp[k]-tmpOld[k];
	tmpOld[k] = tmp[k];
	int offset = k*offset2D+index;
	if (i>1) {
	  dataOut[offset] = data[offset] - _gParams.XLAMBDA*(delta[k]);
	}
      }
      
    } // for i ...
    
  } // for j ...
  
  // laxliu2d evolve : 2nd stage
  for( int i = 2; i < nx+2; ++i) {
    
    float tmp[NVAR_2D]    = {0.0f, 0.0f, 0.0f, 0.0f};
    float tmpOld[NVAR_2D] = {0.0f, 0.0f, 0.0f, 0.0f};
    float delta[NVAR_2D]  = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for( int j = 1; j < ny+2; ++j) { 
      
      unsigned int index = j*pitch+i;
      
      for (unsigned int k=0; k < NVAR_2D; ++k) {
	
	// swap directions :
	int k1=k;
	if (k==1 || k==2)
	  k1=3-k;
	int offset = k1*offset2D+index;
	
	up[k]  = data[offset+pitch];
	um[k]  = data[offset];
	dum[k] = data[offset]        -data[offset-pitch];
	dup[k] = data[offset+2*pitch]-data[offset+pitch];
	du[k]  = data[offset+pitch]  -data[offset];
      }
      
      central_diff_flux<NVAR_2D>(up,um,fc);
      diffusive_flux<NVAR_2D>(up,um,du,dup,dum,df);
      
      for (unsigned int k=0; k < NVAR_2D; ++k) {
	tmp[k]=fc[k]+df[k];
	delta[k]=tmp[k]-tmpOld[k];
	tmpOld[k]=tmp[k];
      }
      
      for (unsigned int k=0; k < NVAR_2D; ++k) {
	int k1=k;
	if (k==1 || k==2)
	  k1=3-k;
	int offset = k*offset2D+index;
	if (j>1) {
	  dataOut[offset] -= _gParams.YLAMBDA*delta[k1];
	  if (k==0 || k==3) {
	    dataOut[offset]  = fmaxf(dataOut[offset], ::gParams.smallr);
	  }
	}
      }
      
    } // for i ...
    
  } // for j ...
  
  
} // HydroRunLaxLiu::laxliu_evolve
#endif // __CUDACC__

/** 
 * \brief Compute a1=(a1+a2)/2
 *
 * Compute (a1+a2)/2, a1 and a2 must have the same dimension (not checked)
 * 
 * @param a1 
 * @param a2 
 */
#ifdef __CUDACC__
void HydroRunLaxLiu::averageArray(DeviceArray<float> &a1, DeviceArray<float> &a2)
{
  
  dim3 dimBlock(BLOCK_DIMX, BLOCK_DIMY);
  dim3 dimGrid(blocksFor(isize, BLOCK_DIMX), blocksFor(jsize, BLOCK_DIMY));
  laxliu_average_kernel<<<dimGrid, dimBlock>>>(a1.data(), a2.data(), a1.pitch(), a1.dimx(), a1.dimy()); 
  
} // HydroRunLaxLiu::averageArray
#else // CPU version
void HydroRunLaxLiu::averageArray(HostArray<float>   &a1, HostArray<float>   &a2)
{
  float* data1 = a1.data();
  float* data2 = a2.data();

  for (unsigned int index=0; index<a1.size(); ++index) {
    data1[index] = 0.5f*(data1[index]+data2[index]);
  }

} // HydroRunLaxLiu::averageArray
#endif

// =======================================================
// =======================================================
/**
 * main routine to start simulation.
 */
void HydroRunLaxLiu::start() {

  /*
   * initial condition
   */
  std::cout << "Initialization\n";
  int configNb = configMap.getInteger("hydro" , "riemann_config_number", 0);
  setRiemannConfId(configNb);
  init_simulation(problem);
  
 
  std::cout << "Starting time integration (LaxLiu method, positive scheme)" << std::endl;
  float t   = 0.0f;
  float dt  = _gParams.XLAMBDA/nx;
  
  if (_gParams.XLAMBDA/ny < dt)
    dt = _gParams.XLAMBDA*1.0f/ny;
  int nStep  = 0;
  int nSteps = (int) ceilf(tEnd/dt);
  dt=tEnd/nSteps;
  _gParams.XLAMBDA = dt*nx;
  _gParams.YLAMBDA = dt*ny;

  std::cout << "NX      : " << nx       << std::endl;
  std::cout << "NY      : " << ny       << std::endl;
  std::cout << "dt      : " << dt       << std::endl;
  std::cout << "nSteps  : " << nSteps   << std::endl;
  std::cout << "XLAMBDA : " << _gParams.XLAMBDA << std::endl;
  std::cout << "YLAMBDA : " << _gParams.YLAMBDA << std::endl;
  

  while(t < tEnd && nStep < nStepmax)
    {

      /* */
#ifdef __CUDACC__
      laxliu_evolve(d_U ,d_tmp1);      
      laxliu_evolve(d_tmp1,d_U1);
      averageArray (d_U ,d_U1);
#else
      laxliu_evolve(h_U ,h_tmp1);
      laxliu_evolve(h_tmp1,h_U1);
      averageArray (h_U ,h_U1);
#endif // __CUDACC__

      /* Output results */
      if((nStep % nOutput)==0) {
	//outputBin(nStep, t);
	outputXsm(h_U, nStep, ID);
	outputXsm(h_U, nStep, IU);
	//outputXsm(h_U, nStep, t, IV);
	outputPng(h_U, nStep, ID);
	std::cout << "step=" << nStep << " t=" << t << " dt=" << dt << std::endl;
      }
      
      /* increase time */
      nStep++;
      t+=dt;
      
    }

} // HydroRunLaxLiu::start

// =======================================================
// =======================================================
/*
 * do one time step integration
 */
void HydroRunLaxLiu::oneStepIntegration(int& nStep, real_t& t, real_t& dt) {

  (void) nStep;
  (void) t;
  (void) dt;

} // HydroRunLaxLiu::oneStepIntegration

// =======================================================
// =======================================================
void HydroRunLaxLiu::init_hydro_jet()
{

  /* initial condition in grid interior */
  memset(h_U.data(),0,h_U.sizeBytes());
  memset(h_U1.data(),0,h_U1.sizeBytes());
  memset(h_tmp1.data(),0,h_tmp1.sizeBytes());

  /* initialize inner grid */
  for (int j=2; j<jsize-2; j++)
    for (int i=2; i<isize-2; i++) {
      //int index = i+isize*j;
      // fill density, U, V and pressure sub-array
      h_U(i,j,ID)=1.0f;
      h_U(i,j,IU)=0.0f;
      h_U(i,j,IV)=0.0f;
      h_U(i,j,IP)=1.0f/(_gParams.gamma0-1.0f);
    }

#ifdef __CUDACC__
  d_U.copyFromHost(h_U); // load data into the VRAM
#endif // __CUDACC__

} // HydroRunLaxLiu::init_hydro_jet

} // namespace hydroSimu
