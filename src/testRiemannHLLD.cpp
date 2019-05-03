/**
 * \file testRiemannHLLD.cpp
 * \brief numerical test of HLLD solver
 *
 * $Id: testRiemannHLLD.cpp 3452 2014-06-17 10:09:48Z pkestene $
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>

//#define USE_DOUBLE

#include "./hydro/real_type.h"
#include "./hydro/riemann.h"
#include "./hydro/riemann_mhd.h"

/** Parse command line / configuration parameter file */
#include "hydro/GetPot.h"
#include "utils/config/ConfigMap.h"

#include "hydro/HydroParameters.h" 

#ifdef __CUDACC__
__global__ void testRiemannHLL_gpu(real_t *qleft,
				   real_t *qright,
				   real_t *flux)
{
  real_t gdnv2[NVAR_3D];
  real_t flux2[NVAR_3D];
  riemann_hll<NVAR_3D>(qleft, qright, gdnv2, flux2);

  for (int i=0; i<NVAR_3D; i++) {
    flux[i] = flux2[i];
  }

}

__global__ void testRiemannHLLD_gpu(real_riemann_t *qleft,
				    real_riemann_t *qright,
				    real_riemann_t *flux)
{
  real_riemann_t flux2[NVAR_MHD];
  riemann_hlld(qleft, qright, flux2);

  for (int i=0; i<NVAR_MHD; i++) {
    flux[i] = flux2[i];
  }

}
#endif // __CUDACC__

int main(int argc, char *argv[])
{

  /* parse command line arguments */
  GetPot cl(argc, argv);

  /* search for configuration parameter file */
  const std::string default_input_file = std::string(argv[0])+ ".ini";
  const std::string input_file = cl.follow(default_input_file.c_str(),    "--param");

  /* parse parameters from input file */
  ConfigMap configMap(input_file); 

  HydroParameters * hydroParam = new HydroParameters(configMap);

  /*************/
  /* TEST HLL  */
  /*************/
  { 
    real_t qLeft[NVAR_3D];
    real_t qRight[NVAR_3D];
    real_t qgdnv[NVAR_3D];
    real_t flux[NVAR_3D];
    
    qLeft[ID]  = configMap.getFloat("BrioWu","qleft_ID",1);
    qLeft[IP]  = configMap.getFloat("BrioWu","qleft_IP",1);
    qLeft[IU]  = configMap.getFloat("BrioWu","qleft_IU",0);
    qLeft[IV]  = configMap.getFloat("BrioWu","qleft_IV",0);
    qLeft[IW]  = configMap.getFloat("BrioWu","qleft_IW",0);
    
    qRight[ID] = configMap.getFloat("BrioWu","qright_ID",0.125);
    qRight[IP] = configMap.getFloat("BrioWu","qright_IP",0.1);
    qRight[IU] = configMap.getFloat("BrioWu","qright_IU",0);
    qRight[IV] = configMap.getFloat("BrioWu","qright_IV",0);
    qRight[IW] = configMap.getFloat("BrioWu","qright_IW",0);

#ifdef __CUDACC__
    (void) qgdnv;

    real_t *d_qleft, *d_qright, *d_flux;
    cudaMalloc( (void**)&d_qleft,  NVAR_3D*sizeof(real_t) );
    cudaMalloc( (void**)&d_qright, NVAR_3D*sizeof(real_t) );
    cudaMalloc( (void**)&d_flux,   NVAR_3D*sizeof(real_t) );
    
    cudaMemcpy(d_qleft,  qLeft,  NVAR_3D*sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qright, qRight, NVAR_3D*sizeof(real_t), cudaMemcpyHostToDevice);
    
    testRiemannHLL_gpu<<<1,1>>>(d_qleft, d_qright, d_flux);
    
    cudaMemcpy(flux, d_flux, NVAR_3D*sizeof(real_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_qleft);
    cudaFree(d_qright);
    cudaFree(d_flux);
#else
    riemann_hll<NVAR_3D>(qLeft, qRight, qgdnv, flux);
#endif // __CUDACC__

    // print results on screen
    std::cout << "HLL HYDRO\n";
    for (int i=0; i<NVAR_3D; i++) {
      std::cout << std::setprecision(12) << "flux[" << i << "] = " << flux[i] << std::endl; 
    }
    std::cout << "\n\n";
  }
  
  /*************/
  /* TEST HLLD */
  /*************/
  {
    real_riemann_t qLeft[NVAR_MHD];
    real_riemann_t qRight[NVAR_MHD];
    real_riemann_t flux[NVAR_MHD];
    
    qLeft[ID]  = configMap.getFloat("BrioWu","qleft_ID",1);
    qLeft[IP]  = configMap.getFloat("BrioWu","qleft_IP",1);
    qLeft[IU]  = configMap.getFloat("BrioWu","qleft_IU",0);
    qLeft[IV]  = configMap.getFloat("BrioWu","qleft_IV",0);
    qLeft[IW]  = configMap.getFloat("BrioWu","qleft_IW",0);
    qLeft[IA]  = configMap.getFloat("BrioWu","qleft_IA",1);
    qLeft[IB]  = configMap.getFloat("BrioWu","qleft_IB",0.75);
    qLeft[IC]  = configMap.getFloat("BrioWu","qleft_IC",1);
    
    qRight[ID] = configMap.getFloat("BrioWu","qright_ID",0.125);
    qRight[IP] = configMap.getFloat("BrioWu","qright_IP",0.1);
    qRight[IU] = configMap.getFloat("BrioWu","qright_IU",0);
    qRight[IV] = configMap.getFloat("BrioWu","qright_IV",0);
    qRight[IW] = configMap.getFloat("BrioWu","qright_IW",0);
    qRight[IA] = configMap.getFloat("BrioWu","qright_IA",-1);
    qRight[IB] = configMap.getFloat("BrioWu","qright_IB",0.75);
    qRight[IC] = configMap.getFloat("BrioWu","qright_IC",-1);

#ifdef __CUDACC__
    real_riemann_t *d_qleft, *d_qright, *d_flux;
    cudaMalloc( (void**)&d_qleft,  NVAR_MHD*sizeof(real_riemann_t) );
    cudaMalloc( (void**)&d_qright, NVAR_MHD*sizeof(real_riemann_t) );
    cudaMalloc( (void**)&d_flux,   NVAR_MHD*sizeof(real_riemann_t) );
    
    cudaMemcpy(d_qleft,  qLeft,  NVAR_MHD*sizeof(real_riemann_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qright, qRight, NVAR_MHD*sizeof(real_riemann_t), cudaMemcpyHostToDevice);
    
    testRiemannHLLD_gpu<<<1,1>>>(d_qleft, d_qright, d_flux);
    
    cudaMemcpy(flux, d_flux, NVAR_MHD*sizeof(real_riemann_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_qleft);
    cudaFree(d_qright);
    cudaFree(d_flux);
#else
    riemann_hlld(qLeft, qRight, flux);
#endif // __CUDACC__

    // print results on screen
    std::cout << "HLLD\n";
    for (int i=0; i<NVAR_MHD; i++) {
      std::cout << std::setprecision(12) << "flux[" << i << "] = " << flux[i] << std::endl; 
    }
    std::cout << "\n\n";
  }


  // clean and exit
  delete hydroParam;

  return EXIT_SUCCESS;

}
