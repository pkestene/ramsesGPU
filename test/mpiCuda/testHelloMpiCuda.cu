/**
 * \file testHelloMpiCuda.cu
 * \brief A simple program to test MPI+Cuda
 *
 * \date 8 Oct 2010
 * \author Pierre Kestener
 */

// MPI-related includes
#include <GlobalMpiSession.h>
#include <MpiComm.h>

// CUDA-C includes
#include <cuda_runtime_api.h>

#include <cstdio>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  
  // initialize MPI session
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
  int myMpiRank = hydroSimu::MpiComm::world().getRank();
  int nMpiProc  = hydroSimu::MpiComm::world().getNProc();

  // initialize cuda
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
    printf("cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
    printf("\nFAILED\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
    printf("There is no device supporting CUDA\n");

  // grab information about current GPU device
  cudaDeviceProp deviceProp;
  int deviceId;
  int driverVersion = 0, runtimeVersion = 0;
  cudaSetDevice(myMpiRank%4);
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&deviceProp, deviceId);

  // grab information about CPU node / MPI process:
  int nameLen;
  char procName[MPI_MAX_PROCESSOR_NAME+1];
  int mpierr = ::MPI_Get_processor_name(procName,&nameLen);

  // dump information
  if (myMpiRank >= 0) {
    if (deviceProp.major == 9999 && deviceProp.minor == 9999)
      printf("There is no device supporting CUDA.\n");
    else if (deviceCount == 1)
      printf("There is 1 device supporting CUDA associated with MPI of rank 0\n");
    else
      printf("There are %d devices supporting CUDA\n", deviceCount);
    printf("Using Device %d: \"%s\"\n", deviceId, deviceProp.name);
#if CUDART_VERSION >= 2020
    // Console log
    cudaDriverGetVersion(&driverVersion);
    printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
#endif
    printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
    printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
    
    printf("  Total amount of global memory:                 %lu bytes (%lu MBytes)\n", deviceProp.totalGlobalMem, deviceProp.totalGlobalMem/1024/1024);

  }

  printf("MPI process number %d out of %d on machine %s\n",myMpiRank,nMpiProc,procName);
  printf("Working GPU device Id is %d\n",deviceId);

  return EXIT_SUCCESS;

}
