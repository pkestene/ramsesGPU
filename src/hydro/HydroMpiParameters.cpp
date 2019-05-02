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
 * \file HydroMpiParameters.cpp
 * \brief Implements class HydroMpiParameters.
 *
 * \date 6 Oct 2010
 * \author Pierre Kestener
 *
 * $Id: HydroMpiParameters.cpp 2108 2012-05-23 12:07:21Z pkestene $
 */
#include "HydroMpiParameters.h"

#include <unistd.h> // for "gethostname"

#include "utils/mpiUtils/mpiEnums.h" // for DIR_X, ....

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // HydroMpiParameters class methods body
  ////////////////////////////////////////////////////////////////////////////////

  // =======================================================
  // =======================================================
  HydroMpiParameters::HydroMpiParameters(ConfigMap &_configMap) :
    HydroParameters(_configMap, false), mx(0), my(0), mz(0), 
    myRank(0), nProcs(0), myMpiPos(),
    nNeighbors(0), neighborsRank(), neighborsBC()
  {

    // MPI parameters :
    mx = configMap.getInteger("mpi", "mx", 1);
    my = configMap.getInteger("mpi", "my", 1);
    mz = configMap.getInteger("mpi", "mz", 1);

    // copy MPI topology sizes into gParams structure (so that it will also
    // be available as a global constant, usefull for GPU implementation in godunov_unsplit_mhd.cuh).
    _gParams.mx = mx;
    _gParams.my = my;
    _gParams.mz = mz;

    // check that parameters are consistent
    bool error = false;
    error |= (mx < 1);
    error |= (my < 1);
    error |= (mz < 1);
    if (dimType == TWO_D and mz != 1)
      error = true;
    TEST_FOR_EXCEPTION_PRINT(error,
			     std::runtime_error,
			     "Inconsistent geometry; check parameter file for nx, ny, nz and mx, my, mz !\n",
			     &std::cerr);

    // get world communicator size and check it is consistent with mesh grid sizes
    nProcs = MpiComm::world().getNProc();
    TEST_FOR_EXCEPTION_PRINT(nProcs != mx*my*mz,
			     std::runtime_error,
			     "Inconsistent MPI cartesian virtual topology geometry; \n mx*my*mz must match with parameter given to mpirun !!!\n",
			     &std::cerr);

    // create the MPI communicator for our cartesian mesh
    if (dimType == TWO_D) {
      communicator = new MpiCommCart(mx, my, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
      nDim = 2;
    } else {
      communicator = new MpiCommCart(mx, my, mz, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
      nDim = 3;
    }

    // get my MPI rank inside topology
    myRank = communicator->getRank();
    
    // get my coordinates inside topology
    // myMpiPos[0] is between 0 and mx-1
    // myMpiPos[1] is between 0 and my-1
    // myMpiPos[2] is between 0 and mz-1
    myMpiPos.resize(nDim);
    communicator->getMyCoords(&myMpiPos[0]);

    // copy coordinate into gParams structure (so that it will also
    // be available as a global constant, usefull for GPU implementation).
    _gParams.mpiPosX = myMpiPos[0];
    _gParams.mpiPosY = myMpiPos[1];
    _gParams.mpiPosZ = myMpiPos[2];

    /*
     * compute MPI ranks of our neighbors and 
     * set default boundary condition types
     */
    if (dimType == TWO_D) {
      nNeighbors = N_NEIGHBORS_2D;
      neighborsRank.resize(nNeighbors);
      neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
      neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
      neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
      neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
      
      neighborsBC.resize(nNeighbors);
      neighborsBC[X_MIN] = BC_COPY;
      neighborsBC[X_MAX] = BC_COPY;
      neighborsBC[Y_MIN] = BC_COPY;
      neighborsBC[Y_MAX] = BC_COPY;
    } else {
      nNeighbors = N_NEIGHBORS_3D;
      neighborsRank.resize(nNeighbors);
      neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
      neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
      neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
      neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
      neighborsRank[Z_MIN] = communicator->getNeighborRank<Z_MIN>();
      neighborsRank[Z_MAX] = communicator->getNeighborRank<Z_MAX>();

      neighborsBC.resize(nNeighbors);
      neighborsBC[X_MIN] = BC_COPY;
      neighborsBC[X_MAX] = BC_COPY;
      neighborsBC[Y_MIN] = BC_COPY;
      neighborsBC[Y_MAX] = BC_COPY;
      neighborsBC[Z_MIN] = BC_COPY;
      neighborsBC[Z_MAX] = BC_COPY;
    }

    /*
     * identify outside boundaries (no actual communication if we are
     * doing BC_DIRICHLET or BC_NEUMANN)
     *
     * Please notice the duality 
     * XMIN -- boundary_xmax
     * XMAX -- boundary_xmin
     *
     */

    // X_MIN boundary
    if (myMpiPos[MPI_TOPO_DIR_X] == 0)
      neighborsBC[X_MIN] = boundary_xmin;

    // X_MAX boundary
    if (myMpiPos[MPI_TOPO_DIR_X] == mx-1)
      neighborsBC[X_MAX] = boundary_xmax;

    // Y_MIN boundary
    if (myMpiPos[MPI_TOPO_DIR_Y] == 0)
      neighborsBC[Y_MIN] = boundary_ymin;

    // Y_MAX boundary
    if (myMpiPos[MPI_TOPO_DIR_Y] == my-1)
      neighborsBC[Y_MAX] = boundary_ymax;

    if (dimType == THREE_D) {

      // Z_MIN boundary
      if (myMpiPos[MPI_TOPO_DIR_Z] == 0)
	neighborsBC[Z_MIN] = boundary_zmin;
      
      // Y_MAX boundary
      if (myMpiPos[MPI_TOPO_DIR_Z] == mz-1)
	neighborsBC[Z_MAX] = boundary_zmax;
      
    } // end THREE_D

    /*
     * Initialize CUDA device if needed.
     * When running on a Linux machine with mutiple GPU per node, it might be
     * very helpfull if admin has set the CUDA device compute mode to exclusive
     * so that a device is only attached to 1 host thread (i.e. 2 different host
     * thread can not communicate with the same GPU).
     *
     * As a sys-admin, just run for all devices command:
     *   nvidia-smi -g $(DEV_ID) -c 1
     *
     * If compute mode is set to normal mode, we need to use cudaSetDevice, 
     * so that each MPI device is mapped onto a different GPU device.
     * 
     * At CCRT, on machine Titane, each node (2 quadri-proc) "sees" only 
     * half a Tesla S1070, that means cudaGetDeviceCount should return 2.
     * If we want the ration 1 MPI process <-> 1 GPU, we need to allocate
     * N nodes and 2*N tasks (MPI process). 
     */
#ifdef __CUDACC__
    // get device count
    int count;
    cutilSafeCall( cudaGetDeviceCount(&count) );
    
    int devId = myRank % count;
    cutilSafeCall( cudaSetDevice(devId) );
    
    cudaDeviceProp deviceProp;
    int myDevId = -1;
    cutilSafeCall( cudaGetDevice( &myDevId ) );
    cutilSafeCall( cudaGetDeviceProperties( &deviceProp, myDevId ) );
    // faire un cudaSetDevice et cudaGetDeviceProp et aficher le nom
    // ajouter un booleen dans le constructeur pour savoir si on veut faire ca
    // sachant que sur Titane, probablement que le mode exclusif est active
    // a verifier demain

    std::cout << "MPI process " << myRank << " is using GPU device num " << myDevId << std::endl;

#endif //__CUDACC__

    // fix space resolution :
    // need to take into account number of MPI process in each direction
    float    xMax = configMap.getFloat("mesh","xmax",1.0);
    float    yMax = configMap.getFloat("mesh","ymax",1.0);
    float    zMax = configMap.getFloat("mesh","zmax",1.0);
    _gParams.dx = (xMax- _gParams.xMin)/(nx*mx);
    _gParams.dy = (yMax- _gParams.yMin)/(ny*my);
    _gParams.dz = (zMax- _gParams.zMin)/(nz*mz);

    // print information about current setup
    if (myRank == 0) {
      std::cout << "We are about to start simulation with the following characteristics\n";

      std::cout << "Global resolution : " << 
	nx*mx << " x " << ny*my << " x " << nz*mz << "\n";
      std::cout << "Local  resolution : " << 
	nx << " x " << ny << " x " << nz << "\n";
      std::cout << "MPI Cartesian topology : " << mx << "x" << my << "x" << mz << std::endl;
    }

#ifdef __CUDACC__
    char hostname[1024];
    gethostname(hostname, 1023);
    std::cout << "hostname      : " << hostname << std::endl;
    std::cout << hostname << " [MPI] myRank  : " << myRank << std::endl;
    std::cout << hostname << " [GPU] myDevId : " << myDevId << " (" << deviceProp.name << ")" << std::endl;
#endif // __CUDACC__
  } // HydroMpiParameters::HydroMpiParameters
  
  // =======================================================
  // =======================================================
  HydroMpiParameters::~HydroMpiParameters()
  {

    delete communicator;
    
  } // HydroMpiParameters::~HydroMpiParameters
  
  
} // namespace hydroSimu
