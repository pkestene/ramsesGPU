/**
 * \file testMpiBuffer.cpp
 * \brief test transfert buffers (HostArray class) between 2 MPI processes.
 *
 * Note for debug: see http://www.open-mpi.org/faq/?category=debugging
 * 1. mpirun -np 2 xterm -e gdb ./testMpiBuffer
 * 2. mpirun -np 2 valgrind ./testMpiBuffer
 *    mpirun -np 2 ./testMpiBuffer

 * \date 15 Oct 2010
 * \author Pierre Kestener
 */
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <typeinfo>

#include <common_types.h>
#include <Arrays.h>

#include <GlobalMpiSession.h>
#include <HydroMpiParameters.h>
#include <mpiBorderUtils.h>


using namespace hydroSimu;

int main(int argc, char * argv[]) {

  int myRank, numTasks, namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME+1];
  int tag = 1;

  /* Initialize MPI session */
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
  hydroSimu::MpiComm worldComm = hydroSimu::MpiComm::world();
  myRank = worldComm.getRank();
  numTasks = worldComm.getNProc();
  MPI::Get_processor_name(processor_name,namelength);

  MPI_Request reqs[2]; // 1 send + 1 receive
  MPI_Status stats[2]; // 1 send + 1 receive

  try {
    
    // check we only have 2 MPI processes
    if (numTasks == 2) {
      
      // some test data
      int isize = 8, jsize = 16;
      HostArray<real_t> U;
      HostArray<real_t> bSend;
      HostArray<real_t> bRecv;
      
      const int ghostWidth = 3;

      // memory allocation
      U.allocate    ( make_uint3(isize     , jsize, 1) );
      bSend.allocate( make_uint3(ghostWidth, jsize, 1) );
      bRecv.allocate( make_uint3(ghostWidth, jsize, 1) );

      // initialization
      for (uint j=0; j<U.dimy(); ++j)
	for (uint i=0; i<U.dimx(); ++i) {
	  U(i,j,0) = static_cast<real_t>(i*myRank + j*(1-myRank));
	}
      for (uint j=0; j<U.dimy(); ++j) {
	bSend(0,j,0) = 0.25*(myRank+1);
	bSend(1,j,0) = 0.25*(myRank+1);
	if (ghostWidth == 3)
	  bSend(2,j,0) = 0.25*(myRank+1);

	bRecv(0,j,0) = 0.0;
	bRecv(1,j,0) = 0.0;
 	if (ghostWidth == 3)
	  bRecv(2,j,0) = 0.0;
      }
    
      // runtime determination if we are using float ou double
      int data_type = typeid(1.0f).name() == typeid((real_t)1.0f).name() ? hydroSimu::MpiComm::FLOAT : hydroSimu::MpiComm::DOUBLE;


      // before buffer exchange
      std::cout << "Print array U before buffer exchange:\n\n";
      for (int iRank=0; iRank<numTasks; ++iRank) {
	MPI_Barrier(MPI_COMM_WORLD);
	if (iRank == myRank) {
	  std::cout << "#####################\n";
	  std::cout << "myRank : " << myRank << std::endl;;
	  std::cout << U;
	}
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // communication
      if (myRank == 0) {
	reqs[0] = worldComm.Isend(bSend.data(), bSend.size(), data_type, 1, tag);
	reqs[1] = worldComm.Irecv(bRecv.data(), bRecv.size(), data_type, 1, tag);
      } else {
	reqs[0] = worldComm.Isend(bSend.data(), bSend.size(), data_type, 0, tag);
	reqs[1] = worldComm.Irecv(bRecv.data(), bRecv.size(), data_type, 0, tag);
      }
      MPI_Waitall(2, reqs, stats);
      
      // copy buffer
      if (myRank == 0) {
	copyBorderBufRecvToHostArray<XMAX,TWO_D,ghostWidth>(U,bRecv);
      } else {
	copyBorderBufRecvToHostArray<XMIN,TWO_D,ghostWidth>(U,bRecv);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      
      // after buffer exchange
      std::cout << "Print array U after buffer exchange:\n\n";
      for (int iRank=0; iRank<numTasks; ++iRank) {
	MPI_Barrier(MPI_COMM_WORLD);
	if (iRank == myRank) {
	  std::cout << "#####################\n";
	  std::cout << "myRank : " << myRank << std::endl;
	  std::cout << U;
	}
      }
      
      std::cout << "#######################" << std::endl;
      std::cout << "Done !!!               " << std::endl;
      std::cout << "#######################" << std::endl;
      
      
    } else {
      std::cout << "Must specify " << 2 << " MPI processes. Terminating.\n";
    }
    
  } catch (...) {
    
    std::cerr << "Exception caught, something really bad happened...\n\n\n";
    return EXIT_FAILURE;
    
  }
    
  return EXIT_SUCCESS;
  
}
