/**
 * \file testMpiHelloGlobalSession.cpp
 * \brief Mpi helloworld application using CPP wrapper classes from libHydroMpi.
 *
 *
 * To launch: mpirun -n 4 ./testMpiHelloGlobalSession
 *
 * Uses the MpiGlobalSession class to handle Initialize/Finalize operation.
 *
 * \date 1 Oct 2010
 * \author Pierre Kestener
 */
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <ctime>

#include <GlobalMpiSession.h>
#include <MpiComm.h>

int main(int argc,char **argv)
{
  int myRank, numProcs, namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME+1];

  // MPI resources
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);

  numProcs  = hydroSimu::MpiComm::world().getNProc();
  myRank    = hydroSimu::MpiComm::world().getRank();

  MPI_Get_processor_name(processor_name, &namelength);
  
  // print local time on rank 0 machine
  if ( myRank == 0 ) {
    std::time_t curtime = std::time (NULL);
    std::cout << "Local time of process 0 : " << std::asctime (std::localtime (&curtime)) << std::endl;
  }
  
  // print process rank and hostname
  std::cout << "MPI process " << myRank << " of " << numProcs << " is on " <<
    processor_name << std::endl;

  return EXIT_SUCCESS;
}
