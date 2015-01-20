/**
 * \file testMpiHello.cpp
 * \brief Small MPI test using c++ API.
 *
 * To launch: mpirun -n 4 ./testMpiHello
 *
 * !!! WARNING C++ binding are deprecated in MPI 2.2 !!!
 *
 * \ingroup test
 *
 * \date 27 Sept 2010
 * \author Pierre Kestener
 */
// the following pragma does the same as -Wno-unused-parameter given
// in Makefile.am
//#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cstring>

int main(int argc,char **argv)
{
  int myRank, numProcs, namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME+1];

  time_t         curtime;
  struct tm     *loctime;

  // MPI initialize
  MPI::Init(argc,argv);
  numProcs  = MPI::COMM_WORLD.Get_size();
  myRank    = MPI::COMM_WORLD.Get_rank();
  try {
    MPI::Get_processor_name(processor_name,namelength);
  } catch (MPI::Exception e) {
    std::cout << "MPI ERROR: " << e.Get_error_code()	\
	      << " -" << e.Get_error_string()		\
	      << std::endl;
    std::cout << "Unable to get processor name; set to \" unknown\" \n";
    strcpy(processor_name, "unknown");
  }
  
  // print local time on rank 0 machine
  if ( myRank == 0 ) {
    curtime = time (NULL);
    loctime = localtime (&curtime);
    std::cout << "Local time of process 0 : " << asctime (loctime) << std::endl;
  }
  
  // print process rank and hostname
  std::cout << "MPI process " << myRank << " of " << numProcs << " is on " <<
    processor_name << std::endl;
  
  // MPI finalize 
  MPI::Finalize();
  return EXIT_SUCCESS;
}
