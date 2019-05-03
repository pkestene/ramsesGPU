/**
 * \file testMpiTopology_2D.cpp
 * \brief Simple example showing how to use MPI virtual topology (2D)
 *
 *
 * Simple MPI test with a cartesian grid topology.
 * In this example, a 3 by 4 grid is instantiate.
 *
 * !!! WARNING : C++ binding are deprecated in MPI 2.2 !!!
 *
 *
 * IMPORTANT NOTE.
 * The purpose of class MpiCommCart was to avoid the use of the
 * original C++ API which is deprecated in MPI 2.2 standards, but
 * the Teuchos API was far from being complete to be really useable
 * here; so we added some point-to-point communication routines.
 *
 * We already have introduced Cartesian topology inside this
 * framework.
 * As for now (5 october 2010), the question is what the best way to
 * go ? 
 * 1. stick with the standard c++ API
 * 2. go on improving our own Teuchos-inspired C++ wrapping which is 
 * MPI-implementation agnostic (i.e. independant from anything).
 *
 * \date 27 sept 2010
 * \author Pierre Kestener
 *
 */

#include <mpi.h>

#include <iostream>
#include <cstdlib>
#include <unistd.h> // for sleep

#include <GlobalMpiSession.h>
#include <MpiCommCart.h>

#define SIZE_X 2
#define SIZE_Y 2
#define SIZE_Z 4
#define SIZE_2D (SIZE_X * SIZE_Y)
#define SIZE_3D (SIZE_X * SIZE_Y * SIZE_Z)

#define N_NEIGHBORS_2D 4
#define N_NEIGHBORS_3D 6

#define NDIM 2

int main(int argc, char* argv[]) 
{
  int myRank, numTasks, namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME+1];

  int source, dest, outbuf, i, tag=1;
  int inbuf[N_NEIGHBORS_2D]={MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,};
  int nbrs[N_NEIGHBORS_2D];
  int periods=hydroSimu::MPI_CART_PERIODIC_TRUE;
  int reorder=hydroSimu::MPI_REORDER_TRUE;
  int coords[NDIM];
  
  MPI_Request reqs[2*N_NEIGHBORS_2D];
  MPI_Status stats[2*N_NEIGHBORS_2D];

  // MPI resources
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
  hydroSimu::MpiComm worldComm = hydroSimu::MpiComm::world();

  myRank = worldComm.getRank();
  numTasks = worldComm.getNProc();

  MPI_Get_processor_name(processor_name,&namelength);
  
  // print warning
  if ( myRank == 0 ) {
    std::cout << "Take care that MPI Cartesian Topology uses COLUMN MAJOR-FORMAT !!!\n";
    std::cout << "\n";
    std::cout << "In this test, each MPI process of the cartesian grid sends a message\n";
    std::cout << "containing a integer (rank of the current process) to all of its\n";
    std::cout << "neighbors. So you must chech that arrays \"neighbors\" and \"inbuf\"\n";
    std::cout << "contain the same information !\n\n";
  }

    // 2D CARTESIAN MPI MESH
  if (numTasks == SIZE_2D) {
  
    // create the cartesian topology
    hydroSimu::MpiCommCart cartcomm(SIZE_X, SIZE_Y, (int) periods, (int) reorder);
    
    // get rank inside the tolopogy
    myRank = cartcomm.getRank();

    // get 2D coordinates inside topology
    cartcomm.getMyCoords(coords);

    // get rank of source (x-1) and destination (x+1) process
    // take care MPI uses column-major order
    // get rank of source (y-1) and destination (y+1) process
    nbrs[hydroSimu::X_MIN] = cartcomm.getNeighborRank<hydroSimu::X_MIN>();
    nbrs[hydroSimu::X_MAX] = cartcomm.getNeighborRank<hydroSimu::X_MAX>();
    nbrs[hydroSimu::Y_MIN] = cartcomm.getNeighborRank<hydroSimu::Y_MIN>();
    nbrs[hydroSimu::Y_MAX] = cartcomm.getNeighborRank<hydroSimu::Y_MAX>();

    outbuf = myRank;

    // send    my rank to   each of my neighbors
    // receive my rank from each of my neighbors
    // inbuf should contain the rank of all neighbors
   for (i=0; i<N_NEIGHBORS_2D; i++) {
      dest = nbrs[i];
      source = nbrs[i];
      reqs[i               ] = cartcomm.Isend(&outbuf, 1, hydroSimu::MpiComm::INT, dest, tag);
      reqs[i+N_NEIGHBORS_2D] = cartcomm.Irecv(&inbuf[i], 1, hydroSimu::MpiComm::INT, source, tag);
    }
    MPI_Waitall(2*N_NEIGHBORS_2D, reqs, stats);

    printf("rank= %2d coords= %d %d  neighbors(x-,+-,y-,y+) = %2d %2d %2d %2d\n", 
  	   myRank,
  	   coords[0],coords[1], 
  	   nbrs[hydroSimu::X_MIN],
  	   nbrs[hydroSimu::X_MAX], 
  	   nbrs[hydroSimu::Y_MIN], 
  	   nbrs[hydroSimu::Y_MAX]);
    printf("rank= %2d coords= %d %d  inbuf    (x-,x+,y-,y+) = %2d %2d %2d %2d\n", 
  	   myRank,
  	   coords[0],coords[1],
  	   inbuf[hydroSimu::X_MIN],
  	   inbuf[hydroSimu::X_MAX],
  	   inbuf[hydroSimu::Y_MIN],
  	   inbuf[hydroSimu::Y_MAX]);
    
    // print topology
    MPI_Barrier(MPI_COMM_WORLD); 
    sleep(1);

    if (myRank == 0) {
      printf("Print topology (COLUMN MAJOR-ORDER) for %dx%d 2D grid:\n",SIZE_X,SIZE_Y);
      printf(" rank     i     j\n");
    }
    printf("%5d %5d %5d %10d %10d %10d %10d\n",myRank,
  	   coords[0],coords[1],
  	   nbrs[hydroSimu::X_MIN], nbrs[hydroSimu::X_MAX],
  	   nbrs[hydroSimu::Y_MIN], nbrs[hydroSimu::Y_MAX]
  	   );
  } else {
    std::cout << "Must specify " << SIZE_2D << " processors. Terminating.\n";
  }

  return EXIT_SUCCESS;

}
