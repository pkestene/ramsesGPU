/**
 * \file testMpiTopology_2D_Cpp_API.cpp
 * \brief Simple example showing how to use MPI virtual topology (2D),
 * CPP API.
 *
 * Simple MPI test with a cartesian grid topology.
 * In this example, a 3 by 4 grid is instantiate.
 *
 * !!! WARNING : C++ binding are deprecated in MPI 2.2 !!!
 *
 * \date 27 sept 2010
 * \author Pierre Kestener
 *
 */

#include <mpi.h>

#include <iostream>
#include <cstdlib>
#include <unistd.h>

#define SIZE_X 4
#define SIZE_Y 3
#define SIZE_Z 4
#define SIZE_2D (SIZE_X * SIZE_Y)
#define SIZE_3D (SIZE_X * SIZE_Y * SIZE_Z)

// identifying neighbors
enum {
  X_PLUS_1  = 0,
  X_MINUS_1 = 1,
  Y_PLUS_1  = 2,
  Y_MINUS_1 = 3,
  Z_PLUS_1  = 4,
  Z_MINUS_1 = 5
};

// allow processor reordering by the MPI cartesian communicator
static const bool MPI_REORDER_FALSE = false;
static const bool MPI_REORDER_TRUE  = true;

static const bool MPI_CART_PERIODIC_FALSE = false;
static const bool MPI_CART_PERIODIC_TRUE  = true;

// MPI topology directions
enum {
  MPI_TOPO_DIR_X = 0,
  MPI_TOPO_DIR_Y = 1,
  MPI_TOPO_DIR_Z = 2
};

// MPI topology shift direction
enum {
  MPI_SHIFT_NONE = 0,
  MPI_SHIFT_FORWARD = 1
};

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
  int dims[NDIM]={SIZE_X,SIZE_Y};
  bool periods[NDIM]={MPI_CART_PERIODIC_TRUE, MPI_CART_PERIODIC_TRUE};
  bool reorder=MPI_REORDER_TRUE;
  int coords[NDIM];
  
  MPI::Request reqs[2*N_NEIGHBORS_2D];
  MPI::Status stats[2*N_NEIGHBORS_2D];

  // MPI initialize
  MPI::Init(argc,argv);
  numTasks  = MPI::COMM_WORLD.Get_size();
  myRank    = MPI::COMM_WORLD.Get_rank();
  MPI::Get_processor_name(processor_name,namelength);
  
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
    MPI::Cartcomm cartcomm = 
      MPI::COMM_WORLD.Create_cart(NDIM, dims, periods, reorder);
    
    // get rank inside the tolopogy
    myRank = cartcomm.Get_rank();

    // get 2D coordinates inside topology
    cartcomm.Get_coords(myRank, NDIM, coords);

    // get rank of source (x-1) and destination (x+1) process
    // take care MPI uses column-major order
    cartcomm.Shift( (int) MPI_TOPO_DIR_X, (int) MPI_SHIFT_FORWARD, nbrs[X_MINUS_1], nbrs[X_PLUS_1]);
    // get rank of source (y-1) and destination (y+1) process
    cartcomm.Shift( (int) MPI_TOPO_DIR_Y, (int) MPI_SHIFT_FORWARD, nbrs[Y_MINUS_1], nbrs[Y_PLUS_1]);

    outbuf = myRank;

    // send    my rank to   each of my neighbors
    // receive my rank from each of my neighbors
    // inbuf should contain the rank of all neighbors
    for (i=0; i<N_NEIGHBORS_2D; i++) {
      dest = nbrs[i];
      source = nbrs[i];
      reqs[i               ] = cartcomm.Isend(&outbuf, 1, MPI::INT, dest, tag);
      reqs[i+N_NEIGHBORS_2D] = cartcomm.Irecv(&inbuf[i], 1, MPI::INT, source, tag);
    }
    MPI::Request::Waitall(2*N_NEIGHBORS_2D, reqs, stats);

    printf("rank= %2d coords= %d %d  neighbors(x+,x-,y+,y-) = %2d %2d %2d %2d\n", 
	   myRank,
	   coords[0],coords[1], 
	   nbrs[X_PLUS_1], 
	   nbrs[X_MINUS_1], 
	   nbrs[Y_PLUS_1], 
	   nbrs[Y_MINUS_1]);
    printf("rank= %2d coords= %d %d  inbuf    (x+,x-,y+,y-) = %2d %2d %2d %2d\n", 
	   myRank,
	   coords[0],coords[1],
	   inbuf[X_PLUS_1],
	   inbuf[X_MINUS_1],
	   inbuf[Y_PLUS_1],
	   inbuf[Y_MINUS_1]);
    
    // print topology
    cartcomm.Barrier();
 
    sleep(1);

    if (myRank == 0) {
      printf("Print topology (COLUMN MAJOR-ORDER) for %dx%d 2D grid:\n",SIZE_X,SIZE_Y);
      printf(" rank     i     j\n");
    }
    printf("%5d %5d %5d %10d %10d %10d %10d\n",myRank,
	   coords[0],coords[1],
	   nbrs[X_PLUS_1], nbrs[X_MINUS_1],
	   nbrs[Y_PLUS_1], nbrs[Y_MINUS_1]
	   );
  } else {
    std::cout << "Must specify " << SIZE_2D << " processors. Terminating.\n";
  }

  // MPI finalize 
  MPI::Finalize();

  return EXIT_SUCCESS;

}
