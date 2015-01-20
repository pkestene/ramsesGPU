/**
 * \file testMpiTopology_3D_C_API.cpp
 * \brief Simple example showing how to use MPI virtual topology (3D).
 *
 * adapted from See https://computing.llnl.gov/tutorials/mpi/
 * see also http://scv.bu.edu/~kadin/alliance/virtual_topology/codes/
 *
 * \author P. Kestener
 * \date 30 Sept 2010
 */

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

#define SIZE_X 3
#define SIZE_Y 3
#define SIZE_Z 3
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
enum {
  MPI_REORDER_FALSE = 0,
  MPI_REORDER_TRUE  = 1
};

enum {
  MPI_CART_PERIODIC_FALSE = 0,
  MPI_CART_PERIODIC_TRUE  = 1
};

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

#define NDIM 3

int main(int argc, char* argv[])
{
  int numtasks, rank, source, dest, outbuf, i, tag=1;
  int inbuf[N_NEIGHBORS_3D]={MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,};
  int nbrs[N_NEIGHBORS_3D];
  int dims[NDIM]={SIZE_X,SIZE_Y,SIZE_Z};
  int periods[NDIM]={MPI_CART_PERIODIC_TRUE, MPI_CART_PERIODIC_TRUE, MPI_CART_PERIODIC_TRUE};
  int reorder=MPI_REORDER_TRUE;
  int coords[NDIM];
  
  MPI_Request reqs[2*N_NEIGHBORS_3D];
  MPI_Status stats[2*N_NEIGHBORS_3D];
  MPI_Comm cartcomm;
  
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("Take care that MPI Cartesian Topology uses COLUMN MAJOR-FORMAT !!!\n");
    printf("\n");
    printf("In this test, each MPI process of the cartesian grid sends a message\n");
    printf("containing a integer (rank of the current process) to all of its\n");
    printf("neighbors. So you must chech that arrays \"neighbors\" and \"inbuf\"\n");
    printf("contain the same information !\n");
    printf("\n");
  }

  // 3D CARTESIAN MPI MESH
  if (numtasks == SIZE_3D) {
  
    // create the cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, NDIM, dims, periods, reorder, &cartcomm);

    // get rank inside the tolopogy
    MPI_Comm_rank(cartcomm, &rank);

    // get 3D coordinates inside topology
    MPI_Cart_coords(cartcomm, rank, NDIM, coords);

    // get rank of source (x-1) and destination (x+1) process
    // take care MPI uses column-major order
    MPI_Cart_shift(cartcomm, MPI_TOPO_DIR_X, MPI_SHIFT_FORWARD, &nbrs[X_MINUS_1], &nbrs[X_PLUS_1]);
    // get rank of source (y-1) and destination (y+1) process
    MPI_Cart_shift(cartcomm, MPI_TOPO_DIR_Y, MPI_SHIFT_FORWARD, &nbrs[Y_MINUS_1], &nbrs[Y_PLUS_1]);
    // get rank of source (z-1) and destination (z+1) process
    MPI_Cart_shift(cartcomm, MPI_TOPO_DIR_Z, MPI_SHIFT_FORWARD, &nbrs[Z_MINUS_1], &nbrs[Z_PLUS_1]);

    outbuf = rank;
 
    // send    my rank to   each of my neighbors
    // receive my rank from each of my neighbors
    // inbuf should contain the rank of all neighbors
    for (i=0; i<N_NEIGHBORS_3D; i++) {
      dest = nbrs[i];
      source = nbrs[i];
      MPI_Isend(&outbuf, 1, MPI_INT, dest, tag, 
		MPI_COMM_WORLD, &reqs[i]);
      MPI_Irecv(&inbuf[i], 1, MPI_INT, source, tag, 
		MPI_COMM_WORLD, &reqs[i+N_NEIGHBORS_3D]);
    }

    MPI_Waitall(2*N_NEIGHBORS_3D, reqs, stats);
   
    printf("rank= %2d coords= %d %d %d neighbors(x+,x-,y+,y-,z+,z-) = %2d %2d %2d %2d %2d %2d\n", 
	   rank,
	   coords[0],coords[1],coords[2],
	   nbrs[X_PLUS_1], 
	   nbrs[X_MINUS_1], 
	   nbrs[Y_PLUS_1], 
	   nbrs[Y_MINUS_1], 
	   nbrs[Z_PLUS_1], 
	   nbrs[Z_MINUS_1]);
    printf("rank= %2d coords= %d %d %d inbuf    (x+,x-,y+,y-,z+,z-) = %2d %2d %2d %2d %2d %2d\n", 
	   rank,
	   coords[0],coords[1],coords[2],
	   inbuf[X_PLUS_1],
	   inbuf[X_MINUS_1],
	   inbuf[Y_PLUS_1],
	   inbuf[Y_MINUS_1],
	   inbuf[Z_PLUS_1],
	   inbuf[Z_MINUS_1]);

    // print topology
    MPI_Barrier(MPI_COMM_WORLD);
    sleep(1);

    if (rank == 0) {
      printf("Print topology (COLUMN MAJOR-ORDER) for a %dx%dx%d 3D grid:\n",SIZE_X,SIZE_Y,SIZE_Z);
      printf(" rank     i     j     k  rank(i+1)  rank(i-1)  rank(j+1)  rank(j-1)  rank(k+1)  rank(k-1)\n");
    }
    printf("%5d %5d %5d %5d %10d %10d %10d %10d %10d %10d\n",rank,
	   coords[0],coords[1],coords[2],
	   nbrs[X_PLUS_1], nbrs[X_MINUS_1],
	   nbrs[Y_PLUS_1], nbrs[Y_MINUS_1],
	   nbrs[Z_PLUS_1], nbrs[Z_MINUS_1]
	   );
    
  } else {
    printf("Must specify %d processors. Terminating.\n",SIZE_3D);
  }

  MPI_Finalize();
  return 0;
}
