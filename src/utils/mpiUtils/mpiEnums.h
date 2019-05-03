#ifndef MPI_ENUMS_H_
#define MPI_ENUMS_H_

namespace hydroSimu {

  //! defgroup mpi_cartesian
  //!@{
  //! number of dimensions of the cartesian virtual topology
  enum {
    NDIM_2D = 2,
    NDIM_3D = 3
  };

  //! do we allow processor reordering by the MPI cartesian communicator ?
  enum {
    MPI_REORDER_FALSE = 0,
    MPI_REORDER_TRUE  = 1
  };
  
  //! should the cartesian virtual topology be considered periodic ?
  enum {
    MPI_CART_PERIODIC_FALSE = 0,
    MPI_CART_PERIODIC_TRUE  = 1
  };
  
  //! MPI topology directions
  enum {
    MPI_TOPO_DIR_X = 0,
    MPI_TOPO_DIR_Y = 1,
    MPI_TOPO_DIR_Z = 2
  };

  //! MPI topology shift direction
  enum {
    MPI_SHIFT_NONE = 0,
    MPI_SHIFT_FORWARD = 1
  };

  //! identifying neighbors
  enum NeighborLocation {
    X_MIN = 0,
    X_MAX = 1,
    Y_MIN = 2,
    Y_MAX = 3,
    Z_MIN = 4,
    Z_MAX = 5
  };

  //! number of neighbors
  enum {
    N_NEIGHBORS_2D = 4,
    N_NEIGHBORS_3D = 6
  };

  // //! direction
  // enum Dir {
  //   DIR_X=0, 
  //   DIR_Y=1, 
  //   DIR_Z=2
  // };

  //!@}

} // namespace hydroSimu

#endif // MPI_ENUMS_H_
