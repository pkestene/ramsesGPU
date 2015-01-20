/**
 * \file MpiCommCart.cpp
 * \brief Implements class MpiCommCart
 *
 * \date 5 Oct 2010
 * \author Pierre Kestener
 *
 * $Id: MpiCommCart.cpp 1783 2012-02-21 10:20:07Z pkestene $
 */

#include "MpiCommCart.h"

namespace hydroSimu {

  // =======================================================
  // =======================================================
  MpiCommCart::MpiCommCart(int mx, int my, int isPeriodic, int allowReorder)
    : MpiComm(), mx_(mx), my_(my), mz_(0), myCoords_(new int[NDIM_2D]), is2D(true)
  {
    
    int dims[NDIM_2D]    = {mx, my};
    int periods[NDIM_2D] = {isPeriodic, isPeriodic};

    // create virtual topology cartesian 2D
    errCheck( MPI_Cart_create(MPI_COMM_WORLD, NDIM_2D, dims, periods, allowReorder, &comm_), "MPI_Cart_create" );;

    // fill nProc_ and myRank_
    init();

    // get cartesian coordinates (myCoords_) of current process (myRank_)
    getCoords(myRank_, NDIM_2D, myCoords_);
  }
  
  // =======================================================
  // =======================================================
  MpiCommCart::MpiCommCart(int mx, int my, int mz, int isPeriodic, int allowReorder)
    : MpiComm(), mx_(mx), my_(my), mz_(mz), myCoords_(new int[NDIM_3D]), is2D(false)
  {
    int dims[NDIM_3D]    = {mx, my, mz};
    int periods[NDIM_3D] = {isPeriodic, isPeriodic, isPeriodic};

    // create virtual topology cartesian 3D
    errCheck( MPI_Cart_create(MPI_COMM_WORLD, NDIM_3D, dims, periods, allowReorder, &comm_), "MPI_Cart_create" );

    // fill nProc_ and myRank_
    init();

    // get cartesian coordinates (myCoords_) of current process (myRank_)
    getCoords(myRank_, NDIM_3D, myCoords_);
  }
  
  // =======================================================
  // =======================================================
  MpiCommCart::~MpiCommCart()
  {
    delete [] myCoords_;
  }

} // namespace hydroSimu
