/**
 * \file HydroMpiParameters.h
 * \brief Defines a C++ class gathering MPI-related parameters.
 *
 * HydroMpiParameters class inherits HydroParameters which is the serial
 * version (1 CPU/GPU).
 *
 * \author P. Kestener
 * \date 6 Oct 2010
 *
 * $Id: HydroMpiParameters.h 1847 2012-03-19 08:35:51Z pkestene $
 */
#ifndef HYDRO_MPI_PARAMETERS_H_
#define HYDRO_MPI_PARAMETERS_H_

#include "HydroParameters.h"

#include "utils/mpiUtils/MpiCommCart.h"
#include "utils/mpiUtils/TestForException.h"

#include <vector>

namespace hydroSimu {

  /**
   * \class HydroMpiParameters HydroMpiParameters.h
   * \brief This is the base class containing all usefull methods to hydro
   * simulations (handling array initializations, boundary
   * computations, output files) in a pure MPI or MPI/CUDA environnement.
   *
   * All classes effectively implementing hydro simulations should
   * inherit from this base class.
   */
  class HydroMpiParameters : public HydroParameters
  {
  public:
    HydroMpiParameters(ConfigMap &_configMap);
    virtual ~HydroMpiParameters();
    
  protected:
    //! size of the MPI cartesian grid
    int mx,my,mz;
    
    //! MPI communicator in a cartesian virtual topology
    MpiCommCart *communicator;
    
    //! number of dimension
    int nDim;

    //! MPI rank of current process
    int myRank;

    //! number of MPI processes
    int nProcs;

    //! MPI cartesian coordinates inside MPI topology
    std::vector<int> myMpiPos;

    //! number of MPI process neighbors (4 in 2D and 6 in 3D)
    int nNeighbors;
    
    //! MPI rank of adjacent MPI processes
    std::vector<int> neighborsRank;
    
    //! boundary condition type with adjacent domains (corresponding to
    //! neighbor MPI processes)
    std::vector<BoundaryConditionType> neighborsBC;

  };

  //! print information about current Mpi parameters
  // std::ostream& operator<<( std::ostream& s, const HydroMpiParameters& param );

} // namespace hydroSimu

#endif // HYDRO_MPI_PARAMETERS_H_
