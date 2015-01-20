/**
 * \file GlobalMpiSession.cpp
 * \brief Implements class GlobalMpiSession.
 * Slightly adapted from the Trilinos package named Teuchos :
 * http://trilinos.sandia.gov/packages/teuchos/
 *
 * - change namespace into hydroSimu
 * - remove macro HAVE_MPI since this file will only be compiled if
 *   MPI wanted
 *
 * \author Pierre Kestener
 * \date 1 Oct 2010
 *
 * $Id: GlobalMpiSession.cpp 1783 2012-02-21 10:20:07Z pkestene $
 */
// ***********************************************************************
// 
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ***********************************************************************

#include "GlobalMpiSession.h"
#include "TestForException.h"

namespace hydroSimu {
  
  bool GlobalMpiSession::haveMPIState_ = false;
  bool GlobalMpiSession::mpiIsFinalized_ = false;
  int GlobalMpiSession::rank_ = 0 ;
  int GlobalMpiSession::nProc_ = 1 ;

  // =======================================================
  // =======================================================
  GlobalMpiSession::GlobalMpiSession( int* argc, char*** argv, std::ostream *out )
  {
    std::ostringstream oss;
    
    // Above is used to create all output before sending to *out to avoid
    // jumbled parallel output between processors
    
    // initialize MPI
    int mpiHasBeenStarted = 0, mpierr = 0;
    MPI_Initialized(&mpiHasBeenStarted);
    TEST_FOR_EXCEPTION_PRINT(
			     mpiHasBeenStarted, std::runtime_error
			     ,"Error, you can only call this constructor once!"
			     ,out
			     );
    
    mpierr = ::MPI_Init (argc, (char ***) argv);
    TEST_FOR_EXCEPTION_PRINT(
			     mpierr != 0, std::runtime_error
			     ,"Error code=" << mpierr << " detected in GlobalMpiSession::GlobalMpiSession(argc,argv)"
			     ,out
			     );
    
    initialize(out); // Get NProc_ and rank_
    
    int nameLen;
    char procName[MPI_MAX_PROCESSOR_NAME];
    mpierr = ::MPI_Get_processor_name(procName,&nameLen);
    TEST_FOR_EXCEPTION_PRINT(
			     mpierr != 0, std::runtime_error
			     ,"Error code=" << mpierr << " detected in MPI_Get_processor_name()"
			     ,out
			     );
    
    oss << "Teuchos::GlobalMpiSession::GlobalMpiSession(): started processor with name "
	<< procName << " and rank " << rank_ << "!" << std::endl;
        
  } // GlobalMpiSession::GlobalMpiSession
  
  // =======================================================
  // =======================================================
  GlobalMpiSession::~GlobalMpiSession()
  {
    haveMPIState_ = false;
    mpiIsFinalized_ = true;
    
    int mpierr = ::MPI_Finalize();
    TEST_FOR_EXCEPTION_PRINT(
			     mpierr != 0, std::runtime_error
			     ,"Error code=" << mpierr << " detected in MPI_Finalize()"
			     ,&std::cerr
			     );

  } // GlobalMpiSession::~GlobalMpiSession
  
  // =======================================================
  // =======================================================
  bool GlobalMpiSession::mpiIsInitialized() {
    if(!haveMPIState_)
      initialize(&std::cerr);
    return haveMPIState_;
  } // GlobalMpiSession::mpiIsInitialized
  
  // =======================================================
  // =======================================================
  bool GlobalMpiSession::mpiIsFinalized()
  {
    return mpiIsFinalized_;
  } // GlobalMpiSession::mpiIsFinalized
  
  // =======================================================
  // =======================================================
  int GlobalMpiSession::getRank()
  {
    if(!haveMPIState_)
      initialize(&std::cerr);
    return rank_;
  } // GlobalMpiSession::getRank
  
  // =======================================================
  // =======================================================
  int GlobalMpiSession::getNProc() {
    if(!haveMPIState_)
      initialize(&std::cerr);
    return nProc_;
  } // GlobalMpiSession::getNProc
  
  // private
  
  // =======================================================
  // =======================================================
  void GlobalMpiSession::initialize( std::ostream *out )
  {
    
    if(mpiIsFinalized_) {
      // MPI has aleady been finalized so we have a serial machine again!
      rank_ = 0;
      nProc_ = 1;
      return;
    }
    
    if(haveMPIState_)
      return; // We already have what we need!
    
    // We don't have the state of MPI so the constructor for this class must not
    // have been called.  However, if MPI has been called in another way we
    // can still get the state of MPI_COMM_WORLD here.
    
    int mpiHasBeenStarted = 0, mpierr = 0;
    MPI_Initialized(&mpiHasBeenStarted);
    
    if(!mpiHasBeenStarted)
      return;  // We have to give up and just leave NProc_ and rank_ at the default values.
    
    // Get the state of MPI
    
    mpierr = ::MPI_Comm_rank( MPI_COMM_WORLD, &rank_ );
    TEST_FOR_EXCEPTION_PRINT(
			     mpierr != 0, std::runtime_error
			     ,"Error code=" << mpierr << " detected in MPI_Comm_rank()"
			     ,out
			     );
    
    mpierr = ::MPI_Comm_size( MPI_COMM_WORLD, &nProc_ );
    TEST_FOR_EXCEPTION_PRINT(
			     mpierr != 0, std::runtime_error
			     ,"Error code=" << mpierr << " detected in MPI_Comm_size()"
			     ,out
			     );
    
    haveMPIState_ = true;
    mpiIsFinalized_ = false;
        
  } // GlobalMpiSession::initialize
  
} // namespace hydroSimu
