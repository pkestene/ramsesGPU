/**
 * \file GlobalMpiSession.h
 * \brief A MPI utilities class, providing methods for initializing,
 *        finalizing, and querying the global MPI session
 *
 * This is slightly adapted from the Trilinos package named Teuchos :
 * http://trilinos.sandia.gov/packages/teuchos/
 * 
 * List of changes:
 * - clean TEUCHOS_LIB_DLL_EXPORT macro which does not seem to exist anymore ?
 * - change namespace into hydroSimu
 *
 * I found this class usefull, considering official MPI C++ API is
 * deprecated, and also to avoid using boost::mpi
 *
 * \author Pierre Kestener
 * \date 1 Oct 2010
 *
 * $Id: GlobalMpiSession.h 1783 2012-02-21 10:20:07Z pkestene $
 *
 * See copyright of the original version.
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


#ifndef GLOBAL_MPI_SESSION_H_
#define GLOBAL_MPI_SESSION_H_

#include "common_config.h"

namespace hydroSimu {

/** \brief This class provides methods for initializing, finalizing, and
 * querying the global MPI session.
 *
 * This class is primarilly designed to insulate basic <tt>main()</tt>
 * program type of code from having to know if MPI is enabled or not.
 *
 * ToDo: Give examples!
 */
class GlobalMpiSession
{
public:
  
  //! @name Public constructor and destructor 
  //@{
  
  /** \brief Calls <tt>MPI_Init()</tt> if MPI is enabled.
   *
   * \param argc  [in] Argment passed into <tt>main(argc,argv)</tt>
   * \param argv  [in] Argment passed into <tt>main(argc,argv)</tt>
   * \param out   [in] If <tt>out!=NULL</tt>, then a small message on each
   *              processor will be printed to this stream.  The default is <tt>&std::cout</tt>.
   *
   * <b>Warning!</b> This constructor can only be called once per
   * executable or an error is printed to <tt>*out</tt> and an std::exception will
   * be thrown!
   */
  GlobalMpiSession( int* argc, char*** argv, std::ostream *out = &std::cout );
  
  /** \brief Calls <tt>MPI_Finalize()</tt> if MPI is enabled.
   */
  ~GlobalMpiSession();
    
  //@}
    
  //! @name Static functions 
  //@{

  /** \brief Return wether MPI was initialized. */
  static bool mpiIsInitialized();

  /** \brief Return wether MPI was already finalized. */
  static bool mpiIsFinalized();
  
  /** \brief Returns the process rank relative to <tt>MPI_COMM_WORLD</tt>
   *
   * Returns <tt>0</tt> if MPI is not enabled.
   *
   * Note, this function can be called even if the above constructor was never
   * called so it is safe to use no matter how <tt>MPI_Init()</tt> got called
   * (but it must have been called somewhere).
   */
  static int getRank();

  /** \brief Returns the number of processors relative to
   * <tt>MPI_COMM_WORLD</tt>
   *
   * Returns <tt>1</tt> if MPI is not enabled.
   *
   * Note, this function can be called even if the above constructor was never
   * called so it is safe to use no matter how <tt>MPI_Init()</tt> got called
   * (but it must have been called somewhere).
   */
  static int getNProc();

  //@}
  
private:
  
  static bool haveMPIState_;
  static bool mpiIsFinalized_;
  static int rank_;
  static int nProc_;

  static void initialize( std::ostream *out );

  static void justInTimeInitialize();

}; // class GlobalMpiSession

} // namespace hydroSimu

#endif // GLOBAL_MPI_SESSION_H_
