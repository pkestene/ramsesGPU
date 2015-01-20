/**
 * \file MpiComm.cpp
 * \brief Implements class MpiComm
 * 
 * Adapted from Teuchos package (Trilinos)
 * See : http://trilinos.sandia.gov/packages/teuchos/
 *
 * Major modification : add virtual topology (cartesian)
 *
 * \date 5 Oct 2010
 * \author Pierre Kestener
 *
 * $Id: MpiComm.cpp 1859 2012-03-20 17:12:12Z pkestene $
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


#include "MpiComm.h"
#include "ErrorPolling.h"

namespace hydroSimu {
  
  const int MpiComm::INT = 1;
  const int MpiComm::FLOAT = 2;
  const int MpiComm::DOUBLE = 3;
  const int MpiComm::CHAR = 4;
  
  const int MpiComm::SUM = 5;
  const int MpiComm::MIN = 6;
  const int MpiComm::MAX = 7;
  const int MpiComm::PROD = 8;

  // =======================================================
  // =======================================================
  MpiComm::MpiComm()
    : comm_(MPI_COMM_WORLD),
      nProc_(0), myRank_(0)
  {
    init();
  }
  
  // =======================================================
  // =======================================================
  MpiComm::MpiComm(MPI_Comm comm)
    : comm_(comm), nProc_(0), myRank_(0)
  {
    init();
  }
  
  // =======================================================
  // =======================================================
  MpiComm::~MpiComm()
  {
  }

  // =======================================================
  // =======================================================
  MpiComm& MpiComm::world()
  {
    static MpiComm w = MpiComm();
    return w;
  }
  
  // =======================================================
  // =======================================================
  MpiComm& MpiComm::self()
  {
    static MpiComm w = MpiComm(MPI_COMM_SELF);
    return w;
  }
  
  // =======================================================
  // =======================================================
  void MpiComm::synchronize() const 
  {

    //mutex_.lock();
    {
      if (mpiIsRunning())
	{
	  /* test whether errors have been detected on another proc before
	   * doing the collective operation. */
	  POLL_FOR_FAILURES(*this);
	  /* if we're to this point, all processors are OK */
        
	  errCheck(::MPI_Barrier(comm_), "Barrier");
	}
    }
    //mutex_.unlock();

  }

  // =======================================================
  // =======================================================
  void MpiComm::allToAll(void* sendBuf, int sendCount, int sendType,
			 void* recvBuf, int recvCount, int recvType) const
  {
    //mutex_.lock();
    {
      MPI_Datatype mpiSendType = getDataType(sendType);
      MPI_Datatype mpiRecvType = getDataType(recvType);


      if (mpiIsRunning())
	{
	  /* test whether errors have been detected on another proc before
	   * doing the collective operation. */
	  POLL_FOR_FAILURES(*this);
	  /* if we're to this point, all processors are OK */
        
	  errCheck(::MPI_Alltoall(sendBuf, sendCount, mpiSendType,
				  recvBuf, recvCount, mpiRecvType,
				  comm_), "Alltoall");
	}
    }
    //mutex_.unlock();
  }

  // =======================================================
  // =======================================================
  void MpiComm::allToAllv(void* sendBuf, int* sendCount, 
			  int* sendDisplacements, int sendType,
			  void* recvBuf, int* recvCount, 
			  int* recvDisplacements, int recvType) const
  {
    //mutex_.lock();
    {
      MPI_Datatype mpiSendType = getDataType(sendType);
      MPI_Datatype mpiRecvType = getDataType(recvType);

      if (mpiIsRunning())
	{
	  /* test whether errors have been detected on another proc before
	   * doing the collective operation. */
	  POLL_FOR_FAILURES(*this);
	  /* if we're to this point, all processors are OK */		
        
	  errCheck(::MPI_Alltoallv(sendBuf, sendCount, sendDisplacements, mpiSendType,
				   recvBuf, recvCount, recvDisplacements, mpiRecvType,
				   comm_), "Alltoallv");
	}
    }
    //mutex_.unlock();
  }

  // =======================================================
  // =======================================================
  void MpiComm::allReduce(void* input, void* result, int inputCount, 
			  int type, int op) const
  {
    //mutex_.lock();
    {
      MPI_Op mpiOp = getOp(op);
      MPI_Datatype mpiType = getDataType(type);
		
      if (mpiIsRunning())
	{
	  errCheck(::MPI_Allreduce(input, result, inputCount, mpiType,
				   mpiOp, comm_), 
		   "Allreduce");
	}
    }
    //mutex_.unlock();
  }

  // =======================================================
  // =======================================================
  void MpiComm::gather(void* sendBuf, int sendCount, int sendType,
		       void* recvBuf, int recvCount, int recvType,
		       int root) const
  {
    //mutex_.lock();
    {
      MPI_Datatype mpiSendType = getDataType(sendType);
      MPI_Datatype mpiRecvType = getDataType(recvType);


      if (mpiIsRunning())
	{
	  /* test whether errors have been detected on another proc before
	   * doing the collective operation. */
	  POLL_FOR_FAILURES(*this);
	  /* if we're to this point, all processors are OK */
        
	  errCheck(::MPI_Gather(sendBuf, sendCount, mpiSendType,
				recvBuf, recvCount, mpiRecvType,
				root, comm_), "Gather");
	}
    }
    //mutex_.unlock();
  }

  // =======================================================
  // =======================================================
  void MpiComm::gatherv(void* sendBuf, int sendCount, int sendType,
			void* recvBuf, int* recvCount, int* displacements, int recvType,
			int root) const
  {
    //mutex_.lock();
    {
      MPI_Datatype mpiSendType = getDataType(sendType);
      MPI_Datatype mpiRecvType = getDataType(recvType);
		
      if (mpiIsRunning())
	{
	  /* test whether errors have been detected on another proc before
	   * doing the collective operation. */
	  POLL_FOR_FAILURES(*this);
	  /* if we're to this point, all processors are OK */
        
	  errCheck(::MPI_Gatherv(sendBuf, sendCount, mpiSendType,
				 recvBuf, recvCount, displacements, mpiRecvType,
				 root, comm_), "Gatherv");
	}
    }
    //mutex_.unlock();
  }

  // =======================================================
  // =======================================================
  void MpiComm::allGather(void* sendBuf, int sendCount, int sendType,
			  void* recvBuf, int recvCount, 
			  int recvType) const
  {
    //mutex_.lock();
    {
      MPI_Datatype mpiSendType = getDataType(sendType);
      MPI_Datatype mpiRecvType = getDataType(recvType);
		
      if (mpiIsRunning())
	{
	  /* test whether errors have been detected on another proc before
	   * doing the collective operation. */
	  POLL_FOR_FAILURES(*this);
	  /* if we're to this point, all processors are OK */
        
	  errCheck(::MPI_Allgather(sendBuf, sendCount, mpiSendType,
				   recvBuf, recvCount, 
				   mpiRecvType, comm_), 
		   "AllGather");
	}
    }
    //mutex_.unlock();
  }


  // =======================================================
  // =======================================================
  void MpiComm::allGatherv(void* sendBuf, int sendCount, int sendType,
			   void* recvBuf, int* recvCount, 
			   int* recvDisplacements,
			   int recvType) const
  {
    //mutex_.lock();
    {
      MPI_Datatype mpiSendType = getDataType(sendType);
      MPI_Datatype mpiRecvType = getDataType(recvType);
    
      if (mpiIsRunning())
	{
	  /* test whether errors have been detected on another proc before
	   * doing the collective operation. */
	  POLL_FOR_FAILURES(*this);
	  /* if we're to this point, all processors are OK */
        
	  errCheck(::MPI_Allgatherv(sendBuf, sendCount, mpiSendType,
				    recvBuf, recvCount, recvDisplacements,
				    mpiRecvType, 
				    comm_), 
		   "AllGatherv");
	}
    }
    //mutex_.unlock();
  }


  // =======================================================
  // =======================================================
  void MpiComm::bcast(void* msg, int length, int type, int src) const
  {
    //mutex_.lock();
    {
      if (mpiIsRunning())
	{
	  /* test whether errors have been detected on another proc before
	   * doing the collective operation. */
	  POLL_FOR_FAILURES(*this);
	  /* if we're to this point, all processors are OK */
        
	  MPI_Datatype mpiType = getDataType(type);
	  errCheck(::MPI_Bcast(msg, length, mpiType, src, 
			       comm_), "Bcast");
	}
    }
    //mutex_.unlock();
  }

  // =======================================================
  // =======================================================
  void MpiComm::send(void *sendBuf, int sendCount, int sendType, 
		int dest, int tag) const 
  {
    MPI_Datatype mpiSendType = getDataType(sendType);
    
    if (mpiIsRunning())
      {
	errCheck(::MPI_Send(sendBuf, sendCount, mpiSendType, dest, tag, comm_), "MPI_Send");
      }
  }

  // =======================================================
  // =======================================================
  void MpiComm::recv(void *recvBuf, int recvCount, int recvType, 
		     int source, int tag) const 
  {
    MPI_Datatype mpiRecvType = getDataType(recvType);
    
    if (mpiIsRunning())
      {
	errCheck(::MPI_Recv(recvBuf, recvCount, mpiRecvType, source, tag, comm_, MPI_STATUS_IGNORE), "MPI_Send");
      }
  }

  // =======================================================
  // =======================================================
  void MpiComm::sendrecv(void *sendBuf, int sendCount, int sendType, 
			 int dest, int sendtag,
			 void *recvBuf, int recvCount, int recvType, 
			 int src, int recvtag) const 
  {
    MPI_Datatype mpiSendType = getDataType(sendType);
    MPI_Datatype mpiRecvType = getDataType(recvType);
    
    if (mpiIsRunning())
      {
	errCheck(::MPI_Sendrecv(sendBuf, sendCount, mpiSendType, dest, sendtag,
				recvBuf, recvCount, mpiRecvType, src, recvtag, 
				comm_, MPI_STATUS_IGNORE), "MPI_Sendrecv");
      }
  }

  // =======================================================
  // =======================================================
  MPI_Request MpiComm::Isend(void *sendBuf, int sendCount, int sendType, 
			     int dest, int tag) const
  {
    MPI_Request request;
    MPI_Datatype mpiSendType = getDataType(sendType);

    if (mpiIsRunning())
      errCheck( ::MPI_Isend(sendBuf, sendCount, mpiSendType, 
			    dest, tag, comm_, &request), "MPI_Isend");
    return request;
  }

  // =======================================================
  // =======================================================
  MPI_Request MpiComm::Ibsend(void *sendBuf, int sendCount, int sendType, 
			      int dest, int tag) const
  {
    MPI_Request request;
    MPI_Datatype mpiSendType = getDataType(sendType);

    if (mpiIsRunning())
      errCheck( ::MPI_Ibsend(sendBuf, sendCount, mpiSendType, 
			    dest, tag, comm_, &request), "MPI_Ibsend");
    return request;
  }

  // =======================================================
  // =======================================================
  MPI_Request MpiComm::Issend(void *sendBuf, int sendCount, int sendType, 
			      int dest, int tag) const
  {
    MPI_Request request;
    MPI_Datatype mpiSendType = getDataType(sendType);

    if (mpiIsRunning())
      errCheck( ::MPI_Issend(sendBuf, sendCount, mpiSendType, 
			     dest, tag, comm_, &request), "MPI_Issend");
    return request;
  }

  // =======================================================
  // =======================================================
  MPI_Request MpiComm::Irsend(void *sendBuf, int sendCount, int sendType, 
			      int dest, int tag) const
  {
    MPI_Request request;
    MPI_Datatype mpiSendType = getDataType(sendType);

    if (mpiIsRunning())
      errCheck( ::MPI_Irsend(sendBuf, sendCount, mpiSendType, 
			     dest, tag, comm_, &request), "MPI_Irsend");
    return request;
  }

  // =======================================================
  // =======================================================
  MPI_Request MpiComm::Irecv(void *recvBuf, int recvCount, int recvType, 
			     int source, int tag) const
  {
    MPI_Request request;
    MPI_Datatype mpiRecvType = getDataType(recvType);

    if (mpiIsRunning())
      errCheck( ::MPI_Irecv(recvBuf, recvCount, mpiRecvType, 
			    source, tag, comm_, &request), "MPI_Irecv");
    return request;
  }

  // =======================================================
  // =======================================================
  void MpiComm::errCheck(int errCode, const std::string& methodName)
  {
    TEST_FOR_EXCEPTION(errCode != 0, std::runtime_error,
		       "MPI function MPI_" << methodName 
		       << " returned error code=" << errCode);
  }

  // =======================================================
  // =======================================================
  MPI_Datatype MpiComm::getDataType(int type)
  {
    TEST_FOR_EXCEPTION(
		       !(type == INT || type==FLOAT 
			 || type==DOUBLE || type==CHAR),
		       std::range_error,
		       "invalid type " << type << " in MpiComm::getDataType");
  
    if(type == INT) return MPI_INT;
    if(type == FLOAT) return MPI_FLOAT;
    if(type == DOUBLE) return MPI_DOUBLE;
  
    return MPI_CHAR;
  }

  // =======================================================
  // =======================================================
  MPI_Op MpiComm::getOp(int op)
  {

    TEST_FOR_EXCEPTION(
		       !(op == SUM || op==MAX 
			 || op==MIN || op==PROD),
		       std::range_error,
		       "invalid operator " 
		       << op << " in MpiComm::getOp");

    if( op == SUM) return MPI_SUM;
    else if( op == MAX) return MPI_MAX;
    else if( op == MIN) return MPI_MIN;
    return MPI_PROD;
  }

  // =======================================================
  // =======================================================
  void MpiComm::init()
  {
    if (mpiIsRunning())
      {
	errCheck(MPI_Comm_rank(comm_, &myRank_), "Comm_rank");
	errCheck(MPI_Comm_size(comm_, &nProc_), "Comm_size");
      }
    else
      {
	nProc_ = 1;
	myRank_ = 0;
      }
  }

  // =======================================================
  // =======================================================
  int MpiComm::mpiIsRunning() const
  {
    int mpiStarted = 0;
    MPI_Initialized(&mpiStarted);
    return mpiStarted;
  }

} // namespace hydroSimu
