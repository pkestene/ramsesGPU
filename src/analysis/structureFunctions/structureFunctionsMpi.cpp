/**
 * \file structureFunctionsMPI.cpp
 *
 * A simple standalone program to structure functions ( longitudinal and transversal)
 * of a large data from turbulence simulation run. Data are stored in a netcdf file
 * (use Parallel-NetCDF to load data).
 * There are 2 possibilities:
 * - use a 2D (X,Y) domain decomposition, read 3D sub-domain along
 * Z-axis
 * - use a 3D domain decomposition that exactly math the 3D global domain
 *
 * Algorithm:
 * Since for very large data (typically 2000^3), we cannot use all
 * possible pair of points, we adopt the following approach:
 *
 * 1. read data, and sub-sample sub-domain and select random reference
 * points (store global coordinates x,y,z,rho,vx,vy,vz)
 *
 * 2. MPI communication, so that all MPI proc have all sub-sample
 * reference point data
 *
 * 3. re-read data and compute partial structure functions between all
 * reference points and some local data.
 * 
 * 4. MPI comm to collect all partial structure functions.
 *
 * 5. Output structure functions in npy file format (to ease reload by python)
 *
 * \author P. Kestener
 * \date 27 August 2013
 *
 * $Id: structureFunctionsMpi.cpp 3463 2014-06-27 13:26:12Z pkestene $
 */

#include <math.h>
#include <iostream>
#include <fstream>
#include <limits>     // numeric limits

#include "hydro/GetPot.h"
#include "utils/config/ConfigMap.h"
#include "utils/cnpy/cnpy.h"

#include "hydro/Arrays.h"
using hydroSimu::HostArray;

#include "hydro/constants.h"

#include "pnetcdf_io.h"
#include "common_sf.h"

void init_array(HostArray<double> &data, double val);
void print_help(int argc, char* argv[]);

/** ------------------------- randnum --------------------------------
 ** returns a random number in [0;1]
 ** ------------------------------------------------------------------ */
inline double randnum(void)
{
  return static_cast<double>(rand())/static_cast<double>(RAND_MAX);
}

/** ------------------------- randnum_int -----------------------------
 ** returns a random number in [0, N [
 ** assumes N < RAND_MAX (= 2147483647 = 2^31-1)
 ** ------------------------------------------------------------------ */
inline int randnum_int(int N)
{
  return rand() % N;
}

/* ####################################### */
/* ####################################### */
/* ####################################### */
int main(int argc, char **argv){

#ifndef USE_PNETCDF
  std::cout << "Parallel-NetCDF is not available; please enable to build this application\n";
  return 0;

#else
 
  /* parse command line arguments */
  GetPot cl(argc, argv);

  /* search for multiple options with the same meaning HELP */
  if( cl.search(3, "--help", "-h", "--sos") ) {
    print_help(argc,argv);
    exit(0);
  }

  /* set default configuration parameter fileName */
  const std::string default_param_file = "strucFunc.ini";
  const std::string param_file = cl.follow(default_param_file.c_str(),    "--param");

  /* parse parameters from input file */
  ConfigMap configMap(param_file);
  
  const std::string input_file    = cl.follow("data.nc", "--in");
  const std::string output_prefix = cl.follow("sf",      "--out");

  /* ******************************************* */
  int myRank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // check input file 
  if (input_file.size() == 0) {
    std::cout << "Wrong input.\n";
  } else {
    if (myRank==0) std::cout << "input file used : " << input_file << std::endl;
  }

  /* 
   * Sanity check
   */
  // read mpi geometry
  int mx,my,mz;
  mx=configMap.getInteger("mpi","mx",1);
  my=configMap.getInteger("mpi","my",1);
  mz=configMap.getInteger("mpi","mz",1);

  bool pieceByPieceAlongZ = configMap.getBool("mesh","pieceByPieceAlongZ",false);

  int nbMpiProc;
  MPI_Comm_size(MPI_COMM_WORLD, &nbMpiProc);
  
  if (pieceByPieceAlongZ) {
    if (mx*my != nbMpiProc) {
      if (myRank==0) std::cerr << "Invalid configuration : check parameter file\n";
      if (myRank==0) std::cerr << "mx*my must be equal to the number of MPI proc (2D domain decomp).\n";
      return -1;
    }
    if (myRank==0) std::cout << "Use a 2D domain decomposition with mx=" << mx << ", my=" << my 
			     << "\n";
  } else {
    if (mx*my*mz != nbMpiProc) {
      if (myRank==0) std::cerr << "Invalid configuration : check parameter file\n";
      if (myRank==0) std::cerr << "mx*my*mz must be equal to the number of MPI proc (3D domain decomp).\n";
      return -1;
    }
    if (myRank==0) std::cout << "Use a 3D domain decomposition with mx=" << mx 
			     << ", my=" << my 
			     << ", mz=" << mz << "\n";
  } // end pieceByPieceAlongZ

  /*
   * Read parameter file
   */
  // read local domain sizes
  int nx=configMap.getInteger("mesh","nx",32);
  int ny=configMap.getInteger("mesh","ny",32);
  int nz=configMap.getInteger("mesh","nz",32);

  // global sizes
  int NX=nx*mx, NY=ny*my, NZ=nz*mz;

  int ghostWidth = configMap.getInteger("mesh","ghostWidth",3);
  
  // MPI cartesian coordinates
  // myRank = mpiCoord[0] + mx*mpiCoord[1] + mx*my*mpiCoord[2]
  int mpiCoord[3];
  {
    if (pieceByPieceAlongZ) { // 2D domain decomposition
      mpiCoord[2] = 0;
      mpiCoord[1] = myRank/mx;
      mpiCoord[0] = myRank - mpiCoord[1]*mx;
    } else {
      mpiCoord[2] = myRank/(mx*my);
      mpiCoord[1] = (myRank-mx*my*mpiCoord[2])/mx;
      mpiCoord[0] = myRank - mx*my*mpiCoord[2]-mx*mpiCoord[1];
    }
  }

  /* ******************************************* */
  /*               Initialization                */
  /* ******************************************* */

  // Max SF order
  MaxSFOrder = configMap.getInteger("structureFunctions","max_q",5);

  // number of types of SF
  const int NumberOfTypes = 5;

  // max distance between 2 points (assuming periodic boundary conditions)
  double maxLength = NX/2.0*sqrt(3.0);
  
  const double onethird = 1.0/3.0;

  // number of distance bins
  const int MAX_NUM_BINS = 4096;
  const double numeric_epsilon = 10*std::numeric_limits<double>::epsilon();
  int    numberOfBins = 0;
  double distanceGrid[MAX_NUM_BINS] = {0};
  double length = 1.0;

  // construct bins grid
  //distanceGrid[0] = 0.0;
  while (distanceGrid[numberOfBins] < maxLength) // max distance
    {
      numberOfBins++;
      distanceGrid     [numberOfBins] = length;
      //distanceGrid_stag[numberOfBins] = length + 0.5;
      length = length + 1.0;
      if (numberOfBins > MAX_NUM_BINS) {
	if (myRank==0) std::cout << "ERROR. numberOfBins exceeds MaximumNumberofBins!" << std::endl;
	MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
  numberOfBins++;
  if (myRank==0) std::cout << "Using maxLength   =" << maxLength << "\n";
  if (myRank==0) std::cout << "Using numberOfBins=" << numberOfBins << "\n";

  // main data 
  HostArray<double> data_local; // 4 variables: rho, rho*vx, rho*vy, rho*vz
  data_local.allocate(make_uint4(nx, ny, nz, 4));
  
  // number of reference points
  int nSampleDomain     = configMap.getInteger("structureFunctions","nSampleTotal",1000);
  int nSampleSubDomain   = nSampleDomain/(mx*my*mz);
  int nSampleSubDomainZ  = nSampleDomain/(mx*my); // for all z pieces
  if (myRank==0) {
    std::cout << "Using total number of samples : " << nSampleDomain << std::endl;
    std::cout << "Using number of samples per sub-domain : " << nSampleSubDomain << std::endl;
    if (pieceByPieceAlongZ)
      std::cout << "Using number of samples per sub-domain along Z: " << nSampleSubDomainZ << std::endl;
  }

  // ref points: cartesian coordinates (x,y,z)
  HostArray<int> refPointsCoordSubDomain; 
  HostArray<int> refPointsCoordDomain;

  // ref points: data (rho, rho_vx, rho_vy, rho_vz)
  HostArray<double> refPointsDataSubDomain;
  HostArray<double> refPointsDataDomain;

  if (pieceByPieceAlongZ) {
    // memory allocation must take into account all pieces along Z
    refPointsCoordSubDomain.allocate(make_uint4(nSampleSubDomainZ,1,1,3));
    refPointsDataSubDomain.allocate( make_uint4(nSampleSubDomainZ,1,1,4));
  } else {
    refPointsCoordSubDomain.allocate(make_uint4(nSampleSubDomain,1,1,3));
    refPointsDataSubDomain.allocate( make_uint4(nSampleSubDomain,1,1,4));
  }

  // memory allocation for array containing all reference points
  // every MPI task will have a copy of these 2 arrays
  refPointsCoordDomain.allocate   (make_uint4(nSampleDomain   ,1,1,3));
  refPointsDataDomain.allocate    (make_uint4(nSampleDomain   ,1,1,4));

  // Structure Functions (SF) global counters
  // SF dimensions: type, order, distance bin
  HostArray<long int> sf_bin_counter_long;
  HostArray<long int> sf_bin_counter_tran;
  HostArray<double>   sf_bin_sum_long;
  HostArray<double>   sf_bin_sum_tran;
  HostArray<double>   sf_bin_sumsq_long;
  HostArray<double>   sf_bin_sumsq_tran;

  sf_bin_counter_long.allocate(make_uint4(NumberOfTypes, MaxSFOrder, numberOfBins, 1));
  sf_bin_counter_tran.allocate(make_uint4(NumberOfTypes, MaxSFOrder, numberOfBins, 1));
  sf_bin_sum_long.allocate(make_uint4(NumberOfTypes, MaxSFOrder, numberOfBins, 1));
  sf_bin_sum_tran.allocate(make_uint4(NumberOfTypes, MaxSFOrder, numberOfBins, 1));
  sf_bin_sumsq_long.allocate(make_uint4(NumberOfTypes, MaxSFOrder, numberOfBins, 1));
  sf_bin_sumsq_tran.allocate(make_uint4(NumberOfTypes, MaxSFOrder, numberOfBins, 1));

  sf_bin_counter_long.reset();
  sf_bin_counter_tran.reset();
  sf_bin_sum_long.init(numeric_epsilon);
  sf_bin_sum_tran.init(numeric_epsilon);
  sf_bin_sumsq_long.init(numeric_epsilon);
  sf_bin_sumsq_tran.init(numeric_epsilon);

  // /////////////////////////////////////////////////////////////
  // 1. read sub-domain data and generate reference points
  // 2. MPI comm (MPI_Allgather) to have all ref point everywhere
  // 3. Re-read data and compute partial structure functions between all
  //    reference points and some local data.
  // /////////////////////////////////////////////////////////////
  
  // reading parameter (offset and size)
  MPI_Offset         starts[3] = {0};
  MPI_Offset         counts[3] = {nz, ny, nx};
  int                offset[3] = {0};
  
  // init random generator
  int randomSeed      = configMap.getInteger("structureFunctions","randomSeed",0);
  int randomSeedDelta = configMap.getInteger("structureFunctions","randomSeedDelta",1);
  
  // make random seed different on each MPI proc
  randomSeed = randomSeed + myRank*randomSeedDelta;
  srand(static_cast<unsigned int>(randomSeed));

  if (pieceByPieceAlongZ) { // 2D

    for (int zPiece=0; zPiece<mz; zPiece++) {
      
      // take care that row-major / column major format
      starts[IZ] = ghostWidth+mpiCoord[0]*nx;
      starts[IY] = ghostWidth+mpiCoord[1]*ny;
      starts[IX] = ghostWidth+zPiece     *nz;

      // used to compute global coordinate
      // Please note the revert order from above
      offset[IX] = mpiCoord[0]*nx;
      offset[IY] = mpiCoord[1]*ny;
      offset[IZ] = zPiece     *nz;

      // read data
      if (myRank==0) 
	std::cout << "[Build ref points] read zPiece: " << zPiece << std::endl;
      read_pnetcdf(input_file,starts,counts,data_local);
      
      //std::cout << "[" << myRank << "]" << data_local(0,0,0,IID) << std::endl;
      //if (myRank==3) std::cout << data_local << std::endl;

      // TODO: scale data

      // generate nSampleSubDomain random reference points inside sub-domain
      // for the current zPiece
      for (int iRef = zPiece*nSampleSubDomain; 
	   iRef     < (zPiece+1)*nSampleSubDomain; 
	   iRef++) {

	// local coordinate
	int i = randnum_int(nx); // random integer between 0 and nx-1
	int j = randnum_int(ny); // random integer between 0 and ny-1
	int k = randnum_int(nz); // random integer between 0 and nz-1
	
	// global coordinates
	refPointsCoordSubDomain(iRef+IX*nSampleSubDomainZ) =  i + offset[IX];
	refPointsCoordSubDomain(iRef+IY*nSampleSubDomainZ) =  j + offset[IY];
	refPointsCoordSubDomain(iRef+IZ*nSampleSubDomainZ) =  k + offset[IZ];
	
	double rho = data_local(i,j,k,IID);
	refPointsDataSubDomain(iRef+IID*nSampleSubDomainZ) = rho;
	refPointsDataSubDomain(iRef+IIU*nSampleSubDomainZ) = data_local(i,j,k,IIU)/rho;
	refPointsDataSubDomain(iRef+IIV*nSampleSubDomainZ) = data_local(i,j,k,IIV)/rho;
	refPointsDataSubDomain(iRef+IIW*nSampleSubDomainZ) = data_local(i,j,k,IIW)/rho;

      } // end for iRef

    } // end for zPiece

    // all reference point are OK, now we can
    // distribute them so that every MPI proc has them all

    // unfortunately, every component must be exchanged separately (if not
    // everything is interleaved...)
    MPI_Allgather(&(refPointsCoordSubDomain(0,0,0,IX)), 
		  nSampleSubDomainZ, MPI_INT,
		  &(refPointsCoordDomain(0,0,0,IX))   , 
		  nSampleSubDomainZ, MPI_INT, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsCoordSubDomain(0,0,0,IY)), 
		  nSampleSubDomainZ, MPI_INT,
		  &(refPointsCoordDomain(0,0,0,IY))   , 
		  nSampleSubDomainZ, MPI_INT, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsCoordSubDomain(0,0,0,IZ)), 
		  nSampleSubDomainZ, MPI_INT,
		  &(refPointsCoordDomain(0,0,0,IZ))   , 
		  nSampleSubDomainZ, MPI_INT, 
		  MPI_COMM_WORLD);


    MPI_Allgather(&(refPointsDataSubDomain(0,0,0,IID)), 
		  nSampleSubDomainZ, MPI_DOUBLE,
		  &(refPointsDataDomain(0,0,0,IID))   , 
		  nSampleSubDomainZ, MPI_DOUBLE, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsDataSubDomain(0,0,0,IIU)), 
		  nSampleSubDomainZ, MPI_DOUBLE,
		  &(refPointsDataDomain(0,0,0,IIU))   , 
		  nSampleSubDomainZ, MPI_DOUBLE, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsDataSubDomain(0,0,0,IIV)), 
		  nSampleSubDomainZ, MPI_DOUBLE,
		  &(refPointsDataDomain(0,0,0,IIV))   , 
		  nSampleSubDomainZ, MPI_DOUBLE, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsDataSubDomain(0,0,0,IIW)), 
		  nSampleSubDomainZ, MPI_DOUBLE,
		  &(refPointsDataDomain(0,0,0,IIW))   , 
		  nSampleSubDomainZ, MPI_DOUBLE, 
		  MPI_COMM_WORLD);

    // debug
    /*MPI_Barrier(MPI_COMM_WORLD);
      if (myRank==3) std::cout << refPointsCoordDomain << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      if (myRank==1) std::cout << refPointsDataDomain << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);*/

    if (myRank==0)
      std::cout << "#############\n";

    /*
     * 3. re-read data and compute SF
     */

    int MaxBufIndex = NumberOfTypes * MaxSFOrder * numberOfBins;
    
    // buffer used to store partial results (bin counts, and SF sums)
    {

      // buf1 used for accumulation along zPiece for all reference points
      long    *buf1_bin_counter_long   = new long   [MaxBufIndex];
      long    *buf1_bin_counter_tran   = new long   [MaxBufIndex];
      double  *buf1_binsum_long        = new double [MaxBufIndex];
      double  *buf1_binsum_tran        = new double [MaxBufIndex];
      double  *buf1_binsumsq_long      = new double [MaxBufIndex];
      double  *buf1_binsumsq_tran      = new double [MaxBufIndex];

      // buf2 used for local computations inside a zPiece for a given reference point
      long    *buf2_bin_counter_long   = new long   [MaxBufIndex];
      long    *buf2_bin_counter_tran   = new long   [MaxBufIndex];
      double  *buf2_binsum_long        = new double [MaxBufIndex];
      double  *buf2_binsum_tran        = new double [MaxBufIndex];
      double  *buf2_binsumsq_long      = new double [MaxBufIndex];
      double  *buf2_binsumsq_tran      = new double [MaxBufIndex];

      // initialize buffers
      for (int bufIndex = 0; bufIndex < MaxBufIndex; bufIndex++) {
	buf1_bin_counter_long    [bufIndex] = 0;
	buf1_bin_counter_tran    [bufIndex] = 0;
	buf1_binsum_long         [bufIndex] = numeric_epsilon;
	buf1_binsum_tran         [bufIndex] = numeric_epsilon;
	buf1_binsumsq_long       [bufIndex] = numeric_epsilon;
	buf1_binsumsq_tran       [bufIndex] = numeric_epsilon;
      } // end for bufIndex

      // re-read data
      for (int zPiece=0; zPiece<mz; zPiece++) {
	
	// take care that row-major / column major format
	starts[IZ] = ghostWidth+mpiCoord[0]*nx;
	starts[IY] = ghostWidth+mpiCoord[1]*ny;
	starts[IX] = ghostWidth+zPiece     *nz;

	// used to compute global coordinate
	// Please note the revert order from above
	offset[IX] = mpiCoord[0]*nx;
	offset[IY] = mpiCoord[1]*ny;
	offset[IZ] = zPiece     *nz;
	
	// re-read data
	if (myRank==0) 
	  std::cout << "[Compute SF] re-read zPiece: " << zPiece << std::endl;
	read_pnetcdf(input_file,starts,counts,data_local);
	
	// loop over all reference point
	for (long iRefPoint = 0; 
	     iRefPoint < nSampleDomain; 
	     iRefPoint++) {

	  // clear buf2
	  for (int bufIndex = 0; bufIndex < MaxBufIndex; bufIndex++) {
	    buf2_bin_counter_long [bufIndex] = 0;
	    buf2_bin_counter_tran [bufIndex] = 0;
	    buf2_binsum_long      [bufIndex] = numeric_epsilon;
	    buf2_binsum_tran      [bufIndex] = numeric_epsilon;
	    buf2_binsumsq_long    [bufIndex] = numeric_epsilon;
	    buf2_binsumsq_tran    [bufIndex] = numeric_epsilon;
	  }

	  /// get coordinates of the reference point
	  int i1 = refPointsCoordDomain(iRefPoint,0,0,IX);
	  int j1 = refPointsCoordDomain(iRefPoint,0,0,IY);
	  int k1 = refPointsCoordDomain(iRefPoint,0,0,IZ);

	  // experimental
	  double randnum_i = 0.5 + randnum();
	  double randnum_j = 0.5 + randnum();
	  double randnum_k = 0.5 + randnum();

	  // get ref point data
	  double rho1 = refPointsDataDomain(iRefPoint,0,0,IID);
	  double pow3rho1 = pow(rho1,onethird);
	  //double lnrho1 = log(rho1);
	  
	  double u1 = refPointsDataDomain(iRefPoint,0,0,IIU);
	  double v1 = refPointsDataDomain(iRefPoint,0,0,IIV);
	  double w1 = refPointsDataDomain(iRefPoint,0,0,IIW);

	  double DX[NumberOfTypes] = {0}; 
	  double DY[NumberOfTypes] = {0}; 
	  double DZ[NumberOfTypes] = {0};
	  
	  
	  double incr_long[NumberOfTypes] = {0}; 
	  double incr_tran[NumberOfTypes] = {0};
	  double incr_long_pow[NumberOfTypes] = {0}; 
	  double incr_tran_pow[NumberOfTypes] = {0};

	  // distance
	  double distance = 0.0;
	  long distance_sqr = 0;

	  // distance coordinates
	  int dk = 0; int loopcounter_k = 0;
	  int dj = 0; int loopcounter_j = 0;
	  int di = 0; int loopcounter_i = 0;
	  int factor_k = 0; int factor_j = 0; int factor_i = 0;

	  // second point (global) coordinates
	  int i2 = 0; int j2 = 0; int k2 = 0;

	  // second point (local) coordinates
	  int i2sub = 0; int j2sub = 0; int k2sub = 0;

	  // ?
	  int bin = 0; int bin1 = 0; int bin2 = 0;

	  while ( dk < NZ/2 ) {
	    loopcounter_k++; dj = 0; loopcounter_j = 0;
	    while ( dj < NY/2 ) {
	      loopcounter_j++; di = 0; loopcounter_i = 0;
	      while ( di < NX/2 ) {
		loopcounter_i++;
		
		distance_sqr = di*di + dj*dj + dk*dk;
		if (distance_sqr == 0) {
		  di = loopcounter_i;
		  continue;
		}
		distance = sqrt(static_cast<double>(distance_sqr));
		
		for (factor_k = -1; factor_k <= 1; factor_k += 2)
		  for (factor_j = -1; factor_j <= 1; factor_j += 2)
		    for (factor_i = -1; factor_i <= 1; factor_i += 2) {
		      
		      // Periodic Border Condition
		      i2 = i1 + factor_i*di; if (i2 >= NX) i2 -= NX; else if (i2 < 0) i2 += NX;
		      j2 = j1 + factor_j*dj; if (j2 >= NY) j2 -= NY; else if (j2 < 0) j2 += NY;
		      k2 = k1 + factor_k*dk; if (k2 >= NZ) k2 -= NZ; else if (k2 < 0) k2 += NZ;
		      
		      // recover local coordinate inside sub-domain
		      i2sub = i2 - offset[IX];
		      if ((i2sub < 0) || (i2sub >= nx)) continue;
		      
		      j2sub = j2 - offset[IY];
		      if ((j2sub < 0) || (j2sub >= ny)) continue;
		      
		      k2sub = k2 - offset[IZ];
		      if ((k2sub < 0) || (k2sub >= nz)) continue;
		      
		      // get current point data (at i2sub,j2sub,k2sub)
		      double rho2 = data_local(i2sub,j2sub,k2sub,IID);
		      double pow3rho2 = pow(rho2,onethird);
		      //double lnrho2 = log(rho2);

		      double u2 = data_local(i2sub,j2sub,k2sub,IIU)/rho2;
		      double v2 = data_local(i2sub,j2sub,k2sub,IIV)/rho2;
		      double w2 = data_local(i2sub,j2sub,k2sub,IIW)/rho2;

		      for (int t = 0; t < NumberOfTypes; t++) {
			switch(t) {
			case 0: // v
			  {
			    DX[t] = u2 - u1;
			    DY[t] = v2 - v1;
			    DZ[t] = w2 - w1;
			    break;
			  }
			case 1: // norm(v)
			  {
			    DX[t] = u2 - u1;
			    DY[t] = v2 - v1;
			    DZ[t] = w2 - w1;
			    break;
			  }
			case 2: /// rho^(1/3)*v
			  {
			    DX[t] = pow3rho2*u2 - pow3rho1*u1;
			    DY[t] = pow3rho2*v2 - pow3rho1*v1;
			    DZ[t] = pow3rho2*w2 - pow3rho1*w1;
			    break;
			  }
			case 3: // log(increm(v))
			  {
			    DX[t] = u2 - u1;
			    DY[t] = v2 - v1;
			    DZ[t] = w2 - w1;
			    break;
			  }
			case 4: // log(increm(rho^(1/3)*v))
			  {
			    DX[t] = pow3rho2*u2 - pow3rho1*u1;
			    DY[t] = pow3rho2*v2 - pow3rho1*v1;
			    DZ[t] = pow3rho2*w2 - pow3rho1*w1;
			    break;
			  }
			default:
			  {
			    if (myRank == 0) std::cout << "Compute SF:  something is wrong with the structure function type! ." << std::endl;
			    break;
			  }
			} // end switch t
		  
			// scalar product with normalized distance vector
			incr_long[t] = fabs( di*DX[t] + 
					     dj*DY[t] + 
					     dk*DZ[t] ) / distance;
		  
			// Pythagoras
			incr_tran[t] = sqrt( ( DX[t]*DX[t] + 
					       DY[t]*DY[t] + 
					       DZ[t]*DZ[t] ) - 
					     incr_long[t]*incr_long[t] ); 
		  
			// for scalar field, one of DX, DY, DZ is non-zero
			if (t == 1) {
		    
			  incr_long[t]      = sqrt(DX[t]*DX[t] +
						   DY[t]*DY[t] + 
						   DZ[t]*DZ[t] );
		    
			  // incr_tran has no meaning for scalar value
			  incr_tran[t]      = 0;
			}

			if (t == 3 or t == 4) { // log increment

			  incr_long[t]      = log( sqrt(DX[t]*DX[t] +
							DY[t]*DY[t] + 
							DZ[t]*DZ[t] ) );
		      
			  // incr_tran has no meaning for log(increment)
			  incr_tran[t]      = 0;
		    
			}
		  
			incr_long_pow[t] = incr_long[t];
			incr_tran_pow[t] = incr_tran[t];
		  
		      } // end for t
		
		      // add to the appropriate bin (nested intervals)
		      bin1 = 0; bin2 = numberOfBins; bin = 0;
		      while ((bin2 - bin1) > 1)
			{
			  bin = bin1 + (bin2 - bin1)/2;
			  if (distance < distanceGrid[bin])
			    bin2 = bin;
			  else
			    bin1 = bin;
			}
		      bin = bin1;
		
		      // compute the higher oder structure functions (be cautious with numerics)
		      for (int i = 0; i < MaxSFOrder; i++) 
			for (int t = 0; t < NumberOfTypes; t++)
			  {
		      
			    int bufIndex = bin*NumberOfTypes*MaxSFOrder+i*NumberOfTypes+t;
			    if( fabs(incr_long_pow[t]*incr_long_pow[t]/buf2_binsumsq_long[bufIndex]) > numeric_epsilon)
			      {
				buf2_binsumsq_long   [bufIndex] += incr_long_pow[t]*incr_long_pow[t];
			      }
			    if ( fabs(incr_long_pow[t]/buf2_binsum_long[bufIndex]) > numeric_epsilon)
			      {
				buf2_bin_counter_long[bufIndex]++;
				buf2_binsum_long     [bufIndex] += incr_long_pow[t];
			      }
			    else // immediately reduce to buf1 and clear buf2
			      {
				buf1_bin_counter_long          [bufIndex] += buf2_bin_counter_long[bufIndex];
				buf1_binsum_long               [bufIndex] += buf2_binsum_long     [bufIndex];
				buf1_binsumsq_long             [bufIndex] += buf2_binsumsq_long   [bufIndex];
				//buf1_numeric_error_counter_long[bufIndex]++;
				buf2_bin_counter_long          [bufIndex] = 0;
				buf2_binsum_long               [bufIndex] = numeric_epsilon;
				buf2_binsumsq_long             [bufIndex] = numeric_epsilon;
			      }

			    if( fabs(incr_tran_pow[t]*incr_tran_pow[t]/buf2_binsumsq_tran[bufIndex]) > numeric_epsilon)
			      {
				buf2_binsumsq_tran   [bufIndex] += incr_tran_pow[t]*incr_tran_pow[t];
			      }
			    if ( fabs(incr_tran_pow[t]/buf2_binsum_tran[bufIndex]) > numeric_epsilon)
			      {
				buf2_bin_counter_tran[bufIndex]++;
				buf2_binsum_tran     [bufIndex] += incr_tran_pow[t];
			      }
			    else // immediately reduce to buf1 and clear buf2
			      {
				buf1_bin_counter_tran          [bufIndex] += buf2_bin_counter_tran[bufIndex];
				buf1_binsum_tran               [bufIndex] += buf2_binsum_tran     [bufIndex];
				buf1_binsumsq_tran             [bufIndex] += buf2_binsumsq_tran   [bufIndex];
				//buf1_numeric_error_counter_tran[bufIndex]++;
				buf2_bin_counter_tran          [bufIndex] = 0;
				buf2_binsum_tran               [bufIndex] = numeric_epsilon;
				buf2_binsumsq_tran             [bufIndex] = numeric_epsilon;
			      }
		      
			    incr_long_pow[t] *= incr_long[t];
			    incr_tran_pow[t] *= incr_tran[t];
		      
			  } // end for i,t

		    } // end : for factor

		// increase the distance
		di = loopcounter_i*loopcounter_i/randnum_i;
	      } // end while di

	      // increase the distance
	      dj = loopcounter_j*loopcounter_j/randnum_j; 
	    } // end while dj

	    // increase the distance
	    dk = loopcounter_k*loopcounter_k/randnum_k;
	  } // end while dk
	  
	  // reduce / accumulate buf2 to buf1
	  for (int bufIndex = 0; bufIndex < MaxBufIndex; bufIndex++)
	    {
	      buf1_bin_counter_long[bufIndex] += buf2_bin_counter_long[bufIndex];
	      buf1_binsum_long     [bufIndex] += buf2_binsum_long     [bufIndex];
	      buf1_binsumsq_long   [bufIndex] += buf2_binsumsq_long   [bufIndex];
	      buf1_bin_counter_tran[bufIndex] += buf2_bin_counter_tran[bufIndex];
	      buf1_binsum_tran     [bufIndex] += buf2_binsum_tran     [bufIndex];
	      buf1_binsumsq_tran   [bufIndex] += buf2_binsumsq_tran   [bufIndex];
	    }
	  
	} // end for iRefPoint
	
	if (myRank==0) std::cout << "compute SF for zPiece " << zPiece << " done ..." << std::endl;
	
      } // end for zPiece

      // buf1 can now be reduced over all MPI processes
      MPI_Reduce(buf1_bin_counter_long, sf_bin_counter_long.data(), MaxBufIndex, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(buf1_bin_counter_tran, sf_bin_counter_tran.data(), MaxBufIndex, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

      MPI_Reduce(buf1_binsum_long, sf_bin_sum_long.data(), MaxBufIndex, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(buf1_binsum_tran, sf_bin_sum_tran.data(), MaxBufIndex, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(buf1_binsumsq_long, sf_bin_sumsq_long.data(), MaxBufIndex, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(buf1_binsumsq_tran, sf_bin_sumsq_tran.data(), MaxBufIndex, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      // free memory
      delete[] buf1_bin_counter_long; buf1_bin_counter_long = 0;
      delete[] buf1_bin_counter_tran; buf1_bin_counter_tran = 0;
      delete[] buf1_binsum_long; buf1_binsum_long = 0;
      delete[] buf1_binsum_tran; buf1_binsum_tran = 0;
      delete[] buf1_binsumsq_long; buf1_binsumsq_long = 0;
      delete[] buf1_binsumsq_tran; buf1_binsumsq_tran = 0;
      //delete[] buf1_numeric_error_counter_long; buf1_numeric_error_counter_long = 0;
      //delete[] buf1_numeric_error_counter_tran; buf1_numeric_error_counter_tran = 0;
      delete[] buf2_bin_counter_long; buf2_bin_counter_long = 0;
      delete[] buf2_bin_counter_tran; buf2_bin_counter_tran = 0;
      delete[] buf2_binsum_long; buf2_binsum_long = 0;
      delete[] buf2_binsum_tran; buf2_binsum_tran = 0;
      delete[] buf2_binsumsq_long; buf2_binsumsq_long = 0;
      delete[] buf2_binsumsq_tran; buf2_binsumsq_tran = 0;

    } // end SF computation

  } else { // 3D domain decomposition

    // take care that row-major / column major format
    starts[IZ] = ghostWidth+mpiCoord[0]*nx;
    starts[IY] = ghostWidth+mpiCoord[1]*ny;
    starts[IX] = ghostWidth+mpiCoord[2]*nz;
    
    // used to compute global coordinate
    // Please note the revert order from above
    offset[IX] = mpiCoord[0]*nx;
    offset[IY] = mpiCoord[1]*ny;
    offset[IZ] = mpiCoord[2]*nz;

    // read local sub-domain data
    read_pnetcdf(input_file,starts,counts,data_local);

    // generate nSampleSubDomain random reference points inside sub-domain
    for (int iRef = 0;
	 iRef     < nSampleSubDomain; 
	 iRef++) {

      // local coordinate
      int i = randnum_int(nx); // random integer between 0 and nx-1
      int j = randnum_int(ny); // random integer between 0 and ny-1
      int k = randnum_int(nz); // random integer between 0 and nz-1
      
      // global coordinates
      refPointsCoordSubDomain(iRef+IX*nSampleSubDomain) =  i + offset[IX];
      refPointsCoordSubDomain(iRef+IY*nSampleSubDomain) =  j + offset[IY];
      refPointsCoordSubDomain(iRef+IZ*nSampleSubDomain) =  k + offset[IZ];
      
      double rho = data_local(i,j,k,IID);
      refPointsDataSubDomain(iRef+IID*nSampleSubDomain) = rho;
      refPointsDataSubDomain(iRef+IIU*nSampleSubDomain) = data_local(i,j,k,IIU)/rho;
      refPointsDataSubDomain(iRef+IIV*nSampleSubDomain) = data_local(i,j,k,IIV)/rho;
      refPointsDataSubDomain(iRef+IIW*nSampleSubDomain) = data_local(i,j,k,IIW)/rho;


    } // end for iRef

    // all reference point are OK, now we can
    // distribute them so that every MPI proc has them all
    
    // unfortunately, every component must be exchanged separately (if not
    // everything is interleaved...)
    MPI_Allgather(&(refPointsCoordSubDomain(0,0,0,IX)), 
		  nSampleSubDomain, MPI_INT,
		  &(refPointsCoordDomain(0,0,0,IX))   , 
		  nSampleSubDomain, MPI_INT, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsCoordSubDomain(0,0,0,IY)), 
		  nSampleSubDomain, MPI_INT,
		  &(refPointsCoordDomain(0,0,0,IY))   , 
		  nSampleSubDomain, MPI_INT, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsCoordSubDomain(0,0,0,IZ)), 
		  nSampleSubDomain, MPI_INT,
		  &(refPointsCoordDomain(0,0,0,IZ))   , 
		  nSampleSubDomain, MPI_INT, 
		  MPI_COMM_WORLD);
    
    
    MPI_Allgather(&(refPointsDataSubDomain(0,0,0,IID)), 
		  nSampleSubDomain, MPI_DOUBLE,
		  &(refPointsDataDomain(0,0,0,IID))   , 
		  nSampleSubDomain, MPI_DOUBLE, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsDataSubDomain(0,0,0,IIU)), 
		  nSampleSubDomain, MPI_DOUBLE,
		  &(refPointsDataDomain(0,0,0,IIU))   , 
		  nSampleSubDomain, MPI_DOUBLE, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsDataSubDomain(0,0,0,IIV)), 
		  nSampleSubDomain, MPI_DOUBLE,
		  &(refPointsDataDomain(0,0,0,IIV))   , 
		  nSampleSubDomain, MPI_DOUBLE, 
		  MPI_COMM_WORLD);
    MPI_Allgather(&(refPointsDataSubDomain(0,0,0,IIW)), 
		  nSampleSubDomain, MPI_DOUBLE,
		  &(refPointsDataDomain(0,0,0,IIW))   , 
		  nSampleSubDomain, MPI_DOUBLE, 
		  MPI_COMM_WORLD);
    
    /*
     * 3. re-read data and compute SF
     */
    
    int MaxBufIndex = NumberOfTypes * MaxSFOrder * numberOfBins;
    
    // buffer used to store partial results (bin counts, and SF sums)
    { // start SF computations

      // buf1 used for accumulation along zPiece for all reference points
      long    *buf1_bin_counter_long   = new long   [MaxBufIndex];
      long    *buf1_bin_counter_tran   = new long   [MaxBufIndex];
      double  *buf1_binsum_long        = new double [MaxBufIndex];
      double  *buf1_binsum_tran        = new double [MaxBufIndex];
      double  *buf1_binsumsq_long      = new double [MaxBufIndex];
      double  *buf1_binsumsq_tran      = new double [MaxBufIndex];

      // buf2 used for local computations inside a zPiece for a given reference point
      long    *buf2_bin_counter_long   = new long   [MaxBufIndex];
      long    *buf2_bin_counter_tran   = new long   [MaxBufIndex];
      double  *buf2_binsum_long        = new double [MaxBufIndex];
      double  *buf2_binsum_tran        = new double [MaxBufIndex];
      double  *buf2_binsumsq_long      = new double [MaxBufIndex];
      double  *buf2_binsumsq_tran      = new double [MaxBufIndex];

      // initialize buffers
      for (int bufIndex = 0; bufIndex < MaxBufIndex; bufIndex++) {
	buf1_bin_counter_long    [bufIndex] = 0;
	buf1_bin_counter_tran    [bufIndex] = 0;
	buf1_binsum_long         [bufIndex] = numeric_epsilon;
	buf1_binsum_tran         [bufIndex] = numeric_epsilon;
	buf1_binsumsq_long       [bufIndex] = numeric_epsilon;
	buf1_binsumsq_tran       [bufIndex] = numeric_epsilon;
      } // end for bufIndex
      
      // re-read data
      // take care that row-major / column major format
      starts[IZ] = ghostWidth+mpiCoord[0]*nx;
      starts[IY] = ghostWidth+mpiCoord[1]*ny;
      starts[IX] = ghostWidth+mpiCoord[2]*nz;
      
      // used to compute global coordinate
      // Please note the revert order from above
      offset[IX] = mpiCoord[0]*nx;
      offset[IY] = mpiCoord[1]*ny;
      offset[IZ] = mpiCoord[2]*nz;
      
      // re-read data
      read_pnetcdf(input_file,starts,counts,data_local);
      
      // loop over all reference point
      for (long iRefPoint = 0; 
	   iRefPoint < nSampleDomain; 
	   iRefPoint++) {
	
	// clear buf2
	for (int bufIndex = 0; bufIndex < MaxBufIndex; bufIndex++) {
	  buf2_bin_counter_long [bufIndex] = 0;
	  buf2_bin_counter_tran [bufIndex] = 0;
	  buf2_binsum_long      [bufIndex] = numeric_epsilon;
	  buf2_binsum_tran      [bufIndex] = numeric_epsilon;
	  buf2_binsumsq_long    [bufIndex] = numeric_epsilon;
	  buf2_binsumsq_tran    [bufIndex] = numeric_epsilon;
	}
	
	// get coordinates of the reference point
	int i1 = refPointsCoordDomain(iRefPoint,0,0,IX);
	int j1 = refPointsCoordDomain(iRefPoint,0,0,IY);
	int k1 = refPointsCoordDomain(iRefPoint,0,0,IZ);
	
	// experimental
	double randnum_i = 0.5 + randnum();
	double randnum_j = 0.5 + randnum();
	double randnum_k = 0.5 + randnum();
	
	// get ref point data
	double rho1 = refPointsDataDomain(iRefPoint,0,0,IID);
	double pow3rho1 = pow(rho1,onethird);
	//double lnrho1 = log(rho1);
	
	double u1 = refPointsDataDomain(iRefPoint,0,0,IIU);
	double v1 = refPointsDataDomain(iRefPoint,0,0,IIV);
	double w1 = refPointsDataDomain(iRefPoint,0,0,IIW);
	
	double DX[NumberOfTypes] = {0}; 
	double DY[NumberOfTypes] = {0}; 
	double DZ[NumberOfTypes] = {0};
	
	
	double incr_long[NumberOfTypes] = {0}; 
	double incr_tran[NumberOfTypes] = {0};
	double incr_long_pow[NumberOfTypes] = {0}; 
	double incr_tran_pow[NumberOfTypes] = {0};
	
	// distance
	double distance = 0.0;
	long distance_sqr = 0;
	
	// distance coordinates
	int dk = 0; int loopcounter_k = 0;
	int dj = 0; int loopcounter_j = 0;
	int di = 0; int loopcounter_i = 0;
	int factor_k = 0; 
	int factor_j = 0; 
	int factor_i = 0;
	
	// second point (global) coordinates
	int i2 = 0; 
	int j2 = 0; 
	int k2 = 0;
	
	// second point (local) coordinates
	int i2sub = 0; 
	int j2sub = 0; 
	int k2sub = 0;
	
	// ?
	int bin = 0; int bin1 = 0; int bin2 = 0;
	
	while ( dk < NZ/2 ) {
	  loopcounter_k++; dj = 0; loopcounter_j = 0;
	  while ( dj < NY/2 ) {
	    loopcounter_j++; di = 0; loopcounter_i = 0;
	    while ( di < NX/2 ) {
	      loopcounter_i++;
	      
	      distance_sqr = di*di + dj*dj + dk*dk;
	      if (distance_sqr == 0) {
		di = loopcounter_i;
		continue;
	      }
	      distance = sqrt(static_cast<double>(distance_sqr));
	      
	      for (factor_k = -1; factor_k <= 1; factor_k += 2)
		for (factor_j = -1; factor_j <= 1; factor_j += 2)
		  for (factor_i = -1; factor_i <= 1; factor_i += 2) {
		    
		    // Periodic Border Condition
		    i2 = i1 + factor_i*di; if (i2 >= NX) i2 -= NX; else if (i2 < 0) i2 += NX;
		    j2 = j1 + factor_j*dj; if (j2 >= NY) j2 -= NY; else if (j2 < 0) j2 += NY;
		    k2 = k1 + factor_k*dk; if (k2 >= NZ) k2 -= NZ; else if (k2 < 0) k2 += NZ;
		    
		    // recover local coordinate inside sub-domain
		    i2sub = i2 - offset[IX];
		    if ((i2sub < 0) || (i2sub >= nx)) continue;
		    
		    j2sub = j2 - offset[IY];
		    if ((j2sub < 0) || (j2sub >= ny)) continue;
		    
		    k2sub = k2 - offset[IZ];
		    if ((k2sub < 0) || (k2sub >= nz)) continue;
		    
		    // get current point data (at i2sub,j2sub,k2sub)
		    double rho2 = data_local(i2sub,j2sub,k2sub,IID);
		    double pow3rho2 = pow(rho2,onethird);
		    //double lnrho2 = log(rho2);
		    
		    double u2 = data_local(i2sub,j2sub,k2sub,IIU)/rho2;
		    double v2 = data_local(i2sub,j2sub,k2sub,IIV)/rho2;
		    double w2 = data_local(i2sub,j2sub,k2sub,IIW)/rho2;
		    
		    for (int t = 0; t < NumberOfTypes; t++) {
		      switch(t) {
		      case 0: // v
			{
			  DX[t] = u2 - u1;
			  DY[t] = v2 - v1;
			  DZ[t] = w2 - w1;
			  break;
			}
		      case 1: // norm(v)
			{
			  DX[t] = u2 - u1;
			  DY[t] = v2 - v1;
			  DZ[t] = w2 - w1;
			  break;
			}
		      case 2: /// rho^(1/3)*v
			{
			  DX[t] = pow3rho2*u2 - pow3rho1*u1;
			  DY[t] = pow3rho2*v2 - pow3rho1*v1;
			  DZ[t] = pow3rho2*w2 - pow3rho1*w1;
			  break;
			}
		      case 3: // log(increm(v))
			{
			  DX[t] = u2 - u1;
			  DY[t] = v2 - v1;
			  DZ[t] = w2 - w1;
			  break;
			}
		      case 4: // log(increm(rho^(1/3)*v))
			{
			  DX[t] = pow3rho2*u2 - pow3rho1*u1;
			  DY[t] = pow3rho2*v2 - pow3rho1*v1;
			  DZ[t] = pow3rho2*w2 - pow3rho1*w1;
			  break;
			}
		      default:
			{
			  if (myRank == 0) std::cout << "Compute SF:  something is wrong with the structure function type! ." << std::endl;
			  break;
			}
		      } // end switch t
		      
		      // scalar product with normalized distance vector
		      incr_long[t] = fabs( di*DX[t] + 
					   dj*DY[t] + 
					   dk*DZ[t] ) / distance;
		      
		      // Pythagoras
		      incr_tran[t] = sqrt( ( DX[t]*DX[t] + 
					     DY[t]*DY[t] + 
					     DZ[t]*DZ[t] ) - 
					   incr_long[t]*incr_long[t] ); 
		      
		      // for scalar field, one of DX, DY, DZ is non-zero
		      if (t == 1) {
			
			incr_long[t]      = sqrt(DX[t]*DX[t] +
						 DY[t]*DY[t] + 
						 DZ[t]*DZ[t] );
			
			// incr_tran has no meaning for scalar value
			incr_tran[t]      = 0;
		      }
		      
		      if (t == 3 or t == 4) { // log increment
			
			incr_long[t]      = log( sqrt(DX[t]*DX[t] +
						      DY[t]*DY[t] + 
						      DZ[t]*DZ[t] ) );
			
			// incr_tran has no meaning for log(increment)
			incr_tran[t]      = 0;
			
		      }
		      
		      incr_long_pow[t] = incr_long[t];
		      incr_tran_pow[t] = incr_tran[t];
		      
		    } // end for t
		    
		    // add to the appropriate bin (nested intervals)
		    bin1 = 0; bin2 = numberOfBins; bin = 0;
		    while ((bin2 - bin1) > 1)
		      {
			bin = bin1 + (bin2 - bin1)/2;
			if (distance < distanceGrid[bin])
			  bin2 = bin;
			else
			  bin1 = bin;
		      }
		    bin = bin1;
		    
		    // compute the higher oder structure functions (be cautious with numerics)
		    for (int i = 0; i < MaxSFOrder; i++) 
		      for (int t = 0; t < NumberOfTypes; t++)
			{
			  
			  int bufIndex = bin*NumberOfTypes*MaxSFOrder+i*NumberOfTypes+t;
			  if( fabs(incr_long_pow[t]*incr_long_pow[t]/buf2_binsumsq_long[bufIndex]) > numeric_epsilon)
			    {
			      buf2_binsumsq_long   [bufIndex] += incr_long_pow[t]*incr_long_pow[t];
			    }
			  if ( fabs(incr_long_pow[t]/buf2_binsum_long[bufIndex]) > numeric_epsilon)
			    {
			      buf2_bin_counter_long[bufIndex]++;
			      buf2_binsum_long     [bufIndex] += incr_long_pow[t];
			    }
			  else // immediately reduce to buf1 and clear buf2
			    {
			      buf1_bin_counter_long          [bufIndex] += buf2_bin_counter_long[bufIndex];
			      buf1_binsum_long               [bufIndex] += buf2_binsum_long     [bufIndex];
			      buf1_binsumsq_long             [bufIndex] += buf2_binsumsq_long   [bufIndex];
			      //buf1_numeric_error_counter_long[bufIndex]++;
			      buf2_bin_counter_long          [bufIndex] = 0;
			      buf2_binsum_long               [bufIndex] = numeric_epsilon;
			      buf2_binsumsq_long             [bufIndex] = numeric_epsilon;
			    }
			  
			  if( fabs(incr_tran_pow[t]*incr_tran_pow[t]/buf2_binsumsq_tran[bufIndex]) > numeric_epsilon)
			    {
			      buf2_binsumsq_tran   [bufIndex] += incr_tran_pow[t]*incr_tran_pow[t];
			    }
			  if ( fabs(incr_tran_pow[t]/buf2_binsum_tran[bufIndex]) > numeric_epsilon)
			    {
				buf2_bin_counter_tran[bufIndex]++;
				buf2_binsum_tran     [bufIndex] += incr_tran_pow[t];
			    }
			  else // immediately reduce to buf1 and clear buf2
			    {
			      buf1_bin_counter_tran          [bufIndex] += buf2_bin_counter_tran[bufIndex];
			      buf1_binsum_tran               [bufIndex] += buf2_binsum_tran     [bufIndex];
			      buf1_binsumsq_tran             [bufIndex] += buf2_binsumsq_tran   [bufIndex];
			      //buf1_numeric_error_counter_tran[bufIndex]++;
			      buf2_bin_counter_tran          [bufIndex] = 0;
			      buf2_binsum_tran               [bufIndex] = numeric_epsilon;
			      buf2_binsumsq_tran             [bufIndex] = numeric_epsilon;
			    }
			  
			  incr_long_pow[t] *= incr_long[t];
			  incr_tran_pow[t] *= incr_tran[t];
			  
			} // end for i,t
		    
		  } // end : for factor
	      
	      // increase the distance
	      di = loopcounter_i*loopcounter_i/randnum_i;
	    } // end while di
	    
	    // increase the distance
	    dj = loopcounter_j*loopcounter_j/randnum_j; 
	  } // end while dj
	  
	  // increase the distance
	  dk = loopcounter_k*loopcounter_k/randnum_k;
	} // end while dk
	
	// reduce / accumulate buf2 to buf1
	for (int bufIndex = 0; bufIndex < MaxBufIndex; bufIndex++)
	  {
	    buf1_bin_counter_long[bufIndex] += buf2_bin_counter_long[bufIndex];
	    buf1_binsum_long     [bufIndex] += buf2_binsum_long     [bufIndex];
	    buf1_binsumsq_long   [bufIndex] += buf2_binsumsq_long   [bufIndex];
	    buf1_bin_counter_tran[bufIndex] += buf2_bin_counter_tran[bufIndex];
	    buf1_binsum_tran     [bufIndex] += buf2_binsum_tran     [bufIndex];
	    buf1_binsumsq_tran   [bufIndex] += buf2_binsumsq_tran   [bufIndex];
	  }
	
      } // end for iRefPoint
      
      // buf1 can now be reduced over all MPI processes
      MPI_Reduce(buf1_bin_counter_long, sf_bin_counter_long.data(), MaxBufIndex, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(buf1_bin_counter_tran, sf_bin_counter_tran.data(), MaxBufIndex, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      
      MPI_Reduce(buf1_binsum_long, sf_bin_sum_long.data(), MaxBufIndex, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(buf1_binsum_tran, sf_bin_sum_tran.data(), MaxBufIndex, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(buf1_binsumsq_long, sf_bin_sumsq_long.data(), MaxBufIndex, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(buf1_binsumsq_tran, sf_bin_sumsq_tran.data(), MaxBufIndex, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      
      // free memory
      delete[] buf1_bin_counter_long; buf1_bin_counter_long = 0;
      delete[] buf1_bin_counter_tran; buf1_bin_counter_tran = 0;
      delete[] buf1_binsum_long; buf1_binsum_long = 0;
      delete[] buf1_binsum_tran; buf1_binsum_tran = 0;
      delete[] buf1_binsumsq_long; buf1_binsumsq_long = 0;
      delete[] buf1_binsumsq_tran; buf1_binsumsq_tran = 0;
      //delete[] buf1_numeric_error_counter_long; buf1_numeric_error_counter_long = 0;
      //delete[] buf1_numeric_error_counter_tran; buf1_numeric_error_counter_tran = 0;
      delete[] buf2_bin_counter_long; buf2_bin_counter_long = 0;
      delete[] buf2_bin_counter_tran; buf2_bin_counter_tran = 0;
      delete[] buf2_binsum_long; buf2_binsum_long = 0;
      delete[] buf2_binsum_tran; buf2_binsum_tran = 0;
      delete[] buf2_binsumsq_long; buf2_binsumsq_long = 0;
      delete[] buf2_binsumsq_tran; buf2_binsumsq_tran = 0;

    } // end SF computation

  } // end if (pieceByPieceAlongZ)


  // we shall now proceed to saving SF to files !!!!
  if (myRank==0) {
    
    // write distance grid
    {
      std::string output_file=output_prefix+"_distance_grid.npy";
      const unsigned int shape[] = {(unsigned int) numberOfBins};
      cnpy::npy_save(output_file.c_str(),distanceGrid,shape,1,"w");
    }
    
    // write sf longitudinal
    {
      std::string output_file=output_prefix+"_long.npy";
      const unsigned int shape[] = {(unsigned int) numberOfBins, 
				    (unsigned int) MaxSFOrder, 
				    (unsigned int) NumberOfTypes};
      cnpy::npy_save(output_file.c_str(), sf_bin_sum_long.data(),shape,3,"w");
    }
    
    // write sf square longitudinal
    {
      std::string output_file=output_prefix+"_long_sq.npy";
      const unsigned int shape[] = {(unsigned int) numberOfBins, 
				    (unsigned int) MaxSFOrder, 
				    (unsigned int) NumberOfTypes};
      cnpy::npy_save(output_file.c_str(), sf_bin_sumsq_long.data(),shape,3,"w");
    }
    
    // write counter longitudinal
    {
      std::string output_file=output_prefix+"_long_count.npy";
      const unsigned int shape[] = {(unsigned int) numberOfBins, 
				    (unsigned int) MaxSFOrder, 
				    (unsigned int) NumberOfTypes};
      cnpy::npy_save(output_file.c_str(), sf_bin_counter_long.data(),shape,3,"w");
    }
    
    // write sf transversal
    {
      std::string output_file=output_prefix+"_tran.npy";
      const unsigned int shape[] = {(unsigned int) numberOfBins, 
				    (unsigned int) MaxSFOrder, 
				    (unsigned int) NumberOfTypes};
      cnpy::npy_save(output_file.c_str(), sf_bin_sum_tran.data(),shape,3,"w");
    }
    
    // write sf square transversal
    {
      std::string output_file=output_prefix+"_tran_sq.npy";
      const unsigned int shape[] = {(unsigned int) numberOfBins, 
				    (unsigned int) MaxSFOrder, 
				    (unsigned int) NumberOfTypes};
      cnpy::npy_save(output_file.c_str(), sf_bin_sumsq_tran.data(),shape,3,"w");
    }
    
    // write counter transversal
    {
      std::string output_file=output_prefix+"_tran_count.npy";
      const unsigned int shape[] = {(unsigned int) numberOfBins, 
				    (unsigned int) MaxSFOrder, 
				    (unsigned int) NumberOfTypes};
      cnpy::npy_save(output_file.c_str(), sf_bin_counter_tran.data(),shape,3,"w");
    }
  }
  
  MPI_Finalize();

  if (myRank==0) printf("MPI finalized...\n");

  return 0;

#endif // defined(USE_PNETCDF)

} // end main

/*
 *
 */
void init_array(HostArray<double> &data, double val)
{

  for (unsigned int i=0; i<data.size(); i++) {
    
    data(i) = val;

  }

} // end init_array

/*
 *
 */
void print_help(int argc, char* argv[]) {
  
  using std::cerr;
  using std::cout;
  using std::endl;

  cout << endl;
  cout << argv[0] << " computes structure functions" << endl;
  cout << endl; 
  cout << "USAGE:" << endl;
  cout << "--help, -h, --sos" << endl;
  cout << "        get some help about this program." << endl << endl;
  cout << "--param [string]" << endl;
  cout << "        specify parameter file (INI format)" << endl;
  cout << "--in [string (default=./data.nc)]" << endl;
  cout << "        filename: input ParallelNetCDF data" << endl;
  cout << "--out [string (default=sf)]" << endl;
  cout << "        prefix to output file containing structure functions" << endl;
  cout << endl << endl;       

} // print_help
