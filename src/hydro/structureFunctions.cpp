/**
 * \file structureFunctions.cpp
 *
 * Compute structure functions for mono-CPU or mono-GPU.
 * See src/analysis/structureFunctionsMpi.cpp for algorithm description.
 *
 * \author Pierre Kestener
 * \date 29 June 2014
 *
 * $Id: structureFunctions.cpp 3469 2014-07-02 12:39:45Z pkestene $
 */
#include "structureFunctions.h"

#include "utils/cnpy/cnpy.h"
#include <limits> // for std::numeric_limits

#include <cstdlib>
#include "RandomGen.h" // for old randnum_int
//#include <random> // for c++11 random number generator

namespace hydroSimu {

  // =======================================================
  // =======================================================
  void structure_functions_hydro(int nStep, 
				 ConfigMap &configMap, 
				 GlobalConstants &_gParams,
				 HostArray<real_t> &U)
  {

    // structure function computation
    int maxSFOrder = configMap.getInteger("structureFunctions","max_q",5);

    // number of types of SF
    const int numberOfTypes = 5;
    
    // global sizes
    int &nx = _gParams.nx;
    int &ny = _gParams.ny;
    int &nz = _gParams.nz;

    int &NX = _gParams.nx;
    int &NY = _gParams.ny;
    int &NZ = _gParams.nz;

    // ghostWidth
    int g = 2; // for hydro

    // max distance between 2 points (assuming periodic boundary conditions)
    double maxLength = NX/2.0*sqrt(3.0);

    // some constants
    const double oneThird = 1.0/3.0;
    const double numeric_epsilon = 10*std::numeric_limits<double>::epsilon();

    // number of distance bins
    int    numberOfBins = 0;
    const int MAX_NUM_BINS = 4096;
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
	  std::cerr << "ERROR. numberOfBins exceeds MaximumNumberofBins!" << std::endl;
	}
      }
    numberOfBins++;

    // number of reference points
    int nSampleDomain    = configMap.getInteger("structureFunctions","nSampleTotal",1000);

    // ref points: cartesian coordinates (x,y,z)
    HostArray<int>    refPointsCoordDomain;
    
    // ref points: data (rho, rho_vx, rho_vy, rho_vz)
    HostArray<double> refPointsDataDomain;
    
    // Structure Functions (SF) global counters
    // SF dimensions: type, order, distance bin
    // HostArray<long int> sf_bin_counter_long;
    // HostArray<long int> sf_bin_counter_tran;
    // HostArray<double>   sf_bin_sum_long;
    // HostArray<double>   sf_bin_sum_tran;
    // HostArray<double>   sf_bin_sumsq_long;
    // HostArray<double>   sf_bin_sumsq_tran;
    
    // sf_bin_counter_long.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_counter_tran.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_sum_long.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_sum_tran.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_sumsq_long.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_sumsq_tran.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    
    // sf_bin_counter_long.reset();
    // sf_bin_counter_tran.reset();

    // // sum 
    // sf_bin_sum_long.init(numeric_epsilon);
    // sf_bin_sum_tran.init(numeric_epsilon);

    // // sum of squares
    // sf_bin_sumsq_long.init(numeric_epsilon);
    // sf_bin_sumsq_tran.init(numeric_epsilon);
  
    /*
     * init random generator (old way)
     */
    int randomSeed      = configMap.getInteger("structureFunctions","randomSeed",0);
    srand(static_cast<unsigned int>(randomSeed));

    /*
     * random generator c++11 way
     */
    // int randomSeed      = configMap.getInteger("structureFunctions","randomSeed",0);
    // std::default_random_engine generator(randomSeed);
    // std::uniform_int_distribution<int> unif_dist_x(0,nx-1);
    // std::uniform_int_distribution<int> unif_dist_y(0,ny-1);
    // std::uniform_int_distribution<int> unif_dist_z(0,nz-1);
    // std::uniform_real_distribution<double> unif_dist(0.0,1.0);

    int maxBufIndex = numberOfTypes * maxSFOrder * numberOfBins;

    { // start SF computations

      // buf1 used for accumulation for all reference points
      long    *buf1_bin_counter_long   = new long   [maxBufIndex];
      long    *buf1_bin_counter_tran   = new long   [maxBufIndex];
      double  *buf1_binsum_long        = new double [maxBufIndex];
      double  *buf1_binsum_tran        = new double [maxBufIndex];
      double  *buf1_binsumsq_long      = new double [maxBufIndex];
      double  *buf1_binsumsq_tran      = new double [maxBufIndex];
      
      // buf2 used for local computations for a given reference point
      long    *buf2_bin_counter_long   = new long   [maxBufIndex];
      long    *buf2_bin_counter_tran   = new long   [maxBufIndex];
      double  *buf2_binsum_long        = new double [maxBufIndex];
      double  *buf2_binsum_tran        = new double [maxBufIndex];
      double  *buf2_binsumsq_long      = new double [maxBufIndex];
      double  *buf2_binsumsq_tran      = new double [maxBufIndex];

      // initialize buffers
      for (int bufIndex = 0; bufIndex < maxBufIndex; bufIndex++) {
	buf1_bin_counter_long    [bufIndex] = 0;
	buf1_bin_counter_tran    [bufIndex] = 0;
	buf1_binsum_long         [bufIndex] = numeric_epsilon;
	buf1_binsum_tran         [bufIndex] = numeric_epsilon;
	buf1_binsumsq_long       [bufIndex] = numeric_epsilon;
	buf1_binsumsq_tran       [bufIndex] = numeric_epsilon;
      } // end for bufIndex

      // loop over all reference point
      for (long iRefPoint = 0; 
	   iRefPoint < nSampleDomain; 
	   iRefPoint++) {

	// clear buf2
	for (int bufIndex = 0; bufIndex < maxBufIndex; bufIndex++) {
	  buf2_bin_counter_long [bufIndex] = 0;
	  buf2_bin_counter_tran [bufIndex] = 0;
	  buf2_binsum_long      [bufIndex] = numeric_epsilon;
	  buf2_binsum_tran      [bufIndex] = numeric_epsilon;
	  buf2_binsumsq_long    [bufIndex] = numeric_epsilon;
	  buf2_binsumsq_tran    [bufIndex] = numeric_epsilon;
	}

	// get coordinates of the reference point
	int i1 = randnum_int(NX);
	int j1 = randnum_int(NY);
	int k1 = randnum_int(NZ);
	//int i1 = unif_dist_x(generator); // random integer between 0 and nx-1
	//int j1 = unif_dist_y(generator); // random integer between 0 and ny-1
	//int k1 = unif_dist_z(generator); // random integer between 0 and nz-1

	// experimental
	double randnum_i = 0.5 + randnum();
	double randnum_j = 0.5 + randnum();
	double randnum_k = 0.5 + randnum();
	// double randnum_i = 0.5 + unif_dist(generator);
	// double randnum_j = 0.5 + unif_dist(generator);
	// double randnum_k = 0.5 + unif_dist(generator);
	
	// get ref point data
	double rho1 = U(i1+g, j1+g, k1+g, ID);
	double u1   = U(i1+g, j1+g, k1+g, IU)/rho1;
	double v1   = U(i1+g, j1+g, k1+g, IV)/rho1;
	double w1   = U(i1+g, j1+g, k1+g, IW)/rho1;

	double pow3rho1 = pow(rho1,oneThird);
	//double lnrho1 = log(rho1);
	
	double DX[numberOfTypes] = {0}; 
	double DY[numberOfTypes] = {0}; 
	double DZ[numberOfTypes] = {0};
	
	double incr_long[numberOfTypes] = {0}; 
	double incr_tran[numberOfTypes] = {0};
	double incr_long_pow[numberOfTypes] = {0}; 
	double incr_tran_pow[numberOfTypes] = {0};

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
		    i2sub = i2 /*- offset[IX]*/;
		    if ((i2sub < 0) || (i2sub >= nx)) continue;
		    
		    j2sub = j2 /*- offset[IY]*/;
		    if ((j2sub < 0) || (j2sub >= ny)) continue;
		    
		    k2sub = k2 /*- offset[IZ]*/;
		    if ((k2sub < 0) || (k2sub >= nz)) continue;
		    
		    // get current point data (at i2sub,j2sub,k2sub)
		    double rho2 = U(i2sub+g, j2sub+g, k2sub+g, ID);
		    double u2   = U(i2sub+g, j2sub+g, k2sub+g, IU)/rho2;
		    double v2   = U(i2sub+g, j2sub+g, k2sub+g, IV)/rho2;
		    double w2   = U(i2sub+g, j2sub+g, k2sub+g, IW)/rho2;
		    
		    double pow3rho2 = pow(rho2,oneThird);
		    //double lnrho2 = log(rho2);

		    for (int t = 0; t < numberOfTypes; t++) {
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
			  std::cout << "Compute SF:  something is wrong with the structure function type! ." << std::endl;
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
		    
		    // compute the higher order structure functions 
		    // (be cautious with numerics)
		    for (int i = 0; i < maxSFOrder; i++) 
		      for (int t = 0; t < numberOfTypes; t++)
			{
			  
			  int bufIndex = bin*numberOfTypes*maxSFOrder+i*numberOfTypes+t;
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
	for (int bufIndex = 0; bufIndex < maxBufIndex; bufIndex++)
	  {
	    buf1_bin_counter_long[bufIndex] += buf2_bin_counter_long[bufIndex];
	    buf1_binsum_long     [bufIndex] += buf2_binsum_long     [bufIndex];
	    buf1_binsumsq_long   [bufIndex] += buf2_binsumsq_long   [bufIndex];
	    buf1_bin_counter_tran[bufIndex] += buf2_bin_counter_tran[bufIndex];
	    buf1_binsum_tran     [bufIndex] += buf2_binsum_tran     [bufIndex];
	    buf1_binsumsq_tran   [bufIndex] += buf2_binsumsq_tran   [bufIndex];
	  }

      } // end for iRefPoint

      // free memory
      delete[] buf2_bin_counter_long; buf2_bin_counter_long = 0;
      delete[] buf2_bin_counter_tran; buf2_bin_counter_tran = 0;
      delete[] buf2_binsum_long; buf2_binsum_long = 0;
      delete[] buf2_binsum_tran; buf2_binsum_tran = 0;
      delete[] buf2_binsumsq_long; buf2_binsumsq_long = 0;
      delete[] buf2_binsumsq_tran; buf2_binsumsq_tran = 0;

      // we shall now proceed to saving SF to files !!!!
      std::string outputDir    = configMap.getString("output", "outputDir", "./");
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
      std::ostringstream outNum;
      outNum.width(7);
      outNum.fill('0');
      outNum << nStep;
      
      outputPrefix = outputDir+"/"+outputPrefix+"_"+outNum.str()+"_sf";

      std::string output_file=outputPrefix+".npz";

      // write distance grid
      {
	
	const unsigned int shape[] = {(unsigned int) numberOfBins};
	cnpy::npz_save(output_file.c_str(),"distance",distanceGrid,shape,1,"w");
      }
      
      // write sf longitudinal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "binsum_long", buf1_binsum_long,shape,3,"a");
      }
      
      // write sf square longitudinal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "binsum_sq_long", buf1_binsumsq_long,shape,3,"a");
      }
      
      // write counter longitudinal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "count_long", buf1_bin_counter_long,shape,3,"a");
      }
      
      // write sf transversal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "binsum_tran", buf1_binsum_tran,shape,3,"a");
      }
      
      // write sf square transversal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "binsum_sq_tran", buf1_binsumsq_tran,shape,3,"a");
      }
      
      // write counter transversal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "count_tran", buf1_bin_counter_tran,shape,3,"a");
      }
 
      // write some comments
      {
	std::string comments = "";

	comments += "About SF types\n";
	comments += "Type 0: velocity increments\n";
	comments += "Type 1: norm(velocity increments)\n";
	comments += "Type 2: rho^(1/3)*v increments\n";
	comments += "Type 3: log(delta(v))\n";
	comments += "Type 4: log(delta(rho^(1/3)*v))\n";

	unsigned int shape[] = { (unsigned int) comments.size() };

	cnpy::npz_save(output_file.c_str(), "comments", comments.c_str(),shape,1,"a");
      }
     
      delete[] buf1_bin_counter_long; buf1_bin_counter_long = 0;
      delete[] buf1_bin_counter_tran; buf1_bin_counter_tran = 0;
      delete[] buf1_binsum_long; buf1_binsum_long = 0;
      delete[] buf1_binsum_tran; buf1_binsum_tran = 0;
      delete[] buf1_binsumsq_long; buf1_binsumsq_long = 0;
      delete[] buf1_binsumsq_tran; buf1_binsumsq_tran = 0;
      //delete[] buf1_numeric_error_counter_long; buf1_numeric_error_counter_long = 0;
      //delete[] buf1_numeric_error_counter_tran; buf1_numeric_error_counter_tran = 0;

    } // end SF computation

  } // structure_functions_hydro



  // =======================================================
  // =======================================================
  void structure_functions_mhd(int nStep,
			       ConfigMap &configMap,
			       GlobalConstants &_gParams,
			       HostArray<real_t> &U)
  {

    // structure function computation
    int maxSFOrder = configMap.getInteger("structureFunctions","max_q",5);

    // number of types of SF
    const int numberOfTypes = 13;
    
    // global sizes
    int &nx = _gParams.nx;
    int &ny = _gParams.ny;
    int &nz = _gParams.nz;

    int &NX = _gParams.nx;
    int &NY = _gParams.ny;
    int &NZ = _gParams.nz;

    // ghostWidth
    int g = 3; // for MHD

    // max distance between 2 points (assuming periodic boundary conditions)
    double maxLength = NX/2.0*sqrt(3.0);

    // some constants
    const double oneThird = 1.0/3.0;
    const double numeric_epsilon = 10*std::numeric_limits<double>::epsilon();

    // number of distance bins
    int    numberOfBins = 0;
    const int MAX_NUM_BINS = 4096;
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
	  std::cerr << "ERROR. numberOfBins exceeds MaximumNumberofBins!" << std::endl;
	}
      }
    numberOfBins++;

    // number of reference points
    int nSampleDomain    = configMap.getInteger("structureFunctions","nSampleTotal",1000);

    // ref points: cartesian coordinates (x,y,z)
    HostArray<int>    refPointsCoordDomain;
    
    // ref points: data (rho, rho_vx, rho_vy, rho_vz)
    HostArray<double> refPointsDataDomain;
    
    // Structure Functions (SF) global counters
    // SF dimensions: type, order, distance bin
    // HostArray<long int> sf_bin_counter_long;
    // HostArray<long int> sf_bin_counter_tran;
    // HostArray<double>   sf_bin_sum_long;
    // HostArray<double>   sf_bin_sum_tran;
    // HostArray<double>   sf_bin_sumsq_long;
    // HostArray<double>   sf_bin_sumsq_tran;
    
    // sf_bin_counter_long.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_counter_tran.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_sum_long.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_sum_tran.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_sumsq_long.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    // sf_bin_sumsq_tran.allocate(make_uint4(numberOfTypes, maxSFOrder, numberOfBins, 1));
    
    // sf_bin_counter_long.reset();
    // sf_bin_counter_tran.reset();

    // // sum 
    // sf_bin_sum_long.init(numeric_epsilon);
    // sf_bin_sum_tran.init(numeric_epsilon);

    // // sum of squares
    // sf_bin_sumsq_long.init(numeric_epsilon);
    // sf_bin_sumsq_tran.init(numeric_epsilon);
  
    /*
     * init random generator (old way)
     */
    int randomSeed      = configMap.getInteger("structureFunctions","randomSeed",0);
    srand(static_cast<unsigned int>(randomSeed));

    /*
     * random generator c++11 way
     */
    // int randomSeed      = configMap.getInteger("structureFunctions","randomSeed",0);
    // std::default_random_engine generator(randomSeed);
    // std::uniform_int_distribution<int> unif_dist_x(0,nx-1);
    // std::uniform_int_distribution<int> unif_dist_y(0,ny-1);
    // std::uniform_int_distribution<int> unif_dist_z(0,nz-1);
    // std::uniform_real_distribution<double> unif_dist(0.0,1.0);

    int maxBufIndex = numberOfTypes * maxSFOrder * numberOfBins;

    { // start SF computations

      // buf1 used for accumulation for all reference points
      long    *buf1_bin_counter_long   = new long   [maxBufIndex];
      long    *buf1_bin_counter_tran   = new long   [maxBufIndex];
      double  *buf1_binsum_long        = new double [maxBufIndex];
      double  *buf1_binsum_tran        = new double [maxBufIndex];
      double  *buf1_binsumsq_long      = new double [maxBufIndex];
      double  *buf1_binsumsq_tran      = new double [maxBufIndex];
      
      // buf2 used for local computations for a given reference point
      long    *buf2_bin_counter_long   = new long   [maxBufIndex];
      long    *buf2_bin_counter_tran   = new long   [maxBufIndex];
      double  *buf2_binsum_long        = new double [maxBufIndex];
      double  *buf2_binsum_tran        = new double [maxBufIndex];
      double  *buf2_binsumsq_long      = new double [maxBufIndex];
      double  *buf2_binsumsq_tran      = new double [maxBufIndex];

      // initialize buffers
      for (int bufIndex = 0; bufIndex < maxBufIndex; bufIndex++) {
	buf1_bin_counter_long    [bufIndex] = 0;
	buf1_bin_counter_tran    [bufIndex] = 0;
	buf1_binsum_long         [bufIndex] = numeric_epsilon;
	buf1_binsum_tran         [bufIndex] = numeric_epsilon;
	buf1_binsumsq_long       [bufIndex] = numeric_epsilon;
	buf1_binsumsq_tran       [bufIndex] = numeric_epsilon;
      } // end for bufIndex

      // loop over all reference point
      for (long iRefPoint = 0; 
	   iRefPoint < nSampleDomain; 
	   iRefPoint++) {

	// clear buf2
	for (int bufIndex = 0; bufIndex < maxBufIndex; bufIndex++) {
	  buf2_bin_counter_long [bufIndex] = 0;
	  buf2_bin_counter_tran [bufIndex] = 0;
	  buf2_binsum_long      [bufIndex] = numeric_epsilon;
	  buf2_binsum_tran      [bufIndex] = numeric_epsilon;
	  buf2_binsumsq_long    [bufIndex] = numeric_epsilon;
	  buf2_binsumsq_tran    [bufIndex] = numeric_epsilon;
	}

	// get coordinates of the reference point
	int i1 = randnum_int(NX);
	int j1 = randnum_int(NY);
	int k1 = randnum_int(NZ);
	// int i1 = unif_dist_x(generator); // random integer between 0 and nx-1
	// int j1 = unif_dist_y(generator); // random integer between 0 and ny-1
	// int k1 = unif_dist_z(generator); // random integer between 0 and nz-1

	// experimental
	double randnum_i = 0.5 + randnum();
	double randnum_j = 0.5 + randnum();
	double randnum_k = 0.5 + randnum();
	//double randnum_i = 0.5 + unif_dist(generator);
	//double randnum_j = 0.5 + unif_dist(generator);
	//double randnum_k = 0.5 + unif_dist(generator);
	
	// get ref point data
	double rho1 = U(i1+g, j1+g, k1+g, ID);
	double u1   = U(i1+g, j1+g, k1+g, IU)/rho1;
	double v1   = U(i1+g, j1+g, k1+g, IV)/rho1;
	double w1   = U(i1+g, j1+g, k1+g, IW)/rho1;
	double bx1  = U(i1+g, j1+g, k1+g, IBX);
	double by1  = U(i1+g, j1+g, k1+g, IBY);
	double bz1  = U(i1+g, j1+g, k1+g, IBZ);

	double pow3rho1 = pow(rho1,oneThird);
	//double lnrho1 = log(rho1);
	
	double DX[numberOfTypes] = {0}; 
	double DY[numberOfTypes] = {0}; 
	double DZ[numberOfTypes] = {0};
	
	double incr_long[numberOfTypes] = {0}; 
	double incr_tran[numberOfTypes] = {0};
	double incr_long_pow[numberOfTypes] = {0}; 
	double incr_tran_pow[numberOfTypes] = {0};

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
		    i2sub = i2 /*- offset[IX]*/;
		    if ((i2sub < 0) || (i2sub >= nx)) continue;
		    
		    j2sub = j2 /*- offset[IY]*/;
		    if ((j2sub < 0) || (j2sub >= ny)) continue;
		    
		    k2sub = k2 /*- offset[IZ]*/;
		    if ((k2sub < 0) || (k2sub >= nz)) continue;
		    
		    // get current point data (at i2sub,j2sub,k2sub)
		    double rho2 = U(i2sub+g, j2sub+g, k2sub+g, ID);
		    double u2   = U(i2sub+g, j2sub+g, k2sub+g, IU)/rho2;
		    double v2   = U(i2sub+g, j2sub+g, k2sub+g, IV)/rho2;
		    double w2   = U(i2sub+g, j2sub+g, k2sub+g, IW)/rho2;
		    double bx2  = U(i2sub+g, j2sub+g, k2sub+g, IBX);
		    double by2  = U(i2sub+g, j2sub+g, k2sub+g, IBY);
		    double bz2  = U(i2sub+g, j2sub+g, k2sub+g, IBZ);
		    
		    double pow3rho2 = pow(rho2,oneThird);
		    //double lnrho2 = log(rho2);

		    for (int t = 0; t < numberOfTypes; t++) {
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
		      case 2: // rho^(1/3)*v
			{
			  DX[t] = pow3rho2*u2 - pow3rho1*u1;
			  DY[t] = pow3rho2*v2 - pow3rho1*v1;
			  DZ[t] = pow3rho2*w2 - pow3rho1*w1;
			  break;
			}
		      case 3: // increm Elsasser +
			{
			  DX[t] = u2+bx2/sqrt(rho2) - ( u1+bx1/sqrt(rho1) );
			  DY[t] = v2+by2/sqrt(rho2) - ( v1+by1/sqrt(rho1) );
			  DZ[t] = w2+bz2/sqrt(rho2) - ( w1+bz1/sqrt(rho1) );
			  break;
			}
		      case 4: // increm Elsasser -
			{
			  DX[t] = u2-bx2/sqrt(rho2) - ( u1-bx1/sqrt(rho1) );
			  DY[t] = v2-by2/sqrt(rho2) - ( v1-by1/sqrt(rho1) );
			  DZ[t] = w2-bz2/sqrt(rho2) - ( w1-bz1/sqrt(rho1) );
			  break;
			}
		      case 5: // increm rho^(1/3)*Elsasser +
			{
			  DX[t] = pow3rho2*( u2+bx2/sqrt(rho2) ) 
			    -     pow3rho1*( u1+bx1/sqrt(rho1) );
			  DY[t] = pow3rho2*( v2+by2/sqrt(rho2) ) 
			    -     pow3rho1*( v1+by1/sqrt(rho1) );
			  DZ[t] = pow3rho2*( w2+bz2/sqrt(rho2) ) 
			    -     pow3rho1*( w1+bz1/sqrt(rho1) );
			  break;
			}
		      case 6: // increm rho^(1/3)*Elsasser -
			{
			  DX[t] = pow3rho2*( u2-bx2/sqrt(rho2) ) 
			    -     pow3rho1*( u1-bx1/sqrt(rho1) );
			  DY[t] = pow3rho2*( v2-by2/sqrt(rho2) ) 
			    -     pow3rho1*( v1-by1/sqrt(rho1) );
			  DZ[t] = pow3rho2*( w2-bz2/sqrt(rho2) ) 
			    -     pow3rho1*( w1-bz1/sqrt(rho1) );
			  break;
			}
		      case 7: // log(increm(v))
			{
			  DX[t] = u2 - u1;
			  DY[t] = v2 - v1;
			  DZ[t] = w2 - w1;
			  break;
			}
		      case 8: // log(increm(rho^(1/3)*v))
			{
			  DX[t] = pow3rho2*u2 - pow3rho1*u1;
			  DY[t] = pow3rho2*v2 - pow3rho1*v1;
			  DZ[t] = pow3rho2*w2 - pow3rho1*w1;
			  break;
			}
		      case 9: // log increm Elsasser +
			{
			  DX[t] = u2+bx2/sqrt(rho2) - ( u1+bx1/sqrt(rho1) );
			  DY[t] = v2+by2/sqrt(rho2) - ( v1+by1/sqrt(rho1) );
			  DZ[t] = w2+bz2/sqrt(rho2) - ( w1+bz1/sqrt(rho1) );
			  break;
			}
		      case 10: // log increm Elsasser -
			{
			  DX[t] = u2-bx2/sqrt(rho2) - ( u1-bx1/sqrt(rho1) );
			  DY[t] = v2-by2/sqrt(rho2) - ( v1-by1/sqrt(rho1) );
			  DZ[t] = w2-bz2/sqrt(rho2) - ( w1-bz1/sqrt(rho1) );
			  break;
			}
		      case 11: // log increm rho^(1/3)*Elsasser +
			{
			  DX[t] = pow3rho2*( u2+bx2/sqrt(rho2) ) 
			    -     pow3rho1*( u1+bx1/sqrt(rho1) );
			  DY[t] = pow3rho2*( v2+by2/sqrt(rho2) ) 
			    -     pow3rho1*( v1+by1/sqrt(rho1) );
			  DZ[t] = pow3rho2*( w2+bz2/sqrt(rho2) ) 
			    -     pow3rho1*( w1+bz1/sqrt(rho1) );
			  break;
			}
		      case 12: // log increm rho^(1/3)*Elsasser -
			{
			  DX[t] = pow3rho2*( u2-bx2/sqrt(rho2) ) 
			    -     pow3rho1*( u1-bx1/sqrt(rho1) );
			  DY[t] = pow3rho2*( v2-by2/sqrt(rho2) ) 
			    -     pow3rho1*( v1-by1/sqrt(rho1) );
			  DZ[t] = pow3rho2*( w2-bz2/sqrt(rho2) ) 
			    -     pow3rho1*( w1-bz1/sqrt(rho1) );
			  break;
			}
		      default:
			{
			  std::cout << "Compute SF:  something is wrong with the structure function type! ." << std::endl;
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
		      
		      if (t >= 7) { // log increment
			
			incr_long[t] = log( incr_long[t] );
			
			incr_tran[t] = log( incr_tran[t] );
			
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
		    
		    // compute the higher order structure functions 
		    // (be cautious with numerics)
		    for (int i = 0; i < maxSFOrder; i++) 
		      for (int t = 0; t < numberOfTypes; t++)
			{
			  
			  int bufIndex = bin*numberOfTypes*maxSFOrder+i*numberOfTypes+t;
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
	for (int bufIndex = 0; bufIndex < maxBufIndex; bufIndex++)
	  {
	    buf1_bin_counter_long[bufIndex] += buf2_bin_counter_long[bufIndex];
	    buf1_binsum_long     [bufIndex] += buf2_binsum_long     [bufIndex];
	    buf1_binsumsq_long   [bufIndex] += buf2_binsumsq_long   [bufIndex];
	    buf1_bin_counter_tran[bufIndex] += buf2_bin_counter_tran[bufIndex];
	    buf1_binsum_tran     [bufIndex] += buf2_binsum_tran     [bufIndex];
	    buf1_binsumsq_tran   [bufIndex] += buf2_binsumsq_tran   [bufIndex];
	  }

      } // end for iRefPoint

      // free memory
      delete[] buf2_bin_counter_long; buf2_bin_counter_long = 0;
      delete[] buf2_bin_counter_tran; buf2_bin_counter_tran = 0;
      delete[] buf2_binsum_long; buf2_binsum_long = 0;
      delete[] buf2_binsum_tran; buf2_binsum_tran = 0;
      delete[] buf2_binsumsq_long; buf2_binsumsq_long = 0;
      delete[] buf2_binsumsq_tran; buf2_binsumsq_tran = 0;

      // we shall now proceed to saving SF to files !!!!
      std::string outputDir    = configMap.getString("output", "outputDir", "./");
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
      std::ostringstream outNum;
      outNum.width(7);
      outNum.fill('0');
      outNum << nStep;
      
      outputPrefix = outputDir+"/"+outputPrefix+"_"+outNum.str()+"_sf";

      std::string output_file=outputPrefix+".npz";

      // write distance grid
      {
	
	const unsigned int shape[] = {(unsigned int) numberOfBins};
	cnpy::npz_save(output_file.c_str(),"distance",distanceGrid,shape,1,"w");
      }
      
      // write sf longitudinal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "binsum_long", buf1_binsum_long,shape,3,"a");
      }
      
      // write sf square longitudinal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "binsum_sq_long", buf1_binsumsq_long,shape,3,"a");
      }
      
      // write counter longitudinal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "count_long", buf1_bin_counter_long,shape,3,"a");
      }
      
      // write sf transversal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "binsum_tran", buf1_binsum_tran,shape,3,"a");
      }
      
      // write sf square transversal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "binsum_sq_tran", buf1_binsumsq_tran,shape,3,"a");
      }
      
      // write counter transversal
      {
	const unsigned int shape[] = {(unsigned int) numberOfBins, 
				      (unsigned int) maxSFOrder, 
				      (unsigned int) numberOfTypes};
	cnpy::npz_save(output_file.c_str(), "count_tran", buf1_bin_counter_tran,shape,3,"a");
      }
 
      // write some comments
      {
	std::string comments = "";

	comments += "About SF types\n";
	comments += "Type  0: velocity increments\n";
	comments += "Type  1: norm(velocity increments)\n";
	comments += "Type  2: rho^(1/3)*v increments\n";
	comments += "Type  3: increm Elsasser +\n";
	comments += "Type  4: increm Elsasser -\n";
	comments += "Type  5: increm rho^(1/3)*Elsasser +\n";
	comments += "Type  6: increm rho^(1/3)*Elsasser -\n";
	comments += "Type  7: log increm v\n";
	comments += "Type  8: log increm rho^(1/3)*v\n";
	comments += "Type  9: log increm Elsasser +\n";
	comments += "Type 10: log increm Elsasser -\n";
	comments += "Type 11: log increm rho^(1/3)*Elsasser+\n";
	comments += "Type 12: log increm rho^(1/3)*Elsasser-\n";

	unsigned int shape[] = { (unsigned int) comments.size() };

	cnpy::npz_save(output_file.c_str(), "comments", comments.c_str(),shape,1,"a");
      }
     
      delete[] buf1_bin_counter_long; buf1_bin_counter_long = 0;
      delete[] buf1_bin_counter_tran; buf1_bin_counter_tran = 0;
      delete[] buf1_binsum_long; buf1_binsum_long = 0;
      delete[] buf1_binsum_tran; buf1_binsum_tran = 0;
      delete[] buf1_binsumsq_long; buf1_binsumsq_long = 0;
      delete[] buf1_binsumsq_tran; buf1_binsumsq_tran = 0;
      //delete[] buf1_numeric_error_counter_long; buf1_numeric_error_counter_long = 0;
      //delete[] buf1_numeric_error_counter_tran; buf1_numeric_error_counter_tran = 0;

    } // end SF computation

  } // structure_functions_mhd

} // namespace hydroSimu
