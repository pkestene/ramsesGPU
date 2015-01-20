/*
 * Copyright CEA / Maison de la Simulation
 * Contributors: Pierre Kestener, Sebastien Fromang (May 22, 2012)
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use, 
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info". 
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 */

/**
 * \file turbulenceInit.cpp
 * \brief Create random fiels used to initialize velocity field in a turbulence simulation.
 *
 * This turbulence initialization is adapted from Enzo (by A. Kritsuk) in file
 * turboinit.f
 *
 * \author P. Kestener
 * \date 01/09/2012
 *
 * $Id: turbulenceInit.cpp 2868 2013-06-07 13:26:56Z pkestene $
 *
 */

#include "turbulenceInit.h"
#include <math.h>
#include <stdio.h>

/**
 * Create random field for turbulence forcing.
 *
 * \param[in] sizeX physical (i.e. ghost included) array size along X
 * \param[in] sizeY physical (i.e. ghost included) array size along Y
 * \param[in] sizeZ physical (i.e. ghost included) array size along Z
 * \param[in] offsetX used for ghost width and MPI sub-domain
 * \param[in] offsetY used for ghost width and MPI sub-domain
 * \param[in] offsetZ used for ghost width and MPI sub-domain
 * \param[in] nbox logical size of global domain (nx*mz in the MPI case)
 * \param[in] randomForcingMachNumber
 * \param[out] u X component of the generated velocity field
 * \param[out] v Y component of the generated velocity field
 * \param[out] w Z component of the generated velocity field
 *
 */
void turbulenceInit(int sizeX,   int sizeY,   int sizeZ,
		    int offsetX, int offsetY, int offsetZ,
		    int nbox, real_t randomForcingMachNumber,
		    real_t *u, real_t *v, real_t *w)
{

  const int nMode = 16;
  int mode[nMode][3] = 
    { 
      {1,1,1},
      {-1,1,1},
      {1,-1,1},
      {1,1,-1},
      {0,0,1},
      {0,1,0},
      {1,0,0},
      {0,1,1},
      {1,0,1},
      {1,1,0},
      {0,-1,1},
      {-1,0,1},
      {-1,1,0},
      {0,0,2},
      {0,2,0},
      {2,0,0} 
    };

  // A set of randomly selected phases for seed=12398L that provide good 
  // isotropy
  // Phases are uniformly sampled from [0, 2pi)
  // Phases for x, y, and z velocities for each mode
  
  real_t phax[nMode] = 
    { 4.88271710 , 4.55016280 , 3.68972560  , 5.76067300, 
      2.02647730 , 0.832007770, 1.93749010 , 0.0141755510, 
      5.13556960 , 2.77787590 , 2.02909450 , 0.663769130, 
      1.80512500 , 3.31305960 , 1.05063310 , 1.75230850};

  real_t phay[nMode] =
    { 1.40113130, 5.71809960 , 3.82072880 , 1.00265060, 
      2.26816680, 2.81446220 , 0.990584490, 2.94580650, 
      3.92715640, 0.896237970, 1.85357800 , 2.84606100, 
      1.63463330, 3.46619220 , 5.58599570 , 1.59481430 };

  real_t phaz[nMode] =
    { 5.60595510, 4.13909050, 6.22733640, 5.92633250, 
      3.51874880, 5.42229180, 5.77061890, 4.95180180, 
      4.46144340, 5.29367540, 5.50741860, 2.39496800, 
      4.59486870, 2.23851540, 3.19591550, 4.47066500 };

  // Random Gaussian amplitudes for each mode for seed=12398L, solenoidalized
  real_t amp[3][nMode] =
    {
      { 0.0755957220, -1.35724380,   0.378455820, -0.383104000,  // X
	0.116980840,  -1.16079680,   0.0,         -0.0280965080, // X
	0.0,           0.0,         -0.232798780,  0.0,          // X
	0.0,          -0.879534360, -0.604585950,  0.0 },        // X
      { 1.03223790,    0.530986910, -0.242943420, -0.832715270,  // Y
	-0.607103350,   0.0,         -0.278135540,  0.0,         // Y
	-1.18019080,    0.0,          0.0,          0.976678430, // Y
	0.0,          -0.694509390,  0.0,         -0.608007610}, // Y
      { 1.01825800,   -0.966076610,  0.211956020, -0.605923650,  // Z
	0.0,           0.314906060,  0.109417880,  0.0,          // Z
	0.0,          -1.53612340,   0.0,          0.0,          // Z
	0.813212160,   0.0,         -0.368619380, -0.371489380}  // Z
    };

  // signs of choice in eqs. (10.6) and (10.7) in Crockett (2005), p.96
  real_t sign1[4] = { 1.0,-1.0,-1.0, 1.0};
  real_t sign2[4] = {-1.0,-1.0, 1.0, 1.0};

  // some variables
  real_t  aa, phayy, phazz, k1;
  const real_t pi = (real_t) (2.0*asin(1.0)); 

  // this is for large-scale force 1<k<2
  aa = 2.0*pi/nbox;

  /*
   * fill-in the velocity arrays
   */
  for (int k=0; k<sizeZ; ++k) {
    for (int j=0; j<sizeY; ++j) {
      for (int i=0; i<sizeX; ++i) {

	int index = i + sizeX * (j + sizeY * k);
	/*
	 * fill in 0s first
	 */
	u[index] = 0.0;
	v[index] = 0.0;
	w[index] = 0.0;
	  
	/*
	 * start with first four modes
	 */
	for (int imo=0; imo<4; ++imo) {
	  k1 = 
	    mode[imo][0] * (i+offsetX+1) + 
	    mode[imo][1] * (j+offsetY+1) + 
	    mode[imo][2] * (k+offsetZ+1);
	  u[index] = u[index] + amp[0][imo] * cos(aa*k1 + phax[imo]);

	  /*
	   * get solenoidal corrections for y- and z-phases of modes with
	   * k=(1,1,1), (-1,1,1), (1,-1,1), and (1,1,-1)
	   */ 
	  phayy = phax[imo] + sign1[imo]*
	    acos( (amp[2][imo]*amp[2][imo]-
		   amp[0][imo]*amp[0][imo]-
		   amp[1][imo]*amp[1][imo])/2.0/
		  amp[0][imo]/mode[imo][0]/mode[imo][1]/amp[1][imo]);
	  
	  v[index] = v[index] + 
	    amp[1][imo]*cos(aa*k1 + phayy);
	  
	  phazz = phax[imo] + sign2[imo]*
	    acos( (amp[1][imo]*amp[1][imo]-
		   amp[0][imo]*amp[0][imo]-
		   amp[2][imo]*amp[2][imo])/2.0/
		  amp[0][imo]/mode[imo][0]/mode[imo][2]/amp[2][imo]);
	  
	  w[index] = w[index] + 
	    amp[2][imo]*cos(aa*k1 + phazz);
	}
	
	/*
	 * continue with other modes
	 */
	for (int imo=4; imo<nMode; ++imo) {
	  k1 = 
	    mode[imo][0] * (i+offsetX+1) + 
	    mode[imo][1] * (j+offsetY+1) + 
	    mode[imo][2] * (k+offsetZ+1);

	  u[index] = u[index] + amp[0][imo] * cos(aa*k1 + phax[imo]);
	  v[index] = v[index] + amp[1][imo] * cos(aa*k1 + phay[imo]);
	  w[index] = w[index] + amp[2][imo] * cos(aa*k1 + phaz[imo]);
	}
	
	/*
	 * normalize to get rms 3D Mach = 3.0
	 */
	u[index] = u[index] / 2.848320 * randomForcingMachNumber;
	v[index] = v[index] / 2.848320 * randomForcingMachNumber;
	w[index] = w[index] / 2.848320 * randomForcingMachNumber;

      } // end for i
    } // end for j
  } // end for k

} // turbulenceInit
