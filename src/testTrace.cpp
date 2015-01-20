/**
 * \file testTrace.cpp
 * \brief numerical test of reconstruction step
 *
 * $Id$
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>

//#define USE_DOUBLE

#include "./hydro/real_type.h"
#include "./hydro/trace_mhd.h"

/** Parse command line / configuration parameter file */
#include "GetPot.h"
#include <ConfigMap.h>

#include <HydroParameters.h> 

/* some input parameter */

real_t q[NVAR_MHD] = {0.78085453230400148, 0.78085453230400148, 
		      1.7482990149652375, 1.431649141690978, 
		      -1.6489703029434373, 0.96664668393322351, 
		      0.45950695670762798, -0.83734007613353567};

real_t dq[3][NVAR_MHD] = {{-0.20004523946836317, -0.20004523946836317, -0.073880482150031207, -0.10480637362090355, 
			   0.15073338522700153, -0.53717455527658486, -0.1597735588049945, 0.092140452263844097}, 
			  {-0.2442286152441237, -0.2442286152441237, -0.068687856494275196, -0.099121662109250691, 
			   0.15381051628001347, -0.24808016252608234, -0.2011688411426639, 0.091373013001669889}, 
			  {-0.54002379330827266, -0.54002379330827266, -0.04273529612352038, -0.040499860187924119,
			   0.089364945980171834, -0.40881892596044966, 0.079419629709805969, 0.14568920282309172}};

real_t bfNb[6] = {0.93024795905089264, 1.0030454088155545, 
		  0.442596799777559, 0.4764171136376969, 
		  -0.78403119432113488, -0.89064895794593657};

real_t dbf[12] = {0.19655484116022154, -0, -0.063166365419217829, 0.16154482778326049, 
		  0.0035521953184123323, -0.10797247358474393, 0.20889819517277602,-0,
		  -0.17366142021062936, 0.31824156182477314, 0.062986152988710353, 0};

real_t elecFields[3][2][2] = {{{-0.34371331413681028, -0.3982705071033722}, 
			       {-0.53446391973021579, -0.35180107596471411}}, 
			      {{-0.19044704408885793, 0.036575569327717217}, 
			       {-0.45759436562143385, -0.18324907665978296}},
			      {{-0.33216337592034451, -0.47886436975190949},
			       {-0.7921176818152742, -1.050387151409131}}};

real_t dtdx = 0.013867197924410549;
real_t dtdy = 0.013867197924410549;
real_t dtdz = 0.013867197924410549;

void testTrace_cpu(real_t (&qm)[3][NVAR_MHD], 
		   real_t (&qp)[3][NVAR_MHD],
		   real_t (&qEdge)[4][3][NVAR_MHD])
{
  real_t xPos=0.0;

  // call trace
  trace_unsplit_mhd_3d_simpler(q, dq, bfNb, dbf, elecFields, dtdx, dtdy, dtdz, xPos, qm, qp, qEdge);

}


int main(int argc, char *argv[])
{

  /* parse command line arguments */
  GetPot cl(argc, argv);

  /* search for configuration parameter file */
  const std::string default_input_file = std::string(argv[0])+ ".ini";
  const std::string input_file = cl.follow(default_input_file.c_str(),    "--param");

  /* parse parameters from input file */
  ConfigMap configMap(input_file); 

  HydroParameters * hydroParam = new HydroParameters(configMap);


  /**************/
  /* TEST TRACE */
  /**************/
  {
    real_t qm[3][NVAR_MHD];
    real_t qp[3][NVAR_MHD];
    real_t qEdge[4][3][NVAR_MHD];

    // CPU
    testTrace_cpu(qm,qp,qEdge);
 
    // print results on screen
    std::cout << "test trace_unsplit_mhd_3d_simpler\n";
    for (int i=0; i<NVAR_MHD; i++) {
      printf("qm[0][%d] = %e\t qm[1][%d] = %e\t qm[2][%d] = %e\n",i,(double)qm[0][i],i,(double)qm[1][i],i,(double)qm[2][i]);
    }
    for (int i=0; i<NVAR_MHD; i++) {
      printf("qp[0][%d] = %e\t qp[1][%d] = %e\t qp[2][%d] = %e\n",i,(double)qp[0][i],i,(double)qp[1][i],i,(double)qp[2][i]);
    }
    for (int i=0; i<NVAR_MHD; i++) {
      printf("qEdge[0][0][%d] = %e\t qEdge[0][1][%d] = %e\t qEdge[0][2][%d] = %e\n",i,(double)qEdge[0][0][i],i,(double)qEdge[0][1][i],i,(double)qEdge[0][2][i]);
      printf("qEdge[1][0][%d] = %e\t qEdge[1][1][%d] = %e\t qEdge[1][2][%d] = %e\n",i,(double)qEdge[1][0][i],i,(double)qEdge[1][1][i],i,(double)qEdge[1][2][i]);
      printf("qEdge[2][0][%d] = %e\t qEdge[2][1][%d] = %e\t qEdge[2][2][%d] = %e\n",i,(double)qEdge[2][0][i],i,(double)qEdge[2][1][i],i,(double)qEdge[2][2][i]);
      printf("qEdge[3][0][%d] = %e\t qEdge[3][1][%d] = %e\t qEdge[3][2][%d] = %e\n",i,(double)qEdge[3][0][i],i,(double)qEdge[3][1][i],i,(double)qEdge[3][2][i]);
    }
    std::cout << "\n\n";
    
  }

  // clean and exit
  delete hydroParam;

  return EXIT_SUCCESS;

}
