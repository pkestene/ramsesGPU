/*
 * g++ -o RandomGenTest RandomGenTest.cpp RandomGen.cpp -lm
 *
 */

#include "RandomGen.h"

#include <cstdlib>
#include <cstdio>

int main (int argc, char *argv[])
{

  int ncpu=4, ndim=3,i,nmode=31,init_rand=600;
  double mode[ndim][nmode];
  double fourforce[ndim][nmode];
  double projtens[ndim][ndim][nmode];
  int seed_gauss[ncpu][4];
  int forcseed[4];

  RandomGen randomGen = RandomGen();

  randomGen.rans(ncpu,init_rand,seed_gauss);

  for (int i=0; i<ncpu; i++) {
    
    printf("%d %d %d %d\n",
	   seed_gauss[i][0],
	   seed_gauss[i][1],
	   seed_gauss[i][2],
	   seed_gauss[i][3]);
    
  }

  forcseed[0] = seed_gauss[0][0];
  forcseed[1] = seed_gauss[0][1];
  forcseed[2] = seed_gauss[0][2];
  forcseed[3] = seed_gauss[0][3];

  for (int imode=0; imode<nmode; imode++) {
    double randomnumber;
    randomGen.gaussDev(forcseed, randomnumber);
    printf("%d %.20g\n",imode+1,randomnumber);
  }

  return 0;
}
