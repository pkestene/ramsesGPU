#ifndef RANDOM_GEN_H_
#define RANDOM_GEN_H_

#include <stdlib.h> // for rand


/**
 * \struct RandomGen RandomGen.h
 *
 * Rewrite some of the random generator function used in Ramses
 * which I believe are borrowed from ASC Sequoia benchmark / SPhot.
 *
 */
struct RandomGen {

  static const int IRandNumSize = 4;
  static const int IBinarySize  = 48;
  static const int Mod4096DigitSize = 12;
  static const int IZero = 0 ;
  static const int NPoissonLimit = 10;

  static const int Multiplier[IRandNumSize];
  static const int DefaultSeed[IRandNumSize];
  
  static const double Divisor[IRandNumSize];

  int IGauss;
  double GaussBak;

  RandomGen() : IGauss(0), GaussBak(0.0) {};

  void ranf( int (&Seed)[IRandNumSize], 
	     double &RandNum );
  
  void rans( int N, 
	     int StartVal, 
	     int SeedArray[][4] );

  void rans( int N, 
	     int StartVal, 
	     int *SeedArray );

  void ranfatok( const int (&a)[IRandNumSize], 
		 int (&KBinary)[IBinarySize],
		 int (&atothek)[IRandNumSize] );
  
  void ranfk( int N, int (&K)[IRandNumSize] );
  
  void ranfkBinary( int (&K)[IRandNumSize], 
		    int (&KBinary)[IBinarySize] );
  
  void ranfModMult( const int (&A)[IRandNumSize], 
		    const int (&B)[IRandNumSize],
		    int       (&C)[IRandNumSize]);

  void poissDev( int (&Seed)[IRandNumSize],
		 double AverNum,
		 int &PoissNum);

  void gaussDev( int (&Seed)[IRandNumSize], 
		 double &GaussNum );

}; // struct RandomGen

int iRanfOdd( int N );
int iRanfEven( int N );


/* ======================== */
/* ======================== */
/* ------------------------- randnum --------------------------------
 * returns a random number in [0;1]
 * ------------------------------------------------------------------ */
double randnum(void);
  
/* ------------------------- randnum_int -----------------------------
 * returns a random number in [0, N [
 * assumes N < RAND_MAX (= 2147483647 = 2^31-1)
 * ------------------------------------------------------------------ */
int randnum_int(int N);

#endif // RANDOM_GEN_H_
