/**
 * \file RandomGen.cpp
 * \brief Adapt random.f90 from original ramses code.
 *
 * \note See also C++11 random capabilities :
 *       http://en.cppreference.com/w/cpp/numeric/random
 *
 * \date Dec 19, 2013
 *
 * $Id: RandomGen.cpp 3465 2014-06-29 21:28:48Z pkestene $
 */

/*
 *=======================================================================
 *
 *   P A R A L L E L   R A N D O M   N U M B E R   G E N E R A T O R
 *
 *=======================================================================
 * Here's how to use these functions:
 *       Initialization: call Rans( N, StartVal, Seeds )
 * This returns an array of N seeds;
 * the returned seeds have values that partition the basic
 * ranf cycle of 2**46 pseudorandom reals in [0,1] into independent sub-
 * sequences. The second argument, StartVal, can be a cycle-starting seed
 * for the full-period generator; if it is zero, a special seed will be
 * used that produces statistically desirable behavior.
 *       Use: call Ranf( Seed, RandNum )
 * This returns a pseudorandom real and a seed to be passed back in the
 * next invocation. The returned seed carries the state normally hidden
 * in imperative generators.
 */
/*=======================================================================*/

#include "RandomGen.h"
#include <math.h>
#include <cstdlib>

// static member declaration
const int RandomGen::IRandNumSize;
const int RandomGen::IBinarySize;
const int RandomGen::Mod4096DigitSize;
const int RandomGen::IZero;
const int RandomGen::NPoissonLimit;

const int RandomGen::Multiplier[IRandNumSize]  = {373, 3707, 1442, 647};
const int RandomGen::DefaultSeed[IRandNumSize] = {3281, 4041, 595, 2376};

const double RandomGen::Divisor[IRandNumSize] = {281474976710656.0,
						 68719476736.0,
						 16777216.0,
						 4096.0};

//=======================================================================
//=======================================================================
/**
 * Random generator (linear congruential), uniformly distributed in [0,1].
 *
 * \param[in,out] Seed random seed array (will be modified)
 * \param[out]    RandNum the output
 */
void RandomGen::ranf( int (&Seed)[IRandNumSize], 
		      double &RandNum )
{
  int OutSeed[IRandNumSize];
  
  RandNum = 
    (float)( Seed[ 3 ] ) / Divisor[ 3 ] + 
    (float)( Seed[ 2 ] ) / Divisor[ 2 ] + 
    (float)( Seed[ 1 ] ) / Divisor[ 1 ] + 
    (float)( Seed[ 0 ] ) / Divisor[ 0 ];
  
  ranfModMult( Multiplier, Seed, OutSeed );
  
  Seed[0] = OutSeed[0];
  Seed[1] = OutSeed[1];
  Seed[2] = OutSeed[2];
  Seed[3] = OutSeed[3];
  
} // RandomGen::ranf


//======================================================================
//======================================================================
/**
 * Pseudo-random number generator with poisson distribution.
 *
 * \param[in,out] Seed     will be modified
 * \param[in]     AverNum  poisson distribution average
 * \param[out]    PoissNum poisoon random number
 */
void RandomGen::poissDev( int (&Seed)[IRandNumSize],
			  double AverNum,
			  int &PoissNum)
{
  
  double Norm, Repar, Proba;
  double RandNum, GaussNum;
  
  if(AverNum <= (double)(NPoissonLimit)) {
    
    Norm=exp(-AverNum);
    Repar=1.0;
    PoissNum=0;
    Proba=1.0;
    ranf(Seed,RandNum);
    do {
      PoissNum=PoissNum+1;
      Proba=Proba*AverNum/PoissNum;
      Repar=Repar+Proba;
    } while(Repar*Norm <= RandNum && PoissNum <= 10*NPoissonLimit );

  } else {
    
    gaussDev(Seed,GaussNum);
    GaussNum=GaussNum*sqrt(AverNum)-0.5+AverNum;
    if(GaussNum<=0.0) GaussNum=0.0;
    PoissNum=(int)(rint(GaussNum));
    
  }
  
} // RandomGen::poissDev


//======================================================================
//======================================================================
/**
 * Pseudo-random number generator with normal distribution with
 * zero mean and unit variance.
 *
 * Applies the polar form of Box-Muller transform to produce pseudo-random
 * numbers with Gaussian (normal) distribution which has a zero mean and
 * standard deviation of one.
 * Box GEP, Muller ME. A note on the generation of random normal deviates.
 * Annals of Mathematical Statistics, Volume 29, Issue 2, 1958, 610-611.
 * Available from JSTOR http://www.jstor.org/
 *
 * \param[in,out] Seed     random seed
 * \param[out]    GaussNum random number
 */
void RandomGen::gaussDev( int (&Seed)[IRandNumSize], 
			  double &GaussNum ) 
{
  
  double fac,rsq,v1,v2;
  
  if (IGauss == IZero) {

    rsq=0.0;
    while (rsq >= 1.0 || rsq <= 0.0) {
      ranf(Seed,v1);
      ranf(Seed,v2);
      v1  = 2.0*v1 - 1.0;
      v2  = 2.0*v2 - 1.0;
      rsq = v1*v1 + v2*v2;
    }
    fac = sqrt(-2.0*log(rsq)/rsq);
    GaussBak=v1*fac;
    GaussNum=v2*fac;
    IGauss=1;

  } else {

    GaussNum=GaussBak;
    IGauss=0;

  }
  
} // RandomGen::gaussDev

//======================================================================
//======================================================================
/**
 * Main routine to initialize N random seeds.
 *
 * \param[in]   N         number of random seed to generate
 * \param[in]   StartVal  Initial seed array value
 * \param[out]  SeedArray array of random generator seeds
 */
void RandomGen::rans( int N, 
		      int StartVal, 
		      int SeedArray[][4] )
{
  
  /* SeedArray should be allocated as a 2D array of sizes N,4 */

  int atothek[IRandNumSize];
  int K      [IRandNumSize];
  int InSeed [IRandNumSize];
  int OutSeed[IRandNumSize];
  int KBinary[IBinarySize];

  int I;

  if( StartVal == IZero ) {
    SeedArray[0][ 0 ] = DefaultSeed[ 0 ];
    SeedArray[0][ 1 ] = DefaultSeed[ 1 ];
    SeedArray[0][ 2 ] = DefaultSeed[ 2 ];
    SeedArray[0][ 3 ] = DefaultSeed[ 3 ];
  } else {
    SeedArray[0][ 0 ] = abs( StartVal );
    SeedArray[0][ 1 ] = IZero;
    SeedArray[0][ 2 ] = IZero;
    SeedArray[0][ 3 ] = IZero;
  }
    
  if( N == 1 ) {
    atothek[ 0 ] = Multiplier[ 0 ];
    atothek[ 1 ] = Multiplier[ 1 ];
    atothek[ 2 ] = Multiplier[ 2 ];
    atothek[ 3 ] = Multiplier[ 3 ];
  } else {
    ranfk( N, K );
    ranfkBinary( K, KBinary );
    ranfatok( Multiplier, KBinary, atothek );
    for ( I = 1; I < N; I++) {
      InSeed[ 0 ] = SeedArray[ I-1][ 0 ];
      InSeed[ 1 ] = SeedArray[ I-1][ 1 ];
      InSeed[ 2 ] = SeedArray[ I-1][ 2 ];
      InSeed[ 3 ] = SeedArray[ I-1][ 3 ];
      ranfModMult( InSeed, atothek, OutSeed );
      SeedArray[I][ 0 ] = OutSeed[ 0 ];
      SeedArray[I][ 1 ] = OutSeed[ 1 ];
      SeedArray[I][ 2 ] = OutSeed[ 2 ];
      SeedArray[I][ 3 ] = OutSeed[ 3 ];
    }
  }
} // RandomGen::rans

//======================================================================
//======================================================================
/**
 * Main routine to initialize N random  seeds.
 *
 * \param[in]   N         number of random seed to generate
 * \param[in]   StartVal  Initial seed array value
 * \param[out]  SeedArray array of random generator seeds
 */
void RandomGen::rans( int N, 
		      int StartVal, 
		      int *SeedArray )
{
  
  /* SeedArray should be allocated as a 1D array of sizes N*4 */

  int atothek[IRandNumSize];
  int K      [IRandNumSize];
  int InSeed [IRandNumSize];
  int OutSeed[IRandNumSize];
  int KBinary[IBinarySize];

  int I;

  if( StartVal == IZero ) {
    SeedArray[0] = DefaultSeed[0];
    SeedArray[1] = DefaultSeed[1];
    SeedArray[2] = DefaultSeed[2];
    SeedArray[3] = DefaultSeed[3];
  } else {
    SeedArray[0] = abs( StartVal );
    SeedArray[1] = IZero;
    SeedArray[2] = IZero;
    SeedArray[3] = IZero;
  }
    
  if( N == 1 ) {
    atothek[ 0 ] = Multiplier[ 0 ];
    atothek[ 1 ] = Multiplier[ 1 ];
    atothek[ 2 ] = Multiplier[ 2 ];
    atothek[ 3 ] = Multiplier[ 3 ];
  } else {
    ranfk( N, K );
    ranfkBinary( K, KBinary );
    ranfatok( Multiplier, KBinary, atothek );
    for ( I = 1; I < N; I++) {
      InSeed[ 0 ] = SeedArray[(I-1)*4+0];
      InSeed[ 1 ] = SeedArray[(I-1)*4+1];
      InSeed[ 2 ] = SeedArray[(I-1)*4+2];
      InSeed[ 3 ] = SeedArray[(I-1)*4+3];
      ranfModMult( InSeed, atothek, OutSeed );
      SeedArray[I*4+0] = OutSeed[ 0 ];
      SeedArray[I*4+1] = OutSeed[ 1 ];
      SeedArray[I*4+2] = OutSeed[ 2 ];
      SeedArray[I*4+3] = OutSeed[ 3 ];
    }
  }
} // RandomGen::rans

//======================================================================
//======================================================================
/**
 * This routine computes a to the Kth power, mod 2**48. K is a binary number.
 * It returns the calculated value as an array of four modulo-4096 digits.
 */
void RandomGen::ranfatok( const int (&a)[IRandNumSize], 
			  int (&KBinary)[IBinarySize],
			  int (&atothek)[IRandNumSize] )
{
  
  int asubi[IRandNumSize];
  
  int I;
  
  asubi[ 0 ] = a[ 0 ];
  asubi[ 1 ] = a[ 1 ];
  asubi[ 2 ] = a[ 2 ];
  asubi[ 3 ] = a[ 3 ];
  
  atothek[ 0 ] = 1;
  atothek[ 1 ] = IZero;
  atothek[ 2 ] = IZero;
  atothek[ 3 ] = IZero;
  
  for ( I = 0; I<= 44; I++) {
    if( KBinary[ I ] != IZero ) {
      ranfModMult( atothek, asubi, atothek );
    }
    ranfModMult( asubi, asubi, asubi );
  }
  
} // RandomGen::ranfatok


//======================================================================
//======================================================================
/**
 * This routine calculates 2**46/N, which should be the period of each of the
 * subsequences of random numbers that are being created. Both the numerator
 * and the result of this calculation are represented as an array of four
 * integers, each of which is one digit of a four-digit moduo-4096 number.  The
 * numerator is represented as (1024, 0, 0, 0 ), using base ten digits.
 *   
 *    It returns the result of the division.
 *
 * \param[in]  N
 * \param[out] K
 */
void RandomGen::ranfk( int N, int (&K)[IRandNumSize] )
{
  
  int nn, r4, r3, r2, q4, q3, q2, q1;
  
  nn = N + iRanfEven( N );
  
  q4 = 1024 / nn;
  r4 = 1024 - (nn * q4);
  q3 = (r4 * 4096) / nn;
  r3 = (r4 * 4096) - (nn * q3);
  q2 = (r3 * 4096) / nn;
  r2 = (r3 * 4096) - (nn * q2);
  q1 = (r2 * 4096) / nn;
  
  K[ 0 ] = q1;
  K[ 1 ] = q2;
  K[ 2 ] = q3;
  K[ 3 ] = q4;
} // RandomGen::ranfk


//======================================================================
//======================================================================
/**
 * This routine calculates the binary expansion of the argument
 * K, which is a 48-bit integer represented as an array of four 12-bit
 * integers.
 *
 * It returns an array of 48 binary values (KBinary).
 *
 * \param[in]  K
 * \param[out] KBinary
 */
void RandomGen::ranfkBinary( int (&K)[IRandNumSize], 
			     int (&KBinary)[IBinarySize] )
{
  
  int Bits[Mod4096DigitSize];
  int X;
  
  for (int I=0; I<4; I++) {
    X = K[I] / 2;
    Bits[0] = iRanfOdd (K[I]);
    for(int J = 1; J < Mod4096DigitSize; J++) {
      Bits[J] = iRanfOdd (X);
      X = X / 2;
    }
    for (int J = 0; J< Mod4096DigitSize; J++) {
      KBinary[ I*Mod4096DigitSize + J ] = Bits[J];
    }
  }
  
} // RandomGen::ranfkBinary


//======================================================================
//======================================================================
/**
 * This routine computes the product of the first two arguments. 
 * All three arguments are represented as arrays of 12-bit
 * integers, each making up the digits of one radix-4096
 * number. The multiplication is done piecemeal.
 *
 * It returns the product in the third argument.
 * 
 * \param[in]  A
 * \param[in]  B
 * \param[out] C
 */
void RandomGen::ranfModMult( const int (&A)[IRandNumSize], 
			     const int (&B)[IRandNumSize],
			     int       (&C)[IRandNumSize]) 
{
  
  int j0, j1, j2, j3, k0, k1, k2, k3;
  
  j0 = A[0] * B[0];
  j1 = A[0] * B[1] + A[1] * B[0];
  j2 = A[0] * B[2] + A[1] * B[1] + A[2] * B[0];
  j3 = A[0] * B[3] + A[1] * B[2] + A[2] * B[1] + A[3] * B[0];
  
  k0 = j0;
  k1 = j1 + k0 / 4096;
  k2 = j2 + k1 / 4096;
  k3 = j3 + k2 / 4096;
    
  C[0] = k0 % 4096;
  C[1] = k1 % 4096;
  C[2] = k2 % 4096;
  C[3] = k3 % 4096;

} // RandomGen::ranfModMult


//======================================================================
//======================================================================
/**
 * This function checks the parity of the argument integer.
 *
 * It returns one if the argument is odd and zero otherwise.
 */
int iRanfOdd( int N )
{
  
  int val;
  if( N%2 == 0 ) {
    val = 0;
  } else {
    val = 1;
  }	
  return val;

} // iRandOdd


//======================================================================
//======================================================================
/**
 * This function checks the parity of the argument integer.
 *
 * It returns one if the argument is even and zero otherwise.
 */
int iRanfEven( int N )
{

  int val;
  if( N%2 == 0 ) {
    val = 1;
  } else {
    val = 0;
  }
  return val;

} // iRanfEven

/** ------------------------- randnum --------------------------------
 ** returns a random number in [0;1]
 ** ------------------------------------------------------------------ */
double randnum(void)
{
  return static_cast<double>(rand())/static_cast<double>(RAND_MAX);
}

/** ------------------------- randnum_int -----------------------------
 ** returns a random number in [0, N [
 ** assumes N < RAND_MAX (= 2147483647 = 2^31-1)
 ** ------------------------------------------------------------------ */
int randnum_int(int N)
{
  return rand() % N;
}

