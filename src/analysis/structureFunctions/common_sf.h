#ifndef COMMON_SF_H_
#define COMMON_SF_H_

enum SF_TYPES {
  SF_TYPE_V     = 0, /* v         */
  SF_TYPE_RHO2V = 1, /* rho^1/2*v */
  SF_TYPE_RHO3V = 2  /* rho^1/3*v */
};

//! number of SF type (velocity, rho^1/2*v, rho^1/3*v)
//extern const int NumberOfTypes;

//! number of SF order (q=1,2, ..., MaxSFOrder), initialisation from param file
extern int MaxSFOrder;

enum DataIndex {
  IID=0,  /*!< ID Density field index */
  IIU=1,  /*!< X velocity / momentum index */
  IIV=2,  /*!< Y velocity / momentum index */
  IIW=3,  /*!< Z velocity / momentum index */
};

#endif // COMMON_SF_H_
