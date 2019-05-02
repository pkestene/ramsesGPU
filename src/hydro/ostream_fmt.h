/**
 * \file ostream_fmt.h
 * \brief Simplifies the use of iomanip
 *
 * Taken from the book "Scientific Software Design" by D. Rouson et al.
 *
 * \date January 30, 2012
 *
 * $Id: ostream_fmt.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef OSTREAM_FMT_H_
#define OSTREAM_FMT_H_

#include "real_type.h"
#include <vector>
typedef std::vector<real_t> crd_t;

#include <iostream>
#include <iomanip>

// The fmt(...) helper class helps hide the mess that is in <iomanip>
struct fmt {
  
  explicit fmt(real_t value, int width=12, int prec=8) :
    v_(1, value), w_(width), p_(prec)
  {}
  
  explicit fmt(crd_t value,  int width=12, int prec=8) :
    v_(value), w_(width), p_(prec)
  {}

  const crd_t v_;
  const int w_, p_;

}; // struct fmt

inline std::ostream& operator<<(std::ostream &os, const fmt &v) {

  // Store format flags for the stream.
  std::ios_base::fmtflags flags = os.flags();

  // Force our own weird format.
  for(crd_t::const_iterator it = v.v_.begin();
      it != v.v_.end();
      ++it) {
    os << " " << std::setw(v.w_) << std::setprecision(v.p_) << std::fixed << *it;
  }

  // Restore original format flags.
  os.flags(flags);
  return os;
}

#endif // OSTREAM_FMT_H_
