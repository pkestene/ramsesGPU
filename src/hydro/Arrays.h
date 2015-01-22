/**
 * \file Arrays.h
 * \brief Provides CPU/GPU C++ array classes.
 *
 * \author F. Chateau and P. Kestener
 *
 * $Id: Arrays.h 3444 2014-06-15 20:30:40Z pkestene $
 */
#ifndef ARRAYS_H_
#define ARRAYS_H_

#include <stdexcept>
#include <iostream>
#include <cstring>

// the following defines types for cuda compatibility when using g++
// instead of nvcc
#include "common_types.h"

#ifdef __CUDACC__
#include "cutil_inline.h"
#endif // __CUDACC__

namespace hydroSimu {

/**
 * \class HostArray Arrays.h 
 * \brief Provides an array object with memory allocated on CPU.
 *
 * HostArray is a storage class for 1d/2d/3d vector data. The number of
 * vector component is specified by nvar (number of variables).
 * In the case of finite volume simulation of Euler equations,
 * nvar should return 4 in 2D and 5 in 3D, as we use the following scalar fields : 
 * rho, E, u, v and w (primitive variables).
 */
template<typename T>
class HostArray
{
public:
  /** enumeration only used in the CUDA implementation */
  enum HostMemoryAllocType {
    PAGEABLE, /**< enum PAGEABLE (standard allocation using new) */
    PINNED    /**< enum PINNED (allocation using cudaMallocHost) */
  };
  HostArray();
  ~HostArray();

  /** memory allocation for 1D data */
  void allocate(int length, int numVar, HostMemoryAllocType memAllocType=PAGEABLE);
  /** memory allocation for 2D data */
  void allocate(uint3 dim, HostMemoryAllocType memAllocType=PAGEABLE);
  /** memory allocation for 3D data */
  void allocate(uint4 dim, HostMemoryAllocType memAllocType=PAGEABLE);
  /** memory free */
  void free();

  /** copy from another array (make a call to allocate and then copy data) */
  void copyHard(HostArray<T>& src);

  /** copy to another existing array of the same size */
  void copyTo(HostArray<T>& src);

  uint dimx() const	{ return _dim.x; }
  uint dimy() const	{ return _dim.y; }
  uint dimz() const	{ return _dim.z; }
  uint nvar() const	{ return _dim.w; }

  uint pitch() const	{ return _dim.x; }
  uint section() const	{ return pitch() * _dim.y * _dim.z; }
  uint size() const	{ return pitch() * _dim.y * _dim.z * _dim.w; }

  uint dimXBytes() const	{ return dimx() * sizeof(T); }
  uint pitchBytes() const 	{ return pitch() * sizeof(T); }
  uint sectionBytes() const	{ return section() * sizeof(T); }
  uint sizeBytes() const	{ return size()  * sizeof(T); }

  bool usePinnedMemory() const  { return _usePinnedMemory; }

  T* data()		{ return _data; }
  const T* data() const	{ return _data; }

  /** access 1d data (only valied if _dim.y and _dim.z are 1)*/
  T& operator() (int i, int ivar) { 
    return _data[i+_dim.x*ivar]; }
  T  operator() (int i, int ivar) const { 
    return _data[i+_dim.x*ivar]; }

  /** access 2d data (only valid if _dim.z=1) */
  T& operator() (int i, int j, int ivar) { 
    return _data[i+_dim.x*(j+_dim.y*ivar)]; }
  T  operator() (int i, int j, int ivar) const { 
    return _data[i+_dim.x*(j+_dim.y*ivar)]; }

  /** access 3d data */
  T& operator() (int i, int j, int k, int ivar) { 
    return _data[i+_dim.x*(j+_dim.y*(k+_dim.z*ivar))]; }
  T  operator() (int i, int j, int k, int ivar) const { 
    return _data[i+_dim.x*(j+_dim.y*(k+_dim.z*ivar))]; }

  /** access data directly */
  T& operator() (int i) { return _data[i]; }
  T  operator() (int i) const { return _data[i]; }

  /** other operators */
  HostArray<T> &operator+=(const HostArray<T>& operand);
  HostArray<T> &operator-=(const HostArray<T>& operand);
  HostArray<T> &operator*=(const HostArray<T>& operand);
  HostArray<T> &operator/=(const HostArray<T>& operand);

  /** other methods */
  void reset() {memset((void*) _data, 0, sizeBytes());  };

  /** init with constant value */
  void init(T value) {
    for (int idx=0; idx<this->size(); idx++)
      _data[idx] = value;
  };

private:
  T*	_data;
  uint4	_dim;
  bool  _usePinnedMemory;

public:
  /** is memory allocated ? */
  bool isAllocated;

  /** total allocated memory in bytes */
  static unsigned long int totalAllocMemoryInKB;



}; // class HostArray

  template<typename T> unsigned long int HostArray<T>::totalAllocMemoryInKB = 0;

  template<typename T> bool arraysHaveSameShape(const HostArray<T> &array1,
						const HostArray<T> &array2) {
    if (array1.dimx() != array2.dimx() ||
	array1.dimy() != array2.dimy() ||
	array1.dimz() != array2.dimz() ||
	array1.nvar() != array2.nvar() )
      return false;
    return true;
  } // arraysHaveSameShape

  // =======================================================
  // =======================================================
  /**
   * A simple routine to print a HostArray to an ostream
   */
#define PRECISION 7
#define WIDTH     10
  template<typename T>
  std::ostream& operator<<(std::ostream& os, const HostArray<T>& U) 
  {
    
    // print a HostArray
    os << "HostArray values:" << std::endl;
    {
      for (uint nVar = 0; nVar < U.nvar(); ++nVar) {
	os << "nVar = " << nVar << std::endl;
	for (uint k = 0; k < U.dimz(); ++k) {
	  os << "k = " << k << std::endl;
	  for (uint j = 0; j < U.dimy(); ++j) {
	    for (uint i = 0; i < U.dimx(); ++i) {
	      os.precision(PRECISION); os.width(WIDTH);
	      os << static_cast<T>(U(i,j,k,nVar)) << " ";
	    }
	    os << std::endl;
	  }
	  os << std::endl;
	}
	os << std::endl;
      }
      return os;
    }
  } // operator<<

#ifdef __CUDACC__
/**
 * \class DeviceArray Arrays.h
 * \brief Provides an array object with memory allocated on GPU.
 *
 * This class is symetric of HostArray, but allocated in GPU global
 * memory.
 * We take care of alignment constraints by using cudaMallocPitch
 * routine for memory allocation.
 */
template<typename T>
class DeviceArray
{
public:
  /** enumeration only used in the CUDA implementation */
  enum DeviceMemoryAllocType {
    LINEAR,  /**< enum LINEAR  (standard allocation using cudaMalloc) */
    PITCHED  /**< enum PITCHED (allocation using cudaMallocPitch) */
  };
  DeviceArray();
  ~DeviceArray();

  /** memory allocation for 1D data */
  void allocate(int length, int numVar, DeviceMemoryAllocType memAllocType=LINEAR);
  /** memory allocation for 2D data */
  void allocate(uint3 dim, DeviceMemoryAllocType memAllocType=PITCHED);
  /** memory allocation for 3D data */
  void allocate(uint4 dim, DeviceMemoryAllocType memAllocType=PITCHED);
  void free();

  void copyFromHost(const HostArray<T>& src);
  void copyToHost(HostArray<T>& dest);
  void copyTo(DeviceArray<T>& dest);

  uint dimx() const	{ return _dim.x; }
  uint dimy() const	{ return _dim.y; }
  uint dimz() const	{ return _dim.z; }
  uint nvar() const	{ return _dim.w; }

  uint pitch() const	{ return _pitch; }
  uint section() const	{ return pitch() * _dim.y * _dim.z; }
  uint size() const	{ return pitch() * _dim.y * _dim.z * _dim.w; }

  uint dimXBytes() const	{ return dimx() * sizeof(T); }
  uint pitchBytes() const 	{ return pitch() * sizeof(T); }
  uint sectionBytes() const	{ return section() * sizeof(T); }
  uint sizeBytes() const	{ return size()  * sizeof(T); }

  bool usePitchedMemory() const { return _usePitchedMemory; }

  T* data()		{ return _data; }
  const T* data() const	{ return _data; }
  /** access 2d data (only valid if _dim.z=1) */
  T& operator() (int i, int j, int ivar) { 
    return _data[i+_pitch*(j+_dim.y*ivar)]; }
  T  operator() (int i, int j, int ivar) const { 
    return _data[i+_pitch*(j+_dim.y*ivar)]; }
  /** access 3d data */
  T& operator() (int i, int j, int k, int ivar) { 
    return _data[i+_pitch*(j+_dim.y*(k+_dim.z*ivar))]; }
  T  operator() (int i, int j, int k, int ivar) const { 
    return _data[i+_pitch*(j+_dim.y*(k+_dim.z*ivar))]; }
  /** access data directly */
  T& operator() (int i) { return _data[i]; }

  /** other methods */
  void reset();

  /** total allocated memory in bytes on GPU device */
  static unsigned long int totalAllocMemoryInKB;

private:
  T*	_data;
  uint4	_dim;
  uint	_pitch;
  bool  _usePitchedMemory;
}; // class DeviceArray

  template<typename T> unsigned long int DeviceArray<T>::totalAllocMemoryInKB = 0;

#endif // __CUDACC__



////////////////////////////////////////////////////////////////////////////////
// HostArray class methods body
////////////////////////////////////////////////////////////////////////////////

  // =======================================================
  // =======================================================
template<typename T>
HostArray<T>::HostArray()
  : _data(0), _dim(make_uint4(0, 0, 0, 0)), _usePinnedMemory(false), isAllocated(false)
{
}

  // =======================================================
  // =======================================================
template<typename T>
HostArray<T>::~HostArray()
{
  free();
}

  // =======================================================
  // =======================================================
  /* 1d data allocation */
  template<typename T>
  void HostArray<T>::allocate(int length, 
			      int numVar, 
			      HostMemoryAllocType memAllocType)
  {
    
#ifdef __CUDACC__
    
    if (memAllocType == PINNED) {
      _usePinnedMemory = true;
    }  
    _dim.x = length;
    _dim.y = 1;
    _dim.z = 1;
    _dim.w = numVar;
    if (_usePinnedMemory) {
      cutilSafeCall( cudaMallocHost((void**)&_data, length * numVar * sizeof(T)) );
    } else {
      free();
      _data = new T[length * numVar];
    }
    
#else // standard version (non-CUDA)
    
    (void) memAllocType; // avoid compiler warning
    
    free();
    _dim.x = length;
    _dim.y = 1;
    _dim.z = 1;
    _dim.w = numVar;
    _data = new T[length * numVar];
    
#endif // __CUDACC__
    
    isAllocated = true;
    totalAllocMemoryInKB += (length * numVar * sizeof(T) / 1024);   

  } // HostArray<T>::allocate for 1D data
  
  // =======================================================
  // =======================================================
  /* 2d data allocation */
  template<typename T>
  void HostArray<T>::allocate(uint3 dim, HostMemoryAllocType memAllocType)
{
#ifdef __CUDACC__

  if (memAllocType == PINNED) {
    _usePinnedMemory = true;
  }  
  _dim.x = dim.x;
  _dim.y = dim.y;
  _dim.z = 1;
  _dim.w = dim.z;
  if (_usePinnedMemory) {
    cutilSafeCall( cudaMallocHost((void**)&_data, dim.x * dim.y * dim.z * sizeof(T)) );
  } else {
    free();
    _data = new T[dim.x * dim.y * dim.z];
  }

#else // standard version (non-CUDA)

  (void) memAllocType; // avoid compiler warning

  free();
  _dim.x = dim.x;
  _dim.y = dim.y;
  _dim.z = 1;
  _dim.w = dim.z;
  _data = new T[dim.x * dim.y * dim.z];

#endif // __CUDACC__

  isAllocated = true;

  totalAllocMemoryInKB += (dim.x * dim.y * dim.z * sizeof(T) / 1024);

} // void HostArray<T>::allocate for 2D data

  // =======================================================
  // =======================================================
/* 3d data allocation */
template<typename T>
void HostArray<T>::allocate(uint4 dim, HostMemoryAllocType memAllocType)
{
#ifdef __CUDACC__

  if (memAllocType == PINNED) {
    _usePinnedMemory = true;
  }
  _dim = dim;
  if (_usePinnedMemory) {
    cutilSafeCall( cudaMallocHost((void**)&_data, dim.x * dim.y * dim.z * dim.w * sizeof(T)) );
  } else {
    free();
    _data = new T[dim.x * dim.y * dim.z * dim.w];
  }

#else // standard version (non-CUDA)

  (void) memAllocType; // avoid compiler warning

  free();
  _dim = dim;
  _data = new T[dim.x * dim.y * dim.z * dim.w];

#endif // __CUDACC__

  isAllocated = true;

  totalAllocMemoryInKB += (dim.x * dim.y * dim.z * dim.w * sizeof(T) / 1024);

} // void HostArray<T>::allocate for 3D data

  // =======================================================
  // =======================================================
template<typename T>
void HostArray<T>::free()
{
#ifdef __CUDACC__

  if (_usePinnedMemory) {
    cutilSafeCall( cudaFreeHost(_data) );
  } else {
    delete[] _data;
  }

#else // standard version (non-CUDA)

  delete[] _data;

#endif // __CUDACC__

  isAllocated = false;

} // HostArray<T>::free

  // =======================================================
  // =======================================================
template<typename T>
void HostArray<T>::copyHard(HostArray<T>& src)
{
  uint4	src_dim;
  HostMemoryAllocType srcMemAllocType = (src.usePinnedMemory()) ? PINNED : PAGEABLE;

  src_dim.x = src.dimx();
  src_dim.y = src.dimy();
  src_dim.z = src.dimz();
  src_dim.w = src.nvar();
  
  // memory allocation (eventually free before allocate, if previously allocated)
  if (src_dim.y == 1 and src_dim.z == 1) { // ONE_D
    this->allocate(src_dim.x, src_dim.w, srcMemAllocType);
  } else if (src_dim.y != 1 and src_dim.z == 1) { // TWO_D
    this->allocate(make_uint3(src_dim.x, src_dim.y, src_dim.w), srcMemAllocType);
  } else { // THREE_D
    this->allocate(src_dim, srcMemAllocType);
  }
  
  // copy data
  //T* src_data = src.data();
  for (unsigned int i=0; i<src.size(); i++) {
    this->_data[i] = src(i);
  }

} // HostArray<T>::copyHard

  // =======================================================
  // =======================================================
template<typename T>
void HostArray<T>::copyTo(HostArray<T>& dest)
{
  //uint4	dest_dim;
  //HostMemoryAllocType destMemAllocType = (dest.usePinnedMemory()) ? PINNED : PAGEABLE;

  if (_dim.x != dest.dimx() or
      _dim.y != dest.dimy() or
      _dim.z != dest.dimz() or
      _dim.w != dest.nvar() ) {
    std::cerr << "HostArray dimensions do not match ! abort...\n";
    return;
  }
  
  // copy data
  //T* dest_data = dest.data();
  for (unsigned int i=0; i<dest.size(); i++) {
    dest(i) = this->_data[i];
  }

} // HostArray<T>::copyTo

  // =======================================================
  // =======================================================
template<typename T>
HostArray<T> &HostArray<T>::operator+=(const HostArray<T> &operand)
{

  // check arrays have same shape
  if (!arraysHaveSameShape(*this, operand)) {
    std::cerr << "HostArray<T>::operator+= : arrays do not have same shape\n";
    return *this;
  }
   
  // apply operator
  for (unsigned int i=0; i<size(); i++) {
    this->_data[i] += operand(i);
  }

  return *this;

} // HostArray<T>::operator+=

  // =======================================================
  // =======================================================
template<typename T>
HostArray<T> &HostArray<T>::operator-=(const HostArray<T> &operand)
{

  // check arrays have same shape
  if (!arraysHaveSameShape(*this, operand)) {
    std::cerr << "HostArray<T>::operator-= : arrays do not have same shape\n";
    return *this;
  }
   
  // apply operator
  for (unsigned int i=0; i<size(); i++) {
    this->_data[i] -= operand(i);
  }

  return *this;

} // HostArray<T>::operator-=

  // =======================================================
  // =======================================================
template<typename T>
HostArray<T> &HostArray<T>::operator*=(const HostArray<T> &operand)
{

  // check arrays have same shape
  if (!arraysHaveSameShape(*this, operand)) {
    std::cerr << "HostArray<T>::operator*= : arrays do not have same shape\n";
    return *this;
  }
   
  // apply operator
  for (unsigned int i=0; i<size(); i++) {
    this->_data[i] *= operand(i);
  }

  return *this;

} // HostArray<T>::operator*=

  // =======================================================
  // =======================================================
template<typename T>
HostArray<T> &HostArray<T>::operator/=(const HostArray<T> &operand)
{

  // check arrays have same shape
  if (!arraysHaveSameShape(*this, operand)) {
    std::cerr << "HostArray<T>::operator/= : arrays do not have same shape\n";
    return *this;
  }
   
  // apply operator
  for (unsigned int i=0; i<size(); i++) {
    this->_data[i] /= operand(i);
  }

  return *this;

} // HostArray<T>::operator/=

////////////////////////////////////////////////////////////////////////////////
// DeviceArray class methods body
////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__
template<typename T>
DeviceArray<T>::DeviceArray()
  : _data(0), _dim(make_uint4(0, 0, 0, 0)), _pitch(0), _usePitchedMemory(true), , isAllocated(false)
{
}

template<typename T>
DeviceArray<T>::~DeviceArray()
{
  free();
}

  /**
   * 1D memory allocation on device.
   * @param[in] length : dimension of the 1D array
   * @param[in] numVar : number of fields
   * @param[in] memAllocType : triggers pitched or linear memory type.
   */
template<typename T>
void DeviceArray<T>::allocate(int length, 
			      int numVar, 
			      DeviceMemoryAllocType memAllocType)
{
  // most probably here, we will never use PITCHED allocation, but just in case....

  free();
  _dim.x = length;
  _dim.y = 1;
  _dim.z = 1;
  _dim.w = numVar;
  const uint rows = nvar();
  const uint dimXBytes = dimx() * sizeof(T);

  if (memAllocType == LINEAR) {
    _usePitchedMemory = false;
    cutilSafeCall( cudaMalloc((void**) &_data, rows * dimXBytes) );
    cutilSafeCall( cudaMemset((void* )  _data, 0, rows*dimXBytes) );
    _pitch = length;
  } else { // PITCHED
    size_t pitchBytes;
    cutilSafeCall( cudaMallocPitch((void**) &_data, &pitchBytes, dimXBytes, rows) );
    cutilSafeCall( cudaMemset2D(   (void* )  _data,  pitchBytes, 0, dimXBytes, rows) );
    _pitch = pitchBytes / sizeof(T);
  }

  isAllocated = true;

  totalAllocMemoryInKB += (_pitch * _dim.y * _dim.z * _dim.w * sizeof(T) / 1024);
  //std::cout << "Device memory allocated : " << totalAllocMemoryInKB/1000 << "MB\n";

} // DeviceArray<T>::allocate for 1D

  /**
   * 2D memory allocation on device.
   * @param[in] dim : dimension of the 2D array
   * @param[in] memAllocType : triggers pitched or linear memory type.
   */
template<typename T>
void DeviceArray<T>::allocate(uint3 dim, DeviceMemoryAllocType memAllocType)
{
  free();
  _dim.x = dim.x;
  _dim.y = dim.y;
  _dim.z = 1;
  _dim.w = dim.z;
  const uint rows = dimy() * nvar();
  const uint dimXBytes = dimx() * sizeof(T);

  if (memAllocType == LINEAR) {
    _usePitchedMemory = false;
    cutilSafeCall( cudaMalloc((void**) &_data, rows * dimXBytes) );
    cutilSafeCall( cudaMemset((void* )  _data, 0, rows*dimXBytes) );
    _pitch = dim.x;
  } else { // PITCHED
    size_t pitchBytes;
    cutilSafeCall( cudaMallocPitch((void**) &_data, &pitchBytes, dimXBytes, rows) );
    cutilSafeCall( cudaMemset2D(   (void* )  _data,  pitchBytes, 0, dimXBytes, rows) );
    _pitch = pitchBytes / sizeof(T);
  }

  isAllocated = true;

  totalAllocMemoryInKB += (_pitch * _dim.y * _dim.z * _dim.w * sizeof(T) / 1024);
  //std::cout << "Device memory allocated : " << totalAllocMemoryInKB/1000 << "MB\n";

} // DeviceArray<T>::allocate for 2D

/**
 * 3D memory allocation on device.
 * @param[in] dim : dimension of the 3D array
 * @param[in] memAllocType : triggers pitched or linear memory type.
 */
template<typename T>
void DeviceArray<T>::allocate(uint4 dim, DeviceMemoryAllocType memAllocType)
{
  free();
  _dim = dim;
  const uint rows = dimy() * dimz() * nvar();
  const uint dimXBytes = dimx() * sizeof(T);

  if (memAllocType == LINEAR) {
    _usePitchedMemory = false;
    cutilSafeCall( cudaMalloc((void**) &_data, rows * dimXBytes) );
    cutilSafeCall( cudaMemset((void*)   _data, 0, rows*dimXBytes) );
    _pitch = dim.x;
  } else { // PITCHED
    size_t pitchBytes;
    cutilSafeCall( cudaMallocPitch((void**) &_data, &pitchBytes, dimXBytes, rows) );
    cutilSafeCall( cudaMemset2D(   (void*)   _data,  pitchBytes, 0, dimXBytes, rows) );
    _pitch = pitchBytes / sizeof(T);
  }

  isAllocated = true;

  totalAllocMemoryInKB += (_pitch * _dim.y * _dim.z * _dim.w * sizeof(T) / 1024);
  //std::cout << "Device memory allocated : " << totalAllocMemoryInKB/1000 << "MB\n";

} // DeviceArray<T>::allocate for 3D

template<typename T>
void DeviceArray<T>::free()
{
  cutilSafeCall( cudaFree(_data) );

  isAllocated = false;
} // DeviceArray<T>::free

  /**
   * Copy data from Host to Device memory.
   * @param[in] src : HostArray source data
   *
   * We use the host source array property usePinnedMemory to trigger the
   * use of asynchronous memory copy routines.
   * We use the device destination array property usePitchedMemory to
   * trigger linear or pitched version of the copy routines.
   */
template<typename T>
void DeviceArray<T>::copyFromHost(const HostArray<T>& src)
{
  if(dimx() != src.dimx() or
     dimy() != src.dimy() or
     dimz() != src.dimz() or
     nvar() != src.nvar())
    {
      throw std::runtime_error("copyFromHost: non-matching array size");
    }

  // host memory was allocated by cudaMallocHost, so we use async mem copy
  if ( src.usePinnedMemory() ) { 
    if (_usePitchedMemory) {
      cutilSafeCall( cudaMemcpy2DAsync(data(), pitchBytes(), 
				       src.data(), src.dimXBytes(),
				       src.dimXBytes(), src.dimy() * src.dimz() * src.nvar(), 
				       cudaMemcpyHostToDevice) );
    } else { // linear device memory
      cutilSafeCall( cudaMemcpyAsync(data(),
				     src.data(), 
				     src.sizeBytes(),
				     cudaMemcpyHostToDevice) );
    }
  } else { // host memory was allocated by new
    if (_usePitchedMemory) {
      cutilSafeCall( cudaMemcpy2D(data(), pitchBytes(), 
				  src.data(), src.dimXBytes(),
				  src.dimXBytes(), src.dimy() * src.dimz() * src.nvar(), 
				  cudaMemcpyHostToDevice) );
    } else { // linear device memory
      cutilSafeCall( cudaMemcpy(data(),
				src.data(), 
				src.sizeBytes(),
				cudaMemcpyHostToDevice) );
    }
  }
} // DeviceArray<T>::copyFromHost

  /**
   * Copy data from Device to Host memory.
   * @param[in] src : HostArray destination data
   *
   * We use the host source array property usePinnedMemory to trigger the
   * use of asynchronous memory copy routines.
   * We use the device destination array property usePitchedMemory to
   * trigger linear or pitched version of the copy routines.
   */
template<typename T>
void DeviceArray<T>::copyToHost(HostArray<T>& dest)
{
  if(dimx() != dest.dimx() or
     dimy() != dest.dimy() or
     dimz() != dest.dimz() or 
     nvar() != dest.nvar())
    {
      throw std::runtime_error("copyToHost: non-matching array size");
    }
  
  // host memory was allocated by cudaMallocHost, so we use async mem copy
  if ( dest.usePinnedMemory() ) { 
    if (_usePitchedMemory) {
      cutilSafeCall( cudaMemcpy2DAsync(dest.data(), dest.dimXBytes(), 
				       data(), pitchBytes(),
				       dimXBytes(), dimy() * dimz() * nvar(), 
				       cudaMemcpyDeviceToHost) );
    } else { // linear device memory
      cutilSafeCall( cudaMemcpyAsync(dest.data(), 
				     data(),
				     sizeBytes(),
				     cudaMemcpyDeviceToHost) );      
    }
  } else { // host memory was allocated by new
    if (_usePitchedMemory) {
      cutilSafeCall( cudaMemcpy2D(dest.data(), dest.dimXBytes(), 
				  data(), pitchBytes(),
				  dimXBytes(), dimy() * dimz() * nvar(), 
				  cudaMemcpyDeviceToHost) );
    } else { // linear device memory
      cutilSafeCall( cudaMemcpy(dest.data(), 
				data(),
				sizeBytes(),
				cudaMemcpyDeviceToHost) );
    }
  }
} // DeviceArray<T>::copyToHost

  /**
   * Copy data from Device to Device memory.
   * @param[in] dest : DeviceArray destination data
   *
   */
template<typename T>
void DeviceArray<T>::copyTo(DeviceArray<T>& dest)
{
  if(dimx() != dest.dimx() or
     dimy() != dest.dimy() or
     dimz() != dest.dimz() or 
     nvar() != dest.nvar())
    {
      throw std::runtime_error("DeviceArray::copyTo: non-matching array size");
    }
  
  if (_usePitchedMemory) {
    cutilSafeCall( cudaMemcpy2D(dest.data(), dest.pitchBytes(), 
				data(), pitchBytes(),
				dimXBytes(), dimy() * dimz() * nvar(), 
				cudaMemcpyDeviceToDevice) );
  } else { // linear device memory
    cutilSafeCall( cudaMemcpy(dest.data(), 
			      data(),
			      sizeBytes(),
			      cudaMemcpyDeviceToDevice) );
  }

} // DeviceArray<T>::copyTo

template<typename T>
void DeviceArray<T>::reset()
{

  if (!_usePitchedMemory) { // LINEAR memory

    cutilSafeCall( cudaMemset((void* )  _data, 0, sizeBytes()) );

  } else { // PITCHED memory

    const uint rows      = dimy() * dimz() * nvar();
    const uint dimXBytes = dimx() * sizeof(T);
    cutilSafeCall( cudaMemset2D(   (void* )  _data,  _pitch*sizeof(T), 0, dimXBytes, rows) );

  }

} // DeviceArray<T>::reset

#endif // __CUDACC__

} // namespace hydroSimu

#endif /*ARRAYS_H_*/
