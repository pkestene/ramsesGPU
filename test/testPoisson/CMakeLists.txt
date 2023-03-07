#######################################
add_executable(testPoissonCpuFFTW2d "")
target_sources(testPoissonCpuFFTW2d
  PUBLIC
  testPoissonCpuFFTW2d.cpp)

target_include_directories(testPoissonCpuFFTW2d
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/hydro)
target_link_libraries(testPoissonCpuFFTW2d
  PUBLIC
  RamsesGPU::cnpy
  RamsesGPU::fftw)

#######################################
add_executable(testPoissonCpuFFTW3d "")
target_sources(testPoissonCpuFFTW3d
  PUBLIC
  testPoissonCpuFFTW3d.cpp)

target_include_directories(testPoissonCpuFFTW3d
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/hydro)
target_link_libraries(testPoissonCpuFFTW3d
  PUBLIC
  RamsesGPU::cnpy
  RamsesGPU::fftw)

if(USE_CUDA)

  # linking to cuda libs with modern cmake
  # https://gitlab.kitware.com/cmake/cmake/issues/17816

  # TODO : analyze why target CUDAlibs::fftw, CUDAlibs::fft are not ok
  # for some reason flag '-LCUFFT' is added on command line => link error
  # library not found

  add_executable(testPoissonGpuCuFFT2d "")
  target_sources(testPoissonGpuCuFFT2d
    PUBLIC
    testPoissonGpuCuFFT2d.cu)

  target_include_directories(testPoissonGpuCuFFT2d
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/hydro)
  target_link_libraries(testPoissonGpuCuFFT2d
    PUBLIC
    RamsesGPU::cnpy
    RamsesGPU::fftw
    CUDA::cufft
    )

  add_executable(testPoissonGpuCuFFT3d "")
  target_sources(testPoissonGpuCuFFT3d
    PUBLIC
    testPoissonGpuCuFFT3d.cu)

  target_include_directories(testPoissonGpuCuFFT3d
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/hydro)
  target_link_libraries(testPoissonGpuCuFFT3d
    PUBLIC
    RamsesGPU::cnpy
    RamsesGPU::fftw
    CUDA::cufft
    )

endif(USE_CUDA)
