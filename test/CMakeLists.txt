add_executable(makeConfigHydro makeConfigHydro.cpp)
target_compile_options(makeConfigHydro PRIVATE -I${CMAKE_SOURCE_DIR}/src/hydro)

configure_file(test_run.sh.in test_run.sh)
configure_file(computeL2relatif.py.in computeL2relatif.py)
