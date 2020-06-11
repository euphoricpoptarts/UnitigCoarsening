cmake_minimum_required(VERSION 3.13)
project(sgpar CXX)

#kokkos and kokkos-kernels assumed to be in parallel directories to this project
add_subdirectory(../kokkos)
add_subdirectory(../kokkos-kernels)

add_library(sgparLib sgpar.c sgpar.h ExperimentLoggerUtil.cpp)
target_link_libraries(sgparLib Kokkos::kokkoskernels)