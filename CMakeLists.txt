cmake_minimum_required(VERSION 3.13)
project(sgpar CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#change the first parameter to wherever your kokkos and kokkos-kernels are located
#add_subdirectory(../kokkos bin_kokkos/)
#add_subdirectory(../kokkos-kernels bin_kernels/)
find_package(KokkosKernels REQUIRED)

add_executable(sgpar sgpar.cpp)
add_compile_definitions(_KOKKOS EXPERIMENT USE_GNU_PARALLELMODE)
target_compile_options(sgpar PRIVATE -O3)
#target_compile_options(sgpar PRIVATE -g)
target_link_libraries(sgpar Kokkos::kokkos Kokkos::kokkoskernels)