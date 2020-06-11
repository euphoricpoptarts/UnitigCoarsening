cmake_minimum_required(VERSION 3.13)
project(sgpar CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#change the first parameter to wherever your kokkos and kokkos-kernels are located
add_subdirectory(../kokkos bin_kokkos/)
add_subdirectory(../kokkos-kernels bin_kernels/)

set_source_files_properties(sgpar.c PROPERTIES LANGUAGE CXX)
add_executable(sgpar sgpar.c)
target_compile_options(sgpar PRIVATE -D_KOKKOS -O3)
target_link_libraries(sgpar kokkos Kokkos::kokkoskernels)