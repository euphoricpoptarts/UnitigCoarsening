cmake_minimum_required(VERSION 3.13)
project(sgpar)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#change the first parameter to wherever your kokkos and kokkos-kernels are located
add_subdirectory(../kokkos bin_kokkos/)
add_subdirectory(../kokkos-kernels bin_kernels/)

add_executable(sgpar sgpar.c)
target_compile_options(sgpar PRIVATE -D_KOKKOS)
target_link_libraries(sgpar Kokkos::kokkoskernels)
target_include_directories(sgpar PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/../kokkos-kernels")