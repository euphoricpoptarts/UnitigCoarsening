cmake_minimum_required(VERSION 3.13)
project(sgpar)

#change the first parameter to wherever your kokkos and kokkos-kernels are located
add_subdirectory(../kokkos bin_kokkos/)
add_subdirectory(../kokkos-kernels bin_kernels/)

add_executable(sgpar sgpar.c)
target_compile_options(sgpar PRIVATE -D_KOKKOS)
target_link_libraries(sgpar Kokkos::kokkoskernels)