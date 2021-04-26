#IMPORTANT: Use the cmake flag -DCMAKE_CXX_COMPILER=/path/to/your/nvcc_wrapper
cmake_minimum_required(VERSION 3.18)
project(unitig CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#change the first parameter to wherever your kokkos and kokkos-kernels are located
#add_subdirectory(../kokkos bin_kokkos/)
#add_subdirectory(../kokkos-kernels bin_kernels/)
find_package(KokkosKernels REQUIRED)
link_libraries(Kokkos::kokkos Kokkos::kokkoskernels)
add_compile_options(-O3)
#add_compile_options(-g -DDEBUG)

add_executable(main driver.cpp)
add_executable(huge driver.cpp)

#target_compile_definitions(main PUBLIC _KOKKOS EXPERIMENT HEC RADIX)
target_compile_definitions(huge PUBLIC HUGE)
